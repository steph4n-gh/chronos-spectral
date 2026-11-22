#!/usr/bin/env node
/**
 * Gate I0 & I4 Verification Suite:
 * - Fail-closed budget scale validation & PARTIAL reporting when scaled > 1.0
 * - Static 448-page arena, zero imports, dual static-end bounds, and full 12 x 921,600 B ring disjointness
 * - chronos_set_ingest_meta fail-closed validation, atomic phase guards, and authoritative staged clocks
 * - 32-bit t_src_us_lo -> hi carry, u64 rebase, equal-PTS distinct commit, isolated gap resets + frozen BOS ref
 * - Engine u32::MAX -> 0 rollover (FRAME_GAP_WRAP + overflow-free ring slots) & byte-identical determinism replay
 * - Real product/per-slot Hero stamp refresh + exact held identity preservation across Micro/Grad/FTLE/Modal/HeroPhys/BOS
 * - Real web/worker.js rejection & backpressure test (bad byteLength, duplicate, unmapped ptr -> ingest_rejected)
 * - Gate I4 1,000-frame HD K=0 textured workload (dense swarm + active textured masked PCG BOS + Hero physics)
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { performance } from 'node:perf_hooks';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const DEFAULT_WASM = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const WASM_PATH = path.resolve(process.argv.find((a) => a.startsWith('--wasm='))?.split('=')[1] || DEFAULT_WASM);
const WORKER_PATH = path.join(ROOT, 'web', 'worker.js');

const PAGE_BYTES = 65536;
const EXPECTED_PAGES = 448;
const EXPECTED_MEM_BYTES = EXPECTED_PAGES * PAGE_BYTES; // 29,360,128
const MAX_ARENA_BYTES = 29_100_000;
const WAVE1_MAX_STATIC_END = 29_294_592;
const FULL_RING_SLOT_BYTES = 640 * 360 * 4; // 921,600 B
const RING_LEN = 12;
const RING_META_STRIDE = 24;
const ROW_LUMA_H = 360;
const WAVE1_TELEM_LEN = 192;
const BOS_W = 256;
const BOS_H = 128;
const BOS_N = BOS_W * BOS_H;

const STAMP_VALID = 1 << 0;
const STAMP_HELD = 1 << 1;

const GAP_SEQ = 1 << 0;
const GAP_NONINC_PTS = 1 << 1;
const GAP_EXCESS_DT = 1 << 2;
const GAP_SEEK_RESTART = 1 << 3;
const GAP_GEOM_CHANGE = 1 << 4;
const GAP_SCENE_CUT = 1 << 5;
const GAP_WRAP = 1 << 6;

function assert(cond, msg) {
  if (!cond) throw new Error(`FAIL [Gate I0/I4]: ${msg}`);
}

function parseBudgetScale() {
  const raw = process.env.CHRONOS_CI_BUDGET_SCALE;
  if (raw === undefined || raw === '') return 1.0;
  const val = Number(raw);
  assert(Number.isFinite(val) && val >= 1.0, `Invalid nonfinite or <1 CHRONOS_CI_BUDGET_SCALE="${raw}"`);
  return val;
}

function solveFiedlerLanczos(exp, laplacian, fiedler) {
  if (typeof exp.chronos_solve_fiedler_lanczos === 'function') {
    return exp.chronos_solve_fiedler_lanczos();
  }
  const N = 32;
  const active = [];
  for (let i = 0; i < N; i++) {
    if (laplacian[i * N + i] > 1e-6) active.push(i);
  }
  const n = active.length;
  fiedler.fill(0.0);
  if (n < 2) {
    exp.chronos_set_fiedler_spectrum(0.0, 0.0);
    return 0.0;
  }
  const invN = 1.0 / n;
  const q = new Float64Array(n);
  let mean0 = 0.0;
  for (let a = 0; a < n; a++) {
    q[a] = a + 1 - 0.5 * (n + 1);
    mean0 += q[a];
  }
  mean0 *= invN;
  let norm0 = 0.0;
  for (let a = 0; a < n; a++) {
    q[a] -= mean0;
    norm0 += q[a] * q[a];
  }
  const invNorm0 = 1.0 / Math.sqrt(Math.max(norm0, 1e-18));
  for (let a = 0; a < n; a++) q[a] *= invNorm0;
  let lam2 = 0.0;
  for (let it = 0; it < 20; it++) {
    let mean = 0.0;
    const w = new Float64Array(n);
    for (let a = 0; a < n; a++) {
      let s = 0.0;
      const ia = active[a] * N;
      for (let b = 0; b < n; b++) s += laplacian[ia + active[b]] * q[b];
      w[a] = 10.0 * q[a] - s;
      mean += w[a];
    }
    mean *= invN;
    let n2 = 0.0;
    for (let a = 0; a < n; a++) {
      w[a] -= mean;
      n2 += w[a] * w[a];
    }
    const inv = 1.0 / Math.sqrt(Math.max(n2, 1e-18));
    for (let a = 0; a < n; a++) q[a] = w[a] * inv;
  }
  for (let a = 0; a < n; a++) {
    fiedler[active[a]] = q[a];
    let s = 0.0;
    const ia = active[a] * N;
    for (let b = 0; b < n; b++) s += laplacian[ia + active[b]] * q[b];
    lam2 += q[a] * s;
  }
  const val = Math.max(0.0, lam2);
  exp.chronos_set_fiedler_spectrum(val, 0.25 * val);
  return val;
}

function buildTexturedBosBackdrop(w, h, withWhiteoutMask = false) {
  const bg = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      const gx = Math.sin(x * 0.43 + Math.cos(y * 0.19) * 0.8);
      const gy = Math.cos(y * 0.47 - Math.sin(x * 0.23) * 0.8);
      let lum;
      if (withWhiteoutMask) {
        // High-contrast textured BOS perimeter with backlit saturated window strictly outside gauge ROI [4..24, 4..60)
        const inSwarmSky = x >= 250 && x < 540 && y >= 76 && y < 274;
        const inHeroPocket = x >= 150 && x < 196 && y >= 148 && y < 214;
        lum = (inSwarmSky || inHeroPocket)
          ? 255
          : Math.round(126 + 56 * gx * gy + 20 * Math.sin((x + y) * 0.61));
      } else {
        const inBorderAnchor = x < 140 || y < 32 || y >= h - 32;
        lum = inBorderAnchor
          ? Math.round(126 + 56 * gx * gy + 20 * Math.sin((x + y) * 0.61))
          : Math.round(28 + 11 * gx * gy + 5 * Math.sin((x + y) * 0.53));
      }
      lum = Math.max(8, Math.min(255, lum));
      bg[o] = lum;
      bg[o + 1] = lum;
      bg[o + 2] = lum;
      bg[o + 3] = 255;
    }
  }
  return bg;
}

function paintTexturedSwarmAndHero(buf, bg, w, h, f, count = 1120) {
  buf.set(bg);
  // Dark silhouetted oscillating Hero disc inside pocket [150..196, 148..214] (outside BOS gauge)
  const heroX1 = 173.0 + 8.0 * Math.sin(f * 0.14);
  const heroY1 = 181.0 + 16.0 * Math.cos(f * 0.14);
  for (let oy = -7; oy <= 7; oy++) {
    for (let ox = -7; ox <= 7; ox++) {
      if (ox * ox + oy * oy > 52) continue;
      const px = Math.round(heroX1 + ox);
      const py = Math.round(heroY1 + oy);
      if (px >= 150 && px < 196 && py >= 148 && py < 214) {
        const o = (py * w + px) * 4;
        buf[o] = 18;
        buf[o + 1] = 24;
        buf[o + 2] = 42;
        buf[o + 3] = 255;
      }
    }
  }

  // Dense 3.0 Hz coherent opposing-lobe dark micro silhouettes in backlit sky cavity [250..540, 76..274]
  const cols = 42;
  const rows = Math.ceil(count / cols);
  const xMin = 260;
  const xSpan = 270;
  const yMin = 84;
  const ySpan = 182;
  const dx = xSpan / Math.max(1, cols - 1);
  const dy = ySpan / Math.max(1, rows - 1);
  const modalOmega = 2.0 * Math.PI * 3.0 * (1.0 / 30.0);
  let idx = 0;
  for (let r = 0; r < rows && idx < count; r++) {
    for (let c = 0; c < cols && idx < count; c++, idx++) {
      const sign = c < (cols >> 1) ? 1.0 : -1.0;
      const shear = 0.04 * (r - rows * 0.5);
      const cx = Math.round(xMin + c * dx + sign * 1.2 * Math.cos(f * 0.17 + shear));
      const cy = Math.round(yMin + r * dy + sign * 1.8 * Math.sin(f * modalOmega));
      if (cx >= 252 && cx + 1 < 538 && cy >= 78 && cy + 1 < 272) {
        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            const o = ((cy + oy) * w + (cx + ox)) * 4;
            buf[o] = 18;
            buf[o + 1] = 22;
            buf[o + 2] = 28;
            buf[o + 3] = 255;
          }
        }
      }
    }
  }
}

function readStampTuple(w1, w1f, base) {
  return [
    w1[base] >>> 0,
    w1[base + 1] >>> 0,
    w1[base + 2] >>> 0,
    w1[base + 3] >>> 0,
    w1[base + 4] >>> 0,
    w1[base + 5] >>> 0,
    w1f[base + 6],
    w1[base + 7] >>> 0,
  ];
}

function instantiateWasm(wasmBytes) {
  const mod = new WebAssembly.Module(wasmBytes);
  assert(WebAssembly.Module.imports(mod).length === 0, 'Wasm module must have 0 imports');
  const inst = new WebAssembly.Instance(mod, {});
  const exp = inst.exports;
  const mem = exp.memory;
  assert(mem.buffer.byteLength === EXPECTED_MEM_BYTES, `Expected ${EXPECTED_MEM_BYTES} B, got ${mem.buffer.byteLength}`);
  return { mod, inst, exp, mem };
}

function createViews(exp, mem) {
  return {
    telem: new Float32Array(mem.buffer, exp.chronos_get_telemetry_ptr(), 1568),
    w1: new Uint32Array(mem.buffer, exp.chronos_get_wave1_telemetry_ptr(), WAVE1_TELEM_LEN),
    w1f: new Float32Array(mem.buffer, exp.chronos_get_wave1_telemetry_ptr(), WAVE1_TELEM_LEN),
    ringMeta: new Uint32Array(mem.buffer, exp.chronos_get_ring_meta_ptr(), RING_LEN * RING_META_STRIDE),
    ringMetaF32: new Float32Array(mem.buffer, exp.chronos_get_ring_meta_ptr(), RING_LEN * RING_META_STRIDE),
    rowLuma: new Float32Array(mem.buffer, exp.chronos_get_row_luma_ptr(), RING_LEN * ROW_LUMA_H),
    lap: new Float32Array(mem.buffer, exp.chronos_get_laplacian_ptr(), 1024),
    fiedler: new Float32Array(mem.buffer, exp.chronos_get_fiedler_ptr(), 32),
    bosTelem: typeof exp.chronos_get_bos_telemetry_ptr === 'function'
      ? new Uint32Array(mem.buffer, exp.chronos_get_bos_telemetry_ptr(), 32)
      : null,
    bosF32: typeof exp.chronos_get_bos_telemetry_ptr === 'function'
      ? new Float32Array(mem.buffer, exp.chronos_get_bos_telemetry_ptr(), 32)
      : null,
    bosPlanes: typeof exp.chronos_get_bos_planes_ptr === 'function'
      ? new Float32Array(mem.buffer, exp.chronos_get_bos_planes_ptr(), 6 * BOS_N)
      : null,
    microTracks: typeof exp.chronos_get_micro_tracks_ptr === 'function'
      ? new Float32Array(mem.buffer, exp.chronos_get_micro_tracks_ptr(), 3072 * 8)
      : null,
    microCov: typeof exp.chronos_get_micro_cov_ptr === 'function'
      ? new Float32Array(mem.buffer, exp.chronos_get_micro_cov_ptr(), 3 * 3072)
      : null,
  };
}

async function runWorkerRejectionHarness(wasmBytes) {
  const workerSource = fs.readFileSync(WORKER_PATH, 'utf8');
  const postedMessages = [];
  const createDummyCtx = () => ({
    fillStyle: '#000',
    strokeStyle: '#000',
    lineWidth: 1,
    font: '12px sans-serif',
    globalAlpha: 1,
    textAlign: 'left',
    textBaseline: 'alphabetic',
    imageSmoothingEnabled: false,
    setLineDash() {},
    measureText() {
      return { width: 10 };
    },
    translate() {},
    scale() {},
    rotate() {},
    clip() {},
    rect() {},
    roundRect() {},
    createLinearGradient() {
      return { addColorStop() {} };
    },
    setTransform() {},
    save() {},
    restore() {},
    fillRect() {},
    clearRect() {},
    drawImage() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    stroke() {},
    arc() {},
    ellipse() {},
    fill() {},
    closePath() {},
    fillText() {},
    strokeRect() {},
    createImageData(w, h) {
      return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) };
    },
    getImageData(_x, _y, w, h) {
      return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) };
    },
    putImageData() {},
  });
  class MockOffscreenCanvas {
    constructor(w, h) {
      this.width = w;
      this.height = h;
      this._ctx = createDummyCtx();
    }
    getContext() {
      return this._ctx;
    }
  }
  class MockImageData {
    constructor(dataOrW, wOrH, hOpt) {
      if (dataOrW instanceof Uint8ClampedArray) {
        this.data = dataOrW;
        this.width = wOrH;
        this.height = hOpt || (dataOrW.length / (wOrH * 4));
      } else {
        this.width = dataOrW;
        this.height = wOrH;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      }
    }
  }
  const fakeSelf = {
    postMessage(msg) {
      postedMessages.push(msg);
    },
    onmessage: null,
  };
  const pendingTimers = new Set();
  const sandbox = {
    self: fakeSelf,
    OffscreenCanvas: MockOffscreenCanvas,
    ImageData: MockImageData,
    WebAssembly,
    performance,
    Math,
    Number,
    Array,
    Uint8Array,
    Uint8ClampedArray,
    Uint32Array,
    Int32Array,
    Float32Array,
    Float64Array,
    ArrayBuffer,
    Error,
    Promise,
    JSON,
    console,
    setTimeout: (fn, ms) => {
      const id = setTimeout(() => {
        pendingTimers.delete(id);
        fn();
      }, ms);
      pendingTimers.add(id);
      return id;
    },
    clearTimeout: (id) => {
      pendingTimers.delete(id);
      clearTimeout(id);
    },
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: () => {},
    fetch: async () => ({
      ok: true,
      status: 200,
      arrayBuffer: async () => wasmBytes.buffer.slice(wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength),
    }),
  };
  vm.runInNewContext(workerSource, sandbox, { filename: 'web/worker.js' });
  assert(typeof fakeSelf.onmessage === 'function', 'worker.js must attach self.onmessage');

  const dummyCanvas = () => new MockOffscreenCanvas(640, 360);
  await fakeSelf.onmessage({
    data: {
      type: 'init',
      wasmUrl: './chronos_spectral.wasm',
      hudCanvas: dummyCanvas(),
      pipCanvas: dummyCanvas(),
      hilbertCanvas: dummyCanvas(),
      spectralCanvas: dummyCanvas(),
      persistenceCanvas: dummyCanvas(),
      physicsPhaseCanvas: dummyCanvas(),
    },
  });
  const readyMsg = postedMessages.find((m) => m.type === 'ready');
  assert(readyMsg && readyMsg.pages === EXPECTED_PAGES, 'Worker harness must post ready with 448 pages');

  await fakeSelf.onmessage({
    data: { type: 'set_resolution', resMode: 1 },
  });
  await fakeSelf.onmessage({
    data: { type: 'set_source', sourceMode: 'video', fpsHint: 30 },
  });

  const hdBytes = 640 * 360 * 4;
  const goodFrame = new Uint8Array(hdBytes).fill(128);
  const countType = (type) => postedMessages.filter((m) => m.type === type).length;

  // 1. Valid video_rgba -> emits frame_ack
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 10,
      seq: 1,
      width: 640,
      height: 360,
      buffer: goodFrame.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.0,
      tSrcUs: 0,
    },
  });
  assert(countType('frame_ack') === 1 && countType('ingest_rejected') === 0, `Worker must emit 1 frame_ack on valid video_rgba, got ${JSON.stringify(postedMessages.slice(-2))}`);
  const commitsAfter1 = vm.runInContext('wave1TelemView[4] >>> 0', sandbox);
  assert(commitsAfter1 === 1, `Expected commits=1 after valid worker frame, got ${commitsAfter1}`);

  // 2. Duplicate (epoch=10, seq=1) -> emits ingest_rejected, zero new frame_ack, unchanged commits
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 10,
      seq: 1,
      width: 640,
      height: 360,
      buffer: goodFrame.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.033333,
      tSrcUs: 33333,
    },
  });
  assert(countType('frame_ack') === 1 && countType('ingest_rejected') === 1, 'Worker must emit ingest_rejected on duplicate');
  assert((vm.runInContext('wave1TelemView[4] >>> 0', sandbox)) === 1, 'Duplicate worker reject must not increment commits');

  // 3. Invalid byteLength -> emits ingest_rejected, unchanged commits
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 10,
      seq: 2,
      width: 640,
      height: 360,
      buffer: new Uint8Array(1024).buffer,
      dtSec: 1 / 30,
      mediaTimeSec: 0.033333,
      tSrcUs: 33333,
    },
  });
  assert(countType('frame_ack') === 1 && countType('ingest_rejected') === 2, 'Worker must emit ingest_rejected on invalid byteLength');
  assert((vm.runInContext('wave1TelemView[4] >>> 0', sandbox)) === 1, 'Bad byteLength must not increment commits');

  // 4. Unmapped ingest pointer thrown during processIngestedFrame -> emits ingest_rejected, zero deadlock!
  const preUnmapped = countType('frame_ack');
  const prevSlot0Off = vm.runInContext('wasmExports.chronos_get_ring_slot_ptr(0)', sandbox);
  vm.runInContext('ringSlotByteOffsets.fill(-1);', sandbox);
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 10,
      seq: 3,
      width: 640,
      height: 360,
      buffer: goodFrame.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.066666,
      tSrcUs: 66666,
    },
  });
  assert(countType('frame_ack') === preUnmapped && countType('ingest_rejected') === 3, 'Worker must emit ingest_rejected on unmapped ptr');
  assert((vm.runInContext('wave1TelemView[4] >>> 0', sandbox)) === 1, 'Unmapped ptr must not increment commits');

  // Restore ringSlotByteOffsets & verify zero backpressure deadlock on subsequent valid frame!
  for (let s = 0; s < RING_LEN; s++) {
    vm.runInContext(`ringSlotByteOffsets[${s}] = wasmExports.chronos_get_ring_slot_ptr(${s});`, sandbox);
  }
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 10,
      seq: 4,
      width: 640,
      height: 360,
      buffer: goodFrame.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.1,
      tSrcUs: 100000,
    },
  });
  assert(countType('frame_ack') === 2 && countType('ingest_rejected') === 3, 'Worker must recover and ack next valid frame');
  assert((vm.runInContext('wave1TelemView[4] >>> 0', sandbox)) === 2, `Expected commits=2 after recovery, got ${vm.runInContext('wave1TelemView[4] >>> 0', sandbox)}`);

  // 5. Mid-tick throw between chronos_pre_tick and chronos_post_tick recovers cleanly without status=1, commit increment, or phase/sync wedge
  vm.runInContext(`
    const __origWasmExports = wasmExports;
    let __throwOnce = true;
    wasmExports = Object.assign({}, __origWasmExports, {
      chronos_solve_fiedler_lanczos() {
        if (__throwOnce) { __throwOnce = false; throw new Error('injected_mid_tick_fault'); }
        return __origWasmExports.chronos_solve_fiedler_lanczos();
      },
    });
  `, sandbox);
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 10,
      seq: 5,
      width: 640,
      height: 360,
      buffer: goodFrame.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.133333,
      tSrcUs: 133333,
    },
  });
  const midReject = postedMessages[postedMessages.length - 1];
  assert(midReject && midReject.type === 'ingest_rejected' && midReject.status !== 1, `Mid-tick throw must emit ingest_rejected with status!=1, got ${JSON.stringify(midReject)}`);
  assert((vm.runInContext('wave1TelemView[4] >>> 0', sandbox)) === 3, 'Mid-tick throw preserves pre-solver commit count=3');
  assert((vm.runInContext('wave1TelemView[13] >>> 0', sandbox)) === 0, 'Mid-tick throw reset must clear split_phase w1[13]=0');
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 10,
      seq: 6,
      width: 640,
      height: 360,
      buffer: goodFrame.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.166666,
      tSrcUs: 166666,
    },
  });
  const postMidAck = postedMessages[postedMessages.length - 1];
  assert(postMidAck && postMidAck.type === 'frame_ack' && postMidAck.status === 1 && postMidAck.committed === true, `Next valid frame after mid-tick throw must commit cleanly, got ${JSON.stringify(postMidAck)}`);
  assert((vm.runInContext('wave1TelemView[4] >>> 0', sandbox)) === 4, 'Next frame after mid-tick throw must advance commits=4');

  // 6. Clock gating: !PTS_KNOWN -> t_src_us_lo == 0 && t_src_us_hi == 0 (never workerSimTimeSec)
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 11,
      seq: 1,
      width: 640,
      height: 360,
      buffer: goodFrame.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: -1.0,
    },
  });
  assert(postedMessages[postedMessages.length - 1].type === 'frame_ack', 'No-PTS frame must still ingest cleanly');
  const noPtsMeta = vm.runInContext(`(() => {
    const sl = wave1TelemView[2] >>> 0;
    const o = sl * 24;
    return { flags: ringMetaView[o] >>> 0, tLo: ringMetaView[o + 4] >>> 0, tHi: ringMetaView[o + 5] >>> 0 };
  })()`, sandbox);
  assert((noPtsMeta.flags & 2) === 0 && noPtsMeta.tLo === 0 && noPtsMeta.tHi === 0, `Unknown PTS must zero t_src_us, got ${JSON.stringify(noPtsMeta)}`);

  // 7. Dimensionless video_rgba with preScaled: true & roiOrientation=4 (h-flip) executes extractAndOrientRgba and preserves fallbackScale
  await fakeSelf.onmessage({ data: { type: 'set_resolution', resMode: 0 } });
  await fakeSelf.onmessage({ data: { type: 'set_roi', native1to1: true, orientation: 4, scanDir: 0xffffffff } });
  const asymSd = new Uint8Array(320 * 180 * 4);
  for (let y = 0; y < 180; y++) {
    for (let x = 0; x < 320; x++) {
      const o = (y * 320 + x) * 4;
      asymSd[o] = x < 160 ? 220 : 25;
      asymSd[o + 1] = x < 160 ? 220 : 25;
      asymSd[o + 2] = x < 160 ? 220 : 25;
      asymSd[o + 3] = 255;
    }
  }
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 12,
      seq: 1,
      buffer: asymSd.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.0,
      tSrcUs: 0,
      preScaled: true,
    },
  });
  const dimlessChk = vm.runInContext(`(() => {
    const sl = wave1TelemView[2] >>> 0;
    const o = sl * 24;
    const rgbaPtr = wasmExports.chronos_get_ring_slot_ptr(sl);
    const px = new Uint8Array(wasmMemory.buffer, rgbaPtr, 320 * 180 * 4);
    return {
      ack: lastIngestGeom.fallbackScale,
      nat: lastIngestGeom.native1to1,
      metaNat: ringMetaView[o + 22] >>> 0,
      leftR: px[(90 * 320 + 10) * 4],
      rightR: px[(90 * 320 + 310) * 4],
    };
  })()`, sandbox);
  assert(dimlessChk.ack === true && dimlessChk.nat === 0 && dimlessChk.metaNat === 0, `Dimensionless preScaled must preserve fallbackScale/native1to1=0, got ${JSON.stringify(dimlessChk)}`);
  assert(dimlessChk.leftR === 25 && dimlessChk.rightR === 220, `Dimensionless orientation=4 must mirror image, got L=${dimlessChk.leftR}, R=${dimlessChk.rightR}`);

  // 8. Non-zero visibleRect separates coded pixel reads from visible-relative RingMeta coordinates
  await fakeSelf.onmessage({ data: { type: 'set_roi', native1to1: true, orientation: 0, scanDir: 0xffffffff, x0: 10, y0: 5 } });
  const visRectBuf = new Uint8Array(640 * 360 * 4);
  for (let y = 0; y < 360; y++) {
    for (let x = 0; x < 640; x++) {
      const o = (y * 640 + x) * 4;
      const inCrop = x >= 50 && x < 370 && y >= 25 && y < 205;
      visRectBuf[o] = inCrop ? 190 : 12;
      visRectBuf[o + 1] = inCrop ? 190 : 12;
      visRectBuf[o + 2] = inCrop ? 190 : 12;
      visRectBuf[o + 3] = 255;
    }
  }
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 13,
      seq: 1,
      width: 640,
      height: 360,
      buffer: visRectBuf.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.033333,
      tSrcUs: 33333,
      geom: { srcW: 640, srcH: 360, visibleRect: { x: 40, y: 20, width: 560, height: 320 } },
    },
  });
  const vrChk = vm.runInContext(`(() => {
    const sl = wave1TelemView[2] >>> 0;
    const o = sl * 24;
    const rgbaPtr = wasmExports.chronos_get_ring_slot_ptr(sl);
    const px = new Uint8Array(wasmMemory.buffer, rgbaPtr, 320 * 180 * 4);
    return {
      srcW: ringMetaView[o + 7] >>> 0,
      srcH: ringMetaView[o + 8] >>> 0,
      cropX0: ringMetaView[o + 9] >>> 0,
      cropY0: ringMetaView[o + 10] >>> 0,
      cropW: ringMetaView[o + 11] >>> 0,
      cropH: ringMetaView[o + 12] >>> 0,
      sampleTL: px[0],
      sampleBR: px[(179 * 320 + 319) * 4],
    };
  })()`, sandbox);
  assert(vrChk.srcW === 560 && vrChk.srcH === 320 && vrChk.cropX0 === 10 && vrChk.cropY0 === 5 && vrChk.cropW === 320 && vrChk.cropH === 180, `visibleRect RingMeta coords mismatch: ${JSON.stringify(vrChk)}`);
  assert(vrChk.sampleTL === 190 && vrChk.sampleBR === 190, `visibleRect must sample coded pixels (50,25): ${JSON.stringify(vrChk)}`);

  // 9. Repeat set_resolution(0 -> 0) mid-stream must preserve workerExpectedEngineFrame and keep committing
  const preRepeatCommits = vm.runInContext('wave1TelemView[4] >>> 0', sandbox);
  const expBeforeRepeat = vm.runInContext('workerExpectedEngineFrame >>> 0', sandbox);
  await fakeSelf.onmessage({ data: { type: 'set_resolution', resMode: 0 } });
  const expAfterRepeat = vm.runInContext('workerExpectedEngineFrame >>> 0', sandbox);
  assert(expAfterRepeat === expBeforeRepeat && expAfterRepeat === 2, `Repeat set_resolution(0->0) must preserve workerExpectedEngineFrame=${expBeforeRepeat}, got ${expAfterRepeat}`);
  const sdSolid = new Uint8Array(320 * 180 * 4).fill(99);
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 13,
      seq: 2,
      width: 320,
      height: 180,
      buffer: sdSolid.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.066666,
      tSrcUs: 66666,
    },
  });
  const repeatAck = postedMessages[postedMessages.length - 1];
  assert(repeatAck && repeatAck.type === 'frame_ack' && (vm.runInContext('wave1TelemView[4] >>> 0', sandbox)) === preRepeatCommits + 1, `Repeat set_resolution must not wedge ingest: ${JSON.stringify(repeatAck)}`);

  // 10. Post-commit HUD scheduling fault must not turn an already-committed tick into ingest_rejected
  vm.runInContext(`
    const __origSchedHud = scheduleOffscreenHudPresent;
    let __throwHudOnce = true;
    scheduleOffscreenHudPresent = function(...args) {
      if (__throwHudOnce) { __throwHudOnce = false; throw new Error('injected_hud_fault'); }
      return __origSchedHud(...args);
    };
  `, sandbox);
  const preHudErrCommits = vm.runInContext('wave1TelemView[4] >>> 0', sandbox);
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 13,
      seq: 3,
      width: 320,
      height: 180,
      buffer: sdSolid.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 0.1,
      tSrcUs: 100000,
    },
  });
  const hudErrAck = postedMessages[postedMessages.length - 1];
  assert(hudErrAck && hudErrAck.type === 'frame_ack' && hudErrAck.status === 1 && (vm.runInContext('wave1TelemView[4] >>> 0', sandbox)) === preHudErrCommits + 1, `Post-commit HUD exception must still emit frame_ack: ${JSON.stringify(hudErrAck)}`);

  // 11. After set_source reset, paused set_roi must not redraw stale slot 0 pixels
  vm.runInContext(`
    var __hudDrawCount = 0;
    const __origRenderMainHud = renderMainHud;
    renderMainHud = function(...args) {
      __hudDrawCount++;
      return __origRenderMainHud(...args);
    };
  `, sandbox);
  await fakeSelf.onmessage({ data: { type: 'set_source', sourceMode: 'video', fpsHint: 30 } });
  const resetState = vm.runInContext(`({ disp: wave1TelemView[3] >>> 0, hasFrame: hasAnyRenderedFrame(), draws: __hudDrawCount })`, sandbox);
  await fakeSelf.onmessage({ data: { type: 'set_roi', native1to1: true, x0: 0, y0: 0, orientation: 0, scanDir: 0 } });
  const afterResetRoi = vm.runInContext(`(() => {
    pendingHudPresentDirty = true;
    pendingHudPresentSlot = 0;
    flushPendingHudRender();
    return { disp: wave1TelemView[3] >>> 0, hasFrame: hasAnyRenderedFrame(), draws: __hudDrawCount };
  })()`, sandbox);
  assert(resetState.disp === 0xffffffff && !afterResetRoi.hasFrame && afterResetRoi.draws === resetState.draws, `Reset paused set_roi & flushPendingHudRender must not render stale HUD: ${JSON.stringify({ resetState, afterResetRoi })}`);

  // 12. Host ptsKnown:false gating, normalized tUs -> latestCausalPtsMs, alreadyOriented:true unrotated provenance, and u32 lookahead wrap
  await fakeSelf.onmessage({ data: { type: 'set_resolution', resMode: 1 } });
  await fakeSelf.onmessage({ data: { type: 'set_roi', native1to1: true, x0: 0, y0: 0, orientation: 0, scanDir: 0xffffffff } });
  const hdProbe = new Uint8Array(640 * 360 * 4);
  for (let y = 0; y < 360; y++) {
    for (let x = 0; x < 640; x++) {
      const o = (y * 640 + x) * 4;
      const v = x < 320 ? 210 : 20;
      hdProbe[o] = v; hdProbe[o + 1] = v; hdProbe[o + 2] = v; hdProbe[o + 3] = 255;
    }
  }
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 14,
      seq: 1,
      width: 640,
      height: 360,
      buffer: hdProbe.buffer.slice(0),
      dtSec: 1 / 30,
      ptsKnown: false,
      mediaTimeSec: -1,
      tSrcUs: 160000,
    },
  });
  const ptsFalseChk = vm.runInContext(`(() => {
    const sl = wave1TelemView[2] >>> 0;
    const o = sl * 24;
    return { flags: ringMetaView[o] >>> 0, tLo: ringMetaView[o + 4] >>> 0, tHi: ringMetaView[o + 5] >>> 0, causalPts: latestCausalPtsMs };
  })()`, sandbox);
  assert((ptsFalseChk.flags & 2) === 0 && ptsFalseChk.tLo === 0 && ptsFalseChk.tHi === 0 && ptsFalseChk.causalPts === -1.0, `ptsKnown:false must suppress PTS_KNOWN even with tSrcUs>0: ${JSON.stringify(ptsFalseChk)}`);

  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 14,
      seq: 2,
      width: 640,
      height: 360,
      buffer: hdProbe.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: 8.0,
      tSrcUs: 123456,
      geom: { srcW: 640, srcH: 360, alreadyOriented: true, orientation: 1 },
    },
  });
  const aoChk = vm.runInContext(`(() => {
    const sl = wave1TelemView[2] >>> 0;
    const o = sl * 24;
    const rgbaPtr = wasmExports.chronos_get_ring_slot_ptr(sl);
    const px = new Uint8Array(wasmMemory.buffer, rgbaPtr, 640 * 360 * 4);
    const lkWrap = (() => {
      const savedIdx = ringMetaView[o + 3];
      ringMetaView[o + 3] = 0xfffffffb;
      const res = computeLookaheadTiming(8, sl, 3);
      ringMetaView[o + 3] = savedIdx;
      return res;
    })();
    return {
      tLo: ringMetaView[o + 4] >>> 0,
      causalPts: latestCausalPtsMs,
      srcW: ringMetaView[o + 7] >>> 0,
      srcH: ringMetaView[o + 8] >>> 0,
      cropW: ringMetaView[o + 11] >>> 0,
      cropH: ringMetaView[o + 12] >>> 0,
      orient: ringMetaView[o + 13] >>> 0,
      nat: ringMetaView[o + 22] >>> 0,
      leftR: px[(180 * 640 + 10) * 4],
      rightR: px[(180 * 640 + 630) * 4],
      lkWrap,
    };
  })()`, sandbox);
  assert(aoChk.tLo === 123456 && Math.abs(aoChk.causalPts - 123.456) < 1e-6, `Normalized tUs must drive both Wasm PTS and latestCausalPtsMs: ${JSON.stringify(aoChk)}`);
  assert(aoChk.srcW === 360 && aoChk.srcH === 640 && aoChk.cropW === 360 && aoChk.cropH === 640 && aoChk.orient === 1 && aoChk.nat === 1, `alreadyOriented:true orientation:1 must map unrotated 360x640 RingMeta: ${JSON.stringify(aoChk)}`);
  assert(aoChk.leftR === 210 && aoChk.rightR === 20, `alreadyOriented:true must not double-rotate pixels: ${JSON.stringify(aoChk)}`);
  assert(aoChk.lkWrap.lookaheadSource === 'source_pts' && aoChk.lkWrap.displayedFrameIdx === 0xfffffffb, `computeLookaheadTiming must handle u32 wrap via ringMeta: ${JSON.stringify(aoChk.lkWrap)}`);

  // 13. Non-blocking HUD flush prefers authoritative wave1TelemView[3], plus u32 2^24+1 & u32::MAX->0 telemetry + lookahead + JSON export fidelity
  await fakeSelf.onmessage({ data: { type: 'lab_action', action: 'reset' } });
  await fakeSelf.onmessage({ data: { type: 'set_resolution', resMode: 1 } });
  await fakeSelf.onmessage({ data: { type: 'set_roi', native1to1: true, x0: 0, y0: 0, orientation: 0, scanDir: 0xffffffff } });
  await fakeSelf.onmessage({ data: { type: 'params', lookaheadK: 2, ecaAdaptRate: 0.035, fistaLambda: 1.45, eulerianGain: 25.0, phaseOffsetRad: -Math.PI * 0.5, labMode: 0 } });
  await fakeSelf.onmessage({ data: { type: 'set_swarm_mode', swarmMode: 2 } });
  await fakeSelf.onmessage({ data: { type: 'rec_control', action: 'start', clear: true } });

  const locateWarmCount = 14;
  let warmT = 0;
  for (let i = 1; i <= locateWarmCount; i++) {
    warmT += 33333;
    await fakeSelf.onmessage({
      data: {
        type: 'video_rgba',
        epoch: 15,
        seq: i,
        width: 640,
        height: 360,
        buffer: hdProbe.buffer.slice(0),
        dtSec: 1 / 30,
        mediaTimeSec: warmT / 1e6,
        tSrcUs: warmT,
      },
    });
    assert(postedMessages[postedMessages.length - 1].type === 'frame_ack', `Warmup frame ${i} failed`);
  }
  const hudFlushChk = vm.runInContext(`(() => {
    const wDisp = wave1TelemView[3] >>> 0;
    const wLast = wave1TelemView[2] >>> 0;
    let drawnSlot = -1;
    const __origRenderForFlush = renderMainHud;
    renderMainHud = function(slotArg) {
      drawnSlot = slotArg;
      return __origRenderForFlush(slotArg);
    };
    pendingHudPresentDirty = true;
    pendingHudPresentSlot = wLast;
    flushPendingHudRender();
    renderMainHud = __origRenderForFlush;
    return { wDisp, wLast, drawnSlot };
  })()`, sandbox);
  assert(hudFlushChk.wDisp !== hudFlushChk.wLast && hudFlushChk.drawnSlot === hudFlushChk.wDisp, `flushPendingHudRender must prefer wave1TelemView[3]: ${JSON.stringify(hudFlushChk)}`);

  const cand14Hex = vm.runInContext(`(() => {
    const ring0 = wasmExports.chronos_get_ring_slot_ptr(0);
    const u32 = new Uint32Array(wasmMemory.buffer, ring0, (wasmMemory.buffer.byteLength - ring0) >>> 2);
    const c = [];
    for (let wi = 2; wi < u32.length - 4; wi++) {
      if (u32[wi - 2] === 14 && u32[wi - 1] === 0 && u32[wi] === 1 && u32[wi + 1] === 14 && u32[wi + 3] === 2) c.push(wi);
    }
    return c;
  })()`, sandbox);
  warmT += 33333;
  await fakeSelf.onmessage({
    data: {
      type: 'video_rgba',
      epoch: 15,
      seq: 15,
      width: 640,
      height: 360,
      buffer: hdProbe.buffer.slice(0),
      dtSec: 1 / 30,
      mediaTimeSec: warmT / 1e6,
      tSrcUs: warmT,
    },
  });
  const resWordIdx = vm.runInContext(`(() => {
    const ring0 = wasmExports.chronos_get_ring_slot_ptr(0);
    const u32 = new Uint32Array(wasmMemory.buffer, ring0, (wasmMemory.buffer.byteLength - ring0) >>> 2);
    const cands = ${JSON.stringify(cand14Hex)};
    const hits = cands.filter((wi) => u32[wi - 2] === 15 && u32[wi - 1] === 0 && u32[wi + 1] === 15);
    return hits.length === 1 ? hits[0] : -1;
  })()`, sandbox);
  assert(resWordIdx > 0, `Worker VM failed to locate unique Arena (ring_cursor, res_mode, frame_idx): ${resWordIdx}`);

  const latestOfType = (type) => {
    for (let i = postedMessages.length - 1; i >= 0; i--) {
      if (postedMessages[i] && postedMessages[i].type === type) return postedMessages[i];
    }
    return null;
  };

  // Jump past 2^24 (f = 16777214 .. 16777217) where f32 telemView[0] loses precision on odd 2^24+1=16777217
  vm.runInContext(`(() => {
    const ring0 = wasmExports.chronos_get_ring_slot_ptr(0);
    const u32 = new Uint32Array(wasmMemory.buffer, ring0, (wasmMemory.buffer.byteLength - ring0) >>> 2);
    // frame_idx only: the ring cursor is passed back unchanged.
    wasmExports.chronos_test_hook_set_frame_counters(16777214, u32[${resWordIdx} - 2], u32[${resWordIdx} - 1]);
    if (u32[${resWordIdx} + 1] !== 16777214) throw new Error('chronos_test_hook_set_frame_counters did not set frame_idx');
    workerExpectedEngineFrame = 16777214;
  })()`, sandbox);
  for (let i = 0; i < 4; i++) {
    warmT += 33333;
    await fakeSelf.onmessage({
      data: {
        type: 'video_rgba',
        epoch: 15,
        seq: 16 + i,
        width: 640,
        height: 360,
        buffer: hdProbe.buffer.slice(0),
        dtSec: 1 / 30,
        mediaTimeSec: warmT / 1e6,
        tSrcUs: warmT,
      },
    });
    assert(postedMessages[postedMessages.length - 1].type === 'frame_ack', `2^24 step ${i} ack failed`);
  }
  const t24 = latestOfType('telemetry');
  assert(
    t24 &&
      t24.frameIdx === 16777217 &&
      t24.chronos1000 &&
      t24.chronos1000.causalFrameIdx === 16777217 &&
      t24.chronos1000.displayedFrameIdx === 16777215 &&
      t24.chronos1000.lookaheadSource === 'source_pts',
    `Worker telemetry failed u32 exact precision at 2^24+1: ${JSON.stringify({ f: t24?.frameIdx, c1k: t24?.chronos1000 })}`
  );
  await fakeSelf.onmessage({ data: { type: 'export_telemetry', format: 'json' } });
  const exp24 = latestOfType('telemetry_export_data');
  const rows24 = new Float64Array(exp24.buffer);
  const lastRowFrame24 = rows24[(exp24.rows - 1) * exp24.cols];
  assert(
    lastRowFrame24 === 16777217 &&
      exp24.meta.chronos1000.causalFrameIdx === 16777217 &&
      exp24.meta.chronos1000.displayedFrameIdx === 16777215 &&
      exp24.meta.chronos1000.lookaheadSource === 'source_pts',
    `export_telemetry truncated 2^24+1: ${JSON.stringify({ lastRowFrame24, c1k: exp24.meta.chronos1000 })}`
  );

  // Hero HUD trail continuity at 0x80000000 boundary (Uint32Array trailLastFrame + unsigned delta)
  const trail0x80 = vm.runInContext(`(() => {
    const outMid = [];
    const outWrap = [];
    const lastSlot = wave1TelemView[2] >>> 0;
    const dSlot = wave1TelemView[3] >>> 0;
    const ts = 7 & (TRAIL_SLOTS - 1);
    const savedFrame = ringMetaView[lastSlot * RING_META_STRIDE + 3];
    const savedT0 = telemView[32], savedT1 = telemView[33], savedT2 = telemView[34], savedT3 = telemView[35], savedT10 = telemView[32 + 10], savedK = telemView[1], savedCam = telemView[TELEM_GLOBAL + 9], savedShow = showHeroTargets;
    showHeroTargets = true;
    telemView[1] = 2.0;
    telemView[TELEM_GLOBAL + 9] = 0.0;
    const runSeq = (frames, dst) => {
      trailOwnerTid.fill(-1);
      trailCount.fill(0);
      trailLastFrame.fill(0);
      for (let i = 0; i < frames.length; i++) {
        ringMetaView[lastSlot * RING_META_STRIDE + 3] = frames[i] >>> 0;
        telemView[32] = 1.0;
        telemView[33] = 7.0;
        telemView[34] = 200.0 + i * 2.0;
        telemView[35] = 120.0;
        telemView[32 + 10] = 1.0;
        renderMainHud(dSlot);
        dst.push(trailCount[ts]);
      }
    };
    runSeq([0x7ffffffd, 0x7ffffffe, 0x7fffffff, 0x80000000, 0x80000001, 0x80000002, 0x80000003], outMid);
    runSeq([0xfffffffd, 0xfffffffe, 0xffffffff, 0, 1, 2], outWrap);
    ringMetaView[lastSlot * RING_META_STRIDE + 3] = savedFrame;
    telemView[32] = savedT0; telemView[33] = savedT1; telemView[34] = savedT2; telemView[35] = savedT3; telemView[32 + 10] = savedT10;
    telemView[1] = savedK; telemView[TELEM_GLOBAL + 9] = savedCam; showHeroTargets = savedShow;
    trailOwnerTid.fill(-1);
    trailCount.fill(0);
    trailLastFrame.fill(0);
    return { outMid, outWrap };
  })()`, sandbox);
  assert(
    trail0x80.outMid.join(',') === '1,2,3,4,5,6,7' && trail0x80.outWrap.join(',') === '1,2,3,4,5,6',
    `Hero trail must remain continuous across 0x80000000 and u32::MAX->0: ${JSON.stringify(trail0x80)}`
  );

  // Jump across u32::MAX -> 0 (0xfffffffd .. 0xffffffff -> 0 -> 1 -> 2) and assert live telemetry + lookahead + strike age + export
  vm.runInContext(`(() => {
    const ring0 = wasmExports.chronos_get_ring_slot_ptr(0);
    const u32 = new Uint32Array(wasmMemory.buffer, ring0, (wasmMemory.buffer.byteLength - ring0) >>> 2);
    wasmExports.chronos_test_hook_set_frame_counters(0xfffffffd, u32[${resWordIdx} - 2], u32[${resWordIdx} - 1]);
    if (u32[${resWordIdx} + 1] !== 0xfffffffd) throw new Error('chronos_test_hook_set_frame_counters did not set frame_idx');
    workerExpectedEngineFrame = 0xfffffffd;
    const __origRosterForStrike = updateSlotRosterEma;
    let __strikeInjectPending = true;
    updateSlotRosterEma = function() {
      __origRosterForStrike();
      if (__strikeInjectPending && (ringMetaView[(wave1TelemView[2] >>> 0) * RING_META_STRIDE + 3] >>> 0) === 0xffffffff) {
        __strikeInjectPending = false;
        telemView[15] = 1.0;
        telemView[TELEM_EXCUR + 0] = 1.0;
        telemView[TELEM_EXCUR + 1] = 0.0;
        telemView[TELEM_EXCUR + 2] = 1.0; // leadFrames = 1 -> scheduled at (0xffffffff + 1) >>> 0 = 0
        telemView[TELEM_EXCUR + 3] = 5.0;
        telemView[TELEM_EXCUR + 4] = 100.0;
        telemView[TELEM_EXCUR + 5] = 100.0;
        telemView[TELEM_KINETIC + 3] = 45.0;
        telemView[TELEM_KINETIC + 4] = 0.8;
        telemView[TELEM_KINETIC + 5] = 10.0;
        updateSlotRosterEma = __origRosterForStrike;
      }
    };
  })()`, sandbox);
  let tAtMax = null;
  let expAtMax = null;
  let tAtZero = null;
  let expAtZero = null;
  let gapAtZero = 0;
  const wrapStrikeAges = [];
  for (let i = 0; i < 6; i++) {
    warmT += 33333;
    await fakeSelf.onmessage({
      data: {
        type: 'video_rgba',
        epoch: 16,
        seq: i + 1,
        width: 640,
        height: 360,
        buffer: hdProbe.buffer.slice(0),
        dtSec: 1 / 30,
        mediaTimeSec: warmT / 1e6,
        tSrcUs: warmT,
      },
    });
    assert(postedMessages[postedMessages.length - 1].type === 'frame_ack', `u32::MAX wrap step ${i} ack failed`);
    const tCur = latestOfType('telemetry');
    if (i >= 2) wrapStrikeAges.push(tCur ? tCur.lastStrikeAgeFrames : -99);
    if (i === 2) {
      tAtMax = tCur;
      await fakeSelf.onmessage({ data: { type: 'export_telemetry', format: 'json' } });
      expAtMax = latestOfType('telemetry_export_data');
    } else if (i === 3) {
      tAtZero = tCur;
      gapAtZero = vm.runInContext('ringMetaView[(wave1TelemView[2] >>> 0) * 24 + 21] >>> 0', sandbox);
      await fakeSelf.onmessage({ data: { type: 'export_telemetry', format: 'json' } });
      expAtZero = latestOfType('telemetry_export_data');
    }
  }
  const rowsMax = new Float64Array(expAtMax.buffer);
  const lastRowMax = rowsMax[(expAtMax.rows - 1) * expAtMax.cols];
  assert(
    tAtMax &&
      tAtMax.frameIdx === 0xffffffff &&
      tAtMax.chronos1000.causalFrameIdx === 0xffffffff &&
      tAtMax.chronos1000.displayedFrameIdx === 0xfffffffd &&
      tAtMax.chronos1000.lookaheadSource === 'source_pts' &&
      lastRowMax === 0xffffffff &&
      expAtMax.meta.chronos1000.causalFrameIdx === 0xffffffff &&
      expAtMax.meta.chronos1000.displayedFrameIdx === 0xfffffffd &&
      expAtMax.meta.chronos1000.lookaheadSource === 'source_pts',
    `Worker failed u32::MAX exact telemetry/export: ${JSON.stringify({ tMax: tAtMax?.chronos1000, lastRowMax, expMax: expAtMax?.meta?.chronos1000 })}`
  );
  const rowsWrap = new Float64Array(expAtZero.buffer);
  const lastRowWrap = rowsWrap[(expAtZero.rows - 1) * expAtZero.cols];
  assert(
    tAtZero &&
      tAtZero.frameIdx === 0 &&
      tAtZero.chronos1000.causalFrameIdx === 0 &&
      tAtZero.chronos1000.displayedFrameIdx === 0xfffffffe &&
      tAtZero.chronos1000.lookaheadSource === 'source_pts' &&
      (gapAtZero & 0x40) !== 0 &&
      lastRowWrap === 0 &&
      expAtZero.meta.chronos1000.causalFrameIdx === 0 &&
      expAtZero.meta.chronos1000.displayedFrameIdx === 0xfffffffe &&
      expAtZero.meta.chronos1000.lookaheadSource === 'source_pts',
    `Worker failed post-u32::MAX wrap telemetry/export: ${JSON.stringify({ tWrap: tAtZero?.chronos1000, gapAtZero, lastRowWrap, expWrap: expAtZero?.meta?.chronos1000 })}`
  );
  assert(wrapStrikeAges.join(',') === '0,0,1,2', `u32::MAX leadFrames=1 strike age must transition 0->0->1->2 across wrap, got ${JSON.stringify(wrapStrikeAges)}`);
  // Also test wrap-safe scatterAgeFrames on fIdxNow=0xffffffff / 0 / 1 / 2 with scatterFrame = 0xffffffff
  const scAges = vm.runInContext(`(() => {
    const out = [];
    const lastSlot = wave1TelemView[2] >>> 0;
    const savedF = ringMetaView[lastSlot * RING_META_STRIDE + 3];
    const savedSc = lastScatterFrame;
    lastScatterFrame = 0xffffffff;
    for (const f of [0xfffffffe, 0xffffffff, 0, 1, 2]) {
      ringMetaView[lastSlot * RING_META_STRIDE + 3] = f >>> 0;
      const d = ((f >>> 0) - (lastScatterFrame >>> 0)) >>> 0;
      out.push(d < 0x80000000 ? d : 0);
    }
    ringMetaView[lastSlot * RING_META_STRIDE + 3] = savedF;
    lastScatterFrame = savedSc;
    return out;
  })()`, sandbox);
  assert(scAges.join(',') === '0,0,1,2,3', `scatterAgeFrames wrap math mismatch: ${JSON.stringify(scAges)}`);
  await fakeSelf.onmessage({ data: { type: 'rec_control', action: 'clear' } });

  for (const id of pendingTimers) clearTimeout(id);
  pendingTimers.clear();
  void prevSlot0Off;
}

function runGateI4Workload(wasmBytes, hdTex, budgetScale, flagsBase, hdBytes) {
  const g4 = instantiateWasm(wasmBytes);
  const g4e = g4.exp;
  const g4v = createViews(g4e, g4.mem);
  g4e.chronos_init();
  g4e.chronos_set_resolution(1);
  g4e.chronos_set_params(0, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
  g4e.chronos_set_swarm_mode(2);
  g4e.chronos_set_micro_metrology_mode(2);
  g4e.chronos_set_bos_mode(1);
  g4e.chronos_set_bos_gauge_roi(8, 8, 48, 120);

  const g4Ring0Ptr = g4e.chronos_get_ring_slot_ptr(0);
  const hdRing = Array.from({ length: RING_LEN }, (_, s) => new Uint8Array(g4.mem.buffer, g4e.chronos_get_ring_slot_ptr(s), hdBytes));
  const scratchFrame = new Uint8Array(hdBytes);
  const warmTimes = [];
  let heldGradSeen = 0;
  let heldHeroPhysSeen = 0;
  let heldFtleSeen = 0;
  let freshHeroPhysSeen = 0;
  let bosReadyFrames = 0;
  let bosPcgFrames = 0;
  let pcgConvergedFrames = 0;
  let bosDisturbedFrames = 0;
  let confMicroSum = 0;
  let prevCommits = 0;
  const swarmTelem = new Float32Array(g4.mem.buffer, g4e.chronos_get_swarm_telemetry_ptr(), 64);

  const FRAMES = 1000;
  for (let f = 0; f < FRAMES; f++) {
    paintTexturedSwarmAndHero(scratchFrame, hdTex, 640, 360, f, 1120);
    if (f >= 20 && (f % 40) < 6) {
      let activeTid = 0;
      for (let m = 0; m < 16; m++) {
        if (g4v.telem[32 + m * 56] > 0.5) {
          activeTid = Math.round(g4v.telem[32 + m * 56 + 1]);
          break;
        }
      }
      if (activeTid > 0) g4e.chronos_set_focus_target(activeTid);
    } else if (f >= 20 && (f % 40) === 6) {
      g4e.chronos_set_focus_target(0);
    }
    const t0 = performance.now();
    const slotPtr = g4e.chronos_get_ingest_rgba_ptr();
    const slotIdx = Math.round((slotPtr - g4Ring0Ptr) / FULL_RING_SLOT_BYTES);
    assert(slotIdx >= 0 && slotIdx < RING_LEN && hdRing[slotIdx].byteOffset === slotPtr, `Invalid HD ingest ptr at f=${f}`);

    const seq = f + 1;
    const tUs = Math.round(f * (1_000_000 / 30));
    const tLo = tUs >>> 0;
    const tHi = Math.floor(tUs / 4294967296) >>> 0;
    assert(
      g4e.chronos_set_ingest_meta(77, seq, tLo, tHi, 1.25, flagsBase | 4, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1) === 1,
      `HD stage failed at f=${f}`
    );
    hdRing[slotIdx].set(scratchFrame);

    g4e.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(g4e, g4v.lap, g4v.fiedler);
    g4e.chronos_post_tick(1 / 30);

    const commitsNow = (g4v.w1[5] >>> 0) * 4294967296 + (g4v.w1[4] >>> 0);
    assert(commitsNow === prevCommits + 1, `100% ring ingest failed at f=${f}: ${prevCommits} -> ${commitsNow}`);
    prevCommits = commitsNow;
    const ms = performance.now() - t0;
    if (f >= 20) warmTimes.push(ms);

    assert(g4v.w1[8] === 1 && g4v.w1[10] === 360, `Bad Wave-1 status/rows at f=${f}`);
    if ((g4v.w1[24] & STAMP_VALID) !== 0 && (g4v.w1[24] & STAMP_HELD) !== 0) heldGradSeen++;
    if ((g4v.w1[32] & STAMP_VALID) !== 0 && (g4v.w1[32] & STAMP_HELD) !== 0) heldFtleSeen++;
    if ((g4v.w1[48] & STAMP_VALID) !== 0) {
      if ((g4v.w1[48] & STAMP_HELD) !== 0) heldHeroPhysSeen++;
      else freshHeroPhysSeen++;
    }
    if (g4v.bosTelem[2] === 2) bosReadyFrames++;
    if (g4v.bosTelem[2] === 3) bosDisturbedFrames++;
    if (g4v.bosTelem[2] === 2 && g4v.bosTelem[9] === 2) bosPcgFrames++;
    if (g4v.bosTelem[2] === 2 && g4v.bosTelem[9] === 2 && (g4v.bosTelem[3] & 64) === 0 && g4v.bosTelem[7] > 0 && g4v.bosF32[16] <= 1e-6) {
      pcgConvergedFrames++;
    }
    confMicroSum += swarmTelem[4];
  }

  assert(bosReadyFrames >= 980, `Gate I4 BOS must be READY >= 980/1000 frames, got ${bosReadyFrames} (disturbed=${bosDisturbedFrames})`);
  assert(bosPcgFrames >= 900, `Gate I4 must run real masked PCG BOS solves (>= 900 frames solver_path=2), got ${bosPcgFrames}`);
  assert(pcgConvergedFrames >= 980 && pcgConvergedFrames === bosPcgFrames, `Gate I4 BOS PCG must converge cleanly >= 980 & match bosPcgFrames: ${pcgConvergedFrames}/${bosPcgFrames}`);
  assert(freshHeroPhysSeen >= 10 && heldHeroPhysSeen >= 1, `Gate I4 must compute Hero physics (${freshHeroPhysSeen} fresh, ${heldHeroPhysSeen} held)`);
  const meanConfMicro = confMicroSum / FRAMES;
  assert(meanConfMicro >= 900, `Gate I4 must track >= 900 confirmed swarm targets, got ${meanConfMicro.toFixed(1)}`);

  warmTimes.sort((a, b) => a - b);
  const medMs = warmTimes[Math.floor(warmTimes.length * 0.5)];
  const p95Ms = warmTimes[Math.floor(warmTimes.length * 0.95)];
  const maxMs = warmTimes[warmTimes.length - 1];
  const budgetMs = 33.333 * budgetScale;
  assert(maxMs <= budgetMs, `Gate I4 max frame time ${maxMs.toFixed(2)} ms > ${budgetMs.toFixed(2)} ms`);
  return {
    medMs,
    p95Ms,
    maxMs,
    budgetMs,
    bosReadyFrames,
    bosPcgFrames,
    pcgConvergedFrames,
    meanConfMicro,
    heldGradSeen,
    heldFtleSeen,
    heldHeroPhysSeen,
    freshHeroPhysSeen,
    prevCommits,
  };
}

async function main() {
  const budgetScale = parseBudgetScale();
  const isPartial = budgetScale > 1.0;
  const wasmBytes = fs.readFileSync(WASM_PATH);
  const { exp, mem } = instantiateWasm(wasmBytes);

  const requiredWave1 = [
    'chronos_set_ingest_meta',
    'chronos_get_ring_meta_ptr',
    'chronos_get_row_luma_ptr',
    'chronos_get_wave1_telemetry_ptr',
  ];
  const requiredWave2 = [
    'chronos_get_wave2_telemetry_ptr',
    'chronos_set_rs_calib',
    'chronos_get_audio_staging_ptr',
    'chronos_push_audio_block',
    'chronos_get_vibro_telemetry_ptr',
    'chronos_get_vibro_planes_ptr',
    'chronos_set_hodge_mode',
    'chronos_get_hodge_telemetry_ptr',
    'chronos_get_hodge_planes_ptr',
    'chronos_get_hodge_inject_ptr',
    'chronos_hodge_solve_injected',
    'chronos_get_conservation_telemetry_ptr',
    'chronos_get_conservation_input_ptr',
    'chronos_eval_occluded_pair',
    'chronos_seed_target',
    'chronos_get_fovea_rgba_ptr',
    'chronos_set_fovea_meta',
    'chronos_get_fovea_request_ptr',
    'chronos_set_depth_mode',
    'chronos_get_depth_mode',
    'chronos_set_depth_calib',
    'chronos_depth_capture_stereo_left',
    'chronos_set_depth_external_meta',
    'chronos_get_depth_telemetry_ptr',
    'chronos_get_depth_field_ptr',
    'chronos_get_depth_tracks_ptr',
    'chronos_get_depth_track_flags_ptr',
    'chronos_get_depth_hero_ptr',
    'chronos_get_depth_ingest_ptr',
  ];
  const exportNames = Object.keys(exp);
  assert(exportNames.length === 91, `Expected exact pinned Wasm export count 91, got ${exportNames.length}`);
  assert(typeof exp.chronos_get_micro_metrology_mode === 'undefined', 'Unpinned export chronos_get_micro_metrology_mode must not exist');
  for (const name of [...requiredWave1, ...requiredWave2]) {
    assert(typeof exp[name] === 'function', `Missing required export: ${name}`);
  }

  // Run Gate I4 Workload FIRST before multi-instance Wasm allocations to eliminate munmap/GC jitter spikes
  const flagsBase = 2 | 16; // PTS known + Geom known
  const hdBytes = 640 * 360 * 4;
  const hdTex = buildTexturedBosBackdrop(640, 360, true);
  const g4Res = runGateI4Workload(wasmBytes, hdTex, budgetScale, flagsBase, hdBytes);

  // 1. Check SD pointer disjointness including ALL 12 full 921,600 B rgba_ring slots & Wave-1 sidecars
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  const v = createViews(exp, mem);
  assert(v.w1[2] === 0xffffffff && v.w1[3] === 0xffffffff, `chronos_init must set w1[2..3]=u32::MAX, got (${v.w1[2]}, ${v.w1[3]})`);
  for (let s = 0; s < RING_LEN; s++) {
    assert(v.ringMeta[s * RING_META_STRIDE] === 0, `init ringMeta[${s}].flags must be 0`);
  }
  const arenaBytes = v.w1[1] >>> 0;
  assert(arenaBytes > 0 && arenaBytes <= MAX_ARENA_BYTES, `Arena ${arenaBytes} > ${MAX_ARENA_BYTES}`);
  assert(v.w1[0] === 1 && v.w1[11] === 24, `Expected Wave-1 ABI=1, stride=24, got (${v.w1[0]}, ${v.w1[11]})`);
  const ring0Ptr = exp.chronos_get_ring_slot_ptr(0);
  const linkedStaticEnd = v.w1[12] >>> 0;
  const dualStaticEnd = Math.max(ring0Ptr + arenaBytes, linkedStaticEnd);
  assert(dualStaticEnd <= WAVE1_MAX_STATIC_END, `Dual static-end breach: ${dualStaticEnd} > ${WAVE1_MAX_STATIC_END}`);

  const ranges = [];
  for (let s = 0; s < RING_LEN; s++) {
    const p = exp.chronos_get_ring_slot_ptr(s);
    assert((p & 3) === 0, `ring[${s}] unaligned`);
    ranges.push({ name: `ring_${s}`, start: p, end: p + FULL_RING_SLOT_BYTES });
  }
  ranges.push({ name: 'ring_meta', start: exp.chronos_get_ring_meta_ptr(), end: exp.chronos_get_ring_meta_ptr() + RING_LEN * 24 * 4 });
  ranges.push({ name: 'row_luma', start: exp.chronos_get_row_luma_ptr(), end: exp.chronos_get_row_luma_ptr() + RING_LEN * 360 * 4 });
  ranges.push({ name: 'wave1_telem', start: exp.chronos_get_wave1_telemetry_ptr(), end: exp.chronos_get_wave1_telemetry_ptr() + 192 * 4 });
  if (typeof exp.chronos_get_bos_planes_ptr === 'function') {
    ranges.push({ name: 'bos_planes', start: exp.chronos_get_bos_planes_ptr(), end: exp.chronos_get_bos_planes_ptr() + 6 * BOS_N * 4 });
  }
  if (typeof exp.chronos_get_micro_cov_ptr === 'function') {
    ranges.push({ name: 'micro_cov', start: exp.chronos_get_micro_cov_ptr(), end: exp.chronos_get_micro_cov_ptr() + 3 * 3072 * 4 });
  }
  ranges.push({ name: 'wave2_telem', start: exp.chronos_get_wave2_telemetry_ptr(), end: exp.chronos_get_wave2_telemetry_ptr() + 64 * 4 });
  ranges.push({ name: 'audio_staging', start: exp.chronos_get_audio_staging_ptr(), end: exp.chronos_get_audio_staging_ptr() + 4096 * 4 });
  ranges.push({ name: 'vibro_telem', start: exp.chronos_get_vibro_telemetry_ptr(), end: exp.chronos_get_vibro_telemetry_ptr() + 32 * 4 });
  ranges.push({ name: 'vibro_planes', start: exp.chronos_get_vibro_planes_ptr(), end: exp.chronos_get_vibro_planes_ptr() + (3 * 2304 + 128) * 4 });
  ranges.push({ name: 'hodge_telem', start: exp.chronos_get_hodge_telemetry_ptr(), end: exp.chronos_get_hodge_telemetry_ptr() + 32 * 4 });
  ranges.push({ name: 'hodge_planes', start: exp.chronos_get_hodge_planes_ptr(), end: exp.chronos_get_hodge_planes_ptr() + 7 * 2304 * 4 });
  ranges.push({ name: 'hodge_inject', start: exp.chronos_get_hodge_inject_ptr(), end: exp.chronos_get_hodge_inject_ptr() + 3 * 2304 * 4 });
  ranges.push({ name: 'conservation_telem', start: exp.chronos_get_conservation_telemetry_ptr(), end: exp.chronos_get_conservation_telemetry_ptr() + 32 * 4 });
  ranges.push({ name: 'conservation_input', start: exp.chronos_get_conservation_input_ptr(), end: exp.chronos_get_conservation_input_ptr() + 32 * 4 });
  ranges.push({ name: 'depth_telem', start: exp.chronos_get_depth_telemetry_ptr(), end: exp.chronos_get_depth_telemetry_ptr() + 32 * 4 });
  ranges.push({ name: 'depth_field', start: exp.chronos_get_depth_field_ptr(), end: exp.chronos_get_depth_field_ptr() + 64 * 36 * 2 * 4 });
  ranges.push({ name: 'depth_tracks', start: exp.chronos_get_depth_tracks_ptr(), end: exp.chronos_get_depth_tracks_ptr() + 3072 * 2 * 4 });
  ranges.push({ name: 'depth_track_flags', start: exp.chronos_get_depth_track_flags_ptr(), end: exp.chronos_get_depth_track_flags_ptr() + 3072 });
  ranges.push({ name: 'depth_hero', start: exp.chronos_get_depth_hero_ptr(), end: exp.chronos_get_depth_hero_ptr() + 16 * 4 * 4 });
  ranges.push({ name: 'depth_ingest', start: exp.chronos_get_depth_ingest_ptr(), end: exp.chronos_get_depth_ingest_ptr() + 128 * 64 * 4 });
  ranges.sort((a, b) => a.start - b.start);
  for (let i = 1; i < ranges.length; i++) {
    assert(ranges[i - 1].end <= ranges[i].start, `Overlap ${ranges[i - 1].name} vs ${ranges[i].name}`);
  }

  // 2. Fail-closed chronos_set_ingest_meta matrix + atomic phase guards
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, -1.0, flagsBase, 0, 180, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 0) === 0 && v.w1[8] === 2, 'Reject src_w=0');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, -1.0, flagsBase, 640, 360, 0xfffffff0, 0, 32, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 0) === 0 && v.w1[8] === 2, 'Reject overflow x0+cw');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 320, 180, 8, 0xffffffff, 0, 0, 320, 180, 0, 0) === 0 && v.w1[8] === 2, 'Reject orientation > 7');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 320, 180, 0, 4, 0, 0, 320, 180, 0, 0) === 0 && v.w1[8] === 2, 'Reject scan_dir=4');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, Number.NaN, flagsBase | 4, 640, 360, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 0) === 0 && v.w1[8] === 2, 'Reject NaN age');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, 2.0, flagsBase | 12, 640, 360, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 0) === 0 && v.w1[8] === 2, 'Reject capture+handoff bit conflict');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, 1.5, flagsBase, 640, 360, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 0) === 0 && v.w1[8] === 2, 'Reject age!=-1 when age flags 0');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, -1.0, flagsBase | 1, 640, 360, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 0) === 0 && v.w1[8] === 2, 'Reject host setting committed bit0');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0x40, 0) === 0 && v.w1[8] === 2, 'Reject host gap_mask=0x40 (kernel wrap bit)');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0x80, 0) === 0 && v.w1[8] === 2, 'Reject host gap_mask=0x80');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 321, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 1) === 0 && v.w1[8] === 2, 'Reject native 1:1 crop mismatch');
  assert(exp.chronos_set_ingest_meta(1, 0, 0, 0, -1.0, flagsBase, 640, 360, 10, 10, 180, 320, 1, 0xffffffff, 0, 0, 320, 180, 0, 1) === 1, 'Accept rotated 90-deg native 1:1 (180x320 -> 320x180)');

  // Orphan post_tick MUST set PHASE_ERR=4 without mutating frame_idx, commits, or valid stamps
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  exp.chronos_set_swarm_mode(2);
  exp.chronos_set_micro_metrology_mode(2);
  assert(exp.chronos_set_ingest_meta(9, 1, 0, 0, -1.0, flagsBase, 320, 180, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 1) === 1, 'Seed valid stamp before orphan post_tick');
  exp.chronos_pre_tick(1 / 30);
  solveFiedlerLanczos(exp, v.lap, v.fiedler);
  exp.chronos_post_tick(1 / 30);
  assert((v.w1[16] & STAMP_VALID) !== 0 && v.w1[4] === 1, 'Seed frame must commit micro stamp before orphan check');
  exp.chronos_post_tick(1 / 30);
  assert(v.w1[8] === 4 && v.w1[4] === 1 && v.telem[0] === 0.0, `Orphan post_tick mutated Arena: w1[8]=${v.w1[8]}, commits=${v.w1[4]}, f=${v.telem[0]}`);
  assert((v.w1[16] & STAMP_VALID) !== 0, 'Orphan post_tick phase error must be status-only without invalidating valid stamps');

  // Double pre_tick + open chronos_tick guard + deferred commit until post_tick + reset_tracks clear
  const sdBytes = 320 * 180 * 4;
  const sdPattern = new Uint8Array(sdBytes);
  for (let y = 0; y < 180; y++) {
    const val = Math.min(255, Math.round((y / 179) * 255));
    for (let x = 0; x < 320; x++) {
      const o = (y * 320 + x) * 4;
      sdPattern[o] = val;
      sdPattern[o + 1] = val;
      sdPattern[o + 2] = val;
      sdPattern[o + 3] = 255;
    }
  }
  exp.chronos_reset_tracks();
  assert(v.w1[2] === 0xffffffff && v.w1[3] === 0xffffffff && v.w1[4] === 1, `chronos_reset_tracks must set w1[2..3]=u32::MAX and preserve commits=1, got (${v.w1[2]}, ${v.w1[3]}, ${v.w1[4]})`);
  for (let s = 0; s < RING_LEN; s++) {
    assert(v.ringMeta[s * RING_META_STRIDE] === 0, `reset_tracks must clear ringMeta[${s}].flags`);
  }
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  assert(v.w1[2] === 0xffffffff && v.w1[3] === 0xffffffff, 'init must reset w1[2..3]=u32::MAX');
  new Uint8Array(mem.buffer, exp.chronos_get_ingest_rgba_ptr(), sdBytes).set(sdPattern);
  assert(exp.chronos_set_ingest_meta(10, 0, 0xfffffe00, 0, 4.25, flagsBase | 8, 640, 360, 100, 50, 320, 180, 0, 2, 0, 0, 320, 180, 0, 1) === 1, 'Stage frame 0');
  exp.chronos_pre_tick(1 / 30);
  assert(v.w1[8] === 1 && v.w1[4] === 1 && (v.ringMeta[0] & 1) === 1 && v.w1[13] === 1, `Pre-solver pre_tick must commit ring/OK/w1[13]=1, got w1[8]=${v.w1[8]}, commits=${v.w1[4]}, flags=${v.ringMeta[0]}, split=${v.w1[13]}`);
  const openTickRet = exp.chronos_tick(1 / 30);
  assert(openTickRet === 4 && v.w1[8] === 4 && v.w1[4] === 1 && v.w1[13] === 1, `chronos_tick while CTRL_TICK_PHASE!=0 must return PHASE_ERR=4 without corrupting commits: ret=${openTickRet}, w1[8]=${v.w1[8]}, commits=${v.w1[4]}`);
  exp.chronos_pre_tick(1 / 30); // Double pre_tick!
  assert(v.w1[8] === 4 && v.w1[4] === 1 && v.ringMeta[3] === 0 && v.w1[13] === 1, `Double pre_tick must set PHASE_ERR=4 without mutating commits=${v.w1[4]} or ringMeta[3]=${v.ringMeta[3]}`);
  assert(exp.chronos_set_ingest_meta(10, 1, 0x00008035, 1, -1.0, flagsBase, 640, 360, 100, 50, 320, 180, 0, 2, 0, 0, 320, 180, 0, 1) === 0 && v.w1[8] === 4, 'Reject mid-tick set_ingest_meta');
  solveFiedlerLanczos(exp, v.lap, v.fiedler);
  exp.chronos_post_tick(1 / 30);
  assert(v.w1[13] === 0 && v.w1[8] === 1 && v.w1[4] === 1 && v.w1[10] === 180 && (v.ringMeta[0] & 1) === 1 && v.ringMeta[22] === 1, `Frame 0 post_tick must clear w1[13]=0 and restore OK status=1 without double-incrementing commits: w1[13]=${v.w1[13]}, w1[8]=${v.w1[8]}, commits=${v.w1[4]}`);
  assert(Math.abs(v.ringMetaF32[6] - 4.25) < 1e-4, `age_ms mismatch ${v.ringMetaF32[6]}`);
  assert(v.rowLuma[0] < 0.01 && Math.abs(v.rowLuma[179] - 1.0) < 0.01, `SD row_luma gradient bad: [0]=${v.rowLuma[0]}, [179]=${v.rowLuma[179]}`);
  for (let r = 180; r < 360; r++) {
    assert(v.rowLuma[r] === 0.0, `SD trailing row ${r} must be 0.0, got ${v.rowLuma[r]}`);
  }
  // Duplicate (10, 0) rejected with status=3
  assert(exp.chronos_set_ingest_meta(10, 0, 0x00008035, 1, -1.0, flagsBase, 640, 360, 100, 50, 320, 180, 0, 2, 0, 0, 320, 180, 0, 1) === 0 && v.w1[8] === 3, 'Reject duplicate (10, 0)');

  // 3. t_src_us_lo -> hi 32-bit carry + Authoritative staged meta vs conflicting legacy PTS + Equal-PTS commit + Isolated gaps + BOS frozen ref!
  // Step seq=1 across u32 low wrap (0xfffffe00, 0) -> (0x00008035, 1) = exactly +33,333 us, while calling conflicting chronos_set_ingest_pts(5.0)!
  const epCarry = v.w1[6];
  new Uint8Array(mem.buffer, exp.chronos_get_ingest_rgba_ptr(), sdBytes).set(sdPattern);
  assert(exp.chronos_set_ingest_meta(10, 1, 0x00008035, 1, -1.0, flagsBase, 640, 360, 100, 50, 320, 180, 0, 2, 0, 0, 320, 180, 0, 1) === 1, 'Stage 32-bit carry frame');
  exp.chronos_set_ingest_pts(5.0); // MUST be ignored because staged IngestSlotMeta is authoritative!
  exp.chronos_pre_tick(1 / 30);
  solveFiedlerLanczos(exp, v.lap, v.fiedler);
  exp.chronos_post_tick(1 / 30);
  assert(v.w1[6] === epCarry && v.ringMeta[RING_META_STRIDE + 21] === 0, `32-bit lo->hi carry must keep continuity_epoch=${epCarry} (got ${v.w1[6]}) & gap=0 (got ${v.ringMeta[RING_META_STRIDE + 21]})`);
  assert(v.ringMeta[RING_META_STRIDE + 4] === 0x00008035 && v.ringMeta[RING_META_STRIDE + 5] === 1, 'ringMeta must keep staged u64 PTS carry');

  // Distinct equal-PTS frame (seq=2, same t_us=(0x00008035, 1)) MUST commit and bump epoch with NONINC_PTS
  const epBeforeEq = v.w1[6];
  const commitsBeforeEq = v.w1[4];
  new Uint8Array(mem.buffer, exp.chronos_get_ingest_rgba_ptr(), sdBytes).set(sdPattern);
  assert(exp.chronos_set_ingest_meta(10, 2, 0x00008035, 1, -1.0, flagsBase, 640, 360, 100, 50, 320, 180, 0, 2, 0, 0, 320, 180, 0, 1) === 1, 'Stage equal-PTS distinct frame');
  exp.chronos_pre_tick(1 / 30);
  solveFiedlerLanczos(exp, v.lap, v.fiedler);
  exp.chronos_post_tick(1 / 30);
  assert(v.w1[4] === commitsBeforeEq + 1, 'Distinct equal-PTS frame must commit');
  assert((v.ringMeta[2 * RING_META_STRIDE + 21] & GAP_NONINC_PTS) !== 0 && v.w1[6] === epBeforeEq + 1, 'Distinct equal-PTS must emit NONINC_PTS and bump epoch');

  // Test 48-frame SD & 48-frame HD 12-wrap + isolated GAP_SEQ, GAP_EXCESS_DT preserving frozen BOS ref + GAP_SEEK_RESTART resetting BOS
  const sdTex = buildTexturedBosBackdrop(320, 180, false);
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  const sdHits = new Int32Array(RING_LEN);
  let tUsBig = 1_700_000_000_000_000n; // ~1.7e15 us Unix-epoch magnitude
  for (let f = 0; f < 48; f++) {
    const slotPtr = exp.chronos_get_ingest_rgba_ptr();
    const slotIdx = Math.round((slotPtr - ring0Ptr) / FULL_RING_SLOT_BYTES);
    sdHits[slotIdx]++;
    new Uint8Array(mem.buffer, slotPtr, sdBytes).set(sdTex);
    const tLo = Number(tUsBig & 0xffffffffn) >>> 0;
    const tHi = Number((tUsBig >> 32n) & 0xffffffffn) >>> 0;
    assert(exp.chronos_set_ingest_meta(20, f + 1, tLo, tHi, -1.0, flagsBase, 320, 180, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 1) === 1, `SD 48f stage f=${f}`);
    exp.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(exp, v.lap, v.fiedler);
    exp.chronos_post_tick(1 / 30);
    tUsBig += 33333n;
  }
  for (let s = 0; s < RING_LEN; s++) {
    assert(sdHits[s] === 4, `SD wrap slot ${s} expected 4 visits, got ${sdHits[s]}`);
  }
  assert(v.w1[6] === 1, `Unix-epoch 1.7e15 us u64 rebase must preserve epoch=1 over 48 continuous frames, got ${v.w1[6]}`);

  // HD 48f + BOS READY + Isolated SEQ & EXCESS_DT preserving frozen BOS ref
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  exp.chronos_set_bos_mode(1);
  exp.chronos_set_bos_gauge_roi(8, 8, 48, 120);
  const hdHits = new Int32Array(RING_LEN);
  let seqHd = 100;
  let tHdUs = 0;
  for (let f = 0; f < 48; f++) {
    const ptr = exp.chronos_get_ingest_rgba_ptr();
    const slotIdx = Math.round((ptr - ring0Ptr) / FULL_RING_SLOT_BYTES);
    hdHits[slotIdx]++;
    new Uint8Array(mem.buffer, ptr, hdBytes).set(hdTex);
    assert(exp.chronos_set_ingest_meta(30, seqHd, tHdUs >>> 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1) === 1, `HD stage f=${f}`);
    exp.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(exp, v.lap, v.fiedler);
    exp.chronos_post_tick(1 / 30);
    seqHd++;
    tHdUs += 33333;
  }
  for (let s = 0; s < RING_LEN; s++) {
    assert(hdHits[s] === 4, `HD wrap slot ${s} expected 4 visits, got ${hdHits[s]}`);
  }
  assert(v.bosTelem && v.bosTelem[2] === 2 && v.bosTelem[10] === 16 && v.w1[6] === 1, `Expected BOS READY (warmup=16) before gap tests: state=${v.bosTelem?.[2]}, warm=${v.bosTelem?.[10]}, ep=${v.w1[6]}`);

  // Inject isolated FRAME_GAP_SEQ (seq skips +1, same dt=33333)
  seqHd += 1; // skip!
  {
    const ptr = exp.chronos_get_ingest_rgba_ptr();
    const slotIdx = Math.round((ptr - ring0Ptr) / FULL_RING_SLOT_BYTES);
    new Uint8Array(mem.buffer, ptr, hdBytes).set(hdTex);
    assert(exp.chronos_set_ingest_meta(30, seqHd, tHdUs >>> 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1) === 1, 'Stage isolated SEQ gap');
    exp.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(exp, v.lap, v.fiedler);
    exp.chronos_post_tick(1 / 30);
    const mOff = slotIdx * RING_META_STRIDE;
    assert(v.ringMeta[mOff + 21] === GAP_SEQ, `Expected isolated GAP_SEQ=1, got 0x${v.ringMeta[mOff + 21].toString(16)}`);
    assert(v.w1[6] === 2, `Isolated GAP_SEQ must advance continuity_epoch 1 -> 2, got ${v.w1[6]}`);
    assert(v.bosTelem[2] === 2 && v.bosTelem[10] === 16, 'Timing/seq-only GAP_SEQ must preserve frozen BOS READY ref!');
    seqHd++;
    tHdUs += 80000; // next step: EXCESS_DT (> 1.5 * 33333)
  }

  // Inject isolated FRAME_GAP_EXCESS_DT (contiguous seq, dt=80000 us)
  {
    const ptr = exp.chronos_get_ingest_rgba_ptr();
    const slotIdx = Math.round((ptr - ring0Ptr) / FULL_RING_SLOT_BYTES);
    new Uint8Array(mem.buffer, ptr, hdBytes).set(hdTex);
    assert(exp.chronos_set_ingest_meta(30, seqHd, tHdUs >>> 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1) === 1, 'Stage isolated EXCESS_DT gap');
    exp.chronos_pre_tick(0.08);
    solveFiedlerLanczos(exp, v.lap, v.fiedler);
    exp.chronos_post_tick(0.08);
    const mOff = slotIdx * RING_META_STRIDE;
    assert(v.ringMeta[mOff + 21] === GAP_EXCESS_DT, `Expected isolated GAP_EXCESS_DT=4, got 0x${v.ringMeta[mOff + 21].toString(16)}`);
    assert(v.w1[6] === 3, `Isolated GAP_EXCESS_DT must advance continuity_epoch 2 -> 3, got ${v.w1[6]}`);
    assert(v.bosTelem[2] === 2 && v.bosTelem[10] === 16, 'Timing-only GAP_EXCESS_DT must preserve frozen BOS READY ref!');
    seqHd++;
    tHdUs += 33333;
  }

  // source_seq strict wrap adjacency: 0xfffffffe -> 1 MUST emit GAP_SEQ (0x01) without GAP_WRAP (0x40),
  // whereas exact 0xffffffff -> 0 yields gap=0 and preserves continuity_epoch
  {
    const ptrPre = exp.chronos_get_ingest_rgba_ptr();
    new Uint8Array(mem.buffer, ptrPre, hdBytes).set(hdTex);
    exp.chronos_set_ingest_meta(30, 0xfffffffe, tHdUs >>> 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1);
    exp.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(exp, v.lap, v.fiedler);
    exp.chronos_post_tick(1 / 30);
    tHdUs += 33333;

    const ptrSkip = exp.chronos_get_ingest_rgba_ptr();
    const slotSkip = Math.round((ptrSkip - ring0Ptr) / FULL_RING_SLOT_BYTES);
    new Uint8Array(mem.buffer, ptrSkip, hdBytes).set(hdTex);
    exp.chronos_set_ingest_meta(30, 1, tHdUs >>> 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1);
    exp.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(exp, v.lap, v.fiedler);
    exp.chronos_post_tick(1 / 30);
    const skipGap = v.ringMeta[slotSkip * RING_META_STRIDE + 21];
    assert((skipGap & GAP_SEQ) !== 0 && (skipGap & GAP_WRAP) === 0, `source_seq 0xfffffffe->1 must emit GAP_SEQ (0x01) without GAP_WRAP (0x40), got 0x${skipGap.toString(16)}`);
    tHdUs += 33333;

    const ptrA = exp.chronos_get_ingest_rgba_ptr();
    new Uint8Array(mem.buffer, ptrA, hdBytes).set(hdTex);
    exp.chronos_set_ingest_meta(30, 0xffffffff, tHdUs >>> 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1);
    exp.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(exp, v.lap, v.fiedler);
    exp.chronos_post_tick(1 / 30);
    const epBeforeSrcWrap = v.w1[6];
    tHdUs += 33333;

    const ptrB = exp.chronos_get_ingest_rgba_ptr();
    const slotB = Math.round((ptrB - ring0Ptr) / FULL_RING_SLOT_BYTES);
    new Uint8Array(mem.buffer, ptrB, hdBytes).set(hdTex);
    exp.chronos_set_ingest_meta(30, 0, tHdUs >>> 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1);
    exp.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(exp, v.lap, v.fiedler);
    exp.chronos_post_tick(1 / 30);
    const mOffB = slotB * RING_META_STRIDE;
    assert(v.ringMeta[mOffB + 21] === 0 && v.w1[6] === epBeforeSrcWrap, `source_seq 0xffffffff->0 must yield gap=0 and preserve continuity_epoch=${epBeforeSrcWrap} (got gap=0x${v.ringMeta[mOffB + 21].toString(16)}, ep=${v.w1[6]})`);
    tHdUs += 33333;
  }

  // Seek/restart (epoch 30 -> 31) resets BOS
  {
    const ptr = exp.chronos_get_ingest_rgba_ptr();
    const slotIdx = Math.round((ptr - ring0Ptr) / FULL_RING_SLOT_BYTES);
    new Uint8Array(mem.buffer, ptr, hdBytes).set(hdTex);
    exp.chronos_set_ingest_meta(31, 0, tHdUs >>> 0, 0, -1.0, flagsBase, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1);
    exp.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(exp, v.lap, v.fiedler);
    exp.chronos_post_tick(1 / 30);
    const mOff = slotIdx * RING_META_STRIDE;
    assert((v.ringMeta[mOff + 21] & GAP_SEEK_RESTART) !== 0 && v.bosTelem[2] === 3 && v.bosTelem[10] === 0, 'Seek/restart must set GAP_SEEK_RESTART and reset BOS warmup');
  }

  // Sustained FPS step-down (60 fps -> 30 fps on same source_epoch=32 with contiguous seq)
  // relearns CTRL_NOM_DT_US after one GAP_EXCESS_DT transition instead of wedging every frame
  {
    let tRateUs = tHdUs + 33333;
    let rateSeq = 0;
    for (let k = 0; k < 8; k++) {
      const ptr = exp.chronos_get_ingest_rgba_ptr();
      new Uint8Array(mem.buffer, ptr, hdBytes).set(hdTex);
      assert(
        exp.chronos_set_ingest_meta(
          32,
          rateSeq++,
          tRateUs >>> 0,
          Math.floor(tRateUs / 4294967296) >>> 0,
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
        ) === 1,
        'Stage 60fps warmup frame'
      );
      exp.chronos_pre_tick(1 / 60);
      solveFiedlerLanczos(exp, v.lap, v.fiedler);
      exp.chronos_post_tick(1 / 60);
      tRateUs += 16667;
    }
    tRateUs += 16666; // First step-down interval = 33333 us (2x 60fps nominal)
    const epAt60 = v.w1[6];
    for (let k = 0; k < 7; k++) {
      const ptr = exp.chronos_get_ingest_rgba_ptr();
      const slotIdx = Math.round((ptr - ring0Ptr) / FULL_RING_SLOT_BYTES);
      new Uint8Array(mem.buffer, ptr, hdBytes).set(hdTex);
      assert(
        exp.chronos_set_ingest_meta(
          32,
          rateSeq++,
          tRateUs >>> 0,
          Math.floor(tRateUs / 4294967296) >>> 0,
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
        ) === 1,
        'Stage 30fps step-down frame'
      );
      exp.chronos_pre_tick(1 / 30);
      solveFiedlerLanczos(exp, v.lap, v.fiedler);
      exp.chronos_post_tick(1 / 30);
      const mOff = slotIdx * RING_META_STRIDE;
      if (k === 0) {
        assert(v.ringMeta[mOff + 21] === GAP_EXCESS_DT, `First 60->30fps frame must flag GAP_EXCESS_DT=4, got 0x${v.ringMeta[mOff + 21].toString(16)}`);
        assert(v.w1[6] === epAt60 + 1, `First 60->30fps frame must bump continuity_epoch ${epAt60}->${epAt60 + 1}, got ${v.w1[6]}`);
      } else {
        assert(v.ringMeta[mOff + 21] === 0, `Sustained 30fps frame #${k + 1} must relearn nominal dt with gap=0, got 0x${v.ringMeta[mOff + 21].toString(16)}`);
        assert(v.w1[6] === epAt60 + 1, `Sustained 30fps frame #${k + 1} must hold continuity_epoch=${epAt60 + 1}, got ${v.w1[6]}`);
      }
      tRateUs += 33333;
    }
  }

  // 4. Engine counter u32::MAX -> 0 rollover check on isolated Wasm instance with monotonic u64 ring_cursor and K=8
  {
    const iso = instantiateWasm(wasmBytes);
    iso.exp.chronos_init();
    iso.exp.chronos_set_resolution(0);
    const iv = createViews(iso.exp, iso.mem);
    const isoRing0 = iso.exp.chronos_get_ring_slot_ptr(0);
    const arenaU32 = new Uint32Array(iso.mem.buffer, isoRing0, arenaBytes >> 2);
    for (let f = 0; f < 13; f++) {
      iso.exp.chronos_pre_tick(1 / 30);
      solveFiedlerLanczos(iso.exp, iv.lap, iv.fiedler);
      iso.exp.chronos_post_tick(1 / 30);
    }
    const cand13 = [];
    for (let wi = 2; wi < arenaU32.length - 4; wi++) {
      if (arenaU32[wi - 2] === 13 && arenaU32[wi - 1] === 0 && arenaU32[wi] === 0 && arenaU32[wi + 1] === 13 && arenaU32[wi + 3] === 8) cand13.push(wi);
    }
    iso.exp.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(iso.exp, iv.lap, iv.fiedler);
    iso.exp.chronos_post_tick(1 / 30);
    const resWords = cand13.filter((wi) => arenaU32[wi - 2] === 14 && arenaU32[wi - 1] === 0 && arenaU32[wi + 1] === 14);
    assert(resWords.length === 1, `Expected unique Arena (ring_cursor, res_mode, frame_idx), got ${resWords.length}`);
    const rcLoWord = resWords[0] - 2;
    const rcHiWord = resWords[0] - 1;
    const fIdxWord = resWords[0] + 1;
    // Through the test-hook export, not a raw write, so recorded traces of this gate replay natively.
    iso.exp.chronos_test_hook_set_frame_counters(0xfffffffd, 0xfffffffd, 0);
    assert(
      arenaU32[rcLoWord] === 0xfffffffd && arenaU32[rcHiWord] === 0 && arenaU32[fIdxWord] === 0xfffffffd,
      'chronos_test_hook_set_frame_counters must set exactly the located (ring_cursor, frame_idx) words'
    );
    const seenRoll = [];
    for (let k = 0; k < 5; k++) {
      const fCur = arenaU32[fIdxWord] >>> 0;
      const rcCur = (BigInt(arenaU32[rcHiWord] >>> 0) << 32n) | BigInt(arenaU32[rcLoWord] >>> 0);
      const expectedSlot = Number(rcCur % BigInt(RING_LEN));
      const expectedDisp = (expectedSlot + RING_LEN - 8) % RING_LEN;
      const ptr = iso.exp.chronos_get_ingest_rgba_ptr();
      const gotSlot = Math.round((ptr - isoRing0) / FULL_RING_SLOT_BYTES);
      assert(gotSlot === expectedSlot, `Rollover f=0x${fCur.toString(16)} ingest slot mismatch: ${gotSlot} != ${expectedSlot}`);
      assert(iso.exp.chronos_set_ingest_meta(50, k + 1, k * 33333, 0, -1.0, flagsBase, 320, 180, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 1) === 1, 'Rollover stage failed');
      iso.exp.chronos_pre_tick(1 / 30);
      solveFiedlerLanczos(iso.exp, iv.lap, iv.fiedler);
      iso.exp.chronos_post_tick(1 / 30);
      seenRoll.push({
        fCur,
        effK: iv.telem[1],
        slot: iv.w1[2] >>> 0,
        disp: iv.w1[3] >>> 0,
        expectedSlot,
        expectedDisp,
        gap: iv.ringMeta[expectedSlot * RING_META_STRIDE + 21],
        eng: iv.ringMeta[expectedSlot * RING_META_STRIDE + 3],
      });
    }
    assert(seenRoll.map((r) => r.slot).join(',') === '1,2,3,4,5', `Slot must remain continuous 1..5 across u32::MAX->0 wrap, got ${seenRoll.map((r) => r.slot).join(',')}`);
    assert(seenRoll.map((r) => r.disp).join(',') === '5,6,7,8,9', `Display slot must remain continuous 5..9 across u32::MAX->0 wrap, got ${seenRoll.map((r) => r.disp).join(',')}`);
    assert(seenRoll.every((r) => r.effK === 8), `eff_k must remain 8 across u32::MAX->0 wrap, got ${seenRoll.map((r) => r.effK).join(',')}`);
    assert(seenRoll[3].fCur === 0 && seenRoll[3].eng === 0 && (seenRoll[3].gap & GAP_WRAP) !== 0 && (seenRoll[3].gap & GAP_SEQ) === 0,
      `Engine wrap f=u32::MAX->0 must emit GAP_WRAP (0x40), got ${JSON.stringify(seenRoll[3])}`);
    assert(iv.w1[6] === iv.ringMeta[seenRoll[0].expectedSlot * RING_META_STRIDE + 23], 'Engine wrap f=u32::MAX->0 must preserve continuity_epoch');
  }

  // 5. All Product & Per-Slot Hero Stamp Refresh & Held Identity Preservation + Metrology Enable Bit Matrix
  {
    const stampWasm = instantiateWasm(wasmBytes);
    const se = stampWasm.exp;
    const sv = createViews(se, stampWasm.mem);
    se.chronos_init();
    se.chronos_set_resolution(1);
    se.chronos_set_params(0, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    se.chronos_set_swarm_mode(2);
    se.chronos_set_micro_metrology_mode(0);
    // Check mode=0 with swarm active: w1[7] bit 0 must be 0, PROD_MICRO must be invalid!
    const tmpScratch = new Uint8Array(hdBytes);
    paintTexturedSwarmAndHero(tmpScratch, hdTex, 640, 360, 0, 1120);
    new Uint8Array(stampWasm.mem.buffer, se.chronos_get_ingest_rgba_ptr(), hdBytes).set(tmpScratch);
    se.chronos_set_ingest_meta(60, 1, 0, 0, 1.5, flagsBase | 4, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1);
    se.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(se, sv.lap, sv.fiedler);
    se.chronos_post_tick(1 / 30);
    assert((sv.w1[7] & 1) === 0 && sv.w1[16] === 0, `With met=0, enabled bit0 must be 0 (${sv.w1[7]}) and PROD_MICRO invalid (${sv.w1[16]})`);

    // Now test swarm=0 with met=2: w1[7] bit 0 must be 1, PROD_MICRO stays invalid while inactive
    se.chronos_set_swarm_mode(0);
    se.chronos_set_micro_metrology_mode(2);
    new Uint8Array(stampWasm.mem.buffer, se.chronos_get_ingest_rgba_ptr(), hdBytes).set(tmpScratch);
    se.chronos_set_ingest_meta(60, 2, 33333, 0, 1.5, flagsBase | 4, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1);
    se.chronos_pre_tick(1 / 30);
    solveFiedlerLanczos(se, sv.lap, sv.fiedler);
    se.chronos_post_tick(1 / 30);
    assert((sv.w1[7] & 1) === 1 && sv.w1[16] === 0, `With swarm=0, met=2, enabled bit0 must be 1 (${sv.w1[7]}) and PROD_MICRO invalid (${sv.w1[16]})`);

    // Enable full Wave-1 suite (swarm=2, met=2, bos=1) and verify EVERY product + per-slot Hero stamp refresh + exact held identity
    se.chronos_init();
    se.chronos_set_resolution(1);
    se.chronos_set_params(0, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    se.chronos_set_swarm_mode(2);
    se.chronos_set_micro_metrology_mode(2);
    se.chronos_set_bos_mode(1);
    se.chronos_set_bos_gauge_roi(8, 8, 48, 120);

    const lastFreshProd = new Array(6).fill(null);
    const prodFreshCount = new Int32Array(6);
    const prodHeldCount = new Int32Array(6);
    const lastFreshHero = new Array(16).fill(null);
    let heroSlotFreshCount = 0;
    let heroSlotHeldCount = 0;

    for (let f = 0; f < 56; f++) {
      // Temporarily focus confirmed Hero at f=36..41 to guarantee FRESH Hero physics, then unfocus at f=42 to force HELD Hero physics!
      let activeTid = 0;
      for (let m = 0; m < 16; m++) {
        if (sv.telem[32 + m * 56] > 0.5) {
          activeTid = Math.round(sv.telem[32 + m * 56 + 1]);
          break;
        }
      }
      if (f >= 36 && f <= 41 && activeTid > 0) {
        se.chronos_set_focus_target(activeTid);
      } else if (f === 42) {
        se.chronos_set_focus_target(0);
      }

      paintTexturedSwarmAndHero(tmpScratch, hdTex, 640, 360, f, 1120);
      new Uint8Array(stampWasm.mem.buffer, se.chronos_get_ingest_rgba_ptr(), hdBytes).set(tmpScratch);
      const seq = f + 1;
      const tUs = Math.round(f * (1_000_000 / 30));
      assert(se.chronos_set_ingest_meta(61, seq, tUs >>> 0, 0, 2.5, flagsBase | 4, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1) === 1, `Stamp suite stage f=${f}`);
      se.chronos_pre_tick(1 / 30);
      solveFiedlerLanczos(se, sv.lap, sv.fiedler);
      se.chronos_post_tick(1 / 30);

      for (let p = 0; p < 6; p++) {
        const tup = readStampTuple(sv.w1, sv.w1f, 16 + p * 8);
        if ((tup[0] & STAMP_VALID) !== 0) {
          if ((tup[0] & STAMP_HELD) !== 0) {
            assert(lastFreshProd[p] !== null, `Product ${p} emitted HELD before ever FRESH at f=${f}`);
            for (let wi = 1; wi <= 7; wi++) {
              assert(tup[wi] === lastFreshProd[p][wi], `Product ${p} HELD word[${wi}] mismatch at f=${f}: ${tup[wi]} != ${lastFreshProd[p][wi]}`);
            }
            prodHeldCount[p]++;
          } else {
            assert(tup[7] === f && tup[3] === seq, `Product ${p} FRESH identity mismatch at f=${f}`);
            lastFreshProd[p] = tup;
            prodFreshCount[p]++;
          }
        } else {
          assert(sv.w1f[16 + p * 8 + 6] === -1.0, `Product ${p} INVALID must have age=-1.0`);
        }
      }
      for (let m = 0; m < 16; m++) {
        const tup = readStampTuple(sv.w1, sv.w1f, 64 + m * 8);
        if ((tup[0] & STAMP_VALID) !== 0) {
          if ((tup[0] & STAMP_HELD) !== 0) {
            assert(lastFreshHero[m] !== null, `Hero slot ${m} HELD before FRESH at f=${f}`);
            for (let wi = 1; wi <= 7; wi++) {
              assert(tup[wi] === lastFreshHero[m][wi], `Hero slot ${m} HELD word[${wi}] mismatch at f=${f}`);
            }
            heroSlotHeldCount++;
          } else {
            assert(tup[7] === f && tup[3] === seq, `Hero slot ${m} FRESH identity mismatch at f=${f}`);
            lastFreshHero[m] = tup;
            heroSlotFreshCount++;
          }
        } else {
          assert(sv.w1f[64 + m * 8 + 6] === -1.0, `Hero slot ${m} INVALID must have age=-1.0`);
          lastFreshHero[m] = null;
        }
      }
    }
    for (let p = 0; p < 6; p++) {
      assert(prodFreshCount[p] >= 1, `Product ${p} never went FRESH across 45 frames (counts=${Array.from(prodFreshCount)})`);
    }
    assert(prodHeldCount[1] >= 1 && prodHeldCount[2] >= 1 && prodHeldCount[3] >= 1 && prodHeldCount[4] >= 1,
      `Expected HELD frames for staggered products Grad/FTLE/Modal/HeroPhys: ${Array.from(prodHeldCount)}`);
    // Assert continuity_epoch bump inside chronos_pre_tick immediately invalidates held Wave-1 stamps before post_tick
    let heldSlotWord = -1;
    for (let extra = 45; extra < 52 && heldSlotWord < 0; extra++) {
      paintTexturedSwarmAndHero(tmpScratch, hdTex, 640, 360, extra, 1120);
      new Uint8Array(stampWasm.mem.buffer, se.chronos_get_ingest_rgba_ptr(), hdBytes).set(tmpScratch);
      const tUs = Math.round(extra * (1_000_000 / 30));
      assert(se.chronos_set_ingest_meta(61, extra + 1, tUs >>> 0, 0, 2.5, flagsBase | 4, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1) === 1);
      se.chronos_pre_tick(1 / 30);
      solveFiedlerLanczos(se, sv.lap, sv.fiedler);
      se.chronos_post_tick(1 / 30);
      for (const wordOff of [24, 32, 40, 48]) {
        if ((sv.w1[wordOff] & (STAMP_VALID | STAMP_HELD)) === (STAMP_VALID | STAMP_HELD)) {
          heldSlotWord = wordOff;
          break;
        }
      }
    }
    assert(heldSlotWord >= 0, 'Expected at least one STAMP_HELD slot before testing pre_tick epoch invalidation');
    const epBeforeDiscont = sv.w1[6];
    paintTexturedSwarmAndHero(tmpScratch, hdTex, 640, 360, 52, 1120);
    new Uint8Array(stampWasm.mem.buffer, se.chronos_get_ingest_rgba_ptr(), hdBytes).set(tmpScratch);
    assert(se.chronos_set_ingest_meta(61, 53, 1, 0, 2.5, flagsBase | 4, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1) === 1);
    se.chronos_pre_tick(1 / 30);
    assert(sv.w1[6] === epBeforeDiscont + 1, `Expected pre_tick continuity_epoch bump on NONINC_PTS: ${epBeforeDiscont} -> ${sv.w1[6]}`);
    assert((sv.w1[heldSlotWord] & STAMP_VALID) === 0, `Epoch bump inside pre_tick must immediately invalidate held stamp word=${heldSlotWord}, got flags=${sv.w1[heldSlotWord]}`);
    solveFiedlerLanczos(se, sv.lap, sv.fiedler);
    se.chronos_post_tick(1 / 30);
    assert(heroSlotFreshCount >= 1 && heroSlotHeldCount >= 1, `Expected per-slot Hero FRESH & HELD stamps (${heroSlotFreshCount}/${heroSlotHeldCount})`);
  }

  // 6. Byte-identical determinism across two independent Wasm instances
  {
    const runReplay = () => {
      const r = instantiateWasm(wasmBytes);
      r.exp.chronos_init();
      r.exp.chronos_set_resolution(1);
      r.exp.chronos_set_params(0, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
      r.exp.chronos_set_swarm_mode(2);
      r.exp.chronos_set_micro_metrology_mode(2);
      r.exp.chronos_set_bos_mode(1);
      r.exp.chronos_set_bos_gauge_roi(8, 8, 48, 120);
      const rv = createViews(r.exp, r.mem);
      const scratch = new Uint8Array(hdBytes);
      for (let f = 0; f < 48; f++) {
        paintTexturedSwarmAndHero(scratch, hdTex, 640, 360, f, 1120);
        new Uint8Array(r.mem.buffer, r.exp.chronos_get_ingest_rgba_ptr(), hdBytes).set(scratch);
        r.exp.chronos_set_ingest_meta(88, f + 1, f * 33333, 0, 1.0, flagsBase | 4, 640, 360, 0, 0, 640, 360, 0, 0xffffffff, 0, 0, 640, 360, 0, 1);
        r.exp.chronos_pre_tick(1 / 30);
        solveFiedlerLanczos(r.exp, rv.lap, rv.fiedler);
        r.exp.chronos_post_tick(1 / 30);
      }
      return Buffer.concat([
        Buffer.from(rv.telem.buffer, rv.telem.byteOffset, rv.telem.byteLength),
        Buffer.from(rv.w1.buffer, rv.w1.byteOffset, rv.w1.byteLength),
        Buffer.from(rv.ringMeta.buffer, rv.ringMeta.byteOffset, rv.ringMeta.byteLength),
        Buffer.from(rv.rowLuma.buffer, rv.rowLuma.byteOffset, rv.rowLuma.byteLength),
        Buffer.from(rv.bosPlanes.buffer, rv.bosPlanes.byteOffset, rv.bosPlanes.byteLength),
        Buffer.from(rv.microTracks.buffer, rv.microTracks.byteOffset, rv.microTracks.byteLength),
        Buffer.from(rv.microCov.buffer, rv.microCov.byteOffset, rv.microCov.byteLength),
      ]);
    };
    const bufA = runReplay();
    const bufB = runReplay();
    assert(Buffer.compare(bufA, bufB) === 0, 'Two identical Wasm replays diverged byte-for-byte!');
  }

  // 7. Real web/worker.js rejection & backpressure test
  await runWorkerRejectionHarness(wasmBytes);

  const tag = isPartial ? '[PARTIAL PASS - SCALED BUDGET]' : '[PASS]';
  console.log(
    `  [PASS] Gate I0 (Wave-1 Ingest, Clocks, Phase, Determinism & Worker Reject): commits=${g4Res.prevCommits}/1000, held={grad:${g4Res.heldGradSeen}, ftle:${g4Res.heldFtleSeen}, heroPhys:${g4Res.heldHeroPhysSeen}, heroFresh:${g4Res.freshHeroPhysSeen}}, dualStaticEnd=${dualStaticEnd}`
  );
  console.log(
    `  ${tag} Gate I4 (1,000f HD K=0 Textured BOS PCG + Dense Swarm): med=${g4Res.medMs.toFixed(2)} ms, p95=${g4Res.p95Ms.toFixed(2)} ms, rawMax=${g4Res.maxMs.toFixed(2)} ms (budget=${g4Res.budgetMs.toFixed(2)} ms, scale=${budgetScale.toFixed(2)}, bosReady=${g4Res.bosReadyFrames}, bosPcgConv=${g4Res.pcgConvergedFrames}/${g4Res.bosPcgFrames}, meanMicro=${g4Res.meanConfMicro.toFixed(0)})`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
