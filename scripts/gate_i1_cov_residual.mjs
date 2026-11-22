#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate I1: Calibrated Track Covariance, LOO Bench Flow
// Residual, 3-Chord Pinhole Looming Suppression, and Hero Hysteresis Gate
//
// Usage:
//   node scripts/gate_i1_cov_residual.mjs [--wasm=/path/to/chronos.wasm]
//   CHRONOS_I1_WASM=/path/to/chronos.wasm node scripts/gate_i1_cov_residual.mjs
//
// Honest Monocular Scope Disclosure:
//   3-chord pinhole looming rejection models stationary-camera constant-3D-velocity
//   X(t) = X_0 + V t and planar parallax. Arbitrary monocular depth-scaled curves
//   lambda(t) X(t) produce identical 2D image projections; this test verifies
//   declared-regime decoy FAR and maneuvering promotion without claiming universal
//   monocular FAR outside that physical operating model.
// ============================================================================

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const W = 640;
const H = 360;
const MAX_MICRO = 3072;
const MICRO_STRIDE = 8;

const QUAL_EST_MASK = 0x7;
const QUAL_COV_USABLE = 1 << 3;
const QUAL_BENCH_RES_USABLE = 1 << 4;
const QUAL_COV_BOUND_EXCEEDED = 1 << 7;
const QUAL_NEIGHBOR_SHIFT = 8;
const QUAL_BENCH_TRANSLATION_ONLY = 1 << 16;
const QUAL_LOOM_USABLE = 1 << 17;
const QUAL_PROMO_CANDIDATE = 1 << 18;
const QUAL_PROMO_BOUND_HERO = 1 << 19;

const COV_TARGET_1D = 0.682689;
const COV_TARGET_2D = 1.0 - Math.exp(-0.5); // 0.39346934
const COV_TOL = 0.03;

function resolveWasmPath() {
  const arg = process.argv.find((a) => a.startsWith('--wasm='));
  if (arg) return path.resolve(arg.slice('--wasm='.length));
  if (process.env.CHRONOS_I1_WASM) return path.resolve(process.env.CHRONOS_I1_WASM);
  const candidates = [
    path.join(ROOT, 'web', 'chronos_spectral.wasm'),
    path.join(ROOT, 'build', 'chronos.wasm'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return candidates[0];
}

function resolveFfmpeg() {
  if (fs.existsSync('/opt/homebrew/bin/ffmpeg')) return '/opt/homebrew/bin/ffmpeg';
  const which = spawnSync('which', ['ffmpeg'], { encoding: 'utf8' });
  if (which.status === 0 && which.stdout.trim()) return which.stdout.trim();
  throw new Error('ffmpeg is required for Gate I1 H.264 verification');
}

function makeRng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  return Math.sqrt(-2 * Math.log(Math.max(1e-12, rng()))) * Math.cos(2 * Math.PI * rng());
}

function lgamma(z) {
  const c = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.001208650973866179, -0.000005395239384953,
  ];
  let x = z;
  let y = z;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

function betacf(a, b, x) {
  const fpmin = 1e-30;
  const qab = a + b;
  const qap = a + 1.0;
  const qam = a - 1.0;
  let c = 1.0;
  let d = 1.0 - (qab * x) / qap;
  if (Math.abs(d) < fpmin) d = fpmin;
  d = 1.0 / d;
  let h = d;
  for (let m = 1; m <= 200; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (Math.abs(d) < fpmin) d = fpmin;
    c = 1.0 + aa / c;
    if (Math.abs(c) < fpmin) c = fpmin;
    d = 1.0 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (Math.abs(d) < fpmin) d = fpmin;
    c = 1.0 + aa / c;
    if (Math.abs(c) < fpmin) c = fpmin;
    d = 1.0 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1.0) < 1e-13) break;
  }
  return h;
}

function betainc(x, a, b) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1.0 - x));
  if (x < (a + 1.0) / (a + b + 2.0)) return (bt * betacf(a, b, x)) / a;
  return 1.0 - (bt * betacf(b, a, 1.0 - x)) / b;
}

function tCrit95(df) {
  // Exact two-sided 95% Student-t critical value for any df >= 1 via regularized incomplete beta
  if (df <= 0) return 1.95996398454;
  let lo = 1.95;
  let hi = 13.0;
  for (let it = 0; it < 60; it++) {
    const mid = 0.5 * (lo + hi);
    const pTwoTail = betainc(df / (df + mid * mid), 0.5 * df, 0.5);
    if (pTwoTail > 0.05) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

function runClusterSummary(values) {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  let ss = 0;
  for (const v of values) ss += (v - mean) * (v - mean);
  const sd = n > 1 ? Math.sqrt(ss / (n - 1)) : 0;
  const se = n > 1 ? sd / Math.sqrt(n) : 0;
  const half = tCrit95(n - 1) * se;
  return { mean, lo: mean - half, hi: mean + half, se };
}

function clopperPearsonUpper95(k, n) {
  if (k === 0) return 1.0 - Math.pow(0.05, 1.0 / n);
  // Binomial tail search for P(X <= k | n, p) = 0.05
  let lo = k / n;
  let hi = 1.0;
  const logFact = (m) => {
    let s = 0;
    for (let i = 2; i <= m; i++) s += Math.log(i);
    return s;
  };
  const lnN = logFact(n);
  const binomCdf = (p) => {
    let cdf = 0;
    for (let i = 0; i <= k; i++) {
      const lp = lnN - logFact(i) - logFact(n - i) + i * Math.log(p) + (n - i) * Math.log(1 - p);
      cdf += Math.exp(lp);
    }
    return cdf;
  };
  for (let it = 0; it < 50; it++) {
    const mid = 0.5 * (lo + hi);
    if (binomCdf(mid) > 0.05) lo = mid;
    else hi = mid;
  }
  return hi;
}

class WasmHarness {
  constructor(wasmBytes, mode) {
    this.wasmBytes = wasmBytes;
    this.mode = mode;
    this.ex = null;
    this.seq = 0;
  }

  async init() {
    const { instance } = await WebAssembly.instantiate(this.wasmBytes, {});
    const ex = instance.exports;
    for (const name of [
      'chronos_init',
      'chronos_set_resolution',
      'chronos_set_swarm_mode',
      'chronos_tick',
      'chronos_get_telemetry_ptr',
      'chronos_get_slot_meta_ptr',
      'chronos_get_micro_tracks_ptr',
      'chronos_set_micro_metrology_mode',
      'chronos_get_wave1_telemetry_ptr',
      'chronos_get_micro_cov_ptr',
      'chronos_get_micro_residual_ptr',
      'chronos_get_micro_loom_ptr',
      'chronos_get_micro_quality_ptr',
      'chronos_get_hero_candidates_ptr',
    ]) {
      assert.equal(typeof ex[name], 'function', `Missing Wasm export: ${name}`);
    }
    ex.chronos_init();
    ex.chronos_set_resolution(1);
    ex.chronos_set_swarm_mode(2);
    const readWave1ModeBit = () => new Uint32Array(ex.memory.buffer, ex.chronos_get_wave1_telemetry_ptr(), 16)[7] & 1;
    assert.equal(readWave1ModeBit(), 0, 'wave1_telem[7] bit0 must start 0 after init');
    assert.equal(ex.chronos_set_micro_metrology_mode(3), 0, 'invalid mode=3 must reject');
    assert.equal(readWave1ModeBit(), 0, 'mode bit must stay 0 after reject');
    assert.equal(ex.chronos_set_micro_metrology_mode(this.mode), 1, 'valid mode must accept');
    assert.equal(readWave1ModeBit(), this.mode !== 0 ? 1 : 0, 'wave1_telem[7] bit0 must reflect active mode');
    this.ex = ex;
    this.hasIngestMeta = typeof ex.chronos_set_ingest_meta === 'function';
    this.hasProbeClock = typeof ex.chronos_probe_set_ingest_clock === 'function';
    this.hasRingSlots = typeof ex.chronos_get_ring_slot_ptr === 'function' && this.hasIngestMeta;
    this.hasWave1Telem = true;
  }

  step(rgba, frameIdx, dtSec = 1 / 30, epoch = 1) {
    const ex = this.ex;
    const mem = ex.memory.buffer;
    const tUs = BigInt(Math.round(frameIdx * dtSec * 1_000_000));
    const tLo = Number(tUs & 0xffffffffn);
    const tHi = Number((tUs >> 32n) & 0xffffffffn);

    if (this.hasRingSlots && this.hasWave1Telem) {
      const slot = this.seq % 12;
      this.seq++;
      const ptr =
        typeof ex.chronos_get_ingest_rgba_ptr === 'function'
          ? ex.chronos_get_ingest_rgba_ptr()
          : ex.chronos_get_ring_slot_ptr(slot);
      new Uint8Array(mem, ptr, W * H * 4).set(rgba);
      // flags: bit1 (source PTS known) | bit2 (capture age) | bit4 (decoded geom known) = 22
      const ok = ex.chronos_set_ingest_meta(
        epoch,
        this.seq,
        tLo,
        tHi,
        1.0,
        22,
        W,
        H,
        0,
        0,
        W,
        H,
        0,
        0,
        0,
        0,
        W,
        H,
        0,
        1,
      );
      assert.equal(ok, 1, 'chronos_set_ingest_meta rejected valid HD 1:1 metadata');
      ex.chronos_tick(dtSec);
      return;
    }

    const rgbaPtr = ex.chronos_get_ingest_rgba_ptr();
    new Uint8Array(mem, rgbaPtr, W * H * 4).set(rgba);
    if (typeof ex.chronos_set_ingest_pts === 'function') {
      ex.chronos_set_ingest_pts(frameIdx * dtSec);
    }
    if (this.hasProbeClock) {
      ex.chronos_probe_set_ingest_clock(tLo, tHi, epoch);
    }
    ex.chronos_tick(dtSec);
  }

  readFrame() {
    const ex = this.ex;
    const mem = ex.memory.buffer;
    const tr = new Float32Array(mem, ex.chronos_get_micro_tracks_ptr(), MAX_MICRO * MICRO_STRIDE);
    const cov = new Float32Array(mem, ex.chronos_get_micro_cov_ptr(), 3 * MAX_MICRO);
    const res = new Float32Array(mem, ex.chronos_get_micro_residual_ptr(), 3 * MAX_MICRO);
    const loom = new Float32Array(mem, ex.chronos_get_micro_loom_ptr(), MAX_MICRO);
    const qual = new Uint32Array(mem, ex.chronos_get_micro_quality_ptr(), MAX_MICRO);
    const hc = new Uint32Array(mem, ex.chronos_get_hero_candidates_ptr(), 4 + 16 * 8);
    const hcDv = new DataView(hc.buffer, hc.byteOffset, hc.byteLength);

    const tracks = [];
    for (let i = 0; i < MAX_MICRO; i++) {
      const b = i * MICRO_STRIDE;
      const tid = Math.round(tr[b]);
      const state = Math.round(tr[b + 5]);
      if (tid <= 0 || state <= 0) continue;
      const q = qual[i];
      const cxx = cov[i];
      const cxy = cov[MAX_MICRO + i];
      const cyy = cov[2 * MAX_MICRO + i];
      const rx = res[i];
      const ry = res[MAX_MICRO + i];
      const d2Bench = res[2 * MAX_MICRO + i];
      const d2Loom = loom[i];
      assert.ok(
        Number.isFinite(cxx) && Number.isFinite(cxy) && Number.isFinite(cyy),
        `Non-finite covariance at slot ${i}`
      );
      assert.ok(
        Number.isFinite(rx) && Number.isFinite(ry) && Number.isFinite(d2Bench) && Number.isFinite(d2Loom),
        `Non-finite residual/loom at slot ${i}`
      );
      if ((q & QUAL_COV_USABLE) !== 0) {
        const trC = cxx + cyy;
        const diff = cxx - cyy;
        const disc = Math.sqrt(Math.max(0, diff * diff + 4 * cxy * cxy));
        const lMax = 0.5 * (trC + disc);
        const lMin = 0.5 * (trC - disc);
        assert.ok(
          lMin >= 0.99e-4 && lMax <= 256.01 && (q & QUAL_COV_BOUND_EXCEEDED) === 0,
          `Eigenvalue bounds violated when cov_usable=1: [${lMin}, ${lMax}]`
        );
      }
      tracks.push({
        slot: i,
        tid,
        x: tr[b + 1],
        y: tr[b + 2],
        vx: tr[b + 3],
        vy: tr[b + 4],
        state,
        cxx,
        cxy,
        cyy,
        rx,
        ry,
        d2Bench,
        d2Loom,
        q,
        est: q & QUAL_EST_MASK,
        covUsable: (q & QUAL_COV_USABLE) !== 0,
        benchUsable: (q & QUAL_BENCH_RES_USABLE) !== 0,
        transOnly: (q & QUAL_BENCH_TRANSLATION_ONLY) !== 0,
        loomUsable: (q & QUAL_LOOM_USABLE) !== 0,
        promo: (q & QUAL_PROMO_CANDIDATE) !== 0,
        bound: (q & QUAL_PROMO_BOUND_HERO) !== 0,
        nb: (q >>> QUAL_NEIGHBOR_SHIFT) & 0xff,
      });
    }

    const nCands = Math.min(16, hc[0]);
    const cands = [];
    for (let k = 0; k < nCands; k++) {
      const o = 4 + k * 8;
      cands.push({
        tid: hc[o],
        slot: hc[o + 1],
        flags: hc[o + 2],
        score: hcDv.getFloat32((o + 3) * 4, true),
        d2Bench: hcDv.getFloat32((o + 4) * 4, true),
        d2Loom: hcDv.getFloat32((o + 5) * 4, true),
        x: hcDv.getFloat32((o + 6) * 4, true),
        y: hcDv.getFloat32((o + 7) * 4, true),
      });
    }
    return { tracks, cands, numCands: hc[0], promoEpoch: hc[1], boundCnt: hc[2] };
  }

  readTargetsTelemetry() {
    const ex = this.ex;
    const mem = ex.memory.buffer;
    const telem = new Float32Array(mem, ex.chronos_get_telemetry_ptr(), 32 + 16 * 56);
    const slotMeta = new Float32Array(mem, ex.chronos_get_slot_meta_ptr(), 16 * 8);
    const slots = [];
    for (let m = 0; m < 16; m++) {
      const tb = 32 + m * 56;
      const mb = m * 8;
      slots.push({
        slot: m,
        active: telem[tb],
        tid: telem[tb + 1],
        x: telem[tb + 2],
        y: telem[tb + 3],
        posX: telem[tb + 8],
        posY: telem[tb + 9],
        age: slotMeta[mb],
        coasting: slotMeta[mb + 1],
        visEma: slotMeta[mb + 2],
        metaZero: slotMeta.subarray(mb, mb + 8).every((v) => v === 0),
      });
    }
    return { activeConfirmed: telem[3], slots };
  }
}

const BASE_BG_LUM = (() => {
  const lum = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      lum[y * W + x] = 0.15 + 0.02 * Math.sin(0.02 * x + 0.025 * y);
    }
  }
  return lum;
})();

const BASE_BG_RGBA = (() => {
  const rgba = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const u8 = Math.max(0, Math.min(255, Math.round(BASE_BG_LUM[i] * 255)));
    const p = i * 4;
    rgba[p] = u8;
    rgba[p + 1] = u8;
    rgba[p + 2] = u8;
    rgba[p + 3] = 255;
  }
  return rgba;
})();

function renderBlobsFrame(blobs, sigmaNoise, rng) {
  const rgba = new Uint8Array(BASE_BG_RGBA);
  const lum = new Float32Array(BASE_BG_LUM);
  const boxes = [];
  for (const b of blobs) {
    const cx = b.x;
    const cy = b.y;
    const ca = Math.cos(b.theta ?? 0);
    const sa = Math.sin(b.theta ?? 0);
    const su = b.su ?? 1.4;
    const sv = b.sv ?? 1.4;
    const inv2Su = 1 / (2 * su * su);
    const inv2Sv = 1 / (2 * sv * sv);
    const x0 = Math.max(0, Math.floor(cx - 12));
    const x1 = Math.min(W - 1, Math.ceil(cx + 12));
    const y0 = Math.max(0, Math.floor(cy - 12));
    const y1 = Math.min(H - 1, Math.ceil(cy + 12));
    boxes.push(x0, x1, y0, y1);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const u = ca * dx + sa * dy;
        const v = -sa * dx + ca * dy;
        const g = Math.exp(-(u * u * inv2Su + v * v * inv2Sv));
        const r2 = dx * dx + dy * dy;
        const env = r2 < 12.25 ? Math.min(1.0, Math.max(0.0, g * 2.2)) : 0;
        lum[y * W + x] += (b.amp ?? 0.6) * g + (sigmaNoise > 0 ? env * sigmaNoise * randn(rng) : 0);
      }
    }
  }
  for (let bi = 0; bi < boxes.length; bi += 4) {
    const x0 = boxes[bi];
    const x1 = boxes[bi + 1];
    const y0 = boxes[bi + 2];
    const y1 = boxes[bi + 3];
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * W + x;
        const u8 = Math.max(0, Math.min(255, Math.round(lum[i] * 255)));
        const p = i * 4;
        rgba[p] = u8;
        rgba[p + 1] = u8;
        rgba[p + 2] = u8;
        rgba[p + 3] = 255;
      }
    }
  }
  return rgba;
}

function runSpawnPipe(cmd, args, stdinBuf) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    const outChunks = [];
    const errChunks = [];
    proc.stdout.on('data', (c) => outChunks.push(c));
    proc.stderr.on('data', (c) => errChunks.push(c));
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`${cmd} exited ${code}: ${Buffer.concat(errChunks).toString()}`));
      } else {
        resolve(Buffer.concat(outChunks));
      }
    });
    if (stdinBuf) proc.stdin.end(stdinBuf);
    else proc.stdin.end();
  });
}

async function encodeDecodeH264Async(ffmpeg, frames, crf, jobId) {
  const tmpMp4 = path.join(os.tmpdir(), `chronos_i1_${process.pid}_${jobId}_${crf}.mp4`);
  try {
    const rawIn = Buffer.concat(frames.map((f) => Buffer.from(f.buffer, f.byteOffset, f.byteLength)));
    await runSpawnPipe(
      ffmpeg,
      [
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgba',
        '-s', `${W}x${H}`,
        '-r', '30',
        '-threads', '1',
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', String(crf),
        '-pix_fmt', 'yuv420p',
        tmpMp4,
      ],
      rawIn
    );
    const decoded = await runSpawnPipe(
      ffmpeg,
      ['-threads', '1', '-i', tmpMp4, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'],
      null
    );
    const out = [];
    const frameBytes = W * H * 4;
    for (let i = 0; i < frames.length; i++) {
      out.push(new Uint8Array(decoded.subarray(i * frameBytes, (i + 1) * frameBytes)));
    }
    return out;
  } finally {
    await fs.promises.unlink(tmpMp4).catch(() => {});
  }
}

async function mapPool(items, concurrency, fn) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const idx = next++;
      if (idx >= items.length) return;
      results[idx] = await fn(items[idx], idx);
    }
  }
  const workers = [];
  for (let w = 0; w < Math.min(concurrency, items.length); w++) workers.push(worker());
  await Promise.all(workers);
  return results;
}

async function testCoverageSuite(wasmBytes, ffmpeg, mode, useH264, numRuns, seedBase) {
  const runCovX = [];
  const runCovY = [];
  const runCov2D = [];
  const runVelCovX = [];
  const runVelCovY = [];
  const runVelCov2D = [];
  const runVelRatioCons = [];
  const runAvail = [];
  let totalEval = 0;
  let totalVelEval = 0;
  let totalNullBench = 0;
  let totalNullBenchHi = 0;
  let totalNullPromo = 0;
  let totalNullEval = 0;
  let sumNullBenchD2 = 0;

  const nFrames = 56;
  const sigNoise = useH264 ? 0.004 : 0.007;
  const runConfigs = Array.from({ length: numRuns }, (_, run) => {
    const rng = makeRng(seedBase + run * 409);
    const dir = rng() * 2 * Math.PI;
    const speed = 0.45 + 0.95 * rng();
    const vx = speed * Math.cos(dir);
    const vy = speed * Math.sin(dir);
    const blobs = [];
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 8; c++) {
        blobs.push({
          x0: 65 + c * 68 + (rng() - 0.5) * 6,
          y0: 55 + r * 60 + (rng() - 0.5) * 6,
          vx,
          vy,
          theta: rng() * Math.PI,
          su: 1.25 + rng() * 0.35,
          sv: 1.15 + rng() * 0.3,
          amp: 0.45 + 0.3 * rng(),
        });
      }
    }
    const rawFrames = [];
    for (let t = 0; t < nFrames; t++) {
      const fblobs = blobs.map((b) => ({ ...b, x: b.x0 + b.vx * t, y: b.y0 + b.vy * t }));
      rawFrames.push(renderBlobsFrame(fblobs, sigNoise, rng));
    }
    return { run, blobs, rawFrames, crf: 21 + (run % 3) };
  });

  const decodedRuns = useH264
    ? await mapPool(runConfigs, 6, async (cfg) => ({
        ...cfg,
        frames: await encodeDecodeH264Async(ffmpeg, cfg.rawFrames, cfg.crf, `cov_${cfg.run}`),
      }))
    : runConfigs.map((cfg) => ({ ...cfg, frames: cfg.rawFrames }));

  for (const { run, blobs, frames } of decodedRuns) {
    const h = new WasmHarness(wasmBytes, mode);
    await h.init();
    const hist = new Map();
    const prevByTid = new Map();
    let tObs = 0;
    let uObs = 0;
    let vc1x = 0;
    let vc1y = 0;
    let vc2d = 0;
    let vCnt = 0;
    let vRatioSum = 0;

    for (let t = 0; t < nFrames; t++) {
      h.step(frames[t], t, 1 / 30, 1);
      const { tracks } = h.readFrame();
      for (const tr of tracks) {
        if (t >= 16 && (tr.est === 2 || tr.est === 3 || tr.est === 4)) {
          tObs++;
          if (tr.covUsable) uObs++;
        }
        if (t >= 16) {
          totalNullEval++;
          if (tr.promo) totalNullPromo++;
        }
        if (!tr.covUsable) {
          prevByTid.delete(tr.tid);
          continue;
        }
        let bestB = -1;
        let bestD2 = 25;
        for (let bi = 0; bi < blobs.length; bi++) {
          const bx = blobs[bi].x0 + blobs[bi].vx * t;
          const by = blobs[bi].y0 + blobs[bi].vy * t;
          const d2 = (tr.x - bx) ** 2 + (tr.y - by) ** 2;
          if (d2 < bestD2) {
            bestD2 = d2;
            bestB = bi;
          }
        }
        if (bestB < 0) {
          prevByTid.delete(tr.tid);
          continue;
        }
        const bx = blobs[bestB].x0 + blobs[bestB].vx * t;
        const by = blobs[bestB].y0 + blobs[bestB].vy * t;
        if (!hist.has(tr.tid)) hist.set(tr.tid, []);
        hist.get(tr.tid).push({
          t,
          bb: bestB,
          ex: tr.x - bx,
          ey: tr.y - by,
          cxx: tr.cxx,
          cxy: tr.cxy,
          cyy: tr.cyy,
        });

        const prev = prevByTid.get(tr.tid);
        if (
          prev &&
          prev.t === t - 1 &&
          prev.bb === bestB &&
          t >= 17 &&
          (tr.est === 2 || tr.est === 3) &&
          (prev.est === 2 || prev.est === 3)
        ) {
          if (t >= 34 && bestD2 < 1.0 && prev.d2 < 1.0 && tr.benchUsable && !tr.transOnly && Number.isFinite(tr.d2Bench)) {
            totalNullBench++;
            sumNullBenchD2 += tr.d2Bench;
            if (tr.d2Bench >= 18.42) totalNullBenchHi++;
          }
          const invDt = 30.0;
          const invDt2Cons = 2.0 * invDt * invDt; // matches COV_VEL_CONSERVATIVE_MUL = 2.0
          const evx = (tr.x - prev.x) * invDt - blobs[bestB].vx * invDt;
          const evy = (tr.y - prev.y) * invDt - blobs[bestB].vy * invDt;
          const cvxx = (tr.cxx + prev.cxx) * invDt2Cons;
          const cvxy = (tr.cxy + prev.cxy) * invDt2Cons;
          const cvyy = (tr.cyy + prev.cyy) * invDt2Cons;
          const detV = cvxx * cvyy - cvxy * cvxy;
          if (detV > 1e-12 && cvxx > 0 && cvyy > 0) {
            if (Math.abs(evx) <= Math.sqrt(cvxx)) vc1x++;
            if (Math.abs(evy) <= Math.sqrt(cvyy)) vc1y++;
            const d2v = (cvyy * evx * evx - 2 * cvxy * evx * evy + cvxx * evy * evy) / detV;
            if (d2v <= 1.0) vc2d++;
            vRatioSum += 0.5 * d2v;
            vCnt++;
          }
        }
        prevByTid.set(tr.tid, {
          t,
          bb: bestB,
          d2: bestD2,
          x: tr.x,
          y: tr.y,
          cxx: tr.cxx,
          cxy: tr.cxy,
          cyy: tr.cyy,
          est: tr.est,
        });
      }
    }

    let c1x = 0;
    let c1y = 0;
    let c2d = 0;
    let cnt = 0;
    for (const [, arr] of hist) {
      const fit = arr.filter((s) => s.t >= 16 && s.t <= 32);
      const val = arr.filter((s) => s.t >= 34 && s.t < nFrames);
      if (fit.length < 12 || val.length < 12) continue;
      const b0 = fit[0].bb;
      if (!fit.every((s) => s.bb === b0) || !val.every((s) => s.bb === b0)) continue;
      const mx = fit.reduce((a, b) => a + b.ex, 0) / fit.length;
      const my = fit.reduce((a, b) => a + b.ey, 0) / fit.length;
      if (mx * mx + my * my > 1.2) continue;
      const adj = 1.0 + 1.0 / fit.length;
      for (const s of val) {
        const dx = s.ex - mx;
        const dy = s.ey - my;
        const cxx = s.cxx * adj;
        const cxy = s.cxy * adj;
        const cyy = s.cyy * adj;
        const det = cxx * cyy - cxy * cxy;
        if (!(det > 1e-12)) continue;
        if (Math.abs(dx) <= Math.sqrt(cxx)) c1x++;
        if (Math.abs(dy) <= Math.sqrt(cyy)) c1y++;
        const d2 = (cyy * dx * dx - 2 * cxy * dx * dy + cxx * dy * dy) / det;
        if (d2 <= 1.0) c2d++;
        cnt++;
      }
    }
    assert.ok(cnt >= 100 && vCnt >= 100 && tObs > 0, `Run ${run} insufficient samples (${cnt}, vel=${vCnt})`);
    runCovX.push(c1x / cnt);
    runCovY.push(c1y / cnt);
    runCov2D.push(c2d / cnt);
    runVelCovX.push(vc1x / vCnt);
    runVelCovY.push(vc1y / vCnt);
    runVelCov2D.push(vc2d / vCnt);
    runVelRatioCons.push(vRatioSum / vCnt);
    runAvail.push(uObs / tObs);
    totalEval += cnt;
    totalVelEval += vCnt;
  }

  const sx = runClusterSummary(runCovX);
  const sy = runClusterSummary(runCovY);
  const s2d = runClusterSummary(runCov2D);
  const vsx = runClusterSummary(runVelCovX);
  const vsy = runClusterSummary(runVelCovY);
  const vs2d = runClusterSummary(runVelCov2D);
  const vRatioCons = runClusterSummary(runVelRatioCons);
  const availMean = runAvail.reduce((a, b) => a + b, 0) / runAvail.length;
  const nullOutlierRate = totalNullBench > 0 ? totalNullBenchHi / totalNullBench : 0;
  const meanNullD2 = totalNullBench > 0 ? sumNullBenchD2 / totalNullBench : 0;
  const label = useH264 ? 'H264 (mode=2)' : 'Lossless (mode=1)';

  console.log(
    `[DATA] ${label}: runs=${numRuns} (t95=${tCrit95(numRuns - 1).toFixed(6)}) n=${totalEval} avail=${(availMean * 100).toFixed(2)}% ` +
      `covX=${sx.mean.toFixed(4)} [${sx.lo.toFixed(4)},${sx.hi.toFixed(4)}] ` +
      `covY=${sy.mean.toFixed(4)} [${sy.lo.toFixed(4)},${sy.hi.toFixed(4)}] ` +
      `cov2D=${s2d.mean.toFixed(4)} [${s2d.lo.toFixed(4)},${s2d.hi.toFixed(4)}]`
  );
  console.log(
    `[DATA] ${label} VelocityCov(2x): nVel=${totalVelEval} ` +
      `vCovX=${vsx.mean.toFixed(4)} [${vsx.lo.toFixed(4)},${vsx.hi.toFixed(4)}] ` +
      `vCovY=${vsy.mean.toFixed(4)} [${vsy.lo.toFixed(4)},${vsy.hi.toFixed(4)}] ` +
      `vCov2D=${vs2d.mean.toFixed(4)} [${vs2d.lo.toFixed(4)},${vs2d.hi.toFixed(4)}] ` +
      `E[d2v/2]=${vRatioCons.mean.toFixed(4)} | InKernelNull(n=${totalNullBench}) meanD2=${meanNullD2.toFixed(3)} hiRate=${(nullOutlierRate * 100).toFixed(3)}%`
  );

  assert.ok(availMean >= 0.90, `${label} availability ${(availMean * 100).toFixed(2)}% < 90%`);
  for (const [axis, st, target] of [
    ['covX', sx, COV_TARGET_1D],
    ['covY', sy, COV_TARGET_1D],
    ['cov2D', s2d, COV_TARGET_2D],
  ]) {
    assert.ok(
      Math.abs(st.mean - target) <= COV_TOL,
      `${label} ${axis} mean ${st.mean.toFixed(4)} outside ${target.toFixed(4)} +/- ${COV_TOL}`
    );
    assert.ok(
      st.lo >= target - COV_TOL && st.hi <= target + COV_TOL,
      `${label} ${axis} 95% CI [${st.lo.toFixed(4)}, ${st.hi.toFixed(4)}] outside ${target.toFixed(4)} +/- ${COV_TOL}`
    );
  }
  assert.ok(
    vsx.lo >= COV_TARGET_1D - COV_TOL && vsx.hi <= 0.98,
    `${label} conservative velCovX CI [${vsx.lo.toFixed(4)}, ${vsx.hi.toFixed(4)}] out of conservative bounds`
  );
  assert.ok(
    vsy.lo >= COV_TARGET_1D - COV_TOL && vsy.hi <= 0.98,
    `${label} conservative velCovY CI [${vsy.lo.toFixed(4)}, ${vsy.hi.toFixed(4)}] out of conservative bounds`
  );
  assert.ok(
    vs2d.lo >= COV_TARGET_2D - COV_TOL && vs2d.hi <= 0.92,
    `${label} conservative velCov2D CI [${vs2d.lo.toFixed(4)}, ${vs2d.hi.toFixed(4)}] out of conservative bounds`
  );
  assert.ok(vRatioCons.mean <= 1.15, `${label} conservative velocity covariance E[d2v/2]=${vRatioCons.mean.toFixed(3)} > 1.15 (undercoverage)`);
  const nullPromoRate = totalNullEval > 0 ? totalNullPromo / totalNullEval : 0;
  assert.ok(
    nullPromoRate <= 0.001,
    `${label} constant-flow null promo rate ${(nullPromoRate * 100).toFixed(4)}% (${totalNullPromo}/${totalNullEval}) > 0.1%`
  );
  assert.ok(totalNullBench >= 1000 && nullOutlierRate <= 0.01 && meanNullD2 >= 0.25 && meanNullD2 <= 3.0,
    `${label} in-kernel steady null d2Bench invalid: n=${totalNullBench}, mean=${meanNullD2.toFixed(3)}, hiRate=${(nullOutlierRate * 100).toFixed(3)}%`
  );
}

async function testRotatedAnisoSign(wasmBytes) {
  const rng = makeRng(4411);
  const res = {};
  for (const sign of [+1, -1]) {
    const theta = sign * (35 * Math.PI / 180);
    const blobs = [];
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 6; c++) {
        blobs.push({ x0: 90 + c * 90, y0: 70 + r * 75, vx: 0.55, vy: 0.25, theta, su: 2.5, sv: 1.0, amp: 0.65 });
      }
    }
    const h = new WasmHarness(wasmBytes, 1);
    await h.init();
    const cxyPred = [];
    const hist = new Map();
    for (let t = 0; t < 50; t++) {
      const fblobs = blobs.map((b) => ({ ...b, x: b.x0 + b.vx * t, y: b.y0 + b.vy * t }));
      h.step(renderBlobsFrame(fblobs, 0.006, rng), t, 1 / 30, 1);
      const { tracks } = h.readFrame();
      for (const tr of tracks) {
        if (!tr.covUsable) continue;
        let bb = -1;
        let bd2 = 16;
        for (let bi = 0; bi < fblobs.length; bi++) {
          const d2 = (tr.x - fblobs[bi].x) ** 2 + (tr.y - fblobs[bi].y) ** 2;
          if (d2 < bd2) {
            bd2 = d2;
            bb = bi;
          }
        }
        if (bb < 0) continue;
        if (t >= 20) cxyPred.push(tr.cxy);
        if (!hist.has(tr.tid)) hist.set(tr.tid, []);
        hist.get(tr.tid).push({ t, bb, dx: tr.x - fblobs[bb].x, dy: tr.y - fblobs[bb].y });
      }
    }
    const empProd = [];
    for (const [, arr] of hist) {
      const val = arr.filter((s) => s.t >= 20);
      if (val.length < 15 || !val.every((s) => s.bb === val[0].bb)) continue;
      const mx = val.reduce((a, b) => a + b.dx, 0) / val.length;
      const my = val.reduce((a, b) => a + b.dy, 0) / val.length;
      for (const s of val) empProd.push((s.dx - mx) * (s.dy - my));
    }
    const meanPred = cxyPred.reduce((a, b) => a + b, 0) / cxyPred.length;
    const meanEmp = empProd.reduce((a, b) => a + b, 0) / empProd.length;
    res[sign] = { meanPred, meanEmp };
  }
  console.log(
    `[PASS] Rotated anisotropic covariance sign: +35deg predCxy=${res[1].meanPred.toFixed(6)} emp=${res[1].meanEmp.toFixed(6)} | -35deg predCxy=${res[-1].meanPred.toFixed(6)} emp=${res[-1].meanEmp.toFixed(6)}`
  );
  assert.ok(res[1].meanPred > 0 && res[1].meanEmp > 0, '+35deg C_xy and empirical cov_xy must both be positive');
  assert.ok(res[-1].meanPred < 0 && res[-1].meanEmp < 0, '-35deg C_xy and empirical cov_xy must both be negative');
}

async function testLooInvarianceAndSupport(wasmBytes) {
  // 1. LOO invariance with >= 6 clean surrounding neighbors
  const runScene = async (perturbVelX) => {
    const rng = makeRng(5521);
    const bg = [];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 4; c++) {
        bg.push({ x0: 140 + c * 95, y0: 95 + r * 80, vx: 0.45, vy: 0.15 });
      }
    }
    const h = new WasmHarness(wasmBytes, 1);
    await h.init();
    let sample = null;
    for (let t = 0; t < 24; t++) {
      const qx = 310 + 0.45 * t + (t >= 18 ? perturbVelX * (t - 17) : 0);
      const qy = 180 + 0.15 * t;
      const fblobs = bg.map((b) => ({ x: b.x0 + b.vx * t, y: b.y0 + b.vy * t, su: 1.4, sv: 1.4, amp: 0.65 }));
      fblobs.push({ x: qx, y: qy, su: 1.4, sv: 1.4, amp: 0.7 });
      h.step(renderBlobsFrame(fblobs, 0.0, rng), t, 1 / 30, 1);
      if (t === 22) {
        const { tracks } = h.readFrame();
        sample = tracks.find((tr) => (tr.x - qx) ** 2 + (tr.y - qy) ** 2 < 16);
      }
    }
    return sample;
  };
  const baseQ = await runScene(0.0);
  const pertQ = await runScene(1.6); // +48 px/s perturbation on query target ONLY
  assert.ok(baseQ && pertQ && baseQ.benchUsable && pertQ.benchUsable, 'LOO target must have bench_residual_usable=1');
  assert.ok(!baseQ.transOnly && !pertQ.transOnly, '>=6 neighbors must use full affine (not translation_only)');
  assert.ok(baseQ.nb >= 6 && pertQ.nb >= 6);
  const vObsBase = baseQ.vx; // Note: export_micro stores EMA vel, let's check prediction consistency:
  // Predictor invariance: in uniform background flow, LOO predicted v_hat = 0.45*30 = 13.5 px/s stays unchanged while d2_bench increases monotonically
  assert.ok(pertQ.rx > baseQ.rx + 35.0, `LOO residual rx must shift by perturbation: base=${baseQ.rx.toFixed(2)}, pert=${pertQ.rx.toFixed(2)}`);
  assert.ok(pertQ.d2Bench > baseQ.d2Bench + 200.0 && baseQ.d2Bench < 4.0, `LOO d2_bench invalid: base=${baseQ.d2Bench.toFixed(2)}, pert=${pertQ.d2Bench.toFixed(2)}`);
  // Also verify v_hat_x(x_i) computed from neighbor pool stayed unchanged when query shifted by 48 px/s:
  // Note: residual rx = obs_v_x - v_hat_x, obs_v_x = (1.6 + 0.45)*30 = 61.5 vs 13.5 -> expected rx diff ~ 48.0 px/s
  const vHatDiff = Math.abs((pertQ.rx - baseQ.rx) - 1.6 * 30);
  assert.ok(vHatDiff < 2.0, `LOO background prediction shifted under query perturbation: diff=${vHatDiff.toFixed(2)} px/s`);

  // 2. Translation-only (3..5 clean neighbors): exactly 4 clusters total (3 LOO neighbors -> bit 16 = 1)
  {
    const rng = makeRng(5522);
    const h = new WasmHarness(wasmBytes, 1);
    await h.init();
    const pts = [
      { x0: 270, y0: 150, vx: 0.4, vy: 0.1 },
      { x0: 360, y0: 150, vx: 0.4, vy: 0.1 },
      { x0: 270, y0: 220, vx: 0.4, vy: 0.1 },
      { x0: 360, y0: 220, vx: 0.4, vy: 0.1 },
    ];
    let hit = null;
    for (let t = 0; t < 20; t++) {
      const fblobs = pts.map((p, idx) => ({
        x: p.x0 + p.vx * t + (idx === 0 && t >= 15 ? 1.5 * (t - 14) : 0),
        y: p.y0 + p.vy * t,
        su: 1.4,
        sv: 1.4,
        amp: 0.65,
      }));
      h.step(renderBlobsFrame(fblobs, 0.0, rng), t, 1 / 30, 1);
      if (t === 19) {
        const { tracks } = h.readFrame();
        hit = tracks.filter((tr) => tr.benchUsable);
      }
    }
    assert.ok(hit && hit.length === 4, `Expected 4 bench tracks with 3 LOO neighbors, got ${hit?.length}`);
    for (const tr of hit) {
      assert.equal(tr.nb, 3, '4-track cluster must report nb=3 LOO neighbors');
      assert.ok(tr.transOnly, 'nb=3 must set QUAL_BENCH_TRANSLATION_ONLY (bit 16)');
      assert.ok(!tr.promo, 'translation_only track must never auto-promote to Hero');
    }
  }

  // 3. Insufficient support (< 3 clean neighbors): 2 isolated tracks -> bench_usable=0 and zeroed residual
  {
    const rng = makeRng(5523);
    const h = new WasmHarness(wasmBytes, 1);
    await h.init();
    const pts = [
      { x0: 280, y0: 180, vx: 0.5, vy: 0.2 },
      { x0: 340, y0: 180, vx: 0.5, vy: 0.2 },
    ];
    let trks = [];
    for (let t = 0; t < 16; t++) {
      const fblobs = pts.map((p) => ({ x: p.x0 + p.vx * t, y: p.y0 + p.vy * t, su: 1.4, sv: 1.4, amp: 0.65 }));
      h.step(renderBlobsFrame(fblobs, 0.0, rng), t, 1 / 30, 1);
      if (t === 15) trks = h.readFrame().tracks;
    }
    assert.ok(trks.length >= 2);
    for (const tr of trks) {
      assert.ok(!tr.benchUsable && !tr.transOnly, '<3 neighbors must clear bench_residual_usable');
      assert.equal(tr.rx, 0, '<3 neighbors must zero rx');
      assert.equal(tr.ry, 0, '<3 neighbors must zero ry');
      assert.equal(tr.d2Bench, 0, '<3 neighbors must zero d2_bench');
    }
  }
  // 4. Collinear >= 6 neighbors rank-failure check (ill-conditioned normalized X^T W X must reject)
  {
    const rng = makeRng(5524);
    const h = new WasmHarness(wasmBytes, 1);
    await h.init();
    const pts = [];
    for (let c = 0; c < 8; c++) {
      pts.push({ x0: 255 + c * 15, y0: 180, vx: 0.45, vy: 0.15 });
    }
    let trks = [];
    for (let t = 0; t < 20; t++) {
      const fblobs = pts.map((p) => ({ x: p.x0 + p.vx * t, y: p.y0 + p.vy * t, su: 1.4, sv: 1.4, amp: 0.68 }));
      h.step(renderBlobsFrame(fblobs, 0.0, rng), t, 1 / 30, 1);
      if (t === 19) trks = h.readFrame().tracks;
    }
    assert.ok(trks.length >= 7, `Expected >= 7 collinear tracks, got ${trks.length}`);
    assert.ok(trks.some((tr) => tr.nb >= 6), 'Expected at least some collinear tracks to have >= 6 neighbors');
    for (const tr of trks) {
      assert.ok(!tr.benchUsable, `Collinear >=6 neighbors must reject ill-conditioned affine (nb=${tr.nb})`);
      assert.equal(tr.rx, 0, 'Collinear rank-failure must zero rx');
      assert.equal(tr.ry, 0, 'Collinear rank-failure must zero ry');
      assert.equal(tr.d2Bench, 0, 'Collinear rank-failure must zero d2_bench');
    }
  }
  console.log(
    `[PASS] LOO invariance & support tiers: affine nb=${pertQ.nb} d2=${baseQ.d2Bench.toFixed(2)}->${pertQ.d2Bench.toFixed(1)} | vHatDiff=${vHatDiff.toFixed(2)} px/s | trans_only(nb=3)=ok | <3 zeroed=ok | collinear>=6 rejected=ok`
  );
}

async function testLoomingAndParallaxFar(wasmBytes, ffmpeg) {
  let evalCount = 0;
  let falsePromos = 0;
  let episodePromos = 0;
  const totalEpisodes = 300; // 150 lossless + 150 H.264 -> 0/300 Clopper-Pearson 95% UB = 0.994% <= 1.0%
  const d2LoomAll = [];
  let minEpHighBench = Infinity;
  let maxBenchOverall = 0;
  const nFrames = 52;
  const fx = 520;
  const cx0 = 320;
  const cy0 = 180;

  const buildEpisodeSpec = (ep) => {
    const useH264 = ep >= 150;
    const mode = useH264 ? 2 : 1;
    const rng = makeRng(6600 + ep * 997);
    const kind = ep % 3; // 0: radial looming, 1: off-axis 3D flyby, 2: planar parallax (Vz=0)
    const sx = rng() < 0.5 ? -1 : 1;
    const sy = rng() < 0.5 ? -1 : 1;
    const X0 = kind === 0 ? sx * (0.28 + 0.14 * rng()) : sx * (0.38 + 0.22 * rng());
    const Y0 = kind === 0 ? sy * (0.16 + 0.12 * rng()) : sy * (0.14 + 0.12 * rng());
    const Z0 = 3.0 + 0.6 * rng();
    const Vx = kind === 0 ? 0.0 : -sx * (0.48 + 0.28 * rng());
    const Vy = kind === 0 ? 0.0 : -sy * (0.20 + 0.12 * rng());
    const Vz = kind === 2 ? 0.0 : -(0.80 + 0.30 * rng());

    const decoyXY = [];
    for (let t = 0; t < nFrames; t++) {
      const ts = t / 30;
      const Z = Z0 + Vz * ts;
      decoyXY.push({
        px: cx0 + (fx * (X0 + Vx * ts)) / Z,
        py: cy0 + (fx * (Y0 + Vy * ts)) / Z,
      });
    }

    const bgVx = 0.28 + 0.22 * (rng() - 0.3);
    const bgVy = 0.06 + 0.16 * (rng() - 0.5);
    const bg = [];
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 6; c++) {
        const bx0 = 65 + c * 98 + (rng() - 0.5) * 4;
        const by0 = 55 + r * 80 + (rng() - 0.5) * 4;
        let clear = true;
        for (let t = 0; t < nFrames; t++) {
          const d2 = (bx0 + bgVx * t - decoyXY[t].px) ** 2 + (by0 + bgVy * t - decoyXY[t].py) ** 2;
          if (d2 < 18 * 18) {
            clear = false;
            break;
          }
        }
        if (clear) bg.push({ x0: bx0, y0: by0, vx: bgVx, vy: bgVy });
      }
    }

    const rawFrames = [];
    for (let t = 0; t < nFrames; t++) {
      const { px, py } = decoyXY[t];
      const fblobs = bg.map((b) => ({ x: b.x0 + b.vx * t, y: b.y0 + b.vy * t, su: 1.5, sv: 1.5, amp: 0.6 }));
      fblobs.push({ x: px, y: py, su: 1.4, sv: 1.4, amp: 0.78 });
      rawFrames.push(renderBlobsFrame(fblobs, useH264 ? 0.003 : 0.005, rng));
    }
    return { ep, mode, useH264, crf: 21 + (ep % 3), decoyXY, rawFrames };
  };

  const chunkSize = 25;
  for (let start = 0; start < totalEpisodes; start += chunkSize) {
    const end = Math.min(totalEpisodes, start + chunkSize);
    const specs = [];
    for (let ep = start; ep < end; ep++) specs.push(buildEpisodeSpec(ep));
    const prepared = await mapPool(specs, 8, async (spec) => ({
      ...spec,
      frames: spec.useH264
        ? await encodeDecodeH264Async(ffmpeg, spec.rawFrames, spec.crf, `far_${spec.ep}`)
        : spec.rawFrames,
    }));

    for (const { mode, decoyXY, frames } of prepared) {
      const h = new WasmHarness(wasmBytes, mode);
      await h.init();
      let epFalse = false;
      let epHighBench = 0;
      for (let t = 0; t < nFrames; t++) {
        h.step(frames[t], t, 1 / 30, 1);
        const { tracks, cands } = h.readFrame();
        const { px, py } = decoyXY[t];
        for (const tr of tracks) {
          if ((tr.x - px) ** 2 + (tr.y - py) ** 2 < 20 && tr.benchUsable && tr.loomUsable) {
            evalCount++;
            d2LoomAll.push(tr.d2Loom);
            if (tr.d2Bench >= 18.42) epHighBench++;
            if (tr.d2Bench > maxBenchOverall) maxBenchOverall = tr.d2Bench;
            if (tr.promo || cands.some((c) => c.tid === tr.tid)) {
              falsePromos++;
              epFalse = true;
            }
          }
        }
      }
      if (epHighBench < minEpHighBench) minEpHighBench = epHighBench;
      if (epFalse) episodePromos++;
    }
  }

  assert.ok(evalCount >= 5000, `Expected >= 5000 eligible decoy evaluations across 300 episodes, got ${evalCount}`);
  d2LoomAll.sort((a, b) => a - b);
  const med = d2LoomAll[Math.floor(d2LoomAll.length / 2)];
  const p95 = d2LoomAll[Math.floor(d2LoomAll.length * 0.95)];
  const maxD2 = d2LoomAll[d2LoomAll.length - 1];
  const ub95Ep = clopperPearsonUpper95(episodePromos, totalEpisodes);

  console.log(
    `[PASS] Looming & parallax decoys (lossless+H264): episodes=${totalEpisodes} epFalse=${episodePromos} ` +
      `epOneSided95%UB=${(ub95Ep * 100).toFixed(3)}% evals=${evalCount} falsePromos=${falsePromos} ` +
      `minEpHighBench=${minEpHighBench} maxD2Bench=${maxBenchOverall.toFixed(1)} loomD2(med/p95/max)=${med.toFixed(3)}/${p95.toFixed(3)}/${maxD2.toFixed(3)}`
  );
  console.log(
    '[INFO] Monocular scope: constant-3D-velocity X(t)=X_0+Vt pinhole looming & planar parallax rejected; ' +
      'arbitrary depth-scaled trajectories lambda(t)X(t) remain observationally indistinguishable in monocular 2D projections.'
  );
  assert.ok(minEpHighBench >= 10, `Every decoy episode must strongly trigger LOO residual >= 18.42 (minEpHighBench=${minEpHighBench})`);
  assert.ok(
    episodePromos === 0 && falsePromos === 0 && ub95Ep <= 0.01,
    `Episode decoy FAR exceeded 1% ceiling (epFalse=${episodePromos}/${totalEpisodes}, UB=${(ub95Ep * 100).toFixed(3)}%)`
  );
}

async function testManeuverHysteresisAndDisable(wasmBytes) {
  const rng = makeRng(7707);
  const bg = [];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 6; c++) {
      bg.push({ x0: 70 + c * 98, y0: 55 + r * 80, vx: 0.35, vy: 0.1 });
    }
  }
  const h = new WasmHarness(wasmBytes, 1);
  await h.init();

  let promoStart = -1;
  let promoEnd = -1;
  let targetTid = 0;
  let dropTime = -1;
  let boundHeroSlot = -1;
  let ownedBoundSlot = -1;
  let sawAutoHeroBound = false;
  let sawOwnedSlotRetired = false;
  let sawArenaTargetSlot = false;
  let sawArenaTargetAge = false;
  let prevActiveSet = new Set();
  const traj = [];

  for (let t = 0; t < 75; t++) {
    let ox = 200 + 0.35 * t;
    let oy = 180 + 0.1 * t;
    if (t >= 16 && t <= 42) {
      const tau = t - 16;
      ox += 24 * (1 - Math.cos(0.32 * tau));
      oy += 22 * Math.sin(0.35 * tau);
    } else if (t > 42) {
      ox = 200 + 0.35 * 42 + 24 * (1 - Math.cos(0.32 * 26)) + 0.35 * (t - 42);
      oy = 180 + 0.1 * 42 + 22 * Math.sin(0.35 * 26) + 0.1 * (t - 42);
    }
    const fblobs = bg.map((b) => ({ x: b.x0 + b.vx * t, y: b.y0 + b.vy * t, su: 1.5, sv: 1.5, amp: 0.6 }));
    fblobs.push({ x: ox, y: oy, su: 1.4, sv: 1.4, amp: 0.8 });
    h.step(renderBlobsFrame(fblobs, 0.003, rng), t, 1 / 30, 1);
    const { tracks, cands, boundCnt } = h.readFrame();
    const targetsTelem = h.readTargetsTelemetry();
    const tr = tracks.find((z) => (z.x - ox) ** 2 + (z.y - oy) ** 2 < 25);
    if (t >= 2) assert.ok(tr, `Lost maneuvering outlier at t=${t}`);
    if (!tr) {
      prevActiveSet = new Set(targetsTelem.slots.filter((s) => s.active > 0.5).map((s) => s.slot));
      continue;
    }
    if (t === 15) targetTid = tr.tid;
    if (tr.promo) {
      if (promoStart < 0) promoStart = t;
      promoEnd = t;
      const cand = cands.find((c) => c.tid === tr.tid);
      assert.ok(cand, `Promoted track ${tr.tid} missing from hero_candidates at t=${t}`);
      assert.ok(targetsTelem.activeConfirmed >= 1, `Arena.targets activeConfirmed=0 during promotion at t=${t}`);
      if (t > promoStart) {
        assert.ok(boundCnt >= 1 && (cand.flags & 2) !== 0 && tr.bound, `Promoted track ${tr.tid} not bound to Arena.targets at t=${t}`);
        sawAutoHeroBound = true;
        if (ownedBoundSlot < 0) {
          const bornSlot = targetsTelem.slots.find(
            (s) =>
              s.active > 0.5 &&
              !prevActiveSet.has(s.slot) &&
              ((s.posX - tr.x) ** 2 + (s.posY - tr.y) ** 2 < 400 ||
                (s.x - tr.x) ** 2 + (s.y - tr.y) ** 2 < 400)
          );
          if (bornSlot) ownedBoundSlot = bornSlot.slot;
        }
      }
      const hSlot = targetsTelem.slots.find(
        (s) =>
          s.active > 0.5 &&
          s.tid > 0 &&
          ((s.posX - tr.x) ** 2 + (s.posY - tr.y) ** 2 < 400 ||
            (s.x - tr.x) ** 2 + (s.y - tr.y) ** 2 < 400)
      );
      if (hSlot) {
        sawArenaTargetSlot = true;
        boundHeroSlot = hSlot.slot;
        if (t > promoStart && hSlot.age >= 1) sawArenaTargetAge = true;
      }
    } else if (promoStart >= 0 && dropTime < 0 && t > 42) {
      dropTime = t;
      assert.ok(
        !cands.some((c) => c.tid === targetTid),
        `Exited track ${targetTid} still in hero_candidates at t=${t}`
      );
      assert.equal(boundCnt, 0, `Exited track ${targetTid} still counted in boundCnt at t=${t}`);
      assert.ok(!tr.bound, `Exited track ${targetTid} still marked bound at t=${t}`);
      if (ownedBoundSlot >= 0) {
        assert.equal(
          targetsTelem.slots[ownedBoundSlot].active,
          0,
          `Wave-1 owned Arena.targets[${ownedBoundSlot}] did not retire on exit at t=${t}`
        );
        sawOwnedSlotRetired = true;
      }
    }
    prevActiveSet = new Set(targetsTelem.slots.filter((s) => s.active > 0.5).map((s) => s.slot));
    traj.push({ t, promo: tr.promo, d2b: tr.d2Bench, d2l: tr.d2Loom });
  }

  assert.ok(promoStart >= 18 && promoStart <= 24, `Expected Hero auto-promotion around t=19..24, got t=${promoStart}`);
  assert.ok(sawAutoHeroBound, 'Expected Wave-1 owned auto-Hero binding on Arena.targets');
  assert.ok(sawOwnedSlotRetired, 'Expected Wave-1 owned Arena.targets slot retirement on hysteresis exit');
  assert.ok(sawArenaTargetSlot, 'Expected matching active Arena.targets Hero slot during promotion');
  assert.ok(sawArenaTargetAge, 'Expected Arena.targets slot_meta.age >= 1 on frames while auto-Hero bound');
  // After maneuver ends at t=42, t=43..46 are first 4 non-hold frames (must stay held!), exit on frame 5 (t=47)!
  assert.ok(dropTime >= 47 && dropTime <= 49, `Expected hysteresis 5-frame exit around t=47, got t=${dropTime}`);

  // Verify continuity epoch reset clears temporal residuals/candidates cleanly
  const bblobs = bg.map((b) => ({ x: b.x0 + b.vx * 75, y: b.y0 + b.vy * 75, su: 1.5, sv: 1.5, amp: 0.6 }));
  h.step(renderBlobsFrame(bblobs, 0.002, rng), 75, 1 / 30, 2);
  const afterEpoch = h.readFrame();
  assert.equal(afterEpoch.numCands, 0, 'Continuity epoch change must clear hero_candidates');
  for (const tr of afterEpoch.tracks) {
    assert.ok(!tr.benchUsable && !tr.loomUsable && !tr.promo, 'Epoch change must reset temporal metrology flags');
  }

  // Verify mode=0 disable clears all sidecars and preserves legacy tracking
  assert.equal(h.ex.chronos_set_micro_metrology_mode(0), 1);
  assert.equal(
    new Uint32Array(h.ex.memory.buffer, h.ex.chronos_get_wave1_telemetry_ptr(), 16)[7] & 1,
    0,
    'wave1_telem[7] bit0 must clear on mode=0'
  );
  const afterOff = h.readFrame();
  assert.equal(afterOff.numCands, 0);
  assert.ok(afterOff.tracks.length >= 20, 'Disabling metrology must keep legacy tracks intact');
  for (const tr of afterOff.tracks) {
    assert.equal(tr.cxx, 0);
    assert.equal(tr.cxy, 0);
    assert.equal(tr.cyy, 0);
    assert.equal(tr.d2Bench, 0);
    assert.equal(tr.d2Loom, 0);
    assert.equal(tr.q, 0);
  }
  console.log(
    `[PASS] Maneuver auto-promotion & hysteresis: tid=${targetTid} arenaSlot=${boundHeroSlot} entered at t=${promoStart}, held through t=${promoEnd}, exited at t=${dropTime} (5-frame exit + Arena.targets release verified), epoch reset & mode=0 cleanup verified`
  );

  // Coasting pause & translation-only on_streak reset verification
  {
    const rng2 = makeRng(7708);
    const h2 = new WasmHarness(wasmBytes, 1);
    await h2.init();
    let seenOffStreak2 = -1;
    let coastOffStreakA = -1;
    let coastOffStreakB = -1;
    let oxEnd = 200;
    let oyEnd = 180;
    for (let t = 0; t < 42; t++) {
      let ox = 200 + 0.35 * t;
      let oy = 180 + 0.1 * t;
      if (t >= 16 && t <= 26) {
        const tau = t - 16;
        ox += 24 * (1 - Math.cos(0.32 * tau));
        oy += 22 * Math.sin(0.35 * tau);
        oxEnd = ox;
        oyEnd = oy;
      } else if (t > 26) {
        ox = oxEnd + 0.35 * (t - 26);
        oy = oyEnd + 0.1 * (t - 26);
      }
      const occludeTarget = t === 29 || t === 30;
      const fblobs = bg.map((b) => ({ x: b.x0 + b.vx * t, y: b.y0 + b.vy * t, su: 1.5, sv: 1.5, amp: 0.6 }));
      if (!occludeTarget) fblobs.push({ x: ox, y: oy, su: 1.4, sv: 1.4, amp: 0.8 });
      h2.step(renderBlobsFrame(fblobs, 0.002, rng2), t, 1 / 30, 1);
      const { cands } = h2.readFrame();
      const cand = cands.find((c) => (c.x - ox) ** 2 + (c.y - oy) ** 2 < 400);
      if (t === 28 && cand) seenOffStreak2 = (cand.flags >>> 16) & 0xff;
      if (t === 29 && cand) coastOffStreakA = (cand.flags >>> 16) & 0xff;
      if (t === 30 && cand) coastOffStreakB = (cand.flags >>> 16) & 0xff;
    }
    assert.ok(seenOffStreak2 >= 1, `Expected eligible non-hold frames to increment off_streak (got ${seenOffStreak2})`);
    assert.equal(coastOffStreakA, seenOffStreak2, `Coast frame 1 must not increment off_streak (${coastOffStreakA} != ${seenOffStreak2})`);
    assert.equal(coastOffStreakB, seenOffStreak2, `Coast frame 2 must not increment off_streak (${coastOffStreakB} != ${seenOffStreak2})`);
  }
  {
    const rng3 = makeRng(7709);
    const h3 = new WasmHarness(wasmBytes, 1);
    await h3.init();
    const corePts = [
      { x0: 280, y0: 150 },
      { x0: 360, y0: 150 },
      { x0: 280, y0: 220 },
      { x0: 360, y0: 220 },
    ];
    const extraPts = [
      { x0: 255, y0: 185 },
      { x0: 410, y0: 185 },
      { x0: 320, y0: 120 },
      { x0: 320, y0: 245 },
    ];
    let firstPromoAfterUnlock = -1;
    for (let t = 0; t < 48; t++) {
      let ox = 318 + 0.35 * t;
      let oy = 185 + 0.1 * t;
      if (t >= 16) {
        const tau = t - 16;
        ox += 18 * (1 - Math.cos(0.32 * tau));
        oy += 16 * Math.sin(0.35 * tau);
      }
      // t < 24: only 4 background points -> nb <= 4 (translation_only, must keep on_streak=0!)
      const activeBg = t < 24 ? corePts : corePts.concat(extraPts);
      const fblobs = activeBg.map((p) => ({ x: p.x0 + 0.35 * t, y: p.y0 + 0.1 * t, su: 1.5, sv: 1.5, amp: 0.65 }));
      fblobs.push({ x: ox, y: oy, su: 1.4, sv: 1.4, amp: 0.8 });
      h3.step(renderBlobsFrame(fblobs, 0.002, rng3), t, 1 / 30, 1);
      const { numCands } = h3.readFrame();
      if (t < 24) {
        assert.equal(numCands, 0, `Translation-only phase at t=${t} must never promote`);
      } else if (numCands > 0 && firstPromoAfterUnlock < 0) {
        firstPromoAfterUnlock = t;
      }
    }
    // extraPts spawn at t=24 and need confirmation + 0.075s velocity + 3 eligible fresh frames -> promo >= t=29
    assert.ok(
      firstPromoAfterUnlock >= 29,
      `on_streak failed to unlock or carried across translation-only frames (promo at t=${firstPromoAfterUnlock})`
    );
  }
  console.log('[PASS] Hysteresis edge-cases: coast frames preserve off_streak; translation-only resets on_streak');
}

async function main() {
  assert.ok(Math.abs(tCrit95(47) - 2.0117405) < 1e-5, `tCrit95(47) mismatch: ${tCrit95(47)}`);
  assert.ok(Math.abs(tCrit95(30) - 2.042272) < 1e-5, `tCrit95(30) mismatch: ${tCrit95(30)}`);
  assert.ok(Math.abs(tCrit95(1) - 12.706205) < 1e-4, `tCrit95(1) mismatch: ${tCrit95(1)}`);
  const wasmPath = resolveWasmPath();
  assert.ok(fs.existsSync(wasmPath), `Wasm binary not found: ${wasmPath}`);
  const wasmBytes = fs.readFileSync(wasmPath);
  const ffmpeg = resolveFfmpeg();
  console.log(`Gate I1 target Wasm: ${wasmPath} (${(wasmBytes.length / 1024).toFixed(1)} KiB)`);

  await testRotatedAnisoSign(wasmBytes);
  await testCoverageSuite(wasmBytes, ffmpeg, 1, false, 48, 110000);
  await testCoverageSuite(wasmBytes, ffmpeg, 2, true, 48, 210000);
  await testLooInvarianceAndSupport(wasmBytes);
  await testLoomingAndParallaxFar(wasmBytes, ffmpeg);
  await testManeuverHysteresisAndDisable(wasmBytes);
  console.log('ALL GATE I1 CHECKS PASSED.');
}

main().catch((err) => {
  console.error('[FAIL] Gate I1 failed:', err);
  process.exit(1);
});
