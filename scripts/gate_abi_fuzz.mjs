#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate X: adversarial C-ABI fuzz
//
// Calls the exports of web/chronos_spectral.wasm in random order with hostile
// arguments (0, -1, u32/i32 extremes, NaN, +-Infinity, subnormals, 3.4e38, frame
// sizes +-1, ...), hostile pixels (noise, black, white, strobe, alpha 0, moving
// discs) and hostile staging buffers (audio, Hodge injection, conservation input,
// Laplacian, Fiedler vector filled with NaN, +-1e30, extremes), interleaved with
// in-order and out-of-order ticks at hostile time steps, then reads every pointer
// getter and the f32 telemetry views.
//
//   Gate:     zero traps. A trap is a wasm trap or JS exception from any export
//             call, or a change of the fixed 448-page memory size.
//   Reported: non-finite values in the f32 telemetry views (telemetry,
//             swarm_telemetry, conservation_telemetry, slot_meta), per slot with
//             the first occurrence and the calls before it. Not gated.
//
// Deterministic: trial k draws everything from its own seeded PRNG, so a given
// wasm makes exactly the same calls on every run. Bounded: a fixed number of
// trials x frames, run in a worker thread under a wall-clock watchdog, so a
// kernel call that never returns fails the gate (naming the trial and the call
// count reached) instead of stalling the build.
//
// Usage:
//   node scripts/gate_abi_fuzz.mjs [--wasm=<path>] [--trials=N] [--frames=N] [--timeout=SECONDS]
//   (defaults: web/chronos_spectral.wasm, 300 trials, 48 frames per trial, 300 s)
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

if (isMainThread) {
  const args = process.argv.slice(2);
  const opt = (name, dflt) => {
    const a = args.find((x) => x.startsWith(`--${name}=`));
    return a ? a.slice(name.length + 3) : dflt;
  };
  const wasmPath = path.resolve(opt('wasm', path.join(ROOT, 'web', 'chronos_spectral.wasm')));
  const trials = Number(opt('trials', '300'));
  const frames = Number(opt('frames', '48'));
  const timeoutSec = Number(opt('timeout', '300'));
  if (![trials, frames, timeoutSec].every((v) => Number.isInteger(v) && v > 0)) {
    console.error('usage: node scripts/gate_abi_fuzz.mjs [--wasm=<path>] [--trials=N] [--frames=N] [--timeout=SECONDS]');
    process.exit(2);
  }
  // [trial, frame, export calls]: written by the worker, read here if the watchdog fires.
  const progress = new Int32Array(new SharedArrayBuffer(12));
  const t0 = Date.now();
  const worker = new Worker(fileURLToPath(import.meta.url), {
    workerData: { wasmBytes: fs.readFileSync(wasmPath), trials, frames, progress },
  });
  const fail = (msg) => {
    console.error(`  [FAIL] Gate X (ABI fuzz): ${msg}`);
    process.exit(1);
  };
  const watchdog = setTimeout(() => {
    const [trial, frame, calls] = progress;
    worker.terminate();
    fail(`no result after ${timeoutSec} s; stuck in trial ${trial}, frame ${frame}, after ${calls} export calls (a kernel call that does not return?)`);
  }, timeoutSec * 1000);
  worker.on('error', (e) => {
    clearTimeout(watchdog);
    fail(`fuzz worker failed: ${e?.stack || e}`);
  });
  worker.on('message', (r) => {
    clearTimeout(watchdog);
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    const fmt = (v) => JSON.stringify(v);
    const what = `${r.trials} trials x ${r.frames} frames, ${r.ticks.toLocaleString('en-US')} frames ticked, ${r.calls.toLocaleString('en-US')} export calls`;
    for (const [slot, v] of r.nonFinite) {
      console.log(`  [INFO] non-finite ${slot}: x${v.count}, first in trial ${v.first.trial} frame ${v.first.frame} = ${v.first.value}; calls before it: ${fmt(v.first.tail)}`);
    }
    if (r.traps.length) {
      for (const [key, v] of r.traps) console.error(`  [TRAP] x${v.count} ${key}; first in trial ${v.first.trial}; calls up to it: ${fmt(v.first.tail)}`);
      fail(`${r.traps.length} distinct trap(s) (${what}, ${secs} s)`);
    }
    console.log(
      `  [PASS] Gate X (Adversarial C-ABI fuzz over ${r.exports} exported functions): ${what}, 0 traps; ` +
        `non-finite telemetry: ${r.nonFinite.length} slot(s) (reported, not gated) (${secs} s)`
    );
  });
} else {
  const { wasmBytes, trials: TRIALS, frames: FRAMES, progress } = workerData;
  const mod = new WebAssembly.Module(wasmBytes);
  const exportNames = WebAssembly.Module.exports(mod).filter((e) => e.kind === 'function').map((e) => e.name);

  // mulberry32
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const HOSTILE = [
    0, 1, -1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 255, 256, 359, 360, 639, 640, 641, 1535, 1536, 1537,
    65535, 65536, 0x7fffffff, 0x80000000, 0xffffffff, NaN, Infinity, -Infinity, 1e-38, -1e-38, 1e38,
    -1e38, 0.5, -0.5, 1 / 30, 1 / 60, 1e-6, 1e6, 3.4e38, 44100, 48000, 96000, 192000, 1e-45,
  ];
  const DT = [1 / 30, 1 / 30, 1 / 30, 1 / 60, 1 / 24, 1 / 10, 0, -1 / 30, NaN, Infinity, -Infinity, 1e-9, 1e9, 5, 1e-40];

  // Exports that take arguments and mutate state; pointer getters are read separately.
  const noTick = exportNames.filter((n) => !/^chronos_(tick|pre_tick|post_tick|init)$/.test(n) && !/_ptr$/.test(n));
  const ptrGetters = exportNames.filter((n) => /_ptr$/.test(n));

  const F32_VIEWS = [
    ['chronos_get_telemetry_ptr', 1568],
    ['chronos_get_swarm_telemetry_ptr', 64],
    ['chronos_get_conservation_telemetry_ptr', 32],
    ['chronos_get_slot_meta_ptr', 128],
  ];
  const STAGING = [
    ['chronos_get_audio_staging_ptr', 4096],
    ['chronos_get_hodge_inject_ptr', 2640],
    ['chronos_get_conservation_input_ptr', 32],
    ['chronos_get_laplacian_ptr', 1024],
    ['chronos_get_fiedler_ptr', 32],
  ];

  const traps = new Map();
  const nonFinite = new Map();
  let calls = 0;
  let ticks = 0;

  function note(map, key, detail) {
    const e = map.get(key);
    if (e) e.count++;
    else map.set(key, { count: 1, first: detail });
  }

  function fillPixels(u8, w, h, kind, f, r) {
    const n = w * h * 4;
    switch (kind) {
      case 0:
        for (let i = 0; i < n; i++) u8[i] = (r() * 256) | 0;
        break;
      case 1:
        u8.fill(0, 0, n);
        break;
      case 2:
        u8.fill(255, 0, n);
        break;
      case 3: {
        // moving bright discs on a textured background
        for (let y = 0; y < h; y++)
          for (let x = 0; x < w; x++) {
            const o = (y * w + x) * 4;
            const bg = 60 + (((x * 7) ^ (y * 13)) & 31);
            u8[o] = bg;
            u8[o + 1] = bg;
            u8[o + 2] = bg;
            u8[o + 3] = 255;
          }
        for (let k = 0; k < 24; k++) {
          const cx = (((k * 53 + f * (3 + (k % 5))) % w) + w) % w;
          const cy = (((k * 31 + f * (2 + (k % 3))) % h) + h) % h;
          for (let dy = -3; dy <= 3; dy++)
            for (let dx = -3; dx <= 3; dx++) {
              const x = cx + dx;
              const y = cy + dy;
              if (x < 0 || y < 0 || x >= w || y >= h || dx * dx + dy * dy > 9) continue;
              const o = (y * w + x) * 4;
              u8[o] = 250;
              u8[o + 1] = 240;
              u8[o + 2] = 40;
            }
        }
        break;
      }
      case 4:
        // exposure strobe: alternating black / white frames
        u8.fill(f & 1 ? 255 : 0, 0, n);
        break;
      default:
        // vertical stripes shifting 1 px per frame, alpha 0
        for (let y = 0; y < h; y++)
          for (let x = 0; x < w; x++) {
            const o = (y * w + x) * 4;
            const v = ((x + f) >> 2) & 1 ? 230 : 20;
            u8[o] = v;
            u8[o + 1] = v;
            u8[o + 2] = v;
            u8[o + 3] = 0;
          }
    }
  }

  for (let trial = 0; trial < TRIALS; trial++) {
    const r = rng(0x9e3779b9 ^ (trial * 2654435761));
    const inst = new WebAssembly.Instance(mod, {});
    const e = inst.exports;
    const mem = e.memory;
    const hist = [];
    Atomics.store(progress, 0, trial);
    const call = (name, args) => {
      calls++;
      Atomics.store(progress, 2, calls);
      hist.push(`${name}(${args.map((v) => (Object.is(v, -0) ? '-0' : String(v))).join(',')})`);
      if (hist.length > 12) hist.shift();
      try {
        return e[name](...args);
      } catch (err) {
        note(traps, `${name}: ${err.message}`, { trial, tail: hist.slice() });
        throw err;
      }
    };
    try {
      call('chronos_init', []);
      if (r() < 0.5) call('chronos_set_resolution', [1]);
      const pixKind = (r() * 6) | 0;
      const hostility = r(); // fraction of frames that get hostile mutator calls
      for (let f = 0; f < FRAMES; f++) {
        Atomics.store(progress, 1, f);
        const w = e.chronos_get_frame_w();
        const h = e.chronos_get_frame_h();
        const u8 = new Uint8Array(mem.buffer, e.chronos_get_ingest_rgba_ptr(), w * h * 4);
        fillPixels(u8, w, h, r() < 0.9 ? pixKind : (r() * 6) | 0, f, r);

        if (r() < hostility) {
          const nCalls = 1 + ((r() * 4) | 0);
          for (let c = 0; c < nCalls; c++) {
            // hostile staging contents
            if (r() < 0.4) {
              const [g, len] = STAGING[(r() * STAGING.length) | 0];
              if (e[g]) {
                const v = new Float32Array(mem.buffer, e[g](), len);
                const mode = (r() * 4) | 0;
                for (let i = 0; i < len; i++)
                  v[i] = mode === 0 ? NaN : mode === 1 ? (r() - 0.5) * 1e30 : mode === 2 ? HOSTILE[(r() * HOSTILE.length) | 0] : r() - 0.5;
              }
            }
            const name = noTick[(r() * noTick.length) | 0];
            const arity = e[name].length;
            const args = [];
            for (let i = 0; i < arity; i++) args.push(HOSTILE[(r() * HOSTILE.length) | 0]);
            call(name, args);
          }
        }
        const dt = r() < 0.85 ? 1 / 30 : DT[(r() * DT.length) | 0];
        const style = r();
        if (style < 0.8) call('chronos_tick', [dt]);
        else if (style < 0.9) {
          call('chronos_pre_tick', [dt]);
          call('chronos_post_tick', [dt]);
        } else if (style < 0.95) {
          call('chronos_post_tick', [dt]); // out of order
          call('chronos_pre_tick', [dt]);
          call('chronos_pre_tick', [dt]);
        } else {
          call('chronos_pre_tick', [dt]);
          call('chronos_tick', [dt]);
          call('chronos_post_tick', [dt]);
        }
        ticks++;
        for (const g of ptrGetters) e[g](g === 'chronos_get_ring_slot_ptr' ? HOSTILE[(r() * HOSTILE.length) | 0] : undefined);
        for (const [g, len] of F32_VIEWS) {
          if (!e[g]) continue;
          const v = new Float32Array(mem.buffer, e[g](), len);
          for (let i = 0; i < len; i++) {
            if (!Number.isFinite(v[i])) {
              note(nonFinite, `${g.replace(/^chronos_get_|_ptr$/g, '')}[${i}]`, { trial, frame: f, value: String(v[i]), tail: hist.slice() });
            }
          }
        }
        if (mem.buffer.byteLength !== 448 * 65536) note(traps, 'memory size changed', { trial, tail: hist.slice() });
      }
    } catch {
      /* recorded above; the next trial uses a fresh instance */
    }
  }

  parentPort.postMessage({
    trials: TRIALS,
    frames: FRAMES,
    exports: exportNames.length,
    ticks,
    calls,
    traps: [...traps],
    nonFinite: [...nonFinite].sort((a, b) => b[1].count - a[1].count),
  });
}
