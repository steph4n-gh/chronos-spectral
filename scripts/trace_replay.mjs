#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Reference trace replayer (CSTRACE v1).
//
// Replays recorded kernel sessions against a fresh instance with no
// JavaScript scene code: apply each host write (region + offset, region
// address resolved by calling its getter now), make each call with the
// recorded arguments, and check the recorded return value and per-region
// XXH32 digests. This is the reference the native Rust replayer follows.
//
// Usage:
//   node scripts/trace_replay.mjs [--wasm=<path>] [--stats] [--portable] [--max-errors=N]
//        [--allow-wasm-mismatch] <trace|dir>...
//   node scripts/trace_replay.mjs --bench [--warmup=N] [--passes=N] [--bench-reconcile=K]
//        [--bench-verbose] [--json=FILE] [--bench-cpu-prof=FILE] [--wasm=<path>] <trace|dir>...
//
// --bench times the kernel calls instead of verifying them, measuring exactly
// what `chronos_replay --bench` measures natively (see benchTrace below and
// docs/VERIFICATION_AND_BENCHMARKS.md, "Native vs wasm kernel timing").
//
// Exit status: 0 when every trace replays with 0 mismatches, 1 otherwise.
// UNMAPPED (raw-address) host writes are applied and reported as non-portable;
// --portable turns them into failures.
//
// --allow-wasm-mismatch replays traces recorded against a different wasm build
// (for example sessions recorded on the previous release, replayed on a new
// one) instead of refusing them. Everything is still checked, pointer results
// included: equal pointers mean every region is where it was, which UNMAPPED
// writes rely on. Whole-memory digests cover the data section too, so they are
// skipped (and counted) on a different build.
// ============================================================================

import crypto from 'node:crypto';
import fs from 'node:fs';
import inspector from 'node:inspector';
import path from 'node:path';
import { DEFAULT_WASM, END_REASON, FFLAG, RFLAG, TraceReader, decodeValues, fmtValue, xxh32, xxh32Masked } from './trace_lib.mjs';

// If trace_hook.mjs was preloaded, use the unpatched API so the replay itself is not recorded.
const WA = globalThis[Symbol.for('chronos.trace.originals')] || WebAssembly;

const args = process.argv.slice(2);
const wasmPath = path.resolve((args.find((a) => a.startsWith('--wasm=')) || `--wasm=${DEFAULT_WASM}`).slice(7));
const maxErrors = Number((args.find((a) => a.startsWith('--max-errors=')) || '--max-errors=1').slice(13));
const showStats = args.includes('--stats');
const requirePortable = args.includes('--portable');
const allowWasmMismatch = args.includes('--allow-wasm-mismatch');
const inputs = args.filter((a) => !a.startsWith('--'));
const optVal = (k, d) => {
  const a = args.find((x) => x.startsWith(`--${k}=`));
  return a === undefined ? d : a.slice(k.length + 3);
};
const benchMode = args.includes('--bench');
if (!inputs.length) {
  console.error('usage: node scripts/trace_replay.mjs [--wasm=<path>] [--stats] [--max-errors=N] <trace|dir>...');
  process.exit(2);
}
const files = inputs.flatMap((p) =>
  fs.statSync(p).isDirectory()
    ? fs.readdirSync(p).filter((f) => f.endsWith('.cstrace')).sort().map((f) => path.join(p, f))
    : [p]
);
if (!files.length) {
  console.error(`FAIL: no .cstrace files found in: ${inputs.join(', ')} (pass each scenario directory or trace file)`);
  process.exit(2);
}

const wasmBytes = fs.readFileSync(wasmPath);
const wasmSha = crypto.createHash('sha256').update(wasmBytes).digest('hex');
const wasmModule = new WA.Module(wasmBytes);

function replay(file) {
  const rd = new TraceReader(file);
  const h = rd.header;
  const name = path.basename(file);
  const otherWasm = h.wasmSha256 !== wasmSha;
  if (otherWasm && !allowWasmMismatch) {
    throw new Error(`${name}: trace was recorded against wasm ${h.wasmSha256}, replay wasm is ${wasmSha} (--allow-wasm-mismatch replays it anyway)`);
  }

  const E = new WA.Instance(wasmModule, {}).exports;
  const mem = new Uint8Array(E.memory.buffer);
  if (mem.length !== h.memBytes) throw new Error(`${name}: memory ${mem.length} B != recorded ${h.memBytes} B`);
  const pristine = Uint8Array.from(mem); // fresh-instance image, for RESET records
  const fns = h.funcs.map((f) => {
    if (typeof E[f.name] !== 'function') throw new Error(`${name}: export ${f.name} missing`);
    return E[f.name];
  });
  // Region address = its getter's return value right now (ingest_rgba moves with the ring cursor).
  const regionAddr = (ri) => {
    const r = h.regions[ri];
    const g = fns[r.getterIndex];
    return (r.hasArg ? g(r.arg) : g()) >>> 0;
  };
  const digested = h.digested.map((ri) => h.regions[ri]);
  // Format 1.1: the recorder zeroed these platform-descriptive words before hashing.
  const mask = h.meta.digestMask || {};
  for (const [rn, words] of Object.entries(mask)) {
    const r = h.regions.find((x) => x.name === rn);
    if (!r || r.elem === 1 || !words.every((w) => Number.isInteger(w) && w >= 0 && 4 * w + 4 <= r.byteLen)) {
      throw new Error(`${name}: invalid meta.digestMask entry ${rn}: ${JSON.stringify(words)}`);
    }
  }
  const maskWords = digested.map((r) => mask[r.name] || null);
  const maskScratch = new Uint8Array(Math.max(0, ...digested.filter((r, i) => maskWords[i]).map((r) => r.byteLen)));
  // Digest cache: a region's hash is reused while its bytes are unchanged.
  const prevBytes = digested.map(() => null);
  const prevHash = new Uint32Array(digested.length);

  const st = { firstUnmappedCall: -1, resets: 0, memChecks: 0, memSkipped: 0, calls: 0, digestChecks: 0, writes: 0, writeBytes: 0, unmapped: 0, errors: 0, perRegionWriteBytes: new Map() };
  const fail = (msg) => {
    st.errors++;
    if (st.errors <= maxErrors) console.error(`  [MISMATCH] ${name}: ${msg}`);
  };
  let end = null;
  let eof = null;
  for (const rec of rd.records()) {
    if (rec.tag === 'W') {
      const r = h.regions[rec.region];
      if (!r) throw new Error(`${name}: bad region ${rec.region} at byte ${rec.pos}`);
      if (rec.offset + rec.data.length > r.byteLen) throw new Error(`${name}: write overruns region ${r.name} at byte ${rec.pos}`);
      mem.set(rec.data, regionAddr(rec.region) + rec.offset);
      st.writes++;
      st.writeBytes += rec.data.length;
      st.perRegionWriteBytes.set(r.name.replace(/\[\d+\]$/, '[*]'), (st.perRegionWriteBytes.get(r.name.replace(/\[\d+\]$/, '[*]')) || 0) + rec.data.length);
    } else if (rec.tag === 'R') {
      // The host restored the fresh-instance image (natively: a fresh kernel).
      mem.set(pristine);
      st.resets++;
    } else if (rec.tag === 'U') {
      // Raw-address write: replayable only against this exact wasm build, never natively.
      mem.set(rec.data, rec.addr);
      if (st.unmapped++ === 0) st.firstUnmappedCall = st.calls;
    } else if (rec.tag === 'C') {
      const f = h.funcs[rec.func];
      if (rec.seq !== st.calls) throw new Error(`${name}: call sequence ${rec.seq} != expected ${st.calls}`);
      const callArgs = decodeValues(rec.args, f.params);
      const where = `call #${rec.seq} ${f.name}(${callArgs.map((v, k) => fmtValue(f.params[k], v)).join(', ')})`;
      let result;
      let status = 0;
      try {
        result = fns[rec.func](...callArgs);
      } catch (e) {
        status = 1;
        result = e;
      }
      st.calls++;
      if (status !== rec.status) {
        fail(`${where}: recorded ${rec.status ? 'trap' : 'return'}, replay ${status ? `trap (${result?.message})` : 'returned'}`);
      } else if (rec.result) {
        const expect = decodeValues(rec.result, f.results)[0];
        const t = f.results[0];
        // NaN payloads are not observable through the JS API: compare NaN as NaN.
        const same = t === 0x7f ? (result | 0) === expect : Object.is(result, expect) || (Number.isNaN(result) && Number.isNaN(expect));
        if (!same) fail(`${where}: return ${fmtValue(t, result)} != recorded ${fmtValue(t, expect)}${f.flags & FFLAG.POINTER_RESULT ? ' (pointer)' : ''}`);
      }
      if (rec.digests) {
        st.digestChecks++;
        for (let i = 0; i < digested.length; i++) {
          const r = digested[i];
          const a = regionAddr(h.digested[i]);
          const cur = mem.subarray(a, a + r.byteLen);
          let hash;
          if (prevBytes[i] && Buffer.compare(prevBytes[i], cur) === 0) hash = prevHash[i];
          else {
            hash = xxh32Masked(mem, a, r.byteLen, maskWords[i], maskScratch);
            prevBytes[i] = Buffer.from(cur);
            prevHash[i] = hash;
          }
          if (hash !== rec.digests[i]) {
            fail(`${where}: region ${r.name} digest ${hash.toString(16).padStart(8, '0')} != recorded ${rec.digests[i].toString(16).padStart(8, '0')}`);
          }
        }
      }
      if (rec.memDigest !== null && otherWasm) {
        st.memSkipped++;
      } else if (rec.memDigest !== null) {
        // Validation traces: all memory above the shadow stack must match too.
        const lo = h.meta.stackTop;
        const hash = xxh32(mem, lo, mem.length - lo);
        st.memChecks++;
        if (hash !== rec.memDigest) fail(`${where}: whole-memory digest (above ${lo}) differs: a host write was not captured`);
      }
      if (st.errors >= maxErrors) break;
    } else if (rec.tag === 'E') {
      end = rec;
    } else if (rec.tag === 'EOF') {
      eof = rec;
    }
  }
  rd.close();
  if (end && st.errors === 0) {
    if (end.nCalls !== st.calls || end.nWrites !== st.writes || end.nUnmapped !== st.unmapped) {
      fail(`END record counts (calls=${end.nCalls}, writes=${end.nWrites}, unmapped=${end.nUnmapped}) disagree with replay (${st.calls}, ${st.writes}, ${st.unmapped})`);
    }
  }
  const size = fs.statSync(file).size;
  const endTxt = end
    ? END_REASON[end.reason]
    : !eof
      ? 'stopped at mismatch'
      : eof.torn
        ? 'torn tail (prefix replayed)'
        : 'no END record (prefix replayed)';
  const ok = st.errors === 0 && (st.unmapped === 0 || !requirePortable);
  console.log(
    `${ok ? '[PASS]' : '[FAIL]'} ${name}: ${st.calls} calls (${st.digestChecks} digest-checked x ${digested.length} regions` +
      `${st.memChecks ? `, ${st.memChecks} whole-memory` : ''}), ` +
      `${st.writes} host writes (${(st.writeBytes / 1e6).toFixed(2)} MB), ${st.resets} resets, ${st.unmapped} unmapped, ${(size / 1e6).toFixed(1)} MB, end=${endTxt}, mismatches=${st.errors}`
  );
  if (otherWasm) {
    console.log(
      `  [NOTE] ${name}: recorded against wasm ${h.wasmSha256.slice(0, 16)}..., replayed on ${wasmSha.slice(0, 16)}... (--allow-wasm-mismatch)` +
        `${st.memSkipped ? `; ${st.memSkipped} whole-memory digest(s) skipped` : ''}`
    );
  }
  if (st.unmapped) {
    console.log(
      `  [${requirePortable ? 'FAIL' : 'WARN'}] ${name}: ${st.unmapped} UNMAPPED raw-address write(s); ` +
        `portable to a native build only up to call #${st.firstUnmappedCall}`
    );
  }
  if (showStats) {
    const rows = [...st.perRegionWriteBytes].sort((a, b) => b[1] - a[1]);
    for (const [r, b] of rows) console.log(`    write bytes ${r.padEnd(24)} ${b}`);
  }
  return ok;
}

// ----------------------------------------------------------------------------
// --bench: per-call kernel timing (the wasm side of scripts/bench_native_vs_wasm.mjs).
//
// Mirrors native/replay/bench.rs record for record:
//  * Load (untimed): records are read once into memory. Loading stops where a
//    native replay must stop, at the first UNMAPPED (raw-address) write or the
//    first recorded trap, so both sides time the identical call sequence.
//  * Each pass restores the fresh-instance image (as a RESET record does) and
//    replays every op. Host writes are applied untimed; each kernel call is
//    timed alone with performance.now() (the monotonic clock: on macOS the same
//    24 MHz mach timebase as the native Instant). Return values are compared
//    with the recording outside the timed region; digests are not computed.
//  * --warmup passes are discarded (they also let V8 tier the hot functions up
//    from Liftoff to TurboFan); --passes passes are measured.
//  * --bench-reconcile=K: K passes without per-call timers and K passes that
//    apply the writes but make no call; their wall-clock difference must agree
//    with the summed per-call times.
// The mean empty-interval cost of the timer is measured at start-up and
// subtracted from every sample (clamped at 0).
// ----------------------------------------------------------------------------

function calibrateTimer() {
  const N = 200000;
  let sum = 0;
  let minNz = Infinity;
  for (let rep = 0; rep < 2; rep++) {
    sum = 0;
    for (let i = 0; i < N; i++) {
      const t0 = performance.now();
      const t1 = performance.now();
      const d = (t1 - t0) * 1e6;
      sum += d;
      if (d > 0 && d < minNz) minNz = d;
    }
  }
  return { overheadNs: sum / N, resolutionNs: minNz };
}

function benchLoad(file) {
  const rd = new TraceReader(file);
  const h = rd.header;
  const ops = [];
  let calls = 0;
  let writes = 0;
  let writeBytes = 0;
  let stop = 'end';
  for (const rec of rd.records()) {
    if (rec.tag === 'W') {
      ops.push({ k: 0, region: rec.region, offset: rec.offset, data: rec.data });
      writes++;
      writeBytes += rec.data.length;
    } else if (rec.tag === 'R') {
      ops.push({ k: 1 });
    } else if (rec.tag === 'U') {
      stop = `prefix@${calls} (UNMAPPED write)`;
      break;
    } else if (rec.tag === 'C') {
      if (rec.status !== 0) {
        stop = `prefix@${calls} (recorded trap)`;
        break;
      }
      const f = h.funcs[rec.func];
      ops.push({
        k: 2,
        func: rec.func,
        args: decodeValues(rec.args, f.params),
        result: rec.result ? decodeValues(rec.result, f.results)[0] : undefined,
        isF32: f.results[0] === 0x7d,
      });
      calls++;
    }
  }
  rd.close();
  return { h, ops, calls, writes, writeBytes, stop };
}

function invokeExport(fn, a) {
  switch (a.length) {
    case 0: return fn();
    case 1: return fn(a[0]);
    case 2: return fn(a[0], a[1]);
    case 3: return fn(a[0], a[1], a[2]);
    case 4: return fn(a[0], a[1], a[2], a[3]);
    case 5: return fn(a[0], a[1], a[2], a[3], a[4]);
    case 6: return fn(a[0], a[1], a[2], a[3], a[4], a[5]);
    default: return fn(...a);
  }
}

const MODE_TIMED = 0;
const MODE_UNTIMED = 1;
const MODE_APPLY_ONLY = 2;

function benchPass(ctx, ld, mode, overheadNs, samples) {
  const { mem, pristine, fns, regionAddr } = ctx;
  const ops = ld.ops;
  let kernelNs = 0;
  let mismatches = 0;
  mem.set(pristine);
  const w0 = performance.now();
  for (let i = 0; i < ops.length; i++) {
    const op = ops[i];
    if (op.k === 0) {
      mem.set(op.data, regionAddr(op.region) + op.offset);
    } else if (op.k === 1) {
      mem.set(pristine);
    } else {
      if (mode === MODE_APPLY_ONLY) continue;
      const fn = fns[op.func];
      let r;
      if (mode === MODE_TIMED) {
        const t0 = performance.now();
        r = invokeExport(fn, op.args);
        const t1 = performance.now();
        let ns = (t1 - t0) * 1e6 - overheadNs;
        if (ns < 0) ns = 0;
        kernelNs += ns;
        if (samples) samples[op.func].push(ns);
      } else {
        r = invokeExport(fn, op.args);
      }
      if (op.result !== undefined) {
        const same = op.isF32 ? Object.is(r, op.result) || (Number.isNaN(r) && Number.isNaN(op.result)) : (r | 0) === op.result;
        if (!same) mismatches++;
      }
    }
  }
  return { wallNs: (performance.now() - w0) * 1e6, kernelNs, mismatches };
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
const medianOf = (v) => quantile([...v].sort((a, b) => a - b), 0.5);

function benchMain() {
  const warmup = Number(optVal('warmup', '2'));
  const passes = Number(optVal('passes', '5'));
  const reconcile = Number(optVal('bench-reconcile', '0'));
  const verbose = args.includes('--bench-verbose');
  const jsonOut = optVal('json', null);
  if (!(passes >= 1) || !(warmup >= 0) || !(reconcile >= 0)) {
    console.error('bad --passes / --warmup / --bench-reconcile');
    process.exit(2);
  }
  const { overheadNs, resolutionNs } = calibrateTimer();
  const engine = `node ${process.version} (V8 ${process.versions.v8})${process.execArgv.length ? ` ${process.execArgv.join(' ')}` : ''}`;
  console.log(
    `trace_replay --bench: wasm ${wasmSha.slice(0, 16)} on ${engine}; warmup ${warmup} + ${passes} measured pass(es) per trace; ` +
      `timer: performance.now(), empty-interval mean ${overheadNs.toFixed(1)} ns (subtracted per call), resolution ${resolutionNs.toFixed(1)} ns`
  );
  const out = [];
  let failed = false;
  for (const file of files) {
    const name = path.basename(file);
    const tl = performance.now();
    const ld = benchLoad(file);
    const loadS = (performance.now() - tl) / 1e3;
    const h = ld.h;
    if (h.wasmSha256 !== wasmSha && !allowWasmMismatch) {
      console.log(`[FAIL] ${name}: trace was recorded against wasm ${h.wasmSha256}, replay wasm is ${wasmSha} (--allow-wasm-mismatch times it anyway)`);
      failed = true;
      continue;
    }
    const E = new WA.Instance(wasmModule, {}).exports;
    const mem = new Uint8Array(E.memory.buffer);
    const pristine = Uint8Array.from(mem);
    const fns = h.funcs.map((f) => E[f.name]);
    const getters = h.regions.map((r) => {
      const g = fns[r.getterIndex];
      return r.hasArg ? () => g(r.arg) >>> 0 : () => g() >>> 0;
    });
    const ctx = { mem, pristine, fns, regionAddr: (ri) => getters[ri]() };
    const samples = h.funcs.map(() => []);
    // --bench-cpu-prof: V8 CPU profile of the measured passes only (after tier-up).
    const cpuProf = optVal('bench-cpu-prof', null);
    let session = null;
    const passKernel = [];
    const passWall = [];
    const warmKernel = [];
    let mismatches = 0;
    for (let p = 0; p < warmup + passes; p++) {
      if (typeof globalThis.gc === 'function') globalThis.gc();
      const measured = p >= warmup;
      if (cpuProf && p === warmup) {
        session = new inspector.Session();
        session.connect();
        // The inspector answers synchronously while the main thread is busy in post().
        session.post('Profiler.enable');
        session.post('Profiler.setSamplingInterval', { interval: Number(optVal('bench-cpu-prof-interval', '100')) });
        session.post('Profiler.start');
      }
      const o = benchPass(ctx, ld, MODE_TIMED, overheadNs, measured ? samples : null);
      mismatches += o.mismatches;
      if (measured) {
        passKernel.push(o.kernelNs);
        passWall.push(o.wallNs);
      } else warmKernel.push(o.kernelNs);
    }
    if (session) {
      session.post('Profiler.stop', (e, r) => {
        if (e) throw e;
        fs.writeFileSync(files.length > 1 ? `${cpuProf}.${name}.cpuprofile` : cpuProf, JSON.stringify(r.profile));
      });
      session.disconnect();
      session = null;
    }
    const recUntimed = [];
    const recApply = [];
    for (let k = 0; k < reconcile; k++) {
      if (typeof globalThis.gc === 'function') globalThis.gc();
      const u = benchPass(ctx, ld, MODE_UNTIMED, overheadNs, null);
      mismatches += u.mismatches;
      recUntimed.push(u.wallNs);
      if (typeof globalThis.gc === 'function') globalThis.gc();
      recApply.push(benchPass(ctx, ld, MODE_APPLY_ONLY, overheadNs, null).wallNs);
    }
    if (mismatches) failed = true;
    const km = medianOf(passKernel);
    const wm = medianOf(passWall);
    console.log(
      `[${mismatches ? 'FAIL ' : 'BENCH'}] ${name}: ${ld.calls} calls, ${ld.writes} host writes (${(ld.writeBytes / 1e6).toFixed(1)} MB), ` +
        `stop=${ld.stop}, loaded in ${loadS.toFixed(1)} s; kernel ${(km / 1e6).toFixed(3)} ms/pass (median of ${passKernel.length}; ` +
        `min ${(Math.min(...passKernel) / 1e6).toFixed(3)}, max ${(Math.max(...passKernel) / 1e6).toFixed(3)}), pass wall ${(wm / 1e6).toFixed(3)} ms, ` +
        `return mismatches ${mismatches}`
    );
    if (reconcile) {
      const u = medianOf(recUntimed);
      const a = medianOf(recApply);
      console.log(
        `    reconcile: untimed pass ${(u / 1e6).toFixed(3)} ms - apply-only pass ${(a / 1e6).toFixed(3)} ms = ${((u - a) / 1e6).toFixed(3)} ms ` +
          `kernel by wall-clock vs ${(km / 1e6).toFixed(3)} ms summed per call (${(km - (u - a) >= 0 ? '+' : '') + ((100 * (km - (u - a))) / Math.max(1, u - a)).toFixed(1)}%); ` +
          `timed pass wall ${(wm / 1e6).toFixed(3)} ms`
      );
    }
    const exportsOut = [];
    samples.forEach((s, fi) => {
      if (!s.length) return;
      s.sort((a, b) => a - b);
      const n = s.length;
      const sum = s.reduce((a, b) => a + b, 0);
      exportsOut.push({
        name: h.funcs[fi].name,
        count: Math.floor(n / passes),
        median_ns: quantile(s, 0.5),
        p95_ns: quantile(s, 0.95),
        mean_ns: sum / n,
        total_ns_per_pass: sum / passes,
        min_ns: s[0],
        max_ns: s[n - 1],
        // Every measured sample (sorted), so statistics can be pooled exactly across traces and runs.
        samples_ns: s.map((x) => Math.round(x)),
      });
    });
    exportsOut.sort((a, b) => b.total_ns_per_pass - a.total_ns_per_pass);
    if (verbose) {
      console.log(`    ${'export'.padEnd(36)} ${'calls'.padStart(8)} ${'median us'.padStart(12)} ${'p95 us'.padStart(12)} ${'mean us'.padStart(12)} ${'share'.padStart(9)}`);
      for (const e of exportsOut) {
        console.log(
          `    ${e.name.padEnd(36)} ${String(e.count).padStart(8)} ${(e.median_ns / 1e3).toFixed(3).padStart(12)} ${(e.p95_ns / 1e3).toFixed(3).padStart(12)} ` +
            `${(e.mean_ns / 1e3).toFixed(3).padStart(12)} ${((100 * e.total_ns_per_pass) / Math.max(1, km)).toFixed(1).padStart(8)}%`
        );
      }
    }
    out.push({
      name,
      calls: ld.calls,
      writes: ld.writes,
      write_bytes: ld.writeBytes,
      stop: ld.stop,
      mismatches,
      warmup_kernel_ns: warmKernel,
      pass_kernel_ns: passKernel,
      pass_wall_ns: passWall,
      reconcile_untimed_ns: recUntimed,
      reconcile_apply_ns: recApply,
      exports: exportsOut,
    });
  }
  if (jsonOut) {
    fs.writeFileSync(
      jsonOut,
      JSON.stringify({
        side: 'wasm',
        backend: 'wasm-simd128',
        engine,
        wasm_sha256: wasmSha,
        timer: { clock: 'performance.now', overhead_ns: overheadNs, resolution_ns: resolutionNs },
        warmup,
        passes,
        traces: out,
      }) + '\n'
    );
  }
  process.exit(failed ? 1 : 0);
}

if (benchMode) benchMain();

let allOk = true;
for (const f of files) {
  try {
    allOk = replay(f) && allOk;
  } catch (e) {
    allOk = false;
    console.log(`[FAIL] ${path.basename(f)}: ${e.message}`);
  }
}
console.log(allOk ? `[PASS] ${files.length} trace(s) replayed with 0 mismatches` : '[FAIL] trace replay found mismatches');
process.exit(allOk ? 0 : 1);
