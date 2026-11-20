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
//   node scripts/trace_replay.mjs [--wasm=<path>] [--stats] [--portable] [--max-errors=N] <trace|dir>...
//
// Exit status: 0 when every trace replays with 0 mismatches, 1 otherwise.
// UNMAPPED (raw-address) host writes are applied and reported as non-portable;
// --portable turns them into failures.
// ============================================================================

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { DEFAULT_WASM, END_REASON, FFLAG, RFLAG, TraceReader, decodeValues, fmtValue, xxh32, xxh32Masked } from './trace_lib.mjs';

// If trace_hook.mjs was preloaded, use the unpatched API so the replay itself is not recorded.
const WA = globalThis[Symbol.for('chronos.trace.originals')] || WebAssembly;

const args = process.argv.slice(2);
const wasmPath = path.resolve((args.find((a) => a.startsWith('--wasm=')) || `--wasm=${DEFAULT_WASM}`).slice(7));
const maxErrors = Number((args.find((a) => a.startsWith('--max-errors=')) || '--max-errors=1').slice(13));
const showStats = args.includes('--stats');
const requirePortable = args.includes('--portable');
const inputs = args.filter((a) => !a.startsWith('--'));
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
  if (h.wasmSha256 !== wasmSha) throw new Error(`${name}: trace was recorded against wasm ${h.wasmSha256}, replay wasm is ${wasmSha}`);

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

  const st = { firstUnmappedCall: -1, resets: 0, memChecks: 0, calls: 0, digestChecks: 0, writes: 0, writeBytes: 0, unmapped: 0, errors: 0, perRegionWriteBytes: new Map() };
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
      if (rec.memDigest !== null) {
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
