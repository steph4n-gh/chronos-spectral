#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Dump a kernel region from a wasm replay, and diff it
// against the native replay driver's dump of the same region at the same call.
//
//   node scripts/trace_region_diff.mjs <trace> --call=N --region=NAME[,NAME...]
//        [--native=<file>] [--out-dir=DIR] [--wasm=<path>] [--max=N]
//
// Replays <trace> against the wasm (default web/chronos_spectral.wasm) up to and
// including call N, then writes each region's raw bytes to
// <out-dir>/<trace-stem>.c<N>.<region>.wasm.bin (out-dir defaults to the
// directory of --native, else target/replay-dumps). With --native (one region),
// compares the two dumps element by element (u8 / u32 / f32 by the region's
// element type) and prints every differing index up to --max (default 40),
// with values and bit patterns. The native driver (native/replay) prints the
// exact command for each region whose digest differs.
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { DEFAULT_WASM, PLATFORM_WORDS, ROOT, TraceReader, decodeValues } from './trace_lib.mjs';

const WA = globalThis[Symbol.for('chronos.trace.originals')] || WebAssembly;

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const a = args.find((x) => x.startsWith(`--${name}=`));
  return a ? a.slice(name.length + 3) : dflt;
};
const traceFile = args.find((a) => !a.startsWith('--'));
const callN = Number(opt('call', NaN));
const regionNames = (opt('region', '') || '').split(',').filter(Boolean);
const nativeFile = opt('native', null);
const wasmPath = path.resolve(opt('wasm', DEFAULT_WASM));
const maxShow = Number(opt('max', '40'));
if (!traceFile || !Number.isInteger(callN) || callN < 0 || !regionNames.length || (nativeFile && regionNames.length !== 1)) {
  console.error('usage: node scripts/trace_region_diff.mjs <trace> --call=N --region=NAME[,NAME...] [--native=<file>] [--out-dir=DIR] [--wasm=<path>] [--max=N]');
  process.exit(2);
}
const outDir = path.resolve(opt('out-dir', nativeFile ? path.dirname(nativeFile) : path.join(ROOT, 'target', 'replay-dumps')));

const rd = new TraceReader(traceFile);
const h = rd.header;
const E = new WA.Instance(new WA.Module(fs.readFileSync(wasmPath)), {}).exports;
const mem = new Uint8Array(E.memory.buffer);
const pristine = Uint8Array.from(mem);
const fns = h.funcs.map((f) => E[f.name]);
const regionAddr = (ri) => {
  const r = h.regions[ri];
  return (r.hasArg ? fns[r.getterIndex](r.arg) : fns[r.getterIndex]()) >>> 0;
};
const wanted = regionNames.map((n) => {
  const ri = h.regions.findIndex((r) => r.name === n);
  if (ri < 0) throw new Error(`unknown region ${n}; regions: ${h.regions.map((r) => r.name).join(', ')}`);
  return ri;
});

let reached = false;
for (const rec of rd.records()) {
  if (rec.tag === 'W') mem.set(rec.data, regionAddr(rec.region) + rec.offset);
  else if (rec.tag === 'R') mem.set(pristine);
  else if (rec.tag === 'U') mem.set(rec.data, rec.addr);
  else if (rec.tag === 'C') {
    const f = h.funcs[rec.func];
    try {
      fns[rec.func](...decodeValues(rec.args, f.params));
    } catch {
      // A recorded trap replays as a trap; the state after it is what the recording saw.
    }
    if (rec.seq === callN) {
      reached = true;
      break;
    }
  }
}
rd.close();
if (!reached) {
  console.error(`trace has no call #${callN}`);
  process.exit(1);
}

fs.mkdirSync(outDir, { recursive: true });
const stem = path.basename(traceFile).replace(/\.cstrace$/, '');
const safe = (s) => s.replace(/[^A-Za-z0-9_-]/g, '_');
const dumps = wanted.map((ri) => {
  const r = h.regions[ri];
  const a = regionAddr(ri);
  const bytes = Buffer.from(mem.subarray(a, a + r.byteLen));
  const file = path.join(outDir, `${stem}.c${callN}.${safe(r.name)}.wasm.bin`);
  fs.writeFileSync(file, bytes);
  console.log(`wasm ${r.name} after call #${callN}: ${r.byteLen} B -> ${file}`);
  return { r, bytes };
});

if (nativeFile) {
  const { r, bytes: w } = dumps[0];
  const n = fs.readFileSync(nativeFile);
  if (n.length !== w.length) {
    console.log(`[DIFF] sizes differ: wasm ${w.length} B, native ${n.length} B`);
    process.exit(1);
  }
  const masked = new Set(PLATFORM_WORDS.filter((p) => p.region === r.name).map((p) => p.word));
  const es = r.elem === 1 ? 1 : 4;
  const count = w.length / es;
  const diffs = [];
  for (let i = 0; i < count; i++) {
    const o = i * es;
    if (es === 1 ? w[o] !== n[o] : w.readUInt32LE(o) !== n.readUInt32LE(o)) diffs.push(i);
  }
  const fmt = (b, i) => {
    const o = i * es;
    if (r.elem === 1) return String(b[o]);
    const bits = b.readUInt32LE(o).toString(16).padStart(8, '0');
    return r.elem === 3 ? `${b.readFloatLE(o)} (0x${bits})` : `${b.readUInt32LE(o)} (0x${bits})`;
  };
  const real = diffs.filter((i) => !(es === 4 && masked.has(i)));
  console.log(`${r.name}: ${diffs.length} of ${count} ${['', 'u8', 'u32', 'f32'][r.elem]} elements differ` + (diffs.length !== real.length ? ` (${diffs.length - real.length} of them allowlisted platform words, masked in digests)` : ''));
  for (const i of diffs.slice(0, maxShow)) {
    const pix = r.elem === 1 && r.name.includes('rgba') ? ` (pixel ${Math.floor(i / 4)}, channel ${'RGBA'[i % 4]})` : '';
    console.log(`  [${i}]${pix}${masked.has(i) && es === 4 ? ' [platform word]' : ''}: wasm ${fmt(w, i)}  native ${fmt(n, i)}`);
  }
  if (diffs.length > maxShow) console.log(`  ... ${diffs.length - maxShow} more (last index ${diffs[diffs.length - 1]})`);
  process.exit(real.length ? 1 : 0);
}
