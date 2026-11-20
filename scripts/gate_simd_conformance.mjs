#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate S: SIMD Layer Conformance (scalar reference vs real Wasm SIMD128)
//
// Verifies, inside this Node (V8) engine:
//   1. Source hygiene (zero em dashes, zero banned codenames) of this script and src/simd/
//   2. A conformance build (`--features simd-conformance`, own target dir, never web/) with
//      zero imports. LLVM's auto-vectorisers are switched off for it, so `simd::scalar` runs as
//      scalar code and is compared against the engine's SIMD128, not against itself
//   3. Every item of the SIMD layer has a conformance entry
//   4. Every op of `core::arch::wasm32` as executed by the engine matches `simd::scalar` on
//      edge-value and seeded random lanes (NaN results compare as "both NaN" where wasm leaves
//      the payload open; everything else bitwise)
//   5. Every negative control (a deliberately wrong scalar op) is detected, and only in the
//      op it breaks, so a green result means something
//
// Usage:
//   node scripts/gate_simd_conformance.mjs [--wasm=<path>] [--no-build] [--seed=<hex>]
//                                           [--random=<n>] [--negative-random=<n>]
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TARGET_DIR = path.join(ROOT, 'target', 'simd-conformance');
const DEFAULT_WASM = path.join(TARGET_DIR, 'wasm32-unknown-unknown', 'release', 'chronos_spectral.wasm');

const args = process.argv.slice(2);
const argValue = (name) => {
  const a = args.find((x) => x.startsWith(`--${name}=`));
  return a ? a.slice(name.length + 3) : undefined;
};
const WASM_PATH = argValue('wasm') ? path.resolve(argValue('wasm')) : DEFAULT_WASM;
const BUILD = !args.includes('--no-build') && argValue('wasm') === undefined;
const SEED = BigInt(`0x${(argValue('seed') ?? '5EEDC0DE20261119').replace(/^0x/i, '')}`);
const N_RANDOM = Number(argValue('random') ?? 10000);
const N_RANDOM_NEGATIVE = Number(argValue('negative-random') ?? 2000);

function assert(cond, msg) {
  if (!cond) {
    throw new Error(`Gate S FAIL: ${msg}`);
  }
}

// ----------------------------------------------------------------------------
// 1. Source hygiene
// ----------------------------------------------------------------------------
{
  const bannedRe = new RegExp(
    `(${['CHRONOS', 'IMPOSSIBLE'].join('-')}|${['CHRONOS', '1000'].join('-')})`
  );
  const files = [
    fileURLToPath(import.meta.url),
    ...fs.readdirSync(path.join(ROOT, 'src', 'simd')).map((f) => path.join(ROOT, 'src', 'simd', f)),
  ];
  for (const f of files) {
    const txt = fs.readFileSync(f, 'utf8');
    const rel = path.relative(ROOT, f);
    assert(!txt.includes('\u2014'), `${rel}: contains forbidden em dash`);
    assert(!bannedRe.test(txt), `${rel}: contains forbidden internal codename`);
  }
}

// ----------------------------------------------------------------------------
// 2. Conformance build (separate target dir: never touches web/ or the default artifact)
// ----------------------------------------------------------------------------
if (BUILD) {
  const t0 = performance.now();
  // `cargo rustc` adds the flags to this crate only, on top of the wasm32 rustflags in
  // .cargo/config.toml. Without them LLVM vectorises parts of `simd::scalar` into the very
  // SIMD128 instructions it is supposed to be checked against.
  const r = spawnSync(
    'cargo',
    [
      'rustc', '--lib', '--release', '--features', 'simd-conformance', '--target-dir', TARGET_DIR,
      '--', '-C', 'no-vectorize-slp', '-C', 'no-vectorize-loops',
    ],
    { cwd: ROOT, stdio: ['ignore', 'inherit', 'inherit'] }
  );
  assert(r.status === 0, `cargo rustc --features simd-conformance failed (exit ${r.status})`);
  console.log(`  [INFO] Built conformance module in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}
assert(fs.existsSync(WASM_PATH), `conformance Wasm not found: ${WASM_PATH}`);
assert(
  path.resolve(WASM_PATH) !== path.join(ROOT, 'web', 'chronos_spectral.wasm'),
  'refusing to treat the shipped web/chronos_spectral.wasm as the conformance build'
);

const wasmBytes = fs.readFileSync(WASM_PATH);
const module = await WebAssembly.compile(wasmBytes);
assert(WebAssembly.Module.imports(module).length === 0, 'conformance module must have zero imports');
const instance = await WebAssembly.instantiate(module, {});
const exp = instance.exports;
const mem = exp.memory;
for (const name of [
  'run', 'negative_control', 'negative_control_count', 'negative_control_target_ptr',
  'negative_control_target_len', 'report_ptr', 'stats_offset', 'records_offset', 'record_size',
  'entry_count', 'entry_name_ptr', 'entry_name_len', 'entry_variant_ptr', 'entry_variant_len',
  'entry_check', 'entry_lane_bytes', 'entry_is_float', 'layer_item_count',
  'uncovered_layer_items', 'distinct_ops',
]) {
  assert(typeof exp[`chronos_simd_conformance_${name}`] === 'function', `missing export chronos_simd_conformance_${name}`);
}
const C = (name, ...a) => exp[`chronos_simd_conformance_${name}`](...a);

const utf8 = new TextDecoder();
const str = (ptr, len) => utf8.decode(new Uint8Array(mem.buffer, ptr, len));
const CHECK_NAMES = ['exact', 'f32 lanes (NaN == NaN)', 'f64 lanes (NaN == NaN)'];

const nEntries = C('entry_count');
const entries = [];
for (let i = 0; i < nEntries; i++) {
  entries.push({
    name: str(C('entry_name_ptr', i), C('entry_name_len', i)),
    variant: str(C('entry_variant_ptr', i), C('entry_variant_len', i)),
    check: C('entry_check', i),
    laneBytes: C('entry_lane_bytes', i),
    isFloat: C('entry_is_float', i) === 1,
  });
}

// ----------------------------------------------------------------------------
// 3. Coverage
// ----------------------------------------------------------------------------
const nOps = C('distinct_ops');
const nLayer = C('layer_item_count');
assert(C('uncovered_layer_items') === 0, `${C('uncovered_layer_items')} layer items have no conformance entry`);
assert(nOps === nLayer - 1, `table covers ${nOps} ops but the layer has ${nLayer - 1} functions`);
console.log(`  [PASS] Coverage: all ${nLayer - 1} layer functions (+ type v128) have entries; ${nEntries} entries incl. every const-generic instantiation the kernel uses`);

// ----------------------------------------------------------------------------
// Report decoding
// ----------------------------------------------------------------------------
function readReport() {
  const base = C('report_ptr');
  const dv = new DataView(mem.buffer);
  const statsOff = base + C('stats_offset');
  const recOff = base + C('records_offset');
  const recSize = C('record_size');
  const r = {
    entries: dv.getUint32(base, true),
    recorded: dv.getUint32(base + 4, true),
    mismatches: dv.getUint32(base + 8, true),
    vectors: dv.getUint32(base + 12, true),
    stats: [],
    records: [],
  };
  for (let i = 0; i < r.entries; i++) {
    r.stats.push({ vectors: dv.getUint32(statsOff + 8 * i, true), mismatches: dv.getUint32(statsOff + 8 * i + 4, true) });
  }
  const bytes = (off) => new Uint8Array(mem.buffer.slice(off, off + 16));
  for (let k = 0; k < r.recorded; k++) {
    const o = recOff + k * recSize;
    r.records.push({
      entry: dv.getUint32(o, true),
      vector: dv.getUint32(o + 4, true),
      v: [bytes(o + 8), bytes(o + 24), bytes(o + 40)],
      s: bytes(o + 56),
      outA: bytes(o + 72),
      outB: bytes(o + 88),
    });
  }
  return r;
}

function lanes(b, width, isFloat) {
  const dv = new DataView(b.buffer, b.byteOffset, 16);
  const out = [];
  for (let i = 0; i < 16 / width; i++) {
    const o = i * width;
    let hex;
    let val;
    if (width === 1) {
      hex = dv.getUint8(o).toString(16).padStart(2, '0');
      val = dv.getInt8(o);
    } else if (width === 2) {
      hex = dv.getUint16(o, true).toString(16).padStart(4, '0');
      val = dv.getInt16(o, true);
    } else if (width === 4) {
      hex = dv.getUint32(o, true).toString(16).padStart(8, '0');
      val = isFloat ? dv.getFloat32(o, true) : dv.getInt32(o, true);
    } else {
      hex = dv.getBigUint64(o, true).toString(16).padStart(16, '0');
      val = isFloat ? dv.getFloat64(o, true) : dv.getBigInt64(o, true);
    }
    out.push(`${hex}(${val})`);
  }
  return `[${out.join(' ')}]`;
}

function describe(rec) {
  const e = entries[rec.entry];
  const w = e.laneBytes;
  return [
    `    ${e.name}${e.variant} vector #${rec.vector} (check: ${CHECK_NAMES[e.check]})`,
    `      v0      = ${lanes(rec.v[0], w, e.isFloat)}`,
    `      v1      = ${lanes(rec.v[1], w, e.isFloat)}`,
    `      v2      = ${lanes(rec.v[2], w, e.isFloat)}`,
    `      scalars = ${lanes(rec.s, w, e.isFloat)}`,
    `      wasm    = ${lanes(rec.outA, w, e.isFloat)}`,
    `      ref     = ${lanes(rec.outB, w, e.isFloat)}`,
  ].join('\n');
}

const seedLo = Number(SEED & 0xFFFFFFFFn);
const seedHi = Number((SEED >> 32n) & 0xFFFFFFFFn);

// ----------------------------------------------------------------------------
// 4. core::arch::wasm32 (engine SIMD128) vs simd::scalar
// ----------------------------------------------------------------------------
{
  const t0 = performance.now();
  const status = C('run', seedLo, seedHi, N_RANDOM);
  const ms = performance.now() - t0;
  assert(status !== 0xFFFFFFFF, 'tables disagree in shape (entry order / names)');
  const r = readReport();
  assert(r.entries === nEntries, `report covers ${r.entries} of ${nEntries} entries`);
  const per = r.stats.map((s) => s.vectors);
  const byCheck = [0, 0, 0];
  entries.forEach((e) => { byCheck[e.check] += 1; });
  console.log(`  [INFO] Engine: Node ${process.version} (V8 ${process.versions.v8}), seed 0x${SEED.toString(16)}, ${N_RANDOM} random vectors per entry`);
  console.log(`  [INFO] ${nOps} ops / ${nEntries} entries (${byCheck[0]} exact, ${byCheck[1]} f32 NaN-tolerant, ${byCheck[2]} f64 NaN-tolerant); ${r.vectors} vectors, ${Math.min(...per)}..${Math.max(...per)} per entry; ${ms.toFixed(0)} ms`);
  if (r.mismatches !== 0) {
    const bad = r.stats.map((s, i) => [i, s.mismatches]).filter(([, m]) => m > 0);
    console.error(`  [FAIL] ${r.mismatches} mismatches in ${bad.length} entries:`);
    for (const [i, m] of bad) {
      console.error(`    ${entries[i].name}${entries[i].variant}: ${m} / ${r.stats[i].vectors}`);
    }
    console.error('  First recorded mismatches (wasm = core::arch::wasm32 in this engine, ref = simd::scalar):');
    for (const rec of r.records) console.error(describe(rec));
    throw new Error(`Gate S FAIL: simd::scalar disagrees with Wasm SIMD128 (${r.mismatches} mismatches)`);
  }
  console.log(`  [PASS] simd::scalar matches Wasm SIMD128 executed by V8: 0 mismatches across ${nOps} ops, ${r.vectors} vectors`);
}

// ----------------------------------------------------------------------------
// 5. Negative controls
// ----------------------------------------------------------------------------
{
  const n = C('negative_control_count');
  assert(n >= 5, `expected at least 5 negative controls, found ${n}`);
  for (let i = 0; i < n; i++) {
    const target = str(C('negative_control_target_ptr', i), C('negative_control_target_len', i));
    const status = C('negative_control', i, seedLo, seedHi, N_RANDOM_NEGATIVE);
    assert(status !== 0xFFFFFFFF, `negative control ${target}: tables disagree in shape`);
    const r = readReport();
    let inTarget = 0;
    let outside = 0;
    r.stats.forEach((s, j) => {
      if (entries[j].name === target) inTarget += s.mismatches; else outside += s.mismatches;
    });
    assert(r.mismatches > 0 && inTarget > 0, `negative control for ${target} was NOT detected`);
    assert(outside === 0, `negative control for ${target} leaked ${outside} mismatches into other ops`);
    const first = r.records[0];
    const why = first ? ` (first: vector #${first.vector})` : '';
    console.log(`  [PASS] Negative control ${target.padEnd(24)} detected: ${inTarget} mismatches, all in ${target}${why}`);
  }
}

console.log('  [PASS] Gate S: SIMD layer conformance verified');
