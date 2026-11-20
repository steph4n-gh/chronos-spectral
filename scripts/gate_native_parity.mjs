#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate N: whole-kernel parity, native build vs the wasm build
//
// Self-contained (no pre-recorded traces, no platform assumptions):
//   1. Builds the native replay driver (native/, bin `chronos_replay`, profile `checked`:
//      overflow checks and debug assertions on) for the host.
//   2. Records a small bounded trace set fresh from web/chronos_spectral.wasm by running
//      existing harnesses under scripts/trace_hook.mjs: real-MP4 sparse hero tracking
//      (calibrate_mp4.mjs, juggling_ballistic, SD), discrete Hodge (gate_i3_hodge.mjs),
//      occluded-pair conservation (gate_i5b_conservation.mjs) and the first 24 MiB of the
//      1,200-target vortex swarm (benchmark_1000.mjs --only=Q1).
//   3. Replays every trace on the native kernel: every return value and every per-region
//      XXH32 digest must match bit for bit (only the documented platform words of
//      PLATFORM_WORDS in scripts/trace_lib.mjs are masked, on both sides). Each scenario must
//      reach a minimum number of calls, so a harness that records little cannot pass vacuously.
//   4. Negative control: the same driver must reject a copy of the Hodge trace with one host
//      write byte flipped, at or after the call that write precedes.
//
// Usage:
//   node scripts/gate_native_parity.mjs [--no-build] [--keep] [--out=<dir>] [--features=<list>]
//
//   --out       trace directory (default target/native-parity; removed after a pass unless --keep)
//   --features  cargo features for the native build (default: none, i.e. the backend
//               `crate::simd` selects for the host; e.g. --features=simd-force-scalar)
// ============================================================================

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const HOOK = path.join(ROOT, 'scripts', 'trace_hook.mjs');
const WASM = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const MIB = 1024 * 1024;

const args = process.argv.slice(2);
const argValue = (name) => {
  const a = args.find((x) => x.startsWith(`--${name}=`));
  return a ? a.slice(name.length + 3) : undefined;
};
const OUT = path.resolve(argValue('out') ?? path.join(ROOT, 'target', 'native-parity'));
const KEEP = args.includes('--keep');
const BUILD = !args.includes('--no-build');
const FEATURES = argValue('features') ?? '';

// Each scenario runs one existing harness unmodified. `minCalls` is the least number of
// replayed kernel calls the scenario must contribute.
const SCENARIOS = [
  { id: 'hero_sd', argv: ['scripts/calibrate_mp4.mjs', '--clip=juggling_ballistic'], maxInstances: 2, capMiB: 64, minCalls: 1000 },
  { id: 'hodge', argv: ['scripts/gate_i3_hodge.mjs'], capMiB: 32, minCalls: 150 },
  { id: 'conservation', argv: ['scripts/gate_i5b_conservation.mjs'], capMiB: 32, minCalls: 1000 },
  { id: 'swarm', argv: ['scripts/benchmark_1000.mjs', '--only=Q1'], maxInstances: 1, capMiB: 24, minCalls: 300 },
];

function fail(msg) {
  console.error(`  [FAIL] Gate N: ${msg}`);
  console.error(`         traces kept in ${OUT}`);
  process.exit(1);
}

const t0 = Date.now();
const secs = (t) => ((Date.now() - t) / 1000).toFixed(1);

// ----------------------------------------------------------------------------
// 1. Native replay driver
// ----------------------------------------------------------------------------
const host = spawnSync('rustc', ['-vV'], { encoding: 'utf8' }).stdout?.match(/^host: (.+)$/m)?.[1];
if (!host) fail('cannot determine the host target (rustc -vV)');
const driver = path.join(ROOT, 'native', 'target', host, 'checked', process.platform === 'win32' ? 'chronos_replay.exe' : 'chronos_replay');
if (BUILD) {
  const tb = Date.now();
  const cargoArgs = ['build', '--quiet', '--manifest-path', 'native/Cargo.toml', '--target', host, '--profile', 'checked', '--bin', 'chronos_replay'];
  if (FEATURES) cargoArgs.push('--features', FEATURES);
  const r = spawnSync('cargo', cargoArgs, {
    cwd: ROOT,
    stdio: 'inherit',
    env: { ...process.env, RUSTFLAGS: '-D warnings' },
  });
  if (r.status !== 0) fail('native replay driver build failed');
  console.log(`  [PASS] Native replay driver built for ${host} (profile checked${FEATURES ? `, features ${FEATURES}` : ''}) in ${secs(tb)} s`);
}
if (!fs.existsSync(driver)) fail(`native replay driver not found at ${driver}`);

// ----------------------------------------------------------------------------
// 2. Record
// ----------------------------------------------------------------------------
const wasmSha = crypto.createHash('sha256').update(fs.readFileSync(WASM)).digest('hex');
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });

function record(s) {
  const dir = path.join(OUT, s.id);
  fs.mkdirSync(dir, { recursive: true });
  const env = {
    ...process.env,
    NODE_OPTIONS: `${process.env.NODE_OPTIONS || ''} --import=${HOOK}`.trim(),
    CHRONOS_TRACE_DIR: dir,
    CHRONOS_TRACE_LABEL: s.id,
    CHRONOS_TRACE_MAX_BYTES: String(s.capMiB * MIB),
    CHRONOS_TRACE_TOTAL_MAX_BYTES: String(4 * s.capMiB * MIB),
    // Recording slows latency-budget harnesses; their budgets are scaled, not changed.
    CHRONOS_CI_BUDGET_SCALE: '32',
    CHRONOS_TRACE_LOG: '0',
  };
  if (s.maxInstances) env.CHRONOS_TRACE_MAX_INSTANCES = String(s.maxInstances);
  const t = Date.now();
  return new Promise((resolve) => {
    const child = spawn(process.execPath, s.argv, { cwd: ROOT, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let log = '';
    child.stdout.on('data', (d) => (log += d));
    child.stderr.on('data', (d) => (log += d));
    child.on('close', (code, signal) => {
      fs.writeFileSync(path.join(OUT, `${s.id}.log`), log);
      const idx = path.join(dir, 'index.jsonl');
      const traces = fs.existsSync(idx) ? fs.readFileSync(idx, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)) : [];
      resolve({ s, code, signal, seconds: (Date.now() - t) / 1000, traces });
    });
  });
}

const tr = Date.now();
const recorded = await Promise.all(SCENARIOS.map(record));
let totalBytes = 0;
let totalCalls = 0;
for (const r of recorded) {
  const calls = r.traces.reduce((a, t) => a + t.calls, 0);
  const bytes = r.traces.reduce((a, t) => a + t.bytes, 0);
  totalBytes += bytes;
  totalCalls += calls;
  if (r.code !== 0) fail(`${r.s.id}: harness ${r.s.argv.join(' ')} exited ${r.code ?? r.signal} while recording (see ${r.s.id}.log)`);
  if (calls < r.s.minCalls) fail(`${r.s.id}: recorded ${calls} kernel calls, expected at least ${r.s.minCalls}`);
  if (r.traces.some((t) => t.unmapped)) fail(`${r.s.id}: trace has UNMAPPED raw-address writes; the gate needs fully portable traces`);
  console.log(`  [PASS] Recorded ${r.s.id.padEnd(12)} ${String(r.traces.length).padStart(2)} trace(s), ${String(calls).padStart(5)} calls, ${(bytes / 1e6).toFixed(1).padStart(5)} MB in ${r.seconds.toFixed(1)} s`);
}
console.log(`  [INFO] Recorded ${totalCalls} calls, ${(totalBytes / 1e6).toFixed(1)} MB in ${secs(tr)} s (wasm sha256 ${wasmSha.slice(0, 16)}...)`);

// ----------------------------------------------------------------------------
// 3. Native replay
// ----------------------------------------------------------------------------
const dumpDir = path.join(OUT, 'dumps');
const runDriver = (extra) =>
  spawnSync(driver, [`--dump-dir=${dumpDir}`, `--wasm-sha256=${wasmSha}`, ...extra], { cwd: ROOT, encoding: 'utf8', maxBuffer: 64 * MIB });
const tp = Date.now();
const rep = runDriver(['--require-full', ...SCENARIOS.map((s) => path.join(OUT, s.id))]);
process.stdout.write(rep.stdout.replace(/^/gm, '    '));
if (rep.stderr) process.stderr.write(rep.stderr.replace(/^/gm, '    '));
if (rep.status !== 0) fail('native replay diverged from the wasm recording (first divergence and region dumps above)');
const digestChecks = [...rep.stdout.matchAll(/(\d+) region digests/g)].reduce((a, m) => a + Number(m[1]), 0);
console.log(`  [PASS] Native kernel matches wasm bit for bit: ${totalCalls} calls, ${digestChecks} region digests, 0 mismatches (replay ${secs(tp)} s)`);

// ----------------------------------------------------------------------------
// 4. Negative control
// ----------------------------------------------------------------------------
const hodgeTrace = fs.readdirSync(path.join(OUT, 'hodge')).filter((f) => f.endsWith('.cstrace')).map((f) => path.join(OUT, 'hodge', f))[0];
const mutated = path.join(OUT, 'negative-control.cstrace');
const mut = spawnSync(process.execPath, ['scripts/trace_tool.mjs', 'mutate-write', hodgeTrace, mutated], { cwd: ROOT, encoding: 'utf8' });
const before = Number(mut.stdout.match(/before call #(\d+)/)?.[1]);
if (mut.status !== 0 || !Number.isInteger(before)) fail(`could not build the negative-control trace: ${mut.stderr || mut.stdout}`);
const neg = runDriver([mutated]);
const at = Number(neg.stdout.match(/^ {4}call #(\d+) /m)?.[1]);
if (neg.status === 0 || !Number.isInteger(at) || at < before) {
  process.stdout.write(neg.stdout.replace(/^/gm, '    '));
  fail(`negative control not detected as expected (${mut.stdout.trim()})`);
}
const region = neg.stdout.match(/region (\S+): native/)?.[1] ?? '?';
console.log(`  [PASS] Negative control detected: ${mut.stdout.trim()}; driver reports call #${at}, region ${region}`);

if (!KEEP) fs.rmSync(OUT, { recursive: true, force: true });
console.log(`  [PASS] Gate N: native/wasm whole-kernel parity verified (${secs(t0)} s total, ${(totalBytes / 1e6).toFixed(1)} MB of traces)`);
