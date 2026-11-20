#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Native vs wasm kernel timing on identical recorded inputs.
//
// Replays the same CSTRACE traces on the wasm kernel in V8
// (`node scripts/trace_replay.mjs --bench`) and on one or more native builds
// (`chronos_replay --bench`), times every kernel call on both sides, and prints
// per workload: wasm median, native median, ratio, spread, repetitions.
//
// Usage:
//   node scripts/bench_native_vs_wasm.mjs [options] <trace-dir | trace>...
//
// Options:
//   --native=LABEL=PATH  a chronos_replay binary to time (repeatable). Default:
//                        native/target/<host>/release/chronos_replay as "native".
//   --wasm=PATH          wasm binary (default web/chronos_spectral.wasm)
//   --node-flag=FLAG     extra node/V8 flag for the wasm side (repeatable),
//                        for example --node-flag=--no-liftoff
//   --wasm-variant=LABEL=FLAGS  time the wasm under several V8 configurations
//                        side by side (repeatable; FLAGS space-separated, may be
//                        empty), e.g. --wasm-variant=wasm= --wasm-variant=turbofan=--no-liftoff
//   --rounds=R           interleaved rounds (default 5)
//   --warmup=W           discarded passes per process (default 2)
//   --passes=N           measured passes per process (default 3)
//   --reconcile=K        also run K untimed + K apply-only passes per process
//   --out=DIR            raw per-run JSON, logs and summary.{json,md}
//                        (default target/bench/<timestamp>)
//   --exports=a,b,...    exports in the per-export table (default: the tick
//                        exports and the solvers)
//   --no-wasm            native sides only (ratios relative to the first one)
//
// Method (docs/VERIFICATION_AND_BENCHMARKS.md, "Native vs wasm kernel timing"):
//  * one process per (round, workload, side); every process loads the trace
//    into memory, discards W warm-up passes (V8 tiers hot functions up from
//    Liftoff to TurboFan) and measures N passes, each from the fresh-instance
//    state; only kernel calls are timed, host writes are applied untimed,
//    digests are off and return values are still checked;
//  * rounds are interleaved (round-major: every workload and side runs once per
//    round) and the side order alternates (A B C, C B A, ...), so slow drift on
//    a shared machine hits every side alike;
//  * per workload the headline is the median of the R x N per-pass kernel
//    totals; "spread" is the range of the R per-round medians relative to the
//    overall median; the ratio's range is taken over per-round ratios (native
//    and wasm medians from the same round);
//  * the script fails if the two sides did not replay the identical call
//    sequence (per-trace and per-export call counts) or if any return value
//    differed from the recording.
// ============================================================================

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const opt = (k, d) => {
  const a = argv.find((x) => x.startsWith(`--${k}=`));
  return a === undefined ? d : a.slice(k.length + 3);
};
const multi = (k) => argv.filter((x) => x.startsWith(`--${k}=`)).map((x) => x.slice(k.length + 3));
const inputs = argv.filter((a) => !a.startsWith('--'));
if (!inputs.length) {
  console.error('usage: node scripts/bench_native_vs_wasm.mjs [--native=LABEL=PATH]... [--rounds=R] [--warmup=W] [--passes=N] <trace-dir|trace>...');
  process.exit(2);
}

const rounds = Number(opt('rounds', '5'));
const warmup = Number(opt('warmup', '2'));
const passes = Number(opt('passes', '3'));
const reconcile = Number(opt('reconcile', '0'));
const wasmPath = path.resolve(opt('wasm', path.join(ROOT, 'web/chronos_spectral.wasm')));
const nodeFlags = multi('node-flag');
const withWasm = !argv.includes('--no-wasm');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outDir = path.resolve(opt('out', path.join(ROOT, 'target/bench', stamp)));
const DEFAULT_EXPORTS = [
  'chronos_tick',
  'chronos_pre_tick',
  'chronos_post_tick',
  'chronos_solve_fiedler_lanczos',
  'chronos_hodge_solve_injected',
  'chronos_bos_solve_manufactured_test',
  'chronos_push_audio_block',
  'chronos_eval_occluded_pair',
  'chronos_init',
];
const tableExports = (opt('exports', null) || DEFAULT_EXPORTS.join(',')).split(',').filter(Boolean);
const FRAME_EXPORTS = new Set(['chronos_tick', 'chronos_pre_tick']);

let natives = multi('native').map((s) => {
  const i = s.indexOf('=');
  if (i <= 0) throw new Error(`--native=${s}: expected LABEL=PATH`);
  return { label: s.slice(0, i), bin: path.resolve(s.slice(i + 1)) };
});
if (!natives.length) {
  const host = spawnSync('rustc', ['--print', 'host-tuple'], { encoding: 'utf8' }).stdout.trim();
  natives = [{ label: 'native', bin: path.join(ROOT, 'native/target', host, 'release/chronos_replay') }];
}
for (const n of natives) {
  if (!fs.existsSync(n.bin)) {
    console.error(
      `FAIL: ${n.bin} not found. Build it with:\n  cargo build --release --manifest-path native/Cargo.toml --target host-tuple --bin chronos_replay`
    );
    process.exit(2);
  }
}
const wasmVariants = multi('wasm-variant').map((v) => {
  const i = v.indexOf('=');
  if (i <= 0) throw new Error(`--wasm-variant=${v}: expected LABEL=FLAGS`);
  return { label: v.slice(0, i), kind: 'wasm', flags: v.slice(i + 1).split(' ').filter(Boolean) };
});
const sides = [
  ...(withWasm ? (wasmVariants.length ? wasmVariants : [{ label: 'wasm', kind: 'wasm', flags: nodeFlags }]) : []),
  ...natives.map((n) => ({ ...n, kind: 'native' })),
];
const base = sides[0].label;

const workloads = inputs.map((p) => {
  const abs = path.resolve(p);
  const st = fs.statSync(abs);
  const name = st.isDirectory() ? path.basename(abs) : path.basename(abs).replace(/\.cstrace$/, '');
  return { name, path: abs };
});

fs.mkdirSync(outDir, { recursive: true });
const benchArgs = [`--warmup=${warmup}`, `--passes=${passes}`, ...(reconcile ? [`--bench-reconcile=${reconcile}`] : [])];

function runOne(side, w, r) {
  const json = path.join(outDir, `${w.name}.${side.label}.r${r}.json`);
  const log = path.join(outDir, `${w.name}.${side.label}.r${r}.log`);
  const [cmd, args] =
    side.kind === 'wasm'
      ? [process.execPath, [...side.flags, '--expose-gc', path.join(ROOT, 'scripts/trace_replay.mjs'), '--bench', `--wasm=${wasmPath}`, ...benchArgs, `--json=${json}`, w.path]]
      : [side.bin, ['--bench', ...benchArgs, `--json=${json}`, w.path]];
  const t0 = Date.now();
  const res = spawnSync(cmd, args, { encoding: 'utf8', maxBuffer: 1 << 28 });
  fs.writeFileSync(log, `$ ${cmd} ${args.join(' ')}\n${res.stdout || ''}${res.stderr || ''}`);
  if (res.status !== 0 || !fs.existsSync(json)) {
    throw new Error(`${side.label} on ${w.name} (round ${r}) failed with status ${res.status}; see ${log}`);
  }
  const doc = JSON.parse(fs.readFileSync(json, 'utf8'));
  doc.elapsed_s = (Date.now() - t0) / 1e3;
  return doc;
}

// ---------------------------------------------------------------------------
// Run.
// ---------------------------------------------------------------------------
console.log(
  `bench_native_vs_wasm: ${workloads.length} workload(s) x ${sides.length} side(s) [${sides.map((s) => s.label).join(', ')}] x ${rounds} round(s); ` +
    `${warmup} warm-up + ${passes} measured passes per process; out ${outDir}`
);
console.log(`machine: ${os.cpus()[0]?.model} x ${os.cpus().length}, ${os.platform()}/${os.arch()}, node ${process.version}, load avg ${os.loadavg().map((x) => x.toFixed(1)).join(' ')}`);
const results = {}; // results[w][side] = [doc per round]
const tStart = Date.now();
for (let r = 0; r < rounds; r++) {
  const order = r % 2 === 0 ? sides : [...sides].reverse();
  for (const w of workloads) {
    for (const side of order) {
      const doc = runOne(side, w, r);
      ((results[w.name] ||= {})[side.label] ||= [])[r] = doc;
      const k = doc.traces.reduce((a, t) => a + median(t.pass_kernel_ns), 0);
      console.log(
        `  [round ${r + 1}/${rounds}] ${w.name.padEnd(16)} ${side.label.padEnd(10)} kernel ${(k / 1e6).toFixed(3).padStart(10)} ms/pass  (${doc.elapsed_s.toFixed(1)} s, load avg ${os.loadavg()[0].toFixed(1)})`
      );
    }
  }
}
const elapsed = (Date.now() - tStart) / 1e3;

// ---------------------------------------------------------------------------
// Aggregate.
// ---------------------------------------------------------------------------
function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
function median(v) {
  return quantile([...v].sort((a, b) => a - b), 0.5);
}

const problems = [];
const summary = { machine: os.cpus()[0]?.model, node: process.version, rounds, warmup, passes, elapsed_s: elapsed, sides: {}, workloads: [] };
for (const s of sides) {
  const d = results[workloads[0].name][s.label][0];
  summary.sides[s.label] = d.side === 'wasm' ? { engine: d.engine, wasm_sha256: d.wasm_sha256, timer: d.timer } : { build: d.build, backend: d.backend, bin: s.bin, timer: d.timer };
}

for (const w of workloads) {
  const R = results[w.name];
  const ref = R[base][0];
  // Identical call sequence on every side and round.
  for (const s of sides) {
    for (let r = 0; r < rounds; r++) {
      const d = R[s.label][r];
      if (d.traces.length !== ref.traces.length) problems.push(`${w.name}: ${s.label} r${r} replayed ${d.traces.length} traces, ${base} ${ref.traces.length}`);
      d.traces.forEach((t, i) => {
        const rt = ref.traces[i];
        if (!rt || t.name !== rt.name || t.calls !== rt.calls || t.writes !== rt.writes || t.stop.split(' ')[0] !== rt.stop.split(' ')[0]) {
          problems.push(`${w.name}/${t.name}: ${s.label} r${r} replayed ${t.calls} calls / ${t.writes} writes / stop ${t.stop}; ${base}: ${rt?.calls} / ${rt?.writes} / ${rt?.stop}`);
        }
        if (t.mismatches) problems.push(`${w.name}/${t.name}: ${s.label} r${r}: ${t.mismatches} return-value mismatches`);
        const cnt = (tr) => Object.fromEntries(tr.exports.map((e) => [e.name, e.count]));
        const a = cnt(t);
        const b = cnt(rt);
        for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
          if (a[k] !== b[k]) problems.push(`${w.name}/${t.name}: ${k} called ${a[k]} times per pass on ${s.label}, ${b[k]} on ${base}`);
        }
      });
    }
  }

  const row = { workload: w.name, traces: ref.traces.length, calls: ref.traces.reduce((a, t) => a + t.calls, 0), frames: 0, sides: {}, exports: {} };
  for (const e of ref.traces.flatMap((t) => t.exports)) if (FRAME_EXPORTS.has(e.name)) row.frames += e.count;
  row.stops = ref.traces.filter((t) => t.stop !== 'end').map((t) => `${t.name}: ${t.stop}`);
  for (const s of sides) {
    const docs = R[s.label];
    const perRound = docs.map((d) => {
      // Per-pass workload totals (traces summed pass by pass).
      const tot = [];
      for (let p = 0; p < passes; p++) tot.push(d.traces.reduce((a, t) => a + t.pass_kernel_ns[p], 0));
      return tot;
    });
    const pooled = perRound.flat();
    const rm = perRound.map(median);
    const med = median(pooled);
    const recU = docs.map((d) => d.traces.reduce((a, t) => a + (t.reconcile_untimed_ns.length ? median(t.reconcile_untimed_ns) : 0), 0));
    const recA = docs.map((d) => d.traces.reduce((a, t) => a + (t.reconcile_apply_ns.length ? median(t.reconcile_apply_ns) : 0), 0));
    const wall = docs.map((d) => d.traces.reduce((a, t) => a + median(t.pass_wall_ns), 0));
    const warm = docs.map((d) => d.traces.reduce((a, t) => a + (t.warmup_kernel_ns[0] ?? NaN), 0));
    row.sides[s.label] = {
      median_ms: med / 1e6,
      round_medians_ms: rm.map((x) => x / 1e6),
      spread_pct: (100 * (Math.max(...rm) - Math.min(...rm))) / med,
      p95_pass_ms: quantile([...pooled].sort((a, b) => a - b), 0.95) / 1e6,
      n: pooled.length,
      per_frame_ms: row.frames ? med / 1e6 / row.frames : null,
      first_warmup_pass_ms: median(warm) / 1e6,
      pass_wall_ms: median(wall) / 1e6,
      reconcile_kernel_by_wall_ms: reconcile ? (median(recU) - median(recA)) / 1e6 : null,
    };
    // Per-export pooled statistics.
    for (const name of new Set(ref.traces.flatMap((t) => t.exports.map((e) => e.name)))) {
      const samples = [];
      let count = 0;
      const roundMed = [];
      docs.forEach((d, r) => {
        const rs = [];
        for (const t of d.traces) {
          const e = t.exports.find((x) => x.name === name);
          if (!e) continue;
          if (r === 0) count += e.count;
          for (const x of e.samples_ns) rs.push(x);
        }
        roundMed.push(median(rs));
        for (const x of rs) samples.push(x);
      });
      samples.sort((a, b) => a - b);
      const sum = samples.reduce((a, b) => a + b, 0);
      (row.exports[name] ||= { count, sides: {} }).sides[s.label] = {
        median_us: quantile(samples, 0.5) / 1e3,
        p95_us: quantile(samples, 0.95) / 1e3,
        mean_us: sum / samples.length / 1e3,
        share_pct: (100 * (sum / (rounds * passes))) / med,
        round_medians_us: roundMed.map((x) => x / 1e3),
      };
    }
  }
  for (const s of sides) {
    if (s.label === base) continue;
    const ratios = row.sides[s.label].round_medians_ms.map((x, r) => x / row.sides[base].round_medians_ms[r]);
    row.sides[s.label].ratio = row.sides[s.label].median_ms / row.sides[base].median_ms;
    row.sides[s.label].ratio_round_range = [Math.min(...ratios), Math.max(...ratios)];
  }
  summary.workloads.push(row);
}
summary.problems = problems;

// ---------------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------------
const f3 = (x) => (x == null || Number.isNaN(x) ? '-' : x >= 100 ? x.toFixed(1) : x >= 10 ? x.toFixed(2) : x.toFixed(3));
const others = sides.filter((s) => s.label !== base);
const md = [];
const lines = [];
const emit = (cells, widths) => lines.push(cells.map((c, i) => (i === 0 ? String(c).padEnd(widths[i]) : String(c).padStart(widths[i]))).join('  '));

md.push(`Kernel time per pass (ms; median of ${rounds} rounds x ${passes} passes = ${rounds * passes} per side; ratio = side / ${base}, < 1 means faster than ${base})`, '');
const head = ['workload', 'frames', ...sides.map((s) => `${s.label} ms`), ...sides.map((s) => `${s.label} spread`), ...others.flatMap((s) => [`${s.label}/${base}`, 'round range'])];
md.push(`| ${head.join(' | ')} |`, `| ${head.map((h, i) => (i === 0 ? ':---' : '---:')).join(' | ')} |`);
const widths = head.map((h) => Math.max(h.length, 9));
widths[0] = Math.max(16, ...workloads.map((w) => w.name.length));
emit(head, widths);
let geo = Object.fromEntries(others.map((s) => [s.label, 0]));
for (const row of summary.workloads) {
  const cells = [
    row.workload,
    row.frames || '-',
    ...sides.map((s) => f3(row.sides[s.label].median_ms)),
    ...sides.map((s) => `${row.sides[s.label].spread_pct.toFixed(1)}%`),
    ...others.flatMap((s) => {
      const x = row.sides[s.label];
      geo[s.label] += Math.log(x.ratio);
      return [x.ratio.toFixed(3), `${x.ratio_round_range[0].toFixed(3)}-${x.ratio_round_range[1].toFixed(3)}`];
    }),
  ];
  emit(cells, widths);
  md.push(`| ${cells.join(' | ')} |`);
}
const totals = sides.map((s) => summary.workloads.reduce((a, r) => a + r.sides[s.label].median_ms, 0));
const tcells = ['TOTAL (sum)', '', ...totals.map(f3), ...sides.map(() => ''), ...others.flatMap((s, i) => [(totals[sides.indexOf(s)] / totals[0]).toFixed(3), `geomean ${Math.exp(geo[s.label] / summary.workloads.length).toFixed(3)}`])];
emit(tcells, widths);
md.push(`| ${tcells.join(' | ')} |`);

// Per-frame line for workloads with ticks.
lines.push('', `Per frame (kernel ms per chronos_pre_tick / chronos_tick, all calls of the frame included):`);
md.push('', 'Kernel time per frame (ms; pass total / frames):', '', `| workload | frames | ${sides.map((s) => s.label).join(' | ')} |`, `| :--- | ---: | ${sides.map(() => '---:').join(' | ')} |`);
for (const row of summary.workloads) {
  if (!row.frames) continue;
  lines.push(`  ${row.workload.padEnd(widths[0])} ${String(row.frames).padStart(6)} frames  ${sides.map((s) => `${s.label} ${f3(row.sides[s.label].per_frame_ms)}`).join('  ')}`);
  md.push(`| ${row.workload} | ${row.frames} | ${sides.map((s) => f3(row.sides[s.label].per_frame_ms)).join(' | ')} |`);
}

// Per-export table.
lines.push('', `Per call (us; median / p95 / mean over every call of every measured pass and round):`);
md.push('', 'Per call (µs; median / p95 / mean, pooled over all measured passes and rounds):', '');
const eh = ['workload', 'export', 'calls/pass', ...sides.map((s) => `${s.label} med/p95/mean`), ...others.map((s) => `${s.label}/${base} med`), ...others.map((s) => `${s.label}/${base} mean`)];
md.push(`| ${eh.join(' | ')} |`, `| ${eh.map((h, i) => (i < 2 ? ':---' : '---:')).join(' | ')} |`);
for (const row of summary.workloads) {
  for (const name of tableExports) {
    const e = row.exports[name];
    if (!e) continue;
    const cells = [
      row.workload,
      name.replace(/^chronos_/, ''),
      e.count,
      ...sides.map((s) => `${f3(e.sides[s.label].median_us)} / ${f3(e.sides[s.label].p95_us)} / ${f3(e.sides[s.label].mean_us)}`),
      ...others.map((s) => (e.sides[s.label].median_us / e.sides[base].median_us).toFixed(3)),
      ...others.map((s) => (e.sides[s.label].mean_us / e.sides[base].mean_us).toFixed(3)),
    ];
    lines.push(`  ${cells[0].padEnd(widths[0])} ${cells[1].padEnd(28)} ${String(cells[2]).padStart(6)}  ${cells.slice(3).map((c) => String(c).padStart(26)).join(' ')}`);
    md.push(`| ${cells.join(' | ')} |`);
  }
}

if (reconcile) {
  lines.push('', 'Reconciliation (ms per pass: summed per-call kernel time vs untimed-pass wall minus apply-only-pass wall):');
  for (const row of summary.workloads) {
    lines.push(
      `  ${row.workload.padEnd(widths[0])} ${sides
        .map((s) => {
          const x = row.sides[s.label];
          return `${s.label} ${f3(x.median_ms)} vs ${f3(x.reconcile_kernel_by_wall_ms)} (${(100 * (x.median_ms / x.reconcile_kernel_by_wall_ms - 1)).toFixed(1)}%)`;
        })
        .join('   ')}`
    );
  }
}

const stopped = summary.workloads.filter((r) => r.stops.length);
if (stopped.length) {
  lines.push('', 'Native-portable prefixes (both sides stop at the same record):');
  for (const r of stopped) for (const s of r.stops) lines.push(`  ${r.workload}: ${s}`);
}
lines.push('', 'Sides:');
for (const [k, v] of Object.entries(summary.sides)) lines.push(`  ${k}: ${v.engine || v.build}; timer ${v.timer.clock} overhead ${v.timer.overhead_ns.toFixed(1)} ns, resolution ${v.timer.resolution_ns.toFixed(1)} ns`);
lines.push(`elapsed ${(elapsed / 60).toFixed(1)} min; raw results in ${outDir}`);

console.log('');
console.log(lines.join('\n'));
fs.writeFileSync(path.join(outDir, 'summary.json'), JSON.stringify(summary, null, 1));
fs.writeFileSync(path.join(outDir, 'summary.md'), md.join('\n') + '\n');
if (problems.length) {
  console.log(`\n[FAIL] ${problems.length} problem(s):`);
  for (const p of problems.slice(0, 40)) console.log(`  ${p}`);
  process.exit(1);
}
console.log('\n[PASS] identical call sequences on every side and round; 0 return-value mismatches');