#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Record the canonical kernel trace set.
//
// Runs the existing harnesses, unmodified, under scripts/trace_hook.mjs and
// writes one CSTRACE file per kernel instance into <out>/<scenario>/. Traces
// are large temporary artifacts: write them outside the repository.
//
// Usage:
//   node scripts/record_canonical_traces.mjs --out=<dir> [--only=a,b] [--list]
//        [--instance-mb=N] [--total-mb=N] [--budget-scale=S] [--strict] [--replay]
//
//   --instance-mb   per-instance cap (default 1024); the trace ends cleanly
//   --total-mb      cap over the whole set (default 8192); each scenario gets
//                   what is left, scenarios after exhaustion are skipped
//   --budget-scale  CHRONOS_CI_BUDGET_SCALE for the recorded runs (default 32;
//                   recording slows latency-budget gates, budgets are unchanged)
//   --strict        CHRONOS_TRACE_STRICT=1 (diff + digest around every call)
//   --memdigest=N   CHRONOS_TRACE_MEMDIGEST=N (whole-memory digest every Nth digested call)
//   --replay        replay every trace afterwards with scripts/trace_replay.mjs
// ============================================================================

import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './trace_lib.mjs';

const HOOK = path.join(ROOT, 'scripts', 'trace_hook.mjs');
const MB = 1024 * 1024;

// Each scenario runs one existing harness. `maxInstances` bounds harnesses
// that build many kernel instances (the rest of the run is unrecorded).
const SCENARIOS = [
  {
    id: 'hero_sd',
    what: 'sparse hero tracking, SD 320x180, real MP4 (calibrate_mp4.mjs, juggling_ballistic)',
    argv: ['scripts/calibrate_mp4.mjs', '--clip=juggling_ballistic'],
    maxInstances: 2,
  },
  {
    id: 'hero_hd',
    what: 'sparse hero tracking, HD 640x360, real MP4 (calibrate_mp4.mjs --hd, juggling_ballistic)',
    argv: ['scripts/calibrate_mp4.mjs', '--hd', '--clip=juggling_ballistic'],
    maxInstances: 2,
  },
  {
    id: 'hero_suite',
    what: 'synthetic sparse hero gates A..P incl. SD<->HD switch and SD/HD pillar trials (build.sh [3/5] heredoc, read-only)',
    buildShHeredoc: true,
    maxInstances: 1,
  },
  {
    id: 'mp4_occlusion',
    what: 'real MP4 with occlusions (calibrate_mp4.mjs, roundabout_occlusion)',
    argv: ['scripts/calibrate_mp4.mjs', '--clip=roundabout_occlusion'],
    maxInstances: 2,
  },
  {
    id: 'swarm_1200',
    what: '1,200-target vortex swarm (benchmark_1000.mjs --only=Q1)',
    argv: ['scripts/benchmark_1000.mjs', '--only=Q1'],
    maxInstances: 2,
  },
  { id: 'bos', what: 'background-oriented schlieren (gate_i5_bos.mjs)', argv: ['scripts/gate_i5_bos.mjs'] },
  { id: 'hodge', what: 'discrete Hodge decomposition (gate_i3_hodge.mjs)', argv: ['scripts/gate_i3_hodge.mjs'] },
  {
    id: 'rs_vibrometry',
    what: 'rolling-shutter vibrometry with audio (gate_i2_rs_welch.mjs)',
    argv: ['scripts/gate_i2_rs_welch.mjs'],
  },
  {
    id: 'conservation',
    what: 'occluded-pair conservation (gate_i5b_conservation.mjs)',
    argv: ['scripts/gate_i5b_conservation.mjs'],
  },
  {
    id: 'covariance',
    what: 'micro covariance / residual metrology (gate_i1_cov_residual.mjs)',
    argv: ['scripts/gate_i1_cov_residual.mjs'],
    // 406 instances of ~12 MB; sample the raw, H.264 and looming/parallax suites.
    instances: '0-1,48-49,96-97,200-201,404-405',
  },
  {
    id: 'ingest_roi',
    what: 'ingest ring, ROI, worker ABI (gate_i0_i4_ingest.mjs)',
    argv: ['scripts/gate_i0_i4_ingest.mjs'],
  },
  {
    id: 'depth',
    what: 'zero-ML depth sensing: mono parallax, micro/wide stereo BOS, external plane (gate_i7_depth.mjs --parity-lite)',
    argv: ['scripts/gate_i7_depth.mjs', '--parity-lite'],
  },
];

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const a = args.find((x) => x.startsWith(`--${name}=`));
  return a ? a.slice(name.length + 3) : dflt;
};
if (args.includes('--list') || args.includes('--help')) {
  for (const s of SCENARIOS) console.log(`${s.id.padEnd(15)} ${s.what}`);
  process.exit(0);
}
const outArg = opt('out', null);
if (!outArg) {
  console.error('usage: node scripts/record_canonical_traces.mjs --out=<dir> [--only=a,b] [--instance-mb=N] [--total-mb=N] [--budget-scale=S] [--strict] [--replay]');
  process.exit(2);
}
const OUT = path.resolve(outArg);
if (OUT === ROOT || OUT.startsWith(ROOT + path.sep)) {
  console.error(`[FAIL] --out must be outside the repository (traces must never be committed): ${OUT}`);
  process.exit(2);
}
const only = opt('only', null)?.split(',').filter(Boolean);
if (only) {
  const unknown = only.filter((id) => !SCENARIOS.some((s) => s.id === id));
  if (unknown.length) {
    console.error(`[FAIL] unknown scenario(s): ${unknown.join(', ')} (see --list)`);
    process.exit(2);
  }
}
const instanceBytes = Number(opt('instance-mb', '1024')) * MB;
const totalBytes = Number(opt('total-mb', '8192')) * MB;
const budgetScale = opt('budget-scale', '32');
const strict = args.includes('--strict');
const memDigest = opt('memdigest', null);

function buildShHeredoc() {
  // The [3/5] Node gate suite embedded in build.sh: the first `node --input-type=module - <<'EOF'` block.
  const sh = fs.readFileSync(path.join(ROOT, 'build.sh'), 'utf8');
  const m = sh.match(/\nnode --input-type=module - <<'EOF'\n([\s\S]*?)\nEOF\n/);
  if (!m) throw new Error('could not find the Node heredoc in build.sh');
  return m[1] + '\n';
}

function dirBytes(d) {
  if (!fs.existsSync(d)) return 0;
  return fs.readdirSync(d).reduce((a, f) => a + fs.statSync(path.join(d, f)).size, 0);
}

function runScenario(s, remaining) {
  const dir = path.join(OUT, s.id);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const env = {
    ...process.env,
    NODE_OPTIONS: `${process.env.NODE_OPTIONS || ''} --import=${HOOK}`.trim(),
    CHRONOS_TRACE_DIR: dir,
    CHRONOS_TRACE_LABEL: s.id,
    CHRONOS_TRACE_MAX_BYTES: String(Math.min(instanceBytes, remaining)),
    CHRONOS_TRACE_TOTAL_MAX_BYTES: String(remaining),
    CHRONOS_CI_BUDGET_SCALE: budgetScale,
  };
  if (s.maxInstances) env.CHRONOS_TRACE_MAX_INSTANCES = String(s.maxInstances);
  if (s.instances) env.CHRONOS_TRACE_INSTANCES = s.instances;
  if (strict) env.CHRONOS_TRACE_STRICT = '1';
  if (memDigest) env.CHRONOS_TRACE_MEMDIGEST = memDigest;
  const argv = s.buildShHeredoc ? ['--input-type=module', '-'] : s.argv;
  const t0 = Date.now();
  return new Promise((resolve) => {
    const child = spawn(process.execPath, argv, { cwd: ROOT, env, stdio: ['pipe', 'pipe', 'pipe'] });
    const log = fs.createWriteStream(path.join(OUT, `${s.id}.log`));
    child.stdout.pipe(log);
    child.stderr.on('data', (d) => {
      log.write(d);
      for (const line of d.toString().split('\n')) if (line.startsWith('[chronos-trace]')) console.log(`  ${line}`);
    });
    if (s.buildShHeredoc) child.stdin.end(buildShHeredoc());
    else child.stdin.end();
    child.on('close', (code, signal) => {
      log.end();
      resolve({ code, signal, seconds: (Date.now() - t0) / 1000, bytes: dirBytes(dir) });
    });
  });
}

fs.mkdirSync(OUT, { recursive: true });
console.log(`Recording canonical traces into ${OUT} (instance cap ${instanceBytes / MB} MB, total cap ${totalBytes / MB} MB, CHRONOS_CI_BUDGET_SCALE=${budgetScale}${strict ? ', strict' : ''}${memDigest ? `, memdigest every ${memDigest}` : ''})`);
const summary = [];
let used = 0;
for (const s of SCENARIOS) {
  if (only && !only.includes(s.id)) continue;
  const remaining = totalBytes - used;
  if (remaining < 16 * MB) {
    console.log(`[SKIP] ${s.id}: total cap exhausted`);
    summary.push({ scenario: s.id, skipped: 'total-cap' });
    continue;
  }
  console.log(`[RUN ] ${s.id}: ${s.what}`);
  const r = await runScenario(s, remaining);
  used += r.bytes;
  const indexFile = path.join(OUT, s.id, 'index.jsonl');
  const traces = fs.existsSync(indexFile)
    ? fs.readFileSync(indexFile, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l))
    : [];
  const status = r.code === 0 ? 'harness exit 0' : `harness exit ${r.code ?? r.signal} (see ${s.id}.log)`;
  console.log(`[DONE] ${s.id}: ${traces.length} trace(s), ${(r.bytes / 1e6).toFixed(1)} MB, ${r.seconds.toFixed(1)} s, ${status}`);
  summary.push({ scenario: s.id, what: s.what, harnessExit: r.code, seconds: r.seconds, bytes: r.bytes, traces });
}

fs.writeFileSync(path.join(OUT, 'summary.json'), JSON.stringify(summary, null, 2));
console.log('\nscenario         trace                                       calls   host-writes  resets  unmapped      MB  end');
for (const s of summary) {
  for (const t of s.traces || []) {
    console.log(
      `${s.scenario.padEnd(16)} ${t.file.padEnd(42)} ${String(t.calls).padStart(7)} ${String(t.hostWrites).padStart(13)} ${String(t.resets).padStart(7)} ${String(t.unmapped).padStart(9)} ${(t.bytes / 1e6).toFixed(1).padStart(7)}  ${t.end}`
    );
  }
}
console.log(`total ${(used / 1e6).toFixed(1)} MB`);

let failed = summary.some((s) => s.harnessExit !== undefined && s.harnessExit !== 0);
if (args.includes('--replay')) {
  console.log('\nReplaying...');
  const dirs = summary.filter((s) => s.traces?.length).map((s) => path.join(OUT, s.scenario));
  const r = spawnSync(process.execPath, [path.join(ROOT, 'scripts', 'trace_replay.mjs'), ...dirs], { stdio: 'inherit' });
  failed ||= r.status !== 0;
}
process.exit(failed ? 1 : 0);
