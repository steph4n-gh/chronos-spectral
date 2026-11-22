#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL v3.5 // Static UI Contract & Discipline Linter
// Verifies:
//   1. Every ID in `CONTROL_REGISTRY` appears exactly once in `web/index.html`
//   2. Every `[data-ctl]` element in `web/index.html` is registered (zero orphans)
//   3. Zero duplicate `id="..."` attributes in `web/index.html`
//   4. All 22 worker-transferred `<canvas>` IDs + `#auditTimelineCanvas` exist
//   5. All ES modules under `web/ui/` pass Node syntax verification
//   6. Zero em dashes (`\u2014`) in new v3.5 UI modules, CSS, and `web/index.html`
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { CONTROL_REGISTRY, BAY_PANEL_OWNERSHIP } from '../web/ui/core/registry.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const HTML_PATH = path.join(ROOT, 'web/index.html');
const html = fs.readFileSync(HTML_PATH, 'utf8');

let failures = 0;
function fail(msg) {
  console.error(`  [FAIL] ${msg}`);
  failures++;
}
function pass(msg) {
  console.log(`  [PASS] ${msg}`);
}

console.log('=== CHRONOS-SPECTRAL v3.5 UI Contract Lint ===');

// 1. Unique IDs across web/index.html
const idMatches = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
const idCounts = new Map();
for (const id of idMatches) idCounts.set(id, (idCounts.get(id) || 0) + 1);
const dupIds = [...idCounts.entries()].filter(([, c]) => c > 1);
if (dupIds.length > 0) {
  fail(`Duplicate DOM IDs in web/index.html: ${dupIds.map(([k, c]) => `${k}(${c})`).join(', ')}`);
} else {
  pass(`Zero duplicate DOM IDs (${idCounts.size} unique IDs in web/index.html)`);
}

// 2. Every CONTROL_REGISTRY entry exists in web/index.html with matching data-ctl
const regIds = new Set(CONTROL_REGISTRY.map((c) => c.id));
if (regIds.size !== CONTROL_REGISTRY.length) {
  fail(`Duplicate IDs inside CONTROL_REGISTRY`);
}
const missingReg = [];
for (const id of regIds) {
  if (!idCounts.has(id)) missingReg.push(id);
}
if (missingReg.length > 0) {
  fail(`Missing CONTROL_REGISTRY IDs in web/index.html: ${missingReg.join(', ')}`);
} else {
  pass(`All ${regIds.size} CONTROL_REGISTRY IDs present in web/index.html`);
}

// 3. Every [data-ctl="..."] in web/index.html is either in CONTROL_REGISTRY or meta bosBusStrip
const dataCtls = [...html.matchAll(/\bdata-ctl="([^"]+)"/g)].map((m) => m[1]);
const unregistered = dataCtls.filter((id) => !regIds.has(id) && id !== 'bosBusStrip');
if (unregistered.length > 0) {
  fail(`Unregistered data-ctl attributes in web/index.html: ${unregistered.join(', ')}`);
} else {
  pass(`All ${dataCtls.length} [data-ctl] elements are registered`);
}

// 4. Every BAY_PANEL_OWNERSHIP ID exists in web/index.html
const missingPanels = Object.keys(BAY_PANEL_OWNERSHIP).filter((id) => !idCounts.has(id));
if (missingPanels.length > 0) {
  fail(`Missing BAY_PANEL_OWNERSHIP IDs in web/index.html: ${missingPanels.join(', ')}`);
} else {
  pass(`All ${Object.keys(BAY_PANEL_OWNERSHIP).length} Scope Bay panel/canvas IDs present`);
}

// 5. All 22 worker-transferred canvases + auditTimelineCanvas exist
const requiredCanvases = [
  'hudCanvas', 'pipCanvas', 'persistenceCanvas', 'hilbertCanvas', 'spectralCanvas', 'physicsPhaseCanvas', 'auditTimelineCanvas',
  ...Array.from({ length: 16 }, (_, i) => `slotPip${i}`),
];
const missingCanvases = requiredCanvases.filter((id) => !idCounts.has(id));
if (missingCanvases.length > 0) {
  fail(`Missing required <canvas> elements: ${missingCanvases.join(', ')}`);
} else {
  pass(`All ${requiredCanvases.length} static <canvas> elements present`);
}

// 6. Syntax-check every JS file under web/ui and web/worker.js
function walkJs(dir) {
  const out = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) out.push(...walkJs(full));
    else if (ent.name.endsWith('.js') || ent.name.endsWith('.mjs')) out.push(full);
  }
  return out;
}
const jsFiles = [...walkJs(path.join(ROOT, 'web/ui')), path.join(ROOT, 'web/worker.js')];
let syntaxErrs = 0;
for (const f of jsFiles) {
  try {
    execFileSync(process.execPath, ['--check', f], { stdio: 'pipe' });
  } catch (err) {
    syntaxErrs++;
    fail(`Syntax error in ${path.relative(ROOT, f)}: ${err.stderr?.toString() || err.message}`);
  }
}
if (syntaxErrs === 0) {
  pass(`Node --check passed on all ${jsFiles.length} browser JS/worker modules`);
}

// 6b. Worker-UI contract checks (scopeVisibilityMask bit alignment & showKalmanBaseline)
const workerSrc = fs.readFileSync(path.join(ROOT, 'web/worker.js'), 'utf8');
const benchShellSrc = fs.readFileSync(path.join(ROOT, 'web/ui/views/bench-shell.js'), 'utf8');
if (/\bshowKalman\s*=/.test(workerSrc)) {
  fail('web/worker.js assigns to undeclared showKalman instead of showKalmanBaseline');
} else if (!/track:\s*0x09/.test(benchShellSrc) || !/phase:\s*0x0a/.test(benchShellSrc) || !/physics:\s*0x0c/.test(benchShellSrc)) {
  fail('BAY_SCOPE_MASK in web/ui/views/bench-shell.js must keep bit3 (Roster 0x08) on and match worker bit0=Track, bit1=Phase, bit2=Physics');
} else {
  pass('Worker-UI message contracts (scopeVisibilityMask bits & showKalmanBaseline) verified');
}

// 7. Zero em dashes in v3.5 HTML, CSS, and UI JS modules
const checkFiles = [
  HTML_PATH,
  ...fs.readdirSync(path.join(ROOT, 'web/styles')).filter((n) => n.endsWith('.css')).map((n) => path.join(ROOT, 'web/styles', n)),
  ...walkJs(path.join(ROOT, 'web/ui')),
];
const emDashHits = [];
for (const f of checkFiles) {
  const txt = fs.readFileSync(f, 'utf8');
  if (txt.includes('\u2014')) emDashHits.push(path.relative(ROOT, f));
}
if (emDashHits.length > 0) {
  fail(`Em dash (\\u2014) found in: ${emDashHits.join(', ')}`);
} else {
  pass(`Zero em dashes across ${checkFiles.length} v3.5 UI files`);
}

if (failures > 0) {
  console.error(`\nUI contract lint FAILED with ${failures} error(s).`);
  process.exit(1);
}
console.log('\nALL v3.5 UI CONTRACT LINT CHECKS PASSED.');
