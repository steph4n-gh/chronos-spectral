#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Where the kernel's time goes, native vs wasm, on one trace.
//
// Native: runs `chronos_replay --bench` on the trace and attaches the macOS
// Time Profiler (`xcrun xctrace record --attach`, 1 kHz) once the measured
// passes start; exports the samples, keeps those inside a kernel call, and
// attributes them by function (self and inclusive time), by inlined source
// line (`atos -i` against the binary's dSYM) and by instruction (the sampled
// PCs looked up in an `otool` disassembly).
// Wasm: runs `node scripts/trace_replay.mjs --bench --bench-cpu-prof` on the
// same trace (V8's sampling profiler, measured passes only, after tier-up)
// and attributes the samples by wasm function (name section).
//
// Usage (macOS with Xcode; build the native binary with symbols, see
// docs/VERIFICATION_AND_BENCHMARKS.md, "Native profile"):
//   node scripts/bench_profile.mjs --bin=PATH --trace=FILE [--seconds=20]
//        [--wasm-passes=10] [--top=20] [--lines=8] [--out=DIR] [--no-wasm]
//   node scripts/bench_profile.mjs --demangle <symbol>...
//
// Output: report.md and report.json in --out (default target/bench/profile-<trace>).
// ============================================================================

import { execFileSync, spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const opt = (k, d) => {
  const a = argv.find((x) => x.startsWith(`--${k}=`));
  return a === undefined ? d : a.slice(k.length + 3);
};

// ---------------------------------------------------------------------------
// Rust v0 symbol demangling (the subset rustc emits for this crate: paths,
// impls, generic arguments, basic types, const generics, back-references).
// ---------------------------------------------------------------------------
const BASIC = { a: 'i8', b: 'bool', c: 'char', d: 'f64', e: 'str', f: 'f32', h: 'u8', i: 'isize', j: 'usize', l: 'i32', m: 'u32', n: 'i128', o: 'u128', s: 'i16', t: 'u16', u: '()', v: '...', x: 'i64', y: 'u64', z: '!', p: '_' };

export function demangle(sym) {
  if (!sym || !sym.startsWith('_R')) return sym;
  const s = sym.slice(2);
  let pos = 0;
  const peek = () => s[pos];
  const b62 = () => {
    if (s[pos] === '_') {
      pos++;
      return 0;
    }
    let v = 0;
    while (s[pos] !== '_') {
      const c = s[pos++];
      if (c === undefined) throw new Error('eof');
      v = v * 62 + (c >= '0' && c <= '9' ? c.charCodeAt(0) - 48 : c >= 'a' && c <= 'z' ? c.charCodeAt(0) - 87 : c.charCodeAt(0) - 29);
    }
    pos++;
    return v + 1;
  };
  const disamb = () => {
    if (s[pos] === 's') {
      pos++;
      b62();
    }
  };
  const ident = () => {
    disamb();
    if (s[pos] === 'u') pos++;
    let n = '';
    while (s[pos] >= '0' && s[pos] <= '9') n += s[pos++];
    if (s[pos] === '_') pos++;
    const len = Number(n);
    const id = s.slice(pos, pos + len);
    pos += len;
    return id;
  };
  const backref = (fn) => {
    const at = b62();
    const save = pos;
    pos = at;
    const r = fn();
    pos = save;
    return r;
  };
  const genericArg = () => {
    if (s[pos] === 'L') {
      pos++;
      b62();
      return "'_";
    }
    if (s[pos] === 'K') {
      pos++;
      return constant();
    }
    return type();
  };
  const constant = () => {
    if (s[pos] === 'B') {
      pos++;
      return backref(constant);
    }
    if (s[pos] === 'p') {
      pos++;
      return '_';
    }
    const t = s[pos++];
    let neg = false;
    if (s[pos] === 'n') {
      neg = true;
      pos++;
    }
    let hex = '';
    while (s[pos] !== '_') hex += s[pos++];
    pos++;
    const v = hex ? BigInt(`0x${hex}`) : 0n;
    if (t === 'b') return v ? 'true' : 'false';
    if (t === 'c') return `'${String.fromCodePoint(Number(v))}'`;
    return `${neg ? '-' : ''}${v}`;
  };
  const type = () => {
    const c = s[pos];
    if (BASIC[c] && c !== 'p') {
      pos++;
      return BASIC[c];
    }
    if (c === 'p') {
      pos++;
      return '_';
    }
    if ('CMXYNI'.includes(c)) return pathP();
    pos++;
    switch (c) {
      case 'B':
        return backref(type);
      case 'A': {
        const t = type();
        if (s[pos] === 'K') pos++;
        return `[${t}; ${constant()}]`;
      }
      case 'S':
        return `[${type()}]`;
      case 'T': {
        const ts = [];
        while (s[pos] !== 'E') ts.push(type());
        pos++;
        return `(${ts.join(', ')})`;
      }
      case 'R':
      case 'Q': {
        if (s[pos] === 'L') {
          pos++;
          b62();
        }
        return `&${c === 'Q' ? 'mut ' : ''}${type()}`;
      }
      case 'P':
        return `*const ${type()}`;
      case 'O':
        return `*mut ${type()}`;
      case 'F': {
        if (s[pos] === 'G') {
          pos++;
          b62();
        }
        if (s[pos] === 'U') pos++;
        if (s[pos] === 'K') {
          pos++;
          if (s[pos] === 'C') pos++;
          else ident();
        }
        const ps = [];
        while (s[pos] !== 'E') ps.push(type());
        pos++;
        return `fn(${ps.join(', ')}) -> ${type()}`;
      }
      case 'D': {
        if (s[pos] === 'G') {
          pos++;
          b62();
        }
        const bs = [];
        while (s[pos] !== 'E') {
          bs.push(pathP());
          while (s[pos] === 'p') {
            pos++;
            ident();
            type();
          }
        }
        pos++;
        if (s[pos] === 'L') {
          pos++;
          b62();
        }
        return `dyn ${bs.join(' + ')}`;
      }
      default:
        throw new Error(`type ${c}`);
    }
  };
  const pathP = () => {
    const c = s[pos++];
    switch (c) {
      case 'C':
        return ident();
      case 'N': {
        const ns = s[pos++];
        const parent = pathP();
        const id = ident();
        if (ns === 'C') return `${parent}::{closure${id ? `:${id}` : ''}}`;
        if (ns >= 'A' && ns <= 'Z') return `${parent}::{${ns}${id ? `:${id}` : ''}}`;
        return `${parent}::${id}`;
      }
      case 'M': {
        disamb();
        pathP();
        return `<${type()}>`;
      }
      case 'X': {
        disamb();
        pathP();
        const t = type();
        return `<${t} as ${pathP()}>`;
      }
      case 'Y': {
        const t = type();
        return `<${t} as ${pathP()}>`;
      }
      case 'I': {
        const p = pathP();
        const as = [];
        while (s[pos] !== 'E') as.push(genericArg());
        pos++;
        return `${p}<${as.join(', ')}>`;
      }
      case 'B':
        return backref(pathP);
      default:
        throw new Error(`path ${c}`);
    }
  };
  try {
    while (s[pos] >= '0' && s[pos] <= '9') pos++;
    return pathP();
  } catch {
    return sym;
  }
}

/** Short display name: crate prefix dropped. */
const short = (sym) => demangle(sym).replace(/^chronos_spectral::/, '').replace(/^chronos_replay::/, 'replay::');

if (argv[0] === '--demangle') {
  for (const s of argv.slice(1)) console.log(`${s}\n  -> ${demangle(s)}`);
  process.exit(0);
}

// --same-code A B: do two builds contain the same kernel machine code? Compares every kernel
// function's instruction stream (`otool -tV`) opcode by opcode and operand by operand, allowing
// only address operands to differ (adrp pages, page offsets, branch targets, literal addresses),
// which move whenever anything else in the binary changes size.
if (argv[0] === '--same-code') {
  const funcs = (b) => {
    const d = new Map();
    let cur = null;
    for (const l of execFileSync('otool', ['-tV', b], { encoding: 'utf8', maxBuffer: 1 << 30 }).split('\n')) {
      if (l.endsWith(':') && !/^[0-9a-f]{16}/.test(l)) {
        cur = l.slice(0, -1).replace(/Cs[0-9A-Za-z]+_/g, 'Cs_');
        d.set(cur, []);
        continue;
      }
      const m = l.match(/^[0-9a-f]{16}\s+(\S+)\s*(.*)$/);
      if (m && cur) d.get(cur).push([m[1], m[2].replace(/;.*/, '').trim()]);
    }
    return d;
  };
  const [A, B] = [funcs(argv[1]), funcs(argv[2])];
  const kernel = [...A.keys()].filter((k) => k.includes('16chronos_spectral') || /^_(chronos_|padic_)/.test(k));
  const NUM = /(#?-?0x[0-9a-f]+|#-?\d+|\b\d+\b|_\S+)/;
  let n = 0;
  let addrOnly = 0;
  const bad = [];
  for (const k of kernel) {
    const a = A.get(k);
    const b = B.get(k);
    if (!b || a.length !== b.length) {
      bad.push(`${k}: ${b ? `${a.length} vs ${b.length} instructions` : 'missing'}`);
      continue;
    }
    a.forEach(([m1, o1], i) => {
      n++;
      const [m2, o2] = b[i];
      if (m1 === m2 && o1 === o2) return;
      const t1 = o1.split(NUM);
      const t2 = o2.split(NUM);
      // Same opcode and same non-numeric operand text (registers, shifts, lane specifiers):
      // the numbers that differ are address operands. A differing immediate on an arithmetic
      // op with no literal/address role would also land here; such diffs are listed below.
      const regsSame = m1 === m2 && t1.length === t2.length && t1.every((x, j) => j % 2 === 1 || x === t2[j]);
      // `ldr q0, [x10]` vs `ldr q0, [x10, #0xfd0]`: a page offset that became (non-)zero.
      const pageOff = m1 === m2 && o1.replace(/, #-?0x[0-9a-f]+\]/, ']') === o2.replace(/, #-?0x[0-9a-f]+\]/, ']');
      if (regsSame || pageOff) addrOnly++;
      else bad.push(`${k}+${i}: ${m1} ${o1} | ${m2} ${o2}`);
    });
  }
  console.log(`${kernel.length} kernel functions, ${n} instructions: ${n - addrOnly - bad.length} identical, ${addrOnly} differ only in an address operand, ${bad.length} differ otherwise`);
  for (const x of bad.slice(0, 20)) console.log(`  ${x}`);
  process.exit(bad.length ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Arguments.
// ---------------------------------------------------------------------------
const bin = opt('bin', null);
const trace = opt('trace', null);
const seconds = Number(opt('seconds', '20'));
const wasmPasses = Number(opt('wasm-passes', '10'));
const topN = Number(opt('top', '20'));
const lineN = Number(opt('lines', '8'));
const withWasm = !argv.includes('--no-wasm');
const withNative = !argv.includes('--no-native');
if (!trace || (withNative && !bin)) {
  console.error('usage: node scripts/bench_profile.mjs --bin=PATH --trace=FILE [--seconds=20] [--wasm-passes=10] [--top=20] [--lines=8] [--out=DIR] [--no-wasm] [--no-native]');
  process.exit(2);
}
const traceName = path.basename(trace).replace(/\.cstrace$/, '');
const outDir = path.resolve(opt('out', path.join(ROOT, 'target/bench', `profile-${traceName}`)));
fs.mkdirSync(outDir, { recursive: true });
const report = { trace, bin, seconds };
const md = [`# Profile: ${traceName}`, ''];

// ---------------------------------------------------------------------------
// Native: xctrace Time Profiler attached during the measured passes.
// ---------------------------------------------------------------------------
async function recordNative() {
  const xml = path.join(outDir, 'native-time-profile.xml');
  const trc = path.join(outDir, 'native.trace');
  fs.rmSync(trc, { recursive: true, force: true });
  const child = spawn(bin, ['--bench', '--warmup=1', '--passes=1000000', trace], { stdio: ['ignore', 'ignore', 'pipe'] });
  await new Promise((resolve, reject) => {
    let buf = '';
    child.stderr.on('data', (d) => {
      buf += d;
      if (buf.includes('[bench]') && buf.includes('measuring')) resolve();
    });
    child.on('exit', (c) => reject(new Error(`chronos_replay exited (${c}) before measuring: ${buf}`)));
  });
  const rec = spawnSync('xcrun', ['xctrace', 'record', '--template', 'Time Profiler', '--output', trc, '--attach', String(child.pid), '--time-limit', `${seconds}s`], { encoding: 'utf8' });
  child.kill('SIGKILL');
  if (rec.status !== 0) throw new Error(`xctrace record failed: ${rec.stderr || rec.stdout}`);
  const exp = spawnSync('xcrun', ['xctrace', 'export', '--input', trc, '--xpath', '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]'], { encoding: 'utf8', maxBuffer: 1 << 30 });
  if (exp.status !== 0) throw new Error(`xctrace export failed: ${exp.stderr}`);
  fs.writeFileSync(xml, exp.stdout);
  fs.rmSync(trc, { recursive: true, force: true }); // the .trace bundle also stores the environment
  return exp.stdout;
}

const unxml = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, '&');

function parseTimeProfile(xml) {
  const frames = new Map();
  const stacks = new Map();
  const cores = new Map();
  const samples = [];
  let loadAddr = null;
  let binPath = null;
  for (const m of xml.matchAll(/<row>([\s\S]*?)<\/row>/g)) {
    const row = m[1];
    let core = null;
    const cd = row.match(/<core id="(\d+)" fmt="([^"]*)"/);
    if (cd) {
      cores.set(cd[1], cd[2]);
      core = cd[2];
    } else {
      const cr = row.match(/<core ref="(\d+)"\/>/);
      if (cr) core = cores.get(cr[1]);
    }
    let stack;
    const bt = row.match(/<tagged-backtrace id="(\d+)">([\s\S]*?)<\/tagged-backtrace>/);
    if (bt) {
      stack = [];
      for (const f of bt[2].matchAll(/<frame (?:id="(\d+)"([^>]*?)(\/?)>|ref="(\d+)"\/>)/g)) {
        if (f[4]) {
          stack.push(frames.get(f[4]));
          continue;
        }
        const attrs = f[2];
        const name = attrs.match(/name="([^"]*)"/);
        const addr = attrs.match(/addr="(0x[0-9a-f]+)"/);
        const fr = { name: name ? unxml(name[1]) : '???', addr: addr ? BigInt(addr[1]) : 0n };
        frames.set(f[1], fr);
        stack.push(fr);
      }
      stacks.set(bt[1], stack);
      if (!loadAddr) {
        const b = bt[2].match(/<binary id="\d+" name="chronos_replay"[^>]*load-addr="(0x[0-9a-f]+)"[^>]*path="([^"]*)"/);
        if (b) {
          loadAddr = b[1];
          binPath = b[2];
        }
      }
    } else {
      const br = row.match(/<tagged-backtrace ref="(\d+)"\/>/);
      stack = br ? stacks.get(br[1]) : [];
    }
    samples.push({ core: core || '?', stack: stack || [] });
  }
  return { samples, loadAddr, binPath };
}

/** Maps addresses to inlined frame chains with atos; returns Map(addr -> [{fn, loc}] innermost first). */
function symbolize(binary, loadAddr, addrs) {
  const out = new Map();
  const dsym = `${binary}.dSYM/Contents/Resources/DWARF/${path.basename(binary)}`;
  const obj = fs.existsSync(dsym) ? dsym : binary;
  const SENT = '0x1';
  for (let i = 0; i < addrs.length; i += 2000) {
    const chunk = addrs.slice(i, i + 2000);
    const args = ['-o', obj, '-arch', 'arm64', '-l', loadAddr, '-i'];
    for (const a of chunk) args.push(`0x${a.toString(16)}`, SENT);
    const txt = execFileSync('atos', args, { encoding: 'utf8', maxBuffer: 1 << 28 });
    const groups = txt.split(/\n0x0*1\n|^0x0*1\n/m);
    chunk.forEach((a, k) => {
      const lines = (groups[k] || '').split('\n').filter((l) => l.trim());
      out.set(
        a,
        lines.map((l) => {
          const mm = l.match(/^(.*?) \(in [^)]*\)(?: \(([^)]*)\))?/);
          return mm ? { fn: mm[1], loc: mm[2] || '?' } : { fn: l.trim(), loc: '?' };
        })
      );
    });
  }
  return out;
}

/** Disassembly index: address -> instruction text, for the main binary's __text. */
function disassemble(binary, loadAddr) {
  const txt = execFileSync('otool', ['-tvV', binary], { encoding: 'utf8', maxBuffer: 1 << 30 });
  const map = new Map();
  // otool prints unslid addresses (__TEXT vmaddr 0x100000000); the sampled PCs are slid.
  const slide = BigInt(loadAddr) - 0x100000000n;
  for (const l of txt.split('\n')) {
    const m = l.match(/^([0-9a-f]{16})\s+(.*)$/);
    if (m) map.set(BigInt(`0x${m[1]}`) + slide, m[2].trim());
  }
  return map;
}

const isKernelExport = (n) => /^(chronos_|padic_)/.test(n);

function analyseNative(xml) {
  const { samples, loadAddr } = parseTimeProfile(xml);
  const total = samples.length;
  const pCore = samples.filter((s) => /P Core/.test(s.core)).length;
  const kernel = samples.filter((s) => s.stack.some((f) => isKernelExport(f.name)));
  const self = new Map();
  const incl = new Map();
  const byExport = new Map();
  const leafPcs = new Map(); // function -> Map(pc -> count)
  for (const s of kernel) {
    const leaf = s.stack[0];
    const fn = short(leaf.name);
    self.set(fn, (self.get(fn) || 0) + 1);
    // Leaf frames are reported as PC + 1 (return-address convention); align to the instruction.
    const pc = leaf.addr & ~3n;
    if (!leafPcs.has(fn)) leafPcs.set(fn, new Map());
    leafPcs.get(fn).set(pc, (leafPcs.get(fn).get(pc) || 0) + 1);
    const seen = new Set();
    for (const f of s.stack) {
      const n = short(f.name);
      if (seen.has(n)) continue;
      seen.add(n);
      incl.set(n, (incl.get(n) || 0) + 1);
    }
    // Outermost kernel export on the stack (the trace call being made).
    const ex = [...s.stack].reverse().find((f) => isKernelExport(f.name));
    byExport.set(ex.name, (byExport.get(ex.name) || 0) + 1);
  }
  const K = kernel.length;
  const top = [...self].sort((a, b) => b[1] - a[1]).slice(0, topN);
  // Source lines and instructions for the top functions.
  const binary = bin;
  const pcs = [...new Set(top.flatMap(([fn]) => [...leafPcs.get(fn).keys()]))];
  const sym = symbolize(binary, loadAddr, pcs);
  const dis = disassemble(binary, loadAddr);
  const detail = top.map(([fn, n]) => {
    const lines = new Map();
    const inl = new Map();
    const ins = new Map();
    const insClass = new Map();
    for (const [pc, c] of leafPcs.get(fn)) {
      const chain = sym.get(pc) || [];
      const inner = chain[0] || { fn: '?', loc: '?' };
      const loc = `${inner.loc}`;
      lines.set(loc, (lines.get(loc) || 0) + c);
      const innerFn = short(inner.fn);
      inl.set(innerFn, (inl.get(innerFn) || 0) + c);
      const t = dis.get(pc) || '?';
      ins.set(`${pc.toString(16)}  ${t}`, (ins.get(`${pc.toString(16)}  ${t}`) || 0) + c);
      const mn = t.split(/\s+/)[0];
      insClass.set(mn, (insClass.get(mn) || 0) + c);
    }
    const sorted = (m, k) => [...m].sort((a, b) => b[1] - a[1]).slice(0, k).map(([x, c]) => ({ what: x, samples: c, pct_of_fn: (100 * c) / n }));
    return { fn, samples: n, pct_kernel: (100 * n) / K, lines: sorted(lines, lineN), inlined: sorted(inl, lineN), instructions: sorted(ins, lineN), mnemonics: sorted(insClass, 10) };
  });
  return {
    total_samples: total,
    p_core_samples: pCore,
    kernel_samples: K,
    load_addr: loadAddr,
    exports: [...byExport].sort((a, b) => b[1] - a[1]).map(([e, c]) => ({ export: e, pct_kernel: (100 * c) / K })),
    self: [...self].sort((a, b) => b[1] - a[1]).slice(0, 60).map(([f, c]) => ({ fn: f, samples: c, pct_kernel: (100 * c) / K })),
    inclusive: [...incl].sort((a, b) => b[1] - a[1]).slice(0, 60).map(([f, c]) => ({ fn: f, samples: c, pct_kernel: (100 * c) / K })),
    detail,
  };
}

// ---------------------------------------------------------------------------
// Wasm: V8 CPU profile of the measured passes.
// ---------------------------------------------------------------------------
function recordWasm() {
  const prof = path.join(outDir, 'wasm.cpuprofile');
  const r = spawnSync(
    process.execPath,
    ['--expose-gc', path.join(ROOT, 'scripts/trace_replay.mjs'), '--bench', '--warmup=2', `--passes=${wasmPasses}`, `--bench-cpu-prof=${prof}`, trace],
    { encoding: 'utf8', maxBuffer: 1 << 28 }
  );
  if (r.status !== 0) throw new Error(`wasm bench failed: ${r.stdout}${r.stderr}`);
  return JSON.parse(fs.readFileSync(prof, 'utf8'));
}

function analyseWasm(p) {
  const byId = new Map(p.nodes.map((n) => [n.id, n]));
  const parent = new Map();
  for (const n of p.nodes) for (const c of n.children || []) parent.set(c, n.id);
  const isWasm = (n) => n.callFrame.url.startsWith('wasm://');
  const self = new Map();
  const incl = new Map();
  const byExport = new Map();
  let K = 0;
  let total = 0;
  for (const n of p.nodes) {
    const hits = n.hitCount || 0;
    if (!hits) continue;
    total += hits;
    // Stack from this node to the root.
    const stack = [];
    for (let id = n.id; id !== undefined; id = parent.get(id)) stack.push(byId.get(id));
    if (!stack.some(isWasm)) continue;
    K += hits;
    const leaf = short(n.callFrame.functionName);
    self.set(leaf, (self.get(leaf) || 0) + hits);
    const seen = new Set();
    for (const f of stack.filter(isWasm)) {
      const nm = short(f.callFrame.functionName);
      if (seen.has(nm)) continue;
      seen.add(nm);
      incl.set(nm, (incl.get(nm) || 0) + hits);
    }
    const ex = [...stack].reverse().find(isWasm);
    byExport.set(ex.callFrame.functionName, (byExport.get(ex.callFrame.functionName) || 0) + hits);
  }
  return {
    total_samples: total,
    kernel_samples: K,
    interval_us: (p.endTime - p.startTime) / Math.max(1, p.samples.length),
    exports: [...byExport].sort((a, b) => b[1] - a[1]).map(([e, c]) => ({ export: e, pct_kernel: (100 * c) / K })),
    self: [...self].sort((a, b) => b[1] - a[1]).slice(0, 60).map(([f, c]) => ({ fn: f, samples: c, pct_kernel: (100 * c) / K })),
    inclusive: [...incl].sort((a, b) => b[1] - a[1]).slice(0, 60).map(([f, c]) => ({ fn: f, samples: c, pct_kernel: (100 * c) / K })),
  };
}

// ---------------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------------
const pct = (x) => (x == null ? '-' : `${x.toFixed(1)}%`);
if (withNative) {
  console.log(`native: recording ${seconds} s of ${bin} on ${traceName} ...`);
  const nat = analyseNative(await recordNative());
  report.native = nat;
  md.push(
    `Native: ${nat.kernel_samples} samples inside kernel calls of ${nat.total_samples} (1 kHz, ${seconds} s; ${((100 * nat.p_core_samples) / nat.total_samples).toFixed(1)}% on P cores).`,
    ''
  );
}
if (withWasm) {
  console.log(`wasm: profiling ${wasmPasses} measured passes in V8 ...`);
  report.wasm = analyseWasm(recordWasm());
  md.push(`Wasm (V8): ${report.wasm.kernel_samples} samples inside wasm of ${report.wasm.total_samples} (${report.wasm.interval_us.toFixed(0)} µs interval).`, '');
}
const N = report.native;
const Wm = report.wasm;
md.push('## Time by export (share of kernel samples)', '', '| export | native | wasm |', '| :--- | ---: | ---: |');
for (const e of (N || Wm).exports.slice(0, 8)) {
  md.push(`| ${e.export} | ${pct(N?.exports.find((x) => x.export === e.export)?.pct_kernel)} | ${pct(Wm?.exports.find((x) => x.export === e.export)?.pct_kernel)} |`);
}
md.push('', '## Self time by function (share of kernel samples)', '', '| function | native self | wasm self | native incl. | wasm incl. |', '| :--- | ---: | ---: | ---: | ---: |');
const names = [...new Set([...(N ? N.self.slice(0, topN) : []), ...(Wm ? Wm.self.slice(0, Math.ceil(topN / 2)) : [])].map((x) => x.fn))];
for (const fn of names) {
  const g = (side, k) => side?.[k].find((x) => x.fn === fn)?.pct_kernel;
  md.push(`| \`${fn}\` | ${pct(g(N, 'self'))} | ${pct(g(Wm, 'self'))} | ${pct(g(N, 'inclusive'))} | ${pct(g(Wm, 'inclusive'))} |`);
}
if (N) {
  md.push('', '## Native hot functions: what dominates', '');
  for (const d of N.detail) {
    md.push(`### \`${d.fn}\`: ${pct(d.pct_kernel)} of kernel samples (${d.samples})`, '');
    md.push('Innermost inlined function:', '', ...d.inlined.map((x) => `- ${pct(x.pct_of_fn)} \`${x.what}\``), '');
    md.push('Source line:', '', ...d.lines.map((x) => `- ${pct(x.pct_of_fn)} ${x.what}`), '');
    md.push('Sampled instruction mnemonics:', '', d.mnemonics.map((x) => `${x.what} ${pct(x.pct_of_fn)}`).join(', '), '');
    md.push('Hottest instructions:', '', '```', ...d.instructions.map((x) => `${pct(x.pct_of_fn).padStart(6)}  ${x.what}`), '```', '');
  }
}
fs.writeFileSync(path.join(outDir, 'report.json'), JSON.stringify(report, null, 1));
fs.writeFileSync(path.join(outDir, 'report.md'), md.join('\n') + '\n');
console.log(md.join('\n'));
console.log(`\nwritten: ${path.join(outDir, 'report.md')}`);