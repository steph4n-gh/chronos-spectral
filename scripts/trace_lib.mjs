// ============================================================================
// CHRONOS-SPECTRAL // Kernel trace library (shared by trace_hook.mjs,
// trace_replay.mjs and record_canonical_traces.mjs). Zero dependencies.
//
//   - xxh32():            the digest hash (standard XXH32, seed 0)
//   - PLATFORM_WORDS / xxh32Masked(): the platform-descriptive words zeroed
//                         before hashing (format 1.1, meta.digestMask)
//   - parseWasmExports(): export name -> parameter/result value types
//   - buildRegionTable(): the named memory regions, derived from src/*.rs
//   - encodeHeader() / TraceReader: the binary trace format (CSTRACE v1)
//
// The format is specified in docs/VERIFICATION_AND_BENCHMARKS.md
// ("Kernel record-and-replay traces"). Everything is little-endian.
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const DEFAULT_WASM = path.join(ROOT, 'web', 'chronos_spectral.wasm');

// ---------------------------------------------------------------------------
// XXH32 (Yann Collet's xxHash, 32-bit variant, seed 0). Region digests are
// XXH32 over the region's raw bytes in memory order.
// ---------------------------------------------------------------------------
const P1 = 0x9e3779b1 | 0;
const P2 = 0x85ebca77 | 0;
const P3 = 0xc2b2ae3d | 0;
const P4 = 0x27d4eb2f | 0;
const P5 = 0x165667b1 | 0;

function rotl(x, r) {
  return (x << r) | (x >>> (32 - r));
}

function round(acc, lane) {
  acc = (acc + Math.imul(lane, P2)) | 0;
  return Math.imul(rotl(acc, 13), P1);
}

/** XXH32 of u8[off, off+len). `u8` is a Uint8Array; returns an unsigned 32-bit number. */
export function xxh32(u8, off = 0, len = u8.length - off, seed = 0) {
  const abs = u8.byteOffset + off;
  // Fast path: aligned u32 view over the same buffer (little-endian hosts only).
  const w = (abs & 3) === 0 && LITTLE_ENDIAN ? new Int32Array(u8.buffer, abs, len >>> 2) : null;
  const dv = w ? null : new DataView(u8.buffer, abs, len);
  const word = w ? (i) => w[i] : (i) => dv.getInt32(i * 4, true);
  let h;
  let i = 0; // word index
  const nStripes = len >>> 4;
  if (len >= 16) {
    let v1 = (seed + P1 + P2) | 0;
    let v2 = (seed + P2) | 0;
    let v3 = seed | 0;
    let v4 = (seed - P1) | 0;
    if (w) {
      for (let s = 0; s < nStripes; s++, i += 4) {
        v1 = Math.imul(rotl((v1 + Math.imul(w[i], P2)) | 0, 13), P1);
        v2 = Math.imul(rotl((v2 + Math.imul(w[i + 1], P2)) | 0, 13), P1);
        v3 = Math.imul(rotl((v3 + Math.imul(w[i + 2], P2)) | 0, 13), P1);
        v4 = Math.imul(rotl((v4 + Math.imul(w[i + 3], P2)) | 0, 13), P1);
      }
    } else {
      for (let s = 0; s < nStripes; s++, i += 4) {
        v1 = round(v1, word(i));
        v2 = round(v2, word(i + 1));
        v3 = round(v3, word(i + 2));
        v4 = round(v4, word(i + 3));
      }
    }
    h = (rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18)) | 0;
  } else {
    h = (seed + P5) | 0;
  }
  h = (h + len) | 0;
  let p = i * 4;
  while (p + 4 <= len) {
    h = (h + Math.imul(word(p >>> 2), P3)) | 0;
    h = Math.imul(rotl(h, 17), P4);
    p += 4;
  }
  while (p < len) {
    h = (h + Math.imul(u8[off + p], P5)) | 0;
    h = Math.imul(rotl(h, 11), P1);
    p++;
  }
  h ^= h >>> 15;
  h = Math.imul(h, P2);
  h ^= h >>> 13;
  h = Math.imul(h, P3);
  h ^= h >>> 16;
  return h >>> 0;
}

export const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

// ---------------------------------------------------------------------------
// Minimal wasm binary parser: type (1), import (2), function (3), export (7).
// ---------------------------------------------------------------------------
export const VT = { 0x7f: 'i32', 0x7e: 'i64', 0x7d: 'f32', 0x7c: 'f64', 0x7b: 'v128' };
export const VT_SIZE = { 0x7f: 4, 0x7e: 8, 0x7d: 4, 0x7c: 8 };

/** Returns Map(exportName -> { kind, index, params:[u8], results:[u8] }) plus memory info. */
export function parseWasmExports(bytes) {
  const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let p = 8;
  if (b[0] !== 0 || b[1] !== 0x61 || b[2] !== 0x73 || b[3] !== 0x6d) throw new Error('not a wasm module');
  const leb = () => {
    let r = 0;
    let shift = 0;
    for (;;) {
      const x = b[p++];
      r += (x & 0x7f) * 2 ** shift;
      shift += 7;
      if ((x & 0x80) === 0) return r;
    }
  };
  const name = () => {
    const n = leb();
    const s = new TextDecoder().decode(b.subarray(p, p + n));
    p += n;
    return s;
  };
  const types = [];
  const funcTypes = []; // function index -> type index (imports first)
  let importedFuncs = 0;
  let memPages = null;
  const exportsMap = new Map();
  while (p < b.length) {
    const id = b[p++];
    const size = leb();
    const end = p + size;
    if (id === 1) {
      const n = leb();
      for (let k = 0; k < n; k++) {
        if (b[p++] !== 0x60) throw new Error('bad functype');
        const np = leb();
        const params = Array.from(b.subarray(p, p + np));
        p += np;
        const nr = leb();
        const results = Array.from(b.subarray(p, p + nr));
        p += nr;
        types.push({ params, results });
      }
    } else if (id === 2) {
      const n = leb();
      for (let k = 0; k < n; k++) {
        name();
        name();
        const kind = b[p++];
        if (kind === 0) {
          funcTypes.push(leb());
          importedFuncs++;
        } else if (kind === 1) {
          p++;
          const fl = leb();
          leb();
          if (fl & 1) leb();
        } else if (kind === 2) {
          const fl = leb();
          memPages = leb();
          if (fl & 1) leb();
        } else if (kind === 3) {
          p += 2;
        } else throw new Error(`unsupported import kind ${kind}`);
      }
    } else if (id === 3) {
      const n = leb();
      for (let k = 0; k < n; k++) funcTypes.push(leb());
    } else if (id === 5) {
      const n = leb();
      if (n > 0) {
        const fl = leb();
        memPages = leb();
        if (fl & 1) leb();
      }
    } else if (id === 7) {
      const n = leb();
      for (let k = 0; k < n; k++) {
        const nm = name();
        const kind = b[p++];
        const index = leb();
        exportsMap.set(nm, { kind, index });
      }
    }
    p = end;
  }
  for (const e of exportsMap.values()) {
    if (e.kind === 0) {
      const t = types[funcTypes[e.index]];
      e.params = t.params;
      e.results = t.results;
    }
  }
  return { exports: exportsMap, importedFuncs, memPages };
}

// ---------------------------------------------------------------------------
// Region table. Each entry names the getter export that returns the region's
// pointer and the Rust backing field whose declared array type gives its
// length; lengths are evaluated from the `const` items in src/*.rs, so a
// constant change in Rust changes the table. Direction comes from the getter's
// Rust return type (*const -> out, *mut -> both), with `in` overrides for the
// pure host-input staging buffers. `alias` regions are moving views onto a ring
// slot: host writes may be addressed through them, but they are not digested
// separately (the ring_slot[k] digests cover the same bytes).
// ---------------------------------------------------------------------------
const REGION_SPECS = [
  { name: 'ring_slot', getter: 'chronos_get_ring_slot_ptr', argCount: 'RING_LEN', file: 'lib.rs', field: 'rgba_ring', inner: true },
  { name: 'ingest_rgba', getter: 'chronos_get_ingest_rgba_ptr', file: 'lib.rs', field: 'rgba_ring', inner: true, alias: true, dir: 'in' },
  { name: 'display_rgba', getter: 'chronos_get_display_rgba_ptr', file: 'lib.rs', field: 'rgba_ring', inner: true, alias: true },
  { name: 'eca_res', getter: 'chronos_get_eca_res_ptr', file: 'lib.rs', field: 'eca_res' },
  { name: 'pip_rgba', getter: 'chronos_get_pip_rgba_ptr', file: 'lib.rs', field: 'pip_rgba' },
  { name: 'all_pip_rgba', getter: 'chronos_get_all_pip_rgba_ptr', file: 'lib.rs', field: 'all_pip_rgba' },
  { name: 'slot_meta', getter: 'chronos_get_slot_meta_ptr', file: 'lib.rs', field: 'slot_meta' },
  { name: 'laplacian', getter: 'chronos_get_laplacian_ptr', file: 'lib.rs', field: 'laplacian' },
  { name: 'fiedler', getter: 'chronos_get_fiedler_ptr', file: 'lib.rs', field: 'fiedler_vec' },
  { name: 'telemetry', getter: 'chronos_get_telemetry_ptr', file: 'lib.rs', field: 'telemetry' },
  { name: 'micro_tracks', getter: 'chronos_get_micro_tracks_ptr', file: 'soa_tracks.rs', field: 'export_micro' },
  { name: 'continuum_field', getter: 'chronos_get_continuum_field_ptr', file: 'continuum_physics.rs', field: 'export_field' },
  { name: 'rigid_bodies', getter: 'chronos_get_rigid_bodies_ptr', file: 'continuum_physics.rs', field: 'export_bodies' },
  { name: 'swarm_telemetry', getter: 'chronos_get_swarm_telemetry_ptr', file: 'soa_tracks.rs', field: 'swarm_telem' },
  { name: 'modal_field', getter: 'chronos_get_modal_field_ptr', file: 'continuum_physics.rs', field: 'export_modal_field' },
  { name: 'ring_meta', getter: 'chronos_get_ring_meta_ptr', file: 'lib.rs', field: 'ring_meta' },
  { name: 'row_luma', getter: 'chronos_get_row_luma_ptr', file: 'lib.rs', field: 'row_luma_means' },
  { name: 'wave1_telemetry', getter: 'chronos_get_wave1_telemetry_ptr', file: 'lib.rs', field: 'wave1_telem' },
  { name: 'bos_telemetry', getter: 'chronos_get_bos_telemetry_ptr', file: 'bos.rs', field: 'telem' },
  { name: 'bos_planes', getter: 'chronos_get_bos_planes_ptr', file: 'bos.rs', field: 'planes' },
  { name: 'bos_valid', getter: 'chronos_get_bos_valid_ptr', file: 'bos.rs', field: 'valid_mask' },
  { name: 'bos_rgba', getter: 'chronos_get_bos_rgba_ptr', file: 'bos.rs', field: 'rgba' },
  { name: 'micro_cov', getter: 'chronos_get_micro_cov_ptr', file: 'soa_tracks.rs', field: 'export_micro_cov' },
  { name: 'micro_residual', getter: 'chronos_get_micro_residual_ptr', file: 'soa_tracks.rs', field: 'export_micro_residual' },
  { name: 'micro_loom', getter: 'chronos_get_micro_loom_ptr', file: 'soa_tracks.rs', field: 'export_micro_loom_d2' },
  { name: 'micro_quality', getter: 'chronos_get_micro_quality_ptr', file: 'soa_tracks.rs', field: 'export_micro_quality' },
  { name: 'hero_candidates', getter: 'chronos_get_hero_candidates_ptr', file: 'soa_tracks.rs', field: 'export_hero_candidates' },
  { name: 'wave2_telemetry', getter: 'chronos_get_wave2_telemetry_ptr', file: 'lib.rs', field: 'wave2_telem' },
  { name: 'audio_staging', getter: 'chronos_get_audio_staging_ptr', file: 'vibrometry.rs', field: 'audio_staging', dir: 'in' },
  { name: 'vibro_telemetry', getter: 'chronos_get_vibro_telemetry_ptr', file: 'vibrometry.rs', field: 'telem' },
  { name: 'vibro_planes', getter: 'chronos_get_vibro_planes_ptr', file: 'vibrometry.rs', field: 'planes' },
  { name: 'hodge_telemetry', getter: 'chronos_get_hodge_telemetry_ptr', file: 'hodge.rs', field: 'telem' },
  { name: 'hodge_planes', getter: 'chronos_get_hodge_planes_ptr', file: 'hodge.rs', field: 'planes' },
  { name: 'hodge_inject', getter: 'chronos_get_hodge_inject_ptr', file: 'hodge.rs', field: 'inject_buf', dir: 'in' },
  { name: 'conservation_telemetry', getter: 'chronos_get_conservation_telemetry_ptr', file: 'physics_id.rs', field: 'telem' },
  { name: 'conservation_input', getter: 'chronos_get_conservation_input_ptr', file: 'physics_id.rs', field: 'input', dir: 'in' },
  { name: 'fovea_rgba', getter: 'chronos_get_fovea_rgba_ptr', argCount: '1', file: 'lib.rs', field: 'rgba', dir: 'in' },
  { name: 'fovea_request', getter: 'chronos_get_fovea_request_ptr', file: 'lib.rs', field: 'request_meta' },
  { name: 'depth_telemetry', getter: 'chronos_get_depth_telemetry_ptr', file: 'depth.rs', field: 'telem' },
  { name: 'depth_field', getter: 'chronos_get_depth_field_ptr', file: 'depth.rs', field: 'export_field' },
  { name: 'depth_tracks', getter: 'chronos_get_depth_tracks_ptr', file: 'depth.rs', field: 'export_track_depth' },
  { name: 'depth_track_flags', getter: 'chronos_get_depth_track_flags_ptr', file: 'depth.rs', field: 'export_track_flags' },
  { name: 'depth_hero', getter: 'chronos_get_depth_hero_ptr', file: 'depth.rs', field: 'export_hero_depth' },
  { name: 'depth_ingest', getter: 'chronos_get_depth_ingest_ptr', file: 'depth.rs', field: 'ext_depth', dir: 'in' },
];

export const ELEM = { u8: 1, u32: 2, f32: 3 };
export const ELEM_NAME = { 1: 'u8', 2: 'u32', 3: 'f32' };
export const ELEM_SIZE = { 1: 1, 2: 4, 3: 4 };
export const RFLAG = { HOST_WRITABLE: 1, DIGESTED: 2, ALIAS: 4, INPUT_ONLY: 8 };

function readRustSources(srcDir) {
  const files = new Map();
  for (const f of fs.readdirSync(srcDir)) if (f.endsWith('.rs')) files.set(f, fs.readFileSync(path.join(srcDir, f), 'utf8'));
  return files;
}

/** Evaluates `const NAME: usize = EXPR;` items, resolving names per file first, then crate-wide. */
function makeConstEvaluator(files) {
  const defs = new Map(); // file -> Map(name -> expr)
  const global = new Map();
  const re = /^\s*(?:pub(?:\([^)]*\))?\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:\s*(?:usize|u32|u64|i32)\s*=\s*([^;]+);/gm;
  for (const [f, txt] of files) {
    const m = new Map();
    const cleaned = txt.replace(/#\[cfg\(all\(feature\s*=\s*"hires-720p"[^\n]*\)\)\]\s*\n[^\n]*/g, '');
    for (const x of cleaned.matchAll(re)) {
      m.set(x[1], x[2].trim());
      if (!global.has(x[1])) global.set(x[1], { file: f, expr: x[2].trim() });
    }
    defs.set(f, m);
  }
  const sizeOf = new Map();
  for (const txt of files.values()) {
    for (const x of txt.matchAll(/size_of::<([A-Za-z0-9_]+)>\(\)\s*==\s*([0-9_]+)/g)) sizeOf.set(x[1], Number(x[2].replace(/_/g, '')));
  }
  const evalExpr = (expr, file, depth = 0) => {
    if (depth > 32) throw new Error(`const recursion too deep in ${expr}`);
    const toks = expr.replace(/\bas\s+(usize|u32|u64|i32)\b/g, '').match(/[A-Za-z_][A-Za-z0-9_:]*|[0-9][0-9_]*|[()+\-*/]/g) || [];
    let i = 0;
    const atom = () => {
      const t = toks[i++];
      if (t === '(') {
        const v = sum();
        if (toks[i++] !== ')') throw new Error(`unbalanced ${expr}`);
        return v;
      }
      if (/^[0-9]/.test(t)) return Number(t.replace(/_/g, ''));
      const nm = t.split('::').pop();
      const local = defs.get(file)?.get(nm);
      if (local !== undefined) return evalExpr(local, file, depth + 1);
      const g = global.get(nm);
      if (g) return evalExpr(g.expr, g.file, depth + 1);
      throw new Error(`unknown Rust constant ${t} (in ${file}: ${expr})`);
    };
    const prod = () => {
      let v = atom();
      while (toks[i] === '*' || toks[i] === '/') v = toks[i++] === '*' ? v * atom() : Math.floor(v / atom());
      return v;
    };
    const sum = () => {
      let v = prod();
      while (toks[i] === '+' || toks[i] === '-') v = toks[i++] === '+' ? v + prod() : v - prod();
      return v;
    };
    const v = sum();
    if (i !== toks.length || !Number.isSafeInteger(v)) throw new Error(`cannot evaluate Rust expr "${expr}"`);
    return v;
  };
  return { evalExpr, sizeOf };
}

/** Parses `[T; N]`, `[[T; N]; M]` into { elem, outer, inner } (elements of T per outer element). */
function parseArrayType(ty, file, ev) {
  const m2 = ty.match(/^\[\s*\[\s*([A-Za-z0-9_]+)\s*;\s*([^\]]+)\]\s*;\s*([^\]]+)\]$/);
  if (m2) return { elem: m2[1], inner: ev.evalExpr(m2[2], file), outer: ev.evalExpr(m2[3], file) };
  const m1 = ty.match(/^\[\s*([A-Za-z0-9_]+)\s*;\s*([^\]]+)\]$/);
  if (m1) return { elem: m1[1], inner: 1, outer: ev.evalExpr(m1[2], file) };
  throw new Error(`unsupported field type ${ty}`);
}

/**
 * Builds the region table from src/*.rs. Returns
 * [{ name, getter, hasArg, arg, elem, byteLen, flags, dir, source }] in table
 * order (ring_slot expands to ring_slot[0..RING_LEN)).
 */
export function buildRegionTable(srcDir = path.join(ROOT, 'src')) {
  const files = readRustSources(srcDir);
  const ev = makeConstEvaluator(files);
  const lib = files.get('lib.rs');
  // Every pointer-returning export in lib.rs must be covered.
  const getterSig = new Map();
  for (const x of lib.matchAll(/extern\s+"C"\s+fn\s+(chronos_[a-z0-9_]+)\s*\(([^)]*)\)\s*->\s*\*(const|mut)\s+([a-z0-9]+)/g)) {
    getterSig.set(x[1], { args: x[2].trim(), mut: x[3] === 'mut', ty: x[4] });
  }
  const covered = new Set(REGION_SPECS.map((s) => s.getter));
  const missing = [...getterSig.keys()].filter((g) => !covered.has(g));
  if (missing.length) throw new Error(`pointer getters without a region spec in scripts/trace_lib.mjs: ${missing.join(', ')}`);
  const table = [];
  for (const s of REGION_SPECS) {
    const sig = getterSig.get(s.getter);
    if (!sig) throw new Error(`region getter ${s.getter} not found in src/lib.rs`);
    const txt = files.get(s.file);
    const fm = txt.match(new RegExp(`^\\s*pub\\s+${s.field}\\s*:\\s*(\\[[^\\n]*\\])\\s*,`, 'm'));
    if (!fm) throw new Error(`field ${s.field} not found in src/${s.file}`);
    const at = parseArrayType(fm[1].trim(), s.file, ev);
    let elemBytes;
    if (['u8', 'u32', 'f32'].includes(at.elem)) elemBytes = at.elem === 'u8' ? 1 : 4;
    else if (ev.sizeOf.has(at.elem)) elemBytes = ev.sizeOf.get(at.elem);
    else throw new Error(`unknown element size for ${at.elem}`);
    const total = (s.inner ? at.inner : at.inner * at.outer) * elemBytes;
    const elem = ELEM[sig.ty];
    if (!elem) throw new Error(`getter ${s.getter} returns unsupported *${sig.ty}`);
    if (total % ELEM_SIZE[elem] !== 0) throw new Error(`${s.name}: ${total} B not a multiple of ${sig.ty}`);
    const dir = s.dir || (sig.mut ? 'both' : 'out');
    let flags = 0;
    if (sig.mut) flags |= RFLAG.HOST_WRITABLE;
    if (!s.alias) flags |= RFLAG.DIGESTED;
    if (s.alias) flags |= RFLAG.ALIAS;
    if (dir === 'in') flags |= RFLAG.INPUT_ONLY;
    const source = `src/${s.file}: ${s.field}: ${fm[1].trim()}`;
    if (s.argCount) {
      if (!/^\w+\s*:\s*u32$/.test(sig.args)) throw new Error(`${s.getter} expected one u32 arg`);
      const n = ev.evalExpr(s.argCount, 'lib.rs');
      for (let k = 0; k < n; k++) table.push({ name: `${s.name}[${k}]`, getter: s.getter, hasArg: true, arg: k, elem, byteLen: total, flags, dir, source });
    } else {
      if (sig.args !== '') throw new Error(`${s.getter} expected no args`);
      table.push({ name: s.name, getter: s.getter, hasArg: false, arg: 0, elem, byteLen: total, flags, dir, source });
    }
  }
  return table;
}

/**
 * Platform-descriptive words: 32-bit words of digested regions whose value
 * describes the build (pointer width, linker layout), not the computation, so
 * the wasm32 kernel and a native kernel legitimately disagree on them. The
 * recorder zeroes exactly these words before hashing a region and writes the
 * list into every trace as `meta.digestMask`; replayers hash the same way. The
 * native replay driver refuses a trace whose mask is not exactly its own copy
 * (native/replay/allowlist.rs), so a trace cannot widen the exemption.
 */
export const PLATFORM_WORDS = [
  { region: 'telemetry', word: 20, why: 'size_of::<Arena>() as f32: the arena holds usize fields, so its size depends on pointer width' },
  { region: 'wave1_telemetry', word: 1, why: 'size_of::<Arena>() as u32 (pointer-width dependent, as above)' },
  { region: 'wave1_telemetry', word: 12, why: "the wasm linker's __heap_base (end of the static image); 0 in a native build" },
  { region: 'wave2_telemetry', word: 1, why: 'size_of::<Arena>() as u32 (pointer-width dependent, as above)' },
];

/** `{ region: [word, ...] }` for PLATFORM_WORDS, words ascending (the `meta.digestMask` value). */
export function digestMask() {
  const m = {};
  for (const p of PLATFORM_WORDS) (m[p.region] ||= []).push(p.word);
  for (const k of Object.keys(m)) m[k].sort((a, b) => a - b);
  return m;
}

/**
 * XXH32 of region bytes u8[off, off+len) with the 32-bit words listed in
 * `words` (word indices within the region) read as zero. `scratch` is a
 * reusable Uint8Array of at least `len` bytes.
 */
export function xxh32Masked(u8, off, len, words, scratch) {
  if (!words || !words.length) return xxh32(u8, off, len);
  const s = scratch.subarray(0, len);
  s.set(u8.subarray(off, off + len));
  for (const w of words) s.fill(0, 4 * w, 4 * w + 4);
  return xxh32(s, 0, len);
}

/**
 * Exports that do not write kernel memory (getters; padic_ultrametric_val is a
 * leaf xor/clz). Their calls carry no digests. CHRONOS_TRACE_STRICT=1 plus
 * CHRONOS_TRACE_MEMDIGEST=1 verifies this on recorded sessions.
 */
export function isPureExport(name) {
  return name.startsWith('chronos_get_') || name === 'padic_ultrametric_val';
}

// ---------------------------------------------------------------------------
// Trace format (CSTRACE v1)
// ---------------------------------------------------------------------------
export const MAGIC = 'CSTRACE\0';
export const VERSION_MAJOR = 1;
// 1.1: region digests are computed with the words listed in meta.digestMask zeroed.
export const VERSION_MINOR = 1;
export const TAG = { WRITE: 0x57, UNMAPPED: 0x55, RESET: 0x52, CALL: 0x43, END: 0x45 };
export const FFLAG = { POINTER_RESULT: 1, PURE: 2 };
export const HFLAG = { STRICT: 1, MEMDIGEST: 2 };
export const END_REASON = { 0: 'exit', 1: 'instance-cap', 2: 'total-cap', 3: 'instance-collected' };

class ByteWriter {
  constructor() {
    this.buf = Buffer.alloc(4096);
    this.n = 0;
  }
  need(k) {
    if (this.n + k > this.buf.length) {
      const nb = Buffer.alloc(Math.max(this.buf.length * 2, this.n + k));
      this.buf.copy(nb, 0, 0, this.n);
      this.buf = nb;
    }
  }
  u8(v) { this.need(1); this.buf[this.n++] = v; }
  u16(v) { this.need(2); this.buf.writeUInt16LE(v, this.n); this.n += 2; }
  u32(v) { this.need(4); this.buf.writeUInt32LE(v >>> 0, this.n); this.n += 4; }
  bytes(b) { this.need(b.length); this.buf.set(b, this.n); this.n += b.length; }
  str16(s) { const b = Buffer.from(s, 'utf8'); this.u16(b.length); this.bytes(b); }
  out() { return this.buf.subarray(0, this.n); }
}

/** funcs: [{name, params:[u8], results:[u8], flags}], regions: buildRegionTable() entries with getterIndex. */
export function encodeHeader({ wasmSha256, memBytes, flags, meta, funcs, regions }) {
  const w = new ByteWriter();
  w.bytes(Buffer.from(MAGIC, 'latin1'));
  w.u16(VERSION_MAJOR);
  w.u16(VERSION_MINOR);
  w.u32(0); // header length, patched below
  w.bytes(wasmSha256);
  w.u32(memBytes);
  w.u32(flags);
  const mj = Buffer.from(JSON.stringify(meta), 'utf8');
  w.u32(mj.length);
  w.bytes(mj);
  w.u16(funcs.length);
  for (const f of funcs) {
    w.str16(f.name);
    w.u8(f.params.length);
    for (const t of f.params) w.u8(t);
    w.u8(f.results.length);
    for (const t of f.results) w.u8(t);
    w.u8(f.flags);
  }
  w.u16(regions.length);
  for (const r of regions) {
    w.str16(r.name);
    w.u16(r.getterIndex);
    w.u8(r.hasArg ? 1 : 0);
    w.u32(r.arg);
    w.u8(r.elem);
    w.u32(r.byteLen);
    w.u8(r.flags);
  }
  const out = Buffer.from(w.out());
  out.writeUInt32LE(out.length, 12);
  return out;
}

/**
 * Streaming trace reader. `header` is parsed eagerly; records() yields
 *   { tag:'W', pos, region, offset, data }
 *   { tag:'U', pos, addr, data }
 *   { tag:'R', pos }   (memory := fresh-instance image)
 *   { tag:'C', pos, seq, func, args:Buffer, status, result:Buffer|null, digests:Uint32Array|null, memDigest:number|null }
 *   { tag:'E', pos, reason, nCalls, nWrites, nUnmapped }
 * and finally { tag:'EOF', clean:boolean, pos } (clean = an END record was seen and nothing follows).
 */
export class TraceReader {
  constructor(file) {
    this.fd = fs.openSync(file, 'r');
    this.size = fs.fstatSync(this.fd).size;
    this.chunk = Buffer.alloc(0);
    this.chunkPos = 0; // file offset of chunk[0]
    this.pos = 0;
    this.header = this.readHeader();
  }
  ensure(n) {
    // Guarantees chunk covers [pos, pos+n); returns false at EOF.
    if (this.pos + n > this.size) return false;
    if (this.pos >= this.chunkPos && this.pos + n <= this.chunkPos + this.chunk.length) return true;
    const len = Math.min(this.size - this.pos, Math.max(n, 64 << 20));
    this.chunk = Buffer.allocUnsafe(len);
    fs.readSync(this.fd, this.chunk, 0, len, this.pos);
    this.chunkPos = this.pos;
    return true;
  }
  take(n) {
    if (!this.ensure(n)) throw new RangeError('truncated');
    const o = this.pos - this.chunkPos;
    this.pos += n;
    return this.chunk.subarray(o, o + n);
  }
  u8() { return this.take(1)[0]; }
  u16() { return this.take(2).readUInt16LE(0); }
  u32() { return this.take(4).readUInt32LE(0); }
  str16() { return this.take(this.u16()).toString('utf8'); }
  readHeader() {
    const magic = this.take(8).toString('latin1');
    if (magic !== MAGIC) throw new Error('not a CSTRACE file');
    const major = this.u16();
    const minor = this.u16();
    if (major !== VERSION_MAJOR) throw new Error(`unsupported trace version ${major}.${minor}`);
    const headerLen = this.u32();
    const wasmSha256 = Buffer.from(this.take(32)).toString('hex');
    const memBytes = this.u32();
    const flags = this.u32();
    const meta = JSON.parse(this.take(this.u32()).toString('utf8'));
    const funcs = [];
    for (let n = this.u16(), k = 0; k < n; k++) {
      const name = this.str16();
      const params = [...this.take(this.u8())];
      const results = [...this.take(this.u8())];
      const fl = this.u8();
      funcs.push({ name, params, results, flags: fl });
    }
    const regions = [];
    for (let n = this.u16(), k = 0; k < n; k++) {
      regions.push({
        name: this.str16(),
        getterIndex: this.u16(),
        hasArg: this.u8() === 1,
        arg: this.u32(),
        elem: this.u8(),
        byteLen: this.u32(),
        flags: this.u8(),
      });
    }
    if (this.pos !== headerLen) throw new Error(`header length mismatch ${this.pos} != ${headerLen}`);
    const digested = regions.map((r, i) => i).filter((i) => regions[i].flags & RFLAG.DIGESTED);
    return { major, minor, headerLen, wasmSha256, memBytes, flags, meta, funcs, regions, digested };
  }
  *records() {
    const h = this.header;
    let ended = false;
    for (;;) {
      const pos = this.pos;
      if (pos >= this.size) {
        yield { tag: 'EOF', clean: ended, pos };
        return;
      }
      let rec;
      try {
        const tag = this.u8();
        if (tag === TAG.WRITE) {
          const region = this.u16();
          const offset = this.u32();
          const data = this.take(this.u32());
          rec = { tag: 'W', pos, region, offset, data };
        } else if (tag === TAG.RESET) {
          rec = { tag: 'R', pos };
        } else if (tag === TAG.UNMAPPED) {
          const addr = this.u32();
          rec = { tag: 'U', pos, addr, data: this.take(this.u32()) };
        } else if (tag === TAG.CALL) {
          const seq = this.u32();
          const func = this.u16();
          const f = h.funcs[func];
          if (!f) throw new Error(`bad function index ${func}`);
          const args = this.take(f.params.reduce((a, t) => a + VT_SIZE[t], 0));
          const status = this.u8();
          const result = status === 0 && f.results.length ? this.take(VT_SIZE[f.results[0]]) : null;
          const hasDigests = this.u8();
          const digests = hasDigests & 1 ? new Uint32Array(Uint8Array.from(this.take(4 * h.digested.length)).buffer) : null;
          const memDigest = hasDigests & 2 ? this.u32() : null;
          rec = { tag: 'C', pos, seq, func, args, status, result, digests, memDigest };
        } else if (tag === TAG.END) {
          rec = { tag: 'E', pos, reason: this.u8(), nCalls: this.u32(), nWrites: this.u32(), nUnmapped: this.u32() };
          ended = true;
        } else {
          throw new Error(`bad record tag 0x${tag.toString(16)} at byte ${pos}`);
        }
      } catch (e) {
        if (e instanceof RangeError) {
          // A torn final record (process killed mid-flush): stop at the last whole record.
          yield { tag: 'EOF', clean: false, pos, torn: true };
          return;
        }
        throw e;
      }
      if (ended && this.pos < this.size) throw new Error(`data after END record at byte ${this.pos}`);
      yield rec;
    }
  }
  close() {
    fs.closeSync(this.fd);
  }
}

/** Decodes raw little-endian argument bytes into JS values for the given wasm param types. */
export function decodeValues(buf, types) {
  const out = [];
  let o = 0;
  for (const t of types) {
    if (t === 0x7f) { out.push(buf.readInt32LE(o)); o += 4; }
    else if (t === 0x7d) { out.push(buf.readFloatLE(o)); o += 4; }
    else if (t === 0x7c) { out.push(buf.readDoubleLE(o)); o += 8; }
    else if (t === 0x7e) { out.push(buf.readBigInt64LE(o)); o += 8; }
    else throw new Error(`unsupported value type 0x${t.toString(16)}`);
  }
  return out;
}

/** Encodes one JS value (already converted as wasm would) into `buf` at `o`; returns new offset. */
export function encodeValue(buf, o, t, v) {
  if (t === 0x7f) { buf.writeInt32LE(v | 0, o); return o + 4; }
  if (t === 0x7d) { buf.writeFloatLE(v, o); return o + 4; }
  if (t === 0x7c) { buf.writeDoubleLE(v, o); return o + 8; }
  if (t === 0x7e) { buf.writeBigInt64LE(BigInt.asIntN(64, v), o); return o + 8; }
  throw new Error(`unsupported value type 0x${t.toString(16)}`);
}

/** Formats one decoded value for messages. */
export function fmtValue(t, v) {
  if (t === 0x7f) return `${v} (0x${(v >>> 0).toString(16)})`;
  return String(v);
}
