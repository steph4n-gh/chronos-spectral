// ============================================================================
// CHRONOS-SPECTRAL // Transparent kernel session recorder (Node preload).
//
//   CHRONOS_TRACE_DIR=/tmp/traces node --import ./scripts/trace_hook.mjs scripts/gate_i3_hodge.mjs
//   CHRONOS_TRACE_DIR=/tmp/traces NODE_OPTIONS="--import=$PWD/scripts/trace_hook.mjs" node ...
//
// Inert unless CHRONOS_TRACE_DIR is set. When active it patches
// WebAssembly.{Module, Instance, compile, compileStreaming, instantiate,
// instantiateStreaming} so every instance of the CHRONOS kernel (recognised by
// a `chronos_init` export) gets wrapped exports that record, per instance, a
// CSTRACE v1 file (format: scripts/trace_lib.mjs and
// docs/VERIFICATION_AND_BENCHMARKS.md). Other wasm modules are untouched.
//
// Per call the recorder:
//   1. diffs the whole linear memory against the snapshot taken after the
//      previous kernel call, so every byte the host changed since then is
//      found (kernel writes are already in the snapshot);
//   2. names each changed range by region + offset (region = getter export +
//      argument). If some bytes fall outside every region but the memory
//      equals the fresh-instance image plus region writes (a pooled instance
//      being reset), that is recorded as RESET + region writes; anything else
//      outside every region is recorded as UNMAPPED and reported on stderr;
//   3. calls the real export with the arguments converted exactly as the wasm
//      JS API converts them, records them and the result;
//   4. re-snapshots the memory and records the XXH32 digest of every digested
//      region (with the platform-descriptive words of PLATFORM_WORDS in
//      trace_lib.mjs zeroed; the mask is written into the header metadata as
//      `digestMask`).
// Getters (`chronos_get_*`, `padic_ultrametric_val`) do not write kernel
// memory, so by default they skip step 4 (only the shadow stack is
// re-synchronised after them, and they carry no digests).
// CHRONOS_TRACE_STRICT=1 re-snapshots and digests after every call; a getter
// that wrote outside the shadow stack would then show up as a digest change
// (and, in default mode, as a host write before the next call).
//
// Environment:
//   CHRONOS_TRACE_DIR              output directory (activates the hook)
//   CHRONOS_TRACE_LABEL            file-name prefix (default: script basename)
//   CHRONOS_TRACE_MAX_BYTES        per-instance cap (default 4 GiB)
//   CHRONOS_TRACE_TOTAL_MAX_BYTES  per-process cap over all instances (default 16 GiB)
//   CHRONOS_TRACE_MAX_INSTANCES    record at most N kernel instances per process
//   CHRONOS_TRACE_INSTANCES        only record these kernel instances, by creation
//                                  order in the process, e.g. "0-3,100,300-303"
//   CHRONOS_TRACE_STRICT=1         digest and diff around every call (validation)
//   CHRONOS_TRACE_MEMDIGEST=N      validation: every Nth digested call also carries an
//                                  XXH32 of all memory above the shadow stack, so a
//                                  replay proves no host write was missed anywhere
//   CHRONOS_TRACE_LOG=0            silence the per-trace summary lines on stderr
// When a cap is reached the trace ends cleanly (END record) at the last whole
// call; the program keeps running unrecorded.
// ============================================================================

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { threadId } from 'node:worker_threads';
import {
  buildRegionTable,
  digestMask,
  encodeHeader,
  encodeValue,
  isPureExport,
  parseWasmExports,
  xxh32,
  xxh32Masked,
  FFLAG,
  HFLAG,
  RFLAG,
  TAG,
  VT_SIZE,
} from './trace_lib.mjs';

const TRACE_DIR = process.env.CHRONOS_TRACE_DIR;
const ORIGINALS_KEY = Symbol.for('chronos.trace.originals');

if (TRACE_DIR && !globalThis[ORIGINALS_KEY]) install();

function envInt(name, dflt) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return dflt;
  const v = Number(raw);
  if (!Number.isFinite(v) || v <= 0) throw new Error(`[chronos-trace] invalid ${name}=${raw}`);
  return Math.floor(v);
}

function parseSelection(raw) {
  if (!raw) return null;
  const ranges = raw.split(',').filter(Boolean).map((t) => {
    const m = t.trim().match(/^(\d+)(?:-(\d+))?$/);
    if (!m) throw new Error(`[chronos-trace] invalid CHRONOS_TRACE_INSTANCES item "${t}"`);
    return [Number(m[1]), Number(m[2] ?? m[1])];
  });
  return (i) => ranges.some(([a, b]) => i >= a && i <= b);
}

function install() {
  const WA = globalThis.WebAssembly;
  const orig = {
    Module: WA.Module,
    Instance: WA.Instance,
    compile: WA.compile,
    compileStreaming: WA.compileStreaming,
    instantiate: WA.instantiate,
    instantiateStreaming: WA.instantiateStreaming,
  };
  globalThis[ORIGINALS_KEY] = orig; // the replayer uses these to stay unrecorded

  const cfg = {
    dir: path.resolve(TRACE_DIR),
    label: (process.env.CHRONOS_TRACE_LABEL || path.basename(process.argv[1] || 'stdin').replace(/\.[cm]?js$/, '')).replace(/[^A-Za-z0-9_.-]/g, '_'),
    maxBytes: envInt('CHRONOS_TRACE_MAX_BYTES', 4 * 2 ** 30),
    totalMaxBytes: envInt('CHRONOS_TRACE_TOTAL_MAX_BYTES', 16 * 2 ** 30),
    maxInstances: envInt('CHRONOS_TRACE_MAX_INSTANCES', Number.MAX_SAFE_INTEGER),
    select: parseSelection(process.env.CHRONOS_TRACE_INSTANCES),
    strict: process.env.CHRONOS_TRACE_STRICT === '1',
    memDigestEvery: envInt('CHRONOS_TRACE_MEMDIGEST', 0),
    log: process.env.CHRONOS_TRACE_LOG !== '0',
  };
  fs.mkdirSync(cfg.dir, { recursive: true });

  const proc = { seen: 0, instances: 0, totalBytes: 0, skippedLogged: false, open: new Set() };
  const moduleInfo = new WeakMap(); // Module -> { bytes } (kernel modules only)
  let regionTable = null;

  const isKernelModule = (mod) => orig.Module.exports(mod).some((e) => e.name === 'chronos_init' && e.kind === 'function');
  const noteModule = (mod, bytesLike) => {
    if (!moduleInfo.has(mod) && isKernelModule(mod)) {
      const src = ArrayBuffer.isView(bytesLike)
        ? new Uint8Array(bytesLike.buffer, bytesLike.byteOffset, bytesLike.byteLength)
        : new Uint8Array(bytesLike);
      moduleInfo.set(mod, { bytes: Uint8Array.from(src) });
    }
    return mod;
  };

  const wrap = (instance, mod) => {
    if (!instance?.exports || typeof instance.exports.chronos_init !== 'function') return instance;
    const info = moduleInfo.get(mod);
    if (!info) {
      console.error('[chronos-trace] ERROR: kernel instance from a module whose bytes were not seen; NOT recorded');
      return instance;
    }
    const ordinal = proc.seen++;
    if (cfg.select && !cfg.select(ordinal)) return instance;
    if (proc.instances >= cfg.maxInstances) {
      if (!proc.skippedLogged && cfg.log) console.error(`[chronos-trace] CHRONOS_TRACE_MAX_INSTANCES=${cfg.maxInstances} reached; further instances unrecorded`);
      proc.skippedLogged = true;
      return instance;
    }
    if (!info.parsed) {
      info.parsed = parseWasmExports(info.bytes);
      info.sha256 = crypto.createHash('sha256').update(info.bytes).digest();
    }
    if (!regionTable) regionTable = buildRegionTable();
    proc.instances++;
    const seq = ordinal;
    const rec = new Recorder(instance.exports, info, regionTable, cfg, proc, seq);
    Object.defineProperty(instance, 'exports', { value: rec.wrappedExports, enumerable: true, configurable: true });
    return instance;
  };

  WA.Module = new Proxy(orig.Module, {
    construct(target, args, newTarget) {
      return noteModule(Reflect.construct(target, args, newTarget), args[0]);
    },
  });
  WA.Instance = new Proxy(orig.Instance, {
    construct(target, args, newTarget) {
      return wrap(Reflect.construct(target, args, newTarget), args[0]);
    },
  });
  WA.compile = async function compile(bytes, ...rest) {
    return noteModule(await orig.compile.call(WA, bytes, ...rest), bytes);
  };
  WA.instantiate = async function instantiate(source, ...rest) {
    const out = await orig.instantiate.call(WA, source, ...rest);
    if (out instanceof orig.Instance) return wrap(out, source); // (Module, imports) overload
    noteModule(out.module, source);
    wrap(out.instance, out.module);
    return out;
  };
  const streamingBytes = async (source) => {
    const resp = await source;
    return resp.arrayBuffer();
  };
  if (orig.compileStreaming) {
    WA.compileStreaming = async function compileStreaming(source) {
      return WA.compile(await streamingBytes(source));
    };
  }
  if (orig.instantiateStreaming) {
    WA.instantiateStreaming = async function instantiateStreaming(source, imports) {
      return WA.instantiate(await streamingBytes(source), imports);
    };
  }

  process.on('exit', () => {
    for (const w of [...proc.open]) w.finish(0);
    if (cfg.log && proc.unmappedTotal) console.error(`[chronos-trace] WARNING: ${proc.unmappedTotal} UNMAPPED host write range(s) recorded in this process`);
  });
  const finalizer = new FinalizationRegistry((writer) => writer.finish(3));
  proc.finalizer = finalizer;
}

// ---------------------------------------------------------------------------
// Trace writer: buffered appends, caps, END record, index.jsonl summary line.
// ---------------------------------------------------------------------------
class TraceWriter {
  constructor(file, header, cfg, proc, meta) {
    this.file = file;
    this.cfg = cfg;
    this.proc = proc;
    this.meta = meta;
    this.pending = [header];
    this.pendingBytes = header.length;
    this.bytes = header.length;
    this.calls = 0;
    this.writes = 0;
    this.writeBytes = 0;
    this.unmapped = 0;
    this.resets = 0;
    this.digested = 0;
    this.done = false;
    this.endReason = null;
    proc.totalBytes += header.length;
    fs.writeFileSync(file, Buffer.alloc(0));
    proc.open.add(this);
  }
  /** Returns false (and ends the trace) when the step would exceed a cap. */
  commit(step, stats) {
    if (this.done) return false;
    const n = step.length;
    if (this.bytes + n + 14 > this.cfg.maxBytes) return this.finish(1), false;
    if (this.proc.totalBytes + n + 14 > this.cfg.totalMaxBytes) return this.finish(2), false;
    this.pending.push(step);
    this.pendingBytes += n;
    this.bytes += n;
    this.proc.totalBytes += n;
    this.calls += 1;
    this.writes += stats.writes;
    this.writeBytes += stats.writeBytes;
    this.unmapped += stats.unmapped;
    this.resets += stats.resets;
    this.digested += stats.digested;
    if (this.pendingBytes >= 8 << 20) this.flush();
    return true;
  }
  flush() {
    if (!this.pending.length) return;
    fs.appendFileSync(this.file, Buffer.concat(this.pending, this.pendingBytes));
    this.pending = [];
    this.pendingBytes = 0;
  }
  finish(reason) {
    if (this.done) return;
    this.done = true;
    this.endReason = reason;
    const end = Buffer.alloc(14);
    end[0] = TAG.END;
    end[1] = reason;
    end.writeUInt32LE(this.calls, 2);
    end.writeUInt32LE(this.writes, 6);
    end.writeUInt32LE(this.unmapped, 10);
    this.pending.push(end);
    this.pendingBytes += end.length;
    this.bytes += end.length;
    this.flush();
    this.pending = null;
    this.proc.open.delete(this);
    const summary = {
      file: path.basename(this.file),
      label: this.meta.label,
      instance: this.meta.instance,
      pid: this.meta.pid,
      calls: this.calls,
      digestedCalls: this.digested,
      hostWrites: this.writes,
      hostWriteBytes: this.writeBytes,
      unmapped: this.unmapped,
      resets: this.resets,
      bytes: this.bytes,
      end: ['exit', 'instance-cap', 'total-cap', 'instance-collected'][reason],
    };
    fs.appendFileSync(path.join(this.cfg.dir, 'index.jsonl'), JSON.stringify(summary) + '\n');
    if (this.cfg.log) {
      console.error(`[chronos-trace] ${summary.file}: ${summary.calls} calls, ${summary.hostWrites} host writes, ${summary.resets} resets, ${summary.unmapped} unmapped, ${(summary.bytes / 1e6).toFixed(1)} MB (${summary.end})`);
    }
  }
}

/** Growable byte buffer for one step (host writes + one CALL record). */
class StepBuf {
  constructor() {
    this.buf = Buffer.allocUnsafe(1 << 16);
    this.n = 0;
  }
  reset() { this.n = 0; }
  need(k) {
    if (this.n + k <= this.buf.length) return;
    const nb = Buffer.allocUnsafe(Math.max(this.buf.length * 2, this.n + k));
    this.buf.copy(nb, 0, 0, this.n);
    this.buf = nb;
  }
  u8(v) { this.need(1); this.buf[this.n++] = v; }
  u16(v) { this.need(2); this.buf.writeUInt16LE(v, this.n); this.n += 2; }
  u32(v) { this.need(4); this.buf.writeUInt32LE(v >>> 0, this.n); this.n += 4; }
  copyFrom(src, a, b) { this.need(b - a); src.copy(this.buf, this.n, a, b); this.n += b - a; }
  out() { return Buffer.from(this.buf.subarray(0, this.n)); }
}

const CHUNK = 4096;
const BLOCK = 65536;
const MERGE_GAP_WORDS = 8;

// ---------------------------------------------------------------------------
// Per-instance recorder.
// ---------------------------------------------------------------------------
class Recorder {
  constructor(E, info, regionTable, cfg, proc, seq) {
    this.E = E;
    this.cfg = cfg;
    this.proc = proc;
    this.mem = E.memory;
    this.memBytes = this.mem.buffer.byteLength;
    this.stackTop = stackTopOf(info.bytes, this.memBytes);

    // Function table: every function export, in export order.
    this.funcs = [];
    const fidx = new Map();
    const pointerGetters = new Set(regionTable.map((r) => r.getter));
    for (const [name, x] of info.parsed.exports) {
      if (x.kind !== 0) continue;
      let flags = 0;
      if (pointerGetters.has(name)) flags |= FFLAG.POINTER_RESULT;
      if (isPureExport(name)) flags |= FFLAG.PURE;
      fidx.set(name, this.funcs.length);
      this.funcs.push({ name, params: x.params, results: x.results, flags, argBytes: x.params.reduce((a, t) => a + VT_SIZE[t], 0) });
    }
    this.regions = regionTable.map((r, index) => {
      if (!fidx.has(r.getter)) throw new Error(`[chronos-trace] region getter ${r.getter} is not exported by the wasm`);
      return { ...r, index, getterIndex: fidx.get(r.getter), fn: E[r.getter], addr: -1 };
    });
    this.digested = this.regions.filter((r) => r.flags & RFLAG.DIGESTED);
    this.writeOrder = [
      ...this.regions.filter((r) => r.name === 'ingest_rgba'),
      ...this.digested,
      ...this.regions.filter((r) => (r.flags & RFLAG.ALIAS) && r.name !== 'ingest_rgba'),
    ];
    this.resolveRegions();
    this.validateRegions();
    this.buildChunkIndex();

    const meta = {
      label: cfg.label,
      script: process.argv[1] || null,
      argv: process.argv.slice(2),
      pid: process.pid,
      threadId,
      instance: seq,
      node: process.version,
      created: new Date().toISOString(),
      strict: cfg.strict,
      stackTop: this.stackTop,
      budgetScale: process.env.CHRONOS_CI_BUDGET_SCALE || null,
      // Platform-descriptive words zeroed before hashing (format 1.1).
      digestMask: digestMask(),
    };
    const header = encodeHeader({
      wasmSha256: info.sha256,
      memBytes: this.memBytes,
      flags: (cfg.strict ? HFLAG.STRICT : 0) | (cfg.memDigestEvery ? HFLAG.MEMDIGEST : 0),
      meta,
      funcs: this.funcs,
      regions: this.regions,
    });
    const tid = threadId ? `.t${threadId}` : '';
    const file = path.join(cfg.dir, `${cfg.label}.p${process.pid}${tid}.i${String(seq).padStart(3, '0')}.cstrace`);
    // The writer must not reference the recorder: it is the FinalizationRegistry
    // held value, and a back-reference would keep the instance alive.
    this.writer = new TraceWriter(file, header, cfg, proc, meta);

    this.memU8 = new Uint8Array(this.mem.buffer);
    this.memBuf = Buffer.from(this.mem.buffer, 0, this.memBytes);
    this.memI32 = new Int32Array(this.mem.buffer);
    this.snap = Buffer.allocUnsafe(this.memBytes);
    this.memBuf.copy(this.snap);
    // Fresh-instance memory image, shared by all instances of the module.
    if (!info.pristine) {
      const buf = Buffer.from(this.snap);
      info.pristine = { buf, i32: new Int32Array(buf.buffer, buf.byteOffset, this.memBytes >>> 2) };
    }
    this.pristine = info.pristine;
    this.snapI32 = new Int32Array(this.snap.buffer, this.snap.byteOffset, this.memBytes >>> 2);
    const mask = meta.digestMask;
    this.maskWords = this.digested.map((r) => mask[r.name] || null);
    this.maskScratch = new Uint8Array(Math.max(0, ...this.digested.filter((r, i) => this.maskWords[i]).map((r) => r.byteLen)));
    this.hashes = new Uint32Array(this.digested.length);
    this.dirty = new Uint8Array(this.digested.length).fill(1);
    this.step = new StepBuf();
    this.callSeq = 0;
    this.active = true;

    // Wrapped exports: same keys, same order, same Memory object.
    const wrapped = Object.create(null);
    for (const key of Object.keys(E)) {
      const v = E[key];
      if (typeof v === 'function' && fidx.has(key)) wrapped[key] = this.makeWrapper(fidx.get(key), v);
      else wrapped[key] = v;
    }
    this.wrappedExports = Object.freeze(wrapped);
    proc.finalizer.register(this.wrappedExports, this.writer);
  }

  release() {
    this.active = false;
    this.snap = this.snapI32 = null;
    this.step = null;
  }

  makeWrapper(fi, fn) {
    const rec = this;
    const w = function (...args) {
      return rec.active ? rec.invoke(fi, fn, args) : fn(...args);
    };
    Object.defineProperty(w, 'length', { value: fn.length });
    Object.defineProperty(w, 'name', { value: fn.name });
    return w;
  }

  ensureViews() {
    // Memory is fixed-size by design; a grow would detach the old buffer.
    if (this.memU8.buffer !== this.mem.buffer || this.mem.buffer.byteLength !== this.memBytes) {
      throw new Error('[chronos-trace] kernel memory changed size; recording assumes a fixed-size memory');
    }
  }

  resolveRegions() {
    for (const r of this.regions) r.addr = (r.hasArg ? r.fn(r.arg) : r.fn()) >>> 0;
  }

  validateRegions() {
    for (const r of this.regions) {
      const es = r.elem === 1 ? 1 : 4;
      if (r.addr % es !== 0 || r.addr < this.stackTop || r.addr + r.byteLen > this.memBytes) {
        throw new Error(`[chronos-trace] region ${r.name} [${r.addr}, +${r.byteLen}) is outside memory (${this.memBytes} B) or misaligned`);
      }
    }
  }

  buildChunkIndex() {
    this.digestAddrs = this.digested.map((r) => r.addr);
    const nChunks = Math.ceil(this.memBytes / CHUNK);
    this.chunkRegions = new Array(nChunks);
    this.digested.forEach((r, i) => {
      for (let c = Math.floor(r.addr / CHUNK); c * CHUNK < r.addr + r.byteLen; c++) (this.chunkRegions[c] ||= []).push(i);
    });
  }

  markDirtyRange(a, b) {
    for (let c = Math.floor(a / CHUNK); c * CHUNK < b; c++) {
      const list = this.chunkRegions[c];
      if (list) for (const i of list) this.dirty[i] = 1;
    }
  }

  /** Exact byte ranges in [lo, hi) where memory differs from `snap` (default: the snapshot). */
  diffRanges(lo, hi, snap = this.snap, si = this.snapI32) {
    const out = [];
    const mem = this.memBuf;
    const mi = this.memI32;
    for (let blk = lo; blk < hi; blk += BLOCK) {
      const be = Math.min(hi, blk + BLOCK);
      if (mem.compare(snap, blk, be, blk, be) === 0) continue;
      for (let c = blk; c < be; c += CHUNK) {
        const ce = Math.min(be, c + CHUNK);
        if (mem.compare(snap, c, ce, c, ce) === 0) continue;
        const we = ce >>> 2;
        for (let w = c >>> 2; w < we; w++) {
          if (mi[w] === si[w]) continue;
          let e = w + 1;
          let gap = 0;
          for (let k = e; k < we && gap <= MERGE_GAP_WORDS; k++) {
            if (mi[k] !== si[k]) { e = k + 1; gap = 0; } else gap++;
          }
          let a = w * 4;
          let b = e * 4;
          while (mem[a] === snap[a]) a++;
          while (mem[b - 1] === snap[b - 1]) b--;
          const last = out[out.length - 1];
          if (last && a - last[1] <= MERGE_GAP_WORDS * 4) last[1] = b;
          else out.push([a, b]);
          w = e - 1;
        }
      }
    }
    return out;
  }

  /** Copies changed chunks of [lo, hi) into the snapshot; marks digests dirty. */
  syncSnapshot(lo, hi) {
    const mem = this.memBuf;
    const snap = this.snap;
    for (let blk = lo; blk < hi; blk += BLOCK) {
      const be = Math.min(hi, blk + BLOCK);
      if (mem.compare(snap, blk, be, blk, be) === 0) continue;
      for (let c = blk; c < be; c += CHUNK) {
        const ce = Math.min(be, c + CHUNK);
        if (mem.compare(snap, c, ce, c, ce) === 0) continue;
        mem.copy(snap, c, c, ce);
        const list = this.chunkRegions[c / CHUNK];
        if (list) for (const i of list) this.dirty[i] = 1;
      }
    }
  }

  /** Splits byte ranges into region pieces; region null = outside every region. */
  mapRanges(ranges) {
    const pieces = [];
    for (const [ra, rb] of ranges) {
      let a = ra;
      while (a < rb) {
        let region = null;
        for (const r of this.writeOrder) {
          if (a >= r.addr && a < r.addr + r.byteLen) { region = r; break; }
        }
        let b;
        if (region) b = Math.min(rb, region.addr + region.byteLen);
        else {
          b = rb;
          for (const r of this.regions) if (r.addr > a && r.addr < b) b = r.addr;
        }
        pieces.push({ region, a, b });
        a = b;
      }
    }
    return pieces;
  }

  /** Records host writes made since the last snapshot into the step buffer. */
  captureHostWrites(stats) {
    const ranges = this.diffRanges(0, this.memBytes);
    if (!ranges.length) return;
    this.resolveRegions();
    for (let i = 0; i < this.digested.length; i++) {
      if (this.digested[i].addr !== this.digestAddrs[i]) {
        this.buildChunkIndex();
        this.dirty.fill(1);
        break;
      }
    }
    let pieces = this.mapRanges(ranges);
    const st = this.step;
    if (pieces.some((p) => !p.region)) {
      // A host that restores the fresh-instance memory image (instance pooling)
      // is recorded as RESET plus region writes relative to that image.
      const pr = this.pristine;
      const fromPristine = this.mapRanges(this.diffRanges(0, this.memBytes, pr.buf, pr.i32));
      if (fromPristine.every((p) => p.region)) {
        st.u8(TAG.RESET);
        stats.resets++;
        pieces = fromPristine;
      }
    }
    for (const { region, a, b } of pieces) {
      if (region) {
        st.u8(TAG.WRITE);
        st.u16(region.index);
        st.u32(a - region.addr);
        st.u32(b - a);
        st.copyFrom(this.memBuf, a, b);
        stats.writes++;
        stats.writeBytes += b - a;
      } else {
        st.u8(TAG.UNMAPPED);
        st.u32(a);
        st.u32(b - a);
        st.copyFrom(this.memBuf, a, b);
        stats.unmapped++;
        this.proc.unmappedTotal = (this.proc.unmappedTotal || 0) + 1;
        if (this.proc.unmappedTotal <= 20) {
          let below = null;
          for (const r of this.regions) if (r.addr + r.byteLen <= a && (!below || r.addr > below.addr)) below = r;
          const near = below ? `, ${a - below.addr - below.byteLen} B past the end of ${below.name}` : '';
          console.error(
            `[chronos-trace] WARNING: host write OUTSIDE every known region: addr=${a} len=${b - a}${near}, ` +
              `before call #${this.callSeq} (${path.basename(this.writer.file)})`
          );
        }
      }
    }
    for (const [ra, rb] of ranges) {
      this.memBuf.copy(this.snap, ra, ra, rb);
      this.markDirtyRange(ra, rb);
    }
    // The resolution getters above may have touched the shadow stack.
    this.syncSnapshot(0, this.stackTop);
  }

  invoke(fi, fn, args) {
    this.ensureViews();
    const f = this.funcs[fi];
    const params = f.params;
    const conv = new Array(params.length);
    for (let k = 0; k < params.length; k++) {
      const t = params[k];
      const v = args[k];
      // Exactly the ToWebAssemblyValue conversions; the converted primitive is
      // what the export receives (re-converting it is the identity).
      conv[k] = t === 0x7f ? v | 0 : t === 0x7d ? Math.fround(v) : t === 0x7c ? +v : BigInt.asIntN(64, v);
    }
    const impure = this.cfg.strict || !(f.flags & FFLAG.PURE);
    const st = this.step;
    st.reset();
    const stats = { writes: 0, writeBytes: 0, unmapped: 0, digested: 0, resets: 0 };
    // Host writes are captured before every call, getters included: a getter
    // can read state the host just changed (e.g. a poked ring cursor).
    this.captureHostWrites(stats);

    let status = 0;
    let result;
    let error;
    try {
      result = fn(...conv);
    } catch (e) {
      status = 1;
      error = e;
    }

    st.u8(TAG.CALL);
    st.u32(this.callSeq++);
    st.u16(fi);
    st.need(f.argBytes);
    let o = st.n;
    for (let k = 0; k < params.length; k++) o = encodeValue(st.buf, o, params[k], conv[k]);
    st.n = o;
    st.u8(status);
    if (status === 0 && f.results.length) {
      st.need(8);
      st.n = encodeValue(st.buf, st.n, f.results[0], result);
    }
    if (impure) {
      this.syncSnapshot(0, this.memBytes);
      for (let i = 0; i < this.digested.length; i++) {
        if (this.dirty[i]) {
          const r = this.digested[i];
          this.hashes[i] = xxh32Masked(this.memU8, r.addr, r.byteLen, this.maskWords[i], this.maskScratch);
          this.dirty[i] = 0;
        }
      }
      const withMem = this.cfg.memDigestEvery && this.writer.digested % this.cfg.memDigestEvery === 0;
      st.u8(withMem ? 3 : 1);
      st.need(4 * this.hashes.length);
      for (let i = 0; i < this.hashes.length; i++) st.u32(this.hashes[i]);
      if (withMem) st.u32(xxh32(this.memU8, this.stackTop, this.memBytes - this.stackTop));
      stats.digested = 1;
    } else {
      this.syncSnapshot(0, this.stackTop);
      st.u8(0);
    }
    if (!this.writer.commit(st.out(), stats)) this.release();
    if (status) throw error;
    return result;
  }
}

/** Initial value of the mutable i32 global 0 (__stack_pointer) when the stack sits below it. */
function stackTopOf(bytes, memBytes) {
  const b = bytes;
  let p = 8;
  const leb = () => {
    let r = 0;
    let s = 0;
    for (;;) {
      const x = b[p++];
      r += (x & 0x7f) * 2 ** s;
      s += 7;
      if ((x & 0x80) === 0) return r;
    }
  };
  while (p < b.length) {
    const id = b[p++];
    const size = leb();
    const end = p + size;
    if (id === 6 && leb() > 0 && b[p] === 0x7f && b[p + 1] === 1 && b[p + 2] === 0x41) {
      p += 3;
      const v = leb();
      if (v > 0 && v < memBytes) return v;
    }
    p = end;
  }
  return 0;
}
