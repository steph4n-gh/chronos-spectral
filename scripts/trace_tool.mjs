#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // CSTRACE inspection and negative-control mutation tool.
//
//   node scripts/trace_tool.mjs info <trace>
//       header, function table, region table, record counts
//   node scripts/trace_tool.mjs mutate-write <in> <out> [--nth=K]
//       copy <in> with one payload byte of the K-th host WRITE flipped (default: middle write)
//   node scripts/trace_tool.mjs mutate-arg <in> <out> [--nth=K] [--func=NAME] [--arg=I]
//       copy <in> with argument I (default 0) of the K-th call (default: middle one)
//       that has arguments (optionally only calls to NAME) changed
//
// Mutations print the call index and region they touched so a replay failure
// can be checked against them.
// ============================================================================

import fs from 'node:fs';
import { ELEM_NAME, FFLAG, RFLAG, TraceReader, VT, decodeValues } from './trace_lib.mjs';

const [cmd, ...rest] = process.argv.slice(2);
const pos = rest.filter((a) => !a.startsWith('--'));
const nthArg = rest.find((a) => a.startsWith('--nth='));
const nth = nthArg ? Number(nthArg.slice(6)) : null;
const funcArg = rest.find((a) => a.startsWith('--func='))?.slice(7);
const argIdx = Number(rest.find((a) => a.startsWith('--arg='))?.slice(6) ?? 0);

function scan(file) {
  const rd = new TraceReader(file);
  const recs = [];
  let calls = 0;
  for (const r of rd.records()) {
    if (r.tag === 'C') calls++;
    recs.push({ tag: r.tag, pos: r.pos, seq: r.seq, func: r.func, region: r.region, len: r.data?.length, nArgs: r.args?.length, digests: !!r.digests, callsSoFar: calls, offset: r.offset, reason: r.reason, clean: r.clean });
  }
  rd.close();
  return { header: rd.header, recs };
}

if (cmd === 'info') {
  const { header: h, recs } = scan(pos[0]);
  console.log(`CSTRACE v${h.major}.${h.minor}  header ${h.headerLen} B  wasm sha256 ${h.wasmSha256}  memory ${h.memBytes} B  flags ${h.flags}`);
  console.log(`meta ${JSON.stringify(h.meta)}`);
  console.log(`functions (${h.funcs.length}):`);
  h.funcs.forEach((f, i) => {
    const fl = [f.flags & FFLAG.POINTER_RESULT ? 'ptr' : '', f.flags & FFLAG.PURE ? 'pure' : ''].filter(Boolean).join(',');
    console.log(`  ${String(i).padStart(3)} ${f.name}(${f.params.map((t) => VT[t]).join(', ')}) -> ${f.results.map((t) => VT[t]).join(', ') || 'void'} ${fl}`);
  });
  console.log(`regions (${h.regions.length}, ${h.digested.length} digested):`);
  h.regions.forEach((r, i) => {
    const fl = [r.flags & RFLAG.HOST_WRITABLE ? 'mut' : 'const', r.flags & RFLAG.DIGESTED ? 'digested' : '', r.flags & RFLAG.ALIAS ? 'alias' : '', r.flags & RFLAG.INPUT_ONLY ? 'input' : ''].filter(Boolean).join(',');
    console.log(`  ${String(i).padStart(3)} ${r.name.padEnd(24)} ${h.funcs[r.getterIndex].name}(${r.hasArg ? r.arg : ''}) ${ELEM_NAME[r.elem]} ${r.byteLen} B ${fl}`);
  });
  const count = (t) => recs.filter((r) => r.tag === t).length;
  const last = recs[recs.length - 1];
  const end = recs.find((r) => r.tag === 'E');
  console.log(`records: ${count('C')} calls (${recs.filter((r) => r.digests).length} with digests), ${count('W')} writes, ${count('R')} resets, ${count('U')} unmapped, end=${end ? ['exit', 'instance-cap', 'total-cap', 'instance-collected'][end.reason] : 'none'}, eof clean=${last.clean}`);
} else if (cmd === 'mutate-write' || cmd === 'mutate-arg') {
  const [inFile, outFile] = pos;
  const { header: h, recs } = scan(inFile);
  const buf = fs.readFileSync(inFile);
  if (cmd === 'mutate-write') {
    const writes = recs.filter((r) => r.tag === 'W');
    const w = writes[nth ?? Math.floor(writes.length / 2)];
    const at = w.pos + 1 + 2 + 4 + 4 + Math.floor(w.len / 2);
    buf[at] ^= 0x5a;
    fs.writeFileSync(outFile, buf);
    console.log(`flipped byte ${Math.floor(w.len / 2)} of a ${w.len}-byte host write to region ${h.regions[w.region].name} (offset ${w.offset}), applied before call #${w.callsSoFar}`);
  } else {
    const withArgs = recs.filter((r) => r.tag === 'C' && r.nArgs > 0 && (!funcArg || h.funcs[r.func].name === funcArg));
    if (!withArgs.length) throw new Error(`no matching calls with arguments${funcArg ? ` to ${funcArg}` : ''}`);
    const c = withArgs[nth ?? Math.floor(withArgs.length / 2)];
    const f = h.funcs[c.func];
    if (argIdx >= f.params.length) throw new Error(`${f.name} has ${f.params.length} argument(s)`);
    const t = f.params[argIdx];
    const at = c.pos + 1 + 4 + 2 + f.params.slice(0, argIdx).reduce((a, p) => a + (p === 0x7e || p === 0x7c ? 8 : 4), 0);
    const before = decodeValues(buf.subarray(at, at + 8), [t])[0];
    if (t === 0x7f) buf.writeInt32LE(before ^ 1, at);
    else if (t === 0x7d) buf.writeFloatLE(before === 0 ? 1 : before * 1.5, at);
    else if (t === 0x7c) buf.writeDoubleLE(before === 0 ? 1 : before * 1.5, at);
    const after = decodeValues(buf.subarray(at, at + 8), [t])[0];
    fs.writeFileSync(outFile, buf);
    console.log(`changed argument ${argIdx} of call #${c.seq} ${f.name} from ${before} to ${after}`);
  }
} else {
  console.error('usage: trace_tool.mjs info <trace> | mutate-write <in> <out> [--nth=K] | mutate-arg <in> <out> [--nth=K]');
  process.exit(2);
}
