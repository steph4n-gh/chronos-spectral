#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"

free_stale_chronos_port() {
  local target_port="$1"
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -tiTCP:"${target_port}" -sTCP:LISTEN 2>/dev/null || true)"
    for pid in $pids; do
      local proc_cwd
      proc_cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)"
      if [ "$proc_cwd" = "$ROOT_DIR" ]; then
        echo "  [INFO] Stopping older CHRONOS server on port ${target_port} (PID ${pid})..."
        kill "$pid" 2>/dev/null || true
        sleep 0.2
      fi
    done
  fi
}

if [ "${1:-}" = "--serve" ]; then
  if [ ! -f "web/chronos_spectral.wasm" ]; then
    cargo build --release
    mkdir -p web
    cp "target/wasm32-unknown-unknown/release/chronos_spectral.wasm" "web/chronos_spectral.wasm.tmp.$$"
    mv -f "web/chronos_spectral.wasm.tmp.$$" "web/chronos_spectral.wasm"
  fi
  PORT="${PORT:-8080}"
  free_stale_chronos_port "$PORT"
  echo "Launching CHRONOS-SPECTRAL Command Deck at http://127.0.0.1:${PORT}"
  echo "  (RFC 7233 HTTP 206 byte-range + Live FFmpeg fMP4 MSE Relay server)"
  echo "  Press Ctrl+C to stop server."
  exec env PORT="$PORT" node scripts/dev_server.mjs
fi

echo "========================================================================"
echo " CHRONOS-SPECTRAL // Zero-Allocation Wasm32 SIMD128 Build & Audit"
echo "========================================================================"

echo "[1/5] Compiling #![no_std] Rust kernel (wasm32-unknown-unknown +simd128)..."
cargo build --release

WASM_SRC="target/wasm32-unknown-unknown/release/chronos_spectral.wasm"
WASM_DST="web/chronos_spectral.wasm"
mkdir -p web
cp "$WASM_SRC" "${WASM_DST}.tmp.$$"
mv -f "${WASM_DST}.tmp.$$" "$WASM_DST"
ls -lh "$WASM_DST"

if command -v sha256sum >/dev/null 2>&1; then
  WASM_SHA256="$(sha256sum "$WASM_DST" | cut -d' ' -f1)"
else
  WASM_SHA256="$(shasum -a 256 "$WASM_DST" | cut -d' ' -f1)"
fi
echo "  [INFO] $(rustc --version) -> sha256 ${WASM_SHA256}"

# The committed web/chronos_spectral.wasm is the binary GitHub Pages deploys. CI sets
# CHRONOS_VERIFY_COMMITTED_WASM=1 so every gate below runs against exactly that binary.
if [ "${CHRONOS_VERIFY_COMMITTED_WASM:-0}" = "1" ]; then
  if ! git diff --quiet HEAD -- "$WASM_DST"; then
    echo "FAIL: rebuilt ${WASM_DST} (sha256 ${WASM_SHA256}) differs from the committed binary." >&2
    echo "      Rebuild with the toolchain pinned in rust-toolchain.toml and commit the result." >&2
    exit 1
  fi
  echo "  [PASS] Rebuilt Wasm is byte-identical to the committed ${WASM_DST}"
fi

# Source hygiene claims published in README.md and docs/: keep the docs and these counts in step.
CONTINUUM_UNSAFE_EXPECTED=6
CONTINUUM_UNSAFE="$(grep -ow 'unsafe' src/continuum_physics.rs | wc -l | tr -d ' ')"
if [ "$CONTINUUM_UNSAFE" != "$CONTINUUM_UNSAFE_EXPECTED" ]; then
  echo "FAIL: src/continuum_physics.rs has ${CONTINUUM_UNSAFE} unsafe tokens, docs state ${CONTINUUM_UNSAFE_EXPECTED}." >&2
  exit 1
fi
if grep -q 'MaybeUninit' src/*.rs; then
  echo "FAIL: MaybeUninit found in src/*.rs, docs state 0." >&2
  exit 1
fi
echo "  [PASS] Source hygiene: ${CONTINUUM_UNSAFE} unsafe tokens in continuum_physics.rs, 0 MaybeUninit across src/*.rs"

echo ""
echo "[2/5] Auditing Wasm binary layout & SIMD128 / CLZ opcodes via wasm-objdump..."
if ! command -v wasm-objdump >/dev/null 2>&1; then
  echo "FAIL: wasm-objdump is required in PATH (fail-closed binary audit gate)." >&2
  exit 1
fi

HEADERS="$(wasm-objdump -x "$WASM_DST")"
DISASM="$(wasm-objdump -d "$WASM_DST")"

if echo "$HEADERS" | grep -q "^Import\["; then
  echo "FAIL: Unexpected imports detected in Wasm module!" >&2
  echo "$HEADERS" | grep -A 10 "^Import\[" >&2
  exit 1
fi
echo "  [PASS] Zero imports verified (0 env / malloc / free allocator imports)"

MEM_LINE="$(echo "$HEADERS" | grep "memory\[0\]" || true)"
if ! echo "$MEM_LINE" | grep -q "initial=448 max=448"; then
  echo "FAIL: Expected fixed Wasm linear memory initial=448 max=448, got: ${MEM_LINE}" >&2
  exit 1
fi
echo "  [PASS] Contiguous Linear Memory Arena:${MEM_LINE}"

AUDIT_REPORT="$(python3 -c '
import re, sys
hdrs = open(sys.argv[1]).read()
disasm = sys.stdin.read()

if re.search(r"\bmemory\.grow\b", disasm):
    print("FAIL memory.grow opcode detected in disassembly")
    sys.exit(0)

alloc_re = re.compile(r"(__rust_alloc|__rust_dealloc|__rust_realloc|__rust_alloc_zeroed|__rdl_|__rg_|dlmalloc|wee_alloc)")
for ln in hdrs.splitlines():
    if alloc_re.search(ln):
        print(f"FAIL forbidden allocator symbol in headers: {ln.strip()}")
        sys.exit(0)

max_data_end = 0
for m in re.finditer(r"segment\[\d+\][^\n]*?size=(\d+)[^\n]*?(?:init\s+i32=|i32\.const\s+)(\d+)", hdrs):
    end = int(m.group(2)) + int(m.group(1))
    if end > max_data_end:
        max_data_end = end
if max_data_end <= 0 or max_data_end > 29294592:
    print(f"FAIL static data segment end={max_data_end} outside (0, 29294592]")
    sys.exit(0)
if max_data_end > 24248320:
    print(f"FAIL Wave-2 static data segment end={max_data_end} exceeds 24248320")
    sys.exit(0)

fn_re = re.compile(r"^[0-9a-f]+\s+func\[(\d+)\]\s+<([^>]+)>:")
call_re = re.compile(r"\|\s*call\s+(\d+)\b")
ind_re = re.compile(r"\|\s*call_indirect\b")
frames, names, callees = {}, {}, {}
has_indirect, cur = False, None
lines = disasm.splitlines()
for i, ln in enumerate(lines):
    m = fn_re.match(ln)
    if m:
        cur = int(m.group(1))
        names[cur] = m.group(2)
        frames[cur] = 0
        callees[cur] = set()
        if alloc_re.search(names[cur]):
            print(f"FAIL forbidden allocator function in disasm: {names[cur]}")
            sys.exit(0)
        continue
    if cur is None:
        continue
    if "global.get 0" in ln and i + 2 < len(lines) and "i32.sub" in lines[i + 2]:
        cm = re.search(r"i32\.const\s+(\d+)", lines[i + 1])
        if cm:
            frames[cur] = max(frames[cur], int(cm.group(1)))
    cm2 = call_re.search(ln)
    if cm2:
        callee = int(cm2.group(1))
        callees[cur].add(callee)
    if ind_re.search(ln):
        has_indirect = True

if not frames:
    print("FAIL incomplete wasm-objdump disassembly (0 analyzed functions)")
    sys.exit(0)

for u, cs in callees.items():
    for v in cs:
        if alloc_re.search(names.get(v, "")):
            print(f"FAIL forbidden allocator call target func[{v}] <{names.get(v)}> from func[{u}]")
            sys.exit(0)

memo, vis, has_cycle = {}, set(), False
def dfs(u):
    global has_cycle
    if u in memo:
        return memo[u]
    if u in vis:
        has_cycle = True
        return frames.get(u, 0)
    vis.add(u)
    sub = max((dfs(v) for v in callees.get(u, ())), default=0)
    vis.remove(u)
    memo[u] = frames.get(u, 0) + sub
    return memo[u]
mx = max((dfs(u) for u in frames), default=0)
print(f"OK {mx} {int(has_indirect)} {int(has_cycle)} {max_data_end} {len(frames)}")
' <(printf '%s' "$HEADERS") <<< "$DISASM")"

if [ "${AUDIT_REPORT%% *}" != "OK" ]; then
  echo "FAIL: Wasm binary static audit failed: ${AUDIT_REPORT}" >&2
  exit 1
fi
STACK_MAX_PATH="$(echo "$AUDIT_REPORT" | awk '{print $2}')"
STACK_INDIRECT="$(echo "$AUDIT_REPORT" | awk '{print $3}')"
STACK_CYCLE="$(echo "$AUDIT_REPORT" | awk '{print $4}')"
RODATA_END="$(echo "$AUDIT_REPORT" | awk '{print $5}')"
ANALYZED_FUNCS="$(echo "$AUDIT_REPORT" | awk '{print $6}')"
if [ "$STACK_INDIRECT" -ne 0 ] || [ "$STACK_CYCLE" -ne 0 ] || [ "$STACK_MAX_PATH" -gt 14336 ]; then
  echo "FAIL: Wasm call-graph stack check failed (max_path=${STACK_MAX_PATH} B, indirect=${STACK_INDIRECT}, cycle=${STACK_CYCLE}, budget=14336/16384 B)!" >&2
  exit 1
fi
echo "  [PASS] Zero memory.grow / zero __rust_alloc symbols or calls (${ANALYZED_FUNCS} funcs, .rodata end=${RODATA_END} <= 24248320 <= 29294592 B)"
echo "  [PASS] Wasm call-graph max live shadow stack: ${STACK_MAX_PATH} / 16384 B (indirect=${STACK_INDIRECT}, cycles=${STACK_CYCLE})"

V128_COUNT="$(echo "$DISASM" | grep -cE "v128\.|f32x4\.|u32x4\.|i32x4\.")"
CLZ_COUNT="$(echo "$DISASM" | grep -c "i32\.clz")"
COPYSIGN_COUNT="$(echo "$DISASM" | grep -c "f32\.copysign")"

if [ "$V128_COUNT" -lt 20 ] || [ "$CLZ_COUNT" -lt 1 ] || [ "$COPYSIGN_COUNT" -lt 1 ]; then
  echo "FAIL: Missing expected hardware SIMD128 or i32.clz / f32.copysign instructions!" >&2
  exit 1
fi
echo "  [PASS] Emitted instructions: ${V128_COUNT} SIMD128 ops, ${CLZ_COUNT} i32.clz ops, ${COPYSIGN_COUNT} f32.copysign ops"

# --native-check: the build, the static audit, Gate S and Gate N only (the native-backend CI jobs,
# e.g. macOS arm64 for NEON). Steps 3, Q, W1, W2 and X exercise the Wasm alone and run in --check.
if [ "${1:-}" = "--native-check" ]; then
  echo ""
  echo "[3/5] --native-check: skipping the Wasm-only suites (step 3, Gates Q, W1, W2, X; run them with --check)"
else
echo ""
echo "[3/5] Running Mathematical Counterexample & Ablation Audit Suite..."
if command -v sysctl >/dev/null 2>&1; then
  CPU_BRAND="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
  CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
  echo "  Host: ${CPU_BRAND} (${CORES} cores) | Load: $(uptime | awk -F'load averages?: ' '{print $2}')"
fi

node --input-type=module - <<'EOF'
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';

const W = 320;
const H = 180;
const PIXELS = W * H;
const BYTES = PIXELS * 4;
const GRAPH_NODES = 32;
const MAX_TARGETS = 16;
const MAX_KRYLOV = 31;
const TELEMETRY_LEN = 1568;
const TELEM_PHYSICS_START = 1280;
const TELEM_ENSEMBLE_START = 1536;

const wasmBytes = fs.readFileSync('./web/chronos_spectral.wasm');
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const exp = instance.exports;
const mem = exp.memory;

if (mem.buffer.byteLength !== 448 * 65536) {
  throw new Error(`Expected 448 pages (29360128 B), got ${mem.buffer.byteLength}`);
}
let grewPastMax = false;
try { mem.grow(1); grewPastMax = true; } catch {}
if (grewPastMax) {
  throw new Error('Expected fixed max=448 Wasm memory (mem.grow(1) must fail)');
}

const telem = new Float32Array(mem.buffer, exp.chronos_get_telemetry_ptr(), TELEMETRY_LEN);
const laplacian = new Float32Array(mem.buffer, exp.chronos_get_laplacian_ptr(), 1024);
const fiedler = new Float32Array(mem.buffer, exp.chronos_get_fiedler_ptr(), 32);
const pipBuf = new Uint8Array(mem.buffer, exp.chronos_get_pip_rgba_ptr(), 16384);

// Exact production pre-allocated Shifted Lanczos + Invariant-Subspace Restart solver from web/worker.js
const activeNodeIdx = new Int32Array(GRAPH_NODES);
const lanczosQ = new Float64Array(MAX_KRYLOV * GRAPH_NODES);
const lanczosAlpha = new Float64Array(MAX_KRYLOV);
const lanczosBeta = new Float64Array(MAX_KRYLOV);
const lanczosW = new Float64Array(GRAPH_NODES);
const jacobiA = new Float64Array(MAX_KRYLOV * MAX_KRYLOV);
const jacobiV = new Float64Array(MAX_KRYLOV * MAX_KRYLOV);
const prevFiedlerTrack = new Float64Array(MAX_TARGETS);
const eigVecBuf = new Float64Array(GRAPH_NODES);

function findOrthogonalBasisSeed(n, numExisting, outOffset) {
  for (let cand = 0; cand < n; cand++) {
    const invN = 1.0 / n;
    for (let a = 0; a < n; a++) {
      lanczosW[a] = (a === cand ? 1.0 : 0.0) - invN;
    }
    for (let pass = 0; pass < 2; pass++) {
      for (let k = 0; k < numExisting; k++) {
        const kOff = k * GRAPH_NODES;
        let dot = 0.0;
        for (let a = 0; a < n; a++) dot += lanczosQ[kOff + a] * lanczosW[a];
        for (let a = 0; a < n; a++) lanczosW[a] -= dot * lanczosQ[kOff + a];
      }
      let mean = 0.0;
      for (let a = 0; a < n; a++) mean += lanczosW[a];
      mean /= n;
      for (let a = 0; a < n; a++) lanczosW[a] -= mean;
    }
    let norm2 = 0.0;
    for (let a = 0; a < n; a++) norm2 += lanczosW[a] * lanczosW[a];
    if (norm2 > 1e-12) {
      const invNorm = 1.0 / Math.sqrt(norm2);
      for (let a = 0; a < n; a++) lanczosQ[outOffset + a] = lanczosW[a] * invNorm;
      return true;
    }
  }
  return false;
}

function solveShiftedLanczosFiedler() {
  let n = 0;
  for (let i = 0; i < GRAPH_NODES; i++) {
    if (laplacian[i * GRAPH_NODES + i] > 1e-6) {
      activeNodeIdx[n++] = i;
    }
  }
  fiedler.fill(0.0);
  if (n < 2) {
    exp.chronos_set_fiedler_spectrum(0.0, 0.0);
    return 0.0;
  }

  let mu = 0.0;
  for (let a = 0; a < n; a++) {
    const ia = activeNodeIdx[a];
    let rowSum = laplacian[ia * GRAPH_NODES + ia];
    for (let b = 0; b < n; b++) {
      if (a !== b) rowSum += Math.abs(laplacian[ia * GRAPH_NODES + activeNodeIdx[b]]);
    }
    if (rowSum > mu) mu = rowSum;
  }
  mu += 0.1;

  const mSteps = Math.min(n - 1, MAX_KRYLOV);
  let mean0 = 0.0;
  for (let a = 0; a < n; a++) {
    const val = (a + 1) - 0.5 * (n + 1);
    lanczosQ[a] = val;
    mean0 += val;
  }
  mean0 /= n;
  let norm0 = 0.0;
  for (let a = 0; a < n; a++) {
    lanczosQ[a] -= mean0;
    norm0 += lanczosQ[a] * lanczosQ[a];
  }
  const invNorm0 = 1.0 / Math.sqrt(Math.max(norm0, 1e-18));
  for (let a = 0; a < n; a++) lanczosQ[a] *= invNorm0;

  let actualM = 0;
  for (let j = 0; j < mSteps; j++) {
    const qOff = j * GRAPH_NODES;
    for (let a = 0; a < n; a++) {
      const ia = activeNodeIdx[a];
      let sumL = 0.0;
      for (let b = 0; b < n; b++) {
        sumL += laplacian[ia * GRAPH_NODES + activeNodeIdx[b]] * lanczosQ[qOff + b];
      }
      lanczosW[a] = mu * lanczosQ[qOff + a] - sumL;
    }
    let wMean = 0.0;
    for (let a = 0; a < n; a++) wMean += lanczosW[a];
    wMean /= n;
    for (let a = 0; a < n; a++) lanczosW[a] -= wMean;

    if (j > 0 && lanczosBeta[j - 1] > 0.0) {
      const prevOff = (j - 1) * GRAPH_NODES;
      const betaPrev = lanczosBeta[j - 1];
      for (let a = 0; a < n; a++) lanczosW[a] -= betaPrev * lanczosQ[prevOff + a];
    }

    let alpha = 0.0;
    for (let a = 0; a < n; a++) alpha += lanczosQ[qOff + a] * lanczosW[a];
    lanczosAlpha[j] = alpha;
    for (let a = 0; a < n; a++) lanczosW[a] -= alpha * lanczosQ[qOff + a];

    for (let pass = 0; pass < 2; pass++) {
      for (let k = 0; k <= j; k++) {
        const kOff = k * GRAPH_NODES;
        let dot = 0.0;
        for (let a = 0; a < n; a++) dot += lanczosQ[kOff + a] * lanczosW[a];
        for (let a = 0; a < n; a++) lanczosW[a] -= dot * lanczosQ[kOff + a];
      }
      let wm = 0.0;
      for (let a = 0; a < n; a++) wm += lanczosW[a];
      wm /= n;
      for (let a = 0; a < n; a++) lanczosW[a] -= wm;
    }

    let beta2 = 0.0;
    for (let a = 0; a < n; a++) beta2 += lanczosW[a] * lanczosW[a];
    const beta = Math.sqrt(Math.max(0.0, beta2));
    actualM = j + 1;

    if (j + 1 >= mSteps) {
      lanczosBeta[j] = 0.0;
      break;
    }
    const nextOff = (j + 1) * GRAPH_NODES;
    if (beta < 1e-8) {
      lanczosBeta[j] = 0.0;
      if (!findOrthogonalBasisSeed(n, j + 1, nextOff)) break;
    } else {
      lanczosBeta[j] = beta;
      const invBeta = 1.0 / beta;
      for (let a = 0; a < n; a++) lanczosQ[nextOff + a] = lanczosW[a] * invBeta;
    }
  }

  jacobiA.fill(0.0);
  jacobiV.fill(0.0);
  for (let r = 0; r < actualM; r++) {
    jacobiV[r * actualM + r] = 1.0;
    jacobiA[r * actualM + r] = lanczosAlpha[r];
    if (r + 1 < actualM) {
      const b = lanczosBeta[r];
      jacobiA[r * actualM + (r + 1)] = b;
      jacobiA[(r + 1) * actualM + r] = b;
    }
  }

  for (let sweep = 0; sweep < 20; sweep++) {
    let maxOff = 0.0;
    for (let p = 0; p < actualM; p++) {
      for (let q = p + 1; q < actualM; q++) {
        const apq = jacobiA[p * actualM + q];
        const absApq = Math.abs(apq);
        if (absApq > maxOff) maxOff = absApq;
        if (absApq < 1e-12) continue;

        const app = jacobiA[p * actualM + p];
        const aqq = jacobiA[q * actualM + q];
        const tau = (aqq - app) / (2.0 * apq);
        const t = tau >= 0.0
          ? 1.0 / (tau + Math.sqrt(1.0 + tau * tau))
          : -1.0 / (-tau + Math.sqrt(1.0 + tau * tau));
        const c = 1.0 / Math.sqrt(1.0 + t * t);
        const s = t * c;

        jacobiA[p * actualM + p] = app - t * apq;
        jacobiA[q * actualM + q] = aqq + t * apq;
        jacobiA[p * actualM + q] = 0.0;
        jacobiA[q * actualM + p] = 0.0;

        for (let k = 0; k < actualM; k++) {
          if (k !== p && k !== q) {
            const akp = jacobiA[k * actualM + p];
            const akq = jacobiA[k * actualM + q];
            const nkp = c * akp - s * akq;
            const nkq = s * akp + c * akq;
            jacobiA[k * actualM + p] = nkp;
            jacobiA[p * actualM + k] = nkp;
            jacobiA[k * actualM + q] = nkq;
            jacobiA[q * actualM + k] = nkq;
          }
          const vkp = jacobiV[k * actualM + p];
          const vkq = jacobiV[k * actualM + q];
          jacobiV[k * actualM + p] = c * vkp - s * vkq;
          jacobiV[k * actualM + q] = s * vkp + c * vkq;
        }
      }
    }
    if (maxOff < 1e-11) break;
  }

  let bestCol = 0;
  let thetaMax = -1e30;
  let thetaSecond = -1e30;
  for (let c = 0; c < actualM; c++) {
    const val = jacobiA[c * actualM + c];
    if (val > thetaMax) {
      thetaSecond = thetaMax;
      thetaMax = val;
      bestCol = c;
    } else if (val > thetaSecond) {
      thetaSecond = val;
    }
  }

  const lambda2 = Math.max(0.0, mu - thetaMax);
  const lambda3 = actualM >= 2 ? Math.max(lambda2, mu - thetaSecond) : lambda2;
  const spectralGap = Math.max(0.0, lambda3 - lambda2);

  let dotPrev = 0.0;
  for (let a = 0; a < n; a++) {
    let comp = 0.0;
    for (let j = 0; j < actualM; j++) {
      comp += lanczosQ[j * GRAPH_NODES + a] * jacobiV[j * actualM + bestCol];
    }
    eigVecBuf[a] = comp;
    const nodeIdx = activeNodeIdx[a];
    if (nodeIdx < MAX_TARGETS) dotPrev += comp * prevFiedlerTrack[nodeIdx];
  }

  const signFix = (dotPrev < -1e-4 || (Math.abs(dotPrev) <= 1e-4 && eigVecBuf[0] < 0.0)) ? -1.0 : 1.0;
  for (let a = 0; a < n; a++) {
    const nodeIdx = activeNodeIdx[a];
    const val = eigVecBuf[a] * signFix;
    fiedler[nodeIdx] = val;
    if (nodeIdx < MAX_TARGETS) prevFiedlerTrack[nodeIdx] = val;
  }

  exp.chronos_set_fiedler_spectrum(lambda2, spectralGap);
  return { lambda2, spectralGap, n };
}

function eigenResidual64(n, lambda2) {
  let res2 = 0.0;
  let sumV = 0.0;
  for (let a = 0; a < n; a++) {
    const ia = activeNodeIdx[a];
    let lv = 0.0;
    for (let b = 0; b < n; b++) {
      lv += laplacian[ia * GRAPH_NODES + activeNodeIdx[b]] * eigVecBuf[b];
    }
    const diff = lv - lambda2 * eigVecBuf[a];
    res2 += diff * diff;
    sumV += eigVecBuf[a];
  }
  return { resNorm: Math.sqrt(res2), nullDot: Math.abs(sumV) };
}

// ----------------------------------------------------------------------------
// Gate A: Certified Shifted Lanczos Fiedler Solver (Path + Reflection Symmetry)
// ----------------------------------------------------------------------------
exp.chronos_init();
laplacian.fill(0.0);
for (let i = 0; i < 6; i++) {
  let deg = 0.0;
  for (let j = 0; j < 6; j++) {
    if (i === j) continue;
    const w = (Math.abs(i - j) === 1 ? 1.0 : 0.0) + 0.03;
    laplacian[i * GRAPH_NODES + j] = -w;
    deg += w;
  }
  laplacian[i * GRAPH_NODES + i] = deg;
}
const gPath = solveShiftedLanczosFiedler();
const exactPathL2 = (2.0 - Math.sqrt(3.0)) + 6.0 * 0.03; // 0.4479491924311227
const pathCert = eigenResidual64(6, gPath.lambda2);
if (Math.abs(gPath.lambda2 - exactPathL2) > 1e-6 || pathCert.resNorm > 1e-7 || pathCert.nullDot > 1e-11) {
  throw new Error(`Gate A1 (Path6 Lanczos) failed: l2=${gPath.lambda2} vs ${exactPathL2}, res=${pathCert.resNorm}`);
}

// Reflection-symmetric 2-cluster graph (0,3) vs (1,2) with eps=0.01 (ramp seed is 100% orthogonal to v2!)
laplacian.fill(0.0);
prevFiedlerTrack.fill(0.0);
const symClusters = [0, 1, 1, 0];
for (let i = 0; i < 4; i++) {
  let deg = 0.0;
  for (let j = 0; j < 4; j++) {
    if (i === j) continue;
    const w = symClusters[i] === symClusters[j] ? 1.0 : 0.01;
    laplacian[i * GRAPH_NODES + j] = -w;
    deg += w;
  }
  laplacian[i * GRAPH_NODES + i] = deg;
}
const gSym = solveShiftedLanczosFiedler();
const symCert = eigenResidual64(4, gSym.lambda2);
if (Math.abs(gSym.lambda2 - 0.04) > 1e-6 || Math.abs(gSym.spectralGap - 1.98) > 1e-5 || symCert.resNorm > 1e-7) {
  throw new Error(`Gate A2 (Symmetric Cluster Restart) failed: l2=${gSym.lambda2} (expected 0.04), gap=${gSym.spectralGap} (expected 1.98), res=${symCert.resNorm}`);
}
console.log(`  [PASS] Gate A (Shifted Lanczos + Orthogonal Restart): Path6 err=${Math.abs(gPath.lambda2 - exactPathL2).toExponential(2)}, SymCluster l2=${gSym.lambda2.toFixed(6)} (gap=${gSym.spectralGap.toFixed(4)}, ||Lv-lv||=${symCert.resNorm.toExponential(2)})`);

// Helper to draw static grid background & gray/colored discs
const staticBg = new Uint8Array(BYTES);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const idx = (y * W + x) * 4;
    const grid = (x % 24 === 0 || y % 24 === 0) ? 14 : 0;
    const base = 18 + ((x * 7 + y * 11) >> 6) + grid;
    staticBg[idx] = base;
    staticBg[idx + 1] = base + 4;
    staticBg[idx + 2] = base + 10;
    staticBg[idx + 3] = 255;
  }
}

function stampDisc(buf, cx, cy, r, g, b, cr, cg, cb) {
  const x0 = Math.max(0, Math.floor(cx - 8)), x1 = Math.min(W - 1, Math.ceil(cx + 8));
  const y0 = Math.max(0, Math.floor(cy - 8)), y1 = Math.min(H - 1, Math.ceil(cy + 8));
  for (let py = y0; py <= y1; py++) {
    for (let px = x0; px <= x1; px++) {
      const r2 = (px - cx) ** 2 + (py - cy) ** 2;
      if (r2 <= 60) {
        const idx = (py * W + px) * 4;
        const c = r2 <= 14;
        buf[idx] = c ? cr : r;
        buf[idx + 1] = c ? cg : g;
        buf[idx + 2] = c ? cb : b;
        buf[idx + 3] = 255;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Gate B: Pre-Assignment Spectral Fiedler Participation Ablation (gamma=0 vs gamma>0)
// Two identical neutral-gray targets (identical 2-adic & photometric templates) on lanes
// y=72 and y=100: when lane-0 target briefly dims/slows while lane-1 surges in ECA energy,
// gamma=0 suffers cross-lane hijack whereas gamma=0.55 preserves lane partition via Fiedler cut.
// ----------------------------------------------------------------------------
function runTwinGrayAblation(spectralGamma) {
  exp.chronos_init();
  exp.chronos_set_spectral_weight(spectralGamma);
  prevFiedlerTrack.fill(0.0);
  const ringViews = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), BYTES));
  const buf = new Uint8Array(BYTES);

  let crossLaneHijacks = 0;
  let trackLaneMap = new Map();

  for (let f = 0; f < 40; f++) {
    buf.set(staticBg);
    const x0 = f <= 20 ? 50 + f * 2.5 : 100 + (f - 20) * 0.25; // slows down after f=20 -> low ECA energy
    const x1 = f <= 20 ? 50 + f * 2.5 : 100 + (f - 20) * 5.2;  // surges at f=21 -> high ECA reward!
    const shade0 = f >= 21 && f <= 25 ? 182 : 210;              // slight photometric dip on lane 0
    stampDisc(buf, x0, 72,  shade0, shade0, shade0, 235, 235, 235);
    stampDisc(buf, x1, 102, 210,    210,    210,    235, 235, 235);

    ringViews[f % 12].set(buf);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);

    if (f >= 12) {
      for (let m = 0; m < MAX_TARGETS; m++) {
        const base = 32 + m * 56;
        if (telem[base] < 0.5 || telem[base + 44 + 11] < 0.5) continue;
        const tid = Math.round(telem[base + 1]);
        const iy = telem[base + 9];
        const lane = Math.abs(iy - 72) < Math.abs(iy - 102) ? 0 : 1;
        if (!trackLaneMap.has(tid)) trackLaneMap.set(tid, lane);
        else if (trackLaneMap.get(tid) !== lane) {
          crossLaneHijacks++;
          trackLaneMap.set(tid, lane);
        }
      }
    }
  }
  return { crossLaneHijacks, lambda2: telem[7], gap: telem[24] };
}

const ablZero = runTwinGrayAblation(0.0);
const ablSpec = runTwinGrayAblation(0.55);
if (ablZero.crossLaneHijacks < 1 || ablSpec.crossLaneHijacks !== 0) {
  throw new Error(`Gate B (Spectral Ablation) failed: expected gamma=0 hijacks>=1 and gamma=0.55 hijacks=0, got gamma=0:${ablZero.crossLaneHijacks}, gamma=0.55:${ablSpec.crossLaneHijacks}`);
}
console.log(`  [PASS] Gate B (Twin-Gray Spectral Ablation): gamma=0 cross-lane hijacks=${ablZero.crossLaneHijacks} vs gamma=0.55 hijacks=${ablSpec.crossLaneHijacks} (l2=${ablSpec.lambda2.toFixed(4)}, gap=${ablSpec.gap.toFixed(4)})`);

// ----------------------------------------------------------------------------
// Gate C: Certified Dual Projected-FISTA + Extended (>12-Frame) All-Occluded Safety
// ----------------------------------------------------------------------------
exp.chronos_init();
prevFiedlerTrack.fill(0.0);
{
  const ringViews = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), BYTES));
  const buf = new Uint8Array(BYTES);
  let maxGapVis = 0.0;
  let maxGapAllOcc = 0.0;

  for (let f = 0; f < 52; f++) {
    buf.set(staticBg);
    const x = 40 + f * 3.2;
    // Completely omit disc for 18 frames (f = 16..33 > 12-slot ring buffer!)
    if (f < 16 || f >= 34) {
      stampDisc(buf, x, 90, 245, 65, 70, 255, 235, 115);
    }
    ringViews[f % 12].set(buf);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);

    for (let k = 0; k < TELEMETRY_LEN; k++) {
      if (!Number.isFinite(telem[k])) {
        throw new Error(`Gate C: Non-finite telemetry[${k}]=${telem[k]} at frame ${f}`);
      }
      if (k >= 1550 && telem[k] !== 0.0) {
        throw new Error(`Gate C: Expected reserved tail telemetry[${k}]=0.0, got ${telem[k]} at frame ${f}`);
      }
    }
    if (f >= 10 && f < 16) maxGapVis = Math.max(maxGapVis, telem[25]);
    if (f >= 29 && f <= 33) maxGapAllOcc = Math.max(maxGapAllOcc, telem[25]);
  }
  if (maxGapVis < 0.0 || maxGapVis > 2.5 || maxGapAllOcc < 0.0 || maxGapAllOcc > 0.10) {
    throw new Error(`Gate C (Dual Projected-FISTA) gap out of certified range: vis=${maxGapVis}, allOcc=${maxGapAllOcc}`);
  }
  console.log(`  [PASS] Gate C (Certified Dual Projected-FISTA + 18-Frame Total Occlusion): 0 NaNs, certified P(X)-D(u) vis=${maxGapVis.toFixed(4)}, all-occluded=${maxGapAllOcc.toFixed(4)}`);
}

// ----------------------------------------------------------------------------
// Gate D: 2D Isotropic Principal-Axis Tremor + Equalized Hilbert Quadrature (60Hz & 120Hz) + K<7 Gate
// ----------------------------------------------------------------------------
function runAntiDiagonalHilbertCheck(fps, lookaheadKVal) {
  exp.chronos_init();
  exp.chronos_set_params(lookaheadKVal, 0.035, 2.2, 25.0, -Math.PI * 0.5, 1);
  prevFiedlerTrack.fill(0.0);
  const ringViews = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), BYTES));
  const buf = new Uint8Array(BYTES);

  let sumR2 = 0.0, sumH2 = 0.0, validFrames = 0;
  const totalFrames = Math.round(fps * 1.5);
  for (let f = 0; f < totalFrames; f++) {
    buf.set(staticBg);
    const t = f / fps;
    // Anti-diagonal (1, -1) 10 Hz tremor on top of steady drift (killed old dx + dy / dx + 0.5 dy projections!)
    const osc = 1.35 * Math.sin(2.0 * Math.PI * 10.0 * t);
    const cx = 60.0 + 45.0 * t + osc;
    const cy = 90.0 - osc;
    stampDisc(buf, cx, cy, 40, 230, 250, 255, 255, 255);
    ringViews[f % 12].set(buf);
    exp.chronos_pre_tick(1 / fps);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / fps);

    if (f >= Math.round(fps * 0.5)) {
      if (telem[26] > 0.5) {
        validFrames++;
        sumR2 += telem[9] * telem[9];
        sumH2 += telem[10] * telem[10];
      }
    }
  }
  const rRms = Math.sqrt(sumR2 / Math.max(1, validFrames));
  const hRms = Math.sqrt(sumH2 / Math.max(1, validFrames));
  return { validFrames, rRms, hRms, ratio: hRms / Math.max(1e-6, rRms) };
}

const hilb60 = runAntiDiagonalHilbertCheck(60, 8);
const hilb120 = runAntiDiagonalHilbertCheck(120, 8);
const hilbK4 = runAntiDiagonalHilbertCheck(60, 4);

if (hilb60.validFrames < 20 || hilb60.rRms < 0.35 || Math.abs(hilb60.ratio - 1.0) > 0.28) {
  throw new Error(`Gate D (60Hz Anti-Diag Hilbert) failed: valid=${hilb60.validFrames}, rRms=${hilb60.rRms}, H/R=${hilb60.ratio}`);
}
if (hilb120.validFrames < 40 || hilb120.rRms < 0.35 || Math.abs(hilb120.ratio - 1.0) > 0.28) {
  throw new Error(`Gate D (120Hz Anti-Diag Hilbert) failed: valid=${hilb120.validFrames}, rRms=${hilb120.rRms}, H/R=${hilb120.ratio}`);
}
if (hilbK4.validFrames !== 0) {
  throw new Error(`Gate D (K=4 DC-Leakage Hilbert Gate) failed: expected 0 valid frames, got ${hilbK4.validFrames}`);
}
console.log(`  [PASS] Gate D (Anti-Diag PCA + Equalized Hilbert): 60Hz H/R=${hilb60.ratio.toFixed(3)} (rRms=${hilb60.rRms.toFixed(2)}px), 120Hz H/R=${hilb120.ratio.toFixed(3)}, K=4 validFrames=${hilbK4.validFrames}`);

// ----------------------------------------------------------------------------
// Gate E: Mode C Physical Strike Deduplication (K=10 & K=11) + Reset Hygiene
// ----------------------------------------------------------------------------
for (const testK of [10, 11]) {
  exp.chronos_init();
  exp.chronos_set_params(testK, 0.035, 1.45, 25.0, -Math.PI * 0.5, 2);
  prevFiedlerTrack.fill(0.0);
  const ringViews = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), BYTES));
  const buf = new Uint8Array(BYTES);

  let strikeCount = 0;
  let cx = 60.0;
  for (let f = 0; f < 45; f++) {
    buf.set(staticBg);
    // Sharp air-percussion velocity snap at f=22
    const vx = f < 22 ? 4.5 : -4.5;
    cx += vx;
    stampDisc(buf, cx, 90, 50, 242, 112, 235, 255, 235);
    ringViews[f % 12].set(buf);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
    if (Math.round(telem[15]) !== 0) strikeCount++;
  }
  if (strikeCount !== 1) {
    throw new Error(`Gate E (Mode C K=${testK} Strike Dedup) failed: expected exactly 1 strike for single snap, got ${strikeCount}`);
  }
}
exp.chronos_reset_tracks();
if (exp.chronos_get_ingest_rgba_ptr() !== exp.chronos_get_ring_slot_ptr(0)) {
  throw new Error('Gate E (Reset Hygiene) failed: frame_idx / ingest slot did not reset to slot 0');
}
exp.chronos_post_tick(1 / 60);
if (Math.round(telem[3]) !== 0 || Math.round(telem[5]) !== 0 || Math.round(telem[6]) !== 0) {
  throw new Error(`Gate E (Reset Hygiene) failed: expected 0 active targets & swaps after reset, got active=${telem[3]}, swaps=${telem[5]}/${telem[6]}`);
}
console.log(`  [PASS] Gate E (Mode C K=10 & K=11 Physical Strike Dedup + Reset Hygiene): 1 impulse -> 1 trigger, clean state reset`);

// ----------------------------------------------------------------------------
// Gate F: Rank-2 ECA Affine Illumination Cancellation & 2-Adic Strong Triangle Inequality
// ----------------------------------------------------------------------------
{
  exp.chronos_init();
  const ringViews = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), BYTES));
  const ecaRes = new Float32Array(mem.buffer, exp.chronos_get_eca_res_ptr(), PIXELS);

  // Frame 0: seed structured static clutter
  ringViews[0].set(staticBg);
  exp.chronos_pre_tick(1 / 60);
  exp.chronos_post_tick(1 / 60);

  // Frame 1: apply global illumination offset (+5 RGB levels across entire frame; rank-1 fails, rank-2 cancels)
  const litBg = new Uint8Array(BYTES);
  for (let i = 0; i < PIXELS; i++) {
    litBg[i * 4] = Math.min(255, staticBg[i * 4] + 5);
    litBg[i * 4 + 1] = Math.min(255, staticBg[i * 4 + 1] + 5);
    litBg[i * 4 + 2] = Math.min(255, staticBg[i * 4 + 2] + 5);
    litBg[i * 4 + 3] = 255;
  }
  ringViews[1].set(litBg);
  exp.chronos_pre_tick(1 / 60);
  let maxAffineRes = 0.0;
  for (let i = 0; i < PIXELS; i++) {
    if (ecaRes[i] > maxAffineRes) maxAffineRes = ecaRes[i];
  }
  if (maxAffineRes > 1e-4) {
    throw new Error(`Gate F1 (Rank-2 ECA Illumination Offset Cancellation) failed: maxRes=${maxAffineRes}`);
  }

  // Verify strong ultrametric triangle inequality d_2(A, C) <= max(d_2(A, B), d_2(B, C)) across 1,000 triples
  let seed = 0x9E3779B9;
  const nextU32 = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return seed >>> 0;
  };
  const d2 = (a, b) => {
    const v = exp.padic_ultrametric_val(a, b) >>> 0;
    return v >= 32 ? 0.0 : Math.pow(2.0, -v);
  };
  for (let k = 0; k < 1000; k++) {
    const a = nextU32(), b = nextU32(), c = nextU32();
    if (d2(a, c) > Math.max(d2(a, b), d2(b, c)) + 1e-12) {
      throw new Error(`Gate F2 (2-Adic Strong Triangle Inequality) violated for (${a}, ${b}, ${c})`);
    }
  }
  console.log(`  [PASS] Gate F (Rank-2 ECA Offset Cancellation & 1000-Triple Ultrametric Check): maxAffineRes=${maxAffineRes.toExponential(2)}`);
}

// ----------------------------------------------------------------------------
// Gate G: Live Dual-Resolution Arena Switch (320x180 SD <-> 640x360 HD, 0 memory.grow)
// ----------------------------------------------------------------------------
{
  exp.chronos_init();
  if (exp.chronos_get_resolution() !== 0 || exp.chronos_get_frame_w() !== 320 || exp.chronos_get_frame_h() !== 180 || exp.chronos_get_pip_dim() !== 32) {
    throw new Error('Gate G: Unexpected initial SD dimensions');
  }
  exp.chronos_set_resolution(1);
  if (exp.chronos_get_resolution() !== 1 || exp.chronos_get_frame_w() !== 640 || exp.chronos_get_frame_h() !== 360 || exp.chronos_get_pip_dim() !== 64) {
    throw new Error('Gate G: Unexpected HD dimensions after chronos_set_resolution(1)');
  }
  const hdBytes = 640 * 360 * 4;
  const hdRing = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), hdBytes));
  const hdFrame = new Uint8Array(hdBytes);
  for (let i = 0; i < 640 * 360; i++) {
    hdFrame[i * 4] = 22;
    hdFrame[i * 4 + 1] = 28;
    hdFrame[i * 4 + 2] = 38;
    hdFrame[i * 4 + 3] = 255;
  }
  for (let f = 0; f < 24; f++) {
    const cx = 120 + f * 7;
    const cy = 180;
    hdRing[f % 12].set(hdFrame);
    const dst = hdRing[f % 12];
    for (let py = cy - 16; py <= cy + 16; py++) {
      for (let px = cx - 16; px <= cx + 16; px++) {
        if ((px - cx) ** 2 + (py - cy) ** 2 <= 240) {
          const idx = (py * 640 + px) * 4;
          dst[idx] = 248; dst[idx + 1] = 64; dst[idx + 2] = 72; dst[idx + 3] = 255;
        }
      }
    }
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
  }
  if (mem.buffer.byteLength !== 448 * 65536) {
    throw new Error(`Gate G: Arena grew unexpectedly to ${mem.buffer.byteLength}`);
  }
  if (telem[30] !== 640 || telem[31] !== 360 || pipBuf[16383] !== 255) {
    throw new Error(`Gate G: Expected HD telemetry (640x360) and 64x64 PiP alpha=255, got ${telem[30]}x${telem[31]}, alpha=${pipBuf[16383]}`);
  }
  exp.chronos_set_resolution(0);
  if (exp.chronos_get_frame_w() !== 320 || exp.chronos_get_pip_dim() !== 32) {
    throw new Error('Gate G: Failed to restore 320x180 SD mode');
  }
  console.log('  [PASS] Gate G (Dual-Resolution 320x180 SD <-> 640x360 HD Switch): 448 static pages, 64x64 PiP verified');
}

// ----------------------------------------------------------------------------
// Gate H: Ballistic Quadratic Gravity, CTRV Curved Coasting, Kasa Virtual Pivot & Collision Angles
// ----------------------------------------------------------------------------
{
  const sdBytes = 320 * 180 * 4;
  const sdBg = new Uint8Array(sdBytes);
  for (let y = 0; y < 180; y++) {
    for (let x = 0; x < 320; x++) {
      const idx = (y * 320 + x) * 4;
      const g = (x % 20 === 0 || y % 20 === 0) ? 10 : 0;
      sdBg[idx] = 18 + g; sdBg[idx + 1] = 22 + g; sdBg[idx + 2] = 30 + g; sdBg[idx + 3] = 255;
    }
  }
  const stampSubpxDisc = (dst, cx, cy) => {
    const x0 = Math.max(0, Math.floor(cx - 8)), x1 = Math.min(319, Math.ceil(cx + 8));
    const y0 = Math.max(0, Math.floor(cy - 8)), y1 = Math.min(179, Math.ceil(cy + 8));
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const d2 = (px - cx) ** 2 + (py - cy) ** 2;
        if (d2 <= 42.0) {
          const w = Math.max(0, Math.min(1, (42.0 - d2) / 10.0));
          const c = d2 <= 10.0;
          const tr = c ? 255 : 246, tg = c ? 240 : 72, tb = c ? 120 : 62;
          const idx = (py * 320 + px) * 4;
          dst[idx] = Math.round(dst[idx] * (1 - w) + tr * w);
          dst[idx + 1] = Math.round(dst[idx + 1] * (1 - w) + tg * w);
          dst[idx + 2] = Math.round(dst[idx + 2] * (1 - w) + tb * w);
        }
      }
    }
  };

  // H1: Full-segment closed-form quadratic gravity & apex theta=0 crossing
  exp.chronos_init();
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
  const ring = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), sdBytes));
  let maxAscTheta = -999, minDescTheta = 999, apexThetaAbs = 999, bestApexYErr = 999, gHat = 0;
  for (let f = 0; f < 56; f++) {
    const dst = ring[f % 12];
    dst.set(sdBg);
    if (f >= 4) {
      const u = f - 4;
      const cx = 36.0 + 4.2 * u;
      const cy = 152.0 - 6.5 * u + 0.5 * 0.25 * u * u; // g_true = 900 px/s^2, apex at u=26 (cy=67.5)
      stampSubpxDisc(dst, cx, cy);
    }
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
    if (telem[1248] > 0) gHat = telem[1248];
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56, kbase = 1120 + m * 8;
      if (telem[base] > 0.5) {
        if (telem[base + 17] > 1.0) {
          bestApexYErr = Math.min(bestApexYErr, Math.abs(telem[base + 16] - 67.5));
        }
        const th = telem[kbase];
        const fDisp = f - Math.round(telem[1]);
        if (fDisp >= 10 && fDisp <= 20) maxAscTheta = Math.max(maxAscTheta, th);
        if (fDisp === 30) apexThetaAbs = Math.min(apexThetaAbs, Math.abs(th));
        if (fDisp >= 38 && fDisp <= 46) minDescTheta = Math.min(minDescTheta, th);
      }
    }
  }
  if (Math.abs(gHat - 900.0) > 55.0 || bestApexYErr > 3.0 || maxAscTheta < 35.0 || apexThetaAbs > 6.5 || minDescTheta > -30.0) {
    throw new Error(`Gate H1 (Ballistic Quadratic & Theta): gHat=${gHat.toFixed(1)}, apexYErr=${bestApexYErr.toFixed(2)}, asc=${maxAscTheta.toFixed(1)}, apexTh=${apexThetaAbs.toFixed(2)}, desc=${minDescTheta.toFixed(1)}`);
  }

  // H2: CTRV Curved-Arc Occlusion Coasting behind opaque pillar (+ negative pivot/Hz check)
  exp.chronos_init();
  let arcTrackId = -1, arcIdConsistent = true, arcBridges = 0, falsePivotFrames = 0, falseHzFrames = 0;
  for (let f = 0; f < 54; f++) {
    const dst = ring[f % 12];
    dst.set(sdBg);
    if (f >= 4) {
      const ang = -2.35 + 0.048 * (f - 4); // Smooth circular turn R=96 around (160, 172)
      const cx = 160.0 + 96.0 * Math.cos(ang);
      const cy = 172.0 + 96.0 * Math.sin(ang);
      stampSubpxDisc(dst, cx, cy);
    }
    // Opaque pillar occluding curved arc near top of turn x in [145, 177] (~7 frames)
    for (let py = 20; py < 140; py++) {
      for (let px = 145; px < 177; px++) {
        const idx = (py * 320 + px) * 4;
        dst[idx] = 36; dst[idx + 1] = 42; dst[idx + 2] = 52; dst[idx + 3] = 255;
      }
    }
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
    if (telem[13] > 0.01) falseHzFrames++;
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56, kbase = 1120 + m * 8;
      if (telem[base] > 0.5) {
        const tid = Math.round(telem[base + 1]);
        if (arcTrackId < 0) arcTrackId = tid;
        else if (tid !== arcTrackId) arcIdConsistent = false;
        if (telem[base + 11] > 0.5) arcBridges++;
        if (telem[kbase + 6] > -1e5) falsePivotFrames++;
      }
    }
  }
  if (arcTrackId < 0 || !arcIdConsistent || arcBridges < 4 || falsePivotFrames !== 0 || falseHzFrames !== 0) {
    throw new Error(`Gate H2 (CTRV Curved Occlusion Coasting / No False Pivot-Hz): tid=${arcTrackId}, consistent=${arcIdConsistent}, bridges=${arcBridges}, falsePivots=${falsePivotFrames}, falseHz=${falseHzFrames}`);
  }

  // H3: Mode B Kasa 2x2 Virtual Pendulum Pivot Recovery
  exp.chronos_init();
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 1);
  let bestPivotErr = 999;
  for (let f = 0; f < 56; f++) {
    const dst = ring[f % 12];
    dst.set(sdBg);
    if (f >= 4) {
      const th = 0.56 * Math.sin(2.0 * Math.PI * 0.85 * ((f - 4) / 60.0) - 0.8);
      const cx = 160.0 + 112.0 * Math.sin(th);
      const cy = 22.0 + 112.0 * Math.cos(th);
      stampSubpxDisc(dst, cx, cy);
    }
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56, kbase = 1120 + m * 8;
      if (telem[base] > 0.5 && telem[kbase + 6] > -1e5) {
        const err = Math.hypot(telem[kbase + 6] - 160.0, telem[kbase + 7] - 22.0);
        bestPivotErr = Math.min(bestPivotErr, err);
      }
    }
  }
  if (bestPivotErr > 4.0) {
    throw new Error(`Gate H3 (Kasa Virtual Pendulum Pivot): bestPivotErr=${bestPivotErr.toFixed(2)} px > 4.0 px`);
  }

  // H4: Mode C Deflection Angle Protractor & Coefficient of Restitution
  exp.chronos_init();
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 2);
  let strikeCount = 0, measDeflect = 0, measRest = 0;
  const rad60 = -Math.PI / 3.0; // -60 deg downward-right deflection
  for (let f = 0; f < 46; f++) {
    const dst = ring[f % 12];
    dst.set(sdBg);
    if (f >= 4) {
      const u = f - 4;
      let cx, cy;
      if (u <= 20) {
        cx = 64.0 + 4.5 * u;
        cy = 68.0;
      } else {
        const du = u - 20;
        cx = 154.0 + 3.6 * Math.cos(rad60) * du;
        cy = 68.0 - 3.6 * Math.sin(rad60) * du; // e_true = 3.6 / 4.5 = 0.80
      }
      stampSubpxDisc(dst, cx, cy);
    }
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56, ebase = 992 + m * 8, kbase = 1120 + m * 8;
      if (telem[base] > 0.5) {
        if (telem[ebase] > 0.5) strikeCount++;
        if (Math.abs(telem[kbase + 3]) > 1.0) {
          measDeflect = telem[kbase + 3];
          measRest = telem[kbase + 4];
        }
      }
    }
  }
  if (strikeCount < 1 || Math.abs(Math.abs(measDeflect) - 60.0) > 3.5 || Math.abs(measRest - 0.80) > 0.08) {
    throw new Error(`Gate H4 (Collision Deflection & Restitution): strikes=${strikeCount}, deflect=${measDeflect.toFixed(2)} deg, e=${measRest.toFixed(3)}`);
  }
  console.log(`  [PASS] Gate H (Quadratic Gravity g=${gHat.toFixed(1)}, CTRV Arc Coast, Kasa Pivot err=${bestPivotErr.toFixed(2)}px, Deflect=${measDeflect.toFixed(1)}° v+/v-=${measRest.toFixed(2)})`);

  // --------------------------------------------------------------------------
  // Gate I: Variable-PTS Velocity Invariance, Multi-Arc Auto-g Consensus & Live Camera Pan/Cut Guard
  // --------------------------------------------------------------------------
  // I1: Variable-PTS jitter spike at t must NOT distort historical t-K speed; dt > 0.25s increments discont_count
  exp.chronos_init();
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
  let pts = 0.0;
  let speedBeforeSpike = 0.0, speedAtSpike = 0.0;
  for (let f = 0; f < 36; f++) {
    const dt = f === 0 ? (1 / 60) : (f === 28 ? (2.6 / 60) : (f === 34 ? 0.40 : (1 / 60)));
    if (f > 0) pts += dt;
    const dst = ring[f % 12];
    dst.set(sdBg);
    const cx = Math.min(306.0, 28.0 + 180.0 * Math.min(pts, 1.2));
    stampSubpxDisc(dst, cx, 92.0);
    exp.chronos_set_ingest_pts(pts);
    exp.chronos_pre_tick(dt);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(dt);
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56, kbase = 1120 + m * 8;
      if (telem[base] > 0.5) {
        const spd = telem[kbase + 1];
        if (f === 27) speedBeforeSpike = spd;
        if (f === 28) speedAtSpike = spd;
      }
    }
  }
  const spikeSpeedErrPct = (100.0 * Math.abs(speedAtSpike - 180.0)) / 180.0;
  const discontObserved = Math.round(telem[1263]);
  if (Math.abs(speedBeforeSpike - 180.0) > 4.5 || spikeSpeedErrPct > 2.5 || discontObserved < 1) {
    throw new Error(`Gate I1 (Variable-PTS Invariance & Discont): pre=${speedBeforeSpike.toFixed(2)}, spike=${speedAtSpike.toFixed(2)} (${spikeSpeedErrPct.toFixed(2)}%), discont=${discontObserved}`);
  }

  // I2: Multi-arc Auto-g lock (1st arc stays provisional < 0.50; 2nd independent arc locks >= 0.50)
  exp.chronos_init();
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
  let confAfterArc1 = 0.0, confAfterArc2 = 0.0, arcsAfter2 = 0;
  for (let f = 0; f < 92; f++) {
    const dst = ring[f % 12];
    dst.set(sdBg);
    if (f >= 4 && f <= 40) {
      const u = f - 4;
      stampSubpxDisc(dst, 32.0 + 4.0 * u, 150.0 - 6.2 * u + 0.5 * 0.25 * u * u);
    } else if (f >= 50 && f <= 86) {
      const u = f - 50;
      stampSubpxDisc(dst, 288.0 - 4.0 * u, 150.0 - 6.2 * u + 0.5 * 0.25 * u * u);
    }
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
    if (f === 42) confAfterArc1 = telem[1249];
    if (f >= 74) {
      confAfterArc2 = Math.max(confAfterArc2, telem[1249]);
      arcsAfter2 = Math.max(arcsAfter2, Math.round(telem[1259]));
    }
  }
  if (confAfterArc1 >= 0.50 || confAfterArc2 < 0.50 || arcsAfter2 < 2) {
    throw new Error(`Gate I2 (Multi-Arc Auto-g Consensus): conf1=${confAfterArc1.toFixed(3)} (must be <0.50), conf2=${confAfterArc2.toFixed(3)} (must be >=0.50), arcs=${arcsAfter2}`);
  }

  // I3: Camera Motion Detector (Global +15 RGB -> 0, Rigid Pan -> 1, Hard Cut -> 2 + ECA/Track Reset)
  exp.chronos_init();
  const makeTexturedScene = (shiftX, variant) => {
    const out = new Uint8Array(sdBytes);
    for (let y = 0; y < 180; y++) {
      for (let x = 0; x < 320; x++) {
        const idx = (y * 320 + x) * 4;
        const sx = (x + shiftX + 3200) % 320;
        let v;
        if (variant === 0) {
          const block = (((sx >> 4) * 37 + (y >> 4) * 53) & 63);
          const edge = (sx % 16 < 2 || y % 16 < 2) ? 48 : 0;
          v = 24 + block + edge;
        } else {
          const diag = (((x + y) >> 3) & 1) ? 195 : 22;
          v = diag + ((x * 13) & 15);
        }
        out[idx] = v; out[idx + 1] = Math.min(255, v + 6); out[idx + 2] = Math.min(255, v + 12); out[idx + 3] = 255;
      }
    }
    return out;
  };
  let fi = 0;
  for (let w = 0; w < 6; w++) {
    const dst = ring[fi % 12];
    dst.set(makeTexturedScene(0, 0));
    stampSubpxDisc(dst, 60 + w * 4, 90);
    exp.chronos_pre_tick(1 / 60);
    exp.chronos_post_tick(1 / 60);
    fi++;
  }
  // Global +15 RGB exposure jump -> stationary (state 0)
  {
    const lit = makeTexturedScene(0, 0);
    for (let i = 0; i < 320 * 180; i++) {
      lit[i * 4] = Math.min(255, lit[i * 4] + 15);
      lit[i * 4 + 1] = Math.min(255, lit[i * 4 + 1] + 15);
      lit[i * 4 + 2] = Math.min(255, lit[i * 4 + 2] + 15);
    }
    ring[fi % 12].set(lit);
    exp.chronos_pre_tick(1 / 60);
    exp.chronos_post_tick(1 / 60);
    fi++;
  }
  const stateExposure = Math.round(telem[1257]);
  // Coherent rigid pan (4 px per frame across whole scene) -> pan hold (state 1)
  let sawPanHold = false;
  for (let p = 1; p <= 3; p++) {
    ring[fi % 12].set(makeTexturedScene(p * 4, 0));
    exp.chronos_pre_tick(1 / 60);
    exp.chronos_post_tick(1 / 60);
    fi++;
    if (Math.round(telem[1257]) === 1) sawPanHold = true;
  }
  // Hard scene cut -> state 2 + tracks flushed
  ring[fi % 12].set(makeTexturedScene(0, 1));
  exp.chronos_pre_tick(1 / 60);
  exp.chronos_post_tick(1 / 60);
  const stateHardCut = Math.round(telem[1257]);
  const activeAfterCut = Math.round(telem[3]);
  if (stateExposure !== 0 || !sawPanHold || stateHardCut !== 2 || activeAfterCut !== 0) {
    throw new Error(`Gate I3 (Camera Motion/Cut Guard): exposureState=${stateExposure}, sawPanHold=${sawPanHold}, cutState=${stateHardCut}, activeAfterCut=${activeAfterCut}`);
  }
  console.log(`  [PASS] Gate I (PTS Jitter Err=${spikeSpeedErrPct.toFixed(2)}%, 2-Arc Auto-g conf=${confAfterArc1.toFixed(2)}->${confAfterArc2.toFixed(2)}, Cam Pan/Cut Guard 0/1/2)`);
}

// ----------------------------------------------------------------------------
// Gate J: Ego-Motion Pan-Warped ECA & Mid-Pan Target Custody (SD 320x180 & HD 640x360)
// Proves pre-R&D failure (0% mid-pan spawn, 25 dB pan clutter collapse, 53 HD pan ghosts)
// is structurally eliminated across both resolutions.
// ----------------------------------------------------------------------------
{
  const runPanGate = (resMode) => {
    exp.chronos_init();
    exp.chronos_set_resolution(resMode);
    prevFiedlerTrack.fill(0.0);
    const w = exp.chronos_get_frame_w();
    const h = exp.chronos_get_frame_h();
    const s = w / 320.0;
    const fBytes = w * h * 4;
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    const rViews = Array.from({ length: 12 }, (_, slot) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), fBytes));
    const blkShift = resMode === 0 ? 4 : 5;
    const gPeriod = Math.round(24 * s);
    const gThick = Math.round(2 * s);
    const rad = 5.5 * s;
    const hist = [];
    let matchA = 0, totA = 0, matchB = 0, totB = 0, minClutter = 999.0, fpAfterSpawn = 0;

    for (let f = 0; f < 76; f++) {
      const camX = f <= 16 ? 0 : (f - 16) * (4 * s);
      const camY = f <= 16 ? 0 : (f - 16) * (2 * s);
      const tA = { wx: (50 + f * 5.2) * s, wy: (84 + 18 * Math.sin(f * 0.12)) * s, r: 245, g: 70, b: 55 };
      const tB = f >= 24 ? { wx: camX + (70 + (f - 24) * 3.8) * s, wy: camY + (132 - (f - 24) * 1.4) * s, r: 55, g: 220, b: 245 } : null;
      hist.push({ camX, camY, tA, tB });

      const out = rViews[f % 12];
      for (let y = 0; y < h; y++) {
        const wy = y + camY + 3600;
        for (let x = 0; x < w; x++) {
          const wx = x + camX + 6400;
          const blk = (((wx >> blkShift) * 37 + (wy >> blkShift) * 53) & 63);
          const grd = ((wx % gPeriod) < gThick || (wy % gPeriod) < gThick) ? 42 : 0;
          const v = 28 + blk + grd;
          const idx = (y * w + x) * 4;
          out[idx] = v; out[idx + 1] = v + 6; out[idx + 2] = v + 12; out[idx + 3] = 255;
        }
      }
      for (const t of (tB ? [tA, tB] : [tA])) {
        const sx = t.wx - camX, sy = t.wy - camY;
        const x0 = Math.max(0, Math.floor(sx - rad - 2)), x1 = Math.min(w - 1, Math.ceil(sx + rad + 2));
        const y0 = Math.max(0, Math.floor(sy - rad - 2)), y1 = Math.min(h - 1, Math.ceil(sy + rad + 2));
        for (let py = y0; py <= y1; py++) {
          for (let px = x0; px <= x1; px++) {
            const d = Math.hypot(px - sx, py - sy);
            if (d <= rad + 1.0) {
              const a = Math.max(0.0, Math.min(1.0, rad + 0.5 - d));
              const idx = (py * w + px) * 4;
              out[idx] = Math.round(out[idx] * (1 - a) + t.r * a);
              out[idx + 1] = Math.round(out[idx + 1] * (1 - a) + t.g * a);
              out[idx + 2] = Math.round(out[idx + 2] * (1 - a) + t.b * a);
            }
          }
        }
      }
      exp.chronos_pre_tick(1 / 60);
      solveShiftedLanczosFiedler();
      exp.chronos_post_tick(1 / 60);

      if (f >= 20 && telem[22] > 0.5) minClutter = Math.min(minClutter, telem[4]);
      const fDisp = f - Math.round(telem[1]);
      if (fDisp >= 20 && fDisp < hist.length) {
        const ref = hist[fDisp];
        const preds = [];
        for (let m = 0; m < MAX_TARGETS; m++) {
          const base = 32 + m * 56;
          if (telem[base] > 0.5) preds.push({ x: telem[base + 2], y: telem[base + 3] });
        }
        const used = new Set();
        totA++;
        let ai = -1, ad = 18 * s;
        preds.forEach((p, idx) => { const d = Math.hypot(p.x - (ref.tA.wx - ref.camX), p.y - (ref.tA.wy - ref.camY)); if (d < ad) { ad = d; ai = idx; } });
        if (ai >= 0) { matchA++; used.add(ai); }
        if (ref.tB) {
          if (fDisp >= 30) totB++;
          let bi = -1, bd = 18 * s;
          preds.forEach((p, idx) => {
            if (used.has(idx)) return;
            const d = Math.hypot(p.x - (ref.tB.wx - ref.camX), p.y - (ref.tB.wy - ref.camY));
            if (d < bd) { bd = d; bi = idx; }
          });
          if (bi >= 0) { if (fDisp >= 30) matchB++; used.add(bi); }
          fpAfterSpawn += preds.length - used.size;
        }
      }
    }
    return { minClutter, recA: matchA / Math.max(1, totA), recB: matchB / Math.max(1, totB), fpAfterSpawn, panSpd: telem[1258] };
  };

  const panSd = runPanGate(0);
  const panHd = runPanGate(1);
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  prevFiedlerTrack.fill(0.0);

  if (panSd.minClutter < 65.0 || panHd.minClutter < 65.0 || panSd.recA < 0.78 || panHd.recA < 0.78 || panSd.recB < 0.95 || panHd.recB < 0.95 || panSd.fpAfterSpawn > 2 || panHd.fpAfterSpawn > 2) {
    throw new Error(`Gate J (Pan-Warped ECA & Mid-Pan Custody): SD(clut=${panSd.minClutter.toFixed(1)}dB, A=${(100*panSd.recA).toFixed(1)}%, B=${(100*panSd.recB).toFixed(1)}%, fp=${panSd.fpAfterSpawn}) HD(clut=${panHd.minClutter.toFixed(1)}dB, A=${(100*panHd.recA).toFixed(1)}%, B=${(100*panHd.recB).toFixed(1)}%, fp=${panHd.fpAfterSpawn})`);
  }
  console.log(`  [PASS] Gate J (Pan-Warped ECA & Mid-Pan Spawn): SD=${panSd.minClutter.toFixed(1)}dB B=${(100*panSd.recB).toFixed(0)}% fp=${panSd.fpAfterSpawn} | HD=${panHd.minClutter.toFixed(1)}dB B=${(100*panHd.recB).toFixed(0)}% fp=${panHd.fpAfterSpawn}`);
}

// Slow fractional pans (1.25*s px/f + 0.5*s micro-bob) sit below the 2*s pyramid stride
// and previously collapsed clutter to 23.9 dB while spawning 308 HD background false positives.
{
  const runSlowSubpxPanGate = (resMode) => {
    exp.chronos_init();
    exp.chronos_set_resolution(resMode);
    prevFiedlerTrack.fill(0.0);
    const w = exp.chronos_get_frame_w();
    const h = exp.chronos_get_frame_h();
    const s = w / 320.0;
    const fBytes = w * h * 4;
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    const rViews = Array.from({ length: 12 }, (_, slot) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), fBytes));

    let camX = 0.0, camY = 0.0;
    let minClutter = Infinity, minHalfClutter = Infinity;
    let validClutter = 0, validHalfClutter = 0;
    let fpTracks = 0, dupFrames = 0;
    let lockedA = 0, totalA = 0, lockedB = 0, totalB = 0;
    let idA = -1, idB = -1, swapsA = 0, swapsB = 0;

    for (let f = 0; f < 118; f++) {
      if (f >= 30 && f < 80) {
        camX += 1.25;
        camY += 0.5 * Math.sin(f * 0.25);
      } else if (f >= 80) {
        camX -= 0.50 / s;
      }
      const discA = { wx: 52 + f * 1.45, wy: 64, r: 245, g: 60, b: 65 };
      const discB = f >= 45 ? { wx: camX + 88 + (f - 45) * 1.85, wy: camY + 118, r: 45, g: 235, b: 245 } : null;
      const out = rViews[f % 12];

      for (let y = 0; y < h; y++) {
        const wy = y / s + camY;
        for (let x = 0; x < w; x++) {
          const wx = x / s + camX;
          const v1 = Math.sin(wx * 0.17) * Math.cos(wy * 0.19) * 38.0;
          const v2 = Math.sin((wx + wy) * 0.07) * 22.0;
          const base = Math.max(12, Math.min(220, Math.round(85 + v1 + v2)));
          const idx = (y * w + x) * 4;
          out[idx] = base; out[idx + 1] = base + 6; out[idx + 2] = base + 12; out[idx + 3] = 255;
        }
      }
      for (const d of [discA, discB]) {
        if (!d) continue;
        const sx = (d.wx - camX) * s, sy = (d.wy - camY) * s;
        const rad = 7 * s, r2Max = 49 * s * s;
        for (let py = Math.max(0, Math.floor(sy - rad)); py <= Math.min(h - 1, Math.ceil(sy + rad)); py++) {
          for (let px = Math.max(0, Math.floor(sx - rad)); px <= Math.min(w - 1, Math.ceil(sx + rad)); px++) {
            if ((px - sx) ** 2 + (py - sy) ** 2 <= r2Max) {
              const idx = (py * w + px) * 4;
              out[idx] = d.r; out[idx + 1] = d.g; out[idx + 2] = d.b; out[idx + 3] = 255;
            }
          }
        }
      }

      exp.chronos_pre_tick(1 / 60);
      solveShiftedLanczosFiedler();
      exp.chronos_post_tick(1 / 60);

      if (f >= 35 && telem[22] > 0.5) {
        if (!Number.isFinite(telem[4])) minClutter = NaN;
        else minClutter = Math.min(minClutter, telem[4]);
        validClutter++;
        if (f >= 84) {
          minHalfClutter = Math.min(minHalfClutter, telem[4]);
          validHalfClutter++;
        }
      }
      let hitsA = 0, hitsB = 0;
      for (let m = 0; m < MAX_TARGETS; m++) {
        const base = 32 + m * 56;
        if (telem[base] > 0.5) {
          const tid = Math.round(telem[base + 1]);
          const tx = telem[base + 8], ty = telem[base + 9];
          const dA = Math.hypot(tx - (discA.wx - camX) * s, ty - (discA.wy - camY) * s);
          const dB = discB ? Math.hypot(tx - (discB.wx - camX) * s, ty - (discB.wy - camY) * s) : 9999;
          if (dA < 18 * s) {
            hitsA++;
            if (f >= 15) {
              if (idA < 0) idA = tid;
              else if (tid !== idA) { swapsA++; idA = tid; }
            }
          }
          if (discB && dB < 18 * s) {
            hitsB++;
            if (f >= 52) {
              if (idB < 0) idB = tid;
              else if (tid !== idB) { swapsB++; idB = tid; }
            }
          }
          if (dA > 24 * s && dB > 24 * s && f >= 35) fpTracks++;
        }
      }
      if (hitsA > 1 || hitsB > 1) dupFrames++;
      if (f >= 15) {
        totalA++;
        if (hitsA >= 1) lockedA++;
      }
      if (f >= 52) {
        totalB++;
        if (hitsB >= 1) lockedB++;
      }
    }
    return {
      minClutter,
      minHalfClutter,
      validClutter,
      validHalfClutter,
      recA: lockedA / Math.max(1, totalA),
      recB: lockedB / Math.max(1, totalB),
      swaps: swapsA + swapsB + (idA > 0 && idA === idB ? 1 : 0),
      dupFrames,
      fpTracks,
    };
  };

  const slowSd = runSlowSubpxPanGate(0);
  const slowHd = runSlowSubpxPanGate(1);
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  prevFiedlerTrack.fill(0.0);

  const gateKOk = (r) =>
    Number.isFinite(r.minClutter) && r.minClutter >= 44.0 && r.minClutter < 200.0 &&
    Number.isFinite(r.minHalfClutter) && r.minHalfClutter >= 44.0 && r.minHalfClutter < 200.0 &&
    r.validClutter >= 65 && r.validHalfClutter >= 25 &&
    r.recA >= 0.90 && r.recB >= 0.88 &&
    r.swaps === 0 && r.dupFrames === 0 && r.fpTracks <= 2;

  if (!gateKOk(slowSd) || !gateKOk(slowHd)) {
    throw new Error(
      `Gate K (Slow & Sub-Pixel Pan Custody): ` +
      `SD(clut=${slowSd.minClutter.toFixed(1)}dB, half=${slowSd.minHalfClutter.toFixed(1)}dB, n=${slowSd.validClutter}/${slowSd.validHalfClutter}, A=${(100*slowSd.recA).toFixed(1)}%, B=${(100*slowSd.recB).toFixed(1)}%, sw=${slowSd.swaps}, dup=${slowSd.dupFrames}, fp=${slowSd.fpTracks}) ` +
      `HD(clut=${slowHd.minClutter.toFixed(1)}dB, half=${slowHd.minHalfClutter.toFixed(1)}dB, n=${slowHd.validClutter}/${slowHd.validHalfClutter}, A=${(100*slowHd.recA).toFixed(1)}%, B=${(100*slowHd.recB).toFixed(1)}%, sw=${slowHd.swaps}, dup=${slowHd.dupFrames}, fp=${slowHd.fpTracks})`
    );
  }
  console.log(
    `  [PASS] Gate K (Slow & Sub-Pixel Pan): ` +
    `SD=${slowSd.minClutter.toFixed(1)}dB (0.5px=${slowSd.minHalfClutter.toFixed(1)}dB) A=${(100*slowSd.recA).toFixed(0)}% B=${(100*slowSd.recB).toFixed(0)}% sw=0 dup=0 fp=${slowSd.fpTracks} | ` +
    `HD=${slowHd.minClutter.toFixed(1)}dB (0.5px=${slowHd.minHalfClutter.toFixed(1)}dB) A=${(100*slowHd.recA).toFixed(0)}% B=${(100*slowHd.recB).toFixed(0)}% sw=0 dup=0 fp=${slowHd.fpTracks}`
  );
}

{
  const runTwinValleyAndRigidBodyGate = (resMode) => {
    exp.chronos_init();
    exp.chronos_set_resolution(resMode);
    prevFiedlerTrack.fill(0.0);
    const w = exp.chronos_get_frame_w();
    const h = exp.chronos_get_frame_h();
    const s = w / 320.0;
    const fBytes = w * h * 4;
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    const rViews = Array.from({ length: 12 }, (_, slot) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), fBytes));

    const signX0 = Math.round(146 * s), signX1 = Math.round(178 * s);
    const signY0 = Math.round(121 * s), signY1 = Math.round(147 * s);
    const stampGantrySign = (buf) => {
      for (let py = signY0; py < signY1; py++) {
        for (let px = signX0; px < signX1; px++) {
          const idx = (py * w + px) * 4;
          const border = (px < signX0 + 2 * s || px >= signX1 - 2 * s || py < signY0 + 2 * s || py >= signY1 - 2 * s);
          const glyph = ((((px / s) | 0) * 5 + ((py / s) | 0) * 7) & 7) === 0 ? 44 : 0;
          const base = border ? 168 : (52 + glyph);
          buf[idx] = base; buf[idx + 1] = Math.min(255, base + 22); buf[idx + 2] = Math.min(255, base + 12); buf[idx + 3] = 255;
        }
      }
    };

    const bg = new Uint8Array(fBytes);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const v = 26 + ((((x / s) | 0) * 13 + ((y / s) | 0) * 19) & 15);
        bg[idx] = v; bg[idx + 1] = v + 4; bg[idx + 2] = v + 8; bg[idx + 3] = 255;
      }
    }
    stampGantrySign(bg);

    const hist = [];
    let lockA = 0, lockB = 0, lockBoth = 0, lockCar = 0, evalFrames = 0;
    let idA = -1, idB = -1, idCar = -1, idCarPre = -1, idCarPost = -1;
    let swaps = 0, carDupFrames = 0, carBridges = 0, signPins = 0, fp = 0;

    for (let f = 0; f < 82; f++) {
      const out = rViews[f % 12];
      out.set(bg);
      const tx = (30 + f * 2.4) * s;
      const tyA = (54 + 3.0 * Math.sin(f * 0.08)) * s;
      const tyB = tyA + 18.0 * s;
      const carX = (284 - f * 2.7) * s;
      const carY = 134 * s;
      hist.push({ tx, tyA, tyB, carX, carY });

      const rad = 5 * s, r2Max = 25 * s * s;
      for (const cy of [tyA, tyB]) {
        for (let py = Math.max(0, Math.floor(cy - rad)); py <= Math.min(h - 1, Math.ceil(cy + rad)); py++) {
          for (let px = Math.max(0, Math.floor(tx - rad)); px <= Math.min(w - 1, Math.ceil(tx + rad)); px++) {
            const r2 = (px - tx) ** 2 + (py - cy) ** 2;
            if (r2 <= r2Max) {
              const idx = (py * w + px) * 4;
              const core = r2 <= 6 * s * s ? 25 : 0;
              out[idx] = Math.min(255, 230 + core);
              out[idx + 1] = Math.min(255, 185 + core);
              out[idx + 2] = 65;
              out[idx + 3] = 255;
            }
          }
        }
      }

      const hw = Math.round(13 * s), hh = Math.round(5.5 * s);
      for (let py = Math.max(0, Math.floor(carY - hh)); py <= Math.min(h - 1, Math.ceil(carY + hh)); py++) {
        for (let px = Math.max(0, Math.floor(carX - hw)); px <= Math.min(w - 1, Math.ceil(carX + hw)); px++) {
          const idx = (py * w + px) * 4;
          const relX = (px - carX) / s;
          const tex = (((px >> (resMode === 1 ? 2 : 1)) + (py >> (resMode === 1 ? 2 : 1))) & 1) ? 26 : 0;
          const cabin = Math.abs(relX) < 5.0 ? 32 : 0;
          const v = 135 + cabin + tex;
          out[idx] = v; out[idx + 1] = v + 6; out[idx + 2] = v + 12; out[idx + 3] = 255;
        }
      }
      stampGantrySign(out);

      exp.chronos_pre_tick(1 / 60);
      solveShiftedLanczosFiedler();
      exp.chronos_post_tick(1 / 60);

      const dispF = f - 8;
      if (dispF >= 16) {
        const gt = hist[dispF];
        evalFrames++;
        let hitA = false, hitB = false, carHits = 0;
        for (let m = 0; m < MAX_TARGETS; m++) {
          const base = 32 + m * 56;
          if (telem[base] > 0.5) {
            const tid = Math.round(telem[base + 1]);
            const mx = telem[base + 2], my = telem[base + 3];
            const isBridged = telem[base + 11] > 0.5;
            const dA = Math.hypot(mx - gt.tx, my - gt.tyA);
            const dB = Math.hypot(mx - gt.tx, my - gt.tyB);
            const dCar = Math.hypot((mx - gt.carX) * 0.7, my - gt.carY);
            if (dA <= dB && dA < 9.0 * s) {
              hitA = true;
              if (idA < 0) idA = tid;
              else if (tid !== idA) { swaps++; idA = tid; }
            } else if (dB < dA && dB < 9.0 * s) {
              hitB = true;
              if (idB < 0) idB = tid;
              else if (tid !== idB) { swaps++; idB = tid; }
            } else if (dCar < 17.0 * s) {
              carHits++;
              if (isBridged) carBridges++;
              if (gt.carX > 186 * s && idCarPre < 0) idCarPre = tid;
              if (gt.carX < 132 * s) idCarPost = tid;
              if (idCar < 0) idCar = tid;
              else if (tid !== idCar) { swaps++; idCar = tid; }
            } else {
              fp++;
              if (gt.carX < 126 * s && mx >= (signX0 - 8 * s) && mx <= (signX1 + 8 * s) && my >= (signY0 - 8 * s) && my <= (signY1 + 8 * s)) {
                signPins++;
              }
            }
          }
        }
        if (hitA) lockA++;
        if (hitB) lockB++;
        if (hitA && hitB) lockBoth++;
        if (carHits >= 1) lockCar++;
        if (carHits > 1) carDupFrames++;
      }
    }
    if (idA > 0 && idA === idB) swaps++;
    if (idCarPre < 0 || idCarPost < 0 || idCarPre !== idCarPost) swaps++;
    return {
      bothPct: lockBoth / Math.max(1, evalFrames),
      carPct: lockCar / Math.max(1, evalFrames),
      swaps,
      carDupFrames,
      carBridges,
      signPins,
      fp,
    };
  };

  const twinSd = runTwinValleyAndRigidBodyGate(0);
  const twinHd = runTwinValleyAndRigidBodyGate(1);
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  prevFiedlerTrack.fill(0.0);

  const gateLOk = (r) =>
    r.bothPct >= 0.90 &&
    r.carPct >= 0.90 &&
    r.swaps === 0 &&
    r.carDupFrames <= 2 &&
    r.carBridges >= 2 &&
    r.signPins === 0 &&
    r.fp <= 2;

  if (!gateLOk(twinSd) || !gateLOk(twinHd)) {
    throw new Error(
      `Gate L (Valley Connectivity, Gantry Sign Bridge & Rigid-Body Custody): ` +
      `SD(both=${(100*twinSd.bothPct).toFixed(1)}%, car=${(100*twinSd.carPct).toFixed(1)}%, br=${twinSd.carBridges}, pins=${twinSd.signPins}, sw=${twinSd.swaps}, dup=${twinSd.carDupFrames}, fp=${twinSd.fp}) ` +
      `HD(both=${(100*twinHd.bothPct).toFixed(1)}%, car=${(100*twinHd.carPct).toFixed(1)}%, br=${twinHd.carBridges}, pins=${twinHd.signPins}, sw=${twinHd.swaps}, dup=${twinHd.carDupFrames}, fp=${twinHd.fp})`
    );
  }
  console.log(
    `  [PASS] Gate L (Valley Connectivity, Gantry Sign Bridge & Rigid-Body Custody): ` +
    `SD(twins=${(100*twinSd.bothPct).toFixed(0)}% car=${(100*twinSd.carPct).toFixed(0)}% br=${twinSd.carBridges} pins=0 dup=${twinSd.carDupFrames} sw=0) | ` +
    `HD(twins=${(100*twinHd.bothPct).toFixed(0)}% car=${(100*twinHd.carPct).toFixed(0)}% br=${twinHd.carBridges} pins=0 dup=${twinHd.carDupFrames} sw=0)`
  );
}

{
  const runOpticalExposurePanGate = (resMode) => {
    exp.chronos_init();
    exp.chronos_set_resolution(resMode);
    prevFiedlerTrack.fill(0.0);
    const w = exp.chronos_get_frame_w();
    const h = exp.chronos_get_frame_h();
    const s = w / 320.0;
    const fBytes = w * h * 4;
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    const rViews = Array.from({ length: 12 }, (_, slot) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), fBytes));

    let camX = 0.0, camY = 0.0;
    let minClutter = 999.0, minEvalRatio = 1.0, panQualFrames = 0;
    let hitACnt = 0, totA = 0, hitBCnt = 0, totB = 0;
    let idA = -1, idB = -1, swaps = 0, bgFp = 0;

    for (let f = 0; f < 90; f++) {
      const vxCam = f >= 25 ? 1.35 : 0.0;
      const vyCam = f >= 25 ? 0.35 * Math.cos(f * 0.2) : 0.0;
      camX += vxCam;
      camY += vyCam;
      const gain = f >= 25 ? 1.0 + 0.22 * Math.sin((f - 25) * 0.09) : 1.0;
      const ped = f >= 25 ? 10.0 * Math.cos((f - 25) * 0.11) : 0.0;
      const discA = { wx: 45.0 + f * 1.6, wy: 74.0 };
      const hasB = f >= 34;
      const discB = { wx: camX + 88.0 + Math.max(0, f - 34) * 1.65, wy: camY + 122.0 };

      const out = rViews[f % 12];
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let acc = 0.0;
          for (const tau of [-0.35, 0.0, 0.35]) {
            const wx = x / s + camX + tau * vxCam;
            const wy = y / s + camY + tau * vyCam;
            acc += 88.0 + Math.sin(wx * 0.17) * Math.cos(wy * 0.19) * 38.0 + Math.sin((wx + wy) * 0.07) * 22.0;
          }
          const base = Math.max(6, Math.min(245, Math.round((acc / 3.0) * gain + ped)));
          const idx = (y * w + x) * 4;
          out[idx] = base;
          out[idx + 1] = Math.min(255, base + 5);
          out[idx + 2] = Math.min(255, base + 10);
          out[idx + 3] = 255;
        }
      }

      const drawDisc = (sx, sy, cr, cg, cb) => {
        const rad = 7 * s, r2m = 49 * s * s;
        for (let py = Math.max(0, Math.floor(sy - rad)); py <= Math.min(h - 1, Math.ceil(sy + rad)); py++) {
          for (let px = Math.max(0, Math.floor(sx - rad)); px <= Math.min(w - 1, Math.ceil(sx + rad)); px++) {
            if ((px - sx) ** 2 + (py - sy) ** 2 <= r2m) {
              const idx = (py * w + px) * 4;
              out[idx] = cr; out[idx + 1] = cg; out[idx + 2] = cb; out[idx + 3] = 255;
            }
          }
        }
      };

      const sxA = (discA.wx - camX) * s, syA = (discA.wy - camY) * s;
      drawDisc(sxA, syA, 245, 60, 65);
      const sxB = (discB.wx - camX) * s, syB = (discB.wy - camY) * s;
      if (hasB) drawDisc(sxB, syB, 55, 225, 235);

      exp.chronos_pre_tick(1 / 60);
      solveShiftedLanczosFiedler();
      exp.chronos_post_tick(1 / 60);

      if (f >= 32 && telem[22] > 0.5) {
        panQualFrames++;
        minClutter = Math.min(minClutter, telem[4]);
        minEvalRatio = Math.min(minEvalRatio, telem[23]);
      }

      let hitA = false, hitB = false;
      for (let m = 0; m < MAX_TARGETS; m++) {
        const base = 32 + m * 56;
        if (telem[base] > 0.5) {
          const tid = Math.round(telem[base + 1]);
          const tx = telem[base + 8], ty = telem[base + 9];
          const dA = Math.hypot(tx - sxA, ty - syA);
          const dB = hasB ? Math.hypot(tx - sxB, ty - syB) : 1e9;
          if (dA <= dB && dA < 18.0 * s) {
            hitA = true;
            if (idA < 0) idA = tid;
            else if (tid !== idA) { swaps++; idA = tid; }
          } else if (hasB && dB < 18.0 * s) {
            hitB = true;
            if (idB < 0) idB = tid;
            else if (tid !== idB) { swaps++; idB = tid; }
          } else if (f >= 32 && dA > 24.0 * s && (!hasB || dB > 24.0 * s)) {
            bgFp++;
          }
        }
      }
      if (f >= 32) { totA++; if (hitA) hitACnt++; }
      if (f >= 44) { totB++; if (hitB) hitBCnt++; }
    }
    if (idA > 0 && idA === idB) swaps++;
    return {
      panQualFrames,
      minClutter,
      minEvalRatio,
      recA: hitACnt / Math.max(1, totA),
      recB: hitBCnt / Math.max(1, totB),
      swaps,
      bgFp,
    };
  };

  const optSd = runOpticalExposurePanGate(0);
  const optHd = runOpticalExposurePanGate(1);
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  prevFiedlerTrack.fill(0.0);

  const gateMOk = (r) =>
    r.panQualFrames >= 45 &&
    r.minClutter >= 43.5 &&
    r.minEvalRatio >= 0.85 &&
    r.recA >= 0.95 &&
    r.recB >= 0.92 &&
    r.swaps === 0 &&
    r.bgFp === 0;

  if (!gateMOk(optSd) || !gateMOk(optHd)) {
    throw new Error(
      `Gate M (Exposure- & Blur-Compensated Sub-Pixel Camera Pan): ` +
      `SD(clut=${optSd.minClutter.toFixed(2)}dB, eval=${(100*optSd.minEvalRatio).toFixed(1)}%, A=${(100*optSd.recA).toFixed(1)}%, B=${(100*optSd.recB).toFixed(1)}%, sw=${optSd.swaps}, fp=${optSd.bgFp}) ` +
      `HD(clut=${optHd.minClutter.toFixed(2)}dB, eval=${(100*optHd.minEvalRatio).toFixed(1)}%, A=${(100*optHd.recA).toFixed(1)}%, B=${(100*optHd.recB).toFixed(1)}%, sw=${optHd.swaps}, fp=${optHd.bgFp})`
    );
  }
  console.log(
    `  [PASS] Gate M (Exposure- & Blur-Compensated Optical Pan): ` +
    `SD=${optSd.minClutter.toFixed(1)}dB (eval=${(100*optSd.minEvalRatio).toFixed(0)}%) A=${(100*optSd.recA).toFixed(0)}% B=${(100*optSd.recB).toFixed(0)}% sw=0 fp=0 | ` +
    `HD=${optHd.minClutter.toFixed(1)}dB (eval=${(100*optHd.minEvalRatio).toFixed(0)}%) A=${(100*optHd.recA).toFixed(0)}% B=${(100*optHd.recB).toFixed(0)}% sw=0 fp=0`
  );
}

{
  const runStationaryExpAndPanRampGate = (resMode) => {
    exp.chronos_init();
    exp.chronos_set_resolution(resMode);
    prevFiedlerTrack.fill(0.0);
    const w = exp.chronos_get_frame_w();
    const h = exp.chronos_get_frame_h();
    const s = w / 320.0;
    const fBytes = w * h * 4;
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    const rViews = Array.from({ length: 12 }, (_, slot) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), fBytes));

    let camX = 0.0, camY = 0.0;
    let minClutter = 999.0, minEvalRatio = 1.0, qualFrames = 0;
    let hitCnt = 0, totCnt = 0, idMain = -1, swaps = 0, bgFp = 0;
    let statExpPanTrig = 0;

    for (let f = 0; f < 105; f++) {
      const rampFrac = f < 50 ? 0.0 : Math.min(1.0, (f - 50) / 35.0);
      const vxCam = 1.45 * rampFrac;
      const vyCam = 0.85 * rampFrac;
      camX += vxCam;
      camY += vyCam;
      const gain = f < 20 ? 1.0 : 1.0 + 0.30 * Math.sin((f - 20) * 0.11);
      const ped = f < 20 ? 0.0 : 12.0 * Math.sin((f - 20) * 0.09);
      const wxT = 40.0 + f * 1.55;
      const wyT = 45.0 + f * 0.90;

      const out = rViews[f % 12];
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let acc = 0.0;
          for (const tau of [-0.35, 0.0, 0.35]) {
            const wx = x / s + camX + tau * vxCam;
            const wy = y / s + camY + tau * vyCam;
            acc += 92.0 + Math.sin(wx * 0.19) * Math.cos(wy * 0.23) * 34.0 + Math.cos((wx + wy) * 0.09) * 20.0 + Math.sin((wx - wy) * 0.31) * 12.0;
          }
          const qStep = Math.round((acc / 3.0) * gain + ped);
          const base = Math.max(6, Math.min(246, qStep));
          const idx = (y * w + x) * 4;
          out[idx] = base;
          out[idx + 1] = Math.min(255, base + 4);
          out[idx + 2] = Math.min(255, base + 8);
          out[idx + 3] = 255;
        }
      }

      const sxT = (wxT - camX) * s, syT = (wyT - camY) * s;
      const rad = 7 * s, r2m = 49 * s * s;
      for (let py = Math.max(0, Math.floor(syT - rad)); py <= Math.min(h - 1, Math.ceil(syT + rad)); py++) {
        for (let px = Math.max(0, Math.floor(sxT - rad)); px <= Math.min(w - 1, Math.ceil(sxT + rad)); px++) {
          if ((px - sxT) ** 2 + (py - syT) ** 2 <= r2m) {
            const idx = (py * w + px) * 4;
            out[idx] = 240; out[idx + 1] = 65; out[idx + 2] = 65; out[idx + 3] = 255;
          }
        }
      }

      exp.chronos_pre_tick(1 / 30);
      solveShiftedLanczosFiedler();
      exp.chronos_post_tick(1 / 30);

      if (f >= 20 && f < 50 && Math.round(telem[1257]) !== 0) statExpPanTrig++;
      if (f >= 24 && telem[22] > 0.5) {
        qualFrames++;
        minClutter = Math.min(minClutter, telem[4]);
        minEvalRatio = Math.min(minEvalRatio, telem[23]);
      }

      let hit = false;
      for (let m = 0; m < MAX_TARGETS; m++) {
        const base = 32 + m * 56;
        if (telem[base] > 0.5) {
          const tid = Math.round(telem[base + 1]);
          const tx = telem[base + 8], ty = telem[base + 9];
          const d = Math.hypot(tx - sxT, ty - syT);
          if (d < 18.0 * s) {
            hit = true;
            if (idMain < 0) idMain = tid;
            else if (tid !== idMain) { swaps++; idMain = tid; }
          } else if (f >= 24 && d > 24.0 * s) {
            bgFp++;
          }
        }
      }
      if (f >= 24) { totCnt++; if (hit) hitCnt++; }
    }
    return { qualFrames, minClutter, minEvalRatio, rec: hitCnt / Math.max(1, totCnt), swaps, bgFp, statExpPanTrig };
  };

  const n1Sd = runStationaryExpAndPanRampGate(0);
  const n1Hd = runStationaryExpAndPanRampGate(1);
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  prevFiedlerTrack.fill(0.0);

  const gateN1Ok = (r) =>
    r.qualFrames >= 75 &&
    r.statExpPanTrig === 0 &&
    r.minClutter >= 42.0 &&
    r.minEvalRatio >= 0.70 &&
    r.rec >= 0.90 &&
    r.swaps === 0 &&
    r.bgFp <= 2;

  if (!gateN1Ok(n1Sd) || !gateN1Ok(n1Hd)) {
    throw new Error(
      `Gate N1 (Stationary Exposure + 0->1.45px/f Urban Pan Ramp): ` +
      `SD(clut=${n1Sd.minClutter.toFixed(2)}dB, eval=${(100*n1Sd.minEvalRatio).toFixed(1)}%, rec=${(100*n1Sd.rec).toFixed(1)}%, sw=${n1Sd.swaps}, fp=${n1Sd.bgFp}, panTrig=${n1Sd.statExpPanTrig}) ` +
      `HD(clut=${n1Hd.minClutter.toFixed(2)}dB, eval=${(100*n1Hd.minEvalRatio).toFixed(1)}%, rec=${(100*n1Hd.rec).toFixed(1)}%, sw=${n1Hd.swaps}, fp=${n1Hd.bgFp}, panTrig=${n1Hd.statExpPanTrig})`
    );
  }
  console.log(
    `  [PASS] Gate N1 (Stationary Exposure + 0->1.45px/f Urban Pan Ramp): ` +
    `SD=${n1Sd.minClutter.toFixed(1)}dB (eval=${(100*n1Sd.minEvalRatio).toFixed(0)}%) rec=${(100*n1Sd.rec).toFixed(0)}% sw=0 | ` +
    `HD=${n1Hd.minClutter.toFixed(1)}dB (eval=${(100*n1Hd.minEvalRatio).toFixed(0)}%) rec=${(100*n1Hd.rec).toFixed(0)}% sw=0`
  );
}

{
  const runRotatedOcclusionGate = (resMode, angleDeg) => {
    exp.chronos_init();
    exp.chronos_set_resolution(resMode);
    prevFiedlerTrack.fill(0.0);
    const w = exp.chronos_get_frame_w();
    const h = exp.chronos_get_frame_h();
    const s = w / 320.0;
    const fBytes = w * h * 4;
    const effK = 8;
    exp.chronos_set_params(effK, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    exp.chronos_set_spectral_weight(25.0);
    const rViews = Array.from({ length: 12 }, (_, slot) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), fBytes));

    const cx = (w - 1) * 0.5;
    const cy = (h - 1) * 0.5;
    const rad = (angleDeg * Math.PI) / 180.0;
    const ca = Math.cos(rad);
    const sa = Math.sin(rad);

    const rotPoint = (ux, uy) => {
      const dx = ux - cx;
      const dy = uy - cy;
      return { x: cx + dx * ca - dy * sa, y: cy + dx * sa + dy * ca };
    };

    const bgRgba = new Uint8Array(fBytes);
    const roadY0 = 48 * s;
    const roadY1 = 132 * s;
    const divY = 90 * s;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const ux = cx + dx * ca + dy * sa;
        const uy = cy - dx * sa + dy * ca;
        const idx = (y * w + x) * 4;
        if (ux >= 150 * s && ux <= 170 * s && uy >= 38 * s && uy <= 142 * s) {
          const panel = ((Math.floor(ux / s) + Math.floor(uy / s)) % 7 === 0) ? 10 : 0;
          bgRgba[idx] = 36 + panel;
          bgRgba[idx + 1] = 42 + panel;
          bgRgba[idx + 2] = 50 + panel;
          bgRgba[idx + 3] = 255;
        } else if (uy >= roadY0 && uy <= roadY1) {
          const grain = Math.round(6 * Math.sin(ux * 0.35 / s) * Math.cos(uy * 0.35 / s));
          const dash = Math.abs(uy - divY) <= 1.2 * s && (Math.floor(ux / (12 * s)) & 1) === 0;
          if (dash) {
            bgRgba[idx] = 175; bgRgba[idx + 1] = 165; bgRgba[idx + 2] = 120;
          } else {
            bgRgba[idx] = 48 + grain; bgRgba[idx + 1] = 50 + grain; bgRgba[idx + 2] = 53 + grain;
          }
          bgRgba[idx + 3] = 255;
        } else {
          const tex = Math.round(14 * Math.sin(ux * 0.14 / s) + 12 * Math.cos(uy * 0.17 / s));
          bgRgba[idx] = Math.max(12, Math.min(220, 62 + tex));
          bgRgba[idx + 1] = Math.max(12, Math.min(220, 88 + tex));
          bgRgba[idx + 2] = Math.max(12, Math.min(220, 56 + tex));
          bgRgba[idx + 3] = 255;
        }
      }
    }

    const drawCar = (out, uxC, uyC, rgb) => {
      const p = rotPoint(uxC, uyC);
      const radPx = Math.ceil(7 * s);
      const x0 = Math.max(0, Math.floor(p.x - radPx));
      const x1 = Math.min(w - 1, Math.ceil(p.x + radPx));
      const y0 = Math.max(0, Math.floor(p.y - radPx));
      const y1 = Math.min(h - 1, Math.ceil(p.y + radPx));
      const r2Max = 38.0 * s * s;
      for (let py = y0; py <= y1; py++) {
        for (let px = x0; px <= x1; px++) {
          const dx = px - p.x;
          const dy = py - p.y;
          const ldx = dx * ca + dy * sa;
          const ldy = -dx * sa + dy * ca;
          const ux = uxC + ldx;
          if (ux >= 150 * s && ux <= 170 * s && uyC >= 38 * s && uyC <= 142 * s) continue;
          if (ldx * ldx * 0.72 + ldy * ldy * 1.15 <= r2Max) {
            const idx = (py * w + px) * 4;
            const shade = ldx > 2 * s ? 0.78 : 1.0;
            out[idx] = Math.round(rgb[0] * shade);
            out[idx + 1] = Math.round(rgb[1] * shade);
            out[idx + 2] = Math.round(rgb[2] * shade);
            out[idx + 3] = 255;
          }
        }
      }
    };

    let hitA = 0, totA = 0, hitB = 0, totB = 0;
    let idA = -1, idB = -1, swaps = 0, fp = 0, bridgeFrames = 0;

    for (let f = 0; f < 116; f++) {
      const out = rViews[f % 12];
      out.set(bgRgba);
      const uxA = (90.0 + 1.22 * f) * s;
      const uyA = 66.0 * s;
      const uxB = (230.0 - 1.22 * f) * s;
      const uyB = 114.0 * s;
      drawCar(out, uxA, uyA, [235, 72, 62]);
      drawCar(out, uxB, uyB, [230, 175, 48]);

      exp.chronos_set_ingest_pts(f / 30.0);
      exp.chronos_pre_tick(1 / 30);
      solveShiftedLanczosFiedler();
      exp.chronos_post_tick(1 / 30);

      const fd = f - effK;
      if (fd < 16) continue;
      const uxAd = (90.0 + 1.22 * fd) * s;
      const uyAd = 66.0 * s;
      const uxBd = (230.0 - 1.22 * fd) * s;
      const uyBd = 114.0 * s;
      const gA = rotPoint(uxAd, uyAd);
      const gB = rotPoint(uxBd, uyBd);
      const visA = uxAd < 146 * s || uxAd > 174 * s;
      const visB = uxBd < 146 * s || uxBd > 174 * s;

      let bestA = { d: 18.0 * s, tid: -1 };
      let bestB = { d: 18.0 * s, tid: -1 };
      for (let m = 0; m < MAX_TARGETS; m++) {
        const base = 32 + m * 56;
        if (telem[base] <= 0.5) continue;
        const tid = Math.round(telem[base + 1]);
        const tx = telem[base + 2], ty = telem[base + 3];
        const isBridged = telem[base + 11] > 0.5;
        const dA = Math.hypot(tx - gA.x, ty - gA.y);
        const dB = Math.hypot(tx - gB.x, ty - gB.y);
        if (dA <= dB && dA < bestA.d) {
          bestA = { d: dA, tid };
          if (isBridged) bridgeFrames++;
        } else if (dB < dA && dB < bestB.d) {
          bestB = { d: dB, tid };
          if (isBridged) bridgeFrames++;
        } else if (fd >= 18 && dA > 26.0 * s && dB > 26.0 * s) {
          fp++;
        }
      }
      if (visA) {
        totA++;
        if (bestA.tid >= 0) {
          hitA++;
          if (idA < 0) idA = bestA.tid;
          else if (bestA.tid !== idA) { swaps++; idA = bestA.tid; }
        }
      }
      if (visB) {
        totB++;
        if (bestB.tid >= 0) {
          hitB++;
          if (idB < 0) idB = bestB.tid;
          else if (bestB.tid !== idB) { swaps++; idB = bestB.tid; }
        }
      }
    }
    if (idA > 0 && idA === idB) swaps++;

    return {
      resMode,
      angleDeg,
      recA: hitA / Math.max(1, totA),
      recB: hitB / Math.max(1, totB),
      bridges: bridgeFrames,
      swaps,
      fp,
    };
  };

  const n2Runs = [
    runRotatedOcclusionGate(0, 68),
    runRotatedOcclusionGate(0, 180),
    runRotatedOcclusionGate(1, 68),
    runRotatedOcclusionGate(1, 180),
  ];
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  prevFiedlerTrack.fill(0.0);

  const gateN2Ok = (r) => r.recA >= 0.90 && r.recB >= 0.90 && r.bridges >= 2 && r.swaps === 0 && r.fp <= 2;
  if (!n2Runs.every(gateN2Ok)) {
    const desc = n2Runs.map((r) => `${r.resMode ? 'HD' : 'SD'}@${r.angleDeg}deg(A=${(100*r.recA).toFixed(1)}%,B=${(100*r.recB).toFixed(1)}%,br=${r.bridges},sw=${r.swaps},fp=${r.fp})`).join(' ');
    throw new Error(`Gate N2 (SO(2) Arbitrary-Angle 68° & 180° Multi-Lane Occlusion Invariance): ${desc}`);
  }
  const n2Summary = n2Runs.map((r) => `${r.resMode ? 'HD' : 'SD'}@${r.angleDeg}°(A=${(100*r.recA).toFixed(0)}%,B=${(100*r.recB).toFixed(0)}%,br=${r.bridges})`).join(' ');
  console.log(`  [PASS] Gate N2 (SO(2) Arbitrary-Angle 68° & 180° Multi-Lane Occlusion Invariance): ${n2Summary} sw=0`);
}

// ----------------------------------------------------------------------------
// Gate O: 16-Slot Multi-Target Swarm Lock + 32x32/64x64 Border-Clamped PiPs + 160f Dormant Re-ID
// ----------------------------------------------------------------------------
{
  const ringO = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), BYTES));
  const bufO = new Uint8Array(BYTES);

  // O1a: 12 simultaneous moving targets across 3 rows x 4 cols in 320x180 SD (>= 10 confirmed locks)
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  prevFiedlerTrack.fill(0.0);
  const swarmPalette = [
    [60, 220, 140, 220, 255, 225],
    [230, 110, 80, 255, 215, 190],
    [80, 175, 240, 215, 240, 255],
    [235, 195, 70, 255, 245, 200],
    [210, 95, 225, 250, 210, 255],
    [95, 230, 215, 210, 255, 245],
    [240, 145, 70, 255, 225, 190],
    [150, 230, 85, 230, 255, 200],
    [130, 145, 245, 220, 225, 255],
    [240, 95, 165, 255, 205, 235],
    [205, 215, 95, 245, 250, 205],
    [95, 205, 185, 205, 250, 240],
  ];
  for (let f = 0; f < 28; f++) {
    bufO.set(staticBg);
    let idx = 0;
    for (let row = 0; row < 3; row++) {
      const cy = 32 + row * 58;
      for (let col = 0; col < 4; col++) {
        const cx = 26 + col * 68 + f * 1.15;
        const p = swarmPalette[idx++];
        stampDisc(bufO, cx, cy, p[0], p[1], p[2], p[3], p[4], p[5]);
      }
    }
    ringO[f % 12].set(bufO);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
  }
  let locked12 = 0;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telem[base] >= 0.5 && telem[base + 44 + 11] >= 0.5) locked12++;
  }
  if (locked12 < 10) {
    throw new Error(`Gate O1a (12-Disc Swarm) failed: expected >= 10 simultaneous locks (>8 legacy limit), got ${locked12}`);
  }
  if (exp.chronos_get_pip_dim() !== 32 || pipBuf[4095] !== 255) {
    throw new Error(`Gate O1a: Expected 32x32 SD PiP with alpha=255, got dim=${exp.chronos_get_pip_dim()}, alpha=${pipBuf[4095]}`);
  }

  // O1b: Border-clamped 32x32 SD & 64x64 HD PiPs near top-left corner
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  prevFiedlerTrack.fill(0.0);
  for (let f = 0; f < 16; f++) {
    bufO.set(staticBg);
    stampDisc(bufO, 10 + f * 1.0, 10, 64, 225, 160, 230, 255, 240);
    ringO[f % 12].set(bufO);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
  }
  let sdBorderActive = 0;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telem[base] >= 0.5 && telem[base + 44 + 11] >= 0.5) sdBorderActive++;
  }
  if (sdBorderActive < 1 || exp.chronos_get_pip_dim() !== 32 || pipBuf[4095] !== 255) {
    throw new Error(`Gate O1b (SD border 32x32 PiP) failed: active=${sdBorderActive}, dim=${exp.chronos_get_pip_dim()}, alpha=${pipBuf[4095]}`);
  }

  const hdBytesO = 640 * 360 * 4;
  const hdRingO = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), hdBytesO));
  const hdBgO = new Uint8Array(hdBytesO);
  const hdBufO = new Uint8Array(hdBytesO);
  for (let y = 0; y < 360; y++) {
    const sy = y >> 1;
    for (let x = 0; x < 640; x++) {
      const si = (sy * 320 + (x >> 1)) * 4;
      const di = (y * 640 + x) * 4;
      hdBgO[di] = staticBg[si];
      hdBgO[di + 1] = staticBg[si + 1];
      hdBgO[di + 2] = staticBg[si + 2];
      hdBgO[di + 3] = 255;
    }
  }
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  prevFiedlerTrack.fill(0.0);
  for (let f = 0; f < 16; f++) {
    hdBufO.set(hdBgO);
    stampDiscScaled(hdBufO, 640, 360, 2, 20 + f * 2.0, 20, 64, 225, 160, 230, 255, 240);
    hdRingO[f % 12].set(hdBufO);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
  }
  let hdBorderActive = 0;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telem[base] >= 0.5 && telem[base + 44 + 11] >= 0.5) hdBorderActive++;
  }
  if (hdBorderActive < 1 || exp.chronos_get_pip_dim() !== 64 || pipBuf[16383] !== 255) {
    throw new Error(`Gate O1b (HD border 64x64 PiP) failed: active=${hdBorderActive}, dim=${exp.chronos_get_pip_dim()}, alpha=${pipBuf[16383]}`);
  }
  exp.chronos_set_resolution(0);

  // O2: 160-frame Dormant Re-ID (> COAST_MAX=90/48, <= DORMANT_MAX_AGE=600) preserves original track_id
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  prevFiedlerTrack.fill(0.0);
  for (let f = 0; f < 16; f++) {
    ringO[f % 12].set(staticBg);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
  }
  let tid0 = 0;
  for (let f = 16; f < 51; f++) {
    bufO.set(staticBg);
    stampDisc(bufO, 40 + (f - 16) * 1.4, 90, 48, 224, 92, 236, 252, 168);
    ringO[f % 12].set(bufO);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
    if (f === 50) {
      for (let m = 0; m < MAX_TARGETS; m++) {
        const base = 32 + m * 56;
        if (telem[base] >= 0.5 && telem[base + 44 + 11] >= 0.5) {
          tid0 = Math.round(telem[base + 1]);
          break;
        }
      }
    }
  }
  if (tid0 <= 0) {
    throw new Error(`Gate O2 (Dormant Re-ID) failed: initial track failed to confirm (tid0=${tid0})`);
  }

  for (let f = 51; f < 211; f++) {
    ringO[f % 12].set(staticBg);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
  }
  let activeDuringGap = 0;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telem[base] >= 0.5 && telem[base + 44 + 11] >= 0.5) activeDuringGap++;
  }
  if (activeDuringGap !== 0) {
    throw new Error(`Gate O2 (Dormant Re-ID) failed: expected active confirmed count=0 after 160 empty frames, got ${activeDuringGap}`);
  }

  let reacqTid = 0;
  for (let f = 211; f < 241; f++) {
    bufO.set(staticBg);
    stampDisc(bufO, 166 + (f - 211) * 1.4, 90, 48, 224, 92, 236, 252, 168);
    ringO[f % 12].set(bufO);
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56;
      if (telem[base] >= 0.5 && telem[base + 44 + 11] >= 0.5) {
        reacqTid = Math.round(telem[base + 1]);
      }
    }
  }
  const reidEvents = Math.round(telem[1266]);
  if (reacqTid !== tid0 || reidEvents < 1) {
    throw new Error(`Gate O2 (Dormant Re-ID) failed: expected recovered tid0=${tid0} with reidEvents>=1 after 160f absence, got reacqTid=${reacqTid}, reidEvents=${reidEvents}`);
  }
  console.log(`  [PASS] Gate O (16-Slot Swarm Lock + 32x32/64x64 Clamped PiPs + 160f Dormant Re-ID): locked=${locked12}/12, reID=T${tid0}->T${reacqTid} (events=${reidEvents})`);
}

// ----------------------------------------------------------------------------
// Gate P (P1..P5): Wave-3 Physics Discovery, Collision, Kuramoto, Chaos & SD/HD ABI Audit
// ----------------------------------------------------------------------------
{
  const hdBytesP = 640 * 360 * 4;
  const ringSdP = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), BYTES));
  const ringHdP = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), hdBytesP));
  const hdBgP = new Uint8Array(hdBytesP);
  for (let y = 0; y < 360; y++) {
    const sy = y >> 1;
    for (let x = 0; x < 640; x++) {
      const si = (sy * 320 + (x >> 1)) * 4;
      const di = (y * 640 + x) * 4;
      hdBgP[di] = staticBg[si];
      hdBgP[di + 1] = staticBg[si + 1];
      hdBgP[di + 2] = staticBg[si + 2];
      hdBgP[di + 3] = 255;
    }
  }

  function stampDiscSubP(buf, w, h, cx, cy, radius, r0, g0, b0, r1, g1, b1) {
    const x0 = Math.max(0, Math.floor(cx - radius - 1.5));
    const x1 = Math.min(w - 1, Math.ceil(cx + radius + 1.5));
    const y0 = Math.max(0, Math.floor(cy - radius - 1.5));
    const y1 = Math.min(h - 1, Math.ceil(cy + radius + 1.5));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.hypot(x - cx, y - cy);
        if (d >= radius + 1.0) continue;
        const cov = d <= radius - 0.8 ? 1.0 : Math.max(0.0, Math.min(1.0, (radius + 0.8 - d) / 1.6));
        const core = d < radius * 0.48 ? 1.0 : 0.0;
        const dr = core ? r1 : r0;
        const dg = core ? g1 : g0;
        const db = core ? b1 : b0;
        const idx = (y * w + x) * 4;
        buf[idx]     = Math.round(buf[idx]     * (1.0 - cov) + dr * cov);
        buf[idx + 1] = Math.round(buf[idx + 1] * (1.0 - cov) + dg * cov);
        buf[idx + 2] = Math.round(buf[idx + 2] * (1.0 - cov) + db * cov);
        buf[idx + 3] = 255;
      }
    }
  }

  function checkAbiP(label, f) {
    if (telem.length < TELEMETRY_LEN) {
      throw new Error(`Gate P1 (${label}): telemetry len=${telem.length} < ${TELEMETRY_LEN}`);
    }
    if (mem.buffer.byteLength !== 448 * 65536) {
      throw new Error(`Gate P1 (${label}): Wasm memory bytes=${mem.buffer.byteLength} != ${448 * 65536}`);
    }
    const arenaBytes = telem[20];
    if (!Number.isFinite(arenaBytes) || arenaBytes <= 0 || arenaBytes > 24_117_248) {
      throw new Error(`Gate P1 (${label}): Arena size telem[20]=${arenaBytes} exceeds 24_117_248`);
    }
    const w1TelemPtr = typeof exp.chronos_get_wave1_telemetry_ptr === 'function' ? exp.chronos_get_wave1_telemetry_ptr() : 0;
    const w1LinkedEnd = w1TelemPtr > 0 ? (new Uint32Array(mem.buffer, w1TelemPtr, 192)[12] >>> 0) : 0;
    const staticDataEnd = Math.max(exp.chronos_get_ring_slot_ptr(0) + arenaBytes, w1LinkedEnd);
    if (!Number.isFinite(staticDataEnd) || staticDataEnd > 29_294_592) {
      throw new Error(`Gate P1 (${label}): Static data end=${staticDataEnd} exceeds 29_294_592 (< 65536 B reserve)`);
    }
    if (staticDataEnd > 24_248_320) {
      throw new Error(`Gate P1 (${label}): Wave-2 static end=${staticDataEnd} exceeds 24_248_320 (headroom breach)`);
    }
    for (let k = 0; k < TELEMETRY_LEN; k++) {
      if (!Number.isFinite(telem[k])) {
        throw new Error(`Gate P1 (${label}): Non-finite telemetry[${k}]=${telem[k]} at f=${f}`);
      }
      if (k >= 1550 && telem[k] !== 0.0) {
        throw new Error(`Gate P1 (${label}): Reserved telemetry[${k}]=${telem[k]} != 0.0 at f=${f}`);
      }
    }
  }

  function stepFrameP(buf, ring, f, dtSec, label) {
    ring[f % 12].set(buf);
    exp.chronos_pre_tick(dtSec);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(dtSec);
    checkAbiP(label, f);
  }

  function readActiveTargetsP() {
    const out = [];
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56;
      if (telem[base] < 0.5 || telem[base + 44 + 11] < 0.5) continue;
      const pb = TELEM_PHYSICS_START + m * 16;
      out.push({
        m,
        tid: Math.round(telem[base + 1]),
        x: telem[base + 2],
        y: telem[base + 3],
        imm_cv: telem[pb + 0],
        imm_bal: telem[pb + 1],
        imm_osc: telem[pb + 2],
        knots: telem[pb + 3],
        regime: Math.round(telem[pb + 4]),
        conf: telem[pb + 5],
        r2: telem[pb + 6],
        beta: telem[pb + 7],
        thrust: telem[pb + 8],
        ell_T: telem[pb + 9],
        zeta: telem[pb + 10],
        q: telem[pb + 11],
        E_spec: telem[pb + 12],
        L_spec: telem[pb + 13],
        e_n: telem[pb + 14],
        jump: Math.round(telem[pb + 15]),
      });
    }
    return out;
  }

  function agm(a, b) {
    let x = a, y = b;
    for (let i = 0; i < 12; i++) {
      const nx = 0.5 * (x + y);
      const ny = Math.sqrt(x * y);
      x = nx; y = ny;
    }
    return x;
  }

  // P2a: Large-angle damped pendulum RK4 (theta0=0.92 rad, zeta=0.035, g=400, L=54)
  let pendMatch = null;
  let maxLyapPend = 0.0;
  let p2aT0 = 0.0, p2aTagm = 0.0;
  {
    const pivX = 160.0, pivY = 44.0, pendL = 54.0, grav = 400.0;
    const omega0 = Math.sqrt(grav / pendL);
    const zetaTrue = 0.035;
    const theta0 = 0.92;
    p2aT0 = (2.0 * Math.PI) / omega0;
    p2aTagm = p2aT0 / agm(1.0, Math.cos(theta0 * 0.5));
    const dtFrame = 1.0 / 30.0;
    const sub = 6;
    const h = dtFrame / sub;
    let th = theta0, w = 0.0;
    function accP(angle, angVel) {
      return -(grav / pendL) * Math.sin(angle) - 2.0 * zetaTrue * omega0 * angVel;
    }

    exp.chronos_init();
    exp.chronos_set_resolution(0);
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    prevFiedlerTrack.fill(0.0);
    const buf = new Uint8Array(BYTES);

    for (let f = 0; f < 210; f++) {
      const cx = pivX + pendL * Math.sin(th);
      const cy = pivY + pendL * Math.cos(th);
      buf.set(staticBg);
      stampDiscSubP(buf, 320, 180, cx, cy, 5.8, 250, 95, 65, 255, 225, 185);
      stepFrameP(buf, ringSdP, f, dtFrame, 'P2a-pendulum');

      const ly = telem[TELEM_ENSEMBLE_START + 11];
      if (ly > maxLyapPend) maxLyapPend = ly;
      if (f >= 90) {
        const ts = readActiveTargetsP();
        const tp = ts.find((t) => t.regime === 5 && t.ell_T > 0 && t.zeta > 0);
        if (tp) {
          pendMatch = {
            ...tp,
            f,
            taubinDelta: telem[TELEM_ENSEMBLE_START + 12],
            taubinCurv: telem[TELEM_ENSEMBLE_START + 13],
          };
        }
      }

      for (let st = 0; st < sub; st++) {
        const k1t = w, k1w = accP(th, w);
        const k2t = w + 0.5 * h * k1w, k2w = accP(th + 0.5 * h * k1t, k2t);
        const k3t = w + 0.5 * h * k2w, k3w = accP(th + 0.5 * h * k2t, k3t);
        const k4t = w + h * k3w, k4w = accP(th + h * k3t, k4t);
        th += (h / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
        w  += (h / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
      }
    }
    if (!pendMatch) {
      throw new Error('Gate P2a (Damped Elliptic Pendulum) failed: never observed confirmed PHYS_REGIME_PENDULUM with ell_T>0 & zeta>0');
    }
    const relErrT = Math.abs(pendMatch.ell_T - p2aTagm) / p2aTagm;
    if (
      pendMatch.r2 < 0.80 ||
      pendMatch.ell_T <= p2aT0 * 1.015 ||
      relErrT > 0.15 ||
      pendMatch.zeta < 0.003 ||
      pendMatch.zeta > 0.25 ||
      pendMatch.q <= 1.0 ||
      pendMatch.q >= 999.0 ||
      pendMatch.taubinCurv <= 0.0 ||
      !Number.isFinite(pendMatch.taubinDelta) ||
      pendMatch.E_spec <= 0.0
    ) {
      throw new Error(
        `Gate P2a failed: r2=${pendMatch.r2.toFixed(3)}, T_ell=${pendMatch.ell_T.toFixed(3)} (T0=${p2aT0.toFixed(3)}, Tagm=${p2aTagm.toFixed(3)}, relErr=${(relErrT*100).toFixed(1)}%), zeta=${pendMatch.zeta.toFixed(4)}, Q=${pendMatch.q.toFixed(1)}, curv=${pendMatch.taubinCurv.toFixed(4)}, E=${pendMatch.E_spec.toFixed(1)}`
      );
    }
  }

  // P2b & P5 helper: RK4 Ballistic + Quadratic Drag arc in SD (scale=1) and 2x HD (scale=2)
  function runBallisticDragArc(resMode) {
    const s = resMode === 1 ? 2 : 1;
    const w = 320 * s, hPx = 180 * s;
    const ring = resMode === 1 ? ringHdP : ringSdP;
    const bg = resMode === 1 ? hdBgP : staticBg;
    const buf = new Uint8Array(bg.length);
    const dtFrame = 1.0 / 30.0;
    const g = 300.0, beta = 0.0055;
    let x = 42.0, y = 140.0, vx = 145.0, vy = -170.0;
    const sub = 6, hStep = dtFrame / sub;

    exp.chronos_init();
    exp.chronos_set_resolution(resMode);
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    prevFiedlerTrack.fill(0.0);

    let lastBal = null;
    let maxLyap = 0.0;
    for (let f = 0; f < 48; f++) {
      buf.set(bg);
      stampDiscSubP(buf, w, hPx, x * s, y * s, 5.8 * s, 245, 200, 65, 255, 250, 200);
      stepFrameP(buf, ring, f, dtFrame, resMode === 1 ? 'P5-HD' : 'P2b-SD');
      const ly = telem[TELEM_ENSEMBLE_START + 11];
      if (ly > maxLyap) maxLyap = ly;
      if (f >= 24) {
        const ts = readActiveTargetsP();
        if (ts.length > 0) lastBal = ts[0];
      }
      for (let st = 0; st < sub; st++) {
        const sp1 = Math.hypot(vx, vy);
        const ax1 = -beta * sp1 * vx, ay1 = g - beta * sp1 * vy;
        const vx2 = vx + 0.5 * hStep * ax1, vy2 = vy + 0.5 * hStep * ay1;
        const sp2 = Math.hypot(vx2, vy2);
        const ax2 = -beta * sp2 * vx2, ay2 = g - beta * sp2 * vy2;
        const vx3 = vx + 0.5 * hStep * ax2, vy3 = vy + 0.5 * hStep * ay2;
        const sp3 = Math.hypot(vx3, vy3);
        const ax3 = -beta * sp3 * vx3, ay3 = g - beta * sp3 * vy3;
        const vx4 = vx + hStep * ax3, vy4 = vy + hStep * ay3;
        const sp4 = Math.hypot(vx4, vy4);
        const ax4 = -beta * sp4 * vx4, ay4 = g - beta * sp4 * vy4;
        x  += (hStep / 6.0) * (vx + 2.0 * vx2 + 2.0 * vx3 + vx4);
        y  += (hStep / 6.0) * (vy + 2.0 * vy2 + 2.0 * vy3 + vy4);
        vx += (hStep / 6.0) * (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4);
        vy += (hStep / 6.0) * (ay1 + 2.0 * ay2 + 2.0 * ay3 + ay4);
      }
    }
    return { lastBal, maxLyap };
  }

  const p2bSd = runBallisticDragArc(0);
  if (
    !p2bSd.lastBal ||
    (p2bSd.lastBal.regime !== 1 && p2bSd.lastBal.regime !== 2) ||
    p2bSd.lastBal.r2 < 0.85 ||
    p2bSd.lastBal.imm_bal < 0.35 ||
    p2bSd.maxLyap !== 0.0
  ) {
    throw new Error(
      `Gate P2b (Ballistic+Drag) failed: bal=${JSON.stringify(p2bSd.lastBal)}, maxLyap=${p2bSd.maxLyap}`
    );
  }

  // P2c: Straight constant-velocity negative gate
  let cvLast = null;
  let maxLyapCv = 0.0;
  {
    exp.chronos_init();
    exp.chronos_set_resolution(0);
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    prevFiedlerTrack.fill(0.0);
    const buf = new Uint8Array(BYTES);
    for (let f = 0; f < 60; f++) {
      buf.set(staticBg);
      stampDiscSubP(buf, 320, 180, 34.0 + f * 3.1, 90.0, 5.8, 70, 225, 150, 220, 255, 235);
      stepFrameP(buf, ringSdP, f, 1.0 / 30.0, 'P2c-CV');
      const ly = telem[TELEM_ENSEMBLE_START + 11];
      if (ly > maxLyapCv) maxLyapCv = ly;
      if (f >= 40) {
        const ts = readActiveTargetsP();
        if (ts.length > 0) cvLast = { ...ts[0], kurR: telem[TELEM_ENSEMBLE_START] };
      }
    }
    if (
      !cvLast ||
      cvLast.regime !== 0 ||
      cvLast.ell_T !== 0.0 ||
      cvLast.zeta !== 0.0 ||
      cvLast.q !== 0.0 ||
      cvLast.jump !== 0 ||
      cvLast.kurR !== 0.0 ||
      maxLyapCv !== 0.0
    ) {
      throw new Error(`Gate P2c (CV negative gate) failed: cv=${JSON.stringify(cvLast)}, maxLyapCv=${maxLyapCv}`);
    }
  }

  // P3a: Two-body glancing inelastic collision (m2/m1 = 1.6, e_n = 0.72, labMode = 2)
  let colEvent = null;
  {
    exp.chronos_init();
    exp.chronos_set_resolution(0);
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 2);
    prevFiedlerTrack.fill(0.0);
    const buf = new Uint8Array(BYTES);
    const fc = 34;
    const xAc = 136.0, yAc = 92.0, xBc = 144.0, yBc = 102.0;
    const vxAp = 2.2, vyAp = 0.5, vxBp = -0.9, vyBp = -0.15;
    const vxAq = 0.5823, vyAq = -1.5221, vxBq = 0.1111, vyBq = 1.1138;

    for (let f = 0; f < 75; f++) {
      const df = f - fc;
      const xA = df <= 0 ? xAc + df * vxAp : xAc + df * vxAq;
      const yA = df <= 0 ? yAc + df * vyAp : yAc + df * vyAq;
      const xB = df <= 0 ? xBc + df * vxBp : xBc + df * vxBq;
      const yB = df <= 0 ? yBc + df * vyBp : yBc + df * vyBq;
      buf.set(staticBg);
      stampDiscSubP(buf, 320, 180, xA, yA, 5.5, 255, 85, 65, 255, 220, 200);
      stampDiscSubP(buf, 320, 180, xB, yB, 5.5, 60, 225, 255, 215, 255, 255);
      stepFrameP(buf, ringSdP, f, 1.0 / 60.0, 'P3a-collision');

      if (f >= 42) {
        const pairFrame = Math.round(telem[TELEM_ENSEMBLE_START + 10]);
        const ts = readActiveTargetsP();
        const colT = ts.find((t) => t.regime === 7 && t.jump === 4 && t.e_n > 0);
        if (pairFrame > 0 && colT) {
          colEvent = {
            f,
            pairFrame,
            massRatio: telem[TELEM_ENSEMBLE_START + 5],
            momResid: telem[TELEM_ENSEMBLE_START + 6],
            energyRet: telem[TELEM_ENSEMBLE_START + 7],
            e_n: colT.e_n,
          };
        }
      }
    }
    if (
      !colEvent ||
      colEvent.e_n < 0.45 ||
      colEvent.e_n > 0.92 ||
      colEvent.massRatio < 1.10 ||
      colEvent.massRatio > 2.30 ||
      colEvent.momResid > 0.40 ||
      colEvent.energyRet <= 0.0 ||
      colEvent.energyRet > 1.10
    ) {
      throw new Error(`Gate P3a (Two-Body Collision) failed: colEvent=${JSON.stringify(colEvent)}`);
    }
  }

  // P3b: Parallel overtaking non-colliding lanes (must emit ZERO collisions)
  {
    exp.chronos_init();
    exp.chronos_set_resolution(0);
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 2);
    prevFiedlerTrack.fill(0.0);
    const buf = new Uint8Array(BYTES);
    for (let f = 0; f < 75; f++) {
      buf.set(staticBg);
      stampDiscSubP(buf, 320, 180, 30.0 + f * 3.2, 60.0, 5.5, 255, 85, 65, 255, 220, 200);
      stampDiscSubP(buf, 320, 180, 55.0 + f * 1.4, 120.0, 5.5, 60, 225, 255, 215, 255, 255);
      stepFrameP(buf, ringSdP, f, 1.0 / 60.0, 'P3b-noncol');
      if (telem[TELEM_ENSEMBLE_START + 10] > 0) {
        throw new Error(`Gate P3b (Non-Collision Negative) failed: false pair at f=${f}, pairFrame=${telem[TELEM_ENSEMBLE_START + 10]}`);
      }
      for (const t of readActiveTargetsP()) {
        if (t.jump === 4 || t.e_n > 0.0) {
          throw new Error(`Gate P3b (Non-Collision Negative) failed: target ${t.tid} emitted jump=${t.jump}, e_n=${t.e_n} at f=${f}`);
        }
      }
    }
  }

  // P4a: Kuramoto order R_sync (in-phase >= 0.75 vs anti-phase <= 0.40, labMode = 1)
  function runKuramotoPair(phaseOffset) {
    exp.chronos_init();
    exp.chronos_set_resolution(0);
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 1);
    prevFiedlerTrack.fill(0.0);
    const buf = new Uint8Array(BYTES);
    const dtFrame = 1.0 / 30.0;
    const samples = [];
    for (let f = 0; f < 140; f++) {
      const t = f * dtFrame;
      const th0 = 0.42 * Math.sin(2.0 * Math.PI * 1.05 * t);
      const th1 = 0.42 * Math.sin(2.0 * Math.PI * 1.05 * t + phaseOffset);
      const x0 = 96.0 + 52.0 * Math.sin(th0);
      const y0 = 42.0 + 52.0 * Math.cos(th0);
      const x1 = 224.0 + 52.0 * Math.sin(th1);
      const y1 = 42.0 + 52.0 * Math.cos(th1);
      buf.set(staticBg);
      stampDiscSubP(buf, 320, 180, x0, y0, 5.8, 255, 195, 70, 255, 248, 210);
      stampDiscSubP(buf, 320, 180, x1, y1, 5.8, 80, 225, 190, 220, 255, 245);
      stepFrameP(buf, ringSdP, f, dtFrame, 'P4a-kuramoto');
      if (f >= 90 && telem[TELEM_ENSEMBLE_START + 2] >= 2) {
        samples.push(telem[TELEM_ENSEMBLE_START]);
      }
    }
    if (samples.length === 0) {
      throw new Error(`Gate P4a (phaseOffset=${phaseOffset.toFixed(2)}) failed: kuramoto_osc_count never reached >=2`);
    }
    samples.sort((a, b) => a - b);
    return samples[Math.floor(samples.length / 2)];
  }
  const kurIn = runKuramotoPair(0.08);
  const kurAnti = runKuramotoPair(Math.PI);
  if (kurIn < 0.75 || kurAnti > 0.40) {
    throw new Error(`Gate P4a (Kuramoto) failed: expected R_in>=0.75 & R_anti<=0.40, got R_in=${kurIn.toFixed(3)}, R_anti=${kurAnti.toFixed(3)}`);
  }

  // P4b: Bounded Chaotic Rossler Attractor (0.05 < lambda_max < 20.0, while P2a<=0.05 & P2c==0)
  let maxLyapChaos = 0.0;
  {
    let rx = 1.0, ry = 1.0, rz = 1.0;
    const ra = 0.2, rb = 0.2, rc = 5.7, hR = 0.05;
    function rosslerStep() {
      const k1x = -ry - rz, k1y = rx + ra * ry, k1z = rb + rz * (rx - rc);
      const x2 = rx + 0.5 * hR * k1x, y2 = ry + 0.5 * hR * k1y, z2 = rz + 0.5 * hR * k1z;
      const k2x = -y2 - z2, k2y = x2 + ra * y2, k2z = rb + z2 * (x2 - rc);
      const x3 = rx + 0.5 * hR * k2x, y3 = ry + 0.5 * hR * k2y, z3 = rz + 0.5 * hR * k2z;
      const k3x = -y3 - z3, k3y = x3 + ra * y3, k3z = rb + z3 * (x3 - rc);
      const x4 = rx + hR * k3x, y4 = ry + hR * k3y, z4 = rz + hR * k3z;
      const k4x = -y4 - z4, k4y = x4 + ra * y4, k4z = rb + z4 * (x4 - rc);
      rx += (hR / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
      ry += (hR / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
      rz += (hR / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
    for (let w = 0; w < 260; w++) rosslerStep();

    exp.chronos_init();
    exp.chronos_set_resolution(0);
    exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
    prevFiedlerTrack.fill(0.0);
    const buf = new Uint8Array(BYTES);
    for (let f = 0; f < 168; f++) {
      const px = 160.0 + rx * 3.2;
      const py = 92.0 + ry * 2.6;
      buf.set(staticBg);
      stampDiscSubP(buf, 320, 180, px, py, 5.8, 225, 105, 245, 255, 220, 255);
      stepFrameP(buf, ringSdP, f, 1.0 / 30.0, 'P4b-chaos');
      const ly = telem[TELEM_ENSEMBLE_START + 11];
      if (f >= 64 && ly > maxLyapChaos) maxLyapChaos = ly;
      for (let st = 0; st < 5; st++) rosslerStep();
    }
    if (maxLyapChaos <= 0.05 || maxLyapChaos >= 20.0 || maxLyapPend > 0.05 || maxLyapCv !== 0.0) {
      throw new Error(
        `Gate P4b (Lyapunov Chaos Bounds) failed: chaos=${maxLyapChaos.toFixed(3)}, pend=${maxLyapPend.toFixed(3)}, cv=${maxLyapCv.toFixed(3)}`
      );
    }
  }

  // P5: Cross-Resolution SD vs 2x HD Invariance (identical ballistic+drag trajectory)
  const p5Hd = runBallisticDragArc(1);
  if (
    !p5Hd.lastBal ||
    p5Hd.lastBal.regime !== p2bSd.lastBal.regime ||
    Math.abs(p5Hd.lastBal.r2 - p2bSd.lastBal.r2) > 0.08 ||
    Math.abs(p5Hd.lastBal.conf - p2bSd.lastBal.conf) > 0.12
  ) {
    throw new Error(
      `Gate P5 (SD vs 2x HD Invariance) failed: SD=${JSON.stringify(p2bSd.lastBal)} vs HD=${JSON.stringify(p5Hd.lastBal)}`
    );
  }
  exp.chronos_set_resolution(0);

  console.log(
    `  [PASS] Gate P (P1 ABI=${TELEMETRY_LEN}/448pg/Arena=${Math.round(telem[20])}B | P2 Pend T=${pendMatch.ell_T.toFixed(3)}s vs AGM=${p2aTagm.toFixed(3)}s, R2=${pendMatch.r2.toFixed(3)}, zeta=${pendMatch.zeta.toFixed(4)} | P3 Col m2/m1=${colEvent.massRatio.toFixed(2)}, e=${colEvent.e_n.toFixed(2)}, res=${colEvent.momResid.toFixed(3)} | P4 R=${kurIn.toFixed(2)}/${kurAnti.toFixed(2)}, lyap=${maxLyapChaos.toFixed(2)} vs ${maxLyapPend.toFixed(2)} | P5 SD==HD R2=${p2bSd.lastBal.r2.toFixed(3)}/${p5Hd.lastBal.r2.toFixed(3)})`
  );
}

console.log('');
console.log('[4/5] Running 5x2 Interleaved Blind Pixel Pillar Benchmark (320x180 SD vs 640x360 HD)...');

const TRIALS = 5;
const FRAMES_PER_TRIAL = 480;
const basePillars = [
  { x0: 88, x1: 136, y0: 14, y1: 166 },
  { x0: 184, x1: 232, y0: 14, y1: 166 },
];

function buildStaticGridBg(w, h, s) {
  const out = new Uint8Array(w * h * 4);
  const gStep = 24 * s;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const grid = (x % gStep === 0 || y % gStep === 0) ? 14 : 0;
      const base = 18 + ((((x / s) | 0) * 7 + ((y / s) | 0) * 11) >> 6) + grid;
      out[idx] = base;
      out[idx + 1] = base + 4;
      out[idx + 2] = base + 10;
      out[idx + 3] = 255;
    }
  }
  return out;
}

function stampDiscScaled(buf, w, h, s, cx, cy, r, g, b, cr, cg, cb) {
  const rad = 8 * s;
  const outerR2 = 60 * s * s;
  const innerR2 = 14 * s * s;
  const x0 = Math.max(0, Math.floor(cx - rad)), x1 = Math.min(w - 1, Math.ceil(cx + rad));
  const y0 = Math.max(0, Math.floor(cy - rad)), y1 = Math.min(h - 1, Math.ceil(cy + rad));
  for (let py = y0; py <= y1; py++) {
    for (let px = x0; px <= x1; px++) {
      const r2 = (px - cx) ** 2 + (py - cy) ** 2;
      if (r2 <= outerR2) {
        const idx = (py * w + px) * 4;
        const c = r2 <= innerR2;
        buf[idx] = c ? cr : r;
        buf[idx + 1] = c ? cg : g;
        buf[idx + 2] = c ? cb : b;
        buf[idx + 3] = 255;
      }
    }
  }
}

const bgByMode = [buildStaticGridBg(320, 180, 1), buildStaticGridBg(640, 360, 2)];
const workBufByMode = [new Uint8Array(320 * 180 * 4), new Uint8Array(640 * 360 * 4)];
const oracleTrackOwner = new Int16Array(65536);
const oracleCandTrackSlot = new Int32Array(64);
const oracleCandDiscIdx = new Int32Array(64);
const oracleCandDist = new Float32Array(64);

function makeBenchAccumulator(mode, label, budgetMs) {
  return {
    mode,
    label,
    budgetMs,
    trialMeans: [],
    trialP50s: [],
    trialP99s: [],
    totalCollisions: 0,
    totalKalmanSwaps: 0,
    totalPadicSwaps: 0,
    totalOracleSwaps: 0,
    visibleDiscChecks: 0,
    lockedDiscChecks: 0,
    finalClutterDb: 0,
    finalEvalRatio: 0,
    finalActiveTargets: 0,
    pipChecksum: 0,
  };
}

const benchStats = [
  makeBenchAccumulator(0, '320x180 SD (32x32 Patch/PiP)', 0.35),
  makeBenchAccumulator(1, '640x360 HD (64x64 Foveated)', 1.00),
];

function runSinglePillarTrial(acc) {
  const mode = acc.mode;
  const s = mode === 1 ? 2 : 1;
  const fw = 320 * s;
  const fh = 180 * s;
  const fBytes = fw * fh * 4;
  const bg = bgByMode[mode];
  const frameBuf = workBufByMode[mode];

  exp.chronos_init();
  exp.chronos_set_resolution(mode);
  prevFiedlerTrack.fill(0.0);
  oracleTrackOwner.fill(-1);

  const ringViews = Array.from({ length: 12 }, (_, slot) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), fBytes));

  const discs = [
    { id: 0, x: 52 * s,  y: 56 * s,  vx: 3.0 * s,  r: 248, g: 58,  b: 68,  cr: 255, cg: 230, cb: 110 },
    { id: 1, x: 172 * s, y: 56 * s,  vx: -3.0 * s, r: 30,  g: 225, b: 250, cr: 15,  cg: 45,  cb: 85  },
    { id: 2, x: 148 * s, y: 124 * s, vx: 3.0 * s,  r: 50,  g: 242, b: 112, cr: 235, cg: 255, cb: 235 },
    { id: 3, x: 268 * s, y: 124 * s, vx: -3.0 * s, r: 232, g: 64,  b: 248, cr: 90,  cg: 245, cb: 255 },
  ];

  const frameLatencies = [];
  let collisions = 0;

  for (let f = 0; f < FRAMES_PER_TRIAL; f++) {
    frameBuf.set(bg);
    for (const d of discs) {
      d.x += d.vx;
      d.ry = d.y + 0.9 * s * Math.sin(2 * Math.PI * 9.5 * (f / 60) + d.id * 1.2);
      if (d.x < 18 * s) { d.x = 18 * s; d.vx = Math.abs(d.vx); }
      if (d.x > fw - 18 * s) { d.x = fw - 18 * s; d.vx = -Math.abs(d.vx); }
    }
    for (const [i, j] of [[0, 1], [2, 3]]) {
      const a = discs[i], b = discs[j];
      const dx = b.x - a.x;
      if (Math.abs(dx) < 14.0 * s && (a.vx - b.vx) * dx > 0) {
        const tmp = a.vx; a.vx = b.vx; b.vx = tmp;
        const mx = 0.5 * (a.x + b.x);
        if ((mx >= 88 * s && mx <= 136 * s) || (mx >= 184 * s && mx <= 232 * s)) collisions++;
      }
    }
    for (const d of discs) {
      stampDiscScaled(frameBuf, fw, fh, s, d.x, d.ry, d.r, d.g, d.b, d.cr, d.cg, d.cb);
    }
    for (const p of basePillars) {
      const x0 = p.x0 * s, x1 = p.x1 * s, y0 = p.y0 * s, y1 = p.y1 * s;
      for (let py = y0; py < y1; py++) {
        for (let px = x0; px < x1; px++) {
          const idx = (py * fw + px) * 4;
          const v = (px === x0 || px === x1 - 1) ? 58 : 34 + (((py / s) >> 3) & 3);
          frameBuf[idx] = v; frameBuf[idx + 1] = v + 5; frameBuf[idx + 2] = v + 12; frameBuf[idx + 3] = 255;
        }
      }
    }

    ringViews[f % 12].set(frameBuf);

    // Measure complete production compute tick: Wasm pre_tick + Worker Shifted Lanczos + Wasm post_tick
    const t0 = performance.now();
    exp.chronos_pre_tick(1 / 60);
    solveShiftedLanczosFiedler();
    exp.chronos_post_tick(1 / 60);
    const dt = performance.now() - t0;

    acc.pipChecksum = (acc.pipChecksum + pipBuf[mode === 1 ? 8192 : 2048]) | 0;
    if (f >= 16) frameLatencies.push(dt);

    // Strict 1-to-1 nearest-match Oracle & visible lock coverage audit
    let nPairs = 0;
    for (let di = 0; di < discs.length; di++) {
      const d = discs[di];
      const occ = (d.x > 82 * s && d.x < 142 * s) || (d.x > 178 * s && d.x < 238 * s);
      if (occ) continue;
      if (f >= 20) acc.visibleDiscChecks++;

      for (let m = 0; m < MAX_TARGETS; m++) {
        const base = 32 + m * 56;
        if (telem[base] < 0.5 || telem[base + 44 + 11] < 0.5) continue;
        const dist = Math.hypot(telem[base + 8] - d.x, telem[base + 9] - d.ry);
        if (dist < 16.0 * s) {
          oracleCandTrackSlot[nPairs] = m;
          oracleCandDiscIdx[nPairs] = di;
          oracleCandDist[nPairs] = dist;
          nPairs++;
        }
      }
    }

    for (let a = 0; a < nPairs; a++) {
      for (let b = a + 1; b < nPairs; b++) {
        if (oracleCandDist[b] < oracleCandDist[a]) {
          const td = oracleCandDist[a]; oracleCandDist[a] = oracleCandDist[b]; oracleCandDist[b] = td;
          const tm = oracleCandTrackSlot[a]; oracleCandTrackSlot[a] = oracleCandTrackSlot[b]; oracleCandTrackSlot[b] = tm;
          const ti = oracleCandDiscIdx[a]; oracleCandDiscIdx[a] = oracleCandDiscIdx[b]; oracleCandDiscIdx[b] = ti;
        }
      }
    }

    let usedTracks = 0;
    let usedDiscs = 0;
    for (let k = 0; k < nPairs; k++) {
      const m = oracleCandTrackSlot[k];
      const di = oracleCandDiscIdx[k];
      if ((usedTracks & (1 << m)) !== 0 || (usedDiscs & (1 << di)) !== 0) continue;
      usedTracks |= 1 << m;
      usedDiscs |= 1 << di;
      if (f >= 20) acc.lockedDiscChecks++;

      const tid = Math.round(telem[32 + m * 56 + 1]);
      const discId = discs[di].id;
      const prevOwner = oracleTrackOwner[tid];
      if (prevOwner === -1) oracleTrackOwner[tid] = discId;
      else if (prevOwner !== discId) {
        acc.totalOracleSwaps++;
        oracleTrackOwner[tid] = discId;
      }
    }
  }

  frameLatencies.sort((a, b) => a - b);
  acc.trialMeans.push(frameLatencies.reduce((a, b) => a + b, 0) / frameLatencies.length);
  acc.trialP50s.push(frameLatencies[Math.floor(frameLatencies.length * 0.5)]);
  acc.trialP99s.push(frameLatencies[Math.floor(frameLatencies.length * 0.99)]);

  acc.totalCollisions += collisions;
  acc.totalPadicSwaps += Math.round(telem[5]);
  acc.totalKalmanSwaps += Math.round(telem[6]);
  acc.finalClutterDb = telem[4];
  acc.finalEvalRatio = telem[23];
  acc.finalActiveTargets = Math.round(telem[3]);
}

// Interleave SD and HD across 5 pairs (SD, HD, SD, HD, ...) to cancel thermal / scheduler drift
for (let t = 0; t < TRIALS; t++) {
  runSinglePillarTrial(benchStats[0]);
  runSinglePillarTrial(benchStats[1]);
}

for (const acc of benchStats) {
  acc.trialMeans.sort((a, b) => a - b);
  acc.trialP50s.sort((a, b) => a - b);
  acc.trialP99s.sort((a, b) => a - b);

  acc.medMean = acc.trialMeans[Math.floor(TRIALS / 2)];
  acc.minMean = acc.trialMeans[0];
  acc.maxMean = acc.trialMeans[TRIALS - 1];
  acc.medP50 = acc.trialP50s[Math.floor(TRIALS / 2)];
  acc.medP99 = acc.trialP99s[Math.floor(TRIALS / 2)];
  acc.coveragePct = (100.0 * acc.lockedDiscChecks) / Math.max(1, acc.visibleDiscChecks);

  console.log(`  --- ${acc.label} (${TRIALS} interleaved trials x ${FRAMES_PER_TRIAL} frames, PiP guard=${acc.pipChecksum}) ---`);
  console.log(`  Full Compute Tick   : median-of-means = ${acc.medMean.toFixed(4)} ms (range ${acc.minMean.toFixed(4)}..${acc.maxMean.toFixed(4)} ms | p50=${acc.medP50.toFixed(4)} ms, p99=${acc.medP99.toFixed(4)} ms)`);
  console.log(`  Uncensored ECA Supp.: ${acc.finalClutterDb.toFixed(2)} dB over ${(acc.finalEvalRatio * 100).toFixed(1)}% stationary background (> 60 dB target)`);
  console.log(`  Auto-Locked Targets : ${acc.finalActiveTargets} / 4 physical discs (${acc.coveragePct.toFixed(1)}% visible-frame lock coverage)`);
  console.log(`  In-Pillar Collisions: ${acc.totalCollisions} across ${TRIALS} trials`);
  console.log(`  1-to-1 Oracle Swaps : ${acc.totalOracleSwaps} (Internal Wasm Conflict Audit: ${acc.totalPadicSwaps})`);
  console.log(`  2x2 Kalman Swaps    : ${acc.totalKalmanSwaps} (Proves causal kinematic drift hijack)`);

  if (acc.totalPadicSwaps !== 0 || acc.totalOracleSwaps !== 0) {
    throw new Error(`[${acc.label}] 2-Adic ID Swap invariant violated: wasm=${acc.totalPadicSwaps}, oracle=${acc.totalOracleSwaps}`);
  }
  if (acc.coveragePct < 90.0) {
    throw new Error(`[${acc.label}] Expected >= 90% visible disc lock coverage, got ${acc.coveragePct.toFixed(1)}%`);
  }
  if (acc.totalKalmanSwaps === 0) {
    throw new Error(`[${acc.label}] Expected > 0 Causal Kalman ID swaps during in-pillar collisions`);
  }
  if (acc.finalClutterDb < 60.0) {
    throw new Error(`[${acc.label}] Expected > 60 dB uncensored ECA clutter suppression, got ${acc.finalClutterDb.toFixed(2)} dB`);
  }
  const rawScaleEnv = process.env.CHRONOS_CI_BUDGET_SCALE;
  const budgetScale = rawScaleEnv === undefined || rawScaleEnv === '' ? 1.0 : Number(rawScaleEnv);
  if (!Number.isFinite(budgetScale) || budgetScale < 1.0) {
    throw new Error(`Invalid CHRONOS_CI_BUDGET_SCALE="${rawScaleEnv}" (must be finite >= 1.0)`);
  }
  const effectiveBudgetMs = acc.budgetMs * budgetScale;
  if (acc.medMean >= effectiveBudgetMs) {
    throw new Error(`[${acc.label}] Per-frame compute budget exceeded: ${acc.medMean.toFixed(4)} ms >= ${effectiveBudgetMs.toFixed(2)} ms`);
  }
}

const rawOutScaleEnv = process.env.CHRONOS_CI_BUDGET_SCALE;
const outScale = rawOutScaleEnv === undefined || rawOutScaleEnv === '' ? 1.0 : Number(rawOutScaleEnv);
if (!Number.isFinite(outScale) || outScale < 1.0) {
  throw new Error(`Invalid CHRONOS_CI_BUDGET_SCALE="${rawOutScaleEnv}" (must be finite >= 1.0)`);
}
const hdRatio = benchStats[1].medMean / Math.max(1e-6, benchStats[0].medMean);
console.log(`  Scaling Limiter     : HD/SD compute ratio = ${hdRatio.toFixed(2)}x for 4.00x pixel count (planar SIMD128 RGBA/ECA memory bandwidth scales 4x; 2x2-pooled 32x32 patch SAD + Lanczos scale sub-linearly)`);
console.log(`  [${outScale > 1.0 ? 'PARTIAL - SCALED BUDGET' : 'PASS'}] ALL SD (< ${(0.35 * outScale).toFixed(2)} ms) & HD (< ${(1.00 * outScale).toFixed(2)} ms) MATHEMATICAL & LATENCY INVARIANTS VERIFIED`);
EOF

echo ""
echo "[Gate Q] Running CHRONOS-1000 Swarm & Continuum Gate Suite (scripts/benchmark_1000.mjs --gate)..."
node scripts/benchmark_1000.mjs --gate

echo ""
echo "[Gate W1] Running Wave-1 Core Gates (I0/I4 Ingest+ROI, I5 BOS, I1 Metrology, I6 Captures)..."
node scripts/gate_i0_i4_ingest.mjs
node scripts/gate_i5_bos.mjs
node scripts/gate_i1_cov_residual.mjs
if [ "${1:-}" = "--e2e" ]; then
  node scripts/gate_i6_captures.mjs
else
  node scripts/gate_i6_captures.mjs --no-chrome
fi

echo ""
echo "[Gate W2] Running Wave-2 Physics & Identifiability Gates (I2 RS+Welch, I3 Mesh Hodge, I5b Occluded Conservation)..."
node scripts/gate_i2_rs_welch.mjs
node scripts/gate_i3_hodge.mjs
node scripts/gate_i5b_conservation.mjs

echo ""
echo "[Gate X] Running Adversarial C-ABI Fuzz (random exports with hostile arguments, pixels and staging buffers; 0 traps)..."
node scripts/gate_abi_fuzz.mjs
fi

echo ""
echo "[Gate S] Running SIMD Layer Conformance (simd::scalar vs Wasm SIMD128 in V8, native host suite, negative controls)..."
node scripts/gate_simd_conformance.mjs
# Native host: the same conformance table (selected backend vs simd::scalar), scalar math vs std,
# negative controls and coverage checks; then the native rlib build linked into a std test binary.
# `-D warnings` goes in as a `target.<cfg>.rustflags` entry, which Cargo joins with the ones in
# .cargo/config.toml. A RUSTFLAGS variable would replace them, dropping the x86_64 SSSE3/SSE4.1
# flags so that x86_64 hosts silently test simd::scalar instead of simd::sse.
DENY_WARNINGS="target.'cfg(all())'.rustflags = ['-D', 'warnings']"
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS cargo test --quiet --config "$DENY_WARNINGS" --target host-tuple
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS cargo test --quiet --config "$DENY_WARNINGS" --manifest-path native/Cargo.toml --target host-tuple
echo "  [PASS] Native host ($(rustc -vV | sed -n 's/^host: //p')) conformance suite & native library link verified with zero warnings"

echo ""
echo "[Gate N] Running Native/Wasm Whole-Kernel Parity (fresh CSTRACE recording replayed bit for bit on the native build, negative control)..."
if [ "${1:-}" = "--native-check" ] && ! command -v ffmpeg >/dev/null 2>&1; then
  # The real-MP4 scenario decodes its clip with ffmpeg; --native-check also runs on hosts
  # without it (the macOS CI job). --check and --e2e always run every scenario.
  echo "  [SKIP] Gate N scenario hero_sd (real MP4): ffmpeg not on PATH (--native-check only)"
  node scripts/gate_native_parity.mjs --scenarios=hodge,conservation,swarm
else
  node scripts/gate_native_parity.mjs
fi

CERT_PARTIAL=0
if [ -n "${CHRONOS_CI_BUDGET_SCALE:-}" ]; then
  CERT_PARTIAL=$(node -e "const s = Number(process.env.CHRONOS_CI_BUDGET_SCALE); if (!Number.isFinite(s) || s < 1.0) { process.exit(2); } process.stdout.write(s > 1.0 ? '1' : '0');")
fi

if [ "${1:-}" = "--check" ] || [ "${1:-}" = "--native-check" ]; then
  echo ""
  if [ "$CERT_PARTIAL" = "1" ]; then
    echo "[5/5] [PARTIAL - SCALED BUDGET scale=${CHRONOS_CI_BUDGET_SCALE}] ${1} mode complete. Exiting cleanly."
  else
    echo "[5/5] ${1} mode complete. Exiting cleanly."
  fi
  exit 0
fi

if [ "${1:-}" = "--e2e" ]; then
  echo ""
  echo "[5/5] Running Real-Camera MP4 Calibration (SD + HD) & Headless Chrome CDP E2E Suite..."
  node scripts/calibrate_mp4.mjs --gate
  node scripts/calibrate_mp4.mjs --hd --gate
  node scripts/e2e_browser.mjs
  echo ""
  if [ "$CERT_PARTIAL" = "1" ]; then
    echo "[PARTIAL - SCALED BUDGET scale=${CHRONOS_CI_BUDGET_SCALE}] --e2e full-stack verification complete. Exiting cleanly."
  else
    echo "[PASS] --e2e full-stack verification complete. Exiting cleanly."
  fi
  exit 0
fi

PORT="${PORT:-8080}"
free_stale_chronos_port "$PORT"
echo ""
echo "[5/5] Launching CHRONOS-SPECTRAL Command Deck at http://127.0.0.1:${PORT}"
echo "      (RFC 7233 HTTP 206 byte-range + Live FFmpeg fMP4 MSE Relay server)"
echo "      Press Ctrl+C to stop server."
exec env PORT="$PORT" node scripts/dev_server.mjs
