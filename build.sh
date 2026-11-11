#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PATH="/opt/homebrew/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo "========================================================================"
echo " CHRONOS-SPECTRAL // Zero-Allocation Wasm32 SIMD128 Build & Audit"
echo "========================================================================"

echo "[1/5] Compiling #![no_std] Rust kernel (wasm32-unknown-unknown +simd128)..."
cargo build --release

WASM_SRC="target/wasm32-unknown-unknown/release/chronos_spectral.wasm"
WASM_DST="web/chronos_spectral.wasm"
mkdir -p web
cp "$WASM_SRC" "$WASM_DST"
ls -lh "$WASM_DST"

echo ""
echo "[2/5] Auditing Wasm binary layout & SIMD128 / CLZ opcodes via wasm-objdump..."
if command -v wasm-objdump >/dev/null 2>&1; then
  HEADERS="$(wasm-objdump -x "$WASM_DST")"
  DISASM="$(wasm-objdump -d "$WASM_DST")"

  if echo "$HEADERS" | grep -q "^Import\["; then
    echo "FAIL: Unexpected imports detected in Wasm module!" >&2
    echo "$HEADERS" | grep -A 10 "^Import\[" >&2
    exit 1
  fi
  echo "  [PASS] Zero imports verified (0 env / malloc / free allocator imports)"

  MEM_LINE="$(echo "$HEADERS" | grep "memory\[0\]" || true)"
  echo "  [PASS] Contiguous Linear Memory Arena:${MEM_LINE}"

  V128_COUNT="$(echo "$DISASM" | grep -cE "v128\.|f32x4\.|u32x4\.|i32x4\.")"
  CLZ_COUNT="$(echo "$DISASM" | grep -c "i32\.clz")"
  COPYSIGN_COUNT="$(echo "$DISASM" | grep -c "f32\.copysign")"

  if [ "$V128_COUNT" -lt 20 ] || [ "$CLZ_COUNT" -lt 1 ] || [ "$COPYSIGN_COUNT" -lt 1 ]; then
    echo "FAIL: Missing expected hardware SIMD128 or i32.clz / f32.copysign instructions!" >&2
    exit 1
  fi
  echo "  [PASS] Emitted instructions: ${V128_COUNT} SIMD128 ops, ${CLZ_COUNT} i32.clz ops, ${COPYSIGN_COUNT} f32.copysign ops"
else
  echo "  [WARN] wasm-objdump not found in PATH; skipping static disassembly gate."
fi

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
const GRAPH_NODES = 16;
const MAX_TARGETS = 8;
const MAX_KRYLOV = 15;
const TELEMETRY_LEN = 648;

const wasmBytes = fs.readFileSync('./web/chronos_spectral.wasm');
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const exp = instance.exports;
const mem = exp.memory;

if (mem.buffer.byteLength !== 256 * 65536) {
  throw new Error(`Expected 256 pages (16777216 B), got ${mem.buffer.byteLength}`);
}

const telem = new Float32Array(mem.buffer, exp.chronos_get_telemetry_ptr(), TELEMETRY_LEN);
const laplacian = new Float32Array(mem.buffer, exp.chronos_get_laplacian_ptr(), 256);
const fiedler = new Float32Array(mem.buffer, exp.chronos_get_fiedler_ptr(), 16);
const pipBuf = new Uint8Array(mem.buffer, exp.chronos_get_pip_rgba_ptr(), 4096);

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
    laplacian[i * 16 + j] = -w;
    deg += w;
  }
  laplacian[i * 16 + i] = deg;
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
    laplacian[i * 16 + j] = -w;
    deg += w;
  }
  laplacian[i * 16 + i] = deg;
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
      for (let m = 0; m < 8; m++) {
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
  if (exp.chronos_get_resolution() !== 0 || exp.chronos_get_frame_w() !== 320 || exp.chronos_get_frame_h() !== 180 || exp.chronos_get_pip_dim() !== 16) {
    throw new Error('Gate G: Unexpected initial SD dimensions');
  }
  exp.chronos_set_resolution(1);
  if (exp.chronos_get_resolution() !== 1 || exp.chronos_get_frame_w() !== 640 || exp.chronos_get_frame_h() !== 360 || exp.chronos_get_pip_dim() !== 32) {
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
  if (mem.buffer.byteLength !== 256 * 65536) {
    throw new Error(`Gate G: Arena grew unexpectedly to ${mem.buffer.byteLength}`);
  }
  if (telem[30] !== 640 || telem[31] !== 360 || pipBuf[4095] !== 255) {
    throw new Error(`Gate G: Expected HD telemetry (640x360) and 32x32 PiP alpha=255, got ${telem[30]}x${telem[31]}, alpha=${pipBuf[4095]}`);
  }
  exp.chronos_set_resolution(0);
  if (exp.chronos_get_frame_w() !== 320 || exp.chronos_get_pip_dim() !== 16) {
    throw new Error('Gate G: Failed to restore 320x180 SD mode');
  }
  console.log('  [PASS] Gate G (Dual-Resolution 320x180 SD <-> 640x360 HD Switch): 256 static pages, 32x32 PiP verified');
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
    if (telem[640] > 0) gHat = telem[640];
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56, kbase = 576 + m * 8;
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

  // H2: CTRV Curved-Arc Occlusion Coasting behind opaque pillar
  exp.chronos_init();
  let arcTrackId = -1, arcIdConsistent = true, arcBridges = 0;
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
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56;
      if (telem[base] > 0.5) {
        const tid = Math.round(telem[base + 1]);
        if (arcTrackId < 0) arcTrackId = tid;
        else if (tid !== arcTrackId) arcIdConsistent = false;
        if (telem[base + 11] > 0.5) arcBridges++;
      }
    }
  }
  if (arcTrackId < 0 || !arcIdConsistent || arcBridges < 4) {
    throw new Error(`Gate H2 (CTRV Curved Occlusion Coasting): tid=${arcTrackId}, consistent=${arcIdConsistent}, bridges=${arcBridges}`);
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
      const base = 32 + m * 56, kbase = 576 + m * 8;
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
      const base = 32 + m * 56, ebase = 512 + m * 8, kbase = 576 + m * 8;
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
  console.log(`  [PASS] Gate H (Quadratic Gravity g=${gHat.toFixed(1)}, CTRV Arc Coast, Kasa Pivot err=${bestPivotErr.toFixed(2)}px, Deflect=${measDeflect.toFixed(1)}° e=${measRest.toFixed(2)})`);
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
const oracleCandTrackSlot = new Int32Array(32);
const oracleCandDiscIdx = new Int32Array(32);
const oracleCandDist = new Float32Array(32);

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
  makeBenchAccumulator(0, '320x180 SD (16x16 Patch/PiP)', 0.25),
  makeBenchAccumulator(1, '640x360 HD (32x32 Foveated)', 1.00),
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

    acc.pipChecksum = (acc.pipChecksum + pipBuf[mode === 1 ? 2048 : 512]) | 0;
    if (f >= 16) frameLatencies.push(dt);

    // Strict 1-to-1 nearest-match Oracle & visible lock coverage audit
    let nPairs = 0;
    for (let di = 0; di < discs.length; di++) {
      const d = discs[di];
      const occ = (d.x > 82 * s && d.x < 142 * s) || (d.x > 178 * s && d.x < 238 * s);
      if (occ) continue;
      if (f >= 20) acc.visibleDiscChecks++;

      for (let m = 0; m < 8; m++) {
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
  const budgetScale = Math.max(1.0, Number(process.env.CHRONOS_CI_BUDGET_SCALE || '1.0'));
  const effectiveBudgetMs = acc.budgetMs * budgetScale;
  if (acc.medMean >= effectiveBudgetMs) {
    throw new Error(`[${acc.label}] Per-frame compute budget exceeded: ${acc.medMean.toFixed(4)} ms >= ${effectiveBudgetMs.toFixed(2)} ms`);
  }
}

const outScale = Math.max(1.0, Number(process.env.CHRONOS_CI_BUDGET_SCALE || '1.0'));
const hdRatio = benchStats[1].medMean / Math.max(1e-6, benchStats[0].medMean);
console.log(`  Scaling Limiter     : HD/SD compute ratio = ${hdRatio.toFixed(2)}x for 4.00x pixel count (planar SIMD128 RGBA/ECA memory bandwidth scales 4x; 2x2-pooled 32x32 patch SAD + Lanczos scale sub-linearly)`);
console.log(`  [PASS] ALL SD (< ${(0.25 * outScale).toFixed(2)} ms) & HD (< ${(1.00 * outScale).toFixed(2)} ms) MATHEMATICAL & LATENCY INVARIANTS VERIFIED`);
EOF

if [ "${1:-}" = "--check" ]; then
  echo ""
  echo "[5/5] --check mode complete. Exiting cleanly."
  exit 0
fi

if [ "${1:-}" = "--e2e" ]; then
  echo ""
  echo "[5/5] Running Real-Camera MP4 Calibration (SD + HD) & Headless Chrome CDP E2E Suite..."
  node scripts/calibrate_mp4.mjs --gate
  node scripts/calibrate_mp4.mjs --hd --gate
  node scripts/e2e_browser.mjs
  echo ""
  echo "[PASS] --e2e full-stack verification complete. Exiting cleanly."
  exit 0
fi

PORT="${PORT:-8080}"
echo ""
echo "[5/5] Launching CHRONOS-SPECTRAL Command Deck at http://127.0.0.1:${PORT}"
echo "      (RFC 7233 HTTP 206 byte-range server for WebKit/Chromium MP4 + Wasm streaming)"
echo "      Press Ctrl+C to stop server."
exec env PORT="$PORT" node --input-type=module - <<'SERVE_EOF'
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

const root = path.resolve('web');
const port = Number(process.env.PORT || 8080);
const mimes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.mp4': 'video/mp4',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.css': 'text/css; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};

http.createServer((req, res) => {
  const reqUrl = new URL(req.url || '/', 'http://127.0.0.1');
  const rel = decodeURIComponent(reqUrl.pathname === '/' ? '/index.html' : reqUrl.pathname);
  const filePath = path.normalize(path.join(root, rel));
  if (!filePath.startsWith(root) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    res.writeHead(404);
    res.end('Not found');
    return;
  }
  const stat = fs.statSync(filePath);
  const type = mimes[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
  const range = req.headers.range;
  if (range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (!m) {
      res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
      res.end();
      return;
    }
    const start = m[1] ? parseInt(m[1], 10) : 0;
    const end = m[2] ? Math.min(stat.size - 1, parseInt(m[2], 10)) : stat.size - 1;
    if (start >= stat.size || start > end) {
      res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
      res.end();
      return;
    }
    res.writeHead(206, {
      'Content-Type': type,
      'Content-Length': end - start + 1,
      'Content-Range': `bytes ${start}-${end}/${stat.size}`,
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'no-store',
    });
    fs.createReadStream(filePath, { start, end }).pipe(res);
    return;
  }
  res.writeHead(200, {
    'Content-Type': type,
    'Content-Length': stat.size,
    'Accept-Ranges': 'bytes',
    'Cache-Control': 'no-store',
  });
  fs.createReadStream(filePath).pipe(res);
}).listen(port, '127.0.0.1');
SERVE_EOF
