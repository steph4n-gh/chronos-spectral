#!/usr/bin/env node
/**
 * Gate I7: Zero-ML Depth Sensing, FoE/TTC, Calibrated Stereo BOS, External Depth-Plane ABI,
 * Adversarial Hardening Regressions & Worker K>0 / WebCodecs Harness
 *
 * Usage:
 *   node scripts/gate_i7_depth.mjs [--wasm=<path>] [--parity-lite]
 *
 * Suites:
 *   1. Micro-baseline 256x128 calibrated stereo (d = 0.6..1.71 px): AbsRel < 5%, delta1 >= 98%, epiRms < 0.06 px
 *   2. Wide-baseline 256x128 calibrated stereo (d = 6.25..25 px + occlusion + Right-image export coordinates)
 *   3. Monocular lateral motion parallax (160 static 3D landmarks + 4 transverse movers; note movers strictly
 *      along the epipolar line are indistinguishable from static depth by monocular parallax alone)
 *   4. Fail-closed negative controls: pure tripod yaw (REFUSED_PURE_ROTATION) & stationary (REFUSED_NO_BASELINE)
 *   5. Real H.264 clip (perspective_flyby_decoy.mp4) intrinsics-free FoE & optical looming TTC
 *      (FoE within 55 px of (393, 205), >= 2 approaching tracks, best TTC R^2 >= 0.98)
 *   6. External depth-plane ingest (128x64) & Provenance Firewall + Hero-16 depth export
 *   7. Monocular adversarial regressions (mixed yaw+trans, oblique T, wrong/unknown f, vertical flow,
 *      5 fps / 240 ms unclamped dt, shallow-scene hysteresis, synthetic radial + known-TTC vs receding)
 *   8. Stereo adversarial regressions (12 px periodic texture rejection, strict max_disp, 0.35 px vertical
 *      misalignment, sign=+1/0, SD 320x180, unpaired mono pan, scene-cut invalidation, BOS<->Stereo isolation)
 *   9. External staleness, rejected-metadata isolation, reset/res-switch invalidation & AUTO_FUSED fallthrough
 *  10. Worker K=8 display-delay depth ring + WebCodecs fMP4 (+omit_tfhd_offset) / elst / rejected-commit harness
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const WASM_PATH = path.resolve(
  process.argv.find((a) => a.startsWith('--wasm='))?.split('=')[1] ||
    path.join(ROOT, 'web', 'chronos_spectral.wasm')
);
const PARITY_LITE = process.argv.includes('--parity-lite');

const W = 640;
const H = 360;
const BOS_W = 256;
const BOS_H = 128;
const BOS_N = BOS_W * BOS_H;
const DEPTH_W = 64;
const DEPTH_H = 36;
const DEPTH_CELLS = DEPTH_W * DEPTH_H;
const MAX_MICRO = 3072;

const STATE_WARMING = 1;
const STATE_READY_PARALLAX = 2;
const STATE_READY_TTC_ONLY = 3;
const STATE_REFUSED_PURE_ROTATION = 4;
const STATE_REFUSED_NO_BASELINE = 5;
const STATE_READY_STEREO = 6;
const STATE_READY_EXTERNAL = 7;

const PROV_NONE = 0;
const PROV_MONO_UP_TO_SCALE = 1;
const PROV_MONO_METRIC = 2;
const PROV_INTRINSICS_FREE_TTC = 3;
const PROV_SURVEYED_STEREO = 4;
const PROV_SURVEYED_LIDAR = 5;
const PROV_ASSUMED_NEURAL_PRIOR = 6;

const REASON_EXT_STALE = 1 << 4;
const REASON_ASSUMED_NEURAL_PRIOR = 1 << 5;
const REASON_MIXED_ROT_TRANS = 1 << 7;

const FLAG_VALID = 1 << 0;
const FLAG_METRIC = 1 << 1;
const FLAG_TTC_VALID = 1 << 2;
const FLAG_MOVING_OUTLIER = 1 << 3;

function assert(cond, msg) {
  if (!cond) throw new Error(`FAIL [Gate I7]: ${msg}`);
}

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function median(arr) {
  if (!arr.length) return NaN;
  const a = Float64Array.from(arr).sort();
  const m = a.length >> 1;
  return a.length & 1 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

function pearsonR(xs, ys) {
  const n = xs.length;
  if (n < 3) return 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let num = 0;
  let dx2 = 0;
  let dy2 = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx2 += a * a;
    dy2 += b * b;
  }
  return dx2 > 0 && dy2 > 0 ? num / Math.sqrt(dx2 * dy2) : 0;
}

function loadKernel(resMode = 1) {
  const bytes = fs.readFileSync(WASM_PATH);
  const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {});
  const exp = inst.exports;
  const buf = exp.memory.buffer;
  exp.chronos_init();
  exp.chronos_set_resolution(resMode);
  const views = {
    dTelemU32: new Uint32Array(buf, exp.chronos_get_depth_telemetry_ptr(), 32),
    dTelemF32: new Float32Array(buf, exp.chronos_get_depth_telemetry_ptr(), 32),
    dField: new Float32Array(buf, exp.chronos_get_depth_field_ptr(), DEPTH_CELLS * 2),
    dTracks: new Float32Array(buf, exp.chronos_get_depth_tracks_ptr(), MAX_MICRO * 2),
    dFlags: new Uint8Array(buf, exp.chronos_get_depth_track_flags_ptr(), MAX_MICRO),
    dHero: new Float32Array(buf, exp.chronos_get_depth_hero_ptr(), 16 * 4),
    dIngest: new Float32Array(buf, exp.chronos_get_depth_ingest_ptr(), 128 * 64),
    bosTelemU32: new Uint32Array(buf, exp.chronos_get_bos_telemetry_ptr(), 32),
    bosPlanes: new Float32Array(buf, exp.chronos_get_bos_planes_ptr(), 6 * BOS_N),
    bosValid: new Uint8Array(buf, exp.chronos_get_bos_valid_ptr(), BOS_N),
    bosRgba: new Uint8Array(buf, exp.chronos_get_bos_rgba_ptr(), 4 * BOS_N),
    wave2TelemU32: new Uint32Array(buf, exp.chronos_get_wave2_telemetry_ptr(), 48),
    micro: new Float32Array(buf, exp.chronos_get_micro_tracks_ptr(), MAX_MICRO * 8),
    telem: new Float32Array(buf, exp.chronos_get_telemetry_ptr(), 1568),
  };
  return { exp, buf, views };
}

function bosTexture(u, v) {
  const s =
    0.5 +
    0.19 * Math.sin(u * 0.37 + Math.cos(v * 0.23) * 1.1) +
    0.15 * Math.cos(v * 0.41 - Math.sin(u * 0.29) * 0.9) +
    0.09 * Math.sin((u + v) * 0.67) +
    0.05 * Math.cos((u - 1.3 * v) * 0.89);
  return Math.max(0.06, Math.min(0.94, s));
}

function makeNonPeriodicSampler(w, h, seed = 0x57e2e001) {
  const texW = w + 64;
  const texH = h + 8;
  const rng = makeRng(seed);
  const raw = new Float32Array(texW * texH);
  for (let i = 0; i < raw.length; i++) raw[i] = rng();
  const smoothOnce = (src) => {
    const dst = new Float32Array(src.length);
    for (let y = 1; y < texH - 1; y++) {
      for (let x = 1; x < texW - 1; x++) {
        const o = y * texW + x;
        dst[o] =
          (src[o - texW - 1] + 2 * src[o - texW] + src[o - texW + 1] +
            2 * src[o - 1] + 4 * src[o] + 2 * src[o + 1] +
            src[o + texW - 1] + 2 * src[o + texW] + src[o + texW + 1]) / 16.0;
      }
    }
    return dst;
  };
  const tex = smoothOnce(smoothOnce(raw));
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < tex.length; i++) {
    if (tex[i] < mn) mn = tex[i];
    if (tex[i] > mx) mx = tex[i];
  }
  const span = Math.max(1e-4, mx - mn);
  return (fx, fy) => {
    const x = Math.max(1, Math.min(texW - 3, fx + 32));
    const y = Math.max(1, Math.min(texH - 3, fy + 4));
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    const tx = x - ix;
    const ty = y - iy;
    const o = iy * texW + ix;
    const v0 = tex[o] + tx * (tex[o + 1] - tex[o]);
    const v1 = tex[o + texW] + tx * (tex[o + texW + 1] - tex[o + texW]);
    return 0.12 + 0.76 * ((v0 + ty * (v1 - v0) - mn) / span);
  };
}

function makeBaseLandmarks(fPx = 500.0, zMin = 4.5, zSpan = 14.5, seed = 0xdecaf007) {
  const rng = makeRng(seed);
  const pts = [];
  let tries = 0;
  while (pts.length < 160 && tries < 6000) {
    tries++;
    const Z = zMin + rng() * zSpan;
    const u0 = 40 + rng() * (W - 80);
    const v0 = 32 + rng() * (H - 64);
    let ok = true;
    for (const q of pts) {
      if ((q.u0 - u0) ** 2 + (q.v0 - v0) ** 2 < 24 * 24) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    pts.push({
      X: ((u0 - W * 0.5) * Z) / fPx,
      Y: ((v0 - H * 0.5) * Z) / fPx,
      Z,
      u0,
      v0,
      moving: false,
      r: 3.1,
      col: [215, 222, 232],
    });
  }
  return pts;
}

function renderScene3D(u8, pts3d, camPos, yawRad, fPx, pitchRad = 0.0) {
  u8.fill(18);
  for (let i = 3; i < u8.length; i += 4) u8[i] = 255;
  const cx = W * 0.5;
  const cy = H * 0.5;
  const cyaw = Math.cos(yawRad);
  const syaw = Math.sin(yawRad);
  const cp = Math.cos(pitchRad);
  const sp = Math.sin(pitchRad);
  const proj = [];
  for (let i = 0; i < pts3d.length; i++) {
    const p = pts3d[i];
    const dx = p.X - camPos[0];
    const dy = p.Y - camPos[1];
    const dz = p.Z - camPos[2];
    const X1 = cyaw * dx - syaw * dz;
    const Z1 = syaw * dx + cyaw * dz;
    const Xc = X1;
    const Yc = cp * dy - sp * Z1;
    const Zc = sp * dy + cp * Z1;
    if (Zc < 1.0) continue;
    const u = cx + (fPx * Xc) / Zc;
    const v = cy + (fPx * Yc) / Zc;
    if (u < 6 || u > W - 7 || v < 6 || v > H - 7) continue;
    proj.push({ idx: i, u, v, Zc, moving: p.moving, r: p.r, col: p.col });
  }
  proj.sort((a, b) => b.Zc - a.Zc);
  for (const q of proj) {
    const r = q.r;
    const x0 = Math.max(0, Math.floor(q.u - r - 1));
    const x1 = Math.min(W - 1, Math.ceil(q.u + r + 1));
    const y0 = Math.max(0, Math.floor(q.v - r - 1));
    const y1 = Math.min(H - 1, Math.ceil(q.v + r + 1));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d2 = (x - q.u) ** 2 + (y - q.v) ** 2;
        const g = Math.exp(-d2 / (0.55 * r * r));
        if (g < 0.03) continue;
        const o = (y * W + x) * 4;
        u8[o] = Math.min(255, Math.round(18 + g * q.col[0]));
        u8[o + 1] = Math.min(255, Math.round(18 + g * q.col[1]));
        u8[o + 2] = Math.min(255, Math.round(18 + g * q.col[2]));
      }
    }
  }
  return proj;
}

// ============================================================================
// Suite 1: Micro-Baseline 256x128 Calibrated Stereo (d = 0.6..1.71 px)
// ============================================================================
function runSuite1MicroStereo() {
  const { exp, buf, views } = loadKernel();
  const fPx = 500.0;
  const B = 0.012; // f*B = 6.0 m*px -> Z in [3.5..10.0 m] => d in [0.6..1.71 px]

  const depthTrue = (x, y) => {
    const cx1 = 210, cy1 = 180;
    const cx2 = 430, cy2 = 175;
    if ((x - cx1) ** 2 + (y - cy1) ** 2 <= 68 * 68) return 3.6;
    if (Math.abs(x - cx2) <= 58 && Math.abs(y - cy2) <= 56) return 5.2;
    return 9.6 - 1.8 * (y / H);
  };

  assert(exp.chronos_set_depth_mode(2) === 1, 'set_depth_mode(2) failed');
  assert(exp.chronos_set_depth_calib(fPx, 0.0, B, 2.2, -1) === 1, 'set_depth_calib micro failed');

  const leftU8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = Math.round(bosTexture(x, y) * 255);
      const o = (y * W + x) * 4;
      leftU8[o] = v; leftU8[o + 1] = v; leftU8[o + 2] = v; leftU8[o + 3] = 255;
    }
  }
  assert(exp.chronos_depth_capture_stereo_left() === 1, 'capture_stereo_left failed');

  const rightU8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let xL = x + (fPx * B) / (9.6 - 1.8 * (y / H));
      const dBox = (fPx * B) / 5.2;
      if (Math.abs(x + dBox - 430) <= 58 && Math.abs(y - 175) <= 56) xL = x + dBox;
      const dDisc = (fPx * B) / 3.6;
      if ((x + dDisc - 210) ** 2 + (y - 180) ** 2 <= 68 * 68) xL = x + dDisc;
      const v = Math.round(bosTexture(xL, y) * 255);
      const o = (y * W + x) * 4;
      rightU8[o] = v; rightU8[o + 1] = v; rightU8[o + 2] = v; rightU8[o + 3] = 255;
    }
  }

  const t0 = performance.now();
  exp.chronos_tick(1 / 30);
  const dtMs = performance.now() - t0;

  assert(views.dTelemU32[2] === STATE_READY_STEREO, `Suite 1 expected READY_STEREO(6), got ${views.dTelemU32[2]}`);
  assert(views.dTelemU32[4] === PROV_SURVEYED_STEREO, `Suite 1 expected SURVEYED_STEREO(4), got ${views.dTelemU32[4]}`);
  assert(views.wave2TelemU32[40] === 1, `Suite 1 expected PROD_W2_DEPTH stamp valid=1, got ${views.wave2TelemU32[40]}`);
  const epiRms = views.dTelemF32[28];

  const ests = [];
  const trues = [];
  let absRelSum = 0;
  let d1Hit = 0;
  for (let j = 2; j < BOS_H - 2; j++) {
    for (let i = 2; i < BOS_W - 2; i++) {
      const u = j * BOS_W + i;
      if ((views.bosValid[u] & 2) === 0) continue;
      const cx = (i + 0.5) * (W / BOS_W) - 0.5;
      const cy = (j + 0.5) * (H / BOS_H) - 0.5;
      const zc = depthTrue(cx, cy);
      if (
        Math.abs(depthTrue(cx - 5, cy) - zc) > 0.3 ||
        Math.abs(depthTrue(cx + 5, cy) - zc) > 0.3 ||
        Math.abs(depthTrue(cx, cy - 5) - zc) > 0.3 ||
        Math.abs(depthTrue(cx, cy + 5) - zc) > 0.3
      ) {
        continue;
      }
      const zEst = views.bosPlanes[2 * BOS_N + u];
      if (!Number.isFinite(zEst) || zEst <= 0) continue;
      ests.push(zEst);
      trues.push(zc);
      absRelSum += Math.abs(zEst - zc) / zc;
      if (Math.max(zEst / zc, zc / zEst) < 1.25) d1Hit++;
    }
  }
  assert(ests.length > 20000, `Suite 1 too few interior stereo nodes: ${ests.length}`);
  const absRel = absRelSum / ests.length;
  const delta1 = d1Hit / ests.length;
  const r = pearsonR(ests, trues);
  assert(absRel < 0.05, `Suite 1 AbsRel=${(absRel * 100).toFixed(2)}% >= 5%`);
  assert(delta1 >= 0.98, `Suite 1 delta1=${(delta1 * 100).toFixed(2)}% < 98%`);
  assert(r > 0.96, `Suite 1 Pearson r=${r.toFixed(4)} <= 0.96`);
  assert(epiRms < 0.06, `Suite 1 epiRms=${epiRms.toFixed(4)} px >= 0.06 px`);

  let nonZeroAlpha = 0;
  for (let u = 0; u < BOS_N; u++) if (views.bosRgba[u * 4 + 3] > 0) nonZeroAlpha++;
  assert(nonZeroAlpha > 20000 && nonZeroAlpha <= views.dTelemU32[13], `Suite 1 Right-frame bosRgba alpha count ${nonZeroAlpha} out of expected range (validNodes=${views.dTelemU32[13]})`);

  console.log(
    `  [PASS] I7.1 Micro-Baseline Stereo (256x128): nodes=${views.dTelemU32[13]}/${BOS_N}, AbsRel=${(absRel * 100).toFixed(2)}%, delta1=${(delta1 * 100).toFixed(2)}%, r=${r.toFixed(4)}, epiRms=${epiRms.toFixed(4)} px (${dtMs.toFixed(2)} ms)`
  );
}

// ============================================================================
// Suite 2: Wide-Baseline 256x128 Calibrated Stereo (d = 6.25..25 px + occlusion + Right-frame export)
// ============================================================================
function runSuite2WideStereo() {
  const { exp, buf, views } = loadKernel();
  const fPx = 500.0;
  const B = 0.10; // f*B = 50.0 m*px -> Z=2.0m (d=25px), Z=4.0m (d=12.5px), Z=8.0m (d=6.25px)
  const sampleTex = makeNonPeriodicSampler(W, H, 0x57e2e001);

  const layers = [
    { z: 8.0, d: 50.0 / 8.0, test: () => true },
    { z: 4.0, d: 50.0 / 4.0, test: (x, y) => x >= 370 && x <= 520 && y >= 90 && y <= 270 },
    { z: 2.0, d: 50.0 / 2.0, test: (x, y) => x >= 130 && x <= 270 && y >= 95 && y <= 265 },
  ];
  const leftDepthAt = (x, y) => {
    for (let k = layers.length - 1; k >= 0; k--) {
      if (layers[k].test(x, y)) return layers[k].z;
    }
    return 8.0;
  };

  assert(exp.chronos_set_depth_mode(2) === 1, 'set_depth_mode(2) failed');
  assert(exp.chronos_set_depth_calib(fPx, 0.0, B, 36.0, -1) === 1, 'set_depth_calib wide failed');

  const leftU8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const z = leftDepthAt(x, y);
      const tOff = z < 3.0 ? 19 : z < 6.0 ? 7 : 0;
      const v = Math.round(sampleTex(x + tOff, y + tOff) * 255);
      const o = (y * W + x) * 4;
      leftU8[o] = v; leftU8[o + 1] = v; leftU8[o + 2] = v; leftU8[o + 3] = 255;
    }
  }
  assert(exp.chronos_depth_capture_stereo_left() === 1, 'capture_stereo_left wide failed');

  const rightU8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let chosenLayer = layers[0];
      let xL = x + layers[0].d;
      for (let k = layers.length - 1; k >= 1; k--) {
        const candXL = x + layers[k].d;
        if (layers[k].test(candXL, y)) {
          chosenLayer = layers[k];
          xL = candXL;
          break;
        }
      }
      const tOff = chosenLayer.z < 3.0 ? 19 : chosenLayer.z < 6.0 ? 7 : 0;
      const v = Math.round(sampleTex(xL + tOff, y + tOff) * 255);
      const o = (y * W + x) * 4;
      rightU8[o] = v; rightU8[o + 1] = v; rightU8[o + 2] = v; rightU8[o + 3] = 255;
    }
  }

  const t0 = performance.now();
  exp.chronos_tick(1 / 30);
  const dtMs = performance.now() - t0;

  assert(views.dTelemU32[2] === STATE_READY_STEREO, `Suite 2 expected READY_STEREO(6), got ${views.dTelemU32[2]}`);
  const epiRms = views.dTelemF32[28];

  const ests = [];
  const trues = [];
  let absRelSum = 0;
  let d1Hit = 0;
  const nearZ = [];
  const midZ = [];
  const farZ = [];
  let occlStripTotal = 0;
  let occlStripRejected = 0;

  for (let j = 4; j < BOS_H - 4; j++) {
    for (let i = 16; i < BOS_W - 6; i++) {
      const u = j * BOS_W + i;
      const cx = (i + 0.5) * (W / BOS_W) - 0.5;
      const cy = (j + 0.5) * (H / BOS_H) - 0.5;
      const valid = (views.bosValid[u] & 2) !== 0;

      if (cx >= 113 && cx <= 125 && cy >= 108 && cy <= 252) {
        occlStripTotal++;
        if (!valid) occlStripRejected++;
        continue;
      }

      if (!valid) continue;
      const zc = leftDepthAt(cx, cy);
      if (
        leftDepthAt(cx - 10, cy) !== zc ||
        leftDepthAt(cx + 10, cy) !== zc ||
        leftDepthAt(cx, cy - 10) !== zc ||
        leftDepthAt(cx, cy + 10) !== zc
      ) {
        continue;
      }
      const zEst = views.bosPlanes[2 * BOS_N + u];
      ests.push(zEst);
      trues.push(zc);
      absRelSum += Math.abs(zEst - zc) / zc;
      if (Math.max(zEst / zc, zc / zEst) < 1.25) d1Hit++;
      if (zc === 2.0) nearZ.push(zEst);
      else if (zc === 4.0) midZ.push(zEst);
      else farZ.push(zEst);
    }
  }
  assert(nearZ.length > 400 && midZ.length > 400 && farZ.length > 2500, `Suite 2 layer counts too low: near=${nearZ.length} mid=${midZ.length} far=${farZ.length}`);
  const absRel = absRelSum / ests.length;
  const delta1 = d1Hit / ests.length;
  const occlRejectPct = occlStripRejected / Math.max(1, occlStripTotal);
  const medNear = median(nearZ);
  const medMid = median(midZ);
  const medFar = median(farZ);

  assert(absRel < 0.05, `Suite 2 wide stereo AbsRel=${(absRel * 100).toFixed(2)}% >= 5%`);
  assert(delta1 >= 0.95, `Suite 2 wide stereo delta1=${(delta1 * 100).toFixed(2)}% < 95%`);
  assert(Math.abs(medNear - 2.0) < 0.08, `Suite 2 near layer med=${medNear.toFixed(3)} m (expected 2.00)`);
  assert(Math.abs(medMid - 4.0) < 0.15, `Suite 2 mid layer med=${medMid.toFixed(3)} m (expected 4.00)`);
  assert(Math.abs(medFar - 8.0) < 0.25, `Suite 2 far layer med=${medFar.toFixed(3)} m (expected 8.00)`);
  assert(occlRejectPct >= 0.90, `Suite 2 half-occlusion rejection=${(occlRejectPct * 100).toFixed(1)}% < 90%`);

  // Verify Right-image reprojected 64x36 export_field alignment:
  // In Left image, x=255 is inside the Z=2.0m box [130..270].
  // In Right image, the Z=2.0m box projects to x_R in [105..245], while x_R in (245..265) is
  // Left-occluded background and x_R >= 275 is unoccluded Z=8.0m background.
  const shadowCell = 18 * DEPTH_W + Math.floor((255 * DEPTH_W) / W);
  const bgCell = 18 * DEPTH_W + Math.floor((285 * DEPTH_W) / W);
  const fgCell = 18 * DEPTH_W + Math.floor((180 * DEPTH_W) / W);
  const shadowIz = views.dField[shadowCell * 2];
  const bgIz = views.dField[bgCell * 2];
  const fgIz = views.dField[fgCell * 2];
  assert(shadowIz === 0 || 1.0 / shadowIz > 6.0, `Suite 2 Right-frame (255,180) must NOT leak 2.0m foreground, got ${(1 / shadowIz).toFixed(2)}m`);
  assert(bgIz > 0 && Math.abs(1.0 / bgIz - 8.0) < 0.6, `Suite 2 Right-frame (285,180) must read ~8.0m background, got ${(1 / bgIz).toFixed(2)}m`);
  assert(fgIz > 0 && Math.abs(1.0 / fgIz - 2.0) < 0.15, `Suite 2 Right-frame (180,180) must read ~2.0m foreground, got ${(1 / fgIz).toFixed(2)}m`);

  console.log(
    `  [PASS] I7.2 Wide-Baseline Stereo (d=6.25..25 px): Z=[${medNear.toFixed(2)}, ${medMid.toFixed(2)}, ${medFar.toFixed(2)}] m, Right(255,180)=${shadowIz > 0 ? (1 / shadowIz).toFixed(2) + 'm' : 'occluded'}, Right(285,180)=${(1 / bgIz).toFixed(2)}m, AbsRel=${(absRel * 100).toFixed(2)}%, delta1=${(delta1 * 100).toFixed(2)}%, occlReject=${(occlRejectPct * 100).toFixed(1)}%, epiRms=${epiRms.toFixed(4)} px (${dtMs.toFixed(2)} ms)`
  );
}

// ============================================================================
// Suites 3 & 4: Monocular Motion Parallax + Independent Movers + Fail-Closed Gates
// ============================================================================
function runSuite3And4MonoAndNegativeGates() {
  const fPx = 500.0;
  const dt = 1 / 30;
  const basePts = makeBaseLandmarks(fPx, 4.5, 14.5, 0xdecaf007);
  const movers = [
    { X: -0.9, Y: -0.5, Z: 7.0, vx: 0.0, vy: 1.5, vz: 0.0, r: 3.5, col: [240, 235, 150] },
    { X: 0.9, Y: 0.5, Z: 8.5, vx: 0.0, vy: -1.6, vz: 0.0, r: 3.5, col: [240, 235, 150] },
    { X: -0.2, Y: -0.8, Z: 6.5, vx: 1.8, vy: 1.2, vz: 0.0, r: 3.5, col: [240, 235, 150] },
    { X: 0.3, Y: 0.8, Z: 9.0, vx: -1.7, vy: -1.3, vz: 0.0, r: 3.5, col: [240, 235, 150] },
  ];

  // --- Suite 3: Lateral Motion Parallax (v = 0.30 m/s) ---
  {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    assert(exp.chronos_set_depth_mode(1) === 1, 'set_depth_mode(1) failed');
    assert(exp.chronos_set_depth_calib(fPx, 0.30, 0.0, 32.0, 0) === 1, 'set_depth_calib mono failed');

    let lastProj = [];
    for (let f = 0; f < 36; f++) {
      const t = f * dt;
      const pts = basePts.concat(
        movers.map((m) => ({
          X: m.X + m.vx * t,
          Y: m.Y + m.vy * t,
          Z: m.Z + m.vz * t,
          moving: true,
          r: m.r,
          col: m.col,
        }))
      );
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      lastProj = renderScene3D(u8, pts, [0.30 * t, 0, 0], 0.0, fPx);
      exp.chronos_tick(dt);
    }

    assert(views.dTelemU32[2] === STATE_READY_PARALLAX, `Suite 3 expected READY_PARALLAX(2), got ${views.dTelemU32[2]}`);
    assert(views.dTelemU32[4] === PROV_MONO_METRIC, `Suite 3 expected MONO_METRIC(2), got ${views.dTelemU32[4]}`);
    assert(views.dTelemU32[9] > 100, `Suite 3 expected >100 populated 64x36 field cells, got ${views.dTelemU32[9]}`);

    const ests = [];
    const trues = [];
    let absRelSum = 0;
    let d1Hit = 0;
    let moverTracked = 0;
    let moverRejected = 0;

    for (let i = 0; i < MAX_MICRO; i++) {
      const fl = views.dFlags[i];
      if ((fl & (FLAG_VALID | FLAG_MOVING_OUTLIER)) === 0) continue;
      const mb = i * 8;
      if (views.micro[mb + 5] !== 1) continue;
      const px = views.micro[mb + 1];
      const py = views.micro[mb + 2];
      let best = null;
      let bestD2 = 9.0 * 9.0;
      for (const q of lastProj) {
        const d2 = (q.u - px) ** 2 + (q.v - py) ** 2;
        if (d2 < bestD2) {
          bestD2 = d2;
          best = q;
        }
      }
      if (!best) continue;
      if (best.moving) {
        moverTracked++;
        if ((fl & FLAG_MOVING_OUTLIER) !== 0 && (fl & FLAG_VALID) === 0) moverRejected++;
      } else if ((fl & FLAG_VALID) !== 0) {
        const invZ = views.dTracks[i * 2];
        if (invZ > 1e-6) {
          const zEst = 1.0 / invZ;
          ests.push(zEst);
          trues.push(best.Zc);
          absRelSum += Math.abs(zEst - best.Zc) / best.Zc;
          if (Math.max(zEst / best.Zc, best.Zc / zEst) < 1.25) d1Hit++;
        }
      }
    }

    assert(ests.length >= 60, `Suite 3 too few matched static tracks: ${ests.length}`);
    const absRel = absRelSum / ests.length;
    const delta1 = d1Hit / ests.length;
    const r = pearsonR(ests, trues);
    assert(absRel < 0.12, `Suite 3 mono parallax AbsRel=${(absRel * 100).toFixed(2)}% >= 12%`);
    assert(delta1 >= 0.90, `Suite 3 mono parallax delta1=${(delta1 * 100).toFixed(2)}% < 90%`);
    assert(r > 0.96, `Suite 3 mono parallax Pearson r=${r.toFixed(4)} <= 0.96`);
    assert(moverTracked >= 3 && moverRejected === moverTracked, `Suite 3 moving outlier rejection ${moverRejected}/${moverTracked}`);

    console.log(
      `  [PASS] I7.3 Monocular Parallax (in-kernel): ${ests.length} static tracks, AbsRel=${(absRel * 100).toFixed(2)}%, delta1=${(delta1 * 100).toFixed(2)}%, r=${r.toFixed(4)}, movers rejected=${moverRejected}/${moverTracked}, rotResid=${views.dTelemF32[16].toFixed(2)} px/s`
    );
  }

  // --- Suite 4A: Negative Control: Pure Tripod Yaw Rotation (t = 0) ---
  {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(fPx, 0.30, 0.0, 32.0, 0);

    for (let f = 0; f < 30; f++) {
      const t = f * dt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      renderScene3D(u8, basePts, [0, 0, 0], 0.034 * t, fPx);
      exp.chronos_tick(dt);
      assert(views.dTelemU32[5] === 0, `Suite 4A frame ${f} leaked ${views.dTelemU32[5]} depths on pure rotation`);
    }

    assert(
      views.dTelemU32[2] === STATE_REFUSED_PURE_ROTATION,
      `Suite 4A pure rotation expected REFUSED_PURE_ROTATION(4), got state=${views.dTelemU32[2]} rotResid=${views.dTelemF32[16].toFixed(3)} rotRatio=${views.dTelemF32[17].toFixed(3)}`
    );
    assert(views.dTelemU32[5] === 0 && views.dTelemU32[9] === 0, 'Suite 4A must zero all depths and field cells on pure rotation');
    console.log(
      `  [PASS] I7.4A Fail-Closed Pure Rotation Gate: state=REFUSED_PURE_ROTATION, medRotResid=${views.dTelemF32[16].toFixed(3)} px/s, rotRatio=${views.dTelemF32[17].toFixed(4)}, depths=0`
    );
  }

  // --- Suite 4B: Negative Control: Stationary Camera ---
  {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(fPx, 0.30, 0.0, 32.0, 0);

    for (let f = 0; f < 26; f++) {
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      const camX = f < 12 ? 0.30 * (f * dt) : 0.30 * (11 * dt);
      renderScene3D(u8, basePts, [camX, 0, 0], 0.0, fPx);
      exp.chronos_tick(dt);
    }

    assert(
      views.dTelemU32[2] === STATE_REFUSED_NO_BASELINE,
      `Suite 4B stationary expected REFUSED_NO_BASELINE(5), got state=${views.dTelemU32[2]} medSpeed=${views.dTelemF32[18].toFixed(3)}`
    );
    assert(views.dTelemU32[5] === 0 && views.dTelemU32[9] === 0, 'Suite 4B must zero all depths on stationary camera');
    console.log(
      `  [PASS] I7.4B Fail-Closed No-Baseline Gate: state=REFUSED_NO_BASELINE, medSpeed=${views.dTelemF32[18].toFixed(3)} px/s, depths=0`
    );
  }
}

// ============================================================================
// Suite 5: Real H.264 Clip Intrinsics-Free FoE & 1/r(t) Optical Looming TTC
// ============================================================================
function runSuite5RealClipFoeTtc() {
  const clipPath = path.join(ROOT, 'web', 'samples', 'perspective_flyby_decoy.mp4');
  if (!fs.existsSync(clipPath)) {
    console.log('  [SKIP] I7.5 Real H.264 FoE/TTC: sample clip missing');
    return;
  }
  const raw = execFileSync(
    'ffmpeg',
    ['-v', 'error', '-i', clipPath, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${W}x${H}`, '-vframes', '120', '-'],
    { maxBuffer: 140 * 1024 * 1024 }
  );
  const frameBytes = W * H * 4;
  const nFrames = Math.floor(raw.length / frameBytes);
  assert(nFrames >= 80, `Suite 5 decoded too few frames: ${nFrames}`);

  const { exp, buf, views } = loadKernel();
  exp.chronos_set_swarm_mode(2);
  exp.chronos_set_depth_mode(1);
  exp.chronos_set_depth_calib(0.0, 0.0, 0.0, 32.0, 0);

  let maxFoeInliers = 0;
  let maxTtcTracks = 0;
  let bestR2Ever = 0;
  let bestFoe = [0, 0];
  let bestMinTtc = Infinity;

  for (let f = 0; f < nFrames; f++) {
    const dst = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), frameBytes);
    dst.set(raw.subarray(f * frameBytes, (f + 1) * frameBytes));
    exp.chronos_tick(1 / 25);

    const st = views.dTelemU32[2];
    const ttcCnt = views.dTelemU32[7];
    const foeIn = views.dTelemU32[8];
    const r2 = views.dTelemF32[25];
    const minTtc = views.dTelemF32[24];
    // Uncalibrated f=0 must NEVER fabricate parallax depth!
    assert(views.dTelemU32[5] === 0 && st !== STATE_READY_PARALLAX, `Suite 5 uncalibrated f=0 must not emit parallax depth (frame ${f}, st=${st})`);
    if (foeIn > maxFoeInliers) {
      maxFoeInliers = foeIn;
      bestFoe = [views.dTelemF32[22], views.dTelemF32[23]];
    }
    if (ttcCnt > maxTtcTracks) maxTtcTracks = ttcCnt;
    if (ttcCnt > 0 && r2 > bestR2Ever) {
      bestR2Ever = r2;
      if (minTtc > 0 && minTtc < bestMinTtc) bestMinTtc = minTtc;
    }
    assert(
      st === STATE_WARMING || st === STATE_READY_TTC_ONLY || st === STATE_REFUSED_NO_BASELINE,
      `Suite 5 unexpected depth state ${st}`
    );
  }

  const foeErrPx = Math.hypot(bestFoe[0] - 393.0, bestFoe[1] - 205.0);
  assert(maxFoeInliers >= 4, `Suite 5 FoE inliers too low: ${maxFoeInliers}`);
  assert(foeErrPx < 55.0, `Suite 5 FoE (${bestFoe[0].toFixed(1)}, ${bestFoe[1].toFixed(1)}) too far from (393, 205): err=${foeErrPx.toFixed(1)} px`);
  assert(maxTtcTracks >= 2, `Suite 5 expected >= 2 valid TTC tracks, got ${maxTtcTracks}`);
  assert(bestR2Ever >= 0.98, `Suite 5 expected best TTC R^2 >= 0.98, got ${bestR2Ever.toFixed(4)}`);
  assert(Number.isFinite(bestMinTtc) && bestMinTtc > 0.1 && bestMinTtc < 30.0, `Suite 5 invalid minTTC=${bestMinTtc}`);

  console.log(
    `  [PASS] I7.5 Real H.264 Intrinsics-Free FoE & TTC (perspective_flyby_decoy.mp4): FoE=(${bestFoe[0].toFixed(1)}, ${bestFoe[1].toFixed(1)}) px (inliers=${maxFoeInliers}), TTC tracks=${maxTtcTracks}, best R^2=${bestR2Ever.toFixed(4)}, minTTC=${bestMinTtc.toFixed(2)} s`
  );
}

// ============================================================================
// Suite 6: External Depth-Plane Ingest (128x64) & Provenance Firewall + Hero-16 Export
// ============================================================================
function runSuite6ExternalProvenanceAndHero() {
  const { exp, buf, views } = loadKernel();
  assert(exp.chronos_set_depth_mode(3) === 1, 'set_depth_mode(3) failed');

  const paintTwoDiscs = (u8) => {
    u8.fill(32);
    for (let i = 3; i < u8.length; i += 4) u8[i] = 255;
    for (const [cx, cy] of [[160, 180], [480, 180]]) {
      for (let dy = -12; dy <= 12; dy++) {
        for (let dx = -12; dx <= 12; dx++) {
          if (dx * dx + dy * dy > 144) continue;
          const o = ((cy + dy) * W + (cx + dx)) * 4;
          u8[o] = 245; u8[o + 1] = 240; u8[o + 2] = 60;
        }
      }
    }
  };
  paintTwoDiscs(new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4));
  exp.chronos_tick(1 / 30);
  const tidFront = exp.chronos_seed_target(140, 160, 40, 40);
  const tidBack = exp.chronos_seed_target(460, 160, 40, 40);
  assert(tidFront > 0 && tidBack > 0 && tidFront !== tidBack, `Suite 6 hero seed failed: ${tidFront}, ${tidBack}`);

  const ew = 128, eh = 64;
  for (let y = 0; y < eh; y++) {
    for (let x = 0; x < ew; x++) {
      if (x < 4 && y < 4) {
        views.dIngest[y * ew + x] = x === 0 ? NaN : (x === 1 ? -5.0 : 0.0);
      } else {
        views.dIngest[y * ew + x] = x < 64 ? 2.0 : 10.0;
      }
    }
  }

  assert(exp.chronos_set_depth_external_meta(ew, eh, 500.0, 1, 42) === 1, 'set_depth_external_meta LiDAR failed');
  paintTwoDiscs(new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4));
  exp.chronos_tick(1 / 30);

  assert(views.dTelemU32[2] === STATE_READY_EXTERNAL, `Suite 6A expected READY_EXTERNAL(7), got ${views.dTelemU32[2]}`);
  assert(views.dTelemU32[4] === PROV_SURVEYED_LIDAR, `Suite 6A expected SURVEYED_LIDAR(5), got ${views.dTelemU32[4]}`);
  assert((views.dTelemU32[3] & REASON_ASSUMED_NEURAL_PRIOR) === 0, 'Suite 6A SURVEYED_LIDAR must NOT assert neural prior reason bit');
  assert(views.dTelemU32[12] === 42, `Suite 6A seq mismatch: ${views.dTelemU32[12]}`);

  assert(views.dField[0] === 0 && views.dField[1] === 0, 'Suite 6A invalid top-left NaN/<=0 cell must stay 0');
  const leftIz = views.dField[(18 * DEPTH_W + 16) * 2];
  const rightIz = views.dField[(18 * DEPTH_W + 48) * 2];
  assert(Math.abs(1.0 / leftIz - 2.0) < 0.02, `Suite 6A left Z=${(1 / leftIz).toFixed(3)} != 2.0`);
  assert(Math.abs(1.0 / rightIz - 10.0) < 0.05, `Suite 6A right Z=${(1 / rightIz).toFixed(3)} != 10.0`);

  let zFront = 0, zBack = 0;
  for (let m = 0; m < 16; m++) {
    const base = 32 + m * 56;
    if (views.telem[base] < 0.5) continue;
    const tid = Math.round(views.telem[base + 1]);
    const hz = views.dHero[m * 4];
    const hProv = Math.round(views.dHero[m * 4 + 3]);
    if (tid === tidFront) {
      zFront = hz;
      assert(hProv === PROV_SURVEYED_LIDAR, `Suite 6A front hero prov=${hProv}`);
    } else if (tid === tidBack) {
      zBack = hz;
      assert(hProv === PROV_SURVEYED_LIDAR, `Suite 6A back hero prov=${hProv}`);
    }
  }
  assert(Math.abs(zFront - 2.0) < 0.1 && Math.abs(zBack - 10.0) < 0.25 && zFront < zBack, `Suite 6A Hero depth export failed: front=${zFront}, back=${zBack}`);

  // Case B: ASSUMED_NEURAL_PRIOR
  assert(exp.chronos_set_depth_external_meta(ew, eh, 500.0, 3, 43) === 1, 'set_depth_external_meta Neural failed');
  paintTwoDiscs(new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4));
  exp.chronos_tick(1 / 30);

  assert(views.dTelemU32[4] === PROV_ASSUMED_NEURAL_PRIOR, `Suite 6B expected ASSUMED_NEURAL_PRIOR(6), got ${views.dTelemU32[4]}`);
  assert((views.dTelemU32[3] & REASON_ASSUMED_NEURAL_PRIOR) !== 0, 'Suite 6B provenance firewall MUST assert REASON_ASSUMED_NEURAL_PRIOR');

  // Strict parameter validation
  assert(exp.chronos_set_depth_calib(5.0, 0.3, 0.0, 32.0, 0) === 0, 'f_px in (0,10) must fail');
  assert(exp.chronos_set_depth_calib(500.0, 0.3, 0.0, 0.5, 0) === 0, 'max_disp_px < 1 must fail');
  assert(exp.chronos_set_depth_calib(500.0, 0.3, 0.0, 65.0, 0) === 0, 'max_disp_px > 64 must fail');
  assert(exp.chronos_set_depth_calib(500.0, -0.1, 0.0, 32.0, 0) === 0, 'negative speed must fail');
  assert(exp.chronos_set_depth_calib(500.0, 0.0, -0.1, 32.0, 0) === 0, 'negative baseline must fail');
  assert(exp.chronos_set_depth_external_meta(256, 64, 500.0, 1, 44) === 0, 'Oversized external w must fail');
  assert(exp.chronos_set_depth_external_meta(64, 32, 500.0, 0, 44) === 0, 'Provenance=0 must fail');
  assert(exp.chronos_set_depth_external_meta(64, 32, NaN, 1, 44) === 0, 'NaN f_px must fail');
  assert(exp.chronos_set_depth_mode(5) === 0, 'Invalid depth mode=5 must fail');

  console.log(
    `  [PASS] I7.6 External Depth Plane & Provenance Firewall: 128x64->64x36 OK, Hero Z=[${zFront.toFixed(2)}m < ${zBack.toFixed(2)}m], LiDAR vs ASSUMED_NEURAL_PRIOR + strict arg validation verified`
  );
}

// ============================================================================
// Suite 7: Monocular Adversarial & Precision Regressions (Sol 6.2 & Opus 5.5)
// ============================================================================
function runSuite7MonoAdversarial() {
  const fPx = 500.0;
  const dt = 1 / 30;
  const basePts = makeBaseLandmarks(fPx, 4.5, 14.5, 0xdecaf007);

  // 7A. Mixed lateral + yaw (incl. small 0.003 / -0.004 rad/s) & lateral + pitch (0.01 rad/s) MUST fail closed
  for (const [yawRate, pitchRate] of [[0.003, 0], [-0.004, 0], [0.01, 0], [0.02, 0], [-0.02, 0], [0.034, 0], [0, 0.01]]) {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(fPx, 0.30, 0.0, 32.0, 0);
    for (let f = 0; f < 30; f++) {
      const t = f * dt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      renderScene3D(u8, basePts, [0.30 * t, 0, 0], yawRate * t, fPx, pitchRate * t);
      exp.chronos_tick(dt);
      assert(views.dTelemU32[5] === 0, `Suite 7A mixed lateral+(yaw=${yawRate},pitch=${pitchRate}) frame ${f} leaked ${views.dTelemU32[5]} depths (state=${views.dTelemU32[2]}, prov=${views.dTelemU32[4]})`);
    }
    assert(views.dTelemU32[2] === STATE_REFUSED_PURE_ROTATION, `Suite 7A (yaw=${yawRate},pitch=${pitchRate}) expected REFUSED_PURE_ROTATION(4), got ${views.dTelemU32[2]}`);
    assert((views.dTelemU32[3] & REASON_MIXED_ROT_TRANS) !== 0 || views.dTelemU32[3] !== 0, `Suite 7A (yaw=${yawRate},pitch=${pitchRate}) missing refusal reason bit`);
  }

  // 7B. Oblique translation T=(0.30, 0, 0.30) with zero yaw MUST NOT emit biased lateral metric depth
  {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(fPx, 0.30, 0.0, 32.0, 0);
    for (let f = 0; f < 30; f++) {
      const t = f * dt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      renderScene3D(u8, basePts, [0.30 * t, 0, 0.30 * t], 0.0, fPx);
      exp.chronos_tick(dt);
      assert(views.dTelemU32[5] === 0, `Suite 7B oblique T=(0.3,0,0.3) frame ${f} leaked ${views.dTelemU32[5]} metric depths`);
    }
  }

  // 7C. Pure yaw with wrong/unknown focal length (f_true=250, calib f=0 and calib f=500) MUST refuse
  const widePts = makeBaseLandmarks(250.0, 4.5, 14.5, 0xdecaf009);
  for (const calibF of [0.0, 500.0]) {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(calibF, 0.30, 0.0, 32.0, 0);
    for (let f = 0; f < 30; f++) {
      const t = f * dt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      renderScene3D(u8, widePts, [0, 0, 0], 0.04 * t, 250.0);
      exp.chronos_tick(dt);
      assert(views.dTelemU32[5] === 0, `Suite 7C pure yaw f_true=250 calibF=${calibF} frame ${f} leaked ${views.dTelemU32[5]} depths`);
    }
  }

  // 7D. Exact vertical lateral translation (Tx=0, Ty=0.30 m/s) proves stable principal eigenvector
  {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(fPx, 0.30, 0.0, 32.0, 0);
    let lastProj = [];
    for (let f = 0; f < 30; f++) {
      const t = f * dt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      lastProj = renderScene3D(u8, basePts, [0, 0.30 * t, 0], 0.0, fPx);
      exp.chronos_tick(dt);
    }
    assert(views.dTelemU32[2] === STATE_READY_PARALLAX && views.dTelemU32[4] === PROV_MONO_METRIC, `Suite 7D vertical flow failed: st=${views.dTelemU32[2]} prov=${views.dTelemU32[4]}`);
    const ratios = [];
    for (let i = 0; i < MAX_MICRO; i++) {
      if ((views.dFlags[i] & FLAG_VALID) === 0) continue;
      const mb = i * 8;
      if (views.micro[mb + 5] !== 1) continue;
      const px = views.micro[mb + 1], py = views.micro[mb + 2];
      let best = null, bestD2 = 81;
      for (const q of lastProj) {
        const d2 = (q.u - px) ** 2 + (q.v - py) ** 2;
        if (d2 < bestD2) { bestD2 = d2; best = q; }
      }
      if (best && views.dTracks[i * 2] > 1e-6) ratios.push((1.0 / views.dTracks[i * 2]) / best.Zc);
    }
    const medRatio = median(ratios);
    assert(ratios.length >= 40 && Math.abs(medRatio - 1.0) < 0.06, `Suite 7D vertical flow med(Z_est/Z_true)=${medRatio.toFixed(3)} (n=${ratios.length})`);
  }

  // 7E. Low-FPS 5 fps (200 ms) and 240 ms unclamped dt accuracy (no 150 ms clamp bias!)
  for (const slowDt of [0.20, 0.24]) {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    // Keep per-frame pixel displacement <= ~4 px so tracker tracks cleanly at 5 fps: v = 0.06 m/s
    const vSlow = 0.06;
    exp.chronos_set_depth_calib(fPx, vSlow, 0.0, 32.0, 0);
    let lastProj = [];
    for (let f = 0; f < 22; f++) {
      const t = f * slowDt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      lastProj = renderScene3D(u8, basePts, [vSlow * t, 0, 0], 0.0, fPx);
      exp.chronos_tick(slowDt);
    }
    assert(views.dTelemU32[2] === STATE_READY_PARALLAX && views.dTelemU32[4] === PROV_MONO_METRIC, `Suite 7E dt=${slowDt}s failed: st=${views.dTelemU32[2]}`);
    const ratios = [];
    for (let i = 0; i < MAX_MICRO; i++) {
      if ((views.dFlags[i] & FLAG_VALID) === 0) continue;
      const mb = i * 8;
      if (views.micro[mb + 5] !== 1) continue;
      const px = views.micro[mb + 1], py = views.micro[mb + 2];
      let best = null, bestD2 = 81;
      for (const q of lastProj) {
        const d2 = (q.u - px) ** 2 + (q.v - py) ** 2;
        if (d2 < bestD2) { bestD2 = d2; best = q; }
      }
      if (best && views.dTracks[i * 2] > 1e-6) ratios.push((1.0 / views.dTracks[i * 2]) / best.Zc);
    }
    const medRatio = median(ratios);
    assert(ratios.length >= 40 && medRatio >= 0.94 && medRatio <= 1.06, `Suite 7E dt=${slowDt}s biased med(Z_est/Z_true)=${medRatio.toFixed(3)} (expected ~1.00, NOT 0.75/0.625)`);
  }

  // 7F. Shallow scene Z in [10, 14] m hysteresis stability (no rapid state oscillation)
  {
    const shallowPts = makeBaseLandmarks(fPx, 10.0, 4.0, 0x54a11001);
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(fPx, 0.30, 0.0, 32.0, 0);
    const states = [];
    for (let f = 0; f < 34; f++) {
      const t = f * dt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      renderScene3D(u8, shallowPts, [0.30 * t, 0, 0], 0.0, fPx);
      exp.chronos_tick(dt);
      if (f >= 16) states.push(views.dTelemU32[2]);
    }
    let flips = 0;
    for (let i = 1; i < states.length; i++) if (states[i] !== states[i - 1]) flips++;
    assert(flips <= 2, `Suite 7F shallow scene flickered ${flips} times across steady frames: [${states.join(',')}]`);
  }

  // 7G. Synthetic pure forward approach (known TTC + radial metric depth) vs receding (0 TTC)
  {
    const fwdPts = makeBaseLandmarks(fPx, 6.0, 8.0, 0xf02d0001);
    const vz = 1.2;
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(fPx, vz, 0.0, 32.0, 0);
    let lastProj = [];
    const nF = 44;
    for (let f = 0; f < nF; f++) {
      const t = f * dt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      lastProj = renderScene3D(u8, fwdPts, [0, 0, vz * t], 0.0, fPx);
      exp.chronos_tick(dt);
    }
    const tEnd = (nF - 1) * dt;
    assert(views.dTelemU32[2] === STATE_READY_PARALLAX && views.dTelemU32[4] === PROV_MONO_METRIC, `Suite 7G forward radial failed: st=${views.dTelemU32[2]} prov=${views.dTelemU32[4]}`);
    assert(views.dTelemU32[7] >= 4, `Suite 7G forward approach expected >=4 TTC tracks, got ${views.dTelemU32[7]}`);
    // Compare matched radial depth & TTC against exact Zc and Zc / vz
    const zErrs = [];
    const ttcErrs = [];
    for (let i = 0; i < MAX_MICRO; i++) {
      const fl = views.dFlags[i];
      if ((fl & (FLAG_VALID | FLAG_TTC_VALID)) === 0) continue;
      const mb = i * 8;
      if (views.micro[mb + 5] !== 1) continue;
      const px = views.micro[mb + 1], py = views.micro[mb + 2];
      let best = null, bestD2 = 81;
      for (const q of lastProj) {
        const d2 = (q.u - px) ** 2 + (q.v - py) ** 2;
        if (d2 < bestD2) { bestD2 = d2; best = q; }
      }
      if (!best) continue;
      if ((fl & FLAG_VALID) !== 0 && views.dTracks[i * 2] > 1e-6) {
        assert((fl & FLAG_METRIC) !== 0, 'Suite 7G radial metric track missing FLAG_METRIC');
        zErrs.push(Math.abs(1.0 / views.dTracks[i * 2] - best.Zc) / best.Zc);
      }
      if ((fl & FLAG_TTC_VALID) !== 0 && views.dTracks[i * 2 + 1] > 0) {
        const trueTtc = best.Zc / vz;
        ttcErrs.push(Math.abs(views.dTracks[i * 2 + 1] - trueTtc) / trueTtc);
      }
    }
    const medZErr = median(zErrs);
    const medTtcErr = median(ttcErrs);
    assert(zErrs.length >= 25 && medZErr < 0.12, `Suite 7G radial depth medAbsRel=${(medZErr * 100).toFixed(2)}% (n=${zErrs.length})`);
    assert(ttcErrs.length >= 4 && medTtcErr < 0.15, `Suite 7G known TTC medAbsRel=${(medTtcErr * 100).toFixed(2)}% (n=${ttcErrs.length}, tEnd=${tEnd.toFixed(2)})`);

    // Receding camera (no contact) must yield 0 TTC tracks
    exp.chronos_reset_tracks();
    for (let f = 0; f < 36; f++) {
      const t = f * dt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      renderScene3D(u8, fwdPts, [0, 0, -vz * t], 0.0, fPx);
      exp.chronos_tick(dt);
    }
    assert(views.dTelemU32[7] === 0, `Suite 7G receding camera must yield 0 TTC tracks, got ${views.dTelemU32[7]}`);
  }

  // 7H. Opus H1: Forward motion + yaw/pitch (incl. 90-frame shallow Z[10,14], Z[8,12], Z[12,16]!) MUST refuse,
  // while pure centered/off-center 3D controls STILL pass with low p90 error (near-FoE excluded)!
  {
    const shallow10_14 = makeBaseLandmarks(fPx, 10.0, 4.0, 0x54a11001);
    const shallow8_12 = makeBaseLandmarks(fPx, 8.0, 4.0, 0x54a11002);
    const shallow12_16 = makeBaseLandmarks(fPx, 12.0, 4.0, 0x54a11003);
    const badRadialCases = [
      ['fwd0.6+yaw0.01', basePts, [0, 0, 0.6], 0.01, 0.0, 0.6, 40],
      ['fwd0.6+yaw0.03', basePts, [0, 0, 0.6], 0.03, 0.0, 0.6, 40],
      ['fwd0.6+pitch0.01', basePts, [0, 0, 0.6], 0.0, 0.01, 0.6, 40],
      ['shallow10_14+yaw0.01_90f', shallow10_14, [0, 0, 0.6], 0.01, 0.0, 0.6, 90],
      ['shallow10_14+yaw0.0075_60f', shallow10_14, [0, 0, 0.6], 0.0075, 0.0, 0.6, 60],
      ['shallow8_12+yaw0.01_60f', shallow8_12, [0, 0, 0.6], 0.01, 0.0, 0.6, 60],
      ['shallow12_16+yaw-0.01_60f', shallow12_16, [0, 0, 0.6], -0.01, 0.0, 0.6, 60],
      ['shallow10_14+0.85+yaw0.015', shallow10_14, [0, 0, 0.85], 0.015, 0.0, 0.85, 60],
    ];
    for (const [name, pts, vel, yawR, pitchR, spd, nF] of badRadialCases) {
      const { exp, buf, views } = loadKernel();
      exp.chronos_set_swarm_mode(2);
      exp.chronos_set_depth_mode(1);
      exp.chronos_set_depth_calib(fPx, spd, 0.0, 32.0, 0);
      for (let f = 0; f < nF; f++) {
        const t = f * dt;
        const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
        renderScene3D(u8, pts, [vel[0] * t, vel[1] * t, vel[2] * t], yawR * t, fPx, pitchR * t);
        exp.chronos_tick(dt);
        assert(views.dTelemU32[5] === 0, `Suite 7H ${name} frame ${f} leaked ${views.dTelemU32[5]} metric depths (st=${views.dTelemU32[2]}, prov=${views.dTelemU32[4]})`);
      }
    }

    // Pure centered & off-center radial controls MUST still emit accurate MONO_METRIC with p90 < 10%!
    for (const [name, vel] of [['pure-fwd', [0, 0, 0.60]], ['offcenter-x', [0.15, 0, 0.60]], ['offcenter-xy', [0.20, 0.10, 0.60]]]) {
      const { exp, buf, views } = loadKernel();
      const spd = Math.hypot(vel[0], vel[1], vel[2]);
      exp.chronos_set_swarm_mode(2);
      exp.chronos_set_depth_mode(1);
      exp.chronos_set_depth_calib(fPx, spd, 0.0, 32.0, 0);
      let lastProj = [];
      for (let f = 0; f < 36; f++) {
        const t = f * dt;
        const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
        lastProj = renderScene3D(u8, basePts, [vel[0] * t, vel[1] * t, vel[2] * t], 0.0, fPx);
        exp.chronos_tick(dt);
      }
      assert(views.dTelemU32[2] === STATE_READY_PARALLAX && views.dTelemU32[4] === PROV_MONO_METRIC,
        `Suite 7H pure control ${name} refused: st=${views.dTelemU32[2]} prov=${views.dTelemU32[4]}`);
      const errs = [];
      for (let i = 0; i < MAX_MICRO; i++) {
        if ((views.dFlags[i] & FLAG_VALID) === 0 || views.dTracks[i * 2] <= 1e-6) continue;
        const mb = i * 8;
        if (views.micro[mb + 5] !== 1) continue;
        const px = views.micro[mb + 1], py = views.micro[mb + 2];
        let best = null, bestD2 = 81;
        for (const q of lastProj) {
          const d2 = (q.u - px) ** 2 + (q.v - py) ** 2;
          if (d2 < bestD2) { bestD2 = d2; best = q; }
        }
        if (best) errs.push(Math.abs(1.0 / views.dTracks[i * 2] - best.Zc) / best.Zc);
      }
      errs.sort((a, b) => a - b);
      const medErr = median(errs);
      const p90Err = errs[Math.floor(errs.length * 0.9)];
      assert(errs.length >= 25 && medErr < 0.06 && p90Err < 0.10,
        `Suite 7H pure control ${name} med=${(medErr * 100).toFixed(2)}% p90=${(p90Err * 100).toFixed(2)}% (n=${errs.length})`);
    }
  }

  // 7I. Sol B: Uninterrupted 145-frame forward approach MUST NOT suffer frame-120 TTC anchor-refresh dropout
  {
    const longPts = makeBaseLandmarks(fPx, 10.0, 14.0, 0x10490001);
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(0.0, 0.0, 0.0, 32.0, 0);
    let minTtcAfter115 = Infinity;
    for (let f = 0; f < 145; f++) {
      const t = f * dt;
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      renderScene3D(u8, longPts, [0, 0, 1.0 * t], 0.0, fPx);
      exp.chronos_tick(dt);
      if (f >= 115) {
        if (views.dTelemU32[7] < minTtcAfter115) minTtcAfter115 = views.dTelemU32[7];
      }
    }
    assert(minTtcAfter115 >= 4, `Suite 7I frame-120 TTC dropout regression: min ttcTracks in frames 115..144 was ${minTtcAfter115}`);
  }

  console.log(
    '  [PASS] I7.7 Monocular Adversarial Regressions: small yaw/pitch + fwd+rot refused, off-center pure controls OK, oblique/wrong-f refused, 5fps/240ms unbiased, shallow stable, 145f TTC continuous'
  );
}

// ============================================================================
// Suite 8: Stereo Adversarial, Uniqueness, Sign, Alignment, SD & BOS Isolation
// ============================================================================
function runSuite8StereoAdversarialAndBosIsolation() {
  const fPx = 500.0;

  // 8A. Periodic textures MUST be rejected for auto AND forced sign across even, odd, non-integer, sharp-noise & half-image periods (HD + SD)!
  for (const [resMode, period, shift, maxD, sign, halfTiled, sharpKind] of [
    [1, 12.0, 6.0, 36.0, -1, false, 'smooth'],
    [1, 42.0, 30.0, 32.0, 0, false, 'smooth'],
    [1, 42.0, 30.0, 32.0, 1, false, 'smooth'],
    [1, 35.0, 30.0, 32.0, 0, false, 'smooth'],
    [1, 45.0, 30.0, 32.0, 0, false, 'smooth'],
    [1, 55.0, 30.0, 32.0, 0, false, 'smooth'],
    [1, 65.0, 30.0, 64.0, 0, false, 'smooth'],
    [1, 37.3, 33.3, 32.0, 0, false, 'bilinear'],
    [1, 41.5, 37.5, 32.0, 0, false, 'bilinear'],
    [1, 50.7, 46.7, 32.0, 1, false, 'bilinear'],
    [1, 39.5, -29.5, 32.0, -1, false, 'nearest'],
    [1, 41.5, 37.5, 32.0, 0, false, 'nearest'],
    [1, 35.0, 30.0, 32.0, 0, true, 'smooth'],
    [1, 41.5, 37.5, 32.0, 0, true, 'bilinear'],
    [0, 41.5, -17.5, 32.0, 0, false, 'nearest'],
    [0, 65.5, -61.35, 64.0, 0, false, 'nearest'],
    [0, 97.5, -93.35, 64.0, 0, true, 'bilinear'],
    [0, 127.0, -122.0, 64.0, -1, true, 'bilinear'],
  ]) {
    const fw = resMode === 0 ? 320 : W;
    const fh = resMode === 0 ? 180 : H;
    const { exp, buf, views } = loadKernel(resMode);
    exp.chronos_set_depth_mode(2);
    exp.chronos_set_depth_calib(fPx, 0.0, 0.10, maxD, sign);
    const tileNoise = makeNonPeriodicSampler(Math.ceil(period) + 8, fh, 0x711e0001);
    const pCeil = Math.ceil(period) + 4;
    const rawTile = new Float32Array(pCeil * fh);
    let rSeed = 0x711e0009;
    for (let i = 0; i < rawTile.length; i++) {
      rSeed = (Math.imul(rSeed, 1664525) + 1013904223) >>> 0;
      rawTile[i] = 0.05 + 0.90 * (rSeed / 4294967296);
    }
    const periodic = (x, y) => {
      if (halfTiled && x >= fw * 0.5) return 0.5;
      const px = ((x % period) + period) % period;
      if (sharpKind === 'smooth') return tileNoise(px, y);
      const x0 = Math.floor(px);
      if (sharpKind === 'nearest') return rawTile[y * pCeil + x0];
      const frac = px - x0;
      const x1 = (x0 + 1) % Math.max(1, Math.floor(period));
      return (1 - frac) * rawTile[y * pCeil + x0] + frac * rawTile[y * pCeil + x1];
    };
    const leftU8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fw * fh * 4);
    for (let y = 0; y < fh; y++) {
      for (let x = 0; x < fw; x++) {
        const v = Math.round(Math.max(0.05, Math.min(0.95, periodic(x, y))) * 255);
        const o = (y * fw + x) * 4;
        leftU8[o] = v; leftU8[o + 1] = v; leftU8[o + 2] = v; leftU8[o + 3] = 255;
      }
    }
    exp.chronos_depth_capture_stereo_left();
    const rightU8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fw * fh * 4);
    for (let y = 0; y < fh; y++) {
      for (let x = 0; x < fw; x++) {
        const v = Math.round(Math.max(0.05, Math.min(0.95, periodic(x + shift, y))) * 255);
        const o = (y * fw + x) * 4;
        rightU8[o] = v; rightU8[o + 1] = v; rightU8[o + 2] = v; rightU8[o + 3] = 255;
      }
    }
    exp.chronos_tick(1 / 30);
    assert(views.dTelemU32[2] !== STATE_READY_STEREO && views.dTelemU32[13] <= 32,
      `Suite 8A periodic res=${resMode} ${period}px (${sharpKind}, sign=${sign}, shift=${shift}, half=${halfTiled}) must be rejected, got st=${views.dTelemU32[2]} nodes=${views.dTelemU32[13]} medZ=${views.dTelemF32[29].toFixed(2)}m`);
  }

  // 8A1. Sharp aperiodic SD white-noise at half-pixel disparity (md=64) MUST reach accurate READY_STEREO
  {
    const fw = 320, fh = 180;
    const { exp, buf, views } = loadKernel(0);
    exp.chronos_set_depth_mode(2);
    exp.chronos_set_depth_calib(fPx, 0.0, 0.12, 64.0, 0);
    const texW = fw + 160;
    const raw = new Float32Array(texW * fh);
    let rSeed = 0xa100beef;
    for (let i = 0; i < raw.length; i++) {
      rSeed = (Math.imul(rSeed, 1664525) + 1013904223) >>> 0;
      raw[i] = 0.05 + 0.90 * (rSeed / 4294967296);
    }
    const sampleSharp = (x, y) => {
      const fx = x + 80;
      const x0 = Math.floor(fx), t = fx - x0;
      return raw[y * texW + x0] * (1 - t) + raw[y * texW + x0 + 1] * t;
    };
    const uL = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fw * fh * 4);
    for (let y = 0; y < fh; y++) for (let x = 0; x < fw; x++) {
      const v = Math.round(sampleSharp(x, y) * 255), o = (y * fw + x) * 4;
      uL[o] = v; uL[o + 1] = v; uL[o + 2] = v; uL[o + 3] = 255;
    }
    exp.chronos_depth_capture_stereo_left();
    const uR = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fw * fh * 4);
    for (let y = 0; y < fh; y++) for (let x = 0; x < fw; x++) {
      const v = Math.round(sampleSharp(x - 41.5, y) * 255), o = (y * fw + x) * 4;
      uR[o] = v; uR[o + 1] = v; uR[o + 2] = v; uR[o + 3] = 255;
    }
    exp.chronos_tick(1 / 30);
    const zTrue = (fPx * 0.12) / 41.5;
    assert(views.dTelemU32[2] === STATE_READY_STEREO && views.dTelemU32[13] >= 2000 && Math.abs(views.dTelemF32[29] - zTrue) / zTrue < 0.02,
      `Suite 8A1 sharp SD half-pixel control failed: st=${views.dTelemU32[2]} nodes=${views.dTelemU32[13]} medZ=${views.dTelemF32[29].toFixed(3)}m vs ${zTrue.toFixed(3)}m`);
  }

  // 8A2. Wrong disparity sign MUST NOT produce sparse false READY_STEREO (< 512 nodes coverage floor)
  {
    const sampleTex = makeNonPeriodicSampler(W, H, 0x8a200001);
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_depth_mode(2);
    exp.chronos_set_depth_calib(fPx, 0.0, 0.10, 32.0, 1); // expects +disparity, feed -12 px!
    const uL = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = Math.round(sampleTex(x, y) * 255);
        const o = (y * W + x) * 4;
        uL[o] = v; uL[o + 1] = v; uL[o + 2] = v; uL[o + 3] = 255;
      }
    }
    exp.chronos_depth_capture_stereo_left();
    const uR = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = Math.round(sampleTex(x + 12.0, y) * 255);
        const o = (y * W + x) * 4;
        uR[o] = v; uR[o + 1] = v; uR[o + 2] = v; uR[o + 3] = 255;
      }
    }
    exp.chronos_tick(1 / 30);
    assert(views.dTelemU32[2] !== STATE_READY_STEREO && views.dTelemU32[13] === 0,
      `Suite 8A2 wrong-sign stereo emitted sparse false READY_STEREO: st=${views.dTelemU32[2]} nodes=${views.dTelemU32[13]}`);
  }

  // 8B. Strict max_disp=2.0 rejects 2.35 px shift; 0.35 px vertical misalignment rejected; sign=+1 / 0 & SD 320x180 work
  {
    const sampleTex = makeNonPeriodicSampler(W, H, 0x8b001234);
    const runPair = (resMode, dx, dy, maxDisp, sign, B) => {
      const { exp, buf, views } = loadKernel(resMode);
      const fw = exp.chronos_get_frame_w();
      const fh = exp.chronos_get_frame_h();
      exp.chronos_set_depth_mode(2);
      assert(exp.chronos_set_depth_calib(fPx, 0.0, B, maxDisp, sign) === 1, 'calib failed');
      const uL = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fw * fh * 4);
      for (let y = 0; y < fh; y++) {
        for (let x = 0; x < fw; x++) {
          const v = Math.round(sampleTex(x, y) * 255);
          const o = (y * fw + x) * 4;
          uL[o] = v; uL[o + 1] = v; uL[o + 2] = v; uL[o + 3] = 255;
        }
      }
      exp.chronos_depth_capture_stereo_left();
      const uR = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fw * fh * 4);
      for (let y = 0; y < fh; y++) {
        for (let x = 0; x < fw; x++) {
          const v = Math.round(sampleTex(x - dx, y - dy) * 255);
          const o = (y * fw + x) * 4;
          uR[o] = v; uR[o + 1] = v; uR[o + 2] = v; uR[o + 3] = 255;
        }
      }
      exp.chronos_tick(1 / 30);
      return { state: views.dTelemU32[2], nodes: views.dTelemU32[13], medZ: views.dTelemF32[29] };
    };

    // d = -2.35 px exceeds strict max_disp = 2.0
    const overDisp = runPair(1, -2.35, 0.0, 2.0, -1, 0.012);
    assert(overDisp.nodes === 0 && overDisp.state !== STATE_READY_STEREO, `Suite 8B max_disp=2.0 must reject 2.35 px shift, got nodes=${overDisp.nodes}`);

    // dy = 0.35 px vertical misalignment exceeds 0.25 px epipolar gate
    const vertErr = runPair(1, -1.2, 0.35, 2.0, -1, 0.012);
    assert(vertErr.nodes < 200, `Suite 8B 0.35px vertical misalignment should be rejected, got nodes=${vertErr.nodes}`);

    // Positive disparity sign = +1 (Right shifted +1.2 px relative to Left -> ex = +1.2)
    const posSign = runPair(1, 1.2, 0.0, 2.0, 1, 0.012);
    assert(posSign.state === STATE_READY_STEREO && Math.abs(posSign.medZ - 5.0) < 0.25, `Suite 8B sign=+1 failed: st=${posSign.state} medZ=${posSign.medZ}`);

    // Auto sign = 0 with positive disparity
    const autoSign = runPair(1, 1.2, 0.0, 2.0, 0, 0.012);
    assert(autoSign.state === STATE_READY_STEREO && Math.abs(autoSign.medZ - 5.0) < 0.25, `Suite 8B sign=0 failed: st=${autoSign.state} medZ=${autoSign.medZ}`);

    // SD 320x180 stereo
    const sdRun = runPair(0, -1.2, 0.0, 2.0, -1, 0.012);
    assert(sdRun.state === STATE_READY_STEREO && Math.abs(sdRun.medZ - 5.0) < 0.30, `Suite 8B SD 320x180 failed: st=${sdRun.state} medZ=${sdRun.medZ}`);
  }

  // 8C. Unpaired mono pan in STEREO mode (NO capture call) must yield 0 stereo nodes; staged scene cut invalidates left ref
  {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_depth_mode(2);
    exp.chronos_set_depth_calib(fPx, 0.0, 0.012, 2.0, -1);
    for (let f = 0; f < 4; f++) {
      const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const v = Math.round(bosTexture(x + f * 1.2, y) * 255);
          const o = (y * W + x) * 4;
          u8[o] = v; u8[o + 1] = v; u8[o + 2] = v; u8[o + 3] = 255;
        }
      }
      exp.chronos_tick(1 / 30);
      assert(views.dTelemU32[2] !== STATE_READY_STEREO && views.dTelemU32[13] === 0, `Suite 8C unpaired mono pan frame ${f} fabricated stereo! nodes=${views.dTelemU32[13]}`);
    }

    // Now capture left, stage a scene cut (source_epoch jump 1 -> 2), tick right -> MUST reject!
    const uL = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = Math.round(bosTexture(x, y) * 255);
        const o = (y * W + x) * 4;
        uL[o] = v; uL[o + 1] = v; uL[o + 2] = v; uL[o + 3] = 255;
      }
    }
    assert(exp.chronos_depth_capture_stereo_left() === 1, 'capture failed');
    const uR = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = Math.round(bosTexture(x + 1.2, y) * 255);
        const o = (y * W + x) * 4;
        uR[o] = v; uR[o + 1] = v; uR[o + 2] = v; uR[o + 3] = 255;
      }
    }
    // Stage explicit scene cut flag (bit 5 = 32) in ingest meta (exact 20-arg ABI; flags=18=PTS_KNOWN|GEOM_KNOWN)
    assert(
      exp.chronos_set_ingest_meta(
        1, 99, 100000, 0, -1.0, 18, W, H, 0, 0, W, H, 0, 0xffffffff, 0, 0, W, H, 32, 1
      ) === 1,
      'Suite 8C set_ingest_meta scene cut staging failed'
    );
    exp.chronos_tick(1 / 30);
    assert(views.dTelemU32[2] !== STATE_READY_STEREO && views.dTelemU32[13] === 0, `Suite 8C scene cut must invalidate stereo ref, got nodes=${views.dTelemU32[13]}`);

    // 8C2. Sol A4: Internally detected visual scene cut (no ingest meta flags!) between Scene A and Scene B MUST invalidate stereo ref!
    {
      const k2 = loadKernel();
      k2.exp.chronos_set_depth_mode(2);
      k2.exp.chronos_set_depth_calib(fPx, 0.0, 0.012, 2.0, -1);
      const w1 = new Uint32Array(k2.buf, k2.exp.chronos_get_wave1_telemetry_ptr(), 16);
      const texA = makeNonPeriodicSampler(W, H, 0x11112222);
      const paintSceneA = (u8) => {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const v = Math.round(texA(x, y) * 255);
            const o = (y * W + x) * 4;
            u8[o] = v; u8[o + 1] = v; u8[o + 2] = v; u8[o + 3] = 255;
          }
        }
      };
      const paintSceneB = (u8, shift) => {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const v = Math.round(bosTexture(x + shift, y) * 255);
            const o = (y * W + x) * 4;
            u8[o] = v; u8[o + 1] = v; u8[o + 2] = v; u8[o + 3] = 255;
          }
        }
      };
      for (let f = 0; f < 3; f++) {
        paintSceneA(new Uint8Array(k2.buf, k2.exp.chronos_get_ingest_rgba_ptr(), W * H * 4));
        k2.exp.chronos_tick(1 / 30);
      }
      const epBefore = w1[6];
      // Even when Left & Right are a valid Scene B stereo pair, the Scene A -> Scene B visual cut on tick MUST invalidate!
      paintSceneB(new Uint8Array(k2.buf, k2.exp.chronos_get_ingest_rgba_ptr(), W * H * 4), 0.0);
      assert(k2.exp.chronos_depth_capture_stereo_left() === 1, 'Suite 8C2 capture failed');
      paintSceneB(new Uint8Array(k2.buf, k2.exp.chronos_get_ingest_rgba_ptr(), W * H * 4), 1.2);
      k2.exp.chronos_tick(1 / 30);
      assert(w1[6] > epBefore, `Suite 8C2 expected internal visual scene cut to bump continuity epoch (${epBefore} -> ${w1[6]})`);
      assert(k2.views.dTelemU32[2] !== STATE_READY_STEREO && k2.views.dTelemU32[13] === 0,
        `Suite 8C2 internal visual scene cut failed to invalidate stereo ref! st=${k2.views.dTelemU32[2]} nodes=${k2.views.dTelemU32[13]}`);
    }
  }

  // 8D. BOS <-> Stereo mutual exclusion & clean recovery (Stereo -> Off -> BOS Mode 1 -> READY)
  {
    const { exp, buf, views } = loadKernel();
    exp.chronos_set_bos_mode(1);
    exp.chronos_set_bos_gauge_roi(8, 8, 248, 120);
    exp.chronos_set_depth_mode(2);
    exp.chronos_set_depth_calib(fPx, 0.0, 0.012, 2.0, -1);

    const paintBosFrame = (u8, shift) => {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const v = Math.round(bosTexture(x + shift, y) * 255);
          const o = (y * W + x) * 4;
          u8[o] = v; u8[o + 1] = v; u8[o + 2] = v; u8[o + 3] = 255;
        }
      }
    };
    paintBosFrame(new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4), 0.0);
    exp.chronos_depth_capture_stereo_left();
    paintBosFrame(new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4), 1.2);
    exp.chronos_tick(1 / 30);
    assert(views.dTelemU32[2] === STATE_READY_STEREO, 'Suite 8D stereo step failed');
    assert(views.bosTelemU32[2] === 0, `Suite 8D stereo leaked BOS state=${views.bosTelemU32[2]} (expected 0=OFF)`);

    // Turn depth OFF -> BOS must cleanly reset and recover to BOS_STATE_READY (2) within 22 frames!
    exp.chronos_set_depth_mode(0);
    let bosReady = false;
    for (let f = 0; f < 24; f++) {
      paintBosFrame(new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4), 0.0);
      exp.chronos_tick(1 / 30);
      if (views.bosTelemU32[2] === 2) bosReady = true;
    }
    assert(bosReady, `Suite 8D BOS failed to recover after leaving stereo: bosState=${views.bosTelemU32[2]} reason=${views.bosTelemU32[3]}`);

    // BOS warmup invalidated stereo_ref_ready: re-entering stereo WITHOUT new capture must NOT produce stereo!
    exp.chronos_set_depth_mode(2);
    paintBosFrame(new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4), 1.2);
    exp.chronos_tick(1 / 30);
    assert(views.dTelemU32[2] !== STATE_READY_STEREO && views.dTelemU32[13] === 0, 'Suite 8D BOS ref must never be reused by stereo');
  }

  console.log(
    '  [PASS] I7.8 Stereo Adversarial & BOS Isolation: 12px periodic rejected, max_disp/vertical strict, sign=+1/0 & SD OK, unpaired/cut rejected, BOS recovers cleanly'
  );
}

// ============================================================================
// Suite 9: External Staleness, Rejected-Meta Isolation & AUTO_FUSED Fallthrough
// ============================================================================
function runSuite9ExternalAndAutoFused() {
  const { exp, buf, views } = loadKernel();
  exp.chronos_set_depth_mode(3);
  const ew = 64, eh = 32;

  // 9A. Commit 2m as SURVEYED_LIDAR seq 42, mutate buffer to 9m, submit INVALID neural meta -> MUST NOT publish 9m as LiDAR 42!
  views.dIngest.fill(2.0);
  assert(exp.chronos_set_depth_external_meta(ew, eh, 500.0, 1, 42) === 1, 'LiDAR meta failed');
  exp.chronos_tick(1 / 30);
  assert(views.dTelemU32[2] === STATE_READY_EXTERNAL && views.dTelemU32[4] === PROV_SURVEYED_LIDAR && views.dTelemU32[12] === 42, 'Suite 9A initial LiDAR commit failed');

  views.dIngest.fill(9.0);
  assert(exp.chronos_set_depth_external_meta(256, eh, 500.0, 3, 43) === 0, 'invalid width must return 0');
  exp.chronos_tick(1 / 30);
  assert(views.dTelemU32[2] !== STATE_READY_EXTERNAL && views.dTelemU32[4] === PROV_NONE, `Suite 9A rejected meta leaked prior LiDAR provenance! st=${views.dTelemU32[2]} prov=${views.dTelemU32[4]} seq=${views.dTelemU32[12]}`);
  assert((views.dTelemU32[3] & REASON_EXT_STALE) !== 0, 'Suite 9A expected REASON_EXT_STALE');

  // 9B. Consume-once staleness + reset_tracks + HD->SD invalidation
  views.dIngest.fill(4.0);
  assert(exp.chronos_set_depth_external_meta(ew, eh, 500.0, 1, 50) === 1, 'meta 50 failed');
  exp.chronos_tick(1 / 30);
  assert(views.dTelemU32[2] === STATE_READY_EXTERNAL && views.dTelemU32[12] === 50, 'Suite 9B commit 50 failed');
  // Next tick with no new metadata must fail closed
  exp.chronos_tick(1 / 30);
  assert(views.dTelemU32[2] !== STATE_READY_EXTERNAL && views.dTelemU32[4] === PROV_NONE, 'Suite 9B stale plane was reused');

  // Stage fresh external meta then call reset_tracks -> must clear before tick
  assert(exp.chronos_set_depth_external_meta(ew, eh, 500.0, 1, 51) === 1, 'meta 51 failed');
  exp.chronos_reset_tracks();
  exp.chronos_tick(1 / 30);
  assert(views.dTelemU32[2] !== STATE_READY_EXTERNAL, 'Suite 9B reset_tracks must clear staged external plane');

  // Stage fresh external meta then switch HD->SD -> must clear before tick
  assert(exp.chronos_set_depth_external_meta(ew, eh, 500.0, 1, 52) === 1, 'meta 52 failed');
  exp.chronos_set_resolution(0);
  exp.chronos_tick(1 / 30);
  assert(views.dTelemU32[2] !== STATE_READY_EXTERNAL, 'Suite 9B HD->SD must clear staged external plane');

  // 9C. AUTO_FUSED fallthrough: Stereo yields 0 matches -> falls through to valid fresh External plane!
  exp.chronos_set_resolution(1);
  exp.chronos_set_depth_mode(4);
  exp.chronos_set_depth_calib(500.0, 0.0, 0.10, 32.0, -1); // speed_m_s = 0.0, baseline_m = 0.10!
  // Capture blank Left (0 texture -> 0 stereo matches) and stage valid External plane
  new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4).fill(128);
  exp.chronos_depth_capture_stereo_left();
  views.dIngest.fill(3.5);
  assert(exp.chronos_set_depth_external_meta(ew, eh, 500.0, 1, 60) === 1, 'meta 60 failed');
  new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4).fill(128);
  exp.chronos_tick(1 / 30);
  assert(views.dTelemU32[2] === STATE_READY_EXTERNAL && views.dTelemU32[4] === PROV_SURVEYED_LIDAR && views.dTelemU32[12] === 60, `Suite 9C AUTO_FUSED must fall through from empty stereo to external, got st=${views.dTelemU32[2]} prov=${views.dTelemU32[4]}`);

  // 9D. AUTO_FUSED fallback to mono with speed_m_s=0, baseline_m=0.10 MUST NOT misread baseline_m as speed_m_s!
  exp.chronos_set_swarm_mode(2);
  const basePts = makeBaseLandmarks(500.0, 4.5, 14.5, 0xdecaf007);
  for (let f = 0; f < 26; f++) {
    const t = f / 30.0;
    const u8 = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
    renderScene3D(u8, basePts, [0.30 * t, 0, 0], 0.0, 500.0);
    exp.chronos_tick(1 / 30);
  }
  assert(views.dTelemU32[2] === STATE_READY_PARALLAX && views.dTelemU32[4] === PROV_MONO_UP_TO_SCALE, `Suite 9D AUTO_FUSED with speed=0, baseline=0.10 MUST stay UP_TO_SCALE(1), got st=${views.dTelemU32[2]} prov=${views.dTelemU32[4]}`);

  // 9E. Opus H3 + Sol A3: AUTO_FUSED stereo wins frame N over staged 2m LiDAR seq 42 -> mutating buffer to 9m and ticking N+1 MUST NOT publish 9m as LiDAR 42!
  {
    const k3 = loadKernel();
    k3.exp.chronos_set_depth_mode(4);
    k3.exp.chronos_set_depth_calib(500.0, 0.0, 0.012, 2.0, -1);
    const paintTex = (u8, shift) => {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const v = Math.round(bosTexture(x + shift, y) * 255);
          const o = (y * W + x) * 4;
          u8[o] = v; u8[o + 1] = v; u8[o + 2] = v; u8[o + 3] = 255;
        }
      }
    };
    paintTex(new Uint8Array(k3.buf, k3.exp.chronos_get_ingest_rgba_ptr(), W * H * 4), 0.0);
    assert(k3.exp.chronos_depth_capture_stereo_left() === 1, 'Suite 9E capture failed');
    k3.views.dIngest.fill(2.0);
    assert(k3.exp.chronos_set_depth_external_meta(ew, eh, 500.0, 1, 42) === 1, 'Suite 9E LiDAR stage failed');
    paintTex(new Uint8Array(k3.buf, k3.exp.chronos_get_ingest_rgba_ptr(), W * H * 4), 1.2);
    k3.exp.chronos_tick(1 / 30);
    assert(k3.views.dTelemU32[2] === STATE_READY_STEREO && k3.views.dTelemU32[4] === PROV_SURVEYED_STEREO,
      `Suite 9E frame N expected stereo win, got st=${k3.views.dTelemU32[2]} prov=${k3.views.dTelemU32[4]}`);

    // Now mutate ingest buffer to 9m and tick frame N+1 with NO new external meta!
    k3.views.dIngest.fill(9.0);
    paintTex(new Uint8Array(k3.buf, k3.exp.chronos_get_ingest_rgba_ptr(), W * H * 4), 1.2);
    k3.exp.chronos_tick(1 / 30);
    assert(k3.views.dTelemU32[2] !== STATE_READY_EXTERNAL && k3.views.dTelemU32[4] !== PROV_SURVEYED_LIDAR,
      `Suite 9E stale LiDAR leaked on frame N+1 after stereo won frame N! st=${k3.views.dTelemU32[2]} prov=${k3.views.dTelemU32[4]} seq=${k3.views.dTelemU32[12]}`);
  }

  // 9F. Opus H2: AUTO_FUSED with External winning every 2nd or 3rd frame MUST NOT inflate mono velocities / shrink metric depth!
  for (const period of [2, 3]) {
    const k4 = loadKernel();
    k4.exp.chronos_set_swarm_mode(2);
    k4.exp.chronos_set_depth_mode(4);
    k4.exp.chronos_set_depth_calib(500.0, 0.30, 0.0, 32.0, 0);
    for (let f = 0; f < 36; f++) {
      const t = f / 30.0;
      const u8 = new Uint8Array(k4.buf, k4.exp.chronos_get_ingest_rgba_ptr(), W * H * 4);
      const proj = renderScene3D(u8, basePts, [0.30 * t, 0, 0], 0.0, 500.0);
      if (f % period === 1) {
        k4.views.dIngest.fill(6.0);
        assert(k4.exp.chronos_set_depth_external_meta(ew, eh, 500.0, 1, 100 + f) === 1, 'Suite 9F ext stage failed');
      }
      k4.exp.chronos_tick(1 / 30);
      if (k4.views.dTelemU32[4] === PROV_MONO_METRIC) {
        const ratios = [];
        for (let i = 0; i < MAX_MICRO; i++) {
          if ((k4.views.dFlags[i] & FLAG_VALID) === 0 || k4.views.dTracks[i * 2] <= 1e-6) continue;
          const mb = i * 8;
          if (k4.views.micro[mb + 5] !== 1) continue;
          const px = k4.views.micro[mb + 1], py = k4.views.micro[mb + 2];
          let best = null, bestD2 = 81;
          for (const q of proj) {
            const d2 = (q.u - px) ** 2 + (q.v - py) ** 2;
            if (d2 < bestD2) { bestD2 = d2; best = q; }
          }
          if (best) ratios.push((1.0 / k4.views.dTracks[i * 2]) / best.Zc);
        }
        const medR = median(ratios);
        assert(ratios.length === 0 || (medR >= 0.90 && medR <= 1.10),
          `Suite 9F AUTO_FUSED period=${period} frame ${f} shrunk/biased MONO_METRIC: med(Z_est/Z_true)=${medR.toFixed(3)}`);
      }
    }
  }

  console.log(
    '  [PASS] I7.9 External Staleness & AUTO_FUSED Fallthrough: rejected meta isolated, stereo-win expires external, interleaved ext/mono unbiased, stereo->external fallthrough & split baseline/speed verified'
  );
}

// ============================================================================
// Suite 10: Worker K=8 Depth Ring + WebCodecs fMP4 (+omit_tfhd_offset) / elst / Commit-Count
// ============================================================================
async function runSuite10WorkerAndWebCodecs() {
  const workerPath = path.join(ROOT, 'web', 'worker.js');
  const workerSource = fs.readFileSync(workerPath, 'utf8');
  const wasmBytes = fs.readFileSync(WASM_PATH);
  const postedMessages = [];

  const createDummyCtx = () => ({
    save() {}, restore() {}, clearRect() {}, fillRect() {}, strokeRect() {},
    beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, arc() {}, stroke() {}, fill() {},
    fillText() {}, measureText(t) { return { width: String(t).length * 6 }; },
    drawImage() {}, setLineDash() {}, translate() {}, rotate() {}, scale() {}, setTransform() {}, resetTransform() {},
    createImageData(w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; },
    getImageData(_x, _y, w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; },
    putImageData() {},
  });
  class MockOffscreenCanvas {
    constructor(w, h) { this.width = w; this.height = h; this._ctx = createDummyCtx(); }
    getContext() { return this._ctx; }
  }
  class MockImageData {
    constructor(a, b, c) {
      if (a instanceof Uint8ClampedArray) {
        this.data = a; this.width = b; this.height = c || (a.length / (b * 4));
      } else {
        this.width = a; this.height = b; this.data = new Uint8ClampedArray(a * b * 4);
      }
    }
  }
  const fakeSelf = {
    postMessage(msg) { postedMessages.push(msg); },
    onmessage: null,
  };
  const pendingTimers = new Set();
  const sandbox = {
    self: fakeSelf,
    OffscreenCanvas: MockOffscreenCanvas,
    ImageData: MockImageData,
    WebAssembly, performance, Math, Number, Array, Map, Set, DataView, String, Object, Boolean,
    Uint8Array, Uint8ClampedArray, Uint16Array, Uint32Array, Int16Array, Int32Array, Float32Array, Float64Array,
    ArrayBuffer, Error, Promise, JSON, console,
    setTimeout: (fn, ms) => {
      const id = setTimeout(() => { pendingTimers.delete(id); fn(); }, ms);
      pendingTimers.add(id);
      return id;
    },
    clearTimeout: (id) => { pendingTimers.delete(id); clearTimeout(id); },
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: () => {},
    fetch: async () => ({
      ok: true, status: 200,
      arrayBuffer: async () => wasmBytes.buffer.slice(wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength),
    }),
  };
  vm.runInNewContext(workerSource, sandbox, { filename: 'web/worker.js' });

  try {
    const dummyCanvas = () => new MockOffscreenCanvas(640, 360);
    await fakeSelf.onmessage({
      data: {
        type: 'init',
        wasmUrl: './chronos_spectral.wasm',
        hudCanvas: dummyCanvas(), pipCanvas: dummyCanvas(), hilbertCanvas: dummyCanvas(),
        spectralCanvas: dummyCanvas(), persistenceCanvas: dummyCanvas(), physicsPhaseCanvas: dummyCanvas(),
      },
    });
    await fakeSelf.onmessage({ data: { type: 'set_resolution', resMode: 1 } });
    await fakeSelf.onmessage({ data: { type: 'set_source', sourceMode: 'video', fpsHint: 30 } });
    await fakeSelf.onmessage({ data: { type: 'set_depth_config', depthMode: 3 } });
    // Worker defaults to K=8 lookahead delay
    await fakeSelf.onmessage({ data: { type: 'set_config', lookaheadK: 8 } });

    const frameBuf = new Uint8Array(640 * 360 * 4).fill(120);
    for (let seq = 1; seq <= 10; seq++) {
      vm.runInContext(`
        depthIngestView.fill(${(1.0 + seq * 0.5).toFixed(2)});
        wasmExports.chronos_set_depth_external_meta(64, 32, 500.0, 1, ${seq});
      `, sandbox);
      await fakeSelf.onmessage({
        data: {
          type: 'video_rgba',
          epoch: 1,
          seq,
          width: 640,
          height: 360,
          buffer: frameBuf.buffer.slice(0),
          dtSec: 1 / 30,
          mediaTimeSec: (seq - 1) / 30,
          ptsKnown: true,
          tSrcUs: Math.round(((seq - 1) / 30) * 1_000_000),
        },
      });
    }
    // On tick 10 (cursor=10, K=8), displayed video frame is frame 2 (seq=2), NOT seq=10!
    const snap = vm.runInContext('buildWave2Snapshot(currentDisplaySlot())', sandbox);
    assert(snap.depth && snap.depth.extSeq === 2, `Suite 10A K=8 depth alignment failed: expected display extSeq=2, got ${snap.depth && snap.depth.extSeq}`);
    assert(snap.stamps && snap.stamps.depth && snap.stamps.depth.valid === true, 'Suite 10A PROD_W2_DEPTH stamp missing in worker snapshot');

    // 10B. Classic MP4 edts/elst alignment on juggling_ballistic.mp4 (0 ms, not 80 ms!)
    const jugPath = path.join(ROOT, 'web', 'samples', 'juggling_ballistic.mp4');
    if (fs.existsSync(jugPath)) {
      const jugBytes = fs.readFileSync(jugPath);
      sandbox.__testMp4Buf = jugBytes.buffer.slice(jugBytes.byteOffset, jugBytes.byteOffset + jugBytes.byteLength);
      const parsedJug = vm.runInContext('parseMp4TrackMultiCodec(__testMp4Buf)', sandbox);
      assert(parsedJug && parsedJug.samples.length > 10, 'Suite 10B juggling_ballistic.mp4 parse failed');
      assert(Math.abs(parsedJug.samples[0].timestampUs) <= 1000, `Suite 10B juggling_ballistic.mp4 edit-list shift failed: t0=${parsedJug.samples[0].timestampUs} us (expected 0 us)`);
    }

    // 10C. Audio-first fMP4 with +frag_keyframe+empty_moov+omit_tfhd_offset MUST match ffprobe 100%
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chronos-fmp4-'));
    try {
      const fmp4Path = path.join(tmpDir, 'audio_first_omit_tfhd.mp4');
      execFileSync('ffmpeg', [
        '-y', '-v', 'error',
        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.5',
        '-f', 'lavfi', '-i', 'testsrc=size=160x90:rate=24:duration=0.5',
        '-map', '0:a:0', '-map', '1:v:0',
        '-c:a', 'aac', '-c:v', 'libx264', '-g', '6', '-pix_fmt', 'yuv420p',
        '-movflags', '+frag_keyframe+empty_moov+omit_tfhd_offset',
        fmp4Path,
      ]);
      const probeJson = JSON.parse(execFileSync('ffprobe', [
        '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'packet=pos,size', '-of', 'json', fmp4Path,
      ], { encoding: 'utf8' }));
      const fmp4Bytes = fs.readFileSync(fmp4Path);
      sandbox.__testMp4Buf = fmp4Bytes.buffer.slice(fmp4Bytes.byteOffset, fmp4Bytes.byteOffset + fmp4Bytes.byteLength);
      const parsedFmp4 = vm.runInContext('parseMp4TrackMultiCodec(__testMp4Buf)', sandbox);
      assert(parsedFmp4 && parsedFmp4.samples.length === probeJson.packets.length && parsedFmp4.samples.length >= 10,
        `Suite 10C fMP4 sample count mismatch: parsed=${parsedFmp4 ? parsedFmp4.samples.length : 0} vs ffprobe=${probeJson.packets.length}`);
      for (let i = 0; i < probeJson.packets.length; i++) {
        const expPos = Number(probeJson.packets[i].pos);
        const expSize = Number(probeJson.packets[i].size);
        const got = parsedFmp4.samples[i];
        assert(got.offset === expPos && got.size === expSize,
          `Suite 10C fMP4 packet ${i} mismatch: got (${got.offset}, ${got.size}) vs ffprobe (${expPos}, ${expSize})`);
      }

      // 10D. runWorkerWebCodecsFastPass MUST NOT count {committed:false} frames as INDEXED!
      const prevMsgCount = postedMessages.length;
      sandbox.__fakeMp4Bytes = fmp4Bytes.buffer.slice(fmp4Bytes.byteOffset, fmp4Bytes.byteOffset + fmp4Bytes.byteLength);
      sandbox.EncodedVideoChunk = class EncodedVideoChunk {
        constructor(init) { Object.assign(this, init); }
      };
      sandbox.VideoDecoder = class VideoDecoder {
        static async isConfigSupported() { return { supported: true }; }
        constructor(init) { this._init = init; this.state = 'unconfigured'; this.decodeQueueSize = 0; }
        configure() { this.state = 'configured'; }
        decode(chunk) {
          this._init.output({
            displayWidth: 640, displayHeight: 360, codedWidth: 640, codedHeight: 360,
            timestamp: chunk.timestamp,
            close() {},
          });
        }
        async flush() {}
        close() { this.state = 'closed'; }
      };
      await vm.runInContext(`(async () => {
        const origProcess = processIngestedFrame;
        processIngestedFrame = () => ({ committed: false, status: 4 });
        try {
          await runWorkerWebCodecsFastPass({
            file: { arrayBuffer: async () => __fakeMp4Bytes },
            epoch: 99,
            fps: 24,
          });
        } finally {
          processIngestedFrame = origProcess;
        }
      })()`, sandbox);
      const doneMsg = postedMessages.slice(prevMsgCount).find((m) => m.type === 'fast_pass_done');
      assert(doneMsg && doneMsg.processed === 0 && doneMsg.rejected === probeJson.packets.length && String(doneMsg.label).startsWith('FAST-PASS ERR'),
        `Suite 10D rejected fast-pass lied about indexing: ${JSON.stringify(doneMsg)}`);

      // 10E. Sol A8: Stream-copy-trimmed MP4 with edts/elst preroll MUST decode preroll but drop frames outside [editStartUs, editEndUs)
      const fullPath = path.join(tmpDir, 'full_24fps.mp4');
      const trimPath = path.join(tmpDir, 'trimmed_elst.mp4');
      execFileSync('ffmpeg', [
        '-y', '-v', 'error',
        '-f', 'lavfi', '-i', 'testsrc=size=160x90:rate=24:duration=1.0',
        '-c:v', 'libx264', '-g', '24', '-bf', '0', '-pix_fmt', 'yuv420p',
        fullPath,
      ]);
      execFileSync('ffmpeg', [
        '-y', '-v', 'error',
        '-ss', '0.25', '-i', fullPath,
        '-t', '0.50', '-c', 'copy',
        trimPath,
      ]);
      const trimBytes = fs.readFileSync(trimPath);
      sandbox.__trimMp4Buf = trimBytes.buffer.slice(trimBytes.byteOffset, trimBytes.byteOffset + trimBytes.byteLength);
      const parsedTrim = vm.runInContext('parseMp4TrackMultiCodec(__trimMp4Buf)', sandbox);
      assert(parsedTrim && parsedTrim.samples.length === 18, `Suite 10E trimmed MP4 parse failed: samples=${parsedTrim ? parsedTrim.samples.length : 0}`);
      assert(parsedTrim.samples[0].timestampUs < -100000, `Suite 10E expected negative preroll sample timestamp, got ${parsedTrim.samples[0].timestampUs}`);
      assert(Number.isFinite(parsedTrim.editStartUs) && Number.isFinite(parsedTrim.editEndUs), `Suite 10E missing edit window: [${parsedTrim.editStartUs}, ${parsedTrim.editEndUs}]`);

      const seenPts = [];
      sandbox.__seenPts = seenPts;
      const prevTrimMsgCount = postedMessages.length;
      await vm.runInContext(`(async () => {
        const origProcess = processIngestedFrame;
        processIngestedFrame = (_rgba, _dtSec, _mediaTimeSec, metaOpts) => {
          __seenPts.push(metaOpts && typeof metaOpts.tSrcUs === 'number' ? metaOpts.tSrcUs : -999999);
          return { committed: true, status: 0 };
        };
        try {
          await runWorkerWebCodecsFastPass({
            file: { arrayBuffer: async () => __trimMp4Buf },
            epoch: 100,
            fps: 24,
          });
        } finally {
          processIngestedFrame = origProcess;
        }
      })()`, sandbox);
      const trimDone = postedMessages.slice(prevTrimMsgCount).find((m) => m.type === 'fast_pass_done');
      assert(trimDone && trimDone.processed === 12 && seenPts.length === 12,
        `Suite 10E elst preroll/tail filter failed: processed=${trimDone && trimDone.processed}, seen=${seenPts.length}, totalPackets=${parsedTrim.samples.length}`);
      assert(seenPts[0] === 0 && Math.abs(seenPts[11] - 458333) <= 1000,
        `Suite 10E timestamps mismatch: first=${seenPts[0]} last=${seenPts[11]}`);
      for (let i = 0; i < seenPts.length; i++) {
        assert(seenPts[i] >= 0 && seenPts[i] < parsedTrim.editEndUs, `Suite 10E frame ${i} out of edit window: ${seenPts[i]} us`);
        if (i > 0) assert(seenPts[i] > seenPts[i - 1], `Suite 10E non-monotonic pts: ${seenPts[i - 1]} -> ${seenPts[i]}`);
      }

      // 10E2. Exact [editStartUs, editEndUs) boundaries: -250us preroll rejected AND 458333us < 458500us retained!
      {
        const edgePts = [];
        sandbox.__edgePts = edgePts;
        await vm.runInContext(`(async () => {
          const origParse = parseMp4TrackMultiCodec;
          const origProcess = processIngestedFrame;
          parseMp4TrackMultiCodec = () => ({
            codec: 'avc1.64001e',
            codecTag: 'H.264',
            codedWidth: 160,
            codedHeight: 90,
            timescale: 1000000,
            editStartUs: 0,
            editEndUs: 458500,
            bytes: new Uint8Array(64),
            samples: [
              { offset: 0, size: 8, isKey: true, timestampUs: -250, durationUs: 41667 },
              { offset: 8, size: 8, isKey: false, timestampUs: 0, durationUs: 41667 },
              { offset: 16, size: 8, isKey: false, timestampUs: 458333, durationUs: 41667 },
              { offset: 24, size: 8, isKey: false, timestampUs: 458500, durationUs: 41667 },
            ],
          });
          processIngestedFrame = (_rgba, _dtSec, _mediaTimeSec, metaOpts) => {
            __edgePts.push(metaOpts.tSrcUs);
            return { committed: true, status: 0 };
          };
          try {
            await runWorkerWebCodecsFastPass({
              file: { arrayBuffer: async () => new ArrayBuffer(64) },
              epoch: 101,
              fps: 24,
            });
          } finally {
            parseMp4TrackMultiCodec = origParse;
            processIngestedFrame = origProcess;
          }
        })()`, sandbox);
        assert(edgePts.length === 2 && edgePts[0] === 0 && edgePts[1] === 458333,
          `Suite 10E2 exact [editStartUs, editEndUs) boundary check failed: got [${edgePts.join(',')}] (expected [0,458333])`);
      }

      // 10F. Worker reports depth_calib_rejected when calibration is invalid
      const prevCalibMsg = postedMessages.length;
      await fakeSelf.onmessage({ data: { type: 'set_depth_config', fPx: 5.0, reqId: 777 } });
      const rejMsg = postedMessages.slice(prevCalibMsg).find((m) => m.type === 'depth_calib_rejected');
      assert(rejMsg && rejMsg.reqId === 777, `Suite 10F expected depth_calib_rejected(reqId=777), got ${JSON.stringify(rejMsg)}`);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  } finally {
    for (const id of pendingTimers) clearTimeout(id);
  }

  console.log(
    '  [PASS] I7.10 Worker K=8 Depth Ring & WebCodecs Demux/Commit-Count: K=8 seq=2 aligned, elst=0ms & trimmed preroll dropped, audio-first fMP4 100% ffprobe match, rejected frames & calib handled'
  );
}

// ============================================================================
// Fast Parity-Lite Mode for Gate N (exercises ALL depth regimes & f64x2 stereo path)
// ============================================================================
function runParityLite() {
  const { exp, buf, views } = loadKernel(0); // 320x180 SD for compact, fast deterministic trace
  const fw = exp.chronos_get_frame_w();
  const fh = exp.chronos_get_frame_h();
  const fPx = 250.0;

  // 1. External depth with NaN/Inf sanitization
  assert(exp.chronos_set_depth_mode(3) === 1, 'parity-lite ext mode failed');
  for (let i = 0; i < 64 * 32; i++) {
    views.dIngest[i] = i === 0 ? NaN : (i === 1 ? Infinity : (i % 64 < 32 ? 2.5 : 8.0));
  }
  assert(exp.chronos_set_depth_external_meta(64, 32, fPx, 1, 7) === 1, 'parity-lite ext meta failed');
  exp.chronos_tick(1 / 30);
  assert(views.dTelemU32[2] === STATE_READY_EXTERNAL, 'parity-lite ext state failed');

  // 2. Micro-baseline stereo (d = 1.2 px) + Wide-baseline stereo (d = 10 px, exercises f64x2 pyramid/refinement)
  exp.chronos_reset_tracks();
  const sampleTex = makeNonPeriodicSampler(fw, fh, 0x9a217001);
  assert(exp.chronos_set_depth_mode(2) === 1, 'parity-lite stereo mode failed');
  for (const [disp, B, maxD] of [[1.2, 0.024, 2.0], [10.0, 0.20, 24.0]]) {
    assert(exp.chronos_set_depth_calib(fPx, 0.0, B, maxD, -1) === 1, 'parity-lite stereo calib failed');
    const uL = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fw * fh * 4);
    for (let y = 0; y < fh; y++) {
      for (let x = 0; x < fw; x++) {
        const v = Math.round(sampleTex(x, y) * 255);
        const o = (y * fw + x) * 4;
        uL[o] = v; uL[o + 1] = v; uL[o + 2] = v; uL[o + 3] = 255;
      }
    }
    assert(exp.chronos_depth_capture_stereo_left() === 1, 'parity-lite capture failed');
    const uR = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fw * fh * 4);
    for (let y = 0; y < fh; y++) {
      for (let x = 0; x < fw; x++) {
        const v = Math.round(sampleTex(x + disp, y) * 255);
        const o = (y * fw + x) * 4;
        uR[o] = v; uR[o + 1] = v; uR[o + 2] = v; uR[o + 3] = 255;
      }
    }
    exp.chronos_tick(1 / 30);
    assert(views.dTelemU32[2] === STATE_READY_STEREO, `parity-lite stereo d=${disp} failed`);
  }

  // 3. Monocular lateral motion parallax (22 frames at SD)
  exp.chronos_reset_tracks();
  exp.chronos_set_swarm_mode(2);
  assert(exp.chronos_set_depth_mode(1) === 1, 'parity-lite mono mode failed');
  assert(exp.chronos_set_depth_calib(fPx, 0.30, 0.0, 32.0, 0) === 1, 'parity-lite mono calib failed');
  const cx = fw * 0.5, cy = fh * 0.5;
  for (let f = 0; f < 22; f++) {
    const camX = 0.30 * (f / 30.0);
    const dst = new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fw * fh * 4);
    dst.fill(20);
    for (let i = 3; i < dst.length; i += 4) dst[i] = 255;
    for (let idx = 0; idx < 32; idx++) {
      const gx = idx % 8, gy = Math.floor(idx / 8);
      const z = 4.5 + ((idx * 37) % 11);
      const u0 = 28 + gx * 36 + (idx % 3) * 4;
      const v0 = 24 + gy * 40 + (idx % 2) * 5;
      const u = cx + (fPx * (((u0 - cx) * z) / fPx - camX)) / z;
      const v = cy + (fPx * (((v0 - cy) * z) / fPx)) / z;
      const x0 = Math.max(0, Math.floor(u - 3)), x1 = Math.min(fw - 1, Math.ceil(u + 3));
      const y0 = Math.max(0, Math.floor(v - 3)), y1 = Math.min(fh - 1, Math.ceil(v + 3));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const g = Math.exp(-((x - u) ** 2 + (y - v) ** 2) / 2.2);
          if (g < 0.05) continue;
          const o = (y * fw + x) * 4;
          const lum = Math.min(255, Math.round(20 + g * 225));
          dst[o] = lum; dst[o + 1] = lum; dst[o + 2] = lum;
        }
      }
    }
    exp.chronos_tick(1 / 30);
  }
  assert(views.dTelemU32[2] === STATE_READY_PARALLAX, `parity-lite mono failed: st=${views.dTelemU32[2]}`);
  console.log('  [PASS] I7 parity-lite: External + Micro Stereo + Wide Stereo + Mono Parallax verified');
}

if (PARITY_LITE) {
  runParityLite();
} else {
  console.log('[Gate I7] Zero-ML Depth Sensing, FoE/TTC, Calibrated Stereo BOS, External Depth-Plane ABI & Adversarial Suite');
  runSuite1MicroStereo();
  runSuite2WideStereo();
  runSuite3And4MonoAndNegativeGates();
  runSuite5RealClipFoeTtc();
  runSuite6ExternalProvenanceAndHero();
  runSuite7MonoAdversarial();
  runSuite8StereoAdversarialAndBosIsolation();
  runSuite9ExternalAndAutoFused();
  await runSuite10WorkerAndWebCodecs();
  console.log('[PASS] Gate I7: ALL 10 DEPTH & ADVERSARIAL SUITES VERIFIED');
}
