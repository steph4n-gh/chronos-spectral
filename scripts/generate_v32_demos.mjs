#!/usr/bin/env node
/**
 * Generate & Kernel-Verify the 5 v3.2.0 Interactive Mode Demo MP4s in web/samples/:
 *   1. depth_parallax_dolly.mp4      (640x360, 30 fps, 9.0 s) - Calibrated lateral parallax + moving outliers + pure-yaw fail-closed gate
 *   2. depth_foe_approach.mp4        (640x360, 30 fps, 8.5 s) - Forward runway approach: FoE lock + radial metric depth + optical looming TTC
 *   3. hodge_cylinder_circulation.mp4 (640x360, 30 fps, 8.0 s) - Dense PIV tracer flow around solid cylinder (beta_1>=1 harmonic + coexact + exact)
 *   4. rs_vibrometry_247hz.mp4       (640x360, 30 fps, 8.0 s) - Split static-tripod target with 247.5 Hz rolling-shutter vibration (left) vs quiet (right)
 *   5. fovea_720p_micro_detail.mp4   (1280x720, 30 fps, 8.0 s) - Native 720p sweeping carriers + micro-orbiting clusters & 1:1 Vernier scale
 *
 * Usage:
 *   node scripts/generate_v32_demos.mjs [--verify-only]
 */
import fs from 'node:fs';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SAMPLES_DIR = path.join(ROOT, 'web', 'samples');
const WASM_PATH = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const VERIFY_ONLY = process.argv.includes('--verify-only');

function assert(cond, msg) {
  if (!cond) throw new Error(`FAIL [v3.2.0 demos]: ${msg}`);
}

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function encodeRawRgbaToMp4(outPath, w, h, fps, nFrames, crf, renderFrame) {
  return new Promise((resolve, reject) => {
    const ff = spawn(
      'ffmpeg',
      [
        '-y', '-v', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${w}x${h}`, '-r', String(fps), '-i', 'pipe:0',
        '-c:v', 'libx264', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
        '-crf', String(crf), '-preset', 'medium', '-g', '15', '-bf', '0',
        '-movflags', '+faststart',
        outPath,
      ],
      { stdio: ['pipe', 'ignore', 'pipe'] }
    );
    let errTxt = '';
    ff.stderr.on('data', (d) => { errTxt += d.toString(); });
    ff.on('error', reject);
    ff.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg exited ${code} for ${outPath}: ${errTxt}`));
    });

    const buf = new Uint8Array(w * h * 4);
    let f = 0;
    const pump = () => {
      while (f < nFrames) {
        renderFrame(buf, f, f / fps);
        f++;
        if (!ff.stdin.write(buf)) {
          ff.stdin.once('drain', pump);
          return;
        }
      }
      ff.stdin.end();
    };
    pump();
  });
}

function drawDisc(buf, w, h, cx, cy, r, col, bgAdd = false) {
  const x0 = Math.max(0, Math.floor(cx - r - 1.5));
  const x1 = Math.min(w - 1, Math.ceil(cx + r + 1.5));
  const y0 = Math.max(0, Math.floor(cy - r - 1.5));
  const y1 = Math.min(h - 1, Math.ceil(cy + r + 1.5));
  const inv2s2 = 1.0 / (0.48 * r * r);
  for (let y = y0; y <= y1; y++) {
    const dy = y - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      const g = Math.exp(-(dx * dx + dy * dy) * inv2s2);
      if (g < 0.03) continue;
      const o = (y * w + x) * 4;
      if (bgAdd) {
        buf[o] = Math.min(255, Math.round(buf[o] + g * col[0]));
        buf[o + 1] = Math.min(255, Math.round(buf[o + 1] + g * col[1]));
        buf[o + 2] = Math.min(255, Math.round(buf[o + 2] + g * col[2]));
      } else {
        buf[o] = Math.min(255, Math.round(buf[o] * (1 - g) + g * col[0]));
        buf[o + 1] = Math.min(255, Math.round(buf[o + 1] * (1 - g) + g * col[1]));
        buf[o + 2] = Math.min(255, Math.round(buf[o + 2] * (1 - g) + g * col[2]));
      }
    }
  }
}

// ============================================================================
// 1. depth_parallax_dolly.mp4 (640x360, 30 fps, 9.0 s = 270 frames)
//    0.0-6.5 s: Pure lateral dolly v = 0.30 m/s past 3 depth strata (4.5, 9.0, 16.5 m)
//               + 4 bright amber transverse independent movers (epipolar outlier proof)
//    6.5-9.0 s: Pure tripod yaw rotation omega_y = 0.042 rad/s (fail-closed refusal proof)
// ============================================================================
async function buildDepthParallaxDolly() {
  const outPath = path.join(SAMPLES_DIR, 'depth_parallax_dolly.mp4');
  const W = 640, H = 360, fps = 30, nFrames = 270, fPx = 500.0;
  const rng = makeRng(0xde974001);
  const landmarks = [];
  // Distribute across X in [-5.5, 14.5] m so 6.5s * 0.30 = 1.95 m travel always stays densely populated
  const strata = [
    { zBase: 4.5, zSpan: 1.4, count: 140, r: 3.6, col: [248, 178, 92] },
    { zBase: 8.6, zSpan: 2.6, count: 170, r: 3.2, col: [148, 228, 205] },
    { zBase: 14.8, zSpan: 4.2, count: 190, r: 2.9, col: [128, 174, 245] },
  ];
  for (const st of strata) {
    let tries = 0;
    let added = 0;
    while (added < st.count && tries < 8000) {
      tries++;
      const Z = st.zBase + rng() * st.zSpan;
      const halfX = (W * 0.55 * Z) / fPx;
      const halfY = (H * 0.44 * Z) / fPx;
      const X = -halfX + rng() * (2 * halfX + 2.4);
      const Y = -halfY + rng() * (2 * halfY);
      let ok = true;
      for (const q of landmarks) {
        if (Math.abs(q.Z - Z) < 2.0) {
          const du = (fPx * (q.X - X)) / Z;
          const dv = (fPx * (q.Y - Y)) / Z;
          if (du * du + dv * dv < 20 * 20) { ok = false; break; }
        }
      }
      if (!ok) continue;
      landmarks.push({ X, Y, Z, r: st.r, col: st.col });
      added++;
    }
  }
  const movers = [
    { x0: -0.6, y0: -1.4, z: 7.0, vx: 0.0, vy: 1.35 },
    { x0: 1.2, y0: 1.4, z: 8.2, vx: 0.0, vy: -1.45 },
    { x0: 0.3, y0: -1.2, z: 6.4, vx: 0.95, vy: 1.15 },
    { x0: 2.0, y0: 1.1, z: 9.2, vx: -0.85, vy: -1.25 },
  ];

  await encodeRawRgbaToMp4(outPath, W, H, fps, nFrames, 18, (buf, f, t) => {
    // Subtle dark gradient backdrop
    for (let y = 0; y < H; y++) {
      const bgR = 11 + Math.round((y / H) * 8);
      const bgG = 16 + Math.round((y / H) * 11);
      const bgB = 26 + Math.round((y / H) * 16);
      const rowOff = y * W * 4;
      for (let x = 0; x < W; x++) {
        const o = rowOff + x * 4;
        buf[o] = bgR; buf[o + 1] = bgG; buf[o + 2] = bgB; buf[o + 3] = 255;
      }
    }
    const tSwitch = 6.5;
    const camX = t <= tSwitch ? 0.30 * t : 0.30 * tSwitch;
    const yaw = t <= tSwitch ? 0.0 : 0.042 * (t - tSwitch);
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const proj = [];
    for (const p of landmarks) {
      const dx = p.X - camX;
      const dy = p.Y;
      const dz = p.Z;
      const xc = cy * dx - sy * dz;
      const zc = sy * dx + cy * dz;
      if (zc < 1.2) continue;
      const u = W * 0.5 + (fPx * xc) / zc;
      const v = H * 0.5 + (fPx * dy) / zc;
      if (u < 5 || u > W - 6 || v < 5 || v > H - 6) continue;
      proj.push({ u, v, zc, r: p.r, col: p.col });
    }
    for (const m of movers) {
      const cyc = (t % 2.6) - 1.3;
      const mx = m.x0 + camX + m.vx * cyc;
      const my = m.y0 + m.vy * cyc;
      const dx = mx - camX;
      const xc = cy * dx - sy * m.z;
      const zc = sy * dx + cy * m.z;
      if (zc < 1.2) continue;
      const u = W * 0.5 + (fPx * xc) / zc;
      const v = H * 0.5 + (fPx * my) / zc;
      if (u < 6 || u > W - 7 || v < 6 || v > H - 7) continue;
      proj.push({ u, v, zc, r: 4.1, col: [255, 224, 82] });
    }
    proj.sort((a, b) => b.zc - a.zc);
    for (const q of proj) drawDisc(buf, W, H, q.u, q.v, q.r, q.col, false);
  });
  return outPath;
}

// ============================================================================
// 2. depth_foe_approach.mp4 (640x360, 30 fps, 8.5 s = 255 frames)
//    Pure forward flight along (Tx=0.060, Ty=-0.015, Tz=0.85) m/s -> FoE = (355.3, 171.2)
//    Exact non-wrapping 3D beacon corridor (Z0 in [9.0..25.0] m) proving continuous
//    FoE lock, radial MONO_METRIC depth, and optical looming TTC countdowns
// ============================================================================
async function buildDepthFoeApproach() {
  const outPath = path.join(SAMPLES_DIR, 'depth_foe_approach.mp4');
  const W = 640, H = 360, fps = 30, nFrames = 255, fPx = 500.0;
  const rng = makeRng(0xf0e12345);
  const vx = 0.060, vy = -0.015, vz = 0.85;
  const pts = [];
  let tries = 0;
  while (pts.length < 320 && tries < 12000) {
    tries++;
    const Z = 9.0 + rng() * 16.0; // stays in [1.8..25.0] m across 8.5s * 0.85 = 7.225 m travel (zero wrapping!)
    const ang = rng() * Math.PI * 2;
    const rad = 0.75 + rng() * 5.4;
    const X = Math.cos(ang) * rad + vx * 4.25;
    const Y = Math.sin(ang) * rad * 0.60 + vy * 4.25;
    const u0 = W * 0.5 + (fPx * X) / Z;
    const v0 = H * 0.5 + (fPx * Y) / Z;
    if (u0 < 24 || u0 > W - 24 || v0 < 20 || v0 > H - 20) continue;
    let ok = true;
    for (const q of pts) {
      if (Math.abs(q.Z - Z) < 3.0) {
        const qu = W * 0.5 + (fPx * q.X) / q.Z;
        const qv = H * 0.5 + (fPx * q.Y) / q.Z;
        if ((qu - u0) ** 2 + (qv - v0) ** 2 < 20 * 20) { ok = false; break; }
      }
    }
    if (!ok) continue;
    const warm = pts.length % 3 === 0 ? [255, 198, 110] : (pts.length % 3 === 1 ? [130, 232, 255] : [185, 245, 170]);
    pts.push({ X, Y, Z, col: warm });
  }

  await encodeRawRgbaToMp4(outPath, W, H, fps, nFrames, 18, (buf, _f, t) => {
    for (let y = 0; y < H; y++) {
      const bgR = 9 + Math.round((y / H) * 7);
      const bgG = 14 + Math.round((y / H) * 10);
      const bgB = 24 + Math.round((y / H) * 14);
      const rowOff = y * W * 4;
      for (let x = 0; x < W; x++) {
        const o = rowOff + x * 4;
        buf[o] = bgR; buf[o + 1] = bgG; buf[o + 2] = bgB; buf[o + 3] = 255;
      }
    }
    const proj = [];
    for (const p of pts) {
      const xc = p.X - vx * t;
      const yc = p.Y - vy * t;
      const zc = p.Z - vz * t;
      if (zc < 3.2) continue;
      const u = W * 0.5 + (fPx * xc) / zc;
      const v = H * 0.5 + (fPx * yc) / zc;
      if (u < 6 || u > W - 7 || v < 6 || v > H - 7) continue;
      const r = Math.max(2.4, Math.min(4.8, 28.0 / zc));
      proj.push({ u, v, zc, r, col: p.col });
    }
    proj.sort((a, b) => b.zc - a.zc);
    for (const q of proj) drawDisc(buf, W, H, q.u, q.v, q.r, q.col, false);
  });
  return outPath;
}

// ============================================================================
// 3. hodge_cylinder_circulation.mp4 (640x360, 30 fps, 8.0 s = 240 frames)
//    ~1,938 uniform jittered-grid PIV tracer beads flowing around a solid cylinder (R=54 px)
//    Superposition of:
//      - Harmonic circulation Gamma around the cylinder hole (beta_1 >= 1)
//      - Coexact localized vortices (strong nonzero curl!)
//      - Exact potential doublet channel inflow (grad phi!)
// ============================================================================
async function buildHodgeCylinderCirculation() {
  const outPath = path.join(SAMPLES_DIR, 'hodge_cylinder_circulation.mp4');
  const W = 640, H = 360, fps = 30, nFrames = 240;
  const dt = 1 / fps;
  const cx = 320.0, cy = 180.0;
  const cylR = 54.0;
  const holeMargin = 58.0;

  const velAt = (x, y) => {
    const dx = x - cx;
    const dy = y - cy;
    const r2 = Math.max(cylR * cylR, dx * dx + dy * dy);
    const invR2 = 1.0 / r2;
    const invR4 = invR2 * invR2;
    const R2 = cylR * cylR;
    // 1. Exact potential flow past cylinder (U0 = 15 px/s left-to-right)
    const U0 = 15.0;
    const uExact = U0 * (1.0 - R2 * (dx * dx - dy * dy) * invR4);
    const vExact = U0 * (-2.0 * R2 * dx * dy * invR4);
    // 2. Harmonic circulation around cylinder hole
    const gammaFac = 3000.0 * invR2;
    const uHarm = -gammaFac * dy;
    const vHarm = gammaFac * dx;
    // 3. Coexact wake vortices (strong localized swirl)
    let uCo = 0.0, vCo = 0.0;
    const vort = [
      { vx: 145, vy: 110, g: 105.0, s2: 56 * 56 },
      { vx: 145, vy: 255, g: -95.0, s2: 54 * 54 },
      { vx: 495, vy: 110, g: -105.0, s2: 56 * 56 },
      { vx: 495, vy: 255, g: 105.0, s2: 56 * 56 },
    ];
    for (const vo of vort) {
      const rx = x - vo.vx;
      const ry = y - vo.vy;
      const d2 = rx * rx + ry * ry;
      const w = vo.g * Math.exp(-d2 / (2.0 * vo.s2));
      uCo += (-ry / 48.0) * w;
      vCo += (rx / 48.0) * w;
    }
    return [uExact + uHarm + uCo, vExact + vHarm + vCo];
  };

  // Jittered-grid seeding ensures dense 64x36 exterior coverage around the cylinder void (beta_1 >= 1).
  const rng = makeRng(0x40d9e001);
  const particles = [];
  const nx = 60, ny = 34;
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const x = 6 + ((ix + 0.15 + 0.7 * rng()) / nx) * (W - 12);
      const y = 6 + ((iy + 0.15 + 0.7 * rng()) / ny) * (H - 12);
      if (Math.hypot(x - cx, y - cy) < holeMargin) continue;
      particles.push({
        x,
        y,
        col: particles.length % 3 === 0 ? [150, 238, 255] : [228, 240, 250],
      });
    }
  }

  await encodeRawRgbaToMp4(outPath, W, H, fps, nFrames, 21, (buf) => {
    for (let y = 0; y < H; y++) {
      const dy = y - cy;
      const rowOff = y * W * 4;
      for (let x = 0; x < W; x++) {
        const dx = x - cx;
        const r = Math.hypot(dx, dy);
        const o = rowOff + x * 4;
        if (r < cylR - 1.5) {
          buf[o] = 24; buf[o + 1] = 30; buf[o + 2] = 42; buf[o + 3] = 255;
        } else if (r <= cylR + 1.5) {
          buf[o] = 74; buf[o + 1] = 96; buf[o + 2] = 122; buf[o + 3] = 255;
        } else {
          buf[o] = 10; buf[o + 1] = 15; buf[o + 2] = 24; buf[o + 3] = 255;
        }
      }
    }
    for (const p of particles) {
      drawDisc(buf, W, H, p.x, p.y, 2.25, p.col, true);
      const [u1, v1] = velAt(p.x, p.y);
      const mx = p.x + 0.5 * u1 * dt;
      const my = p.y + 0.5 * v1 * dt;
      const [u2, v2] = velAt(mx, my);
      p.x += u2 * dt;
      p.y += v2 * dt;
      const dx = p.x - cx, dy = p.y - cy;
      const r = Math.hypot(dx, dy);
      if (r < holeMargin) {
        const sc = holeMargin / Math.max(1e-3, r);
        p.x = cx + dx * sc;
        p.y = cy + dy * sc;
      }
      if (p.x < 6) p.x += W - 12;
      if (p.x > W - 6) p.x -= W - 12;
      if (p.y < 6) p.y += H - 12;
      if (p.y > H - 6) p.y -= H - 12;
    }
  });
  return outPath;
}

// ============================================================================
// 4. rs_vibrometry_247hz.mp4 (640x360, 30 fps, 8.0 s = 240 frames)
//    Static tripod split target:
//      Active region (28 < x < 276, 24 < y < 335): High-contrast pattern vibrating horizontally at 247.5 Hz
//                                                  (8.25 cycles/frame -> exact 4-frame quadrature cancellation in ref!)
//                                                  with exact rolling-shutter row time t(y) = t_n + y*40us & 200us shutter
//      Surround / right half:                      Identical stationary pattern (0 Hz control)
// ============================================================================
async function buildRsVibrometry247Hz() {
  const outPath = path.join(SAMPLES_DIR, 'rs_vibrometry_247hz.mp4');
  const W = 640, H = 360, fps = 30, nFrames = 240;
  const fVibHz = 247.5;
  const dtRowSec = 40.0e-6;
  const expSec = 200.0e-6;
  const ampPx = 0.65;
  const texLuma = (xs, ys) => {
    const v =
      128.0 +
      42.0 * Math.sin(0.27 * xs + 0.13 * ys) +
      36.0 * Math.cos(0.19 * xs - 0.31 * ys) +
      24.0 * Math.sin(0.37 * xs + 0.29 * ys);
    return Math.max(12, Math.min(244, v));
  };
  const glOff = [0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992];
  const glWt = [0.11846344, 0.23931434, 0.28444444, 0.23931434, 0.11846344];

  await encodeRawRgbaToMp4(outPath, W, H, fps, nFrames, 16, (buf, _f, tFrame) => {
    for (let y = 0; y < H; y++) {
      const tRow = tFrame + y * dtRowSec;
      const shifts = glOff.map((u) => ampPx * Math.sin(2.0 * Math.PI * fVibHz * (tRow + u * expSec)));
      const rowOff = y * W * 4;
      for (let x = 0; x < W; x++) {
        const o = rowOff + x * 4;
        const inBorder = (x >= 28 && x <= 276 && (y === 24 || y === H - 25)) || (y >= 24 && y <= H - 25 && (x === 28 || x === 276));
        if (inBorder) {
          buf[o] = 56; buf[o + 1] = 210; buf[o + 2] = 245; buf[o + 3] = 255;
          continue;
        }
        const active = x > 28 && x < 276 && y > 24 && y < H - 25;
        let luma = 0.0;
        if (active) {
          for (let g = 0; g < 5; g++) luma += glWt[g] * texLuma(x - shifts[g], y);
        } else {
          luma = texLuma(x, y);
        }
        const u = Math.round(luma);
        if (active) {
          buf[o] = u; buf[o + 1] = Math.min(255, u + 4); buf[o + 2] = Math.max(0, u - 6);
        } else {
          buf[o] = Math.max(0, u - 4); buf[o + 1] = u; buf[o + 2] = Math.min(255, u + 6);
        }
        buf[o + 3] = 255;
      }
    }
  });
  return outPath;
}

// ============================================================================
// 5. fovea_720p_micro_detail.mp4 (1280x720, 30 fps, 8.0 s = 240 frames)
//    True 1280x720 HD optical bench showcasing 1:1 Fovea, PTZ ROI Slew & hires-720p:
//      - 2 sweeping carrier clusters carrying tight micro-orbiting satellite discs (r=3.8 px @ 720p)
//      - Crisp 1:1 Vernier calibration reticle & fine multi-frequency resolution test bars
// ============================================================================
async function buildFovea720pMicroDetail() {
  const outPath = path.join(SAMPLES_DIR, 'fovea_720p_micro_detail.mp4');
  const W = 1280, H = 720, fps = 30, nFrames = 240;

  // Pre-render static 720p optical bench backdrop with Vernier grid & 1:1 resolution bar patch
  const bg = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    const rowOff = y * W * 4;
    for (let x = 0; x < W; x++) {
      const o = rowOff + x * 4;
      let r = 14, g = 20, b = 31;
      // Subtle 80 px major / 20 px minor optical table grid
      if (x % 80 === 0 || y % 80 === 0) { r = 34; g = 46; b = 64; }
      else if (x % 20 === 0 || y % 20 === 0) { r = 20; g = 28; b = 42; }
      // High-frequency 1:1 resolution bar chart in top-right quadrant (2px, 3px, 4px line pairs)
      if (x >= 1020 && x < 1230 && y >= 50 && y < 170) {
        const lx = x - 1020;
        const ly = y - 50;
        let bar = 0;
        if (ly >= 12 && ly < 42) bar = ((lx >> 1) & 1) ? 215 : 22;
        else if (ly >= 50 && ly < 80) bar = ((Math.floor(lx / 3)) & 1) ? 215 : 22;
        else if (ly >= 88 && ly < 112) bar = ((lx >> 2) & 1) ? 215 : 22;
        if (bar > 0) { r = bar; g = bar; b = Math.min(255, bar + 12); }
      }
      bg[o] = r; bg[o + 1] = g; bg[o + 2] = b; bg[o + 3] = 255;
    }
  }

  await encodeRawRgbaToMp4(outPath, W, H, fps, nFrames, 18, (buf, _f, t) => {
    buf.set(bg);
    // Carrier A: sweeping Lissajous with 3 tight micro-orbiting satellites (r=3.8 px, orbit R=18 px)
    const ax = 640 + 360 * Math.sin(0.85 * t);
    const ay = 360 + 185 * Math.sin(1.70 * t);
    drawDisc(buf, W, H, ax, ay, 11.5, [255, 196, 78], false);
    for (let k = 0; k < 3; k++) {
      const th = 4.2 * t + (k * 2 * Math.PI) / 3;
      drawDisc(buf, W, H, ax + 18.0 * Math.cos(th), ay + 18.0 * Math.sin(th), 3.8, [96, 244, 255], false);
    }

    // Carrier B: elliptical path with 4 twin-frequency micro-beacons
    const bx = 640 - 320 * Math.cos(0.75 * t + 0.5);
    const by = 360 + 210 * Math.sin(0.75 * t + 0.5);
    drawDisc(buf, W, H, bx, by, 10.5, [120, 232, 162], false);
    for (let k = 0; k < 4; k++) {
      const th = -3.4 * t + (k * Math.PI) / 2;
      drawDisc(buf, W, H, bx + 20.0 * Math.cos(th), by + 20.0 * Math.sin(th), 3.6, [255, 128, 178], false);
    }

    // 18 subtle ambient background drift beads for swarm texture
    for (let i = 0; i < 18; i++) {
      const px = ((i * 73 + t * (26 + (i % 5) * 9)) % (W - 80)) + 40;
      const py = 90 + ((i * 109) % (H - 180)) + 16 * Math.sin(1.3 * t + i);
      drawDisc(buf, W, H, px, py, 3.2, [190, 214, 240], false);
    }
  });
  return outPath;
}

// ============================================================================
// Direct In-Kernel Wasm Verification of ALL 5 Generated MP4s!
// ============================================================================
function decodeMp4Rgba(mp4Path, w, h, maxFrames) {
  return execFileSync(
    'ffmpeg',
    ['-v', 'error', '-i', mp4Path, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${w}x${h}`, '-vframes', String(maxFrames), '-'],
    { maxBuffer: 256 * 1024 * 1024 }
  );
}

function loadWasm() {
  const inst = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(WASM_PATH)), {});
  const exp = inst.exports;
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  return { exp, buf: exp.memory.buffer };
}

function verifyAllDemos() {
  console.log('\n[Verify] Running all 5 generated MP4s through web/chronos_spectral.wasm...');
  const W = 640, H = 360, fBytes = W * H * 4;

  // 1. depth_parallax_dolly.mp4
  {
    const p = path.join(SAMPLES_DIR, 'depth_parallax_dolly.mp4');
    const raw = decodeMp4Rgba(p, W, H, 260);
    const { exp, buf } = loadWasm();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(500.0, 0.30, 0.0, 32.0, 0);
    const dU32 = new Uint32Array(buf, exp.chronos_get_depth_telemetry_ptr(), 32);
    const dF32 = new Float32Array(buf, exp.chronos_get_depth_telemetry_ptr(), 32);
    let metricFrames = 0, maxValid = 0, lastMinZ = 0, lastMaxZ = 0, refusedYawFrames = 0, yawLeaked = 0;
    for (let f = 0; f < 260; f++) {
      new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fBytes).set(raw.subarray(f * fBytes, (f + 1) * fBytes));
      exp.chronos_tick(1 / 30);
      if (f >= 20 && f < 190) {
        if (dU32[2] === 2 && dU32[4] === 2) {
          metricFrames++;
          lastMinZ = dF32[30];
          lastMaxZ = dF32[31];
        }
        if (dU32[5] > maxValid) maxValid = dU32[5];
      } else if (f >= 210 && f < 260) {
        if (dU32[2] === 4 && dU32[5] === 0) refusedYawFrames++;
        if (dU32[5] > 0) yawLeaked++;
      }
    }
    assert(metricFrames >= 120 && maxValid >= 40, `depth_parallax_dolly Phase 1 failed: metricFrames=${metricFrames}, maxValid=${maxValid}`);
    assert(refusedYawFrames >= 30 && yawLeaked === 0, `depth_parallax_dolly Phase 2 yaw guard failed: refused=${refusedYawFrames}, leaked=${yawLeaked}`);
    console.log(`  [PASS] depth_parallax_dolly.mp4: metricFrames=${metricFrames}/170 (maxTracks=${maxValid}, lateralZ=[${lastMinZ.toFixed(2)}..${lastMaxZ.toFixed(2)}]m), pure-yaw refused=${refusedYawFrames}/50 (leaked=0)`);
  }

  // 2. depth_foe_approach.mp4
  {
    const p = path.join(SAMPLES_DIR, 'depth_foe_approach.mp4');
    const raw = decodeMp4Rgba(p, W, H, 160);
    const { exp, buf } = loadWasm();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_depth_mode(1);
    exp.chronos_set_depth_calib(500.0, 0.8522, 0.0, 32.0, 0);
    const dU32 = new Uint32Array(buf, exp.chronos_get_depth_telemetry_ptr(), 32);
    const dF32 = new Float32Array(buf, exp.chronos_get_depth_telemetry_ptr(), 32);
    let maxTtc = 0, bestR2 = 0, radialMetric = 0, bestFoeErr = Infinity;
    for (let f = 0; f < 160; f++) {
      new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fBytes).set(raw.subarray(f * fBytes, (f + 1) * fBytes));
      exp.chronos_tick(1 / 30);
      if (dU32[2] === 2 && dU32[4] === 2) radialMetric++;
      if (dU32[7] > maxTtc) maxTtc = dU32[7];
      if (dU32[7] > 0 && dF32[25] > bestR2) bestR2 = dF32[25];
      if (dU32[8] >= 4) {
        const e = Math.hypot(dF32[22] - 355.3, dF32[23] - 171.2);
        if (e < bestFoeErr) bestFoeErr = e;
      }
    }
    assert(radialMetric >= 80 && maxTtc >= 10 && bestR2 >= 0.98 && bestFoeErr < 25,
      `depth_foe_approach failed: radialMetric=${radialMetric}, maxTtc=${maxTtc}, bestR2=${bestR2.toFixed(4)}, foeErr=${bestFoeErr.toFixed(1)}px`);
    console.log(`  [PASS] depth_foe_approach.mp4: radialMetric=${radialMetric}/160, maxTtcTracks=${maxTtc}, bestR2=${bestR2.toFixed(4)}, bestFoeErr=${bestFoeErr.toFixed(1)} px`);
  }

  // 3. hodge_cylinder_circulation.mp4
  {
    const p = path.join(SAMPLES_DIR, 'hodge_cylinder_circulation.mp4');
    const raw = decodeMp4Rgba(p, W, H, 90);
    const { exp, buf } = loadWasm();
    exp.chronos_set_swarm_mode(2);
    exp.chronos_set_hodge_mode(1);
    const hU32 = new Uint32Array(buf, exp.chronos_get_hodge_telemetry_ptr(), 32);
    const hF32 = new Float32Array(buf, exp.chronos_get_hodge_telemetry_ptr(), 32);
    let readyFrames = 0, b1Hits = 0, lastEx = 0, lastCo = 0, lastHa = 0, lastB1 = 0;
    for (let f = 0; f < 90; f++) {
      new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fBytes).set(raw.subarray(f * fBytes, (f + 1) * fBytes));
      exp.chronos_tick(1 / 30);
      if (hU32[0] === 2) {
        readyFrames++;
        if (hU32[7] >= 1) b1Hits++;
        lastB1 = hU32[7];
        lastEx = hF32[12]; lastCo = hF32[13]; lastHa = hF32[14];
      }
    }
    assert(readyFrames >= 40 && b1Hits >= 20 && lastHa > 0.05 && lastCo > 0.10,
      `hodge_cylinder_circulation failed: ready=${readyFrames}, b1Hits=${b1Hits}, b1=${lastB1}, Ex=${lastEx.toFixed(2)}, Co=${lastCo.toFixed(2)}, Ha=${lastHa.toFixed(2)}`);
    console.log(`  [PASS] hodge_cylinder_circulation.mp4: ready=${readyFrames}/90, b1=${lastB1}, Ex=${(lastEx*100).toFixed(0)}% Co=${(lastCo*100).toFixed(0)}% Ha=${(lastHa*100).toFixed(0)}%`);
  }

  // 4. rs_vibrometry_247hz.mp4
  {
    const p = path.join(SAMPLES_DIR, 'rs_vibrometry_247hz.mp4');
    const raw = decodeMp4Rgba(p, W, H, 80);
    const { exp, buf } = loadWasm();
    exp.chronos_set_swarm_mode(0);
    assert(exp.chronos_set_rs_calib(1, 40.0, 200.0, 48000.0, 500.0) === 1, 'set_rs_calib failed');
    const vU32 = new Uint32Array(buf, exp.chronos_get_vibro_telemetry_ptr(), 32);
    const vF32 = new Float32Array(buf, exp.chronos_get_vibro_telemetry_ptr(), 32);
    const vPl = new Float32Array(buf, exp.chronos_get_vibro_planes_ptr(), 3 * 64 * 36 + 128);
    for (let f = 0; f < 80; f++) {
      new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fBytes).set(raw.subarray(f * fBytes, (f + 1) * fBytes));
      const tUs = BigInt(Math.round((f / 30) * 1_000_000));
      exp.chronos_set_ingest_meta(
        1, f + 1, Number(BigInt.asUintN(32, tUs)), Number(BigInt.asUintN(32, tUs >> 32n)),
        -1.0, 18, W, H, 0, 0, W, H, 0, 0, 0, 0, W, H, 0, 1
      );
      exp.chronos_tick(1 / 30);
    }
    let leftAmp = 0, rightAmp = 0;
    for (let y = 4; y < 32; y++) {
      for (let x = 5; x < 25; x++) leftAmp += vPl[y * 64 + x];
      for (let x = 32; x < 60; x++) rightAmp += vPl[y * 64 + x];
    }
    leftAmp /= 28 * 20;
    rightAmp /= 28 * 28;
    assert(vU32[0] === 3 && Math.abs(vF32[5] - 247.5) < 3.0 && leftAmp > 0.35 && rightAmp < 0.05,
      `rs_vibrometry_247hz failed: st=${vU32[0]} reason=0x${vU32[2].toString(16)} domF=${vF32[5].toFixed(2)}Hz amp=${vF32[12].toFixed(3)}px left=${leftAmp.toFixed(3)} right=${rightAmp.toFixed(3)}`);
    console.log(`  [PASS] rs_vibrometry_247hz.mp4: READY domF=${vF32[5].toFixed(2)} Hz (amp=${vF32[12].toFixed(3)} px, H=${vF32[8].toFixed(3)}), leftAmp=${leftAmp.toFixed(3)} vs quietRight=${rightAmp.toFixed(4)}`);
  }

  // 5. fovea_720p_micro_detail.mp4
  {
    const p = path.join(SAMPLES_DIR, 'fovea_720p_micro_detail.mp4');
    const info = execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,nb_frames', '-of', 'csv=p=0', p], { encoding: 'utf8' }).trim();
    assert(info.startsWith('1280,720,'), `fovea_720p_micro_detail expected 1280x720, got ${info}`);
    const raw = decodeMp4Rgba(p, W, H, 60);
    const { exp, buf } = loadWasm();
    exp.chronos_set_swarm_mode(1);
    const telem = new Float32Array(buf, exp.chronos_get_telemetry_ptr(), 1568);
    let maxActive = 0;
    for (let f = 0; f < 60; f++) {
      new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), fBytes).set(raw.subarray(f * fBytes, (f + 1) * fBytes));
      exp.chronos_tick(1 / 30);
      let act = 0;
      for (let m = 0; m < 16; m++) if (telem[32 + m * 56] > 0.5) act++;
      if (act > maxActive) maxActive = act;
    }
    assert(maxActive >= 2, `fovea_720p_micro_detail expected >=2 active Hero tracks, got ${maxActive}`);
    console.log(`  [PASS] fovea_720p_micro_detail.mp4: 1280x720 H.264 verified (${info}), maxHeroActive=${maxActive}`);
  }
}

if (!VERIFY_ONLY) {
  console.log('[Generate] Building 5 v3.2.0 interactive mode demo MP4s in web/samples/...');
  const files = [];
  files.push(await buildDepthParallaxDolly());
  files.push(await buildDepthFoeApproach());
  files.push(await buildHodgeCylinderCirculation());
  files.push(await buildRsVibrometry247Hz());
  files.push(await buildFovea720pMicroDetail());
  let total = 0;
  for (const f of files) {
    const sz = fs.statSync(f).size;
    total += sz;
    console.log(`  wrote ${path.basename(f)}: ${(sz / 1024).toFixed(1)} KiB`);
  }
  console.log(`  combined size: ${(total / (1024 * 1024)).toFixed(2)} MiB`);
}
verifyAllDemos();
console.log(VERIFY_ONLY
  ? '[PASS] All 5 v3.2.0 demo MP4s kernel-verified.'
  : '[PASS] All 5 v3.2.0 demo MP4s generated and kernel-verified.'
);
