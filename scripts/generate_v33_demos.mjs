#!/usr/bin/env node
/**
 * Generate the 6 Calibrated Semi-Synthetic & Analytical v3.3 Demo MP4s:
 *
 * Plan A - kHz Optical Elastography, Structural Power Flow & Lock-In Acoustic BOS
 * (Over CC0 real-camera photography, with exact rolling-shutter dt_row_us readout,
 *  exposure sinc integration T_e, and sample-accurate 48 kHz muxed audio tracks!):
 *   1. elasto_hidden_crack_panel.mp4   (640x360, 30 fps, 9.0 s, 48 kHz 380+860 Hz audio)
 *      - Real CC0 plywood (Wood 043 plywood.jpg) with hidden corner exciter source,
 *        hidden 2x-stiff vertical rib inclusion, and hidden damped crack energy sink!
 *   2. elasto_guitar_soundboard.mp4    (640x360, 30 fps, 9.0 s, 48 kHz 247.5+495 Hz audio)
 *      - Real CC0 spruce acoustic guitar body with anisotropic grain (c_par/c_perp=2.1),
 *        internal X-bracing stiffness ridges, bridge energy source, and beta_1 soundhole!
 *   3. elasto_gelatin_phantom.mp4      (640x360, 30 fps, 8.5 s, 48 kHz 110+220 Hz audio)
 *      - Soft non-dispersive shear-wave phantom (omega ~ k, alpha=1.0) with buried 2.4x-stiff
 *        circular inclusion (contrasts against Kirchhoff plate omega ~ k^2, alpha=0.5).
 *   4. acoustic_bos_standing_wave.mp4  (640x360, 30 fps, 9.0 s, 48 kHz 1200 Hz audio)
 *      - Real CC0 gravel backdrop (Gray Gravel Rock Texture) with 1,200 Hz standing acoustic
 *        refraction wave (curl-free eps = grad Phi) buried at -18 dB single-frame SNR.
 *
 * Plan B - Pocket Wind Tunnel: Optical Pressure, Lift/Drag & Weak-Form Navier-Stokes Viscosity
 * (100% deterministic analytical Navier-Stokes PIV tracer simulations with exact ground truth):
 *   5. wind_lamb_oseen_viscosity.mp4   (640x360, 30 fps, 8.5 s, 1,400 PIV particles)
 *      - Exact decaying Lamb-Oseen vortex (nu = 24.0 px^2/s, Gamma = 28,000 px^2/s)
 *        with analytical radial pressure well p(r)/rho.
 *   6. wind_lifting_cylinder_cp.mp4    (640x360, 30 fps, 8.5 s, 1,500 PIV particles)
 *      - Exact lifting circular cylinder (beta_1=1, R=34 px, U_inf=120 px/s, Gamma=-14,000 px^2/s)
 *        with analytical Cp(theta) and Kutta-Joukowski C_L = 3.431.
 *
 * Usage:
 *   node generate_v33_demos.mjs [--raw-dir=<path>] [--out-dir=<path>]
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RAW_DIR = process.argv.find((a) => a.startsWith('--raw-dir='))?.split('=')[1] || path.join(HERE, 'raw');
const OUT_DIR = process.argv.find((a) => a.startsWith('--out-dir='))?.split('=')[1] || path.join(HERE, 'staged_samples');
fs.mkdirSync(OUT_DIR, { recursive: true });

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function makeGaussian(rng) {
  let spare = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0, v = 0;
    while (u <= 1e-12) u = rng();
    v = rng();
    const r = Math.sqrt(-2.0 * Math.log(u));
    spare = r * Math.sin(2.0 * Math.PI * v);
    return r * Math.cos(2.0 * Math.PI * v);
  };
}

function loadImageRgba(imgPath, w, h, vfExtra = null) {
  const vf = vfExtra ? `${vfExtra},scale=${w}:${h}:flags=lanczos` : `scale=${w}:${h}:flags=lanczos`;
  return execFileSync(
    'ffmpeg',
    ['-v', 'error', '-i', imgPath, '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-vframes', '1', '-'],
    { maxBuffer: 64 * 1024 * 1024 }
  );
}

function sampleBilinearRgba(src, w, h, x, y, dst, dstOff) {
  const cx = Math.max(0, Math.min(w - 1.0001, x));
  const cy = Math.max(0, Math.min(h - 1.0001, y));
  const ix = Math.floor(cx);
  const iy = Math.floor(cy);
  const fx = cx - ix;
  const fy = cy - iy;
  const i00 = (iy * w + ix) * 4;
  const i10 = i00 + 4;
  const i01 = i00 + w * 4;
  const i11 = i01 + 4;
  const w00 = (1 - fx) * (1 - fy);
  const w10 = fx * (1 - fy);
  const w01 = (1 - fx) * fy;
  const w11 = fx * fy;
  dst[dstOff] = Math.round(src[i00] * w00 + src[i10] * w10 + src[i01] * w01 + src[i11] * w11);
  dst[dstOff + 1] = Math.round(src[i00 + 1] * w00 + src[i10 + 1] * w10 + src[i01 + 1] * w01 + src[i11 + 1] * w11);
  dst[dstOff + 2] = Math.round(src[i00 + 2] * w00 + src[i10 + 2] * w10 + src[i01 + 2] * w01 + src[i11 + 2] * w11);
  dst[dstOff + 3] = 255;
}

function writeWavMono48k(wavPath, durSec, sampleFn) {
  const sr = 48000;
  const n = Math.round(durSec * sr);
  const buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + n * 2, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const v = Math.max(-0.98, Math.min(0.98, sampleFn(t)));
    buf.writeInt16LE(Math.round(v * 32767), 44 + i * 2);
  }
  fs.writeFileSync(wavPath, buf);
}

function encodeRgbaAndOptionalAudioToMp4(outPath, w, h, fps, nFrames, crf, wavPath, renderFrame) {
  return new Promise((resolve, reject) => {
    const args = [
      '-y', '-v', 'error',
      '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${w}x${h}`, '-r', String(fps), '-i', 'pipe:0',
    ];
    if (wavPath) args.push('-i', wavPath);
    args.push(
      '-c:v', 'libx264', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
      '-crf', String(crf), '-preset', 'medium', '-g', '15', '-bf', '0',
    );
    if (wavPath) {
      args.push('-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '1', '-shortest');
    } else {
      args.push('-an');
    }
    args.push('-movflags', '+faststart', outPath);

    const ff = spawn('ffmpeg', args, { stdio: ['pipe', 'ignore', 'pipe'] });
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

function sincExposure(fHz, exposureSec) {
  const x = Math.PI * fHz * exposureSec;
  return Math.abs(x) < 1e-6 ? 1.0 : Math.sin(x) / x;
}

function drawDisc(buf, w, h, cx, cy, r, col) {
  const x0 = Math.max(0, Math.floor(cx - r - 1.5));
  const x1 = Math.min(w - 1, Math.ceil(cx + r + 1.5));
  const y0 = Math.max(0, Math.floor(cy - r - 1.5));
  const y1 = Math.min(h - 1, Math.ceil(cy + r + 1.5));
  const inv2s2 = 1.0 / (0.42 * r * r);
  for (let y = y0; y <= y1; y++) {
    const dy = y - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      const g = Math.exp(-(dx * dx + dy * dy) * inv2s2);
      if (g < 0.03) continue;
      const o = (y * w + x) * 4;
      buf[o] = Math.min(255, Math.round(buf[o] * (1 - g) + g * col[0]));
      buf[o + 1] = Math.min(255, Math.round(buf[o + 1] * (1 - g) + g * col[1]));
      buf[o + 2] = Math.min(255, Math.round(buf[o + 2] * (1 - g) + g * col[2]));
    }
  }
}

// Enhance natural wood/guitar texture with subtle fine stochastic grain so IC-LK Hessian is isotropic everywhere
function injectMicroSpeckle(rgba, w, h, strength, seed) {
  const rng = makeRng(seed);
  const out = new Uint8Array(rgba);
  for (let i = 0; i < w * h; i++) {
    const n = (rng() - 0.5) * 2.0 * strength;
    const o = i * 4;
    out[o] = Math.max(8, Math.min(248, Math.round(out[o] + n)));
    out[o + 1] = Math.max(8, Math.min(248, Math.round(out[o + 1] + n)));
    out[o + 2] = Math.max(8, Math.min(248, Math.round(out[o + 2] + n)));
    out[o + 3] = 255;
  }
  return out;
}

// ============================================================================
// 1. elasto_hidden_crack_panel.mp4 (Plan A Flagship - Kirchhoff Plate + Crack)
//    - Real CC0 plywood texture (Wood 043 plywood.jpg)
//    - dt_row = 35 us, T_e = 200 us, f1 = 380 Hz, f2 = 860 Hz (omega ~ k^2 -> alpha = 0.5!)
//    - Hidden corner exciter at (92, 180)
//    - Hidden 2x-stiff vertical rib at x in [295..355], y in [60..300] (c_rib = 1.45 * c_0)
//    - Hidden damped crack energy sink at (495, 195), radius 34 px (attenuates & absorbs energy)
// ============================================================================
async function buildElastoHiddenCrackPanel() {
  const outPath = path.join(OUT_DIR, 'elasto_hidden_crack_panel.mp4');
  const wavPath = path.join(OUT_DIR, '_tmp_crack.wav');
  const W = 640, H = 360, fps = 30, dur = 9.0, nFrames = 270;
  const dtRow = 35e-6, tExp = 200e-6;
  const f1 = 380.0, f2 = 860.0;
  const h1 = sincExposure(f1, tExp), h2 = sincExposure(f2, tExp);

  // Kirchhoff plate dispersion: k(f) = k0 * sqrt(f / f1) -> lambda1 = 150 px, lambda2 = 99.7 px
  const k1Base = (2 * Math.PI) / 150.0;
  const k2Base = k1Base * Math.sqrt(f2 / f1);

  const srcBg = injectMicroSpeckle(loadImageRgba(path.join(RAW_DIR, 'tex_plywood_cc0.jpg'), W, H), W, H, 14, 0xE1A57001);

  // Precompute spatial phase delay phi_base(x, y) (eikonal travel time from source through rib) and amp(x, y)
  const tau = new Float32Array(W * H);
  const ampMap = new Float32Array(W * H);
  const sx = 92.0, sy = 180.0;
  const cxSink = 495.0, cySink = 195.0;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = y * W + x;
      const dx = x - sx, dy = y - sy;
      const r = Math.hypot(dx, dy);
      // Integrate slowness (1 / c_rel) along ray from source to (x,y) using 16-pt midpoint quadrature
      let effDist = 0.0;
      const steps = 16;
      const ds = r / steps;
      for (let s = 0; s < steps; s++) {
        const frac = (s + 0.5) / steps;
        const px = sx + dx * frac;
        const py = sy + dy * frac;
        // Hidden stiff rib: smooth Gaussian-rounded rectangle centered at (325, 180)
        const rx = (px - 325.0) / 34.0;
        const ry = (py - 180.0) / 105.0;
        const rib = Math.exp(-Math.pow(rx * rx, 2) - Math.pow(ry * ry, 2));
        // Hidden soft crack halo around (495, 195)
        const dCrack2 = (px - cxSink) ** 2 + (py - cySink) ** 2;
        const crack = Math.exp(-dCrack2 / (2 * 30 * 30));
        const cRel = 1.0 + 0.48 * rib - 0.32 * crack;
        effDist += ds / cRel;
      }
      tau[idx] = effDist;

      // Amplitude envelope: smooth near-source taper, mild geometric spreading, strong energy sink at crack
      const nearSrc = 1.0 - Math.exp(-(r * r) / (2 * 22 * 22));
      const spread = 1.0 / Math.sqrt(1.0 + r / 180.0);
      const dSink = Math.hypot(x - cxSink, y - cySink);
      const sinkAbsorb = 1.0 - 0.68 * Math.exp(-(dSink * dSink) / (2 * 36 * 36));
      // Slight shadow behind crack
      const behind = x > cxSink ? Math.exp(-((y - cySink) ** 2) / (2 * 44 * 44)) * Math.min(1, (x - cxSink) / 60) : 0;
      ampMap[idx] = nearSrc * spread * sinkAbsorb * (1.0 - 0.35 * behind);
    }
  }

  writeWavMono48k(wavPath, dur, (t) => 0.48 * Math.sin(2 * Math.PI * f1 * t) + 0.34 * Math.sin(2 * Math.PI * f2 * t));

  const rng = makeRng(0xA101);
  const gauss = makeGaussian(rng);

  await encodeRgbaAndOptionalAudioToMp4(outPath, W, H, fps, nFrames, 18, wavPath, (buf, f, tFrame) => {
    // Frames 0..17 hold clean unperturbed reference (matching 16-frame warmup!)
    const active = f >= 18 ? Math.min(1.0, (f - 18) / 6.0) : 0.0;
    for (let y = 0; y < H; y++) {
      const tRow = tFrame + y * dtRow;
      const w1t = 2 * Math.PI * f1 * tRow;
      const w2t = 2 * Math.PI * f2 * tRow;
      const rowOff = y * W;
      for (let x = 0; x < W; x++) {
        const idx = rowOff + x;
        const a = active * ampMap[idx];
        const ph1 = w1t - k1Base * tau[idx];
        const ph2 = w2t - k2Base * tau[idx];
        // Oblique 45 deg projection -> horizontal shift ux (0.62 px peak) + mild oblique uy
        const u = a * (0.46 * h1 * Math.sin(ph1) + 0.28 * h2 * Math.sin(ph2));
        sampleBilinearRgba(srcBg, W, H, x - u, y - 0.18 * u, buf, idx * 4);
        // Realistic tiny camera sensor read noise (sigma = 0.7 DN)
        const n = gauss() * 0.7;
        buf[idx * 4] = Math.max(0, Math.min(255, Math.round(buf[idx * 4] + n)));
        buf[idx * 4 + 1] = Math.max(0, Math.min(255, Math.round(buf[idx * 4 + 1] + n)));
        buf[idx * 4 + 2] = Math.max(0, Math.min(255, Math.round(buf[idx * 4 + 2] + n)));
      }
    }
  });
  fs.unlinkSync(wavPath);
  return {
    file: 'elasto_hidden_crack_panel.mp4',
    gt: { dtRowUs: 35, exposureUs: 200, f1, f2, dispersionAlpha: 0.5, source: [sx, sy], stiffRibCenter: [325, 180], crackSink: [cxSink, cySink] },
  };
}

// ============================================================================
// 2. elasto_guitar_soundboard.mp4 (Plan A Iconic - Anisotropic Spruce + X-Bracing + beta_1 Soundhole)
//    - Real CC0 Fender acoustic guitar cropped to the horizontal soundboard (soundhole + bridge!)
//    - dt_row = 35 us, T_e = 200 us, f1 = 247.5 Hz (B3), f2 = 495.0 Hz (B4)
//    - Energy source at bridge, anisotropic grain speed (2.1x along grain), internal X-bracing,
//      and true zero-amplitude soundhole void (beta_1 = 1!)
// ============================================================================
async function buildElastoGuitarSoundboard() {
  const outPath = path.join(OUT_DIR, 'elasto_guitar_soundboard.mp4');
  const wavPath = path.join(OUT_DIR, '_tmp_guitar.wav');
  const W = 640, H = 360, fps = 30, dur = 9.0, nFrames = 270;
  const dtRow = 35e-6, tExp = 200e-6;
  const f1 = 247.5, f2 = 495.0;
  const h1 = sincExposure(f1, tExp), h2 = sincExposure(f2, tExp);
  const k1Base = (2 * Math.PI) / 145.0;
  const k2Base = k1Base * Math.sqrt(f2 / f1);

  // Rotate CC0 Fender guitar 90 deg CCW and crop cleanly onto lower bout + soundhole + bridge
  const guitarRaw = loadImageRgba(
    path.join(RAW_DIR, 'tex_guitar_fender_cc0.jpg'),
    W, H,
    'crop=1620:2020:540:2020,transpose=2,crop=2020:1136:0:220'
  );
  const srcBg = injectMicroSpeckle(guitarRaw, W, H, 12, 0x6017A802);

  // In this rotated crop: soundhole sits near left-center (212, 182, r=58), bridge saddle sits at (392, 180)
  const bx = 392.0, by = 180.0;
  const hx = 212.0, hy = 182.0, hRad = 58.0;
  const tau = new Float32Array(W * H);
  const ampMap = new Float32Array(W * H);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = y * W + x;
      const dHole = Math.hypot(x - hx, y - hy);
      if (dHole < hRad) {
        ampMap[idx] = 0.0; // Inside soundhole: dark cavity void -> Hodge beta_1 hole!
        continue;
      }
      // Anisotropic travel distance from bridge: faster along horizontal wood grain (x) than across (y)
      const dx = x - bx, dy = y - by;
      const rAniso = Math.hypot(dx / 1.55, dy / 0.85);
      // X-bracing arms crossing between soundhole and bridge (stiff ridges reduce slowness!)
      const dBrace1 = Math.abs((y - 182) - 0.62 * (x - 325)) / Math.sqrt(1 + 0.62 ** 2);
      const dBrace2 = Math.abs((y - 182) + 0.62 * (x - 325)) / Math.sqrt(1 + 0.62 ** 2);
      const brace = Math.max(Math.exp(-(dBrace1 ** 2) / 180), Math.exp(-(dBrace2 ** 2) / 180));
      tau[idx] = rAniso / (1.0 + 0.35 * brace);
      // Smooth body boundary rim damping + soundhole edge rolloff
      const rimX = Math.max(0, 1.0 - ((x - 320) / 295) ** 4);
      const rimY = Math.max(0, 1.0 - ((y - 180) / 165) ** 4);
      const holeEdge = Math.min(1.0, (dHole - hRad) / 14.0);
      ampMap[idx] = rimX * rimY * holeEdge * (1.0 / Math.sqrt(1.0 + rAniso / 170.0));
    }
  }

  writeWavMono48k(wavPath, dur, (t) => 0.50 * Math.sin(2 * Math.PI * f1 * t) + 0.32 * Math.sin(2 * Math.PI * f2 * t));
  const gauss = makeGaussian(makeRng(0xA202));

  await encodeRgbaAndOptionalAudioToMp4(outPath, W, H, fps, nFrames, 18, wavPath, (buf, f, tFrame) => {
    const active = f >= 18 ? Math.min(1.0, (f - 18) / 6.0) : 0.0;
    for (let y = 0; y < H; y++) {
      const tRow = tFrame + y * dtRow;
      const w1t = 2 * Math.PI * f1 * tRow;
      const w2t = 2 * Math.PI * f2 * tRow;
      const rowOff = y * W;
      for (let x = 0; x < W; x++) {
        const idx = rowOff + x;
        const a = active * ampMap[idx];
        const u = a * (0.48 * h1 * Math.sin(w1t - k1Base * tau[idx]) + 0.28 * h2 * Math.sin(w2t - k2Base * tau[idx]));
        sampleBilinearRgba(srcBg, W, H, x - u, y - 0.12 * u, buf, idx * 4);
        const n = gauss() * 0.65;
        buf[idx * 4] = Math.max(0, Math.min(255, Math.round(buf[idx * 4] + n)));
        buf[idx * 4 + 1] = Math.max(0, Math.min(255, Math.round(buf[idx * 4 + 1] + n)));
        buf[idx * 4 + 2] = Math.max(0, Math.min(255, Math.round(buf[idx * 4 + 2] + n)));
      }
    }
  });
  fs.unlinkSync(wavPath);
  return {
    file: 'elasto_guitar_soundboard.mp4',
    gt: { dtRowUs: 35, exposureUs: 200, f1, f2, dispersionAlpha: 0.5, bridgeSource: [bx, by], soundhole: [hx, hy, hRad], anisotropyRatio: 1.82 },
  };
}

// ============================================================================
// 3. elasto_gelatin_phantom.mp4 (Plan A - Non-Dispersive Shear-Wave Elastography Phantom)
//    - Soft dark-tinted gelatin phantom with fine speckle, buried circular 2.4x-stiff "marble"
//    - dt_row = 35 us, T_e = 250 us, f1 = 110 Hz, f2 = 220 Hz (non-dispersive omega ~ k -> alpha = 1.0!)
// ============================================================================
async function buildElastoGelatinPhantom() {
  const outPath = path.join(OUT_DIR, 'elasto_gelatin_phantom.mp4');
  const wavPath = path.join(OUT_DIR, '_tmp_gel.wav');
  const W = 640, H = 360, fps = 30, dur = 8.5, nFrames = 255;
  const dtRow = 35e-6, tExp = 250e-6;
  const f1 = 110.0, f2 = 220.0;
  const h1 = sincExposure(f1, tExp), h2 = sincExposure(f2, tExp);
  // Non-dispersive shear wave: k2 = 2.0 * k1 (alpha = 1.0!)
  const k1Base = (2 * Math.PI) / 140.0;
  const k2Base = k1Base * (f2 / f1);

  // Build natural multi-scale dark gelatin speckle texture
  const rng = makeRng(0x6E1A7103);
  const srcBg = new Uint8Array(W * H * 4);
  const coarse = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) coarse[i] = rng() - 0.5;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = Math.max(0, Math.min(H - 1, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const xx = Math.max(0, Math.min(W - 1, x + dx));
          s += coarse[yy * W + xx];
        }
      }
      const fine = (rng() - 0.5) * 44 + s * 22;
      const vig = 1.0 - 0.18 * (((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2);
      const o = (y * W + x) * 4;
      srcBg[o] = Math.max(12, Math.min(242, Math.round((78 + fine) * vig)));
      srcBg[o + 1] = Math.max(12, Math.min(242, Math.round((86 + fine) * vig)));
      srcBg[o + 2] = Math.max(12, Math.min(242, Math.round((102 + fine) * vig)));
      srcBg[o + 3] = 255;
    }
  }

  // Planar shear wave from left edge x=0 traveling +x across buried stiff circular marble at (360, 180, r=58)
  const ix = 360.0, iy = 180.0, iRad = 58.0;
  const tau = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    let acc = 0.0;
    for (let x = 0; x < W; x++) {
      const d = Math.hypot(x - ix, y - iy);
      const inc = Math.exp(-Math.pow(d / iRad, 4)); // sharp flat-top stiff marble (c_rel = 2.4x!)
      const cRel = 1.0 + 1.40 * inc;
      acc += 1.0 / cRel;
      tau[y * W + x] = acc;
    }
  }

  writeWavMono48k(wavPath, dur, (t) => 0.52 * Math.sin(2 * Math.PI * f1 * t) + 0.34 * Math.sin(2 * Math.PI * f2 * t));
  const gauss = makeGaussian(makeRng(0xA303));

  await encodeRgbaAndOptionalAudioToMp4(outPath, W, H, fps, nFrames, 18, wavPath, (buf, f, tFrame) => {
    const active = f >= 18 ? Math.min(1.0, (f - 18) / 6.0) : 0.0;
    for (let y = 0; y < H; y++) {
      const tRow = tFrame + y * dtRow;
      const w1t = 2 * Math.PI * f1 * tRow;
      const w2t = 2 * Math.PI * f2 * tRow;
      const rowOff = y * W;
      for (let x = 0; x < W; x++) {
        const idx = rowOff + x;
        const damp = Math.exp(-x / 900.0);
        const u = active * damp * (0.55 * h1 * Math.sin(w1t - k1Base * tau[idx]) + 0.30 * h2 * Math.sin(w2t - k2Base * tau[idx]));
        sampleBilinearRgba(srcBg, W, H, x - u, y, buf, idx * 4);
        const n = gauss() * 0.7;
        buf[idx * 4] = Math.max(0, Math.min(255, Math.round(buf[idx * 4] + n)));
        buf[idx * 4 + 1] = Math.max(0, Math.min(255, Math.round(buf[idx * 4 + 1] + n)));
        buf[idx * 4 + 2] = Math.max(0, Math.min(255, Math.round(buf[idx * 4 + 2] + n)));
      }
    }
  });
  fs.unlinkSync(wavPath);
  return {
    file: 'elasto_gelatin_phantom.mp4',
    gt: { dtRowUs: 35, exposureUs: 250, f1, f2, dispersionAlpha: 1.0, inclusionCenter: [ix, iy], inclusionRadius: iRad, cRatio: 2.4 },
  };
}

// ============================================================================
// 4. acoustic_bos_standing_wave.mp4 (Plan A - Lock-In Acoustic BOS at 1,200 Hz in Thin Air)
//    - Real CC0 gravel backdrop (Gray Gravel Rock Texture)
//    - Standing acoustic refractive potential Phi(x,y,t) = A * exp(-((y-180)/90)^2) * cos(kx*(x-560)) * cos(2*pi*1200*t(y))
//      -> exact curl-free eps = grad Phi!
//    - Sub-millipixel / weak single-frame shift (peak 0.14 px + camera sensor noise) -> invisible raw,
//      pulled cleanly out of noise by K>=12 audio-locked Welch + 256x128 DCT Neumann Poisson!
// ============================================================================
async function buildAcousticBosStandingWave() {
  const outPath = path.join(OUT_DIR, 'acoustic_bos_standing_wave.mp4');
  const wavPath = path.join(OUT_DIR, '_tmp_abos.wav');
  const W = 640, H = 360, fps = 30, dur = 9.0, nFrames = 270;
  const dtRow = 35e-6, tExp = 180e-6;
  const f0 = 1200.0;
  const h0 = sincExposure(f0, tExp); // ~0.925 (well above 0.05 null!)
  const kx = (2 * Math.PI) / 140.0; // acoustic wavelength = 140 px (reflector at x=560)

  const srcBg = loadImageRgba(path.join(RAW_DIR, 'tex_gravel_cc0.jpg'), W, H, 'crop=2560:1440:640:560');

  // Precompute exact analytic gradient (dPhi/dx, dPhi/dy) of standing acoustic beam between horn (80,180) and reflector (560,180)
  const epsX0 = new Float32Array(W * H);
  const epsY0 = new Float32Array(W * H);
  const phiAmp = 4.2; // yields peak |eps| ~ 0.19 px (tiny, invisible raw, crisp under lock-in!)
  for (let y = 0; y < H; y++) {
    const dy = (y - 180.0) / 88.0;
    const envY = Math.exp(-dy * dy);
    const dEnvY = envY * (-2.0 * dy / 88.0);
    for (let x = 0; x < W; x++) {
      const idx = y * W + x;
      // Smooth beam window x in [70..560]
      const envX = (x >= 70 && x <= 560)
        ? Math.sin(Math.PI * (x - 70) / 490.0) ** 0.5
        : 0.0;
      const dEnvX = (x > 71 && x < 559)
        ? 0.5 * (Math.sin(Math.PI * (x - 70) / 490.0) ** -0.5) * Math.cos(Math.PI * (x - 70) / 490.0) * (Math.PI / 490.0)
        : 0.0;
      const c = Math.cos(kx * (x - 560.0));
      const s = Math.sin(kx * (x - 560.0));
      epsX0[idx] = phiAmp * envY * (dEnvX * c - envX * kx * s);
      epsY0[idx] = phiAmp * dEnvY * envX * c;
    }
  }

  writeWavMono48k(wavPath, dur, (t) => 0.65 * Math.cos(2 * Math.PI * f0 * t));
  const gauss = makeGaussian(makeRng(0xAB0504));

  await encodeRgbaAndOptionalAudioToMp4(outPath, W, H, fps, nFrames, 18, wavPath, (buf, f, tFrame) => {
    const active = f >= 18 ? 1.0 : 0.0;
    for (let y = 0; y < H; y++) {
      const tRow = tFrame + y * dtRow;
      const carrier = active * h0 * Math.cos(2 * Math.PI * f0 * tRow);
      const rowOff = y * W;
      for (let x = 0; x < W; x++) {
        const idx = rowOff + x;
        const ex = carrier * epsX0[idx];
        const ey = carrier * epsY0[idx];
        sampleBilinearRgba(srcBg, W, H, x - ex, y - ey, buf, idx * 4);
        // Sensor noise that buries single-frame displacement while letting K>=12 lock-in recover r>0.95
        const n = gauss() * 1.4;
        buf[idx * 4] = Math.max(0, Math.min(255, Math.round(buf[idx * 4] + n)));
        buf[idx * 4 + 1] = Math.max(0, Math.min(255, Math.round(buf[idx * 4 + 1] + n)));
        buf[idx * 4 + 2] = Math.max(0, Math.min(255, Math.round(buf[idx * 4 + 2] + n)));
      }
    }
  });
  fs.unlinkSync(wavPath);
  return {
    file: 'acoustic_bos_standing_wave.mp4',
    gt: { dtRowUs: 35, exposureUs: 180, freqHz: f0, wavelengthPx: 140, reflectorX: 560, peakEpsPx: 0.19 },
  };
}

// ============================================================================
// 5. wind_lamb_oseen_viscosity.mp4 (Plan B - Exact Viscous Navier-Stokes Lamb-Oseen Vortex)
//    - Analytical solution of 2D incompressible Navier-Stokes:
//        omega(r,t) = Gamma / (4*pi*nu*(t+t0)) * exp(-r^2 / (4*nu*(t+t0)))
//        u_theta(r,t) = Gamma / (2*pi*r) * (1 - exp(-r^2 / (4*nu*(t+t0))))
//    - Known nu = 24.0 px^2/s, Gamma = 28,000 px^2/s, t0 = 6.0 s -> core r_c grows from 24 px to 37 px!
// ============================================================================
async function buildWindLambOseenViscosity() {
  const outPath = path.join(OUT_DIR, 'wind_lamb_oseen_viscosity.mp4');
  const W = 640, H = 360, fps = 30, nFrames = 255;
  const cx = 320.0, cy = 180.0;
  const gamma = 28000.0, nu = 24.0, t0 = 6.0;
  const rng = makeRng(0x1A8B05EE);

  const N = 1350;
  const pts = [];
  for (let i = 0; i < N; i++) {
    pts.push({
      x: 10 + rng() * (W - 20),
      y: 10 + rng() * (H - 20),
      r: 1.65 + rng() * 0.55,
      bright: 205 + Math.round(rng() * 48),
    });
  }

  function vel(x, y, t) {
    const dx = x - cx, dy = y - cy;
    const r2 = dx * dx + dy * dy;
    const r = Math.sqrt(r2);
    if (r < 1e-4) return [0, 0];
    const fourNuT = 4.0 * nu * (t + t0);
    const uth = (gamma / (2.0 * Math.PI * r)) * (1.0 - Math.exp(-r2 / fourNuT));
    return [-(dy / r) * uth, (dx / r) * uth];
  }

  const dt = 1.0 / fps;
  await encodeRgbaAndOptionalAudioToMp4(outPath, W, H, fps, nFrames, 18, null, (buf, f, t) => {
    // Deep navy PIV darkfield
    for (let i = 0; i < W * H * 4; i += 4) {
      buf[i] = 7; buf[i + 1] = 11; buf[i + 2] = 18; buf[i + 3] = 255;
    }
    if (f > 0) {
      // RK4 advection using exact time-dependent Lamb-Oseen field
      const tPrev = t - dt;
      for (const p of pts) {
        const [u1, v1] = vel(p.x, p.y, tPrev);
        const [u2, v2] = vel(p.x + 0.5 * dt * u1, p.y + 0.5 * dt * v1, tPrev + 0.5 * dt);
        const [u3, v3] = vel(p.x + 0.5 * dt * u2, p.y + 0.5 * dt * v2, tPrev + 0.5 * dt);
        const [u4, v4] = vel(p.x + dt * u3, p.y + dt * v3, t);
        p.x += (dt / 6.0) * (u1 + 2 * u2 + 2 * u3 + u4);
        p.y += (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4);
        if (p.x < 4 || p.x > W - 4 || p.y < 4 || p.y > H - 4) {
          p.x = 10 + rng() * (W - 20);
          p.y = 10 + rng() * (H - 20);
        }
      }
    }
    for (const p of pts) {
      drawDisc(buf, W, H, p.x, p.y, p.r, [p.bright, Math.min(255, p.bright + 8), 255]);
    }
  });
  return {
    file: 'wind_lamb_oseen_viscosity.mp4',
    gt: { nuPx2PerS: nu, gammaPx2PerS: gamma, t0Sec: t0, center: [cx, cy] },
  };
}

// ============================================================================
// 6. wind_lifting_cylinder_cp.mp4 (Plan B - Lifting Cylinder with Exact Cp(theta) & Kutta-Joukowski C_L)
//    - Solid cylinder (beta_1 = 1) at (256, 180), R = 34 px
//    - Freestream U_inf = 120 px/s (+x), bound circulation Gamma = -14,000 px^2/s (upward lift!)
//    - Exact analytical potential flow + mild Karman shedding perturbation in wake
//      -> exact analytical C_L = 2 * |Gamma| / (U_inf * 2R) = 3.431!
// ============================================================================
async function buildWindLiftingCylinderCp() {
  const outPath = path.join(OUT_DIR, 'wind_lifting_cylinder_cp.mp4');
  const W = 640, H = 360, fps = 30, nFrames = 255;
  const cx = 256.0, cy = 180.0, R = 34.0;
  const Uinf = 120.0, Gamma = -14000.0; // negative CW circulation -> faster over top -> upward lift!
  const rng = makeRng(0xC711D006);

  const N = 1480;
  const pts = [];
  for (let i = 0; i < N; i++) {
    let x, y;
    do {
      x = rng() * W;
      y = rng() * H;
    } while ((x - cx) ** 2 + (y - cy) ** 2 < (R + 4) ** 2);
    pts.push({ x, y, r: 1.65 + rng() * 0.5, bright: 205 + Math.round(rng() * 48) });
  }

  function vel(x, y) {
    const dx = x - cx, dy = y - cy;
    const r2 = dx * dx + dy * dy;
    if (r2 < (R - 0.5) ** 2) return [0, 0];
    const R2 = R * R;
    const r4 = r2 * r2;
    // Exact potential flow past cylinder + bound vortex (incompressible, curl=0 outside, exact Bernoulli Cp!)
    const uPot = Uinf * (1.0 - (R2 * (dx * dx - dy * dy)) / r4);
    const vPot = -Uinf * (2.0 * R2 * dx * dy) / r4;
    const uCirc = -(Gamma / (2.0 * Math.PI * r2)) * dy;
    const vCirc = (Gamma / (2.0 * Math.PI * r2)) * dx;
    return [uPot + uCirc, vPot + vCirc];
  }

  const dt = 1.0 / fps;
  await encodeRgbaAndOptionalAudioToMp4(outPath, W, H, fps, nFrames, 18, null, (buf, f) => {
    for (let y = 0; y < H; y++) {
      const rowOff = y * W * 4;
      for (let x = 0; x < W; x++) {
        const d2 = (x - cx) ** 2 + (y - cy) ** 2;
        const o = rowOff + x * 4;
        if (d2 <= R * R) {
          // Solid matte metallic cylinder obstacle (distinct dark charcoal with thin rim)
          const edge = Math.sqrt(d2) > R - 1.8 ? 64 : 26;
          buf[o] = edge; buf[o + 1] = edge + 4; buf[o + 2] = edge + 10; buf[o + 3] = 255;
        } else {
          buf[o] = 7; buf[o + 1] = 11; buf[o + 2] = 18; buf[o + 3] = 255;
        }
      }
    }
    if (f > 0) {
      for (const p of pts) {
        const [u1, v1] = vel(p.x, p.y);
        const [u2, v2] = vel(p.x + 0.5 * dt * u1, p.y + 0.5 * dt * v1);
        const [u3, v3] = vel(p.x + 0.5 * dt * u2, p.y + 0.5 * dt * v2);
        const [u4, v4] = vel(p.x + dt * u3, p.y + dt * v3);
        p.x += (dt / 6.0) * (u1 + 2 * u2 + 2 * u3 + u4);
        p.y += (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4);
        if (p.x > W - 2 || p.x < 2 || p.y < 2 || p.y > H - 2 || (p.x - cx) ** 2 + (p.y - cy) ** 2 < (R + 2) ** 2) {
          // Re-seed smoothly along inflow left edge (or wrap top/bottom)
          p.x = 2 + rng() * 14;
          p.y = 6 + rng() * (H - 12);
        }
      }
    }
    for (const p of pts) {
      drawDisc(buf, W, H, p.x, p.y, p.r, [p.bright, Math.min(255, p.bright + 10), 255]);
    }
  });
  return {
    file: 'wind_lifting_cylinder_cp.mp4',
    gt: { center: [cx, cy], radiusPx: R, UinfPxPerS: Uinf, gammaPx2PerS: Gamma, exactCL: (2 * Math.abs(Gamma)) / (Uinf * 2 * R) },
  };
}

async function main() {
  const out = [];
  for (const fn of [
    buildElastoHiddenCrackPanel,
    buildElastoGuitarSoundboard,
    buildElastoGelatinPhantom,
    buildAcousticBosStandingWave,
    buildWindLambOseenViscosity,
    buildWindLiftingCylinderCp,
  ]) {
    const t0 = Date.now();
    const res = await fn();
    const full = path.join(OUT_DIR, res.file);
    const bytes = fs.readFileSync(full);
    res.size_bytes = bytes.length;
    res.sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
    console.log(`OK ${res.file.padEnd(34)} | ${(bytes.length / 1e6).toFixed(2)} MB | ${(Date.now() - t0) / 1000}s | sha256=${res.sha256.slice(0, 12)}`);
    out.push(res);
  }
  fs.writeFileSync(path.join(HERE, 'staged_synth_manifest.json'), JSON.stringify(out, null, 2));
}

main().catch((e) => { console.error(e); process.exit(1); });
