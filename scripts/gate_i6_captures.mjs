#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate I6 Physical Captures, Metadata & Failure Suite
// Verifies the three Wave-1 sample MP4 fixtures (bos_thermal_plume.mp4,
// perspective_flyby_decoy.mp4, bos_handheld_failure.mp4):
//   1. Byte-size budget (~3-4 MiB combined) and SHA-256 integrity.
//   2. FFprobe stream/frame metadata (640x360, H.264 yuv420p, <= 15s,
//      strictly monotonic display PTS) and full RGBA decode integrity.
//   3. Headless Chromium <video> decode, seek, and pixel render check via
//      dev_server.mjs + Chrome CDP (pass --no-chrome to skip browser step).
//   4. Attribution & doc integrity (ATTRIBUTION.md, credits.html,
//      ACQUISITION_AND_FAILURE_MODES.md, zero em dashes, zero internal tags).
//   5. Reproducible Wasm physics & failure-mode checks across all 3 clips
//      (plus automatic Wave-1 BOS/I1 assertions when those exports exist).
// ============================================================================

import crypto from 'node:crypto';
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startChronosServer } from './dev_server.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const SAMPLES_DIR = path.join(ROOT, 'web', 'samples');
const WASM_PATH = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const ATTRIBUTION_PATH = path.join(SAMPLES_DIR, 'ATTRIBUTION.md');
const CREDITS_PATH = path.join(SAMPLES_DIR, 'credits.html');
const DOC_PATH = path.join(ROOT, 'docs', 'ACQUISITION_AND_FAILURE_MODES.md');

const skipChrome = process.argv.includes('--no-chrome');
const jsonFlag = process.argv.includes('--json');

const FIXTURES = [
  {
    file: 'bos_thermal_plume.mp4',
    expectedFps: 25,
    expectedFrames: 225,
    maxBytes: 2 * 1024 * 1024,
    maxDurationSec: 15.0,
    role: 'positive_bos',
  },
  {
    file: 'perspective_flyby_decoy.mp4',
    expectedFps: 25,
    expectedFrames: 225,
    maxBytes: 2 * 1024 * 1024,
    maxDurationSec: 15.0,
    role: 'perspective_decoy',
  },
  {
    file: 'bos_handheld_failure.mp4',
    expectedFps: 25,
    expectedFrames: 200,
    maxBytes: 2 * 1024 * 1024,
    maxDurationSec: 15.0,
    role: 'handheld_failure',
  },
];

function assert(cond, msg) {
  if (!cond) throw new Error(`Gate I6 FAIL: ${msg}`);
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function probeClip(mp4Path) {
  const metaRaw = execFileSync(
    'ffprobe',
    [
      '-v', 'error',
      '-show_entries', 'format=duration,size,bit_rate:stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames',
      '-of', 'json',
      mp4Path,
    ],
    { encoding: 'utf8' }
  );
  const meta = JSON.parse(metaRaw);
  const stream = (meta.streams || [])[0];
  assert(stream, `${mp4Path}: missing video stream`);
  assert(stream.codec_name === 'h264', `${mp4Path}: expected h264, got ${stream.codec_name}`);
  assert(stream.pix_fmt === 'yuv420p', `${mp4Path}: expected yuv420p, got ${stream.pix_fmt}`);
  assert(stream.width === 640 && stream.height === 360, `${mp4Path}: expected 640x360, got ${stream.width}x${stream.height}`);

  const duration = Number(meta.format?.duration);
  assert(Number.isFinite(duration) && duration > 0 && duration <= 15.0, `${mp4Path}: invalid duration ${duration}`);

  // Verify strictly monotonic decoded frame presentation timestamps (best_effort_timestamp_time)
  const framesRaw = execFileSync(
    'ffprobe',
    [
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_entries', 'frame=best_effort_timestamp_time,pkt_pts_time',
      '-of', 'csv=p=0',
      mp4Path,
    ],
    { encoding: 'utf8' }
  );
  const lines = framesRaw.trim().split('\n').filter(Boolean);
  let prevPts = -1.0;
  for (let i = 0; i < lines.length; i++) {
    const valStr = lines[i].split(',')[0];
    const pts = Number(valStr);
    assert(Number.isFinite(pts) && pts >= 0, `${mp4Path}: invalid frame[${i}] pts=${valStr}`);
    assert(pts > prevPts, `${mp4Path}: non-monotonic pts at frame[${i}]: ${pts} <= ${prevPts}`);
    prevPts = pts;
  }

  return {
    width: stream.width,
    height: stream.height,
    codec: stream.codec_name,
    pixFmt: stream.pix_fmt,
    rFrameRate: stream.r_frame_rate,
    nbFrames: lines.length,
    duration,
    lastPts: prevPts,
  };
}

function decodeRgbaFrames(mp4Path, w = 640, h = 360) {
  const frameBytes = w * h * 4;
  return new Promise((resolve, reject) => {
    const proc = spawn(
      'ffmpeg',
      ['-v', 'error', '-i', mp4Path, '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'],
      { stdio: ['ignore', 'pipe', 'pipe'] }
    );
    const frames = [];
    let scratch = Buffer.allocUnsafe(frameBytes);
    let carry = 0;
    let errText = '';
    proc.stderr.on('data', (d) => { errText += d.toString('utf8'); });
    proc.stdout.on('data', (chunk) => {
      let off = 0;
      while (off < chunk.length) {
        const take = Math.min(frameBytes - carry, chunk.length - off);
        chunk.copy(scratch, carry, off, off + take);
        carry += take;
        off += take;
        if (carry === frameBytes) {
          frames.push(new Uint8Array(scratch));
          scratch = Buffer.allocUnsafe(frameBytes);
          carry = 0;
        }
      }
    });
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0) reject(new Error(`ffmpeg decode failed (${code}): ${errText}`));
      else if (carry !== 0) reject(new Error(`truncated trailing ${carry}/${frameBytes} bytes`));
      else resolve(frames);
    });
  });
}

function findChromeBinary() {
  if (process.env.CHROME_BIN && fs.existsSync(process.env.CHROME_BIN)) return process.env.CHROME_BIN;
  const candidates = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/opt/google/chrome/chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium-browser',
    '/usr/bin/chromium',
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

class SimpleCdp {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
  }
  async connect() {
    await new Promise((resolve, reject) => {
      this.ws.addEventListener('open', () => resolve(), { once: true });
      this.ws.addEventListener('error', (e) => reject(e.error || new Error('CDP error')), { once: true });
    });
    this.ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString('utf8'));
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject, method } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`CDP ${method}: ${msg.error.message}`));
        else resolve(msg.result);
      }
    });
  }
  send(method, params = {}, sessionId = undefined) {
    const id = this.nextId++;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, method });
      this.ws.send(JSON.stringify(payload));
    });
  }
  async evalJs(sessionId, expression) {
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId);
    if (r?.exceptionDetails) {
      throw new Error(`Browser JS exception: ${r.exceptionDetails.text || r.exceptionDetails.exception?.description}`);
    }
    return r?.result?.value;
  }
  close() {
    try { this.ws.close(); } catch {}
  }
}

async function verifyChromiumVideoDecode(fixtureList) {
  const chromeBin = findChromeBinary();
  assert(chromeBin, 'Google Chrome / Chromium binary not found (set CHROME_BIN or pass --no-chrome)');
  const srv = await startChronosServer({ port: 0, host: '127.0.0.1' });
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chronos-i6-chrome-'));
  const chromeArgs = [
    '--headless=new',
    '--remote-debugging-port=0',
    `--user-data-dir=${userDataDir}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--autoplay-policy=no-user-gesture-required',
    '--mute-audio',
    '--window-size=1280,720',
  ];
  if (process.platform === 'linux') {
    chromeArgs.push('--no-sandbox', '--disable-dev-shm-usage', '--disable-features=dbus');
  }
  const proc = spawn(chromeBin, chromeArgs, { stdio: ['ignore', 'ignore', 'pipe'] });
  let cdp = null;
  try {
    const wsUrl = await new Promise((resolve, reject) => {
      let stderrBuf = '';
      const timer = setTimeout(() => reject(new Error(`Chrome DevTools timeout: ${stderrBuf}`)), 25000);
      proc.stderr.on('data', (d) => {
        stderrBuf += d.toString('utf8');
        const m = stderrBuf.match(/DevTools listening on (ws:\/\/[^\s]+)/);
        if (m) {
          clearTimeout(timer);
          resolve(m[1]);
        }
      });
      proc.on('error', (e) => { clearTimeout(timer); reject(e); });
    });
    cdp = new SimpleCdp(wsUrl);
    await cdp.connect();
    const { targetId } = await cdp.send('Target.createTarget', { url: `http://127.0.0.1:${srv.port}/samples/credits.html` });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    await cdp.send('Runtime.enable', {}, sessionId);
    await cdp.send('Page.enable', {}, sessionId);

    // Wait for credits.html readiness
    for (let t = 0; t < 40; t++) {
      const ready = await cdp.evalJs(sessionId, 'document.readyState === "complete" && document.body.innerText.includes("Karen Roe")');
      if (ready) break;
      await new Promise((r) => setTimeout(r, 100));
    }

    const resList = [];
    for (const fx of fixtureList) {
      const out = await cdp.evalJs(
        sessionId,
        `(async () => {
          const v = document.createElement('video');
          v.muted = true;
          v.playsInline = true;
          v.preload = 'auto';
          v.src = './${fx.file}';
          await new Promise((resolve, reject) => {
            const to = setTimeout(() => reject(new Error('video load timeout for ${fx.file}')), 10000);
            v.addEventListener('loadeddata', () => { clearTimeout(to); resolve(); }, { once: true });
            v.addEventListener('error', () => { clearTimeout(to); reject(new Error('video error code ' + (v.error ? v.error.code : 'unknown'))); }, { once: true });
          });
          const targetT = Math.min(v.duration * 0.5, 4.0);
          v.currentTime = targetT;
          await new Promise((resolve, reject) => {
            const to = setTimeout(() => reject(new Error('video seek timeout for ${fx.file}')), 8000);
            v.addEventListener('seeked', () => { clearTimeout(to); resolve(); }, { once: true });
          });
          const c = document.createElement('canvas');
          c.width = v.videoWidth;
          c.height = v.videoHeight;
          const ctx = c.getContext('2d');
          ctx.drawImage(v, 0, 0);
          const px = ctx.getImageData(0, 0, 64, 64).data;
          let sum = 0, sumSq = 0;
          for (let i = 0; i < px.length; i += 4) {
            const luma = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
            sum += luma;
            sumSq += luma * luma;
          }
          const n = px.length / 4;
          const mean = sum / n;
          const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
          return {
            width: v.videoWidth,
            height: v.videoHeight,
            duration: v.duration,
            readyState: v.readyState,
            currentTime: v.currentTime,
            lumaStd: Number(std.toFixed(3)),
          };
        })()`
      );
      assert(out.width === 640 && out.height === 360, `Chrome <video> ${fx.file}: expected 640x360, got ${out.width}x${out.height}`);
      assert(Number.isFinite(out.duration) && out.duration > 5.0 && out.duration <= fx.maxDurationSec, `Chrome <video> ${fx.file}: bad duration ${out.duration}`);
      assert(out.readyState >= 2, `Chrome <video> ${fx.file}: readyState=${out.readyState}`);
      assert(out.lumaStd > 5.0, `Chrome <video> ${fx.file}: blank/flat decoded frame lumaStd=${out.lumaStd}`);
      resList.push({ file: fx.file, ...out });
    }
    return resList;
  } finally {
    if (cdp) cdp.close();
    try { proc.kill('SIGKILL'); } catch {}
    await srv.close();
    try { fs.rmSync(userDataDir, { recursive: true, force: true }); } catch {}
  }
}

function checkAttributionAndDocs(fileStats) {
  const attrText = fs.readFileSync(ATTRIBUTION_PATH, 'utf8');
  const creditsText = fs.readFileSync(CREDITS_PATH, 'utf8');
  const docText = fs.readFileSync(DOC_PATH, 'utf8');
  const selfText = fs.readFileSync(fileURLToPath(import.meta.url), 'utf8');

  const bannedCodenameRe = new RegExp(
    `(${['CHRONOS', 'IMPOSSIBLE'].join('-')}|${['CHRONOS', '1000'].join('-')})`
  );
  for (const [label, txt] of [
    ['ATTRIBUTION.md', attrText],
    ['credits.html', creditsText],
    ['ACQUISITION_AND_FAILURE_MODES.md', docText],
    ['gate_i6_captures.mjs', selfText],
  ]) {
    assert(!txt.includes('\u2014'), `${label}: contains forbidden em dash`);
    assert(!bannedCodenameRe.test(txt), `${label}: contains forbidden internal codename`);
  }

  assert(creditsText.includes('Bernard de Go Mars') && creditsText.includes('Karen Roe'), 'credits.html missing legacy verified credits');

  for (const st of fileStats) {
    assert(attrText.includes(st.file), `ATTRIBUTION.md missing ${st.file}`);
    assert(creditsText.includes(st.file), `credits.html missing ${st.file}`);
    assert(docText.includes(st.file), `ACQUISITION_AND_FAILURE_MODES.md missing ${st.file}`);
    assert(docText.includes(st.sha256), `ACQUISITION_AND_FAILURE_MODES.md missing exact sha256 for ${st.file}`);
  }

  // Validate all https:// links appearing in ATTRIBUTION.md and credits.html parse cleanly as URLs
  for (const [label, txt] of [['ATTRIBUTION.md', attrText], ['credits.html', creditsText]]) {
    const urls = txt.match(/https:\/\/[^\s)"'<>]+/g) || [];
    assert(urls.length >= 20, `${label}: expected >= 20 attribution URLs, found ${urls.length}`);
    for (const raw of urls) {
      let parsed;
      try {
        parsed = new URL(raw);
      } catch {
        throw new Error(`${label}: malformed URL ${raw}`);
      }
      assert(parsed.protocol === 'https:' && parsed.hostname.length > 3, `${label}: invalid URL ${raw}`);
    }
  }
}

function lumaDiffRegionRms(rgbaA, rgbaB, w, h, xStart, xEnd, yStart, yEnd) {
  let sumSq = 0.0;
  let count = 0;
  for (let y = yStart; y < yEnd; y++) {
    const rowOff = y * w;
    for (let x = xStart; x < xEnd; x++) {
      const i = (rowOff + x) * 4;
      const la = 0.299 * rgbaA[i] + 0.587 * rgbaA[i + 1] + 0.114 * rgbaA[i + 2];
      const lb = 0.299 * rgbaB[i] + 0.587 * rgbaB[i + 1] + 0.114 * rgbaB[i + 2];
      const d = la - lb;
      sumSq += d * d;
      count++;
    }
  }
  return Math.sqrt(sumSq / Math.max(1, count));
}

function runWasmDiagnostics(wasmModule, frames, fps, opts = {}) {
  const {
    swarmMode = 1,
    metrologyMode = 2,
    bosMode = 1,
    maxFrames = frames.length,
    sampleBosAtFrame = 140,
  } = opts;
  const inst = new WebAssembly.Instance(wasmModule, {});
  const exp = inst.exports;
  const mem = exp.memory;
  exp.chronos_init();
  if (typeof exp.chronos_set_resolution === 'function') exp.chronos_set_resolution(1);
  const w = typeof exp.chronos_get_frame_w === 'function' ? exp.chronos_get_frame_w() : 640;
  const h = typeof exp.chronos_get_frame_h === 'function' ? exp.chronos_get_frame_h() : 360;
  assert(w === 640 && h === 360, `Expected 640x360 HD Wasm session, got ${w}x${h}`);
  const frameBytes = w * h * 4;
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);
  if (typeof exp.chronos_set_swarm_mode === 'function') exp.chronos_set_swarm_mode(swarmMode);
  if (typeof exp.chronos_set_micro_metrology_mode === 'function') exp.chronos_set_micro_metrology_mode(metrologyMode);
  if (typeof exp.chronos_set_bos_mode === 'function') exp.chronos_set_bos_mode(bosMode);
  if (bosMode !== 0 && typeof exp.chronos_set_bos_gauge_roi === 'function') {
    assert(exp.chronos_set_bos_gauge_roi(4, 4, 18, 60) === 1, 'chronos_set_bos_gauge_roi(4,4,18,60) failed');
  }

  const telem = new Float32Array(mem.buffer, exp.chronos_get_telemetry_ptr(), 1568);
  const swarmTelem = typeof exp.chronos_get_swarm_telemetry_ptr === 'function'
    ? new Float32Array(mem.buffer, exp.chronos_get_swarm_telemetry_ptr(), 64)
    : null;
  const hasIngestPtr = typeof exp.chronos_get_ingest_rgba_ptr === 'function';
  const ringViews = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), frameBytes));

  const hasBosApi = typeof exp.chronos_get_bos_telemetry_ptr === 'function'
    && typeof exp.chronos_get_bos_planes_ptr === 'function'
    && typeof exp.chronos_get_bos_valid_ptr === 'function';
  const bosPtr = hasBosApi ? exp.chronos_get_bos_telemetry_ptr() : 0;
  const bosTelemU32 = hasBosApi && bosPtr > 0 ? new Uint32Array(mem.buffer, bosPtr, 32) : null;
  const bosTelemF32 = hasBosApi && bosPtr > 0 ? new Float32Array(mem.buffer, bosPtr, 32) : null;
  const BOS_W = 256;
  const BOS_H = 128;
  const BOS_N = BOS_W * BOS_H; // 32,768
  const MAX_MICRO = 3072;
  const bosPlanes = hasBosApi ? new Float32Array(mem.buffer, exp.chronos_get_bos_planes_ptr(), 6 * BOS_N) : null;
  const bosValid = hasBosApi ? new Uint8Array(mem.buffer, exp.chronos_get_bos_valid_ptr(), BOS_N) : null;

  const hasHeroCandApi = typeof exp.chronos_get_hero_candidates_ptr === 'function';
  const heroCandU32 = hasHeroCandApi ? new Uint32Array(mem.buffer, exp.chronos_get_hero_candidates_ptr(), 4 + 16 * 8) : null;
  const microQualU32 = typeof exp.chronos_get_micro_quality_ptr === 'function'
    ? new Uint32Array(mem.buffer, exp.chronos_get_micro_quality_ptr(), MAX_MICRO)
    : null;

  const dtSec = 1.0 / fps;
  const limit = Math.min(frames.length, maxFrames);
  let nanCount = 0;
  const camFrames = [];
  let maxHeroCount = 0;
  let heroTrackFrames = 0;
  let prePanHeroFrames = 0;
  let maxMicroConfirmed = 0;
  let maxHeroCands = 0;
  let maxAutoHeroCount = 0;
  let promoCandFrames = 0;
  let bosWarmFrames = 0;
  let bosReadyFrames = 0;
  let bosDisturbedFrames = 0;
  let bosDisturbedShockFrames = 0;
  let maxBosRmsDisp = 0.0;
  let bosSample = null;

  for (let f = 0; f < limit; f++) {
    if (hasIngestPtr) {
      new Uint8Array(mem.buffer, exp.chronos_get_ingest_rgba_ptr(), frameBytes).set(frames[f]);
    } else {
      ringViews[f % 12].set(frames[f]);
    }
    exp.chronos_set_ingest_pts(f * dtSec);
    exp.chronos_pre_tick(dtSec);
    exp.chronos_set_fiedler_spectrum(0.0, 0.0);
    exp.chronos_post_tick(dtSec);

    for (let i = 0; i < 1568; i++) {
      if (!Number.isFinite(telem[i])) nanCount++;
    }
    const cam = Math.round(telem[1248 + 9]);
    if (cam > 0) camFrames.push(f);

    let activeHeroes = 0;
    for (let m = 0; m < 16; m++) {
      if (telem[32 + m * 56 + 0] > 0.5 && telem[32 + m * 56 + 1] > 0) activeHeroes++;
    }
    if (activeHeroes > maxHeroCount) maxHeroCount = activeHeroes;
    heroTrackFrames += activeHeroes;
    if (f < 135) prePanHeroFrames += activeHeroes;

    if (swarmTelem && swarmTelem[4] > maxMicroConfirmed) maxMicroConfirmed = swarmTelem[4];
    if (heroCandU32) {
      if (heroCandU32[0] > maxHeroCands) maxHeroCands = heroCandU32[0];
      if (heroCandU32[2] > maxAutoHeroCount) maxAutoHeroCount = heroCandU32[2];
    }
    if (microQualU32) {
      for (let i = 0; i < MAX_MICRO; i++) {
        if ((microQualU32[i] & (1 << 18)) !== 0) {
          promoCandFrames++;
          break;
        }
      }
    }

    if (bosTelemU32 && bosTelemF32) {
      const bosState = bosTelemU32[2]; // 0=off, 1=warm, 2=ready, 3=disturbed
      if (bosState === 1) bosWarmFrames++;
      if (bosState === 2) bosReadyFrames++;
      if (bosState === 3) {
        bosDisturbedFrames++;
        if (f >= 65 && f <= 160) bosDisturbedShockFrames++;
      }
      const rmsDisp = bosTelemF32[23];
      if (Number.isFinite(rmsDisp) && rmsDisp > maxBosRmsDisp) maxBosRmsDisp = rmsDisp;

      if (f === sampleBosAtFrame && bosPlanes && bosValid) {
        let validDispCnt = 0;
        let centerMaxDisp = 0.0;
        let centerPhiMax = 0.0;
        let quietSq = 0.0;
        let quietCnt = 0;
        for (let gy = 0; gy < BOS_H; gy++) {
          for (let gx = 0; gx < BOS_W; gx++) {
            const idx = gy * BOS_W + gx;
            if ((bosValid[idx] & 1) === 0) continue;
            validDispCnt++;
            const ex = bosPlanes[idx];
            const ey = bosPlanes[BOS_N + idx];
            const phi = bosPlanes[2 * BOS_N + idx];
            assert(Number.isFinite(ex) && Number.isFinite(ey), `Non-finite BOS displacement at (${gx},${gy})`);
            const mag = Math.hypot(ex, ey);
            if (gx >= 88 && gx <= 168 && gy >= 20 && gy <= 112) {
              if (mag > centerMaxDisp) centerMaxDisp = mag;
              if ((bosValid[idx] & 2) !== 0 && Math.abs(phi) > centerPhiMax) {
                centerPhiMax = Math.abs(phi);
              }
            } else if (gx < 36 || gx >= 220) {
              quietSq += mag * mag;
              quietCnt++;
            }
          }
        }
        const quietRms = Math.sqrt(quietSq / Math.max(1, quietCnt));
        bosSample = {
          frame: f,
          bosState,
          reasonMask: bosTelemU32[3],
          validDispCnt,
          centerMaxDisp: Number(centerMaxDisp.toFixed(4)),
          quietRms: Number(quietRms.toFixed(4)),
          dispSnr: Number((centerMaxDisp / Math.max(1e-6, quietRms)).toFixed(2)),
          centerPhiMax: Number(centerPhiMax.toFixed(3)),
          curlRms: Number(bosTelemF32[17].toFixed(5)),
        };
      }
    }
  }

  assert(nanCount === 0, `Wasm telemetry contained ${nanCount} non-finite values`);
  return {
    frames: limit,
    camMotionFrames: camFrames.length,
    firstCamFrame: camFrames[0] ?? null,
    lastCamFrame: camFrames[camFrames.length - 1] ?? null,
    camFramesInEarlySteady: camFrames.filter((idx) => idx < 60).length,
    camFramesInShockSway: camFrames.filter((idx) => idx >= 65 && idx <= 160).length,
    camFramesInPan: camFrames.filter((idx) => idx >= 150).length,
    camFramesBeforePan: camFrames.filter((idx) => idx < 135).length,
    maxHeroCount,
    heroTrackFrames,
    prePanHeroFrames,
    maxMicroConfirmed,
    hasHeroCandApi,
    maxHeroCands,
    maxAutoHeroCount,
    promoCandFrames,
    hasBosApi,
    bosWarmFrames,
    bosReadyFrames,
    bosDisturbedFrames,
    bosDisturbedShockFrames,
    maxBosRmsDisp,
    bosSample,
  };
}

async function main() {
  const fileStats = [];
  let totalBytes = 0;

  for (const fx of FIXTURES) {
    const mp4Path = path.join(SAMPLES_DIR, fx.file);
    assert(fs.existsSync(mp4Path), `Missing sample fixture ${mp4Path}`);
    const stat = fs.statSync(mp4Path);
    assert(stat.size > 50_000 && stat.size <= fx.maxBytes, `${fx.file}: size ${stat.size} outside (50KB, ${fx.maxBytes}]`);
    totalBytes += stat.size;
    const sha256 = sha256File(mp4Path);
    const probe = probeClip(mp4Path);
    assert(probe.nbFrames === fx.expectedFrames, `${fx.file}: expected ${fx.expectedFrames} frames, got ${probe.nbFrames}`);
    fileStats.push({ ...fx, mp4Path, bytes: stat.size, sha256, probe });
  }

  assert(
    totalBytes >= 1.5 * 1024 * 1024 && totalBytes <= 4.5 * 1024 * 1024,
    `Combined size ${totalBytes} B outside Wave-1 target [1.5 MiB, 4.5 MiB]`
  );

  checkAttributionAndDocs(fileStats);

  let chromeResults = null;
  if (!skipChrome) {
    chromeResults = await verifyChromiumVideoDecode(FIXTURES);
  }

  assert(fs.existsSync(WASM_PATH), `Missing compiled Wasm at ${WASM_PATH}`);
  const wasmModule = new WebAssembly.Module(fs.readFileSync(WASM_PATH));

  const decodedByFile = {};
  const wasmByFile = {};
  for (const st of fileStats) {
    const rgbaFrames = await decodeRgbaFrames(st.mp4Path, 640, 360);
    assert(rgbaFrames.length === st.expectedFrames, `${st.file}: decoded RGBA frame count ${rgbaFrames.length} !== ${st.expectedFrames}`);
    decodedByFile[st.file] = rgbaFrames;
    const isBosClip = st.role === 'positive_bos' || st.role === 'handheld_failure';
    wasmByFile[st.file] = runWasmDiagnostics(wasmModule, rgbaFrames, st.expectedFps, {
      swarmMode: 1,
      metrologyMode: 2,
      bosMode: isBosClip ? 1 : 0,
    });
  }

  // 1. Positive BOS check (bos_thermal_plume.mp4):
  //    - Zero false camera motion across all 225 frames
  //    - Clear-air raw appearance (luma ratio >= 1.5x, raw core RMS < 15 DN) + 0 false discrete Hero targets
  //    - Primary SNR check directly via Wasm chronos_get_bos_planes_ptr() (eps_x, eps_y, phi)
  const plumeFrames = decodedByFile['bos_thermal_plume.mp4'];
  const plumeWasm = wasmByFile['bos_thermal_plume.mp4'];
  assert(plumeWasm.camMotionFrames === 0, `bos_thermal_plume.mp4 triggered ${plumeWasm.camMotionFrames} false camera-motion frames`);
  assert(plumeWasm.maxHeroCount === 0, `bos_thermal_plume.mp4 triggered ${plumeWasm.maxHeroCount} false Hero targets on refractive shimmer`);
  const plumeCenterRms = lumaDiffRegionRms(plumeFrames[5], plumeFrames[140], 640, 360, 260, 380, 60, 310);
  const plumeQuietLeft = lumaDiffRegionRms(plumeFrames[5], plumeFrames[140], 640, 360, 10, 70, 20, 340);
  const plumeQuietRight = lumaDiffRegionRms(plumeFrames[5], plumeFrames[140], 640, 360, 570, 630, 20, 340);
  const plumeQuietRms = 0.5 * (plumeQuietLeft + plumeQuietRight);
  const plumeContrastSnr = plumeCenterRms / Math.max(1e-6, plumeQuietRms);
  assert(plumeContrastSnr >= 1.5, `bos_thermal_plume.mp4 core/margin luma ratio ${plumeContrastSnr.toFixed(2)} < 1.5`);
  assert(plumeCenterRms < 15.0, `bos_thermal_plume.mp4 raw core luma RMS ${plumeCenterRms.toFixed(2)} too high (should look clear)`);
  assert(plumeWasm.hasBosApi && plumeWasm.bosSample, 'Missing Wave-1 BOS planes/telemetry exports');
  assert(plumeWasm.bosWarmFrames === 16, `bos_thermal_plume.mp4 expected 16 warmup frames, got ${plumeWasm.bosWarmFrames}`);
  assert(plumeWasm.bosReadyFrames >= 190, `bos_thermal_plume.mp4 Wave-1 BOS ready frames ${plumeWasm.bosReadyFrames} < 190`);
  assert(plumeWasm.bosDisturbedFrames === 0, `bos_thermal_plume.mp4 Wave-1 BOS reported ${plumeWasm.bosDisturbedFrames} disturbed frames`);
  const bs = plumeWasm.bosSample;
  assert(bs.bosState === 2 && bs.reasonMask === 0, `bos_thermal_plume.mp4 f=140 BOS state=${bs.bosState} reason=${bs.reasonMask}`);
  assert(bs.validDispCnt >= 30000, `bos_thermal_plume.mp4 BOS validDispCnt=${bs.validDispCnt} < 30000`);
  assert(bs.centerMaxDisp >= 0.45, `bos_thermal_plume.mp4 Wasm BOS center max |eps|=${bs.centerMaxDisp} px < 0.45 px`);
  assert(bs.quietRms <= 0.12, `bos_thermal_plume.mp4 Wasm BOS quiet margin RMS=${bs.quietRms} px > 0.12 px`);
  assert(bs.dispSnr >= 8.0, `bos_thermal_plume.mp4 Wasm BOS displacement SNR=${bs.dispSnr} < 8.0`);
  assert(bs.centerPhiMax >= 1.5, `bos_thermal_plume.mp4 Wasm BOS center max |phi|=${bs.centerPhiMax} < 1.5`);
  assert(bs.curlRms <= 0.05, `bos_thermal_plume.mp4 Wasm BOS curl RMS=${bs.curlRms} > 0.05`);

  // 2. Controlled handheld failure check (bos_handheld_failure.mp4):
  //    - Zero camera motion during steady tripod baseline (f < 60)
  //    - Reproducible camera motion / BOS disturbed violation during shock+sway (f in 65..160)
  //    - Displaced rest pose (f=180 vs f=5) breaks outer margin stationarity vs steady clip
  const handFrames = decodedByFile['bos_handheld_failure.mp4'];
  const handWasm = wasmByFile['bos_handheld_failure.mp4'];
  assert(handWasm.camFramesInEarlySteady === 0, `bos_handheld_failure.mp4 tripped during steady baseline (${handWasm.camFramesInEarlySteady})`);
  assert(handWasm.camFramesInShockSway >= 40, `bos_handheld_failure.mp4 failed to trigger reproducible shock/sway motion (${handWasm.camFramesInShockSway} < 40)`);
  assert(handWasm.hasBosApi && handWasm.bosReadyFrames >= 40, `bos_handheld_failure.mp4 expected >= 40 steady BOS ready frames, got ${handWasm.bosReadyFrames}`);
  assert(handWasm.bosDisturbedShockFrames >= 60, `bos_handheld_failure.mp4 Wave-1 BOS disturbed frames during shock/sway ${handWasm.bosDisturbedShockFrames} < 60`);
  const handStaleMarginRms = lumaDiffRegionRms(handFrames[5], handFrames[180], 640, 360, 10, 70, 20, 340);
  const staleRatio = handStaleMarginRms / Math.max(1e-6, plumeQuietRms);
  assert(staleRatio >= 8.0, `bos_handheld_failure.mp4 displaced rest margin ratio ${staleRatio.toFixed(2)} < 8.0`);

  // 3. Perspective flyby decoy check (perspective_flyby_decoy.mp4):
  //    - Swarm AUTO (mode=1) + Metrology (mode=2) across full 225-frame clip:
  //      non-vacuous tracking (>= 4 peak Heroes, >= 300 tracked Hero frames), 0 false pre-pan flags,
  //      >= 30 pan-locked frames in Phase 2, and 0 false I1 auto-promoted Hero alarms.
  //    - Swarm Forced Active (mode=2) + Metrology (mode=2) over stationary pinhole looming phase (0..134):
  //      non-vacuous MicroTrack tracking (>= 20 confirmed micro-tracks) with 0 false I1 alarms.
  const flybyWasm = wasmByFile['perspective_flyby_decoy.mp4'];
  assert(flybyWasm.camFramesBeforePan === 0, `perspective_flyby_decoy.mp4 falsely flagged pan during stationary looming (${flybyWasm.camFramesBeforePan})`);
  assert(flybyWasm.camFramesInPan >= 30, `perspective_flyby_decoy.mp4 failed to lock Phase-2 camera pan (${flybyWasm.camFramesInPan} < 30)`);
  assert(flybyWasm.maxHeroCount >= 4 && flybyWasm.heroTrackFrames >= 300, `perspective_flyby_decoy.mp4 vacuous AUTO Hero tracking (max=${flybyWasm.maxHeroCount}, frames=${flybyWasm.heroTrackFrames})`);
  assert(flybyWasm.hasHeroCandApi, 'Missing Wave-1 hero_candidates export');
  assert(flybyWasm.maxAutoHeroCount === 0 && flybyWasm.maxHeroCands === 0 && flybyWasm.promoCandFrames === 0, `perspective_flyby_decoy.mp4 AUTO triggered false I1 promotions (auto=${flybyWasm.maxAutoHeroCount}, cands=${flybyWasm.maxHeroCands}, promoFrames=${flybyWasm.promoCandFrames})`);

  const flybyLoomMicro = runWasmDiagnostics(wasmModule, decodedByFile['perspective_flyby_decoy.mp4'], 25, {
    swarmMode: 2,
    metrologyMode: 2,
    bosMode: 0,
    maxFrames: 135,
  });
  assert(flybyLoomMicro.maxMicroConfirmed >= 20, `perspective_flyby_decoy.mp4 vacuous MicroTrack looming tracking (maxMicroConfirmed=${flybyLoomMicro.maxMicroConfirmed} < 20)`);
  assert(flybyLoomMicro.maxAutoHeroCount === 0 && flybyLoomMicro.maxHeroCands === 0 && flybyLoomMicro.promoCandFrames === 0, `perspective_flyby_decoy.mp4 MicroTrack looming triggered false I1 promotions (auto=${flybyLoomMicro.maxAutoHeroCount}, cands=${flybyLoomMicro.maxHeroCands}, promoFrames=${flybyLoomMicro.promoCandFrames})`);

  const summary = {
    status: 'PASS',
    totalBytes,
    totalMiB: Number((totalBytes / (1024 * 1024)).toFixed(3)),
    chromeVerified: !skipChrome,
    fixtures: fileStats.map((s) => ({
      file: s.file,
      bytes: s.bytes,
      sha256: s.sha256,
      duration: s.probe.duration,
      frames: s.probe.nbFrames,
      resolution: `${s.probe.width}x${s.probe.height}`,
      wasm: wasmByFile[s.file],
    })),
    metrics: {
      plumeCenterRms: Number(plumeCenterRms.toFixed(3)),
      plumeQuietRms: Number(plumeQuietRms.toFixed(3)),
      plumeContrastSnr: Number(plumeContrastSnr.toFixed(2)),
      wasmBosSample: bs,
      handStaleMarginRms: Number(handStaleMarginRms.toFixed(3)),
      staleMarginRatio: Number(staleRatio.toFixed(2)),
      flybyAutoMaxHero: flybyWasm.maxHeroCount,
      flybyAutoHeroTrackFrames: flybyWasm.heroTrackFrames,
      flybyLoomMicroMaxConf: flybyLoomMicro.maxMicroConfirmed,
    },
    chromeResults,
  };

  if (jsonFlag) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    console.log('========================================================================');
    console.log(' CHRONOS-SPECTRAL // Gate I6 Semi-Synthetic Captures & Failure Matrix');
    console.log('========================================================================');
    for (const fx of summary.fixtures) {
      console.log(`  [PASS] ${fx.file} (${fx.resolution}, ${fx.frames}f, ${fx.duration.toFixed(2)}s, ${(fx.bytes / 1024).toFixed(1)} KiB, sha256=${fx.sha256.slice(0, 12)}...)`);
    }
    console.log(`  [PASS] Combined Fixture Size: ${summary.totalMiB.toFixed(2)} MiB (${summary.totalBytes} B) within [1.5, 4.5] MiB`);
    console.log(`  [PASS] BOS Plume (Wasm Planes f=140): max|eps|=${bs.centerMaxDisp} px vs quiet=${bs.quietRms} px (${bs.dispSnr}x SNR) | max|phi|=${bs.centerPhiMax} | curl=${bs.curlRms} | ready=${plumeWasm.bosReadyFrames}f | camMot=${plumeWasm.camMotionFrames} | hero=${plumeWasm.maxHeroCount}`);
    console.log(`  [PASS] BOS Handheld Failure: steadyCam=${handWasm.camFramesInEarlySteady}, shockSwayCam=${handWasm.camFramesInShockSway}, bosDisturbedShock=${handWasm.bosDisturbedShockFrames}f, displacedRestRatio=${summary.metrics.staleMarginRatio}x`);
    console.log(`  [PASS] Perspective Flyby Decoy: AUTO maxHero=${flybyWasm.maxHeroCount} (${flybyWasm.heroTrackFrames} track-frames), loomMicro=${flybyLoomMicro.maxMicroConfirmed}, prePanCam=${flybyWasm.camFramesBeforePan}, panLockCam=${flybyWasm.camFramesInPan}, falseAutoHero=0`);
    if (chromeResults) {
      console.log(`  [PASS] Headless Chromium <video> decode & seek verified (${chromeResults.length}/3 clips at 640x360)`);
    } else {
      console.log('  [INFO] Headless Chromium step skipped via --no-chrome');
    }
    console.log('[Gate I6] ALL CHECKS PASSED.');
  }
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
