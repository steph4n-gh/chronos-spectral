#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL v3.5.0 // Deterministic Headless Chrome CDP Documentation Media Capture
// Generates:
//   - docs/assets/demos/demo_starling_swarm.gif    (Starling murmuration, Hero 16 HIDDEN)
//   - docs/assets/demos/demo_model_rockets.gif     (Model rockets launch, separation & chutes)
//   - docs/assets/demos/demo_bos_thermal_plume.gif (256x128 BOS thermal plume refraction)
//   - docs/assets/demos/demo_hodge_vortex.gif      (64x36 Hodge coexact curl* psi on vortex swarm)
//   - docs/assets/panels/01..17.png                (2x Retina screenshots of the v3.5.0 Instrument)
// ============================================================================

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { startChronosServer } from './dev_server.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const WEB_DIR = path.join(ROOT_DIR, 'web');
const DEMOS_DIR = path.join(ROOT_DIR, 'docs', 'assets', 'demos');
const PANELS_DIR = path.join(ROOT_DIR, 'docs', 'assets', 'panels');

const VIEWPORT_W = 1440;
const VIEWPORT_H = 900;
const DPR = 2;
const PANELS_ONLY = process.argv.includes('--panels-only');

function findChromeBinary() {
  if (process.env.CHROME_BIN && fs.existsSync(process.env.CHROME_BIN)) {
    return process.env.CHROME_BIN;
  }
  const candidates = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium-browser',
    '/usr/bin/chromium',
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  throw new Error('Could not locate Google Chrome / Chromium binary. Set CHROME_BIN.');
}

function findFfmpegBinary() {
  const candidates = ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/usr/bin/ffmpeg', 'ffmpeg'];
  for (const c of candidates) {
    if (c === 'ffmpeg' || fs.existsSync(c)) return c;
  }
  throw new Error('Could not locate ffmpeg binary.');
}

function launchHeadlessChrome(chromeBin, userDataDir) {
  const args = [
    '--headless=new',
    '--remote-debugging-port=0',
    `--user-data-dir=${userDataDir}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--autoplay-policy=no-user-gesture-required',
    '--mute-audio',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
    `--window-size=${VIEWPORT_W},${VIEWPORT_H}`,
  ];
  if (process.platform === 'linux' || process.env.CI) {
    args.push('--no-sandbox', '--disable-dev-shm-usage');
  }
  const proc = spawn(chromeBin, args, { stdio: ['ignore', 'ignore', 'pipe'] });
  return new Promise((resolve, reject) => {
    let stderrBuf = '';
    const timer = setTimeout(() => {
      proc.kill('SIGKILL');
      reject(new Error(`Timed out waiting for Chrome DevTools endpoint. stderr:\n${stderrBuf}`));
    }, 15000);
    proc.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
    proc.stderr.on('data', (chunk) => {
      stderrBuf += chunk.toString('utf8');
      const m = stderrBuf.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (m) {
        clearTimeout(timer);
        resolve({ proc, wsUrl: m[1] });
      }
    });
  });
}

class CdpClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
  }

  async connect() {
    await new Promise((resolve, reject) => {
      this.ws.addEventListener('open', () => resolve(), { once: true });
      this.ws.addEventListener('error', (e) => reject(e.error || new Error('CDP WebSocket error')), { once: true });
    });
    this.ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString('utf8'));
      if (msg.id) {
        const entry = this.pending.get(msg.id);
        if (entry) {
          this.pending.delete(msg.id);
          if (msg.error) entry.reject(new Error(`CDP ${entry.method}: ${msg.error.message}`));
          else entry.resolve(msg.result);
        }
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
    const res = await this.send(
      'Runtime.evaluate',
      { expression, awaitPromise: true, returnByValue: true },
      sessionId
    );
    if (res.exceptionDetails) {
      const desc =
        res.exceptionDetails.exception?.description ||
        res.exceptionDetails.text ||
        JSON.stringify(res.exceptionDetails);
      throw new Error(`Page JS exception: ${desc}`);
    }
    return res.result ? res.result.value : undefined;
  }

  close() {
    try {
      this.ws.close();
    } catch (_e) {
      // ignore
    }
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForCondition(cdp, sessionId, description, jsExpr, timeoutMs = 22000) {
  const deadline = Date.now() + timeoutMs;
  let lastVal = null;
  while (Date.now() < deadline) {
    lastVal = await cdp.evalJs(sessionId, jsExpr);
    if (lastVal && lastVal.ok) return lastVal;
    await sleep(80);
  }
  throw new Error(`Timeout waiting for [${description}]. Last state: ${JSON.stringify(lastVal)}`);
}

async function setViewportSize(cdp, sessionId, width = VIEWPORT_W, height = VIEWPORT_H, dpr = DPR) {
  await cdp.send(
    'Emulation.setDeviceMetricsOverride',
    { width, height, deviceScaleFactor: dpr, mobile: false },
    sessionId
  );
}

async function captureBoundingSelectors(cdp, sessionId, selectors, outPath, options = {}) {
  const { pad = 6, dpr = DPR, waitMs = 220 } = options;
  const list = Array.isArray(selectors) ? selectors : [selectors];

  const clip = await cdp.evalJs(
    sessionId,
    `(() => {
      const sels = ${JSON.stringify(list)};
      const els = sels.map((s) => document.querySelector(s)).filter(Boolean);
      if (!els.length) return null;
      return new Promise((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => {
          setTimeout(() => {
            let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
            for (const el of els) {
              const r = el.getBoundingClientRect();
              x0 = Math.min(x0, r.left);
              y0 = Math.min(y0, r.top);
              x1 = Math.max(x1, r.right);
              y1 = Math.max(y1, r.bottom);
            }
            const x = Math.max(0, Math.floor(x0 - ${pad}));
            const y = Math.max(0, Math.floor(y0 - ${pad}));
            const w = Math.min(window.innerWidth - x, Math.ceil(x1 - x0 + ${pad * 2}));
            const h = Math.min(window.innerHeight - y, Math.ceil(y1 - y0 + ${pad * 2}));
            resolve({ x, y, width: w, height: h, scale: 1 });
          }, ${waitMs});
        }));
      });
    })()`
  );
  if (!clip || clip.width <= 0 || clip.height <= 0) {
    throw new Error(`Missing or empty selector group: ${list.join(', ')}`);
  }

  const shot = await cdp.send(
    'Page.captureScreenshot',
    { format: 'png', captureBeyondViewport: false, clip },
    sessionId
  );
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.from(shot.data, 'base64'));
  const bytes = fs.statSync(outPath).size;
  console.log(`  [PNG] ${path.relative(ROOT_DIR, outPath)} (${clip.width}x${clip.height} @${dpr}x, ${(bytes / 1024).toFixed(1)} KiB)`);
}

async function encodeGifFromFrames(ffmpegBin, frameDir, outGifPath, fps, scaleWidth = 640) {
  fs.mkdirSync(path.dirname(outGifPath), { recursive: true });
  const pattern = path.join(frameDir, 'frame_%04d.png');
  const filter = `fps=${fps},scale=${scaleWidth}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=2:diff_mode=rectangle`;
  execFileSync(
    ffmpegBin,
    ['-y', '-framerate', String(fps), '-i', pattern, '-vf', filter, outGifPath],
    { stdio: 'pipe' }
  );
  const bytes = fs.statSync(outGifPath).size;
  console.log(`  [GIF] ${path.relative(ROOT_DIR, outGifPath)} (${(bytes / 1048576).toFixed(2)} MiB, ${fps} fps, ${scaleWidth}px)`);
  if (bytes > 8 * 1048576) {
    console.warn(`  [WARN] ${path.basename(outGifPath)} exceeded 8 MiB (${(bytes / 1048576).toFixed(2)} MiB).`);
  }
}

async function setHeroToggle(cdp, sessionId, visible) {
  await cdp.evalJs(
    sessionId,
    `(() => {
      const cur = window.__chronosUi?.getState()?.overlays?.hero;
      if (Boolean(cur) !== ${Boolean(visible)}) {
        document.getElementById('btnHudHeroToggle')?.click();
      }
      return true;
    })()`
  );
  await waitForCondition(
    cdp,
    sessionId,
    `Hero toggle -> ${visible}`,
    `(() => ({
      ok: Boolean(window.__chronosUi?.getState()?.overlays?.hero) === ${Boolean(visible)},
      cur: window.__chronosUi?.getState()?.overlays?.hero
    }))()`
  );
}

async function setWipePct(cdp, sessionId, pct) {
  await cdp.evalJs(
    sessionId,
    `(() => {
      const sl = document.getElementById('showcaseWipeSlider');
      if (sl) {
        sl.value = String(${pct});
        sl.dispatchEvent(new Event('input', { bubbles: true }));
      }
      return true;
    })()`
  );
  await sleep(120);
}

async function pauseVideo(cdp, sessionId) {
  await cdp.evalJs(
    sessionId,
    `(() => {
      const v = document.getElementById('hiddenVideoSource');
      if (v && !v.paused) document.getElementById('btnVidPlayPause')?.click();
      return true;
    })()`
  );
  await waitForCondition(
    cdp,
    sessionId,
    'video paused',
    `(() => ({ ok: document.getElementById('hiddenVideoSource')?.paused === true }))()`,
    8000
  );
}

async function stepForwardNFrames(cdp, sessionId, count) {
  for (let i = 0; i < count; i++) {
    await waitForCondition(
      cdp,
      sessionId,
      `Video ready before +1f #${i + 1}`,
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        return { ok: Boolean(v && v.readyState >= 2 && !v.seeking), readyState: v?.readyState, seeking: v?.seeking };
      })()`,
      8000
    );
    const beforeTime = await cdp.evalJs(sessionId, `document.getElementById('hiddenVideoSource')?.currentTime ?? -1`);
    await cdp.evalJs(sessionId, `document.getElementById('btnVidStepFwd').click(); true;`);
    await waitForCondition(
      cdp,
      sessionId,
      `Video step +1f #${i + 1}`,
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        return {
          ok: Boolean(v && v.paused && !v.seeking && v.readyState >= 2 && Math.abs(v.currentTime - ${beforeTime}) > 0.008),
          t: v?.currentTime,
          readyState: v?.readyState,
          seeking: v?.seeking
        };
      })()`,
      8000
    );
  }
  await sleep(65);
}

async function getHudStageClip(cdp, sessionId) {
  return cdp.evalJs(
    sessionId,
    `(() => {
      const stage = document.getElementById('hudCanvasStage');
      return new Promise((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const r = stage.getBoundingClientRect();
          resolve({
            x: Math.max(0, Math.floor(r.left)),
            y: Math.max(0, Math.floor(r.top)),
            width: Math.ceil(r.width),
            height: Math.ceil(r.height),
            scale: 1,
          });
        }));
      });
    })()`
  );
}

async function captureHudSequenceGif(cdp, sessionId, ffmpegBin, cfg) {
  const { frameDir, outGifPath, recipeKey, startTimeSec, totalGifFrames, strideFrames, gifFps, scaleWidth, heroVisible, wipePct = 0, beforePauseCheck, onMidFrameCapture } = cfg;
  fs.mkdirSync(frameDir, { recursive: true });

  await cdp.evalJs(sessionId, `window.__chronosUi.loadRecipe(${JSON.stringify(recipeKey)}); true;`);
  await setHeroToggle(cdp, sessionId, heroVisible);
  await setWipePct(cdp, sessionId, wipePct);

  if (beforePauseCheck) {
    await waitForCondition(cdp, sessionId, `${recipeKey} pre-capture condition`, beforePauseCheck, 24000);
  }

  await pauseVideo(cdp, sessionId);
  if (typeof startTimeSec === 'number' && startTimeSec <= 0.05) {
    await cdp.evalJs(sessionId, `document.getElementById('btnVidReplay').click(); true;`);
    await sleep(180);
    await pauseVideo(cdp, sessionId);
  }

  const clip = await getHudStageClip(cdp, sessionId);
  for (let i = 0; i < totalGifFrames; i++) {
    await stepForwardNFrames(cdp, sessionId, strideFrames);
    await cdp.evalJs(sessionId, `new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))`);
    const shot = await cdp.send(
      'Page.captureScreenshot',
      { format: 'png', captureBeyondViewport: false, clip },
      sessionId
    );
    fs.writeFileSync(path.join(frameDir, `frame_${String(i).padStart(4, '0')}.png`), Buffer.from(shot.data, 'base64'));
    if (onMidFrameCapture) {
      await onMidFrameCapture(i, totalGifFrames);
    }
  }

  await encodeGifFromFrames(ffmpegBin, frameDir, outGifPath, gifFps, scaleWidth);
}

async function main() {
  console.log('========================================================================');
  console.log(' CHRONOS-SPECTRAL v3.5.0 // Documentation Media Capture');
  console.log('========================================================================');

  const chromeBin = findChromeBinary();
  const ffmpegBin = findFfmpegBinary();
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chronos-docs-media-'));
  const chromeUserDir = path.join(tmpDir, 'chrome');
  fs.mkdirSync(chromeUserDir, { recursive: true });

  const srv = await startChronosServer({ rootDir: WEB_DIR, port: 0, host: '127.0.0.1' });
  const appUrl = `http://127.0.0.1:${srv.port}/index.html`;
  let chromeProc = null;
  let cdp = null;

  try {
    const launched = await launchHeadlessChrome(chromeBin, chromeUserDir);
    chromeProc = launched.proc;
    cdp = new CdpClient(launched.wsUrl);
    await cdp.connect();

    const { targetId } = await cdp.send('Target.createTarget', { url: appUrl });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Runtime.enable', {}, sessionId);
    await setViewportSize(cdp, sessionId, VIEWPORT_W, VIEWPORT_H, DPR);

    await waitForCondition(
      cdp,
      sessionId,
      'Wasm boot ready',
      `(() => ({ ok: Boolean(window.__chronosUi) && (document.getElementById('pillKernel')?.textContent || '').includes('0 HEAP ALLOC') }))()`
    );

    if (!PANELS_ONLY) {
      console.log('\n[Phase 1] Capturing Starling Murmuration GIF (Hero 16 HIDDEN)...');
      await captureHudSequenceGif(cdp, sessionId, ffmpegBin, {
        frameDir: path.join(tmpDir, 'gif_starling'),
        outGifPath: path.join(DEMOS_DIR, 'demo_starling_swarm.gif'),
        recipeKey: 'starling_1000',
        totalGifFrames: 75,
        strideFrames: 1,
        gifFps: 18,
        scaleWidth: 640,
        heroVisible: false,
        wipePct: 0,
        beforePauseCheck: `(() => {
          const v = document.getElementById('hiddenVideoSource');
          const conf = parseInt(((document.getElementById('kpiSwarmConfirmed')?.textContent || '0').split('/')[0] || '0').replace(/,/g, ''), 10);
          return { ok: Boolean(v && v.currentTime >= 2.0 && conf >= 1050), t: v?.currentTime || 0, conf };
        })()`,
      });

      console.log('\n[Phase 2] Capturing Model Rockets Full-Flight GIF (Hero 16 ON)...');
      await captureHudSequenceGif(cdp, sessionId, ffmpegBin, {
        frameDir: path.join(tmpDir, 'gif_rockets'),
        outGifPath: path.join(DEMOS_DIR, 'demo_model_rockets.gif'),
        recipeKey: 'model_rockets',
        startTimeSec: 0.0,
        totalGifFrames: 92,
        strideFrames: 4,
        gifFps: 14,
        scaleWidth: 640,
        heroVisible: true,
        wipePct: 0,
        beforePauseCheck: `(() => {
          const v = document.getElementById('hiddenVideoSource');
          return { ok: Boolean(v && v.readyState >= 2) };
        })()`,
      });

      console.log('\n[Phase 2b] Capturing 256x128 BOS Thermal Plume GIF...');
      await captureHudSequenceGif(cdp, sessionId, ffmpegBin, {
        frameDir: path.join(tmpDir, 'gif_bos_plume'),
        outGifPath: path.join(DEMOS_DIR, 'demo_bos_thermal_plume.gif'),
        recipeKey: 'bos_plume',
        totalGifFrames: 64,
        strideFrames: 2,
        gifFps: 16,
        scaleWidth: 640,
        heroVisible: false,
        wipePct: 0,
        beforePauseCheck: `(() => {
          const v = document.getElementById('hiddenVideoSource');
          const bos = window.__chronosWave1Probe?.getLastWave1()?.bos;
          return { ok: Boolean(v && v.currentTime >= 1.1 && bos && bos.state === 2), t: v?.currentTime, state: bos?.state };
        })()`,
      });

      console.log('\n[Phase 2c] Capturing 64x36 Mesh Hodge Coexact Vortex GIF...');
      await cdp.evalJs(sessionId, `window.__chronosUi.loadRecipe('vortex_1200'); true;`);
      await setHeroToggle(cdp, sessionId, false);
      await setWipePct(cdp, sessionId, 0);
      await cdp.evalJs(
        sessionId,
        `(() => {
          window.__chronosUi.applyIntent({ kind: 'SET_HODGE', mode: 2 });
          window.__chronosUi.applyIntent({ kind: 'SET_FIELD', field: 9 });
          return true;
        })()`
      );
      await waitForCondition(
        cdp,
        sessionId,
        '64x36 Mesh Hodge converged on vortex_1200',
        `(() => {
          const h = window.__chronosWave2Probe?.getLastWave2()?.hodge;
          return { ok: Boolean(h && h.state === 2 && h.converged === true && h.coexactFrac > 0.85), state: h?.state, co: h?.coexactFrac };
        })()`,
        22000
      );
      const hodgeClip = await getHudStageClip(cdp, sessionId);
      const hodgeFrameDir = path.join(tmpDir, 'gif_hodge_vortex');
      fs.mkdirSync(hodgeFrameDir, { recursive: true });
      for (let i = 0; i < 56; i++) {
        await sleep(62);
        await cdp.evalJs(sessionId, `new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))`);
        const shot = await cdp.send(
          'Page.captureScreenshot',
          { format: 'png', captureBeyondViewport: false, clip: hodgeClip },
          sessionId
        );
        fs.writeFileSync(path.join(hodgeFrameDir, `frame_${String(i).padStart(4, '0')}.png`), Buffer.from(shot.data, 'base64'));
      }
      await encodeGifFromFrames(ffmpegBin, hodgeFrameDir, path.join(DEMOS_DIR, 'demo_hodge_vortex.gif'), 16, 640);
    }

    console.log('\n[Phase 3] Capturing v3.5.0 Optical Physics Instrument Panels (01..17)...');

    // 03: Starling Murmuration HUD (Hero 16 OFF, 64x36 Velocity field)
    await cdp.evalJs(sessionId, `window.__chronosUi.loadRecipe('starling_1000'); true;`);
    await setHeroToggle(cdp, sessionId, false);
    await setWipePct(cdp, sessionId, 0);
    await waitForCondition(
      cdp,
      sessionId,
      'Starling swarm >= 1,000 confirmed',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const conf = parseInt(((document.getElementById('kpiSwarmConfirmed')?.textContent || '0').split('/')[0] || '0').replace(/,/g, ''), 10);
        return { ok: Boolean(v && v.currentTime >= 2.2 && conf >= 1000), t: v?.currentTime, conf };
      })()`,
      24000
    );
    await pauseVideo(cdp, sessionId);
    await captureBoundingSelectors(cdp, sessionId, '#hudPanel', path.join(PANELS_DIR, '03_hud_starling_swarm.png'));

    // 01, 02, 04, 05, 13, 14, 10: Model Rockets full instrument overview + rail + HUD + Track bay + Rack tabs
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__chronosUi.loadRecipe('model_rockets');
        window.__chronosUi.setRackTab('engines');
        window.__chronosUi.setBayTab('track');
        return true;
      })()`
    );
    await setHeroToggle(cdp, sessionId, true);
    await waitForCondition(
      cdp,
      sessionId,
      'Model rockets playing',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        return { ok: Boolean(v && v.readyState >= 2 && !v.paused) };
      })()`,
      12000
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnRailRec')?.click(); true;`);
    await waitForCondition(
      cdp,
      sessionId,
      'Model rockets active flight with targets & speed',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const t = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const speed = document.getElementById('kpiKinSpeed')?.textContent || '';
        return {
          ok: Boolean(v && v.currentTime >= 2.1 && t >= 1 && (speed.endsWith('m/s') || speed.endsWith('px/s'))),
          cur: v?.currentTime, t, speed
        };
      })()`,
      18000
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnRailRec')?.click(); true;`);
    await pauseVideo(cdp, sessionId);
    await sleep(240);

    // 01: Full Instrument overview (1440x900 @2x, showcasing live split-wipe at 24%)
    const fullShot = await cdp.send(
      'Page.captureScreenshot',
      { format: 'png', captureBeyondViewport: false, clip: { x: 0, y: 0, width: VIEWPORT_W, height: VIEWPORT_H, scale: 1 } },
      sessionId
    );
    fs.mkdirSync(PANELS_DIR, { recursive: true });
    fs.writeFileSync(path.join(PANELS_DIR, '01_deck_overview.png'), Buffer.from(fullShot.data, 'base64'));
    console.log(`  [PNG] docs/assets/panels/01_deck_overview.png (${VIEWPORT_W}x${VIEWPORT_H} @${DPR}x)`);

    // 02: Top 36px Status Rail
    await captureBoundingSelectors(cdp, sessionId, 'header.bench-rail', path.join(PANELS_DIR, '02_header_telemetry.png'), { pad: 2 });
    // 04: Model Rockets HUD
    await captureBoundingSelectors(cdp, sessionId, '#hudPanel', path.join(PANELS_DIR, '04_hud_model_rockets.png'));
    // 05: Scope Bay Track tab (Split PiP + 12 KPIs + Custody Timeline)
    await captureBoundingSelectors(cdp, sessionId, '.bench-bay', path.join(PANELS_DIR, '05_split_pip_and_primary_kpis.png'), { pad: 2 });
    // 13: Left Rack Engines tab (Scratch Bus P1..P5 + Lab Mode + Continuum Substrate + Optical Organs)
    await captureBoundingSelectors(cdp, sessionId, 'aside.bench-rack', path.join(PANELS_DIR, '13_showcases_input_and_modes.png'), { pad: 2 });

    // 14: Left Rack Calibrate tab (with Surveyed Coordinates expanded)
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__chronosUi.setRackTab('calibrate');
        const d = document.getElementById('calKeyboardDisclosure');
        if (d) d.open = true;
        return true;
      })()`
    );
    await captureBoundingSelectors(cdp, sessionId, 'aside.bench-rack', path.join(PANELS_DIR, '14_calibration_hyperparams_recorder.png'), { pad: 2 });
    await cdp.evalJs(sessionId, `(() => { const d = document.getElementById('calKeyboardDisclosure'); if (d) d.open = false; return true; })()`);

    // 10: Left Rack Record & Variational Tuning tab
    await cdp.evalJs(sessionId, `window.__chronosUi.setRackTab('record'); true;`);
    await captureBoundingSelectors(cdp, sessionId, 'aside.bench-rack', path.join(PANELS_DIR, '10_variational_kinematic_matrix.png'), { pad: 2 });
    await cdp.evalJs(sessionId, `window.__chronosUi.setRackTab('engines'); true;`);

    // 17: Presets Drawer (53 recipes)
    await cdp.evalJs(sessionId, `window.__chronosUi.openPresets(true); true;`);
    await sleep(220);
    await captureBoundingSelectors(cdp, sessionId, '#presetsDrawer', path.join(PANELS_DIR, '17_presets_drawer.png'), { pad: 4 });
    await cdp.evalJs(sessionId, `window.__chronosUi.openPresets(false); true;`);

    // 11: Scope Bay Physics ODE tab on Galileo Pendulum
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__chronosUi.loadRecipe('galileo');
        window.__chronosUi.setBayTab('physics');
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Galileo pendulum Physics ODE fit',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const r2 = parseFloat((document.getElementById('kpiPhysicsR2')?.textContent || '0').trim()) || 0;
        const ell = document.getElementById('kpiPhysicsEllipticT')?.textContent || '--';
        const reg = document.getElementById('physRegimeBadge')?.textContent || '';
        return { ok: Boolean(v && v.currentTime >= 2.2 && r2 >= 0.70 && !ell.startsWith('--') && !reg.startsWith('R0') && reg !== 'STANDBY'), cur: v?.currentTime, r2, ell, reg };
      })()`,
      20000
    );
    await sleep(500);
    await pauseVideo(cdp, sessionId);
    await captureBoundingSelectors(cdp, sessionId, '.bench-bay', path.join(PANELS_DIR, '11_physics_id_and_phase_portrait.png'), { pad: 2 });

    // 08: Scope Bay Phase tab on Nikko Metronome 0.80 Hz
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__chronosUi.loadRecipe('metronome');
        window.__chronosUi.setBayTab('phase');
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Metronome 0.80 Hz Hilbert quadrature lock',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const meta = document.getElementById('hilbertMeta')?.textContent || '';
        const hz = parseFloat(document.getElementById('kpiTremorHz')?.textContent || '0') || 0;
        return { ok: Boolean(v && v.currentTime >= 3.2 && meta.includes('COH') && hz >= 0.65 && hz <= 0.95), meta, hz };
      })()`,
      18000
    );
    await pauseVideo(cdp, sessionId);
    await captureBoundingSelectors(cdp, sessionId, '.bench-bay', path.join(PANELS_DIR, '08_hilbert_quadrature_scope.png'), { pad: 2 });

    // 06, 07, 09: 16-Channel Roster, Custody Timeline, and Lanczos Spectral scope on 12-target dense swarm
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__chronosUi.loadRecipe('bench_multi_target_swarm');
        document.getElementById('btnResHd')?.click();
        window.__chronosUi.setBayTab('track');
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      '12-target swarm locks & custody',
      `(() => {
        const t = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const uniq = parseInt(document.getElementById('kpiPersistUnique')?.textContent || '0', 10) || 0;
        return { ok: t >= 8 && uniq >= 8, t, uniq };
      })()`,
      16000
    );
    await sleep(1600);
    await captureBoundingSelectors(cdp, sessionId, 'aside.bench-roster', path.join(PANELS_DIR, '06_target_roster_16slot.png'), { pad: 2 });
    await captureBoundingSelectors(cdp, sessionId, '#persistencePanel', path.join(PANELS_DIR, '07_session_persistence.png'), { pad: 4 });

    await cdp.evalJs(sessionId, `window.__chronosUi.setBayTab('phase'); true;`);
    await waitForCondition(
      cdp,
      sessionId,
      'Lanczos Fiedler lambda2 active',
      `(() => {
        const spec = document.getElementById('spectralMeta')?.textContent || '';
        const m = spec.match(/= ([-\\d.]+)/);
        const l2 = m ? parseFloat(m[1]) : 0;
        return { ok: l2 > 0.05, spec, l2 };
      })()`,
      12000
    );
    await sleep(600);
    await cdp.evalJs(sessionId, `document.getElementById('spectralCanvas')?.closest('.scope-frame')?.setAttribute('id', 'captureSpectralFrame'); true;`);
    await captureBoundingSelectors(cdp, sessionId, '#captureSpectralFrame', path.join(PANELS_DIR, '09_lanczos_fiedler_timeline.png'), { pad: 4 });

    // 12: Scope Bay Field & Diag tab (murmuration_1200 + 64x36 Hodge + click-to-inspect + Expert Drawer open)
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__chronosUi.loadRecipe('bench_murmuration_1200');
        window.__chronosUi.applyIntent({ kind: 'SET_HODGE', mode: 2 });
        window.__chronosUi.setBayTab('field');
        document.getElementById('expertDiagnosticsDrawer').open = true;
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'murmuration_1200 >= 1,100 confirmed + SE(2) bodies + FTLE',
      `(() => {
        const conf = parseInt(((document.getElementById('kpiSwarmConfirmed')?.textContent || '0').split('/')[0] || '0').replace(/,/g, ''), 10);
        const rigid = document.getElementById('kpiRigidBodies')?.textContent || '';
        const ftle = document.getElementById('kpiContinuumFtle')?.textContent || '';
        return {
          ok: conf >= 1100 && /^[1-9]\\d*/.test(rigid) && !ftle.startsWith('WARM') && !ftle.startsWith('--'),
          conf, rigid, ftle
        };
      })()`,
      20000
    );
    for (const [nx, ny] of [[0.35, 0.42], [0.50, 0.50], [0.65, 0.58], [0.30, 0.55], [0.62, 0.40]]) {
      await cdp.evalJs(
        sessionId,
        `(() => {
          const hud = document.getElementById('hudCanvas');
          const r = hud.getBoundingClientRect();
          hud.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, clientX: r.left + r.width * ${nx}, clientY: r.top + r.height * ${ny} }));
          return true;
        })()`
      );
      await sleep(250);
      const txt = await cdp.evalJs(sessionId, `(document.getElementById('swarmInspectReadout')?.textContent || '').trim()`);
      if (txt.startsWith('INSP ')) break;
    }
    await sleep(300);
    await captureBoundingSelectors(cdp, sessionId, '.bench-bay', path.join(PANELS_DIR, '12_continuum_swarm_panel.png'), { pad: 2 });
    await cdp.evalJs(sessionId, `document.getElementById('expertDiagnosticsDrawer').open = false; true;`);

    // 16: Scope Bay Audit A/B tab (Slot A Real Ref PASS + Slot B Stutter FAIL)
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__chronosUi.setBayTab('audit');
        document.getElementById('btnAuditSampleRef')?.click();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Audit Slot A complete',
      `(() => {
        const txt = document.getElementById('auditGradeA')?.textContent || '';
        return { ok: txt.includes('PASS') || txt.includes('WARN') || txt.includes('FAIL'), txt };
      })()`,
      30000
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnAuditSampleStutter')?.click(); true;`);
    await waitForCondition(
      cdp,
      sessionId,
      'Audit Slot B complete',
      `(() => {
        const txt = document.getElementById('auditGradeB')?.textContent || '';
        return { ok: txt.includes('FAIL') || txt.includes('WARN'), txt };
      })()`,
      30000
    );
    await sleep(250);
    await captureBoundingSelectors(cdp, sessionId, '.bench-bay', path.join(PANELS_DIR, '16_audit_ab_bench.png'), { pad: 2 });

    // 15: Live Stream 3-Tier Modal
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__chronosUi.setRackTab('acquire');
        document.getElementById('btnSrcLiveStream')?.click();
        return true;
      })()`
    );
    await waitForCondition(cdp, sessionId, 'Live Stream Modal Open', `(() => ({ ok: document.getElementById('liveStreamDialog')?.open === true }))()`);
    await captureBoundingSelectors(cdp, sessionId, '#liveStreamDialog', path.join(PANELS_DIR, '15_live_stream_modal.png'));
    await cdp.evalJs(sessionId, `document.getElementById('liveStreamDialog').close(); true;`);

    console.log('\n[DONE] All v3.5.0 documentation media captured cleanly.');
  } finally {
    if (cdp) cdp.close();
    if (chromeProc) chromeProc.kill('SIGKILL');
    await srv.close();
    try {
      fs.rmSync(tmpDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 80 });
    } catch (_e) {
      // ignore transient macOS lock
    }
  }
}

main().catch((err) => {
  console.error('MEDIA CAPTURE FAILED:', err.message || err);
  process.exit(1);
});
