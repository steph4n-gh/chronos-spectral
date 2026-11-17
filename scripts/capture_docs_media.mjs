#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Deterministic Headless Chrome CDP Documentation Media Capture
// Generates:
//   - docs/assets/demos/demo_starling_swarm.gif (Starling murmuration, Hero 16 HIDDEN)
//   - docs/assets/demos/demo_model_rockets.gif  (Model rockets launch, separation & chutes)
//   - docs/assets/panels/01..15.png             (2x Retina screenshots of every Command Deck panel)
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

const VIEWPORT_W = 1720;
const VIEWPORT_H = 1080;
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

async function waitForCondition(cdp, sessionId, description, jsExpr, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  let lastVal = null;
  while (Date.now() < deadline) {
    lastVal = await cdp.evalJs(sessionId, jsExpr);
    if (lastVal && lastVal.ok) return lastVal;
    await sleep(80);
  }
  throw new Error(`Timeout waiting for [${description}]. Last state: ${JSON.stringify(lastVal)}`);
}

async function setViewportHeight(cdp, sessionId, height, dpr = DPR) {
  await cdp.send(
    'Emulation.setDeviceMetricsOverride',
    { width: VIEWPORT_W, height, deviceScaleFactor: dpr, mobile: false },
    sessionId
  );
}

async function captureBoundingSelectors(cdp, sessionId, selectors, outPath, options = {}) {
  const { pad = 8, dpr = DPR, waitMs = 240 } = options;
  const list = Array.isArray(selectors) ? selectors : [selectors];

  const meta = await cdp.evalJs(
    sessionId,
    `(() => {
      const sels = ${JSON.stringify(list)};
      const els = sels.map((s) => document.querySelector(s)).filter(Boolean);
      if (!els.length) return null;
      const isFixed = getComputedStyle(els[0]).position === 'fixed';
      const maxDocBottom = isFixed
        ? Math.ceil(Math.max(...els.map((e) => e.getBoundingClientRect().bottom)))
        : Math.ceil(Math.max(
            document.documentElement.scrollHeight,
            ...els.map((e) => e.getBoundingClientRect().bottom + window.scrollY)
          ));
      return { isFixed, maxDocBottom };
    })()`
  );
  if (!meta) {
    throw new Error(`Missing selector group: ${list.join(', ')}`);
  }

  const targetVpH = meta.isFixed ? VIEWPORT_H : Math.max(VIEWPORT_H, meta.maxDocBottom + 120);
  await setViewportHeight(cdp, sessionId, targetVpH, dpr);

  const clip = await cdp.evalJs(
    sessionId,
    `(() => {
      if (!${meta.isFixed}) window.scrollTo(0, 0);
      const sels = ${JSON.stringify(list)};
      const els = sels.map((s) => document.querySelector(s)).filter(Boolean);
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
            const w = Math.min(${VIEWPORT_W} - x, Math.ceil(x1 - x0 + ${pad * 2}));
            const h = Math.min(${targetVpH} - y, Math.ceil(y1 - y0 + ${pad * 2}));
            resolve({ x, y, width: w, height: h, scale: 1 });
          }, ${waitMs});
        }));
      });
    })()`
  );

  const shot = await cdp.send(
    'Page.captureScreenshot',
    { format: 'png', captureBeyondViewport: false, clip },
    sessionId
  );
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.from(shot.data, 'base64'));
  if (targetVpH !== VIEWPORT_H) {
    await setViewportHeight(cdp, sessionId, VIEWPORT_H, dpr);
  }
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
      const cur = window.__chronosUiState?.overlays?.showHeroTargets;
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
      ok: Boolean(window.__chronosUiState?.overlays?.showHeroTargets) === ${Boolean(visible)},
      cur: window.__chronosUiState?.overlays?.showHeroTargets
    }))()`
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

async function captureHudSequenceGif(cdp, sessionId, ffmpegBin, cfg) {
  const { frameDir, outGifPath, showcase, startTimeSec, totalGifFrames, strideFrames, gifFps, scaleWidth, heroVisible, beforePauseCheck, onMidFrameCapture } = cfg;
  fs.mkdirSync(frameDir, { recursive: true });

  await setViewportHeight(cdp, sessionId, VIEWPORT_H, DPR);
  await cdp.evalJs(
    sessionId,
    `(() => {
      document.querySelector(${JSON.stringify(`.showcase-btn[data-showcase="${showcase}"]`)})?.click();
      return true;
    })()`
  );
  await setHeroToggle(cdp, sessionId, heroVisible);

  if (beforePauseCheck) {
    await waitForCondition(cdp, sessionId, `${showcase} pre-capture condition`, beforePauseCheck, 22000);
  }

  // Pause video deterministically and optionally replay/seek cleanly
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
    `${showcase} paused`,
    `(() => ({ ok: document.getElementById('hiddenVideoSource')?.paused === true }))()`
  );

  if (typeof startTimeSec === 'number') {
    if (startTimeSec <= 0.05) {
      await cdp.evalJs(sessionId, `document.getElementById('btnVidReplay').click(); true;`);
      await sleep(180);
      await cdp.evalJs(sessionId, `(() => { const v = document.getElementById('hiddenVideoSource'); if (v && !v.paused) document.getElementById('btnVidPlayPause')?.click(); return true; })()`);
      await waitForCondition(cdp, sessionId, `${showcase} replayed & paused`, `(() => ({ ok: document.getElementById('hiddenVideoSource')?.paused === true }))()`);
    }
  }

  const clip = await cdp.evalJs(
    sessionId,
    `(() => {
      const hud = document.getElementById('hudPanel');
      hud.scrollIntoView({ block: 'start', inline: 'nearest' });
      window.scrollTo(0, Math.max(0, Math.floor(hud.getBoundingClientRect().top + window.scrollY - 8)));
      return new Promise((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const r = hud.getBoundingClientRect();
          resolve({
            x: Math.max(0, Math.floor(r.left - 4)),
            y: Math.max(0, Math.floor(r.top - 4)),
            width: Math.ceil(r.width + 8),
            height: Math.ceil(r.height + 8),
            scale: 1,
          });
        }));
      });
    })()`
  );

  for (let i = 0; i < totalGifFrames; i++) {
    await stepForwardNFrames(cdp, sessionId, strideFrames);
    await cdp.evalJs(sessionId, `new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))`);
    const shot = await cdp.send(
      'Page.captureScreenshot',
      { format: 'png', captureBeyondViewport: false, clip },
      sessionId
    );
    const frameFile = path.join(frameDir, `frame_${String(i).padStart(4, '0')}.png`);
    fs.writeFileSync(frameFile, Buffer.from(shot.data, 'base64'));
    if (onMidFrameCapture) {
      await onMidFrameCapture(i, totalGifFrames);
    }
  }

  await encodeGifFromFrames(ffmpegBin, frameDir, outGifPath, gifFps, scaleWidth);
  await setViewportHeight(cdp, sessionId, VIEWPORT_H, DPR);
}

async function main() {
  console.log('========================================================================');
  console.log(' CHRONOS-SPECTRAL // Documentation Media Capture (GIFs + Panel PNGs)');
  console.log('========================================================================');

  const chromeBin = findChromeBinary();
  const ffmpegBin = findFfmpegBinary();
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chronos-docs-media-'));
  const chromeUserDir = path.join(tmpDir, 'chrome');
  fs.mkdirSync(chromeUserDir, { recursive: true });

  const srv = await startChronosServer({ rootDir: WEB_DIR, port: 0, host: '127.0.0.1' });
  const appUrl = `http://127.0.0.1:${srv.port}/index.html?view=deck`;
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
    await setViewportHeight(cdp, sessionId, VIEWPORT_H, DPR);

    await waitForCondition(
      cdp,
      sessionId,
      'Wasm boot ready',
      `(() => ({ ok: (document.getElementById('pillKernel')?.textContent || '').includes('0 HEAP ALLOC') }))()`
    );

    if (!PANELS_ONLY) {
      // ------------------------------------------------------------------------
      // Phase 1: Demo GIF #1 — Starling Murmuration Swarm WITHOUT Hero-16 Trackers
      // ------------------------------------------------------------------------
      console.log('\n[Phase 1] Capturing Starling Murmuration GIF (Hero 16 HIDDEN)...');
      let starlingHudStillCaptured = false;
      await captureHudSequenceGif(cdp, sessionId, ffmpegBin, {
        frameDir: path.join(tmpDir, 'gif_starling'),
        outGifPath: path.join(DEMOS_DIR, 'demo_starling_swarm.gif'),
        showcase: 'starling_1000',
        totalGifFrames: 75,
        strideFrames: 1,
        gifFps: 18,
        scaleWidth: 640,
        heroVisible: false,
        beforePauseCheck: `(() => {
          const v = document.getElementById('hiddenVideoSource');
          const conf = parseInt(((document.getElementById('kpiSwarmConfirmed')?.textContent || '0').split('/')[0] || '0').replace(/,/g, ''), 10);
          const hiddenText = (document.getElementById('btnHudHeroToggle')?.textContent || '').toUpperCase();
          return {
            ok: v && v.currentTime >= 2.0 && conf >= 1150 && hiddenText.includes('HIDDEN') && window.__chronosUiState?.overlays?.showHeroTargets === false,
            t: v?.currentTime || 0,
            conf,
          };
        })()`,
        onMidFrameCapture: async (i, total) => {
          if (!starlingHudStillCaptured && i === Math.floor(total * 0.45)) {
            starlingHudStillCaptured = true;
            await captureBoundingSelectors(cdp, sessionId, '#hudPanel', path.join(PANELS_DIR, '03_hud_starling_swarm.png'));
          }
        },
      });
      await setHeroToggle(cdp, sessionId, true);

      // ------------------------------------------------------------------------
      // Phase 2: Demo GIF #2 — Model Rockets Launch, Stage Sep & Parachute Recovery
      // ------------------------------------------------------------------------
      console.log('\n[Phase 2] Capturing Model Rockets Full-Flight GIF (Hero 16 ON)...');
      let capturedRocketDeckAndPanels = false;
      await captureHudSequenceGif(cdp, sessionId, ffmpegBin, {
        frameDir: path.join(tmpDir, 'gif_rockets'),
        outGifPath: path.join(DEMOS_DIR, 'demo_model_rockets.gif'),
        showcase: 'model_rockets',
        startTimeSec: 0.0,
        totalGifFrames: 92,
        strideFrames: 4, // Advances deterministic 25fps frame-by-frame across 0.0s .. 14.7s (boost, sep, apogee, twin chutes)
        gifFps: 14,
        scaleWidth: 640,
        heroVisible: true,
        beforePauseCheck: `(() => {
          const v = document.getElementById('hiddenVideoSource');
          const btn = document.querySelector('.showcase-btn[data-showcase="model_rockets"]');
          return { ok: Boolean(btn?.getAttribute('aria-pressed') === 'true' && v && v.readyState >= 2) };
        })()`,
        onMidFrameCapture: async (i, total) => {
          if (!capturedRocketDeckAndPanels && i >= 22) {
            const active = await cdp.evalJs(
              sessionId,
              `parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0`
            );
            if (active >= 1 || i === Math.floor(total * 0.3)) {
              capturedRocketDeckAndPanels = true;
              await captureBoundingSelectors(cdp, sessionId, 'header.deck-header', path.join(PANELS_DIR, '02_header_telemetry.png'));
              await captureBoundingSelectors(cdp, sessionId, '#hudPanel', path.join(PANELS_DIR, '04_hud_model_rockets.png'));
            }
          }
        },
      });
    } else {
      // Panels-only refresh: capture 03_hud_starling_swarm.png cleanly with Hero 16 HIDDEN
      console.log('\n[Panels-Only] Capturing 03_hud_starling_swarm.png (Hero 16 HIDDEN)...');
      await cdp.evalJs(sessionId, `document.querySelector('.showcase-btn[data-showcase="starling_1000"]')?.click(); true;`);
      await setHeroToggle(cdp, sessionId, false);
      await waitForCondition(
        cdp,
        sessionId,
        'Starling 1,150+ confirmed for panel 03',
        `(() => {
          const v = document.getElementById('hiddenVideoSource');
          const conf = parseInt(((document.getElementById('kpiSwarmConfirmed')?.textContent || '0').split('/')[0] || '0').replace(/,/g, ''), 10);
          return { ok: Boolean(v && v.currentTime >= 2.2 && conf >= 1150 && window.__chronosUiState?.overlays?.showHeroTargets === false) };
        })()`,
        20000
      );
      await cdp.evalJs(sessionId, `(() => { const v = document.getElementById('hiddenVideoSource'); if (v && !v.paused) document.getElementById('btnVidPlayPause')?.click(); return true; })()`);
      await captureBoundingSelectors(cdp, sessionId, '#hudPanel', path.join(PANELS_DIR, '03_hud_starling_swarm.png'));
      await setHeroToggle(cdp, sessionId, true);
    }

    console.log('\n[Phase 3] Capturing Individual Command Deck Panels...');
    // Load Model Rockets cleanly from t=0, record during ascent/boost so overview/header/PiP/matrix/recorder all populate cleanly
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.querySelector('.showcase-btn[data-showcase="model_rockets"]')?.click();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Model rockets started & ready',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const btn = document.querySelector('.showcase-btn[data-showcase="model_rockets"]');
        return { ok: Boolean(btn?.getAttribute('aria-pressed') === 'true' && v && v.readyState >= 2 && !v.paused) };
      })()`,
      12000
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnRecToggle')?.click(); true;`);
    await waitForCondition(
      cdp,
      sessionId,
      'Model rockets active flight for Deck Overview, Header, HUD, PiP & Variational Matrix',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const t = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const speed = document.getElementById('kpiKinSpeed')?.textContent || '';
        const odeR2 = parseFloat((document.getElementById('kpiPhysicsR2')?.textContent || '0').trim()) || 0;
        const eq = document.getElementById('physicsOdeEq')?.textContent || '';
        return {
          ok: Boolean(v && v.currentTime >= 2.0 && t >= 1 && (speed.endsWith('m/s') || speed.endsWith('px/s')) && odeR2 >= 0.35 && !eq.startsWith("x'' = 0, y'' = 0")),
          cur: v?.currentTime, t, speed, odeR2, eq
        };
      })()`,
      16000
    );

    await cdp.evalJs(sessionId, `(() => { document.getElementById('btnRecToggle')?.click(); const v = document.getElementById('hiddenVideoSource'); if (v && !v.paused) document.getElementById('btnVidPlayPause')?.click(); return true; })()`);
    await waitForCondition(cdp, sessionId, 'paused', `(() => ({ ok: document.getElementById('hiddenVideoSource')?.paused === true }))()`).catch(() => {});
    await sleep(220);

    // Ensure metric m/s readout if preset cal stayed clean
    if (PANELS_ONLY) {
      await captureBoundingSelectors(cdp, sessionId, 'header.deck-header', path.join(PANELS_DIR, '02_header_telemetry.png'));
      await captureBoundingSelectors(cdp, sessionId, '#hudPanel', path.join(PANELS_DIR, '04_hud_model_rockets.png'));
    }

    // Full deck overview (01_deck_overview.png)
    await cdp.evalJs(sessionId, `window.scrollTo(0, 0); true;`);
    await sleep(220);
    await setViewportHeight(cdp, sessionId, 1020, DPR);
    const fullShot = await cdp.send(
      'Page.captureScreenshot',
      { format: 'png', captureBeyondViewport: false, clip: { x: 0, y: 0, width: VIEWPORT_W, height: 1020, scale: 1 } },
      sessionId
    );
    fs.mkdirSync(PANELS_DIR, { recursive: true });
    fs.writeFileSync(path.join(PANELS_DIR, '01_deck_overview.png'), Buffer.from(fullShot.data, 'base64'));
    console.log(`  [PNG] docs/assets/panels/01_deck_overview.png (${VIEWPORT_W}x1020 @${DPR}x)`);
    await setViewportHeight(cdp, sessionId, VIEWPORT_H, DPR);

    // Capture full Split PiP panel (parent of #pipCanvas), Variational & Kinematic Matrix (parent of #fistaGapMeta), and Right Column Panels 13 & 14
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.getElementById('pipCanvas')?.closest('.panel')?.setAttribute('id', 'panelPipAudit');
        document.getElementById('fistaGapMeta')?.closest('.panel')?.setAttribute('id', 'panelVariationalMatrix');
        document.getElementById('hilbertCanvas')?.closest('.panel')?.setAttribute('id', 'panelHilbertScope');
        document.getElementById('spectralCanvas')?.closest('.panel')?.setAttribute('id', 'panelSpectralScope');
        document.getElementById('inputResMeta')?.closest('.panel')?.setAttribute('id', 'panelInputSource');
        document.getElementById('labBadge')?.closest('.panel')?.setAttribute('id', 'panelLabMode');
        document.getElementById('calBadge')?.closest('.panel')?.setAttribute('id', 'panelCalibration');
        document.getElementById('phaseOffsetLabel')?.closest('.panel')?.setAttribute('id', 'panelPhaseStrobe');
        document.getElementById('sliderK')?.closest('.panel')?.setAttribute('id', 'panelHyperparams');
        document.getElementById('recStatusLabel')?.closest('.panel')?.setAttribute('id', 'panelRecorder');
        return true;
      })()`
    );

    await captureBoundingSelectors(cdp, sessionId, '#panelPipAudit', path.join(PANELS_DIR, '05_split_pip_and_primary_kpis.png'));
    await captureBoundingSelectors(cdp, sessionId, '#panelVariationalMatrix', path.join(PANELS_DIR, '10_variational_kinematic_matrix.png'));
    await captureBoundingSelectors(cdp, sessionId, ['.showcase-strip', '#panelInputSource', '#panelLabMode'], path.join(PANELS_DIR, '13_showcases_input_and_modes.png'));

    // Expand hyperparams tuning disclosure so 14 captures full hyperparameter controls
    await cdp.evalJs(sessionId, `(() => { const d = document.querySelector('#panelHyperparams details.tuning-disclosure'); if (d) d.open = true; return true; })()`);
    await captureBoundingSelectors(cdp, sessionId, ['#panelCalibration', '#panelPhaseStrobe', '#panelHyperparams', '#panelRecorder'], path.join(PANELS_DIR, '14_calibration_hyperparams_recorder.png'));
    await cdp.evalJs(sessionId, `(() => { const d = document.querySelector('#panelHyperparams details.tuning-disclosure'); if (d) d.open = false; return true; })()`);

    // Region 3D Physics ID: galileo_pendulum_peg.mp4 yields Pendulum_Harmonic (R5), Elliptic T(theta0), Curvature, and phase portrait
    await cdp.evalJs(
      sessionId,
      `(() => {
        const sel = document.getElementById('sampleVideoSelect');
        sel.value = './samples/galileo_pendulum_peg.mp4';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Galileo pendulum Physics ID with strong ODE R2 + Elliptic T + Pivot',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const r2 = parseFloat((document.getElementById('kpiPhysicsR2')?.textContent || '0').trim()) || 0;
        const ell = document.getElementById('kpiPhysicsEllipticT')?.textContent || '-- s';
        const reg = document.getElementById('physRegimeBadge')?.textContent || '';
        return { ok: Boolean(v && v.currentTime >= 2.2 && r2 >= 0.70 && !ell.startsWith('--') && !reg.startsWith('R0')), cur: v?.currentTime, r2, ell, reg };
      })()`,
      16000
    );
    await sleep(600);
    await cdp.evalJs(sessionId, `(() => { const v = document.getElementById('hiddenVideoSource'); if (v && !v.paused) document.getElementById('btnVidPlayPause')?.click(); return true; })()`);
    await captureBoundingSelectors(cdp, sessionId, '#physicsIdPanel', path.join(PANELS_DIR, '11_physics_id_and_phase_portrait.png'));

    // Region 3A Hilbert Quadrature Scope: harmonic_metronome_phase.mp4 locked at 0.80 Hz
    await cdp.evalJs(
      sessionId,
      `(() => {
        const sel = document.getElementById('sampleVideoSelect');
        sel.value = './samples/harmonic_metronome_phase.mp4';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Metronome 0.80 Hz Hilbert Quadrature lock + clean orbit',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const meta = document.getElementById('hilbertMeta')?.textContent || '';
        const hz = parseFloat(document.getElementById('kpiTremorHz')?.textContent || '0') || 0;
        return { ok: v && v.currentTime >= 3.2 && meta.includes('COH') && hz >= 0.70 && hz <= 0.92, meta, hz };
      })()`,
      16000
    );
    await cdp.evalJs(sessionId, `(() => { const v = document.getElementById('hiddenVideoSource'); if (v && !v.paused) document.getElementById('btnVidPlayPause')?.click(); return true; })()`);
    await captureBoundingSelectors(cdp, sessionId, '#panelHilbertScope', path.join(PANELS_DIR, '08_hilbert_quadrature_scope.png'));

    // Region 2C Target Roster, 3B Lanczos Fiedler & Timeline, and Persistence Swimlanes: multi_target_swarm (12 discs) in HD
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.getElementById('btnSrcTestbench').click();
        const sel = document.getElementById('benchScenarioSelect');
        sel.value = 'multi_target_swarm';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        document.getElementById('btnResHd').click();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      '12-target multi_target_swarm active slots + persistence swimlanes + Fiedler spectrum',
      `(() => {
        const t = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const spec = document.getElementById('spectralMeta')?.textContent || '';
        const uniq = parseInt(document.getElementById('kpiPersistUnique')?.textContent || '0', 10) || 0;
        const m = spec.match(/λ₂ = ([-\\d.]+)/);
        const l2 = m ? parseFloat(m[1]) : 0;
        return { ok: t >= 8 && uniq >= 8 && l2 > 0.05, t, uniq, l2 };
      })()`,
      16000
    );
    await sleep(1800); // Let custody swimlanes build 3+ seconds of timeline bars & Fiedler sparkline
    await captureBoundingSelectors(cdp, sessionId, '.roster-panel', path.join(PANELS_DIR, '06_target_roster_16slot.png'));
    await captureBoundingSelectors(cdp, sessionId, '#persistencePanel', path.join(PANELS_DIR, '07_session_persistence.png'));
    await captureBoundingSelectors(cdp, sessionId, '#panelSpectralScope', path.join(PANELS_DIR, '09_lanczos_fiedler_timeline.png'));

    // Region 3E Continuum & Swarm Field Panel (murmuration_1200 + click-to-inspect + Expert Drawer OPEN)
    await cdp.evalJs(
      sessionId,
      `(() => {
        const sel = document.getElementById('benchScenarioSelect');
        sel.value = 'murmuration_1200';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'murmuration_1200 active >= 1100 confirmed + rigid SE(2) bodies + FTLE',
      `(() => {
        const conf = parseInt(((document.getElementById('kpiSwarmConfirmed')?.textContent || '0').split('/')[0] || '0').replace(/,/g, ''), 10);
        const rigid = document.getElementById('kpiRigidBodies')?.textContent || '';
        const ftle = document.getElementById('kpiContinuumFtle')?.textContent || '';
        return {
          ok: conf >= 1100 && /^[1-9]\\d*\\s*\\|/.test(rigid) && !ftle.startsWith('WARM') && !ftle.startsWith('--'),
          conf, rigid, ftle
        };
      })()`,
      18000
    );
    // Probe coordinates on #hudCanvas until #swarmInspectReadout populates with INSP μT...
    for (const [nx, ny] of [[0.35, 0.42], [0.50, 0.50], [0.65, 0.58], [0.30, 0.55], [0.62, 0.40]]) {
      await cdp.evalJs(
        sessionId,
        `(() => {
          const hud = document.getElementById('hudCanvas');
          const r = hud.getBoundingClientRect();
          hud.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, clientX: r.left + r.width * ${nx}, clientY: r.top + r.height * ${ny} }));
          document.getElementById('expertDiagnosticsDrawer').open = true;
          return true;
        })()`
      );
      await sleep(280);
      const txt = await cdp.evalJs(sessionId, `(document.getElementById('swarmInspectReadout')?.textContent || '').trim()`);
      if (txt.startsWith('INSP ')) break;
    }
    await sleep(320);
    await captureBoundingSelectors(cdp, sessionId, '#continuumFieldPanel', path.join(PANELS_DIR, '12_continuum_swarm_panel.png'));
    await cdp.evalJs(sessionId, `document.getElementById('expertDiagnosticsDrawer').open = false; true;`);

    // Region 5 Live Stream Suite modal
    await cdp.evalJs(sessionId, `document.getElementById('btnSrcLiveStream').click(); true;`);
    await waitForCondition(cdp, sessionId, 'Live Stream Modal Open', `(() => ({ ok: document.getElementById('liveStreamDialog')?.open === true }))()`);
    await captureBoundingSelectors(cdp, sessionId, '#liveStreamDialog', path.join(PANELS_DIR, '15_live_stream_modal.png'));
    await cdp.evalJs(sessionId, `document.getElementById('liveStreamDialog').close(); true;`);

    console.log('\n[DONE] All documentation GIFs and panel screenshots generated cleanly.');
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
