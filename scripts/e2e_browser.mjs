#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Zero-Dependency Headless Chrome CDP Full-Stack E2E Gate
// Exercises V8 Web Worker + OffscreenCanvas HUD, 256-page Wasm SIMD128 kernel,
// runtime 320x180 SD <-> 640x360 HD switching, real-camera H.264 MP4 playback,
// exact 1/fps transport step, Nikko 0.80 Hz vs APERIODIC gating, ASSUMED vs
// SURVEYED calibration provenance + Escape preservation, 56-col v1.5.0 export,
// and Live Tier-3 FFmpeg fMP4 -> MSE continuous stream with camera-cut reset.
// ============================================================================

import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { execSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { startChronosServer } from './dev_server.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const WEB_DIR = path.join(ROOT_DIR, 'web');

function parseProofDir() {
  const args = process.argv.slice(2);
  const idx = args.indexOf('--proof-dir');
  if (idx >= 0 && args[idx + 1]) return path.resolve(args[idx + 1]);
  if (process.env.CHRONOS_PROOF_DIR) return path.resolve(process.env.CHRONOS_PROOF_DIR);
  return '';
}

function findChromeBinary() {
  if (process.env.CHROME_BIN && fs.existsSync(process.env.CHROME_BIN)) {
    return process.env.CHROME_BIN;
  }
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
  throw new Error('Could not locate Google Chrome / Chromium binary. Set CHROME_BIN.');
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
    '--window-size=1440,900',
  ];
  if (process.platform === 'linux') {
    args.push('--no-sandbox', '--disable-dev-shm-usage', '--disable-features=dbus');
  }
  if (process.platform === 'linux' || process.env.CI) {
    args.push(
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--disable-accelerated-2d-canvas',
      '--disable-webgl'
    );
  }

  const env = process.platform === 'linux'
    ? { ...process.env, DBUS_SESSION_BUS_ADDRESS: '/dev/null', NO_AT_BRIDGE: '1' }
    : process.env;
  const proc = spawn(chromeBin, args, { stdio: ['ignore', 'ignore', 'pipe'], env });

  return new Promise((resolve, reject) => {
    let stderrBuf = '';
    const timer = setTimeout(() => {
      proc.kill('SIGKILL');
      reject(new Error(`Timed out waiting for Chrome DevTools endpoint. stderr:\n${stderrBuf}`));
    }, 30000);

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
    this.listeners = new Map();
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
      } else if (msg.method) {
        const list = this.listeners.get(msg.method);
        if (list) {
          for (const fn of list) fn(msg.params, msg.sessionId);
        }
      }
    });
  }

  on(method, fn) {
    if (!this.listeners.has(method)) this.listeners.set(method, []);
    this.listeners.get(method).push(fn);
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
      {
        expression,
        awaitPromise: true,
        returnByValue: true,
      },
      sessionId
    );
    if (res.exceptionDetails) {
      const desc = res.exceptionDetails.exception?.description || res.exceptionDetails.text || JSON.stringify(res.exceptionDetails);
      throw new Error(`Page JS exception: ${desc}`);
    }
    return res.result ? res.result.value : undefined;
  }

  async saveScreenshot(sessionId, proofDir, filename) {
    if (!proofDir) return;
    fs.mkdirSync(proofDir, { recursive: true });
    const shot = await this.send('Page.captureScreenshot', { format: 'png' }, sessionId);
    if (shot && shot.data) {
      fs.writeFileSync(path.join(proofDir, filename), Buffer.from(shot.data, 'base64'));
    }
  }

  close() {
    try {
      this.ws.close();
    } catch (_e) {
      // ignore
    }
  }
}

async function waitForCondition(cdp, sessionId, description, jsExpr, timeoutMs = 14000) {
  const ciTimeoutScale = Math.max(1.0, Math.min(2.0, Number(process.env.CHRONOS_CI_BUDGET_SCALE || '1.0') * 0.6));
  const deadline = Date.now() + Math.round(timeoutMs * ciTimeoutScale);
  let lastVal = null;
  while (Date.now() < deadline) {
    lastVal = await cdp.evalJs(sessionId, jsExpr);
    if (lastVal && lastVal.ok) return lastVal;
    await new Promise((r) => setTimeout(r, 90));
  }
  throw new Error(`Timeout waiting for [${description}]. Last state: ${JSON.stringify(lastVal)}`);
}

async function main() {
  console.log('========================================================================');
  console.log(' CHRONOS-SPECTRAL // Headless Chrome CDP Full-Stack E2E Verification');
  console.log('========================================================================');

  const chromeBin = findChromeBinary();
  const proofDir = parseProofDir();
  console.log(`  Browser : ${chromeBin}`);
  if (proofDir) console.log(`  Proofs  : ${proofDir}`);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chronos-e2e-'));
  const downloadDir = path.join(tmpDir, 'downloads');
  fs.mkdirSync(downloadDir, { recursive: true });

  const srv = await startChronosServer({ rootDir: WEB_DIR, port: 0, host: '127.0.0.1' });
  const appUrl = `http://127.0.0.1:${srv.port}/index.html?view=deck`;
  let chromeProc = null;
  let cdp = null;

  try {
    const launched = await launchHeadlessChrome(chromeBin, tmpDir);
    chromeProc = launched.proc;
    cdp = new CdpClient(launched.wsUrl);
    await cdp.connect();

    const pageErrors = [];
    let allowNegativeNetworkLogs = false;
    cdp.on('Runtime.exceptionThrown', (params) => {
      const detail = params.exceptionDetails;
      pageErrors.push(detail.exception?.description || detail.text || 'Uncaught exception');
    });
    cdp.on('Runtime.consoleAPICalled', (params) => {
      if (params.type === 'error') {
        const text = (params.args || []).map((a) => a.value ?? a.description ?? '').join(' ');
        pageErrors.push(`console.error: ${text}`);
      }
    });
    cdp.on('Log.entryAdded', (params) => {
      const entry = params?.entry;
      if (entry && entry.level === 'error') {
        const url = entry.url || '';
        if (allowNegativeNetworkLogs && url.includes('__e2e_neg_')) return;
        pageErrors.push(`Log.error (${entry.source}): ${entry.text} ${url}`.trim());
      }
    });

    const { targetId } = await cdp.send('Target.createTarget', { url: appUrl });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    cdp.on('Target.attachedToTarget', (params) => {
      if (params && params.sessionId) {
        cdp.send('Runtime.enable', {}, params.sessionId).catch(() => {});
      }
    });
    await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: true }, sessionId);
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Log.enable', {}, sessionId);
    await cdp.send('Runtime.enable', {}, sessionId);
    await cdp.send(
      'Emulation.setDeviceMetricsOverride',
      { width: 1440, height: 900, deviceScaleFactor: process.env.CI ? 1 : 2, mobile: false },
      sessionId
    );
    await cdp.send('Browser.setDownloadBehavior', {
      behavior: 'allow',
      downloadPath: downloadDir,
    });

    // 1. Verify Boot + 320-Page Arena + 320x180 SD Blind Pillar Testbench (0 Oracle swaps)
    const sdState = await waitForCondition(
      cdp,
      sessionId,
      '320x180 SD boot + pillar collisions + 0 Oracle swaps',
      `(() => {
        const kernel = document.getElementById('pillKernel')?.textContent || '';
        const arena = document.getElementById('pillArena')?.textContent || '';
        const resMeta = document.getElementById('inputResMeta')?.textContent || '';
        const targets = document.getElementById('kpiTargets')?.textContent || '';
        const swaps = (document.getElementById('kpiPadicSwaps')?.textContent || '').trim();
        const collisions = parseInt(document.getElementById('kpiCollisions')?.textContent || '0', 10);
        return {
          ok: kernel.includes('0 HEAP ALLOC / 0 IMPORTS') &&
              arena.includes('320 PAGES / 20.00 MB') &&
              resMeta === '320×180 SD RGBA' &&
              targets === '4 / 16' &&
              collisions >= 2 &&
              swaps === '◆ 0',
          kernel, arena, resMeta, targets, swaps, collisions,
        };
      })()`
    );
    console.log(`  [PASS] Step 1 (320x180 SD Boot & Pillar Bench): ${sdState.arena}, targets=${sdState.targets}, collisions=${sdState.collisions}, swaps=${sdState.swaps}`);

    // 2. Runtime Switch to 640x360 HD (64x64 PiP)
    await cdp.evalJs(sessionId, `document.getElementById('btnResHd').click(); true;`);
    const hdState = await waitForCondition(
      cdp,
      sessionId,
      '640x360 HD live switch + 4/16 targets + 0 Oracle swaps',
      `(() => {
        const resMeta = document.getElementById('inputResMeta')?.textContent || '';
        const pipEl = document.getElementById('pipCanvas');
        const pipAria = pipEl?.getAttribute('aria-label') || '';
        const targets = document.getElementById('kpiTargets')?.textContent || '';
        const swaps = (document.getElementById('kpiPadicSwaps')?.textContent || '').trim();
        const collisions = parseInt(document.getElementById('kpiCollisions')?.textContent || '0', 10);
        return {
          ok: resMeta === '640×360 HD RGBA' &&
              pipAria.startsWith('64x64') &&
              pipEl?.width === 412 && pipEl?.height === 234 &&
              targets === '4 / 16' &&
              collisions >= 1 &&
              swaps === '◆ 0',
          resMeta, pipAria, targets, swaps, collisions,
        };
      })()`
    );
    console.log(`  [PASS] Step 2 (640x360 HD Live Switch): res=${hdState.resMeta}, pip=${hdState.pipAria}, collisions=${hdState.collisions}, swaps=${hdState.swaps}`);

    // 2b. Verify All-Targets Split 1x Raw vs 25x Eulerian PiPs (Slots 1-16) + Roster Click-to-Pin & Zero Layout Shift
    const rosterState = await waitForCondition(
      cdp,
      sessionId,
      '16-slot Split PiP Roster active on all 4 tracked slots with live telemetry',
      `(() => {
        const cards = Array.from(document.querySelectorAll('.roster-card'));
        if (cards.length !== 16) return { ok: false, count: cards.length };
        const slot0 = document.getElementById('slotPip0');
        const slotDimOk = slot0?.width === 276 && slot0?.height === 148;
        let activeSlots = 0;
        let pickSlot = -1;
        let pickTid = 0;
        let canvasRect = null;
        let cardRect = null;
        for (let i = 0; i < cards.length; i++) {
          const st = cards[i].dataset.state || '';
          const tid = parseInt(cards[i].dataset.tid || '0', 10);
          const spdTxt = cards[i].querySelector('.rs-val-spd')?.textContent || '';
          const ageSec = parseFloat(cards[i].querySelector('.rs-val-age')?.textContent || '0') || 0;
          const idVal = cards[i].querySelector('.rs-val-id')?.textContent || '';
          if (tid > 0 && st !== 'IDLE' && st !== 'HOLD' && !spdTxt.startsWith('--') && idVal.includes('v₂:')) {
            activeSlots++;
            if (pickSlot < 0 && (st === 'FOCUS' || st === 'LOCKED') && ageSec >= 0.8) {
              pickSlot = i;
              pickTid = tid;
              const cr = cards[i].querySelector('.slot-pip-canvas').getBoundingClientRect();
              const ar = cards[i].getBoundingClientRect();
              canvasRect = { x: cr.x + (window.scrollX || 0), y: cr.y + (window.scrollY || 0), width: cr.width, height: cr.height };
              cardRect = { x: ar.x + (window.scrollX || 0), y: ar.y + (window.scrollY || 0), width: ar.width, height: ar.height };
            }
          }
        }
        return {
          ok: slotDimOk && activeSlots === 4 && pickSlot >= 0 && canvasRect && canvasRect.width >= 200 && canvasRect.height >= 100,
          slotDimOk, activeSlots, pickSlot, pickTid, canvasRect, cardRect,
        };
      })()`
    );
    // Verify compositor-rendered pixels inside the active slot's OffscreenCanvas via CDP clip screenshot
    const slotClipShot = await cdp.send(
      'Page.captureScreenshot',
      {
        format: 'png',
        captureBeyondViewport: true,
        clip: {
          x: Math.max(0, rosterState.canvasRect.x),
          y: Math.max(0, rosterState.canvasRect.y),
          width: rosterState.canvasRect.width,
          height: rosterState.canvasRect.height,
          scale: 1,
        },
      },
      sessionId
    );
    const slotPngBytes = slotClipShot?.data ? Buffer.from(slotClipShot.data, 'base64').byteLength : 0;
    if (slotPngBytes < 450) {
      throw new Error(`Expected active .slot-pip-canvas compositor screenshot >= 450 B, got ${slotPngBytes} B`);
    }

    const clickInfo = await cdp.evalJs(
      sessionId,
      `(() => {
        window.__e2eClsScore = 0;
        window.__e2eClsPo = new PerformanceObserver((list) => {
          for (const e of list.getEntries()) window.__e2eClsScore += e.value;
        });
        window.__e2eClsPo.observe({ type: 'layout-shift', buffered: false });

        const snap = (sel) => {
          const el = document.querySelector(sel);
          const r = el.getBoundingClientRect();
          return { x: r.x, y: r.y, width: r.width, height: r.height };
        };
        const anchors = {
          rosterPanel: snap('.roster-panel'),
          pipGrid: snap('.pip-kpi-grid'),
          rosterMeta: snap('#rosterSummaryMeta'),
          btnFocus: snap('#btnFocusCycle'),
          btnAudio: snap('#btnAudioToggle'),
          btnRec: snap('#btnRecToggle'),
          labDesc: snap('#labDescription'),
        };

        const cards = Array.from(document.querySelectorAll('.roster-card'));
        let idx = ${rosterState.pickSlot};
        for (let i = 0; i < cards.length; i++) {
          const st = cards[i].dataset.state || '';
          const tid = parseInt(cards[i].dataset.tid || '0', 10);
          const ageSec = parseFloat(cards[i].querySelector('.rs-val-age')?.textContent || '0') || 0;
          if (tid > 0 && (st === 'FOCUS' || st === 'LOCKED') && ageSec >= 0.8) {
            idx = i;
            break;
          }
        }
        const card = cards[idx];
        const neighborIdx = idx === 0 ? 1 : 0;
        const ar = card.getBoundingClientRect();
        const nr = cards[neighborIdx].getBoundingClientRect();
        const clickedTid = parseInt(card.dataset.tid || '0', 10);
        card.querySelector('.roster-pin-btn').click();
        return {
          idx,
          neighborIdx,
          clickedTid,
          anchors,
          rect: { x: ar.x, y: ar.y, width: ar.width, height: ar.height },
          neighborRect: { x: nr.x, y: nr.y, width: nr.width, height: nr.height },
        };
      })()`
    );
    const pinnedRoster = await waitForCondition(
      cdp,
      sessionId,
      'Roster card click pins focus target cleanly with 0.00px global layout shift',
      `(() => {
        const cards = document.querySelectorAll('.roster-card');
        const card = cards[${clickInfo.idx}];
        const neighbor = cards[${clickInfo.neighborIdx}];
        const focusTxt = document.getElementById('btnFocusCycle')?.textContent || '';
        const pinnedAttr = card?.dataset.pinned || '0';
        const btnPressed = card?.querySelector('.roster-pin-btn')?.getAttribute('aria-pressed') || 'false';
        const ar = card.getBoundingClientRect();
        const nr = neighbor.getBoundingClientRect();
        const sameRect = (a, b) =>
          Math.abs(a.x - b.x) < 0.05 &&
          Math.abs(a.y - b.y) < 0.05 &&
          Math.abs(a.width - b.width) < 0.05 &&
          Math.abs(a.height - b.height) < 0.05;
        const base = ${JSON.stringify(clickInfo.anchors)};
        const zeroShift =
          sameRect(ar, ${JSON.stringify(clickInfo.rect)}) &&
          sameRect(nr, ${JSON.stringify(clickInfo.neighborRect)}) &&
          sameRect(document.querySelector('.roster-panel').getBoundingClientRect(), base.rosterPanel) &&
          sameRect(document.querySelector('.pip-kpi-grid').getBoundingClientRect(), base.pipGrid) &&
          sameRect(document.getElementById('rosterSummaryMeta').getBoundingClientRect(), base.rosterMeta) &&
          sameRect(document.getElementById('btnFocusCycle').getBoundingClientRect(), base.btnFocus) &&
          sameRect(document.getElementById('btnAudioToggle').getBoundingClientRect(), base.btnAudio);
        return {
          ok: focusTxt === '◎ Focus: PIN T${clickInfo.clickedTid}' && pinnedAttr === '1' && btnPressed === 'true' && zeroShift,
          focusTxt, pinnedAttr, btnPressed, zeroShift,
        };
      })()`
    );
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.getElementById('btnAudioToggle').click();
        document.getElementById('btnAudioToggle').click();
        document.querySelector('.lab-btn[data-lab="1"]').click();
        document.querySelector('.lab-btn[data-lab="2"]').click();
        document.querySelector('.lab-btn[data-lab="0"]').click();
        document.getElementById('btnAudioToggle').click();
        document.querySelectorAll('.roster-card')[${clickInfo.idx}].querySelector('.roster-pin-btn').click();
        return true;
      })()`
    );
    const unpinnedZeroCls = await waitForCondition(
      cdp,
      sessionId,
      'Roster unpin + dynamic button/mode toggles preserve 0.00px global geometry & CLS = 0',
      `(() => {
        const focusTxt = document.getElementById('btnFocusCycle')?.textContent || '';
        const base = ${JSON.stringify(clickInfo.anchors)};
        const sameRect = (a, b) =>
          Math.abs(a.x - b.x) < 0.05 &&
          Math.abs(a.y - b.y) < 0.05 &&
          Math.abs(a.width - b.width) < 0.05 &&
          Math.abs(a.height - b.height) < 0.05;
        const geoLocked =
          sameRect(document.querySelector('.roster-panel').getBoundingClientRect(), base.rosterPanel) &&
          sameRect(document.querySelector('.pip-kpi-grid').getBoundingClientRect(), base.pipGrid) &&
          sameRect(document.getElementById('btnFocusCycle').getBoundingClientRect(), base.btnFocus) &&
          sameRect(document.getElementById('btnAudioToggle').getBoundingClientRect(), base.btnAudio) &&
          sameRect(document.getElementById('btnRecToggle').getBoundingClientRect(), base.btnRec) &&
          sameRect(document.getElementById('labDescription').getBoundingClientRect(), base.labDesc);
        const cls = window.__e2eClsScore || 0;
        if (focusTxt.startsWith('◎ Focus: AUTO') && window.__e2eClsPo) {
          window.__e2eClsPo.disconnect();
        }
        return { ok: focusTxt.startsWith('◎ Focus: AUTO') && geoLocked && cls < 0.0001, focusTxt, geoLocked, cls };
      })()`
    );
    console.log(`  [PASS] Step 2b (All-16 Split PiP Roster & Zero-Shift Gate): activeSlots=${rosterState.activeSlots}/16, slotPng=${slotPngBytes}B, pinned=${pinnedRoster.focusTxt} -> AUTO (CLS=${unpinnedZeroCls.cls.toFixed(6)}, 0.00px delta)`);

    // 3a. Juggling 25 fps MP4: Multi-arc Auto-g, exact 1/25s (+0.040s) +1f step, paused calibration sync, APERIODIC guard
    await cdp.evalJs(
      sessionId,
      `(() => {
        const sel = document.getElementById('sampleVideoSelect');
        sel.value = './samples/juggling_ballistic.mp4';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        if (${Boolean(process.env.CI)}) {
          const v = document.getElementById('hiddenVideoSource');
          if (v) {
            v.defaultPlaybackRate = 0.5;
            v.playbackRate = 0.5;
          }
        }
        return true;
      })()`
    );
    const mp4State = await waitForCondition(
      cdp,
      sessionId,
      'Juggling 25fps MP4 playback + multi-arc gravity + APERIODIC guard',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        if (${Boolean(process.env.CI)} && v && v.playbackRate !== 0.5) {
          v.defaultPlaybackRate = 0.5;
          v.playbackRate = 0.5;
        }
        const srcLabel = document.getElementById('hudSourceLabel')?.textContent || '';
        const targetsText = document.getElementById('kpiTargets')?.textContent || '0 / 16';
        const activeCount = parseInt(targetsText.split('/')[0], 10) || 0;
        const clutterText = document.getElementById('kpiClutterDb')?.textContent || '';
        const hzText = document.getElementById('kpiTremorHz')?.textContent || '';
        const gravLbl = document.getElementById('lblKinRestGrav')?.textContent || '';
        const gravVal = document.getElementById('kpiKinRestGrav')?.textContent || '';
        const arcMatch = gravLbl.match(/\((\d+)\s+arcs\)/) || gravLbl.match(/\\((\\d+)\\s+arcs\\)/);
        const arcCount = arcMatch ? parseInt(arcMatch[1], 10) : 0;
        const transportVisible = !document.getElementById('videoTransportBar')?.hidden;
        const simBtnDisabled = document.getElementById('btnLabStimulus')?.getAttribute('aria-disabled') === 'true';
        return {
          ok: srcLabel.includes('SAMPLE MP4') &&
              transportVisible &&
              simBtnDisabled &&
              v && !v.paused && v.currentTime > 1.25 && v.currentTime < 4.40 &&
              activeCount >= 1 &&
              arcCount >= 2 &&
              (hzText === 'APERIODIC' || hzText === 'HOLD') &&
              gravVal.endsWith('px/s²'),
          srcLabel,
          currentTime: v ? Number(v.currentTime.toFixed(3)) : 0,
          activeCount,
          arcCount,
          clutterText,
          hzText,
          gravLbl,
          gravVal,
        };
      })()`,
      18000
    );
    await cdp.saveScreenshot(sessionId, proofDir, 'proof_mode_a_juggling.png');

    // Pause first so rapid juggling catches cannot expire the pinned track mid-click, ensure >=1 active target, pin [FOCUS], step +1f (+0.040s), and verify paused Auto-g KPI sync
    await cdp.evalJs(sessionId, `document.getElementById('btnVidPlayPause').click(); true;`);
    await waitForCondition(cdp, sessionId, 'MP4 paused', `(() => ({ ok: document.getElementById('hiddenVideoSource').paused }))()`);
    for (let guard = 0; guard < 10; guard++) {
      const curTargets = await cdp.evalJs(
        sessionId,
        `parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0`
      );
      if (curTargets >= 1) break;
      await cdp.evalJs(sessionId, `document.getElementById('btnVidStepFwd').click(); true;`);
      await new Promise((r) => setTimeout(r, 120));
    }
    await cdp.evalJs(sessionId, `document.getElementById('btnFocusCycle').click(); true;`);
    const pinState = await waitForCondition(
      cdp,
      sessionId,
      'Paused Focus target pin (PIN T#)',
      `(() => {
        const txt = document.getElementById('btnFocusCycle')?.textContent || '';
        return { ok: txt.includes('PIN T'), txt };
      })()`
    );
    const beforeStepT = await cdp.evalJs(sessionId, `document.getElementById('hiddenVideoSource').currentTime`);
    await cdp.evalJs(sessionId, `document.getElementById('btnVidStepFwd').click(); true;`);
    const stepState = await waitForCondition(
      cdp,
      sessionId,
      'Exact 25fps (+0.040s) +1f MP4 step',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const dt = v ? (v.currentTime - ${beforeStepT}) : 0;
        return { ok: v && v.paused && dt > 0.024 && dt < 0.062, t: v ? v.currentTime : 0, dt };
      })()`
    );
    // While STILL paused, apply Auto-g calibration and verify DOM KPI immediately updates from px/s to m/s!
    await cdp.evalJs(sessionId, `document.getElementById('btnCalAutoG').click(); true;`);
    const pausedAutoG = await waitForCondition(
      cdp,
      sessionId,
      'Paused Auto-g calibration KPI sync (m/s)',
      `(() => {
        const badge = document.getElementById('calBadge')?.textContent || '';
        const speedTxt = document.getElementById('kpiKinSpeed')?.textContent || '';
        return { ok: badge === 'AUTO-g LOCKED' && speedTxt.endsWith('m/s'), badge, speedTxt };
      })()`
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnCalClear').click(); document.getElementById('btnVidPlayPause').click(); true;`);
    console.log(`  [PASS] Step 3a (25fps Juggling + Exact +1f=${(stepState.dt * 1000).toFixed(1)}ms + Paused Auto-g): ${mp4State.gravLbl}=${mp4State.gravVal}, ${pausedAutoG.badge}->${pausedAutoG.speedTxt}, ${pinState.txt}`);

    // 3b. Highway Gantry Bridges (0 false pivots/Hz), Nikko 0.80 Hz Metronome, Galileo Pivot, and Carom Mode C
    await cdp.evalJs(
      sessionId,
      `(() => {
        const sel = document.getElementById('sampleVideoSelect');
        sel.value = './samples/roundabout_occlusion.mp4';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        document.getElementById('hiddenVideoSource').currentTime = 1.6;
        return true;
      })()`
    );
    const gantryState = await waitForCondition(
      cdp,
      sessionId,
      'Highway gantry MP4 Hermite bridges + 0 false pivot/Hz',
      `(() => {
        const bridges = parseInt(document.getElementById('kpiCollisions')?.textContent || '0', 10) || 0;
        const activeCount = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const hzTxt = document.getElementById('kpiTremorHz')?.textContent || '';
        const angLbl = document.getElementById('lblKinAngle')?.textContent || '';
        return {
          ok: activeCount >= 3 &&
              bridges >= 1 &&
              (hzTxt === 'APERIODIC' || hzTxt === 'HOLD') &&
              angLbl !== 'Pendulum Swing θ',
          activeCount, bridges, hzTxt, angLbl,
        };
      })()`,
      15000
    );
    await cdp.saveScreenshot(sessionId, proofDir, 'proof_mode_a_gantry_bridge.png');

    await cdp.evalJs(
      sessionId,
      `(() => {
        const sel = document.getElementById('sampleVideoSelect');
        sel.value = './samples/harmonic_metronome_phase.mp4';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()`
    );
    const metroState = await waitForCondition(
      cdp,
      sessionId,
      'Nikko 96 BPM metronome Mode B true ~0.80 Hz readout',
      `(() => {
        const badge = document.getElementById('labBadge')?.textContent || '';
        const meta = document.getElementById('hilbertMeta')?.textContent || '';
        const hzTxt = document.getElementById('kpiTremorHz')?.textContent || '';
        const hzVal = parseFloat(hzTxt);
        return {
          ok: badge === 'MODE B' &&
              meta.includes('COH') &&
              hzTxt.endsWith(' Hz') &&
              Number.isFinite(hzVal) &&
              hzVal >= 0.68 &&
              hzVal <= 0.92,
          badge, meta, hzTxt, hzVal,
        };
      })()`,
      18000
    );
    await cdp.saveScreenshot(sessionId, proofDir, 'proof_mode_b_metronome.png');

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
      'Galileo clip loaded & playing',
      `(() => ({ ok: (document.getElementById('hudSourceLabel')?.textContent || '').includes('GALILEO') }))()`
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnCalPreset').click(); true;`);
    const galileoState = await waitForCondition(
      cdp,
      sessionId,
      'Galileo pendulum Kåsa virtual pivot & swing angle θ',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const srcLabel = document.getElementById('hudSourceLabel')?.textContent || '';
        const activeCount = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const lbl = document.getElementById('lblKinAngle')?.textContent || '';
        const val = document.getElementById('kpiKinAngle')?.textContent || '';
        const radVal = document.getElementById('kpiKinDeflect')?.textContent || '';
        return {
          ok: srcLabel.includes('GALILEO') &&
              v && !v.paused && v.currentTime > 1.35 &&
              activeCount >= 1 &&
              lbl === 'Pendulum Swing θ' &&
              val.endsWith('°') &&
              radVal.endsWith(' m') &&
              !val.includes('NaN') &&
              !val.includes('--'),
          lbl,
          val,
          radVal,
        };
      })()`,
      15000
    );
    await cdp.saveScreenshot(sessionId, proofDir, 'proof_galileo_peg_pivot.png');

    await cdp.evalJs(
      sessionId,
      `(() => {
        const sel = document.getElementById('sampleVideoSelect');
        sel.value = './samples/carom_billiards_impulses.mp4';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()`
    );
    const caromState = await waitForCondition(
      cdp,
      sessionId,
      'Carom billiards Mode C L1 impact strikes + APERIODIC guard',
      `(() => {
        const badge = document.getElementById('labBadge')?.textContent || '';
        const strikes = parseInt(document.getElementById('kpiCollisions')?.textContent || '0', 10) || 0;
        const hzTxt = document.getElementById('kpiTremorHz')?.textContent || '';
        return {
          ok: badge === 'MODE C' && strikes >= 1 && (hzTxt === 'APERIODIC' || hzTxt === 'HOLD'),
          badge, strikes, hzTxt,
        };
      })()`,
      15000
    );

    // 3c. Calibration Provenance (ASSUMED vs SURVEYED) + Atomic Escape Preservation
    await cdp.evalJs(sessionId, `document.getElementById('btnCalPreset').click(); true;`);
    const calPresetState = await waitForCondition(
      cdp,
      sessionId,
      'Carom Assumed 4-Pt Homography preset + Assumed Speed KPI',
      `(() => {
        const badge = document.getElementById('calBadge')?.textContent || '';
        const quadPressed = document.getElementById('btnCalQuad')?.getAttribute('aria-pressed') || '';
        const speedLbl = document.getElementById('lblKinSpeed')?.textContent || '';
        const speedTxt = document.getElementById('kpiKinSpeed')?.textContent || '';
        const angTxt = document.getElementById('kpiKinAngle')?.textContent || '';
        const deflTxt = document.getElementById('kpiKinDeflect')?.textContent || '';
        const restTxt = document.getElementById('kpiKinRestGrav')?.textContent || '';
        const noNan = ![speedTxt, angTxt, deflTxt, restTxt].some((s) => s.includes('NaN'));
        return {
          ok: badge === 'ASSUMED 4-PT H' &&
              quadPressed === 'true' &&
              speedLbl === 'Assumed Speed' &&
              speedTxt.endsWith(' m/s') &&
              noNan,
          badge, speedLbl, speedTxt, angTxt, deflTxt, restTxt,
        };
      })()`
    );
    await cdp.saveScreenshot(sessionId, proofDir, 'proof_mode_c_billiards.png');
    if (process.env.CI) {
      await cdp.evalJs(
        sessionId,
        `(() => {
          const v = document.getElementById('hiddenVideoSource');
          if (v) {
            v.defaultPlaybackRate = 1.0;
            v.playbackRate = 1.0;
          }
          return true;
        })()`
      );
    }

    // Arm 2-Pt Ruler while ASSUMED 4-PT H is active, then press Escape -> prior calibration MUST survive!
    await cdp.evalJs(sessionId, `document.getElementById('btnCalRuler').click(); true;`);
    await waitForCondition(
      cdp,
      sessionId,
      '2-Pt Ruler armed without wiping active calibration',
      `(() => ({
        ok: document.getElementById('calToolbar')?.getAttribute('role') === 'toolbar' &&
            document.getElementById('btnCalRuler')?.getAttribute('aria-pressed') === 'true' &&
            document.getElementById('calBadge')?.textContent === 'ASSUMED 4-PT H'
      }))()`
    );
    await cdp.evalJs(
      sessionId,
      `document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); true;`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Escape cancels armed 2-Pt Ruler while preserving ASSUMED 4-PT H',
      `(() => ({
        ok: document.getElementById('btnCalRuler')?.getAttribute('aria-pressed') === 'false' &&
            document.getElementById('btnCalQuad')?.getAttribute('aria-pressed') === 'true' &&
            document.getElementById('calBadge')?.textContent === 'ASSUMED 4-PT H'
      }))()`
    );

    // Commit accessible keyboard normalized coordinates -> SURVEYED RULER provenance!
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.getElementById('calKeyboardDisclosure').open = true;
        document.getElementById('calCoordMode').value = 'ruler';
        document.getElementById('calCoordMode').dispatchEvent(new Event('change', { bubbles: true }));
        document.getElementById('coordX1').value = '0.15';
        document.getElementById('coordY1').value = '0.50';
        document.getElementById('coordX2').value = '0.85';
        document.getElementById('coordY2').value = '0.50';
        document.getElementById('calCoordForm').requestSubmit();
        return true;
      })()`
    );
    const surveyedState = await waitForCondition(
      cdp,
      sessionId,
      'Keyboard normalized coordinates commit SURVEYED RULER provenance',
      `(() => {
        const badge = document.getElementById('calBadge')?.textContent || '';
        const speedLbl = document.getElementById('lblKinSpeed')?.textContent || '';
        return { ok: badge === 'SURVEYED RULER' && speedLbl === 'Surveyed Speed', badge, speedLbl };
      })()`
    );
    // Re-apply Assumed 4-Pt Homography for 56-col v1.5.0 telemetry session recording
    await cdp.evalJs(sessionId, `document.getElementById('btnCalPreset').click(); true;`);
    await waitForCondition(cdp, sessionId, 'Re-applied ASSUMED 4-PT H', `(() => ({ ok: document.getElementById('calBadge')?.textContent === 'ASSUMED 4-PT H' }))()`);
    console.log(`  [PASS] Step 3b/c (Nikko=${metroState.hzTxt}, Galileo=${galileoState.val}, Carom=${caromState.strikes}, Provenance=${calPresetState.badge}/${surveyedState.badge} + Esc atomic)`);

    // 4. Session Telemetry Recorder (56-col schemaVersion 1.5.0 CSV & JSON + Disk Download)
    await cdp.evalJs(sessionId, `document.getElementById('btnRecToggle').click(); true;`);
    await waitForCondition(
      cdp,
      sessionId,
      'Session recorder >= 16 frames',
      `(() => {
        const txt = document.getElementById('recStatusLabel')?.textContent || '';
        const m = txt.match(/REC\\s+(\\d+)\\s+FRAMES/);
        const count = m ? parseInt(m[1], 10) : 0;
        return { ok: count >= 16, txt, count };
      })()`
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnRecToggle').click(); true;`);

    await cdp.evalJs(sessionId, `window.__lastTelemetryExport = null; document.getElementById('btnExportCsv').click(); true;`);
    const csvExport = await waitForCondition(
      cdp,
      sessionId,
      '56-column CSV telemetry export payload',
      `(() => {
        const ex = window.__lastTelemetryExport;
        if (!ex || !ex.filename.endsWith('.csv')) return { ok: false };
        const lines = ex.fullText.trim().split('\\n');
        const headerCols = lines[0].split(',');
        const firstRow = lines[1] ? lines[1].split(',').map(Number) : [];
        const allFinite = lines.slice(1).every((l) => l.split(',').every((v) => Number.isFinite(Number(v))));
        return {
          ok: headerCols.length === 56 &&
              headerCols.includes('heading_deg') &&
              headerCols.includes('metric_x_m') &&
              headerCols.includes('cal_segment_id') &&
              headerCols.includes('metric_valid') &&
              headerCols.includes('phys_regime') &&
              headerCols.includes('phys_regime_conf') &&
              headerCols.includes('ode_r2') &&
              headerCols.includes('aero_drag_beta') &&
              headerCols.includes('thrust_accel') &&
              headerCols.includes('damping_zeta') &&
              headerCols.includes('kuramoto_order_r') &&
              headerCols.includes('lyapunov_div_rate') &&
              lines.length >= 17 &&
              allFinite &&
              firstRow[2] === 640 &&
              firstRow[3] === 360,
          filename: ex.filename,
          cols: headerCols.length,
          rows: lines.length - 1,
        };
      })()`
    );

    await cdp.evalJs(sessionId, `window.__lastTelemetryExport = null; document.getElementById('btnExportJson').click(); true;`);
    const jsonExport = await waitForCondition(
      cdp,
      sessionId,
      '56-column v1.5.0 JSON telemetry export payload',
      `(() => {
        const ex = window.__lastTelemetryExport;
        if (!ex || !ex.filename.endsWith('.json')) return { ok: false };
        const parsed = JSON.parse(ex.fullText);
        const hasMetric = parsed.frames.some((f) => Math.abs(f.metric_x_m) > 0.01 || Math.abs(f.metric_y_m) > 0.01);
        return {
          ok: parsed.columns.length === 56 &&
              parsed.columns.includes('phys_regime') &&
              parsed.columns.includes('phys_regime_conf') &&
              parsed.columns.includes('ode_r2') &&
              parsed.columns.includes('aero_drag_beta') &&
              parsed.columns.includes('thrust_accel') &&
              parsed.columns.includes('damping_zeta') &&
              parsed.columns.includes('kuramoto_order_r') &&
              parsed.columns.includes('lyapunov_div_rate') &&
              parsed.meta.schemaVersion === '1.5.0' &&
              parsed.meta.calibration?.mode === 'homography' &&
              parsed.meta.calibration?.provenance === 'assumed' &&
              parsed.meta.calibration?.homographyValid === true &&
              Array.isArray(parsed.meta.calibrationSegments) &&
              parsed.meta.calibrationSegments.length >= 1 &&
              hasMetric &&
              parsed.frameCount >= 16 &&
              parsed.frames.length === parsed.frameCount &&
              parsed.meta.resMode === 1 &&
              parsed.meta.frameW === 640 &&
              parsed.meta.frameH === 360 &&
              parsed.meta.pipDim === 64,
          filename: ex.filename,
          frameCount: parsed.frameCount,
          segCount: parsed.meta.calibrationSegments?.length || 0,
        };
      })()`
    );

    const deadlineDl = Date.now() + 5000;
    let downloadedCsv = null;
    let downloadedJson = null;
    while (Date.now() < deadlineDl) {
      const files = fs.readdirSync(downloadDir).filter((f) => !f.endsWith('.crdownload'));
      downloadedCsv = files.find((f) => f.endsWith('.csv')) || null;
      downloadedJson = files.find((f) => f.endsWith('.json')) || null;
      if (downloadedCsv && downloadedJson) break;
      await new Promise((r) => setTimeout(r, 80));
    }
    if (!downloadedCsv || !downloadedJson) {
      throw new Error(`Expected .csv and .json files on disk in ${downloadDir}, found: ${fs.readdirSync(downloadDir).join(', ')}`);
    }
    const diskCsvLines = fs.readFileSync(path.join(downloadDir, downloadedCsv), 'utf8').trim().split('\n');
    const diskJsonObj = JSON.parse(fs.readFileSync(path.join(downloadDir, downloadedJson), 'utf8'));
    if (diskCsvLines.length - 1 !== csvExport.rows || diskJsonObj.frameCount !== jsonExport.frameCount) {
      throw new Error('Downloaded disk file row counts did not match exported telemetry payload');
    }
    console.log(`  [PASS] Step 4 (v1.5.0 56-Col Export & Disk Download): CSV (${csvExport.rows}x${csvExport.cols}) & JSON (${jsonExport.frameCount}f, calSegments=${jsonExport.segCount})`);

    // 4b. Extended 29s-80s HD Showcase Buttons & Grouped <select> Verification
    const selectMeta = await cdp.evalJs(
      sessionId,
      `(() => {
        const sel = document.getElementById('sampleVideoSelect');
        return {
          optgroups: sel ? sel.querySelectorAll('optgroup').length : 0,
          options: sel ? sel.querySelectorAll('option').length : 0,
        };
      })()`
    );
    if (selectMeta.optgroups !== 3 || selectMeta.options !== 15) {
      throw new Error(`Expected 3 <optgroup> and 15 <option> elements in #sampleVideoSelect, got ${JSON.stringify(selectMeta)}`);
    }

    await cdp.evalJs(sessionId, `document.querySelector('.showcase-btn[data-showcase="model_rockets"]')?.click(); true;`);
    const rocketClipState = await waitForCondition(
      cdp,
      sessionId,
      'Extended 31s Model Rockets full-flight showcase active + ASSUMED RULER',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const btn = document.querySelector('.showcase-btn[data-showcase="model_rockets"]');
        const activeCount = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const badge = document.getElementById('calBadge')?.textContent || '';
        const speedTxt = document.getElementById('kpiKinSpeed')?.textContent || '';
        const angTxt = document.getElementById('kpiKinAngle')?.textContent || '';
        const noNan = ![speedTxt, angTxt].some((s) => s.includes('NaN'));
        return {
          ok: v && Number.isFinite(v.duration) && v.duration >= 29.5 && !v.paused &&
              btn?.getAttribute('aria-pressed') === 'true' &&
              activeCount >= 1 &&
              badge === 'ASSUMED RULER' &&
              noNan,
          duration: v ? Number(v.duration.toFixed(2)) : 0,
          activeCount,
          badge,
        };
      })()`,
      22000
    );

    await cdp.evalJs(sessionId, `document.querySelector('.showcase-btn[data-showcase="rocket_vtvl"]')?.click(); true;`);
    const rocketVtvlState = await waitForCondition(
      cdp,
      sessionId,
      'Extended 80s VTVL Rocket Pad-to-Pad showcase active + ASSUMED RULER + m/s',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const btn = document.querySelector('.showcase-btn[data-showcase="rocket_vtvl"]');
        const activeCount = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const badge = document.getElementById('calBadge')?.textContent || '';
        const speedLbl = document.getElementById('lblKinSpeed')?.textContent || '';
        const speedTxt = document.getElementById('kpiKinSpeed')?.textContent || '';
        const angTxt = document.getElementById('kpiKinAngle')?.textContent || '';
        const noNan = ![speedTxt, angTxt].some((s) => s.includes('NaN'));
        return {
          ok: v && Number.isFinite(v.duration) && v.duration >= 78.0 && !v.paused &&
              btn?.getAttribute('aria-pressed') === 'true' &&
              activeCount >= 1 &&
              badge === 'ASSUMED RULER' &&
              speedLbl === 'Assumed Speed' &&
              speedTxt.endsWith(' m/s') &&
              noNan,
          duration: v ? Number(v.duration.toFixed(2)) : 0,
          activeCount,
          badge,
          speedTxt,
        };
      })()`,
      22000
    );

    await cdp.evalJs(sessionId, `document.querySelector('.showcase-btn[data-showcase="cologne_motorway"]')?.click(); true;`);
    const motorwayState = await waitForCondition(
      cdp,
      sessionId,
      'Extended 45s A4 Cologne Motorway showcase active + ASSUMED 4-PT H',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const btn = document.querySelector('.showcase-btn[data-showcase="cologne_motorway"]');
        const activeCount = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const badge = document.getElementById('calBadge')?.textContent || '';
        const speedTxt = document.getElementById('kpiKinSpeed')?.textContent || '';
        const angTxt = document.getElementById('kpiKinAngle')?.textContent || '';
        const noNan = ![speedTxt, angTxt].some((s) => s.includes('NaN'));
        return {
          ok: v && Number.isFinite(v.duration) && v.duration >= 40.0 && !v.paused &&
              btn?.getAttribute('aria-pressed') === 'true' &&
              activeCount >= 3 &&
              badge === 'ASSUMED 4-PT H' &&
              noNan,
          duration: v ? Number(v.duration.toFixed(2)) : 0,
          activeCount,
          badge,
        };
      })()`,
      18000
    );

    await cdp.evalJs(sessionId, `document.querySelector('.showcase-btn[data-showcase="rally_pan"]')?.click(); true;`);
    const rallyState = await waitForCondition(
      cdp,
      sessionId,
      'Extended 29s Rally Ego-Pan showcase active + PX UNITS',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const btn = document.querySelector('.showcase-btn[data-showcase="rally_pan"]');
        const activeCount = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const badge = document.getElementById('calBadge')?.textContent || '';
        const speedTxt = document.getElementById('kpiKinSpeed')?.textContent || '';
        const angTxt = document.getElementById('kpiKinAngle')?.textContent || '';
        const noNan = ![speedTxt, angTxt].some((s) => s.includes('NaN'));
        return {
          ok: v && Number.isFinite(v.duration) && v.duration >= 28.0 && !v.paused &&
              btn?.getAttribute('aria-pressed') === 'true' &&
              activeCount >= 1 &&
              badge === 'PX UNITS' &&
              noNan,
          duration: v ? Number(v.duration.toFixed(2)) : 0,
          activeCount,
          badge,
        };
      })()`,
      18000
    );
    console.log(`  [PASS] Step 4b (Extended 29s-80s HD Showcases & Grouped Selector): ${selectMeta.optgroups} groups/${selectMeta.options} opts, rockets=${rocketClipState.duration}s (${rocketClipState.activeCount} tgts, ${rocketClipState.badge}), vtvl=${rocketVtvlState.duration}s (${rocketVtvlState.activeCount} tgts, ${rocketVtvlState.speedTxt}), cologne=${motorwayState.duration}s (${motorwayState.activeCount} tgts, ${motorwayState.badge}), rally=${rallyState.duration}s (${rallyState.activeCount} tgts, ${rallyState.badge})`);

    await cdp.evalJs(sessionId, `(() => {
      document.getElementById('btnSrcTestbench')?.click();
      const sel = document.getElementById('benchScenarioSelect');
      if (sel) {
        sel.value = 'model_rocket_flight';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
      }
      return true;
    })()`);
    const rocketBenchState = await waitForCondition(
      cdp,
      sessionId,
      'Synthetic model_rocket_flight Mode A boost, coast & apogee active',
      `(() => {
        const activeCount = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const apexTxt = document.getElementById('kpiApexMs')?.textContent || '';
        const apexNum = parseFloat(apexTxt);
        const swapsTxt = document.getElementById('kpiPadicSwaps')?.textContent || '';
        const swapsNum = parseInt(swapsTxt.replace(/[^\\d-]/g, ''), 10);
        return {
          ok: activeCount >= 3 && activeCount <= 4 && Number.isFinite(apexNum) && swapsNum === 0,
          activeCount,
          apexTxt,
          swapsTxt,
        };
      })()`,
      18000
    );
    await cdp.evalJs(sessionId, `(() => {
      const sel = document.getElementById('benchScenarioSelect');
      if (sel) {
        sel.value = 'pillar_crossing';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
      }
      return true;
    })()`);
    console.log(`  [PASS] Step 4c (Model Rocket Flight Scenario): tgts=${rocketBenchState.activeCount}, T-Apex=${rocketBenchState.apexTxt}, swaps=${rocketBenchState.swapsTxt}`);

    // 5. 3-Tier Live Stream Suite: Open <dialog>, verify /api/live-presets, stream Tier 3 fMP4 MSE across hard scene cut!
    await cdp.evalJs(sessionId, `document.getElementById('btnSrcLiveStream').click(); true;`);
    const dialogState = await waitForCondition(
      cdp,
      sessionId,
      'Live stream <dialog> open + /api/live-presets populated',
      `(() => {
        const dlg = document.getElementById('liveStreamDialog');
        const sel = document.getElementById('tier3PresetSelect');
        const opts = sel ? Array.from(sel.options).map((o) => o.value) : [];
        return {
          ok: Boolean(dlg && dlg.open) && opts.includes('sim_live_cut_demo') && opts.length >= 3,
          optCount: opts.length,
        };
      })()`
    );

    // Connect Tier 3 fMP4 -> MSE live stream (`sim_live_cut_demo`: 1.6s highway -> hard cut -> juggling)
    await cdp.evalJs(
      sessionId,
      `(() => {
        const sel = document.getElementById('tier3PresetSelect');
        sel.value = 'sim_live_cut_demo';
        document.getElementById('tier3CustomHlsUrl').value = '';
        document.getElementById('btnStartTier3Mse').click();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Tier 3 fMP4 MSE stream connected & playing',
      `(() => {
        const dlg = document.getElementById('liveStreamDialog');
        const barVisible = !document.getElementById('liveStreamBar')?.hidden;
        const badge = document.getElementById('liveTierBadge')?.textContent || '';
        const v = document.getElementById('hiddenVideoSource');
        return {
          ok: Boolean(dlg && !dlg.open) && barVisible && badge.includes('TIER 3') && v && !v.paused && v.currentTime > 0.15,
          badge,
          t: v ? Number(v.currentTime.toFixed(2)) : 0,
        };
      })()`,
      15000
    );

    // Arm a surveyed ruler calibration BEFORE the 1.6s hard camera cut, then verify the cut auto-clears calibration to PX UNITS!
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.getElementById('calKeyboardDisclosure').open = true;
        document.getElementById('calCoordForm').requestSubmit();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Pre-cut SURVEYED RULER active on live stream',
      `(() => ({ ok: document.getElementById('calBadge')?.textContent === 'SURVEYED RULER' }))()`,
      5000
    );

    const liveCutState = await waitForCondition(
      cdp,
      sessionId,
      'Live MSE stream crosses 1.6s scene cut -> auto-resets calibration to PX UNITS & locks post-cut targets',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const calBadge = document.getElementById('calBadge')?.textContent || '';
        const activeCount = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
        const srcLbl = document.getElementById('hudSourceLabel')?.textContent || '';
        return {
          ok: v && v.currentTime >= 2.35 && calBadge === 'PX UNITS' && activeCount >= 1 && srcLbl.includes('LIVE fMP4 MSE'),
          t: v ? Number(v.currentTime.toFixed(2)) : 0,
          calBadge,
          activeCount,
          srcLbl,
        };
      })()`,
      18000
    );
    await cdp.saveScreenshot(sessionId, proofDir, 'proof_live_mse_stream.png');
    await cdp.evalJs(sessionId, `document.getElementById('btnStopLiveFeed').click(); true;`);
    console.log(`  [PASS] Step 5 (Live Tier-3 fMP4 MSE Stream + Hard-Cut Auto-Reset): presets=${dialogState.optCount}, t=${liveCutState.t}s, postCutCal=${liveCutState.calBadge}, targets=${liveCutState.activeCount}`);

    // 6. Testbench Scenarios + Lab Modes B & C + Return to 320x180 SD Cleanly
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.getElementById('btnSrcTestbench').click();
        const scSel = document.getElementById('benchScenarioSelect');
        scSel.value = 'short_gap_bridge';
        scSel.dispatchEvent(new Event('change', { bubbles: true }));
        document.querySelector('.lab-btn[data-lab="0"]').click();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Short-gap Hermite bridge scenario lock (4 targets, 0 swaps)',
      `(() => ({
        ok: !document.getElementById('benchScenarioField')?.hidden &&
            document.getElementById('videoTransportBar')?.hidden &&
            document.getElementById('liveStreamBar')?.hidden &&
            (document.getElementById('kpiTargets')?.textContent || '') === '4 / 16' &&
            (document.getElementById('kpiPadicSwaps')?.textContent || '').trim() === '◆ 0'
      }))()`
    );

    // 6a. Multi-Target Swarm Scenario (>=9 / 16 simultaneous active locks) & Multi-Minute Continuity Panel
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__e2eSwarmCls = 0;
        window.__e2eSwarmClsPo = new PerformanceObserver((list) => {
          for (const e of list.getEntries()) window.__e2eSwarmCls += e.value;
        });
        window.__e2eSwarmClsPo.observe({ type: 'layout-shift', buffered: false });
        const scSel = document.getElementById('benchScenarioSelect');
        scSel.value = 'multi_target_swarm';
        scSel.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()`
    );
    const swarmState = await waitForCondition(
      cdp,
      sessionId,
      '12-disc multi_target_swarm locks >= 9 / 16 targets and populates calm #persistencePanel',
      `(() => {
        const targetsTxt = document.getElementById('kpiTargets')?.textContent || '0 / 16';
        const activeCount = parseInt(targetsTxt.split('/')[0], 10) || 0;
        const peakTxt = document.getElementById('kpiPersistActivePeak')?.textContent || '0 / 16';
        const peakCount = parseInt(peakTxt.split('/')[0], 10) || 0;
        const uniqueCount = parseInt(document.getElementById('kpiPersistUnique')?.textContent || '0', 10) || 0;
        const spanTxt = (document.getElementById('kpiPersistSpan')?.textContent || '').trim();
        const panelEl = document.getElementById('persistencePanel');
        const pCanvas = document.getElementById('persistenceCanvas');
        const cr = pCanvas ? pCanvas.getBoundingClientRect() : null;
        const panelVisible = panelEl && getComputedStyle(panelEl).display !== 'none' && panelEl.offsetParent !== null;
        return {
          ok: targetsTxt.endsWith('/ 16') &&
              activeCount >= 9 &&
              peakCount >= 9 &&
              uniqueCount >= 9 &&
              Boolean(spanTxt) && !spanTxt.startsWith('--') &&
              Boolean(panelVisible) &&
              pCanvas?.width === 420 && pCanvas?.height === 204 &&
              cr && cr.width >= 200 && cr.height >= 100,
          targetsTxt, activeCount, peakTxt, peakCount, uniqueCount, spanTxt,
          pRect: cr ? { x: cr.x + (window.scrollX || 0), y: cr.y + (window.scrollY || 0), width: cr.width, height: cr.height } : null,
        };
      })()`,
      15000
    );
    await cdp.evalJs(
      sessionId,
      `document.getElementById('persistenceCanvas')?.scrollIntoView({ block: 'center', inline: 'center' }); true;`
    );
    await new Promise((r) => setTimeout(r, 120));
    const persistRect = await cdp.evalJs(
      sessionId,
      `(() => {
        const r = document.getElementById('persistenceCanvas')?.getBoundingClientRect();
        return r ? { x: Math.max(0, Math.floor(r.left + (window.scrollX || 0))), y: Math.max(0, Math.floor(r.top + (window.scrollY || 0))), width: Math.ceil(r.width), height: Math.ceil(r.height) } : null;
      })()`
    );
    const persistClipShot = await cdp.send(
      'Page.captureScreenshot',
      {
        format: 'png',
        captureBeyondViewport: false,
        clip: {
          x: persistRect.x,
          y: persistRect.y,
          width: persistRect.width,
          height: persistRect.height,
          scale: 1,
        },
      },
      sessionId
    );
    const persistPngBytes = persistClipShot?.data ? Buffer.from(persistClipShot.data, 'base64').byteLength : 0;
    if (persistPngBytes < 600) {
      throw new Error(`Step 6a: Expected non-black #persistenceCanvas screenshot >= 600 B, got ${persistPngBytes} B`);
    }
    const swarmCls = await cdp.evalJs(
      sessionId,
      `(() => {
        if (window.__e2eSwarmClsPo) window.__e2eSwarmClsPo.disconnect();
        return window.__e2eSwarmCls || 0;
      })()`
    );
    if (swarmCls >= 0.0001) {
      throw new Error(`Step 6a: Expected 0.000000 CLS during multi_target_swarm & calm panel updates, got ${swarmCls}`);
    }
    console.log(`  [PASS] Step 6a (12-Disc Swarm >=9/16 Locks & Multi-Minute Continuity Panel): targets=${swarmState.targetsTxt}, peak=${swarmState.peakTxt}, unique=${swarmState.uniqueCount}, span=${swarmState.spanTxt}, persistPng=${persistPngBytes}B, CLS=${swarmCls.toFixed(6)}`);

    // 6b. In-Situ Physics ID Panel, OffscreenCanvas Phase/Conservation Portrait & Zero-CLS Audit
    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__e2ePhysCls = 0;
        if (window.__e2ePhysClsPo) window.__e2ePhysClsPo.disconnect();
        if (typeof PerformanceObserver === 'function') {
          window.__e2ePhysClsPo = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              if (!entry.hadRecentInput) window.__e2ePhysCls += entry.value;
            }
          });
          window.__e2ePhysClsPo.observe({ type: 'layout-shift', buffered: false });
        }
        document.querySelector('.showcase-btn[data-showcase="chaos_pendulum"]')?.click();
        return true;
      })()`
    );
    const physUiState = await waitForCondition(
      cdp,
      sessionId,
      'In-Situ Physics ID Panel & #physicsPhaseCanvas populated cleanly during Chaotic Magnetic Pendulum',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const btn = document.querySelector('.showcase-btn[data-showcase="chaos_pendulum"]');
        const panel = document.getElementById('physicsIdPanel');
        const badge = document.getElementById('physRegimeBadge');
        const jumpBadge = document.getElementById('physJumpBadge');
        const phaseCanvas = document.getElementById('physicsPhaseCanvas');
        if (!v || !btn || !panel || !badge || !jumpBadge || !phaseCanvas) return { ok: false };

        const fieldIds = [
          'physicsIdMeta',
          'physRegimeBadge',
          'physJumpBadge',
          'physicsOdeEq',
          'kpiPhysicsR2',
          'kpiPhysicsConf',
          'kpiImmWeights',
          'kpiPhysicsBeta',
          'kpiPhysicsThrust',
          'kpiPhysicsDamping',
          'kpiPhysicsEllipticT',
          'kpiKuramotoSync',
          'kpiLyapunovMax',
          'kpiTaubinDelta',
          'kpiTaubinKappa',
          'kpiFistaKnots',
          'kpiTwoBodyMassE',
          'kpiTwoBodyMomRes',
          'kpiPhysicsConserv',
        ];
        const values = {};
        for (const id of fieldIds) {
          const el = document.getElementById(id);
          if (!el) return { ok: false, missing: id };
          const txt = (el.textContent || '').trim();
          values[id] = txt;
          if (!txt || /\\bNaN\\b|Infinity|undefined/i.test(txt)) {
            return { ok: false, badId: id, badTxt: txt };
          }
        }

        const panelRect = panel.getBoundingClientRect();
        const cRect = phaseCanvas.getBoundingClientRect();
        const r2Val = parseFloat(values.kpiPhysicsR2.replace('%', ''));
        const active = badge.getAttribute('data-active') === '1';
        const regCode = badge.getAttribute('data-regime') || '0';
        const videoPlaying = Number.isFinite(v.duration) && v.duration >= 40.0 && !v.paused && v.currentTime >= 1.2;
        const ok =
          videoPlaying &&
          btn.getAttribute('aria-pressed') === 'true' &&
          panelRect.width >= 240 &&
          panelRect.height >= 120 &&
          cRect.width >= 200 &&
          cRect.height >= 90 &&
          phaseCanvas.width === 420 &&
          phaseCanvas.height === 210 &&
          active &&
          Number.isFinite(r2Val) &&
          r2Val > 0;
        return {
          ok,
          metaTxt: values.physicsIdMeta,
          regimeTxt: values.physRegimeBadge,
          regCode,
          odeEq: values.physicsOdeEq,
          r2Txt: values.kpiPhysicsR2,
          confTxt: values.kpiPhysicsConf,
          immTxt: values.kpiImmWeights,
          lyapTxt: values.kpiLyapunovMax,
          cRect: {
            x: Math.floor(cRect.left),
            y: Math.floor(cRect.top),
            width: Math.ceil(cRect.width),
            height: Math.ceil(cRect.height),
          },
        };
      })()`,
      22000
    );
    await cdp.evalJs(
      sessionId,
      `document.getElementById('physicsPhaseCanvas')?.scrollIntoView({ block: 'center' }); true;`
    );
    await new Promise((r) => setTimeout(r, 120));
    const physRect = await cdp.evalJs(
      sessionId,
      `(() => {
        const r = document.getElementById('physicsPhaseCanvas')?.getBoundingClientRect();
        return r ? { x: Math.max(0, Math.floor(r.left + (window.scrollX || 0))), y: Math.max(0, Math.floor(r.top + (window.scrollY || 0))), width: Math.ceil(r.width), height: Math.ceil(r.height) } : null;
      })()`
    );
    const phaseClipShot = await cdp.send(
      'Page.captureScreenshot',
      {
        format: 'png',
        captureBeyondViewport: false,
        clip: {
          x: Math.max(0, physRect.x),
          y: Math.max(0, physRect.y),
          width: physRect.width,
          height: physRect.height,
          scale: 1,
        },
      },
      sessionId
    );
    const phasePngBytes = phaseClipShot?.data ? Buffer.from(phaseClipShot.data, 'base64').byteLength : 0;
    if (phasePngBytes < 1000) {
      throw new Error(`Step 6b: Expected non-blank #physicsPhaseCanvas screenshot >= 1000 B, got ${phasePngBytes} B`);
    }
    const physCls = await cdp.evalJs(
      sessionId,
      `(() => {
        if (window.__e2ePhysClsPo) window.__e2ePhysClsPo.disconnect();
        return window.__e2ePhysCls || 0;
      })()`
    );
    if (physCls >= 0.0001) {
      throw new Error(`Step 6b: Expected 0.000000 CLS during Physics ID updates, got ${physCls}`);
    }
    console.log(
      `  [PASS] Step 6b (Physics ID Panel, Phase/Conservation OffscreenCanvas & 56-Col v1.5.0 UI Parity): ${physUiState.regimeTxt}, R2=${physUiState.r2Txt}, eq="${physUiState.odeEq}", phasePng=${phasePngBytes}B, CLS=${physCls.toFixed(6)}`
    );

    await cdp.evalJs(
      sessionId,
      `(() => {
        document.getElementById('btnSrcTestbench')?.click();
        const scSel = document.getElementById('benchScenarioSelect');
        scSel.value = 'pillar_crossing';
        scSel.dispatchEvent(new Event('change', { bubbles: true }));
        document.querySelector('.lab-btn[data-lab="1"]').click();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Testbench re-lock in Mode B',
      `(() => ({ ok: (document.getElementById('kpiTargets')?.textContent || '') === '4 / 16' }))()`
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnLabStimulus').click(); true;`);
    const modeBState = await waitForCondition(
      cdp,
      sessionId,
      'Mode B deterministic saccade onset latency readout',
      `(() => {
        const val = document.getElementById('kpiSaccadeMs')?.textContent || '';
        const ms = parseFloat(val);
        return { ok: val.endsWith(' ms') && Number.isFinite(ms) && ms >= 10 && ms <= 600, val, ms };
      })()`
    );

    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__e2eSecondStimFired = false;
        document.getElementById('btnRecClear').click();
        document.getElementById('btnRecToggle').click();
        document.querySelector('.lab-btn[data-lab="2"]').click();
        document.getElementById('btnLabStimulus').click();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Mode C air-percussion badge, WebAudio auto-enable, and L1 strike telemetry',
      `(() => {
        const badge = document.getElementById('labBadge')?.textContent || '';
        const audioPressed = document.getElementById('btnAudioToggle')?.getAttribute('aria-pressed') || '';
        const recTxt = document.getElementById('recStatusLabel')?.textContent || '';
        const m = recTxt.match(/REC\\s+(\\d+)\\s+FRAMES/);
        const count = m ? parseInt(m[1], 10) : 0;
        if (count >= 18 && !window.__e2eSecondStimFired) {
          window.__e2eSecondStimFired = true;
          document.getElementById('btnLabStimulus').click();
        }
        return { ok: badge === 'MODE C' && audioPressed === 'true' && count >= 38, badge, audioPressed, count };
      })()`
    );
    await cdp.evalJs(sessionId, `window.__lastTelemetryExport = null; document.getElementById('btnRecToggle').click(); document.getElementById('btnExportJson').click(); true;`);
    const modeCStrikeCheck = await waitForCondition(
      cdp,
      sessionId,
      'Mode C perc_mask > 0 in recorded frames',
      `(() => {
        const ex = window.__lastTelemetryExport;
        if (!ex || !ex.filename.endsWith('.json')) return { ok: false };
        const parsed = JSON.parse(ex.fullText);
        const strikes = parsed.frames.filter((f) => f.perc_mask > 0).length;
        return { ok: strikes >= 1, strikes };
      })()`
    );

    await cdp.evalJs(sessionId, `document.getElementById('btnResSd').click(); true;`);
    const backSd = await waitForCondition(
      cdp,
      sessionId,
      'Return to 320x180 SD mode',
      `(() => {
        const resMeta = document.getElementById('inputResMeta')?.textContent || '';
        const pipAria = document.getElementById('pipCanvas')?.getAttribute('aria-label') || '';
        return { ok: resMeta === '320×180 SD RGBA' && pipAria.startsWith('32x32'), resMeta, pipAria };
      })()`
    );
    console.log(`  [PASS] Step 6 (Lab Modes B/C & SD Round-Trip): saccade=${modeBState.val}, modeC_strikes=${modeCStrikeCheck.strikes}, returned=${backSd.resMeta}`);

    // 6c. CHRONOS-1000 Live Dense Continuum Gate (vortex_1200 Synthetic + starling_1000 Real MP4)
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.querySelector('.lab-btn[data-lab="0"]')?.click();
        document.getElementById('btnCalClear')?.click();
        window.__e2eC1kCls = 0;
        if (window.__e2eC1kClsPo) {
          try { window.__e2eC1kClsPo.disconnect(); } catch (_e) {}
        }
        window.__e2eC1kClsPo = new PerformanceObserver((list) => {
          for (const e of list.getEntries()) window.__e2eC1kCls += e.value;
        });
        window.__e2eC1kClsPo.observe({ type: 'layout-shift', buffered: false });
        return true;
      })()`
    );

    const captureClipBytes = async (elementId) => {
      await cdp.evalJs(
        sessionId,
        `document.getElementById(${JSON.stringify(elementId)})?.scrollIntoView({ block: 'center', inline: 'center' }); true;`
      );
      await new Promise((r) => setTimeout(r, 120));
      const rect = await cdp.evalJs(
        sessionId,
        `(() => {
          const el = document.getElementById(${JSON.stringify(elementId)});
          if (!el) return null;
          const r = el.getBoundingClientRect();
          return {
            x: Math.max(0, Math.floor(r.left + (window.scrollX || 0))),
            y: Math.max(0, Math.floor(r.top + (window.scrollY || 0))),
            width: Math.ceil(r.width),
            height: Math.ceil(r.height),
          };
        })()`
      );
      if (!rect || rect.width < 10 || rect.height < 10) {
        throw new Error(`Step 6c: Missing or zero-sized canvas #${elementId}: ${JSON.stringify(rect)}`);
      }
      const shot = await cdp.send(
        'Page.captureScreenshot',
        {
          format: 'png',
          captureBeyondViewport: false,
          clip: { x: rect.x, y: rect.y, width: rect.width, height: rect.height, scale: 1 },
        },
        sessionId
      );
      return shot?.data ? Buffer.from(shot.data, 'base64').byteLength : 0;
    };

    const verifyDenseC1kShowcase = async (cfg) => {
      await cdp.evalJs(
        sessionId,
        `(() => {
          const btn = document.querySelector(${JSON.stringify(cfg.selector)});
          if (!btn) throw new Error('Missing showcase button ' + ${JSON.stringify(cfg.selector)});
          btn.click();
          return true;
        })()`
      );

      const denseState = await waitForCondition(
        cdp,
        sessionId,
        `Step 6c ${cfg.id}: >= 1000 confirmed, active continuum metrics, collapsed drawer, valid latency split`,
        `(() => {
          const btn = document.querySelector(${JSON.stringify(cfg.selector)});
          const meta = document.getElementById('continuumSummaryMeta')?.textContent || '';
          const confTxt = document.getElementById('kpiSwarmConfirmed')?.textContent || '';
          const peakTxt = document.getElementById('kpiSwarmPeak')?.textContent || '';
          const uniqTxt = document.getElementById('kpiSwarmUnique')?.textContent || '';
          const vortTxt = document.getElementById('kpiContinuumVort')?.textContent || '';
          const ftleTxt = document.getElementById('kpiContinuumFtle')?.textContent || '';
          const latTxt = document.getElementById('kpiLatencySplit')?.textContent || '';
          const drawer = document.getElementById('expertDiagnosticsDrawer');
          const panel = document.getElementById('continuumFieldPanel');

          const parseCommaInt = (s) => parseInt(String(s || '0').replace(/,/g, '').trim(), 10);
          const confParts = confTxt.split('/');
          const confirmed = parseCommaInt(confParts[0] || '0');
          const coast = parseCommaInt(confParts[1] || '-1');
          const peak = parseCommaInt(peakTxt || '0');
          const uniq = parseCommaInt(uniqTxt || '0');

          const latMatch = latTxt.trim().match(/^(\\d+(?:\\.\\d+)?)\\s*ms\\s*\\|\\s*(\\d+(?:\\.\\d+)?)\\s*ms$/);
          const kernelMs = latMatch ? parseFloat(latMatch[1]) : NaN;
          const pipeMs = latMatch ? parseFloat(latMatch[2]) : NaN;

          const scanEls = [panel, ...Array.from(document.querySelectorAll('.kpi-grid .kpi-value, [id^="kpi"]'))].filter(Boolean);
          const badTexts = [];
          for (const el of scanEls) {
            const t = el.textContent || '';
            if (/\\b(?:NaN|Infinity|undefined)\\b/i.test(t)) {
              badTexts.push((el.id || el.className) + '=' + t.slice(0, 60));
            }
          }

          const vortActive = !vortTxt.includes('--') && !vortTxt.startsWith('RAW') && vortTxt.includes('s⁻¹');
          const ftleActive = ${cfg.requireFtleNumeric ? "(!ftleTxt.includes('--') && !ftleTxt.startsWith('WARM') && ftleTxt.includes('s⁻¹'))" : "(!ftleTxt.includes('--'))"};

          return {
            ok: btn?.getAttribute('aria-pressed') === 'true' &&
                meta.startsWith('SWARM: ACTIVE') &&
                !meta.includes('DIAG INVALID') &&
                Number.isInteger(confirmed) && confirmed >= 1000 &&
                Number.isInteger(coast) && coast >= 0 &&
                Number.isInteger(peak) && peak >= 1000 &&
                Number.isInteger(uniq) && uniq >= 1000 &&
                vortActive &&
                ftleActive &&
                Boolean(latMatch) && Number.isFinite(kernelMs) && Number.isFinite(pipeMs) && kernelMs >= 0 && pipeMs >= 0 &&
                Boolean(drawer) && drawer.open === false &&
                badTexts.length === 0,
            meta, confTxt, peakTxt, uniqTxt, vortTxt, ftleTxt, latTxt,
            drawerOpen: drawer ? drawer.open : null,
            badTexts,
          };
        })()`,
        26000
      );

      const hudPngBytes = await captureClipBytes('hudCanvas');
      if (hudPngBytes < 2500) {
        throw new Error(`Step 6c (${cfg.id}): Expected non-blank #hudCanvas screenshot >= 2500 B, got ${hudPngBytes} B`);
      }

      // Click a populated noncentral region on #hudCanvas and verify a fresh field inspection update + normalized coordinate mapping
      let inspectState = null;
      const probeCoords = [
        [0.35, 0.42], [0.65, 0.58], [0.33, 0.54], [0.67, 0.46],
        [0.40, 0.35], [0.60, 0.65], [0.38, 0.60], [0.62, 0.40],
        [0.30, 0.50], [0.70, 0.50], [0.42, 0.62], [0.58, 0.38],
        [0.78, 0.30], [0.76, 0.68], [0.54, 0.26], [0.82, 0.48], [0.52, 0.72], [0.72, 0.24],
      ];
      const beforeProbe = await cdp.evalJs(
        sessionId,
        `(() => {
          const readout = (document.getElementById('swarmInspectReadout')?.textContent || '').trim();
          const mc = readout.match(/^INSP\\s+(?:μ|u)T([1-9]\\d*)(?:\\s*·\\s*B\\d+)?\\s*·\\s*\\(([-\\d.]+),\\s*([-\\d.]+)\\)/i);
          const mt = readout.match(/^INSP\\s+(?:μ|u)T([1-9]\\d*)/i);
          const resTxt = (document.getElementById('inputResMeta')?.textContent || '').trim();
          const rm = resTxt.match(/(\\d+)\\s*[×x]\\s*(\\d+)/i);
          const fw = rm ? parseInt(rm[1], 10) : 640;
          const fh = rm ? parseInt(rm[2], 10) : 360;
          return {
            readout,
            tid: mc ? parseInt(mc[1], 10) : (mt ? parseInt(mt[1], 10) : 0),
            nx: mc ? parseFloat(mc[2]) / fw : null,
            ny: mc ? parseFloat(mc[3]) / fh : null,
            fw,
            fh,
          };
        })()`
      );
      const filteredProbeCoords = probeCoords.filter(([nx, ny]) => {
        if (Math.hypot(nx - 0.5, ny - 0.5) < 0.12) return false;
        if (beforeProbe && beforeProbe.nx !== null && beforeProbe.ny !== null) {
          return Math.hypot(nx - beforeProbe.nx, ny - beforeProbe.ny) >= 0.15;
        }
        return true;
      });
      for (const [nx, ny] of (filteredProbeCoords.length ? filteredProbeCoords : probeCoords)) {
        const attemptBefore = await cdp.evalJs(
          sessionId,
          `(() => {
            const readout = (document.getElementById('swarmInspectReadout')?.textContent || '').trim();
            const prevInspectSeq = window.__chronosUiState?.lastInspectAck?.inspectSeq ?? 0;
            const hud = document.getElementById('hudCanvas');
            if (!hud) return null;
            const r = hud.getBoundingClientRect();
            const cx = r.left + r.width * ${nx};
            const cy = r.top + r.height * ${ny};
            hud.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, clientX: cx, clientY: cy }));
            return { readout, prevInspectSeq };
          })()`
        );
        if (!attemptBefore) continue;
        try {
          inspectState = await waitForCondition(
            cdp,
            sessionId,
            `Step 6c ${cfg.id}: fresh #swarmInspectReadout + inspectSeq + coord (${nx},${ny})`,
            `(() => {
              const readout = (document.getElementById('swarmInspectReadout')?.textContent || '').trim();
              const mc = readout.match(/^INSP\\s+(?:μ|u)T([1-9]\\d*)(?:\\s*·\\s*B\\d+)?\\s*·\\s*\\(([-\\d.]+),\\s*([-\\d.]+)\\)/i);
              const ack = window.__chronosUiState?.lastInspectAck;
              if (!ack || !Number.isInteger(ack.inspectSeq) || ack.inspectSeq <= ${attemptBefore.prevInspectSeq}) {
                return { ok: false, readout, tid: 0, inspectSeq: ack?.inspectSeq ?? 0 };
              }
              if (!mc) {
                return { ok: true, matched: false, readout, tid: 0, inspectSeq: ack.inspectSeq };
              }
              const tid = parseInt(mc[1], 10);
              const ackMatchesReq =
                Number.isFinite(ack.inspectReqNormX) &&
                Number.isFinite(ack.inspectReqNormY) &&
                Number.isFinite(ack.inspectReqPxX) &&
                Number.isFinite(ack.inspectReqPxY) &&
                Math.abs(ack.inspectReqNormX - ${nx}) <= 0.005 &&
                Math.abs(ack.inspectReqNormY - ${ny}) <= 0.005 &&
                Math.abs(ack.inspectReqPxX - ${nx} * ${beforeProbe.fw}) <= 0.5 &&
                Math.abs(ack.inspectReqPxY - ${ny} * ${beforeProbe.fh}) <= 0.5 &&
                ack.inspectedTid === tid &&
                ['live', 'pinned'].includes(ack.inspectStatus);
              if (!ackMatchesReq) {
                return { ok: true, matched: false, readout, tid: 0, inspectSeq: ack.inspectSeq };
              }
              const ix = parseFloat(mc[2]);
              const iy = parseFloat(mc[3]);
              const ixNorm = ix / ${beforeProbe.fw};
              const iyNorm = iy / ${beforeProbe.fh};
              const pxDist = Math.hypot(ix - ack.inspectReqPxX, iy - ack.inspectReqPxY);
              const mSpd = readout.match(/·\\s*([\\d.]+)\\s*px\\/s/i);
              const spdPxS = mSpd ? parseFloat(mSpd[1]) : 0;
              const qScale = ${beforeProbe.fw} / 640.0;
              const maxAllowedDist = ack.inspectStatus === 'pinned'
                ? (16.0 * qScale + spdPxS * 0.40 + 4.0)
                : (24.0 * qScale + 0.5);
              const coordMatch = pxDist < maxAllowedDist;
              const freshReadout = readout !== ${JSON.stringify(attemptBefore.readout)} && readout !== ${JSON.stringify(beforeProbe.readout)};
              const shiftedTarget = ${beforeProbe.tid} === 0 || tid !== ${beforeProbe.tid} ||
                (${beforeProbe.nx !== null ? `Math.hypot(ixNorm - ${beforeProbe.nx}, iyNorm - ${beforeProbe.ny}) >= 0.12` : 'false'});
              return { ok: true, matched: tid > 0 && coordMatch && freshReadout && shiftedTarget, readout, tid, ixNorm, iyNorm, inspectSeq: ack.inspectSeq, pxDist };
            })()`,
            2000
          );
          if (inspectState && inspectState.matched) break;
        } catch (_probeErr) {
          // Try next noncentral dense probe point if this coordinate fell outside active track radius
        }
      }
      if (!inspectState || !inspectState.matched) {
        throw new Error(`Step 6c (${cfg.id}): #swarmInspectReadout did not produce fresh INSP μT<tid> matching noncentral #hudCanvas coordinates`);
      }

      const pipPngBytes = await captureClipBytes('pipCanvas');
      if (pipPngBytes < 600) {
        throw new Error(`Step 6c (${cfg.id}): Expected non-blank Hero PiP #pipCanvas screenshot >= 600 B, got ${pipPngBytes} B`);
      }

      return { ...denseState, hudPngBytes, pipPngBytes, readout: inspectState.readout, inspectedTid: inspectState.tid };
    };

    const vortex1200Res = await verifyDenseC1kShowcase({
      id: 'vortex_1200',
      selector: '.showcase-btn[data-showcase="vortex_1200"]',
      requireFtleNumeric: true,
    });
    const starling1000Res = await verifyDenseC1kShowcase({
      id: 'starling_1000',
      selector: '.showcase-btn[data-showcase="starling_1000"]',
      requireFtleNumeric: false,
    });

    const c1kCls = await cdp.evalJs(
      sessionId,
      `(() => {
        if (window.__e2eC1kClsPo) window.__e2eC1kClsPo.disconnect();
        return window.__e2eC1kCls || 0;
      })()`
    );
    if (c1kCls >= 0.0001) {
      throw new Error(`Step 6c: Expected 0.000000 CLS during dense CHRONOS-1000 showcase checks, got ${c1kCls}`);
    }

    console.log(
      `  [PASS] Step 6c (CHRONOS-1000 Live Dense Continuum Gate): vortex_1200=[${vortex1200Res.confTxt}, ${vortex1200Res.latTxt}, ${vortex1200Res.readout}, hud=${vortex1200Res.hudPngBytes}B, pip=${vortex1200Res.pipPngBytes}B], starling_1000=[${starling1000Res.confTxt}, ${starling1000Res.latTxt}, ${starling1000Res.readout}, hud=${starling1000Res.hudPngBytes}B, pip=${starling1000Res.pipPngBytes}B], CLS=${c1kCls.toFixed(6)}`
    );

    // 6d. Wave-2 Deck UI FSM, Hero-16 Toggle, 16:9 Fullscreen Viewport, Shortcut Guards & Calibration Exclusion
    const sendCdpKey = async (key, code, keyCode, modifiers = 0, extra = {}) => {
      const text = key.length === 1 && modifiers === 0 && !extra.autoRepeat ? key : undefined;
      await cdp.send(
        'Input.dispatchKeyEvent',
        {
          type: 'keyDown',
          key,
          code,
          windowsVirtualKeyCode: keyCode,
          nativeVirtualKeyCode: keyCode,
          modifiers,
          ...(extra.autoRepeat ? { autoRepeat: true } : {}),
          ...(text ? { text, unmodifiedText: text } : {}),
        },
        sessionId
      );
      await cdp.send(
        'Input.dispatchKeyEvent',
        {
          type: 'keyUp',
          key,
          code,
          windowsVirtualKeyCode: keyCode,
          nativeVirtualKeyCode: keyCode,
          modifiers,
        },
        sessionId
      );
    };

    const clickDomCenter = async (selector) => {
      const pt = await cdp.evalJs(
        sessionId,
        `(() => {
          const el = document.querySelector(${JSON.stringify(selector)});
          if (!el) return null;
          el.scrollIntoView({ block: 'center', inline: 'center' });
          const r = el.getBoundingClientRect();
          return { x: r.left + r.width * 0.5, y: r.top + r.height * 0.5, w: r.width, h: r.height };
        })()`
      );
      if (!pt || pt.w < 1 || pt.h < 1) {
        throw new Error(`Step 6d: Cannot click missing/collapsed selector ${selector}`);
      }
      await cdp.send(
        'Input.dispatchMouseEvent',
        { type: 'mouseMoved', x: pt.x, y: pt.y, button: 'none' },
        sessionId
      );
      await cdp.send(
        'Input.dispatchMouseEvent',
        { type: 'mousePressed', x: pt.x, y: pt.y, button: 'left', clickCount: 1 },
        sessionId
      );
      await cdp.send(
        'Input.dispatchMouseEvent',
        { type: 'mouseReleased', x: pt.x, y: pt.y, button: 'left', clickCount: 1 },
        sessionId
      );
      return pt;
    };

    const probeSwarmInspectOnHud = async (label, useCdpClick = false) => {
      const coords = [
        [0.35, 0.42], [0.65, 0.58], [0.33, 0.54], [0.67, 0.46],
        [0.40, 0.35], [0.60, 0.65], [0.38, 0.60], [0.62, 0.40],
        [0.30, 0.50], [0.70, 0.50], [0.42, 0.62], [0.58, 0.38],
        [0.78, 0.30], [0.76, 0.68], [0.54, 0.26], [0.82, 0.48], [0.52, 0.72], [0.72, 0.24],
      ];
      const before = await cdp.evalJs(
        sessionId,
        `(() => {
          const readout = (document.getElementById('swarmInspectReadout')?.textContent || '').trim();
          const mc = readout.match(/^INSP\\s+(?:μ|u)T([1-9]\\d*)(?:\\s*·\\s*B\\d+)?\\s*·\\s*\\(([-\\d.]+),\\s*([-\\d.]+)\\)/i);
          const mt = readout.match(/^INSP\\s+(?:μ|u)T([1-9]\\d*)/i);
          const resTxt = (document.getElementById('inputResMeta')?.textContent || '').trim();
          const rm = resTxt.match(/(\\d+)\\s*[×x]\\s*(\\d+)/i);
          const fw = rm ? parseInt(rm[1], 10) : 640;
          const fh = rm ? parseInt(rm[2], 10) : 360;
          return {
            readout,
            tid: mc ? parseInt(mc[1], 10) : (mt ? parseInt(mt[1], 10) : 0),
            nx: mc ? parseFloat(mc[2]) / fw : null,
            ny: mc ? parseFloat(mc[3]) / fh : null,
            fw,
            fh,
          };
        })()`
      );
      const filteredCoords = coords.filter(([nx, ny]) => {
        if (Math.hypot(nx - 0.5, ny - 0.5) < 0.12) return false;
        if (before && before.nx !== null && before.ny !== null) {
          return Math.hypot(nx - before.nx, ny - before.ny) >= 0.15;
        }
        return true;
      });
      for (const [nx, ny] of (filteredCoords.length ? filteredCoords : coords)) {
        const pt = await cdp.evalJs(
          sessionId,
          `(() => {
            const readout = (document.getElementById('swarmInspectReadout')?.textContent || '').trim();
            const prevInspectSeq = window.__chronosUiState?.lastInspectAck?.inspectSeq ?? 0;
            const hud = document.getElementById('hudCanvas');
            if (!hud) return null;
            const r = hud.getBoundingClientRect();
            const x = r.left + r.width * ${nx};
            const y = r.top + r.height * ${ny};
            if (!${Boolean(useCdpClick)}) {
              hud.dispatchEvent(new MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                clientX: x,
                clientY: y,
              }));
            }
            return { x, y, attemptBeforeReadout: readout, prevInspectSeq };
          })()`
        );
        if (!pt) continue;
        if (useCdpClick) {
          await cdp.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: pt.x, y: pt.y, button: 'left', clickCount: 1 }, sessionId);
          await cdp.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: pt.x, y: pt.y, button: 'left', clickCount: 1 }, sessionId);
        }
        try {
          const res = await waitForCondition(
            cdp,
            sessionId,
            `${label} (${nx},${ny})`,
            `(() => {
              const readout = (document.getElementById('swarmInspectReadout')?.textContent || '').trim();
              const mc = readout.match(/^INSP\\s+(?:μ|u)T([1-9]\\d*)(?:\\s*·\\s*B\\d+)?\\s*·\\s*\\(([-\\d.]+),\\s*([-\\d.]+)\\)/i);
              const ack = window.__chronosUiState?.lastInspectAck;
              if (!ack || !Number.isInteger(ack.inspectSeq) || ack.inspectSeq <= ${pt.prevInspectSeq}) {
                return { ok: false, readout, inspectSeq: ack?.inspectSeq ?? 0 };
              }
              if (!mc) {
                return { ok: true, matched: false, readout, inspectSeq: ack.inspectSeq };
              }
              const tid = parseInt(mc[1], 10);
              const ackMatchesReq =
                Number.isFinite(ack.inspectReqNormX) &&
                Number.isFinite(ack.inspectReqNormY) &&
                Number.isFinite(ack.inspectReqPxX) &&
                Number.isFinite(ack.inspectReqPxY) &&
                Math.abs(ack.inspectReqNormX - ${nx}) <= 0.005 &&
                Math.abs(ack.inspectReqNormY - ${ny}) <= 0.005 &&
                Math.abs(ack.inspectReqPxX - ${nx} * ${before.fw}) <= 0.5 &&
                Math.abs(ack.inspectReqPxY - ${ny} * ${before.fh}) <= 0.5 &&
                ack.inspectedTid === tid &&
                ['live', 'pinned'].includes(ack.inspectStatus);
              if (!ackMatchesReq) {
                return { ok: true, matched: false, readout, inspectSeq: ack.inspectSeq };
              }
              const ix = parseFloat(mc[2]);
              const iy = parseFloat(mc[3]);
              const ixNorm = ix / ${before.fw};
              const iyNorm = iy / ${before.fh};
              const pxDist = Math.hypot(ix - ack.inspectReqPxX, iy - ack.inspectReqPxY);
              const mSpd = readout.match(/·\\s*([\\d.]+)\\s*px\\/s/i);
              const spdPxS = mSpd ? parseFloat(mSpd[1]) : 0;
              const qScale = ${before.fw} / 640.0;
              const maxAllowedDist = ack.inspectStatus === 'pinned'
                ? (16.0 * qScale + spdPxS * 0.40 + 4.0)
                : (24.0 * qScale + 0.5);
              const coordMatch = pxDist < maxAllowedDist;
              const freshReadout = readout !== ${JSON.stringify(pt.attemptBeforeReadout)} && readout !== ${JSON.stringify(before.readout)};
              const shiftedTarget = ${before.tid} === 0 || tid !== ${before.tid} ||
                (${before.nx !== null ? `Math.hypot(ixNorm - ${before.nx}, iyNorm - ${before.ny}) >= 0.12` : 'false'});
              return { ok: true, matched: tid > 0 && coordMatch && freshReadout && shiftedTarget, readout, tid, ixNorm, iyNorm, inspectSeq: ack.inspectSeq, pxDist };
            })()`,
            1800
          );
          if (res && res.matched) return res;
        } catch (_probeErr) {
          // Try next noncentral dense region if current coordinate sits over empty sky between starlings
        }
      }
      throw new Error(`${label}: #swarmInspectReadout did not lock onto a fresh track with matching noncentral coordinates`);
    };

    await cdp.evalJs(
      sessionId,
      `(() => {
        window.__e2eHeroRawCls = 0;
        window.__e2eUiFsmCls = 0;
        window.__e2eUiFsmClsEntries = [];
        window.__e2eUiFsmPhase = 'hero';
        window.__e2eUiFsmClsPo = new PerformanceObserver((list) => {
          for (const e of list.getEntries()) {
            if (window.__e2eUiFsmPhase === 'hero') {
              window.__e2eHeroRawCls += e.value;
            }
            if (!e.hadRecentInput) {
              window.__e2eUiFsmCls += e.value;
              window.__e2eUiFsmClsEntries.push({
                phase: window.__e2eUiFsmPhase,
                val: +e.value.toFixed(6),
                nodes: (e.sources || []).map(s => s.node?.id || s.node?.className || s.node?.tagName || '?'),
              });
            }
          }
        });
        window.__e2eUiFsmClsPo.observe({ type: 'layout-shift', buffered: false });
        return true;
      })()`
    );

    const fsmBoot = await cdp.evalJs(
      sessionId,
      `(() => {
        const st = window.__chronosUiState;
        const dock = document.getElementById('hudDockSlot');
        const panel = document.getElementById('hudPanel');
        const hud = document.getElementById('hudCanvas');
        const btnHero = document.getElementById('btnHudHeroToggle');
        const chkHero = document.getElementById('chkHeroTargets');
        const btnFs = document.getElementById('btnHudFullscreen');
        const rHero = btnHero?.getBoundingClientRect();
        const rFs = btnFs?.getBoundingClientRect();
        const rHud = hud?.getBoundingClientRect();
        const csHud = hud ? getComputedStyle(hud) : null;
        const dockAspectErr = rHud && rHud.height > 0 ? Math.abs(rHud.width / rHud.height - 16 / 9) : 1;
        const borderFreeHud = Boolean(
          csHud &&
            csHud.borderLeftWidth === '0px' &&
            csHud.borderTopWidth === '0px' &&
            csHud.borderRightWidth === '0px' &&
            csHud.borderBottomWidth === '0px' &&
            csHud.paddingLeft === '0px' &&
            csHud.paddingTop === '0px' &&
            csHud.paddingRight === '0px' &&
            csHud.paddingBottom === '0px'
        );
        return {
          ok: Boolean(st && dock && panel && hud && btnHero && chkHero && btnFs) &&
              dock.contains(panel) &&
              Object.isFrozen(st) &&
              Object.isFrozen(st.overlays) &&
              Number.isInteger(st.seq) &&
              st.engine === 'SWARM_ACTIVE' &&
              st.overlays.showHeroTargets === true &&
              st.viewport === 'DOCKED' &&
              st.pointerTool === 'INSPECT_TARGET' &&
              panel.dataset.engine === 'SWARM_ACTIVE' &&
              panel.dataset.viewport === 'DOCKED' &&
              panel.dataset.pointerTool === 'INSPECT_TARGET' &&
              btnHero.getAttribute('aria-pressed') === 'true' &&
              chkHero.checked === true &&
              borderFreeHud &&
              dockAspectErr <= 0.005,
          borderFreeHud,
          dockAspectErr,
          heroW: rHero ? rHero.width : 0,
          fsLeft: rFs ? rFs.left : 0,
          seq: st ? st.seq : -1,
        };
      })()`
    );
    if (!fsmBoot.ok) {
      throw new Error(`Step 6d: Invalid Wave-2 DeckUiFsm initial state: ${JSON.stringify(fsmBoot)}`);
    }

    // 6d.1 Toggle Hero-16 OFF via #btnHudHeroToggle, assert 1:1 checkbox sync, fixed button geometry & live inspection
    await clickDomCenter('#btnHudHeroToggle');
    const heroHiddenState = await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Hero-16 toggle OFF updates FSM, #btnHudHeroToggle, #chkHeroTargets & keeps zero button shift',
      `(() => {
        const st = window.__chronosUiState;
        const btnHero = document.getElementById('btnHudHeroToggle');
        const chkHero = document.getElementById('chkHeroTargets');
        const btnFs = document.getElementById('btnHudFullscreen');
        const rHero = btnHero?.getBoundingClientRect();
        const rFs = btnFs?.getBoundingClientRect();
        return {
          ok: Boolean(st && btnHero && chkHero && btnFs && rHero && rFs) &&
              st.overlays.showHeroTargets === false &&
              chkHero.checked === false &&
              btnHero.textContent.trim() === 'HERO 16: HIDDEN' &&
              btnHero.getAttribute('aria-pressed') === 'false' &&
              btnHero.dataset.hiddenSolo === 'false' &&
              Math.abs(rHero.width - ${fsmBoot.heroW}) < 0.1 &&
              Math.abs(rFs.left - ${fsmBoot.fsLeft}) < 0.1,
          text: btnHero?.textContent?.trim(),
          checked: chkHero?.checked,
          hiddenSolo: btnHero?.dataset?.hiddenSolo,
        };
      })()`,
      6000
    );

    const hiddenHeroInspect = await probeSwarmInspectOnHud(
      'Step 6d: Swarm inspect readout stays live while Hero-16 overlay is hidden',
      false
    );
    const hiddenHeroPipBytes = await captureClipBytes('pipCanvas');
    if (hiddenHeroPipBytes < 600) {
      throw new Error(`Step 6d: Hero #pipCanvas unexpectedly blank while Hero-16 overlay hidden (${hiddenHeroPipBytes} B)`);
    }

    // Toggle back ON via #chkHeroTargets click, then verify H keyboard shortcut OFF -> ON
    await clickDomCenter('#chkHeroTargets');
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Clicking #chkHeroTargets syncs Hero-16 back ON',
      `(() => {
        const st = window.__chronosUiState;
        const btnHero = document.getElementById('btnHudHeroToggle');
        const chkHero = document.getElementById('chkHeroTargets');
        return {
          ok: Boolean(st && st.overlays.showHeroTargets === true && chkHero?.checked === true && btnHero?.textContent?.trim() === 'Hero 16: ON' && btnHero?.getAttribute('aria-pressed') === 'true'),
        };
      })()`,
      5000
    );

    await cdp.evalJs(sessionId, `document.activeElement?.blur?.(); true;`);
    await sendCdpKey('h', 'KeyH', 72, 0);
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: H shortcut toggles Hero-16 OFF',
      `(() => ({ ok: window.__chronosUiState?.overlays?.showHeroTargets === false }))()`,
      4000
    );
    await sendCdpKey('h', 'KeyH', 72, 0);
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: H shortcut toggles Hero-16 back ON',
      `(() => ({ ok: window.__chronosUiState?.overlays?.showHeroTargets === true }))()`,
      4000
    );

    // 6d.2 Verify H & F shortcuts are ignored independently inside <input>/<select>/<textarea>/contentEditable, with Ctrl/Alt/Meta/Repeat, and when #liveStreamDialog is open
    const assertGuardedKey = async (guardLabel, key, code, keyCode, modifiers = 0, extra = {}) => {
      const snapUiGuard = `(() => {
        const st = window.__chronosUiState;
        const panel = document.getElementById('hudPanel');
        const chk = document.getElementById('chkHeroTargets');
        const fsElem = document.fullscreenElement || document.webkitFullscreenElement;
        return st ? {
          showHeroTargets: st.overlays.showHeroTargets,
          viewport: st.viewport,
          chkHero: Boolean(chk && chk.checked),
          nativeFs: fsElem === panel,
          fallbackMax: Boolean(panel && panel.classList.contains('hud-maximized')),
        } : null;
      })()`;
      const before = await cdp.evalJs(sessionId, snapUiGuard);
      if (!before) {
        throw new Error(`Step 6d: Missing window.__chronosUiState before guard ${guardLabel}`);
      }
      await sendCdpKey(key, code, keyCode, modifiers, extra);
      await new Promise((r) => setTimeout(r, 40));
      const after = await cdp.evalJs(sessionId, snapUiGuard);
      if (
        !after ||
        after.showHeroTargets !== before.showHeroTargets ||
        after.viewport !== before.viewport ||
        after.chkHero !== before.chkHero ||
        after.nativeFs !== before.nativeFs ||
        after.fallbackMax !== before.fallbackMax
      ) {
        throw new Error(
          `Step 6d: Shortcut guard failed independently for ${guardLabel} (${key}): before=${JSON.stringify(before)}, after=${JSON.stringify(after)}`
        );
      }
    };

    const focusInputOk = await cdp.evalJs(
      sessionId,
      `(() => {
        const el = document.getElementById('calRulerM');
        if (!el) return false;
        el.focus();
        return document.activeElement === el && document.activeElement.tagName === 'INPUT';
      })()`
    );
    if (!focusInputOk) throw new Error('Step 6d: Failed to focus INPUT #calRulerM for shortcut guard test');
    await assertGuardedKey('INPUT', 'h', 'KeyH', 72, 0);
    await assertGuardedKey('INPUT', 'f', 'KeyF', 70, 0);

    const focusSelectOk = await cdp.evalJs(
      sessionId,
      `(() => {
        const el = document.getElementById('swarmModeSelect');
        if (!el) return false;
        el.focus();
        return document.activeElement === el && document.activeElement.tagName === 'SELECT';
      })()`
    );
    if (!focusSelectOk) throw new Error('Step 6d: Failed to focus SELECT #swarmModeSelect for shortcut guard test');
    await assertGuardedKey('SELECT', 'h', 'KeyH', 72, 0);
    await assertGuardedKey('SELECT', 'f', 'KeyF', 70, 0);

    const focusTextareaOk = await cdp.evalJs(
      sessionId,
      `(() => {
        let ta = document.getElementById('__e2eTmpTextarea');
        if (!ta) {
          ta = document.createElement('textarea');
          ta.id = '__e2eTmpTextarea';
          ta.style.cssText = 'position:fixed;left:-9999px;top:-9999px;width:40px;height:20px;opacity:0;';
          document.body.appendChild(ta);
        }
        ta.focus();
        return document.activeElement === ta && document.activeElement.tagName === 'TEXTAREA';
      })()`
    );
    if (!focusTextareaOk) throw new Error('Step 6d: Failed to focus TEXTAREA for shortcut guard test');
    await assertGuardedKey('TEXTAREA', 'h', 'KeyH', 72, 0);
    await assertGuardedKey('TEXTAREA', 'f', 'KeyF', 70, 0);
    await cdp.evalJs(sessionId, `document.getElementById('__e2eTmpTextarea')?.remove(); true;`);

    const focusCeOk = await cdp.evalJs(
      sessionId,
      `(() => {
        let ce = document.getElementById('__e2eTmpCe');
        if (!ce) {
          ce = document.createElement('div');
          ce.id = '__e2eTmpCe';
          ce.contentEditable = 'true';
          ce.tabIndex = 0;
          ce.style.cssText = 'position:fixed;left:-9999px;top:-9999px;width:40px;height:20px;opacity:0;';
          document.body.appendChild(ce);
        }
        ce.focus();
        return document.activeElement === ce && Boolean(document.activeElement.isContentEditable);
      })()`
    );
    if (!focusCeOk) throw new Error('Step 6d: Failed to focus contentEditable element for shortcut guard test');
    await assertGuardedKey('CONTENTEDITABLE', 'h', 'KeyH', 72, 0);
    await assertGuardedKey('CONTENTEDITABLE', 'f', 'KeyF', 70, 0);
    await cdp.evalJs(sessionId, `document.getElementById('__e2eTmpCe')?.remove(); document.activeElement?.blur?.(); true;`);

    await assertGuardedKey('CTRL', 'h', 'KeyH', 72, 2);
    await assertGuardedKey('CTRL', 'f', 'KeyF', 70, 2);
    await assertGuardedKey('ALT', 'h', 'KeyH', 72, 1);
    await assertGuardedKey('ALT', 'f', 'KeyF', 70, 1);
    await assertGuardedKey('META', 'h', 'KeyH', 72, 4);
    await assertGuardedKey('META', 'f', 'KeyF', 70, 4);
    await assertGuardedKey('REPEAT', 'h', 'KeyH', 72, 0, { autoRepeat: true });
    await assertGuardedKey('REPEAT', 'f', 'KeyF', 70, 0, { autoRepeat: true });

    const dialogOpenOk = await cdp.evalJs(
      sessionId,
      `(() => {
        const d = document.getElementById('liveStreamDialog');
        if (d && !d.open && typeof d.showModal === 'function') d.showModal();
        return Boolean(d && d.open);
      })()`
    );
    if (!dialogOpenOk) throw new Error('Step 6d: Failed to open #liveStreamDialog for shortcut guard test');
    await assertGuardedKey('DIALOG', 'h', 'KeyH', 72, 0);
    await assertGuardedKey('DIALOG', 'f', 'KeyF', 70, 0);
    await cdp.evalJs(
      sessionId,
      `(() => {
        const d = document.getElementById('liveStreamDialog');
        if (d && d.open) d.close();
        return Boolean(!d || !d.open);
      })()`
    );

    const heroRawCls = await cdp.evalJs(
      sessionId,
      `(() => {
        const v = window.__e2eHeroRawCls || 0;
        window.__e2eUiFsmPhase = 'fullscreen';
        return v;
      })()`
    );
    if (heroRawCls >= 0.0001) {
      throw new Error(`Step 6d: Expected 0.000000 raw layout-shift across Hero-16 toggles & key guards, got ${heroRawCls}`);
    }

    // Snapshot docked geometry before fullscreen transitions so we verify 0.00px drift on DOCKED restore
    const dockSnapBefore = await cdp.evalJs(
      sessionId,
      `(() => {
        const snap = (id) => {
          const el = document.getElementById(id);
          if (!el) return null;
          const r = el.getBoundingClientRect();
          return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
        };
        return {
          dock: snap('hudDockSlot'),
          panel: snap('hudPanel'),
          canvas: snap('hudCanvas'),
          continuum: snap('continuumFieldPanel'),
        };
      })()`
    );

    // 6d.3 Wavefield Fullscreen enter (#btnHudFullscreen click) -> 16:9 aspect check -> inspect click -> Esc exit -> F toggle
    await clickDomCenter('#btnHudFullscreen');
    const fsEntered = await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Fullscreen enters FULLSCREEN with strict 16:9 #hudCanvas geometry',
      `(() => {
        const st = window.__chronosUiState;
        const panel = document.getElementById('hudPanel');
        const btnFs = document.getElementById('btnHudFullscreen');
        const hud = document.getElementById('hudCanvas');
        const r = hud ? hud.getBoundingClientRect() : null;
        const fsElem = document.fullscreenElement || document.webkitFullscreenElement;
        const isNativeOrFallback = Boolean(panel && (fsElem === panel || panel.classList.contains('hud-maximized')));
        const aspect = r && r.height > 0 ? r.width / r.height : 0;
        const aspectErr = Math.abs(aspect - 16 / 9);
        const csHud = hud ? getComputedStyle(hud) : null;
        const borderFreeHud = Boolean(
          csHud &&
            csHud.borderLeftWidth === '0px' &&
            csHud.borderTopWidth === '0px' &&
            csHud.borderRightWidth === '0px' &&
            csHud.borderBottomWidth === '0px' &&
            csHud.paddingLeft === '0px' &&
            csHud.paddingTop === '0px' &&
            csHud.paddingRight === '0px' &&
            csHud.paddingBottom === '0px'
        );
        const coversViewport = Boolean(r && (r.width >= window.innerWidth * 0.65 || r.height >= window.innerHeight * 0.65));
        const inBounds = Boolean(r && r.left >= -1 && r.top >= -1 && r.right <= window.innerWidth + 1 && r.bottom <= window.innerHeight + 1);
        return {
          ok: Boolean(st && st.viewport === 'FULLSCREEN' && panel?.dataset.viewport === 'FULLSCREEN' &&
              btnFs?.getAttribute('aria-pressed') === 'true' && btnFs?.textContent?.trim() === '✕ Exit Full' &&
              isNativeOrFallback && coversViewport && inBounds && borderFreeHud && aspectErr <= 0.005),
          nativeFs: fsElem === panel,
          fallbackMax: Boolean(panel?.classList.contains('hud-maximized')),
          w: r ? Math.round(r.width) : 0,
          h: r ? Math.round(r.height) : 0,
          aspectErr,
        };
      })()`,
      6000
    );

    await probeSwarmInspectOnHud(
      'Step 6d: Clicking fullscreen #hudCanvas sustains valid Swarm inspection readout',
      true
    );

    await sendCdpKey('Escape', 'Escape', 27, 0);
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Escape exits Wavefield Fullscreen cleanly back to DOCKED',
      `(() => {
        const st = window.__chronosUiState;
        const panel = document.getElementById('hudPanel');
        const dock = document.getElementById('hudDockSlot');
        const btnFs = document.getElementById('btnHudFullscreen');
        const fsElem = document.fullscreenElement || document.webkitFullscreenElement;
        return {
          ok: Boolean(st && st.viewport === 'DOCKED' && !fsElem && panel &&
              !panel.classList.contains('hud-maximized') &&
              panel.dataset.viewport === 'DOCKED' &&
              (!dock || dock.style.blockSize === '') &&
              btnFs?.getAttribute('aria-pressed') === 'false' &&
              btnFs?.textContent?.trim() === '⛶ Fullscreen'),
        };
      })()`,
      5000
    );

    await cdp.evalJs(sessionId, `document.activeElement?.blur?.(); true;`);
    await sendCdpKey('f', 'KeyF', 70, 0);
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: F key enters Wavefield Fullscreen',
      `(() => ({ ok: window.__chronosUiState?.viewport === 'FULLSCREEN' }))()`,
      5000
    );
    await sendCdpKey('f', 'KeyF', 70, 0);
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Second F key exits Wavefield Fullscreen back to DOCKED',
      `(() => ({ ok: window.__chronosUiState?.viewport === 'DOCKED' && !document.fullscreenElement && !document.getElementById('hudPanel')?.classList.contains('hud-maximized') }))()`,
      5000
    );

    const uiFsmGeomAndCls = await cdp.evalJs(
      sessionId,
      `(() => {
        if (window.__e2eUiFsmClsPo) window.__e2eUiFsmClsPo.disconnect();
        const snap = (id) => {
          const el = document.getElementById(id);
          if (!el) return null;
          const r = el.getBoundingClientRect();
          return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
        };
        const after = {
          dock: snap('hudDockSlot'),
          panel: snap('hudPanel'),
          canvas: snap('hudCanvas'),
          continuum: snap('continuumFieldPanel'),
        };
        const before = ${JSON.stringify(null)} || null;
        return {
          cls: window.__e2eUiFsmCls || 0,
          entries: window.__e2eUiFsmClsEntries || [],
          after,
        };
      })()`
    );
    const uiFsmCls = uiFsmGeomAndCls.cls;
    const dockDrift = ['dock', 'panel', 'canvas', 'continuum']
      .map((k) => {
        const b = dockSnapBefore[k];
        const a = uiFsmGeomAndCls.after[k];
        if (!b || !a) return `${k}:missing`;
        const dx = Math.abs(a.x - b.x);
        const dy = Math.abs(a.y - b.y);
        const dw = Math.abs(a.w - b.w);
        const dh = Math.abs(a.h - b.h);
        const maxD = Math.max(dx, dy, dw, dh);
        return maxD > 0.5 ? `${k}:d=${maxD.toFixed(2)}` : null;
      })
      .filter(Boolean);
    if (uiFsmCls >= 0.0001 || dockDrift.length > 0) {
      throw new Error(
        `Step 6d: Expected 0.000000 Web Vitals CLS & 0.00px dock drift across Hero-16 & Fullscreen transitions, got cls=${uiFsmCls}, drift=${JSON.stringify(dockDrift)}, entries=${JSON.stringify(uiFsmGeomAndCls.entries)}`
      );
    }

    // Restore cleanly to legacy SD pillar_crossing before Step 7 (and verify SWARM_OFF data-hidden-solo & calibration tool mutual exclusion)
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.querySelector('.lab-btn[data-lab="0"]')?.click();
        const swarmSel = document.getElementById('swarmModeSelect');
        if (swarmSel) {
          swarmSel.value = '0';
          swarmSel.dispatchEvent(new Event('change', { bubbles: true }));
        }
        const ovSel = document.getElementById('fieldOverlayModeSelect');
        if (ovSel) {
          ovSel.value = '0';
          ovSel.dispatchEvent(new Event('change', { bubbles: true }));
        }
        document.getElementById('btnSrcTestbench')?.click();
        const scSel = document.getElementById('benchScenarioSelect');
        if (scSel) {
          scSel.value = 'pillar_crossing';
          scSel.dispatchEvent(new Event('change', { bubbles: true }));
        }
        document.getElementById('btnResSd')?.click();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Restore legacy 320x180 SD pillar_crossing after Step 6c/6d',
      `(() => {
        const resMeta = document.getElementById('inputResMeta')?.textContent || '';
        const targets = document.getElementById('kpiTargets')?.textContent || '';
        const meta = document.getElementById('continuumSummaryMeta')?.textContent || '';
        const st = window.__chronosUiState;
        const v = document.getElementById('hiddenVideoSource');
        return {
          ok: resMeta === '320×180 SD RGBA' &&
              targets === '4 / 16' &&
              meta.startsWith('SWARM: OFF') &&
              !meta.includes('DIAG INVALID') &&
              Boolean(st && st.source === 'TESTBENCH' && st.engine === 'SWARM_OFF' && st.viewport === 'DOCKED' && st.overlays.showHeroTargets === true) &&
              (!v || v.paused),
          resMeta, targets, meta,
        };
      })()`,
      12000
    );

    // 6d.4 Verify SWARM_OFF solo-hidden warning flag (data-hidden-solo="true") + pointerTool ruler exclusion
    await sendCdpKey('h', 'KeyH', 72, 0);
    const soloWarnState = await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Hiding Hero-16 while SWARM_OFF activates data-hidden-solo="true"',
      `(() => {
        const btnHero = document.getElementById('btnHudHeroToggle');
        const st = window.__chronosUiState;
        return {
          ok: Boolean(st && st.overlays.showHeroTargets === false && btnHero?.dataset?.hiddenSolo === 'true'),
          hiddenSolo: btnHero?.dataset?.hiddenSolo,
        };
      })()`,
      4000
    );
    await sendCdpKey('h', 'KeyH', 72, 0);
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Restoring Hero-16 ON clears data-hidden-solo back to false',
      `(() => {
        const btnHero = document.getElementById('btnHudHeroToggle');
        const st = window.__chronosUiState;
        return { ok: Boolean(st && st.overlays.showHeroTargets === true && btnHero?.dataset?.hiddenSolo === 'false') };
      })()`,
      4000
    );

    await clickDomCenter('#btnCalRuler');
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Arming ruler transitions pointerTool to CAL_RULER_P1',
      `(() => ({ ok: window.__chronosUiState?.pointerTool === 'CAL_RULER_P1' }))()`,
      4000
    );
    await clickDomCenter('#hudCanvas');
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Clicking #hudCanvas during calibration transitions CAL_RULER_P1 -> CAL_RULER_P2 cleanly',
      `(() => ({ ok: window.__chronosUiState?.pointerTool === 'CAL_RULER_P2' }))()`,
      4000
    );
    await sendCdpKey('Escape', 'Escape', 27, 0);
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Escape disarms calibration pointerTool back to INSPECT_TARGET',
      `(() => ({ ok: window.__chronosUiState?.pointerTool === 'INSPECT_TARGET' }))()`,
      4000
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnCalClear')?.click(); true;`);

    // 6d.5 Verify single Escape exits .hud-maximized fallback AND cancels armed Ruler synchronously
    await cdp.evalJs(
      sessionId,
      `(() => {
        const panel = document.getElementById('hudPanel');
        if (!panel) return false;
        window.__e2eHasOwnReqFs = Object.prototype.hasOwnProperty.call(panel, 'requestFullscreen');
        window.__e2eOrigRequestFullscreen = panel.requestFullscreen;
        panel.requestFullscreen = () => Promise.reject(new Error('e2e-fallback-test'));
        return true;
      })()`
    );
    await clickDomCenter('#btnCalRuler');
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Arming ruler transitions pointerTool to CAL_RULER_P1 before fallback fullscreen',
      `(() => ({ ok: window.__chronosUiState?.pointerTool === 'CAL_RULER_P1' }))()`,
      4000
    );
    await clickDomCenter('#btnHudFullscreen');
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Fallback .hud-maximized enters FULLSCREEN while Ruler remains armed',
      `(() => {
        const st = window.__chronosUiState;
        const panel = document.getElementById('hudPanel');
        return {
          ok: Boolean(
            st &&
              st.viewport === 'FULLSCREEN' &&
              st.pointerTool === 'CAL_RULER_P1' &&
              panel?.classList.contains('hud-maximized') &&
              !document.fullscreenElement
          ),
        };
      })()`,
      4000
    );
    await sendCdpKey('Escape', 'Escape', 27, 0);
    await waitForCondition(
      cdp,
      sessionId,
      'Step 6d: Single Escape exits .hud-maximized AND disarms Ruler synchronously',
      `(() => {
        const st = window.__chronosUiState;
        const panel = document.getElementById('hudPanel');
        return {
          ok: Boolean(
            st &&
              st.viewport === 'DOCKED' &&
              st.pointerTool === 'INSPECT_TARGET' &&
              panel &&
              !panel.classList.contains('hud-maximized') &&
              panel.dataset.viewport === 'DOCKED' &&
              panel.dataset.pointerTool === 'INSPECT_TARGET'
          ),
        };
      })()`,
      4000
    );
    await cdp.evalJs(
      sessionId,
      `(() => {
        const panel = document.getElementById('hudPanel');
        if (panel && '__e2eOrigRequestFullscreen' in window) {
          if (window.__e2eHasOwnReqFs) panel.requestFullscreen = window.__e2eOrigRequestFullscreen;
          else delete panel.requestFullscreen;
          delete window.__e2eHasOwnReqFs;
          delete window.__e2eOrigRequestFullscreen;
        }
        document.getElementById('btnCalClear')?.click();
        return true;
      })()`
    );

    console.log(
      `  [PASS] Step 6d (Wave-2 DeckUiFsm & Viewport Gate): heroToggle=${heroHiddenState.text}->ON (readout=${hiddenHeroInspect.readout}, pip=${hiddenHeroPipBytes}B), soloWarn=${soloWarnState.hiddenSolo}, fs=${fsEntered.nativeFs ? 'native' : 'maximized'}(${fsEntered.w}x${fsEntered.h}, aspectErr=${fsEntered.aspectErr.toFixed(4)}), keyGuards=OK, calExclusion=OK, CLS=${uiFsmCls.toFixed(6)}`
    );

    // 7. Server & Browser Hardening Regression Gates (Range FD/Suffix, Multi-Hop HLS SSRF, 429 Cap, Esc-After-Blur, Horizon Homography, MSE Finite Retry)
    {
      const baseUrl = `http://127.0.0.1:${srv.port}`;
      const countFds = () => {
        if (process.platform === 'win32') return 0;
        if (process.platform === 'linux') {
          try {
            return fs.readdirSync('/proc/self/fd').length;
          } catch {
            return 0;
          }
        }
        try {
          return parseInt(execSync(`lsof -p ${process.pid} 2>/dev/null | wc -l`, { encoding: 'utf8' }).trim(), 10) || 0;
        } catch {
          return 0;
        }
      };

      // 7a. Aborted Range FD-leak check + RFC 7233 suffix Range (-256) & invalid Range (bytes=- -> 416)
      await (await fetch(`${baseUrl}/index.html`)).arrayBuffer();
      const fdBefore = countFds();
      for (let i = 0; i < 25; i++) {
        await new Promise((resolve) => {
          const r = http.get(
            `${baseUrl}/samples/roundabout_occlusion.mp4`,
            { headers: { Range: 'bytes=0-' } },
            (res) => {
              res.once('data', () => {
                r.destroy();
                resolve();
              });
            }
          );
          r.on('error', () => resolve());
        });
      }
      await new Promise((r) => setTimeout(r, 120));
      const fdDelta = countFds() - fdBefore;
      if (fdBefore > 0 && fdDelta > 3) {
        throw new Error(`Step 7a: Expected ~0 FD leak across 25 aborted Range streams, got +${fdDelta}`);
      }
      const mp4Disk = fs.readFileSync(path.join(WEB_DIR, 'samples', 'roundabout_occlusion.mp4'));
      const sufRes = await fetch(`${baseUrl}/samples/roundabout_occlusion.mp4`, { headers: { Range: 'bytes=-256' } });
      const sufBuf = Buffer.from(await sufRes.arrayBuffer());
      if (sufRes.status !== 206 || sufBuf.length !== 256 || !sufBuf.equals(mp4Disk.subarray(mp4Disk.length - 256))) {
        throw new Error(`Step 7a: Suffix Range bytes=-256 failed (status=${sufRes.status}, len=${sufBuf.length})`);
      }
      const badRangeRes = await fetch(`${baseUrl}/samples/roundabout_occlusion.mp4`, { headers: { Range: 'bytes=-' } });
      if (badRangeRes.status !== 416) {
        throw new Error(`Step 7a: Expected 416 on Range: bytes=-, got ${badRangeRes.status}`);
      }

      // 7b. Host rebinding, Cross-Site Fetch, Loopback URL, Multi-Hop HLS Segment 302 SSRF Block, and 429 Relay Cap
      const badHostStatus = await new Promise((resolve, reject) => {
        const r = http.get(
          { host: '127.0.0.1', port: srv.port, path: '/api/live-presets', headers: { Host: '127.0.0.1.nip.io' } },
          (res) => { res.resume(); resolve(res.statusCode); }
        );
        r.on('error', reject);
      });
      const csrfRes = await fetch(`${baseUrl}/api/live-presets?refresh=1`, { headers: { 'Sec-Fetch-Site': 'cross-site' } });
      const loopRes = await fetch(`${baseUrl}/api/live-stream?url=${encodeURIComponent('http://127.0.0.1:8080/index.html')}`);
      const proxyDirectRes = await fetch(`${baseUrl}/__chronos_upstream_proxy/invalid/media.m3u8?u=https://example.com/a.m3u8`);
      if (badHostStatus !== 403 || csrfRes.status !== 403 || loopRes.status !== 400 || proxyDirectRes.status !== 403) {
        throw new Error(`Step 7b: Security status mismatch (${badHostStatus}/${csrfRes.status}/${loopRes.status}/${proxyDirectRes.status})`);
      }

      let internalHits = 0;
      const internalSecretSrv = http.createServer((_req, res) => {
        internalHits++;
        res.writeHead(200, { 'Content-Type': 'video/mp2t' });
        res.end('SECRET_INTERNAL_PAYLOAD');
      });
      await new Promise((r) => internalSecretSrv.listen(0, '127.0.0.1', r));
      const internalPort = internalSecretSrv.address().port;

      let publicMockPort = 0;
      const publicMockHlsSrv = http.createServer((req, res) => {
        if (req.url === '/master.m3u8') {
          res.writeHead(200, { 'Content-Type': 'application/vnd.apple.mpegurl' });
          res.end(
            `#EXTM3U\n` +
            `#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="a1",NAME="eng",URI=http://127.0.0.1:${publicMockPort}/split_diff_playlist\n` +
            `#EXT-X-STREAM-INF:BANDWIDTH=800000,AUDIO="a1"\nhttp://127.0.0.1:${publicMockPort}/split_diff_playlist\n`
          );
          return;
        }
        if (req.url === '/split_diff_playlist') {
          // Combines all 3 parser-differential vectors: <7B first TCP chunk ('#EX'), bare-\r lines, and unquoted/single-quoted URI=
          res.socket?.setNoDelay(true);
          res.writeHead(200, { 'Content-Type': 'text/plain' });
          res.write('#EX');
          setTimeout(() => {
            res.end(
              `TM3U\r#EXT-X-VERSION:6\r#EXT-X-TARGETDURATION:2\r` +
              `#EXT-X-MAP:URI=http://127.0.0.1:${internalPort}/unquoted_map.mp4\r` +
              `#EXT-X-KEY:METHOD=AES-128,URI='http://127.0.0.1:${internalPort}/single_quote.key'\r` +
              `#EXTINF:2.0,\rhttp://127.0.0.1:${publicMockPort}/seg_redirect.ts\r` +
              `#EXTINF:2.0,\rhttp://127.0.0.1:${internalPort}/barecr_secret.ts\r#EXT-X-ENDLIST\r`
            );
          }, 20);
          return;
        }
        if (req.url === '/seg_redirect.ts') {
          res.writeHead(302, { Location: `http://127.0.0.1:${internalPort}/redirected_secret.ts` });
          res.end();
          return;
        }
        res.writeHead(404);
        res.end();
      });
      await new Promise((r) => publicMockHlsSrv.listen(0, '127.0.0.1', r));
      publicMockPort = publicMockHlsSrv.address().port;

      const capSrv = await startChronosServer({
        rootDir: WEB_DIR,
        port: 0,
        host: '127.0.0.1',
        maxLiveProcs: 1,
        testAllowHostPorts: [`127.0.0.1:${publicMockPort}`],
      });
      try {
        const capBase = `http://127.0.0.1:${capSrv.port}`;
        const directDiffRes = await fetch(
          `${capBase}/api/live-stream?url=${encodeURIComponent(`http://127.0.0.1:${publicMockPort}/split_diff_playlist`)}`
        );
        await directDiffRes.arrayBuffer().catch(() => {});
        if ((directDiffRes.status !== 400 && directDiffRes.status !== 403) || internalHits !== 0) {
          throw new Error(`Step 7b: Direct parser-differential HLS preflight failed (status=${directDiffRes.status}, internalHits=${internalHits})`);
        }

        const ssrfHlsRes = await fetch(
          `${capBase}/api/live-stream?url=${encodeURIComponent(`http://127.0.0.1:${publicMockPort}/master.m3u8`)}`
        );
        await ssrfHlsRes.arrayBuffer().catch(() => {});
        if (internalHits !== 0) {
          throw new Error(`Step 7b: Multi-hop HLS segment/parser-differential SSRF leaked ${internalHits} request(s) to internal server!`);
        }

        // Verify maxLiveProcs=1 -> 429 Too Many Requests on concurrent relay
        const holdAbort = new AbortController();
        const holdPromise = fetch(`${capBase}/api/live-stream?preset=sim_live_traffic`, { signal: holdAbort.signal });
        const holdRes = await holdPromise;
        const holdReader = holdRes.body.getReader();
        await holdReader.read();
        const secondRes = await fetch(`${capBase}/api/live-stream?preset=sim_live_traffic`);
        await secondRes.arrayBuffer().catch(() => {});
        holdAbort.abort();
        await holdReader.cancel().catch(() => {});
        if (secondRes.status !== 429) {
          throw new Error(`Step 7b: Expected 429 when maxLiveProcs=1 exceeded, got ${secondRes.status}`);
        }
      } finally {
        await capSrv.close();
        publicMockHlsSrv.close();
        internalSecretSrv.close();
      }

      // 7c. Browser UI Esc-After-Blur Preservation & (0,0)-Horizon 4-Pt Homography
      await cdp.evalJs(
        sessionId,
        `(() => {
          document.getElementById('calRulerM').value = '1.00';
          document.getElementById('calRulerM').dispatchEvent(new Event('change', { bubbles: true }));
          document.getElementById('calKeyboardDisclosure').open = true;
          document.getElementById('calCoordMode').value = 'ruler';
          document.getElementById('calCoordMode').dispatchEvent(new Event('change', { bubbles: true }));
          document.getElementById('coordX1').value = '0.20';
          document.getElementById('coordY1').value = '0.50';
          document.getElementById('coordX2').value = '0.80';
          document.getElementById('coordY2').value = '0.50';
          document.getElementById('calCoordForm').requestSubmit();
          return true;
        })()`
      );
      await waitForCondition(
        cdp,
        sessionId,
        '1.00 m SURVEYED RULER committed',
        `(() => {
          const badge = document.getElementById('calBadge')?.textContent || '';
          const live = document.getElementById('calStatusLive')?.textContent || '';
          return { ok: badge === 'SURVEYED RULER' && live.includes('1.00 m'), badge, live };
        })()`
      );
      // Type 2.50 in #calRulerM, blur it by focusing+clicking #btnCalRuler, then press Escape!
      await cdp.evalJs(
        sessionId,
        `(() => {
          const inp = document.getElementById('calRulerM');
          inp.focus();
          inp.value = '2.50';
          inp.dispatchEvent(new Event('input', { bubbles: true }));
          inp.dispatchEvent(new Event('change', { bubbles: true }));
          const armBtn = document.getElementById('btnCalRuler');
          inp.blur();
          armBtn.focus();
          armBtn.click();
          return true;
        })()`
      );
      await waitForCondition(
        cdp,
        sessionId,
        'Arm 2-Pt Ruler active after input blur while keeping 1.00 m active',
        `(() => ({ ok: document.getElementById('btnCalRuler')?.getAttribute('aria-pressed') === 'true' }))()`
      );
      await cdp.evalJs(
        sessionId,
        `document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); true;`
      );
      const escBlurState = await waitForCondition(
        cdp,
        sessionId,
        'Esc-after-blur preserves 1.00 m SURVEYED RULER untouched',
        `(() => {
          const badge = document.getElementById('calBadge')?.textContent || '';
          const live = document.getElementById('calStatusLive')?.textContent || '';
          return { ok: badge === 'SURVEYED RULER' && live.includes('1.00 m') && !live.includes('2.50'), badge, live };
        })()`
      );

      // Submit exact (0,0)-horizon ground quad TL(.35,.3) TR(.65,.3) BR(.95,.9) BL(.05,.9)
      await cdp.evalJs(
        sessionId,
        `(() => {
          document.getElementById('calKeyboardDisclosure').open = true;
          const sel = document.getElementById('calCoordMode');
          sel.value = 'homography';
          sel.dispatchEvent(new Event('change', { bubbles: true }));
          document.getElementById('coordX1').value = '0.35'; document.getElementById('coordY1').value = '0.30';
          document.getElementById('coordX2').value = '0.65'; document.getElementById('coordY2').value = '0.30';
          document.getElementById('coordX3').value = '0.95'; document.getElementById('coordY3').value = '0.90';
          document.getElementById('coordX4').value = '0.05'; document.getElementById('coordY4').value = '0.90';
          document.getElementById('calCoordForm').requestSubmit();
          return true;
        })()`
      );
      const zeroHorizonState = await waitForCondition(
        cdp,
        sessionId,
        '(0,0)-horizon 4-pt homography solves cleanly in browser',
        `(() => {
          const badge = document.getElementById('calBadge')?.textContent || '';
          return { ok: badge === 'SURVEYED 4-PT H', badge };
        })()`
      );
      await cdp.evalJs(sessionId, `document.getElementById('btnCalClear').click(); true;`);

      // 7d. Truncated MSE stream finite retry cap (verifies no infinite reconnect loop on 200 + immediate EOF)
      allowNegativeNetworkLogs = true;
      let mseTruncHits = 0;
      cdp.on('Fetch.requestPaused', (params) => {
        mseTruncHits++;
        cdp.send(
          'Fetch.fulfillRequest',
          {
            requestId: params.requestId,
            responseCode: 200,
            responseHeaders: [{ name: 'Content-Type', value: 'video/mp4' }],
            body: Buffer.alloc(64).toString('base64'),
          },
          sessionId
        ).catch(() => {});
      });
      await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*__e2e_neg_mse*' }] }, sessionId);
      await cdp.evalJs(
        sessionId,
        `(() => {
          document.getElementById('btnSrcLiveStream').click();
          document.getElementById('tier3CustomHlsUrl').value = 'https://example.com/__e2e_neg_mse.m3u8';
          document.getElementById('btnStartTier3Mse').click();
          return true;
        })()`
      );
      const mseBoundedState = await waitForCondition(
        cdp,
        sessionId,
        'Truncated MSE stream stops after 3 bounded retries without looping forever',
        `(() => {
          const txt = document.getElementById('liveStreamMeta')?.textContent || '';
          return { ok: txt.includes('STREAM ENDED'), txt };
        })()`,
        10000
      );
      await new Promise((r) => setTimeout(r, 350));
      if (mseTruncHits !== 4) {
        throw new Error(`Step 7d: Expected exactly 4 MSE fetches (1 initial + 3 retries) on truncated EOF, got ${mseTruncHits}`);
      }
      await cdp.send('Fetch.disable', {}, sessionId);
      await cdp.evalJs(sessionId, `document.getElementById('btnStopLiveFeed').click(); true;`);
      allowNegativeNetworkLogs = false;

      console.log(`  [PASS] Step 7 (Security & Hardening Gates): FD_delta=${fdDelta}, multiHopSSRF_hits=${internalHits}, 429=ok, EscBlur="${escBlurState.live}", (0,0)-H=${zeroHorizonState.badge}, MSE_BoundedRetries=${mseTruncHits} (${mseBoundedState.txt})`);
    }

    // 8. Option A Showcase Landing Surface, Split-Wipe, Local MP4 Drop, Zero-Reload Deck Switch & Static-Host Gate
    {
      const baseUrl = `http://127.0.0.1:${srv.port}`;
      const { targetId: scTargetId } = await cdp.send('Target.createTarget', { url: `${baseUrl}/index.html` });
      const { sessionId: scSession } = await cdp.send('Target.attachToTarget', { targetId: scTargetId, flatten: true });
      try {
        await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: true }, scSession);
        await cdp.send('Page.enable', {}, scSession);
        await cdp.send('Log.enable', {}, scSession);
        await cdp.send('Runtime.enable', {}, scSession);
        await cdp.send(
          'Emulation.setDeviceMetricsOverride',
          { width: 1440, height: 900, deviceScaleFactor: process.env.CI ? 1 : 2, mobile: false },
          scSession
        );

        await waitForCondition(
          cdp,
          scSession,
          'Step 8a: Showcase DOM booted cleanly',
          `(() => ({
            ok: document.body?.dataset?.surface === 'showcase' && Boolean(document.getElementById('showcaseWipeSlider'))
          }))()`,
          8000
        );
        await cdp.evalJs(
          scSession,
          `(() => {
            window.__e2eShowcaseCls = 0;
            window.__e2eShowcaseClsPo = new PerformanceObserver((list) => {
              for (const e of list.getEntries()) {
                if (!e.hadRecentInput) window.__e2eShowcaseCls += e.value;
              }
            });
            window.__e2eShowcaseClsPo.observe({ type: 'layout-shift', buffered: false });
            return true;
          })()`
        );

        const starlingBoot = await waitForCondition(
          cdp,
          scSession,
          'Step 8a: Showcase frame-0 Starling Murmuration (>= 1,000 confirmed, HERO 16 hidden, 28% wipe)',
          `(() => {
            const surface = document.body?.dataset?.surface || '';
            const btnStar = document.getElementById('btnHeroStarlings');
            const wipeSlider = document.getElementById('showcaseWipeSlider');
            const kernelTxt = (document.getElementById('scValKernel')?.textContent || '').trim();
            const tracksTxt = (document.getElementById('scValTracks')?.textContent || '').trim();
            const confRaw = tracksTxt.split(' ')[0] || '0';
            const confirmed = parseInt(confRaw.replace(/,/g, ''), 10) || 0;
            const st = window.__chronosUiState;
            return {
              ok: surface === 'showcase' &&
                  btnStar?.getAttribute('aria-pressed') === 'true' &&
                  confirmed >= 1000 &&
                  st?.overlays?.showHeroTargets === false &&
                  wipeSlider?.value === '28' &&
                  kernelTxt.endsWith('ms / frame') &&
                  !kernelTxt.startsWith('--'),
              surface,
              confirmed,
              kernelTxt,
              tracksTxt,
              wipe: wipeSlider?.value,
            };
          })()`,
          26000
        );

        const scHudRect = await cdp.evalJs(
          scSession,
          `(() => {
            const el = document.getElementById('hudCanvas');
            if (!el) return null;
            el.scrollIntoView({ block: 'center', inline: 'center' });
            const r = el.getBoundingClientRect();
            return {
              x: Math.max(0, Math.floor(r.left + (window.scrollX || 0))),
              y: Math.max(0, Math.floor(r.top + (window.scrollY || 0))),
              width: Math.ceil(r.width),
              height: Math.ceil(r.height),
            };
          })()`
        );
        const scHudShot = await cdp.send(
          'Page.captureScreenshot',
          { format: 'png', captureBeyondViewport: false, clip: { ...scHudRect, scale: 1 } },
          scSession
        );
        const scHudBytes = scHudShot?.data ? Buffer.from(scHudShot.data, 'base64').byteLength : 0;
        if (scHudBytes < 2500) {
          throw new Error(`Step 8a: Expected Showcase #hudCanvas screenshot >= 2500 B, got ${scHudBytes} B`);
        }
        await cdp.saveScreenshot(scSession, proofDir, 'proof_showcase_starlings.png');

        // 8b. Interactive Split-Wipe (55%) + Right-Side Click-to-Inspect
        const wipeState = await cdp.evalJs(
          scSession,
          `(() => {
            const s = document.getElementById('showcaseWipeSlider');
            s.value = '55';
            s.dispatchEvent(new Event('input', { bubbles: true }));
            return {
              val: s.value,
              aria: s.getAttribute('aria-valuetext') || '',
              readout: document.getElementById('showcaseWipeReadout')?.textContent || '',
            };
          })()`
        );
        if (
          wipeState.val !== '55' ||
          wipeState.aria !== '55% raw video, 45% spectral wavefield' ||
          !wipeState.readout.includes('55% RAW')
        ) {
          throw new Error(`Step 8b: Split-wipe sync mismatch: ${JSON.stringify(wipeState)}`);
        }

        let scInspectText = '';
        for (const [nx, ny] of [[0.67, 0.46], [0.65, 0.58], [0.62, 0.40], [0.76, 0.68], [0.78, 0.30], [0.70, 0.50], [0.74, 0.55]]) {
          await cdp.evalJs(
            scSession,
            `(() => {
              const hud = document.getElementById('hudCanvas');
              const r = hud.getBoundingClientRect();
              hud.dispatchEvent(new MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                clientX: r.left + r.width * ${nx},
                clientY: r.top + r.height * ${ny},
              }));
              return true;
            })()`
          );
          try {
            const hit = await waitForCondition(
              cdp,
              scSession,
              `Step 8b: Right-side inspect (${nx},${ny})`,
              `(() => {
                const txt = (document.getElementById('scValInspect')?.textContent || '').trim();
                return { ok: /^INSP\\s+(?:μ|u)T\\d+/i.test(txt), txt };
              })()`,
              1600
            );
            if (hit && hit.ok) {
              scInspectText = hit.txt;
              break;
            }
          } catch (_e) {
            // Try next coordinate on right half
          }
        }
        if (!scInspectText) {
          throw new Error('Step 8b: Clicking right-side Spectral Wavefield did not inspect an active track (#scValInspect)');
        }

        // 8c. 1-Click Counter-Proof: Model Rockets
        await cdp.evalJs(scSession, `document.getElementById('btnHeroRockets')?.click(); true;`);
        const rocketHero = await waitForCondition(
          cdp,
          scSession,
          'Step 8c: Showcase 1-click Model Rockets counter-proof',
          `(() => {
            const btnR = document.getElementById('btnHeroRockets');
            const btnS = document.getElementById('btnHeroStarlings');
            const selVal = document.getElementById('sampleVideoSelect')?.value || '';
            const cal = document.getElementById('calBadge')?.textContent || '';
            const wipe = document.getElementById('showcaseWipeSlider')?.value || '';
            const st = window.__chronosUiState;
            const tgts = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
            return {
              ok: btnR?.getAttribute('aria-pressed') === 'true' &&
                  btnS?.getAttribute('aria-pressed') === 'false' &&
                  selVal.endsWith('model_rocket_full_flights.mp4') &&
                  st?.overlays?.showHeroTargets === true &&
                  cal === 'ASSUMED RULER' &&
                  wipe === '0' &&
                  tgts >= 1,
              selVal,
              cal,
              wipe,
              tgts,
            };
          })()`,
          15000
        );
        await cdp.saveScreenshot(scSession, proofDir, 'proof_showcase_rockets.png');

        // 8d. Zero-Upload Drag-and-Drop Local MP4 onto #hudPanel
        await cdp.evalJs(
          scSession,
          `(async () => {
            const resp = await fetch('./samples/juggling_ballistic.mp4');
            const blob = await resp.blob();
            const file = new File([blob], 'user_clip.mp4', { type: 'video/mp4' });
            const dt = new DataTransfer();
            dt.items.add(file);
            document.getElementById('hudPanel').dispatchEvent(
              new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: dt })
            );
            return true;
          })()`
        );
        const dropState = await waitForCondition(
          cdp,
          scSession,
          'Step 8d: Drag-and-drop user_clip.mp4 tracks locally with scene pills cleared',
          `(() => {
            const srcLbl = document.getElementById('hudSourceLabel')?.textContent || '';
            const tgts = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
            const anyPressed = Array.from(document.querySelectorAll('.hero-scene-btn')).some(
              (b) => b.getAttribute('aria-pressed') === 'true'
            );
            return {
              ok: srcLbl.includes('user_clip.mp4') && !anyPressed && tgts >= 1,
              srcLbl,
              tgts,
              anyPressed,
            };
          })()`,
          15000
        );

        // 8e. Zero-Reload Switch to Full 15-Panel Command Deck & Back
        const seqBeforeSwitch = await cdp.evalJs(scSession, `window.__chronosUiState?.seq ?? -1`);
        await cdp.evalJs(scSession, `document.getElementById('btnOpenFullDeck')?.click(); true;`);
        const deckSwitched = await waitForCondition(
          cdp,
          scSession,
          'Step 8e: Seamless zero-reload transition to Full Command Deck',
          `(() => {
            const surface = document.body?.dataset?.surface || '';
            const roster = document.querySelector('.roster-panel');
            const seq = window.__chronosUiState?.seq ?? -1;
            return {
              ok: surface === 'deck' &&
                  location.search.includes('view=deck') &&
                  Boolean(roster && roster.offsetParent !== null) &&
                  seq >= ${seqBeforeSwitch},
              surface,
              search: location.search,
              seq,
            };
          })()`,
          6000
        );
        await cdp.evalJs(scSession, `document.getElementById('btnReturnShowcase')?.click(); true;`);
        const returnedShowcase = await waitForCondition(
          cdp,
          scSession,
          'Step 8e: Return transition back to Split-Wipe Showcase',
          `(() => ({
            ok: document.body?.dataset?.surface === 'showcase' && !location.search.includes('view=deck'),
            cls: window.__e2eShowcaseCls || 0,
          }))()`,
          6000
        );
        await cdp.evalJs(scSession, `if (window.__e2eShowcaseClsPo) window.__e2eShowcaseClsPo.disconnect(); true;`);
        if (returnedShowcase.cls >= 0.0001) {
          throw new Error(`Step 8e: Expected 0.000000 CLS across Showcase surface flows, got ${returnedShowcase.cls}`);
        }

        // 8f. Static-Host Mode (?static=1) & /deck.html Redirect
        await cdp.send('Page.navigate', { url: `${baseUrl}/index.html?static=1` }, scSession);
        const staticState = await waitForCondition(
          cdp,
          scSession,
          'Step 8f: Static-host mode active (?static=1) without API 404s',
          `(() => ({
            ok: document.body?.dataset?.staticHost === 'true' && document.body?.dataset?.surface === 'showcase',
            staticHost: document.body?.dataset?.staticHost,
          }))()`,
          8000
        );
        await new Promise((r) => setTimeout(r, 250));

        await cdp.send('Page.navigate', { url: `${baseUrl}/deck.html` }, scSession);
        await waitForCondition(
          cdp,
          scSession,
          'Step 8f: /deck.html redirects cleanly to ?view=deck Command Deck',
          `(() => ({
            ok: location.search.includes('view=deck') && document.body?.dataset?.surface === 'deck',
          }))()`,
          8000
        );

        console.log(
          `  [PASS] Step 8 (Option A Showcase, Split-Wipe & Static-Host Gate): starlings=${starlingBoot.confirmed} (${starlingBoot.kernelTxt}, hud=${scHudBytes}B), wipe55%="${wipeState.readout}", inspect="${scInspectText}", rockets=${rocketHero.cal}(${rocketHero.tgts} tgts), drop="${dropState.srcLbl}", deckSwitchSeq=${seqBeforeSwitch}->${deckSwitched.seq}, staticHost=${staticState.staticHost}, CLS=${returnedShowcase.cls.toFixed(6)}`
        );
      } finally {
        await cdp.send('Target.closeTarget', { targetId: scTargetId }).catch(() => {});
      }
    }

    if (pageErrors.length > 0) {
      throw new Error(`Browser reported ${pageErrors.length} error(s):\n${pageErrors.join('\n')}`);
    }
    console.log('  [PASS] Zero uncaught page/worker exceptions and zero console errors.');
  } finally {
    if (cdp) cdp.close();
    if (chromeProc) chromeProc.kill('SIGKILL');
    await srv.close();
    try {
      fs.rmSync(tmpDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 80 });
    } catch (_e) {
      // ignore transient macOS Chrome helper file lock during teardown
    }
  }
}

main().catch((err) => {
  console.error('E2E FAILED:', err.message || err);
  process.exit(1);
});
