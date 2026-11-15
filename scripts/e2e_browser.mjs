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
  const deadline = Date.now() + timeoutMs;
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
  const appUrl = `http://127.0.0.1:${srv.port}/index.html`;
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
      { width: 1440, height: 900, deviceScaleFactor: 2, mobile: false },
      sessionId
    );
    await cdp.send('Browser.setDownloadBehavior', {
      behavior: 'allow',
      downloadPath: downloadDir,
    });

    // 1. Verify Boot + 288-Page Arena + 320x180 SD Blind Pillar Testbench (0 Oracle swaps)
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
              arena.includes('288 PAGES / 18.00 MB') &&
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
        return true;
      })()`
    );
    const mp4State = await waitForCondition(
      cdp,
      sessionId,
      'Juggling 25fps MP4 playback + multi-arc gravity + APERIODIC guard',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
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
    if (selectMeta.optgroups !== 2 || selectMeta.options !== 14) {
      throw new Error(`Expected 2 <optgroup> and 14 <option> elements in #sampleVideoSelect, got ${JSON.stringify(selectMeta)}`);
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
    const persistClipShot = await cdp.send(
      'Page.captureScreenshot',
      {
        format: 'png',
        captureBeyondViewport: true,
        clip: {
          x: Math.max(0, swarmState.pRect.x),
          y: Math.max(0, swarmState.pRect.y),
          width: swarmState.pRect.width,
          height: swarmState.pRect.height,
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
        return r ? { x: Math.floor(r.left), y: Math.floor(r.top), width: Math.ceil(r.width), height: Math.ceil(r.height) } : null;
      })()`
    );
    const phaseClipShot = await cdp.send(
      'Page.captureScreenshot',
      {
        format: 'png',
        captureBeyondViewport: true,
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

    // 7. Server & Browser Hardening Regression Gates (Range FD/Suffix, Multi-Hop HLS SSRF, 429 Cap, Esc-After-Blur, Horizon Homography, MSE Finite Retry)
    {
      const baseUrl = `http://127.0.0.1:${srv.port}`;
      const countFds = () => {
        if (process.platform === 'win32') return 0;
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
