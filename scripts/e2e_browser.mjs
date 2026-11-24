#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Zero-Dependency Headless Chrome CDP Full-Stack E2E Gate
// Exercises V8 Web Worker + OffscreenCanvas HUD, 448-page Wasm SIMD128 kernel,
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
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
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

async function waitForCondition(cdp, sessionId, description, jsExpr, timeoutMs = 14000, allowNavigation = false) {
  const ciTimeoutScale = Math.max(1.0, Math.min(2.0, Number(process.env.CHRONOS_CI_BUDGET_SCALE || '1.0') * 0.6));
  const deadline = Date.now() + Math.round(timeoutMs * ciTimeoutScale);
  let lastVal = null;
  while (Date.now() < deadline) {
    try {
      lastVal = await cdp.evalJs(sessionId, jsExpr);
      if (lastVal && lastVal.ok) return lastVal;
    } catch (err) {
      const msg = String(err && err.message ? err.message : err);
      if (
        allowNavigation &&
        /Inspected target navigated or closed|Execution context was destroyed|Cannot find context/i.test(msg)
      ) {
        lastVal = { navInFlight: msg };
      } else {
        throw err;
      }
    }
    await new Promise((r) => setTimeout(r, 90));
  }
  throw new Error(`Timeout waiting for [${description}]. Last state: ${JSON.stringify(lastVal)}`);
}

const SWARM_INSPECT_PROBE_COORDS = [
  [0.35, 0.42], [0.65, 0.58], [0.33, 0.54], [0.67, 0.46],
  [0.40, 0.35], [0.60, 0.65], [0.38, 0.60], [0.62, 0.40],
  [0.30, 0.50], [0.70, 0.50], [0.42, 0.62], [0.58, 0.38],
  [0.78, 0.30], [0.76, 0.68], [0.54, 0.26], [0.82, 0.48], [0.52, 0.72], [0.72, 0.24],
];

// Clicks noncentral points on #hudCanvas until one inspects a live Swarm track, then verifies the
// worker's ack echoes the clicked point and #swarmInspectReadout shows a fresh, different track
// located at that point. Three details keep this deterministic on slow or loaded machines:
//  - Chrome truncates MouseEvent clientX/clientY (synthetic and CDP Input alike) to whole CSS px, so
//    the click goes to a whole pixel and the expected request is derived from that pixel. A
//    fractional click could miss the 0.5 frame-px request tolerance by construction.
//  - The ack and the readout are captured together, in the microtask after the telemetry frame that
//    applies the ack (both are written in the same synchronous updateContinuumDomUi call). Polling
//    from CDP instead samples the readout hundreds of ms later under load, after the inspected
//    track has moved past the distance allowance.
//  - A click on empty sky, on a track that expired before the causal frame, or just after the
//    looping clip resets every track is not a product failure, so the probe clicks again (cycling
//    the coordinate list) until the budget runs out. Every match check is unchanged.
async function probeSwarmInspectOnHud(cdp, sessionId, label, opts = {}) {
  const {
    useCdpClick = false,
    attemptTimeoutMs = 1800,
    budgetMs = 20000,
    waitFor = waitForCondition,
  } = opts;
  const ciTimeoutScale = Math.max(1.0, Math.min(2.0, Number(process.env.CHRONOS_CI_BUDGET_SCALE || '1.0') * 0.6));
  const deadline = Date.now() + Math.round(budgetMs * ciTimeoutScale);
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
  const filteredCoords = SWARM_INSPECT_PROBE_COORDS.filter(([nx, ny]) => {
    if (Math.hypot(nx - 0.5, ny - 0.5) < 0.12) return false;
    if (before && before.nx !== null && before.ny !== null) {
      return Math.hypot(nx - before.nx, ny - before.ny) >= 0.15;
    }
    return true;
  });
  const coords = filteredCoords.length ? filteredCoords : SWARM_INSPECT_PROBE_COORDS;
  const rejects = {};
  let attempts = 0;
  let lastReject = null;
  for (let pass = 0; pass === 0 || Date.now() < deadline; pass++) {
    for (const [nx, ny] of coords) {
      if (pass > 0 && Date.now() >= deadline) break;
      attempts++;
      const token = `${Date.now()}-${attempts}`;
      const pt = await cdp.evalJs(
        sessionId,
        `(() => {
          const el = document.getElementById('swarmInspectReadout');
          const hud = document.getElementById('hudCanvas');
          if (!el || !hud) return null;
          const readout = (el.textContent || '').trim();
          const prevInspectSeq = window.__chronosUiState?.lastInspectAck?.inspectSeq ?? 0;
          if (window.__e2eInspectObs) window.__e2eInspectObs.disconnect();
          window.__e2eInspectCap = null;
          const obs = new MutationObserver(() => {
            const ack = window.__chronosUiState?.lastInspectAck;
            if (window.__e2eInspectCap || !ack || !Number.isInteger(ack.inspectSeq) || ack.inspectSeq <= prevInspectSeq) return;
            window.__e2eInspectCap = { token: ${JSON.stringify(token)}, ack: { ...ack }, readout: (el.textContent || '').trim() };
            obs.disconnect();
          });
          obs.observe(el, { attributes: true, attributeFilter: ['data-inspect-seq'], childList: true, characterData: true, subtree: true });
          window.__e2eInspectObs = obs;
          const r = hud.getBoundingClientRect();
          const x = Math.round(r.left + r.width * ${nx});
          const y = Math.round(r.top + r.height * ${ny});
          if (!${Boolean(useCdpClick)}) {
            hud.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, clientX: x, clientY: y }));
          }
          return { x, y, effNx: (x - r.left) / r.width, effNy: (y - r.top) / r.height, attemptBeforeReadout: readout, prevInspectSeq };
        })()`
      );
      if (!pt) continue;
      if (useCdpClick) {
        await cdp.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: pt.x, y: pt.y, button: 'left', clickCount: 1 }, sessionId);
        await cdp.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: pt.x, y: pt.y, button: 'left', clickCount: 1 }, sessionId);
      }
      let res = null;
      try {
        res = await waitFor(
          cdp,
          sessionId,
          `${label} (${nx},${ny})`,
          `(() => {
            const cap = window.__e2eInspectCap;
            if (!cap || cap.token !== ${JSON.stringify(token)}) {
              return { ok: false, inspectSeq: window.__chronosUiState?.lastInspectAck?.inspectSeq ?? 0 };
            }
            const { ack, readout } = cap;
            const mc = readout.match(/^INSP\\s+(?:μ|u)T([1-9]\\d*)(?:\\s*·\\s*B\\d+)?\\s*·\\s*\\(([-\\d.]+),\\s*([-\\d.]+)\\)/i);
            if (!mc) {
              return { ok: true, matched: false, reason: 'status:' + ack.inspectStatus, readout, tid: 0, inspectSeq: ack.inspectSeq };
            }
            const tid = parseInt(mc[1], 10);
            const effNx = ${pt.effNx};
            const effNy = ${pt.effNy};
            const ackMatchesReq =
              Number.isFinite(ack.inspectReqNormX) &&
              Number.isFinite(ack.inspectReqNormY) &&
              Number.isFinite(ack.inspectReqPxX) &&
              Number.isFinite(ack.inspectReqPxY) &&
              Math.abs(ack.inspectReqNormX - effNx) <= 0.005 &&
              Math.abs(ack.inspectReqNormY - effNy) <= 0.005 &&
              Math.abs(ack.inspectReqPxX - effNx * ${before.fw}) <= 0.5 &&
              Math.abs(ack.inspectReqPxY - effNy * ${before.fh}) <= 0.5 &&
              ack.inspectedTid === tid &&
              ['live', 'pinned'].includes(ack.inspectStatus);
            if (!ackMatchesReq) {
              return { ok: true, matched: false, reason: 'ack-mismatch:' + ack.inspectStatus, readout, tid: 0, inspectSeq: ack.inspectSeq, ack };
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
            const matched = tid > 0 && coordMatch && freshReadout && shiftedTarget;
            const reason = matched ? 'matched' : !coordMatch ? 'coord:' + ack.inspectStatus : !freshReadout ? 'stale-readout' : 'same-target';
            return { ok: true, matched, reason, readout, tid, ixNorm, iyNorm, inspectSeq: ack.inspectSeq, pxDist, maxAllowedDist };
          })()`,
          attemptTimeoutMs
        );
      } catch (err) {
        if (/^Page JS exception/.test(String(err && err.message))) throw err;
        res = { matched: false, reason: 'no-ack' };
      }
      if (res && res.matched) return { ...res, attempts };
      rejects[res.reason] = (rejects[res.reason] || 0) + 1;
      lastReject = { nx, ny, ...res };
    }
  }
  throw new Error(
    `${label}: #swarmInspectReadout did not lock onto a fresh track with matching noncentral coordinates ` +
      `(attempts=${attempts}, rejects=${JSON.stringify(rejects)}, last=${JSON.stringify(lastReject)})`
  );
}

// Step 10e native gate: fake webcam + Web Audio capture into the RS+Welch chain on a deck page.
async function verifyNativeWebcamAudioGate(cdp, sessionId) {
  const runNativeWebcamWindow = () => cdp.evalJs(
    sessionId,
    `(async () => {
      const origPostMessage = Worker.prototype.postMessage;
      const origGetUserMedia = navigator.mediaDevices && navigator.mediaDevices.getUserMedia
        ? navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
        : null;
        const nativePushMsgs = [];
        const nativeAcks = [];
        const nativeAckWorkers = new WeakSet();
        let nativeCapturing = true;
        const capturedStreams = [];
        // Independent witness for dropped frames: count webcam frames Chrome presented that no
        // rVFC callback saw (presentedFrames jumps), the same evidence the product flags as FRAME_GAP.
        const vidEl = document.getElementById('hiddenVideoSource');
        let rvfcWatching = false;
        let rvfcLastPresented = -1;
        let browserSkippedFrames = 0;
        const onWitnessFrame = (_now, meta) => {
          if (!rvfcWatching) return;
          if (rvfcLastPresented >= 0 && meta.presentedFrames > rvfcLastPresented + 1) {
            browserSkippedFrames += meta.presentedFrames - rvfcLastPresented - 1;
          }
          rvfcLastPresented = meta.presentedFrames;
          vidEl.requestVideoFrameCallback(onWitnessFrame);
        };
        try {
          if (!origGetUserMedia) {
            return { unsupported: true };
          }
          navigator.mediaDevices.getUserMedia = async (constraints) => {
            const st = await origGetUserMedia(constraints);
            capturedStreams.push(st);
            return st;
          };
          Worker.prototype.postMessage = function (data) {
            if (nativeCapturing && data && data.type === 'push_audio_pcm') {
              if (!nativeAckWorkers.has(this)) {
                nativeAckWorkers.add(this);
                this.addEventListener('message', (ev) => {
                  const d = ev && ev.data;
                  if (d && d.type === 'wave2_audio_ack' && typeof d.reqId === 'string' && d.reqId.startsWith('e2e-native-')) {
                    nativeAcks.push(d);
                  }
                });
              }
              data.wantAck = true;
              data.reqId = 'e2e-native-' + nativePushMsgs.length;
              nativePushMsgs.push({
                tEndUs: data.tEndUs,
                estSkewUs: data.estSkewUs,
                len: data.samples ? data.samples.length : 0,
              });
            }
            return origPostMessage.apply(this, arguments);
          };

          const rmSel = document.getElementById('rsVibroModeSelect');
          if (rmSel) {
            rmSel.value = '0';
            rmSel.dispatchEvent(new Event('change', { bubbles: true }));
          }
          await new Promise((r) => setTimeout(r, 40));
          document.getElementById('btnSrcWebcam')?.click();
          await new Promise((r) => setTimeout(r, 220));
          nativePushMsgs.length = 0;
          nativeAcks.length = 0;

          if (rmSel) {
            rmSel.value = '2';
            rmSel.dispatchEvent(new Event('change', { bubbles: true }));
          }
          if (vidEl && typeof vidEl.requestVideoFrameCallback === 'function') {
            rvfcWatching = true;
            vidEl.requestVideoFrameCallback(onWitnessFrame);
          }
          const deadline = performance.now() + 4500;
          while (performance.now() < deadline && nativePushMsgs.length < 40) {
            await new Promise((r) => setTimeout(r, 40));
          }
          nativeCapturing = false;
          rvfcWatching = false;
          const ackDeadline = performance.now() + 2500;
          while (nativeAcks.length < nativePushMsgs.length && performance.now() < ackDeadline) {
            await new Promise((r) => setTimeout(r, 15));
          }
          await new Promise((r) => setTimeout(r, 40));
          const nativeVibro = window.__chronosWave2Probe?.getLastWave2()?.vibro || null;
          if (rmSel) {
            rmSel.value = '0';
            rmSel.dispatchEvent(new Event('change', { bubbles: true }));
          }
          await new Promise((r) => setTimeout(r, 80));
          const nativePushed = nativePushMsgs.length;
          const pushedSamplesSum = nativePushMsgs.reduce((acc, m) => acc + m.len, 0);
          const pushedSkews = nativePushMsgs.map((m) => m.estSkewUs);
          const acceptedDelta = nativeAcks.reduce((acc, a) => acc + (a.accepted ? a.count : 0), 0);
          const liveAudioAfterOff = capturedStreams.reduce(
            (cnt, st) => cnt + st.getAudioTracks().filter((tr) => tr.readyState === 'live').length,
            0
          );
          return {
            unsupported: false,
            nativePushed,
            nativeAckCount: nativeAcks.length,
            pushedSamplesSum,
            acceptedDelta,
            nativeAcceptedAll:
              nativeAcks.length === nativePushed && acceptedDelta === pushedSamplesSum && pushedSamplesSum > 0,
            state: nativeVibro?.state ?? -1,
            reasonMask: nativeVibro?.reasonMask ?? -1,
            liveAudioAfterOff,
            maxAbsSkew: pushedSkews.reduce((mx, v) => Math.max(mx, Math.abs(v)), 0),
            browserSkippedFrames,
          };
      } finally {
        rvfcWatching = false;
        Worker.prototype.postMessage = origPostMessage;
        if (origGetUserMedia) {
          navigator.mediaDevices.getUserMedia = origGetUserMedia;
        }
        const rmSel = document.getElementById('rsVibroModeSelect');
        if (rmSel) {
          rmSel.value = '0';
          rmSel.dispatchEvent(new Event('change', { bubbles: true }));
        }
        document.getElementById('btnSrcTestbench')?.click();
        await new Promise((r) => setTimeout(r, 60));
      }
    })()`
  );
  const nativeWebcamGateFails = (r) =>
    !r ||
    r.unsupported ||
    r.nativePushed < 36 ||
    r.maxAbsSkew > 500 ||
    !r.nativeAcceptedAll ||
    r.state === 4 ||
    (r.reasonMask & 32) !== 0 ||
    r.liveAudioAfterOff !== 0;
  // A window in which Chrome itself dropped webcam frames (a starved main thread) cannot show a
  // gap-free RS+Welch chain: the product is right to report DISTURBED with only RS_REASON_FRAME_GAP.
  // Such a window, and only such a window, is re-run. Every other check must already have passed
  // in it, and any other disturbance reason, a skew, a rejected block, or a FRAME_GAP without
  // witnessed dropped frames fails at once.
  const isWitnessedFrameGapOnly = (r) =>
    Boolean(r) &&
    !r.unsupported &&
    r.state === 4 &&
    r.reasonMask === 8 &&
    r.browserSkippedFrames > 0 &&
    r.nativePushed >= 36 &&
    r.maxAbsSkew <= 500 &&
    r.nativeAcceptedAll &&
    r.liveAudioAfterOff === 0;
  const nativeWebcamAttempts = [];
  let nativeWebcamCheck = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    nativeWebcamCheck = await runNativeWebcamWindow();
    nativeWebcamAttempts.push(nativeWebcamCheck);
    if (!nativeWebcamGateFails(nativeWebcamCheck) || !isWitnessedFrameGapOnly(nativeWebcamCheck) || attempt === 3) break;
    console.log(
      `  [RETRY] Step 10e: capture window ${attempt}/3 was disturbed only by FRAME_GAP after Chrome dropped ${nativeWebcamCheck.browserSkippedFrames} webcam frame(s); taking a fresh window`
    );
  }
  if (nativeWebcamGateFails(nativeWebcamCheck)) {
    throw new Error('Step 10e native Chrome webcam + Web Audio live gate failed: ' + JSON.stringify(nativeWebcamAttempts));
  }
  return { ...nativeWebcamCheck, attempts: nativeWebcamAttempts.length };
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
              arena.includes('448 PAGES / 28.00 MB') &&
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
          ok: slotDimOk && activeSlots === 4 && pickSlot >= 0 && canvasRect && canvasRect.width >= 48 && canvasRect.height >= 24,
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
    if (slotPngBytes < 180) {
      throw new Error(`Expected active .slot-pip-canvas compositor screenshot >= 180 B, got ${slotPngBytes} B`);
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
        const optVals = new Set(sel ? Array.from(sel.querySelectorAll('option')).map((o) => o.value).filter(Boolean) : []);
        const recipes = window.__chronosUi?.recipes || [];
        const sampleRecipes = recipes.filter((r) => r?.source?.kind === 'sample');
        const missingRecipes = sampleRecipes
          .filter((r) => !optVals.has(r.source.url))
          .map((r) => ({ key: r.key, url: r.source.url }));
        return {
          optgroups: sel ? sel.querySelectorAll('optgroup').length : 0,
          options: sel ? sel.querySelectorAll('option').length : 0,
          recipeCount: recipes.length,
          sampleRecipeCount: sampleRecipes.length,
          missingRecipes,
        };
      })()`
    );
    if (
      selectMeta.optgroups !== 7 ||
      selectMeta.options !== 42 ||
      selectMeta.recipeCount !== 53 ||
      selectMeta.sampleRecipeCount < 41 ||
      selectMeta.missingRecipes.length > 0
    ) {
      throw new Error(`Expected 7 <optgroup>, 42 <option>, 53 recipes, and 0 missing recipe mappings in #sampleVideoSelect, got ${JSON.stringify(selectMeta)}`);
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
        const calStatus = document.getElementById('calStatusLive')?.textContent || '';
        const calValid =
          badge === 'ASSUMED RULER' ||
          (badge === 'PX UNITS' && calStatus.includes('CAMERA PTZ PAN'));
        const speedTxt = document.getElementById('kpiKinSpeed')?.textContent || '';
        const angTxt = document.getElementById('kpiKinAngle')?.textContent || '';
        const noNan = ![speedTxt, angTxt].some((s) => s.includes('NaN'));
        return {
          ok: v && Number.isFinite(v.duration) && v.duration >= 29.5 && !v.paused &&
              btn?.getAttribute('aria-pressed') === 'true' &&
              activeCount >= 1 &&
              calValid &&
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
      `window.__chronosUi?.reveal('physicsPhaseCanvas'); document.getElementById('physicsPhaseCanvas')?.scrollIntoView({ block: 'center' }); true;`
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
        `window.__chronosUi?.reveal(${JSON.stringify(elementId)}); document.getElementById(${JSON.stringify(elementId)})?.scrollIntoView({ block: 'center', inline: 'center' }); true;`
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
      const inspectState = await probeSwarmInspectOnHud(
        cdp,
        sessionId,
        `Step 6c (${cfg.id}): fresh #swarmInspectReadout + inspectSeq + noncentral coordinate`,
        { attemptTimeoutMs: 2000 }
      );

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
          if (el.id) window.__chronosUi?.reveal(el.id);
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

    await cdp.evalJs(
      sessionId,
      `(() => {
        if (window.__chronosUiState?.overlays?.showHeroTargets === false) {
          document.getElementById('btnHudHeroToggle')?.click();
        }
        return true;
      })()`
    );
    await new Promise((r) => setTimeout(r, 80));
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
      cdp,
      sessionId,
      'Step 6d: Swarm inspect readout stays live while Hero-16 overlay is hidden'
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
        window.__chronosUi?.reveal('calRulerM');
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
        window.__chronosUi?.reveal('swarmModeSelect');
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
      cdp,
      sessionId,
      'Step 6d: Clicking fullscreen #hudCanvas sustains valid Swarm inspection readout',
      { useCdpClick: true }
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
        if (req.url === '/evil.mpd') {
          res.writeHead(200, { 'Content-Type': 'application/dash+xml' });
          res.end(
            `<MPD><Period><AdaptationSet><Representation><BaseURL>http://127.0.0.1:${internalPort}/dash_secret.mp4</BaseURL></Representation></AdaptationSet></Period></MPD>`.padEnd(1200, ' ')
          );
          return;
        }
        if (req.url === '/oversized.m3u8') {
          res.writeHead(200, { 'Content-Type': 'application/vnd.apple.mpegurl' });
          res.end(`#EXTM3U\n${'# comment\n'.repeat(60000)}`);
          return;
        }
        if (req.url === '/slow_preflight.mp4') {
          const timer = setTimeout(() => {
            if (!res.destroyed && !res.writableEnded) {
              res.writeHead(200, { 'Content-Type': 'video/mp4' });
              fs.createReadStream(path.join(WEB_DIR, 'samples', 'roundabout_occlusion.mp4')).pipe(res);
            }
          }, 220);
          res.once('close', () => clearTimeout(timer));
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

        const nat64Res = await fetch(
          `${capBase}/api/live-stream?url=${encodeURIComponent('http://[64:ff9b::7f00:1]/a.mp4')}`
        );
        await nat64Res.arrayBuffer().catch(() => {});
        if (nat64Res.status !== 400 && nat64Res.status !== 403) {
          throw new Error(`Step 7b: NAT64 IPv6 SSRF target not blocked (status=${nat64Res.status})`);
        }

        const dashRes = await fetch(
          `${capBase}/api/live-stream?url=${encodeURIComponent(`http://127.0.0.1:${publicMockPort}/evil.mpd`)}`
        );
        await dashRes.arrayBuffer().catch(() => {});
        if (dashRes.status === 200 || internalHits !== 0) {
          throw new Error(`Step 7b: DASH manifest SSRF not blocked (status=${dashRes.status}, internalHits=${internalHits})`);
        }

        const bigRes = await fetch(
          `${capBase}/api/live-stream?url=${encodeURIComponent(`http://127.0.0.1:${publicMockPort}/oversized.m3u8`)}`,
          { signal: AbortSignal.timeout(4000) }
        );
        await bigRes.arrayBuffer().catch(() => {});
        if (bigRes.status === 200) {
          throw new Error(`Step 7b: Oversized >512 KB HLS playlist unexpectedly succeeded (${bigRes.status})`);
        }

        // Disconnect client mid-preflight and prove no orphan FFmpeg process consumes the maxLiveProcs=1 slot
        const slowCtrl = new AbortController();
        const slowPromise = fetch(
          `${capBase}/api/live-stream?url=${encodeURIComponent(`http://127.0.0.1:${publicMockPort}/slow_preflight.mp4`)}`,
          { signal: slowCtrl.signal }
        ).catch(() => null);
        await new Promise((r) => setTimeout(r, 40));
        slowCtrl.abort();
        await slowPromise;
        await new Promise((r) => setTimeout(r, 300));

        const ssrfHlsRes = await fetch(
          `${capBase}/api/live-stream?url=${encodeURIComponent(`http://127.0.0.1:${publicMockPort}/master.m3u8`)}`
        );
        await ssrfHlsRes.arrayBuffer().catch(() => {});
        if (internalHits !== 0) {
          throw new Error(`Step 7b: Multi-hop HLS segment/parser-differential SSRF leaked ${internalHits} request(s) to internal server!`);
        }

        // Verify maxLiveProcs=1 -> 200 on first live relay (slot uncorrupted) then 429 Too Many Requests on concurrent relay
        const holdAbort = new AbortController();
        const holdPromise = fetch(`${capBase}/api/live-stream?preset=sim_live_traffic`, { signal: holdAbort.signal });
        const holdRes = await holdPromise;
        if (holdRes.status !== 200) {
          throw new Error(`Step 7b: Expected 200 for first live relay (no preflight orphan leak), got ${holdRes.status}`);
        }
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
          window.__chronosUi?.reveal('calRulerM');
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
        await cdp.send('Target.closeTarget', { targetId }).catch(() => {});
        await cdp.send(
          'Emulation.setDeviceMetricsOverride',
          { width: 1440, height: 900, deviceScaleFactor: process.env.CI ? 1 : 2, mobile: false },
          scSession
        );

        await waitForCondition(
          cdp,
          scSession,
          'Step 8a: Bare URL / boots into v3.5 Instrument-First Idle Bench (zero unsolicited autoplay)',
          `(() => ({
            ok: document.getElementById('welcomeOverlay')?.hidden === false &&
                window.__chronosUi?.getState?.()?.source?.kind === 'IDLE' &&
                Boolean(document.getElementById('showcaseWipeSlider')) &&
                Boolean(document.getElementById('lnkSampleAttribution')?.getAttribute('href')?.endsWith('samples/credits.html')) &&
                document.title.includes('CHRONOS-SPECTRAL')
          }))()`,
          8000,
          true
        );
        const creditsOk = await cdp.evalJs(
          scSession,
          `(async () => {
            const r = await fetch('./samples/credits.html');
            const html = await r.text();
            return r.status === 200 && html.includes('Bernard de Go Mars') && html.includes('Karen Roe');
          })()`
        );
        if (!creditsOk) {
          throw new Error('Step 8a: ./samples/credits.html failed status 200 or missing verified attribution credits');
        }
        await cdp.evalJs(
          scSession,
          `(() => {
            window.__e2eShowcaseCls = 0;
            window.__e2eShowcaseClsPo = new PerformanceObserver((list) => {
              for (const e of list.getEntries()) {
                window.__e2eShowcaseCls += e.value;
              }
            });
            window.__e2eShowcaseClsPo.observe({ type: 'layout-shift', buffered: false });
            document.getElementById('btnHeroStarlings')?.click();
            return true;
          })()`
        );

        const starlingBoot = await waitForCondition(
          cdp,
          scSession,
          'Step 8a: 1-Click Starling Murmuration preset (>= 1,000 confirmed, HERO 16 hidden, 28% wipe)',
          `(() => {
            const wipeSlider = document.getElementById('showcaseWipeSlider');
            const kernelTxt = (document.getElementById('scValKernel')?.textContent || '').trim();
            const tracksTxt = (document.getElementById('scValTracks')?.textContent || '').trim();
            const confRaw = tracksTxt.split(' ')[0] || '0';
            const confirmed = parseInt(confRaw.replace(/,/g, ''), 10) || 0;
            const st = window.__chronosUiState;
            return {
              ok: document.getElementById('welcomeOverlay')?.hidden === true &&
                  st?.recipe?.key === 'starling_1000' &&
                  confirmed >= 1000 &&
                  st?.overlays?.showHeroTargets === false &&
                  wipeSlider?.value === '28' &&
                  kernelTxt.endsWith('ms / frame') &&
                  !kernelTxt.startsWith('--'),
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

        const dragGeom = await cdp.evalJs(
          scSession,
          `(() => {
            const hud = document.getElementById('hudCanvas');
            const stage = document.getElementById('hudCanvasStage');
            const track = document.getElementById('hudWipeTrack');
            const handle = document.getElementById('hudWipeHandle');
            if (!hud || !stage || !track || track.hidden || !handle) return null;
            hud.scrollIntoView({ block: 'center', inline: 'center' });
            const hr = hud.getBoundingClientRect();
            const dr = handle.getBoundingClientRect();
            return {
              wipeCssVar: stage.style.getPropertyValue('--wipe-pct').trim(),
              handleX: dr.left + dr.width * 0.5,
              handleY: dr.top + dr.height * 0.5,
              targetX: hr.left + hr.width * 0.38,
              drW: dr.width,
              drH: dr.height,
            };
          })()`
        );
        if (!dragGeom || dragGeom.wipeCssVar !== '55' || dragGeom.drW < 20 || dragGeom.drH < 40) {
          throw new Error(`Step 8b: On-canvas #hudWipeHandle geometry/sync invalid: ${JSON.stringify(dragGeom)}`);
        }
        await cdp.send(
          'Input.dispatchMouseEvent',
          { type: 'mouseMoved', x: dragGeom.handleX, y: dragGeom.handleY, button: 'none' },
          scSession
        );
        await cdp.send(
          'Input.dispatchMouseEvent',
          { type: 'mousePressed', x: dragGeom.handleX, y: dragGeom.handleY, button: 'left', buttons: 1, clickCount: 1 },
          scSession
        );
        await cdp.send(
          'Input.dispatchMouseEvent',
          { type: 'mouseMoved', x: dragGeom.handleX - 12, y: dragGeom.handleY, button: 'left', buttons: 1 },
          scSession
        );
        await cdp.send(
          'Input.dispatchMouseEvent',
          { type: 'mouseMoved', x: dragGeom.targetX, y: dragGeom.handleY, button: 'left', buttons: 1 },
          scSession
        );
        await cdp.send(
          'Input.dispatchMouseEvent',
          { type: 'mouseReleased', x: dragGeom.targetX, y: dragGeom.handleY, button: 'left', buttons: 0, clickCount: 1 },
          scSession
        );

        const dragAfter = await waitForCondition(
          cdp,
          scSession,
          'Step 8b: On-canvas pointer drag updates split-wipe + slider + status cleanly',
          `(() => {
            const s = document.getElementById('showcaseWipeSlider');
            const handle = document.getElementById('hudWipeHandle');
            const readout = document.getElementById('showcaseWipeReadout')?.textContent || '';
            const status = document.getElementById('showcaseToolbarStatus')?.textContent || '';
            const val = Number(s?.value || 0);
            const draggingLeft = Boolean(handle && handle.classList.contains('dragging'));
            return {
              ok: val >= 37 &&
                  val <= 39 &&
                  !draggingLeft &&
                  readout.includes(val + '% RAW') &&
                  status.includes('Split-wipe divider moved to ' + val + '%'),
              val,
              draggingLeft,
              readout,
              status,
            };
          })()`,
          4000
        );

        const kbdAfter = await cdp.evalJs(
          scSession,
          `(() => {
            const handle = document.getElementById('hudWipeHandle');
            handle.focus();
            handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
            const val = Number(document.getElementById('showcaseWipeSlider')?.value || 0);
            return { val, cssVar: document.getElementById('hudCanvasStage')?.style.getPropertyValue('--wipe-pct').trim() || '' };
          })()`
        );
        if (kbdAfter.val !== dragAfter.val + 2 || kbdAfter.cssVar !== String(kbdAfter.val)) {
          throw new Error(`Step 8b: ArrowRight on #hudWipeHandle expected ${dragAfter.val + 2}, got ${JSON.stringify(kbdAfter)}`);
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

        // 8c. 1-Click Recipe Counter-Proofs: Carom Billiards + Model Rockets
        await cdp.evalJs(scSession, `window.__chronosUi?.loadRecipe('billiards'); true;`);
        const billiardsHero = await waitForCondition(
          cdp,
          scSession,
          'Step 8c: Unified 1-click Carom Billiards (Mode C)',
          `(() => {
            const selVal = document.getElementById('sampleVideoSelect')?.value || '';
            const cal = document.getElementById('calBadge')?.textContent || '';
            const modeC = document.querySelector('.lab-btn[data-lab="2"]')?.getAttribute('aria-pressed') === 'true';
            const wipe = Number(document.getElementById('showcaseWipeSlider')?.value || 0);
            const st = window.__chronosUiState;
            const tgts = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
            return {
              ok: st?.recipe?.key === 'billiards' &&
                  selVal.endsWith('carom_billiards_impulses.mp4') &&
                  st?.overlays?.showHeroTargets === true &&
                  cal === 'ASSUMED 4-PT H' &&
                  modeC &&
                  wipe > 0 &&
                  tgts >= 1,
              selVal,
              cal,
              modeC,
              wipe,
              tgts,
            };
          })()`,
          15000
        );

        await cdp.evalJs(scSession, `window.__chronosUi?.loadRecipe('model_rockets'); true;`);
        const rocketHero = await waitForCondition(
          cdp,
          scSession,
          'Step 8c: Unified 1-click Model Rockets counter-proof',
          `(() => {
            const selVal = document.getElementById('sampleVideoSelect')?.value || '';
            const cal = document.getElementById('calBadge')?.textContent || '';
            const calStatus = document.getElementById('calStatusLive')?.textContent || '';
            const calValid =
              cal === 'ASSUMED RULER' ||
              (cal === 'PX UNITS' && calStatus.includes('CAMERA PTZ PAN'));
            const wipe = Number(document.getElementById('showcaseWipeSlider')?.value || 0);
            const st = window.__chronosUiState;
            const tgts = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
            return {
              ok: st?.recipe?.key === 'model_rockets' &&
                  selVal.endsWith('model_rocket_full_flights.mp4') &&
                  st?.overlays?.showHeroTargets === true &&
                  calValid &&
                  wipe > 0 &&
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

        // 8d. Zero-Upload Drag-and-Drop Local MP4 onto #hudPanel (preserving current instrument config!)
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
          'Step 8d: Drag-and-drop user_clip.mp4 tracks locally with recipe cleared & setup preserved',
          `(() => {
            const srcLbl = document.getElementById('hudSourceLabel')?.textContent || '';
            const tgts = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
            const st = window.__chronosUiState;
            return {
              ok: srcLbl.includes('user_clip.mp4') && st?.recipe?.key === null && tgts >= 1,
              srcLbl,
              tgts,
            };
          })()`,
          15000
        );

        await new Promise((r) => setTimeout(r, 80));
        const totalCls = await cdp.evalJs(
          scSession,
          `(() => {
            if (window.__e2eShowcaseClsPo) window.__e2eShowcaseClsPo.disconnect();
            return window.__e2eShowcaseCls || 0;
          })()`
        );
        if (totalCls >= 0.0001) {
          throw new Error(`Step 8d: Expected < 0.0001 raw CLS across unified bench flows (8a..8d), got ${totalCls}`);
        }

        // 8e. Static-Host Mode (?static=1) & /deck.html Redirect
        await cdp.send('Page.navigate', { url: `${baseUrl}/index.html?static=1` }, scSession);
        const staticState = await waitForCondition(
          cdp,
          scSession,
          'Step 8e: Static-host mode active (?static=1) without API 404s',
          `(() => ({
            ok: document.body?.dataset?.staticHost === 'true' && document.getElementById('welcomeOverlay')?.hidden === false,
            staticHost: document.body?.dataset?.staticHost,
          }))()`,
          8000,
          true
        );
        await new Promise((r) => setTimeout(r, 250));

        await cdp.send('Page.navigate', { url: `${baseUrl}/deck.html` }, scSession);
        await waitForCondition(
          cdp,
          scSession,
          'Step 8e: /deck.html redirects cleanly to ?view=deck SD pillar bench',
          `(() => ({
            ok: location.search.includes('view=deck') &&
                document.getElementById('welcomeOverlay')?.hidden === true &&
                window.__chronosUi?.getState?.()?.source?.kind === 'TESTBENCH',
          }))()`,
          8000,
          true
        );

        console.log(
          `  [PASS] Step 8 (v3.5 Unified Instrument Boot, Split-Wipe, Local Drop & Static-Host Gate): starlings=${starlingBoot.confirmed} (${starlingBoot.kernelTxt}, hud=${scHudBytes}B), wipeDrag=${dragAfter.val}%+ArrowRight=${kbdAfter.val}%, inspect="${scInspectText}", billiards=${billiardsHero.cal}(${billiardsHero.tgts} tgts), rockets=${rocketHero.cal}(${rocketHero.tgts} tgts, wipe=${rocketHero.wipe}%), drop="${dropState.srcLbl}", staticHost=${staticState.staticHost}, rawCLS=${totalCls.toFixed(6)}`
        );
      } finally {
        await cdp.send('Target.closeTarget', { targetId: scTargetId }).catch(() => {});
      }
    }

    // 9. Wave-1 Live Showcase BOS Plume, Neumann Potential / Curl Overlays, Native 1:1 ROI Offset, Tripod Failure Warning & Zero-CLS
    {
      const baseUrl = `http://127.0.0.1:${srv.port}`;
      const { targetId: w1TargetId } = await cdp.send('Target.createTarget', { url: `${baseUrl}/index.html` });
      const { sessionId: w1Session } = await cdp.send('Target.attachToTarget', { targetId: w1TargetId, flatten: true });
      try {
        await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: true }, w1Session);
        await cdp.send('Page.enable', {}, w1Session);
        await cdp.send('Log.enable', {}, w1Session);
        await cdp.send('Runtime.enable', {}, w1Session);
        await cdp.send(
          'Emulation.setDeviceMetricsOverride',
          { width: 1440, height: 900, deviceScaleFactor: process.env.CI ? 1 : 2, mobile: false },
          w1Session
        );

        await waitForCondition(
          cdp,
          w1Session,
          'Step 9a: Unified instrument booted cleanly in IDLE with Wave-1 BOS controls',
          `(() => {
            const st = window.__chronosUi?.getState?.();
            const pillTxt = (document.getElementById('pillKernel')?.textContent || '').trim();
            const sourceKind = st?.source?.kind;
            const workerState = st?.runtime_health?.workerState;
            const welcomeVisible = document.getElementById('welcomeOverlay')?.hidden === false;
            return {
              ok: Boolean(document.getElementById('btnHeroBosPlume')) &&
                  Boolean(document.getElementById('acquisitionStatusBanner')) &&
                  welcomeVisible &&
                  sourceKind === 'IDLE' &&
                  workerState === 'READY' &&
                  pillTxt.includes('0 HEAP ALLOC'),
              sourceKind,
              workerState,
              welcomeVisible,
              pillTxt,
            };
          })()`,
          20000,
          true
        );

        await cdp.evalJs(
          w1Session,
          `(() => {
            window.__e2eW1Cls = 0;
            window.__e2eW1ClsPo = new PerformanceObserver((list) => {
              for (const e of list.getEntries()) {
                if (!e.hadRecentInput) window.__e2eW1Cls += e.value;
              }
            });
            window.__e2eW1ClsPo.observe({ type: 'layout-shift', buffered: false });
            if (${Boolean(process.env.CI)}) {
              const v = document.getElementById('hiddenVideoSource');
              if (v) { v.defaultPlaybackRate = 0.25; v.playbackRate = 0.25; }
            }
            document.getElementById('btnHeroBosPlume')?.click();
            return true;
          })()`
        );

        const bosReadyState = await waitForCondition(
          cdp,
          w1Session,
          'Step 9a: Thermal Plume (BOS) reaches READY + 1:1 Native ROI + fresh stamps',
          `(() => {
            if (${Boolean(process.env.CI)}) {
              const v = document.getElementById('hiddenVideoSource');
              if (v && v.playbackRate !== 0.25) { v.defaultPlaybackRate = 0.25; v.playbackRate = 0.25; }
            }
            const selVal = document.getElementById('sampleVideoSelect')?.value || '';
            const bosMode = document.getElementById('bosModeSelect')?.value || '';
            const ovMode = document.getElementById('fieldOverlayModeSelect')?.value || '';
            const roiChk = document.getElementById('chkNativeRoi')?.checked === true;
            const covChk = document.getElementById('chkCovEllipses')?.checked === true;
            const roiTxt = document.getElementById('roiGeomMeta')?.textContent || '';
            const diagBos = document.getElementById('diagBosState')?.textContent || '';
            const diagIngest = document.getElementById('diagWave1Ingest')?.textContent || '';
            const diagStamps = document.getElementById('diagWave1Stamps')?.textContent || '';
            const banner = document.getElementById('acquisitionStatusBanner');
            const sev = banner?.dataset?.severity || '';
            const st = window.__chronosUiState;
            return {
              ok: st?.recipe?.key === 'bos_plume' &&
                  selVal.endsWith('bos_thermal_plume.mp4') &&
                  bosMode === '1' &&
                  ovMode === '5' &&
                  roiChk &&
                  covChk &&
                  roiTxt.startsWith('1:1 NATIVE (0,0 640x360)') &&
                  diagBos.startsWith('READY · ') &&
                  diagIngest.includes(' · OK · ') &&
                  diagStamps.length >= 6 &&
                  diagStamps[5] === 'V' &&
                  sev === 'ok' &&
                  st?.overlays?.fieldOverlayMode === 5 &&
                  st?.overlays?.showCovEllipses === true,
              roiTxt,
              diagBos,
              diagIngest,
              diagStamps,
              sev,
            };
          })()`,
          process.env.CI ? 35000 : 22000
        );
        if (process.env.CI) {
          await cdp.evalJs(
            w1Session,
            `(() => {
              const v = document.getElementById('hiddenVideoSource');
              if (v) { v.defaultPlaybackRate = 1.0; v.playbackRate = 1.0; }
              return true;
            })()`
          );
        }

        const bosHudBytes = await (async () => {
          const rect = await cdp.evalJs(
            w1Session,
            `(() => {
              const r = document.getElementById('hudCanvas').getBoundingClientRect();
              return { x: Math.max(0, Math.floor(r.left)), y: Math.max(0, Math.floor(r.top)), width: Math.ceil(r.width), height: Math.ceil(r.height), scale: 1 };
            })()`
          );
          const shot = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false, clip: rect }, w1Session);
          return shot?.data ? Buffer.from(shot.data, 'base64').byteLength : 0;
        })();
        if (bosHudBytes < 2500) {
          throw new Error(`Step 9a: Expected BOS #hudCanvas screenshot >= 2500 B, got ${bosHudBytes} B`);
        }

        await new Promise((r) => setTimeout(r, 80));
        const w1ShowcaseCls = await cdp.evalJs(
          w1Session,
          `(() => {
            if (window.__e2eW1ClsPo) window.__e2eW1ClsPo.disconnect();
            return window.__e2eW1Cls || 0;
          })()`
        );
        if (w1ShowcaseCls >= 0.000001) {
          throw new Error(`Step 9a: Expected 0.000000 CLS on BOS flow, got ${w1ShowcaseCls}`);
        }

        // 9b. Attach fresh CLS observer + verify Neumann Potential (6) & Curl Check (7) + SD Native 1:1 ROI non-zero (crop_x0, crop_y0) offset
        await cdp.evalJs(
          w1Session,
          `new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => {
            window.__e2eW1DeckCls = 0;
            window.__e2eW1DeckClsPo = new PerformanceObserver((list) => {
              for (const e of list.getEntries()) {
                if (!e.hadRecentInput) window.__e2eW1DeckCls += e.value;
              }
            });
            window.__e2eW1DeckClsPo.observe({ type: 'layout-shift', buffered: false });
            resolve(true);
          })))`
        );

        for (const modeVal of ['6', '7']) {
          await cdp.evalJs(
            w1Session,
            `(() => {
              const sel = document.getElementById('fieldOverlayModeSelect');
              sel.value = ${JSON.stringify(modeVal)};
              sel.dispatchEvent(new Event('change', { bubbles: true }));
              return true;
            })()`
          );
          await waitForCondition(
            cdp,
            w1Session,
            `Step 9b: BOS fieldOverlayMode=${modeVal} synced cleanly`,
            `(() => ({
              ok: document.getElementById('fieldOverlayModeSelect')?.value === ${JSON.stringify(modeVal)} &&
                  window.__chronosUiState?.overlays?.fieldOverlayMode === ${Number(modeVal)}
            }))()`,
            4000
          );
        }

        await cdp.evalJs(
          w1Session,
          `(() => {
            document.getElementById('btnResSd')?.click();
            const chk = document.getElementById('chkNativeRoi');
            if (chk && !chk.checked) {
              chk.checked = true;
              chk.dispatchEvent(new Event('change', { bubbles: true }));
            }
            const rx = document.getElementById('rangeRoiX0');
            const ry = document.getElementById('rangeRoiY0');
            if (rx) { rx.value = '100'; rx.dispatchEvent(new Event('input', { bubbles: true })); }
            if (ry) { ry.value = '100'; ry.dispatchEvent(new Event('input', { bubbles: true })); }
            return true;
          })()`
        );
        const roiOffsetState = await waitForCondition(
          cdp,
          w1Session,
          'Step 9b: SD Native 1:1 ROI shifts to (320,180 320x180) without implicit rescaling',
          `(() => {
            const roiTxt = document.getElementById('roiGeomMeta')?.textContent || '';
            return { ok: roiTxt === '1:1 NATIVE (320,180 320x180)', roiTxt };
          })()`,
          8000
        );

        await cdp.evalJs(
          w1Session,
          `(() => {
            document.getElementById('btnCalRuler')?.click();
            const hud = document.getElementById('hudCanvas');
            const r = hud.getBoundingClientRect();
            hud.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, clientX: r.left + r.width * 0.25, clientY: r.top + r.height * 0.50 }));
            hud.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, clientX: r.left + r.width * 0.75, clientY: r.top + r.height * 0.50 }));
            return true;
          })()`
        );
        await waitForCondition(
          cdp,
          w1Session,
          'Step 9b-cal: SURVEYED RULER active on cropped 1:1 ROI before ROI shift',
          `(() => ({
            ok: document.getElementById('calBadge')?.textContent === 'SURVEYED RULER',
            badge: document.getElementById('calBadge')?.textContent || '',
          }))()`,
          8000
        );
        await cdp.evalJs(
          w1Session,
          `(() => {
            const rx = document.getElementById('rangeRoiX0');
            if (rx) { rx.value = '0'; rx.dispatchEvent(new Event('input', { bubbles: true })); }
            return true;
          })()`
        );
        await waitForCondition(
          cdp,
          w1Session,
          'Step 9b-cal-reset: Shifting Native 1:1 crop_x0 resets stale SURVEYED RULER to PX UNITS',
          `(() => {
            const roiTxt = document.getElementById('roiGeomMeta')?.textContent || '';
            const badge = document.getElementById('calBadge')?.textContent || '';
            const live = document.getElementById('calStatusLive')?.textContent || '';
            return {
              ok: roiTxt === '1:1 NATIVE (0,180 320x180)' && badge === 'PX UNITS' && live.includes('ROI CROP CHANGED'),
              roiTxt,
              badge,
              live,
            };
          })()`,
          8000
        );

        // 9c. Restore HD and trigger Controlled Handheld Failure -> banner turns danger with SHIFT readout
        await cdp.evalJs(
          w1Session,
          `(() => {
            document.getElementById('btnResHd')?.click();
            const sel = document.getElementById('sampleVideoSelect');
            sel.value = './samples/bos_handheld_failure.mp4';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
          })()`
        );
        const failureBannerState = await waitForCondition(
          cdp,
          w1Session,
          'Step 9c: bos_handheld_failure.mp4 triggers Tripod Failure Warning banner (data-severity="danger")',
          `(() => {
            const banner = document.getElementById('acquisitionStatusBanner');
            const sev = banner?.dataset?.severity || '';
            const txt = banner?.textContent || '';
            const diagBos = document.getElementById('diagBosState')?.textContent || '';
            const diagCov = document.getElementById('diagMetrologyCov')?.textContent || '';
            return {
              ok: sev === 'danger' &&
                  txt.includes('TRIPOD FAILURE / DISTURBED') &&
                  diagBos.startsWith('DISTURBED · '),
              sev,
              txt,
              diagBos,
              diagCov,
            };
          })()`,
          24000
        );

        // 9d. Perspective Flyby Decoy: 2x2 covariance active across multiple frames with 0 false I1 Hero promos
        const epBeforeDecoy = await cdp.evalJs(
          w1Session,
          `(() => {
            const diagIngest = document.getElementById('diagWave1Ingest')?.textContent || '';
            const m = diagIngest.match(/EP\\s+(\\d+)/);
            const curEp = m ? Number(m[1]) : 0;
            const sel = document.getElementById('sampleVideoSelect');
            sel.value = './samples/perspective_flyby_decoy.mp4';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
            return curEp;
          })()`
        );
        await waitForCondition(
          cdp,
          w1Session,
          'Step 9d-init: perspective_flyby_decoy.mp4 booted to seq >= 20 on fresh continuity epoch',
          `(() => {
            const snap = window.__chronosWave1Probe?.getLastWave1?.();
            const diagIngest = document.getElementById('diagWave1Ingest')?.textContent || '';
            const mEp = diagIngest.match(/EP\\s+(\\d+)/);
            const curEp = mEp ? Number(mEp[1]) : 0;
            const seqNum = snap?.wave1?.dispMeta?.sourceSeq || 0;
            const metSel = document.getElementById('metrologyModeSelect')?.value || '';
            const ellOn = document.getElementById('chkCovEllipses')?.checked === true;
            const srcOk = (document.getElementById('hudSourceLabel')?.textContent || '').includes('PERSPECTIVE');
            return {
              ok: curEp > ${epBeforeDecoy} && srcOk && metSel === '2' && ellOn && seqNum >= 20 &&
                  window.__chronosUiState?.overlays?.showCovEllipses === true,
              curEp,
              seqNum,
            };
          })()`,
          24000
        );
        const decoyBudgetScale = Math.max(1.0, Math.min(2.0, Number(process.env.CHRONOS_CI_BUDGET_SCALE || '1.0') * 0.6));
        const decoyDeadline = Date.now() + Math.round(22000 * decoyBudgetScale);
        let decoyDistinct = 0;
        let decoyPrevCommits = -1;
        let decoyPrevSeq = -1;
        let decoyMaxSeq = 0;
        let decoyMaxUsable = 0;
        let decoyLastDiag = '';
        while (Date.now() < decoyDeadline) {
          const sample = await cdp.evalJs(
            w1Session,
            `(() => {
              const snap = window.__chronosWave1Probe?.getLastWave1?.();
              return {
                diagCov: document.getElementById('diagMetrologyCov')?.textContent || '',
                ep: snap?.wave1?.continuityEpoch || 0,
                commits: snap?.wave1?.totalCommits || 0,
                seq: snap?.wave1?.dispMeta?.sourceSeq || 0,
                usable: snap?.metrology?.usableCovCount || 0,
                cand: snap?.metrology?.promotedCandidates ?? -1,
                autoHero: snap?.metrology?.activeAutoHeroes ?? -1,
              };
            })()`
          );
          if (sample.ep <= epBeforeDecoy) {
            throw new Error(`Step 9d: continuity epoch regressed (${sample.ep} <= ${epBeforeDecoy})`);
          }
          if (sample.commits < decoyPrevCommits) {
            throw new Error(`Step 9d: totalCommits went backward (${sample.commits} < ${decoyPrevCommits})`);
          }
          if (sample.cand !== 0 || sample.autoHero !== 0) {
            throw new Error(`Step 9d: transient false Auto-Hero promotion on decoy seq=${sample.seq}: cand=${sample.cand}, autoHero=${sample.autoHero}, diag=${sample.diagCov}`);
          }
          if (sample.seq !== decoyPrevSeq && sample.commits > decoyPrevCommits) {
            decoyDistinct++;
            decoyPrevSeq = sample.seq;
          }
          decoyPrevCommits = sample.commits;
          decoyMaxSeq = Math.max(decoyMaxSeq, sample.seq);
          decoyMaxUsable = Math.max(decoyMaxUsable, sample.usable);
          decoyLastDiag = sample.diagCov;
          if (decoyMaxSeq >= 115 && decoyDistinct >= 8 && decoyMaxUsable >= 20) break;
          await new Promise((r) => setTimeout(r, 90));
        }
        if (decoyMaxSeq < 115 || decoyDistinct < 8 || decoyMaxUsable < 20) {
          throw new Error(`Step 9d: decoy loop incomplete (maxSeq=${decoyMaxSeq}, distinct=${decoyDistinct}, maxUsable=${decoyMaxUsable}, diag=${decoyLastDiag})`);
        }
        const decoyState = { diagCov: `${decoyLastDiag} (${decoyDistinct} samples, maxSeq=${decoyMaxSeq}, maxUsable=${decoyMaxUsable})` };

        // 9e. Real Starling Swarm + H.264 2x2 Covariance Ellipse + Auto-Hero Promotion & Retirement Proof
        const epBeforeStarling = await cdp.evalJs(
          w1Session,
          `(() => {
            const diagIngest = document.getElementById('diagWave1Ingest')?.textContent || '';
            const m = diagIngest.match(/EP\\s+(\\d+)/);
            const curEp = m ? Number(m[1]) : 0;
            const sel = document.getElementById('sampleVideoSelect');
            sel.value = './samples/starling_murmuration_1000.mp4';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
            return curEp;
          })()`
        );
        await waitForCondition(
          cdp,
          w1Session,
          'Step 9e: Starling clip booted on fresh continuity epoch before enabling H.264 2x2 metrology',
          `(() => {
            const diagIngest = document.getElementById('diagWave1Ingest')?.textContent || '';
            const mEp = diagIngest.match(/EP\\s+(\\d+)/);
            const curEp = mEp ? Number(mEp[1]) : 0;
            const srcOk = (document.getElementById('hudSourceLabel')?.textContent || '').includes('STARLING');
            return { ok: curEp > ${epBeforeStarling} && srcOk, curEp, diagIngest };
          })()`,
          15000
        );
        await cdp.evalJs(
          w1Session,
          `(() => {
            if (${Boolean(process.env.CI)}) {
              const v = document.getElementById('hiddenVideoSource');
              if (v) { v.defaultPlaybackRate = 0.25; v.playbackRate = 0.25; }
            }
            const sw = document.getElementById('swarmModeSelect');
            if (sw && sw.value !== '2') { sw.value = '2'; sw.dispatchEvent(new Event('change', { bubbles: true })); }
            const mm = document.getElementById('metrologyModeSelect');
            if (mm) { mm.value = '2'; mm.dispatchEvent(new Event('change', { bubbles: true })); }
            const chk = document.getElementById('chkCovEllipses');
            if (chk && !chk.checked) { chk.checked = true; chk.dispatchEvent(new Event('change', { bubbles: true })); }
            return true;
          })()`
        );
        const denseCovState = await waitForCondition(
          cdp,
          w1Session,
          'Step 9e: Dense starling swarm runs 2x2 covariance + LOO residual Auto-Hero promotion (cand >= 1, tgts in 1..3)',
          `(() => {
            if (${Boolean(process.env.CI)}) {
              const v = document.getElementById('hiddenVideoSource');
              if (v && v.playbackRate !== 0.25) { v.defaultPlaybackRate = 0.25; v.playbackRate = 0.25; }
            }
            const snap = window.__chronosWave1Probe?.getLastWave1?.();
            const diagCov = document.getElementById('diagMetrologyCov')?.textContent || '';
            const mUsable = diagCov.match(/(\\d+)\\s+usable/);
            const mCand = diagCov.match(/(\\d+)\\s+cand/);
            const usable = mUsable ? Number(mUsable[1]) : 0;
            const cand = mCand ? Number(mCand[1]) : 0;
            const autoHero = snap?.metrology?.activeAutoHeroes ?? 0;
            const conf = parseInt((document.getElementById('kpiSwarmConfirmed')?.textContent || '0').split('/')[0].replace(/,/g, ''), 10) || 0;
            const tgts = parseInt((document.getElementById('kpiTargets')?.textContent || '0').split('/')[0], 10) || 0;
            return {
              ok: conf >= 900 &&
                  usable >= 300 &&
                  cand >= 1 &&
                  tgts >= 1 &&
                  tgts <= 3 &&
                  window.__chronosUiState?.overlays?.showCovEllipses === true,
              diagCov,
              conf,
              usable,
              cand,
              autoHero,
              tgts,
            };
          })()`,
          24000
        );
        await cdp.evalJs(
          w1Session,
          `(() => {
            const mm = document.getElementById('metrologyModeSelect');
            if (mm) { mm.value = '0'; mm.dispatchEvent(new Event('change', { bubbles: true })); }
            return true;
          })()`
        );
        const starlingMetOffState = await waitForCondition(
          cdp,
          w1Session,
          'Step 9e-off: Disabling metrology mode=0 clears candidates & enabled bit 0 while keeping legacy tracking intact',
          `(() => {
            const snap = window.__chronosWave1Probe?.getLastWave1?.();
            const autoHero = snap?.metrology?.activeAutoHeroes ?? -1;
            const cand = snap?.metrology?.promotedCandidates ?? -1;
            const enabledBits = snap?.wave1?.enabledBits ?? 1;
            return {
              ok: autoHero === 0 && cand === 0 && (enabledBits & 1) === 0,
              autoHero,
              cand,
              enabledBits,
            };
          })()`,
          12000
        );

        // 9f. Browser 8x4 Orientation x scanDir Matrix, 360x640 Quarter-Turn Native, Undersized Non-Smooth Fallback & Reject
        await cdp.evalJs(
          w1Session,
          `(() => {
            const v = document.getElementById('hiddenVideoSource') || document.querySelector('video');
            if (v) {
              v.defaultPlaybackRate = 1.0;
              v.playbackRate = 1.0;
              v.pause();
            }
            const sel = document.getElementById('sampleVideoSelect');
            if (sel) sel.value = '';
            const probe = window.__chronosWave1Probe;
            probe.postWorkerMsg({ type: 'set_resolution', resMode: 1 });
            probe.postWorkerMsg({ type: 'set_swarm_mode', swarmMode: 0 });
            probe.postWorkerMsg({ type: 'set_bos_config', bosMode: 0 });
            probe.postWorkerMsg({ type: 'set_metrology_config', metrologyMode: 0 });
            probe.postWorkerMsg({ type: 'set_field_overlay_mode', fieldOverlayMode: 0 });
            probe.postWorkerMsg({ type: 'params', lookaheadK: 0, rawWipeRatio: 1.0, showHeroTargets: false, showcaseMinimalScopes: false });
            probe.postWorkerMsg({ type: 'set_source', sourceMode: 'video', fpsHint: 30 });
            return true;
          })()`
        );
        await waitForCondition(
          cdp,
          w1Session,
          'Step 9f-drain: Wait for paused video backpressure to settle before matrix checks',
          `(() => ({ ok: window.__chronosWave1Probe?.getBusyState?.().workerFrameBusy === false }))()`,
          5000
        );
        await new Promise((r) => setTimeout(r, 80));
        const orientMatrixReport = await cdp.evalJs(
          w1Session,
          `(async () => {
            const probe = window.__chronosWave1Probe;
            const hudEl = document.getElementById('hudCanvas');
            const readCanvas = document.createElement('canvas');
            readCanvas.width = 960;
            readCanvas.height = 540;
            const rCtx = readCanvas.getContext('2d', { willReadFrequently: true });
            const srcCanvas = document.createElement('canvas');
            const sCtx = srcCanvas.getContext('2d');
            const waitRafs = () => new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(res)));
            const sampleCornerColors = () => {
              rCtx.drawImage(hudEl, 0, 0);
              const pts = [[200, 135], [760, 135], [200, 405], [760, 405]];
              return pts.map(([x, y]) => {
                const d = rCtx.getImageData(x, y, 1, 1).data;
                if (d[0] > 200 && d[1] < 55 && d[2] < 55) return 'R';
                if (d[0] < 55 && d[1] > 200 && d[2] < 55) return 'G';
                if (d[0] < 55 && d[1] < 55 && d[2] > 200) return 'B';
                if (d[0] > 200 && d[1] > 200 && d[2] < 55) return 'Y';
                return d[0] + ',' + d[1] + ',' + d[2];
              }).join('');
            };
            const paintQuadrants = (w, h) => {
              srcCanvas.width = w;
              srcCanvas.height = h;
              const hw = w >> 1;
              const hh = h >> 1;
              sCtx.fillStyle = '#ff0000'; sCtx.fillRect(0, 0, hw, hh);
              sCtx.fillStyle = '#00ff00'; sCtx.fillRect(hw, 0, w - hw, hh);
              sCtx.fillStyle = '#0000ff'; sCtx.fillRect(0, hh, hw, h - hh);
              sCtx.fillStyle = '#ffff00'; sCtx.fillRect(hw, hh, w - hw, h - hh);
            };
            const expectedCorners = ['RGBY', 'BRYG', 'YBGR', 'GYRB', 'GRYB', 'RBGY', 'BYRG', 'YGBR'];
            let seq = 1;
            const epoch = 901;
            let quarterTurnCount = 0;
            for (let o = 0; o < 8; o++) {
              const swapped = (o & 1) === 1;
              const sw = swapped ? 360 : 640;
              const sh = swapped ? 640 : 360;
              paintQuadrants(sw, sh);
              const rawImgData = (scanDirIdx) => sCtx.getImageData(0, 0, sw, sh).data.buffer.slice(0);
              for (let sd = 0; sd < 4; sd++) {
                probe.postWorkerMsg({ type: 'set_roi', native1to1: true, normX: 0.5, normY: 0.5, orientation: o, scanDir: sd });
                let ack;
                const ts = seq * 33333;
                if ((sd & 1) === 0) {
                  const bmp = await createImageBitmap(srcCanvas);
                  ack = await probe.submitTestFrame({
                    type: 'video_frame',
                    epoch,
                    seq,
                    bitmap: bmp,
                    dtSec: 1 / 30,
                    mediaTimeSec: seq / 30,
                    tSrcUs: ts,
                  }, [bmp]);
                } else {
                  const buf = rawImgData(sd);
                  ack = await probe.submitTestFrame({
                    type: 'video_rgba',
                    epoch,
                    seq,
                    width: sw,
                    height: sh,
                    buffer: buf,
                    dtSec: 1 / 30,
                    mediaTimeSec: seq / 30,
                    tSrcUs: ts,
                  }, [buf]);
                }
                if (!ack || ack.type !== 'frame_ack') {
                  throw new Error('Step 9f matrix frame failed o=' + o + ' sd=' + sd + ': ' + JSON.stringify(ack));
                }
                const snap = probe.getLastWave1();
                if (!snap || !snap.roi || snap.roi.orientation !== o || snap.roi.scanDir !== sd || !snap.roi.native1to1Active) {
                  throw new Error('Step 9f ROI metadata mismatch o=' + o + ' sd=' + sd + ': ' + JSON.stringify(snap?.roi));
                }
                if (swapped) {
                  if (snap.roi.srcW !== 360 || snap.roi.srcH !== 640 || snap.roi.engineW !== 640 || snap.roi.engineH !== 360) {
                    throw new Error('Step 9f quarter-turn dims mismatch o=' + o + ': ' + JSON.stringify(snap.roi));
                  }
                  quarterTurnCount++;
                }
                let gotColors = '';
                for (let rf = 0; rf < 24; rf++) {
                  await waitRafs();
                  gotColors = sampleCornerColors();
                  if (gotColors === expectedCorners[o]) break;
                }
                if (gotColors !== expectedCorners[o]) {
                  throw new Error('Step 9f HUD D4 mismatch o=' + o + ' sd=' + sd + ': got ' + gotColors + ' expected ' + expectedCorners[o]);
                }
                seq++;
              }
            }

            // Undersized 160x90 sharp edge non-smoothed fallback check
            probe.postWorkerMsg({ type: 'set_roi', native1to1: true, normX: 0.5, normY: 0.5, orientation: 0, scanDir: 0 });
            srcCanvas.width = 160;
            srcCanvas.height = 90;
            sCtx.fillStyle = '#ff0000'; sCtx.fillRect(0, 0, 80, 90);
            sCtx.fillStyle = '#0000ff'; sCtx.fillRect(80, 0, 80, 90);
            const smallBmp = await createImageBitmap(srcCanvas);
            const smallAck = await probe.submitTestFrame({
              type: 'video_frame',
              epoch,
              seq,
              bitmap: smallBmp,
              dtSec: 1 / 30,
              mediaTimeSec: seq / 30,
              tSrcUs: seq * 33333,
            }, [smallBmp]);
            seq++;
            if (!smallAck || smallAck.type !== 'frame_ack') throw new Error('Step 9f undersized ack failed');
            const smallSnap = probe.getLastWave1();
            if (!smallSnap?.roi || smallSnap.roi.native1to1Active !== false || smallSnap.roi.fallbackScale !== true) {
              throw new Error('Step 9f undersized fallback flag mismatch: ' + JSON.stringify(smallSnap?.roi));
            }
            let leftPx = null;
            let rightPx = null;
            for (let rf = 0; rf < 24; rf++) {
              await waitRafs();
              rCtx.drawImage(hudEl, 0, 0);
              leftPx = rCtx.getImageData(478, 140, 1, 1).data;
              rightPx = rCtx.getImageData(482, 140, 1, 1).data;
              if (leftPx[1] <= 15 && rightPx[1] <= 15 && leftPx[0] >= 200 && rightPx[2] >= 200) break;
            }
            if (!leftPx || !rightPx || leftPx[0] < 240 || leftPx[2] > 15 || rightPx[2] < 240 || rightPx[0] > 15) {
              throw new Error('Step 9f undersized fallback was smoothed! L=' + Array.from(leftPx || []) + ' R=' + Array.from(rightPx || []));
            }

            // preScaled=true on 640x360 must never mark native1to1Active=true
            srcCanvas.width = 640;
            srcCanvas.height = 360;
            sCtx.fillStyle = '#00ff00'; sCtx.fillRect(0, 0, 640, 360);
            const preBmp = await createImageBitmap(srcCanvas);
            const preAck = await probe.submitTestFrame({
              type: 'video_frame',
              epoch,
              seq,
              bitmap: preBmp,
              preScaled: true,
              dtSec: 1 / 30,
              mediaTimeSec: seq / 30,
              tSrcUs: seq * 33333,
            }, [preBmp]);
            seq++;
            if (!preAck || preAck.type !== 'frame_ack') throw new Error('Step 9f preScaled ack failed');
            const preSnap = probe.getLastWave1();
            if (!preSnap?.roi || preSnap.roi.native1to1Active !== false || preSnap.roi.fallbackScale !== true) {
              throw new Error('Step 9f preScaled native flag mismatch: ' + JSON.stringify(preSnap?.roi));
            }

            // Invalid byteLength video_rgba must reject without deadlocking busy state
            const commitsBeforeRej = preSnap.wave1.totalCommits;
            const badBuf = new Uint8Array(1024).buffer;
            const rejAck = await probe.submitTestFrame({
              type: 'video_rgba',
              epoch,
              seq,
              width: 640,
              height: 360,
              buffer: badBuf,
              dtSec: 1 / 30,
              mediaTimeSec: seq / 30,
              tSrcUs: seq * 33333,
            }, [badBuf]);
            seq++;
            const busyAfterRej = probe.getBusyState().workerFrameBusy;
            if (!rejAck || rejAck.type !== 'ingest_rejected' || busyAfterRej !== false) {
              throw new Error('Step 9f reject contract failed: ' + JSON.stringify({ rejAck, busyAfterRej }));
            }
            const recBmp = await createImageBitmap(srcCanvas);
            const recAck = await probe.submitTestFrame({
              type: 'video_frame',
              epoch,
              seq,
              bitmap: recBmp,
              dtSec: 1 / 30,
              mediaTimeSec: seq / 30,
              tSrcUs: seq * 33333,
            }, [recBmp]);
            const recSnap = probe.getLastWave1();
            if (!recAck || recAck.type !== 'frame_ack' || recSnap.wave1.totalCommits !== commitsBeforeRej + 1) {
              throw new Error('Step 9f post-reject recovery failed: ' + JSON.stringify({ recAck, commitsBeforeRej, after: recSnap?.wave1?.totalCommits }));
            }

            // Browser visibleRect provenance: coded 640x360 with visibleRect (40,20,560,320) + x0=10,y0=5 -> RingMeta (560x320, crop=10,5), sampled at coded (50,25)
            probe.postWorkerMsg({ type: 'set_resolution', resMode: 0 });
            probe.postWorkerMsg({ type: 'set_roi', native1to1: true, x0: 10, y0: 5, orientation: 0, scanDir: 0 });
            srcCanvas.width = 640;
            srcCanvas.height = 360;
            sCtx.fillStyle = '#001000'; sCtx.fillRect(0, 0, 640, 360);
            sCtx.fillStyle = '#ff00ff'; sCtx.fillRect(50, 25, 320, 180);
            const vrBmp = await createImageBitmap(srcCanvas);
            const vrAck = await probe.submitTestFrame({
              type: 'video_frame',
              epoch: epoch + 1,
              seq: 1,
              bitmap: vrBmp,
              dtSec: 1 / 30,
              mediaTimeSec: 0.0,
              tSrcUs: 0,
              geom: { visibleRect: { x: 40, y: 20, width: 560, height: 320 } },
            }, [vrBmp]);
            if (!vrAck || vrAck.type !== 'frame_ack') {
              throw new Error('Step 9f visibleRect frame failed: ' + JSON.stringify(vrAck));
            }
            const vrSnap = probe.getLastWave1();
            if (!vrSnap?.roi || vrSnap.roi.srcW !== 560 || vrSnap.roi.srcH !== 320 || vrSnap.roi.cropX0 !== 10 || vrSnap.roi.cropY0 !== 5 || !vrSnap.roi.native1to1Active) {
              throw new Error('Step 9f visibleRect ROI coords mismatch: ' + JSON.stringify(vrSnap?.roi));
            }
            let vrPx = null;
            for (let rf = 0; rf < 24; rf++) {
              await waitRafs();
              rCtx.drawImage(hudEl, 0, 0);
              vrPx = rCtx.getImageData(480, 270, 1, 1).data;
              if (vrPx[0] > 200 && vrPx[2] > 200 && vrPx[1] < 55) break;
            }
            if (!vrPx || vrPx[0] < 200 || vrPx[2] < 200 || vrPx[1] > 55) {
              throw new Error('Step 9f visibleRect did not sample coded (50,25) magenta region: ' + Array.from(vrPx || []));
            }

            // Synchronous dispatchBitmapFrameToWorker DataCloneError (closed ImageBitmap) clears workerFrameBusy=false and does not wedge
            const deadBmp = await createImageBitmap(srcCanvas);
            deadBmp.close();
            const busyBeforeErr = probe.getBusyState();
            const syncDispatchOk = probe.dispatchBitmapFrame({
              epoch: busyBeforeErr.sourceEpoch,
              seq: busyBeforeErr.frameSeq + 50,
              bitmap: deadBmp,
              dtSec: 1 / 30,
              mediaTimeSec: -1,
              tSrcUs: null,
              captureAgeMs: -1,
              enqueuedEpochMs: performance.now(),
              gapMask: 0,
              geom: null,
            });
            const busyAfterErr = probe.getBusyState();
            if (syncDispatchOk !== false || busyAfterErr.workerFrameBusy !== false) {
              throw new Error('Step 9f synchronous transfer throw leaked busy state: ' + JSON.stringify({ syncDispatchOk, busyAfterErr }));
            }
            const afterDeadBmp = await createImageBitmap(srcCanvas);
            const afterDeadAck = await probe.submitTestFrame({
              type: 'video_frame',
              epoch: epoch + 1,
              seq: 2,
              bitmap: afterDeadBmp,
              dtSec: 1 / 30,
              mediaTimeSec: 1 / 30,
              tSrcUs: 33333,
            }, [afterDeadBmp]);
            if (!afterDeadAck || afterDeadAck.type !== 'frame_ack') {
              throw new Error('Step 9f post-DataCloneError worker recovery failed: ' + JSON.stringify(afterDeadAck));
            }

            // lastPresentedFrames skip-gap preservation when bitmapCaptureBusy drops frame curP+1
            const hiddenVid = document.getElementById('hiddenVideoSource');
            if (!hiddenVid || hiddenVid.readyState < 2) {
              throw new Error('Step 9f expected loaded hiddenVideoSource for capture-busy skip check');
            }
            await hiddenVid.play().catch(() => {});
            probe.postWorkerMsg({ type: 'params', lookaheadK: 0, rawWipeRatio: 1.0, showHeroTargets: false });
            for (let wt = 0; wt < 60 && (probe.getBusyState().lastPresentedFrames < 1 || probe.getBusyState().bitmapCaptureBusy || probe.getBusyState().workerFrameBusy); wt++) {
              await new Promise((res) => setTimeout(res, 25));
            }
            const busy0 = probe.getBusyState();
            if (busy0.lastPresentedFrames < 1 || busy0.bitmapCaptureBusy || busy0.workerFrameBusy) {
              throw new Error('Step 9f video pump did not become ready for skip-gap check: ' + JSON.stringify(busy0));
            }
            const curP = busy0.lastPresentedFrames;
            probe.setCaptureBusyForTest(true);
            const droppedWhileBusy = await probe.forwardVideoFrameForTest(
              -1,
              performance.now(),
              { presentedFrames: curP + 1 }
            );
            const busyDuringDrop = probe.getBusyState();
            probe.setCaptureBusyForTest(false);
            if (droppedWhileBusy !== false || busyDuringDrop.lastPresentedFrames !== curP) {
              hiddenVid.pause();
              throw new Error('Step 9f bitmapCaptureBusy must not advance lastPresentedFrames: ' + JSON.stringify({ droppedWhileBusy, curP, got: busyDuringDrop.lastPresentedFrames }));
            }
            const pNext = probe.forwardVideoFrameForTest(
              -1,
              performance.now() + 33,
              { presentedFrames: curP + 2 }
            );
            hiddenVid.pause();
            const nextSent = await pNext;
            if (nextSent !== true || probe.getBusyState().lastPresentedFrames !== curP + 2) {
              throw new Error('Step 9f post-busy skip capture did not advance lastPresentedFrames to curP+2: ' + JSON.stringify({ nextSent, got: probe.getBusyState().lastPresentedFrames, curP }));
            }
            for (let wt = 0; wt < 60 && probe.getBusyState().workerFrameBusy; wt++) {
              await new Promise((res) => setTimeout(res, 20));
            }
            const skipSnap = probe.getLastWave1();
            if (!skipSnap?.wave1?.dispMeta || (skipSnap.wave1.dispMeta.frameGapFlag & 1) === 0) {
              throw new Error('Step 9f dropped presentedFrame curP+1 failed to emit FRAME_GAP_SEQ (bit 0): ' + JSON.stringify(skipSnap?.wave1?.dispMeta));
            }

            // bitmapCaptureBusy drop without presentedFrames still preserves FRAME_GAP_SEQ (bit 0)
            if (hiddenVid.ended) hiddenVid.currentTime = 0;
            await hiddenVid.play().catch(() => {});
            probe.setCaptureBusyForTest(true);
            const dropNoPres = await probe.forwardVideoFrameForTest(-1, performance.now() + 66, null);
            const busyGapNoPres = probe.getBusyState().pendingVideoGapMask;
            hiddenVid.pause();
            probe.setCaptureBusyForTest(false);
            if (dropNoPres !== false || (busyGapNoPres & 1) === 0) {
              throw new Error('Step 9f bitmapCaptureBusy without presentedFrames failed to set bit0: ' + JSON.stringify({ dropNoPres, busyGapNoPres, paused: hiddenVid.paused, ended: hiddenVid.ended, readyState: hiddenVid.readyState }));
            }

            // Async in-flight createImageBitmap overlap must preserve FRAME_GAP_SEQ across F1 commit for F3
            const origCreateImageBitmap = window.createImageBitmap;
            try {
              // Flush pending bit cleanly via F0 first
              if (hiddenVid.ended) hiddenVid.currentTime = 0;
              await hiddenVid.play().catch(() => {});
              const pF0 = Math.max(curP + 3, probe.getBusyState().lastPresentedFrames + 1);
              const f0Sent = await probe.forwardVideoFrameForTest(-1, performance.now() + 80, { presentedFrames: pF0 });
              hiddenVid.pause();
              for (let wt = 0; wt < 60 && (probe.getBusyState().bitmapCaptureBusy || probe.getBusyState().workerFrameBusy); wt++) {
                await new Promise((res) => setTimeout(res, 20));
              }
              if ((f0Sent !== true && probe.getBusyState().lastPresentedFrames < pF0) || probe.getBusyState().pendingVideoGapMask !== 0) {
                throw new Error('Step 9f async overlap pre-flush failed: ' + JSON.stringify({ f0Sent, busy: probe.getBusyState() }));
              }
              let releaseF1Bitmap = null;
              let rejectF1Bitmap = null;
              let f1Intercepted = false;
              window.createImageBitmap = (src, ...rest) => {
                if (!f1Intercepted && src === hiddenVid) {
                  f1Intercepted = true;
                  return new Promise((res, rej) => {
                    releaseF1Bitmap = () => origCreateImageBitmap(srcCanvas).then(res, rej);
                    rejectF1Bitmap = rej;
                  });
                }
                return origCreateImageBitmap(src, ...rest);
              };
              if (hiddenVid.ended) hiddenVid.currentTime = 0;
              await hiddenVid.play().catch(() => {});
              const pF1 = probe.getBusyState().lastPresentedFrames + 1;
              const f1Promise = probe.forwardVideoFrameForTest(-1, performance.now() + 110, { presentedFrames: pF1 });
              for (let wt = 0; wt < 30 && !releaseF1Bitmap; wt++) {
                await new Promise((res) => setTimeout(res, 10));
              }
              if (!releaseF1Bitmap || !probe.getBusyState().bitmapCaptureBusy) {
                if (rejectF1Bitmap) rejectF1Bitmap(new Error('abort_overlap'));
                hiddenVid.pause();
                throw new Error('Step 9f failed to pause inside in-flight createImageBitmap');
              }
              const f2Drop = await probe.forwardVideoFrameForTest(-1, performance.now() + 125, { presentedFrames: pF1 + 1 });
              const midOverlapGap = probe.getBusyState().pendingVideoGapMask;
              hiddenVid.pause();
              window.createImageBitmap = origCreateImageBitmap;
              releaseF1Bitmap();
              const f1Sent = await f1Promise;
              for (let wt = 0; wt < 60 && (probe.getBusyState().bitmapCaptureBusy || probe.getBusyState().workerFrameBusy); wt++) {
                await new Promise((res) => setTimeout(res, 20));
              }
              const gapAfterF1 = probe.getBusyState().pendingVideoGapMask;
              if (f2Drop !== false || (midOverlapGap & 1) === 0 || f1Sent !== true || (gapAfterF1 & 1) === 0) {
                throw new Error('Step 9f async createImageBitmap overlap lost pendingVideoGapMask: ' + JSON.stringify({ f2Drop, midOverlapGap, f1Sent, gapAfterF1 }));
              }
              const commitsBeforeF3 = probe.getLastWave1()?.wave1?.totalCommits ?? 0;
              await hiddenVid.play().catch(() => {});
              const pF3 = Math.max(pF1 + 2, probe.getBusyState().lastPresentedFrames + 1);
              let f3Sent = false;
              if (probe.getBusyState().pendingVideoGapMask !== 0) {
                f3Sent = await probe.forwardVideoFrameForTest(-1, performance.now() + 145, { presentedFrames: pF3 });
              }
              hiddenVid.pause();
              for (let wt = 0; wt < 60 && (probe.getBusyState().bitmapCaptureBusy || probe.getBusyState().workerFrameBusy); wt++) {
                await new Promise((res) => setTimeout(res, 20));
              }
              const f3Snap = probe.getLastWave1();
              const committedByF3OrRvfc = f3Sent === true || (f3Snap?.wave1?.totalCommits ?? 0) > commitsBeforeF3;
              if (!committedByF3OrRvfc || probe.getBusyState().pendingVideoGapMask !== 0 || !f3Snap?.wave1?.dispMeta || (f3Snap.wave1.dispMeta.frameGapFlag & 1) === 0) {
                throw new Error('Step 9f F3 after async overlap failed to commit FRAME_GAP_SEQ: ' + JSON.stringify({ f3Sent, committedByF3OrRvfc, busy: probe.getBusyState(), dispMeta: f3Snap?.wave1?.dispMeta }));
              }

              // Epoch-switch held-createImageBitmap lock regression: stale epoch-1 finish must NOT unlock epoch-2 capture B or allow C to overtake B
              let releaseEp1 = null;
              let releaseEp2 = null;
              let epTrap = 0;
              window.createImageBitmap = (src, ...rest) => {
                if (src === hiddenVid && epTrap < 2) {
                  epTrap++;
                  const trapIdx = epTrap;
                  return new Promise((res, rej) => {
                    const rel = () => origCreateImageBitmap(srcCanvas).then(res, rej);
                    if (trapIdx === 1) releaseEp1 = rel;
                    else releaseEp2 = rel;
                  });
                }
                return origCreateImageBitmap(src, ...rest);
              };
              const tEp1 = Math.max(20.0, Math.ceil((hiddenVid.currentTime || 0) + 2)) + 0.1;
              if (hiddenVid.ended) hiddenVid.currentTime = 0;
              await hiddenVid.play().catch(() => {});
              const ep1Promise = probe.forwardVideoFrameForTest(tEp1, performance.now() + 180, null);
              for (let wt = 0; wt < 30 && !releaseEp1; wt++) {
                await new Promise((res) => setTimeout(res, 10));
              }
              if (!releaseEp1 || !probe.getBusyState().bitmapCaptureBusy) {
                throw new Error('Step 9f epoch-switch test failed to trap epoch-1 createImageBitmap');
              }
              const ep2 = probe.simulateEpochFlushForTest();
              if (probe.getBusyState().bitmapCaptureBusy !== false) {
                throw new Error('Step 9f epoch-switch flush did not reset bitmapCaptureBusy for new epoch');
              }
              const ep2Promise = probe.forwardVideoFrameForTest(0.2, performance.now() + 200, null);
              for (let wt = 0; wt < 30 && !releaseEp2; wt++) {
                await new Promise((res) => setTimeout(res, 10));
              }
              if (!releaseEp2 || !probe.getBusyState().bitmapCaptureBusy) {
                throw new Error('Step 9f epoch-switch test failed to trap epoch-2 Capture B createImageBitmap');
              }
              // Stale epoch-1 capture finishes first while epoch-2 Capture B is in flight
              releaseEp1();
              const ep1Result = await ep1Promise;
              if (ep1Result !== false || probe.getBusyState().bitmapCaptureBusy !== true) {
                throw new Error('Step 9f stale epoch-1 finish cleared epoch-2 bitmapCaptureBusy lock: ' + JSON.stringify({ ep1Result, busy: probe.getBusyState() }));
              }
              // Capture C arrives while Capture B is still awaiting: MUST NOT leapfrog Capture B!
              const ep2CDrop = await probe.forwardVideoFrameForTest(0.3, performance.now() + 215, null);
              if (ep2CDrop !== false || (probe.getBusyState().pendingVideoGapMask & 1) === 0) {
                throw new Error('Step 9f epoch-2 Capture C overtook in-flight Capture B: ' + JSON.stringify({ ep2CDrop, busy: probe.getBusyState() }));
              }
              hiddenVid.pause();
              window.createImageBitmap = origCreateImageBitmap;
              releaseEp2();
              const ep2BSent = await ep2Promise;
              for (let wt = 0; wt < 60 && (probe.getBusyState().bitmapCaptureBusy || probe.getBusyState().workerFrameBusy); wt++) {
                await new Promise((res) => setTimeout(res, 20));
              }
              const bMeta = probe.getLastWave1()?.wave1?.dispMeta;
              if (ep2BSent !== true || !bMeta || bMeta.sourceEpoch !== ep2 || bMeta.tSrcUs !== 200000) {
                throw new Error('Step 9f epoch-2 Capture B (0.2s) did not land first cleanly: ' + JSON.stringify({ ep2BSent, ep2, bMeta }));
              }
              await hiddenVid.play().catch(() => {});
              const ep2DSent = await probe.forwardVideoFrameForTest(0.3, performance.now() + 235, null);
              hiddenVid.pause();
              for (let wt = 0; wt < 60 && (probe.getBusyState().bitmapCaptureBusy || probe.getBusyState().workerFrameBusy); wt++) {
                await new Promise((res) => setTimeout(res, 20));
              }
              const dMeta = probe.getLastWave1()?.wave1?.dispMeta;
              if (ep2DSent !== true || !dMeta || dMeta.sourceEpoch !== ep2 || dMeta.tSrcUs !== 300000 || dMeta.sourceSeq <= bMeta.sourceSeq || (dMeta.frameGapFlag & 1) === 0) {
                throw new Error('Step 9f epoch-2 Capture D (0.3s) ordering/gap mismatch after B: ' + JSON.stringify({ ep2DSent, ep2, bSeq: bMeta.sourceSeq, dMeta }));
              }
            } finally {
              window.createImageBitmap = origCreateImageBitmap;
              hiddenVid.pause();
            }

            // Live burst + pause settle check on #hudCanvas + repeat set_resolution(1->1) check
            probe.postWorkerMsg({ type: 'set_resolution', resMode: 1 });
            probe.postWorkerMsg({ type: 'set_resolution', resMode: 1 });
            probe.postWorkerMsg({ type: 'set_roi', native1to1: true, normX: 0.5, normY: 0.5, orientation: 0, scanDir: 0 });
            srcCanvas.width = 640;
            srcCanvas.height = 360;
            const burstEp = Math.max(epoch + 2, busy0.sourceEpoch + 2);
            const burstColors = ['#ff0000', '#ff0000', '#ff0000', '#00ffff'];
            for (let bi = 0; bi < burstColors.length; bi++) {
              sCtx.fillStyle = burstColors[bi];
              sCtx.fillRect(0, 0, 640, 360);
              const bBmp = await createImageBitmap(srcCanvas);
              const bAck = await probe.submitTestFrame({
                type: 'video_frame',
                epoch: burstEp,
                seq: bi + 1,
                bitmap: bBmp,
                dtSec: 1 / 30,
                mediaTimeSec: bi / 30,
                tSrcUs: bi * 33333,
              }, [bBmp]);
              if (!bAck || bAck.type !== 'frame_ack') {
                throw new Error('Step 9f burst+pause frame failed bi=' + bi + ': ' + JSON.stringify(bAck));
              }
            }
            let settlePx = null;
            for (let rf = 0; rf < 24; rf++) {
              await waitRafs();
              rCtx.drawImage(hudEl, 0, 0);
              settlePx = rCtx.getImageData(480, 270, 1, 1).data;
              if (settlePx[0] < 35 && settlePx[1] > 210 && settlePx[2] > 210) break;
            }
            if (!settlePx || settlePx[0] > 35 || settlePx[1] < 210 || settlePx[2] < 210) {
              throw new Error('Step 9f burst+pause #hudCanvas failed to settle on final cyan frame: ' + Array.from(settlePx || []));
            }

            // Browser live Wave-1 owned Auto-Hero promotion (activeAutoHeroes >= 1) and mode=0 retirement
            probe.postWorkerMsg({ type: 'set_source', sourceMode: 'video', fpsHint: 30 });
            probe.postWorkerMsg({ type: 'set_roi', native1to1: true, normX: 0.5, normY: 0.5, orientation: 0, scanDir: 0 });
            probe.postWorkerMsg({ type: 'set_swarm_mode', swarmMode: 2 });
            probe.postWorkerMsg({ type: 'set_metrology_config', metrologyMode: 2, showCovEllipses: true });
            probe.postWorkerMsg({ type: 'params', lookaheadK: 0, rawWipeRatio: 0.0, showHeroTargets: true, showcaseMinimalScopes: false });
            const promoEpoch = 902;
            const sheetBuf = new Uint8Array(640 * 360 * 4);
            const drawGaussianDot = (cx, cy, amp) => {
              const x0 = Math.max(0, Math.floor(cx - 5));
              const x1 = Math.min(639, Math.ceil(cx + 5));
              const y0 = Math.max(0, Math.floor(cy - 5));
              const y1 = Math.min(359, Math.ceil(cy + 5));
              for (let y = y0; y <= y1; y++) {
                const dy = y - cy;
                for (let x = x0; x <= x1; x++) {
                  const dx = x - cx;
                  const v = Math.round(amp * 255 * Math.exp(-0.5 * (dx * dx + dy * dy) / (1.4 * 1.4)));
                  const o = (y * 640 + x) * 4;
                  const nv = Math.min(255, sheetBuf[o] + v);
                  sheetBuf[o] = nv;
                  sheetBuf[o + 1] = nv;
                  sheetBuf[o + 2] = nv;
                }
              }
            };
            const fillGaussianSheet = (frameIdx) => {
              for (let i = 0; i < 640 * 360 * 4; i += 4) {
                sheetBuf[i] = 38;
                sheetBuf[i + 1] = 38;
                sheetBuf[i + 2] = 38;
                sheetBuf[i + 3] = 255;
              }
              for (let r = 0; r < 4; r++) {
                for (let c = 0; c < 6; c++) {
                  drawGaussianDot(70 + c * 98 + 0.35 * frameIdx, 55 + r * 80 + 0.1 * frameIdx, 0.28);
                }
              }
              let ox = 200 + 0.35 * frameIdx;
              let oy = 180 + 0.1 * frameIdx;
              if (frameIdx >= 16) {
                const tau = frameIdx - 16;
                ox += 24 * (1 - Math.cos(0.32 * tau));
                oy += 22 * Math.sin(0.35 * tau);
              }
              drawGaussianDot(ox, oy, 0.34);
            };
            let sawAutoPromo = 0;
            let sawActiveTgts = 0;
            for (let t = 0; t < 28; t++) {
              fillGaussianSheet(t);
              const ab = sheetBuf.buffer.slice(0);
              const sAck = await probe.submitTestFrame({
                type: 'video_rgba',
                epoch: promoEpoch,
                seq: t + 1,
                width: 640,
                height: 360,
                buffer: ab,
                dtSec: 1 / 30,
                mediaTimeSec: t / 30,
                tSrcUs: t * 33333,
              }, [ab]);
              if (!sAck || sAck.type !== 'frame_ack') throw new Error('Step 9f-hero frame failed t=' + t);
              const hSnap = probe.getLastWave1();
              if (hSnap?.metrology?.activeAutoHeroes >= 1 && hSnap?.metrology?.promotedCandidates >= 1) {
                sawAutoPromo = Math.max(sawAutoPromo, hSnap.metrology.activeAutoHeroes);
                sawActiveTgts = Math.max(sawActiveTgts, hSnap.activeTargets || 0);
              }
            }
            if (sawAutoPromo < 1 || sawAutoPromo > 3 || sawActiveTgts < 1) {
              throw new Error('Step 9f-hero failed to promote owned Auto-Hero in browser: ' + JSON.stringify({ sawAutoPromo, sawActiveTgts, last: probe.getLastWave1()?.metrology }));
            }
            probe.postWorkerMsg({ type: 'set_metrology_config', metrologyMode: 0, showCovEllipses: false });
            for (let t = 28; t < 30; t++) {
              fillGaussianSheet(t);
              const ab = sheetBuf.buffer.slice(0);
              await probe.submitTestFrame({
                type: 'video_rgba',
                epoch: promoEpoch,
                seq: t + 1,
                width: 640,
                height: 360,
                buffer: ab,
                dtSec: 1 / 30,
                mediaTimeSec: t / 30,
                tSrcUs: t * 33333,
              }, [ab]);
            }
            const retSnap = probe.getLastWave1();
            if (!retSnap || retSnap.metrology.activeAutoHeroes !== 0 || retSnap.metrology.promotedCandidates !== 0 || retSnap.activeTargets !== 0 || (retSnap.wave1.enabledBits & 1) !== 0) {
              throw new Error('Step 9f-hero mode=0 failed to retire owned Auto-Hero: ' + JSON.stringify({ retMet: retSnap?.metrology, tgts: retSnap?.activeTargets, en: retSnap?.wave1?.enabledBits }));
            }
            return { matrixFrames: 32, quarterTurnCount, commitsAfter: retSnap.wave1.totalCommits, sawAutoPromo, retiredAutoHeroes: retSnap.metrology.activeAutoHeroes };
          })()`
        );

        await new Promise((r) => setTimeout(r, 80));
        const w1DeckCls = await cdp.evalJs(
          w1Session,
          `(() => {
            if (window.__e2eW1DeckClsPo) window.__e2eW1DeckClsPo.disconnect();
            return window.__e2eW1DeckCls || 0;
          })()`
        );
        if (w1DeckCls >= 0.000001) {
          throw new Error(`Step 9b-9f: Expected 0.000000 CLS across Deck BOS/ROI/Failure/Metrology/Orientation flows, got ${w1DeckCls}`);
        }
        const totalW1Cls = w1ShowcaseCls + w1DeckCls;

        console.log(
          `  [PASS] Step 9 (Wave-1 BOS Plume, Native 1:1 ROI, Tripod Failure, 2x2 Cov/Decoy & 8x4 D4 Matrix): bos=${bosReadyState.diagBos}, stamps=${bosReadyState.diagStamps}, hud=${bosHudBytes}B, roiOffset="${roiOffsetState.roiTxt}", failure="${failureBannerState.diagBos}" (${failureBannerState.sev}), decoy="${decoyState.diagCov}", denseCov="${denseCovState.diagCov}", metOffCands=${starlingMetOffState.cand}, matrix=${orientMatrixReport.matrixFrames} (qturn=${orientMatrixReport.quarterTurnCount}), ownedAutoPromo=${orientMatrixReport.sawAutoPromo}->${orientMatrixReport.retiredAutoHeroes}, CLS=${totalW1Cls.toFixed(6)}`
        );
      } finally {
        await cdp.send('Target.closeTarget', { targetId: w1TargetId }).catch(() => {});
      }
    }

    // 10. Wave-2 Browser E2E: 40x22 Mesh Hodge, Calibrated RS+Welch (K>=8 + Exposure Null Banner), Occluded AMBIGUOUS & Zero-CLS
    {
      const baseUrl = `http://127.0.0.1:${srv.port}`;
      const { targetId: w2TargetId } = await cdp.send('Target.createTarget', { url: `${baseUrl}/index.html?view=deck` });
      const { sessionId: w2Session } = await cdp.send('Target.attachToTarget', { targetId: w2TargetId, flatten: true });
      try {
        await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: true }, w2Session);
        await cdp.send('Page.enable', {}, w2Session);
        await cdp.send('Log.enable', {}, w2Session);
        await cdp.send('Runtime.enable', {}, w2Session);
        await cdp.send(
          'Emulation.setDeviceMetricsOverride',
          { width: 1440, height: 900, deviceScaleFactor: process.env.CI ? 1 : 2, mobile: false },
          w2Session
        );

        await waitForCondition(
          cdp,
          w2Session,
          'Step 10a: Deck booted cleanly with Wave-2 probe & controls',
          `(() => {
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            return {
              ok: Boolean(document.getElementById('hodgeModeSelect')) &&
                  Boolean(document.getElementById('rsVibroModeSelect')) &&
                  Boolean(document.getElementById('inpRsDtRowUs')) &&
                  Boolean(document.getElementById('inpRsExposureUs')) &&
                  Boolean(document.getElementById('selRsScanDir')) &&
                  Boolean(document.getElementById('diagHodge2Complex')) &&
                  Boolean(w2 && w2.available === true && w2.vibro?.available && w2.hodge?.available && w2.conservation?.available),
              avail: Boolean(w2?.available),
            };
          })()`,
          20000,
          true
        );

        await cdp.evalJs(
          w2Session,
          `(() => {
            window.__e2eW2DeckCls = 0;
            window.__e2eW2DeckClsPo = new PerformanceObserver((list) => {
              for (const e of list.getEntries()) {
                if (!e.hadRecentInput) window.__e2eW2DeckCls += e.value;
              }
            });
            window.__e2eW2DeckClsPo.observe({ type: 'layout-shift', buffered: false });
            const rm = document.getElementById('rsVibroModeSelect');
            if (rm) {
              rm.value = '1';
              rm.dispatchEvent(new Event('change', { bubbles: true }));
            }
            return true;
          })()`
        );

        await waitForCondition(
          cdp,
          w2Session,
          'Step 10a: Fresh Deck UI RS Mode 1 reaches READY without probe overrides',
          `(() => {
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            const v = w2?.vibro;
            const diag = document.getElementById('diagRsWelch')?.textContent || '';
            return {
              ok: Boolean(v && v.state === 3 && diag.startsWith('READY · ')),
              state: v?.state,
              reason: v?.reasonFlags,
              diag,
            };
          })()`,
          15000
        );

        await cdp.evalJs(
          w2Session,
          `(() => {
            const rm = document.getElementById('rsVibroModeSelect');
            if (rm) {
              rm.value = '0';
              rm.dispatchEvent(new Event('change', { bubbles: true }));
            }
            const chkNative = document.getElementById('chkNativeRoi');
            if (chkNative && chkNative.checked) {
              chkNative.checked = false;
              chkNative.dispatchEvent(new Event('change', { bubbles: true }));
            }
            document.querySelector('[data-showcase="vortex_1200"]')?.click();
            return true;
          })()`
        );

        await waitForCondition(
          cdp,
          w2Session,
          'Step 10a: vortex_1200 swarm initialized before enabling Hodge',
          `(() => {
            const btn = document.querySelector('[data-showcase="vortex_1200"]');
            const meta = document.getElementById('continuumSummaryMeta')?.textContent || '';
            return {
              ok: btn?.getAttribute('aria-pressed') === 'true' && meta.startsWith('SWARM: ACTIVE'),
              meta,
            };
          })()`,
          20000
        );

        await cdp.evalJs(
          w2Session,
          `(() => {
            const hm = document.getElementById('hodgeModeSelect');
            hm.value = '2';
            hm.dispatchEvent(new Event('change', { bubbles: true }));
            const ov = document.getElementById('fieldOverlayModeSelect');
            ov.value = '9';
            ov.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
          })()`
        );

        const hodgeState = await waitForCondition(
          cdp,
          w2Session,
          'Step 10a: 40x22 Mesh Hodge on vortex_1200 converges (READY, coexactFrac > 0.85, orthEcRel < 1e-6)',
          `(() => {
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            const h = w2?.hodge;
            const diag = document.getElementById('diagHodge2Complex')?.textContent || '';
            const ok = Boolean(
              h &&
              h.state === 2 &&
              h.converged === true &&
              h.coexactFrac > 0.85 &&
              h.orthEcRel < 1e-6 &&
              typeof h.voidCx === 'number' &&
              typeof h.voidCy === 'number' &&
              Array.isArray(h.wakeBbox) &&
              h.wakeBbox.length === 4 &&
              !('voidR' in h) &&
              !('wakeR' in h) &&
              w2?.stamps?.hodge?.valid === true &&
              diag.startsWith('READY · ')
            );
            return {
              ok,
              state: h?.state,
              converged: h?.converged,
              exactFrac: h?.exactFrac,
              coexactFrac: h?.coexactFrac,
              harmFrac: h?.harmFrac,
              orthEcRel: h?.orthEcRel,
              beta1: h?.beta1,
              diag,
            };
          })()`,
          20000
        );

        for (const omVal of ['8', '10', '9']) {
          await cdp.evalJs(
            w2Session,
            `(() => {
              const ov = document.getElementById('fieldOverlayModeSelect');
              ov.value = ${JSON.stringify(omVal)};
              ov.dispatchEvent(new Event('change', { bubbles: true }));
              return true;
            })()`
          );
          await waitForCondition(
            cdp,
            w2Session,
            `Step 10a: Hodge fieldOverlayMode=${omVal} synced`,
            `(() => ({ ok: window.__chronosUiState?.overlays?.fieldOverlayMode === ${Number(omVal)} }))()`,
            4000
          );
        }

        const w2HudRect = await cdp.evalJs(
          w2Session,
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
        const w2HudShot = await cdp.send(
          'Page.captureScreenshot',
          { format: 'png', captureBeyondViewport: false, clip: { ...w2HudRect, scale: 1 } },
          w2Session
        );
        const w2HudBytes = w2HudShot?.data ? Buffer.from(w2HudShot.data, 'base64').byteLength : 0;
        if (w2HudBytes < 2500) {
          throw new Error(`Step 10a: Expected Hodge #hudCanvas screenshot >= 2500 B, got ${w2HudBytes} B`);
        }

        // Switching to a preset without an overlayMode (model_rockets) must reset orphaned Hodge overlay 9 -> 0
        await cdp.evalJs(
          w2Session,
          `(() => {
            document.querySelector('[data-showcase="model_rockets"]')?.click();
            return true;
          })()`
        );
        await waitForCondition(
          cdp,
          w2Session,
          'Step 10a: Preset without overlayMode resets orphaned Hodge overlay 9 -> 0',
          `(() => {
            const hm = document.getElementById('hodgeModeSelect')?.value;
            const ov = document.getElementById('fieldOverlayModeSelect')?.value;
            return {
              ok: hm === '0' && ov === '0' && window.__chronosUiState?.overlays?.fieldOverlayMode === 0,
              hm,
              ov,
              uiOm: window.__chronosUiState?.overlays?.fieldOverlayMode,
            };
          })()`,
          8000
        );
        await cdp.evalJs(
          w2Session,
          `(() => {
            document.getElementById('btnSrcTestbench')?.click();
            return true;
          })()`
        );

        // 10b. Calibrated RS + Welch (K >= 8, 880 Hz) & Exposure-Null Banner (T_e = 1/880s)
        const rsWelchReport = await cdp.evalJs(
          w2Session,
          `(async () => {
            const w1p = window.__chronosWave1Probe;
            const w2p = window.__chronosWave2Probe;
            w1p.postWorkerMsg({ type: 'set_source', sourceMode: 'video', fpsHint: 30 });
            w1p.postWorkerMsg({ type: 'set_resolution', resMode: 1 });
            w1p.postWorkerMsg({ type: 'set_swarm_mode', swarmMode: 0 });
            w1p.postWorkerMsg({ type: 'set_hodge_config', hodgeMode: 0 });
            w1p.postWorkerMsg({ type: 'set_bos_config', bosMode: 0 });
            w1p.postWorkerMsg({ type: 'set_roi', native1to1: true, normX: 0.5, normY: 0.5, orientation: 0, scanDir: 0 });
            w1p.postWorkerMsg({ type: 'params', lookaheadK: 0, rawWipeRatio: 0.0, showHeroTargets: false });

            const hm = document.getElementById('hodgeModeSelect');
            if (hm) hm.value = '0';
            const ov = document.getElementById('fieldOverlayModeSelect');
            if (ov) { ov.value = '11'; ov.dispatchEvent(new Event('change', { bubbles: true })); }
            const rm = document.getElementById('rsVibroModeSelect');
            if (rm) rm.value = '2';

            const W = 640;
            const H = 360;
            const frameBytes = W * H * 4;
            const rgba = new Uint8Array(frameBytes);

            const ax = 0.27, ay = 0.13, bx = 0.19, by = -0.31, cx = 0.37, cy = 0.29;
            const npix = W * H;
            const sA = new Float32Array(npix);
            const cA = new Float32Array(npix);
            const cB = new Float32Array(npix);
            const sB = new Float32Array(npix);
            const sC = new Float32Array(npix);
            const cC = new Float32Array(npix);
            for (let y = 0; y < H; y++) {
              const row = y * W;
              for (let x = 0; x < W; x++) {
                const idx = row + x;
                const pA = ax * x + ay * y;
                const pB = bx * x + by * y;
                const pC = cx * x + cy * y;
                sA[idx] = Math.sin(pA); cA[idx] = Math.cos(pA);
                cB[idx] = Math.cos(pB); sB[idx] = Math.sin(pB);
                sC[idx] = Math.sin(pC); cC[idx] = Math.cos(pC);
              }
            }

            const renderRsFrame = (tSrcUs, dtRowUs, exposureUs, fTone, ampPx, taps) => {
              const expSec = exposureUs * 1e-6;
              const invT = 1.0 / taps;
              const scA = new Float64Array(taps);
              const ccA = new Float64Array(taps);
              const scB = new Float64Array(taps);
              const ccB = new Float64Array(taps);
              const scC = new Float64Array(taps);
              const ccC = new Float64Array(taps);
              const w2piF = 2.0 * Math.PI * fTone;
              for (let y = 0; y < H; y++) {
                const tMid = (tSrcUs + y * dtRowUs + 0.5 * exposureUs) * 1e-6;
                for (let s = 0; s < taps; s++) {
                  const tS = taps === 1 ? tMid : (tMid + ((s + 0.5) * invT - 0.5) * expSec);
                  const ux = ampPx * Math.cos(w2piF * tS + 0.4);
                  scA[s] = Math.sin(ax * ux); ccA[s] = Math.cos(ax * ux);
                  scB[s] = Math.sin(bx * ux); ccB[s] = Math.cos(bx * ux);
                  scC[s] = Math.sin(cx * ux); ccC[s] = Math.cos(cx * ux);
                }
                const row = y * W;
                for (let x = 0; x < W; x++) {
                  const idx = row + x;
                  let acc = 0.0;
                  for (let s = 0; s < taps; s++) {
                    const vA = sA[idx] * ccA[s] - cA[idx] * scA[s];
                    const vB = cB[idx] * ccB[s] + sB[idx] * scB[s];
                    const vC = sC[idx] * ccC[s] - cC[idx] * scC[s];
                    const l = 128.0 + 38.0 * vA + 34.0 * vB + 22.0 * vC;
                    acc += l < 8.0 ? 8.0 : (l > 247.0 ? 247.0 : l);
                  }
                  const v = Math.round(acc * invT);
                  const o = idx * 4;
                  rgba[o] = v; rgba[o + 1] = v; rgba[o + 2] = v; rgba[o + 3] = 255;
                }
              }
            };

            const pumpRsSession = async ({ epoch, totalFrames, t0Us, fps, dtRowUs, exposureUs, audioRateHz, fTone, taps }) => {
              w2p.postWorkerMsg({
                type: 'set_rs_calib',
                rsMode: 2,
                dtRowUs,
                exposureUs,
                audioRateHz,
                maxSkewUs: 500.0,
              });
              let cursor = 0;
              for (let f = 0; f < totalFrames; f++) {
                const tSrcUs = t0Us + Math.round((f * 1e6) / fps);
                const endSample = Math.round((((f + 1) / fps) * audioRateHz));
                const nAudio = endSample - cursor;
                const pcm = new Array(nAudio);
                for (let i = 0; i < nAudio; i++) {
                  const tS = t0Us * 1e-6 + (cursor + i) / audioRateHz;
                  pcm[i] = 0.5 * Math.cos(2.0 * Math.PI * fTone * tS + 0.1);
                }
                const tEndUs = t0Us + Math.round((endSample * 1e6) / audioRateHz);
                const aAck = await w2p.submitAudioBlock({ samples: pcm, tEndUs, sampleRateHz: audioRateHz, estSkewUs: 0.0 });
                if (!aAck || aAck.accepted !== 1) {
                  throw new Error('Step 10b audio push rejected at frame ' + f + ': ' + JSON.stringify(aAck));
                }
                cursor = endSample;

                renderRsFrame(tSrcUs, dtRowUs, exposureUs, fTone, 0.85, taps);
                const ab = rgba.buffer.slice(0);
                const fAck = await w1p.submitTestFrame({
                  type: 'video_rgba',
                  epoch,
                  seq: f + 1,
                  width: W,
                  height: H,
                  buffer: ab,
                  dtSec: 1 / fps,
                  mediaTimeSec: (f + 1) / fps,
                  tSrcUs,
                }, [ab]);
                if (!fAck || fAck.type !== 'frame_ack') {
                  throw new Error('Step 10b frame rejected at f=' + f + ': ' + JSON.stringify(fAck));
                }
              }
            };

            await pumpRsSession({
              epoch: 910,
              totalFrames: 30,
              t0Us: 1_000_000,
              fps: 30,
              dtRowUs: 25.0,
              exposureUs: 1000.0,
              audioRateHz: 45056.0,
              fTone: 880.0,
              taps: 11,
            });
            await new Promise((r) => setTimeout(r, 60));
            const snapReady = w2p.getLastWave2();
            const diagReady = document.getElementById('diagRsWelch')?.textContent || '';

            await pumpRsSession({
              epoch: 911,
              totalFrames: 28,
              t0Us: 3_000_000,
              fps: 30,
              dtRowUs: 25.0,
              exposureUs: 1e6 / 880.0,
              audioRateHz: 48000.0,
              fTone: 880.0,
              taps: 11,
            });
            await new Promise((r) => setTimeout(r, 60));
            const snapNull = w2p.getLastWave2();
            const banner = document.getElementById('acquisitionStatusBanner');
            const nullSev = banner?.dataset?.severity || '';
            const nullTxt = banner?.textContent || '';

            const chunkPcm = new Array(6000);
            for (let i = 0; i < 6000; i++) {
              chunkPcm[i] = 0.25 * Math.cos(2.0 * Math.PI * 440.0 * (i / 48000.0));
            }
            const chunkAck = await w2p.submitAudioBlock({
              samples: chunkPcm,
              tEndUs: 5_000_000,
              sampleRateHz: 48000.0,
              estSkewUs: 0.0,
            });
            const nanSkewAck = await w2p.submitAudioBlock({
              samples: [0.0, 0.0, 0.0, 0.0],
              tEndUs: 5_050_000,
              sampleRateHz: 48000.0,
              estSkewUs: NaN,
            });
            const f32InfSampleAck = await w2p.submitAudioBlock({
              samples: [1e40],
              tEndUs: 5_100_000,
              sampleRateHz: 48000.0,
              estSkewUs: 0.0,
            });
            const u64WrapTimeAck = await w2p.submitAudioBlock({
              samples: [0.1, 0.2, 0.3, 0.4],
              tEndUs: 2 ** 64,
              sampleRateHz: 48000.0,
              estSkewUs: 0.0,
            });
            await new Promise((r) => setTimeout(r, 30));
            const postNanVibro = w2p.getLastWave2();

            // Delayed-ring fault isolation at lookaheadK=4: rejected 44100 Hz block must show DISTURBED (4), NOT OFF (0)
            w1p.postWorkerMsg({ type: 'params', lookaheadK: 4 });
            await pumpRsSession({
              epoch: 912,
              totalFrames: 30,
              t0Us: 7_000_000,
              fps: 30,
              dtRowUs: 25.0,
              exposureUs: 1000.0,
              audioRateHz: 48000.0,
              fTone: 843.75,
              taps: 11,
            });
            await new Promise((r) => setTimeout(r, 60));
            const preFaultVibro = w2p.getLastWave2();
            await w2p.submitAudioBlock({
              samples: new Array(1600).fill(0.1),
              tEndUs: 8_050_000,
              sampleRateHz: 44100.0,
              estSkewUs: 0.0,
            });
            await new Promise((r) => setTimeout(r, 40));
            const delayedFaultVibro = w2p.getLastWave2();

            // Delayed-ring dropped-block underrun + paused Hodge STANDBY transitions (lookaheadK=4)
            w1p.postWorkerMsg({ type: 'params', lookaheadK: 4 });
            await pumpRsSession({
              epoch: 913,
              totalFrames: 30,
              t0Us: 10_000_000,
              fps: 30,
              dtRowUs: 25.0,
              exposureUs: 1000.0,
              audioRateHz: 48000.0,
              fTone: 843.75,
              taps: 11,
            });
            await new Promise((r) => setTimeout(r, 110));
            const preDropVibro = w2p.getLastWave2();

            return {
              readyVibro: snapReady?.vibro || null,
              diagReady,
              nullVibro: snapNull?.vibro || null,
              nullSev,
              nullTxt,
              chunkAck,
              nanSkewAck,
              f32InfSampleAck,
              u64WrapTimeAck,
              postNanState: postNanVibro?.vibro?.state ?? -1,
              postNanStampValid: Boolean(postNanVibro?.stamps?.vibro?.valid),
              preFaultState: preFaultVibro?.vibro?.state ?? -1,
              delayedFaultState: delayedFaultVibro?.vibro?.state ?? -1,
              delayedFaultReason: delayedFaultVibro?.vibro?.reasonMask ?? 0,
              delayedFaultStampValid: Boolean(delayedFaultVibro?.stamps?.vibro?.valid),
              preDropState: preDropVibro?.vibro?.state ?? -1,
            };
          })()`
        );

        const hudBeforeDropShot = await cdp.send(
          'Page.captureScreenshot',
          { format: 'png', captureBeyondViewport: false, clip: { ...w2HudRect, scale: 1 } },
          w2Session
        );
        const dropGapAndHodge = await cdp.evalJs(
          w2Session,
          `(async () => {
            const w1p = window.__chronosWave1Probe;
            const w2p = window.__chronosWave2Probe;
            const dropGapAck = await w2p.submitAudioBlock({
              samples: new Array(1600).fill(0.15),
              tEndUs: 11_066_667, // skip 33,333 us (1600 samples) after 11_000_000 -> accepted new interval WARMING|AUDIO_UNDERRUN
              sampleRateHz: 48000.0,
              estSkewUs: 0.0,
            });
            await new Promise((r) => setTimeout(r, 150));
            const dropGapSnap = w2p.getLastWave2();
            return {
              dropGapAccepted: dropGapAck?.accepted ?? -1,
              dropGapState: dropGapSnap?.vibro?.state ?? -1,
              dropGapReason: dropGapSnap?.vibro?.reasonMask ?? 0,
              dropGapStampValid: Boolean(dropGapSnap?.stamps?.vibro?.valid),
            };
          })()`
        );
        const hudAfterDropShot = await cdp.send(
          'Page.captureScreenshot',
          { format: 'png', captureBeyondViewport: false, clip: { ...w2HudRect, scale: 1 } },
          w2Session
        );
        const hudBeforeDropBuf = Buffer.from(hudBeforeDropShot?.data || '', 'base64');
        const hudAfterDropBuf = Buffer.from(hudAfterDropShot?.data || '', 'base64');
        if (hudBeforeDropBuf.length < 1000 || hudBeforeDropBuf.equals(hudAfterDropBuf)) {
          throw new Error('Step 10b dropped PCM block while running=true did not redraw paused #hudCanvas from READY to WARMING');
        }
        const pausedHodgeEnableReport = await cdp.evalJs(
          w2Session,
          `(async () => {
            const w1p = window.__chronosWave1Probe;
            const w2p = window.__chronosWave2Probe;
            w1p.postWorkerMsg({ type: 'set_hodge_config', hodgeMode: 0 });
            await new Promise((r) => setTimeout(r, 30));
            w1p.postWorkerMsg({ type: 'set_hodge_config', hodgeMode: 1 });
            await new Promise((r) => setTimeout(r, 50));
            const pausedHodgeEnableSnap = w2p.getLastWave2();
            w1p.postWorkerMsg({ type: 'set_hodge_config', hodgeMode: 0 });
            w1p.postWorkerMsg({ type: 'params', lookaheadK: 0 });
            return {
              pausedHodgeReqMode: pausedHodgeEnableSnap?.hodge?.reqMode ?? -1,
              pausedHodgeState: pausedHodgeEnableSnap?.hodge?.state ?? -1,
            };
          })()`
        );
        Object.assign(rsWelchReport, dropGapAndHodge, pausedHodgeEnableReport);

        if (
          !rsWelchReport.chunkAck ||
          rsWelchReport.chunkAck.accepted !== 1 ||
          rsWelchReport.chunkAck.count !== 6000
        ) {
          throw new Error('Step 10b 6000-sample chunking gate failed: ' + JSON.stringify(rsWelchReport.chunkAck));
        }
        if (
          !rsWelchReport.nanSkewAck ||
          rsWelchReport.nanSkewAck.accepted !== 0 ||
          rsWelchReport.postNanState !== 4 ||
          rsWelchReport.postNanStampValid ||
          !rsWelchReport.f32InfSampleAck ||
          rsWelchReport.f32InfSampleAck.accepted !== 0 ||
          rsWelchReport.f32InfSampleAck.count !== 0 ||
          !rsWelchReport.u64WrapTimeAck ||
          rsWelchReport.u64WrapTimeAck.accepted !== 0 ||
          rsWelchReport.u64WrapTimeAck.count !== 0 ||
          rsWelchReport.preFaultState !== 3 ||
          rsWelchReport.delayedFaultState !== 4 ||
          (rsWelchReport.delayedFaultReason & 1) === 0 ||
          rsWelchReport.delayedFaultStampValid ||
          rsWelchReport.preDropState !== 3 ||
          rsWelchReport.dropGapAccepted !== 1 ||
          rsWelchReport.dropGapState !== 2 ||
          (rsWelchReport.dropGapReason & 0x100) === 0 ||
          rsWelchReport.dropGapStampValid ||
          rsWelchReport.pausedHodgeReqMode !== 1 ||
          rsWelchReport.pausedHodgeState !== 1
        ) {
          throw new Error('Step 10b audio boundary, K=4 fault/drop propagation & paused Hodge STANDBY gate failed: ' + JSON.stringify(rsWelchReport));
        }

        const rv = rsWelchReport.readyVibro;
        if (
          !rv ||
          rv.state !== 3 ||
          rv.welchK < 8 ||
          Math.abs(rv.domFreqHz - 880.0) >= 1.0 ||
          rv.maxCohGamma2 <= 0.90 ||
          rv.coupledCellsCount < 1 ||
          !rsWelchReport.diagReady.startsWith('READY · ')
        ) {
          throw new Error('Step 10b RS+Welch READY gate failed: ' + JSON.stringify({ rv, diagReady: rsWelchReport.diagReady }));
        }
        const nv = rsWelchReport.nullVibro;
        if (
          !nv ||
          nv.exposureNullFlag !== true ||
          Math.abs(nv.sincH) >= 0.05 ||
          rsWelchReport.nullSev !== 'warn' ||
          !rsWelchReport.nullTxt.includes('EXPOSURE_NULL')
        ) {
          throw new Error('Step 10b Exposure-Null banner gate failed: ' + JSON.stringify({ nv, nullSev: rsWelchReport.nullSev, nullTxt: rsWelchReport.nullTxt }));
        }

        // 10c. Occluded Equal-Mass AMBIGUOUS Conservation + UI Check
        const consReport = await cdp.evalJs(
          w2Session,
          `(async () => {
            const ack = await window.__chronosWave2Probe.evalOccludedPair([
              100, 90, 150, 0, 1, 0,
              160, 90, -150, 0, 1, 0,
              100, 90, -150, 0,
              160, 90, 150, 0,
              0.40, 0, 0.02, 1, 1, 1, 3,
              11, 12, 42, 1, 0,
            ]);
            await new Promise((r) => setTimeout(r, 50));
            return {
              ack,
              lastCons: window.__chronosWave2Probe?.getLastWave2()?.conservation || null,
              diagCons: document.getElementById('diagOcclConservation')?.textContent || '',
            };
          })()`
        );
        const oc = consReport.lastCons || consReport.ack?.conservation;
        if (
          !oc ||
          oc.status !== 2 ||
          (oc.ambiguityMask & 6) !== 6 ||
          !consReport.diagCons.includes('AMBIGUOUS')
        ) {
          throw new Error('Step 10c Occluded conservation AMBIGUOUS gate failed: ' + JSON.stringify(consReport));
        }

        // 10d. Zero-CLS Check (leave observer active through Step 10e)
        await new Promise((r) => setTimeout(r, 80));
        const w2DeckCls = await cdp.evalJs(
          w2Session,
          `(() => window.__e2eW2DeckCls || 0)()`
        );
        if (w2DeckCls >= 0.000001) {
          throw new Error(`Step 10d: Expected 0.000000 CLS across Wave-2 Deck flows, got ${w2DeckCls}`);
        }

        // 10e. Webcam Live Audio Anchoring (Native Fake Webcam + Mock Lifecycle Edge Cases)
        await verifyNativeWebcamAudioGate(cdp, w2Session);

        const webcamAudioCheck = await cdp.evalJs(
          w2Session,
          `(async () => {
            const origMediaDevicesDesc = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices');
            const origAudioCtx = window.AudioContext;
            const origWebkitAudioCtx = window.webkitAudioContext;
            const origPostMessage = Worker.prototype.postMessage;
            const origPerfNow = performance.now.bind(performance);
            const canvas = document.createElement('canvas');
            canvas.width = 320;
            canvas.height = 180;
            const cctx = canvas.getContext('2d');
            cctx.fillStyle = '#204060';
            cctx.fillRect(0, 0, 320, 180);
            const realAc = new origAudioCtx();
            const d1 = realAc.createMediaStreamDestination();
            const d2 = realAc.createMediaStreamDestination();
            const d3 = realAc.createMediaStreamDestination();
            const dComb = realAc.createMediaStreamDestination();
            const dStaleA = realAc.createMediaStreamDestination();
            const dStaleB1 = realAc.createMediaStreamDestination();
            const dStaleB2 = realAc.createMediaStreamDestination();
            const dDriftNeg = realAc.createMediaStreamDestination();
            const trackA = d1.stream.getAudioTracks()[0];
            const trackB = d2.stream.getAudioTracks()[0];
            const trackC = d3.stream.getAudioTracks()[0];
            const trackComb = dComb.stream.getAudioTracks()[0];
            const trackStaleA = dStaleA.stream.getAudioTracks()[0];
            const trackStaleB1 = dStaleB1.stream.getAudioTracks()[0];
            const trackStaleB2 = dStaleB2.stream.getAudioTracks()[0];
            const trackDriftNeg = dDriftNeg.stream.getAudioTracks()[0];
            const driftExtraTracks = [];
            let nextDynamicMicTrack = null;
            let nextMockSampleRate = 48000;
            const stream = canvas.captureStream(30);
            const micOnlyStream = new MediaStream([trackA]);
            const combStream = canvas.captureStream(30);
            combStream.addTrack(trackComb);
            const streamStaleA = canvas.captureStream(30);
            const streamStaleB = canvas.captureStream(30);
            let pendingResolveA = null;
            let pendingResolveB1 = null;
            let gumMode = 'normal';
            const gumCalls = [];
            try {
              const ctxSpies = [];
              let firePrePumpOnNextAssign = false;
              let prePumpCount = -1;
              let msgsBeforeComb = 0;
              class MockAudioContext {
                constructor() {
                  this.sampleRate = nextMockSampleRate;
                  this.state = 'running';
                  this.closed = false;
                  this.boundTrackIds = [];
                  this.hwClockFn = null;
                  this.currentTimeFn = null;
                  Object.defineProperty(this, 'currentTime', {
                    configurable: true,
                    get: () => (typeof this.currentTimeFn === 'function' ? this.currentTimeFn() : undefined),
                  });
                  let onProc = null;
                  this.proc = {
                    get onaudioprocess() {
                      return onProc;
                    },
                    set onaudioprocess(fn) {
                      onProc = fn;
                      if (typeof fn === 'function' && firePrePumpOnNextAssign) {
                        firePrePumpOnNextAssign = false;
                        const arr = new Float32Array(2048);
                        fn({ inputBuffer: { numberOfChannels: 1, getChannelData: () => arr } });
                        prePumpCount = pushedAudioMsgs.length - msgsBeforeComb;
                      }
                    },
                    connect: () => {},
                    disconnect: () => {},
                  };
                  ctxSpies.push(this);
                }
                createMediaStreamSource(srcStream) {
                  this.boundTrackIds = srcStream && typeof srcStream.getAudioTracks === 'function'
                    ? srcStream.getAudioTracks().map((tr) => tr.id)
                    : [];
                  return { connect: () => {}, disconnect: () => {} };
                }
                createScriptProcessor() {
                  return this.proc;
                }
                createGain() {
                  return { gain: { value: 0 }, connect: () => {}, disconnect: () => {} };
                }
                getOutputTimestamp() {
                  if (typeof this.hwClockFn === 'function') {
                    return this.hwClockFn();
                  }
                  return { contextTime: 0, performanceTime: 0 };
                }
                resume() {}
                close() {
                  this.closed = true;
                }
              }
              window.AudioContext = MockAudioContext;
              window.webkitAudioContext = MockAudioContext;

              const pushedAudioMsgs = [];
              const driftAckWorkers = new WeakSet();
              let pendingDriftNextWantAckReqId = '';
              let pendingDriftAckReqId = '';
              let pendingDriftSawAck = false;
              let pendingDriftDrainResolve = null;
              let driftLastReqIdSeen = '';
              const resolvePendingDriftDrain = () => {
                if (pendingDriftDrainResolve) {
                  const fn = pendingDriftDrainResolve;
                  pendingDriftDrainResolve = null;
                  fn(driftLastReqIdSeen);
                }
              };
              let virtOffsetMs = 0.0;
              let frozenNowMs = null;
              performance.now = () => (frozenNowMs !== null ? frozenNowMs : origPerfNow()) + virtOffsetMs;
              Worker.prototype.postMessage = function (data) {
                if (data && data.type === 'push_audio_pcm') {
                  if (!driftAckWorkers.has(this)) {
                    driftAckWorkers.add(this);
                    this.addEventListener('message', (ev) => {
                      const d = ev && ev.data;
                      if (!d || !pendingDriftAckReqId) return;
                      if (d.type === 'wave2_audio_ack' && d.reqId === pendingDriftAckReqId) {
                        pendingDriftSawAck = true;
                        driftLastReqIdSeen = d.reqId;
                      } else if (pendingDriftSawAck && d.type === 'telemetry') {
                        pendingDriftAckReqId = '';
                        pendingDriftSawAck = false;
                        resolvePendingDriftDrain();
                      }
                    });
                  }
                  if (pendingDriftNextWantAckReqId) {
                    data.wantAck = true;
                    data.reqId = pendingDriftNextWantAckReqId;
                    pendingDriftAckReqId = pendingDriftNextWantAckReqId;
                    pendingDriftSawAck = false;
                    pendingDriftNextWantAckReqId = '';
                  }
                  pushedAudioMsgs.push({
                    tEndUs: data.tEndUs,
                    estSkewUs: data.estSkewUs,
                    len: data.samples ? data.samples.length : 0,
                  });
                }
                return origPostMessage.apply(this, arguments);
              };
              Object.defineProperty(navigator, 'mediaDevices', {
                configurable: true,
                value: {
                  getUserMedia: async (constraints) => {
                    gumCalls.push({
                      video: Boolean(constraints && constraints.video),
                      audio: Boolean(constraints && constraints.audio),
                    });
                    if (gumMode === 'combined') {
                      return combStream;
                    }
                    if (gumMode === 'stale_a_vid') return streamStaleA;
                    if (gumMode === 'stale_a_mic') {
                      return new Promise((res) => { pendingResolveA = res; });
                    }
                    if (gumMode === 'stale_b_vid') return streamStaleB;
                    if (gumMode === 'stale_b1_mic') {
                      return new Promise((res) => { pendingResolveB1 = res; });
                    }
                    if (gumMode === 'stale_b2_mic') {
                      return new MediaStream([trackStaleB2]);
                    }
                    if (gumMode === 'drift_neg_mic') {
                      return new MediaStream([trackDriftNeg]);
                    }
                    if (gumMode === 'dynamic_mic' && nextDynamicMicTrack) {
                      return new MediaStream([nextDynamicMicTrack]);
                    }
                    return constraints && constraints.audio && !constraints.video ? micOnlyStream : stream;
                  },
                },
              });

              const rmSel = document.getElementById('rsVibroModeSelect');
              if (rmSel) {
                rmSel.value = '0';
                rmSel.dispatchEvent(new Event('change', { bubbles: true }));
              }
              document.getElementById('btnSrcWebcam')?.click();
              for (let i = 0; i < 40 && gumCalls.length === 0; i++) {
                await new Promise((r) => setTimeout(r, 25));
              }
              await new Promise((r) => setTimeout(r, 60));
              const initialGumCount = gumCalls.length;
              const initialGumAudio = Boolean(gumCalls[0] && gumCalls[0].audio);
              const ctxBeforeWelch = ctxSpies.length;

              if (rmSel) {
                rmSel.value = '2';
                rmSel.dispatchEvent(new Event('change', { bubbles: true }));
              }
              for (let i = 0; i < 40 && ctxSpies.length === 0; i++) {
                await new Promise((r) => setTimeout(r, 25));
              }
              const lateGumCount = gumCalls.length;
              const lateGumAudioOnly = Boolean(gumCalls[1] && gumCalls[1].audio && !gumCalls[1].video);
              const c1 = ctxSpies[0];
              const fireAudioCb = (ctxObj, playbackTimeSec = undefined) => {
                if (ctxObj && ctxObj.proc && typeof ctxObj.proc.onaudioprocess === 'function') {
                  const arr = new Float32Array(2048);
                  const ev = {
                    inputBuffer: { numberOfChannels: 1, getChannelData: () => arr },
                  };
                  if (typeof playbackTimeSec === 'number') {
                    ev.playbackTime = playbackTimeSec;
                  }
                  ctxObj.proc.onaudioprocess(ev);
                }
              };

              // Pump frames until rVFC initializes pump ref; dispatch first post-pump callback 5.0 ms late
              for (let f = 0; f < 30 && pushedAudioMsgs.length === 0; f++) {
                cctx.fillStyle = 'rgb(' + (30 + ((f * 7) % 200)) + ',' + ((60 + f * 5) % 220) + ',160)';
                cctx.fillRect(20 + (f % 10) * 6, 20, 48, 48);
                await new Promise((r) => setTimeout(r, 40));
                if (f >= 2) {
                  frozenNowMs = origPerfNow();
                  virtOffsetMs = 5.0;
                  fireAudioCb(c1);
                  if (pushedAudioMsgs.length === 0) {
                    virtOffsetMs = 0.0;
                    frozenNowMs = null;
                  }
                }
              }
              if (pushedAudioMsgs.length === 1) {
                virtOffsetMs = 2048.0 / 48.0;
                fireAudioCb(c1);
                virtOffsetMs = (2.0 * 2048.0) / 48.0;
                fireAudioCb(c1);
                virtOffsetMs = (3.0 * 2048.0) / 48.0;
                fireAudioCb(c1);
              }
              frozenNowMs = null;
              await new Promise((r) => setTimeout(r, 25));
              const postAudioReason = window.__chronosWave2Probe?.getLastWave2()?.vibro?.reasonMask ?? 0;

              // Empty-gap track swap: remove trackA -> callback closes c1 -> addTrack(trackB) restarts capture with zero extra callbacks
              stream.removeTrack(trackA);
              fireAudioCb(c1);
              const ctxCountDuringGap = ctxSpies.length;
              const c1ClosedDuringGap = Boolean(c1 && c1.closed);
              stream.addTrack(trackB);
              await new Promise((r) => setTimeout(r, 25));
              const c2 = ctxSpies[1] || null;
              const ctxCountAfterGapAdd = ctxSpies.length;

              // Ended track kept in stream alongside live trackC: createMediaStreamSource receives ONLY [trackC.id]
              trackB.stop();
              stream.addTrack(trackC);
              await new Promise((r) => setTimeout(r, 25));
              const cLast = ctxSpies[ctxSpies.length - 1] || null;

              const gumCountAfterNormal = gumCalls.length;
              const ctxCountBeforeExtra = ctxSpies.length;

              // Combined webcam+mic start -> trigger synchronous pre-pump callback -> selecting RS Off stops/detaches audio track
              gumMode = 'combined';
              if (rmSel) { rmSel.value = '2'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
              msgsBeforeComb = pushedAudioMsgs.length;
              firePrePumpOnNextAssign = true;
              document.getElementById('btnSrcWebcam')?.click();
              for (let i = 0; i < 40 && ctxSpies.length === ctxCountBeforeExtra; i++) {
                await new Promise((r) => setTimeout(r, 25));
              }
              if (rmSel) { rmSel.value = '0'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
              await new Promise((r) => setTimeout(r, 40));
              const combAudioEndedAfterOff = trackComb.readyState === 'ended';
              const combTracksLeftAfterOff = combStream.getAudioTracks().length;

              // Stale pending mic A resolving while request B1 is pending + Off->Welch B2 stops A & B1 and attaches only B2
              gumMode = 'stale_a_vid';
              document.getElementById('btnSrcWebcam')?.click();
              await new Promise((r) => setTimeout(r, 60));
              gumMode = 'stale_a_mic';
              if (rmSel) { rmSel.value = '2'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
              for (let i = 0; i < 20 && !pendingResolveA; i++) {
                await new Promise((r) => setTimeout(r, 15));
              }
              document.getElementById('btnSrcTestbench')?.click();
              await new Promise((r) => setTimeout(r, 25));
              if (rmSel) { rmSel.value = '0'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
              gumMode = 'stale_b_vid';
              document.getElementById('btnSrcWebcam')?.click();
              await new Promise((r) => setTimeout(r, 60));
              gumMode = 'stale_b1_mic';
              if (rmSel) { rmSel.value = '2'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
              for (let i = 0; i < 20 && !pendingResolveB1; i++) {
                await new Promise((r) => setTimeout(r, 15));
              }
              if (pendingResolveA) pendingResolveA(new MediaStream([trackStaleA]));
              await new Promise((r) => setTimeout(r, 20));
              if (rmSel) { rmSel.value = '0'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
              gumMode = 'stale_b2_mic';
              const ctxBeforeB2 = ctxSpies.length;
              if (rmSel) { rmSel.value = '2'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
              for (let i = 0; i < 40 && ctxSpies.length === ctxBeforeB2; i++) {
                await new Promise((r) => setTimeout(r, 25));
              }
              if (pendingResolveB1) pendingResolveB1(new MediaStream([trackStaleB1]));
              await new Promise((r) => setTimeout(r, 40));
              const cStaleFinal = ctxSpies[ctxSpies.length - 1] || null;
              const staleATracksEnded = trackStaleA.readyState === 'ended';
              const staleB1TracksEnded = trackStaleB1.readyState === 'ended';
              const staleBTracksIds = streamStaleB.getAudioTracks().map((tr) => tr.id);

              const pushedBeforeDriftCount = pushedAudioMsgs.length;
              const sdBtn = document.getElementById('btnResSd');
              if (sdBtn) sdBtn.click();
              const scanSel = document.getElementById('selRsScanDir');
              if (scanSel && scanSel.value !== '0') {
                scanSel.value = '0';
                scanSel.dispatchEvent(new Event('change', { bubbles: true }));
              }
              await new Promise((r) => setTimeout(r, 40));

              // Sustained clock drift & dispatch delay checks (Round 14 & Round 15 P2 regression suite)
              let driftRunSeq = 0;
              const runSustainedDrift = async (ctxObj, totalBlocks, offsetMsFn, hwSetupFn = null, curTimeSetupFn = null, omitPlaybackTime = false, videoDriftRatePerBlock = 0.0) => {
                driftRunSeq++;
                const msgStart = pushedAudioMsgs.length;
                cctx.fillStyle = 'rgb(' + (40 + driftRunSeq * 15) + ',' + (70 + driftRunSeq * 10) + ',110)';
                cctx.fillRect(16, 16, 48, 48);
                await new Promise((r) => setTimeout(r, 40));
                const samplesBefore = window.__chronosWave2Probe?.getLastWave2()?.vibro?.audioTotalSamples ?? 0;
                const sr = ctxObj.sampleRate || 48000;
                const blockDurMs = (2048.0 * 1000.0) / sr;
                let curBlockIdx = 0;
                ctxObj.hwClockFn = typeof hwSetupFn === 'function'
                  ? () => hwSetupFn(curBlockIdx, frozenNowMs !== null ? frozenNowMs : origPerfNow(), blockDurMs)
                  : null;
                ctxObj.currentTimeFn = typeof curTimeSetupFn === 'function'
                  ? () => curTimeSetupFn(curBlockIdx, blockDurMs)
                  : null;
                const hasPlayTime = !omitPlaybackTime && (typeof hwSetupFn === 'function' || typeof curTimeSetupFn === 'function');
                const playTimeOf = (idx) => (hasPlayTime ? 1.0 + ((idx + 1) * 2048.0) / sr : undefined);
                for (let f = 0; f < 30 && pushedAudioMsgs.length === msgStart; f++) {
                  cctx.fillStyle = 'rgb(' + (40 + ((f * 11) % 190)) + ',' + ((80 + f * 7) % 210) + ',180)';
                  cctx.fillRect(24 + (f % 10) * 6, 24, 48, 48);
                  await new Promise((r) => setTimeout(r, 45));
                  if (f >= 2) {
                    if (videoDriftRatePerBlock !== 0.0) {
                      const img = cctx.createImageData(canvas.width, canvas.height);
                      for (let yy = 0; yy < canvas.height; yy++) {
                        for (let xx = 0; xx < canvas.width; xx++) {
                          const off = (yy * canvas.width + xx) * 4;
                          const val = 128 + 38 * Math.sin(0.27 * xx + 0.13 * yy) + 34 * Math.cos(0.19 * xx - 0.31 * yy) + 22 * Math.cos(0.37 * xx + 0.29 * yy);
                          img.data[off] = img.data[off + 1] = img.data[off + 2] = val;
                          img.data[off + 3] = 255;
                        }
                      }
                      cctx.putImageData(img, 0, 0);
                      await new Promise((r) => setTimeout(r, 55));
                    }
                    const probe1 = window.__chronosWave1Probe;
                    const busyDeadline = origPerfNow() + 500;
                    while (origPerfNow() < busyDeadline) {
                      const bNow = probe1?.getBusyState?.();
                      const vNow = window.__chronosWave2Probe?.getLastWave2()?.vibro?.state;
                      if ((!bNow || (!bNow.workerFrameBusy && !bNow.bitmapCaptureBusy)) && vNow === 2) break;
                      await new Promise((r) => setTimeout(r, 15));
                    }
                    const readyStateNow = window.__chronosWave2Probe?.getLastWave2()?.vibro?.state;
                    if (readyStateNow !== 2 && f < 29) continue;
                    curBlockIdx = 0;
                    frozenNowMs = origPerfNow();
                    virtOffsetMs = offsetMsFn(0, blockDurMs);
                    if (totalBlocks === 1) {
                      pendingDriftNextWantAckReqId = 'e2e-drift-' + driftRunSeq;
                    }
                    fireAudioCb(ctxObj, playTimeOf(0));
                    if (pushedAudioMsgs.length === msgStart) {
                      pendingDriftNextWantAckReqId = '';
                      frozenNowMs = null;
                    }
                  }
                }
                let drainPromise = null;
                if (totalBlocks === 1 && pendingDriftAckReqId) {
                  drainPromise = new Promise((resolve) => {
                    const timer = setTimeout(() => {
                      if (pendingDriftDrainResolve) {
                        pendingDriftDrainResolve = null;
                        resolve('timeout');
                      }
                    }, 20000);
                    pendingDriftDrainResolve = (reqId) => {
                      clearTimeout(timer);
                      resolve(reqId);
                    };
                  });
                } else if (pushedAudioMsgs.length === msgStart + 1) {
                  let nextFrame = 1;
                  const framePeriodMs = 1000.0 / 30.0;
                  const vsyncMs = 1000.0 / 60.0;
                  for (let b = 1; b < totalBlocks; b++) {
                    curBlockIdx = b;
                    if (videoDriftRatePerBlock !== 0.0) {
                      while (nextFrame * framePeriodMs + 2.5 * blockDurMs <= b * blockDurMs) {
                        const captureRel = nextFrame * framePeriodMs;
                        const dispatchRel = Math.ceil((captureRel + 0.2) / vsyncMs) * vsyncMs;
                        const videoDriftMs = Math.max(0.0, captureRel / blockDurMs - 320.0) * videoDriftRatePerBlock;
                        const baseEndUs = pushedAudioMsgs[msgStart].tEndUs;
                        const mt = (baseEndUs + (captureRel + videoDriftMs) * 1000.0) / 1e6;
                        const probe = window.__chronosWave1Probe;
                        for (let wi = 0; probe.getBusyState().workerFrameBusy && wi < 200; wi++) {
                          await new Promise((r) => setTimeout(r, 1));
                        }
                        await probe.forwardVideoFrameForTest(mt, frozenNowMs + dispatchRel, {
                          captureTime: frozenNowMs + captureRel,
                        });
                        nextFrame++;
                      }
                    }
                    virtOffsetMs = offsetMsFn(b, blockDurMs);
                    if (b === totalBlocks - 1) {
                      pendingDriftNextWantAckReqId = 'e2e-drift-' + driftRunSeq;
                    }
                    fireAudioCb(ctxObj, playTimeOf(b));
                  }
                  if (pendingDriftAckReqId) {
                    drainPromise = new Promise((resolve) => {
                      const timer = setTimeout(() => {
                        if (pendingDriftDrainResolve) {
                          pendingDriftDrainResolve = null;
                          resolve('timeout');
                        }
                      }, 20000);
                      pendingDriftDrainResolve = (reqId) => {
                        clearTimeout(timer);
                        resolve(reqId);
                      };
                    });
                  }
                }
                pendingDriftNextWantAckReqId = '';
                ctxObj.hwClockFn = null;
                ctxObj.currentTimeFn = null;
                frozenNowMs = null;
                virtOffsetMs = 0.0;
                if (drainPromise) {
                  await drainPromise;
                } else {
                  await new Promise((r) => setTimeout(r, 120));
                }
                pendingDriftAckReqId = '';
                pendingDriftSawAck = false;
                pendingDriftDrainResolve = null;
                const snap = window.__chronosWave2Probe?.getLastWave2() || null;
                const slice = pushedAudioMsgs.slice(msgStart);
                const samplesAfter = snap?.vibro?.audioTotalSamples ?? 0;
                return {
                  pushedCount: slice.length,
                  lastSkew: slice.length > 0 ? slice[slice.length - 1].estSkewUs : 0,
                  maxAbsSkew: slice.reduce((mx, m) => Math.max(mx, Math.abs(m.estSkewUs)), 0),
                  acceptedBlocks: Math.round(((samplesAfter - samplesBefore) >>> 0) / 2048),
                  state: snap?.vibro?.state ?? -1,
                  reasonMask: snap?.vibro?.reasonMask ?? -1,
                  welchK: snap?.vibro?.welchK ?? -1,
                  stampValid: Boolean(snap?.stamps?.vibro?.valid),
                };
              };

              const restartFreshMicSession = async (sampleRate = 48000) => {
                if (rmSel) { rmSel.value = '0'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
                await new Promise((r) => setTimeout(r, 30));
                const dst = realAc.createMediaStreamDestination();
                const trk = dst.stream.getAudioTracks()[0];
                driftExtraTracks.push(trk);
                nextDynamicMicTrack = trk;
                nextMockSampleRate = sampleRate;
                gumMode = 'dynamic_mic';
                const ctxBefore = ctxSpies.length;
                if (rmSel) { rmSel.value = '2'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
                for (let i = 0; i < 40 && ctxSpies.length === ctxBefore; i++) {
                  await new Promise((r) => setTimeout(r, 25));
                }
                nextMockSampleRate = 48000;
                return ctxSpies[ctxSpies.length - 1] || null;
              };

              const driftPos = await runSustainedDrift(cStaleFinal, 320, (b, bd) => b * (bd + 0.1));
              const cDriftNeg = await restartFreshMicSession();
              const driftNeg = await runSustainedDrift(cDriftNeg, 320, (b, bd) => b * (bd - 0.1));
              const cLatePosP4 = await restartFreshMicSession();
              const lateDriftPosP4 = await runSustainedDrift(
                cLatePosP4,
                3320,
                (b, bd) => b * bd + 2.667 * (Math.floor(b / 2) % 2) + 0.02 * Math.max(0, b - 3000)
              );
              const cLateNegP4 = await restartFreshMicSession();
              const lateDriftNegP4 = await runSustainedDrift(
                cLateNegP4,
                3320,
                (b, bd) => b * bd + 2.667 * (Math.floor(b / 2) % 2) - 0.02 * Math.max(0, b - 3000)
              );
              const cDisp5333 = await restartFreshMicSession();
              const dispatchDelay5333 = await runSustainedDrift(
                cDisp5333,
                410,
                (b, bd) => b * bd + Math.min(5.333, 0.1 * Math.max(0, b - 320))
              );
              const cHwDisp25ms = await restartFreshMicSession();
              const hwDispatchDelay25ms = await runSustainedDrift(
                cHwDisp25ms,
                410,
                (b, bd) => b * bd + Math.min(25.0, 0.3 * Math.max(0, b - 320)),
                (b, basePerf, bd) => ({
                  contextTime: 1.0 + ((b + 1) * 2048.0) / 48000.0 - 0.0426667,
                  performanceTime: basePerf + b * bd - 42.6667,
                })
              );
              const cHwLateP4 = await restartFreshMicSession();
              const hwLateDriftP4 = await runSustainedDrift(
                cHwLateP4,
                3320,
                (b, bd) => b * bd + 2.667 * (Math.floor(b / 2) % 2) + 0.02 * Math.max(0, b - 3000),
                (b, basePerf, bd) => ({
                  contextTime: 1.0 + ((b + 1) * 2048.0) / 48000.0 - 0.0426667,
                  performanceTime: basePerf + b * bd + 0.02 * Math.max(0, b - 3000) - 42.6667,
                })
              );
              const cCtxPullDisp = await restartFreshMicSession(48000);
              const ctxPullDisp5333 = await runSustainedDrift(
                cCtxPullDisp,
                410,
                (b, bd) => b * bd + Math.min(5.333, 0.1 * Math.max(0, b - 320)),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0 + Math.floor((Math.min(5.333e-3, 1e-4 * Math.max(0, b - 320)) + 1e-3) / (256.0 / 48000.0)) * (256.0 / 48000.0)
              );
              const cBare44k1Pos = await restartFreshMicSession(44100);
              const bare44k1LatePosP4 = await runSustainedDrift(
                cBare44k1Pos,
                3320,
                (b, bd) => b * bd + 2.667 * (Math.floor(b / 2) % 2) + 0.02 * Math.max(0, b - 3000)
              );
              const cBare44k1Neg = await restartFreshMicSession(44100);
              const bare44k1LateNegP4 = await runSustainedDrift(
                cBare44k1Neg,
                3320,
                (b, bd) => b * bd + 2.667 * (Math.floor(b / 2) % 2) - 0.02 * Math.max(0, b - 3000)
              );
              const cCtx512Pull48k = await restartFreshMicSession(48000);
              const ctx512Pull48k8000 = await runSustainedDrift(
                cCtx512Pull48k,
                410,
                (b, bd) => b * bd + Math.min(8.0, 0.1 * Math.max(0, b - 320)),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0 + Math.floor((Math.min(8e-3, 1e-4 * Math.max(0, b - 320)) + 1e-3) / (512.0 / 48000.0)) * (512.0 / 48000.0)
              );
              const cCtx512Pull96k = await restartFreshMicSession(96000);
              const ctx512Pull96k5333 = await runSustainedDrift(
                cCtx512Pull96k,
                410,
                (b, bd) => b * bd + Math.min(5.333, 0.1 * Math.max(0, b - 320)),
                null,
                (b) => 1.0 + (b * 2048.0) / 96000.0 + Math.floor((Math.min(5.333e-3, 1e-4 * Math.max(0, b - 320)) + 1e-3) / (512.0 / 96000.0)) * (512.0 / 96000.0)
              );
              const cRawTsNoPlay = await restartFreshMicSession(48000);
              const rawTsNoPlay750 = await runSustainedDrift(
                cRawTsNoPlay,
                750,
                (b, bd) => b * bd + Math.min(5.333, 0.1 * Math.max(0, b - 320)),
                (b, basePerf, bd) => {
                  const dMs = Math.min(5.333, 0.1 * Math.max(0, b - 320));
                  const pSec = 256.0 / 48000.0;
                  const pulls = Math.floor(((dMs + 1.0) * 1e-3) / pSec);
                  return {
                    contextTime: 1.0 + (b * 2048.0) / 48000.0 + pulls * pSec,
                    performanceTime: basePerf + b * bd - 1.0 + pulls * pSec * 1000.0,
                  };
                },
                null,
                true
              );
              const cBare48kPosP12 = await restartFreshMicSession(48000);
              const bare48kLatePosP12 = await runSustainedDrift(
                cBare48kPosP12,
                3320,
                (b, bd) => b * bd + 2.667 * (Math.floor(b / 6) % 2) + 0.02 * Math.max(0, b - 3000)
              );
              const cBare48kNegP12 = await restartFreshMicSession(48000);
              const bare48kLateNegP12 = await runSustainedDrift(
                cBare48kNegP12,
                3320,
                (b, bd) => b * bd + 2.667 * (Math.floor(b / 6) % 2) - 0.02 * Math.max(0, b - 3000)
              );
              const cCtx48kPosP12 = await restartFreshMicSession(48000);
              const ctx48kLatePosP12 = await runSustainedDrift(
                cCtx48kPosP12,
                3320,
                (b, bd) => b * bd + 2.667 * (Math.floor(b / 6) % 2) + 0.02 * Math.max(0, b - 3000),
                null,
                (b) => {
                  const jitSec = 2.667e-3 * (Math.floor(b / 6) % 2);
                  const pSec = 256.0 / 48000.0;
                  return 1.0 + (b * 2048.0) / 48000.0 + Math.floor((jitSec + 1e-3) / pSec) * pSec;
                }
              );
              const cIntermittentTs = await restartFreshMicSession(48000);
              const intermittentNullTs4320 = await runSustainedDrift(
                cIntermittentTs,
                4320,
                (b, bd) => b * bd + 0.02 * Math.max(0, b - 3000),
                (b, basePerf, bd) => (
                  b % 16 === 0
                    ? null
                    : {
                        contextTime: 1.0 + (b * 2048.0) / 48000.0,
                        performanceTime: basePerf + b * bd + 0.02 * Math.max(0, b - 3000) - bd,
                      }
                ),
                (b) => 1.0 + (b * 2048.0) / 48000.0
              );
              const makeRandJitArr = (seedInit, len, qMs) => {
                const arr = new Float64Array(len);
                let sd = seedInit >>> 0;
                for (let k = 0; k < len; k++) {
                  sd = (Math.imul(sd, 1664525) + 1013904223) >>> 0;
                  arr[k] = (sd >>> 31) * qMs;
                }
                return arr;
              };
              const makeHeldUnifJitArr = (seedInit, len, hold, qMs) => {
                const arr = new Float64Array(len);
                let sd = seedInit >>> 0;
                let cur = 0.0;
                for (let k = 0; k < len; k++) {
                  sd = (Math.imul(sd, 1664525) + 1013904223) >>> 0;
                  if (k % hold === 0) cur = qMs * (sd / 4294967296.0);
                  arr[k] = cur;
                }
                return arr;
              };
              const randJit1Pos = makeRandJitArr(1, 3320, 2.667);
              const randJit7Neg = makeRandJitArr(7, 3320, 2.667);
              const randJit44kBenign = makeRandJitArr(5, 750, 2.9025);
              const holdJit2Pos = makeHeldUnifJitArr(2, 3320, 20, 2.667);
              const holdJit3Neg = makeHeldUnifJitArr(3, 3320, 20, 2.667);
              const holdJit50Benign = makeHeldUnifJitArr(50, 750, 20, 2.667);
              const cBareRand1Pos = await restartFreshMicSession(48000);
              const bareRandSeed1Pos = await runSustainedDrift(
                cBareRand1Pos,
                3320,
                (b, bd) => b * bd + randJit1Pos[b] + 0.02 * Math.max(0, b - 3000)
              );
              const cBareRand7Neg = await restartFreshMicSession(48000);
              const bareRandSeed7Neg = await runSustainedDrift(
                cBareRand7Neg,
                3320,
                (b, bd) => b * bd + randJit7Neg[b] - 0.02 * Math.max(0, b - 3000)
              );
              const cCtxRand1Pos = await restartFreshMicSession(48000);
              const ctxRandSeed1Pos = await runSustainedDrift(
                cCtxRand1Pos,
                3320,
                (b, bd) => b * bd + randJit1Pos[b] + 0.02 * Math.max(0, b - 3000),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0 + Math.floor((randJit1Pos[b] * 1e-3 + 1e-3) / (256.0 / 48000.0)) * (256.0 / 48000.0)
              );
              const cCtxRand7Neg = await restartFreshMicSession(48000);
              const ctxRandSeed7Neg = await runSustainedDrift(
                cCtxRand7Neg,
                3320,
                (b, bd) => b * bd + randJit7Neg[b] - 0.02 * Math.max(0, b - 3000),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0 + Math.floor((randJit7Neg[b] * 1e-3 + 1e-3) / (256.0 / 48000.0)) * (256.0 / 48000.0)
              );
              const cCtxRandBenign = await restartFreshMicSession(44100);
              const ctxRandBenign512Pull = await runSustainedDrift(
                cCtxRandBenign,
                750,
                (b, bd) => b * bd + randJit44kBenign[b] + Math.min(11.589, 0.1 * Math.max(0, b - 320)),
                null,
                (b) => 1.0 + (b * 2048.0) / 44100.0 + Math.floor((randJit44kBenign[b] * 1e-3 + Math.min(11.589e-3, 1e-4 * Math.max(0, b - 320)) + 1e-3) / (512.0 / 44100.0)) * (512.0 / 44100.0)
              );
              const cBareHold20Pos = await restartFreshMicSession(48000);
              const bareHold20Seed2Pos = await runSustainedDrift(
                cBareHold20Pos,
                3320,
                (b, bd) => b * bd + holdJit2Pos[b] + 0.02 * Math.max(0, b - 3000)
              );
              const cBareHold20Neg = await restartFreshMicSession(48000);
              const bareHold20Seed3Neg = await runSustainedDrift(
                cBareHold20Neg,
                3320,
                (b, bd) => b * bd + holdJit3Neg[b] - 0.02 * Math.max(0, b - 3000)
              );
              const cCtxHold20Pos = await restartFreshMicSession(48000);
              const ctxHold20Seed2Pos = await runSustainedDrift(
                cCtxHold20Pos,
                3320,
                (b, bd) => b * bd + holdJit2Pos[b] + 0.02 * Math.max(0, b - 3000),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0
              );
              const cCtxHold20Neg = await restartFreshMicSession(48000);
              const ctxHold20Seed3Neg = await runSustainedDrift(
                cCtxHold20Neg,
                3320,
                (b, bd) => b * bd + holdJit3Neg[b] - 0.02 * Math.max(0, b - 3000),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0
              );
              const cCtxHold20Benign = await restartFreshMicSession(48000);
              const ctxHold20Seed50Benign = await runSustainedDrift(
                cCtxHold20Benign,
                750,
                (b, bd) => b * bd + holdJit50Benign[b] + Math.min(8.0, 0.1 * Math.max(0, b - 320)),
                null,
                (b) => {
                  const p = 512.0 / 48000.0;
                  const d = Math.min(8.0, 0.1 * Math.max(0, b - 320));
                  return 1.0 + (b * 2048.0) / 48000.0 + Math.floor(((holdJit50Benign[b] + d + 1.0) * 1e-3) / p) * p;
                }
              );
              const cBareRand44kBenign = await restartFreshMicSession(44100);
              const bareRand44kSeed5Benign = await runSustainedDrift(
                cBareRand44kBenign,
                750,
                (b, bd) => b * bd + randJit44kBenign[b] + Math.min(11.589, 0.1 * Math.max(0, b - 320))
              );
              const makeHeldBernJitArr = (seedInit, len, hold, qMs) => {
                const arr = new Float64Array(len);
                let sd = seedInit >>> 0;
                let cur = 0.0;
                for (let k = 0; k < len; k++) {
                  sd = (Math.imul(sd, 1664525) + 1013904223) >>> 0;
                  if (k % hold === 0) cur = (sd >>> 31) * qMs;
                  arr[k] = cur;
                }
                return arr;
              };
              const q48Ms = (128.0 * 1000.0) / 48000.0;
              const bernHold23Seed2 = makeHeldBernJitArr(2, 3320, 23, q48Ms);
              const unifHold14Seed2 = makeHeldUnifJitArr(2, 3320, 14, q48Ms);
              const unifHold25Seed2 = makeHeldUnifJitArr(2, 3320, 25, q48Ms);
              const p512Sec48 = 512.0 / 48000.0;
              const cBareHold23Bern = await restartFreshMicSession(48000);
              const bareHold23BernSeed2Neg20 = await runSustainedDrift(
                cBareHold23Bern,
                3320,
                (b, bd) => b * bd + bernHold23Seed2[b] - 0.02 * Math.max(0, b - 3000)
              );
              const cCtxPh96Hold14 = await restartFreshMicSession(48000);
              const ctx512Ph96Hold14Seed2Pos20 = await runSustainedDrift(
                cCtxPh96Hold14,
                3320,
                (b, bd) => b * bd + unifHold14Seed2[b] + 0.02 * Math.max(0, b - 3000),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0 + p512Sec48 * Math.floor((unifHold14Seed2[b] + 9.6) / (1000.0 * p512Sec48))
              );
              const cCtxPh96Hold25 = await restartFreshMicSession(48000);
              const ctx512Ph96Hold25Seed2Pos60 = await runSustainedDrift(
                cCtxPh96Hold25,
                3320,
                (b, bd) => b * bd + unifHold25Seed2[b] + 0.06 * Math.max(0, b - 3000),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0 + p512Sec48 * Math.floor((unifHold25Seed2[b] + 9.6) / (1000.0 * p512Sec48))
              );
              const cBareSlow5ms = await restartFreshMicSession(48000);
              const bareBenignSlow5ms1100 = await runSustainedDrift(
                cBareSlow5ms,
                1100,
                (b, bd) => b * bd + Math.min(5.0, 0.025 * Math.max(0, b - 320))
              );
              const cCtxSlow5ms = await restartFreshMicSession(48000);
              const ctx512BenignSlow5ms1100 = await runSustainedDrift(
                cCtxSlow5ms,
                1100,
                (b, bd) => b * bd + Math.min(5.0, 0.025 * Math.max(0, b - 320)),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0 + p512Sec48 * Math.floor((Math.min(5.0, 0.025 * Math.max(0, b - 320)) + 1.0) / (1000.0 * p512Sec48))
              );
              const cCtx23p5ms = await restartFreshMicSession(48000);
              const ctx512Benign23p5ms750 = await runSustainedDrift(
                cCtx23p5ms,
                750,
                (b, bd) => b * bd + Math.min(23.5, 0.04 * Math.max(0, b - 320)),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0 + p512Sec48 * Math.floor((Math.min(23.5, 0.04 * Math.max(0, b - 320)) + 0.2) / (1000.0 * p512Sec48))
              );
              const cSlowCleanPos = await restartFreshMicSession(48000);
              const slowCleanPos15us3701 = await runSustainedDrift(
                cSlowCleanPos,
                3701,
                (b, bd) => b * bd + 0.015 * Math.max(0, b - 3000),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0
              );
              const p128Sec48 = 128.0 / 48000.0;
              const cSlowCleanNeg = await restartFreshMicSession(48000);
              const slowCleanNeg15us3701 = await runSustainedDrift(
                cSlowCleanNeg,
                3701,
                (b, bd) => b * bd - 0.015 * Math.max(0, b - 3000),
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0 + p128Sec48 * Math.floor(1.0 / (1000.0 * p128Sec48))
              );
              const q441Ms = (128.0 * 1000.0) / 44100.0;
              const p128Sec441 = 128.0 / 44100.0;
              const p512Sec441 = 512.0 / 44100.0;
              const unifHold14Seed2_441 = makeHeldUnifJitArr(2, 1000, 14, q441Ms);
              const cDisp44k128 = await restartFreshMicSession(44100);
              const dispatch44kHeld128Cap5 = await runSustainedDrift(
                cDisp44k128,
                1000,
                (b, bd) => b * bd + unifHold14Seed2_441[b] + Math.min(5.0, 0.02 * Math.max(0, b - 320)),
                null,
                (b) => 1.0 + (b * 2048.0) / 44100.0 + p128Sec441 * Math.floor((unifHold14Seed2_441[b] + Math.min(5.0, 0.02 * Math.max(0, b - 320))) / (1000.0 * p128Sec441))
              );
              const cDisp44k512 = await restartFreshMicSession(44100);
              const dispatch44kHeld512Cap11p589 = await runSustainedDrift(
                cDisp44k512,
                1000,
                (b, bd) => b * bd + unifHold14Seed2_441[b] + Math.min(11.589, 0.02 * Math.max(0, b - 320)),
                null,
                (b) => 1.0 + (b * 2048.0) / 44100.0 + p512Sec441 * Math.floor((unifHold14Seed2_441[b] + Math.min(11.589, 0.02 * Math.max(0, b - 320)) + 10.448979591836734) / (1000.0 * p512Sec441))
              );
              const q96Ms = (128.0 * 1000.0) / 96000.0;
              const p512Sec96 = 512.0 / 96000.0;
              const unifHold20Seed1_96 = makeHeldUnifJitArr(1, 680, 20, q96Ms);
              const cDrift96k = await restartFreshMicSession(96000);
              const drift96kHold20Neg20 = await runSustainedDrift(
                cDrift96k,
                680,
                (b, bd) => b * bd + unifHold20Seed1_96[b] - 0.02 * Math.max(0, b - 360),
                null,
                (b) => 1.0 + (b * 2048.0) / 96000.0 + p512Sec96 * Math.floor((unifHold20Seed1_96[b] + 5.28) / (1000.0 * p512Sec96))
              );
              const cRvfcCaptureDrift = await restartFreshMicSession(48000);
              const rvfcCaptureVideoDrift650 = await runSustainedDrift(
                cRvfcCaptureDrift,
                650,
                (b, bd) => b * bd,
                null,
                (b) => 1.0 + (b * 2048.0) / 48000.0,
                false,
                0.06
              );

              return {
                initialGumCount,
                initialGumAudio,
                ctxBeforeWelch,
                lateGumCount,
                lateGumAudioOnly,
                gumCountFinal: gumCountAfterNormal,
                ctxCountDuringGap,
                c1ClosedDuringGap,
                ctxCountAfterGapAdd,
                c2Closed: Boolean(c2 && c2.closed),
                ctxCountFinal: ctxCountBeforeExtra,
                c2BoundTrackIds: c2 ? c2.boundTrackIds : [],
                cLastBoundTrackIds: cLast ? cLast.boundTrackIds : [],
                trackBId: trackB.id,
                trackCId: trackC.id,
                prePumpCount,
                pushedCount: pushedBeforeDriftCount,
                pushedSkews: pushedAudioMsgs.slice(0, 4).map((m) => m.estSkewUs),
                postAudioReason,
                combAudioEndedAfterOff,
                combTracksLeftAfterOff,
                staleATracksEnded,
                staleB1TracksEnded,
                staleBTracksIds,
                staleFinalBoundIds: cStaleFinal ? cStaleFinal.boundTrackIds : [],
                trackStaleB2Id: trackStaleB2.id,
                driftPos,
                driftNeg,
                lateDriftPosP4,
                lateDriftNegP4,
                dispatchDelay5333,
                hwDispatchDelay25ms,
                hwLateDriftP4,
                ctxPullDisp5333,
                bare44k1LatePosP4,
                bare44k1LateNegP4,
                ctx512Pull48k8000,
                ctx512Pull96k5333,
                rawTsNoPlay750,
                bare48kLatePosP12,
                bare48kLateNegP12,
                ctx48kLatePosP12,
                intermittentNullTs4320,
                bareRandSeed1Pos,
                bareRandSeed7Neg,
                ctxRandSeed1Pos,
                ctxRandSeed7Neg,
                ctxRandBenign512Pull,
                bareHold20Seed2Pos,
                bareHold20Seed3Neg,
                ctxHold20Seed2Pos,
                ctxHold20Seed3Neg,
                ctxHold20Seed50Benign,
                bareRand44kSeed5Benign,
                bareHold23BernSeed2Neg20,
                ctx512Ph96Hold14Seed2Pos20,
                ctx512Ph96Hold25Seed2Pos60,
                bareBenignSlow5ms1100,
                ctx512BenignSlow5ms1100,
                ctx512Benign23p5ms750,
                slowCleanPos15us3701,
                slowCleanNeg15us3701,
                dispatch44kHeld128Cap5,
                dispatch44kHeld512Cap11p589,
                drift96kHold20Neg20,
                rvfcCaptureVideoDrift650,
              };
            } finally {
              performance.now = origPerfNow;
              Worker.prototype.postMessage = origPostMessage;
              window.AudioContext = origAudioCtx;
              window.webkitAudioContext = origWebkitAudioCtx;
              if (origMediaDevicesDesc) {
                Object.defineProperty(navigator, 'mediaDevices', origMediaDevicesDesc);
              } else {
                delete navigator.mediaDevices;
              }
              stream.getTracks().forEach((tr) => tr.stop());
              combStream.getTracks().forEach((tr) => tr.stop());
              streamStaleA.getTracks().forEach((tr) => tr.stop());
              streamStaleB.getTracks().forEach((tr) => tr.stop());
              trackStaleA.stop();
              trackStaleB1.stop();
              trackStaleB2.stop();
              trackDriftNeg.stop();
              driftExtraTracks.forEach((tr) => tr.stop());
              await realAc.close().catch(() => {});
              const rmSel = document.getElementById('rsVibroModeSelect');
              if (rmSel) { rmSel.value = '0'; rmSel.dispatchEvent(new Event('change', { bubbles: true })); }
              document.getElementById('btnResHd')?.click();
              document.getElementById('btnSrcTestbench')?.click();
              await new Promise((r) => setTimeout(r, 40));
            }
          })()`
        );
        if (
          !webcamAudioCheck ||
          webcamAudioCheck.initialGumCount !== 1 ||
          webcamAudioCheck.initialGumAudio ||
          webcamAudioCheck.ctxBeforeWelch !== 0 ||
          webcamAudioCheck.lateGumCount !== 2 ||
          !webcamAudioCheck.lateGumAudioOnly ||
          webcamAudioCheck.gumCountFinal !== 2 ||
          webcamAudioCheck.prePumpCount !== 0 ||
          webcamAudioCheck.pushedCount !== 4 ||
          webcamAudioCheck.pushedSkews.some((s) => !Number.isFinite(s) || Math.abs(s) > 250.0) ||
          (webcamAudioCheck.postAudioReason & 32) !== 0 ||
          !webcamAudioCheck.c1ClosedDuringGap ||
          webcamAudioCheck.ctxCountDuringGap !== 1 ||
          webcamAudioCheck.ctxCountAfterGapAdd < 2 ||
          !webcamAudioCheck.c2Closed ||
          webcamAudioCheck.ctxCountFinal < 3 ||
          webcamAudioCheck.c2BoundTrackIds.length !== 1 ||
          webcamAudioCheck.c2BoundTrackIds[0] !== webcamAudioCheck.trackBId ||
          webcamAudioCheck.cLastBoundTrackIds.length !== 1 ||
          webcamAudioCheck.cLastBoundTrackIds[0] !== webcamAudioCheck.trackCId ||
          !webcamAudioCheck.combAudioEndedAfterOff ||
          webcamAudioCheck.combTracksLeftAfterOff !== 0 ||
          !webcamAudioCheck.staleATracksEnded ||
          !webcamAudioCheck.staleB1TracksEnded ||
          webcamAudioCheck.staleBTracksIds.length !== 1 ||
          webcamAudioCheck.staleBTracksIds[0] !== webcamAudioCheck.trackStaleB2Id ||
          webcamAudioCheck.staleFinalBoundIds.length !== 1 ||
          webcamAudioCheck.staleFinalBoundIds[0] !== webcamAudioCheck.trackStaleB2Id ||
          !webcamAudioCheck.driftPos ||
          webcamAudioCheck.driftPos.pushedCount !== 320 ||
          webcamAudioCheck.driftPos.maxAbsSkew <= 20000 ||
          webcamAudioCheck.driftPos.lastSkew >= -20000 ||
          webcamAudioCheck.driftPos.acceptedBlocks <= 0 ||
          webcamAudioCheck.driftPos.acceptedBlocks >= 180 ||
          webcamAudioCheck.driftPos.state !== 4 ||
          (webcamAudioCheck.driftPos.reasonMask & 32) === 0 ||
          webcamAudioCheck.driftPos.welchK !== 0 ||
          webcamAudioCheck.driftPos.stampValid ||
          !webcamAudioCheck.driftNeg ||
          webcamAudioCheck.driftNeg.pushedCount !== 320 ||
          webcamAudioCheck.driftNeg.maxAbsSkew <= 20000 ||
          webcamAudioCheck.driftNeg.lastSkew <= 20000 ||
          webcamAudioCheck.driftNeg.acceptedBlocks <= 0 ||
          webcamAudioCheck.driftNeg.acceptedBlocks >= 180 ||
          webcamAudioCheck.driftNeg.state !== 4 ||
          (webcamAudioCheck.driftNeg.reasonMask & 32) === 0 ||
          webcamAudioCheck.driftNeg.welchK !== 0 ||
          webcamAudioCheck.driftNeg.stampValid ||
          !webcamAudioCheck.lateDriftPosP4 ||
          webcamAudioCheck.lateDriftPosP4.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.lateDriftPosP4.lastSkew) <= 500 ||
          webcamAudioCheck.lateDriftPosP4.acceptedBlocks <= 3000 ||
          webcamAudioCheck.lateDriftPosP4.acceptedBlocks >= 3320 ||
          webcamAudioCheck.lateDriftPosP4.state !== 4 ||
          (webcamAudioCheck.lateDriftPosP4.reasonMask & 32) === 0 ||
          webcamAudioCheck.lateDriftPosP4.welchK !== 0 ||
          webcamAudioCheck.lateDriftPosP4.stampValid ||
          !webcamAudioCheck.lateDriftNegP4 ||
          webcamAudioCheck.lateDriftNegP4.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.lateDriftNegP4.lastSkew) <= 500 ||
          webcamAudioCheck.lateDriftNegP4.acceptedBlocks <= 3000 ||
          webcamAudioCheck.lateDriftNegP4.acceptedBlocks >= 3320 ||
          webcamAudioCheck.lateDriftNegP4.state !== 4 ||
          (webcamAudioCheck.lateDriftNegP4.reasonMask & 32) === 0 ||
          webcamAudioCheck.lateDriftNegP4.welchK !== 0 ||
          webcamAudioCheck.lateDriftNegP4.stampValid ||
          !webcamAudioCheck.dispatchDelay5333 ||
          webcamAudioCheck.dispatchDelay5333.pushedCount !== 410 ||
          webcamAudioCheck.dispatchDelay5333.maxAbsSkew > 500 ||
          webcamAudioCheck.dispatchDelay5333.acceptedBlocks !== 410 ||
          webcamAudioCheck.dispatchDelay5333.state === 4 ||
          (webcamAudioCheck.dispatchDelay5333.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.hwDispatchDelay25ms ||
          webcamAudioCheck.hwDispatchDelay25ms.pushedCount !== 410 ||
          webcamAudioCheck.hwDispatchDelay25ms.maxAbsSkew !== 0 ||
          webcamAudioCheck.hwDispatchDelay25ms.acceptedBlocks !== 410 ||
          webcamAudioCheck.hwDispatchDelay25ms.state === 4 ||
          (webcamAudioCheck.hwDispatchDelay25ms.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.hwLateDriftP4 ||
          webcamAudioCheck.hwLateDriftP4.pushedCount !== 3320 ||
          webcamAudioCheck.hwLateDriftP4.lastSkew >= -4000 ||
          webcamAudioCheck.hwLateDriftP4.acceptedBlocks <= 3050 ||
          webcamAudioCheck.hwLateDriftP4.acceptedBlocks >= 3200 ||
          webcamAudioCheck.hwLateDriftP4.state !== 4 ||
          (webcamAudioCheck.hwLateDriftP4.reasonMask & 32) === 0 ||
          webcamAudioCheck.hwLateDriftP4.welchK !== 0 ||
          webcamAudioCheck.hwLateDriftP4.stampValid ||
          !webcamAudioCheck.ctxPullDisp5333 ||
          webcamAudioCheck.ctxPullDisp5333.pushedCount !== 410 ||
          webcamAudioCheck.ctxPullDisp5333.maxAbsSkew !== 0 ||
          webcamAudioCheck.ctxPullDisp5333.acceptedBlocks !== 410 ||
          webcamAudioCheck.ctxPullDisp5333.state === 4 ||
          (webcamAudioCheck.ctxPullDisp5333.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.bare44k1LatePosP4 ||
          webcamAudioCheck.bare44k1LatePosP4.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.bare44k1LatePosP4.lastSkew) <= 500 ||
          webcamAudioCheck.bare44k1LatePosP4.acceptedBlocks <= 3000 ||
          webcamAudioCheck.bare44k1LatePosP4.acceptedBlocks >= 3320 ||
          webcamAudioCheck.bare44k1LatePosP4.state !== 4 ||
          (webcamAudioCheck.bare44k1LatePosP4.reasonMask & 32) === 0 ||
          webcamAudioCheck.bare44k1LatePosP4.welchK !== 0 ||
          webcamAudioCheck.bare44k1LatePosP4.stampValid ||
          !webcamAudioCheck.bare44k1LateNegP4 ||
          webcamAudioCheck.bare44k1LateNegP4.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.bare44k1LateNegP4.lastSkew) <= 500 ||
          webcamAudioCheck.bare44k1LateNegP4.acceptedBlocks <= 3000 ||
          webcamAudioCheck.bare44k1LateNegP4.acceptedBlocks >= 3320 ||
          webcamAudioCheck.bare44k1LateNegP4.state !== 4 ||
          (webcamAudioCheck.bare44k1LateNegP4.reasonMask & 32) === 0 ||
          webcamAudioCheck.bare44k1LateNegP4.welchK !== 0 ||
          webcamAudioCheck.bare44k1LateNegP4.stampValid ||
          !webcamAudioCheck.ctx512Pull48k8000 ||
          webcamAudioCheck.ctx512Pull48k8000.pushedCount !== 410 ||
          webcamAudioCheck.ctx512Pull48k8000.maxAbsSkew !== 0 ||
          webcamAudioCheck.ctx512Pull48k8000.acceptedBlocks !== 410 ||
          webcamAudioCheck.ctx512Pull48k8000.state === 4 ||
          (webcamAudioCheck.ctx512Pull48k8000.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.ctx512Pull96k5333 ||
          webcamAudioCheck.ctx512Pull96k5333.pushedCount !== 410 ||
          webcamAudioCheck.ctx512Pull96k5333.maxAbsSkew !== 0 ||
          webcamAudioCheck.ctx512Pull96k5333.acceptedBlocks !== 410 ||
          webcamAudioCheck.ctx512Pull96k5333.state === 4 ||
          (webcamAudioCheck.ctx512Pull96k5333.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.rawTsNoPlay750 ||
          webcamAudioCheck.rawTsNoPlay750.pushedCount !== 750 ||
          webcamAudioCheck.rawTsNoPlay750.maxAbsSkew !== 0 ||
          webcamAudioCheck.rawTsNoPlay750.acceptedBlocks !== 750 ||
          webcamAudioCheck.rawTsNoPlay750.state === 4 ||
          (webcamAudioCheck.rawTsNoPlay750.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.bare48kLatePosP12 ||
          webcamAudioCheck.bare48kLatePosP12.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.bare48kLatePosP12.lastSkew) <= 500 ||
          webcamAudioCheck.bare48kLatePosP12.acceptedBlocks <= 3000 ||
          webcamAudioCheck.bare48kLatePosP12.acceptedBlocks >= 3320 ||
          webcamAudioCheck.bare48kLatePosP12.state !== 4 ||
          (webcamAudioCheck.bare48kLatePosP12.reasonMask & 32) === 0 ||
          webcamAudioCheck.bare48kLatePosP12.welchK !== 0 ||
          webcamAudioCheck.bare48kLatePosP12.stampValid ||
          !webcamAudioCheck.bare48kLateNegP12 ||
          webcamAudioCheck.bare48kLateNegP12.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.bare48kLateNegP12.lastSkew) <= 500 ||
          webcamAudioCheck.bare48kLateNegP12.acceptedBlocks <= 3000 ||
          webcamAudioCheck.bare48kLateNegP12.acceptedBlocks >= 3320 ||
          webcamAudioCheck.bare48kLateNegP12.state !== 4 ||
          (webcamAudioCheck.bare48kLateNegP12.reasonMask & 32) === 0 ||
          webcamAudioCheck.bare48kLateNegP12.welchK !== 0 ||
          webcamAudioCheck.bare48kLateNegP12.stampValid ||
          !webcamAudioCheck.ctx48kLatePosP12 ||
          webcamAudioCheck.ctx48kLatePosP12.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.ctx48kLatePosP12.lastSkew) <= 500 ||
          webcamAudioCheck.ctx48kLatePosP12.acceptedBlocks <= 3000 ||
          webcamAudioCheck.ctx48kLatePosP12.acceptedBlocks >= 3320 ||
          webcamAudioCheck.ctx48kLatePosP12.state !== 4 ||
          (webcamAudioCheck.ctx48kLatePosP12.reasonMask & 32) === 0 ||
          webcamAudioCheck.ctx48kLatePosP12.welchK !== 0 ||
          webcamAudioCheck.ctx48kLatePosP12.stampValid ||
          !webcamAudioCheck.intermittentNullTs4320 ||
          webcamAudioCheck.intermittentNullTs4320.pushedCount !== 4320 ||
          Math.abs(webcamAudioCheck.intermittentNullTs4320.lastSkew) <= 500 ||
          webcamAudioCheck.intermittentNullTs4320.acceptedBlocks <= 3000 ||
          webcamAudioCheck.intermittentNullTs4320.acceptedBlocks >= 3320 ||
          webcamAudioCheck.intermittentNullTs4320.state !== 4 ||
          (webcamAudioCheck.intermittentNullTs4320.reasonMask & 32) === 0 ||
          webcamAudioCheck.intermittentNullTs4320.welchK !== 0 ||
          webcamAudioCheck.intermittentNullTs4320.stampValid ||
          !webcamAudioCheck.bareRandSeed1Pos ||
          webcamAudioCheck.bareRandSeed1Pos.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.bareRandSeed1Pos.lastSkew) <= 500 ||
          webcamAudioCheck.bareRandSeed1Pos.acceptedBlocks <= 3000 ||
          webcamAudioCheck.bareRandSeed1Pos.acceptedBlocks >= 3320 ||
          webcamAudioCheck.bareRandSeed1Pos.state !== 4 ||
          (webcamAudioCheck.bareRandSeed1Pos.reasonMask & 32) === 0 ||
          webcamAudioCheck.bareRandSeed1Pos.welchK !== 0 ||
          webcamAudioCheck.bareRandSeed1Pos.stampValid ||
          !webcamAudioCheck.bareRandSeed7Neg ||
          webcamAudioCheck.bareRandSeed7Neg.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.bareRandSeed7Neg.lastSkew) <= 500 ||
          webcamAudioCheck.bareRandSeed7Neg.acceptedBlocks <= 3000 ||
          webcamAudioCheck.bareRandSeed7Neg.acceptedBlocks >= 3320 ||
          webcamAudioCheck.bareRandSeed7Neg.state !== 4 ||
          (webcamAudioCheck.bareRandSeed7Neg.reasonMask & 32) === 0 ||
          webcamAudioCheck.bareRandSeed7Neg.welchK !== 0 ||
          webcamAudioCheck.bareRandSeed7Neg.stampValid ||
          !webcamAudioCheck.ctxRandSeed1Pos ||
          webcamAudioCheck.ctxRandSeed1Pos.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.ctxRandSeed1Pos.lastSkew) <= 500 ||
          webcamAudioCheck.ctxRandSeed1Pos.acceptedBlocks <= 3000 ||
          webcamAudioCheck.ctxRandSeed1Pos.acceptedBlocks >= 3320 ||
          webcamAudioCheck.ctxRandSeed1Pos.state !== 4 ||
          (webcamAudioCheck.ctxRandSeed1Pos.reasonMask & 32) === 0 ||
          webcamAudioCheck.ctxRandSeed1Pos.welchK !== 0 ||
          webcamAudioCheck.ctxRandSeed1Pos.stampValid ||
          !webcamAudioCheck.ctxRandSeed7Neg ||
          webcamAudioCheck.ctxRandSeed7Neg.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.ctxRandSeed7Neg.lastSkew) <= 500 ||
          webcamAudioCheck.ctxRandSeed7Neg.acceptedBlocks <= 3000 ||
          webcamAudioCheck.ctxRandSeed7Neg.acceptedBlocks >= 3320 ||
          webcamAudioCheck.ctxRandSeed7Neg.state !== 4 ||
          (webcamAudioCheck.ctxRandSeed7Neg.reasonMask & 32) === 0 ||
          webcamAudioCheck.ctxRandSeed7Neg.welchK !== 0 ||
          webcamAudioCheck.ctxRandSeed7Neg.stampValid ||
          !webcamAudioCheck.ctxRandBenign512Pull ||
          webcamAudioCheck.ctxRandBenign512Pull.pushedCount !== 750 ||
          webcamAudioCheck.ctxRandBenign512Pull.maxAbsSkew !== 0 ||
          webcamAudioCheck.ctxRandBenign512Pull.acceptedBlocks !== 750 ||
          webcamAudioCheck.ctxRandBenign512Pull.state === 4 ||
          (webcamAudioCheck.ctxRandBenign512Pull.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.bareHold20Seed2Pos ||
          webcamAudioCheck.bareHold20Seed2Pos.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.bareHold20Seed2Pos.lastSkew) <= 500 ||
          webcamAudioCheck.bareHold20Seed2Pos.acceptedBlocks <= 3000 ||
          webcamAudioCheck.bareHold20Seed2Pos.acceptedBlocks >= 3320 ||
          webcamAudioCheck.bareHold20Seed2Pos.state !== 4 ||
          (webcamAudioCheck.bareHold20Seed2Pos.reasonMask & 32) === 0 ||
          webcamAudioCheck.bareHold20Seed2Pos.welchK !== 0 ||
          webcamAudioCheck.bareHold20Seed2Pos.stampValid ||
          !webcamAudioCheck.bareHold20Seed3Neg ||
          webcamAudioCheck.bareHold20Seed3Neg.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.bareHold20Seed3Neg.lastSkew) <= 500 ||
          webcamAudioCheck.bareHold20Seed3Neg.acceptedBlocks <= 3000 ||
          webcamAudioCheck.bareHold20Seed3Neg.acceptedBlocks >= 3320 ||
          webcamAudioCheck.bareHold20Seed3Neg.state !== 4 ||
          (webcamAudioCheck.bareHold20Seed3Neg.reasonMask & 32) === 0 ||
          webcamAudioCheck.bareHold20Seed3Neg.welchK !== 0 ||
          webcamAudioCheck.bareHold20Seed3Neg.stampValid ||
          !webcamAudioCheck.ctxHold20Seed2Pos ||
          webcamAudioCheck.ctxHold20Seed2Pos.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.ctxHold20Seed2Pos.lastSkew) <= 500 ||
          webcamAudioCheck.ctxHold20Seed2Pos.acceptedBlocks <= 3000 ||
          webcamAudioCheck.ctxHold20Seed2Pos.acceptedBlocks >= 3320 ||
          webcamAudioCheck.ctxHold20Seed2Pos.state !== 4 ||
          (webcamAudioCheck.ctxHold20Seed2Pos.reasonMask & 32) === 0 ||
          webcamAudioCheck.ctxHold20Seed2Pos.welchK !== 0 ||
          webcamAudioCheck.ctxHold20Seed2Pos.stampValid ||
          !webcamAudioCheck.ctxHold20Seed3Neg ||
          webcamAudioCheck.ctxHold20Seed3Neg.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.ctxHold20Seed3Neg.lastSkew) <= 500 ||
          webcamAudioCheck.ctxHold20Seed3Neg.acceptedBlocks <= 3000 ||
          webcamAudioCheck.ctxHold20Seed3Neg.acceptedBlocks >= 3320 ||
          webcamAudioCheck.ctxHold20Seed3Neg.state !== 4 ||
          (webcamAudioCheck.ctxHold20Seed3Neg.reasonMask & 32) === 0 ||
          webcamAudioCheck.ctxHold20Seed3Neg.welchK !== 0 ||
          webcamAudioCheck.ctxHold20Seed3Neg.stampValid ||
          !webcamAudioCheck.ctxHold20Seed50Benign ||
          webcamAudioCheck.ctxHold20Seed50Benign.pushedCount !== 750 ||
          webcamAudioCheck.ctxHold20Seed50Benign.maxAbsSkew !== 0 ||
          webcamAudioCheck.ctxHold20Seed50Benign.acceptedBlocks !== 750 ||
          webcamAudioCheck.ctxHold20Seed50Benign.state === 4 ||
          (webcamAudioCheck.ctxHold20Seed50Benign.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.bareRand44kSeed5Benign ||
          webcamAudioCheck.bareRand44kSeed5Benign.pushedCount !== 750 ||
          webcamAudioCheck.bareRand44kSeed5Benign.maxAbsSkew !== 0 ||
          webcamAudioCheck.bareRand44kSeed5Benign.acceptedBlocks !== 750 ||
          webcamAudioCheck.bareRand44kSeed5Benign.state === 4 ||
          (webcamAudioCheck.bareRand44kSeed5Benign.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.bareHold23BernSeed2Neg20 ||
          webcamAudioCheck.bareHold23BernSeed2Neg20.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.bareHold23BernSeed2Neg20.lastSkew) <= 500 ||
          webcamAudioCheck.bareHold23BernSeed2Neg20.acceptedBlocks <= 3000 ||
          webcamAudioCheck.bareHold23BernSeed2Neg20.acceptedBlocks >= 3320 ||
          webcamAudioCheck.bareHold23BernSeed2Neg20.state !== 4 ||
          (webcamAudioCheck.bareHold23BernSeed2Neg20.reasonMask & 32) === 0 ||
          webcamAudioCheck.bareHold23BernSeed2Neg20.welchK !== 0 ||
          webcamAudioCheck.bareHold23BernSeed2Neg20.stampValid ||
          !webcamAudioCheck.ctx512Ph96Hold14Seed2Pos20 ||
          webcamAudioCheck.ctx512Ph96Hold14Seed2Pos20.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.ctx512Ph96Hold14Seed2Pos20.lastSkew) <= 500 ||
          webcamAudioCheck.ctx512Ph96Hold14Seed2Pos20.acceptedBlocks <= 3000 ||
          webcamAudioCheck.ctx512Ph96Hold14Seed2Pos20.acceptedBlocks >= 3320 ||
          webcamAudioCheck.ctx512Ph96Hold14Seed2Pos20.state !== 4 ||
          (webcamAudioCheck.ctx512Ph96Hold14Seed2Pos20.reasonMask & 32) === 0 ||
          webcamAudioCheck.ctx512Ph96Hold14Seed2Pos20.welchK !== 0 ||
          webcamAudioCheck.ctx512Ph96Hold14Seed2Pos20.stampValid ||
          !webcamAudioCheck.ctx512Ph96Hold25Seed2Pos60 ||
          webcamAudioCheck.ctx512Ph96Hold25Seed2Pos60.pushedCount !== 3320 ||
          Math.abs(webcamAudioCheck.ctx512Ph96Hold25Seed2Pos60.lastSkew) <= 500 ||
          webcamAudioCheck.ctx512Ph96Hold25Seed2Pos60.acceptedBlocks <= 3000 ||
          webcamAudioCheck.ctx512Ph96Hold25Seed2Pos60.acceptedBlocks >= 3320 ||
          webcamAudioCheck.ctx512Ph96Hold25Seed2Pos60.state !== 4 ||
          (webcamAudioCheck.ctx512Ph96Hold25Seed2Pos60.reasonMask & 32) === 0 ||
          webcamAudioCheck.ctx512Ph96Hold25Seed2Pos60.welchK !== 0 ||
          webcamAudioCheck.ctx512Ph96Hold25Seed2Pos60.stampValid ||
          !webcamAudioCheck.bareBenignSlow5ms1100 ||
          webcamAudioCheck.bareBenignSlow5ms1100.pushedCount !== 1100 ||
          webcamAudioCheck.bareBenignSlow5ms1100.maxAbsSkew !== 0 ||
          webcamAudioCheck.bareBenignSlow5ms1100.acceptedBlocks !== 1100 ||
          webcamAudioCheck.bareBenignSlow5ms1100.state === 4 ||
          (webcamAudioCheck.bareBenignSlow5ms1100.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.ctx512BenignSlow5ms1100 ||
          webcamAudioCheck.ctx512BenignSlow5ms1100.pushedCount !== 1100 ||
          webcamAudioCheck.ctx512BenignSlow5ms1100.maxAbsSkew !== 0 ||
          webcamAudioCheck.ctx512BenignSlow5ms1100.acceptedBlocks !== 1100 ||
          webcamAudioCheck.ctx512BenignSlow5ms1100.state === 4 ||
          (webcamAudioCheck.ctx512BenignSlow5ms1100.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.ctx512Benign23p5ms750 ||
          webcamAudioCheck.ctx512Benign23p5ms750.pushedCount !== 750 ||
          webcamAudioCheck.ctx512Benign23p5ms750.maxAbsSkew !== 0 ||
          webcamAudioCheck.ctx512Benign23p5ms750.acceptedBlocks !== 750 ||
          webcamAudioCheck.ctx512Benign23p5ms750.state === 4 ||
          (webcamAudioCheck.ctx512Benign23p5ms750.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.slowCleanPos15us3701 ||
          webcamAudioCheck.slowCleanPos15us3701.pushedCount !== 3701 ||
          Math.abs(webcamAudioCheck.slowCleanPos15us3701.lastSkew) <= 500 ||
          webcamAudioCheck.slowCleanPos15us3701.acceptedBlocks >= 3701 ||
          webcamAudioCheck.slowCleanPos15us3701.state !== 4 ||
          (webcamAudioCheck.slowCleanPos15us3701.reasonMask & 32) === 0 ||
          webcamAudioCheck.slowCleanPos15us3701.welchK !== 0 ||
          webcamAudioCheck.slowCleanPos15us3701.stampValid ||
          !webcamAudioCheck.slowCleanNeg15us3701 ||
          webcamAudioCheck.slowCleanNeg15us3701.pushedCount !== 3701 ||
          Math.abs(webcamAudioCheck.slowCleanNeg15us3701.lastSkew) <= 500 ||
          webcamAudioCheck.slowCleanNeg15us3701.acceptedBlocks >= 3701 ||
          webcamAudioCheck.slowCleanNeg15us3701.state !== 4 ||
          (webcamAudioCheck.slowCleanNeg15us3701.reasonMask & 32) === 0 ||
          webcamAudioCheck.slowCleanNeg15us3701.welchK !== 0 ||
          webcamAudioCheck.slowCleanNeg15us3701.stampValid ||
          !webcamAudioCheck.dispatch44kHeld128Cap5 ||
          webcamAudioCheck.dispatch44kHeld128Cap5.pushedCount !== 1000 ||
          webcamAudioCheck.dispatch44kHeld128Cap5.maxAbsSkew !== 0 ||
          webcamAudioCheck.dispatch44kHeld128Cap5.acceptedBlocks !== 1000 ||
          webcamAudioCheck.dispatch44kHeld128Cap5.state === 4 ||
          (webcamAudioCheck.dispatch44kHeld128Cap5.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.dispatch44kHeld512Cap11p589 ||
          webcamAudioCheck.dispatch44kHeld512Cap11p589.pushedCount !== 1000 ||
          webcamAudioCheck.dispatch44kHeld512Cap11p589.maxAbsSkew !== 0 ||
          webcamAudioCheck.dispatch44kHeld512Cap11p589.acceptedBlocks !== 1000 ||
          webcamAudioCheck.dispatch44kHeld512Cap11p589.state === 4 ||
          (webcamAudioCheck.dispatch44kHeld512Cap11p589.reasonMask & 32) !== 0 ||
          !webcamAudioCheck.drift96kHold20Neg20 ||
          webcamAudioCheck.drift96kHold20Neg20.pushedCount !== 680 ||
          Math.abs(webcamAudioCheck.drift96kHold20Neg20.lastSkew) <= 500 ||
          webcamAudioCheck.drift96kHold20Neg20.acceptedBlocks >= 680 ||
          webcamAudioCheck.drift96kHold20Neg20.state !== 4 ||
          (webcamAudioCheck.drift96kHold20Neg20.reasonMask & 32) === 0 ||
          webcamAudioCheck.drift96kHold20Neg20.welchK !== 0 ||
          webcamAudioCheck.drift96kHold20Neg20.stampValid ||
          !webcamAudioCheck.rvfcCaptureVideoDrift650 ||
          webcamAudioCheck.rvfcCaptureVideoDrift650.pushedCount !== 650 ||
          Math.abs(webcamAudioCheck.rvfcCaptureVideoDrift650.lastSkew) <= 500 ||
          webcamAudioCheck.rvfcCaptureVideoDrift650.acceptedBlocks >= 650 ||
          webcamAudioCheck.rvfcCaptureVideoDrift650.state !== 4 ||
          (webcamAudioCheck.rvfcCaptureVideoDrift650.reasonMask & 32) === 0 ||
          webcamAudioCheck.rvfcCaptureVideoDrift650.welchK !== 0 ||
          webcamAudioCheck.rvfcCaptureVideoDrift650.stampValid
        ) {
          throw new Error('Step 10e webcam live audio anchoring / track swap gate failed: ' + JSON.stringify(webcamAudioCheck));
        }

        const w2DeckClsPost10e = await cdp.evalJs(
          w2Session,
          `(() => {
            if (window.__e2eW2DeckClsPo) window.__e2eW2DeckClsPo.disconnect();
            return window.__e2eW2DeckCls || 0;
          })()`
        );
        if (w2DeckClsPost10e >= 0.000001) {
          throw new Error(`Step 10e: Expected 0.000000 CLS across Wave-2 Deck + Step 10e flows, got ${w2DeckClsPost10e}`);
        }

        console.log(
          `  [PASS] Step 10 (Wave-2 Hodge 40x22, Calibrated RS+Welch & Exposure Null, Occluded AMBIGUOUS Conservation): hodge=(state=${hodgeState.state}, co=${hodgeState.coexactFrac.toFixed(3)}, orth=${hodgeState.orthEcRel.toExponential(1)}, hud=${w2HudBytes}B), rsWelch=(K=${rv.welchK}, f=${rv.domFreqHz.toFixed(2)}Hz, g2=${rv.maxCohGamma2.toFixed(4)}, cells=${rv.coupledCellsCount}), expNull=(H=${nv.sincH.toFixed(4)}, sev=${rsWelchReport.nullSev}), occl=(st=${oc.status}, mask=${oc.ambiguityMask}, diag="${consReport.diagCons}"), CLS=${w2DeckClsPost10e.toFixed(6)}`
        );
      } finally {
        await cdp.send('Target.closeTarget', { targetId: w2TargetId }).catch(() => {});
      }
    }

    // ------------------------------------------------------------------------
    // Step 11: v3.2.0 Interactive Mode Demo Showcases (Depth Parallax, FoE/TTC,
    //          Hodge 64x36 Circulation, 247.5 Hz RS Vibrometry & 720p Fovea)
    // ------------------------------------------------------------------------
    {
      const { targetId: v32TargetId } = await cdp.send('Target.createTarget', {
        url: new URL('/index.html?view=deck', appUrl).href,
      });
      const { sessionId: v32Session } = await cdp.send('Target.attachToTarget', {
        targetId: v32TargetId,
        flatten: true,
      });
      try {
        await cdp.send('Page.enable', {}, v32Session);
        await cdp.send('Runtime.enable', {}, v32Session);
        await cdp.send(
          'Emulation.setDeviceMetricsOverride',
          { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false },
          v32Session
        );
        await waitForCondition(
          cdp,
          v32Session,
          'v3.2.0 demos deck boot ready',
          `(() => {
            const btn = document.querySelector('.showcase-btn[data-showcase="depth_parallax"]');
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            return { ok: Boolean(btn) && Boolean(w2 && w2.available === true && w2.depth?.available), avail: Boolean(w2?.available) };
          })()`,
          20000
        );
        await cdp.evalJs(
          v32Session,
          `(() => {
            window.__e2eV32Cls = 0;
            window.__e2eV32ClsPo = new PerformanceObserver((list) => {
              for (const e of list.getEntries()) {
                if (!e.hadRecentInput) window.__e2eV32Cls += e.value;
              }
            });
            window.__e2eV32ClsPo.observe({ type: 'layout-shift', buffered: false });
            return true;
          })()`
        );

        // Slow virtualized CI runners drop 30fps HD frames under software rendering, which triggers
        // FRAME_GAP_SEQ -> continuity_epoch bumps and resets Depth/Hodge/RS sliding windows. Match
        // Step 9's CI pacing (0.25x wall-clock, exact 1/30s mediaTimeSec per frame).
        const ciPaceJs = process.env.CI
          ? `(() => { const v = document.getElementById('hiddenVideoSource'); if (v) { v.defaultPlaybackRate = 0.25; v.playbackRate = 0.25; } })()`
          : '';
        const v32WaitMs = process.env.CI ? 45000 : 25000;

        // 11a. Zero-ML Lateral Depth Parallax showcase
        await cdp.evalJs(v32Session, `${ciPaceJs}; document.querySelector('.showcase-btn[data-showcase="depth_parallax"]')?.click(); ${ciPaceJs}; true;`);
        const dpState = await waitForCondition(
          cdp,
          v32Session,
          'depth_parallax READY_PARALLAX MONO_METRIC',
          `(() => {
            ${ciPaceJs};
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            const d = w2 && w2.depth;
            return {
              ok: Boolean(d && d.state === 2 && d.provenance === 2 && d.validTracks >= 20),
              state: d ? d.state : 0,
              prov: d ? d.provenance : 0,
              tracks: d ? d.validTracks : 0,
            };
          })()`,
          v32WaitMs
        );

        // 11a2. Zero-ML Forward Approach FoE & Looming TTC showcase
        await cdp.evalJs(v32Session, `${ciPaceJs}; document.querySelector('.showcase-btn[data-showcase="depth_foe_ttc"]')?.click(); ${ciPaceJs}; true;`);
        const foeState = await waitForCondition(
          cdp,
          v32Session,
          'depth_foe_ttc FoE lock + TTC tracks',
          `(() => {
            ${ciPaceJs};
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            const d = w2 && w2.depth;
            return {
              ok: Boolean(d && d.ttcValidTracks >= 4 && d.foeInliers >= 4 && d.bestTtcR2 >= 0.96),
              state: d ? d.state : 0,
              ttcTracks: d ? d.ttcValidTracks : 0,
              foeInliers: d ? d.foeInliers : 0,
              r2: d ? d.bestTtcR2 : 0,
            };
          })()`,
          v32WaitMs
        );

        // 11b. Mesh Hodge 64x36 Cylinder Circulation showcase
        await cdp.evalJs(v32Session, `${ciPaceJs}; document.querySelector('.showcase-btn[data-showcase="hodge_cylinder"]')?.click(); ${ciPaceJs}; true;`);
        const hcState = await waitForCondition(
          cdp,
          v32Session,
          'hodge_cylinder READY with beta1>=1 and Ex/Co/Ha',
          `(() => {
            ${ciPaceJs};
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            const h = w2 && w2.hodge;
            return {
              ok: Boolean(h && h.state === 2 && h.beta1 >= 1 && h.coexactFrac > 0.05 && h.harmFrac > 0.05),
              state: h ? h.state : 0,
              beta1: h ? h.beta1 : 0,
              ex: h ? h.exactFrac : 0,
              co: h ? h.coexactFrac : 0,
              ha: h ? h.harmFrac : 0,
            };
          })()`,
          v32WaitMs
        );

        // 11c. 247.5 Hz Rolling-Shutter Vibrometry (RS_ONLY) showcase
        await cdp.evalJs(v32Session, `${ciPaceJs}; document.querySelector('.showcase-btn[data-showcase="rs_vibro_247hz"]')?.click(); ${ciPaceJs}; true;`);
        const rsState = await waitForCondition(
          cdp,
          v32Session,
          'rs_vibro_247hz READY near 247.5 Hz',
          `(() => {
            ${ciPaceJs};
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            const v = w2 && w2.vibro;
            return {
              ok: Boolean(v && v.state === 3 && Math.abs(v.domFreqHz - 247.5) < 5.0 && v.domAmpPx > 0.35),
              state: v ? v.state : 0,
              domFreqHz: v ? v.domFreqHz : 0,
              domAmpPx: v ? v.domAmpPx : 0,
            };
          })()`,
          v32WaitMs
        );

        // 11d. 720p Dual-Resolution 1:1 Fovea & PTZ showcase
        await cdp.evalJs(v32Session, `document.querySelector('.showcase-btn[data-showcase="fovea_720p"]')?.click(); true;`);
        const fvState = await waitForCondition(
          cdp,
          v32Session,
          'fovea_720p native 1280x720 + 1:1 ROI active',
          `(() => {
            const v = document.getElementById('hiddenVideoSource');
            const roi = document.getElementById('roiGeomMeta')?.textContent || '';
            return {
              ok: Boolean(v && v.videoWidth === 1280 && v.videoHeight === 720 && roi.includes('1:1 NATIVE')),
              vw: v ? v.videoWidth : 0,
              vh: v ? v.videoHeight : 0,
              roi,
            };
          })()`,
          20000
        );

        const v32Cls = await cdp.evalJs(
          v32Session,
          `(() => {
            if (window.__e2eV32ClsPo) window.__e2eV32ClsPo.disconnect();
            return window.__e2eV32Cls || 0;
          })()`
        );
        if (v32Cls >= 0.000001) {
          throw new Error(`Step 11: Expected 0.000000 CLS across v3.2.0 interactive demos, got ${v32Cls}`);
        }
        console.log(
          `  [PASS] Step 11 (v3.2.0 Interactive Mode Demos): depthParallax=(MONO_METRIC, trk=${dpState.tracks}), foeTtc=(ttc=${foeState.ttcTracks}, R2=${foeState.r2.toFixed(3)}), hodgeCyl=(b1=${hcState.beta1}, Ex=${(hcState.ex*100).toFixed(0)}%/Co=${(hcState.co*100).toFixed(0)}%/Ha=${(hcState.ha*100).toFixed(0)}%), rsVibro=(${rsState.domFreqHz.toFixed(1)}Hz, amp=${rsState.domAmpPx.toFixed(2)}px), fovea720p=(${fvState.vw}x${fvState.vh}, ${fvState.roi}), CLS=${v32Cls.toFixed(6)}`
        );
      } finally {
        await cdp.send('Target.closeTarget', { targetId: v32TargetId }).catch(() => {});
      }
    }

    // ------------------------------------------------------------------------
    // Step 12: v3.3.0 Interactive Mode Demos (Pocket Wind Tunnel Cp + Lift/Drag,
    //          kHz Optical Elastography Crack Panel & 1.2 kHz Lock-In Acoustic BOS)
    // ------------------------------------------------------------------------
    {
      const { targetId: v33TargetId } = await cdp.send('Target.createTarget', {
        url: new URL('/index.html?view=deck', appUrl).href,
      });
      const { sessionId: v33Session } = await cdp.send('Target.attachToTarget', {
        targetId: v33TargetId,
        flatten: true,
      });
      try {
        await cdp.send('Page.enable', {}, v33Session);
        await cdp.send('Runtime.enable', {}, v33Session);
        await cdp.send(
          'Emulation.setDeviceMetricsOverride',
          { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false },
          v33Session
        );
        await waitForCondition(
          cdp,
          v33Session,
          'v3.3.0 demos deck boot ready',
          `(() => {
            const btnW = document.querySelector('.showcase-btn[data-showcase="wind_lifting_cyl"]');
            const btnE = document.querySelector('.showcase-btn[data-showcase="elasto_crack"]');
            const btnA = document.querySelector('.showcase-btn[data-showcase="acoustic_bos"]');
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            return {
              ok: Boolean(btnW && btnE && btnA && w2 && w2.available === true && w2.wind?.available && w2.elasto?.available),
              avail: Boolean(w2?.available),
              windAvail: Boolean(w2?.wind?.available),
              elastoAvail: Boolean(w2?.elasto?.available),
            };
          })()`,
          20000
        );
        await cdp.evalJs(
          v33Session,
          `(() => {
            window.__e2eV33Cls = 0;
            window.__e2eV33ClsPo = new PerformanceObserver((list) => {
              for (const e of list.getEntries()) {
                if (!e.hadRecentInput) window.__e2eV33Cls += e.value;
              }
            });
            window.__e2eV33ClsPo.observe({ type: 'layout-shift', buffered: false });
            return true;
          })()`
        );

        const ciPaceV33Js = process.env.CI
          ? `(() => { const v = document.getElementById('hiddenVideoSource'); if (v) { v.defaultPlaybackRate = 0.25; v.playbackRate = 0.25; } })()`
          : '';
        const v33WaitMs = process.env.CI ? 50000 : 30000;

        // 12a. Pocket Wind Tunnel: Lifting Cylinder Cp & Kutta-Joukowski Lift/Drag
        await cdp.evalJs(v33Session, `${ciPaceV33Js}; document.querySelector('.showcase-btn[data-showcase="wind_lifting_cyl"]')?.click(); ${ciPaceV33Js}; true;`);
        const wtState = await waitForCondition(
          cdp,
          v33Session,
          'wind_lifting_cyl READY_LOADS with Cp + Kutta-Joukowski lift',
          `(() => {
            ${ciPaceV33Js};
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            const w = w2 && w2.wind;
            const om = Number(document.getElementById('fieldOverlayModeSelect')?.value || 0);
            return {
              ok: Boolean(w && w.state === 3 && w.validCells >= 1200 && Math.abs(w.cl) > 1.5 && Math.abs(w.clKj) > 1.5 && om === 13),
              state: w ? w.state : 0,
              cells: w ? w.validCells : 0,
              cl: w ? w.cl : 0,
              cd: w ? w.cd : 0,
              clKj: w ? w.clKj : 0,
              om,
            };
          })()`,
          v33WaitMs
        );

        // 12b. kHz Optical Elastography: Hidden Crack Panel (380+860 Hz Kirchhoff Plate)
        await cdp.evalJs(v33Session, `${ciPaceV33Js}; document.querySelector('.showcase-btn[data-showcase="elasto_crack"]')?.click(); ${ciPaceV33Js}; true;`);
        const elState = await waitForCondition(
          cdp,
          v33Session,
          'elasto_crack READY KIRCHHOFF_PLATE via muxed 48 kHz audio',
          `(() => {
            ${ciPaceV33Js};
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            const e = w2 && w2.elasto;
            const om = Number(document.getElementById('fieldOverlayModeSelect')?.value || 0);
            return {
              ok: Boolean(e && e.state === 3 && e.dispClass === 1 && e.alpha > 0.35 && e.alpha < 0.72 && e.meanCoh > 0.75 && e.validCells >= 1500 && om === 14),
              state: e ? e.state : 0,
              cls: e ? e.dispClass : 0,
              alpha: e ? e.alpha : 0,
              medC: e ? e.medC : 0,
              coh: e ? e.meanCoh : 0,
              cells: e ? e.validCells : 0,
              reason: e ? e.reasonMask : 0,
              om,
            };
          })()`,
          v33WaitMs
        );

        // 12c. Lock-In Acoustic BOS: 1,200 Hz Standing Wave (256x128 |Phi|)
        await cdp.evalJs(v33Session, `${ciPaceV33Js}; document.querySelector('.showcase-btn[data-showcase="acoustic_bos"]')?.click(); ${ciPaceV33Js}; true;`);
        const abosState = await waitForCondition(
          cdp,
          v33Session,
          'acoustic_bos READY 256x128 |Phi| via muxed 48 kHz audio',
          `(() => {
            ${ciPaceV33Js};
            const w2 = window.__chronosWave2Probe?.getLastWave2();
            const e = w2 && w2.elasto;
            const om = Number(document.getElementById('fieldOverlayModeSelect')?.value || 0);
            return {
              ok: Boolean(e && e.state === 3 && e.mode === 4 && e.abosPhiRms > 0.15 && e.abosCurlRms < 0.08 && e.validCells >= 16000 && om === 15),
              state: e ? e.state : 0,
              mode: e ? e.mode : 0,
              phiRms: e ? e.abosPhiRms : 0,
              curlRms: e ? e.abosCurlRms : 0,
              cells: e ? e.validCells : 0,
              reason: e ? e.reasonMask : 0,
              om,
            };
          })()`,
          v33WaitMs
        );

        const v33Cls = await cdp.evalJs(
          v33Session,
          `(() => {
            if (window.__e2eV33ClsPo) window.__e2eV33ClsPo.disconnect();
            return window.__e2eV33Cls || 0;
          })()`
        );
        if (v33Cls >= 0.000001) {
          throw new Error(`Step 12: Expected 0.000000 CLS across v3.3.0 interactive demos, got ${v33Cls}`);
        }
        console.log(
          `  [PASS] Step 12 (v3.3.0 Wind Tunnel, kHz Elastography & Acoustic BOS): wind=(CL=${wtState.cl.toFixed(2)}, KJ=${wtState.clKj.toFixed(2)}, cells=${wtState.cells}), elasto=(KIRCHHOFF, a=${elState.alpha.toFixed(2)}, c=${Math.round(elState.medC)}, coh=${elState.coh.toFixed(2)}), abos=(|Phi|=${abosState.phiRms.toFixed(3)}, curl=${abosState.curlRms.toFixed(3)}, cells=${abosState.cells}), CLS=${v33Cls.toFixed(6)}`
        );
      } finally {
        await cdp.send('Target.closeTarget', { targetId: v33TargetId }).catch(() => {});
      }
    }

    // ------------------------------------------------------------------------
    // Step 13: v3.4.0 Physics Integrity Audit Card (Lazy Worker, A/B Audit, Zero-CLS)
    // ------------------------------------------------------------------------
    {
      const { targetId: v34TargetId } = await cdp.send('Target.createTarget', {
        url: new URL('/index.html', appUrl).href,
      });
      try {
        const { sessionId: v34Session } = await cdp.send('Target.attachToTarget', { targetId: v34TargetId, flatten: true });
        await cdp.send('Page.enable', {}, v34Session);
        await cdp.send('Runtime.enable', {}, v34Session);
        await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false }, v34Session);

        await waitForCondition(
          cdp,
          v34Session,
          'v3.4.0 Physics Audit Card mounted (lazy worker idle)',
          `(() => {
            window.__chronosUi?.reveal('physicsAuditCard');
            const p = window.__chronosAuditProbe;
            const card = document.getElementById('physicsAuditCard');
            return { ok: Boolean(p && card && card.getBoundingClientRect().height > 100), ready: p ? p.workerReady() : null };
          })()`,
          20000
        );

        // Verify lazy worker is NOT spawned on initial page load and install CLS observer
        const preAudit = await cdp.evalJs(
          v34Session,
          `(() => {
            window.__e2eV34Cls = 0;
            window.__e2eV34ClsPo = new PerformanceObserver((list) => {
              for (const e of list.getEntries()) {
                if (!e.hadRecentInput) window.__e2eV34Cls += e.value;
              }
            });
            window.__e2eV34ClsPo.observe({ type: 'layout-shift', buffered: false });
            return {
              workerReadyBefore: window.__chronosAuditProbe.workerReady(),
              cardH: document.getElementById('physicsAuditCard').getBoundingClientRect().height,
            };
          })()`
        );
        if (preAudit.workerReadyBefore !== false) {
          throw new Error(`Step 13: Audit worker must be lazy (workerReady() === false before first audit), got ${preAudit.workerReadyBefore}`);
        }

        // Run Slot A (real juggling MP4 via WebCodecs) and Slot B (synthetic stutter_freeze) in parallel/sequence
        const auditRes = await cdp.evalJs(
          v34Session,
          `(async () => {
            const a = await window.__chronosAuditProbe.runUrl('A', './samples/juggling_ballistic.mp4', { profile: 'quick', maxFrames: 75 });
            const b = await window.__chronosAuditProbe.runSynthetic('B', 'stutter_freeze', { profile: 'quick', maxFrames: 48 });
            const st = window.__chronosAuditProbe.getState();
            const cvs = document.getElementById('auditTimelineCanvas');
            const ctx = cvs.getContext('2d');
            const img = ctx.getImageData(0, 0, cvs.width, cvs.height).data;
            let paintedPx = 0;
            for (let i = 0; i < img.length; i += 4) {
              if (img[i] > 20 || img[i + 1] > 40 || img[i + 2] > 60) paintedPx++;
            }
            if (window.__e2eV34ClsPo) window.__e2eV34ClsPo.disconnect();
            return {
              workerReadyAfter: st.workerReady,
              slotA: { grade: a.verdict.grade, integrity: a.verdict.integrity, frames: a.ingest.frames_committed, viol: a.verdict.violations },
              slotB: { grade: b.verdict.grade, integrity: b.verdict.integrity, frames: b.ingest.frames_committed, viol: b.verdict.violations, codes: b.violations.map((v) => v.code) },
              pillStates: st.pillStates,
              paintedPx,
              cls: window.__e2eV34Cls || 0,
              cardHAfter: document.getElementById('physicsAuditCard').getBoundingClientRect().height,
            };
          })()`,
          45000
        );

        if (!auditRes.workerReadyAfter) {
          throw new Error('Step 13: Audit worker did not report ready after run');
        }
        if (auditRes.slotA.frames < 40 || auditRes.slotA.grade === 'FAIL') {
          throw new Error(`Step 13: Real juggling Slot A unexpected result: ${JSON.stringify(auditRes.slotA)}`);
        }
        if (auditRes.slotB.grade !== 'FAIL' || !auditRes.slotB.codes.includes('FROZEN_MID_MOTION')) {
          throw new Error(`Step 13: Adversarial stutter Slot B expected FAIL + FROZEN_MID_MOTION, got ${JSON.stringify(auditRes.slotB)}`);
        }
        if (auditRes.pillStates.temporal_cadence !== 'fail') {
          throw new Error(`Step 13: Expected temporal_cadence pill data-state="fail", got ${JSON.stringify(auditRes.pillStates)}`);
        }
        if (auditRes.paintedPx < 500) {
          throw new Error(`Step 13: Expected painted audit timeline canvas (>500 px), got ${auditRes.paintedPx}`);
        }
        if (auditRes.cls >= 0.000001 || Math.abs(auditRes.cardHAfter - preAudit.cardH) > 0.5) {
          throw new Error(`Step 13: Expected 0.000000 CLS & locked height (${preAudit.cardH} -> ${auditRes.cardHAfter}), got CLS=${auditRes.cls}`);
        }
        console.log(
          `  [PASS] Step 13 (v3.4.0 Physics Integrity Audit Card): lazyWorker=OK, slotA=(${auditRes.slotA.grade}, I=${auditRes.slotA.integrity}, ${auditRes.slotA.frames}f), slotB=(${auditRes.slotB.grade}, codes=[${auditRes.slotB.codes.join(',')}]), canvasPx=${auditRes.paintedPx}, CLS=${auditRes.cls.toFixed(6)}`
        );
      } finally {
        await cdp.send('Target.closeTarget', { targetId: v34TargetId }).catch(() => {});
      }
    }

    // ------------------------------------------------------------------------
    // Step 14: v3.5.0 Scientific Instrument FSM, 0-CLS Tab/Drawer/Focus Sweep,
    // Live 180-Tuple BOS Kernel Mutex Parity, Dense Wind (windMode=3),
    // Explicit Conflict -> 1-Click Scratch Bus Handoff -> Undo, Settings Only
    // ------------------------------------------------------------------------
    {
      const { targetId: v35TargetId } = await cdp.send('Target.createTarget', {
        url: new URL('/index.html?view=deck', appUrl).href,
      });
      try {
        const { sessionId: v35Session } = await cdp.send('Target.attachToTarget', { targetId: v35TargetId, flatten: true });
        await cdp.send('Page.enable', {}, v35Session);
        await cdp.send('Runtime.enable', {}, v35Session);
        await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false }, v35Session);

        await waitForCondition(
          cdp,
          v35Session,
          'Step 14a: v3.5 Instrument UI ready with 100% [data-ctl] annotated',
          `(() => {
            const ui = window.__chronosUi;
            const total = document.querySelectorAll('[data-ctl]').length;
            const missing = document.querySelectorAll('[data-ctl]:not([data-ctl-state])').length;
            const ready = document.getElementById('pillKernel')?.textContent?.includes('0 HEAP ALLOC');
            return { ok: Boolean(ui && ready && total >= 104 && missing === 0), total, missing };
          })()`,
          20000
        );

        // 14a. Raw Zero-CLS & Per-Step 0.00px Rect Drift Sweep across Rack tabs, Bay tabs, All|Applicable & Presets Drawer
        const zeroShiftSweep = await cdp.evalJs(
          v35Session,
          `new Promise((resolve) => {
            let rawCls = 0;
            const po = new PerformanceObserver((list) => {
              for (const e of list.getEntries()) rawCls += e.value;
            });
            po.observe({ type: 'layout-shift', buffered: false });
            const hud0 = document.getElementById('hudCanvas').getBoundingClientRect();
            const bay0 = document.querySelector('.bench-bay').getBoundingClientRect();
            let maxDrift = 0;
            const sampleDrift = () => {
              const h = document.getElementById('hudCanvas').getBoundingClientRect();
              const b = document.querySelector('.bench-bay').getBoundingClientRect();
              maxDrift = Math.max(
                maxDrift,
                Math.abs(h.x - hud0.x), Math.abs(h.y - hud0.y),
                Math.abs(h.width - hud0.width), Math.abs(h.height - hud0.height),
                Math.abs(b.x - bay0.x), Math.abs(b.y - bay0.y),
                Math.abs(b.width - bay0.width), Math.abs(b.height - bay0.height)
              );
            };
            const ui = window.__chronosUi;
            const steps = [
              () => ui.setRackTab('acquire'),
              () => ui.setRackTab('calibrate'),
              () => ui.setRackTab('record'),
              () => ui.setRackTab('engines'),
              () => ui.setBayTab('phase'),
              () => ui.setBayTab('physics'),
              () => ui.setBayTab('field'),
              () => ui.setBayTab('audit'),
              () => ui.setBayTab('track'),
              () => ui.setVisibilityPolicy('APPLICABLE_ONLY'),
              () => ui.setVisibilityPolicy('ALL_CONSTRAINED'),
              () => ui.openPresets(true),
              () => ui.openPresets(false),
            ];
            let idx = 0;
            const runNext = () => {
              if (idx < steps.length) {
                steps[idx++]();
                requestAnimationFrame(() => {
                  sampleDrift();
                  runNext();
                });
                return;
              }
              requestAnimationFrame(() => {
                po.disconnect();
                sampleDrift();
                const overflowY = document.documentElement.scrollHeight - window.innerHeight;
                const railRight = document.getElementById('btnShortcuts')?.getBoundingClientRect()?.right ?? 9999;
                resolve({ rawCls, drift: maxDrift, overflowY, railRight, vpW: window.innerWidth });
              });
            };
            runNext();
          })`
        );
        if (zeroShiftSweep.rawCls >= 0.000001 || zeroShiftSweep.drift >= 0.05 || zeroShiftSweep.overflowY > 0 || zeroShiftSweep.railRight > zeroShiftSweep.vpW) {
          throw new Error(`Step 14a: Zero-shift bench sweep / 1440px rail check failed: ${JSON.stringify(zeroShiftSweep)}`);
        }

        // 14b. Live 180-Tuple BOS Scratch Mutex Kernel Parity (3 bos x 4 wind x 5 elasto x 3 depth)
        const mutexParity = await cdp.evalJs(
          v35Session,
          `(() => {
            const ui = window.__chronosUi;
            const expectedOwner = (bos, wind, elasto, depth) => {
              if (depth === 2) return 'DEPTH_STEREO';
              if (bos > 0) return 'BOS_THERMAL';
              if (wind > 0) return 'WIND_TUNNEL';
              if (elasto === 4) return 'ELASTO_ACOUSTIC_BOS';
              if (elasto >= 1 && elasto <= 3) return 'ELASTO_SURFACE';
              return 'NONE';
            };
            let checked = 0;
            let mismatches = [];
            for (const depth of [0, 1, 2]) {
              for (const bos of [0, 1, 2]) {
                for (const wind of [0, 1, 2, 3]) {
                  for (const elasto of [0, 1, 2, 3, 4]) {
                    const exp = expectedOwner(bos, wind, elasto, depth);
                    const bosSel = document.getElementById('bosModeSelect');
                    const windSel = document.getElementById('windModeSelect');
                    const elSel = document.getElementById('elastoModeSelect');
                    const depSel = document.getElementById('depthModeSelect');
                    bosSel.value = String(bos); bosSel.dispatchEvent(new Event('change', { bubbles: true }));
                    windSel.value = String(wind); windSel.dispatchEvent(new Event('change', { bubbles: true }));
                    elSel.value = String(elasto); elSel.dispatchEvent(new Event('change', { bubbles: true }));
                    depSel.value = String(depth); depSel.dispatchEvent(new Event('change', { bubbles: true }));
                    const got = ui.getState().bos_owner;
                    if (got !== exp) mismatches.push({ bos, wind, elasto, depth, exp, got });
                    checked++;
                  }
                }
              }
            }
            // Reset all to 0
            for (const id of ['depthModeSelect', 'bosModeSelect', 'windModeSelect', 'elastoModeSelect']) {
              const el = document.getElementById(id);
              el.value = '0';
              el.dispatchEvent(new Event('change', { bubbles: true }));
            }
            return { checked, mismatchCount: mismatches.length, sample: mismatches.slice(0, 3) };
          })()`,
          20000
        );
        if (mutexParity.checked !== 180 || mutexParity.mismatchCount !== 0) {
          throw new Error(`Step 14b: Live 180-tuple BOS mutex parity failed: ${JSON.stringify(mutexParity)}`);
        }

        // 14c. Explicit Conflict (BOS owns P2, user selects Wind=1) -> 1-Click Scratch Bus Handoff (+ live worker ACK) -> 1-Level Undo
        const conflictFlow = await cdp.evalJs(
          v35Session,
          `(async () => {
            const ui = window.__chronosUi;
            const bosSel = document.getElementById('bosModeSelect');
            const windSel = document.getElementById('windModeSelect');
            bosSel.value = '1'; bosSel.dispatchEvent(new Event('change', { bubbles: true }));
            windSel.value = '1'; windSel.dispatchEvent(new Event('change', { bubbles: true }));
            const duringConflict = {
              bosVal: bosSel.value,
              windState: windSel.dataset.ctlState,
              owner: ui.getState().bos_owner,
              statusTxt: document.getElementById('statusWind')?.textContent || '',
            };
            document.getElementById('busSlotWind').click();
            await new Promise((r) => setTimeout(r, 80));
            const afterHandoff = {
              bosVal: bosSel.value,
              windVal: windSel.value,
              swarmVal: document.getElementById('swarmModeSelect')?.value,
              hodgeVal: document.getElementById('hodgeModeSelect')?.value,
              owner: ui.getState().bos_owner,
              workerAckOwner: ui.getState().runtime_health.observedBosOwner,
              undoDisabled: document.getElementById('btnUndoConfig')?.getAttribute('aria-disabled'),
            };
            document.getElementById('btnUndoConfig').click();
            await new Promise((r) => setTimeout(r, 80));
            const afterUndo = {
              bosVal: bosSel.value,
              windVal: windSel.value,
              owner: ui.getState().bos_owner,
              workerAckOwner: ui.getState().runtime_health.observedBosOwner,
            };
            return { duringConflict, afterHandoff, afterUndo };
          })()`
        );
        if (
          conflictFlow.duringConflict.bosVal !== '1' ||
          conflictFlow.duringConflict.windState !== 'refused' ||
          !conflictFlow.duringConflict.statusTxt.includes('Thermal BOS') ||
          conflictFlow.afterHandoff.bosVal !== '0' ||
          conflictFlow.afterHandoff.owner !== 'WIND_TUNNEL' ||
          conflictFlow.afterHandoff.workerAckOwner !== 'WIND_TUNNEL' ||
          conflictFlow.afterHandoff.swarmVal !== '2' ||
          conflictFlow.afterHandoff.hodgeVal !== '1' ||
          conflictFlow.afterHandoff.undoDisabled !== 'false' ||
          conflictFlow.afterUndo.bosVal !== '1' ||
          conflictFlow.afterUndo.owner !== 'BOS_THERMAL' ||
          conflictFlow.afterUndo.workerAckOwner !== 'BOS_THERMAL'
        ) {
          throw new Error(`Step 14c: Conflict -> Scratch Bus handoff -> Undo failed: ${JSON.stringify(conflictFlow)}`);
        }

        // 14d. Live `windMode = 3` (Dense Audit FIELD ONLY) on Real MRI Karman Clip + Zero Raw Hex
        await cdp.evalJs(v35Session, `window.__chronosUi.loadRecipe('wind_karman_mri'); true;`);
        const denseWindState = await waitForCondition(
          cdp,
          v35Session,
          'Step 14d: wind_karman_mri activates windMode=3 FIELD ONLY with live Dense-LK physics & zero raw hex',
          `(() => {
            const wm = document.getElementById('windModeSelect')?.value || '';
            const opt13 = document.querySelector('#fieldOverlayModeSelect option[value="13"]')?.textContent || '';
            const banner = document.getElementById('acquisitionStatusBanner')?.textContent || '';
            const diagW = document.getElementById('diagWindTunnel')?.textContent || '';
            const w2 = window.__chronosWave2Probe?.getLastWave2?.();
            const w = w2?.wind;
            const wState = w?.state ?? 0;
            const rMask = w?.reasonMask ?? 0;
            const densePhysicsLive = Boolean(
              w &&
              w.mode === 3 &&
              wState !== 3 &&
              (w.validCells || 0) >= 64 &&
              (wState === 2 || (wState === 4 && (rMask & ((1 << 3) | (1 << 4))) !== 0))
            );
            return {
              ok: wm === '3' &&
                  opt13.includes('FIELD ONLY') &&
                  !banner.includes('0x') &&
                  !diagW.includes('0x') &&
                  densePhysicsLive,
              wm, opt13, banner, diagW, wState, mode: w?.mode ?? 0, rMask, cells: w?.validCells || 0, prov: w?.provenance || 0,
            };
          })()`,
          22000
        );

        // 14e. `Settings Only` Recipe Application preserves currently loaded clip while applying new organs
        await cdp.evalJs(
          v35Session,
          `window.__chronosUi.loadRecipe('bos_plume', { settingsOnly: true }); true;`
        );
        const settingsOnlyCheck = await waitForCondition(
          cdp,
          v35Session,
          'Step 14e: Settings Only recipe applies BOS onto existing MRI clip without switching source',
          `(() => {
            const url = document.getElementById('sampleVideoSelect')?.value || '';
            const bos = document.getElementById('bosModeSelect')?.value || '';
            const ov = document.getElementById('fieldOverlayModeSelect')?.value || '';
            const wind = document.getElementById('windModeSelect')?.value || '';
            return {
              ok: url.endsWith('real_karman_mri_cylinder.mp4') && bos === '1' && ov === '5' && wind === '0',
              url, bos, ov, wind,
            };
          })()`,
          6000
        );

        console.log(
          `  [PASS] Step 14 (v3.5 Scientific Instrument FSM, 0-CLS, 180-Tuple Mutex, Dense Wind & Settings Only): rawCLS=${zeroShiftSweep.rawCls.toFixed(6)}, drift=${zeroShiftSweep.drift.toFixed(2)}px, mutexParity=${mutexParity.checked}/180, handoff+undo=OK, denseWind=(mode=3, state=${denseWindState.wState}, FIELD_ONLY=true), settingsOnly=OK`
        );
      } finally {
        await cdp.send('Target.closeTarget', { targetId: v35TargetId }).catch(() => {});
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

export { probeSwarmInspectOnHud, verifyNativeWebcamAudioGate };

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main()
    .then(() => {
      process.exit(0);
    })
    .catch((err) => {
      console.error('E2E FAILED:', err.message || err);
      process.exit(1);
    });
}
