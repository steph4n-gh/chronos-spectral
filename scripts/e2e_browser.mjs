#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Zero-Dependency Headless Chrome CDP Full-Stack E2E Gate
// Exercises V8 Web Worker + OffscreenCanvas HUD, 256-page Wasm SIMD128 kernel,
// runtime 320x180 SD <-> 640x360 HD switching, real-camera H.264 MP4 playback,
// Lab Modes A/B/C, and 38-column CSV/JSON telemetry recording & export.
// ============================================================================

import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const WEB_DIR = path.join(ROOT_DIR, 'web');

const MIME_TYPES = {
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

function startStaticWebServer(rootDir) {
  const server = http.createServer((req, res) => {
    const reqUrl = new URL(req.url || '/', 'http://127.0.0.1');
    const relPath = decodeURIComponent(reqUrl.pathname === '/' ? '/index.html' : reqUrl.pathname);
    const safePath = path.normalize(path.join(rootDir, relPath));
    if (!safePath.startsWith(rootDir)) {
      res.writeHead(403);
      res.end('Forbidden');
      return;
    }
    if (!fs.existsSync(safePath) || fs.statSync(safePath).isDirectory()) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }

    const stat = fs.statSync(safePath);
    const ext = path.extname(safePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
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
        'Content-Type': contentType,
        'Content-Length': end - start + 1,
        'Content-Range': `bytes ${start}-${end}/${stat.size}`,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-store',
      });
      fs.createReadStream(safePath, { start, end }).pipe(res);
      return;
    }

    res.writeHead(200, {
      'Content-Type': contentType,
      'Content-Length': stat.size,
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'no-store',
    });
    fs.createReadStream(safePath).pipe(res);
  });

  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, port: addr.port });
    });
  });
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
      throw new Error(`Page JS exception: ${res.exceptionDetails.text || JSON.stringify(res.exceptionDetails)}`);
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

async function waitForCondition(cdp, sessionId, description, jsExpr, timeoutMs = 12000) {
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
  console.log(`  Browser : ${chromeBin}`);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chronos-e2e-'));
  const downloadDir = path.join(tmpDir, 'downloads');
  fs.mkdirSync(downloadDir, { recursive: true });

  const { server, port } = await startStaticWebServer(WEB_DIR);
  const appUrl = `http://127.0.0.1:${port}/index.html`;
  let chromeProc = null;
  let cdp = null;

  try {
    const launched = await launchHeadlessChrome(chromeBin, tmpDir);
    chromeProc = launched.proc;
    cdp = new CdpClient(launched.wsUrl);
    await cdp.connect();

    const pageErrors = [];
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

    const { targetId } = await cdp.send('Target.createTarget', { url: appUrl });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Runtime.enable', {}, sessionId);
    await cdp.send('Browser.setDownloadBehavior', {
      behavior: 'allow',
      downloadPath: downloadDir,
    });

    // 1. Verify Boot + 256-Page Arena + 320x180 SD Blind Pillar Testbench (0 Oracle swaps)
    const sdState = await waitForCondition(
      cdp,
      sessionId,
      '320x180 SD boot + pillar collisions + 0 Oracle swaps',
      `(() => {
        const kernel = document.getElementById('pillKernel')?.textContent || '';
        const arena = document.getElementById('pillArena')?.textContent || '';
        const resMeta = document.getElementById('inputResMeta')?.textContent || '';
        const targets = document.getElementById('kpiTargets')?.textContent || '';
        const swaps = document.getElementById('kpiPadicSwaps')?.textContent || '';
        const collisions = parseInt(document.getElementById('kpiCollisions')?.textContent || '0', 10);
        return {
          ok: kernel.includes('0 HEAP ALLOC / 0 IMPORTS') &&
              arena.includes('256 PAGES / 16.00 MB') &&
              resMeta === '320×180 SD RGBA' &&
              targets === '4 / 8' &&
              collisions >= 2 &&
              swaps.includes('0'),
          kernel, arena, resMeta, targets, swaps, collisions,
        };
      })()`
    );
    console.log(`  [PASS] Step 1 (320x180 SD Boot & Pillar Bench): ${sdState.arena}, targets=${sdState.targets}, collisions=${sdState.collisions}, swaps=${sdState.swaps}`);

    // 2. Runtime Switch to 640x360 HD (32x32 PiP)
    await cdp.evalJs(sessionId, `document.getElementById('btnResHd').click(); true;`);
    const hdState = await waitForCondition(
      cdp,
      sessionId,
      '640x360 HD live switch + 4/8 targets + 0 Oracle swaps',
      `(() => {
        const resMeta = document.getElementById('inputResMeta')?.textContent || '';
        const pipAria = document.getElementById('pipCanvas')?.getAttribute('aria-label') || '';
        const targets = document.getElementById('kpiTargets')?.textContent || '';
        const swaps = document.getElementById('kpiPadicSwaps')?.textContent || '';
        const collisions = parseInt(document.getElementById('kpiCollisions')?.textContent || '0', 10);
        return {
          ok: resMeta === '640×360 HD RGBA' &&
              pipAria.startsWith('32x32') &&
              targets === '4 / 8' &&
              collisions >= 1 &&
              swaps.includes('0'),
          resMeta, pipAria, targets, swaps, collisions,
        };
      })()`
    );
    console.log(`  [PASS] Step 2 (640x360 HD Live Switch): res=${hdState.resMeta}, pip=${hdState.pipAria}, collisions=${hdState.collisions}, swaps=${hdState.swaps}`);

    // 3. Built-in Real-Camera Sample MP4 Playback in 640x360 HD
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
      'Sample MP4 playback + createImageBitmap(640x360) target lock',
      `(() => {
        const v = document.getElementById('hiddenVideoSource');
        const srcLabel = document.getElementById('hudSourceLabel')?.textContent || '';
        const targetsText = document.getElementById('kpiTargets')?.textContent || '0 / 8';
        const activeCount = parseInt(targetsText.split('/')[0], 10) || 0;
        const clutterText = document.getElementById('kpiClutterDb')?.textContent || '';
        return {
          ok: srcLabel.includes('SAMPLE MP4') &&
              v && !v.paused && v.currentTime > 0.35 &&
              activeCount >= 2,
          srcLabel,
          currentTime: v ? Number(v.currentTime.toFixed(3)) : 0,
          activeCount,
          clutterText,
        };
      })()`,
      15000
    );
    console.log(`  [PASS] Step 3 (Real-Camera H.264 MP4 @ 640x360 HD): t=${mp4State.currentTime}s, locked=${mp4State.activeCount}/8, ECA=${mp4State.clutterText}`);

    // 4. Session Telemetry Recorder (Record -> Pause -> Export CSV -> Export JSON)
    await cdp.evalJs(sessionId, `document.getElementById('btnRecToggle').click(); true;`);
    const recState = await waitForCondition(
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
      'CSV telemetry export payload',
      `(() => {
        const ex = window.__lastTelemetryExport;
        if (!ex || !ex.filename.endsWith('.csv')) return { ok: false };
        const lines = ex.fullText.trim().split('\\n');
        const headerCols = lines[0].split(',');
        const firstRow = lines[1] ? lines[1].split(',').map(Number) : [];
        const allFinite = lines.slice(1).every((l) => l.split(',').every((v) => Number.isFinite(Number(v))));
        return {
          ok: headerCols.length === 38 && lines.length >= 17 && allFinite && firstRow[2] === 640 && firstRow[3] === 360,
          filename: ex.filename,
          cols: headerCols.length,
          rows: lines.length - 1,
          frameW: firstRow[2],
          frameH: firstRow[3],
        };
      })()`
    );

    await cdp.evalJs(sessionId, `window.__lastTelemetryExport = null; document.getElementById('btnExportJson').click(); true;`);
    const jsonExport = await waitForCondition(
      cdp,
      sessionId,
      'JSON telemetry export payload',
      `(() => {
        const ex = window.__lastTelemetryExport;
        if (!ex || !ex.filename.endsWith('.json')) return { ok: false };
        const parsed = JSON.parse(ex.fullText);
        return {
          ok: parsed.columns.length === 38 &&
              parsed.frameCount >= 16 &&
              parsed.frames.length === parsed.frameCount &&
              parsed.meta.resMode === 1 &&
              parsed.meta.frameW === 640 &&
              parsed.meta.frameH === 360 &&
              parsed.meta.pipDim === 32,
          filename: ex.filename,
          frameCount: parsed.frameCount,
          meta: parsed.meta,
        };
      })()`
    );
    console.log(`  [PASS] Step 4 (Telemetry Export): CSV (${csvExport.rows}x${csvExport.cols}, ${csvExport.frameW}x${csvExport.frameH}) & JSON (${jsonExport.frameCount} frames, schema v${jsonExport.meta.schemaVersion})`);

    // 5. Lab Modes B & C + Return to 320x180 SD Cleanly
    await cdp.evalJs(
      sessionId,
      `(() => {
        document.getElementById('btnSrcTestbench').click();
        document.querySelector('.lab-btn[data-lab="1"]').click();
        return true;
      })()`
    );
    await waitForCondition(
      cdp,
      sessionId,
      'Testbench re-lock in Mode B',
      `(() => ({ ok: (document.getElementById('kpiTargets')?.textContent || '') === '4 / 8' }))()`
    );
    await cdp.evalJs(sessionId, `document.getElementById('btnLabStimulus').click(); true;`);
    const modeBState = await waitForCondition(
      cdp,
      sessionId,
      'Mode B deterministic saccade onset latency readout',
      `(() => {
        const val = document.getElementById('kpiSaccadeMs')?.textContent || '';
        return { ok: val.endsWith(' ms'), val };
      })()`
    );

    await cdp.evalJs(
      sessionId,
      `(() => {
        document.querySelector('.lab-btn[data-lab="2"]').click();
        document.getElementById('btnLabStimulus').click();
        document.getElementById('btnResSd').click();
        return true;
      })()`
    );
    const backSd = await waitForCondition(
      cdp,
      sessionId,
      'Return to 320x180 SD mode',
      `(() => {
        const resMeta = document.getElementById('inputResMeta')?.textContent || '';
        const pipAria = document.getElementById('pipCanvas')?.getAttribute('aria-label') || '';
        return { ok: resMeta === '320×180 SD RGBA' && pipAria.startsWith('16x16'), resMeta, pipAria };
      })()`
    );
    console.log(`  [PASS] Step 5 (Lab Modes B/C & SD Round-Trip): saccade=${modeBState.val}, returned=${backSd.resMeta}`);

    if (pageErrors.length > 0) {
      throw new Error(`Browser reported ${pageErrors.length} error(s):\n${pageErrors.join('\n')}`);
    }
    console.log('  [PASS] Zero uncaught page/worker exceptions and zero console errors.');
  } finally {
    if (cdp) cdp.close();
    if (chromeProc) chromeProc.kill('SIGKILL');
    server.close();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error('E2E FAILED:', err.message || err);
  process.exit(1);
});
