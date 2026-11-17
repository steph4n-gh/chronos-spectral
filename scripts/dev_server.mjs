#!/usr/bin/env node
/**
 * CHRONOS-SPECTRAL // Loopback HTTP 206 Static & Live fMP4 MSE Relay Server
 * - Serves `web/` with RFC 7233 HTTP 206 byte-range support (WebKit/Chromium MP4 + Wasm).
 * - Exposes `GET /api/live-presets` (health-probed public DOT HLS feeds + instant local live sims).
 * - Exposes `GET /api/live-stream` (`ffmpeg` -> fragmented MP4 `ftyp+moov+moof` for zero-dep MSE).
 * - Hardened strictly to loopback `127.0.0.1`, argv-only `spawn`, DNS SSRF block, & process cap.
 */

import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import dns from 'node:dns/promises';
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';
import path from 'node:path';
import { pipeline } from 'node:stream';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = path.resolve(__dirname, '..', 'web');

function resolveFfmpegBin() {
  if (process.env.FFMPEG_BIN && fs.existsSync(process.env.FFMPEG_BIN)) {
    return process.env.FFMPEG_BIN;
  }
  const candidates = ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/usr/bin/ffmpeg'];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return 'ffmpeg';
}

const FFMPEG_BIN = resolveFfmpegBin();

const MIMES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.mp4': 'video/mp4',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.css': 'text/css; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};

const PUBLIC_HLS_CANDIDATES = [
  {
    id: 'caltrans_d4_i80_ashby',
    label: 'Caltrans D4 · I-80 West of Ashby Ave, Berkeley — Overpass Occlusion (1080p HD Live)',
    category: 'public_hls',
    mode: 0,
    url: 'https://wzmedia.dot.ca.gov/D4/E80_JWO_ASHBY_Av.stream/playlist.m3u8',
  },
  {
    id: 'caltrans_d4_i280_6th',
    label: 'Caltrans D4 · I-280 at 6th St Offramp, San Francisco (Live)',
    category: 'public_hls',
    mode: 0,
    url: 'https://wzmedia.dot.ca.gov/D4/N280_at_6th.stream/playlist.m3u8',
  },
  {
    id: 'caltrans_d4_i880_sr238',
    label: 'Caltrans D4 · I-880 at SR-238 Center-Pole Freeway (1080p HD Live)',
    category: 'public_hls',
    mode: 0,
    url: 'https://wzmedia.dot.ca.gov/D4/S880_JNO_JCT_238.stream/playlist.m3u8',
  },
  {
    id: 'caltrans_d4_i680_i80',
    label: 'Caltrans D4 · I-680 South of I-80 Cordelia (Live)',
    category: 'public_hls',
    mode: 0,
    url: 'https://wzmedia.dot.ca.gov/D4/S680_JSO_JCT_80.stream/playlist.m3u8',
  },
  {
    id: 'caltrans_d4_sr4_lone_tree',
    label: 'Caltrans D4 · SR-4 at Lone Tree Way (Live)',
    category: 'public_hls',
    mode: 0,
    url: 'https://wzmedia.dot.ca.gov/D4/W4_at_Lone_Tree_Way.stream/playlist.m3u8',
  },
];

function parseIpv6Hextets(ip) {
  if (typeof ip !== 'string' || !ip || ip.includes('%')) return null;
  const dbl = ip.indexOf('::');
  if (dbl >= 0 && ip.indexOf('::', dbl + 2) >= 0) return null;
  const rawLeft = dbl >= 0 ? ip.slice(0, dbl) : ip;
  const rawRight = dbl >= 0 ? ip.slice(dbl + 2) : '';
  const parsePart = (partStr) => {
    if (!partStr) return [];
    const segs = partStr.split(':');
    const out = [];
    for (let i = 0; i < segs.length; i++) {
      const s = segs[i];
      if (!s) return null;
      if (s.includes('.')) {
        if (i !== segs.length - 1) return null;
        const octs = s.split('.');
        if (octs.length !== 4) return null;
        const nums = octs.map((o) => (/^\d{1,3}$/.test(o) ? Number(o) : -1));
        if (nums.some((n) => n < 0 || n > 255)) return null;
        out.push((nums[0] << 8) | nums[1], (nums[2] << 8) | nums[3]);
      } else {
        if (!/^[0-9a-f]{1,4}$/i.test(s)) return null;
        out.push(parseInt(s, 16));
      }
    }
    return out;
  };
  const left = parsePart(rawLeft);
  const right = parsePart(rawRight);
  if (!left || !right) return null;
  const total = left.length + right.length;
  if (dbl < 0 ? total !== 8 : total >= 8) return null;
  const zeroFill = dbl < 0 ? 0 : 8 - total;
  const out = new Uint16Array(8);
  let idx = 0;
  for (const v of left) out[idx++] = v;
  idx += zeroFill;
  for (const v of right) out[idx++] = v;
  return out;
}

function isPrivateOrLoopbackIp(ip) {
  const ver = net.isIP(ip);
  if (ver === 4) {
    const parts = ip.split('.').map((n) => parseInt(n, 10));
    const [a, b, c] = parts;
    if (a === 0 || a === 10 || a === 127) return true;
    if (a === 169 && b === 254) return true;
    if (a === 172 && b >= 16 && b <= 31) return true;
    if (a === 192 && b === 168) return true;
    if (a === 192 && b === 0 && (c === 0 || c === 2)) return true;
    if (a === 198 && (b === 18 || b === 19)) return true;
    if (a === 198 && b === 51 && c === 100) return true;
    if (a === 203 && b === 0 && c === 113) return true;
    if (a === 100 && b >= 64 && b <= 127) return true;
    if (a >= 224) return true;
    return false;
  }
  if (ver === 6) {
    const h = parseIpv6Hextets(ip);
    if (!h) return true;
    if ((h[0] & 0xe000) !== 0x2000) return true;
    if (h[0] === 0x2001 && (h[1] === 0x0db8 || h[1] === 0x0000)) return true;
    if (h[0] === 0x2001 && h[1] === 0x0002 && h[2] === 0x0000) return true;
    if (h[0] === 0x2002) return true;
    return false;
  }
  return true;
}

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]', '::1']);

function isLoopbackHostHeader(rawHost) {
  if (typeof rawHost !== 'string' || rawHost.length === 0 || rawHost.length > 256) return false;
  const trimmed = rawHost.trim().toLowerCase();
  if (trimmed.startsWith('[')) {
    const closeIdx = trimmed.indexOf(']');
    if (closeIdx < 0) return false;
    return LOOPBACK_HOSTS.has(trimmed.slice(0, closeIdx + 1));
  }
  const hostOnly = trimmed.split(':')[0];
  return LOOPBACK_HOSTS.has(hostOnly);
}

function verifySameOriginLoopbackApi(req) {
  if (!isLoopbackHostHeader(req.headers.host || '')) {
    return 'Blocked non-loopback Host header';
  }
  const secFetchSite = String(req.headers['sec-fetch-site'] || '').toLowerCase();
  if (secFetchSite && secFetchSite !== 'same-origin' && secFetchSite !== 'none') {
    return `Blocked cross-origin request (Sec-Fetch-Site: ${secFetchSite})`;
  }
  const origin = req.headers.origin;
  if (typeof origin === 'string' && origin.length > 0) {
    try {
      const parsedOrigin = new URL(origin);
      if (!isLoopbackHostHeader(parsedOrigin.host)) {
        return 'Blocked cross-origin Origin header';
      }
    } catch {
      return 'Malformed Origin header';
    }
  }
  return null;
}

const VALIDATED_DNS_CACHE = new Map();
const DNS_PIN_TTL_MS = 30_000;

async function resolveAndPinPublicUrl(rawUrl, allowHostPorts = null) {
  if (typeof rawUrl !== 'string' || rawUrl.length < 10 || rawUrl.length > 2048) {
    throw new Error('Invalid stream URL length');
  }
  if (rawUrl.startsWith('-') || /[\r\n\0]/.test(rawUrl)) {
    throw new Error('Rejected unsafe characters in stream URL');
  }
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error('Malformed stream URL');
  }
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
    throw new Error(`Unsupported protocol ${parsed.protocol}`);
  }
  if (parsed.username || parsed.password) {
    throw new Error('Credentials in stream URL are not permitted');
  }
  if (allowHostPorts && allowHostPorts.has(parsed.host.toLowerCase())) {
    return { parsed, pinnedIp: '127.0.0.1', family: 4 };
  }
  const host = parsed.hostname.toLowerCase();
  if (!host || host === 'localhost' || host.endsWith('.local') || host.endsWith('.internal')) {
    throw new Error('Loopback and internal hostnames are blocked');
  }
  const bareHost = host.startsWith('[') && host.endsWith(']') ? host.slice(1, -1) : host;
  if (net.isIP(bareHost)) {
    if (isPrivateOrLoopbackIp(bareHost)) {
      throw new Error('Private or loopback IP stream targets are blocked');
    }
    return { parsed, pinnedIp: bareHost, family: net.isIPv6(bareHost) ? 6 : 4 };
  }
  const now = Date.now();
  const cached = VALIDATED_DNS_CACHE.get(bareHost);
  if (cached && now - cached.ts < DNS_PIN_TTL_MS) {
    return { parsed, pinnedIp: cached.pinnedIp, family: cached.family };
  }
  const records = await dns.lookup(bareHost, { all: true, verbatim: true });
  if (!records || records.length === 0) {
    throw new Error('DNS resolution returned no records');
  }
  for (const rec of records) {
    if (isPrivateOrLoopbackIp(rec.address)) {
      throw new Error('Stream hostname resolved to a blocked private/loopback IP');
    }
  }
  const v4 = records.find((r) => r.family === 4);
  const chosen = v4 || records[0];
  const family = chosen.family || (net.isIPv6(chosen.address) ? 6 : 4);
  VALIDATED_DNS_CACHE.set(bareHost, { pinnedIp: chosen.address, family, ts: now });
  return {
    parsed,
    pinnedIp: chosen.address,
    family,
  };
}

async function openPinnedHttpStream(rawUrl, hopCount = 0, allowHostPorts = null) {
  if (hopCount > 3) {
    throw new Error('Too many stream redirects (max 3)');
  }
  const { parsed, pinnedIp, family } = await resolveAndPinPublicUrl(rawUrl, allowHostPorts);
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Unsupported HTTP proxy protocol ${parsed.protocol}`);
  }
  const transport = parsed.protocol === 'https:' ? https : http;
  const defaultPort = parsed.protocol === 'https:' ? 443 : 80;

  const { upReq, upRes } = await new Promise((resolve, reject) => {
    const req = transport.request(
      {
        protocol: parsed.protocol,
        hostname: parsed.hostname,
        port: parsed.port ? Number(parsed.port) : defaultPort,
        path: (parsed.pathname || '/') + (parsed.search || ''),
        method: 'GET',
        agent: false,
        timeout: 8000,
        headers: {
          Accept: 'application/vnd.apple.mpegurl,video/*,*/*',
          'User-Agent': 'ChronosSpectralRelay/1.0',
          Connection: 'close',
        },
        lookup: (_hostname, opts, cb) => {
          if (opts && opts.all) {
            cb(null, [{ address: pinnedIp, family }]);
          } else {
            cb(null, pinnedIp, family);
          }
        },
      },
      (res) => resolve({ upReq: req, upRes: res })
    );
    req.on('timeout', () => {
      req.destroy(new Error('Upstream connection timed out'));
    });
    req.on('error', (err) => reject(err));
    req.end();
  });

  const status = upRes.statusCode || 502;
  if (status >= 300 && status < 400) {
    const loc = upRes.headers.location;
    upRes.resume();
    upReq.destroy();
    if (!loc) {
      throw new Error(`Redirect (${status}) missing Location header`);
    }
    const nextUrl = new URL(loc, parsed).toString();
    return openPinnedHttpStream(nextUrl, hopCount + 1, allowHostPorts);
  }

  return { parsed, upReq, upRes };
}

const HLS_TAG_URI_RE = /\bURI\s*=\s*(?:"([^"]*)"|'([^']*)'|([^,\s"']+))/gi;
const HLS_RAW_SCHEME_RE = /\b(?:https?|rtsp|rtsps|rtmp|file|tcp|udp|tls|data|concat|subfile|crypto|ftp|gopher|ws|wss):/i;

function peekHeaderBytes(stream, minBytes = 1024, timeoutMs = 6000) {
  if (stream.readableEnded || (stream.complete && stream.readableLength === 0)) {
    return Promise.resolve({ firstChunk: Buffer.alloc(0), ended: true });
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    const chunks = [];
    let total = 0;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        cleanup();
        reject(new Error('Upstream body read timed out'));
      }
    }, timeoutMs);
    const finish = (ended) => {
      if (settled) return;
      settled = true;
      if (!ended) stream.pause();
      cleanup();
      resolve({
        firstChunk: total > 0 ? Buffer.concat(chunks, total) : Buffer.alloc(0),
        ended: Boolean(ended || stream.readableEnded),
      });
    };
    const onData = (chunk) => {
      if (settled) return;
      const buf = Buffer.from(chunk);
      chunks.push(buf);
      total += buf.length;
      if (total >= minBytes) {
        finish(false);
      }
    };
    const onEnd = () => finish(true);
    const onErr = (err) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    };
    function cleanup() {
      clearTimeout(timer);
      stream.off('data', onData);
      stream.off('end', onEnd);
      stream.off('close', onEnd);
      stream.off('error', onErr);
    }
    stream.on('data', onData);
    stream.once('end', onEnd);
    stream.once('close', onEnd);
    stream.once('error', onErr);
  });
}

function bufferStartsWithExtM3u(buf) {
  if (!buf || buf.length < 7) return false;
  let off = 0;
  if (buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf) off = 3;
  const head = buf.subarray(off, Math.min(buf.length, off + 1024)).toString('utf8').trimStart();
  return head.startsWith('#EXTM3U');
}

function isBlockedManifestPayload(ctype, firstChunk) {
  const lcType = String(ctype || '').toLowerCase();
  if (
    lcType.includes('dash') ||
    lcType.includes('sdp') ||
    lcType.includes('smil') ||
    lcType.includes('smoothstream')
  ) {
    return true;
  }
  if (!firstChunk || firstChunk.length === 0) return false;
  const head = firstChunk
    .subarray(0, Math.min(firstChunk.length, 1024))
    .toString('utf8')
    .replace(/^\uFEFF/, '');
  return (
    /<(?:[\w-]+:)?(?:MPD|SmoothStreamingMedia|smil|asx)\b/i.test(head) ||
    /^\s*ffconcat\s+version\b/im.test(head) ||
    /^\s*v=0\r?\n/m.test(head)
  );
}

async function readBoundedStreamBuffer(stream, firstChunk, maxBytes = 524288) {
  const chunks = firstChunk.length > 0 ? [firstChunk] : [];
  let total = firstChunk.length;
  if (total > maxBytes) throw new Error('HLS playlist exceeded 512 KB cap');
  if (stream.readableEnded || (stream.complete && stream.readableLength === 0)) {
    return Buffer.concat(chunks, total);
  }
  return new Promise((resolve, reject) => {
    let done = false;
    function cleanup() {
      stream.off('data', onData);
      stream.off('end', onEnd);
      stream.off('close', onClose);
      stream.off('error', onErr);
    }
    function onData(chunk) {
      if (done) return;
      total += chunk.length;
      if (total > maxBytes) {
        done = true;
        cleanup();
        const capErr = new Error('HLS playlist exceeded 512 KB cap');
        stream.destroy(capErr);
        reject(capErr);
        return;
      }
      chunks.push(Buffer.from(chunk));
    }
    function onEnd() {
      if (done) return;
      done = true;
      cleanup();
      resolve(Buffer.concat(chunks, total));
    }
    function onClose() {
      if (done) return;
      done = true;
      cleanup();
      if (stream.readableEnded) {
        resolve(Buffer.concat(chunks, total));
      } else {
        reject(new Error('Upstream HLS playlist closed prematurely'));
      }
    }
    function onErr(err) {
      if (done) return;
      done = true;
      cleanup();
      reject(err);
    }
    stream.on('data', onData);
    stream.once('end', onEnd);
    stream.once('close', onClose);
    stream.once('error', onErr);
    stream.resume();
    if (stream.readableEnded) onEnd();
  });
}

async function validatePublicStreamUrl(rawUrl, allowHostPorts = null, signal = null) {
  await resolveAndPinPublicUrl(rawUrl, allowHostPorts);
  if (signal?.aborted) {
    throw new Error('Client disconnected during stream preflight');
  }
  const { parsed, upReq, upRes } = await openPinnedHttpStream(rawUrl, 0, allowHostPorts);
  const onAbort = () => {
    try { upRes.destroy(); } catch {}
    try { upReq.destroy(); } catch {}
  };
  if (signal) {
    if (signal.aborted) onAbort();
    else signal.addEventListener('abort', onAbort, { once: true });
  }
  try {
    const status = upRes.statusCode || 502;
    if (status < 200 || status >= 300) {
      throw new Error(`Upstream returned HTTP ${status}`);
    }
    const { firstChunk, ended } = await peekHeaderBytes(upRes);
    if (signal?.aborted) {
      throw new Error('Client disconnected during stream preflight');
    }
    const ctype = String(upRes.headers['content-type'] || '').toLowerCase();
    const isPlaylist =
      parsed.pathname.toLowerCase().endsWith('.m3u8') ||
      ctype.includes('mpegurl') ||
      bufferStartsWithExtM3u(firstChunk);
    if (!isPlaylist && isBlockedManifestPayload(ctype, firstChunk)) {
      throw new Error('Unsupported manifest format (only HLS .m3u8 and direct media containers are permitted)');
    }
    if (isPlaylist) {
      const fullBuf = ended ? firstChunk : await readBoundedStreamBuffer(upRes, firstChunk);
      if (fullBuf.includes(0)) {
        throw new Error('Blocked NUL byte in HLS playlist');
      }
      const text = fullBuf.toString('utf8');
      const checkedHosts = new Set();
      for (const rawLine of text.split(/\r\n|\r|\n/)) {
        const line = rawLine.trim();
        if (!line) continue;
        const refs = [];
        if (line.startsWith('#')) {
          if (!/^#EXT/i.test(line)) continue;
          HLS_TAG_URI_RE.lastIndex = 0;
          let m;
          while ((m = HLS_TAG_URI_RE.exec(line)) !== null) {
            refs.push(m[1] ?? m[2] ?? m[3] ?? '');
          }
          const strippedCheck = line.replace(HLS_TAG_URI_RE, '');
          if (/\bURI\s*=/i.test(strippedCheck) || HLS_RAW_SCHEME_RE.test(strippedCheck)) {
            throw new Error('Blocked unparseable URI attribute in HLS tag');
          }
        } else {
          refs.push(line);
        }
        for (const ref of refs) {
          if (!ref || ref.includes('\\')) {
            throw new Error('Blocked malformed URI reference in HLS manifest');
          }
          const resolvedRef = new URL(ref, parsed);
          if (resolvedRef.protocol !== 'http:' && resolvedRef.protocol !== 'https:') {
            throw new Error(`Blocked non-HTTP URI in HLS manifest (${resolvedRef.protocol})`);
          }
          const hostKey = resolvedRef.host.toLowerCase();
          if (!checkedHosts.has(hostKey)) {
            await resolveAndPinPublicUrl(resolvedRef.toString(), allowHostPorts);
            checkedHosts.add(hostKey);
          }
        }
      }
    }
    return { url: parsed.toString(), isPlaylist };
  } finally {
    if (signal) signal.removeEventListener('abort', onAbort);
    upRes.resume();
    upReq.destroy();
  }
}

async function probeHlsCandidate(cand) {
  const t0 = Date.now();
  try {
    const res = await fetch(cand.url, {
      method: 'GET',
      headers: { Accept: 'application/vnd.apple.mpegurl,text/plain,*/*' },
      signal: AbortSignal.timeout(2200),
    });
    if (!res.ok) {
      return { ...cand, online: false, latencyMs: Date.now() - t0 };
    }
    const text = await res.text();
    const online = text.trimStart().startsWith('#EXTM3U');
    const corsHeader = res.headers.get('access-control-allow-origin') || '';
    return {
      ...cand,
      online,
      corsOpen: corsHeader === '*',
      latencyMs: Date.now() - t0,
    };
  } catch {
    return { ...cand, online: false, corsOpen: false, latencyMs: Date.now() - t0 };
  }
}

export function startChronosServer({
  rootDir = DEFAULT_ROOT,
  port = Number(process.env.PORT || 8080),
  host = process.env.CHRONOS_BIND_HOST || '127.0.0.1',
  maxLiveProcs = 4,
  testAllowHostPorts = null,
} = {}) {
  const root = path.resolve(rootDir);
  const samplesDir = path.join(root, 'samples');
  const activeProcs = new Set();
  const relaySecret = crypto.randomBytes(24).toString('hex');
  const relayPrefix = `/__chronos_upstream_proxy/${relaySecret}/`;
  const allowHostSet = testAllowHostPorts
    ? new Set(Array.from(testAllowHostPorts, (s) => String(s).toLowerCase()))
    : null;
  let boundPort = port;
  let cachedProbe = { ts: 0, items: null };

  function buildLocalProxyUrl(targetUrl, fallbackExt = '.m3u8') {
    let safeExt = fallbackExt;
    try {
      const ext = path.posix.extname(new URL(targetUrl).pathname).toLowerCase();
      if (/^\.(m3u8|ts|m4s|mp4|aac|mp3|key|cmfv|fmp4)$/.test(ext)) {
        safeExt = ext;
      }
    } catch {}
    return `http://127.0.0.1:${boundPort}${relayPrefix}media${safeExt}?u=${encodeURIComponent(targetUrl)}`;
  }

  function rewriteHlsPlaylistBody(text, baseParsed) {
    const proxyOriginPrefix = `http://127.0.0.1:${boundPort}${relayPrefix}`;
    const lines = text.split(/\r\n|\r|\n/);
    const out = [];
    let expectVariant = false;
    for (const rawLine of lines) {
      const trimmed = rawLine.trim();
      if (!trimmed) {
        out.push('');
        continue;
      }
      if (trimmed.startsWith('#')) {
        if (!/^#EXT/i.test(trimmed)) {
          continue;
        }
        if (/^#EXT-X-STREAM-INF\b/i.test(trimmed)) {
          expectVariant = true;
        }
        const rewrittenTag = trimmed.replace(HLS_TAG_URI_RE, (_full, dqRef, sqRef, uqRef) => {
          const ref = dqRef ?? sqRef ?? uqRef ?? '';
          if (!ref || ref.includes('\\')) {
            throw new Error('Blocked malformed URI attribute in HLS tag');
          }
          const resolved = new URL(ref, baseParsed);
          if (resolved.protocol !== 'http:' && resolved.protocol !== 'https:') {
            throw new Error(`Blocked non-HTTP URI in HLS tag (${resolved.protocol})`);
          }
          return `URI="${buildLocalProxyUrl(resolved.toString(), '.key')}"`;
        });
        const residualTag = rewrittenTag.replace(
          new RegExp(`URI="http://127\\.0\\.0\\.1:${boundPort}${relayPrefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}[^"]*"`, 'g'),
          ''
        );
        if (/\bURI\s*=/i.test(residualTag) || HLS_RAW_SCHEME_RE.test(residualTag)) {
          throw new Error('Blocked unrewritten URI scheme in HLS tag');
        }
        out.push(rewrittenTag);
      } else {
        if (trimmed.includes('\\')) {
          throw new Error('Blocked malformed URI line in HLS playlist');
        }
        const resolved = new URL(trimmed, baseParsed);
        if (resolved.protocol !== 'http:' && resolved.protocol !== 'https:') {
          throw new Error(`Blocked non-HTTP URI in HLS playlist (${resolved.protocol})`);
        }
        const guessExt = expectVariant ? '.m3u8' : '.ts';
        expectVariant = false;
        const proxiedLine = buildLocalProxyUrl(resolved.toString(), guessExt);
        if (!proxiedLine.startsWith(proxyOriginPrefix)) {
          throw new Error('Blocked unproxied HLS segment line');
        }
        out.push(proxiedLine);
      }
    }
    return out.join('\n');
  }

  function getLocalSimPresets() {
    const trafficMp4 = path.join(samplesDir, 'roundabout_occlusion.mp4');
    const jugglingMp4 = path.join(samplesDir, 'juggling_ballistic.mp4');
    const billiardsMp4 = path.join(samplesDir, 'carom_billiards_impulses.mp4');
    const out = [];
    if (fs.existsSync(trafficMp4)) {
      out.push({
        id: 'sim_live_traffic',
        label: 'Live Relay Sim · 24/7 Highway Overpass Loop (30 fps fMP4)',
        category: 'local_sim',
        mode: 0,
        online: true,
        corsOpen: true,
        latencyMs: 1,
      });
    }
    if (fs.existsSync(trafficMp4) && fs.existsSync(jugglingMp4)) {
      out.push({
        id: 'sim_live_cut_demo',
        label: 'Live Relay Sim · Multi-Cam Highway -> Ballistic Cut (Auto-Reset Proof)',
        category: 'local_sim',
        mode: 0,
        online: true,
        corsOpen: true,
        latencyMs: 1,
      });
    }
    if (fs.existsSync(billiardsMp4)) {
      out.push({
        id: 'sim_live_impulses',
        label: 'Live Relay Sim · Continuous 3-Cushion Billiards Feed (Mode C)',
        category: 'local_sim',
        mode: 2,
        online: true,
        corsOpen: true,
        latencyMs: 1,
      });
    }
    return out;
  }

  async function buildPresetsPayload(forceRefresh = false) {
    const skipRemote = process.env.CHRONOS_OFFLINE_PRESETS === '1';
    const now = Date.now();
    if (!skipRemote && forceRefresh) {
      const probed = await Promise.all(PUBLIC_HLS_CANDIDATES.map(probeHlsCandidate));
      cachedProbe = { ts: now, items: probed };
    }
    const defaultOnlineRemote = PUBLIC_HLS_CANDIDATES.map((c) => ({
      ...c,
      online: true,
      corsOpen: true,
      latencyMs: 120,
    }));
    const remoteItems = skipRemote ? [] : (cachedProbe.items && now - cachedProbe.ts < 300_000 ? cachedProbe.items : defaultOnlineRemote);
    return {
      ok: true,
      probedAtIso: new Date().toISOString(),
      presets: [...remoteItems.filter((p) => p.online), ...getLocalSimPresets(), ...remoteItems.filter((p) => !p.online)],
    };
  }

  function buildFfmpegArgsForRequest(presetId, customValidatedUrl, customIsPlaylist = false) {
    const commonEnc = [
      '-an',
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-tune', 'zerolatency',
      '-profile:v', 'baseline',
      '-level', '3.1',
      '-crf', '23',
      '-maxrate', '2200k',
      '-bufsize', '4400k',
      '-pix_fmt', 'yuv420p',
      '-g', '15',
      '-keyint_min', '15',
      '-sc_threshold', '0',
      '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
      '-f', 'mp4',
      'pipe:1',
    ];

    if (presetId === 'sim_live_traffic') {
      const src = path.join(samplesDir, 'roundabout_occlusion.mp4');
      if (!fs.existsSync(src)) throw new Error('Missing sample roundabout_occlusion.mp4');
      return [
        '-v', 'error',
        '-stream_loop', '-1',
        '-i', src,
        '-vf', 'scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,realtime',
        ...commonEnc,
      ];
    }

    if (presetId === 'sim_live_impulses') {
      const src = path.join(samplesDir, 'carom_billiards_impulses.mp4');
      if (!fs.existsSync(src)) throw new Error('Missing sample carom_billiards_impulses.mp4');
      return [
        '-v', 'error',
        '-stream_loop', '-1',
        '-i', src,
        '-vf', 'scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,realtime',
        ...commonEnc,
      ];
    }

    if (presetId === 'sim_live_cut_demo') {
      const srcA = path.join(samplesDir, 'roundabout_occlusion.mp4');
      const srcB = path.join(samplesDir, 'juggling_ballistic.mp4');
      if (!fs.existsSync(srcA) || !fs.existsSync(srcB)) {
        throw new Error('Missing sample MP4s for sim_live_cut_demo');
      }
      return [
        '-v', 'error',
        '-i', srcA,
        '-i', srcB,
        '-filter_complex',
        '[0:v]trim=0:1.6,setpts=PTS-STARTPTS,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v0];' +
        '[1:v]trim=0:4.0,setpts=PTS-STARTPTS,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v1];' +
        '[v0][v1]concat=n=2:v=1:a=0,realtime[vout]',
        '-map', '[vout]',
        ...commonEnc,
      ];
    }

    let remoteUrl = customValidatedUrl;
    if (!remoteUrl && presetId) {
      const found = PUBLIC_HLS_CANDIDATES.find((c) => c.id === presetId);
      if (found) remoteUrl = found.url;
    }
    if (!remoteUrl) {
      throw new Error('Unknown live preset or missing stream URL');
    }

    const isHls = customIsPlaylist || remoteUrl.includes('.m3u8');
    const ffmpegInputUrl = buildLocalProxyUrl(remoteUrl, isHls ? '.m3u8' : '.mp4');
    return [
      '-v', 'error',
      '-nostdin',
      ...(isHls ? [] : ['-fflags', 'nobuffer']),
      '-probesize', '500000',
      '-analyzeduration', '1000000',
      '-protocol_whitelist', 'http,tcp',
      '-format_whitelist',
      isHls
        ? 'hls,mpegts,mov,mp4,m4a,3gp,3g2,mj2,aac,mp3'
        : 'mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,mpegts,flv,avi',
      '-max_redirects', '0',
      '-rw_timeout', '15000000',
      ...(isHls ? ['-live_start_index', '-2'] : []),
      '-i', ffmpegInputUrl,
      '-vf', 'scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,setpts=PTS-STARTPTS',
      ...commonEnc,
    ];
  }

  const rootPrefix = root.endsWith(path.sep) ? root : root + path.sep;

  const server = http.createServer(async (req, res) => {
    try {
      const reqUrl = new URL(req.url || '/', `http://${host}`);

      if (reqUrl.pathname.startsWith('/__chronos_upstream_proxy/')) {
        const remoteAddr = req.socket?.remoteAddress || '';
        const isLocalPeer =
          remoteAddr === '127.0.0.1' ||
          remoteAddr === '::1' ||
          remoteAddr === '::ffff:127.0.0.1';
        if (!isLocalPeer || req.method !== 'GET' || !reqUrl.pathname.startsWith(relayPrefix)) {
          res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
          res.end('Forbidden');
          return;
        }
        const targetRaw = reqUrl.searchParams.get('u') || '';
        let activeUpReq = null;
        let activeUpRes = null;
        try {
          const opened = await openPinnedHttpStream(targetRaw, 0, allowHostSet);
          activeUpReq = opened.upReq;
          activeUpRes = opened.upRes;
          const abortUpstream = () => {
            try { activeUpRes?.destroy(); } catch {}
            try { activeUpReq?.destroy(); } catch {}
          };
          req.on('close', abortUpstream);
          res.on('close', abortUpstream);

          const status = activeUpRes.statusCode || 502;
          if (status < 200 || status >= 300) {
            activeUpRes.resume();
            activeUpReq.destroy();
            res.writeHead(status >= 400 && status < 600 ? status : 502, { 'Content-Type': 'text/plain; charset=utf-8' });
            res.end(`Upstream HTTP ${status}`);
            return;
          }

          const { firstChunk, ended } = await peekHeaderBytes(activeUpRes);
          const ctype = String(activeUpRes.headers['content-type'] || '').toLowerCase();
          const isPlaylist =
            opened.parsed.pathname.toLowerCase().endsWith('.m3u8') ||
            ctype.includes('mpegurl') ||
            bufferStartsWithExtM3u(firstChunk);

          if (isPlaylist) {
            const fullBuf = ended ? firstChunk : await readBoundedStreamBuffer(activeUpRes, firstChunk);
            if (fullBuf.includes(0)) {
              throw new Error('Blocked NUL byte in HLS playlist');
            }
            const rewritten = rewriteHlsPlaylistBody(fullBuf.toString('utf8'), opened.parsed);
            const outBuf = Buffer.from(rewritten, 'utf8');
            res.writeHead(200, {
              'Content-Type': 'application/vnd.apple.mpegurl; charset=utf-8',
              'Content-Length': outBuf.length,
              'Cache-Control': 'no-store',
            });
            res.end(outBuf);
            return;
          }

          if (isBlockedManifestPayload(ctype, firstChunk)) {
            throw new Error('Unsupported manifest format on upstream hop');
          }

          res.writeHead(200, {
            'Content-Type': activeUpRes.headers['content-type'] || 'application/octet-stream',
            'Cache-Control': 'no-store',
          });
          if (firstChunk.length > 0) {
            res.write(firstChunk);
          }
          if (ended || activeUpRes.readableEnded || (activeUpRes.complete && activeUpRes.readableLength === 0)) {
            res.end();
          } else {
            pipeline(activeUpRes, res, () => {});
          }
        } catch (proxyErr) {
          try { activeUpRes?.destroy(); } catch {}
          try { activeUpReq?.destroy(); } catch {}
          if (!res.headersSent) {
            res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
            res.end(`Blocked upstream hop: ${proxyErr?.message || 'forbidden'}`);
          } else if (!res.writableEnded) {
            res.end();
          }
        }
        return;
      }

      if (reqUrl.pathname.startsWith('/api/')) {
        const originErr = verifySameOriginLoopbackApi(req);
        if (originErr) {
          res.writeHead(403, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
          res.end(JSON.stringify({ ok: false, error: originErr }));
          return;
        }
      }

      if (reqUrl.pathname === '/api/live-presets' && req.method === 'GET') {
        const force = reqUrl.searchParams.get('refresh') === '1';
        const payload = await buildPresetsPayload(force);
        const body = JSON.stringify(payload);
        res.writeHead(200, {
          'Content-Type': 'application/json; charset=utf-8',
          'Cache-Control': 'no-store',
        });
        res.end(body);
        return;
      }

      if (reqUrl.pathname === '/api/live-stream' && req.method === 'GET') {
        if (activeProcs.size >= maxLiveProcs) {
          res.writeHead(429, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({ ok: false, error: 'Maximum concurrent live FFmpeg relays reached' }));
          return;
        }
        let clientGone = Boolean(req.aborted || req.destroyed || res.destroyed || res.writableEnded);
        let proc = null;
        let cleaned = false;
        const abortCtrl = new AbortController();
        const cleanup = () => {
          clientGone = true;
          abortCtrl.abort();
          if (cleaned) return;
          cleaned = true;
          if (proc) {
            activeProcs.delete(proc);
            try {
              proc.kill('SIGKILL');
            } catch {}
          }
        };
        req.once('close', cleanup);
        res.once('close', cleanup);
        res.once('error', cleanup);

        const presetId = (reqUrl.searchParams.get('preset') || '').trim();
        const rawCustomUrl = (reqUrl.searchParams.get('url') || '').trim();
        let validatedUrl = '';
        let validatedIsPlaylist = false;
        if (!presetId && rawCustomUrl) {
          const validated = await validatePublicStreamUrl(rawCustomUrl, allowHostSet, abortCtrl.signal);
          validatedUrl = validated.url;
          validatedIsPlaylist = validated.isPlaylist;
        }
        if (clientGone || req.aborted || req.destroyed || res.destroyed || res.writableEnded) {
          cleanup();
          return;
        }
        if (activeProcs.size >= maxLiveProcs) {
          res.writeHead(429, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({ ok: false, error: 'Maximum concurrent live FFmpeg relays reached' }));
          return;
        }
        const ffArgs = buildFfmpegArgsForRequest(presetId, validatedUrl, validatedIsPlaylist);
        proc = spawn(FFMPEG_BIN, ffArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
        activeProcs.add(proc);
        if (clientGone || req.aborted || req.destroyed || res.destroyed || res.writableEnded) {
          cleanup();
          return;
        }

        let headersSent = false;
        let errTail = '';

        proc.stderr.on('data', (chunk) => {
          if (errTail.length < 512) errTail += chunk.toString('utf8');
        });

        proc.on('error', (err) => {
          cleanup();
          if (!headersSent && !res.writableEnded) {
            res.writeHead(502, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(JSON.stringify({ ok: false, error: `Failed to spawn ffmpeg: ${err.message}` }));
          } else if (!res.writableEnded) {
            res.end();
          }
        });

        proc.stdout.on('data', (chunk) => {
          if (!headersSent) {
            headersSent = true;
            res.writeHead(200, {
              'Content-Type': 'video/mp4',
              'Cache-Control': 'no-store, no-transform',
              'X-Content-Type-Options': 'nosniff',
              Connection: 'keep-alive',
            });
          }
          if (!res.writableEnded) {
            const ok = res.write(chunk);
            if (!ok) {
              proc.stdout.pause();
              res.once('drain', () => {
                if (!cleaned) proc.stdout.resume();
              });
            }
          }
        });

        proc.on('close', (code) => {
          activeProcs.delete(proc);
          if (!headersSent && !res.writableEnded) {
            res.writeHead(502, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(JSON.stringify({ ok: false, error: errTail.trim() || `ffmpeg exited (${code})` }));
          } else if (!res.writableEnded) {
            res.end();
          }
        });
        return;
      }

      const rel = decodeURIComponent(reqUrl.pathname === '/' ? '/index.html' : reqUrl.pathname);
      const filePath = path.normalize(path.join(root, rel));
      if (
        (filePath !== root && !filePath.startsWith(rootPrefix)) ||
        !fs.existsSync(filePath) ||
        fs.statSync(filePath).isDirectory()
      ) {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Not found');
        return;
      }
      const stat = fs.statSync(filePath);
      const type = MIMES[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
      const range = req.headers.range;
      if (range) {
        const m = /^bytes=(\d*)-(\d*)$/.exec(range);
        if (!m || (!m[1] && !m[2])) {
          res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
          res.end();
          return;
        }
        let start;
        let end;
        if (!m[1]) {
          const suffixLen = parseInt(m[2], 10);
          if (!Number.isFinite(suffixLen) || suffixLen <= 0 || stat.size === 0) {
            res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
            res.end();
            return;
          }
          start = Math.max(0, stat.size - suffixLen);
          end = stat.size - 1;
        } else {
          start = parseInt(m[1], 10);
          end = m[2] ? Math.min(stat.size - 1, parseInt(m[2], 10)) : stat.size - 1;
        }
        if (start >= stat.size || start > end) {
          res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
          res.end();
          return;
        }
        res.writeHead(206, {
          'Content-Type': type,
          'Content-Length': end - start + 1,
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
          'Cache-Control': 'no-store',
        });
        pipeline(fs.createReadStream(filePath, { start, end }), res, () => {});
        return;
      }
      res.writeHead(200, {
        'Content-Type': type,
        'Content-Length': stat.size,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-store',
      });
      pipeline(fs.createReadStream(filePath), res, () => {});
    } catch (err) {
      if (!res.headersSent) {
        res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ ok: false, error: err?.message || 'Bad request' }));
      } else if (!res.writableEnded) {
        res.end();
      }
    }
  });

  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => {
      const addr = server.address();
      boundPort = typeof addr === 'object' && addr ? addr.port : port;
      resolve({
        server,
        port: boundPort,
        host,
        close: () =>
          new Promise((resClose) => {
            for (const proc of activeProcs) {
              try {
                proc.kill('SIGKILL');
              } catch {}
            }
            activeProcs.clear();
            server.close(() => resClose());
          }),
      });
    });
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const port = Number(process.env.PORT || 8080);
  const host = process.env.CHRONOS_BIND_HOST || '127.0.0.1';
  startChronosServer({ port, host }).then(({ port: boundPort, host: boundHost }) => {
    console.log(`[CHRONOS-SERVER] Listening on http://${boundHost}:${boundPort} (HTTP 206 + Live fMP4 MSE Relay, ffmpeg=${FFMPEG_BIN})`);
  }).catch((err) => {
    if (err && err.code === 'EADDRINUSE') {
      console.error(`[CHRONOS-SERVER] Port ${port} is already occupied by another process. Run "./build.sh --serve" to replace a stale server, or set PORT=8081.`);
    } else {
      console.error(`[CHRONOS-SERVER] Failed to start: ${err.message}`);
    }
    process.exit(1);
  });
}
