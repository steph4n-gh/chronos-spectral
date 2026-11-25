# Security Policy

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `3.5.x` (`main`, `v3.5.0`) | Yes |
| `< 3.5` | No |

## Architecture & Threat Boundary

1. **Static Client-Side Optical Physics Instrument (`web/index.html`, `web/ui/`, `web/styles/`, `web/worker.js`, `web/chronos_spectral.wasm`):**
   - Runs 100% locally inside browser Web Workers over a fixed 448-page (`28.00 MiB` / `29,360,128 B`) `#![no_std]` WebAssembly linear arena (`0` host imports, `0` Cargo/npm runtime dependencies, `0` external CDN scripts or analytics).
   - User-selected MP4/WebM files (`Drag & Drop` or file picker, including Audit A/B clip slots) and webcam/microphone streams stay in browser memory (`0` bytes uploaded).
   - Do not expose sensitive local camera feeds or private footage on shared or untrusted screens.

2. **Native C ABI, CLI & Python Audit SDK (`native/capi/`, `native/cli/`, `native/python/`):**
   - Executes locally against heap-isolated `ChronosEngine` instances with explicit buffer-length checks and `0` network access unless the operator explicitly passes a remote stream URL (such as `rtsp://`) to `chronos` or `ffmpeg`.
   - Treat untrusted video or RTSP inputs according to your host `ffmpeg` security posture.

3. **Local Development & HLS Relay Server (`scripts/dev_server.mjs`):**
   - Designed **strictly for local loopback (`127.0.0.1`) use** alongside local FFmpeg during development or evaluation of Tier-3 HTTP Live Streaming relays.
   - Enforces web-root path-traversal confinement, same-origin loopback `Host`/`Origin`/`Sec-Fetch-Site` verification, DNS-pinned SSRF blocking across IPv4/IPv6 non-global ranges (loopback, private `RFC 1918` / `RFC 4193`, CGNAT `RFC 6598`, link-local, NAT64/6to4/Teredo) on every redirect and HLS manifest hop, 512 KiB playlist size caps, 3-hop redirect limits, FFmpeg `-format_whitelist` demuxer restrictions, and bounded concurrent relay process slots (`maxLiveProcs = 4`, returning HTTP `429` when full).
   - **Do not bind `scripts/dev_server.mjs` to `0.0.0.0` or expose it directly to the public internet.**

## Reporting a Vulnerability

If you discover a security issue in the static web app, WebAssembly kernel, build/verification tooling, or `scripts/dev_server.mjs`:

- Please report it confidentially via [GitHub Private Vulnerability Reporting](https://github.com/steph4n-gh/chronos-spectral/security/advisories/new).
- Please **do not** open a public GitHub issue for undisclosed security vulnerabilities.
- Include reproduction steps, affected commit/file paths, and impact. We aim to acknowledge reports within 72 hours.
