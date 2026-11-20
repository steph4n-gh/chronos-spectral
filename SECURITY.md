# Security Policy

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `3.1.x` (`main`, `v3.1.1`) | Yes |
| `< 3.1` | No |

## Architecture & Threat Boundary

1. **Static Client-Side Showcase & Command Deck (`web/index.html`, `web/worker.js`, `web/chronos_spectral.wasm`):**
   - Runs 100% locally inside a browser Web Worker over a static 448-page (`28.00 MB`) `#![no_std]` WebAssembly linear arena (`0` host imports, `0` Cargo/npm runtime dependencies, `0` external CDN scripts or analytics).
   - User-selected MP4/WebM files (`Drag & Drop` or file picker) and webcam streams stay in browser memory (`0` bytes uploaded).
   - Do not expose sensitive local camera feeds or private footage on shared or untrusted screens.

2. **Local Development & HLS Relay Server (`scripts/dev_server.mjs`):**
   - Designed **strictly for local loopback (`127.0.0.1`) use** alongside local FFmpeg during development or evaluation of Tier-3 HTTP Live Streaming relays.
   - Enforces web-root path-traversal confinement, same-origin loopback `Host`/`Origin`/`Sec-Fetch-Site` verification, DNS-pinned SSRF blocking across IPv4/IPv6 non-global ranges (loopback, private `RFC 1918` / `RFC 4193`, CGNAT `RFC 6598`, link-local, NAT64/6to4/Teredo) on every redirect and HLS manifest hop, 512 KiB playlist size caps, 3-hop redirect limits, FFmpeg `-format_whitelist` demuxer restrictions, and bounded concurrent relay process slots (`maxLiveProcs = 4`, returning HTTP `429` when full).
   - **Do not bind `scripts/dev_server.mjs` to `0.0.0.0` or expose it directly to the public internet.**

## Reporting a Vulnerability

If you discover a security issue in the static web app, WebAssembly kernel, build/verification tooling, or `scripts/dev_server.mjs`:

- Please report it confidentially via [GitHub Private Vulnerability Reporting](https://github.com/steph4n-gh/chronos-spectral/security/advisories/new).
- Please **do not** open a public GitHub issue for undisclosed security vulnerabilities.
- Include reproduction steps, affected commit/file paths, and impact. We aim to acknowledge reports within 72 hours.
