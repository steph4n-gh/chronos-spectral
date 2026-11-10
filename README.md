# CHRONOS-SPECTRAL

`chronos_spectral` is a `#![no_std]` Rust `wasm32-unknown-unknown` computer-vision and biophysical motion tracker that runs inside a dedicated Web Worker with `simd128` intrinsics and an `OffscreenCanvas` telemetry deck.

All pixel ring buffers, background subspace planes, Dual Projected-FISTA state vectors, and spectral graph Laplacians live inside one contiguous `WebAssembly.Memory` arena pre-allocated at module instantiation (256 fixed pages / 16.00 MB). The kernel imports zero host functions, performs zero runtime heap allocations or `memory.grow` calls, and switches live between `320x180` SD and `640x360` HD spatial resolutions.

## Build, verify, and run

Ensure `cargo` with the `wasm32-unknown-unknown` target, `node` (v22+), and `python3` are on `PATH`. Install `wabt` (`wasm-objdump`), `ffmpeg`, and Google Chrome to run the full calibration and headless browser suites.

Run the release compile, `wasm-objdump` opcode/import audit, mathematical counterexample gates (Gates A through G), and 5x2 interleaved SD/HD benchmark without starting a server:

	./build.sh --check

Run the complete verification pipeline, including `--check`, all 5 real-camera MP4 calibration gates in both `320x180` SD and `640x360` HD, and the headless Chrome CDP full-stack E2E suite:

	./build.sh --e2e

Compile `web/chronos_spectral.wasm`, run `--check`, and launch the interactive local server at `http://127.0.0.1:8080`:

	./build.sh

## Spatial resolution modes

| Mode | Frame size | Patch & PiP | Coarse-to-fine steps | Apple M4 Pro median tick | Frame budget |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` (`SD`) | `320x180` (`57,600` px) | `16x16` native | `[3, 1]` px | `0.162 ms` | `< 0.25 ms` |
| `1` (`HD`) | `640x360` (`230,400` px) | `32x32` (`2x2` SIMD128 pooled) | `[6, 2, 1]` px | `0.566 ms` | `< 1.00 ms` |

## Repository layout

- `Cargo.toml` configures the `cdylib` release artifact with `opt-level = 3`, fat LTO, `codegen-units = 1`, and `panic = "abort"`.
- `.cargo/config.toml` sets `target = "wasm32-unknown-unknown"`, enables `+simd128`, sets a 16 KiB stack, and pins `WebAssembly.Memory` to 256 fixed pages (`16,777,216` bytes).
- `src/lib.rs` defines the static `Arena`, C ABI entrypoints, dual-resolution geometry scaling, Track-Peak graph Laplacian builder, Dual Projected-FISTA trend filter, $2 \times 2$ Kalman baseline, 15-tap equalized Hilbert quadrature filter, and 576-float telemetry buffer.
- `src/simd_ops.rs` implements `v128` kernels for RGBA-to-planar conversion, guarded rank-2 Extensive Cancellation Algorithm clutter projection, temporal stationarity masks, `2x2` SIMD128 box-pooled `32x32` Haar-Morton signatures, clamped DC-compensated patch SAD, and sub-pixel photometric SSD.
- `web/worker.js` runs the V8 Web Worker loop, `OffscreenCanvas` HUD scopes, Blind Pixel Pillar physics testbench, 1-to-1 identity swap oracle, shifted Lanczos Fiedler eigensolver, and zero-GC 18,000-frame `Float64Array` telemetry recorder.
- `web/index.html` provides the browser telemetry controls, SD/HD resolution toggle, built-in calibration MP4 selector, CSV/JSON session exporter, `requestVideoFrameCallback` media ingestion, and WebAudio polyphonic synthesizer.
- `web/samples/` stores 5 open-licensed real-camera H.264 calibration MP4 clips documented in `web/samples/ATTRIBUTION.md`.
- `scripts/calibrate_mp4.mjs` decodes `web/samples/*.mp4` via `ffmpeg` into raw RGBA buffers and validates lock persistence, ECA suppression, and zero NaNs across SD and HD modes.
- `scripts/e2e_browser.mjs` drives headless Chrome over the DevTools Protocol (CDP) with HTTP `206 Partial Content` byte-range streaming to test SD/HD switching, MP4 playback, Lab Modes A/B/C, and 38-column CSV/JSON telemetry exports.
- `.github/workflows/ci.yml` executes `./build.sh --e2e` on every push and pull request against `main`.

## Mathematical subsystems

### Non-causal 12-frame lookahead buffer

`Arena.rgba_ring` stores 12 frames up to `640x360` RGBA resolution. `chronos_pre_tick` ingests incoming frames at `frame_idx % 12` while `chronos_post_tick` renders delayed display slot `(frame_idx + 12 - eff_k) % 12` for lookahead `K` from `0` through `11`. When an active target enters an opaque occlusion zone and re-emerges within the 12-frame window, the tracker replaces blind forward coasting with two-point Hermite boundary splines before proximal smoothing.

### Rank-2 Extensive Cancellation Algorithm clutter suppression

`eca_subspace_project_masked_simd128` in `src/simd_ops.rs` projects each incoming luma frame orthogonally against the mean-centered rank-2 background subspace $X = [b - \bar{b}\mathbf{1},\, \mathbf{1}]$, cancelling both static spatial structure and global camera auto-exposure offsets. `update_stat_streak_and_mask_simd128` tracks raw RGB pixel invariance across 12 consecutive frames to measure uncensored clutter attenuation in decibels prior to background adaptation (`> 64 dB` on the Blind Pixel Pillar scene and `48-55 dB` on compressed H.264 camera feeds).

### Morton 2-adic signatures and spectral Fiedler association

`haar_morton_signature_scaled_simd128` maps each `16x16` (SD) or `32x32` (HD `2x2`-pooled) target patch into a 32-bit signature:

- Bit `31` gates chromatic confidence on opponent saturation.
- Bits `30..27` encode opponent `R-G` and `B-Y` thermometer polarity with a `0.055` deadzone.
- Bits `23..16` encode orthogonal 2D Haar quadrant wavelets.
- Bits `15..0` encode Morton-interleaved sub-block polarity relative to parent quadrants.

`padic_ultrametric_val` calls `(code_a ^ code_b).leading_zeros()`, which compiles to `i32.clz` and induces a binary-prefix ultrametric satisfying $d_2(A, C) \le \max(d_2(A, B), d_2(B, C))$. Before patch assignment, `chronos_pre_tick` splits merged blobs via `3x3` local sub-peaks, gates textured peaks on AC patch variance, and writes a `16x16` Track-Peak Laplacian. `solveShiftedLanczosFiedler` in `web/worker.js` computes the Fiedler vector $v_2$ and eigengap $\lambda_3 - \lambda_2$, weighting candidate association by $\gamma_{\text{eff}}(v_m - v_{8+p})^2$.

### Certified Dual Projected-FISTA trend filter

`solve_noncausal_fista_target` solves the weighted second-difference L1 trend-filtering problem across the 12-frame trajectory window:

$$\min_{X \in \mathbb{R}^{12}} \frac{1}{2}\sum_{k=0}^{11} w_k (X_k - Y_k)^2 + \lambda_{\text{eff}} \|D_2 X\|_1$$

where $\lambda_{\text{eff}} = \lambda \cdot s$ scales linearly with the resolution factor $s \in \{1, 2\}$ to preserve trajectory curvature invariance. The kernel solves the dual box problem over $u \in [-\lambda_{\text{eff}}, \lambda_{\text{eff}}]^{10}$ using `soft_threshold_branchless`, Gershgorin Lipschitz step bounds, coordinate warm-start sweeps, and 6 accelerated proximal-gradient steps with gradient restart, exporting $P(X) - D(u)$ to `telemetry[25]`.

### Eulerian micro-motion PiP, Hilbert tremor phase, and telemetry export

`chronos_post_tick` co-registers a `16x16` (SD) or `32x32` (HD) Picture-in-Picture patch at the smoothed `disp_fista` coordinate and amplifies temporal luma contrast by `eulerian_gain`. For Mode B tremor analysis, the kernel projects 2D trajectory residuals onto their principal covariance axis, applies an analytically equalized 15-tap Hamming Hilbert filter when `K >= 7`, and forward-predicts instantaneous phase to compensate lookahead and WebAudio output latency.

During active recording, `web/worker.js` appends 38 numeric columns per frame into a pre-allocated `18,000 x 38` `Float64Array` ring buffer (5 minutes at 60 Hz with zero GC allocations) and transfers a contiguous `ArrayBuffer` snapshot on demand for CSV or JSON download.

## Exported Wasm ABI

`src/lib.rs` exports zero-marshaling C functions over linear memory:

- `chronos_init()` zeroes `ARENA` and sets default tracker parameters in `320x180` SD mode.
- `chronos_reset_tracks()` clears active tracks, background planes, and swap counters while preserving user parameters and resolution mode.
- `chronos_set_resolution(mode)` switches between `0` (`320x180` SD) and `1` (`640x360` HD) inside the static 256-page arena.
- `chronos_get_resolution()`, `chronos_get_frame_w()`, `chronos_get_frame_h()`, and `chronos_get_pip_dim()` return active spatial dimensions.
- `chronos_get_ring_slot_ptr(slot)` returns the linear-memory byte offset of RGBA ring slot `slot % 12`.
- `chronos_set_params(lookahead_k, eca_adapt_rate, fista_lambda, eulerian_gain, phase_offset_rad, lab_mode)` updates runtime controls.
- `chronos_set_spectral_weight(gamma)` and `chronos_set_audio_latency(latency_sec)` configure spectral coupling and phase lead compensation.
- `chronos_trigger_stimulus()` arms Mode B step-response onset latency measurement.
- `chronos_pre_tick(dt_sec)` unpacks RGBA pixels, updates ECA residuals, extracts candidate peaks, and writes the `16x16` Laplacian.
- `chronos_set_fiedler_spectrum(lambda2, spectral_gap)` stores external Lanczos eigenvalues prior to `chronos_post_tick(dt_sec)`.
- `chronos_tick(dt_sec)` executes `chronos_pre_tick`, internal Jacobi Fiedler eigensolve, and `chronos_post_tick` in one call.
- `chronos_get_ingest_rgba_ptr()`, `chronos_get_display_rgba_ptr()`, `chronos_get_eca_res_ptr()`, `chronos_get_pip_rgba_ptr()`, `chronos_get_laplacian_ptr()`, `chronos_get_fiedler_ptr()`, and `chronos_get_telemetry_ptr()` return fixed linear-memory pointers.

## Scope limits

- Behind a 100% opaque occluder where visibility weight $w_k = 0$, two pixel-identical targets that exchange velocities produce the same exterior boundary observations as targets that pass straight through. Distinguishing hidden identical swaps requires distinct appearance signatures or non-crossing priors.
- Six Dual Projected-FISTA iterations run in bounded constant time per frame and report an explicit upper-bound duality gap $P(X) - D(u)$ rather than iterating to machine precision.
- Real-camera MP4 files exercise the identical `<video>` -> `createImageBitmap` -> `OffscreenCanvas` RGBA -> Wasm SIMD128 execution path as live `getUserMedia` feeds, adding H.264 quantization noise and camera auto-exposure drift deterministically, but they do not measure live camera-to-display photon latency on a physical human subject.
- Mode B strobe cues fire at forward-predicted quadrature phase offsets. Confirming biological tremor attenuation on human subjects requires external closed-loop physiological measurement.
