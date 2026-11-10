# CHRONOS-SPECTRAL

`chronos_spectral` is a `#![no_std]` Rust `wasm32-unknown-unknown` computer-vision and biophysical motion tracker that runs inside a dedicated Web Worker with `simd128` intrinsics and an `OffscreenCanvas` telemetry deck.

All pixel ring buffers, background subspace planes, Dual Projected-FISTA state vectors, and spectral graph Laplacians live inside one contiguous `WebAssembly.Memory` arena pre-allocated at module instantiation. The kernel imports zero host functions and performs zero runtime heap allocations.

## Build and run

Ensure `cargo` with the `wasm32-unknown-unknown` target, `node`, and `python3` are on `PATH`. Install `wabt` so `wasm-objdump` can verify opcode and import invariants.

Run the build, binary audit, mathematical counterexample suite, and 5-trial benchmark without starting the web server:

	./build.sh --check

Compile `web/chronos_spectral.wasm`, run the verification suite, and start the local static server at `http://127.0.0.1:8080`:

	./build.sh

## Repository layout

- `Cargo.toml` configures the `cdylib` release artifact with `opt-level = 3`, fat LTO, `codegen-units = 1`, and `panic = "abort"`.
- `.cargo/config.toml` sets `target = "wasm32-unknown-unknown"`, enables `+simd128`, sets a 4 KiB stack, and pins `WebAssembly.Memory` to 64 fixed pages, which equals 4,194,304 bytes.
- `src/lib.rs` defines the static `Arena`, C ABI entrypoints, Track-Peak graph Laplacian builder, Dual Projected-FISTA trend filter, $2 \times 2$ Kalman baseline, 15-tap equalized Hilbert quadrature filter, and flat telemetry array.
- `src/simd_ops.rs` implements `v128` kernels for RGBA-to-planar conversion, guarded rank-2 Extensive Cancellation Algorithm clutter projection, temporal stationarity masks, 2-adic Haar-Morton signatures, `16x16` patch SAD, and sub-pixel photometric SSD.
- `web/worker.js` runs the V8 Web Worker loop, `OffscreenCanvas` HUD scopes, Blind Pixel Pillar physics testbench, 1-to-1 identity swap oracle, and shifted Lanczos Fiedler eigensolver with invariant-subspace restarts.
- `web/index.html` provides the browser telemetry controls, `requestVideoFrameCallback` media ingestion, and WebAudio polyphonic synthesizer.
- `build.sh` compiles the Rust crate, inspects the `.wasm` binary with `wasm-objdump`, runs Gates A through F, and executes the 2,400-frame benchmark.

## Mathematical subsystems

### Non-causal 12-frame lookahead buffer

`Arena.rgba_ring` stores 12 frames at `320x180` RGBA resolution. `chronos_pre_tick` ingests incoming frames at `frame_idx % 12` while `chronos_post_tick` renders delayed display slot `(frame_idx + 12 - eff_k) % 12` for lookahead `K` from `0` through `11`. When an active target enters an opaque occlusion zone and re-emerges within the 12-frame window, the tracker replaces blind forward coasting with two-point Hermite boundary splines before proximal smoothing.

### Rank-2 Extensive Cancellation Algorithm clutter suppression

`eca_subspace_project_masked_simd128` in `src/simd_ops.rs` projects each incoming luma frame orthogonally against the mean-centered rank-2 background subspace $X = [b - \bar{b}\mathbf{1},\, \mathbf{1}]$. `update_stat_streak_and_mask_simd128` tracks raw RGB pixel invariance across 12 consecutive frames to measure uncensored clutter attenuation in decibels prior to background adaptation. On the Blind Pixel Pillar scene, uncensored stationary clutter attenuation reaches `64.58 dB` across `96.9%` of the frame.

### Morton 2-adic signatures and spectral Fiedler association

`haar_morton_signature_simd128` packs each `16x16` target patch into a 32-bit signature:

- Bit `31` gates chromatic confidence on opponent saturation.
- Bits `30..27` encode opponent `R-G` and `B-Y` thermometer polarity with a `0.055` deadzone.
- Bits `23..16` encode orthogonal 2D Haar quadrant wavelets.
- Bits `15..0` encode Morton-interleaved `4x4` sub-block polarity relative to parent quadrants.

`padic_ultrametric_val` calls `(code_a ^ code_b).leading_zeros()`, which compiles to `i32.clz` and induces a binary-prefix ultrametric satisfying $d_2(A, C) \le \max(d_2(A, B), d_2(B, C))$. Before patch assignment, `chronos_pre_tick` builds a `16x16` Track-Peak Laplacian and `solveShiftedLanczosFiedler` in `web/worker.js` computes the Fiedler vector $v_2$ and eigengap $\lambda_3 - \lambda_2$. `chronos_post_tick` weights candidate association by sign-invariant spectral discrepancy $\gamma_{\text{eff}}(v_m - v_{8+p})^2$.

### Certified Dual Projected-FISTA trend filter

`solve_noncausal_fista_target` solves the weighted second-difference L1 trend-filtering problem across the 12-frame trajectory window:

$$\min_{X \in \mathbb{R}^{12}} \frac{1}{2}\sum_{k=0}^{11} w_k (X_k - Y_k)^2 + \lambda \|D_2 X\|_1$$

The kernel solves the dual box problem over $u \in [-\lambda, \lambda]^{10}$ with Moreau box projection via `soft_threshold_branchless`, Gershgorin Lipschitz step bounds, coordinate warm-start sweeps, and 6 accelerated proximal-gradient steps with gradient restart. `eval_primal_from_dual` exports the primal-dual optimality gap $P(X) - D(u)$ to `telemetry[25]`.

### Eulerian micro-motion PiP and Hilbert tremor phase

`chronos_post_tick` co-registers a `16x16` Picture-in-Picture patch at the smoothed `disp_fista` coordinate and amplifies temporal luma contrast by `eulerian_gain`. For Mode B tremor analysis, the kernel projects 2D trajectory residuals onto their principal covariance axis, applies an analytically equalized 15-tap Hamming Hilbert filter when `K >= 7`, and forward-predicts instantaneous phase to compensate lookahead and WebAudio output latency.

## Exported Wasm ABI

`src/lib.rs` exports zero-marshaling C functions over linear memory:

- `chronos_init()` zeroes `ARENA` and sets default tracker parameters.
- `chronos_reset_tracks()` clears active tracks, background planes, and swap counters while keeping user parameters.
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
- Mode B strobe cues fire at forward-predicted quadrature phase offsets. Confirming biological tremor attenuation on human subjects requires external closed-loop physiological measurement.
