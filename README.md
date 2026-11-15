# CHRONOS-SPECTRAL

`chronos_spectral` is a `#![no_std]` Rust `wasm32-unknown-unknown` computer-vision and biophysical motion tracker that runs inside a dedicated Web Worker with `simd128` intrinsics and an `OffscreenCanvas` telemetry deck.

All pixel ring buffers, background subspace planes, Dual Projected-FISTA state vectors, and spectral graph Laplacians live inside one contiguous `WebAssembly.Memory` arena pre-allocated at module instantiation (288 fixed pages / 18.00 MB). The kernel imports zero host functions, performs zero runtime heap allocations or `memory.grow` calls, and switches live between `320x180` SD and `640x360` HD spatial resolutions.

## Build, verify, and run

Ensure `cargo` with the `wasm32-unknown-unknown` target, `node` (v22+), and `python3` are on `PATH`. Install `wabt` (`wasm-objdump`), `ffmpeg`, and Google Chrome to run the full calibration and headless browser suites.

Run the release compile, `wasm-objdump` opcode/import audit, mathematical counterexample gates (Gates A through O), and 5x2 interleaved SD/HD benchmark without starting a server:

	./build.sh --check

Run the complete verification pipeline, including `--check`, all 5 real-camera MP4 calibration gates in both `320x180` SD and `640x360` HD, and the headless Chrome CDP full-stack E2E suite (including the live fragmented-MP4 MediaSource stream & camera-cut audit):

	./build.sh --e2e

Compile `web/chronos_spectral.wasm`, run `--check`, and launch the interactive local server (`scripts/dev_server.mjs`, serving static assets, RFC 7233 HTTP 206 byte ranges, and `/api/live-stream` fMP4 live relays) at `http://127.0.0.1:8080`:

	./build.sh

## Spatial resolution modes

| Mode | Frame size | Split PiP (`1x` vs `25x Δ`) | Coarse-to-fine proposal | Apple M4 Pro median tick | Frame budget |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` (`SD`) | `320x180` (`57,600` px) | `32x32` PiP (`16x16` patch) | `160x90` pyramid + `[2, 1]` px + 2D IC-LK | `0.215 ms` Pillar | `< 0.35 ms` |
| `1` (`HD`) | `640x360` (`230,400` px) | `64x64` PiP (`32x32` `2x2` pooled) | `160x90` pyramid + `[4, 1]` px + 2D IC-LK | `0.647 ms` Pillar | `< 1.00 ms` |

## Real-camera MP4 ground-truth benchmark

`scripts/calibrate_mp4.mjs` evaluates pixel-domain CLEAR-MOT ground-truth trajectories (`scripts/gt_mp4_reference.json`), precision bounds, negative non-periodic / non-pendulum gates, and mode-specific physics-proof invariants across all 5 open-licensed H.264 sample clips (`--gate` enforces per-clip `Recall >= minRecallPct`, `Prec >= minPrecisionPct`, `Teleports == 0`, `StaticPins <= 1`, `Hijacks <= 16`, `clutterDb >= minClutterDb`, p-adic vs Kalman swap ordering, physics/frequency/gravity gates, and SD/HD `2x` gravity consistency while also measuring and reporting `RMSE` and `IDSW`):

| Sample clip | Mathematical proof target | `320x180` SD Recall / RMSE | SD Hijacks / Teleports | `640x360` HD Recall / RMSE | HD Hijacks / Teleports | Verified HD physics & trig proof |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `juggling_ballistic.mp4` (`25 fps`) | Mode A multi-arc $\hat{g}$ & flight $\theta$ | `75.9%` / `9.30 px` | `3` / `0` | `77.6%` / `15.18 px` | `0` / `0` | `6`-arc Auto-$g$ (`1.7%` SD/HD $2\times$ err), `0` false Hz |
| `roundabout_occlusion.mp4` (`24 fps`) | Mode A `< K` overhead gantry bridge + $H$ | `94.0%` / `9.99 px` | `3` / `0` | `93.0%` / `18.22 px` | `4` / `0` | `9f` Hermite bridges, `0` false pivots, `0` false Hz |
| `galileo_pendulum_peg.mp4` (`30 fps`) | Kåsa virtual pendulum pivot & swing $\theta$ | `100.0%` / `7.81 px` | `0` / `0` | `100.0%` / `3.20 px` | `0` / `0` | `61f` synchronized virtual pivot (`0` IDSW) |
| `harmonic_metronome_phase.mp4` (`30 fps`) | Mode B Nikko `96 BPM` (`0.80 Hz`) + pivot | `89.0%` / `14.05 px` | `0` / `0` | `96.0%` / `24.64 px` | `0` / `0` | `108f` `0.82 Hz` lock (`SD 0.80 Hz`) + `110f` pivot |
| `carom_billiards_impulses.mp4` (`30 fps`) | Mode C deflection $\Delta\theta$, $v^+/v^-$, & $H$ | `92.6%` / `9.54 px` | `2` / `0` | `93.3%` / `19.80 px` | `2` / `0` | `30` L1 strikes + `117.3°` (`SD 27.2°`) scatter (`0` false Hz) |

## Repository layout

- `Cargo.toml` configures the `cdylib` release artifact with `opt-level = 3`, fat LTO, `codegen-units = 1`, and `panic = "abort"`.
- `.cargo/config.toml` sets `target = "wasm32-unknown-unknown"`, enables `+simd128`, sets a 16 KiB stack, and pins `WebAssembly.Memory` to 288 fixed pages (`18,874,368` bytes).
- `src/lib.rs` defines the static `Arena`, C ABI entrypoints, `64`-frame trajectory rings (`HIST_RING_LEN = 64`, `HIST_MASK = 63`) and `hist_t_sec` (`f64`) discontinuity invalidation, pixel-domain Camera Cut / PTZ Pan detector (`find_best_win_shift_simd128` with `eval_cam_coarse_tile_simd128` + dense `-4..=4` core + stride-2 + exhaustive fallback plus `eval_ref_pt` exhaustive fine-grid fallback), `M-of-N` track confirmation and static-pin reaper, isoluminant chromatic seeding, viewport-exit ring drain, radial rod/rail de-slipped P-N dual-anchor templates (`apply_radial_deslip_weights`), 4-Mode IMM curved-arc & oscillatory prediction/update (CV + Ballistic-Drag Verlet + CTRV Taylor + Oscillatory modes with pan-advected states), Jonker-Volgenant `16x64` global bipartite matcher (up to `16` concurrent targets, `32`-node Laplacian), non-causal backward `rgba_ring` gap re-tracker with 2-frame capped pre-birth backtrack and `no_scene_grav` / `vel_aligned` / `achromatic_track` coasting guards, `32`-slot / `1200`-frame (`40 s` at `30 fps`) `DormantTrack` Re-ID archive (`16x16` zero-mean foreground templates + 4-mode corridor + 2-adic Haar-Morton + 2-hit fast re-confirmation), Track-Peak graph Laplacian builder, 2-pass IRL1 Dual Projected-FISTA trend filter (`solve_noncausal_fista_irl1_target`) with multi-arc consensus quadratic gravity estimation ($\hat{g}$), `t - K`-synchronized Kåsa $2 \times 2$ and `f64` Jacobi Taubin $3 \times 3$ pivot/curvature solvers, bi-directional turnaround periodicity gate, pre/post-impact deflection & speed-ratio kinematics, close-contact two-body scatter wedge detector, and `1568`-float telemetry buffer (`TELEMETRY_LEN = 1568`).
- `src/physics_id.rs` implements `#![no_std]` in-situ physical law discovery and conservation telemetry inside the fixed arena (`LyapunovScratch`): staggered (`1:4` + event-driven) moment-matched weak-form STRidge + AICc symbolic ODE identification (`phys_regime` `0..8`, `ode_r2`, `aero_drag_beta`, `thrust_accel`), `f64` `3x3` Jacobi Taubin vs Kåsa curvature analysis & Galileo peg jump detector (`constraint_jump_code = 1`), `f64` AGM Jacobi elliptic period $T(\theta_0)$ and log-peak decrement damping ($\zeta, Q$), staggered (`1:8`) `4D` delay-coordinate Rosenstein Lyapunov divergence rate ($\lambda_{\max}$), Kuramoto ensemble synchronization ($R_{\text{sync}}, \Psi, N$), and causal blur-excluded two-body collision momentum/energy conservation ($m_2/m_1, e_n$, momentum residual, energy retention).
- `src/simd_ops.rs` implements `v128` kernels for RGBA-to-planar conversion, anchor-shifted `f64x2` Rank-2 Extensive Cancellation Algorithm clutter projection (`eca_fit_affine_params_simd128`), fused pan `rgb_sum` streak caching and ECA/motion-grid projection (`eca_and_motion_grid_pan_fused_simd128`), coarse camera shift tile evaluation (`eval_cam_coarse_tile_simd128`), an 8-row L1D ring buffer for subpixel Catmull-Rom background advection with overlap-safe border filling, hoisted branch-free exposure/pan streak loops (`span_reset_u8=35`, `span_soft_u8=26`), `rgb_abs_sum_f32x4_simd128` (`extadd_pairwise`) motion-grid updates, `160x90` multi-channel box pyramids, 2D Shi-Tomasi structure-tensor cornerness, `2x2` SIMD128 box-pooled `32x32` Haar-Morton signatures, zero-mean gain-invariant foreground-weighted patch SAD & anchor L1 drift locks (`sad_patch_zm_score_simd128`, `sad_patch_zm_weighted_simd128`, `weighted_anchor_zm_l1_simd128`), `2x2` pooled HD PiP rendering, and luma-only Huber IRLS Inverse-Compositional Lucas-Kanade (`refine_subpixel_ic_lk_huber_simd128`) sub-pixel refinement.
- `web/worker.js` runs the V8 Web Worker loop, aspect-preserving letterbox + normalized Crop ROI staging, `frame_ack` backpressure protocol, `OffscreenCanvas` HUD scopes (flight-angle protractors, Kåsa virtual pendulum pivot bullseyes & swing wedges, CAD collision deflection protractors & 2-body scatter wedges, $3 \times 3$ plane homography $H^{-1}$ iso-metric grids, frozen Euclidean Mode A apex crosshairs, enlarged split `1x RAW [t-K]` vs `25x TEMPORAL Δ` PiPs up to `64x64`, `16`-slot enlarged Split PiP roster cards, calm `2 Hz` Zero-GC multi-minute continuity & concurrency chart, auto-ranging Hilbert quadrature orbit, chronological `t-11 -> VIEW [t-K] -> NOW [t]` timeline, and `15 Hz` `#physicsPhaseCanvas` phase-space portrait & conservation trace), atomic draft-staged 2-point metric ruler / multi-arc Auto-$g$ (`9.81 m/s²`) / 4-point projective homography engine ($J_H$ velocity & angle rectification with `OUT OF PLANE` horizon guard and `ASSUMED` vs `SURVEYED` provenance), 7-scenario Blind Pixel Pillar testbench (including `model_rocket_flight` and 12-disc `multi_target_swarm`), shifted 31-step Lanczos Fiedler eigensolver, and zero-GC `18,000 x 56` `Float64Array` segmented telemetry recorder (`REC_COLUMNS[0..55]`).
- `web/index.html` provides the browser telemetry controls, SD/HD resolution toggle, **3-Tier Continuous Live-Stream Suite** (Tier 1 `getDisplayMedia` Tab/Window Capture + Crop ROI, Tier 2 Direct CORS-probed stream with 1-px readback guard, Tier 3 native `MediaSource` fragmented-MP4 live stream tuner `<dialog>`), built-in real-camera physics/traffic/rocketry MP4 selector, exact `1/fps` MP4 transport (`Pause/Play`, `-1f`, `+1f`, `Replay`), accessible Metric & Perspective Calibration toolbar + keyboard coordinate disclosure, enlarged `#pipCanvas` (`412x234`) & 2-column scrollable 16-slot `#rosterScrollWell` (`276x148` per slot) with click/button `[FOCUS]` target pin, calm `1 Hz` / `2 Hz` `#persistencePanel` multi-minute continuity view, `4 Hz`-throttled `#physicsIdPanel` & `15 Hz` `#physicsPhaseCanvas` in-situ physics ID panel, `56`-column CSV/JSON session exporter (`schemaVersion: "1.5.0"`), epoch-guarded `requestVideoFrameCallback` media ingestion, and WebAudio polyphonic synthesizer.
- `docs/UI_TELEMETRY_GUIDE.md` documents every UI element (`#physicsIdPanel`, `#physicsPhaseCanvas`, and all sub-IDs), `OffscreenCanvas` overlay, dynamic KPI state transition, calibration provenance rule, WebAudio voice, and exported `telemetry[0..1567]` / `56`-column `v1.5.0` recorder field, explaining what each element measures and its diagnostic value.
- `scripts/dev_server.mjs` implements the shared loopback HTTP server (`GET`/`HEAD` static files with RFC 7233 `206 Partial Content`, `GET /api/live-presets` probing public Caltrans `wzmedia.dot.ca.gov` highway HLS feeds alongside 3 deterministic local live/cut feeds, and `GET /api/live-stream` `ffmpeg` fragmented-MP4 live relay with strict loopback & SSRF guards).
- `web/samples/` stores 13 open-licensed real-camera H.264 physics, rocketry, and traffic MP4 clips (8 extended `29` to `80 s` showcase clips including full-flight solid-motor model rockets with stage separation & twin parachutes, `79.9 s` pad-to-pad VTVL rocket ascent & landing, and parachute water-rocket flights, plus 5 pixel-calibrated `5.4` to `6.7 s` ground-truth benchmark clips) documented in `web/samples/ATTRIBUTION.md`.
- `scripts/calibrate_mp4.mjs` decodes `web/samples/*.mp4` via `ffmpeg` into raw RGBA buffers and validates CLEAR-MOT ground-truth metrics (`scripts/gt_mp4_reference.json`), precision bounds, SD/HD `2x` gravity consistency, physics/trigonometric/conservation invariants (plus `--extended` / `--all` / `--clip=<substr>` for the 8 extended HD showcases), ECA suppression, and zero NaNs across SD and HD modes.
- `scripts/e2e_browser.mjs` drives headless Chrome over the DevTools Protocol (CDP) against `scripts/dev_server.mjs` to test SD/HD switching, exact `25 fps` frame-stepping, paused calibration sync, multi-clip MP4 physics & negative periodicity gates, `ASSUMED` vs `SURVEYED` provenance + atomic `Escape` cancel, `56`-column `v1.5.0` CSV/JSON telemetry exports, and live fMP4 `MediaSource` streaming across a hard camera cut.
- `.github/workflows/ci.yml` executes `./build.sh --e2e` on every push and pull request against `main`.

## Mathematical subsystems

### Non-causal 12-frame lookahead buffer & backward ring re-tracking

`Arena.rgba_ring` stores 12 frames up to `640x360` RGBA resolution. `chronos_pre_tick` ingests incoming frames at `frame_idx % 12` while `chronos_post_tick` renders delayed display slot `(frame_idx + 12 - eff_k) % 12` for lookahead `K` from `0` through `11`. When a confirmed target re-emerges from occlusion or finishes `M-of-N` confirmation, `backtrack_gap_in_rgba_ring` unpacks historical ROIs from `rgba_ring` (`t - 1 .. t - K`) to recover missed visible sub-pixel detections before pinning residual opaque gaps with two-point Hermite boundary splines.

### Rank-2 Extensive Cancellation Algorithm clutter suppression

`eca_subspace_project_masked_simd128` in `src/simd_ops.rs` projects each incoming luma frame orthogonally against the mean-centered rank-2 background subspace $X = [b - \bar{b}\mathbf{1},\, \mathbf{1}]$, cancelling both static spatial structure and global camera auto-exposure offsets while guarding stationary background learning (`0.15x` rate on unverified moving pixels) against foreground absorption. `update_stat_streak_and_mask_simd128` tracks raw RGB pixel invariance across 12 consecutive frames to measure uncensored clutter attenuation in decibels prior to background adaptation (`> 64 dB` on the Blind Pixel Pillar scene and `36.5-45.7 dB` on compressed H.264 camera feeds).

### Foreground-masked P-N templates, 2-adic signatures, 4-mode IMM, and Huber IC-LK

`capture_template_scaled` weights each target patch by radial Gaussian times observed ECA motion mask (`sum(tpl_weight) = 1.0`, suppressing outer square-patch background corners by `> 12x`) alongside `apply_radial_deslip_weights` (dampening 1D pendulum/metronome rod peripheral texels along the radial tether axis) and an immutable birth anchor (`anchor_luma/rg/by`) that blocks cumulative P-N template drift. `haar_morton_signature_scaled_simd128` maps each `16x16` (SD) or `32x32` (HD `2x2`-pooled) target patch into a 32-bit signature:

- Bit `31` gates chromatic confidence on opponent saturation.
- Bits `30..27` encode opponent `R-G` and `B-Y` thermometer polarity with a `0.055` deadzone.
- Bits `23..16` encode orthogonal 2D Haar quadrant wavelets.
- Bits `15..0` encode Morton-interleaved sub-block polarity relative to parent quadrants.

`padic_ultrametric_val` calls `(code_a ^ code_b).leading_zeros()`, which compiles to `i32.clz` and induces a binary-prefix ultrametric satisfying $d_2(A, C) \le \max(d_2(A, B), d_2(B, C))$. Candidate seeds must pass `patch16x16_shi_tomasi_simd128` 2D structure-tensor eigenvalue checks ($\lambda_{\min}$ vs $\lambda_{\max}$) and `3`-of-`8` tentative confirmation (or `2`-hit fast confirmation when matching the `32`-slot, `1200`-frame `DormantTrack` archive within `40 s` at `30 fps`) before entering telemetry. A 4-mode Interacting Multiple Model (IMM) filter couples Inertial Constant-Velocity (CV), Ballistic-Drag (Verlet integration under gravity clamped to $[150s, 3600s]\,\text{px/s}^2$), Coordinated-Turn (CTRV with 4th-order small-angle Taylor expansion for $|\psi| < 10^{-2}$), and Oscillatory state hypotheses with pan-advected equilibrium coordinates, exporting collapsed weights $(w_{\text{cv}}, w_{\text{bal}}, w_{\text{turn\_osc}})$. `propose_track_candidates_pyr` searches a `160x90` SIMD128 multi-channel pyramid around curved predictions, `solve_jv_assign_16x64` solves global minimum-cost bipartite matching (including zero-mean gain-invariant foreground SAD `sad_patch_zm_score_simd128` / `sad_patch_zm_weighted_simd128` / `weighted_anchor_zm_l1_simd128` against plume/glare contrast shifts, per-track coasting null columns, and Fiedler spectral penalties $\gamma_{\text{eff}}(v_m - v_{16+p})^2$), and `refine_subpixel_ic_lk_huber_simd128` extracts sub-pixel displacements via luma-only Huber IRLS Inverse-Compositional Lucas-Kanade.

### Certified Dual Projected-FISTA & 2-pass IRL1 trend filter

`solve_noncausal_fista_irl1_target` solves the iteratively reweighted weighted second-difference L1 trend-filtering problem across the 12-frame trajectory window:

$$\min_{X \in \mathbb{R}^{12}} \frac{1}{2}\sum_{k=0}^{11} w_k (X_k - Y_k)^2 + \lambda_{\text{eff}} \sum_{j=0}^{9} w^{\text{IRL1}}_j |(D_2 X)_j|$$

where $\lambda_{\text{eff}} = \lambda \cdot s$ scales linearly with the resolution factor $s \in \{1, 2\}$ to preserve trajectory curvature invariance. Pass 1 solves the smooth dual box problem over $u \in [-\lambda_{\text{eff}}, \lambda_{\text{eff}}]^{10}$ using `soft_threshold_branchless`, Gershgorin Lipschitz step bounds, coordinate warm-start sweeps, and 6 accelerated proximal-gradient steps with gradient restart, taking an immediate early-out when $\min_j w^{\text{IRL1}}_j > 0.48$ (`fista_sharp_knots = 0`). When localized high-impulse knots appear (carom strikes, Galileo peg stops, stage burnouts, or parachute deployments), Pass 2 reweights the per-knot $D_2$ bounds to sharpen corners while exporting $P(X) - D(u)$ to `telemetry[25]` and sharp knot counts to `telemetry[1280 + m*16 + 3]`.

### In-situ weak-form physics identification & ensemble dynamics

`src/physics_id.rs` runs `#![no_std]` in-situ physical law discovery and conservation verification over `HIST_RING_LEN = 64` frames (`TELEM_PHYSICS_START = 1280`, `TELEM_ENSEMBLE_START = 1536`, total `1568` floats):

- **Weak-form STRidge + AICc symbolic ODE identification:** Integrates compact-support smooth test functions $\phi(u) = (1 - u^2)^p$ (`1:4` staggered cadence plus event-driven updates on sharp IRL1 knots; `15`-frame hysteresis hold and `56`-frame cooldown for jump events) across a compact $M \le 12$ library to classify `phys_regime` (`0..8`: Inertial CV, Ballistic, Ballistic AeroDrag, Powered Boost, Parachute Terminal, Pendulum Harmonic, Micro Tremor, Impulsive Collisional, Curved CTRV) alongside `ode_r2`, `aero_drag_beta`, `thrust_accel`, specific energy, and specific angular momentum.
- **Jacobi Taubin curvature & Galileo peg jump:** Computes exact `f64` $3 \times 3$ cyclic Jacobi Taubin circle fits, center divergence against algebraic Kåsa fits (`taubin_vs_kasa_delta_px`), curvature (`taubin_curvature_inv_px`), and constraint jump `1` (3-frame hold) on sudden tether radius changes.
- **AGM Jacobi elliptic period & log-decrement damping:** Evaluates the exact large-amplitude pendulum period $T(\theta_0) = 4\sqrt{L/g}\,K(\sin^2(\theta_0/2))$ (`elliptic_period_s`) via `f64` Arithmetic-Geometric Mean alongside peak-decrement damping $\zeta$ and $Q \in [0, 999]$.
- **Rosenstein Lyapunov divergence & Kuramoto synchronization:** Computes staggered (`1:8`) `4D` delay-coordinate Rosenstein divergence rates (`lyapunov_div_rate`) via a `64`-slot `LyapunovScratch` reservoir on interior oscillatory/CTRV tracks, plus the ensemble Kuramoto complex order parameter $R_{\text{sync}} e^{i\Psi} = \frac{1}{N} \sum_j e^{i\theta_j}$ across $N \ge 2$ active oscillators.
- **Causal two-body collision conservation:** Evaluates pre/post-impact windows once causally cleared (`f_c + 2 + win_w < frame_idx`, excluding `[f_c - 2 .. f_c + 2]` motion blur), latching SNR-conditioned mass ratio $m_2/m_1 \in [0.05, 20.0]$, normal restitution $e_n \in [0, 1]$, normalized momentum residual, and kinetic energy retention alongside constraint jump priority `4 > 1 > 3 > 2`.

### Eulerian micro-motion PiP, Hilbert tremor phase, and telemetry export

`chronos_post_tick` co-registers a border-clamped `32x32` (SD) or `64x64` (HD) Picture-in-Picture patch centered at the smoothed `disp_fista` coordinate and amplifies temporal luma contrast by `eulerian_gain`. For Mode B oscillator/tremor analysis, the kernel resamples the `hist_t_sec`-timestamped trajectory onto a uniform temporal grid, projects residuals onto their principal covariance axis, applies an analytically equalized 15-tap Hamming Hilbert filter when `K >= 7`, gates periodicity on bi-directional turnaround crossings, and forward-predicts instantaneous phase to compensate lookahead and WebAudio output latency.

During active recording, `web/worker.js` appends `56` numeric columns per frame (`REC_COLUMNS[0..55]`: the core `48` kinematics/Hilbert/spectral/calibration columns including `heading_deg`, `speed_phys`, `ang_vel_deg_s`, `pend_swing_deg`, `strike_deflect_deg`, `strike_speed_ratio`, `metric_x_m`, `metric_y_m`, `cal_segment_id`, and `metric_valid`, plus columns `48..55` for `phys_regime`, `phys_regime_conf`, `ode_r2`, `aero_drag_beta`, `thrust_accel`, `damping_zeta`, `kuramoto_order_r`, and `lyapunov_div_rate`) into a pre-allocated `18,000 x 56` `Float64Array` ring buffer (5 minutes at 60 Hz with zero GC allocations) alongside `calibrationSegments` provenance metadata for CSV or JSON download (`schemaVersion: "1.5.0"`).

## Exported Wasm ABI

`src/lib.rs` exports zero-marshaling C functions over linear memory:

- `chronos_init()` zeroes `ARENA` and sets default tracker parameters in `320x180` SD mode.
- `chronos_reset_tracks()` clears active tracks, background planes, and swap counters while preserving user parameters and resolution mode.
- `chronos_reset_calibration_priors()` clears accumulated multi-arc scene gravity priors (`scene_gravity_px_s2`, `scene_gravity_conf`, `scene_gravity_arcs`).
- `chronos_set_ingest_pts(pts_sec)` attaches explicit `f64` media presentation timestamps to the next ingested frame for jitter-invariant velocity, gravity, and Hilbert kinematics.
- - `chronos_set_resolution(mode)` switches between `0` (`320x180` SD) and `1` (`640x360` HD) inside the static 288-page arena.
- `chronos_get_resolution()`, `chronos_get_frame_w()`, `chronos_get_frame_h()`, and `chronos_get_pip_dim()` return active spatial dimensions.
- `chronos_get_ring_slot_ptr(slot)` returns the linear-memory byte offset of RGBA ring slot `slot % 12`.
- `chronos_set_params(lookahead_k, eca_adapt_rate, fista_lambda, eulerian_gain, phase_offset_rad, lab_mode)` updates runtime controls.
- `chronos_set_spectral_weight(gamma)` and `chronos_set_audio_latency(latency_sec)` configure spectral coupling and phase lead compensation.
- `chronos_set_focus_target(track_id)` and `chronos_get_focus_target()` pin or query the user-selected `[FOCUS]` track (`0` = automatic mode-aware selection) for the split Eulerian PiP and Hilbert/kinematic scopes.
- `chronos_trigger_stimulus()` arms Mode B step-response onset latency measurement.
- `chronos_pre_tick(dt_sec)` unpacks RGBA pixels, runs Camera Cut / PTZ Pan detection, updates ECA residuals, extracts candidate peaks, and writes the `32x32` Laplacian.
- `chronos_set_fiedler_spectrum(lambda2, spectral_gap)` stores external Lanczos eigenvalues prior to `chronos_post_tick(dt_sec)`.
- `chronos_tick(dt_sec)` executes `chronos_pre_tick`, internal Jacobi Fiedler eigensolve, and `chronos_post_tick` in one call.
- `chronos_get_ingest_rgba_ptr()`, `chronos_get_display_rgba_ptr()`, `chronos_get_eca_res_ptr()`, `chronos_get_pip_rgba_ptr()`, `chronos_get_laplacian_ptr()`, `chronos_get_fiedler_ptr()`, and `chronos_get_telemetry_ptr()` return fixed linear-memory pointers.

## Scope limits

- Behind a 100% opaque occluder where visibility weight $w_k = 0$, two pixel-identical targets that exchange velocities produce the same exterior boundary observations as targets that pass straight through. Distinguishing hidden identical swaps requires distinct appearance signatures or non-crossing priors.
- Six Dual Projected-FISTA iterations run in bounded constant time per frame and report an explicit upper-bound duality gap $P(X) - D(u)$ rather than iterating to machine precision.
- Real-camera MP4 files exercise the identical `<video>` -> `createImageBitmap` -> `OffscreenCanvas` RGBA -> Wasm SIMD128 execution path as live `getUserMedia` feeds, adding H.264 quantization noise and camera auto-exposure drift deterministically, but they do not measure live camera-to-display photon latency on a physical human subject.
- Mode B strobe cues fire at forward-predicted quadrature phase offsets. Confirming biological tremor attenuation on human subjects requires external closed-loop physiological measurement.
