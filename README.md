# CHRONOS-SPECTRAL

[![Live Showcase](https://img.shields.io/badge/Live_Demo-Split--Wipe_Showcase-00b8d4)](https://steph4n-gh.github.io/chronos-spectral/)
[![CI](https://github.com/steph4n-gh/chronos-spectral/actions/workflows/ci.yml/badge.svg)](https://github.com/steph4n-gh/chronos-spectral/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](./LICENSE)

**[Launch Interactive Split-Wipe Showcase ->](https://steph4n-gh.github.io/chronos-spectral/)** · **[Open Full 15-Panel Command Deck ->](https://steph4n-gh.github.io/chronos-spectral/?view=deck)** · **[Architecture & Math](./docs/ARCHITECTURE_AND_MATH.md)** · **[UI & Telemetry Guide](./docs/UI_TELEMETRY_GUIDE.md)** · **[Benchmarks](./docs/VERIFICATION_AND_BENCHMARKS.md)**

CHRONOS-SPECTRAL turns raw video pixels directly into occlusion-bridged trajectories, symbolic differential equations, and continuum flow fields in under 2 milliseconds per CPU frame. It solves tracking, phase measurement, and system identification from classical signal processing and variational mechanics rather than trained neural networks.

Everything runs inside one `2.13 MiB` (`2,234,437 B`) `#![no_std]` Rust WebAssembly (`wasm32-unknown-unknown` `+simd128`) binary pinned to a fixed `20.00 MB` (`320`-page) static arena. The kernel has zero host imports, zero runtime heap allocations, zero GPU compute dependencies, zero `MaybeUninit` blocks, and zero `unsafe` code inside the continuum physics module.

## Live demonstrations

> **Try it live in your browser (zero install, zero upload):** Open the **[Interactive Split-Wipe Showcase](https://steph4n-gh.github.io/chronos-spectral/)** to drag the `t - K` raw-vs-spectral split-wipe across `1,400+` starlings and full-flight model rockets, drop any local MP4 onto the canvas (`0` bytes leave your device), or switch into the **[Full 15-Panel Command Deck](https://steph4n-gh.github.io/chronos-spectral/?view=deck)**.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="./docs/assets/demos/demo_starling_swarm.gif" alt="Real-camera 1,400+ starling murmuration swarm tracked with Hero-16 reticles hidden" width="100%" /><br />
      <strong>Starling Murmuration Swarm (1,453 peak targets, 1.77 ms HD)</strong><br />
      Real-camera <code>640x360</code> footage with single-object reticles toggled off (<code>HERO 16: HIDDEN</code>), isolating pure micro-track velocity stems and the bilinear <code>40x22</code> Eulerian speed field.
    </td>
    <td width="50%" align="center">
      <img src="./docs/assets/demos/demo_model_rockets.gif" alt="Full-flight solid-motor model rocket tracked through liftoff, stage separation, apogee rollover, and twin parachute descent" width="100%" /><br />
      <strong>Model Rockets: Ignition, Separation, and Parachutes</strong><br />
      Single-core non-causal tracking across pad exhaust, supersonic booster separation, Hermite cloud bridges, apogee rollover, and twin canopy deployment.
    </td>
  </tr>
</table>

*Both animations are unmodified `640x360` H.264 sample clips captured directly from headless Chromium running the `OffscreenCanvas` WebGL2/2D Command Deck (`node scripts/capture_docs_media.mjs`).*

## Why this exists and what makes it unique

Most video perception stacks split into two camps: heavy neural detectors (`YOLO`/`SAM` paired with `ByteTrack`) that fail outside their training distribution and demand gigabytes of VRAM, or classical causal filters (`OpenCV` optical flow + Kalman) that smooth away sharp impacts, lag during phase estimation, and swap IDs whenever identical objects cross behind clutter. CHRONOS-SPECTRAL occupies the missing third quadrant: a deterministic visual physics instrument in `2.13 MiB` of portable WebAssembly.

| Capability | Neural Detector + Sort (`YOLO`/`SAM`) | Classical Causal (`OpenCV` + Kalman) | CHRONOS-SPECTRAL (`v2.0.0`) |
| :--- | :--- | :--- | :--- |
| Object taxonomy / training | Requires labeled classes & weights | Class-agnostic blobs | Zero ML, zero weights: arbitrary unknown targets on frame 1 |
| Auto-exposure & camera flicker | Vulnerable without domain augmentation | Thresholding breaks on exposure steps | Orthogonal Rank-2 ECA subspace (`36-64 dB` background suppression) |
| Crossing identical targets | Re-ID struggles on indistinguishable objects | Spatial proximity causes ID swaps | Fiedler spectral partitioning + `i32.clz` Haar-Morton `v2` affinity |
| Opaque occlusions & sharp impacts | Interpolates offline; Kalman rounds corners | Causal drift; corners smear across hits | 12-frame lookahead (`t - K`): backward re-tracking, Hermite bridges, 2-pass IRL1 FISTA |
| Dense `1,000+` swarm association | Prohibitively slow without server GPUs | `O(N^2)` distance checks bottle at `< 300` | 16-bit Morton radix sort + `3x3` CSR (`21.94x` speedup) + fixed-point `epsilon`-Auction (`< 1.8 ms`) |
| Output representation | 2D bounding boxes | Pixel centroids and `(vx, vy)` | Calibrated trajectories, symbolic ODEs, conservation metrics, vorticity, FTLE, `SE(2)` bodies |
| Runtime footprint | `50 MB - 4 GB` weights + CUDA/ONNX | Variable C++ heap & OS threads | `2.13 MiB` Wasm, `20.00 MB` fixed static arena, zero imports, `0` heap allocs |

Four architectural decisions create that difference:

1. **Deliberate 12-frame non-causal lookahead (`t - K`):** By holding `0..11` future frames (`0..183 ms` at `60 fps`) inside `Arena.rgba_ring`, the tracker inspects pixels *after* an occlusion clears, walks backward through buffered frames to recover missed detections, pins irreducible gaps with smooth Hermite boundary splines, preserves sharp corners during collisions or rocket ignition via 2-pass Iteratively Reweighted L1 (IRL1) Dual Projected-FISTA (exporting a convex primal-dual optimality certificate), and runs a symmetric centered 15-tap FIR Hilbert filter with zero group delay.
2. **From pixels to physical laws rather than bounding boxes:** Trajectory histories feed compact-support weak-form quadrature (`phi(u) = (1 - u^2)^p`) and STRidge/AICc sparse regression (`src/physics_id.rs`) to discover symbolic governing ODEs (distinguishing inertial flight, ballistic drag, powered rocket boost, parachute terminal descent, nonlinear pendulums, and impacts), exact AGM elliptic periods, two-body collision momentum/energy conservation, `40x22` Eulerian vorticity, and coherent FTLE ridges.
3. **Hard-real-time `#![no_std]` determinism:** Every pixel ring, pyramid, track slot, spectral workspace, and continuum field occupies fixed offsets computed at compile time. Nothing allocates, fragments, or calls back into the host operating system (`wasm-objdump` verified: `0` imports, `0` `memory.grow`, max shadow stack `13,712 / 16,384 B`).
4. **Honest measurement provenance (`SURVEYED` vs `ASSUMED` vs `OUT OF PLANE`):** Pixel kinematics stay strictly in pixel units until explicit calibration. Built-in clip presets carry explicit `ASSUMED` tags, 2-point rulers and 4-point DLT homographies transform vectors via the analytical perspective Jacobian `J_H`, camera pans automatically suspend ground-plane calibration, and projective depth near the horizon (`w < 0.14`) refuses to extrapolate unbounded metric speeds.

## Standalone modules worth reusing

Each subsystem is cleanly factored with narrow dependencies under the [Apache License 2.0](./LICENSE), so engineers can extract individual algorithms without adopting the whole browser deck:

| Need in your project | Reusable module | What it implements and why extract it |
| :--- | :--- | :--- |
| Track `1,000-1,536` featureless points in `< 2 ms` on CPU (PIV fluid particles, microscopy cells, LiDAR/radar dots) | [src/soa_tracks.rs](./src/soa_tracks.rs) | Lock-free Structure-of-Arrays pipeline: 16-bit Morton Z-order 2-pass radix sort, `3x3` CSR spatial neighborhood pruning (`21.94x` speedup over dense scan), integer `Q`-scaled Bertsekas forward-reverse `epsilon`-Auction with explicit coasting, and `8x8` Huber IC-LK sub-pixel refinement. |
| Turn sparse point velocities into fluid vorticity, FTLE ridges, rigid bodies, or vibration mode maps | [src/continuum_physics.rs](./src/continuum_physics.rs) | Pure safe Rust (`0` `unsafe` tokens): `k=6` spatial graph strain, Procrustes `SO(2)`/`SE(2)` connected-component clustering (`32` bodies, unwrapped `omega` rad/s), `40x22` Eulerian velocity gradient/curl/divergence, 2-bank Cauchy-Green FTLE, modal PLL resonators with nematic order `R_2`, and explicit per-subsystem validity masking (`0x00..0x1F`). |
| Discover governing ODEs or check conservation laws on noisy time series | [src/physics_id.rs](./src/physics_id.rs) | Weak-form integration avoids taking noisy finite-difference accelerations: STRidge + AICc model selection across `9` physical regimes, exact `f64` cyclic Jacobi Taubin vs algebraic Kasa circle fitting, `f64` Arithmetic-Geometric Mean pendulum elliptic integral `K(m)`, staggered `4D` Rosenstein Lyapunov divergence, Kuramoto order `R`, and blur-excluded two-body collision conservation. |
| Smooth noisy position signals without smearing step impacts or turns | `solve_noncausal_fista_irl1_target` in [src/lib.rs](./src/lib.rs) | Constant-time (`6`-iteration) Dual Projected-FISTA solver for weighted L1 second-difference trend filtering (`12`-sample window), computing an exact convex duality gap `P(X) - D(u)` and branching to IRL1 knot reweighting only when high-impulse corners exist. |
| Suppress camera flicker and track fast targets under pan/exposure changes with SIMD128 | [src/simd_ops.rs](./src/simd_ops.rs) & [src/lib.rs](./src/lib.rs) | `v128` kernels (zero `MaybeUninit`): orthogonal Rank-2 ECA background subspace projection (`36-64 dB` clutter cancellation), Catmull-Rom sub-pixel background pan advection via an 8-row L1D ring, radial de-slipped dual-anchor templates, and single-instruction `i32.clz` Haar-Morton binary-prefix ultrametric signatures (`padic_ultrametric_val`). |
| Build a zero-framework, zero-GC real-time browser instrument panel | [web/worker.js](./web/worker.js), [web/index.html](./web/index.html), [scripts/dev_server.mjs](./scripts/dev_server.mjs) | `OffscreenCanvas` WebGL2 instanced field compositor (`< 0.5 ms` per frame), batched Canvas2D scopes, `DeckUiFsm` finite-state reducer (`CLS = 0.000000`), SSRF-hardened FFmpeg fragmented-MP4 `MediaSource` live relay, `18,000 x 56` zero-GC typed recorder, and latency-compensated WebAudio phase/percussion synthesis. |

## Quickstart

Prerequisites: Rust toolchain with `wasm32-unknown-unknown`, `node` (`v22+`), and `python3`. Install `wabt` (`wasm-objdump`), `ffmpeg`, and Google Chrome to run the full MP4 calibration and headless Chrome E2E verification suites.

Compile `web/chronos_spectral.wasm`, run all static/mathematical/swarm gates (`--check`), and start the local server at `http://127.0.0.1:8080` (serving the interactive Split-Wipe Showcase at `/`, with the full 15-panel Command Deck at `/?view=deck` or `/deck.html`):

```bash
./build.sh
```

Run release build, `wasm-objdump` opcode/import/shadow-stack audit, mathematical counterexample gates (Gates A-P), the `5x2` interleaved SD/HD pillar benchmark, and Gate Q (`1,000+` target swarm & continuum verification) without launching a server:

```bash
./build.sh --check
```

Run the complete verification pipeline (`--check`, SD/HD MP4 CLEAR-MOT & physics calibration gates, and headless Chrome CDP Steps `1-8` covering the full Command Deck, Showcase Split-Wipe surface, and live fMP4 `MediaSource` camera-cut recovery):

```bash
./build.sh --e2e
```

## Command Deck tour

![Full 3-column CHRONOS-SPECTRAL Command Deck tracking solid-motor model rockets](./docs/assets/panels/01_deck_overview.png)

Every visual scope, HUD overlay, KPI state transition, and exported column is documented with high-DPI screenshots of each individual panel in the **[UI and Telemetry Guide](./docs/UI_TELEMETRY_GUIDE.md)**.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="./docs/assets/panels/05_split_pip_and_primary_kpis.png" alt="Split 1x Raw vs 25x Eulerian temporal delta picture-in-picture and primary KPIs" width="100%" /><br />
      <strong>Split <code>1x</code> Raw / <code>25x</code> Eulerian Audit &amp; Primary KPIs</strong><br />
      Target-locked <code>64x64</code> stabilized patch next to its <code>25x</code> temporal-contrast delta and binary-prefix signature (<code>v_2 = 10</code>).
    </td>
    <td width="50%" align="center">
      <img src="./docs/assets/panels/08_hilbert_quadrature_scope.png" alt="Auto-ranging Hilbert quadrature phase portrait and 90-degree separated waveforms" width="100%" /><br />
      <strong>Hilbert Quadrature Scope</strong><br />
      Auto-scaled polar orbit paired with 15-tap centered FIR quadrature waveforms (<code>f = 0.76 Hz</code>, coherence <code>76%</code>) gated on turnaround crossings.
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="./docs/assets/panels/11_physics_id_and_phase_portrait.png" alt="In-situ symbolic ODE identification and phase-space portrait on Galileo pendulum" width="100%" /><br />
      <strong>In-Situ Physics ID &amp; Conservation</strong><br />
      Discovers symbolic governing ODEs (<code>R^2 = 0.882</code>), exact AGM elliptic period (<code>7.282 s</code>), Jacobi Taubin curvature, and live phase/energy conservation traces.
    </td>
    <td width="50%" align="center">
      <img src="./docs/assets/panels/12_continuum_swarm_panel.png" alt="Continuum and swarm field telemetry with expanded expert solver diagnostics" width="100%" /><br />
      <strong>Continuum &amp; Swarm Field + Solver Diagnostics</strong><br />
      Separates kernel median latency (<code>1.70 ms</code>) from pipeline window, exposes click-to-inspect track state, vorticity/FTLE peaks, and <code>SE(2)</code> rigid bodies.
    </td>
  </tr>
</table>

## Certified performance and accuracy

All numbers below come from deterministic Node + Wasmtime/V8/Chrome harnesses on an Apple M4 Pro (`CHRONOS_CI_BUDGET_SCALE = 1.00`, `640x360` HD unless noted). Full methodology, ablations, dual-mode standby overhead proofs, and standalone CLI commands live in **[Verification and Benchmarks](./docs/VERIFICATION_AND_BENCHMARKS.md)**.

### High-density swarm and continuum performance (`Gate Q`)

Evaluated headless via `scripts/benchmark_1000.mjs` with zero ground-truth hints passed to Wasm (`21/21` Gate Q checks passing):

| Suite / Scenario | Peak / P10 Confirmed | Accuracy and Physics Verification | Mean / p50 Kernel | p95 Kernel |
| :--- | :--- | :--- | :--- | :--- |
| `Q1: vortex_swarm_1200` | `1,207` / `1,200` | `HOTA 99.9%`, `IDSW 0`, vorticity Pearson `r = 0.9957`, `ftleCells = 657` | `1.56 ms` / `1.48 ms` | `2.02 ms` |
| `Q2: murmuration_1200` | `1,219` / `1,208` | `HOTA 99.8%`, `IDSW 0`, `3/3` `SE(2)` bodies (`joint 100% / 96% / 94%`) | `1.61 ms` / `1.65 ms` | `1.97 ms` |
| `Q3: modal_beam_1152` | `1,143` / `1,135` | `3.000 Hz` (median `2.999 Hz`), coh `0.977`, `R_2 = 0.953`, `\|Delta phi - pi\| = 0.039` rad | `1.42 ms` / `1.44 ms` | `1.64 ms` |
| `Q4: starling_murmuration_1000.mp4` | `1,453` / `1,032` | Real `30 fps` H.264 flock (`movingDarkFeat1to1Prec = 70.3%`, `parityErr = 0`) | `1.76 ms` / `1.72 ms` | `2.12 ms` |
| `SWEEP: N = 1,536` | `1,536` / `1,473` | `Recall 100.0%`, `MOTA 100.0%`, `RMSE 0.059 px`, `HOTA 100.0%` | `1.45 ms` / `1.48 ms` | `1.81 ms` |

Running the standalone Q2 association ablation (`node scripts/benchmark_1000.mjs --only=Q2,ABLATION`) measures a **`21.94x` speedup** from 16-bit Morton Z-order sorting + `3x3` CSR neighborhood indexing (`13.631 ms` JS Nearest-Greedy down to `0.621 ms` JS Morton-CSR+Greedy), while the full SIMD128 Wasm pipeline (Morton-CSR + integer `epsilon`-Auction + `8x8` Huber IC-LK) lifts accuracy from `96.9%` Recall / `0.178 px` RMSE / `121` IDSW (`94.7%` HOTA) to **`100.0%` Recall / `0.039 px` RMSE / `0` IDSW (`99.8%` HOTA)** in `1.703 ms` median compute.

### Real-camera physics and occlusion calibration clips

`scripts/calibrate_mp4.mjs` checks CLEAR-MOT trajectories, negative non-periodic gates, SD/HD gravity scaling, and physical invariants across compressed open-licensed H.264 clips:

| Sample clip | Mathematical proof target | SD (`320x180`) Recall / RMSE | HD (`640x360`) Recall / RMSE | Verified HD physics proof |
| :--- | :--- | :--- | :--- | :--- |
| [juggling_ballistic.mp4](./web/samples/juggling_ballistic.mp4) (`25 fps`) | Mode A multi-arc `g_hat` and flight angle | `75.9%` / `9.30 px` | `77.6%` / `15.18 px` | `6`-arc Auto-`g` (`1.7%` SD/HD `2x` err), `0` false Hz |
| [roundabout_occlusion.mp4](./web/samples/roundabout_occlusion.mp4) (`24 fps`) | Mode A gantry bridge + homography `H` | `94.0%` / `9.99 px` | `93.0%` / `18.22 px` | `9f` Hermite bridges, `0` false pivots, `0` false Hz |
| [galileo_pendulum_peg.mp4](./web/samples/galileo_pendulum_peg.mp4) (`30 fps`) | Kasa virtual pivot & swing angle | `100.0%` / `7.81 px` | `100.0%` / `3.20 px` | `61f` synchronized virtual pivot (`0` IDSW) |
| [harmonic_metronome_phase.mp4](./web/samples/harmonic_metronome_phase.mp4) (`30 fps`) | Mode B Nikko `96 BPM` (`0.80 Hz`) + pivot | `89.0%` / `14.05 px` | `96.0%` / `24.64 px` | `108f` `0.82 Hz` lock (`SD 0.80 Hz`) + `110f` pivot |
| [carom_billiards_impulses.mp4](./web/samples/carom_billiards_impulses.mp4) (`30 fps`) | Mode C deflection, `v+/v-`, and `H` | `92.6%` / `9.54 px` | `93.3%` / `19.80 px` | `30` L1 strikes + `117.3 deg` (`SD 27.2 deg`) scatter (`0` false Hz) |

In paired sparse-mode parity testing across all 13 non-swarm sample clips (`52/52` SD/HD checks, `624` length-preselected trials, `467,448` warm frames), enabling `Auto-Density` swarm probing imposes only `0.00067 ms` (`< 1 microsecond`) pooled upper-95th-percentile overhead with `0` output mismatches against `Swarm: Off`.

## Documentation map

| Document | Diataxis type | What it covers |
| :--- | :--- | :--- |
| [Architecture and Math](./docs/ARCHITECTURE_AND_MATH.md) | Explanation | Arena memory map (`20.00 MB`), ECA subspace math, 12-frame lookahead, IRL1 FISTA, `i32.clz` Haar-Morton, Lanczos Fiedler, Morton-CSR Auction, weak-form STRidge/AICc, continuum mechanics, and homography rectification. |
| [UI and Telemetry Guide](./docs/UI_TELEMETRY_GUIDE.md) | Reference | Panel-by-panel manual with all 15 individual deck screenshots, `DeckUiFsm` states, dynamic KPI transitions, calibration rules, WebAudio synthesis, C-ABI memory offsets (`telemetry[0..1567]`, `swarm_telem[0..63]`), and `56`-column CSV/JSON export schema. |
| [Verification and Benchmarks](./docs/VERIFICATION_AND_BENCHMARKS.md) | How-to and Reference | CLI verification recipes (`--check`, `--e2e`, `calibrate_mp4.mjs`, `benchmark_1000.mjs`, `capture_docs_media.mjs`), static binary audit gates, Gates A-P, full extended-clip tables, dual-mode parity proof, and Q2 ablation metrics. |

## Repository layout

- [LICENSE](./LICENSE) and [NOTICE](./NOTICE): Apache License 2.0 terms for code, build/test scripts, docs, and `web/chronos_spectral.wasm`, plus Section 4(d) attribution notice separating standalone media clips.
- [Cargo.toml](./Cargo.toml) and [.cargo/config.toml](./.cargo/config.toml): Zero-dependency `cdylib` build (`opt-level = 3`, fat LTO, `panic = "abort"`, `+simd128`, `16 KiB` stack, `initial-memory = max-memory = 320` Wasm pages).
- [src/lib.rs](./src/lib.rs): `Arena` definition, C-ABI entrypoints, pixel-domain camera-cut/pan detector, 4-mode IMM + Jonker-Volgenant `16x64` matcher, backward `rgba_ring` re-tracking + Hermite bridges, 32-slot Dormant Re-ID archive, Lanczos Fiedler eigensolver, and Dual Projected-FISTA + multi-arc consensus Auto-`g`.
- [src/soa_tracks.rs](./src/soa_tracks.rs): `1,536`-slot `MicroTrackSoA`, SIMD128 micro-peak extraction, 16-bit Morton Z-order radix sort, sparse `CSR`, and integer `Q`-scaled Bertsekas `epsilon`-Auction + `8x8` Huber IC-LK.
- [src/continuum_physics.rs](./src/continuum_physics.rs): Zero-`unsafe` `k`-NN strain, `SE(2)` rigid-body clustering, `40x22` Eulerian velocity gradients, 2-bank Cauchy-Green FTLE, modal accelerometers (`R_2`), and validity masking.
- [src/physics_id.rs](./src/physics_id.rs): Weak-form STRidge/AICc symbolic ODE discovery, `f64` Jacobi Taubin circle fitting, AGM elliptic pendulum period, Rosenstein `4D` Lyapunov divergence, Kuramoto synchronization, and two-body collision conservation.
- [src/simd_ops.rs](./src/simd_ops.rs): `v128` SIMD kernels (`0` `MaybeUninit`) for RGBA unpacking, Rank-2 ECA projection, sub-pixel Catmull-Rom pan advection, Shi-Tomasi cornerness, `i32.clz` Haar-Morton wavelets, zero-mean gain-invariant patch SAD, and IC-LK refinement.
- [web/](./web/): Interactive Split-Wipe Showcase and Command Deck UI ([web/index.html](./web/index.html), [web/deck.html](./web/deck.html)), Worker + WebGL2 field compositor ([web/worker.js](./web/worker.js)), prebuilt `web/chronos_spectral.wasm`, and 14 open-licensed `640x360` MP4 clips ([web/samples/ATTRIBUTION.md](./web/samples/ATTRIBUTION.md)).
- [scripts/](./scripts/): Static/math/E2E gates ([build.sh](./build.sh), [scripts/e2e_browser.mjs](./scripts/e2e_browser.mjs), [scripts/calibrate_mp4.mjs](./scripts/calibrate_mp4.mjs), [scripts/benchmark_1000.mjs](./scripts/benchmark_1000.mjs)), HTTP 206 / fMP4 live relay server ([scripts/dev_server.mjs](./scripts/dev_server.mjs)), and documentation media generator ([scripts/capture_docs_media.mjs](./scripts/capture_docs_media.mjs)).
- [.github/workflows/ci.yml](./.github/workflows/ci.yml) and [.github/workflows/pages.yml](./.github/workflows/pages.yml): Full `./build.sh --e2e` verification CI pipeline plus automatic GitHub Pages static deployment of [web/](./web/) for public builds.

## Scope limits

- Behind a 100% opaque occluder where visibility weight `w_k = 0`, two pixel-identical targets that exchange velocities produce identical boundary observations to targets that pass straight through. Distinguishing hidden identical swaps requires distinct appearance signatures or non-crossing priors.
- Six Dual Projected-FISTA iterations run in bounded constant time per frame and export an explicit upper-bound duality gap `P(X) - D(u)` instead of iterating to machine precision.
- Real-camera MP4 files exercise the identical `<video>` to `createImageBitmap` to `OffscreenCanvas` RGBA to Wasm SIMD128 execution path as live `getUserMedia` feeds, deterministically testing H.264 compression and camera auto-exposure drift, but offline files do not measure physical camera-to-display photon latency on human subjects.
- Mode B strobe cues fire at forward-predicted quadrature phase offsets. Confirming biological tremor attenuation requires external closed-loop physiological measurement.

## License and attribution

All software source code, build and verification scripts, text documentation, and compiled WebAssembly artifacts (`web/chronos_spectral.wasm`) are licensed under the **[Apache License, Version 2.0](./LICENSE)** (see [NOTICE](./NOTICE)).

Standalone demonstration and benchmark MP4 clips under [web/samples/](./web/samples/), together with derived demonstration GIFs and video-frame screenshots under `docs/assets/demos/` and `docs/assets/panels/`, retain their respective Creative Commons (`CC0 1.0`, `CC BY 2.0/3.0/4.0`, `CC BY-SA 3.0 nl / 3.0 / 4.0`) and U.S. Government Public Domain terms and attributions documented in [web/samples/ATTRIBUTION.md](./web/samples/ATTRIBUTION.md).
