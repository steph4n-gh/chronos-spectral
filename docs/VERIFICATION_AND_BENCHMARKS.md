# Verification, audit gates, and benchmarks

Every architectural guarantee in CHRONOS-SPECTRAL is enforced automatically by deterministic verification gates across binary layout, mathematical counterexamples, real-camera compressed H.264 MP4s, dense `1,000+` target swarms, and headless Chromium DevTools end-to-end execution.

## How to run the verification suites

Ensure Rust (`wasm32-unknown-unknown`), Node.js (`v22+`), and `python3` are installed. Install `wabt` (`wasm-objdump`), `ffmpeg`, and Google Chrome to exercise the complete binary, video, and browser pipeline.

### Offline build, static audit, math gates, and Gate Q

```bash
./build.sh --check
```

Runs seven sequential offline stages:
1. Release compile of `chronos_spectral.wasm` (`opt-level = 3`, fat LTO, `codegen-units = 1`, `panic = "abort"`, `+simd128`).
2. `wasm-objdump` binary audit (`75` exports, `0` host imports, fixed `initial=448 max=448` linear memory, `0` `memory.grow` opcodes, `0` `call_indirect`, `0` call cycles, `0` allocator symbols, `13,776 B` max call-graph shadow stack below `16,384 B`, `101,655` SIMD128 ops, `8` `i32.clz` ops).
3. Mathematical counterexample gates (`Gates A-P`, including `N1` and `N2`).
4. `5x2` interleaved `320x180` SD vs `640x360` HD Blind Pixel Pillar benchmark.
5. `Gate Q` high-density `1,000+` target swarm and continuum physics suite (`scripts/benchmark_1000.mjs --gate`, `21/21` checks).
6. Wave-1 gates (`gate_i0_i4_ingest.mjs`, `gate_i1_cov_residual.mjs`, `gate_i5_bos.mjs`).
7. Wave-2 gates (`gate_i2_rs_welch.mjs`, `gate_i3_hodge.mjs`, `gate_i5b_conservation.mjs`).

### Full end-to-end pipeline (CI entrypoint)

```bash
./build.sh --e2e
```

Runs everything in `--check`, plus Gate `I6` physical capture fixtures (`scripts/gate_i6_captures.mjs`), SD and HD H.264 MP4 calibration gates (`scripts/calibrate_mp4.mjs`), and headless Chromium CDP full-stack UI and audio/video verification (`scripts/e2e_browser.mjs` Steps `1-10e`).

### Standalone calibration, Wave-1/2 gates, ablation, and media regeneration

```bash
# Core 5 ground-truth physics/occlusion clips (SD + HD)
node scripts/calibrate_mp4.mjs --gate

# Extended 8 real-camera HD showcases (rocketry, VTVL, motorway, chaos pendulum, metronomes, Newton's cradle)
node scripts/calibrate_mp4.mjs --extended --hd --gate

# All 13 non-swarm clips at both SD & HD paired across Swarm: Off vs Auto-Density (52 checks, 624 trials)
node scripts/calibrate_mp4.mjs --dual-mode --all --dual-res --gate

# Real-camera starling murmuration dense swarm gate (640x360 HD)
node scripts/calibrate_mp4.mjs --starling --hd --gate

# Full Gate Q suite plus 6-tier association & Spatial-CSR ablation table
node scripts/benchmark_1000.mjs --gate --ablation

# Standalone Wave-1 & Wave-2 metrology gates
node scripts/gate_i0_i4_ingest.mjs
node scripts/gate_i1_cov_residual.mjs
node scripts/gate_i5_bos.mjs
node scripts/gate_i6_captures.mjs
node scripts/gate_i2_rs_welch.mjs
node scripts/gate_i3_hodge.mjs
node scripts/gate_i5b_conservation.mjs

# Regenerate all Command Deck screenshots (docs/assets/panels/) and 4 demo GIFs (docs/assets/demos/)
node scripts/capture_docs_media.mjs
# Refresh PNG panel screenshots only (skips GIF re-encoding):
node scripts/capture_docs_media.mjs --panels-only
```

For slower shared cloud virtual machines, set `CHRONOS_CI_BUDGET_SCALE=3.5` (matching [ci.yml](../.github/workflows/ci.yml)). Bare-metal local verification enforces unrelaxed `CHRONOS_CI_BUDGET_SCALE=1.00`.

## Static binary and safety invariants

Measured directly on `web/chronos_spectral.wasm` (`2,474,879 B`):

| Property | Measured value | Gate threshold |
| :--- | :--- | :--- |
| Export count & host imports | `75` exports, `0` imports (`0 env`, `0 malloc/free`) | Strictly `0` host imports |
| Wasm linear memory | `initial=448 max=448` (`29,360,128 B`) | `448` fixed pages, `0` `memory.grow` |
| Arena size | `22,644,848 B` | `<= 24,117,248 B` (`Arena` static bound) |
| Static `.rodata` end | `59,216 B` | Unchanged `.rodata` ceiling |
| Static data end (`dualStaticEnd`) | `22,704,064 B` | `<= 24,248,320 B` |
| Max DFS shadow stack | `13,776 B` (`0` call cycles, `0` `call_indirect`) | `<= 14,336 B` soft bound, `<= 16,384 B` hardware bound |
| Emitted vector / bitwise ops | `101,655` SIMD128 ops, `8` `i32.clz` ops | Non-zero SIMD128 & hardware CLZ |
| Source memory hygiene | `0` `MaybeUninit` blocks across `src/*.rs`; exactly `6` `unsafe` tokens in [src/continuum_physics.rs](../src/continuum_physics.rs) (one SIMD128 `k`-NN scan) | Enforced by the `build.sh` source hygiene gate |

## Mathematical counterexample gates (`Gates A-P`)

| Gate | Mathematical invariant verified inside `build.sh --check` |
| :--- | :--- |
| `Gate A` | Shifted Lanczos solver with orthogonal restart matches exact spectrum of a `6`-node path graph (`lambda2 = 2 - sqrt(3)`) and degenerate reflection-symmetric clusters (`resNorm`). |
| `Gate B` | Spectral Fiedler ablation on Twin-Gray crossing clusters (`gamma = 0` produces cross-lane hijacks; `gamma = 0.55` yields `0` cross-lane hijacks). |
| `Gate C` | Dual Projected-FISTA maintains `0` NaNs, certified `P(X) - D(u)` duality gap bounds, and safe all-occluded clamping across `18`-frame blackouts. |
| `Gate D` | Anti-diagonal PCA projection and equalized `15`-tap FIR Hilbert quadrature pass `60 Hz` and `120 Hz` amplitude-ratio checks while gating out `K = 4 < 7`. |
| `Gate E` | Mode C physical strike deduplication emits exactly `1` percussive snap at `K = 10` and `K = 11`, followed by clean zeroed arena reset. |
| `Gate F` | Rank-2 ECA orthogonally cancels global affine gain/offset flicker and verifies `1,000` random triples against the 2-adic strong triangle inequality `d_2(A, C) <= max(d_2(A, B), d_2(B, C))`. |
| `Gate G` | Live runtime switching between `320x180` SD (`32x32` PiP) and `640x360` HD (`64x64` PiP) executes with `0` `memory.grow` ops and valid border-clamped RGBA output. |
| `Gate H` | Parabolic quadratic gravity (`g_hat`), CTRV curved occlusion coasting with zero false pivots or false Hz, `2x2` Kasa virtual pendulum pivot error `< 4.0 px`, and CAD collision deflection / restitution `v+/v-`. |
| `Gate I` | Explicit presentation-timestamp (`chronos_set_ingest_pts`) velocity invariance under frame drops, `2`-arc Auto-`g` consensus lock (`conf < 0.50 -> >= 0.50`), and automatic `0/1/2` camera pan/cut guards. |
| `Gate J` | Ego-motion pan-warped ECA suppression and mid-pan target birth custody across SD and HD. |
| `Gate K` | Slow sub-pixel camera pan clutter suppression and track custody across SD and HD. |
| `Gate L` | Connected-valley occlusion connectivity, gantry sign bridge, and rigid-body custody across SD and HD. |
| `Gate M` | Combined optical exposure step plus motion-blur compensated sub-pixel pan tracking. |
| `Gate N1` | Stationary exposure ramp transitioning cleanly into a `0 -> 1.45 px/f` urban pan ramp. |
| `Gate N2` | `SO(2)` arbitrary-angle (`68 deg` and `180 deg`) multi-lane occlusion invariance (`0` ID swaps). |
| `Gate O` | `16`-slot swarm lock (`12`-disc simultaneous crossings), SD/HD border-clamped PiPs, and `160`-frame (`5.3 s` @ `30 fps`) Dormant Archive re-identification under the original `track_id`. |
| `Gate P (P1-P5)` | `1,568`-float telemetry ABI finiteness & tail zeros (`P1`), damped elliptic pendulum `T(theta0)` vs exact `f64` AGM + damping `zeta`/`Q` + ballistic drag + negative CV divergence checks (`P2`), two-body collision `m2/m1` & `e_n` conservation + non-collision negative guard (`P3`), Kuramoto `R_sync` synchronization + Rosenstein `lambda_max` chaos ordering (`P4`), and `SD` vs `2x` `HD` physical invariance (`P5`). |

## Blind Pixel Pillar `5x2` SD/HD benchmark

Interleaved `5x` alternating runs (`320x180` SD and `640x360` HD) on the pixel-isolated dual-pillar crossing bench verify resolution-scaling determinism:

| Resolution | Static ECA clutter attenuation | Oracle ID swaps | Mean kernel time (`ChronosCI = 1.00`) | Budget |
| :--- | :--- | :--- | :--- | :--- |
| `320x180` SD | `> 64 dB` | `0` | `0.121 ms` | `< 0.35 ms` |
| `640x360` HD | `> 64 dB` | `0` | `0.324 ms` | `< 1.00 ms` |

## Real-camera MP4 ground-truth & extended physics suites

`scripts/calibrate_mp4.mjs` streams decoded raw RGBA frames through Wasm in `O(1)` memory.

### Five CLEAR-MOT ground-truth calibration clips (`--gate`)

| Sample clip | Proven target behavior | SD Recall / RMSE / Swaps | HD Recall / RMSE / Swaps | Verified physics & negative invariants |
| :--- | :--- | :--- | :--- | :--- |
| `juggling_ballistic.mp4` (`25 fps`) | Ballistic parabolas + hand tosses | `75.9%` / `9.30 px` / `3` | `77.6%` / `15.18 px` / `0` | `6`-arc Auto-`g` (`1.7%` SD/HD `2x` err), `0` false Hz |
| `roundabout_occlusion.mp4` (`24 fps`) | Overhead highway gantry occlusion | `94.0%` / `9.99 px` / `3` | `93.0%` / `18.22 px` / `4` | `9f` Hermite bridges, `0` false pivots, `0` false Hz |
| `galileo_pendulum_peg.mp4` (`30 fps`) | Interrupted pendulum + peg impact | `100.0%` / `7.81 px` / `0` | `100.0%` / `3.20 px` / `0` | `61f` Kasa/Taubin virtual pivot (`0` IDSW) |
| `harmonic_metronome_phase.mp4` (`30 fps`) | `96 BPM` (`0.80 Hz`) periodic arm | `89.0%` / `14.05 px` / `0` | `96.0%` / `24.64 px` / `0` | `108f` `0.82 Hz` lock (`SD 0.80 Hz`), `110f` pivot |
| `carom_billiards_impulses.mp4` (`30 fps`) | Multi-cushion elastic impacts | `92.6%` / `9.54 px` / `2` | `93.3%` / `19.80 px` / `2` | `30` L1 strikes, `117.3 deg` scatter (`0` false Hz) |

### Eight extended HD real-camera showcases (`--extended --hd --gate`)

| Sample clip | Duration / FPS | Enforced physics & negative verification criteria |
| :--- | :--- | :--- |
| `model_rocket_full_flights.mp4` | `30.9 s` / `25 fps` | `Boost`, `Burnout` constraint jump (`J2`), `Parachute_Terminal`, `Canopy` jump (`J3`) |
| `rocket_vtvl_full_flight.mp4` | `79.9 s` / `30 fps` | Sustained `Powered_Boost` across pad ignition, crane-boom crossing, and powered landing |
| `water_rocket_parachute_flights.mp4` | `30.2 s` / `25 fps` | Aerodynamic drag (`Ballistic_AeroDrag`) + canopy terminal recovery |
| `cologne_motorway_occlusion.mp4` | `45.0 s` / `30 fps` | `>= 4` persistent vehicles across overpass occlusion; `0` false boost/pendulum/collision/chaos events |
| `rally_ego_pan_tracking.mp4` | `29.0 s` / `30 fps` | Active panning camera tracking (`>= 18 dB` ECA); camera motion guard blocks false physics triggers |
| `chaotic_magnetic_pendulum.mp4` | `50.0 s` / `25 fps` | `>= 80%` lock consistency with positive Rosenstein Lyapunov divergence (`lambda_max > 0.15 s^-1`) |
| `coupled_metronomes_sync.mp4` | `55.0 s` / `30 fps` | `>= 2` synchronized metronomes yielding `kuramoto_osc_count >= 2`, `R_sync >= 0.45`, and `>= 30` valid Hz frames |
| `newtons_cradle_impulses.mp4` | `45.0 s` / `30 fps` | Causal two-body impulse conservation (`m2/m1`, `e_n`) with `>= 4` Mode C impact events |

### Sparse-mode standby parity & overhead gate (`--dual-mode --all --dual-res --gate`)

Evaluates paired interleaved runs comparing `Swarm: Off` (`swarmMode = 0`) against `Auto-Density` (`swarmMode = 1`) on sparse footage (`52/52` per-clip & pooled checks across all `13` non-swarm clips x `SD/HD`, `624` length-preselected trials totaling `467,448` warm frames):
- **Output parity:** `0` trajectory or telemetry output mismatches (`swarmActive = 0` cleanly throughout sparse playback).
- **Pooled overhead upper-95th bound:** `0.00067 ms` (`< 0.00500 ms` threshold) when auto-density coarse probing is idle/standby.

## High-density continuum and swarm benchmark (`Gate Q`)

Evaluated headless in `640x360` HD via `scripts/benchmark_1000.mjs --gate` (`21/21` checks passing on Apple M4 Pro, `CHRONOS_CI_BUDGET_SCALE = 1.00`):

| Scenario | Peak / P10 Confirmed | Ground-truth MOT / Physics agreement | Mean / p50 Kernel | p95 Kernel |
| :--- | :--- | :--- | :--- | :--- |
| `Q1: vortex_swarm_1200` | `1,207` / `1,200` | `HOTA 99.9%`, `IDSW 0`, vorticity `r = 0.9957`, `ftleCells = 657` | `1.56 ms` / `1.48 ms` | `2.02 ms` |
| `Q2: murmuration_1200` | `1,219` / `1,208` | `HOTA 99.8%`, `IDSW 0`, `3/3` `SE(2)` bodies (`joint 100/96/94%`) | `1.61 ms` / `1.65 ms` | `1.97 ms` |
| `Q3: modal_beam_1152` | `1,143` / `1,135` | `3.000 Hz` (`med 2.999 Hz`), coh `0.977`, `R_2 = 0.953`, phase err `0.039` rad | `1.42 ms` / `1.44 ms` | `1.64 ms` |
| `Q4: starling_murmuration_1000.mp4` | `1,453` / `1,032` | `movingDarkFeat1to1Prec = 70.3%`, `parityErr = 0`, `dupTid = 0` | `1.76 ms` / `1.72 ms` | `2.12 ms` |
| `ANALYTIC` (13 cases) | Flow & rigid checks | Uniform translation, pure rotation (`omega`), strain/shear, saddle FTLE, modal split | Per-case analytic gate | `< 2.00 ms` |
| `SWEEP` (`N = 512, 1000, 1200, 1536`) | up to `1,536` / `1,473` | At `N = 1,536`: `Recall 100.0%`, `MOTA 100.0%`, `RMSE 0.059 px`, `HOTA 100.0%` | `1.45 ms` / `1.48 ms` at `1,536` | `1.81 ms` |

### `Q2 (murmuration_1200)` component ablation

Measured directly via `node scripts/benchmark_1000.mjs --only=Q2,ABLATION` (`90` frames @ `30 fps`, `640x360` HD; note running `--only=Q2,ABLATION` isolated from `Q1` warmup showed `warmMean = 1.850 ms`, `p50 = 1.703 ms`, and `p95 = 2.700 ms` cold V8 tier-up on `Q2` compared to warmed `--gate` `warmMean = 1.61 ms`, `p95 = 1.97 ms`):

| Configuration | Recall | Precision | RMSE | IDSW | Frag | HOTA | Mean / p50 Kernel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `JS Nearest-Greedy` (baseline `O(N^2)`) | `96.9%` | `98.0%` | `0.178 px` | `121` | `2,992` | `94.7%` | `13.631 ms` / `13.209 ms` |
| `JS Morton-CSR + Greedy` | `96.9%` | `98.0%` | `0.178 px` | `121` | `2,992` | `94.7%` | `0.621 ms` / `0.617 ms` |
| `JS Morton-CSR + Auction` | `96.9%` | `98.0%` | `0.179 px` | `125` | `2,992` | `94.7%` | `0.732 ms` / `0.618 ms` |
| `JS + Static-Mask` (proxy) | `89.0%` | `99.3%` | `0.189 px` | `1,157` | `6,524` | `84.0%` | `1.387 ms` / `1.050 ms` |
| `JS + Mask + Actual-6NN` (proxy) | `89.0%` | `99.3%` | `0.189 px` | `1,153` | `6,524` | `83.9%` | `2.172 ms` / `1.687 ms` |
| `JS + Mask + 6NN + 1StepHuber` (proxy) | `89.0%` | `99.3%` | `0.190 px` | `1,137` | `6,514` | `84.0%` | `3.717 ms` / `2.205 ms` |
| **`Wasm Morton-CSR + Auction + IC-LK`** (production) | **`100.0%`** | **`99.7%`** | **`0.039 px`** | **`0`** | **`3`** | **`99.8%`** | **`1.850 ms` / `1.703 ms`** |

Key measured deltas:
- **Spatial-CSR Speedup (`13.631 ms -> 0.621 ms`):** **`21.94x` faster** association from 16-bit Morton Z-order sorting + `3x3` sparse CSR neighborhood indexing.
- **Full Production Kernel vs Baseline:** Reduces RMSE by `4.56x` (`0.178 px -> 0.039 px`), cuts identity swaps from `121` to `0`, eliminates `99.9%` of fragmentations (`2,992 -> 3`), and reaches `99.8%` HOTA across `1,208` birds and `3` rotating rigid bodies in `1.703 ms` median compute.

## Wave-1 & Wave-2 metrology gates (`Gates I0-I6`)

| Gate | Script | Verified mathematical, physical, and refusal invariants |
| :--- | :--- | :--- |
| `I0` | [scripts/gate_i0_i4_ingest.mjs](../scripts/gate_i0_i4_ingest.mjs) | `1,000/1,000` frame-ring commits prior to throttled solves, `D4` ROI orientation propagation, determinism, and invalid-ingest refusal. |
| `I4` | [scripts/gate_i0_i4_ingest.mjs](../scripts/gate_i0_i4_ingest.mjs) | `1,000`-frame `640x360` HD `K=0` textured BOS PCG + dense swarm budget compliance (`scale = 1.00`) and standby isolation. |
| `I1` | [scripts/gate_i1_cov_residual.mjs](../scripts/gate_i1_cov_residual.mjs) | `2x2` inverse-Hessian track covariance, LOO flow residual calibration, pinhole looming suppression, and Auto-Hero candidate promotion. |
| `I5` | [scripts/gate_i5_bos.mjs](../scripts/gate_i5_bos.mjs) | `128x64` manufactured BOS inhomogeneous Neumann Poisson accuracy, curl consistency check, gauge-ROI offset invariance, and shock refusal. |
| `I6` | [scripts/gate_i6_captures.mjs](../scripts/gate_i6_captures.mjs) | `bos_thermal_plume.mp4` (`12.55x` SNR over `0.0530 px` quiet floor), `perspective_flyby_decoy.mp4` (`0` false maneuvering alarms), and `bos_handheld_failure.mp4` tripod-shock detection. |
| `I2` | [scripts/gate_i2_rs_welch.mjs](../scripts/gate_i2_rs_welch.mjs) | Rolling-shutter `D4` scan inversion, sinc exposure equalization and `EXPOSURE_NULL` (`0x40`) guard, optical Nyquist aliasing guard, `K = 1` vs `K >= 8` Bonferroni Welch trap, and fail-closed `CLOCK_SKEW` / `NO_PTS` / `NO_SCAN_DIR` / `NOT_NATIVE_1TO1` / `AXIS_TRANSPOSED` refusals. |
| `I3` | [scripts/gate_i3_hodge.mjs](../scripts/gate_i3_hodge.mjs) | `40x22` mesh Hodge `2`-complex (`(880, 1698, 819)`), orthogonal exact/coexact/harmonic solves (`orthEcRel < 1e-6`), enclosed-hole `beta_1` vs open-port flooding, and `UNCONV` non-convergence gating. |
| `I5b` | [scripts/gate_i5b_conservation.mjs](../scripts/gate_i5b_conservation.mjs) | Non-circular four-branch (`H0_KEEP`, `H0_SWAP`, `H1_KEEP`, `H1_SWAP`) occluded pair conservation distinguishing `UNIQUE`, `AMBIGUOUS`, and `VIOLATED`. |

## Headless Chrome CDP full-stack E2E (`scripts/e2e_browser.mjs`)

Exercises the full browser application against `scripts/dev_server.mjs` over DevTools Protocol (`Steps 1-10e`, enforcing zero page/worker exceptions and `CLS = 0.000000` at every transition):

- **Steps 1-2:** Boot (`0 HEAP ALLOC / 0 IMPORTS`, `448 PAGES / 28.00 MB`), `320x180` SD <-> `640x360` HD live resolution switch, `16`-slot Split-PiP Roster activation, click-to-pin target lock, and zero layout shift on pin/unpin/mode toggles.
- **Step 3:** Real H.264 MP4 calibration (`juggling_ballistic.mp4` exact `+1f` step + paused Auto-`g`, `roundabout_occlusion.mp4` Hermite bridges, `harmonic_metronome_phase.mp4` `~0.80 Hz` lock, `galileo_pendulum_peg.mp4` pivot lock, `carom_billiards_impulses.mp4` strikes).
- **Step 4:** Accessible metric calibration provenance (`ASSUMED` vs `SURVEYED`), `Esc` atomic cancel preserving prior calibration, zero-horizon singular homography guard, and `56`-column CSV / `v1.5.0` JSON export validation.
- **Steps 5-6:** Live Tier-3 FFmpeg fMP4 MSE streaming across a synthetic hard camera cut (`sim_live_cut_demo` auto-reset proof), Mode B stimulus latency, Mode C strikes, dense `vortex_1200` + `starling_1000` swarm HUD inspection (`inspect_point` `inspectSeq` correlation), `DeckUiFsm` Hero-16 toggle (`H`), `16:9` fullscreen viewport (`F`), and keyboard shortcut isolation.
- **Step 7:** Relay security and resource hardening (zero leaked file descriptors, multi-hop DNS rebinding / RFC1918 SSRF blocks returning `403`, rate limit / process cap checks, and bounded `<= 3` MSE reconnect retries).
- **Step 8:** Default Split-Wipe Showcase surface (`starling_1000`, `model_rockets`, `bos_plume` presets), drag handle interaction, Showcase <-> Deck switching, and `CLS = 0.000000`.
- **Step 9:** Wave-1 Deck controls (`1:1` Native ROI crop/orientation, `bosModeSelect` + `bos_thermal_plume.mp4` READY refraction overlay, `bos_handheld_failure.mp4` tripod-motion warning) and `CLS = 0.000000`.
- **Steps 10 & 10e:** Wave-2 `40x22` Mesh Hodge (`vortex_1200`, `coexactFrac > 0.85`, `orthEcRel < 1e-6`), RS + Welch (`880 Hz`, `K >= 8`) + `EXPOSURE_NULL` warning banner, occluded equal-mass `AMBIGUOUS` conservation, plus native Chrome webcam/Web Audio Worker acknowledgment checks and 40 mocked audio/`currentTime`/`getOutputTimestamp`/`rVFC` `captureTime` clock-skew lifecycle scenarios (`CLS = 0.000000`).
