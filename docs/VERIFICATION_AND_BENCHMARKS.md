# Verification, audit gates, and benchmarks

Every architectural guarantee in CHRONOS-SPECTRAL is enforced automatically by deterministic verification gates across binary layout, mathematical counterexamples, real-camera compressed H.264 MP4s, dense `1,000+` target swarms, and headless Chromium DevTools end-to-end execution.

## How to run the verification suites

Ensure Rust (`wasm32-unknown-unknown`), Node.js (`v22+`), and `python3` are installed. Install `wabt` (`wasm-objdump`), `ffmpeg`, and Google Chrome to exercise the complete binary, video, and browser pipeline.

### Offline build, static audit, math gates, and Gate Q

```bash
./build.sh --check
```

Runs eight sequential offline stages:
1. Release compile of `chronos_spectral.wasm` (`opt-level = 3`, fat LTO, `codegen-units = 1`, `panic = "abort"`, `+simd128`).
2. `wasm-objdump` binary audit (`75` exports, `0` host imports, fixed `initial=448 max=448` linear memory, `0` `memory.grow` opcodes, `0` `call_indirect`, `0` call cycles, `0` allocator symbols, `13,776 B` max call-graph shadow stack below `16,384 B`, `101,655` SIMD128 ops, `8` `i32.clz` ops).
3. Mathematical counterexample gates (`Gates A-P`, including `N1` and `N2`).
4. `5x2` interleaved `320x180` SD vs `640x360` HD Blind Pixel Pillar benchmark.
5. `Gate Q` high-density `1,000+` target swarm and continuum physics suite (`scripts/benchmark_1000.mjs --gate`, `21/21` checks).
6. Wave-1 gates (`gate_i0_i4_ingest.mjs`, `gate_i1_cov_residual.mjs`, `gate_i5_bos.mjs`).
7. Wave-2 gates (`gate_i2_rs_welch.mjs`, `gate_i3_hodge.mjs`, `gate_i5b_conservation.mjs`).
8. `Gate S` SIMD layer conformance (`scripts/gate_simd_conformance.mjs`): every op of the `src/simd/` layer (`125` ops, `206` entries including every const-generic instantiation the kernel uses) run as real Wasm SIMD128 in V8 against the portable `simd::scalar` reference over edge-value and seeded random lanes (`0` mismatches required, `9` deliberately broken negative controls must each be detected), then the same suite natively (`cargo test --target host-tuple`) and the native library link (`native/Cargo.toml`), both with zero warnings.

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

# SIMD layer conformance: Wasm SIMD128 (V8) vs simd::scalar, then the native host suite
node scripts/gate_simd_conformance.mjs
cargo test --target host-tuple
cargo test --manifest-path native/Cargo.toml --target host-tuple

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

## Kernel record-and-replay traces (`CSTRACE v1`)

Whole-kernel parity between the wasm build and a native build is checked by recording what the existing harnesses do to the kernel, then replaying it without any JavaScript. The scene generators are not ported; their effect on the kernel is recorded.

| File | Role |
| :--- | :--- |
| [scripts/trace_hook.mjs](../scripts/trace_hook.mjs) | Node preload. Inert unless `CHRONOS_TRACE_DIR` is set; then every kernel instance (any module exporting `chronos_init`) is recorded to one trace file, with no change to the harness. |
| [scripts/trace_replay.mjs](../scripts/trace_replay.mjs) | Reference replayer: fresh instance, apply writes, make calls, check every return value and region digest. |
| [scripts/record_canonical_traces.mjs](../scripts/record_canonical_traces.mjs) | Records the canonical scenario set (`--list` shows it). |
| [scripts/trace_lib.mjs](../scripts/trace_lib.mjs) | Shared code: `XXH32`, wasm section parser, region table, format reader. |
| [scripts/trace_tool.mjs](../scripts/trace_tool.mjs) | `info` dump and the negative-control mutations (`mutate-write`, `mutate-arg`). |

```bash
# Record one harness
CHRONOS_TRACE_DIR=/tmp/traces node --import ./scripts/trace_hook.mjs scripts/gate_i3_hodge.mjs
# ...or any command line, including child processes
CHRONOS_TRACE_DIR=/tmp/traces NODE_OPTIONS="--import=$PWD/scripts/trace_hook.mjs" node scripts/gate_i5_bos.mjs
# Record the canonical set outside the repo, then replay all of it
node scripts/record_canonical_traces.mjs --out=/tmp/chronos-traces --replay
node scripts/trace_replay.mjs /tmp/chronos-traces/*/
node scripts/trace_tool.mjs info /tmp/chronos-traces/hodge/<file>.cstrace
```

Hook environment: `CHRONOS_TRACE_LABEL` (file prefix), `CHRONOS_TRACE_MAX_BYTES` (per-instance cap, default 4 GiB), `CHRONOS_TRACE_TOTAL_MAX_BYTES` (per-process cap, default 16 GiB), `CHRONOS_TRACE_MAX_INSTANCES`, `CHRONOS_TRACE_INSTANCES` (for example `0-3,100`), `CHRONOS_TRACE_STRICT=1` (snapshot and digest after every call, getters included), `CHRONOS_TRACE_MEMDIGEST=N` (validation: every Nth digested call also carries an `XXH32` of all memory above the shadow stack), `CHRONOS_TRACE_LOG=0`. When a cap is hit, the trace gets an `END` record at the last whole call and the program continues unrecorded. Each directory also gets an `index.jsonl` line per trace. Latency gates run slower while recording: the canonical recorder sets `CHRONOS_CI_BUDGET_SCALE` (default `32`) and leaves budgets unchanged. Its own caps default to 1 GiB per instance and 8 GiB for the whole set. Traces are temporary artifacts. The recorder refuses an output directory inside the repository.

### How the hook works

Per instance, the hook keeps a full snapshot of linear memory taken after the previous call. Before each export call it compares memory with the snapshot (`Buffer.compare` over 64 KiB blocks and 4 KiB chunks, then 32-bit words, trimmed to exact bytes) to find every byte the host changed. The kernel's own writes are already in the snapshot, so they never appear as host writes. Each changed range is named by **region + offset**, never by raw address. If some changed bytes fall outside every region, the hook checks whether memory now equals the fresh-instance image plus region writes. That is what `calibrate_mp4.mjs` does when it resets a pooled instance by restoring the first 64 KiB and zeroing the rest. If so, the change is recorded as `RESET` followed by those region writes. Anything else outside every region becomes an `UNMAPPED` record and is reported on stderr. The export is called with arguments converted exactly as the JS API converts them (`ToInt32`, `Math.fround`, `ToNumber`). After the call, the snapshot is refreshed and the `XXH32` digest of every digested region is recorded. Getters (`chronos_get_*`, `padic_ultrametric_val`) have their host writes captured like any call, but carry no digests; after a getter only the shadow stack (`[0, __stack_pointer init)`) is re-synchronised. Strict mode confirms getters write nothing else.

### Region table

The table lives in `REGION_SPECS` in [scripts/trace_lib.mjs](../scripts/trace_lib.mjs). Each entry names the getter export and the Rust backing field. The byte length comes from that field's declared array type in `src/*.rs`, evaluated from the Rust `const` items (and `size_of::<IngestSlotMeta>() == 96` for `ring_meta`). Element type and direction come from the getter's Rust return type: `*const` is `out`, `*mut` is `both`, and `in` marks the staging buffers `ingest_rgba`, `audio_staging`, `hodge_inject` and `conservation_input`. Building the table fails if a pointer-returning export in `src/lib.rs` has no entry. At startup each region's pointer is checked for alignment and for lying inside memory, above the shadow stack.

There are 47 regions: `ring_slot[0..12)` (the 12 RGBA ring slots, `chronos_get_ring_slot_ptr(k)`), the two moving aliases `ingest_rgba` and `display_rgba` (host writes can be addressed through them, but they are not digested separately), and 33 fixed regions: `eca_res`, `pip_rgba`, `all_pip_rgba`, `slot_meta`, `laplacian`, `fiedler`, `telemetry`, `micro_tracks`, `continuum_field`, `rigid_bodies`, `swarm_telemetry`, `modal_field`, `ring_meta`, `row_luma`, `wave1_telemetry`, `bos_telemetry`, `bos_planes`, `bos_valid`, `bos_rgba`, `micro_cov`, `micro_residual`, `micro_loom`, `micro_quality`, `hero_candidates`, `wave2_telemetry`, `audio_staging`, `vibro_telemetry`, `vibro_planes`, `hodge_telemetry`, `hodge_planes`, `hodge_inject`, `conservation_telemetry` and `conservation_input`. Each digested call covers 45 regions, 12,715,152 bytes. The full table, with getter, argument, element type, length and flags, is embedded in every trace header, so a replayer never parses Rust.

### Canonical set (measured)

`record_canonical_traces.mjs` with default caps: 11 scenarios, 29 traces, 2.54 GB, recorded in 3 min 46 s (including replay) on Apple Silicon. Replaying all 29 takes 39 s with 0 mismatches.

| Scenario | Harness | Traces | Calls | Host writes | Size |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `hero_sd` | `calibrate_mp4.mjs --clip=juggling_ballistic` (SD) | 2 | 1,126 | 145,042 + 1 reset | 45.2 MB |
| `hero_hd` | `calibrate_mp4.mjs --hd --clip=juggling_ballistic` | 2 | 1,126 | 613,215 + 1 reset | 162.8 MB |
| `hero_suite` | build.sh `[3/5]` Gates A-P + SD/HD pillar trials | 1 | 31,371 | 754,082 | 677.6 MB |
| `mp4_occlusion` | `calibrate_mp4.mjs --clip=roundabout_occlusion` | 2 | 1,241 | 821 + 1 reset | 58.9 MB |
| `swarm_1200` | `benchmark_1000.mjs --only=Q1` | 1 | 719 | 636,687 | 50.0 MB |
| `bos` | `gate_i5_bos.mjs` | 1 | 2,464 | 109,002 + 1 unmapped | 110.2 MB |
| `hodge` | `gate_i3_hodge.mjs` | 1 | 175 | 95,067 | 19.2 MB |
| `rs_vibrometry` | `gate_i2_rs_welch.mjs` | 1 | 6,739 | 401,893 | 1,054.8 MB |
| `conservation` | `gate_i5b_conservation.mjs` | 1 | 1,424 | 15,662 | 14.8 MB |
| `covariance` | `gate_i1_cov_residual.mjs` (10 of 406 instances) | 10 | 4,742 | 338,512 | 127.5 MB |
| `ingest_roi` | `gate_i0_i4_ingest.mjs` | 7 | 7,707 | 351,051 + 3 unmapped | 221.0 MB |

Each `calibrate_mp4.mjs` scenario also has a 0-call trace for its probe instance. Recording costs about 4 ms per kernel call (SD hero tick `0.12 -> 4.1 ms`, Q1 swarm tick `1.3 -> 5.7 ms`). Strict mode with `--memdigest=1` takes about 2.4 times as long as default recording.

Non-portable writes in the set: `gate_i5_bos.mjs` and `gate_i0_i4_ingest.mjs` find the private `Arena` fields `ring_cursor`, `res_mode` and `frame_idx` by scanning memory and overwrite them to test frame-index wraparound. No getter exposes those fields, so the writes are `UNMAPPED`. They replay against the wasm, but a native build can follow those three traces only up to call `#209` (`bos`), `#58` (`ingest_roi` instance 2) and `#710` (`ingest_roi` instance 6). Making them portable would need a test-only export for those fields.

### Format

All integers are little-endian. Header:

| Field | Type |
| :--- | :--- |
| magic | 8 bytes `CSTRACE\0` |
| version major, minor | `u16`, `u16` (`1`, `0`) |
| header length in bytes (from offset 0) | `u32` |
| sha256 of the wasm binary | 32 bytes |
| linear memory size in bytes | `u32` |
| flags (bit 0: strict, bit 1: whole-memory digests present) | `u32` |
| metadata JSON (label, argv, pid, instance ordinal, node version, ...) | `u32` length + UTF-8 |
| function table: `u16` count, then per export: name (`u16` length + UTF-8), `u8` nparams, param types, `u8` nresults, result types, `u8` flags | value types are wasm codes `0x7f` i32, `0x7e` i64, `0x7d` f32, `0x7c` f64; flag bit 0 = returns a pointer, bit 1 = getter (no digests) |
| region table: `u16` count, then per region: name (`u16` length + UTF-8), `u16` getter function index, `u8` has-argument, `u32` argument, `u8` element (`1` u8, `2` u32, `3` f32), `u32` byte length, `u8` flags | flag bit 0 = host-writable (`*mut`), bit 1 = digested, bit 2 = alias, bit 3 = input-only |

Records follow, in session order, each starting with a one-byte tag:

| Tag | Record | Payload |
| :--- | :--- | :--- |
| `0x57` `W` | host write | `u16` region index, `u32` offset in region, `u32` length, bytes |
| `0x55` `U` | unmapped host write | `u32` raw wasm address, `u32` length, bytes |
| `0x52` `R` | reset | none: memory now equals the fresh-instance image (the `W` records that follow are relative to it) |
| `0x43` `C` | export call | `u32` call index (from 0), `u16` function index, arguments as raw bit patterns (4 bytes for i32/f32, 8 for i64/f64), `u8` status (`0` returned, `1` trapped/threw), result bits if status `0` and the function has a result, `u8` digest flags, then if bit 0 is set one `u32` `XXH32` per digested region in table order, then if bit 1 is set one `u32` `XXH32` of memory `[stackTop, memory size)` (validation traces only; `stackTop` is in the metadata JSON) |
| `0x45` `E` | end | `u8` reason (`0` process exit, `1` instance cap, `2` total cap, `3` instance garbage-collected), `u32` calls, `u32` host writes, `u32` unmapped writes |

Replay rules (what [scripts/trace_replay.mjs](../scripts/trace_replay.mjs) does, and what a native replayer must do):

1. For `W`, resolve the region's address now by calling its getter (with its argument), then copy the bytes to address + offset. `ingest_rgba` moves with the ring cursor, so it must be resolved at that point in the sequence, not cached.
2. For `C`, call the export with the decoded arguments and compare status and result. f32 and f64 results compare as bit patterns, except that NaN matches any NaN, because the JS API cannot observe NaN payloads. Pointer results (function flag bit 0) are addresses: they are compared only when replaying the same wasm, and a native replayer skips them.
3. When digest bit 0 is set, hash every digested region and compare it with the recorded value. A mismatch names the call index, the export and the region. Bit 1 (whole-memory digest) is meaningful only against the identical wasm binary. A native replayer reads past it.
4. For `R`, restore the fresh-instance state. The JS replayer copies back the memory image it saved at instantiation. A native replayer starts a fresh kernel (a new process, or a reset entry point if one is added).
5. A `U` record is a raw-address write. It replays only against the identical wasm binary. For a native build, the trace is portable up to the call that follows the first `U` record.
6. A file without an `E` record (process killed) replays up to its last whole record.

### Digest hash: `XXH32`, seed `0`

The digest is standard `XXH32` (xxHash 32-bit) over the region's raw bytes, seed `0`. With all arithmetic mod 2^32, `rotl` = rotate left, and `P1 = 0x9E3779B1`, `P2 = 0x85EBCA77`, `P3 = 0xC2B2AE3D`, `P4 = 0x27D4EB2F`, `P5 = 0x165667B1`:

- If `len >= 16`: `v1 = seed + P1 + P2`, `v2 = seed + P2`, `v3 = seed`, `v4 = seed - P1`. For each 16-byte stripe, read four little-endian `u32` lanes and set `vi = rotl(vi + lane_i * P2, 13) * P1`. Then `h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18)`. Otherwise `h = seed + P5`.
- `h += len`.
- For each remaining 4-byte little-endian word `w`: `h = rotl(h + w * P3, 17) * P4`.
- For each remaining byte `b`: `h = rotl(h + b * P5, 11) * P1`.
- Avalanche: `h ^= h >> 15; h *= P2; h ^= h >> 13; h *= P3; h ^= h >> 16`.

Test vectors (ASCII input, seed `0`), checked against the reference `xxhash` implementation:

| Input | `XXH32` |
| :--- | :--- |
| `""` | `02cc5d05` |
| `"a"` | `550d7456` |
| `"abc"` | `32d153ff` |
| `"xxhash"` | `9a95b70e` |
| `"0123456789abcdef"` (16 bytes, one stripe) | `c2c45b69` |
| `"0123456789abcdefg"` (17 bytes, stripe + tail byte) | `cc79b217` |
| `"Nobody inspects the spammish repetition"` | `e2293b2f` |
