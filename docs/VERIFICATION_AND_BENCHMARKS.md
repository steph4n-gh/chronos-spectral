# Verification, audit gates, and benchmarks

Every architectural guarantee in CHRONOS-SPECTRAL is enforced automatically by deterministic verification gates across binary layout, mathematical counterexamples, real-camera compressed H.264 MP4s, dense `1,000+` target swarms, and headless Chromium DevTools end-to-end execution.

## How to run the verification suites

Ensure Rust (`wasm32-unknown-unknown`), Node.js (`v22+`), and `python3` are installed. Install `wabt` (`wasm-objdump`), `ffmpeg`, and Google Chrome to exercise the complete binary, video, and browser pipeline.

### Offline build, static audit, math gates, and Gate Q

```bash
./build.sh --check
```

Runs nine sequential offline stages:
1. Release compile of `chronos_spectral.wasm` (`opt-level = 3`, fat LTO, `codegen-units = 1`, `panic = "abort"`, `+simd128`).
2. `wasm-objdump` binary audit (`75` exports, `0` host imports, fixed `initial=448 max=448` linear memory, `0` `memory.grow` opcodes, `0` `call_indirect`, `0` call cycles, `0` allocator symbols, `13,776 B` max call-graph shadow stack below `16,384 B`, `101,655` SIMD128 ops, `8` `i32.clz` ops).
3. Mathematical counterexample gates (`Gates A-P`, including `N1` and `N2`).
4. `5x2` interleaved `320x180` SD vs `640x360` HD Blind Pixel Pillar benchmark.
5. `Gate Q` high-density `1,000+` target swarm and continuum physics suite (`scripts/benchmark_1000.mjs --gate`, `21/21` checks).
6. Wave-1 gates (`gate_i0_i4_ingest.mjs`, `gate_i1_cov_residual.mjs`, `gate_i5_bos.mjs`).
7. Wave-2 gates (`gate_i2_rs_welch.mjs`, `gate_i3_hodge.mjs`, `gate_i5b_conservation.mjs`).
8. `Gate S` SIMD layer conformance (`scripts/gate_simd_conformance.mjs`): every op of the `src/simd/` layer (`125` ops, `206` entries including every const-generic instantiation the kernel uses) run as real Wasm SIMD128 in V8 against the portable `simd::scalar` reference over edge-value and seeded random lanes (`0` mismatches required, `9` deliberately broken negative controls must each be detected), then the same suite natively (`cargo test --target host-tuple`) and the native library link (`native/Cargo.toml`), both with zero warnings.
9. `Gate N` native/wasm whole-kernel parity (`scripts/gate_native_parity.mjs`): records a fresh `104 MB` trace set from the wasm (real-MP4 sparse tracking, Hodge, conservation, the first `24 MiB` of the `1,200`-target swarm; `3,088` calls) and replays it on the native build with overflow checks on: every return value and every region digest must match bit for bit, and a corrupted copy of one trace must be rejected at the right call (see [Native replay driver](#native-replay-driver-nativereplay)).

### Full end-to-end pipeline (CI entrypoint)

```bash
./build.sh --e2e
```

Runs everything in `--check`, plus Gate `I6` physical capture fixtures (`scripts/gate_i6_captures.mjs`), SD and HD H.264 MP4 calibration gates (`scripts/calibrate_mp4.mjs`), and headless Chromium CDP full-stack UI and audio/video verification (`scripts/e2e_browser.mjs` Steps `1-10e`).

### Native backends only (`--native-check`)

```bash
./build.sh --native-check
```

Runs stages 1, 2, 8 and 9 of `--check`: the release build, the `wasm-objdump` audit, `Gate S` and `Gate N`. It skips the Wasm-only suites (stages 3 to 7). This is what the macOS arm64 CI job runs to prove the NEON backend. It needs no Chrome. Without `ffmpeg` on `PATH` it skips Gate N's real-MP4 scenario (`hero_sd`) and says so; `--check` and `--e2e` always run every scenario.

The scalar reference backend at kernel level (what CI runs with `simd-force-scalar`; drop `--scenarios` when `ffmpeg` is installed):

```bash
DENY_WARNINGS="target.'cfg(all())'.rustflags = ['-D', 'warnings']"
cargo test --config "$DENY_WARNINGS" --target host-tuple --features simd-force-scalar
cargo test --config "$DENY_WARNINGS" --manifest-path native/Cargo.toml --target host-tuple --features simd-force-scalar
node scripts/gate_native_parity.mjs --features=simd-force-scalar --scenarios=hodge,conservation,swarm
```

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

# Native/wasm whole-kernel parity (Gate N), and the replay driver on any trace set
node scripts/gate_native_parity.mjs
cargo run --release --manifest-path native/Cargo.toml --target host-tuple --bin chronos_replay -- <trace|dir>...

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
| [native/replay/](../native/replay/main.rs) | `chronos_replay`, the native replay driver: replays traces against the native build of the same kernel source (see [Native replay driver](#native-replay-driver-nativereplay)). |
| [src/native.rs](../src/native.rs) | Native-only kernel support for the driver: `reset_to_pristine()` (the `RESET` record) and the native layout of every region (`regions()`). Not compiled for wasm32. |
| [scripts/trace_region_diff.mjs](../scripts/trace_region_diff.mjs) | Replays a trace on the wasm up to call N, dumps a region, and diffs it element by element against the driver's native dump of the same region. |
| [scripts/gate_native_parity.mjs](../scripts/gate_native_parity.mjs) | `Gate N`: fresh bounded recording, native replay, negative control. |
| [native/replay/bench.rs](../native/replay/bench.rs) | `chronos_replay --bench`: times every native kernel call on a trace (see [Native vs wasm kernel timing](#native-vs-wasm-kernel-timing)). `trace_replay.mjs --bench` is the wasm side. |
| [scripts/bench_native_vs_wasm.mjs](../scripts/bench_native_vs_wasm.mjs) | One command for the native/wasm comparison over trace directories: interleaved rounds, medians, spread, ratios. |
| [scripts/bench_profile.mjs](../scripts/bench_profile.mjs) | Where the time goes on one trace: xctrace (native, by function, inlined source line, SIMD layer op and instruction) and V8's CPU profiler (wasm, by function). |

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

Per instance, the hook keeps a full snapshot of linear memory taken after the previous call. Before each export call it compares memory with the snapshot (`Buffer.compare` over 64 KiB blocks and 4 KiB chunks, then 32-bit words, trimmed to exact bytes) to find every byte the host changed. The kernel's own writes are already in the snapshot, so they never appear as host writes. Each changed range is named by **region + offset**, never by raw address. If some changed bytes fall outside every region, the hook checks whether memory now equals the fresh-instance image plus region writes. That is what `calibrate_mp4.mjs` does when it resets a pooled instance by restoring the first 64 KiB and zeroing the rest. If so, the change is recorded as `RESET` followed by those region writes. Anything else outside every region becomes an `UNMAPPED` record and is reported on stderr. The export is called with arguments converted exactly as the JS API converts them (`ToInt32`, `Math.fround`, `ToNumber`). After the call, the snapshot is refreshed and the `XXH32` digest of every digested region is recorded, with the four platform-descriptive words of the [allowlist](#platform-word-allowlist) zeroed before hashing (format 1.1; the mask is written into the header metadata as `digestMask`). Getters (`chronos_get_*`, `padic_ultrametric_val`) have their host writes captured like any call, but carry no digests; after a getter only the shadow stack (`[0, __stack_pointer init)`) is re-synchronised. Strict mode confirms getters write nothing else.

### Region table

The table lives in `REGION_SPECS` in [scripts/trace_lib.mjs](../scripts/trace_lib.mjs). Each entry names the getter export and the Rust backing field. The byte length comes from that field's declared array type in `src/*.rs`, evaluated from the Rust `const` items (and `size_of::<IngestSlotMeta>() == 96` for `ring_meta`). Element type and direction come from the getter's Rust return type: `*const` is `out`, `*mut` is `both`, and `in` marks the staging buffers `ingest_rgba`, `audio_staging`, `hodge_inject` and `conservation_input`. Building the table fails if a pointer-returning export in `src/lib.rs` has no entry. At startup each region's pointer is checked for alignment and for lying inside memory, above the shadow stack.

There are 47 regions: `ring_slot[0..12)` (the 12 RGBA ring slots, `chronos_get_ring_slot_ptr(k)`), the two moving aliases `ingest_rgba` and `display_rgba` (host writes can be addressed through them, but they are not digested separately), and 33 fixed regions: `eca_res`, `pip_rgba`, `all_pip_rgba`, `slot_meta`, `laplacian`, `fiedler`, `telemetry`, `micro_tracks`, `continuum_field`, `rigid_bodies`, `swarm_telemetry`, `modal_field`, `ring_meta`, `row_luma`, `wave1_telemetry`, `bos_telemetry`, `bos_planes`, `bos_valid`, `bos_rgba`, `micro_cov`, `micro_residual`, `micro_loom`, `micro_quality`, `hero_candidates`, `wave2_telemetry`, `audio_staging`, `vibro_telemetry`, `vibro_planes`, `hodge_telemetry`, `hodge_planes`, `hodge_inject`, `conservation_telemetry` and `conservation_input`. Each digested call covers 45 regions, 12,715,152 bytes. The full table, with getter, argument, element type, length and flags, is embedded in every trace header, so a replayer never parses Rust.

### Canonical set (measured)

`record_canonical_traces.mjs` with default caps: 11 scenarios, 29 traces, 2.54 GB, recorded in 3 min 46 s (including replay) on Apple Silicon. Replaying all 29 takes 39 s with 0 mismatches. Re-recorded in format 1.1 (digest mask), the set has identical call and host-write counts, and all 29 again replay against the wasm with 0 mismatches.

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

Non-portable writes in the set: `gate_i5_bos.mjs` and `gate_i0_i4_ingest.mjs` find the private `Arena` fields `ring_cursor`, `res_mode` and `frame_idx` by scanning memory and overwrite them to test frame-index wraparound. No getter exposes those fields, so the writes are `UNMAPPED`. They replay against the wasm, but a native build can follow those three traces only up to call `#209` (`bos`), `#58` (`ingest_roi` instance 2) and `#710` (`ingest_roi` instance 6). Making them portable would need a test-only export for those fields. The native driver stops each of them at that call and reports `portable prefix ended at call N`, which is distinct from both a pass and a failure.

### Format

All integers are little-endian. Header:

| Field | Type |
| :--- | :--- |
| magic | 8 bytes `CSTRACE\0` |
| version major, minor | `u16`, `u16` (`1`, `1`; 1.0 traces have no digest mask) |
| header length in bytes (from offset 0) | `u32` |
| sha256 of the wasm binary | 32 bytes |
| linear memory size in bytes | `u32` |
| flags (bit 0: strict, bit 1: whole-memory digests present) | `u32` |
| metadata JSON (label, argv, pid, instance ordinal, node version, ..., and since 1.1 `digestMask`: `{ region: [word index, ...] }`, the 32-bit words zeroed before hashing that region) | `u32` length + UTF-8 |
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
3. When digest bit 0 is set, hash every digested region and compare it with the recorded value. Before hashing a region named in `meta.digestMask`, zero the listed 32-bit words (in a copy). A mismatch names the call index, the export and the region. Bit 1 (whole-memory digest) is meaningful only against the identical wasm binary. A native replayer reads past it.
4. For `R`, restore the fresh-instance state. The JS replayer copies back the memory image it saved at instantiation. The native driver calls `chronos_spectral::native::reset_to_pristine()`, which zeroes the arena (`ARENA` is initialised to zero bytes and is the kernel's only mutable static).
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

### Native replay driver (`native/replay`)

`chronos_replay` replays traces against the native build of the same `src/lib.rs` (the `native/` package, zero dependencies) and checks them bit for bit. It never uses a raw wasm address: every region is resolved through the kernel's own getter at the moment it is needed.

```bash
# Release build (overflow checks off, as in the shipped wasm)
cargo build --release --manifest-path native/Cargo.toml --target host-tuple --bin chronos_replay
native/target/<host>/release/chronos_replay [options] <trace.cstrace | dir>...
# Overflow checks and debug assertions on (opt-level 1, replays about as fast as release); `dev` is ~100x slower
cargo build --profile checked --manifest-path native/Cargo.toml --target host-tuple --bin chronos_replay
```

Options: `--max-errors=N` (divergent calls reported per trace before it stops, default `1`), `--dump-dir=DIR` (native region dumps, default `target/replay-dumps`), `--dump-at=CALL:REGION[,REGION]` (also dump regions after a matching call), `--require-full` (a portable-prefix or recorded-trap stop fails), `--wasm-sha256=HEX` (every trace must come from this wasm binary), `--info` (print each header), `--no-mask` (hash raw regions, for 1.0 traces), `--in-process` (default: one child process per trace), `--list-allowlist`. `--bench` switches to the timing mode described in [Native vs wasm kernel timing](#native-vs-wasm-kernel-timing).

Per trace:

1. **Validate before touching memory.** The function table must name exactly the native exports, with the wasm parameter and result types implied by the Rust signatures (the dispatch table in `native/replay/kernel.rs` calls every export with arguments of its declared Rust types, so the compiler checks it, and a unit test checks it against every `#[no_mangle]` function in `src/`). Every region must match `chronos_spectral::native::regions()`: the same getter and argument, element type, byte length (the native field's `size_of_val`), host-writable and alias flags, and the native field must lie inside the arena. `meta.digestMask` must be exactly the driver's allowlist. Any difference rejects the trace with a list of what differs, so a layout difference can never cause an out-of-bounds access.
2. **Apply** (`Replayer::apply_write`, `Replayer::reset`). `W`: the region pointer comes from its getter now and must equal the native field (an alias must equal a ring slot); the write must fit the region. `R`: `native::reset_to_pristine()`. The trace starts from a reset too.
3. **Call** (`Replayer::invoke`): the native `chronos_*` function with the recorded argument bits. This phase is kept separate so a timing mode can measure only it.
4. **Verify** (`Replayer::check_result`, `Replayer::check_digests`). i32 results are compared bit for bit. f32 results are compared bit for bit too, except that NaN matches any NaN (rule 2). Pointer results are skipped. Every digested region is hashed (allowlisted words zeroed) and compared. A region whose bytes have not changed since the last call reuses its hash. Whole-memory digests are read past.

Outcomes: `PASS` (every record replayed, 0 mismatches), `PREFIX` (stopped at the first `U` record: "portable prefix ended at call N", everything before it matched), `TRAP` (a recorded trap: kernel exports are `extern "C"`, so a native panic aborts the process and cannot be replayed in-process; none occur in the canonical set), `FAIL` (a divergence, a structural error, or a native panic, which a child process turns into a failed trace naming the call and the panic location). The exit status is `0` when nothing failed.

On the first divergent call the driver prints the call index, the export with decoded arguments, and every region whose digest differs, with both hashes. It writes each differing native region to `<dump-dir>/<trace>.c<call>.<region>.native.bin` and prints the command that dumps the same region from a wasm replay and diffs the two element by element:

```text
[FAIL]   swarm_1200.p93524.i000.cstrace: 149 calls replayed, ... mismatches=1
    call #148 chronos_post_tick(0.033333335): 4 of 45 region digests differ:
      region swarm_telemetry: native d2dced97 != recorded 7408e94d
        native dump: .../swarm_1200.p93524.i000.c148.swarm_telemetry.native.bin
        diff vs wasm: node scripts/trace_region_diff.mjs <trace> --call=148 --region=swarm_telemetry --native=<dump>
$ node scripts/trace_region_diff.mjs ...
swarm_telemetry: 7 of 64 f32 elements differ
  [10]: wasm 0.027136314660310745 (0x3cde4cfa)  native 0.027136456221342087 (0x3cde4d46)
  ...
```

(This example is the negative control below: a scalar backend with `f32x4.nearest` rounding ties away from zero.)

### Platform-word allowlist

Exactly four 32-bit words are exempt from the comparison. Each describes the build, not the computation. The mechanism is masking on both sides: the recorder zeroes these words in a copy of the region before hashing, records the list in every trace (`meta.digestMask`), and every replayer does the same. The driver refuses a trace whose mask differs from its own list (`native/replay/allowlist.rs`, mirrored by `PLATFORM_WORDS` in `scripts/trace_lib.mjs`), so a trace cannot widen the exemption. Masking was chosen over having the driver substitute the wasm's values: the trace does not carry those values, and two of them change during a session (`0` before `chronos_init`, the size after it, `0` again after `RESET`). Substitution would mean recording them per call and inventing expected values on the native side. With masking, both sides compute the same function of memory. The cost is that a replay against the wasm no longer checks these four words either, which is acceptable for build constants.

| Word | wasm32 value | native (aarch64) value | Why it differs |
| :--- | ---: | ---: | :--- |
| `telemetry[20]` | `22644848.0` | `22651136.0` | `size_of::<Arena>() as f32`: the arena holds `usize` fields (for example `LanczosScratch::act_idx: [usize; 32]`), so its size depends on pointer width. |
| `wave1_telemetry[1]` | `22644848` | `22651136` | `size_of::<Arena>() as u32`, as above. |
| `wave1_telemetry[12]` | `22704064` | `0` | `linked_static_end_bytes()`: the wasm linker's `__heap_base`. A native build has no linear memory. |
| `wave2_telemetry[1]` | `22644848` | `22651136` | `size_of::<Arena>() as u32`, as above. |

That is 16 bytes exempt per digest-checked call, out of 12,715,152 bytes hashed (0.000126%): 4 of 6,272 bytes of `telemetry`, 8 of 768 of `wave1_telemetry`, 4 of 256 of `wave2_telemetry`, and none of the other 42 regions. These were the only differences on the first native replay (the masked words, diffed with `trace_region_diff.mjs`). No other word differs anywhere in the canonical set.

### Native parity results (measured)

The canonical set re-recorded in format 1.1 was replayed on the native scalar build (`simd::scalar`, the only native backend so far), Apple Silicon (aarch64), release profile: 66.6 s for 2.54 GB.

| Scenario | Traces | Calls replayed | Digest-checked calls | Region digests | Result |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `hero_sd` | 2 | 1,126 | 934 | 42,030 | pass |
| `hero_hd` | 2 | 1,126 | 934 | 42,030 | pass |
| `hero_suite` | 1 | 31,371 | 25,927 | 1,166,715 | pass |
| `mp4_occlusion` | 2 | 1,241 | 1,026 | 46,170 | pass |
| `swarm_1200` | 1 | 719 | 604 | 27,180 | pass |
| `bos` | 1 | 209 | 172 | 7,740 | portable prefix ended at call 209 |
| `hodge` | 1 | 175 | 125 | 5,625 | pass |
| `rs_vibrometry` | 1 | 6,739 | 5,413 | 243,585 | pass |
| `conservation` | 1 | 1,424 | 958 | 43,110 | pass |
| `covariance` | 10 | 4,742 | 1,086 | 48,870 | pass |
| `ingest_roi` | 7 | 7,525 | 5,556 | 250,020 | 5 pass; instance 2: portable prefix ended at call 58; instance 6: at call 710 |
| **total** | **29** | **56,397** | **42,735** | **1,923,075** | **0 mismatches** (26 pass, 3 portable prefix) |

The `checked` profile (overflow checks and debug assertions, including core's UB precondition checks) gives the same result on all 29 traces with no panic. The unoptimised `dev` profile gives the same result on the subset it was run on (`hero_sd`, `swarm_1200`, `mp4_occlusion`, `bos`, `hodge`, `conservation`, two `covariance` and two `ingest_roi` instances).

Expected divergence sources that turned out not to diverge, and why:

- **`isize` clamps** (`(x + d).min(N - 1) as usize` at `src/soa_tracks.rs` 2945, 2947, 8444, 8446 and `src/continuum_physics.rs` 5488, 5498, 5692, 5711, 8639, 8660). A negative value would give a different `usize` on 32 and 64 bits, but none can be negative. In `soa_tracks.rs` the base is `cell_coord(..)`, which is clamped to `[0, max_cell]`. In `continuum_physics.rs`, `row_s + d_max` has `row >= 0` and `dy_cells` clamped to `[1, FIELD_GRID_H - 1]` by `compute_row_dx_spans`, and every `col + dx_c` is behind a `dx_c >= 0` guard.
- **`src/simd_ops.rs:10504`** aligns the `roi_luma` scratch pointer up to 16 bytes, so the h-cache starts at a different offset in the scratch on each target. That is harmless: the cache keys (`cached_sy`, `cached_fused_sy`) are per-call locals initialised to `i32::MIN`, so every cache row is written before it is read in the same call. The other `roi_luma` users fill their window before reading it. The extent used, at most a few KiB plus 15 bytes of slack, is far inside the 36,864-byte buffer.
- **`usize` fields** change the arena layout (`22,644,848` vs `22,651,136` bytes), but nothing reads the arena as bytes except `chronos_init`'s zeroing and the size words in the allowlist.
- **Scalar backend and NaN payloads.** No scalar-backend op disagreed with V8's SIMD128 on any trace. A NaN census of every digested `f32` region at every digest-checked call (an instrumented throwaway build) found that the kernel never writes a NaN into an output region in the canonical set. The only NaNs present are host-written invalid-input tests (`hodge_inject`, `conservation_input`, `0x7fc00000`), which the driver copies byte for byte. So the payload of generated NaNs, which differs between aarch64 (`0x7fc00000`) and x86 (`0xffc00000`), cannot affect these traces.

**Overflow checks beyond the portable prefix** (an experiment in a throwaway copy, not part of the driver). The three `U` writes set `ring_cursor` and `frame_idx` to `2^32 - 18`, `2^32 - 3` and `2^32 - 2` to test frame-index wraparound, so the prefixes stop just before the arithmetic most likely to wrap. Mapping those 16 bytes onto the native fields (`ring_cursor: u64, res_mode: u32, frame_idx: u32`, contiguous in the `repr(C)` arena on both targets) let all three traces continue under the `checked` profile. Both `ingest_roi` traces then replay to the end bit for bit (83 and 867 calls). `bos` hits exactly one overflow: `src/physics_id.rs:2713` `f += 1` (`u32`) in `extract_compensated_window_from_ring`, once, at `f == u32::MAX` (call `#263`). Treating that one increment as wrapping, the whole `bos` trace (2,464 calls) replays bit for bit. v3.1.0 fixes that loop: see "Frame-index wraparound in physics-ID window loops" below.

#### Frame-index wraparound in physics-ID window loops (fixed in v3.1.0)

The engine frame counter `frame_idx` is a `u32` that `chronos_post_tick` advances with `wrapping_add`, so `u32::MAX` is an ordinary frame index: the 2^32-th frame of a session (about 2.3 years at 60 fps), or immediately in the gates that position the counter just before the wrap. Up to v3.0.0, four window loops in [src/physics_id.rs](../src/physics_id.rs) had the form `while f <= bound { ...; f += 1; }`, which computes `bound + 1` when `bound == u32::MAX`:

| Function (v3.0.0 line of `f += 1`) | Reached from the exports | Effect when the bound was `u32::MAX` |
| :--- | :--- | :--- |
| `extract_compensated_window_from_ring` (2713), bound `head_f` | `chronos_post_tick` / `chronos_tick` (`update_physics_and_telemetry`), for every confirmed track, with `head_f = frame_idx`: every tick at which `frame_idx == u32::MAX`. | With 64 frames in `[start_f, u32::MAX]` the loop ended on its count bound after computing `u32::MAX + 1` (the overflow the `bos` experiment above hit at call `#263`). With fewer (a track born, or a discontinuity, within the last 63 frames before the wrap) the release build wrapped `f` to `0` and appended samples from ring slots `0, 1, ...`, frames from outside the window, until it had 64. |
| `extract_compensated_vis_to_scratch` (2766), bound `head_f` | Not reachable: its only caller, `step_ode_track_from_ring`, has no caller (`#[allow(dead_code)]`, emits no wasm). | As above. |
| `collect_window_samples` (5802), bound `fe` | `chronos_post_tick` / `chronos_tick` through `analyze_two_body_collisions` and `evaluate_pair_at_frame_ref` (and `evaluate_stationary_birth_one_side` from it): the post-impact window of the newest candidate frame `f_c = frame_idx - post_reach` ends at `frame_idx`. Every tick at which `frame_idx == u32::MAX` while two confirmed tracks are close enough for the pair scan to evaluate them. | The window holds 6 or 8 frames and the count bound is 10, so the loop always wrapped: it added samples from the oldest ring slots (64 frames back) to the post-impact fit until it had 10 visible ones, wherever they were in the ring. |
| `analyze_two_body_collisions` pair scan (7118), bound `scan_hi` | Not reachable at `u32::MAX`: `scan_hi = frame_idx - win_w <= u32::MAX - 6`. | None. |

v3.1.0 writes all four, and the candidate loop next to the pair scan (bound `f_hi <= u32::MAX - 8`), as inclusive ranges (`for f in lo..=hi`), which end after `hi` without computing `hi + 1`. Results are unchanged for every bound below `u32::MAX`. Regression tests: `physics_id::frame_wrap_tests` (`cargo test --target host-tuple`, run by Gate S). With overflow checks (the test profile) all four tests panic on the old loops. Without them (the release wasm's arithmetic) the window functions return 64 and 10 samples instead of 5 and 6. Every other `while x <= bound` loop in `src/` is bounded by grid, frame or array dimensions far below the maximum of its type.

### Gate N: native/wasm parity in `./build.sh --check`

[scripts/gate_native_parity.mjs](../scripts/gate_native_parity.mjs) needs no pre-recorded traces and makes no platform assumptions:

1. Builds `chronos_replay` for the host with the `checked` profile and `-D warnings`. The flag goes in through `cargo --config "target.'cfg(all())'.rustflags = [...]"`, and `RUSTFLAGS` / `CARGO_ENCODED_RUSTFLAGS` are removed from the build's environment. Until this was fixed the gate set `RUSTFLAGS=-D warnings`, which replaces the `target.*.rustflags` of `.cargo/config.toml`. On x86_64 Linux (the CI job) that dropped `+ssse3,+sse4.1`, so Gate N there replayed on `simd::scalar` instead of `simd::sse`. `cargo check -v --target x86_64-unknown-linux-gnu` shows the rustc command line with `target-feature=+ssse3,+sse4.1` when `RUSTFLAGS` is unset and without it when `RUSTFLAGS='-D warnings'`.
2. Records, from `web/chronos_spectral.wasm` and in parallel, four existing harnesses under the hook into `target/native-parity/` (removed after a pass; `--keep` keeps it). The scenarios are `calibrate_mp4.mjs --clip=juggling_ballistic` (real-MP4 sparse hero tracking at SD, with a pooled-instance `RESET`), `gate_i3_hodge.mjs`, `gate_i5b_conservation.mjs` and `benchmark_1000.mjs --only=Q1` capped at 24 MiB. Measured: 5 traces, 3,088 calls, 104.0 MB, 3.1 s. Each scenario must reach a minimum call count, and none may contain a `U` record.
3. Replays them with `--require-full --wasm-sha256=<web wasm>`: 3,088 calls, 102,375 region digests, 0 mismatches required (2.4 s). The driver's banner names the SIMD backend it runs, and the gate fails unless that is the one the host and features must select: `neon` on aarch64, `sse` on x86_64, and `scalar` with `simd-force-scalar` or on any other host.
4. Negative control: one byte of a host write in the Hodge trace flipped (`trace_tool.mjs mutate-write`). The driver must fail at or after the call that write precedes (measured: call `#97`, region `ring_slot[9]`).

The swarm scenario is there because it is the one that catches a subtle scalar-backend bug: `f32x4.nearest` rounding ties away from zero passes Hodge, conservation and SD hero tracking, and is caught at swarm call `#148`. The gate takes 121 s on Apple Silicon, almost all of it the native build (115 s; a cached build takes under a second). `--features=...` passes cargo features (`--features=simd-force-scalar` replays on the scalar reference; CI runs it on Linux x86_64 and macOS arm64). `--scenarios=<ids>` runs a subset, which must include `hodge` (the negative control). The `hero_sd` scenario depends on `ffmpeg`, like `gate_i1_cov_residual.mjs`.

## Continuous integration

[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs on every push to and pull request against `main`. Every job installs the toolchain pinned in `rust-toolchain.toml` (the Miri job also installs one pinned nightly) and none sets `RUSTFLAGS`, which would replace the `target.*.rustflags` of `.cargo/config.toml` (see Gate N above).

| Job | Runner | Runs | Proves |
| :--- | :--- | :--- | :--- |
| `verify-wasm-simd128-and-e2e` | `ubuntu-latest` (x86_64) | `./build.sh --e2e` with `CHRONOS_VERIFY_COMMITTED_WASM=1` | The full pipeline against the committed Wasm; Gates S and N natively on the SSE backend. |
| `native-macos-arm64` | `macos-latest` (Apple Silicon; the job fails if `uname -m` is not `arm64`) | `./build.sh --native-check` with `CHRONOS_VERIFY_COMMITTED_WASM=1`, then the scalar commands under [Native backends only](#native-backends-only---native-check) | NEON on real hardware: conformance against the scalar reference and whole-kernel parity with the Wasm (Hodge, conservation, swarm; no `ffmpeg` on the runner). The scalar reference at kernel level on aarch64. The Wasm builds byte-identically on macOS too. |
| `native-scalar-linux-x86_64` | `ubuntu-latest` | Both native test suites and Gate N with `simd-force-scalar`, all four scenarios | The scalar reference at kernel level on x86_64, real-MP4 tracking included. |
| `miri-scalar` | `ubuntu-latest` | `cargo miri test` over the SIMD layer, then `native/tests/kernel_ticks.rs` (see [Miri](#miri)) | No undefined behaviour Miri can detect in that scope. Non-blocking (`continue-on-error: true`). |

A scalar backend that diverges fails both scalar jobs. The conformance suite compares it with NEON or SSE and with `std` math, and Gate N compares the whole kernel with the Wasm call by call. Measured with a deliberately broken scalar `f32x4_nearest` (ties away from zero) and `simd-force-scalar` on aarch64, the scalar step fails three ways. `cargo test` fails `neon_backend_matches_scalar_reference` and `floor_and_nearest_match_std` (on x86_64, `sse_backend_matches_scalar_reference` takes the place of the NEON test). Gate N fails at swarm call `#148` (`chronos_post_tick`, 4 of 45 region digests differ), while Hodge and conservation still pass.

**Caching.** The kernel has no dependencies, so the only thing a native job builds is the crate itself, and Cargo decides its freshness by file mtimes. A checkout makes every source newer than any cached artifact, so a cache restored on a partial key would be rebuilt anyway. Worse, moving the mtimes back after such a restore could make stale artifacts look fresh. The native jobs therefore cache `target/` and `native/target/` under a key that hashes the toolchain file, both lock files, both manifests, `.cargo/config.toml` and every Rust source (`src/**`, `native/replay/**`, `native/tests/**`), with no `restore-keys`. Only on an exact hit do they set every tracked file's mtime to 2000-01-01, so Cargo reuses artifacts built from byte-identical inputs. That covers re-runs and commits that touch only docs, scripts or `web/`, and skips Gate N's native build (about 40 s of its 43 s on Apple Silicon). The full-pipeline job keeps its original cache and always rebuilds the crate, as the deployment gate should.

**Pages.** [.github/workflows/pages.yml](../.github/workflows/pages.yml) runs on `workflow_run` completion of this workflow and deploys only if `github.event.workflow_run.conclusion == 'success'` (for a push to `main` of a public repository). A workflow run concludes `success` only when every job does, so any failing blocking job prevents deployment. The Miri job has `continue-on-error: true`: its failure shows on the job but does not change the run's conclusion, so it cannot block (or permit) a deployment. A cancelled or timed-out run concludes `cancelled` or `failure` and does not deploy either.

## Miri

[Miri](https://github.com/rust-lang/miri) interprets the kernel's MIR and reports undefined behaviour it can observe: out-of-bounds or misaligned raw-pointer accesses, uninitialised reads, invalid values, data races, and aliasing violations under Stacked Borrows (the default) or Tree Borrows. It needs a nightly toolchain, so it uses `nightly-2026-10-19` (with the `miri` and `rust-src` components) for this only; everything else keeps `rust-toolchain.toml`. It cannot interpret most NEON intrinsics, so every Miri run uses `simd-force-scalar`. On x86_64 it does interpret the SSE intrinsics, so there `sse_backend_matches_scalar_reference` runs the SSE backend under Miri too.

```bash
rustup toolchain install nightly-2026-10-19 --profile minimal --component miri,rust-src
HOST="$(rustc +nightly-2026-10-19 -vV | sed -n 's/^host: //p')"   # cargo miri does not accept `host-tuple`
# SIMD layer tests
cargo +nightly-2026-10-19 miri test --target "$HOST" --features simd-force-scalar
# Kernel: chronos_init, chronos_set_params (look-ahead K = 1), two SD frames through chronos_tick
cargo +nightly-2026-10-19 miri test --manifest-path native/Cargo.toml --target "$HOST" \
    --features simd-force-scalar --test kernel_ticks -- --nocapture
# Replay driver on a short recorded trace (reads the trace file, hence no isolation)
MIRIFLAGS=-Zmiri-disable-isolation cargo +nightly-2026-10-19 miri run --manifest-path native/Cargo.toml \
    --target "$HOST" --features simd-force-scalar --bin chronos_replay -- --in-process --require-full <trace>
```

Under `cfg(miri)` the tests run reduced versions of themselves:
- The conformance tests use 4 random vectors per entry instead of 4,096; the edge vectors still run in full.
- The negative-control test runs 1 of its 9 controls.
- The scalar-vs-`std` sweeps cover about 1,700 `f32` inputs and 4,000 `f64` inputs.
- The SSE random-shuffle test uses 500 iterations instead of 100,000.
- `kernel_ticks` runs 2 frames instead of 24.

Two tests are ignored under Miri: `every_kernel_const_generic_instantiation_is_in_the_table` (it reads `src/` from disk) and `neon_backend_matches_scalar_reference` (NEON intrinsics; the first one hit is `llvm.aarch64.neon.fmin.v4f32`).

### Results (Apple M4 Pro, at most two Miri processes running at once)

| Scope | Target | Aliasing model | Result | Wall time |
| :--- | :--- | :--- | :--- | ---: |
| SIMD layer tests (13 run) | `aarch64-apple-darwin` | Stacked Borrows | pass, no UB reported | 2 min 57 s |
| SIMD layer tests (13 run) | `aarch64-apple-darwin` | Tree Borrows | pass, no UB reported | 3 min 55 s |
| SIMD layer tests (13 run, SSE backend included) | `x86_64-unknown-linux-gnu` (cross-interpreted, as in CI) | Stacked Borrows | pass, no UB reported | 6 min 07 s (sysroot build included) |
| `kernel_ticks`: init + 3 SD frames, K = 8 | `aarch64-apple-darwin` | Stacked Borrows | pass, no UB reported | 26 min 28 s |
| `kernel_ticks`: init + 3 SD frames, K = 1 (display stage reached) | `aarch64-apple-darwin` | Stacked Borrows | pass, no UB reported; init 33 s, frames 340 s, 611 s, 617 s | 26 min 42 s |
| `kernel_ticks`: init + 2 SD frames, K = 1 (the CI configuration) | `x86_64-unknown-linux-gnu` | Stacked Borrows | pass, no UB reported; init 33 s, frames 346 s, 601 s | 16 min 25 s |
| `chronos_replay --in-process` on a 9-call trace recorded from the wasm (init, set_resolution, set_params, two frames with ingest meta and `chronos_tick`) | `aarch64-apple-darwin` | Stacked Borrows | PASS bit for bit (7 digest-checked calls, 315 region digests); 2 warnings, below | 39 min 35 s |

Each digest-checked call hashes 12.7 MB of regions, which takes about 4 minutes under Miri, so the replay scope stays a local experiment. The trace was recorded by a small throwaway Node script under `scripts/trace_hook.mjs`; it is not in the repository.

**Findings.** Miri reported no undefined behaviour in any scope above, under either aliasing model where both were run (Tree Borrows was run on the SIMD layer; the kernel and replay runs used Stacked Borrows, the stricter of the two). In particular, the re-entrant `arena_mut()` pattern (a function holding `let a = arena_mut()` while calling another kernel function that calls `arena_mut()` again) did not produce a Stacked Borrows violation on the paths these runs execute. That pattern is only UB if the outer reference is used again after the inner call has retagged the arena. The only diagnostics were two warnings in the replay driver, not the kernel:

- `native/replay/replayer.rs:406` (`std::slice::from_raw_parts(p as *const u8, len)` in `check_digests`) and `native/replay/replayer.rs:344` (`(p + offset as usize) as *mut u8` in `apply_write`): `warning: integer-to-pointer cast`. The driver keeps region addresses as `usize` (from `Ret::Ptr(self as usize)` in `native/replay/kernel.rs`) and casts them back. Under Rust's exposed-provenance rules that is defined behaviour, so these are not hazards. But Miri can then check those accesses less precisely ("Miri might miss pointer bugs"). Keeping the getter results as `*mut u8` would let the driver run under `-Zmiri-strict-provenance`.

**What this does not cover.** Only the paths these runs execute are checked: init, the parameter and ingest setters, and the SD per-frame pipeline in the default modes (swarm, Hodge, BOS, vibrometry and conservation off, `lab_mode = 0`), on the scalar backend, for 2 to 3 frames. HD resolution, the swarm and continuum path, Hodge, BOS, vibrometry and audio, conservation, `chronos_reset_tracks`, the inspect calls and `native::reset_to_pristine` were not run under Miri. Neither was the NEON backend. Nor was any long-run state, such as ring wraparound or track lifetimes.

**CI.** The `miri-scalar` job runs the first two commands on `ubuntu-latest` (x86_64) with Stacked Borrows, and is non-blocking (`continue-on-error: true`, `timeout-minutes: 75`). Expected runtime is about 30 to 45 minutes: 6 + 16 minutes measured here, and GitHub's runners are slower per core than an M4 Pro.

## Native vs wasm kernel timing

The native build exists to be faster than the wasm in V8 on every workload, so it is timed against the wasm on identical inputs: the canonical CSTRACE set (`canon-v11`, the 29 traces of [Canonical set (measured)](#canonical-set-measured)), replayed call for call on both. Parity is checked separately (the 29-trace replay with digests, [Native parity results](#native-parity-results-measured)); a timing run checks only return values.

### What is timed, and how the two sides are kept equal

| | wasm in V8 (`node scripts/trace_replay.mjs --bench`) | native (`chronos_replay --bench`) |
| :--- | :--- | :--- |
| Input | the trace, read into memory once before timing | the same |
| Call sequence | every `C` record up to the first `U` (raw-address) record or recorded trap | the same: a native replay cannot follow a `U` write, so both stop there (`bos` at call 209, `ingest_roi` instances 2 and 6 at calls 58 and 710) |
| Each pass | fresh-instance state (the pristine memory image), then every record | fresh-instance state (`native::reset_to_pristine`), then every record |
| Timed | each kernel call alone: `performance.now()` before and after | each kernel call alone: `Instant::now()` before and after |
| Not timed | host writes (`W`), resets, region getters, record decoding | the same |
| Included in a call | V8's JS-to-wasm wrapper and argument conversion (the app pays it too) | the dispatch adapter that decodes the recorded argument bits (a few ns) |
| Checks | return values compared with the recording outside the timed region; digests off | the same |

Both clocks are monotonic and, on macOS, read the same 24 MHz `mach_absolute_time` timebase: 41.7 ns resolution. The mean empty interval (`t1 - t0` with nothing between) is measured at start-up, 13-15 ns native and 8-18 ns in V8, and subtracted from every sample (clamped at 0). Calls under about 1 µs (getters, setters) are resolution-limited and their medians are quantised; the tick and solver exports take 20 µs to 9 ms, where neither effect matters.

Fairness:

- **Warm-up**: each process discards `--warmup` passes (2 below). V8 compiles every function with Liftoff first and tiers hot ones up to TurboFan during them. Check: V8 with `--no-liftoff` (TurboFan from the first call) gives the same steady state (table under [Sanity checks](#sanity-checks)); the first, cold pass is 1.2 to 1.8 times slower and the second is already within 1% of steady state.
- **Repetition and drift**: `--passes` measured passes per process (3), `--rounds` processes per workload and side (5). Every round runs every workload on every side, side order alternating (`wasm, neon, scalar`, then `scalar, neon, wasm`), so slow drift on a shared machine affects every side alike.
- **Statistics**: per workload, the median of the 15 per-pass kernel totals. *Spread* is the range of the 5 per-round medians relative to the median. The ratio's *round range* is the minimum and maximum of the 5 per-round ratios (each side's median from the same round), which cancels drift between rounds; the headline ratio is of the pooled medians, so it can fall just outside that range. Per export: median, p95 and mean over every call of every measured pass of every round (raw samples are in the JSON).
- **Integrity**: `bench_native_vs_wasm.mjs` fails unless every side replayed the identical sequence (per trace: call and write counts and stop point; per export: call counts) with 0 return-value mismatches, in every round.
- **Machine**: Apple M4 Pro (12 cores), macOS 27.3, Node 26.9.0 (V8 14.6.202.34), Rust 1.99.0. Another agent was running CI and Miri jobs during every measurement (load average 5 to 7 on 12 cores). xctrace recorded 99.8% of the profiled samples on performance cores, and the spreads below show how much noise remained.

### Commands

```bash
# Timed binary (release profile), the scalar reference, and the same code with line tables for profiling
cargo build --release --manifest-path native/Cargo.toml --target host-tuple --bin chronos_replay
cargo build --release --features simd-force-scalar --manifest-path native/Cargo.toml --target host-tuple \
    --bin chronos_replay --target-dir target/native-scalar
cargo build --profile profiling --manifest-path native/Cargo.toml --target host-tuple --bin chronos_replay
R=native/target/aarch64-apple-darwin

# One side on one trace or directory: per-trace totals, per-export table, wall-clock reconciliation
$R/release/chronos_replay --bench --warmup=2 --passes=5 --bench-reconcile=1 --bench-verbose <trace|dir>...
node scripts/trace_replay.mjs --bench --warmup=2 --passes=5 --bench-reconcile=1 --bench-verbose <trace|dir>...

# The comparison below (40 min here with the scalar side; --out=DIR keeps every per-run JSON and log)
node scripts/bench_native_vs_wasm.mjs --native=neon=$R/release/chronos_replay \
    --native=scalar=target/native-scalar/aarch64-apple-darwin/release/chronos_replay \
    --rounds=5 --warmup=2 --passes=3 --reconcile=1 <canon-v11>/*/

# Build profiles side by side (no wasm); V8 tiering check
node scripts/bench_native_vs_wasm.mjs --no-wasm --native=a=<bin> --native=b=<bin> ... <dirs>
node scripts/bench_native_vs_wasm.mjs --wasm-variant=wasm= --wasm-variant=wasm-turbofan=--no-liftoff --native=neon=<bin> <dirs>

# Where the time goes on one trace (macOS with Xcode)
node scripts/bench_profile.mjs --bin=$R/profiling/chronos_replay --trace=<file.cstrace> --seconds=20 --wasm-passes=100
node scripts/bench_profile.mjs --same-code $R/release/chronos_replay $R/profiling/chronos_replay
```

### Results: native NEON vs wasm in V8 (measured)

Final build (`release` profile, NEON backend including the `i32x4_dot_i16x8` change described below; binary sha256 `5e2e719b...`), `web/chronos_spectral.wasm` (sha256 `7ba36a5b...`) in Node 26.9.0, and the scalar backend for reference (same profile, `--features simd-force-scalar`). 5 interleaved rounds x 3 measured passes per side after 2 warm-up passes. Each number is the median kernel time per pass: the sum of every timed kernel call in one replay of the workload's traces. Frames are `chronos_pre_tick` plus `chronos_tick` calls. The run took 40.5 min, with load average 5 to 6.5 from the other agent.

| Workload | Frames | wasm ms/pass | NEON ms/pass | NEON / wasm (round range) | scalar ms/pass | scalar / wasm | spread wasm / NEON / scalar |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | :--- |
| `hero_sd` | 232 | 23.34 | 20.84 | **0.893** (0.865-0.918) | 113.4 | 4.86 | 5.3% / 2.9% / 2.0% |
| `hero_hd` | 232 | 68.49 | 62.07 | **0.906** (0.882-0.931) | 372.7 | 5.44 | 7.5% / 5.6% / 3.4% |
| `hero_suite` | 8,430 | 1,665 | 1,508 | **0.906** (0.903-0.912) | 9,856 | 5.92 | 1.4% / 0.4% / 1.7% |
| `mp4_occlusion` | 255 | 39.82 | 35.17 | **0.883** (0.861-0.905) | 213.8 | 5.37 | 3.6% / 3.0% / 2.1% |
| `swarm_1200` | 138 | 167.3 | 140.5 | **0.840** (0.824-0.851) | 816.8 | 4.88 | 5.5% / 4.8% / 4.2% |
| `bos` | 29 | 125.9 | 105.0 | **0.834** (0.825-0.846) | 521.6 | 4.14 | 1.7% / 1.9% / 3.3% |
| `hodge` | 44 | 80.56 | 70.00 | **0.869** (0.857-0.944) | 251.2 | 3.12 | 6.0% / 11.6% / 4.2% |
| `rs_vibrometry` | 1,322 | 1,101 | 816.6 | **0.742** (0.741-0.754) | 3,183 | 2.89 | 2.5% / 2.5% / 4.5% |
| `conservation` | 460 | 46.31 | 41.19 | **0.889** (0.868-0.911) | 243.0 | 5.25 | 3.2% / 2.7% / 5.3% |
| `covariance` | 518 | 248.7 | 214.8 | **0.864** (0.847-0.890) | 1,345 | 5.41 | 5.7% / 4.0% / 5.0% |
| `ingest_roi` | 1,321 | 13,460 | 10,836 | **0.805** (0.787-0.819) | 31,893 | 2.37 | 4.2% / 4.6% / 4.0% |
| sum | | 17,026 | 13,851 | 0.813 (geomean 0.856) | 48,810 | 2.87 (geomean 4.34) | |

**Native NEON is faster than the wasm in V8 on every workload, in every round.**

- **Sparse tracking gains least:** `hero_sd`, `hero_hd`, `hero_suite` and `mp4_occlusion` run at 0.88 to 0.91, so 10 to 13% less kernel time.
- **Dense workloads gain more:** swarm 0.84, BOS 0.83, covariance 0.86, ingest/ROI 0.81.
- **Rolling-shutter vibrometry gains most:** 0.74.
- **Whole canonical set:** the native kernel needs 13.9 s where the wasm needs 17.0 s (0.813).
- **Worst case:** the worst single round of any workload is 0.944, on `hodge`, the noisiest workload (11.6% spread on the native side).
- **No workload is slower natively.**

The scalar backend is 2.4 to 5.9 times slower than the wasm (geomean 4.3). So the NEON backend buys a factor of 2.9 to 6.5 over portable scalar code. It buys least where the kernel is mostly scalar already (`ingest_roi`, `rs_vibrometry`, `hodge`).

Per export, per call (µs; pooled over all 15 measured passes of each side):

| Workload | Export | Calls/pass | wasm µs median / p95 / mean | NEON µs median / p95 / mean | NEON / wasm median | scalar µs median |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `hero_sd` | `pre_tick` | 232 | 73.77 / 84.44 / 74.20 | 68.07 / 79.53 / 68.98 | 0.923 | 364.6 |
| `hero_sd` | `post_tick` | 232 | 22.82 / 46.61 / 24.75 | 18.28 / 35.37 / 19.51 | 0.801 | 104.9 |
| `hero_sd` | `solve_fiedler_lanczos` | 232 | 0.273 / 1.315 / 0.441 | 0.151 / 0.777 / 0.220 | 0.553 | 1.068 |
| `hero_hd` | `pre_tick` | 232 | 248.2 / 293.1 / 250.8 | 229.4 / 271.5 / 233.2 | 0.924 | 1,373 |
| `hero_hd` | `post_tick` | 232 | 39.69 / 72.84 / 42.49 | 30.57 / 56.03 / 33.26 | 0.770 | 202.7 |
| `hero_hd` | `solve_fiedler_lanczos` | 232 | 0.359 / 2.275 / 0.763 | 0.153 / 1.402 / 0.456 | 0.426 | 1.278 |
| `hero_suite` | `pre_tick` | 8430 | 80.07 / 313.3 / 157.7 | 73.99 / 289.3 / 145.5 | 0.924 | 395.6 |
| `hero_suite` | `post_tick` | 8430 | 33.27 / 81.81 / 38.49 | 27.98 / 68.90 / 32.31 | 0.841 | 214.7 |
| `mp4_occlusion` | `pre_tick` | 255 | 81.57 / 106.7 / 84.91 | 75.11 / 97.90 / 78.65 | 0.921 | 399.0 |
| `mp4_occlusion` | `post_tick` | 255 | 61.03 / 119.4 / 66.21 | 51.36 / 99.15 / 55.77 | 0.842 | 344.1 |
| `mp4_occlusion` | `solve_fiedler_lanczos` | 255 | 3.277 / 6.402 / 3.543 | 2.903 / 5.694 / 3.161 | 0.886 | 24.69 |
| `swarm_1200` | `pre_tick` | 138 | 263.1 / 377.0 / 279.9 | 238.9 / 337.5 / 255.1 | 0.908 | 1,424 |
| `swarm_1200` | `post_tick` | 138 | 951.3 / 1,180 / 928.4 | 782.7 / 968.6 / 761.5 | 0.823 | 4,631 |
| `swarm_1200` | `solve_fiedler_lanczos` | 138 | 0.736 / 3.231 / 1.227 | 0.360 / 2.402 / 0.745 | 0.489 | 2.465 |
| `bos` | `tick` | 29 | 3,700 / 6,607 / 3,703 | 3,059 / 5,467 / 3,093 | 0.827 | 10,252 |
| `bos` | `bos_solve_manufactured_test` | 22 | 313.6 / 2,071 / 680.2 | 229.3 / 1,735 / 580.4 | 0.731 | 7,397 |
| `hodge` | `tick` | 44 | 1,267 / 2,965 / 1,541 | 1,035 / 3,100 / 1,393 | 0.817 | 5,793 |
| `hodge` | `hodge_solve_injected` | 23 | 570.6 / 1,172 / 514.2 | 509.7 / 1,002 / 427.6 | 0.893 | 553.4 |
| `rs_vibrometry` | `pre_tick` | 1322 | 269.0 / 473.5 / 290.6 | 243.6 / 444.0 / 268.8 | 0.906 | 1,448 |
| `rs_vibrometry` | `post_tick` | 1322 | 418.1 / 1,070 / 533.6 | 134.4 / 848.9 / 342.7 | 0.322 | 137.7 |
| `rs_vibrometry` | `push_audio_block` | 1254 | 1.276 / 2.858 / 1.591 | 0.861 / 2.360 / 1.167 | 0.675 | 0.985 |
| `conservation` | `tick` | 460 | 97.85 / 128.4 / 98.14 | 86.98 / 114.8 / 88.27 | 0.889 | 481.7 |
| `conservation` | `eval_occluded_pair` | 22 | 0.901 / 4.504 / 1.139 | 0.276 / 0.770 / 0.312 | 0.306 | 13.11 |
| `covariance` | `tick` | 518 | 469.9 / 650.1 / 478.7 | 400.9 / 564.3 / 414.4 | 0.853 | 2,592 |
| `ingest_roi` | `pre_tick` | 1320 | 291.4 / 348.8 / 283.3 | 262.3 / 329.6 / 258.7 | 0.900 | 1,418 |
| `ingest_roi` | `post_tick` | 1319 | 11,243 / 12,004 / 9,972 | 8,940 / 9,563 / 7,942 | 0.795 | 25,768 |
| `ingest_roi` | `solve_fiedler_lanczos` | 1318 | 2.527 / 4.568 / 2.545 | 1.151 / 2.985 / 1.329 | 0.455 | 3.193 |

- **`pre_tick`** (pixel ingest, pooling, the static/motion grid) gains 8 to 10%.
- **`post_tick`** (tracking, physics, metrology) gains 16 to 23%.
- **The small solver and query exports** gain more, because V8's JS-to-wasm call costs about 0.1 to 0.6 µs that the native call does not pay.
- **Two exports are about 3 times faster natively.** `rs_vibrometry` `post_tick` is 418 µs median in the wasm, 134 µs NEON and even 138 µs scalar: a V8-side slowdown (see the profile). `conservation` `eval_occluded_pair` is the other.

### Sanity checks

**Per-call times add up to the wall-clock.** In every process of the final run, two extra passes ran: one replayed the trace with no per-call timer (one wall-clock reading around the pass), and one applied every host write but made no kernel call. Their difference is the kernel's share of the wall-clock, measured without per-call timers. It agrees with the summed per-call times within -1.4% to +1.2% on all 11 workloads and all three sides. For example, `hero_hd`: wasm 68.49 ms summed vs 67.71 ms by wall-clock, NEON 62.07 vs 62.72. `ingest_roi`: wasm 13,460 vs 13,457, NEON 10,836 vs 10,842. So the timing mode measures the kernel, not the harness, and the timer correction is right.

**The wasm numbers agree with the kernel's own gates.** Here the wasm's sparse-tracking frame costs 0.101 ms (SD) and 0.295 ms (HD) of kernel time. `calibrate_mp4.mjs` reports about 0.13 ms and 0.33 ms per tick, but it also times its JavaScript Lanczos solve (`createLanczosSolver`) inside the tick. That solve is host code, so in the trace it appears as host writes and is not kernel time. Likewise the swarm frame costs 1.21 ms here against about 1.5 ms in `benchmark_1000.mjs` (Q1), which also includes its JavaScript solve.

**Warm-up reaches V8's steady state.** Same harness: the wasm with default tiering (Liftoff, then TurboFan) against the wasm with `--no-liftoff` (TurboFan from the first call), interleaved with NEON, 3 rounds x 3 passes after 2 warm-up passes:

| Workload | wasm ms | wasm `--no-liftoff` ms | ratio (round range) | wasm first pass (cold) ms | second pass ms |
| :--- | ---: | ---: | :--- | ---: | ---: |
| `hero_sd` | 23.81 | 23.40 | 0.983 (0.962-0.995) | 42.7 | 25.2 |
| `hero_hd` | 70.00 | 70.36 | 1.005 (0.985-1.000) | 91.1 | 70.9 |
| `mp4_occlusion` | 40.98 | 40.59 | 0.991 (0.935-1.004) | | |
| `swarm_1200` | 171.2 | 171.2 | 1.000 (0.987-1.014) | 204.5 | 172.8 |
| `bos` | 125.2 | 125.2 | 1.000 (0.991-1.003) | 149.5 | 126.3 |
| `hodge` | 81.74 | 81.19 | 0.993 (0.968-1.004) | 106.4 | 81.5 |
| `conservation` | 46.56 | 46.34 | 0.995 (0.995-1.006) | | |
| `covariance` | 249.2 | 248.2 | 0.996 (0.989-0.997) | | |
| `rs_vibrometry` (2 rounds x 2 passes) | 1,082 | 1,085 | 1.003 (1.002-1.003) | | |

**Timer cost.** The empty-interval mean subtracted per call was 13 to 15 ns natively and 8 to 19 ns in V8 (`performance.now()`; `process.hrtime.bigint()` would cost 35 ns). Against a 70 µs `pre_tick` that is 0.02%.

**Parity of what was timed.** Every binary these numbers come from passes the 29-trace parity replay with digests on (26 pass, 3 portable prefix, 0 mismatches): the final NEON and scalar binaries, the five build-profile variants and the pre-change NEON build. The timing runs check every return value as well (0 mismatches in every round).

**What is not measured.** x86-64 is not timed. On this machine the SSE backend runs under Rosetta 2, an emulator, so its timings would say nothing about x86 hardware. The SSE backend's parity under Rosetta is covered by the replay results, not by this section.

### Native build configuration (measured)

Five configurations of the NEON build, timed against each other (native only, 11 workloads, 3 interleaved rounds x 3 passes, before the `i32x4_dot_i16x8` change below). Every one passes the 29-trace parity replay with digests (26 pass, 3 portable prefix, 0 mismatches).

| Configuration | Settings | Sum of medians | Geomean vs default | Per-workload ratio range |
| :--- | :--- | ---: | ---: | :--- |
| default `release` | `opt-level = 3`, `lto = false` (thin-local), `codegen-units = 16`, `panic = "unwind"` | 13,852 ms | 1.000 | |
| `cgu1` | + `codegen-units = 1` | 13,992 ms | 1.004 | 0.993 (`hero_hd`) to 1.022 (`swarm_1200`) |
| `fat` | + `lto = "fat"`, `codegen-units = 1` | 13,945 ms | 1.005 | 0.995 (`bos`) to 1.023 (`swarm_1200`) |
| `fatabort` (the wasm's profile) | + `panic = "abort"` | 13,947 ms | 1.005 | 0.986 (`bos`) to 1.032 (`hodge`, noisiest workload) |
| `fatabort` + `-C target-cpu=native` (`apple-m4`) | | 13,955 ms | 1.006 | 0.992 (`bos`) to 1.022 (`swarm_1200`) |

Nothing beats Cargo's default release profile. The wasm's own profile is 0.5% slower in geometric mean and 2.2% slower on the 1,200-target swarm in every round (round range 1.022 to 1.035); `target-cpu=native` gains nothing: the default target's features are `apple-m1`'s, `native` (`apple-m4`) adds only `bf16`, `bti` and `i8mm`, none of which the backend uses, and fused multiply-add is never allowed. `native/Cargo.toml` therefore spells the defaults out explicitly as `[profile.release]`, with this measurement in a comment. `[profile.profiling]` adds line tables and a `.dSYM`; `bench_profile.mjs --same-code` confirms that its kernel instructions are those of `release` (504,491 instructions in 702 kernel functions: same opcodes and registers, only address operands differ because the profile name changes Cargo's metadata hash and with it the layout).

### Native profile: where the time goes

`scripts/bench_profile.mjs` profiles one trace on both sides.

**Native side.** It runs the `profiling` build (the timed kernel code plus line tables) under `chronos_replay --bench` and attaches the macOS Time Profiler (`xcrun xctrace record --attach`, 1 kHz, 20 s) once the measured passes start.

- Samples count as kernel time when a kernel export is on the stack.
- Each sampled PC is attributed to its physical function through the binary's symbol table. xctrace's own frame names follow inlining once a dSYM is present, so they are not used for this.
- Each PC is then mapped through `atos -i`, the inline chain from the dSYM, to three things: the SIMD layer op (`src/simd/neon.rs` frame), the innermost kernel function, and its source line. It is also looked up in an `otool` disassembly.

**Wasm side.** The same trace runs under `trace_replay.mjs --bench --bench-cpu-prof`, which profiles only the measured passes with V8's sampler (`node:inspector`, 100 µs requested, about 160 µs effective). Functions are named from the wasm name section.

**Absolute times.** Shares become milliseconds per pass through a short `--bench` run of each side on the same trace.

Profiles of the final build, on the five main workloads (self time: time in the function body, after inlining):

**hero_hd** (native 61.9 ms/pass, wasm 66.8 ms/pass; 16981 native / 92040 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `stat_fused_hd40_grid_loop<true, true>` | 50.7 | 31.4 | 48.4 | 32.3 | 0.97 | v128_load 21%, v128_store 18%, v128_bitselect 16% |
| `rgba_to_planar_and_pool160_simd128` | 21.2 | 13.1 | 21.7 | 14.5 | 0.90 | i32x4_dot_i16x8 41%, v128_load 33%, v128_store 17% |
| `eca_fit_affine_params_ex_simd128` | 4.6 | 2.8 | 4.3 | 2.9 | 0.99 | scalar 26%, f32x4_add 16%, v128_and 13% |
| `stat_fused_hd40_grid_loop<false, true>` | 3.5 | 2.2 | 3.6 | 2.4 | 0.90 | v128_store 31%, v128_load 23%, scalar 9% |
| `compute_row_luma_means_simd128` | 3.4 | 2.1 | 3.0 | 2.0 | 1.05 | v128_load 79%, scalar 12%, f32x4_add 9% |
| `render_eulerian_pip_subpx_impl<false, true>` | 2.0 | 1.2 | 2.6 | 1.7 | 0.72 | i16x8_q15mulr_sat 23%, shuffle 12%, i16x8_add 11% |
| `sad_patch_zm_half_st1_ring_hd_simd128` | 1.4 | 0.9 | 1.5 | 1.0 | 0.88 | scalar 41%, v128_load 31%, f32x4_add 12% |

**swarm_1200** (native 140.3 ms/pass, wasm 163.4 ms/pass; 18761 native / 112618 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `extract_phase1a_raw` | 15.5 | 21.7 | 15.9 | 25.9 | 0.84 | scalar 43%, v128_any_true 37%, f32x4_mul 9% |
| `stat_fused_hd40_grid_loop<true, true>` | 12.4 | 17.4 | 10.9 | 17.8 | 0.98 | v128_store 19%, v128_load 19%, v128_bitselect 17% |
| `build_sparse_knn_graph` | 6.0 | 8.4 | 6.0 | 9.9 | 0.85 | scalar 90%, f32x4_extract_lane 4%, f32x4_splat 2% |
| `rgba_to_planar_and_pool160_simd128` | 5.7 | 8.0 | 5.6 | 9.1 | 0.88 | i32x4_dot_i16x8 43%, v128_load 29%, v128_store 16% |
| `eval_lk_patch_wasm<true, true>` | 4.2 | 5.9 | 3.3 | 5.4 | 1.10 | f32x4_pmin 48%, scalar 26%, f32x4_mul 14% |
| `extract_phase1a_3x3_max` | 3.8 | 5.3 | 4.2 | 6.9 | 0.77 | scalar 77%, i32x4_bitmask 8%, v128_any_true 7% |
| `gather_csr_stage1_track<true>` | 3.8 | 5.3 | 3.7 | 6.0 | 0.88 | scalar 100%, v128_any_true 0%, f32x4_splat 0% |

**bos** (native 104.5 ms/pass, wasm 124.7 ms/pass; 19679 native / 126742 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `build_ref_patch_8x8` | 42.0 | 43.9 | 44.8 | 55.8 | 0.79 | scalar 99%, f32x4_mul 1%, f32x4 0% |
| `eval_cur_patch_8x8` | 21.9 | 22.9 | 22.7 | 28.3 | 0.81 | scalar 89%, f32x4_add 8%, f32x4_max 3% |
| `accumulate_warmup_node_noise` | 5.3 | 5.5 | 3.0 | 3.8 | 1.47 | scalar 100% |
| `apply_laplacian_p_to_lp` | 3.4 | 3.6 | 1.7 | 2.2 | 1.65 | scalar 100% |
| `extract_dense_lk` | 3.2 | 3.3 | 2.2 | 2.8 | 1.20 | scalar 100% |
| `project_comp_zero_mean` | 3.1 | 3.2 | 2.9 | 3.6 | 0.89 | scalar 100% |
| `dct3_2d_solve_z` | 2.3 | 2.4 | 2.6 | 3.3 | 0.72 | scalar 92%, f64x2_add 4%, f64x2_sub 2% |

**rs_vibrometry** (native 814.4 ms/pass, wasm 1100.5 ms/pass; 19370 native / 133560 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `extract_signed_band_displacements` | 31.0 | 252.7 | 32.4 | 356.9 | 0.71 | scalar 100% |
| `eval_segment_optical_quadrature` | 12.9 | 104.8 | 10.5 | 115.6 | 0.91 | scalar 100% |
| `stat_fused_hd40_grid_loop<true, true>` | 11.7 | 94.9 | 9.2 | 101.4 | 0.94 | v128_load 21%, v128_store 18%, v128_bitselect 17% |
| `rgba_to_planar_and_pool160_simd128` | 9.7 | 79.0 | 8.1 | 89.4 | 0.88 | i32x4_dot_i16x8 40%, v128_load 34%, v128_store 17% |
| `stat_fused_hd40_grid_loop<false, true>` | 4.4 | 36.2 | 3.9 | 42.8 | 0.84 | v128_store 26%, v128_load 25%, scalar 11% |
| `warmup_accumulate_and_lock` | 3.2 | 26.0 | 14.2 | 156.6 | 0.17 | scalar 100% |
| `eca_fit_affine_params_ex_simd128` | 2.1 | 17.3 | 1.4 | 15.2 | 1.14 | scalar 44%, f32x4_add 13%, f32x4_mul 12% |

**ingest_roi** (native 9074.7 ms/pass, wasm 11539.1 ms/pass; 19966 native / 156068 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `solve_bench_loo_residuals` | 30.5 | 2767.5 | 33.9 | 3911.3 | 0.71 | scalar 100%, f64x2_sqrt 0% |
| `build_ref_patch_8x8` | 21.2 | 1920.7 | 21.2 | 2445.8 | 0.79 | scalar 99%, f32x4_mul 1%, f32x4 0% |
| `eval_cur_patch_8x8` | 15.7 | 1429.0 | 15.1 | 1745.0 | 0.82 | scalar 92%, f32x4_add 5%, f32x4_max 3% |
| `extract_dense_lk` | 3.2 | 293.2 | 2.0 | 227.7 | 1.29 | scalar 100% |
| `apply_laplacian_p_to_lp` | 2.7 | 242.3 | 1.4 | 166.2 | 1.46 | scalar 100% |
| `dct2_2d_solve_z` | 2.6 | 231.8 | 2.9 | 338.3 | 0.69 | scalar 88%, f64x2_add 6%, f64x2_mul 5% |
| `dct3_2d_solve_z` | 2.5 | 225.4 | 2.8 | 327.9 | 0.69 | scalar 90%, f64x2_add 5%, f64x2_sub 3% |

How to read the per-function ratios:

- **Inlining differs between the builds.** LLVM inlines differently for aarch64 and for wasm32, and V8 inlines small wasm functions too (`--wasm-inlining`, on by default). A function's *self* time therefore only compares cleanly when it is a leaf on both sides.
- **Inlining artifacts.** In `bos`, `accumulate_warmup_node_noise` (self 1.47) and `extract_dense_lk` (self 1.20) look slower natively, but their *inclusive* shares are equal (40.0% vs 40.5%, 30.8% vs 30.5%). In milliseconds both are faster natively; only the split between the function and its callees differs.
- **Tail call.** Natively `chronos_tick` tail-calls `chronos_post_tick`, so its frame disappears from native stacks.

What the profiles say:

- **Sparse tracking (`hero_*`, `mp4_occlusion`).** About 75% of HD kernel time is two streaming passes over the frame in `chronos_pre_tick`: `stat_fused_hd40_grid_loop` (51%) and `rgba_to_planar_and_pool160_simd128` (21%).
  - The grid loop is memory-bound. 41% of its samples are on vector loads and stores to frame-sized buffers, and its 1,322 instructions include only 38 vector spills. The rest is `v128_bitselect` (the background-rate selection, 16%), `u8x16_sub_sat` and `i16x8_narrow_i32x4`.
  - It runs at 0.97 of V8's time: both compilers are close to the memory system's limit. The pooling function was the native loss before the dot-product change and is now 0.90.
- **Swarm.** The time is spread out:
  - `extract_phase1a_raw` (15.5%): the per-pixel candidate scan. Its `v128_any_true` tests take 37% of its samples, the loads they wait on included.
  - The grid loop (12%), the k-NN graph build (6%, scalar) and the LK patch evaluation.
  - `eval_lk_patch_wasm` is the one hot SIMD function still slower natively (1.10). Its samples sit on the `fcmgt` of `f32x4_pmin`, waiting on the loads and the subtract/abs chain. V8 lowers `pmin` to the same two instructions.
- **BOS, ingest/ROI, vibrometry: mostly scalar code.**
  - Samples with no SIMD layer op on the stack make up 85% of `bos`, 82% of `ingest_roi` instance 1 and 54% of `rs_vibrometry`.
  - The hot functions there are scalar f64 loops: `build_ref_patch_8x8` and `eval_cur_patch_8x8` (64% of `bos`), `solve_bench_loo_residuals` (31% of `ingest_roi` instance 1), and `extract_signed_band_displacements` (31% of `rs_vibrometry`).
  - Native wins most here (0.71 to 0.82), from LLVM's aarch64 code generation against V8's.
  - Native also wins where LLVM auto-vectorises plain Rust loops for aarch64 but not for wasm32: `warmup_accumulate_and_lock` has 52 `fmul.4s` natively and no SIMD instruction in the wasm, and runs 6 times faster. That is exact, because LLVM vectorises only element-wise float work, never reductions. The same ratio holds with `--no-liftoff`, so it is not V8 tiering.
- **`ingest_roi` instance 1 dominates the set.** It is 9.1 of the 13.9 s of native kernel time in the canonical set, and almost all of it is `chronos_post_tick`: `solve_bench_loo_residuals` (2.8 s per pass) and the BOS dense-LK node loop.

Which SIMD ops matter, by share of all kernel samples. This is aggregated over the 14 hottest functions of each workload, before the dot-product change:

- **`i32x4_dot_i16x8`** 12.9% of `hero_hd`, 5.9% of `rs_vibrometry`, 3.4% of the swarm. Now `smull` + `smull2` + `addp` as written.
- **Shuffles** up to 8.1%: mostly the `uzp1`/`uzp2` pairs after the dot products, since folded into `addp`.
- **`v128_any_true`** 5.5% of the swarm.
- **Narrows** up to 2.1%.
- **`i32x4_bitmask`** 0.8% at most.
- **`v128_bitselect`** 7.6% of `hero_hd`.

Except for the dot product, each of these lowers to the same instructions V8 emits for the wasm op, and no cheaper NEON sequence gives the exact wasm result for every input.

**Non-inlined helpers with `v128` arguments.** On Rust 1.99 a `v128` passed by value to a non-inlined Rust function goes through memory. 112 kernel functions take or return `v128`. 99 are `#[inline(always)]`. Of the 13 `#[inline(never)]` ones, 7 take `v128` by value (none returns one): `subpx_row_blk_chunk` (7 values), `pan_hd40_band`, `subpx_hd40_band_fused`, `pan_subpx_fused_hd40_no_scratch` (2 each), `pan_fused_grid_le40_row`, `pan_subpx_fused_grid_fallback` (15 each) and `eval_cand_4px` (6). They are called per band, row or edge candidate, never per pixel. Their whole bodies account for 0% (`hero_hd`, swarm, `ingest_roi`), 0.35% (`bos`) and 2.1% (`rs_vibrometry`) of native kernel time. They load 0 to 9 `q` registers from argument pointers per call. Removing the `#[inline(never)]` would buy well under 0.1%.

### NEON backend change: `i32x4_dot_i16x8`

Before this change the profile showed `rgba_to_planar_and_pool160_simd128` (the RGBA-to-planar pooling in `chronos_pre_tick`) 1.16 to 1.27 times slower natively than in V8 (now 0.88 to 0.90), with 45% of its samples on `i32x4_dot_i16x8`. Given `vmull_s16` + `vmull_high_s16` + `vpaddq_s32`, LLVM saw that the constant weight vector's two halves are equal and rewrote the per-4-pixel luma step (`u8` pixels widened to `i16`, two dot products, then `i32x4_shuffle::<0, 2, 4, 6>` + `<1, 3, 5, 7>` + add) into `mov d, v[1]` + `umull` pairs and extra `uzp1` shuffles: 19 instructions per 4 pixels. `src/simd/neon.rs` now emits the same three instructions as inline assembly, which LLVM keeps as written and then folds the shuffle-and-add into one `addp`: 13 instructions per 4 pixels, and the function shrinks from 2,004 to 1,538 instructions. The per-op conformance suite (including the heavy NEON run, 6.4 min) and the 29-trace parity replay pass with 0 mismatches.

Before and after (native only, 3 interleaved rounds x 3 passes):

| Workload | before ms/pass | after ms/pass | after / before | round range |
| :--- | ---: | ---: | ---: | :--- |
| `hero_sd` | 22.63 | 20.81 | 0.920 | 0.913-0.932 |
| `hero_hd` | 67.98 | 62.07 | 0.913 | 0.906-0.919 |
| `hero_suite` | 1647.2 | 1519.2 | 0.922 | 0.918-0.930 |
| `mp4_occlusion` | 37.31 | 35.30 | 0.946 | 0.945-0.947 |
| `swarm_1200` | 146.4 | 142.1 | 0.970 | 0.953-0.972 |
| `bos` | 106.1 | 104.9 | 0.988 | 0.976-1.005 |
| `hodge` | 76.09 | 74.68 | 0.981 | 0.969-1.004 |
| `rs_vibrometry` | 856.2 | 815.9 | 0.953 | 0.949-0.963 |
| `conservation` | 44.89 | 41.63 | 0.927 | 0.902-0.946 |
| `covariance` | 227.2 | 214.4 | 0.943 | 0.938-0.943 |
| `ingest_roi` | 10647.9 | 10638.2 | 0.999 | 0.997-0.999 |
| geomean | | | 0.951 | |

`chronos_pre_tick` alone is 9 to 10% faster on every tick workload (for example `hero_hd` 255.7 -> 231.1 µs median); no workload got slower.

### Optimisation opportunities, ranked (for the next sprint)

Shares are of native kernel time on the workload named, from the profiles above. "Exact" means the change can keep every result bit-identical to the wasm. Every change except the backend one is a kernel change, so it would also rebuild `web/chronos_spectral.wasm`.

| # | What | Where | Measured share | Expected gain | Exact? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Threads for the independent per-item loops** (native only). Per node: the BOS dense-LK and warm-up node loops. Per track: `solve_bench_loo_residuals` and the swarm candidate and LK loops. Per band/cell: the vibrometry loops (their independence is not yet checked). Per row band: the grid loop. Each item is computed exactly as now; cross-item sums (`disp2_sum`, `acc_err2`/`acc_sig2`, ...) are added afterwards in the original order. | `bos.rs` `extract_dense_lk` / `accumulate_warmup_node_noise`; `continuum_physics.rs:10820`; `vibrometry.rs`; `simd_ops.rs` grid loop | `ingest_roi` i001: 31% LOO + 36-39% dense LK. `bos`: 71% (warm-up + dense LK). `rs_vibrometry`: 31% + 13%. `hero_hd`: 51% (grid loop) | Up to the core count on those loops (8 P-cores here). Plausibly 2 to 4 times on `bos` and `ingest_roi`, less on sparse tracking, where frames take 0.09-0.27 ms and thread hand-off costs microseconds | Yes, with fixed partitioning and ordered reductions. Costs: per-thread scratch in the arena (the patch buffers are shared today) and a thread source for a `no_std` kernel (host callback or `std` feature). The wasm stays single-threaded (wasm threads need cross-origin isolation) |
| 2 | **Vectorise the BOS patch builders.** Bilinear sampling, gradients and residuals per pixel in `f32x4`. f64 sums stay scalar, in index order | `bos.rs:504` `build_ref_patch_8x8`, `bos.rs:776` `eval_cur_patch_8x8` | `bos` 42% + 22%; `ingest_roi` i001 21% + 16%. 89-99% scalar code today | Estimate 1.5 to 3 times on these functions (most of their arithmetic is element-wise); not measured | Yes, if the f64 accumulation order is kept. Helps the wasm too |
| 3 | **Reuse reference patches.** `build_ref_patch_8x8` depends only on `ref_luma` and the node. Today it is rebuilt for every node every frame (dense LK, warm-up) and three times per anchor per candidate in `eval_anchor_mse_at` | `bos.rs:1642`, `1778`, `3551`, `1531-1587` | Up to the `build_ref_patch_8x8` share above (42% `bos`, 21% `ingest_roi` i001) while the reference is unchanged | Up to those shares; depends on how often reference maintenance (`bos.rs:1983`, `3382`) rewrites `ref_luma`, which was not measured | Yes, reused values are identical, but validity needs per-node invalidation. All 8,192 `PatchRefData` (~1.4 KB each, 11.5 MB) do not fit the wasm's ~6.6 MB of free memory, so it is native-only or a subset |
| 4 | **Fuse the two streaming passes of `chronos_pre_tick`.** The RGBA-to-planar pooling and the static/motion grid each read the full RGBA frame | `simd_ops.rs` `rgba_to_planar_and_pool160_simd128`, `stat_fused_hd40_grid_loop` | `hero_hd` 21% + 51%; swarm 6% + 12%; `rs_vibrometry` 10% + 16% | One fewer pass over 2 x 0.9 MB of RGBA per HD frame in a memory-bound loop; estimate 10-20% of `pre_tick`, not measured | Yes (per-pixel arithmetic unchanged) |
| 5 | NEON `i32x4_dot_i16x8` as `smull` + `smull2` + `addp` | `src/simd/neon.rs` | was 12.9% of `hero_hd` | **Done**: `pre_tick` 9-10% faster on every tick workload, geomean 0.951 | Yes (conformance + parity pass) |
| 6 | Fewer `v128_any_true` exits in the candidate scan: OR 16-32 pixels of hit masks before one test | `soa_tracks.rs` `extract_phase1a_raw` | swarm 15.5% (any_true 37% of it) | Small, estimate under 5% of the swarm | Yes (control flow only) |
| 7 | Find why `apply_laplacian_p_to_lp` is slower natively | `bos.rs:2681` | 2.7-3.4% of `bos`/`ingest_roi`, 1.46-1.65 times V8 | 1-3% of those workloads | Yes. A leaf on both sides with the same branch and load/store structure as V8's TurboFan code (checked with `--print-wasm-code-function-index=392`); suspected memory dependence on `lp[u + 1]` across iterations, so measure with CPU counters first |
| 8 | Remove `#[inline(never)]` from the 7 by-value-`v128` helpers | `simd_ops.rs`, `soa_tracks.rs:3419` | at most 2.1% including their bodies | Under 0.1% | Yes. Not worth a wasm rebuild |
| 9 | Other multi-instruction NEON ops (`any_true`, `pmin`/`pmax`, narrows, bitmasks) | `src/simd/neon.rs` | 0.8-5.5% each, as V8 | None found: each is already the shortest exact sequence and matches V8's lowering | |
| 10 | 256-bit SIMD | | | None on Apple Silicon: NEON is 128-bit and there is no SVE. SME's streaming SVE needs a mode switch and is not general-purpose; M4 P-cores already issue up to four 128-bit vector ops per cycle. On x86-64, AVX2 would need the kernel to issue two `v128` ops per 256-bit op (or a 256-bit layer). It is exact only with lane-wise identical op sequences, no FMA and unchanged reduction order. Not measurable here (Rosetta) | Possible on x86 with those rules |
| 11 | Build flags: fat LTO, `codegen-units = 1`, `panic = "abort"`, `target-cpu=native` | `native/Cargo.toml` | | **None** (measured, table above) | Yes |

For the wasm itself the profile suggests one more item: LLVM does not auto-vectorise for wasm32 the loops it vectorises for aarch64 (`warmup_accumulate_and_lock`, 6 times slower in V8). Hand-written `f32x4` code there would help the shipped wasm. It would not help native.
