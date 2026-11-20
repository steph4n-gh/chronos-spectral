# Verification, audit gates, and benchmarks

Every architectural guarantee in CHRONOS-SPECTRAL is enforced automatically by deterministic verification gates across binary layout, mathematical counterexamples, real-camera compressed H.264 MP4s, dense `1,000+` target swarms, and headless Chromium DevTools end-to-end execution.

## How to run the verification suites

Ensure Rust (`wasm32-unknown-unknown`), Node.js (`v22+`), and `python3` are installed. Install `wabt` (`wasm-objdump`), `ffmpeg`, and Google Chrome to exercise the complete binary, video, and browser pipeline.

### Offline build, static audit, math gates, and Gate Q

```bash
./build.sh --check
```

Runs ten sequential offline stages:
1. Release compile of `chronos_spectral.wasm` (`opt-level = 3`, fat LTO, `codegen-units = 1`, `panic = "abort"`, `+simd128`).
2. `wasm-objdump` binary audit (`76` exports, `0` host imports, fixed `initial=448 max=448` linear memory, `0` `memory.grow` opcodes, `0` `call_indirect`, `0` call cycles, `0` allocator symbols, `13,776 B` max call-graph shadow stack below `16,384 B`, `101,655` SIMD128 ops, `8` `i32.clz` ops).
3. Mathematical counterexample gates (`Gates A-P`, including `N1` and `N2`).
4. `5x2` interleaved `320x180` SD vs `640x360` HD Blind Pixel Pillar benchmark.
5. `Gate Q` high-density `1,000+` target swarm and continuum physics suite (`scripts/benchmark_1000.mjs --gate`, `21/21` checks).
6. Wave-1 gates (`gate_i0_i4_ingest.mjs`, `gate_i1_cov_residual.mjs`, `gate_i5_bos.mjs`).
7. Wave-2 gates (`gate_i2_rs_welch.mjs`, `gate_i3_hodge.mjs`, `gate_i5b_conservation.mjs`).
8. `Gate X` adversarial C-ABI fuzz (`scripts/gate_abi_fuzz.mjs`): random export calls with hostile arguments, pixels and staging buffers, `0` traps required (see [Adversarial C-ABI fuzz](#adversarial-c-abi-fuzz-gate-x)).
9. `Gate S` SIMD layer conformance (`scripts/gate_simd_conformance.mjs`): every op of the `src/simd/` layer (`125` ops, `206` entries including every const-generic instantiation the kernel uses) run as real Wasm SIMD128 in V8 against the portable `simd::scalar` reference over edge-value and seeded random lanes (`0` mismatches required, `9` deliberately broken negative controls must each be detected), then the same suite natively (`cargo test --target host-tuple`) and the native library link (`native/Cargo.toml`), both with zero warnings.
10. `Gate N` native/wasm whole-kernel parity (`scripts/gate_native_parity.mjs`): records a fresh `104 MB` trace set from the wasm (real-MP4 sparse tracking, Hodge, conservation, the first `24 MiB` of the `1,200`-target swarm; `3,088` calls) and replays it on the native build with overflow checks on: every return value and every region digest must match bit for bit, and a corrupted copy of one trace must be rejected at the right call (see [Native replay driver](#native-replay-driver-nativereplay)).

### Full end-to-end pipeline (CI entrypoint)

```bash
./build.sh --e2e
```

Runs everything in `--check`, plus Gate `I6` physical capture fixtures (`scripts/gate_i6_captures.mjs`), SD and HD H.264 MP4 calibration gates (`scripts/calibrate_mp4.mjs`), and headless Chromium CDP full-stack UI and audio/video verification (`scripts/e2e_browser.mjs` Steps `1-10e`).

### Native backends only (`--native-check`)

```bash
./build.sh --native-check
```

Runs stages 1, 2, 9 and 10 of `--check`: the release build, the `wasm-objdump` audit, `Gate S` and `Gate N`. It skips the Wasm-only suites (stages 3 to 8). This is what the macOS arm64 CI job runs to prove the NEON backend. It needs no Chrome. Without `ffmpeg` on `PATH` it skips Gate N's real-MP4 scenario (`hero_sd`) and says so; `--check` and `--e2e` always run every scenario.

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

# Adversarial C-ABI fuzz (Gate X); --trials, --frames, --timeout, --wasm
node scripts/gate_abi_fuzz.mjs

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

Measured directly on `web/chronos_spectral.wasm` (`2,479,630 B`, v3.1.1):

| Property | Measured value | Gate threshold |
| :--- | :--- | :--- |
| Export count & host imports | `76` exports (`memory` and `75` functions, one of them the test hook `chronos_test_hook_set_frame_counters`), `0` imports (`0 env`, `0 malloc/free`) | Strictly `0` host imports |
| Wasm linear memory | `initial=448 max=448` (`29,360,128 B`) | `448` fixed pages, `0` `memory.grow` |
| Arena size | `23,300,224 B` (v3.1.0: `22,644,848 B`; v3.1.1 appends the `655,376 B` BOS reference-patch cache, `bos::BosRefCache`, after every region) | `<= 24,117,248 B` (`Arena` static bound) |
| Static `.rodata` end | `59,216 B` | Exactly `59,216 B` (`build.sh`): `ARENA` is the only `.bss` symbol and starts there, so every region getter keeps its v3.0.0 address. `RODATA_PAD_BYTES` in [src/lib.rs](../src/lib.rs) absorbs kernel changes that add or remove panic locations, and `build.sh` prints the value to set |
| Static data end (`dualStaticEnd`) | `23,359,440 B` | `<= 24,248,320 B` |
| Max DFS shadow stack | `13,776 B` (`0` call cycles, `0` `call_indirect`) | `<= 14,336 B` soft bound, `<= 16,384 B` hardware bound |
| Emitted vector / bitwise ops | `101,901` SIMD128 ops, `8` `i32.clz` ops | Non-zero SIMD128 & hardware CLZ |
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

## Adversarial C-ABI fuzz (`Gate X`)

[scripts/gate_abi_fuzz.mjs](../scripts/gate_abi_fuzz.mjs) drives `web/chronos_spectral.wasm` the way a careless or hostile host could. Each trial is a fresh instance: `chronos_init`, SD or HD, then 48 frames. Every frame writes pixels into the ingest slot (random noise, black, white, a black/white strobe, alpha-0 stripes, or moving discs on texture), then on a random fraction of frames calls 1 to 4 random exports (every function except the four tick/init exports and the pointer getters, the test hook included) with arguments drawn from a hostile pool (`0`, `-1`, `2^31 - 1`, `2^31`, `2^32 - 1`, frame sizes and slot counts `+-1`, `NaN`, `+-Infinity`, `+-1e-38`, `1e-45`, `+-1e38`, `3.4e38`, audio rates, ...), after filling a random staging buffer (audio, Hodge injection, conservation input, Laplacian or Fiedler vector) with `NaN`, `+-1e30` noise or pool values. It then ticks with `chronos_tick`, `pre_tick` + `post_tick`, the two out of order, or all three, mostly at `1/30 s` and otherwise at a hostile step (`0`, negative, `NaN`, `+-Infinity`, `1e-40`, `1e9`), and finally calls every pointer getter (`chronos_get_ring_slot_ptr` with a pool argument) and reads the `f32` views `telemetry`, `swarm_telemetry`, `conservation_telemetry` and `slot_meta`.

- **Gate:** `0` traps, and every trial ticks all of its frames. A trap is a wasm trap or JS exception thrown by any export call (the frame-size and pointer getters, and the typed-array views over the pointers they return, included), or a change of the fixed `448`-page memory size. Each one is printed with its trial and the 12 calls before it.
- **Reported, not gated:** every non-finite value in those views, by slot, with its first occurrence.
- **Deterministic and bounded:** trial `k` draws everything from its own seeded PRNG, so a given wasm makes the same calls on every run. The default `300` trials x `48` frames run in a worker thread under a `300 s` watchdog, so a call that never returns fails the gate, naming the trial and call count reached.

Measured on the v3.1.0 wasm (Apple M4 Pro): `14,400` frames ticked, `37,202` export calls, `0` traps, `0` non-finite values, about `13 s`. The v3.0.0 wasm gives the same result (it lacks the test hook, so `37,101` calls).

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

Re-recorded on v3.1.1 as `canon-v13` (sha256 `c62846a7...`, 3 min on the same machine): every scenario and every instance has exactly the same call count, host-write count, digest-checked call count and file size as `canon-v12` below (29 traces, 58,838 calls, 3,461,034 host writes, 2,541,506,704 bytes), as expected of a release that changes no output. Replaying all 29 against the v3.1.1 wasm gives 0 mismatches. The table and notes below are of the v3.1.0 recording (`canon-v12`).

`record_canonical_traces.mjs` with default caps: 11 scenarios, 29 traces, 2.54 GB, recorded in 3 min 51 s on Apple Silicon (v3.1.0 wasm, format 1.1). Replaying all 29 against the wasm takes about 40 s with 0 mismatches, also with `--portable` (no `UNMAPPED` record). Compared with the v3.0.0 recording, every scenario has identical call and host-write counts except `bos` (+1 call) and `ingest_roi` (+3 calls): the test-hook calls that replace the raw-address writes (see below).

| Scenario | Harness | Traces | Calls | Host writes | Size |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `hero_sd` | `calibrate_mp4.mjs --clip=juggling_ballistic` (SD) | 2 | 1,126 | 145,042 + 1 reset | 45.2 MB |
| `hero_hd` | `calibrate_mp4.mjs --hd --clip=juggling_ballistic` | 2 | 1,126 | 613,215 + 1 reset | 162.8 MB |
| `hero_suite` | build.sh `[3/5]` Gates A-P + SD/HD pillar trials | 1 | 31,371 | 754,082 | 677.0 MB |
| `mp4_occlusion` | `calibrate_mp4.mjs --clip=roundabout_occlusion` | 2 | 1,241 | 821 + 1 reset | 58.9 MB |
| `swarm_1200` | `benchmark_1000.mjs --only=Q1` | 1 | 719 | 636,687 | 50.0 MB |
| `bos` | `gate_i5_bos.mjs` | 1 | 2,465 | 109,002 | 110.2 MB |
| `hodge` | `gate_i3_hodge.mjs` | 1 | 175 | 95,067 | 19.2 MB |
| `rs_vibrometry` | `gate_i2_rs_welch.mjs` | 1 | 6,739 | 401,893 | 1,054.8 MB |
| `conservation` | `gate_i5b_conservation.mjs` | 1 | 1,424 | 15,662 | 14.8 MB |
| `covariance` | `gate_i1_cov_residual.mjs` (10 of 406 instances) | 10 | 4,742 | 338,512 | 127.5 MB |
| `ingest_roi` | `gate_i0_i4_ingest.mjs` | 7 | 7,710 | 351,051 | 221.0 MB |

Each `calibrate_mp4.mjs` scenario also has a 0-call trace for its probe instance. Recording costs about 4 ms per kernel call (SD hero tick `0.12 -> 4.1 ms`, Q1 swarm tick `1.3 -> 5.7 ms`). Strict mode with `--memdigest=1` takes about 2.4 times as long as default recording.

Portable frame-counter writes: `gate_i5_bos.mjs` and `gate_i0_i4_ingest.mjs` test frame-index wraparound by moving the private `Arena` fields `ring_cursor` and `frame_idx` to just before `u32::MAX`. Up to v3.0.0 they overwrote those fields by raw address (found by scanning memory), which the hook records as `UNMAPPED`, so a native build could follow three canonical traces only up to call `#209` (`bos`), `#58` (`ingest_roi` instance 2) and `#710` (`ingest_roi` instance 6). Since v3.1.0 they make exactly the same change through the test-hook export `chronos_test_hook_set_frame_counters(frame_idx, ring_cursor_lo, ring_cursor_hi)`, which a trace records as an ordinary call, and the canonical set has no `UNMAPPED` record. The native driver still stops a trace that has one at that call and reports `portable prefix ended at call N`, which is distinct from both a pass and a failure.

### Sessions recorded on v3.0.0, replayed on v3.1.0

v3.1.0 is the first change of the shipped wasm since the canonical set was recorded, so the v3.0.0 recording (sha256 `7ba36a5b...`, the 29 traces of the table above as they were before this release: `bos` 2,464 calls, `ingest_roi` 7,707, with the four raw-address writes) was replayed on the v3.1.0 wasm (sha256 `f4ad7ce5...`) with `node scripts/trace_replay.mjs --allow-wasm-mismatch`: 29 traces, 58,834 calls, 44,307 digest-checked calls, 1,993,815 region digests, **0 mismatches**. Every return value matched, pointer results included: all 36 region getters return the same addresses in both builds (the release changes no `Arena` field, and the static data ends at the same address, `22,704,064`), so the four raw-address writes still land on `ring_cursor` and `frame_idx`. That is the expected result. The frame-index fix changes results only when a window loop runs past `u32::MAX` with fewer samples than its count bound. The sessions that tick through `u32::MAX` (`bos` and `ingest_roi` instances 2 and 6) never do: the only window loop that reaches that frame is in `bos` at call `#263`, and it has a full 64-frame window, which the old loop also ended on its count bound. The collapsed padding changes only panic locations, and no session panics.

### Sessions recorded on v3.1.0, replayed on v3.1.1

v3.1.1 changes kernel code for speed only (see [Native vs wasm kernel timing](#native-vs-wasm-kernel-timing)), plus one behaviour change at the frame-counter wrap, so the whole `canon-v12` set (29 traces recorded on v3.1.0, sha256 `f4ad7ce5...`) must replay on it unchanged. `node scripts/trace_replay.mjs --allow-wasm-mismatch` on the v3.1.1 wasm (sha256 `c62846a7...`): 29 traces, 58,838 calls, 44,311 digest-checked calls, 1,993,995 region digests, **0 mismatches**, pointer results included. Every commit of the release was checked the same way before it landed (wasm, NEON and `simd-force-scalar`, 29 traces each), and each commit message names its evidence.

- **Region addresses.** `ARENA` is the wasm's only `.bss` symbol and the linker places it directly after `.rodata`, so its address is `16,384 +` the size of `.rodata`. The first v3.1.1 kernel change removed 39 bounds checks, `.rodata` shrank by 624 bytes, and every pointer result of the replay differed from the recording. `RODATA_PAD` in [src/lib.rs](../src/lib.rs) (a `#[used]`, wasm-only byte array) now keeps `.rodata` at exactly 42,832 bytes, and `build.sh` fails unless it ends at `59,216` and prints the `RODATA_PAD_BYTES` value that restores it. All 36 region getters return their v3.0.0 addresses.
- **Memory layout.** v3.1.1 adds one `Arena` field, `bos_ref_cache` (`bos::BosRefCache`, 655,376 B), after every existing field (`Arena` is `repr(C)`), so no region moves. `size_of::<Arena>()` grows from 22,644,848 to 23,300,224 B (native: 22,651,136 to 23,306,512) and the static data end from 22,704,064 to 23,359,440 B, inside the 24,117,248 and 24,248,320 B bounds and the 448 pages. Those sizes appear only in the [platform words](#platform-word-allowlist), which are masked. `BosCtrl` turns one reserved word into `ref_gen` and stays 128 bytes.
- **Frame-wrap fix.** Two comparisons in [src/physics_id.rs](../src/physics_id.rs), `f_c < last_discont_frame + 5` and `birth_frame + 2 < f_c`, overflowed `u32` within five (two) frames of `u32::MAX`; they are now made in `u64`. The first is reachable at `frame_idx == u32::MAX` from the pair scan, but the event it guards is invalid either way, and the second is not reachable from the exports (its caller requires four post-impact samples of the born track first). So no session can change; the regression tests are `pair_candidate_after_discontinuity_near_u32_max` and `stationary_birth_partner_born_near_u32_max` in `physics_id::frame_wrap_tests`.

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

The JS replayer refuses a trace recorded against a different wasm build unless given `--allow-wasm-mismatch`, which replays sessions recorded on one release against another (see [Sessions recorded on v3.0.0, replayed on v3.1.0](#sessions-recorded-on-v300-replayed-on-v310)). It still checks everything, pointer results included: equal pointers show that every region is where it was, which `U` records rely on. Whole-memory digests cover the data section too, so they are skipped and counted.

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
| `telemetry[20]` | `23300224.0` | `23306512.0` | `size_of::<Arena>() as f32`: the arena holds `usize` fields (for example `LanczosScratch::act_idx: [usize; 32]`), so its size depends on pointer width. |
| `wave1_telemetry[1]` | `23300224` | `23306512` | `size_of::<Arena>() as u32`, as above. |
| `wave1_telemetry[12]` | `23359440` | `0` | `linked_static_end_bytes()`: the wasm linker's `__heap_base`. A native build has no linear memory. |
| `wave2_telemetry[1]` | `23300224` | `23306512` | `size_of::<Arena>() as u32`, as above. |

That is 16 bytes exempt per digest-checked call, out of 12,715,152 bytes hashed (0.000126%): 4 of 6,272 bytes of `telemetry`, 8 of 768 of `wave1_telemetry`, 4 of 256 of `wave2_telemetry`, and none of the other 42 regions. These were the only differences on the first native replay (the masked words, diffed with `trace_region_diff.mjs`). No other word differs anywhere in the canonical set.

### Native parity results (measured)

**v3.1.1.** The v3.1.1 canonical set (`canon-v13`, recorded from the v3.1.1 wasm) replays in full with Gate N's settings on the NEON backend (28.8 s) and on the scalar reference (80.5 s): 29 pass, 0 portable prefix, 0 fail, with the same calls, digest-checked calls and region digests as the table below. The SSE backend (an `x86_64-apple-darwin` build, `+ssse3,+sse4.1`, run under Rosetta 2) replays the v3.1.0 set (`canon-v12`) with the same result: 29 pass. So do NEON and the scalar reference in the `release` profile, before every commit of the release.

**v3.1.0.** The v3.1.0 canonical set (format 1.1, recorded from the v3.1.0 wasm) replayed in full on the native build with Gate N's settings (`checked` profile: overflow checks and debug assertions, including core's UB precondition checks; `--require-full --wasm-sha256=<web wasm>`), Apple M4 Pro (aarch64), on the NEON backend (39.6 s for 2.54 GB) and on the scalar reference (`--features simd-force-scalar`, 99.4 s). Both give the same table:

| Scenario | Traces | Calls replayed | Digest-checked calls | Region digests | Result |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `hero_sd` | 2 | 1,126 | 934 | 42,030 | pass |
| `hero_hd` | 2 | 1,126 | 934 | 42,030 | pass |
| `hero_suite` | 1 | 31,371 | 25,927 | 1,166,715 | pass |
| `mp4_occlusion` | 2 | 1,241 | 1,026 | 46,170 | pass |
| `swarm_1200` | 1 | 719 | 604 | 27,180 | pass |
| `bos` | 1 | 2,465 | 1,685 | 75,825 | pass |
| `hodge` | 1 | 175 | 125 | 5,625 | pass |
| `rs_vibrometry` | 1 | 6,739 | 5,413 | 243,585 | pass |
| `conservation` | 1 | 1,424 | 958 | 43,110 | pass |
| `covariance` | 10 | 4,742 | 1,086 | 48,870 | pass |
| `ingest_roi` | 7 | 7,710 | 5,619 | 252,855 | pass |
| **total** | **29** | **58,838** | **44,311** | **1,993,995** | **0 mismatches** (29 pass, 0 portable prefix, 0 fail) |

The v3.0.0 set gave the same result up to the three portable prefixes (`bos` at call `#209`, `ingest_roi` instance 2 at `#58`, instance 6 at `#710`: 56,397 calls, 26 pass and 3 portable prefix) on the scalar build in the release profile, on `checked` too, and on the unoptimised `dev` profile for the subset it was run on (`hero_sd`, `swarm_1200`, `mp4_occlusion`, `bos`, `hodge`, `conservation`, two `covariance` and two `ingest_roi` instances).

Expected divergence sources that turned out not to diverge, and why:

- **`isize` clamps** (`(x + d).min(N - 1) as usize` at `src/soa_tracks.rs` 2945, 2947, 8444, 8446 and `src/continuum_physics.rs` 5488, 5498, 5692, 5711, 8639, 8660). A negative value would give a different `usize` on 32 and 64 bits, but none can be negative. In `soa_tracks.rs` the base is `cell_coord(..)`, which is clamped to `[0, max_cell]`. In `continuum_physics.rs`, `row_s + d_max` has `row >= 0` and `dy_cells` clamped to `[1, FIELD_GRID_H - 1]` by `compute_row_dx_spans`, and every `col + dx_c` is behind a `dx_c >= 0` guard.
- **`src/simd_ops.rs:10504`** aligns the `roi_luma` scratch pointer up to 16 bytes, so the h-cache starts at a different offset in the scratch on each target. That is harmless: the cache keys (`cached_sy`, `cached_fused_sy`) are per-call locals initialised to `i32::MIN`, so every cache row is written before it is read in the same call. The other `roi_luma` users fill their window before reading it. The extent used, at most a few KiB plus 15 bytes of slack, is far inside the 36,864-byte buffer.
- **`usize` fields** change the arena layout (`23,300,224` vs `23,306,512` bytes in v3.1.1; `22,644,848` vs `22,651,136` in v3.1.0), but nothing reads the arena as bytes except `chronos_init`'s zeroing and the size words in the allowlist.
- **Scalar backend and NaN payloads.** No scalar-backend op disagreed with V8's SIMD128 on any trace. A NaN census of every digested `f32` region at every digest-checked call (an instrumented throwaway build) found that the kernel never writes a NaN into an output region in the canonical set. The only NaNs present are host-written invalid-input tests (`hodge_inject`, `conservation_input`, `0x7fc00000`), which the driver copies byte for byte. So the payload of generated NaNs, which differs between aarch64 (`0x7fc00000`) and x86 (`0xffc00000`), cannot affect these traces.

**Overflow checks across the frame-index wrap.** The `bos` and `ingest_roi` traces move `ring_cursor` and `frame_idx` to `2^32 - 18` and `2^32 - 3` and tick through the wrap, so with the `checked` profile they are the canonical test of frame arithmetic at `u32::MAX`. On v3.0.0 they could not be followed natively past the raw-address writes; an experiment in a throwaway copy that mapped those writes onto the native fields found exactly one overflow, `src/physics_id.rs:2713` `f += 1` in `extract_compensated_window_from_ring` at `f == u32::MAX` (`bos`, call `#263`). On v3.1.0 the traces are portable and pass in full under `checked` (table above). The same v3.1.0 `bos` trace replayed on a driver built from the v3.1.0 sources with only `src/physics_id.rs` reverted to v3.0.0 aborts at call `#264` (`chronos_tick`) with `attempt to add with overflow` at that line. See "Frame-index wraparound in physics-ID window loops" below.

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

The native build exists to be faster than the wasm in V8 on every workload, so it is timed against the wasm on identical inputs: the canonical CSTRACE set (`canon-v12`, the 29 traces recorded on v3.1.0, see [Canonical set (measured)](#canonical-set-measured)), replayed call for call on both. The same harness times one kernel build against another (`--wasm-bin`), which is how every v3.1.1 change was measured. Parity is checked separately (the 29-trace replay with digests, [Native parity results](#native-parity-results-measured)); a timing run checks only return values.

### What is timed, and how the two sides are kept equal

| | wasm in V8 (`node scripts/trace_replay.mjs --bench`) | native (`chronos_replay --bench`) |
| :--- | :--- | :--- |
| Input | the trace, read into memory once before timing | the same |
| Call sequence | every `C` record up to the first `U` (raw-address) record or recorded trap | the same: a native replay cannot follow a `U` write, so both stop there. `canon-v12` has none, so every session runs in full (the v3.0.0 numbers, on `canon-v11`, stopped `bos` at call 209 and `ingest_roi` instances 2 and 6 at calls 58 and 710: `bos` was 126 ms per pass then and is 4.0 s now) |
| Each pass | fresh-instance state (the pristine memory image), then every record | fresh-instance state (`native::reset_to_pristine`), then every record |
| Timed | each kernel call alone: `performance.now()` before and after | each kernel call alone: `Instant::now()` before and after |
| Not timed | host writes (`W`), resets, region getters, record decoding | the same |
| Included in a call | V8's JS-to-wasm wrapper and argument conversion (the app pays it too) | the dispatch adapter that decodes the recorded argument bits (a few ns) |
| Checks | return values compared with the recording outside the timed region; digests off | the same |

Both clocks are monotonic and, on macOS, read the same 24 MHz `mach_absolute_time` timebase: 41.7 ns resolution. The mean empty interval (`t1 - t0` with nothing between) is measured at start-up, 13-15 ns native and 8-18 ns in V8, and subtracted from every sample (clamped at 0). Calls under about 1 µs (getters, setters) are resolution-limited and their medians are quantised; the tick and solver exports take 20 µs to 9 ms, where neither effect matters.

Fairness:

- **Warm-up**: each process discards `--warmup` passes (2 below). V8 compiles every function with Liftoff first and tiers hot ones up to TurboFan during them. Check: V8 with `--no-liftoff` (TurboFan from the first call) gives the same steady state (table under [Sanity checks](#sanity-checks)); the first, cold pass is 1.2 to 1.8 times slower and the second is already within 1% of steady state.
- **Repetition and drift**: `--passes` measured passes per process (3), `--rounds` processes per workload and side (5). Every round runs every workload on every side, side order alternating (`A, B, C, D`, then `D, C, B, A`), so slow drift on a shared machine affects every side alike.
- **Statistics**: per workload, the median of the 15 per-pass kernel totals. *Spread* is the range of the 5 per-round medians relative to the median. The ratio's *round range* is the minimum and maximum of the 5 per-round ratios (each side's median from the same round), which cancels drift between rounds; the headline ratio is of the pooled medians, so it can fall just outside that range. Per export: median, p95 and mean over every call of every measured pass of every round (raw samples are in the JSON).
- **Integrity**: `bench_native_vs_wasm.mjs` fails unless every side replayed the identical sequence (per trace: call and write counts and stop point; per export: call counts) with 0 return-value mismatches, in every round.
- **Machine**: Apple M4 Pro (12 cores), macOS 27.3, Node 26.9.0 (V8 14.6.202.34), Rust 1.99.0. For v3.1.1 no other agent ran, but system processes (a FileProvider check service at ~85% of one core, a backup agent, the editor) kept the load average at 3 to 5 throughout; xctrace recorded 98.6 to 99.5% of the profiled samples on performance cores, and the spreads below show how much noise remained. (The v3.0.0 measurements ran beside another agent's CI and Miri jobs, load average 5 to 7.)

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

# The comparison below: two kernel builds, each as wasm and NEON (about 65 min; --out=DIR keeps every
# per-run JSON and log). --wasm-bin replays with --allow-wasm-mismatch; return values are still checked.
node scripts/bench_native_vs_wasm.mjs --wasm-bin=wasm-v310=<v3.1.0 wasm> --wasm-bin=wasm-v311=web/chronos_spectral.wasm \
    --native=neon-v310=<v3.1.0 chronos_replay> --native=neon-v311=$R/release/chronos_replay \
    --rounds=5 --warmup=2 --passes=3 <canon-v12>/*/
# The scalar reference as a third native side, and wall-clock reconciliation
node scripts/bench_native_vs_wasm.mjs --native=neon=$R/release/chronos_replay \
    --native=scalar=target/native-scalar/aarch64-apple-darwin/release/chronos_replay \
    --rounds=5 --warmup=2 --passes=3 --reconcile=1 <canon-v12>/*/

# Build profiles side by side (no wasm); V8 tiering check
node scripts/bench_native_vs_wasm.mjs --no-wasm --native=a=<bin> --native=b=<bin> ... <dirs>
node scripts/bench_native_vs_wasm.mjs --wasm-variant=wasm= --wasm-variant=wasm-turbofan=--no-liftoff --native=neon=<bin> <dirs>

# Where the time goes on one trace (macOS with Xcode)
node scripts/bench_profile.mjs --bin=$R/profiling/chronos_replay --trace=<file.cstrace> --seconds=20 --wasm-passes=100
node scripts/bench_profile.mjs --same-code $R/release/chronos_replay $R/profiling/chronos_replay
```

### Results: v3.1.1 against v3.1.0, wasm in V8 and native NEON (measured)

v3.1.0 (`web/chronos_spectral.wasm` sha256 `f4ad7ce5...`, and its NEON `chronos_replay` in the `release` profile) against v3.1.1 (sha256 `c62846a7...` and its NEON build), all four interleaved: 5 rounds x 3 measured passes per side after 2 warm-up passes. Each number is the median kernel time per pass: the sum of every timed kernel call in one replay of the workload's traces. Frames are `chronos_pre_tick` plus `chronos_tick` calls per pass. The run took 33 min, load average 3.3 to 5.8 (the system processes above).

| Workload | Frames | wasm v3.1.0 ms/pass | wasm v3.1.1 ms/pass | wasm v3.1.1 / v3.1.0 (round range) | NEON v3.1.0 ms/pass | NEON v3.1.1 ms/pass | NEON v3.1.1 / v3.1.0 (round range) | NEON / wasm, v3.1.1 (round range) | spread wasm 3.1.0 / 3.1.1 / NEON 3.1.0 / 3.1.1 |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | :--- | :--- | :--- |
| `hero_sd` | 232 | 23.06 | 22.90 | **0.993** (0.985-1.023) | 19.93 | 20.42 | **1.025** (0.991-1.032) | 0.892 (0.880-0.907) | 6.5% / 6.8% / 9.2% / 7.1% |
| `hero_hd` | 232 | 66.30 | 66.51 | **1.003** (0.999-1.016) | 60.89 | 60.84 | **0.999** (0.994-1.005) | 0.915 (0.909-0.934) | 4.4% / 5.5% / 6.8% / 5.9% |
| `hero_suite` | 8,430 | 1,643.5 | 1,641.5 | **0.999** (0.998-1.003) | 1,486.3 | 1,487.8 | **1.001** (0.995-1.002) | 0.906 (0.901-0.909) | 5.4% / 5.8% / 5.3% / 4.9% |
| `mp4_occlusion` | 255 | 39.00 | 39.01 | **1.000** (0.995-1.007) | 34.49 | 34.71 | **1.006** (0.982-1.058) | 0.890 (0.873-0.919) | 4.7% / 3.5% / 2.4% / 8.7% |
| `swarm_1200` | 138 | 161.1 | 161.9 | **1.005** (0.994-1.006) | 137.5 | 137.5 | **1.000** (0.997-1.028) | 0.849 (0.851-0.882) | 2.7% / 3.5% / 3.9% / 6.2% |
| `bos` | 772 | 4,099.7 | 3,353.9 | **0.818** (0.814-0.833) | 3,417.9 | 2,799.4 | **0.819** (0.811-0.821) | 0.835 (0.832-0.839) | 5.1% / 3.8% / 5.2% / 4.1% |
| `hodge` | 44 | 76.99 | 76.86 | **0.998** (0.981-1.010) | 67.00 | 70.06 | **1.046** (0.997-1.084) | 0.912 (0.867-0.930) | 3.9% / 3.5% / 6.1% / 8.5% |
| `rs_vibrometry` | 1,322 | 1,091.8 | 792.4 | **0.726** (0.715-0.732) | 806.2 | 714.9 | **0.887** (0.885-0.888) | 0.902 (0.895-0.918) | 3.5% / 5.8% / 3.2% / 3.3% |
| `conservation` | 460 | 45.20 | 45.37 | **1.004** (1.002-1.017) | 40.75 | 41.02 | **1.007** (1.002-1.043) | 0.904 (0.897-0.912) | 3.1% / 3.9% / 1.4% / 4.3% |
| `covariance` | 518 | 243.7 | 243.7 | **1.000** (0.994-1.008) | 209.0 | 208.1 | **0.995** (0.994-1.004) | 0.854 (0.849-0.878) | 3.5% / 3.4% / 5.9% / 6.2% |
| `ingest_roi` | 1,336 | 13,238.0 | 9,900.3 | **0.748** (0.745-0.752) | 10,330.8 | 8,242.0 | **0.798** (0.741-0.798) | 0.833 (0.802-0.836) | 4.4% / 4.8% / 8.0% / 1.1% |
| sum | | 20,728.3 | 16,344.3 | 0.789 (geomean 0.929) | 16,610.7 | 13,816.8 | 0.832 (geomean 0.958) | 0.845 (geomean 0.880) | |

- **wasm in V8, the primary number.** The three workloads that run changed code: `bos` 0.818, `ingest_roi` 0.748 and `rs_vibrometry` 0.726 of their v3.1.0 time, in every round. The whole canonical set needs 16.3 s where v3.1.0 needed 20.7 s (0.789).
- **Native NEON.** `bos` 0.819, `ingest_roi` 0.798, `rs_vibrometry` 0.887; the whole set 13.8 s instead of 16.6 s (0.832).
- **Per export.** `bos` `chronos_tick` 5.57 to 4.30 ms median in V8 (0.772); `ingest_roi` `chronos_post_tick` 11.1 to 8.24 ms (0.741); `rs_vibrometry` `chronos_post_tick` 413 to 235 µs (0.568).
- **The other eight workloads run no changed code**, and measure 0.993 to 1.005 in V8. On NEON this run shows `hodge` 1.046 (0.997-1.084), `hero_sd` 1.025 (0.991-1.032) and `conservation` 1.007 (1.002-1.043), with spreads up to 9%. A re-check of the four small ones, v3.1.0 and v3.1.1 interleaved over 9 rounds x 5 passes, gives (wasm / NEON) `conservation` 1.006 / 0.998, `hero_sd` 1.001 / 1.005, `hodge` 0.995 / 1.012 and `mp4_occlusion` 0.991 / 0.999, every round range straddling 1. A bisect over all eight intermediate NEON builds (7 rounds x 5 passes on `mp4_occlusion`, `hero_sd` and `hodge`) showed no step either. No workload is measurably slower.
- **Native NEON is still faster than the wasm in V8 on every workload, in every round:** NEON / wasm 0.835 to 0.915 (geomean 0.880). The gap narrowed most on `rs_vibrometry` (0.902, was 0.742 in v3.0.0), because two of the changes give the wasm what LLVM's aarch64 auto-vectoriser already did natively.

Per export, per call (µs; pooled over all 15 measured passes of each side):

| Workload | Export | Calls/pass | wasm v3.1.0 µs median | wasm v3.1.1 µs median / p95 / mean | NEON v3.1.1 µs median / p95 / mean | wasm 3.1.1 / 3.1.0 median | NEON / wasm median (v3.1.1) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hero_sd` | `pre_tick` | 232 | 72.23 | 72.27 / 88.19 / 72.88 | 66.57 / 81.20 / 67.42 | 1.001 | 0.921 |
| `hero_sd` | `post_tick` | 232 | 21.89 | 21.94 / 44.44 / 23.82 | 17.40 / 34.86 / 19.09 | 1.002 | 0.793 |
| `hero_sd` | `solve_fiedler_lanczos` | 232 | 0.235 | 0.234 / 1.23 / 0.404 | 0.148 / 0.777 / 0.212 | 0.996 | 0.632 |
| `hero_hd` | `pre_tick` | 232 | 241.6 | 242.1 / 302.9 / 245.0 | 224.6 / 294.2 / 231.9 | 1.002 | 0.928 |
| `hero_hd` | `post_tick` | 232 | 36.59 | 36.77 / 65.24 / 39.34 | 28.13 / 52.41 / 31.02 | 1.005 | 0.765 |
| `hero_hd` | `solve_fiedler_lanczos` | 232 | 0.318 | 0.315 / 1.90 / 0.664 | 0.153 / 1.45 / 0.414 | 0.989 | 0.486 |
| `hero_suite` | `pre_tick` | 8,430 | 81.24 | 81.32 / 327.1 / 156.7 | 74.94 / 308.5 / 144.5 | 1.001 | 0.922 |
| `hero_suite` | `post_tick` | 8,430 | 32.35 | 32.40 / 83.77 / 37.83 | 27.07 / 70.57 / 31.56 | 1.001 | 0.836 |
| `mp4_occlusion` | `pre_tick` | 255 | 80.48 | 80.32 / 107.1 / 83.73 | 73.86 / 98.96 / 77.63 | 0.998 | 0.920 |
| `mp4_occlusion` | `post_tick` | 255 | 59.15 | 59.73 / 117.5 / 64.99 | 49.99 / 99.06 / 55.26 | 1.010 | 0.837 |
| `mp4_occlusion` | `solve_fiedler_lanczos` | 255 | 3.15 | 3.15 / 6.35 / 3.47 | 2.82 / 5.65 / 3.10 | 0.999 | 0.895 |
| `swarm_1200` | `pre_tick` | 138 | 253.4 | 253.5 / 361.6 / 272.4 | 230.5 / 334.0 / 249.2 | 1.000 | 0.909 |
| `swarm_1200` | `post_tick` | 138 | 922.8 | 924.7 / 1,151 / 895.2 | 752.0 / 1,068 / 748.1 | 1.002 | 0.813 |
| `swarm_1200` | `solve_fiedler_lanczos` | 138 | 0.486 | 0.524 / 2.82 / 1.04 | 0.236 / 2.19 / 0.651 | 1.078 | 0.450 |
| `bos` | `tick` | 772 | 5,571 | 4,298 / 5,648 / 4,338 | 3,429 / 5,029 / 3,625 | 0.772 | 0.798 |
| `bos` | `bos_solve_manufactured_test` | 22 | 307.2 | 307.8 / 2,006 / 673.8 | 226.8 / 1,698 / 569.4 | 1.002 | 0.737 |
| `hodge` | `tick` | 44 | 1,211 | 1,249 / 2,869 / 1,490 | 1,027 / 2,961 / 1,351 | 1.031 | 0.823 |
| `hodge` | `hodge_solve_injected` | 23 | 547.7 | 552.2 / 1,064 / 481.8 | 492.2 / 1,020 / 435.1 | 1.008 | 0.891 |
| `rs_vibrometry` | `pre_tick` | 1,322 | 261.6 | 259.7 / 459.2 / 284.8 | 234.8 / 431.3 / 263.2 | 0.993 | 0.904 |
| `rs_vibrometry` | `post_tick` | 1,322 | 413.0 | 234.8 / 798.3 / 308.0 | 126.1 / 724.5 / 272.4 | 0.568 | 0.537 |
| `rs_vibrometry` | `push_audio_block` | 1,254 | 1.15 | 1.11 / 2.73 / 1.51 | 0.778 / 2.28 / 1.08 | 0.964 | 0.701 |
| `conservation` | `tick` | 460 | 94.64 | 94.98 / 132.3 / 96.58 | 84.94 / 120.2 / 87.58 | 1.004 | 0.894 |
| `conservation` | `eval_occluded_pair` | 22 | 0.819 | 0.820 / 3.67 / 0.974 | 0.236 / 0.550 / 0.269 | 1.001 | 0.288 |
| `covariance` | `tick` | 518 | 456.1 | 453.9 / 642.4 / 467.8 | 384.1 / 580.0 / 402.4 | 0.995 | 0.846 |
| `ingest_roi` | `pre_tick` | 1,335 | 264.3 | 262.2 / 354.0 / 264.2 | 234.8 / 292.9 / 233.4 | 0.992 | 0.896 |
| `ingest_roi` | `post_tick` | 1,334 | 11,119 | 8,236 / 8,924 / 7,271 | 6,757 / 7,143 / 5,950 | 0.741 | 0.820 |
| `ingest_roi` | `solve_fiedler_lanczos` | 1,333 | 1.40 | 1.07 / 3.40 / 1.48 | 0.527 / 2.40 / 0.811 | 0.763 | 0.493 |

The scalar backend was not re-timed for v3.1.1 (it is a reference, not a product; in v3.0.0 it was 2.4 to 5.9 times slower than the wasm). Its parity is covered by the replays in [Native parity results](#native-parity-results-measured).

### What changed in v3.1.1 (measured one change at a time)

Six commits speed the kernel up; each was measured against the commit before it with `bench_native_vs_wasm.mjs --wasm-bin` on the workloads whose code it changes (5 interleaved rounds x 3 measured passes after 1 warm-up pass), and each passed the 29-session replay on the wasm (pointer results included), NEON and `simd-force-scalar`, and Gate S, before it was committed. Ratios are new / old; the round range is the minimum and maximum of the per-round ratios.

| Change | What it does | Workload | wasm in V8 | NEON |
| :--- | :--- | :--- | :--- | :--- |
| BOS patch sampling | `build_ref_patch_8x8` / `eval_cur_patch_8x8` read each window row through a fixed-size array view (one bounds check per row instead of four per `f32x4`; on aarch64 the four loads then merge into one vector load, on wasm32 they did not until the vector-load change below); the reference patch's samples, differences and saturation mask four pixels at a time | `bos` | 0.971 (0.966-0.979) | 0.970 (0.968-0.974) |
| | | `ingest_roi` | 0.995 (0.992-1.000) | 0.986 (0.945-0.987) |
| BOS reference-patch reuse | the reductions of each node's reference patch (`mu_t`, variance, gradient means and projections, `H`) are kept per node in `BosRefCache`, tagged with a generation that every change of `ref_luma` advances; a hit rebuilds the patch without the three sequential `f64` reductions | `bos` | 0.910 (0.906-0.921) | 0.857 (0.852-0.865) |
| | | `ingest_roi` | 0.934 (0.930-0.939) | 0.900 (0.898-0.901) |
| LOO neighbour keys | `solve_bench_loo_residuals` keeps its 24 nearest neighbours as one `u64` key each (`d2` bits, CSR index) instead of shifting six parallel arrays per insertion | `ingest_roi` | 0.838 (0.834-0.841) | 0.901 (0.899-0.906) |
| | | `covariance` | 1.005 (0.996-1.007) | 1.003 (0.995-1.006) |
| Vibrometry warm-up | the reference accumulation eight pixels per step in the SIMD layer (LLVM vectorised it for aarch64 only) | `rs_vibrometry` | 0.884 (0.884-0.886) | 0.993 (0.993-0.999) |
| Band displacement rows | per-pixel terms four columns at a time, each sum taking its lanes in column order | `rs_vibrometry` | 0.844 (0.842-0.848) | 0.899 (0.898-0.900) |
| Vector loads on wasm32 | `simd_ops::load_f32x4` / `store_f32x4`: one `v128_load` / `v128_store` for four consecutive `f32` (LLVM's wasm32 backend built `f32x4(a[i], ..)` with one `f32.load` + `replace_lane` per lane) | `bos` | 0.923 (0.923-0.929) | 0.984 (0.986-0.989) |
| | | `ingest_roi` | 0.964 (0.941-0.971) | 0.992 (0.984-0.999) |
| | | `rs_vibrometry` | 0.977 (0.970-1.021) | 0.993 (0.992-1.015) |
| Frame-wrap fix | `u64` comparisons for `last_discont_frame + 5` and `birth_frame + 2` | all | not a speed change | |

Why each is exact: the arithmetic of every output value is the same sequence of IEEE operations as before. Sums keep their order (each lane of a vector is added on its own, in index order), masked-out terms are never added, reused values are the values that were computed, and replaced selection or saturation logic is equal for every input (the commit messages give the argument for each). Two paths that no recorded session reaches have unit tests against the replaced scalar code: the saturating warm-up accumulation (`vibrometry::warmup_accumulate_tests`) and the band row sums (`vibrometry::band_row_sums_tests`).

Tried and not landed:

- **Branch-free masked sums in the Hodge PCG** (`project_zero_mean`, the `pap` dot products: inactive vertices add `+0.0`, exact because the sums start at `+0.0`; the running component sum in a register). 15 rounds x 5 passes on `hodge`: NEON 0.965 (0.889-0.979), wasm 0.993 (0.932-1.055). Inside the noise in V8, so dropped.
- **Fusing `chronos_pre_tick`'s two frame passes.** Not attempted after analysis. The grid pass needs the pan estimate (from the pooled pyramid) and the ECA affine fit (a reduction over the whole `luma` plane) before its first pixel, so one pass would need speculative per-pixel work. And its cost is not the second read of the frame: per 16 pixels it issues about 157 vector ALU operations and 31 loads and stores, about 40 cycles at four vector operations per cycle, which predicts about 128 µs per HD frame against 137 µs measured (inferred from an operation count and the measured time, not from CPU counters).

### Proposals that change output (measured, not landed)

Most of what is left in `bos`, `ingest_roi` and `rs_vibrometry` is sequential floating-point sums: every term waits for the previous add (3 to 4 cycles), and the order of the adds cannot change without changing the last bits of the result. A prototype on a scratch copy, not on any branch, reassociated two of them:

- `eval_cur_patch_8x8`, fast path: `sum_ti`, `bx` and `by` as two partial sums each (even and odd pixels, in `f64x2` lanes), added together at the end.
- `band_row_sums`: the five row sums as four-lane partial sums (`f32x4`), added together after the row.

It was timed against v3.1.1 before the vector-load change (3 interleaved rounds x 2 passes after 1 warm-up pass; return values are not affected, so the call sequence is the same):

| Workload | wasm in V8 | NEON |
| :--- | :--- | :--- |
| `bos` | 0.912 (rounds 0.912-0.916) | 0.920 (0.918-0.922) |
| `ingest_roi` instance 1 | 0.965 (0.963-0.965) | 0.974 (0.949-0.977) |
| `rs_vibrometry` | 0.951 (0.948-0.954) | 0.916 (0.917-0.923) |

Results change, as expected: the first differing region digest is `bos_telemetry` at `bos` call `#219` and `vibro_planes` at `rs_vibrometry` call `#103`. Native/wasm parity would be unaffected (every backend reproduces the `f64x2`/`f32x4` lane ops exactly), but every session that runs BOS or vibrometry would change in its last bits and the canonical set would have to be re-recorded. Whether that is worth about 9% of `bos`, 3 to 4% of `ingest_roi` and 5 to 8% of `rs_vibrometry` is the owner's call; more partial sums (four instead of two in BOS, the dense-LK residual, the reference-patch reductions on a cache miss) would gain more and were not measured.

### Sanity checks

Measured on v3.0.0 (the harness is unchanged since; the `--no-liftoff` check and the wall-clock reconciliation were not repeated for v3.1.1).

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

### Native build configuration (measured on v3.0.0)

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

Profiles of v3.1.1 on the five main workloads (self time: time in the function body, after inlining). `hero_hd` and `swarm_1200` run no code that v3.1.1 changed, so their profiles are of v3.1.0 (taken for this release, with the same harness); `bos`, `rs_vibrometry` and `ingest_roi` instance 1 are profiles of the v3.1.1 build:

**hero_hd** (native 63.6 ms/pass, wasm 71.4 ms/pass; 17261 native / 48063 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `stat_fused_hd40_grid_loop<true, true>` | 50.2% | 31.9 | 47.9% | 34.2 | 0.93 | v128_load 20.1%, v128_store 17.3%, v128_bitselect 17.0% |
| `rgba_to_planar_and_pool160_simd128` | 21.0% | 13.4 | 21.7% | 15.5 | 0.86 | i32x4_dot_i16x8 41.2%, v128_load 32.3%, v128_store 16.8% |
| `eca_fit_affine_params_ex_simd128` | 4.9% | 3.12 | 4.4% | 3.14 | 0.99 | scalar 21.4%, f32x4_add 14.4%, v128_any_true 12.4% |
| `compute_row_luma_means_simd128` | 3.4% | 2.14 | 3.1% | 2.20 | 0.97 | v128_load 79.5%, scalar 10.7%, f32x4_add 9.8% |
| `stat_fused_hd40_grid_loop<false, true>` | 3.4% | 2.13 | 3.6% | 2.60 | 0.82 | v128_store 29.4%, v128_load 26.9%, scalar 11.4% |
| `render_eulerian_pip_subpx_impl<false, true>` | 2.1% | 1.30 | 2.5% | 1.75 | 0.74 | i16x8_q15mulr_sat 21.8%, shuffle 13.8%, i32x4_dot_i16x8 11.3% |
| `sad_patch_zm_half_st1_ring_hd_simd128` | 1.6% | 1.00 | 1.5% | 1.04 | 0.96 | scalar 38.7%, v128_load 31.0%, f32x4_add 14.4% |

**swarm_1200** (native 143.2 ms/pass, wasm 170.6 ms/pass; 18936 native / 70149 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `extract_phase1a_raw` | 15.8% | 22.6 | 15.8% | 26.9 | 0.84 | scalar 43.0%, v128_any_true 35.5%, f32x4_mul 9.9% |
| `stat_fused_hd40_grid_loop<true, true>` | 11.9% | 17.1 | 10.8% | 18.4 | 0.93 | v128_load 20.2%, v128_store 18.4%, v128_bitselect 17.9% |
| `build_sparse_knn_graph` | 6.0% | 8.60 | 6.0% | 10.2 | 0.84 | scalar 90.2%, f32x4_splat 3.2%, f32x4_extract_lane 2.9% |
| `rgba_to_planar_and_pool160_simd128` | 5.5% | 7.91 | 5.5% | 9.46 | 0.84 | i32x4_dot_i16x8 43.2%, v128_load 30.1%, v128_store 15.4% |
| `eval_lk_patch_wasm<true, true>` | 4.2% | 5.97 | 3.3% | 5.57 | 1.07 | f32x4_pmin 48.5%, scalar 24.3%, f32x4_mul 16.8% |
| `refine_ic_lk_8x8_2iter_wasm_impl<true, false>` | 4.0% | 5.74 | 3.7% | 6.39 | 0.90 | scalar 78.5%, f32x4_mul 10.0%, f32x4_add 3.3% |
| `gather_csr_stage1_track<true>` | 3.8% | 5.47 | 3.8% | 6.46 | 0.85 | scalar 99.9%, f32x4_extract_lane 0.1% |

**bos** (native 2779.7 ms/pass, wasm 3340.1 ms/pass; 20218 native / 455473 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `eval_cur_patch_8x8` | 35.4% | 985.1 | 35.3% | 1180.1 | 0.83 | scalar 83.7%, f32x4_max 6.8%, v128_load 4.7% |
| `build_ref_patch_node` | 34.6% | 962.8 | 37.9% | 1264.7 | 0.76 | scalar 79.7%, v128_store 7.6%, v128_load 7.0% |
| `extract_dense_lk` | 9.6% | 267.0 | 6.6% | 219.2 | 1.22 | scalar 100.0% |
| `stat_fused_hd40_grid_loop<true, true>` | 3.5% | 98.2 | 3.2% | 106.6 | 0.92 | v128_load 20.6%, v128_store 19.3%, v128_bitselect 16.7% |
| `apply_laplacian_p_to_lp` | 2.2% | 62.1 | 1.1% | 38.4 | 1.62 | scalar 100.0% |
| `rgba_to_planar_and_pool160_simd128` | 1.8% | 50.7 | 1.8% | 59.6 | 0.85 | i32x4_dot_i16x8 43.1%, v128_load 29.8%, v128_store 17.1% |
| `compute_curl_and_assemble_neumann` | 1.7% | 47.4 | 2.1% | 70.8 | 0.67 | scalar 100.0% |

**rs_vibrometry** (native 712.0 ms/pass, wasm 792.6 ms/pass; 19375 native / 43189 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `extract_signed_band_displacements` | 22.9% | 163.3 | 23.5% | 186.6 | 0.87 | scalar 74.3%, f32x4_extract_lane 13.3%, f32x4_mul 6.5% |
| `eval_segment_optical_quadrature` | 14.8% | 105.2 | 14.3% | 113.3 | 0.93 | scalar 100.0% |
| `stat_fused_hd40_grid_loop<true, true>` | 13.1% | 93.5 | 12.7% | 100.4 | 0.93 | v128_load 19.2%, v128_store 17.8%, v128_bitselect 16.5% |
| `rgba_to_planar_and_pool160_simd128` | 10.5% | 75.1 | 11.0% | 86.9 | 0.86 | i32x4_dot_i16x8 43.3%, v128_load 31.3%, v128_store 17.5% |
| `stat_fused_hd40_grid_loop<false, true>` | 5.1% | 36.0 | 5.3% | 41.9 | 0.86 | v128_store 25.9%, v128_load 24.1%, scalar 11.7% |
| `accumulate_ref_q12` | 2.3% | 16.6 | 2.6% | 20.3 | 0.82 | scalar 93.8%, f32x4_min 3.1%, v128_or 1.6% |
| `subpx_pan_streak_row_exact<false, false, true, true, true, true, 2>` | 2.2% | 15.6 | 2.5% | 19.9 | 0.78 | shuffle 39.9%, u16x8_extmul_high_u8x16 9.2%, scalar 8.3% |

**ingest_roi** (native 7074.1 ms/pass, wasm 8431.6 ms/pass; 19941 native / 172319 V8 samples)

| function (self time) | native % | native ms | wasm % | wasm ms | native / wasm | what dominates natively |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `solve_bench_loo_residuals` | 27.6% | 1952.9 | 25.5% | 2150.0 | 0.91 | scalar 100.0% |
| `eval_cur_patch_8x8` | 18.3% | 1295.2 | 18.4% | 1554.8 | 0.83 | scalar 89.5%, f32x4_max 4.1%, v128_load 3.0% |
| `build_ref_patch_node` | 13.3% | 942.9 | 16.4% | 1381.0 | 0.68 | scalar 78.3%, v128_load 8.4%, v128_store 8.1% |
| `extract_dense_lk` | 4.0% | 280.6 | 2.6% | 221.3 | 1.27 | scalar 100.0% |
| `apply_laplacian_p_to_lp` | 3.6% | 251.2 | 2.1% | 174.0 | 1.44 | scalar 100.0% |
| `solve_poisson_and_pin_gauge` | 3.1% | 220.3 | 3.4% | 287.4 | 0.77 | scalar 100.0% |
| `dct2_2d_solve_z` | 3.1% | 218.9 | 3.9% | 327.7 | 0.67 | scalar 87.8%, f64x2_add 6.2%, f64x2_mul 4.4% |

How to read the per-function ratios:

- **Inlining differs between the builds.** LLVM inlines differently for aarch64 and for wasm32, and V8 inlines small wasm functions too (`--wasm-inlining`, on by default). A function's *self* time therefore only compares cleanly when it is a leaf on both sides.
- **Inlining artifacts.** In `bos`, `accumulate_warmup_node_noise` (self 1.47) and `extract_dense_lk` (self 1.20) look slower natively, but their *inclusive* shares are equal (40.0% vs 40.5%, 30.8% vs 30.5%). In milliseconds both are faster natively; only the split between the function and its callees differs.
- **Tail call.** Natively `chronos_tick` tail-calls `chronos_post_tick`, so its frame disappears from native stacks.

What the profiles say:

- **Sparse tracking (`hero_*`, `mp4_occlusion`) is unchanged.** About 75% of HD kernel time is two streaming passes over the frame in `chronos_pre_tick`: `stat_fused_hd40_grid_loop` (50%) and `rgba_to_planar_and_pool160_simd128` (21%).
  - The grid loop runs at 0.93 of V8's time. Its samples sit on vector loads and stores (37%) and `v128_bitselect` (17%), but by operation count it is close to the vector ALUs' limit (see "Tried and not landed" above), so a second pass over the frame is not what it waits on.
- **Swarm is unchanged.** `extract_phase1a_raw` (16%: the per-pixel candidate scan, 36% of its samples on `v128_any_true` and the loads it waits on), the grid loop (12%), the k-NN graph build (6%, scalar) and the LK patch evaluation. `eval_lk_patch_wasm` is the one hot SIMD function slower natively (1.07). Its samples sit on the `fcmgt` of `f32x4_pmin`, waiting on the loads and the subtract/abs chain, and V8 lowers `pmin` to the same two instructions.
- **BOS and ingest/ROI: what is left is mostly sequential `f64` sums.** `eval_cur_patch_8x8` and `build_ref_patch_node` are 73% (V8) / 70% (native) of `bos` and 35% / 32% of `ingest_roi` instance 1. `eval_cur_patch_8x8` waits on its three sequential `f64` sums (64 adds each, 3 to 4 cycles per add). `build_ref_patch_node` is now mostly the cache-hit path: sampling the window and rebuilding the per-pixel arrays (element-wise `f64`) without the reductions. `solve_bench_loo_residuals` is still 26% of `ingest_roi` instance 1 (2.15 s per pass in V8, down from 3.87 s): the neighbour search over the CSR grid, and its insertion sort, now one `u64` per step.
- **Vibrometry.** `extract_signed_band_displacements` fell from 32% to 24% of `rs_vibrometry` in V8; its five row sums stay sequential. The warm-up accumulation, 14% of the wasm's time in v3.1.0, is 2.6%.
- **Native wins most on scalar code** (0.67 to 0.91 here), from LLVM's aarch64 code generation against V8's, and on loops LLVM auto-vectorises for aarch64 but not for wasm32 (v3.1.1 hand-vectorised the two that showed up: the vibrometry warm-up and the band rows). `apply_laplacian_p_to_lp` and `extract_dense_lk` remain slower natively in self time (1.22 to 1.62), as before.

Which SIMD ops matter, by share of all kernel samples (v3.0.0 profiles, before the dot-product change; aggregated over the 14 hottest functions of each workload):

- **`i32x4_dot_i16x8`** 12.9% of `hero_hd`, 5.9% of `rs_vibrometry`, 3.4% of the swarm. Now `smull` + `smull2` + `addp` as written.
- **Shuffles** up to 8.1%: mostly the `uzp1`/`uzp2` pairs after the dot products, since folded into `addp`.
- **`v128_any_true`** 5.5% of the swarm.
- **Narrows** up to 2.1%.
- **`i32x4_bitmask`** 0.8% at most.
- **`v128_bitselect`** 7.6% of `hero_hd`.

Except for the dot product, each of these lowers to the same instructions V8 emits for the wasm op, and no cheaper NEON sequence gives the exact wasm result for every input.

**Non-inlined helpers with `v128` arguments** (measured on v3.0.0). On Rust 1.99 a `v128` passed by value to a non-inlined Rust function goes through memory. 112 kernel functions take or return `v128`. 99 are `#[inline(always)]`. Of the 13 `#[inline(never)]` ones, 7 take `v128` by value (none returns one): `subpx_row_blk_chunk` (7 values), `pan_hd40_band`, `subpx_hd40_band_fused`, `pan_subpx_fused_hd40_no_scratch` (2 each), `pan_fused_grid_le40_row`, `pan_subpx_fused_grid_fallback` (15 each) and `eval_cand_4px` (6). They are called per band, row or edge candidate, never per pixel. Their whole bodies account for 0% (`hero_hd`, swarm, `ingest_roi`), 0.35% (`bos`) and 2.1% (`rs_vibrometry`) of native kernel time. They load 0 to 9 `q` registers from argument pointers per call. Removing the `#[inline(never)]` would buy well under 0.1%.

### NEON backend change: `i32x4_dot_i16x8` (v3.0.0)

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

### Optimisation opportunities, ranked (after v3.1.1)

Shares are of v3.1.1 kernel time from the profiles above. "Exact" means the change can keep every result bit-identical to the wasm. Excluded by design and not ranked: threads (they would change the memory model and the zero-import design) and wider vectors, AVX or FMA (every backend must reproduce SIMD128 exactly). Done in v3.1.1: reference-patch reuse (was #3), the BOS patch sampling (part of #2: the builders turned out to be bound by their sequential `f64` sums, not by sampling), and three changes that were not on the list (LOO neighbour keys, vibrometry warm-up and band rows).

| # | What | Where | Measured share (v3.1.1) | Expected gain | Exact? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Sequential `f64` sums in BOS and the LOO solver as partial sums** | `eval_cur_patch_8x8` (`sum_ti`, `bx`, `by`), dense-LK residual, reductions on a reference-patch miss | see the profiles above | measured on a prototype, see [Proposals that change output](#proposals-that-change-output-measured-not-landed) | **No**: a different summation order changes the last bits |
| 2 | Pair two BOS nodes in `f64x2` lanes (each lane exactly one node's scalar sequence) | `extract_dense_lk` with `eval_cur_patch_8x8` | as #1 | Limited: each node's chain loop already keeps the FP pipes ~90% busy (11 FP ops per pixel against a 3-cycle add chain), so pairing mostly halves latency the pipes cannot fill; estimate 10% of `eval_cur_patch_8x8`, not measured | Yes, but needs the LK iterations of two nodes in lockstep |
| 3 | BOS reference-patch hit path in `f64x2` (`ref_patch_restore`, two pixels per op) | `bos.rs` | the hit path is most of what is left of `build_ref_patch_node` | Needs `f64x2_promote_low_f32x4` and `f32x4_demote_f64x2_zero` in the SIMD layer (every backend, conformance table); estimate 3-5% of `bos` and `ingest_roi`, not measured | Yes |
| 4 | Invalidate only the nodes a maintenance update touches | `update_mode2_reference` | `bos` only: ~100 of 635 dense-LK frames follow a maintenance update and miss the whole cache | Estimate ~2% of `bos`, not measured | Yes |
| 5 | Fewer `v128_any_true` exits in the candidate scan: OR 16-32 pixels of hit masks before one test | `soa_tracks.rs` `extract_phase1a_raw` | swarm 16%, covariance 11%, hodge 9% (wasm) | Estimate under 5% of those, not measured | Yes (control flow only) |
| 6 | Row-luma means inside the pooling pass; 8-op instead of 11-op tolerance pack in the grid loop | `rgba_to_planar_and_pool160_simd128`, `stat_fused_hd40_row` | `compute_row_luma_means_simd128` 3.4% of `hero_hd`; the pack ~2% of the grid loop | Estimate 2-3% and ~1% of `hero_hd` | Yes, keeping each row's four accumulators and their order. The full pass fusion: see "Tried and not landed" above |
| 7 | Branch-free masked sums in the Hodge PCG | `hodge.rs` | `project_zero_mean` 23% of `hodge` (wasm) | **Measured**: NEON 0.965, wasm 0.993 (inside the noise); not landed | Yes |
| 8 | The scan-along-`ex` branch of the band rows | `vibrometry.rs` | not reached by the canonical set | Same transformation as the landed one; needs a recorded session that uses it first | Yes |
| 9 | Build flags: fat LTO, `codegen-units = 1`, `panic = "abort"`, `target-cpu=native` | `native/Cargo.toml` | | **None** (measured on v3.0.0, table above) | Yes |
