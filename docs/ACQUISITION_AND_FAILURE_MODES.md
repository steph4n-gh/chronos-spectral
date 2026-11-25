# Acquisition requirements, physical assumptions, and failure modes

CHRONOS-SPECTRAL is a deterministic `#![no_std]` WebAssembly and native visual physics kernel operating without pretrained neural weights or labeled training datasets. Physical accuracy therefore depends on explicit acquisition profiles, explicit observability limits, fail-closed refusal bitmasks, and reproducible positive and failure fixtures under [web/samples/](../web/samples/) verified by `scripts/gate_i*.mjs`.

---

## 1. Per-mode acquisition profiles and physical assumptions

Every metrology mode pairs a required imaging geometry with bounded noise calibration and explicit refusal flags ([web/ui/core/reasons.js](../web/ui/core/reasons.js)):

| Capability | Mandatory acquisition profile | Physical assumptions | Primary breakdown modes |
| :--- | :--- | :--- | :--- |
| **Native `1:1` ROI & Fovea tracking** | `1:1` sensor crop window (`crop_x0`, `crop_y0`) with `D4` orientation, scan direction, monotonic `t_src_us`, capture age `age_ms`, and gap flags; `100%` frame-ring ingestion prior to any staggered solve | Known integer pixel scale and deterministic ingest cadence (`t - K` ring buffer) | Upstream H.264/HEVC `4x4`/`8x8` block quantization, deblocking smear, skipped B/P residuals, Bayer aliasing, ISP auto-gain pumps |
| **Ballistic residual & `2x2` covariance** | Stationary tripod or smoothly panning camera; explicit weak-perspective or planar assumptions when interpreting 2D residuals | Leave-one-out neighborhood flow approximates background/group drift; `2x2` inverse-Hessian plus empirical noise reflects local uncertainty | Pinhole perspective radial acceleration ($x = fX/Z \implies \ddot{x} \neq 0$ when $\ddot{\mathbf{X}}_{3\text{D}} \equiv \mathbf{0}$), parallax, camera shake, unmodeled motion blur |
| **Controlled-tripod `256x128` BOS** | Rigid stationary tripod, high-spatial-frequency non-periodic textured backdrop, `16`-frame unperturbed reference before plume onset, zero smoke/tint saturation | Paraxial weak-refraction eikonal limit ($\boldsymbol{\varepsilon} = \nabla\Phi$), persistent stationary texture, designated unperturbed reference gauge ROI | Tripod shock or handheld sway, rotational roll, scale breathing, motion blur, blank walls, heavy compression (`CRF >= 30`), curl-free optical zoom |
| **`64x36` Mesh Hodge `2`-complex** | Conditioned continuum velocity field (`CELL_VALID_GRAD`) from dense `MicroTrackSoA` tracks or injected flow | Consistent positive edge metric `W_1`; active grid domain covers the physical region of interest | Unseeded textureless regions creating occupancy holes indistinguishable from physical obstacles without corroborating evidence, solver non-convergence (`UNCONV`) |
| **`64x36` Rolling-shutter vibrometry** | Calibrated `dt_row_us`, exposure `T_e`, monotonic PTS, `1:1` native scan alignment (`scan_dir`), stationary camera; synchronized `48 kHz` audio for Welch coupling | Sub-pixel optical gradient sensitivity below optical Nyquist (`0.48 / dt_row_us`), stationary lighting, audio-video clock skew within `max_skew_us` | Shutter integration nulls (`|H(f)| < 0.05`), transposed scan axis, rescaled video, frame cadence drops, camera motion, unsynchronized clocks |
| **Zero-ML Depth, FoE/TTC & Stereo BOS** | Translating camera with `>= 2` depth strata (Mono) or calibrated rectified stereo pair (`f_px`, `baseline_m`, explicit Left capture); fixed focal length for FoE/TTC | Rigid static scene majority (epipolar outliers rejected), known scale (`speed_m_s` or `baseline_m`) for metric `Z` | Pure tripod yaw (`REFUSED_PURE_ROTATION`), mixed rotation+translation, insufficient baseline, periodic stereo texture, vertical epipolar error `> 0.25 px`, focal zoom |
| **Pocket Wind Tunnel (`64x36`)** | Quasi-2D incompressible planar PIV flow around an enclosed obstacle (`beta_1 >= 1`) with uniform freestream band (Modes `1`/`2`), or textured planar fluid flow (Mode `3` dense audit) | Divergence-free planar flow (`div_frac < 0.25`), conservative Navier-Stokes forcing (`eta_NS`), known fluid density `rho` for dimensional forces | Strong 3D out-of-plane flow (`HIGH_DIV`), non-conservative forcing (`HIGH_NS_RESIDUAL`), open wake without closed void (`NO_OBSTACLE_VOID`), BOS scratch conflict (`MUTEX_CONFLICT`) |
| **kHz Optical Elastography & Lock-In Acoustic BOS** | Stationary tripod, textured vibrating surface or backdrop, calibrated rolling-shutter readout (`dt_row_us`, `T_e`) and synchronized audio or locked tone list (`f0..f3`) | Traveling or multi-tone wavefield resolved within `64x36` spatial Nyquist (surface) or `256x128` paraxial acoustic refraction (Mode `4`) | Spatial aliasing (`SPATIAL_ALIASING`), wavelength exceeding aperture (`WAVELENGTH_TOO_LONG`), pure standing wave with zero net Poynting flux (`STANDING_WAVE_DOMINANT`), exposure nulls, BOS scratch conflict |

### 1.1 Native `1:1` ROI vs downscaling and codec bounds

Downscaling a `1920x1080` sensor stream to `640x360` discards `3x` linear (`9x` pixel area) spatial detail before derivatives or templates are evaluated. Retaining a `1:1` native `640x360` ROI window preserves full per-axis sampling inside the `12`-frame Wasm ring buffer (`11,059,200 B`) while propagating `(crop_x0, crop_y0)`, orientation, and scan direction downstream. For higher-resolution sources, `FoveaState` adds a `160x160` `1:1` native tile and `INGEST_FLAG_ROI_SLEW` lets virtual PTZ track smoothly without resetting downstream geometry. Native builds with `--features hires-720p` also run `1280x720` directly.

Native ROI cropping is not optically or computationally zero-cost:
- **Upstream staging cost:** Host decode, `VideoFrame` staging, and ROI extraction still scale with upstream sensor dimensions and pixel format.
- **Codec quantization floor:** Subpixel optical flow (`IC-LK` and `BOS`) operates on `8`-bit `yuv420p` H.264/HEVC frames. On `web/samples/bos_thermal_plume.mp4` (`CRF 21`, `640x360` High profile), quiet unperturbed margins yield a residual displacement floor near `~0.05 px` RMS due to `4x4`/`8x8` DCT quantization and `4:2:0` chroma subsampling. Thermal refractive signals below `~0.15 px` cannot be cleanly separated from compressed-video noise.
- **Ingestion integrity:** Staggered downstream PDE solves may throttle compute per tick, but `100%` of incoming frames and source timestamps must enter the ring buffer before any solver skip so velocity finite differences and temporal phase never alias across silent drops.

### 1.2 Controlled tripod + textured-background BOS boundaries

Background-Oriented Schlieren (`BOS`) measures line-of-sight integrated refractive index gradients by cross-correlating apparent background displacement $\boldsymbol{\varepsilon}(x, y)$ on a `256x128` lattice against a persistent stationary textured backdrop:

1. **Inhomogeneous Neumann Poisson formulation:** Recovering scalar refractive potential $\Phi$ minimizes $\int_\Omega |\nabla\Phi - \boldsymbol{\varepsilon}|^2\,dA$, yielding:
   $$\Delta\Phi = \nabla \cdot \boldsymbol{\varepsilon} \quad \text{in } \Omega, \qquad \partial_n \Phi = \boldsymbol{\varepsilon} \cdot \mathbf{n} \quad \text{on } \partial\Omega$$
   Homogeneous Neumann boundaries ($\partial_n \Phi = 0$) collapse constant global shifts $\boldsymbol{\varepsilon} = (a, 0)$ to a flat constant instead of the physical linear ramp $\Phi = ax$.
2. **Curl consistency vs motion oracle:** Discrete curl $\nabla \times \boldsymbol{\varepsilon}$ checks integrability consistency of the gradient field only. Optical zoom, planar radial expansion, and longitudinal strain also satisfy $\nabla \times \boldsymbol{\varepsilon} = 0$, so low curl alone never certifies pure refractive flow.
3. **Uncalibrated physical provenance:** Outputs are reported strictly as apparent pixel displacement $\boldsymbol{\varepsilon}$ (`px`) and gauge-referenced scalar potential $\Phi$ relative to an unperturbed margin ROI. Quantitative gas density $\Delta\rho$ or temperature $\Delta T$ requires external optical path lengths ($Z_D$, $Z_B$) and Gladstone-Dale sensitivity constants.
4. **Why natural stochastic texture is mandatory:** Uniform blank walls or pure sky produce singular local Hessians ($\lambda_{\min} \to 0$), while repeating regular checkerboards risk half-period phase aliasing under shifts exceeding `1-2 px`. Multi-scale natural mineral or stone textures provide isotropic spatial power across every `256x128` BOS block.

### 1.3 Vector `2x2` covariance and perspective residual bounds

Single-scalar Hessian bounds discard directional uncertainty along edges (the aperture problem). Propagating symmetric `2x2` covariance `(c_xx, c_xy, c_yy)` preserves directional error ellipsoids while empirical calibration folds codec noise, differencing variance, interpolation bias, and shared jitter into the realized covariance:

- **Dimensional containment semantics:** A `1`-sigma `1D` marginal confidence interval captures `~68.3%` of Gaussian inliers, whereas a `1`-sigma `2D` Mahalanobis error ellipse captures $1 - \exp(-1/2) = 39.35\%$ of isotropic 2D Gaussian inliers.
- **Pinhole perspective trap and monocular scale ambiguity:** Under perspective projection $x = f X / Z$, an object moving at strictly constant 3D velocity $\mathbf{V} = (V_X, V_Y, V_Z)$ with $V_Z < 0$ produces nonzero 2D image acceleration ($\ddot{x} \neq 0$, $\mathbf{a}_{\text{obs}} \cdot \mathbf{v} > 0$) purely from $1/Z(t)$ depth scaling. Single-camera observations also carry fundamental monocular projective scale ambiguity $\lambda(t)\mathbf{X}(t)$: an object accelerating radially along the same line of sight can reproduce identical 2D pixel positions as an unaccelerated object approaching in depth $Z(t)$. Ballistic-model residual attention therefore pairs leave-one-out local flow covariance and sample-disjoint chord looming tests with explicit monocular ambiguity bounds to suppress looming and pan decoys within false-alert budgets.

---

## 2. Sample corpus and semi-synthetic validation fixtures (`web/samples/`)

[web/samples/ATTRIBUTION.md](../web/samples/ATTRIBUTION.md) documents all `41` sample clips (`27` open-licensed real-camera recordings, `6` semi-synthetic compositions over CC0 and Public Domain photography, and `8` deterministic in-repo physical simulations). The three Wave-1 semi-synthetic fixtures verified by [scripts/gate_i6_captures.mjs](../scripts/gate_i6_captures.mjs) establish analytically known refractive, perspective, and tripod-failure ground truth:

| Fixture | Duration / FPS | Size (`MiB`) | SHA-256 | Metrology role |
| :--- | :--- | :--- | :--- | :--- |
| [bos_thermal_plume.mp4](../web/samples/bos_thermal_plume.mp4) | `9.00 s` / `25 fps` (`225 f`) | `1.21 MiB` (`1,270,459 B`) | `adbe481a503a941e952d8380317445f00954926db055a77bc15ffeacfb30ada0` | Semi-synthetic tripod BOS positive baseline |
| [perspective_flyby_decoy.mp4](../web/samples/perspective_flyby_decoy.mp4) | `9.00 s` / `25 fps` (`225 f`) | `0.95 MiB` (`997,228 B`) | `83d78df42db8e58787642ffe454f26828069235e6331d03db1aeebda34494c1e` | Semi-synthetic constant-3D-velocity looming + pan decoy |
| [bos_handheld_failure.mp4](../web/samples/bos_handheld_failure.mp4) | `8.00 s` / `25 fps` (`200 f`) | `1.20 MiB` (`1,263,102 B`) | `7e672c9a7ac7631af7d83cdb35decc24f7c750ce130ac596f85b2df9a232a936` | Semi-synthetic BOS tripod-shock and sway failure |

### 2.1 Positive BOS composition: `bos_thermal_plume.mp4`

- **Construction:** Semi-synthetic composition built on a CC0 natural gravel backdrop (`3888x2592` cropped and `2x` supersampled before `640x360` area filtering). Frames `0..15` (`0.00-0.60 s`) hold an unperturbed `16`-frame static reference matching the `I5` BOS warmup window. From `0.64 s` onward, a 2D Boussinesq Navier-Stokes thermal buoyancy plume integrates paraxial Gladstone-Dale ray deflections ($\boldsymbol{\varepsilon} = -C\nabla T = \nabla\Phi$, peak refractive shift `1.38 px` inside the `1.50 px` bounded `8x8` affine LK pull-in limit, quiet margin shift `0.00 px`, `sigma = 0.14 DN`, `CRF 21`). Recommended unperturbed gauge ROI on the `256x128` BOS grid scales proportionally (`[8, 8, 36, 120]`, or `[4, 4, 18, 60]` on normalized coordinates).
- **Clear pixels vs resolved eddies:** No opaque smoke, color tint, or particle overlay is added. Raw frames resemble a static gravel wall with subtle convective shimmer, while the `256x128` BOS displacement plane (`chronos_get_bos_planes_ptr()`) separates core refractive displacement well above the compressed quiet-margin floor with low curl, `0` disturbed frames, `cam_motion_state == 0`, and `0` false Hero detections.

### 2.2 Perspective flyby & pan decoy: `perspective_flyby_decoy.mp4`

- **Construction:** Textured outdoor horizon built from CC0 speckled asphalt perspective tarmac and Public Domain stone ridges beneath multi-octave atmospheric clouds. Two multi-object formations traverse strictly constant-3D-velocity rectilinear trajectories ($\ddot{\mathbf{X}}_{3\text{D}} \equiv \mathbf{0}$):
  - **Phase 1 (`0.00-5.40 s`, frames `0..134`):** Stationary tripod (`cam_motion_state == 0`). Nine approaching bodies accelerate radially across the pinhole image plane from `8.2 px/s` to `86.2 px/s` (peaking at `73.1 px/s^2` 2D acceleration) purely from $x = f X / Z$ foreshortening.
  - **Phase 2 (`5.40-9.00 s`, frames `135..224`):** Smooth horizontal camera pan (`cam_motion_state == 1`) inducing background optical flow and parallax across a second constant-3D-velocity formation.
- **Failure prevented:** Naive `a_obs . v > 0` energy heuristics falsely promote looming pinhole approaches as maneuvering objects. Gate `I1` requires zero false maneuvering-object alarms across this decoy, while Gate `I7` uses Phase 1 to verify intrinsics-free Focus of Expansion and optical looming Time-to-Contact (`R^2 >= 0.98`).

### 2.3 Reproducible BOS failure fixture: `bos_handheld_failure.mp4`

- **Construction:** Identical gravel backdrop, sensor noise, and Gladstone-Dale thermal plume physics as `bos_thermal_plume.mp4`, divided into four diagnostic intervals: steady tripod baseline (`0.00-2.56 s`), damped mechanical tripod shock (`2.56-3.36 s`: `+6.2 px` horizontal, `-3.8 px` vertical, `0.65 deg` roll, `+0.4%` scale, `3`-tap shutter blur), physiological handheld sway (`3.36-5.96 s`: `1.75-3.1 Hz` roll and scale breathing), and a displaced rest pose (`5.96-8.00 s`: `+3.1 px, -1.7 px, 0.22 deg`).
- **Why stabilization limits fail reproducibly:** Global camera motion triggers immediately on shock entry, sub-pixel 2D translation alignment cannot repair rotational roll (injecting nonzero curl), scale breathing (injecting curl-free radial strain), or blur, and the displaced post-shock rest pose proves why stale BOS references must trigger re-acquisition after any shock event.

---

## 3. Wave-2 and v3.2 to v3.4 identifiability guards

### 3.1 Rolling-shutter vibrometry and multi-segment Welch coherence (`src/vibrometry.rs`, Gate `I2`)

1. **Calibrated scan-time geometry & hard refusals:** Rolling-shutter scan times are inverted through the committed `IngestSlotMeta` `D4` orientation transform and crop origin to recover sensor scan coordinate `q`. Mapped scan axes exceeding the row buffer (`AXIS_TRANSPOSED`), missing scan direction (`NO_SCAN_DIR`), missing or non-monotonic PTS (`NO_PTS` / `FRAME_GAP`), rescaled non-native ingest (`NOT_NATIVE_1TO1`), global camera motion (`CAMERA_MOTION`), invalid calibration (`BAD_CALIB`), and audio clock skew past `max_skew_us` (`CLOCK_SKEW`) transition vibrometry out of `READY` instead of reporting false attribution.
2. **Exposure sinc envelope, optical Nyquist guard, and nulls:** Eligible `2,048`-point Welch frequency bins (`128` bins across `32` spatial bands on a `64x36` output grid) are restricted strictly below `0.48 / dt_row_us`. Recovered amplitudes compensate shutter integration via $H(f) = \sin(\pi f T_e)/(\pi f T_e)$. When $|H(f)| < 0.05$, the kernel raises `EXPOSURE_NULL` (`0x40`), surfaces a warning banner, and suppresses `1 / H` blow-up.
3. **Welch `K >= 8` vs `K = 1` trap & Bonferroni correction:** Magnitude-squared coherence equals `1.0` identically for a single segment (`K = 1`). Coupled significance stays suppressed until at least `K >= 8` overlapping segments share contiguous optical support and audio timestamps without gaps (`AUDIO_UNDERRUN`), evaluated against the Bonferroni-corrected null threshold across active bands and unambiguous frequency bins.

### 3.2 Bounded `64x36` mesh Hodge `2`-complex (`src/hodge.rs`, Gate `I3`)

1. **Cell-vertex complex & complementary dual topology:** Active vertices require `cell_valid_mask[i] & CELL_VALID_GRAD`, forming up to `(V, E, F) = (2304, 4508, 2205)` over one positive `W_1` edge metric across exact (`grad phi`) and coexact (`curl* psi`) SSOR-PCG solves plus edge-level orthogonality checks.
2. **Enclosed internal holes vs open ports & wake hypotheses:** Flooding complementary plaquettes from the exterior border across absent edges distinguishes enclosed internal holes (`beta_1`, circulation carriers) from open outer ports. Because unseeded regions also create absent vertices, wake deficits and voids are reported strictly as occupancy-gap hypotheses.
3. **Convergence gating:** Failing PCG residual tolerance transitions Hodge state to `UNCONV` (`state = 3`), invalidates `PROD_W2_HODGE`, and suppresses overlay rendering.

### 3.3 Two-body occluded conservation and four-branch ambiguity (`src/physics_id.rs`, Gate `I5b`)

Across co-occluded track pairs (`CollisionTrackView::coasting`), all four branches (`H0_KEEP`, `H0_SWAP`, `H1_KEEP`, `H1_SWAP`) are tested against kinematic continuity, equal-and-opposite impulse balance, linear/angular momentum residuals, and admissible restitution (`0 <= e <= 1`) using declared non-circular mass priors. Symmetric equal-mass elastic head-on collisions behind an occluder admit both `H0_SWAP` and `H1_KEEP` as `AMBIGUOUS` rather than fabricating a unique winner, while unphysical impulse or energy violations emit `VIOLATED`.

### 3.4 Zero-ML depth sensing, stereo BOS, and external depth firewall (`src/depth.rs`, Gate `I7`)

1. **Monocular pure-rotation & mixed-motion refusals:** Pure tripod yaw or stationary views produce zero geometric parallax. The 3-DoF Huber-IRLS rotation gate (`REFUSED_PURE_ROTATION`, `REFUSED_NO_BASELINE`) and lateral/radial cross-flow purity regressions (`MIXED_ROT_TRANS`) fail closed before emitting `MONO_METRIC`. Sub-threshold yaw (`~0.004-0.0075 rad/s`) in low-relief forward scenes sits near the uncalibrated monocular observability floor.
2. **Intrinsics-free optical looming FoE/TTC:** Time-to-contact $\tau = Z / -\dot{Z}$ from radial `1/r(t)` scaling requires fixed focal length $f$ and constant approach speed; zoom changes break looming scale invariance.
3. **Calibrated stereo BOS (`256x128`, `P1`):** Rejects periodic textures (bilateral horizontal autocorrelation screening), vertical epipolar misalignment (`|eps_y| > 0.25 px`), half-occluded scanlines, insufficient coverage (`< 512` nodes or `< 32` cells), and stale references across scene cuts. Mono `speed_m_s` and stereo `baseline_m` remain separate calibration fields.
4. **External depth provenance firewall:** Consume-once `(frame_idx, continuity_epoch)`-bound `<= 128x64` staging expires after its target frame (`EXT_STALE`). External neural depth maps remain tagged `ASSUMED_NEURAL_PRIOR` (`DEPTH_REASON_ASSUMED_NEURAL_PRIOR`) and are never relabeled as surveyed metrology.

### 3.5 Pocket wind tunnel (`src/wind_tunnel.rs`, Gates `I9` & `I10a`)

1. **Planar incompressibility & Navier-Stokes integrability:** Strong 3D out-of-plane divergence (`div_frac > 0.25`, `HIGH_DIV`) or non-conservative momentum residual (`HIGH_NS_RESIDUAL`) refuses pressure and load claims (`REFUSED`).
2. **Load prerequisites:** Aerodynamic lift/drag (`C_L`, `C_D`) require both an enclosed obstacle void (`beta_1 >= 1`, else `NO_OBSTACLE_VOID`) and a uniform freestream band (`NO_FREESTREAM`); viscosity $\nu$ requires sufficient vorticity conditioning (`LOW_VORT_COND`).
3. **Dense-LK audit fallback (Mode `3`):** On unseeded real-camera fluids (for example `real_karman_mri_cylinder.mp4`), Mode `3` extracts a dense `64x36` optical-flow field but structurally caps state at `READY_FIELD_ONLY` (`WIND_PROV_DENSE_LK`) with `C_p`, `C_L`, and `C_D` zeroed so dense optical flow never fabricates pressure-integration lift coefficients. Static decoys and exposure flicker refuse with `NO_CONTINUUM | LOW_SUPPORT`.

### 3.6 kHz optical elastography & lock-in acoustic BOS (`src/elastography.rs`, Gate `I8`)

1. **Dispersion and spatial-scale guards:** Surface wave modes (`1..3`) reject spatially aliased wavelengths (`SPATIAL_ALIASING`), rigid-body or aperture-exceeding wavelengths (`WAVELENGTH_TOO_LONG`), pure standing waves with zero net Poynting energy transport (`STANDING_WAVE_DOMINANT`), exposure nulls (`EXPOSURE_NULL`), and low coherent coverage (`LOW_COH_COVERAGE`).
2. **BOS scratch mutex (`P1..P5`):** Because Stereo (`P1`), Thermal BOS (`P2`), Wind Tunnel (`P3`), Lock-In Acoustic BOS (`P4`), and Surface Elastography (`P5`) share `BosState`'s `256x128` scratch buffers, any lower-priority solver invoked while a higher-priority owner holds the bus refuses cleanly with `MUTEX_CONFLICT` instead of corrupting scratch memory. The v3.5.0 Instrument UI prevents multi-claim configurations atomically via its verified FSM planner (`scripts/verify_ui_fsm.mjs`).

### 3.7 8-axis physics integrity audit (`chronos-audit/1`, Gate `I10n`)

The deterministic auditor ([native/capi/src/audit.rs](../native/capi/src/audit.rs), [native/python/chronos_audit.py](../native/python/chronos_audit.py)) separates **applicability** (`unknown`, `absent`, `insufficient`, `eligible`) from **verdict** (`none`, `pass`, `warn`, `violated`). An axis that lacks evidence (`NO_EVIDENCE`), spans too few frames (`TOO_SHORT`), or suffers camera motion (`CAMERA_MOTION`), hard cuts (`HARD_CUT`), low texture (`LOW_TEXTURE`), symmetry ambiguity (`AMBIGUOUS`), missing audio (`NO_AUDIO`), or missing calibration (`NO_CALIB`) reports `NA` rather than a false `PASS`. When a profile declares `required_axes_mask`, blank or static decoys that evade measurement receive `overall_grade = NA` and anti-gaming `reward_milli = -1000`.

---

## 4. Excluded and out-of-scope regimes

To avoid unverified metrology claims, the following acquisition regimes are explicitly rejected or marked unavailable:
- Handheld smartphone `BOS` without physical tripod stabilization and stationary background reference reset.
- Uncalibrated gas thermometry ($\Delta T$) or densitometry ($\Delta\rho$) without calibrated optical path geometry and Gladstone-Dale constants.
- Rolling-shutter acoustic or vibration phase attribution without calibrated per-row readout timing (`dt_row_us`), exposure $T_e$, known PTS, native `1:1` orientation mapping, and multi-segment capture-clock coherence.
- Dual-camera `SE(3)` triangulation from ungenlocked consumer webcams experiencing uncompensated inter-camera shutter skew.
- Dense optical-flow pressure or lift/drag certification on unseeded 3D flows without enclosed-obstacle PIV boundary support.

---

## 5. Verification commands

Run the automated capture, metrology, depth, elastography, wind-tunnel, and audit verification suites:

```bash
node scripts/gate_i6_captures.mjs
node scripts/gate_i2_rs_welch.mjs
node scripts/gate_i3_hodge.mjs
node scripts/gate_i5b_conservation.mjs
node scripts/gate_i7_depth.mjs
node scripts/gate_i8_elastography.mjs
node scripts/gate_i9_wind_tunnel.mjs
node scripts/gate_i10_physics_audit.mjs
```
