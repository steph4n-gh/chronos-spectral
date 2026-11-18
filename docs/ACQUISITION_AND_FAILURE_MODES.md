# Acquisition requirements, physical assumptions, and failure modes

CHRONOS-SPECTRAL is a deterministic `#![no_std]` WebAssembly visual physics kernel operating without pretrained neural weights or offline labeled training datasets. Physical accuracy therefore depends on explicit acquisition profiles, explicit observability limits, and reproducible failure fixtures under [web/samples/](./../web/samples/) verified by [scripts/gate_i6_captures.mjs](./../scripts/gate_i6_captures.mjs).

---

## 1. Per-mode acquisition profiles and physical assumptions

Every Wave-1 metrology mode pairs an assumed imaging geometry with bounded noise calibration and explicit failure flags:

| Capability | Mandatory acquisition profile | Physical assumptions | Primary breakdown modes |
| :--- | :--- | :--- | :--- |
| **Native `640x360` ROI tracking** | `1:1` sensor crop window (`crop_x0`, `crop_y0`) with orientation, scan direction, monotonic `t_src_us`, capture age `age_ms`, and gap flags; 100% frame-ring ingestion prior to any staggered solve | Known integer pixel scale and deterministic ingest cadence (`t - K` ring buffer) | Upstream H.264/HEVC `4x4`/`8x8` block quantization, deblocking smear, skipped B/P residuals, Bayer aliasing, ISP auto-gain pumps |
| **Ballistic-model residual & `2x2` covariance** | Stationary tripod or smoothly panning camera; explicit weak-perspective / planar assumptions when interpreting 2D residuals | Leave-one-out neighborhood flow approximates background/group drift; `2x2` inverse-Hessian plus empirical noise reflects local uncertainty | Pinhole perspective radial acceleration ($x = fX/Z \implies \ddot{x} \neq 0$ when $\ddot{\mathbf{X}}_{3\text{D}} \equiv \mathbf{0}$), parallax, camera shake, unmodeled motion blur |
| **Controlled tripod BOS** | Rigid stationary tripod, high-spatial-frequency non-periodic textured backdrop, unperturbed reference period before plume onset, zero smoke/tint saturation | Paraxial weak-refraction eikonal limit ($\boldsymbol{\varepsilon} = \nabla\Phi$), persistent stationary background texture, designated unperturbed reference gauge ROI | Tripod shock or handheld sway, rotational roll / breathing scale, motion blur, textureless walls, severe compression (`CRF >= 30`), curl-free optical zoom |

### 1.1 Native `640x360` ROI vs downscaling and codec bounds

Downscaling a `1920x1080` sensor stream to `640x360` discards `3x` linear (`9x` pixel area) spatial detail before derivatives or templates are evaluated. Retaining a `1:1` native `640x360` ROI window preserves full per-axis sampling inside the same `12`-frame Wasm ring buffer (`11,059,200 B`) while propagating `(crop_x0, crop_y0)`, orientation, and scan direction downstream.

Native ROI cropping is not optically or computationally zero-cost:
- **Upstream staging cost:** Host decode, `VideoFrame` staging, and ROI extraction still scale with upstream sensor dimensions and pixel format.
- **Codec quantization floor:** Subpixel optical flow (`IC-LK` and `BOS`) operates on `8`-bit `yuv420p` H.264/HEVC frames. Empirically on `web/samples/bos_thermal_plume.mp4` (`CRF 21`, `640x360` High profile), quiet unperturbed margins yield a residual displacement floor of `0.0530 px` RMS in the Wasm `128x64` BOS solver (`gate_i6_captures.mjs`) due to `4x4`/`8x8` DCT quantization and `4:2:0` chroma subsampling. Thermal refractive signals below `~0.15 px` cannot be cleanly separated from standard compressed-video noise.
- **Ingestion integrity:** Staggered downstream PDE solves may throttle compute per tick, but `100%` of incoming frames and source timestamps must enter the ring buffer before any solver skip so velocity finite differences and temporal phase never alias across silent drops.

### 1.2 Controlled tripod + textured-background BOS boundaries

Background-Oriented Schlieren (`BOS`) measures line-of-sight integrated refractive index gradients by cross-correlating apparent background displacement $\boldsymbol{\varepsilon}(x, y)$ against a persistent stationary textured backdrop:

1. **Inhomogeneous Neumann Poisson formulation:** Recovering scalar refractive potential $\Phi$ minimizes $\int_\Omega |\nabla\Phi - \boldsymbol{\varepsilon}|^2\,dA$, yielding:
   $$\Delta\Phi = \nabla \cdot \boldsymbol{\varepsilon} \quad \text{in } \Omega, \qquad \partial_n \Phi = \boldsymbol{\varepsilon} \cdot \mathbf{n} \quad \text{on } \partial\Omega$$
   Homogeneous Neumann boundaries ($\partial_n \Phi = 0$) collapse constant global shifts $\boldsymbol{\varepsilon} = (a, 0)$ to a flat constant instead of the physical linear ramp $\Phi = ax$.
2. **Curl consistency vs motion oracle:** Discrete curl $\nabla \times \boldsymbol{\varepsilon}$ checks integrability consistency of the gradient field only. Optical zoom, planar radial expansion, and longitudinal strain also satisfy $\nabla \times \boldsymbol{\varepsilon} = 0$, so low curl alone never certifies pure refractive flow.
3. **Uncalibrated physical provenance:** Outputs are reported strictly as apparent pixel displacement $\boldsymbol{\varepsilon}$ (`px`) and gauge-referenced scalar potential $\Phi$ relative to an unperturbed margin ROI. Quantitative gas density $\Delta\rho$ or temperature $\Delta T$ requires external optical path lengths ($Z_D$, $Z_B$) and Gladstone-Dale sensitivity constants.
4. **Why natural stochastic texture is mandatory:** Uniform blank walls or pure sky produce singular local Hessians ($\lambda_{\min} \to 0$), while repeating regular checkerboards risk half-period phase aliasing under shifts exceeding `1-2 px`. Multi-scale natural mineral or stone textures provide isotropic spatial power across every `128x64` BOS block.

### 1.3 Vector `2x2` covariance and perspective residual bounds

Single-scalar Hessian bounds discard directional uncertainty along edges (the aperture problem). Propagating symmetric `2x2` covariance `(c_xx, c_xy, c_yy)` preserves directional error ellipsoids while empirical calibration folds codec noise, differencing variance, interpolation bias, and shared jitter into the realized covariance:

- **Dimensional containment semantics:** A `1`-sigma `1D` marginal confidence interval captures `~68.3%` of Gaussian inliers, whereas a `1`-sigma `2D` Mahalanobis error ellipse captures `1 - exp(-1/2) = 39.35%` of isotropic 2D Gaussian inliers.
- **Pinhole perspective trap and monocular scale ambiguity:** Under perspective projection $x = f X / Z$, an object moving at strictly constant 3D velocity $\mathbf{V} = (V_X, V_Y, V_Z)$ with $V_Z < 0$ produces nonzero 2D image acceleration ($\ddot{x} \neq 0$, $\mathbf{a}_{\text{obs}} \cdot \mathbf{v} > 0$) purely from $1/Z(t)$ depth scaling. Conversely, single-camera observations carry fundamental **monocular projective scale ambiguity** $\lambda(t)\mathbf{X}(t)$: an object accelerating radially along the same line of sight can reproduce identical 2D pixel positions as an unaccelerated object approaching in depth $Z(t)$. Ballistic-model residual attention therefore pairs leave-one-out local flow covariance and sample-disjoint chord looming tests with explicit monocular ambiguity bounds to suppress looming and pan decoys within false-alert budgets.

---

## 2. Wave-1 semi-synthetic capture and failure fixtures (`web/samples/`)

Unlike the 14 real-camera recordings documented in [web/samples/ATTRIBUTION.md](../web/samples/ATTRIBUTION.md), the three compact `640x360` H.264 (`yuv420p`, High profile, `g=15`, `+faststart`) fixtures below (`3.37 MiB` combined) are **semi-synthetic physical compositions rendered over CC0 and Public Domain real-camera textures** (distinct from raw laboratory Schlieren film or raw outdoor camera captures) so refractive potentials, 3D rectilinear kinematics, and tripod-shock intervals remain analytically verifiable in Gate `I6`:

| Fixture | Duration / FPS | Size (`MiB`) | SHA-256 | Metrology role |
| :--- | :--- | :--- | :--- | :--- |
| [bos_thermal_plume.mp4](../web/samples/bos_thermal_plume.mp4) | `9.00 s` / `25 fps` (`225 f`) | `1.21 MiB` (`1,270,459 B`) | `adbe481a503a941e952d8380317445f00954926db055a77bc15ffeacfb30ada0` | Semi-synthetic tripod BOS positive baseline |
| [perspective_flyby_decoy.mp4](../web/samples/perspective_flyby_decoy.mp4) | `9.00 s` / `25 fps` (`225 f`) | `0.95 MiB` (`997,228 B`) | `83d78df42db8e58787642ffe454f26828069235e6331d03db1aeebda34494c1e` | Semi-synthetic constant-3D-velocity looming + pan decoy |
| [bos_handheld_failure.mp4](../web/samples/bos_handheld_failure.mp4) | `8.00 s` / `25 fps` (`200 f`) | `1.20 MiB` (`1,263,102 B`) | `7e672c9a7ac7631af7d83cdb35decc24f7c750ce130ac596f85b2df9a232a936` | Semi-synthetic BOS tripod-shock & sway failure |

### 2.1 Positive BOS composition: `bos_thermal_plume.mp4`

- **Construction:** Semi-synthetic composition built on a real CC0 high-resolution natural gravel backdrop (`3888x2592` cropped and `2x` supersampled before `640x360` area filtering). Frames `0..15` (`0.00-0.60 s`) hold an unperturbed `16`-frame static reference matching the `I5` BOS warmup window so background texture anchors lock cleanly. From `0.64 s` onward, a 2D Boussinesq Navier-Stokes thermal buoyancy plume integrates paraxial Gladstone-Dale ray deflections ($\boldsymbol{\varepsilon} = -C\nabla T = \nabla\Phi$, peak refractive shift `1.38 px` strictly inside the `1.50 px` bounded `8x8` affine LK pull-in limit, quiet margin shift `0.00 px`, `sigma = 0.14 DN` sensor noise, `CRF 21`). Recommended unperturbed gauge ROI on the `128x64` BOS grid: `[4, 4, 18, 60]`.
- **Why raw pixels appear clear while BOS resolves eddies:** No opaque smoke, color tint, or particle overlay is added. Raw frames resemble a static gravel wall with subtle convective shimmer (`6.080 DN` core RMS vs `3.381 DN` quiet-margin RMS at frame `140`), while the Wasm `128x64` BOS displacement plane (`chronos_get_bos_planes_ptr()`) in `gate_i6_captures.mjs` recovers `0.6647 px` peak center displacement against a `0.0530 px` RMS quiet-margin floor (`12.55x` displacement SNR, `max |phi| = 6.907`, `curl_rms = 0.01193`, `209` `READY` frames, `0` disturbed frames, `cam_motion_state == 0` across all `225` frames, and `0` false Hero detections).

### 2.2 Perspective flyby & pan decoy: `perspective_flyby_decoy.mp4`

- **Construction:** Textured outdoor horizon built from real CC0 speckled asphalt perspective tarmac and Public Domain stone ridges beneath multi-octave atmospheric clouds. Two multi-object formations traverse along strictly constant-3D-velocity rectilinear trajectories ($\ddot{\mathbf{X}}_{3\text{D}} \equiv \mathbf{0}$):
  - **Phase 1 (`0.00-5.40 s`, frames `0..134`):** Stationary tripod (`cam_motion_state == 0`). Nine approaching bodies accelerate radially across the pinhole image plane from `8.2 px/s` up to `86.2 px/s` (peaking at `73.1 px/s^2` 2D screen acceleration) purely from $x = f X / Z$ foreshortening while remaining strictly unaccelerated in 3D space.
  - **Phase 2 (`5.40-9.00 s`, frames `135..224`):** Smooth horizontal camera pan (`cam_motion_state == 1` locks cleanly across `86` consecutive frames `139..224`) inducing background optical flow and parallax across a second constant-3D-velocity formation entering continuously from the left border.
- **Failure prevented:** Naive `a_obs . v > 0` energy heuristics falsely promote looming pinhole approaches as maneuvering/propelled objects; Gate `I1` requires zero false maneuvering-object alarms across this decoy.

### 2.3 Reproducible BOS failure fixture: `bos_handheld_failure.mp4`

- **Construction:** Identical optical backdrop, sensor noise, and Gladstone-Dale thermal plume physics as `bos_thermal_plume.mp4`, divided into four controlled diagnostic intervals:
  1. `0.00-2.56 s` (frames `0..64`): Steady tripod baseline (`drift = 0.00 px`), clean BOS reference lock.
  2. `2.56-3.36 s` (frames `65..84`): Damped mechanical tripod shock (`+6.2 px` horizontal, `-3.8 px` vertical, `0.65 deg` roll, `+0.4%` scale, `3`-tap shutter exposure blur).
  3. `3.36-5.96 s` (frames `85..149`): Physiological handheld sway (`1.75-3.1 Hz` oscillations `+/- 3.6 px` horizontal, `+/- 2.4 px` vertical, `0.35 deg` roll, `0.5%` scale breathing, exposure blur).
  4. `5.96-8.00 s` (frames `150..199`): Settled displaced rest pose (`+3.1 px, -1.7 px, 0.22 deg` offset from the original frame-0 reference).
- **Why stabilization limits fail reproducibly:**
  - Global camera motion triggers immediately on frame `67` and persists across `91` consecutive frames (`67..157`), flagging tripod-mount violation.
  - Sub-pixel global 2D translation alignment cannot repair rotational roll (`0.22-0.65 deg`), breathing scale (`0.4-0.5%`), or exposure blur: roll injects genuine rotational curl ($\nabla \times \boldsymbol{\varepsilon} \neq 0$), scale injects curl-free radial strain, and directional blur attenuates high-frequency gravel gradients, burying the `~1 px` Gladstone-Dale thermal signal in stabilization error.
  - Once sway ceases (`frames 158..199`), `cam_motion_state` drops back to `0` while the static texture reference remains offset by `(+3.1, -1.7) px`, proving why stale BOS references must trigger re-acquisition after any shock event.

---

## 3. Excluded and out-of-scope regimes

To avoid unverified metrology claims, the following acquisition regimes are explicitly outside Wave-1 scope:
- Handheld smartphone `BOS` without physical tripod stabilization and stationary background reference reset.
- Uncalibrated gas thermometry (`Delta T`) or densitometry (`Delta rho`) without calibrated optical path geometry and Gladstone-Dale constants.
- Rolling-shutter acoustic/vibration phase attribution without calibrated per-row readout timing ($\Delta t_{\text{row}}$), exposure $T_e$, and multi-segment capture-clock coherence.
- Dual-camera `SE(3)` triangulation from ungenlocked consumer webcams experiencing uncompensated inter-camera shutter skew.

---

## 4. Verification command

Run the automated Gate `I6` capture, metadata, browser decode, attribution, and reproducible physics verification suite:

```bash
node scripts/gate_i6_captures.mjs
```
