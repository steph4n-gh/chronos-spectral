/*
 * CHRONOS-SPECTRAL Multi-Instance Native C ABI (v3.2.0)
 * Apache-2.0 License
 */
#ifndef CHRONOS_SPECTRAL_H
#define CHRONOS_SPECTRAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ChronosEngine ChronosEngine;

typedef enum ChronosElemKind {
    CHRONOS_ELEM_U8  = 1,
    CHRONOS_ELEM_U32 = 2,
    CHRONOS_ELEM_F32 = 3
} ChronosElemKind;

typedef enum ChronosFieldKind {
    CHRONOS_FIELD_TELEMETRY       = 0,
    CHRONOS_FIELD_ECA_RES         = 1,
    CHRONOS_FIELD_MICRO_TRACKS    = 2,
    CHRONOS_FIELD_CONTINUUM       = 3,
    CHRONOS_FIELD_RIGID_BODIES    = 4,
    CHRONOS_FIELD_SWARM_TELEMETRY = 5,
    CHRONOS_FIELD_BOS_PLANES      = 6,
    CHRONOS_FIELD_BOS_VALID       = 7,
    CHRONOS_FIELD_BOS_TELEMETRY   = 8,
    CHRONOS_FIELD_HODGE_PLANES    = 9,
    CHRONOS_FIELD_HODGE_TELEMETRY = 10,
    CHRONOS_FIELD_VIBRO_PLANES    = 11,
    CHRONOS_FIELD_VIBRO_TELEMETRY = 12,
    CHRONOS_FIELD_WAVE1_TELEMETRY = 13,
    CHRONOS_FIELD_WAVE2_TELEMETRY = 14,
    CHRONOS_FIELD_DEPTH_TELEMETRY = 15,
    CHRONOS_FIELD_DEPTH_FIELD     = 16,
    CHRONOS_FIELD_DEPTH_TRACKS    = 17,
    CHRONOS_FIELD_DEPTH_FLAGS     = 18,
    CHRONOS_FIELD_DEPTH_HERO      = 19
} ChronosFieldKind;

typedef struct ChronosFieldView {
    const void* ptr;
    size_t len;
    uint32_t elem_size;
    uint32_t elem_kind;
    uint32_t rows;
    uint32_t cols;
    uint32_t stride;
} ChronosFieldView;

/* Allocate and initialize an independent ChronosEngine (~23.3 MB heap arena). Thread-safe across distinct handles. */
ChronosEngine* chronos_engine_create(void);
void chronos_engine_destroy(ChronosEngine* engine);

/* Configure resolution (0 = 320x180 SD, 1 = 640x360 HD, 2 = 1280x720 with hires-720p), solver params, and metrology modes. */
void chronos_engine_set_resolution(ChronosEngine* engine, uint32_t hd);
void chronos_engine_set_resolution_mode(ChronosEngine* engine, uint32_t mode);
void chronos_engine_set_params(
    ChronosEngine* engine,
    uint32_t lookahead_k,
    float eca_adapt_rate,
    float fista_lambda,
    float eulerian_gain,
    float phase_offset_rad,
    uint32_t lab_mode
);
void chronos_engine_set_swarm_mode(ChronosEngine* engine, uint32_t mode);
uint32_t chronos_engine_set_bos_mode(ChronosEngine* engine, uint32_t mode);
uint32_t chronos_engine_set_bos_gauge_roi(ChronosEngine* engine, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1);
uint32_t chronos_engine_set_micro_metrology_mode(ChronosEngine* engine, uint32_t mode);
uint32_t chronos_engine_set_rs_calib(
    ChronosEngine* engine,
    uint32_t mode,
    float dt_row_us,
    float exposure_us,
    float audio_rate_hz,
    float max_skew_us
);
uint32_t chronos_engine_set_hodge_mode(ChronosEngine* engine, uint32_t mode);
/*
 * Depth Sensing Modes (`chronos_engine_set_depth_mode`):
 *   0 = Off (zero overhead), 1 = Monocular Parallax + FoE/TTC (fail-closed),
 *   2 = Calibrated Stereo BOS (256x128 Left-ref planes; Right-frame 64x36 field/RGBA),
 *   3 = External Depth Plane (consume-once per tick), 4 = Auto-Fused (Stereo -> Ext -> Mono).
 *
 * Calibration (`chronos_engine_set_depth_calib`):
 *   f_px:        focal length in pixels (0 = uncalibrated, or 10..=1e5)
 *   speed_m_s:   monocular camera translational speed in m/s (0 = up-to-scale, or 1e-4..=1e5;
 *                continuous monocular tracking operates over dt in [2 ms, 250 ms]; >250 ms resets epoch;
 *                rotational contamination below ~4% of angular flow speed [0.0012..0.004 rad/s] is below gate floor)
 *   baseline_m:  rectified stereo baseline in meters (0 = unset, or 1e-4..=1e4)
 *   max_disp_px: stereo search bound in pixels (1.0..=64.0)
 *   stereo_sign: -1 (Right camera to right of Left, ex <= 0), +1 (ex >= 0), 0 (auto)
 *
 * External Provenance (`chronos_engine_set_depth_external_meta`):
 *   1 = SURVEYED_LIDAR, 2 = SURVEYED_STEREO, 3 = ASSUMED_NEURAL_PRIOR (asserts reason bit 1<<5).
 *   Bound strictly to the current frame_idx and continuity_epoch; expires after one tick even if stereo wins AUTO_FUSED.
 *
 * Telemetry (`CHRONOS_FIELD_DEPTH_TELEMETRY`, 32 x uint32):
 *   words 0..15 are u32 counters/enums (2=state, 3=reasonMask [bit 3=NO_INTRINSICS/no metric scale, bit 4=EXT_STALE,
 *   bit 5=ASSUMED_NEURAL_PRIOR, bit 6=STEREO_NOT_READY, bit 7=MIXED_ROT_TRANS, bit 8=BAD_DT], 4=provenance,
 *   9=validCells, 12=extSeq, 13=stereoNodes);
 *   words 16..31 carry IEEE-754 f32 bit patterns (22..23=FoE xy, 24=minTTC, 25=bestR2, 28=epiRms, 29..31=Z med/min/max).
 */
uint32_t chronos_engine_set_depth_mode(ChronosEngine* engine, uint32_t mode);
uint32_t chronos_engine_set_depth_calib(
    ChronosEngine* engine,
    float f_px,
    float speed_m_s,
    float baseline_m,
    float max_disp_px,
    int32_t stereo_sign
);
uint32_t chronos_engine_depth_capture_stereo_left(ChronosEngine* engine);
uint32_t chronos_engine_set_depth_external_meta(
    ChronosEngine* engine,
    uint32_t w,
    uint32_t h,
    float f_px,
    uint32_t provenance,
    uint32_t seq
);
float* chronos_engine_depth_ingest_mut_ptr(ChronosEngine* engine, size_t* out_len);

/* Explicitly seed or pin a Hero-16 target inside [x, x+w] x [y, y+h] (engine pixel coords). Returns track_id or 0. */
uint32_t chronos_engine_seed_target(ChronosEngine* engine, float x, float y, float w, float h);
void chronos_engine_reset_tracks(ChronosEngine* engine);

/* Active engine frame dimensions (320x180, 640x360, or 1280x720). */
uint32_t chronos_engine_frame_w(const ChronosEngine* engine);
uint32_t chronos_engine_frame_h(const ChronosEngine* engine);

/* Copy one tightly-packed RGBA8 frame (`frame_w * frame_h * 4` bytes) into the next ring slot. Returns 0 on success. */
int32_t chronos_engine_ingest_rgba(ChronosEngine* engine, const uint8_t* rgba, size_t len);

/* Zero-copy writable pointer to the next RGBA8 ingest ring slot (`*out_len = frame_w * frame_h * 4`). */
uint8_t* chronos_engine_ingest_rgba_mut_ptr(ChronosEngine* engine, size_t* out_len);

/* Advance one frame (pre_tick + in-kernel Shifted Lanczos Fiedler + post_tick). Returns display ring slot index. */
uint32_t chronos_engine_step(ChronosEngine* engine, float dt_sec);
uint32_t chronos_engine_step_with_pts(ChronosEngine* engine, double pts_sec, float dt_sec);

/* Read-only pointer to the 1,568-float flat telemetry array (valid until next step/destroy). */
const float* chronos_engine_telemetry_ptr(const ChronosEngine* engine);
size_t chronos_engine_telemetry_len(void);

/* Zero-copy read-only view onto any internal field buffer (valid until next step/destroy). */
ChronosFieldView chronos_engine_field_view(const ChronosEngine* engine, uint32_t kind);

#ifdef __cplusplus
}
#endif

#endif /* CHRONOS_SPECTRAL_H */
