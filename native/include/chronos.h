/*
 * CHRONOS-SPECTRAL Multi-Instance Native C ABI (v3.5.0, ABI 4)
 * Apache-2.0 License
 */
#ifndef CHRONOS_SPECTRAL_H
#define CHRONOS_SPECTRAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CHRONOS_ABI_VERSION 4u
#define CHRONOS_AUDIO_STAGING_LEN 4096u
#define CHRONOS_AUDIT_AXIS_COUNT 8u

typedef struct ChronosEngine ChronosEngine;

typedef enum ChronosElemKind {
    CHRONOS_ELEM_U8  = 1,
    CHRONOS_ELEM_U32 = 2,
    CHRONOS_ELEM_F32 = 3
} ChronosElemKind;

typedef enum ChronosFieldLayout {
    CHRONOS_LAYOUT_FLAT        = 0,
    CHRONOS_LAYOUT_ROWS_COLS   = 1,
    CHRONOS_LAYOUT_PLANES      = 2, /* stride x rows x cols */
    CHRONOS_LAYOUT_INTERLEAVED = 3  /* rows x cols x stride */
} ChronosFieldLayout;

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
    CHRONOS_FIELD_DEPTH_HERO      = 19,
    CHRONOS_FIELD_WIND_TELEMETRY   = 20,
    CHRONOS_FIELD_WIND_PLANES      = 21,
    CHRONOS_FIELD_ELASTO_TELEMETRY = 22,
    CHRONOS_FIELD_ELASTO_PLANES    = 23,
    CHRONOS_FIELD_ELASTO_VALID     = 24
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

/* Ingest flags matching kernel INGEST_FLAG_* bits */
#define CHRONOS_INGEST_PTS_KNOWN   (1u << 1)
#define CHRONOS_INGEST_CAPTURE_AGE (1u << 2)
#define CHRONOS_INGEST_HANDOFF_AGE (1u << 3)
#define CHRONOS_INGEST_GEOM_KNOWN  (1u << 4)
#define CHRONOS_INGEST_ROI_SLEW    (1u << 6)

typedef struct ChronosIngestMeta {
    uint32_t struct_size;
    uint32_t flags;
    uint64_t pts_us;
    uint32_t epoch;
    uint32_t seq;
    float    age_ms;
    uint32_t src_w;
    uint32_t src_h;
    uint32_t crop_x;
    uint32_t crop_y;
    uint32_t crop_w;
    uint32_t crop_h;
    uint32_t dst_x;
    uint32_t dst_y;
    uint32_t dst_w;
    uint32_t dst_h;
    uint32_t orientation;
    uint32_t scan_dir; /* UINT32_MAX = unknown */
    uint32_t gap_mask;
    uint32_t native_1to1;
    uint32_t reserved[3];
} ChronosIngestMeta; /* 96 bytes */

/* Physics Integrity Audit enums, profile claims, and structs (v3.4.0) */
typedef enum ChronosAuditAxisId {
    CHRONOS_AXIS_TRAJECTORY       = 0,
    CHRONOS_AXIS_COLLISION        = 1,
    CHRONOS_AXIS_DEPTH_PARALLAX   = 2,
    CHRONOS_AXIS_FLUID_DIV        = 3,
    CHRONOS_AXIS_FLUID_MOMENTUM   = 4,
    CHRONOS_AXIS_BG_SHIMMER       = 5,
    CHRONOS_AXIS_TEMPORAL_CADENCE = 6,
    CHRONOS_AXIS_AV_SYNC          = 7
} ChronosAuditAxisId;

typedef enum ChronosAuditApplicability {
    CHRONOS_APPL_UNKNOWN      = 0,
    CHRONOS_APPL_ABSENT       = 1,
    CHRONOS_APPL_INSUFFICIENT = 2,
    CHRONOS_APPL_ELIGIBLE     = 3
} ChronosAuditApplicability;

typedef enum ChronosAuditVerdict {
    CHRONOS_VERDICT_NONE     = 0,
    CHRONOS_VERDICT_PASS     = 1,
    CHRONOS_VERDICT_WARN     = 2,
    CHRONOS_VERDICT_VIOLATED = 3
} ChronosAuditVerdict;

typedef enum ChronosAuditGrade {
    CHRONOS_GRADE_NA   = 0,
    CHRONOS_GRADE_PASS = 1,
    CHRONOS_GRADE_WARN = 2,
    CHRONOS_GRADE_FAIL = 3
} ChronosAuditGrade;

#define CHRONOS_AUDIT_CLAIM_3D_TRANSLATION     (1u << 0)
#define CHRONOS_AUDIT_CLAIM_FIXED_FOCAL        (1u << 1)
#define CHRONOS_AUDIT_CLAIM_PLANAR_FLUID       (1u << 2)
#define CHRONOS_AUDIT_CLAIM_STATIC_BG          (1u << 3)
#define CHRONOS_AUDIT_CLAIM_DIRECT_AUDIO       (1u << 4)
#define CHRONOS_AUDIT_CLAIM_CONTINUOUS_CAPTURE (1u << 5)

#define CHRONOS_NA_NO_EVIDENCE   (1u << 0)
#define CHRONOS_NA_TOO_SHORT     (1u << 1)
#define CHRONOS_NA_CAMERA_MOTION (1u << 2)
#define CHRONOS_NA_HARD_CUT      (1u << 3)
#define CHRONOS_NA_LOW_TEXTURE   (1u << 4)
#define CHRONOS_NA_AMBIGUOUS     (1u << 5)
#define CHRONOS_NA_NO_AUDIO      (1u << 6)
#define CHRONOS_NA_NO_CALIB      (1u << 7)
#define CHRONOS_NA_DISABLED      (1u << 8)

typedef struct ChronosAuditConfig {
    uint32_t struct_size;
    uint32_t profile_flags;
    uint32_t required_axes_mask;
    uint32_t expected_frames;
    uint32_t min_applicable_frames;
    uint32_t span_merge_gap_frames;
    float    fail_severity;
    float    min_coverage;
    uint32_t reserved[2];
} ChronosAuditConfig; /* 40 bytes */

typedef struct ChronosAuditAxis {
    uint32_t axis_id;
    uint32_t applicability;
    uint32_t verdict;
    uint32_t na_reason_mask;
    uint32_t applicable_frames;
    uint32_t warned_frames;
    uint32_t violated_frames;
    uint32_t span_count;
    float    score;          /* valid in [0, 1] iff applicability == ELIGIBLE; -1.0 otherwise */
    float    worst_residual; /* -1.0 when no qualified measurement exists */
} ChronosAuditAxis; /* 40 bytes */

typedef struct ChronosAuditSpan {
    uint32_t axis_id;
    uint32_t code;
    uint32_t f0;
    uint32_t f1;
    uint64_t t0_us;
    uint64_t t1_us;
    int32_t  track_id; /* -1 = scene-level */
    float    severity;
    float    bbox[4];  /* normalized to engine frame */
    float    measured;
    float    expected;
} ChronosAuditSpan; /* 64 bytes */

typedef struct ChronosAuditSummary {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t schema_version;
    uint32_t profile_flags;
    uint32_t required_axes_mask;
    uint32_t known_modalities;
    uint32_t present_modalities;
    uint32_t frames_seen;
    uint32_t frames_committed;
    uint32_t frames_rejected;
    uint32_t shots;
    uint32_t audio_blocks;
    uint32_t audio_underruns;
    uint32_t axis_count;
    uint32_t applicable_axes;
    uint32_t violated_axes;
    uint32_t warned_axes;
    uint32_t span_count;
    uint32_t spans_dropped;
    uint32_t overall_grade;
    uint32_t coverage_permille;
    int32_t  reward_milli; /* INT32_MIN when unprofiled & unscorable */
    float    integrity;    /* -1.0 when applicable_axes == 0 */
    uint32_t reserved;
    ChronosAuditAxis axes[CHRONOS_AUDIT_AXIS_COUNT];
} ChronosAuditSummary; /* 416 bytes */

#if defined(__cplusplus)
static_assert(sizeof(ChronosIngestMeta) == 96, "ChronosIngestMeta size mismatch");
static_assert(sizeof(ChronosAuditConfig) == 40, "ChronosAuditConfig size mismatch");
static_assert(sizeof(ChronosAuditAxis) == 40, "ChronosAuditAxis size mismatch");
static_assert(sizeof(ChronosAuditSpan) == 64, "ChronosAuditSpan size mismatch");
static_assert(sizeof(ChronosAuditSummary) == 416, "ChronosAuditSummary size mismatch");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(ChronosIngestMeta) == 96, "ChronosIngestMeta size mismatch");
_Static_assert(sizeof(ChronosAuditConfig) == 40, "ChronosAuditConfig size mismatch");
_Static_assert(sizeof(ChronosAuditAxis) == 40, "ChronosAuditAxis size mismatch");
_Static_assert(sizeof(ChronosAuditSpan) == 64, "ChronosAuditSpan size mismatch");
_Static_assert(sizeof(ChronosAuditSummary) == 416, "ChronosAuditSummary size mismatch");
#endif

/* ABI metadata & layout query */
uint32_t    chronos_abi_version(void);
const char* chronos_version_string(void);
uint32_t    chronos_field_layout(uint32_t kind);

/* Allocate, reset, and destroy an independent ChronosEngine (~23.3 MB heap arena). Thread-safe across distinct handles. */
ChronosEngine* chronos_engine_create(void);
void chronos_engine_reset(ChronosEngine* engine);
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
/* Wind Tunnel modes: 0 = Off, 1 = Staggered PIV, 2 = Every-Frame PIV, 3 = Dense Optical-Flow Audit (READY_FIELD_ONLY). */
uint32_t chronos_engine_set_wind_mode(ChronosEngine* engine, uint32_t mode);
uint32_t chronos_engine_set_wind_fluid_density(ChronosEngine* engine, float rho_kg_m3);
uint32_t chronos_engine_set_elasto_mode(ChronosEngine* engine, uint32_t mode);
uint32_t chronos_engine_set_elasto_lock_freqs(
    ChronosEngine* engine,
    uint32_t count,
    float f0,
    float f1,
    float f2,
    float f3
);
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

/* Per-engine mono f32 audio ingest (exclusive t_end_us on video timeline). */
float*   chronos_engine_audio_staging_mut_ptr(ChronosEngine* engine, size_t* out_len);
uint32_t chronos_engine_push_audio_block(
    ChronosEngine* engine,
    uint32_t num_samples,
    uint64_t t_end_us,
    float sample_rate_hz,
    float est_skew_us
);
int32_t  chronos_engine_push_audio(
    ChronosEngine* engine,
    const float* mono,
    size_t num_samples,
    uint64_t t_end_us,
    float sample_rate_hz
);

/* Explicit ingest metadata and deterministic integer area-resample + letterbox (fit: 0=letterbox, 1=stretch). */
int32_t chronos_engine_set_ingest_meta(ChronosEngine* engine, const ChronosIngestMeta* meta);
int32_t chronos_engine_ingest_rgba_strided(
    ChronosEngine* engine,
    const uint8_t* rgba,
    uint32_t src_w,
    uint32_t src_h,
    size_t src_stride_bytes,
    uint32_t fit,
    ChronosIngestMeta* inout_meta
);

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

/* Advance one frame with explicit source PTS in seconds (populates Wave-1 ingest metadata automatically). */
uint32_t chronos_engine_step_with_pts(ChronosEngine* engine, double pts_sec, float dt_sec);

/* Zero-copy read-only view of the 1,568-float Hero telemetry array (`chronos_engine_telemetry_len()`). */
const float* chronos_engine_telemetry_ptr(const ChronosEngine* engine);
size_t chronos_engine_telemetry_len(void);

/* Zero-copy read-only view of any kernel field buffer (`kind`: 0..=24). Valid until next engine step/reset/destroy. */
ChronosFieldView chronos_engine_field_view(const ChronosEngine* engine, uint32_t kind);

/* Physics Integrity Audit lifecycle (off by default: zero per-tick overhead until audit_begin). */
int32_t chronos_engine_audit_begin(ChronosEngine* engine, const ChronosAuditConfig* cfg);
void    chronos_engine_audit_end(ChronosEngine* engine);
int32_t chronos_engine_audit_finalize(ChronosEngine* engine);
int32_t chronos_engine_audit_summary(const ChronosEngine* engine, ChronosAuditSummary* out);
size_t  chronos_engine_audit_spans(const ChronosEngine* engine, size_t start, ChronosAuditSpan* out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* CHRONOS_SPECTRAL_H */
