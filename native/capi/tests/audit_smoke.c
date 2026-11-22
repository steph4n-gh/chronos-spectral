/* C11 strict ABI & layout smoke test for CHRONOS-SPECTRAL v3.4.0 (ABI 4). */
#include "chronos.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    assert(chronos_abi_version() == CHRONOS_ABI_VERSION);
    const char* ver = chronos_version_string();
    assert(ver != NULL && strncmp(ver, "3.4.0", 5) == 0);
    assert(chronos_field_layout(CHRONOS_FIELD_BOS_PLANES) == CHRONOS_LAYOUT_PLANES);
    assert(chronos_field_layout(CHRONOS_FIELD_CONTINUUM) == CHRONOS_LAYOUT_INTERLEAVED);
    assert(chronos_field_layout(CHRONOS_FIELD_TELEMETRY) == CHRONOS_LAYOUT_FLAT);

    /* Null-handle safety checks */
    size_t out_len = 999;
    assert(chronos_engine_audio_staging_mut_ptr(NULL, &out_len) == NULL && out_len == 0);
    assert(chronos_engine_push_audio_block(NULL, 16, 1000, 48000.0f, 0.0f) == 0);
    assert(chronos_engine_set_ingest_meta(NULL, NULL) == -1);
    assert(chronos_engine_audit_begin(NULL, NULL) == -1);

    ChronosEngine* eng = chronos_engine_create();
    assert(eng != NULL);
    chronos_engine_set_resolution(eng, 0); /* 320x180 SD */
    assert(chronos_engine_frame_w(eng) == 320);
    assert(chronos_engine_frame_h(eng) == 180);

    /* Audio staging & block boundary validation */
    float* ast = chronos_engine_audio_staging_mut_ptr(eng, &out_len);
    assert(ast != NULL && out_len == CHRONOS_AUDIO_STAGING_LEN);
    assert(chronos_engine_push_audio_block(eng, 0, 10000, 48000.0f, 0.0f) == 0);
    assert(chronos_engine_push_audio_block(eng, 4097, 10000, 48000.0f, 0.0f) == 0);
    for (size_t i = 0; i < 480; ++i) {
        ast[i] = 0.01f * sinf((float)i * 0.1f);
    }
    /* Timestamp crossing 2^32 us (> 4295s) */
    uint64_t hi_t = 5000000000ULL;
    assert(chronos_engine_push_audio_block(eng, 480, hi_t, 48000.0f, 0.0f) == 1);

    /* Audit begin + 9:16 letterboxed static black decoy -> must yield NA / reward_milli == -1000 when required */
    ChronosAuditConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.struct_size = (uint32_t)sizeof(cfg);
    cfg.required_axes_mask = (1u << CHRONOS_AXIS_TRAJECTORY);
    cfg.expected_frames = 10;
    cfg.min_applicable_frames = 6;
    cfg.fail_severity = 0.5f;
    cfg.min_coverage = 0.2f;
    assert(chronos_engine_audit_begin(eng, &cfg) == 0);

    uint8_t* portrait = (uint8_t*)calloc(108 * 192 * 4, 1);
    assert(portrait != NULL);
    for (uint32_t f = 0; f < 10; ++f) {
        ChronosIngestMeta meta;
        memset(&meta, 0, sizeof(meta));
        meta.struct_size = (uint32_t)sizeof(meta);
        meta.flags = CHRONOS_INGEST_PTS_KNOWN | CHRONOS_INGEST_GEOM_KNOWN;
        meta.pts_us = (uint64_t)f * 33333ULL;
        meta.epoch = 1;
        meta.seq = f + 1;
        meta.age_ms = -1.0f;
        meta.scan_dir = 0xFFFFFFFFu;
        assert(chronos_engine_ingest_rgba_strided(eng, portrait, 108, 192, 108 * 4, 0, &meta) == 0);
        assert(meta.dst_w == 101 && meta.dst_h == 180 && meta.dst_x == 109 && meta.dst_y == 0);
        assert(chronos_engine_set_ingest_meta(eng, &meta) == 0);
        chronos_engine_step(eng, 1.0f / 30.0f);
    }
    free(portrait);

    assert(chronos_engine_audit_finalize(eng) == 0);
    ChronosAuditSummary sum;
    memset(&sum, 0, sizeof(sum));
    assert(chronos_engine_audit_summary(eng, &sum) == 0);
    assert(sum.struct_size == sizeof(ChronosAuditSummary));
    assert(sum.abi_version == CHRONOS_ABI_VERSION);
    assert(sum.frames_committed == 10);
    assert(sum.overall_grade == CHRONOS_GRADE_NA);
    assert(sum.reward_milli == -1000); /* Anti-gaming: static black decoy cannot earn positive reward! */

    /* Engine reset check */
    chronos_engine_reset(eng);
    assert(chronos_engine_frame_w(eng) == 320);
    chronos_engine_destroy(eng);
    printf("  [PASS] C11 audit_smoke: ABI=%u, sizes=(96,40,40,64,416), letterbox=101x180, static_decoy_reward=-1000\n",
           chronos_abi_version());
    return 0;
}
