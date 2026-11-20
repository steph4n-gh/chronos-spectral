//! Platform-descriptive words: the only bytes the native/wasm parity check exempts.
//!
//! Each entry is one 32-bit word of a digested region whose value describes the build, not the
//! computation, so it legitimately differs between the wasm32 kernel and a native kernel. The
//! recorder (scripts/trace_hook.mjs) and every replayer zero exactly these words before hashing
//! the region, and the recorder writes the list into each trace (`meta.digestMask`). The driver
//! refuses a trace whose mask is not exactly this list, so a trace can never exempt more.
//!
//! The JavaScript copy is `PLATFORM_WORDS` in scripts/trace_lib.mjs; the two must match.

pub struct PlatformWord {
    pub region: &'static str,
    /// Index of the 32-bit word within the region.
    pub word: usize,
    pub why: &'static str,
}

pub const PLATFORM_WORDS: &[PlatformWord] = &[
    PlatformWord {
        region: "telemetry",
        word: 20,
        why: "size_of::<Arena>() as f32 (src/lib.rs chronos_init / chronos_reset_tracks / tick): the arena \
              holds usize fields, so its byte size depends on the target's pointer width",
    },
    PlatformWord {
        region: "wave1_telemetry",
        word: 1,
        why: "size_of::<Arena>() as u32 (src/lib.rs init_wave1_telemetry / finalize_wave1_stamps): \
              pointer-width dependent, as above",
    },
    PlatformWord {
        region: "wave1_telemetry",
        word: 12,
        why: "linked_static_end_bytes(): the wasm linker's __heap_base (end of the static image in \
              linear memory); a native build has no linear memory and reports 0",
    },
    PlatformWord {
        region: "wave2_telemetry",
        word: 1,
        why: "size_of::<Arena>() as u32 (src/lib.rs init_wave2_telemetry / finalize_wave2_stamps): \
              pointer-width dependent, as above",
    },
];

/// Masked word indices of `region`, ascending.
pub fn words_for(region: &str) -> Vec<usize> {
    let mut v: Vec<usize> = PLATFORM_WORDS
        .iter()
        .filter(|w| w.region == region)
        .map(|w| w.word)
        .collect();
    v.sort_unstable();
    v
}
