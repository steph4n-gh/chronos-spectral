//! `chronos_spectral::native`: the pristine-arena reset and the native region layout the replay
//! driver validates traces against.
//!
//! The kernel is a global singleton, so everything lives in one test function (this file is its
//! own test binary, hence its own process).

use chronos_spectral::native::{self, RegionKind};

#[test]
fn reset_restores_fresh_state_and_regions_match_getters() {
    let regions = native::regions();
    let (lo, hi) = native::arena_range();
    assert_eq!(hi - lo, native::arena_bytes());

    // Names are unique; every fixed region lies inside the arena; fixed regions never overlap.
    for (i, r) in regions.iter().enumerate() {
        assert!(
            regions[i + 1..].iter().all(|o| o.name != r.name),
            "duplicate region {}",
            r.name
        );
        assert!(
            r.field_addr >= lo && r.field_addr + r.byte_len <= hi,
            "{} outside the arena",
            r.name
        );
        assert!(
            r.byte_len > 0 && r.byte_len % if r.elem == 1 { 1 } else { 4 } == 0,
            "{}",
            r.name
        );
        if matches!(r.kind, RegionKind::RingAlias | RegionKind::FixedAlias) {
            continue;
        }
        for o in &regions[i + 1..] {
            if !matches!(o.kind, RegionKind::RingAlias | RegionKind::FixedAlias) {
                let disjoint = r.field_addr + r.byte_len <= o.field_addr
                    || o.field_addr + o.byte_len <= r.field_addr;
                assert!(disjoint, "{} overlaps {}", r.name, o.name);
            }
        }
    }
    let by_name = |n: &str| regions.iter().find(|r| r.name == n).unwrap();

    unsafe {
        chronos_spectral::chronos_init();
        // Getters return the fields the layout table names.
        assert_eq!(
            chronos_spectral::chronos_get_telemetry_ptr() as usize,
            by_name("telemetry").field_addr
        );
        assert_eq!(
            chronos_spectral::chronos_get_hodge_inject_ptr() as usize,
            by_name("hodge_inject").field_addr
        );
        assert_eq!(
            chronos_spectral::chronos_get_ring_slot_ptr(11) as usize,
            by_name("ring_slot[11]").field_addr
        );
        assert_eq!(
            chronos_spectral::chronos_get_ingest_rgba_ptr() as usize,
            by_name("ring_slot[0]").field_addr
        );
        assert_eq!(
            chronos_spectral::chronos_get_elasto_planes_ptr() as usize,
            by_name("bos_planes").field_addr
        );
        assert_eq!(
            chronos_spectral::chronos_get_elasto_valid_ptr() as usize,
            by_name("bos_valid").field_addr
        );

        // chronos_init leaves non-zero state (telemetry[20] = arena bytes) ...
        let telem = std::slice::from_raw_parts(
            chronos_spectral::chronos_get_telemetry_ptr(),
            chronos_spectral::TELEMETRY_LEN,
        );
        assert_eq!(telem[20], native::arena_bytes() as f32);
        // ... and a reset returns every byte of the arena to zero, the fresh-instance image.
        native::reset_to_pristine();
        let arena = std::slice::from_raw_parts(lo as *const u8, hi - lo);
        assert!(
            arena.iter().all(|&b| b == 0),
            "arena not zero after reset_to_pristine"
        );
    }
}
