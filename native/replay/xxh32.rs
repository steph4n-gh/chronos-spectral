//! XXH32 (xxHash, 32-bit variant): the CSTRACE region digest.
//!
//! Specified in docs/VERIFICATION_AND_BENCHMARKS.md ("Digest hash: `XXH32`, seed `0`") and
//! implemented in JavaScript in scripts/trace_lib.mjs; both must agree bit for bit.

const P1: u32 = 0x9E37_79B1;
const P2: u32 = 0x85EB_CA77;
const P3: u32 = 0xC2B2_AE3D;
const P4: u32 = 0x27D4_EB2F;
const P5: u32 = 0x1656_67B1;

#[inline(always)]
fn round(acc: u32, lane: u32) -> u32 {
    acc.wrapping_add(lane.wrapping_mul(P2))
        .rotate_left(13)
        .wrapping_mul(P1)
}

#[inline(always)]
fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// XXH32 of `data` with `seed`.
pub fn xxh32(data: &[u8], seed: u32) -> u32 {
    let len = data.len();
    let mut rest = data;
    let mut h = if len >= 16 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);
        let mut stripes = data.chunks_exact(16);
        for s in &mut stripes {
            v1 = round(v1, le32(&s[0..4]));
            v2 = round(v2, le32(&s[4..8]));
            v3 = round(v3, le32(&s[8..12]));
            v4 = round(v4, le32(&s[12..16]));
        }
        rest = stripes.remainder();
        v1.rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18))
    } else {
        seed.wrapping_add(P5)
    };
    // The length is added mod 2^32, as in the reference implementation.
    h = h.wrapping_add(len as u32);
    let mut words = rest.chunks_exact(4);
    for w in &mut words {
        h = h
            .wrapping_add(le32(w).wrapping_mul(P3))
            .rotate_left(17)
            .wrapping_mul(P4);
    }
    for &b in words.remainder() {
        h = h
            .wrapping_add((b as u32).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
    }
    h ^= h >> 15;
    h = h.wrapping_mul(P2);
    h ^= h >> 13;
    h = h.wrapping_mul(P3);
    h ^= h >> 16;
    h
}

#[cfg(test)]
mod tests {
    use super::xxh32;

    /// The documented test vectors (docs/VERIFICATION_AND_BENCHMARKS.md), ASCII input, seed 0.
    #[test]
    fn documented_vectors() {
        let vectors: [(&str, u32); 7] = [
            ("", 0x02cc_5d05),
            ("a", 0x550d_7456),
            ("abc", 0x32d1_53ff),
            ("xxhash", 0x9a95_b70e),
            ("0123456789abcdef", 0xc2c4_5b69),
            ("0123456789abcdefg", 0xcc79_b217),
            ("Nobody inspects the spammish repetition", 0xe229_3b2f),
        ];
        for (input, want) in vectors {
            assert_eq!(xxh32(input.as_bytes(), 0), want, "XXH32({input:?})");
        }
    }

    /// Every tail shape (0..=3 trailing bytes after 0..=3 trailing words, with and without
    /// stripes) against a straightforward per-byte reference of the same specification.
    #[test]
    fn tail_shapes_match_reference() {
        fn reference(d: &[u8]) -> u32 {
            const P: [u32; 5] = [
                0x9E37_79B1,
                0x85EB_CA77,
                0xC2B2_AE3D,
                0x27D4_EB2F,
                0x1656_67B1,
            ];
            let rd = |i: usize| u32::from_le_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]);
            let mut i = 0;
            let mut h = if d.len() >= 16 {
                let mut v = [P[0].wrapping_add(P[1]), P[1], 0, 0u32.wrapping_sub(P[0])];
                while i + 16 <= d.len() {
                    for (k, vk) in v.iter_mut().enumerate() {
                        *vk = vk
                            .wrapping_add(rd(i + 4 * k).wrapping_mul(P[1]))
                            .rotate_left(13)
                            .wrapping_mul(P[0]);
                    }
                    i += 16;
                }
                v[0].rotate_left(1)
                    .wrapping_add(v[1].rotate_left(7))
                    .wrapping_add(v[2].rotate_left(12))
                    .wrapping_add(v[3].rotate_left(18))
            } else {
                P[4]
            };
            h = h.wrapping_add(d.len() as u32);
            while i + 4 <= d.len() {
                h = h
                    .wrapping_add(rd(i).wrapping_mul(P[2]))
                    .rotate_left(17)
                    .wrapping_mul(P[3]);
                i += 4;
            }
            while i < d.len() {
                h = h
                    .wrapping_add((d[i] as u32).wrapping_mul(P[4]))
                    .rotate_left(11)
                    .wrapping_mul(P[0]);
                i += 1;
            }
            h ^= h >> 15;
            h = h.wrapping_mul(P[1]);
            h ^= h >> 13;
            h = h.wrapping_mul(P[2]);
            h ^ (h >> 16)
        }
        let data: Vec<u8> = (0..300u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        for n in 0..data.len() {
            assert_eq!(xxh32(&data[..n], 0), reference(&data[..n]), "length {n}");
        }
    }
}
