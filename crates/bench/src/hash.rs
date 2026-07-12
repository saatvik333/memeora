//! FNV-1a hashing shared by the embedder (token buckets) and the split (partition).
//!
//! FNV-1a is used instead of `std`'s `DefaultHasher` because its output is
//! specified — stable across runs, platforms, and Rust versions — which is what
//! makes both the embeddings and the dev/held-out partition reproducible.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// 64-bit FNV-1a over `bytes`.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    fold(FNV_OFFSET, bytes)
}

/// 64-bit FNV-1a over `seed` (little-endian) followed by `bytes`.
pub(crate) fn fnv1a_seeded(seed: u64, bytes: &[u8]) -> u64 {
    fold(fold(FNV_OFFSET, &seed.to_le_bytes()), bytes)
}

fn fold(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_input_sensitive() {
        assert_eq!(fnv1a(b"token"), fnv1a(b"token"));
        assert_ne!(fnv1a(b"token"), fnv1a(b"tokeN"));
        assert_eq!(fnv1a_seeded(42, b"q1"), fnv1a_seeded(42, b"q1"));
        assert_ne!(fnv1a_seeded(42, b"q1"), fnv1a_seeded(43, b"q1"));
    }

    #[test]
    fn matches_known_fnv1a_vectors() {
        // Reference values for the 64-bit FNV-1a algorithm.
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    }
}
