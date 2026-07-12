//! Deterministic offline embedder: hashed bag-of-words (the "hashing trick").

use std::collections::HashMap;

use memeora_core::{EmbeddingProvider, EmbeddingSpace};

use crate::hash::fnv1a;

/// A deterministic bag-of-words embedder using signed feature hashing.
///
/// Tokens (lowercased alphanumeric runs) are FNV-1a-hashed; the hash picks one
/// of `dim` buckets and a sign, term counts are square-root damped, and the
/// vector is L2-normalised — so dot product behaves like a damped-TF cosine
/// similarity over token overlap.
///
/// Zero I/O and zero model weights: the same text embeds to the same vector on
/// every machine, which is what makes the benchmark repeatable offline. It
/// measures *lexical* retrieval only (no paraphrase understanding) — treat the
/// numbers as a stable relative signal for engine tuning, not an absolute
/// quality ceiling. Use the `real-embeddings` feature for model-grade vectors.
pub struct HashedBowEmbedder {
    space: EmbeddingSpace,
}

impl HashedBowEmbedder {
    /// Default bucket count: enough to keep token collisions rare at
    /// conversation-scale vocabularies while staying cheap in sqlite-vec.
    pub const DEFAULT_DIM: usize = 512;

    /// Build an embedder with `dim` hash buckets.
    pub fn new(dim: usize) -> Self {
        HashedBowEmbedder {
            space: EmbeddingSpace::new("bench", "hashed-bow", dim),
        }
    }

    fn embed_one(&self, text: &str) -> Vec<f32> {
        let dim = self.space.dim;
        let mut counts: HashMap<u64, u32> = HashMap::new();
        for token in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
            if !token.is_empty() {
                *counts.entry(fnv1a(token.as_bytes())).or_insert(0) += 1;
            }
        }
        let mut v = vec![0.0f32; dim];
        for (h, count) in counts {
            let bucket = (h % dim as u64) as usize;
            let sign = if h >> 63 == 0 { 1.0 } else { -1.0 };
            v[bucket] += sign * (count as f32).sqrt();
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

impl EmbeddingProvider for HashedBowEmbedder {
    fn space(&self) -> &EmbeddingSpace {
        &self.space
    }

    fn embed_documents(&self, texts: &[&str]) -> memeora_core::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn deterministic_unit_vectors_of_configured_dim() {
        let e = HashedBowEmbedder::new(HashedBowEmbedder::DEFAULT_DIM);
        let out = e
            .embed_documents(&["The user adopted a puppy.", "The user adopted a puppy."])
            .unwrap();
        assert_eq!(out[0].len(), HashedBowEmbedder::DEFAULT_DIM);
        assert_eq!(out[0], out[1], "same text must embed identically");
        let norm: f32 = out[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }

    #[test]
    fn token_overlap_beats_disjoint_text() {
        let e = HashedBowEmbedder::new(HashedBowEmbedder::DEFAULT_DIM);
        let out = e
            .embed_documents(&[
                "the golden retriever puppy slept",
                "a golden retriever puppy was adopted",
                "quarterly revenue projections for the finance team",
            ])
            .unwrap();
        assert!(
            cosine(&out[0], &out[1]) > cosine(&out[0], &out[2]),
            "overlapping-token texts must be more similar than disjoint ones"
        );
    }

    #[test]
    fn empty_text_embeds_to_zero_vector() {
        let e = HashedBowEmbedder::new(8);
        let out = e.embed_documents(&["   ...   "]).unwrap();
        assert!(out[0].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn query_embedding_matches_document_embedding() {
        // Symmetric embedder: the default `embed_query` must agree with documents.
        let e = HashedBowEmbedder::new(64);
        let doc = e.embed_documents(&["where is the meeting"]).unwrap();
        let query = e.embed_query("where is the meeting").unwrap();
        assert_eq!(doc[0], query);
    }
}
