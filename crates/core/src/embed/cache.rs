//! A content-addressed cache that wraps any [`EmbeddingProvider`].
//!
//! Embedding is the expensive step; identical text recurs constantly (the same
//! preference re-stated, the same query re-run). [`CachingEmbedder`] memoises by
//! content so only cache *misses* reach the inner provider, and they are embedded
//! in a single batch.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::Result;
use crate::embed::{EmbeddingProvider, EmbeddingSpace};

/// Wraps an [`EmbeddingProvider`] with an in-memory, content-keyed cache.
///
/// The cache is keyed by the document text. Because each `CachingEmbedder` wraps
/// exactly one inner provider (one [`EmbeddingSpace`]), the text alone is a
/// sufficient key — vectors from different spaces never share a cache.
pub struct CachingEmbedder<E> {
    inner: E,
    cache: Mutex<HashMap<String, Vec<f32>>>,
}

impl<E: EmbeddingProvider> CachingEmbedder<E> {
    /// Wrap `inner` with an empty cache.
    pub fn new(inner: E) -> Self {
        CachingEmbedder {
            inner,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop the cache, returning the wrapped provider.
    pub fn into_inner(self) -> E {
        self.inner
    }
}

impl<E: EmbeddingProvider> EmbeddingProvider for CachingEmbedder<E> {
    fn space(&self) -> &EmbeddingSpace {
        self.inner.space()
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut miss_texts: Vec<&str> = Vec::new();
        let mut miss_indices: Vec<usize> = Vec::new();

        {
            let cache = self
                .cache
                .lock()
                .map_err(|_| crate::Error::Embedding("embedding cache lock poisoned".into()))?;
            for (i, &text) in texts.iter().enumerate() {
                match cache.get(text) {
                    Some(vec) => results[i] = Some(vec.clone()),
                    None => {
                        miss_texts.push(text);
                        miss_indices.push(i);
                    }
                }
            }
        }

        if !miss_texts.is_empty() {
            let embedded = self.inner.embed_documents(&miss_texts)?;
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| crate::Error::Embedding("embedding cache lock poisoned".into()))?;
            for (j, vec) in embedded.into_iter().enumerate() {
                let original = miss_indices[j];
                cache.insert(miss_texts[j].to_string(), vec.clone());
                results[original] = Some(vec);
            }
        }

        // Every slot is now filled (cache hit or freshly embedded).
        Ok(results.into_iter().map(|v| v.unwrap_or_default()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Deterministic provider: each text maps to a fixed-length vector derived
    /// from its bytes. Counts how many texts it actually embedded so tests can
    /// assert the cache shielded the inner provider.
    struct CountingEmbedder {
        space: EmbeddingSpace,
        embedded: AtomicUsize,
    }

    impl CountingEmbedder {
        fn new() -> Self {
            CountingEmbedder {
                space: EmbeddingSpace::new("mock", "counting", 4),
                embedded: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.embedded.load(Ordering::SeqCst)
        }
    }

    impl EmbeddingProvider for CountingEmbedder {
        fn space(&self) -> &EmbeddingSpace {
            &self.space
        }

        fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.embedded.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| {
                    let n = t.len() as f32;
                    vec![n, n + 1.0, n + 2.0, n + 3.0]
                })
                .collect())
        }
    }

    #[test]
    fn caches_repeated_text_across_calls() {
        let cache = CachingEmbedder::new(CountingEmbedder::new());

        let first = cache.embed_documents(&["alpha", "beta"]).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(cache.len(), 2);

        // Re-embedding the same texts must hit the cache (no new inner calls).
        let second = cache.embed_documents(&["alpha", "beta"]).unwrap();
        assert_eq!(first, second);

        let inner = cache.into_inner();
        assert_eq!(
            inner.calls(),
            2,
            "second batch should have been fully cached"
        );
    }

    #[test]
    fn only_misses_reach_inner_provider() {
        let cache = CachingEmbedder::new(CountingEmbedder::new());

        cache.embed_documents(&["alpha"]).unwrap();
        // "alpha" is cached, "gamma" is new → only one new inner embed.
        let out = cache.embed_documents(&["alpha", "gamma"]).unwrap();

        assert_eq!(out.len(), 2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.into_inner().calls(), 2);
    }

    #[test]
    fn preserves_input_order() {
        let cache = CachingEmbedder::new(CountingEmbedder::new());
        // Different lengths → distinguishable vectors; assert order is kept.
        let out = cache.embed_documents(&["a", "bbb", "cc"]).unwrap();
        assert_eq!(out[0][0], 1.0);
        assert_eq!(out[1][0], 3.0);
        assert_eq!(out[2][0], 2.0);
    }

    #[test]
    fn embed_query_uses_cache_and_returns_one_vector() {
        let cache = CachingEmbedder::new(CountingEmbedder::new());
        let q = cache.embed_query("alpha").unwrap();
        assert_eq!(q.len(), 4);
        // The query text is now cached; embedding it as a document is free.
        cache.embed_documents(&["alpha"]).unwrap();
        assert_eq!(cache.into_inner().calls(), 1);
    }
}
