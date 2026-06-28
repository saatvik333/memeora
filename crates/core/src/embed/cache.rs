//! A content-addressed cache that wraps any [`EmbeddingProvider`].
//!
//! Embedding is the expensive step; identical text recurs constantly (the same
//! preference re-stated, the same query re-run). [`CachingEmbedder`] memoises by
//! content so only cache *misses* reach the inner provider, and they are embedded
//! in a single batch.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Mutex;

use crate::Result;
use crate::embed::{EmbeddingProvider, EmbeddingSpace};

/// Default entry cap. At ~1.5 KB per 384-d vector this bounds the cache to a few
/// tens of MB — enough to shield hot repeats without growing without limit.
pub const DEFAULT_CAPACITY: usize = 50_000;

/// Bounded, insertion-ordered (FIFO-evicted) content cache.
struct Cache {
    map: HashMap<String, Vec<f32>>,
    /// Keys in insertion order; the front is evicted first when over capacity.
    order: VecDeque<String>,
    capacity: usize,
}

impl Cache {
    fn new(capacity: usize) -> Self {
        Cache {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    fn get(&self, key: &str) -> Option<Vec<f32>> {
        self.map.get(key).cloned()
    }

    fn put(&mut self, key: String, value: Vec<f32>) {
        if let Some(slot) = self.map.get_mut(&key) {
            *slot = value; // refresh in place; keep its existing order position
            return;
        }
        while self.map.len() >= self.capacity {
            // Evict oldest-inserted until there is room for the new entry.
            match self.order.pop_front() {
                Some(old) => {
                    self.map.remove(&old);
                }
                None => break,
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }
}

/// Wraps an [`EmbeddingProvider`] with a bounded, content-keyed cache.
///
/// Keys are namespaced by intent — `d:` for documents, `q:` for queries — because
/// asymmetric models (BGE) embed a query differently from a document (an
/// instruction prefix). A query and an identical-text document therefore never
/// collide, and [`embed_query`](CachingEmbedder::embed_query) reaches the inner
/// provider's query path rather than the passage encoder.
pub struct CachingEmbedder<E> {
    inner: E,
    cache: Mutex<Cache>,
}

impl<E: EmbeddingProvider> CachingEmbedder<E> {
    /// Wrap `inner` with an empty cache of [`DEFAULT_CAPACITY`].
    pub fn new(inner: E) -> Self {
        Self::with_capacity(inner, DEFAULT_CAPACITY)
    }

    /// Wrap `inner` with an empty cache bounded to `capacity` entries.
    pub fn with_capacity(inner: E, capacity: usize) -> Self {
        CachingEmbedder {
            inner,
            cache: Mutex::new(Cache::new(capacity)),
        }
    }

    /// Recover the lock guard even if a previous holder panicked — a poisoned
    /// cache is at worst stale, never unsafe.
    fn lock(&self) -> std::sync::MutexGuard<'_, Cache> {
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.lock().map.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every cached entry.
    pub fn clear(&self) {
        let mut cache = self.lock();
        cache.map.clear();
        cache.order.clear();
    }

    /// Drop the cache, returning the wrapped provider.
    pub fn into_inner(self) -> E {
        self.inner
    }
}

/// Cache key for a document text.
fn doc_key(text: &str) -> String {
    format!("d:{text}")
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
            let cache = self.lock();
            for (i, &text) in texts.iter().enumerate() {
                match cache.get(&doc_key(text)) {
                    Some(vec) => results[i] = Some(vec),
                    None => {
                        miss_texts.push(text);
                        miss_indices.push(i);
                    }
                }
            }
        }

        if !miss_texts.is_empty() {
            let embedded = self.inner.embed_documents(&miss_texts)?;
            if embedded.len() != miss_texts.len() {
                return Err(crate::Error::Embedding(format!(
                    "inner provider returned {} vectors for {} texts",
                    embedded.len(),
                    miss_texts.len()
                )));
            }
            let mut cache = self.lock();
            for (j, vec) in embedded.into_iter().enumerate() {
                let original = miss_indices[j];
                cache.put(doc_key(miss_texts[j]), vec.clone());
                results[original] = Some(vec);
            }
        }

        // Every slot is now filled (cache hit or freshly embedded above).
        results
            .into_iter()
            .map(|v| v.ok_or_else(|| crate::Error::Embedding("cache slot left unfilled".into())))
            .collect()
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        // Queries are cached under a `q:` namespace and embedded via the inner
        // provider's query path (which applies any asymmetric instruction prefix),
        // never routed through `embed_documents` (the passage encoder).
        let key = format!("q:{text}");
        if let Some(vec) = self.lock().get(&key) {
            return Ok(vec);
        }
        let vec = self.inner.embed_query(text)?;
        self.lock().put(key, vec.clone());
        Ok(vec)
    }

    // Forward the inner provider's locality so wrapping a remote provider can't be
    // misreported as local and bypass the consent policy (matches the Box<dyn> impl).
    fn is_local(&self) -> bool {
        self.inner.is_local()
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

    /// Provider whose query embedding differs from its document embedding, to
    /// prove the cache preserves the query/document distinction.
    struct AsymmetricEmbedder(EmbeddingSpace);
    impl EmbeddingProvider for AsymmetricEmbedder {
        fn space(&self) -> &EmbeddingSpace {
            &self.0
        }
        fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
        fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.0, 1.0])
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
    fn query_is_cached_separately_from_documents() {
        let cache = CachingEmbedder::new(CountingEmbedder::new());
        let q1 = cache.embed_query("alpha").unwrap();
        // A second identical query is served from cache (no new inner call).
        let q2 = cache.embed_query("alpha").unwrap();
        assert_eq!(q1, q2);
        // A *document* with the same text is a separate namespace → one more embed.
        cache.embed_documents(&["alpha"]).unwrap();
        assert_eq!(
            cache.into_inner().calls(),
            2,
            "query and document caches must not collide"
        );
    }

    #[test]
    fn query_uses_inner_query_path_not_passage_encoder() {
        let cache =
            CachingEmbedder::new(AsymmetricEmbedder(EmbeddingSpace::new("mock", "asym", 2)));
        // Must return the query-flavored vector, not the document one.
        assert_eq!(cache.embed_query("hello").unwrap(), vec![0.0, 1.0]);
        assert_eq!(
            cache.embed_documents(&["hello"]).unwrap()[0],
            vec![1.0, 0.0]
        );
    }

    #[test]
    fn cache_is_bounded_and_evicts_oldest() {
        let cache = CachingEmbedder::with_capacity(CountingEmbedder::new(), 2);
        cache.embed_documents(&["a", "bb", "ccc"]).unwrap();
        // Capacity 2 → only the two newest survive; growth is bounded.
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }
}
