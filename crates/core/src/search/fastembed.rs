//! Cross-encoder reranking via [`fastembed`]'s `TextRerank` — an optional,
//! local, no-API-key quality upgrade over [`search`](crate::search::search).
//!
//! Gated behind the `fastembed` feature (shares the ONNX Runtime stack with the
//! embedder). The default model is BGE-reranker-base.

use std::path::PathBuf;
use std::sync::Mutex;

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

use crate::error::{Error, Result};
use crate::search::{RerankHit, Reranker};

/// A [`fastembed`]-backed cross-encoder reranker.
///
/// `TextRerank` holds a mutable ONNX session, so it lives behind a [`Mutex`]
/// (same rationale as the embedder). Reranking is CPU-bound — call it from a
/// blocking context, not a tokio worker.
pub struct FastEmbedReranker {
    model: Mutex<TextRerank>,
}

impl FastEmbedReranker {
    /// Load the default reranker (BGE-reranker-base), caching weights under
    /// `cache_dir` (or fastembed's default location when `None`).
    pub fn bge_base(cache_dir: Option<PathBuf>) -> Result<Self> {
        Self::new(RerankerModel::BGERerankerBase, cache_dir)
    }

    /// Load a specific reranker model.
    pub fn new(model: RerankerModel, cache_dir: Option<PathBuf>) -> Result<Self> {
        let mut opts = RerankInitOptions::new(model);
        if let Some(dir) = cache_dir {
            opts = opts.with_cache_dir(dir);
        }
        let model = TextRerank::try_new(opts).map_err(|e| Error::Embedding(e.to_string()))?;
        Ok(FastEmbedReranker {
            model: Mutex::new(model),
        })
    }
}

impl Reranker for FastEmbedReranker {
    fn rerank(&self, query: &str, docs: &[&str], top_k: usize) -> Result<Vec<RerankHit>> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| Error::Embedding("reranker model lock poisoned".into()))?;
        // `false` = don't return document text (we map back by index); results are
        // already sorted by descending relevance.
        let results = model
            .rerank(query, docs, false, None)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        Ok(results
            .into_iter()
            .take(top_k)
            .map(|r| RerankHit {
                index: r.index,
                score: r.score,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "downloads the BGE-reranker-base model from HuggingFace on first run"]
    fn reranks_by_relevance() {
        let reranker = FastEmbedReranker::bge_base(None).unwrap();
        let docs = [
            "the giant panda is a bear endemic to China",
            "i don't know",
            "pandas eat bamboo",
        ];
        let hits = reranker.rerank("what does a panda eat?", &docs, 3).unwrap();
        assert!(!hits.is_empty());
        // The most relevant doc should outrank the filler "i don't know" (index 1).
        assert_ne!(hits[0].index, 1);
    }
}
