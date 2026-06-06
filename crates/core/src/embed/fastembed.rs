//! Local ONNX embeddings via [`fastembed`] — the default, no-API-key backend.
//!
//! Models run on CPU, on-device, with no network at inference time (weights are
//! downloaded once to a cache dir). The default model is BGE-small-en-v1.5 (384-d),
//! a strong general-purpose retrieval embedder small enough for few-millisecond CPU
//! inference.
//!
//! Gated behind the `fastembed` feature so it pulls in ONNX Runtime only where the
//! product actually needs it (the daemon/cli), not in `core`'s fast unit tests.

use std::path::PathBuf;
use std::sync::Mutex;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::embed::{EmbeddingProvider, EmbeddingSpace};
use crate::error::{Error, Result};

/// A [`fastembed`]-backed embedding provider.
///
/// The underlying [`TextEmbedding`] holds a mutable ONNX session, so it lives
/// behind a [`Mutex`]: the trait exposes `&self` methods and the daemon shares one
/// instance across tasks. Inference is CPU-bound, so callers should invoke it from
/// a blocking context (e.g. `spawn_blocking`), never directly on a tokio worker.
pub struct FastEmbedder {
    model: Mutex<TextEmbedding>,
    space: EmbeddingSpace,
}

impl FastEmbedder {
    /// Load the default model (BGE-small-en-v1.5, 384-d), caching weights under
    /// `cache_dir` (or fastembed's default location when `None`).
    pub fn bge_small(cache_dir: Option<PathBuf>) -> Result<Self> {
        Self::new(EmbeddingModel::BGESmallENV15, cache_dir)
    }

    /// Load a specific fastembed model, caching weights under `cache_dir`
    /// (or fastembed's default location when `None`).
    pub fn new(model: EmbeddingModel, cache_dir: Option<PathBuf>) -> Result<Self> {
        let dim = model_dim(&model)?;
        let model_name = format!("{model:?}");

        let mut opts = InitOptions::new(model);
        if let Some(dir) = cache_dir {
            opts = opts.with_cache_dir(dir);
        }
        let embedder = TextEmbedding::try_new(opts).map_err(|e| Error::Embedding(e.to_string()))?;

        Ok(FastEmbedder {
            model: Mutex::new(embedder),
            space: EmbeddingSpace::new("fastembed", model_name, dim),
        })
    }
}

/// Look up a model's output dimensionality from fastembed's model registry.
fn model_dim(model: &EmbeddingModel) -> Result<usize> {
    TextEmbedding::list_supported_models()
        .into_iter()
        .find(|info| &info.model == model)
        .map(|info| info.dim)
        .ok_or_else(|| Error::Embedding(format!("unknown fastembed model: {model:?}")))
}

impl EmbeddingProvider for FastEmbedder {
    fn space(&self) -> &EmbeddingSpace {
        &self.space
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| Error::Embedding("fastembed model lock poisoned".into()))?;
        // `None` batch size uses fastembed's default (256). Vectors are L2-normalised.
        model
            .embed(texts, None)
            .map_err(|e| Error::Embedding(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "downloads the BGE-small model from HuggingFace on first run"]
    fn bge_small_embeds_with_expected_shape() {
        let embedder = FastEmbedder::bge_small(None).unwrap();
        assert_eq!(embedder.dim(), 384);
        assert_eq!(embedder.space().namespace(), "fastembed/BGESmallENV15/384");

        let out = embedder
            .embed_documents(&["the user prefers rust", "deploy with docker"])
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 384);

        // Embeddings are L2-normalised → magnitude ~1.0.
        let norm: f32 = out[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");

        // A query embeds to the same shape.
        let q = embedder.embed_query("which language?").unwrap();
        assert_eq!(q.len(), 384);
    }
}
