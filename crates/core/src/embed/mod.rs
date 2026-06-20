//! Embedding abstraction: the [`EmbeddingProvider`] trait and its data types.
//!
//! Text is turned into dense vectors by an [`EmbeddingProvider`]. The default,
//! local, no-API-key backend is [`fastembed::FastEmbedder`] (behind the `fastembed`
//! feature). Any provider can be wrapped in a [`cache::CachingEmbedder`] to skip
//! re-embedding identical content.
//!
//! Vectors are namespaced by an [`EmbeddingSpace`] (provider + model + dims) so that
//! switching models triggers a *scoped* re-embed rather than silently mixing
//! incompatible vectors in the same index.

pub mod cache;
#[cfg(feature = "fastembed")]
pub mod fastembed;

pub use cache::CachingEmbedder;

use crate::Result;

/// Identifies the vector space a set of embeddings lives in.
///
/// Two embeddings are only comparable if they share the same space. The
/// [`namespace`](EmbeddingSpace::namespace) string is stable and suitable for
/// keying stored vectors / caches so a model switch never corrupts an index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingSpace {
    /// Backend that produced the vectors (e.g. `"fastembed"`).
    pub provider: String,
    /// Model identifier within the provider (e.g. `"bge-small-en-v1.5"`).
    pub model: String,
    /// Output dimensionality.
    pub dim: usize,
}

impl EmbeddingSpace {
    /// Build a space from its parts.
    pub fn new(provider: impl Into<String>, model: impl Into<String>, dim: usize) -> Self {
        EmbeddingSpace {
            provider: provider.into(),
            model: model.into(),
            dim,
        }
    }

    /// Stable identifier `"{provider}/{model}/{dim}"` for keying vectors and caches.
    pub fn namespace(&self) -> String {
        format!("{}/{}/{}", self.provider, self.model, self.dim)
    }
}

/// Turns text into dense embedding vectors.
///
/// Object-safe and `Send + Sync` so the daemon can hold a single
/// `Box<dyn EmbeddingProvider>` and share it across tasks. Implementations that
/// wrap mutable model state (e.g. ONNX sessions) use interior mutability, so all
/// methods take `&self`.
pub trait EmbeddingProvider: Send + Sync {
    /// The vector space these embeddings live in.
    fn space(&self) -> &EmbeddingSpace;

    /// Output dimensionality (convenience for `self.space().dim`).
    fn dim(&self) -> usize {
        self.space().dim
    }

    /// Embed a batch of documents. The returned vectors line up with `texts`
    /// (same length, same order) and each has length [`dim`](Self::dim).
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Embed a single search query. Defaults to treating the query as a document;
    /// providers whose models use a distinct query encoding override this.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed_documents(&[text])?;
        out.pop().ok_or_else(|| {
            crate::Error::Embedding("provider returned no embedding for query".into())
        })
    }

    /// Whether this provider runs entirely on the local machine (no network egress).
    /// Local providers are allowed under local-first; a remote/API provider overrides
    /// this to `false` so the consent policy can gate it. Defaults to local.
    fn is_local(&self) -> bool {
        true
    }
}

impl EmbeddingProvider for Box<dyn EmbeddingProvider> {
    fn space(&self) -> &EmbeddingSpace {
        (**self).space()
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_query(text)
    }

    // Must forward: the default `is_local` returns `true`, so without this a wrapped
    // remote provider would be misreported as local and bypass the consent policy.
    fn is_local(&self) -> bool {
        (**self).is_local()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_is_stable_and_descriptive() {
        let space = EmbeddingSpace::new("fastembed", "bge-small-en-v1.5", 384);
        assert_eq!(space.namespace(), "fastembed/bge-small-en-v1.5/384");
        assert_eq!(space.dim, 384);
    }
}
