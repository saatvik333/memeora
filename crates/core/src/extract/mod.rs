//! Extraction: turn raw conversation text into candidate memories.
//!
//! An [`Extractor`] reads a block of (already-cleaned) text and proposes
//! [`Candidate`] memories — content + classified [`MemoryKind`] — *before* they
//! are embedded, scoped, and stored. The default, model-free backend is
//! [`heuristic::HeuristicExtractor`] (Tier-0). A Tier-1 ONNX NER/relation
//! extractor is planned behind a feature once the `ort` versions align
//! (see `docs/ARCHITECTURE.md`).

pub mod heuristic;
pub mod llm;

pub use heuristic::HeuristicExtractor;
pub use llm::{LlmConfig, LlmExtractor};

use crate::Result;
use crate::store::{Memory, MemoryKind};

/// A proposed memory, extracted from text but not yet embedded or stored.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The memory text (a single statement).
    pub content: String,
    /// Heuristically classified kind.
    pub kind: MemoryKind,
    /// Optional expiry carried from a temporal phrase (Unix seconds).
    pub expires_at: Option<i64>,
    /// Extractor confidence in `[0, 1]` — a filtering signal, not stored.
    pub confidence: f32,
}

impl Candidate {
    /// Finalize into a storable [`Memory`] by assigning identity, scope, and an
    /// embedding (produced elsewhere by an [`crate::EmbeddingProvider`]).
    pub fn into_memory(
        self,
        id: impl Into<String>,
        container_tag: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Memory {
        let mut memory = Memory::new(id, self.content, self.kind, container_tag, embedding);
        memory.expires_at = self.expires_at;
        memory
    }
}

/// Extracts [`Candidate`] memories from text.
///
/// Object-safe and `Send + Sync` so the daemon can hold one
/// `Box<dyn Extractor>` and share it across ingestion tasks.
pub trait Extractor: Send + Sync {
    /// Propose candidate memories from `text`. Returns an empty vec when nothing
    /// in the text looks worth remembering.
    fn extract(&self, text: &str) -> Result<Vec<Candidate>>;
}
