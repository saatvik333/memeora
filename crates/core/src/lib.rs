//! memeora engine core.
//!
//! Houses the local-first storage, embeddings, extraction, graph, and hybrid-search
//! building blocks. **Step 1 (this milestone)** implements the storage layer: a SQLite
//! database with a statically-registered `sqlite-vec` extension for vector KNN plus FTS5
//! for lexical search, exposed through the [`VectorStore`] trait and [`SqliteStore`].
//!
//! Embeddings, extraction, and ranking are added in later steps (see `docs/ARCHITECTURE.md`).

pub mod container_tag;
pub mod db;
pub mod embed;
mod error;
pub mod extract;
pub mod ingest;
pub mod models;
pub mod privacy;
pub mod profile;
pub mod search;
pub mod store;

pub use embed::{CachingEmbedder, EmbeddingProvider, EmbeddingSpace};
pub use error::{Error, Result};
pub use extract::{Candidate, Extractor, HeuristicExtractor};
pub use ingest::{
    IngestOutcome, IngestParams, PreparedCandidate, embed_candidates, ingest, ingest_candidates,
    ingest_prepared,
};
pub use privacy::{redact, sanitize, strip_private};
pub use profile::{Profile, ProfileCache, ProfileParams, build_profile};
pub use search::{RerankHit, Reranker, SearchMode, SearchParams, rerank_memories, search};
pub use store::sqlite::SqliteStore;
pub use store::{
    EdgeKind, GraphData, Memory, MemoryKind, Relationship, ScopeInfo, ScoredMemory, VectorStore,
    now_unix,
};

/// Crate version, surfaced by the daemon's capability handshake.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
