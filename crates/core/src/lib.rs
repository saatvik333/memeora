//! memeora engine core.
//!
//! Houses the `EmbeddingProvider`, `Extractor`, and `VectorStore` traits plus the
//! graph, hybrid search (dense + BM25 + RRF), profiles, and forgetting logic.
//! Implementations are added per the build order in `docs/ARCHITECTURE.md`.

/// Crate version, surfaced by the daemon's capability handshake.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
