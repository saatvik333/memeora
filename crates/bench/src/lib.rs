//! Offline retrieval benchmark harness for the memeora engine.
//!
//! Measures **retrieval recall** (not QA accuracy) of memeora's real search
//! pipeline — [`memeora_core::SqliteStore`] + hybrid [`memeora_core::search`] —
//! over public long-term-memory datasets (LongMemEval, LoCoMo). Every later
//! tuning change to the engine gets measured against this harness.
//!
//! The default embedder is a deterministic, dependency-free hashed bag-of-words
//! ([`embedder::HashedBowEmbedder`]), so a run needs no network and produces the
//! same numbers on every machine. Real model embeddings are an opt-in path
//! behind the `real-embeddings` feature.

pub mod datasets;
pub mod embedder;
pub mod harness;
mod hash;
pub mod metrics;
pub mod report;
pub mod split;

/// Boxed error for the bench harness: dataset, engine, and I/O errors all funnel here.
pub type BoxError = Box<dyn std::error::Error>;
