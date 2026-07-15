//! Error type for the core engine.

use std::fmt;

/// Errors returned by the core storage layer.
#[derive(Debug)]
pub enum Error {
    /// An error from the underlying SQLite layer.
    Sqlite(rusqlite::Error),
    /// A schema-migration error.
    Migration(rusqlite_migration::Error),
    /// An embedding vector did not match the store's configured dimensionality.
    DimMismatch {
        /// Dimensionality the store was created with.
        expected: usize,
        /// Dimensionality of the supplied vector.
        got: usize,
    },
    /// Invalid caller-provided input rejected before it reaches storage.
    Invalid(String),
    /// An embedding provider failed (model load, inference, or a poisoned lock).
    /// Carries a message because the underlying error type varies by backend.
    Embedding(String),
    /// The opt-in LLM extractor's transport/response failed. Non-fatal at the
    /// ingest layer (the extractor falls back to the heuristic); surfaced for logging.
    Llm(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Sqlite(e) => write!(f, "sqlite error: {e}"),
            Error::Migration(e) => write!(f, "migration error: {e}"),
            Error::DimMismatch { expected, got } => {
                write!(f, "embedding dim mismatch: expected {expected}, got {got}")
            }
            Error::Invalid(msg) => write!(f, "invalid input: {msg}"),
            Error::Embedding(msg) => write!(f, "embedding error: {msg}"),
            Error::Llm(msg) => write!(f, "llm extractor error: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Sqlite(e) => Some(e),
            Error::Migration(e) => Some(e),
            Error::DimMismatch { .. } | Error::Invalid(_) | Error::Embedding(_) | Error::Llm(_) => {
                None
            }
        }
    }
}

impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::Sqlite(e)
    }
}

impl From<rusqlite_migration::Error> for Error {
    fn from(e: rusqlite_migration::Error) -> Self {
        Error::Migration(e)
    }
}

/// Result alias for the core engine.
pub type Result<T> = std::result::Result<T, Error>;
