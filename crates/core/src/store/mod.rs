//! Storage abstraction: the [`VectorStore`] trait and its data types.
//!
//! The default implementation is [`sqlite::SqliteStore`] (SQLite + sqlite-vec + FTS5).
//! Other backends (e.g. LanceDB for large scale) can implement the same trait.

pub mod sqlite;

use std::time::{SystemTime, UNIX_EPOCH};

use crate::Result;

/// Current Unix time in seconds.
pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The kind of a memory, classified heuristically during extraction (step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// A stable fact ("Alex is a PM at Stripe").
    Fact,
    /// A preference ("prefers morning meetings").
    Preference,
    /// A time-bound episode ("met Alex for coffee Tuesday").
    Episode,
}

impl MemoryKind {
    /// Stable string form persisted in the DB.
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryKind::Fact => "fact",
            MemoryKind::Preference => "preference",
            MemoryKind::Episode => "episode",
        }
    }

    /// Parse from the persisted string, defaulting to [`MemoryKind::Fact`].
    pub fn from_str_lossy(s: &str) -> MemoryKind {
        match s {
            "preference" => MemoryKind::Preference,
            "episode" => MemoryKind::Episode,
            _ => MemoryKind::Fact,
        }
    }
}

/// A single memory: content plus its embedding, scope, and bookkeeping fields.
///
/// Note: on reads (`get`/`knn`/`text_search`) the `embedding` field is left empty —
/// the vector lives in the `vec0` index and is not hydrated back into [`Memory`].
#[derive(Debug, Clone)]
pub struct Memory {
    /// Stable unique id (caller-provided).
    pub id: String,
    /// The memory text.
    pub content: String,
    /// Heuristic classification.
    pub kind: MemoryKind,
    /// Scope (see [`crate::container_tag`]).
    pub container_tag: String,
    /// Embedding vector (must match the store's dimensionality on write).
    pub embedding: Vec<f32>,
    /// Whether this is the current version (superseded memories set this false).
    pub is_latest: bool,
    /// Reinforcement strength.
    pub strength: f32,
    /// Creation time (Unix seconds).
    pub created_at: i64,
    /// Last access time (Unix seconds).
    pub last_accessed_at: i64,
    /// Optional expiry (Unix seconds) for time-bound memories.
    pub expires_at: Option<i64>,
    /// Opaque JSON metadata.
    pub metadata: String,
}

impl Memory {
    /// Build a memory with sensible defaults (latest, strength 1.0, timestamps = now).
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        kind: MemoryKind,
        container_tag: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        let now = now_unix();
        Memory {
            id: id.into(),
            content: content.into(),
            kind,
            container_tag: container_tag.into(),
            embedding,
            is_latest: true,
            strength: 1.0,
            created_at: now,
            last_accessed_at: now,
            expires_at: None,
            metadata: "{}".to_string(),
        }
    }

    /// Whether this memory has passed its expiry time as of `now` (Unix seconds).
    /// Memories with no `expires_at` never expire.
    pub fn is_expired(&self, now: i64) -> bool {
        matches!(self.expires_at, Some(exp) if exp <= now)
    }
}

/// A memory with a relevance score. For `knn` the score is vector distance
/// (lower is closer); for `text_search` it is the BM25 score (lower is better).
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    /// The matched memory.
    pub memory: Memory,
    /// Distance (knn) or BM25 score (text_search) — lower is more relevant.
    pub score: f32,
}

/// A scoped store of memories supporting vector KNN and lexical search.
pub trait VectorStore {
    /// Insert or replace a memory (matched by `id`). Errors on embedding-dim mismatch.
    fn upsert(&mut self, memory: &Memory) -> Result<()>;

    /// K nearest neighbours to `query` within `container_tag`, closest first.
    fn knn(&self, container_tag: &str, query: &[f32], k: usize) -> Result<Vec<ScoredMemory>>;

    /// Full-text (BM25) search within `container_tag`, most relevant first.
    fn text_search(&self, container_tag: &str, query: &str, k: usize) -> Result<Vec<ScoredMemory>>;

    /// Fetch a memory by id.
    fn get(&self, id: &str) -> Result<Option<Memory>>;

    /// Count memories within `container_tag`.
    fn count(&self, container_tag: &str) -> Result<usize>;

    /// List the latest (current-version) memories in `container_tag`, newest
    /// first, up to `limit`. Embedding is not hydrated (see [`Memory`]).
    fn list_latest(&self, container_tag: &str, limit: usize) -> Result<Vec<Memory>>;

    /// Reinforce a memory: add `delta` to its strength and set `last_accessed_at`
    /// to now. A no-op if `id` is unknown.
    fn reinforce(&mut self, id: &str, delta: f32) -> Result<()>;
}
