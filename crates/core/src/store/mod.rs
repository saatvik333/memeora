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
    /// Version chain: the memory this one directly supersedes (`None` for an original).
    pub parent_id: Option<String>,
    /// Version chain: the lineage root (`None` for an original); indexed for traversal.
    pub root_id: Option<String>,
    /// Bi-temporal valid-time: when the event occurred (vs `created_at` = when learned).
    pub occurred_start: Option<i64>,
    /// Bi-temporal valid-time: end of the occurrence interval, if any.
    pub occurred_end: Option<i64>,
    /// Distinct sources corroborating this memory (observation/consolidation layer).
    pub proof_count: u32,
    /// Durability for decay (Ebbinghaus/Cepeda); grows on spaced reinforcement.
    pub stability: f32,
    /// Times activated/reinforced (Hebbian potentiation).
    pub access_count: u32,
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
            parent_id: None,
            root_id: None,
            occurred_start: None,
            occurred_end: None,
            proof_count: 1,
            stability: 1.0,
            access_count: 0,
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

/// The kind of a directed edge between two memories in the knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// `from` supersedes `to` (contradiction-driven; Tier-1, deferred).
    Updates,
    /// `from` elaborates on a related `to` (moderate similarity).
    Extends,
    /// `from` was derived from `to` (LLM-driven; Tier-2, deferred).
    Derives,
}

impl EdgeKind {
    /// Stable string form persisted in the DB.
    pub fn as_str(self) -> &'static str {
        match self {
            EdgeKind::Updates => "updates",
            EdgeKind::Extends => "extends",
            EdgeKind::Derives => "derives",
        }
    }

    /// Parse from the persisted string, defaulting to [`EdgeKind::Extends`].
    pub fn from_str_lossy(s: &str) -> EdgeKind {
        match s {
            "updates" => EdgeKind::Updates,
            "derives" => EdgeKind::Derives,
            _ => EdgeKind::Extends,
        }
    }
}

/// Summary of one scope/container, for the dashboard's spaces switcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeInfo {
    /// The container tag.
    pub tag: String,
    /// Current-version (`is_latest = 1`) memories in the scope.
    pub latest: usize,
    /// All memories in the scope, including superseded/forgotten ones.
    pub total: usize,
}

/// A scope's full knowledge graph for visualization: every node (all versions,
/// including soft-forgotten) plus the edges among them.
#[derive(Debug, Clone, Default)]
pub struct GraphData {
    /// Memories in the scope (newest first). `is_latest = false` ones are dimmed
    /// in the UI rather than hidden. Embeddings are not hydrated (see [`Memory`]).
    pub nodes: Vec<Memory>,
    /// Directed edges whose endpoints are both in `nodes`.
    pub edges: Vec<Relationship>,
}

/// A directed edge `from_id --kind--> to_id` between two memories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relationship {
    /// Source memory id.
    pub from_id: String,
    /// Target memory id.
    pub to_id: String,
    /// Edge type.
    pub kind: EdgeKind,
    /// Creation time (Unix seconds).
    pub created_at: i64,
}

/// A scoped store of memories supporting vector KNN, lexical search, and a
/// directed knowledge graph over memories.
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

    /// Add a directed `from_id --kind--> to_id` edge. Idempotent (duplicate edges
    /// of the same kind are ignored).
    fn add_edge(&mut self, from_id: &str, to_id: &str, kind: EdgeKind) -> Result<()>;

    /// All outgoing edges from `id`.
    fn edges_from(&self, id: &str) -> Result<Vec<Relationship>>;

    /// Soft-forget a memory: mark it not-latest so retrieval skips it. The row is
    /// never hard-deleted (it stays fetchable by [`get`](VectorStore::get)).
    fn forget(&mut self, id: &str) -> Result<()>;
}
