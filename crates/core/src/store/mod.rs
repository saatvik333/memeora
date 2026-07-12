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

/// A consolidated observation: one canonical belief distilled from a cluster of
/// near-duplicate memories, corroborated by a set of distinct source memories.
///
/// The observation layer sits one level above [`Memory`]: [`crate::consolidate`]
/// groups a scope's near-duplicate memories and writes one observation per cluster.
/// `proof_count` is a denormalized `COUNT(DISTINCT source_memory_id)` over the
/// observation's source set (mirroring the per-memory evidence/proof model), refreshed
/// by [`add_observation_source`](VectorStore::add_observation_source).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    /// Stable, deterministic id (keyed on the cluster's canonical member — see
    /// [`crate::consolidate`] — so re-consolidating the same memories converges).
    pub id: String,
    /// Scope (see [`crate::container_tag`]).
    pub container_tag: String,
    /// The canonical belief text (the synthesizer's output for the cluster).
    pub content: String,
    /// Distinct source memories corroborating this observation.
    pub proof_count: u32,
    /// Creation time (Unix seconds); preserved across re-consolidation.
    pub created_at: i64,
    /// Last time the observation's content or sources changed (Unix seconds).
    pub updated_at: i64,
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

    /// Record a corroborating observation for `memory_id` from `source_id`: insert
    /// `(quote, occurred_at)` into the evidence set (set-union — re-recording the same
    /// `source_id` is a no-op) and refresh `proof_count` to the distinct-source count.
    /// This is the consolidation signal: independent sources raise proof, one source
    /// restating does not. Strength/access are handled separately by
    /// [`reinforce`](VectorStore::reinforce). Default: no-op, for stores without an
    /// evidence table (their `proof_count` stays at its insert-time value).
    fn record_evidence(
        &mut self,
        _memory_id: &str,
        _source_id: &str,
        _quote: &str,
        _occurred_at: i64,
    ) -> Result<()> {
        Ok(())
    }

    /// Insert or update a consolidated [`Observation`] (matched by `id`). On update the
    /// original `created_at` is preserved; `content`, `container_tag`, `proof_count`, and
    /// `updated_at` are refreshed. `proof_count` is authoritatively (re)set by
    /// [`add_observation_source`](VectorStore::add_observation_source); the value written
    /// here is a hint. Default: no-op, for stores without an observation table.
    fn upsert_observation(&mut self, _observation: &Observation) -> Result<()> {
        Ok(())
    }

    /// List observations in `container_tag`, most-recently-updated first, up to `limit`.
    /// Default: empty, for stores without an observation table.
    fn list_observations(&self, _container_tag: &str, _limit: usize) -> Result<Vec<Observation>> {
        Ok(Vec::new())
    }

    /// Record `source_memory_id` as a distinct source for `observation_id`: insert into
    /// the source set (set-union via the composite PK — a repeated source is a no-op) and
    /// refresh the observation's `proof_count` to the distinct-source count. This mirrors
    /// [`record_evidence`](VectorStore::record_evidence) one level up: independent sources
    /// raise proof, re-linking a known source does not — which is what makes re-running
    /// consolidation idempotent. Default: no-op, for stores without an observation table.
    fn add_observation_source(
        &mut self,
        _observation_id: &str,
        _source_memory_id: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Add a directed `from_id --kind--> to_id` edge. Idempotent (duplicate edges
    /// of the same kind are ignored).
    fn add_edge(&mut self, from_id: &str, to_id: &str, kind: EdgeKind) -> Result<()>;

    /// Hebbian potentiation of graph edges co-activated during recall (Phase E). For each
    /// unordered `(a, b)` pair, bump every edge between `a` and `b` (either direction):
    /// `strength += `[`EDGE_POTENTIATION_DELTA`] capped at [`STRENGTH_MAX`], grow
    /// `stability` by [`EDGE_STABILITY_DELTA`] **only** when this activation is
    /// ≥ [`SPACING_SECS`] after the last (Cepeda spacing — rapid bursts don't build
    /// durability), and stamp `last_activated = now`. Unknown pairs are a no-op. Default:
    /// no-op, for stores without edge dynamics.
    ///
    /// **Write path only.** This is the mirror of [`reinforce`](VectorStore::reinforce) for
    /// edges: it mutates, so — like every write — it belongs to the sole-writer daemon.
    /// [`graph_search`](VectorStore::graph_search) runs on `&self` (a read) and must never
    /// call it; the daemon collects the edges a recall traversed and potentiates them on
    /// the recall write-back, off the read path.
    ///
    /// [`EDGE_POTENTIATION_DELTA`]: crate::dynamics::EDGE_POTENTIATION_DELTA
    /// [`EDGE_STABILITY_DELTA`]: crate::dynamics::EDGE_STABILITY_DELTA
    /// [`STRENGTH_MAX`]: crate::dynamics::STRENGTH_MAX
    /// [`SPACING_SECS`]: crate::dynamics::SPACING_SECS
    // ponytail: the recall write-back that feeds this pairs list lives in the daemon's
    // recall vertical (out of this crate's file scope) — wire graph_search's surfaced
    // seed→neighbor edges into a potentiate_edges call there, alongside the existing
    // reinforce write-back.
    fn potentiate_edges(&mut self, _pairs: &[(String, String)]) -> Result<()> {
        Ok(())
    }

    /// All outgoing edges from `id`.
    fn edges_from(&self, id: &str) -> Result<Vec<Relationship>>;

    /// Soft-forget a memory: mark it not-latest so retrieval skips it. The row is
    /// never hard-deleted (it stays fetchable by [`get`](VectorStore::get)).
    fn forget(&mut self, id: &str) -> Result<()>;

    /// Supersede `old_id` with `new`: store `new` as the current version linked into
    /// the version chain (`parent_id = old_id`, `root_id` = the old version's root, or
    /// `old_id` when the old version is itself a root), soft-forget the old version
    /// (kept as history, dropped from active retrieval), and record a
    /// `new --updates--> old` edge. Returns `Ok(false)` if `old_id` is unknown (no-op).
    ///
    /// Nothing is hard-deleted — the prior version stays `get`-able and on the chain.
    /// The default composes the trait's own atomic writes; wrap a standalone call in a
    /// store transaction (e.g. [`SqliteStore::transaction`]) when unit atomicity matters
    /// — the daemon already runs ingestion inside one.
    ///
    /// [`SqliteStore::transaction`]: sqlite::SqliteStore::transaction
    fn supersede(&mut self, old_id: &str, new: &Memory) -> Result<bool> {
        // Self-supersession is meaningless (and would forget the row we just wrote);
        // an exact restatement is a reinforce, handled upstream.
        if new.id == old_id {
            return Ok(false);
        }
        let Some(old) = self.get(old_id)? else {
            return Ok(false);
        };
        let mut linked = new.clone();
        linked.parent_id = Some(old_id.to_string());
        linked.root_id = Some(old.root_id.clone().unwrap_or_else(|| old_id.to_string()));
        self.upsert(&linked)?;
        self.forget(old_id)?;
        self.add_edge(&linked.id, old_id, EdgeKind::Updates)?;
        Ok(true)
    }

    /// The version chain for the lineage rooted at `root_id` (pass the root's own id),
    /// newest first, **including** soft-superseded versions — so callers can show the
    /// full history of a belief. Default: empty, for stores without a version chain.
    fn history(&self, _root_id: &str) -> Result<Vec<Memory>> {
        Ok(Vec::new())
    }

    /// Link `memory_id` to its canonical `entities` within `container_tag`, creating
    /// the entities as needed. Idempotent. Default: no-op, for stores without an
    /// entity index.
    fn link_entities(
        &mut self,
        _memory_id: &str,
        _container_tag: &str,
        _entities: &[String],
    ) -> Result<()> {
        Ok(())
    }

    /// Graph recall channel: latest memories in `container_tag` that share canonical
    /// entities with any of `seed_ids` (seeds excluded), ranked by a bounded activation
    /// score — a saturating shared-entity term plus a bonus when the memory is also
    /// directly graph-linked to a seed. The edge bonus is weighted by the edge's
    /// idle-decayed strength ([`crate::dynamics::decayed_edge_strength`]), so a fresh,
    /// often-reinforced relationship activates more than a long-idle one. Best first,
    /// capped at `k`. Default: empty.
    fn graph_search(
        &self,
        _container_tag: &str,
        _seed_ids: &[String],
        _k: usize,
    ) -> Result<Vec<ScoredMemory>> {
        Ok(Vec::new())
    }

    /// Temporal recall channel: latest memories in `container_tag` whose occurred-time
    /// interval overlaps the query `window` `(start, end?)` (Unix seconds), nearest
    /// first (score = distance from the window midpoint, lower is nearer), capped at
    /// `k`. Default: empty, for stores without bi-temporal columns.
    fn temporal_search(
        &self,
        _container_tag: &str,
        _window: (i64, Option<i64>),
        _k: usize,
    ) -> Result<Vec<ScoredMemory>> {
        Ok(Vec::new())
    }
}
