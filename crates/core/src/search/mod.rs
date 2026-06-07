//! Hybrid retrieval: fuse dense (vector KNN) and lexical (BM25) results.
//!
//! Dense and lexical search each surface different relevant memories — vectors
//! catch paraphrase, BM25 catches exact terms and rare tokens. [`search`] runs
//! both against a [`VectorStore`] and fuses their rankings with **Reciprocal Rank
//! Fusion (RRF)**, which combines ranked lists without needing the two score
//! scales to be comparable. Expired memories are dropped before fusion.
//!
//! Returned scores are RRF relevance: **higher is more relevant** (unlike the raw
//! distance/BM25 scores on [`VectorStore`] reads, where lower is better).

#[cfg(feature = "fastembed")]
pub mod fastembed;

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::Result;
use crate::store::{Memory, ScoredMemory, VectorStore, now_unix};

/// Which retrieval signals to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    /// Fuse dense vector KNN and lexical BM25 (the default, best general recall).
    #[default]
    Hybrid,
    /// Dense vector KNN only.
    Vector,
    /// Lexical BM25 only.
    Text,
}

/// Tuning for a [`search`] call.
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Number of results to return.
    pub k: usize,
    /// Which signals to fuse.
    pub mode: SearchMode,
    /// RRF constant: larger values flatten the contribution of top ranks.
    /// 60 is the value from the original RRF paper and a robust default.
    pub rrf_k: f32,
    /// Per-signal candidate pool = `k * candidate_multiplier`. Over-fetching gives
    /// RRF more to work with and leaves headroom for expiry filtering.
    pub candidate_multiplier: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            k: 10,
            mode: SearchMode::Hybrid,
            rrf_k: 60.0,
            candidate_multiplier: 4,
        }
    }
}

/// Fuse one or more ranked lists (best-first) into a single ranking via RRF.
///
/// Each surviving (non-expired) memory's score is `Σ 1/(rrf_k + rank)` over the
/// lists it appears in (rank is 1-based among survivors). A memory ranked well in
/// *both* signals beats one ranked top in only one — the core value of fusion.
fn rrf_fuse(lists: &[Vec<ScoredMemory>], rrf_k: f32, k: usize, now: i64) -> Vec<ScoredMemory> {
    let mut acc: HashMap<String, (Memory, f32)> = HashMap::new();
    for list in lists {
        let mut rank = 0usize;
        for scored in list {
            if scored.memory.is_expired(now) {
                continue;
            }
            rank += 1;
            let contribution = 1.0 / (rrf_k + rank as f32);
            acc.entry(scored.memory.id.clone())
                .and_modify(|entry| entry.1 += contribution)
                .or_insert_with(|| (scored.memory.clone(), contribution));
        }
    }

    let mut fused: Vec<ScoredMemory> = acc
        .into_values()
        .map(|(memory, score)| ScoredMemory { memory, score })
        .collect();
    // Highest RRF score first; break ties by id for deterministic ordering.
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.memory.id.cmp(&b.memory.id))
    });
    fused.truncate(k);
    fused
}

/// Search `store` within `container_tag`, fusing the signals selected by `params`.
///
/// `query_embedding` drives the dense KNN; `query_text` drives BM25. The caller
/// embeds the query (decoupling search from any particular embedding backend).
/// Results are ordered most-relevant-first by RRF score.
pub fn search(
    store: &dyn VectorStore,
    container_tag: &str,
    query_embedding: &[f32],
    query_text: &str,
    params: &SearchParams,
) -> Result<Vec<ScoredMemory>> {
    let now = now_unix();
    let pool = params
        .k
        .saturating_mul(params.candidate_multiplier)
        .max(params.k);

    let mut lists: Vec<Vec<ScoredMemory>> = Vec::with_capacity(2);
    if params.mode != SearchMode::Text {
        lists.push(store.knn(container_tag, query_embedding, pool)?);
    }
    if params.mode != SearchMode::Vector {
        lists.push(store.text_search(container_tag, query_text, pool)?);
    }

    Ok(rrf_fuse(&lists, params.rrf_k, params.k, now))
}

/// A reranked document: its position in the input slice and its relevance score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankHit {
    /// Index into the documents passed to [`Reranker::rerank`].
    pub index: usize,
    /// Cross-encoder relevance score (higher is more relevant).
    pub score: f32,
}

/// A cross-encoder that re-scores candidates against the query jointly.
///
/// Reranking is an optional quality upgrade applied after [`search`]: it judges
/// query and document *together*, catching relevance that bi-encoder embeddings
/// miss, at higher cost — so it runs only over the fused top candidates.
pub trait Reranker: Send + Sync {
    /// Score `docs` against `query`, returning at most `top_k` hits, best first.
    fn rerank(&self, query: &str, docs: &[&str], top_k: usize) -> Result<Vec<RerankHit>>;
}

/// Reorder `candidates` by a [`Reranker`], keeping the top `top_k`.
///
/// The returned memories carry the reranker's score (higher is more relevant).
pub fn rerank_memories(
    reranker: &dyn Reranker,
    query: &str,
    candidates: &[ScoredMemory],
    top_k: usize,
) -> Result<Vec<ScoredMemory>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let docs: Vec<&str> = candidates
        .iter()
        .map(|c| c.memory.content.as_str())
        .collect();
    let hits = reranker.rerank(query, &docs, top_k)?;
    Ok(hits
        .into_iter()
        .filter_map(|hit| {
            candidates.get(hit.index).map(|c| ScoredMemory {
                memory: c.memory.clone(),
                score: hit.score,
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryKind;

    fn scored(id: &str, expires_at: Option<i64>) -> ScoredMemory {
        let mut m = Memory::new(id, id, MemoryKind::Fact, "tag", Vec::new());
        m.expires_at = expires_at;
        ScoredMemory {
            memory: m,
            score: 0.0,
        }
    }

    #[test]
    fn rrf_rewards_agreement_across_signals() {
        // "both" is rank 2 in each list; "dense_top" is rank 1 in dense only;
        // "text_top" is rank 1 in text only. RRF should rank "both" first because
        // it scores in two lists, even though it tops neither.
        let dense = vec![scored("dense_top", None), scored("both", None)];
        let text = vec![scored("text_top", None), scored("both", None)];

        let fused = rrf_fuse(&[dense, text], 60.0, 10, 0);
        assert_eq!(fused[0].memory.id, "both");
        assert!(fused[0].score > fused[1].score);
        // The two single-signal hits tie and trail the agreed-upon one.
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn expired_memories_are_dropped() {
        // expires_at in the past (1) relative to now (1000) → excluded.
        let dense = vec![scored("fresh", None), scored("stale", Some(1))];
        let fused = rrf_fuse(&[dense], 60.0, 10, 1000);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].memory.id, "fresh");
    }

    #[test]
    fn future_expiry_is_kept() {
        let dense = vec![scored("keep", Some(5000))];
        let fused = rrf_fuse(&[dense], 60.0, 10, 1000);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].memory.id, "keep");
    }

    #[test]
    fn truncates_to_k() {
        let dense = vec![scored("a", None), scored("b", None), scored("c", None)];
        let fused = rrf_fuse(&[dense], 60.0, 2, 0);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn search_over_store_fuses_dense_and_lexical() {
        use crate::SqliteStore;

        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let tag = "t";
        // "rust" matches the text query lexically and sits near the query vector.
        store
            .upsert(&Memory::new(
                "m1",
                "the user prefers rust",
                MemoryKind::Preference,
                tag,
                vec![1.0, 0.0, 0.0],
            ))
            .unwrap();
        store
            .upsert(&Memory::new(
                "m2",
                "deploy with docker compose",
                MemoryKind::Fact,
                tag,
                vec![0.0, 1.0, 0.0],
            ))
            .unwrap();

        let params = SearchParams::default();
        let hits = search(&store, tag, &[0.9, 0.1, 0.0], "rust", &params).unwrap();

        assert_eq!(hits[0].memory.id, "m1");
        // RRF scores are positive relevance (higher = better).
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn vector_only_mode_ignores_text_signal() {
        use crate::SqliteStore;

        let mut store = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        store
            .upsert(&Memory::new(
                "m1",
                "alpha",
                MemoryKind::Fact,
                tag,
                vec![1.0, 0.0],
            ))
            .unwrap();

        let params = SearchParams {
            mode: SearchMode::Vector,
            ..SearchParams::default()
        };
        // Text query that matches nothing; vector still finds m1.
        let hits = search(&store, tag, &[1.0, 0.0], "nonexistentterm", &params).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.id, "m1");
    }
}
