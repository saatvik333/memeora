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
pub mod query;

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

/// How to combine the per-signal ranked lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fusion {
    /// Reciprocal Rank Fusion: sum `1/(rrf_k + rank)` across lists. Rewards items
    /// ranked well in *several* signals — the right default for general recall.
    #[default]
    Rrf,
    /// Round-robin interleave: take each list's #1, then each #2, … RRF's averaging
    /// can bury an item that tops one signal but is absent from the others (a
    /// near-duplicate "twin"); interleave guarantees every signal's best a slot.
    /// Useful when the goal is coverage across signals rather than consensus.
    Interleave,
}

/// Tuning for a [`search`] call.
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Number of results to return.
    pub k: usize,
    /// Which signals to fuse.
    pub mode: SearchMode,
    /// How to combine the per-signal ranked lists.
    pub fusion: Fusion,
    /// RRF constant: larger values flatten the contribution of top ranks.
    /// 60 is the value from the original RRF paper and a robust default.
    pub rrf_k: f32,
    /// Per-signal candidate pool = `k * candidate_multiplier`. Over-fetching gives
    /// RRF more to work with and leaves headroom for expiry filtering.
    pub candidate_multiplier: usize,
    /// Optional token budget: when set, results are filled best-first until the budget
    /// (estimated from content length) would be exceeded, instead of a fixed top-`k`.
    /// `k` still caps the count. `None` ⇒ plain top-`k` (the default).
    pub max_tokens: Option<usize>,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            k: 10,
            mode: SearchMode::Hybrid,
            fusion: Fusion::Rrf,
            rrf_k: 60.0,
            candidate_multiplier: 4,
            max_tokens: None,
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

/// Round-robin interleave of ranked lists: take rank-1 of every list (in list
/// order), then rank-2 of every list, and so on. The first appearance of a memory
/// wins its slot; later duplicates are skipped. Expired memories are dropped.
///
/// Score is a descending rank proxy (`1/slot`) so downstream boost multiplication
/// and the final sort behave; the *order* is what interleave controls, not the
/// magnitude (unlike RRF, these scores aren't a calibrated relevance sum).
fn interleave_fuse(lists: &[Vec<ScoredMemory>], k: usize, now: i64) -> Vec<ScoredMemory> {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<ScoredMemory> = Vec::new();
    let depth = lists.iter().map(Vec::len).max().unwrap_or(0);
    for rank in 0..depth {
        for list in lists {
            let Some(scored) = list.get(rank) else {
                continue;
            };
            if scored.memory.is_expired(now) || !seen.insert(scored.memory.id.clone()) {
                continue;
            }
            let slot = out.len() + 1;
            out.push(ScoredMemory {
                memory: scored.memory.clone(),
                score: 1.0 / slot as f32,
            });
            if out.len() >= k {
                return out;
            }
        }
    }
    out
}

/// Combine ranked lists per the chosen [`Fusion`] strategy.
fn fuse(
    lists: &[Vec<ScoredMemory>],
    params: &SearchParams,
    k: usize,
    now: i64,
) -> Vec<ScoredMemory> {
    match params.fusion {
        Fusion::Rrf => rrf_fuse(lists, params.rrf_k, k, now),
        Fusion::Interleave => interleave_fuse(lists, k, now),
    }
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
    // Strip agent-prepended preamble before it poisons BM25/temporal parsing; the
    // caller already embedded the raw text, so only the lexical/temporal legs use this.
    let query_text = query::sanitize_query(query_text);
    // A date in the query drives the temporal channel and the temporal-proximity boost.
    let window = crate::temporal::parse(query_text, now);
    let pool = params
        .k
        .saturating_mul(params.candidate_multiplier)
        .max(params.k);

    let mut lists: Vec<Vec<ScoredMemory>> = Vec::with_capacity(4);
    if params.mode != SearchMode::Text {
        lists.push(store.knn(container_tag, query_embedding, pool)?);
    }
    if params.mode != SearchMode::Vector {
        lists.push(store.text_search(container_tag, query_text, pool)?);
    }
    // Graph channel (Hybrid only): entity-neighbors of the dense/lexical hits,
    // surfacing related memories that neither signal matched directly.
    if params.mode == SearchMode::Hybrid {
        let seeds: Vec<String> = lists
            .iter()
            .flatten()
            .map(|s| s.memory.id.clone())
            .collect();
        let graph = store.graph_search(container_tag, &seeds, pool)?;
        if !graph.is_empty() {
            lists.push(graph);
        }
        // Temporal channel (Hybrid only, and only when the query named a time):
        // memories whose occurred-interval overlaps the query window, nearest-first.
        if let Some(win) = window {
            let temporal = store.temporal_search(container_tag, win, pool)?;
            if !temporal.is_empty() {
                lists.push(temporal);
            }
        }
    }

    // Fuse to an interim pool, apply bounded multiplicative boosts (recency from
    // stability-aware decay; corroboration from proof_count; temporal proximity to the
    // query window), then fill by token budget (or top-k).
    let mut fused = fuse(&lists, params, pool, now);
    for scored in &mut fused {
        scored.score *= recency_boost(&scored.memory, now)
            * proof_boost(&scored.memory)
            * temporal_boost(&scored.memory, window);
    }
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.memory.id.cmp(&b.memory.id))
    });
    Ok(fill_budget(fused, params.k, params.max_tokens))
}

/// Temporal-proximity boost in `[1-α, 1+α]`: a memory whose occurred-time sits near the
/// query's temporal `window` ranks higher, one far from it lower. Neutral (1.0) when the
/// query named no time or the memory has no occurred-time — a neutral signal can't
/// overpower relevance.
fn temporal_boost(m: &Memory, window: Option<(i64, Option<i64>)>) -> f32 {
    const ALPHA: f32 = 0.2;
    const SCALE_DAYS: f32 = 30.0; // proximity decays to neutral about a month off
    let (Some((w_start, w_end)), Some(m_start)) = (window, m.occurred_start) else {
        return 1.0;
    };
    let w_mid = (w_start + w_end.unwrap_or(w_start)) as f32 / 2.0;
    let m_mid = (m_start + m.occurred_end.unwrap_or(m_start)) as f32 / 2.0;
    let dist_days = (m_mid - w_mid).abs() / 86_400.0;
    let proximity = (1.0 - dist_days / SCALE_DAYS).clamp(0.0, 1.0);
    1.0 + ALPHA * (2.0 * proximity - 1.0)
}

/// Rough token estimate for budget fill (~4 characters per token).
// ponytail: char/4 heuristic; swap for a real tokenizer only if budgets need precision.
fn est_tokens(content: &str) -> usize {
    content.chars().count() / 4 + 1
}

/// Fill results best-first: with a `max_tokens` budget, include memories until adding
/// the next would exceed it (always at least the top one), capped at `k`. Without a
/// budget, a plain top-`k` truncation. Input must already be sorted best-first.
fn fill_budget(
    mut fused: Vec<ScoredMemory>,
    k: usize,
    max_tokens: Option<usize>,
) -> Vec<ScoredMemory> {
    let Some(budget) = max_tokens else {
        fused.truncate(k);
        return fused;
    };
    let mut used = 0usize;
    let mut out = Vec::new();
    for scored in fused {
        if out.len() >= k {
            break;
        }
        let cost = est_tokens(&scored.memory.content);
        if !out.is_empty() && used + cost > budget {
            break;
        }
        used += cost;
        out.push(scored);
    }
    out
}

/// Recency boost in `[1-α, 1+α]` from stability-aware Ebbinghaus decay: a freshly
/// accessed (or high-stability) memory is boosted, a long-idle one discounted.
/// Neutral at a decay ratio of 0.5; a neutral signal can't overpower relevance.
fn recency_boost(m: &Memory, now: i64) -> f32 {
    const ALPHA: f32 = 0.2;
    let ratio = (crate::dynamics::decayed_strength(m, now)
        / m.strength.max(crate::dynamics::STRENGTH_FLOOR))
    .clamp(0.0, 1.0);
    1.0 + ALPHA * (2.0 * ratio - 1.0)
}

/// Corroboration boost in `[1, 1+α]` from proof_count: an independently corroborated
/// belief ranks slightly higher. Neutral (1.0) at proof_count = 1.
fn proof_boost(m: &Memory) -> f32 {
    const ALPHA: f32 = 0.05;
    1.0 + ALPHA * (1.0 - 1.0 / m.proof_count.max(1) as f32)
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
    fn interleave_gives_each_signal_its_best_slot() {
        // "dense_top" tops the dense list only; "text_top" tops text only; "both" is
        // rank-2 in each. RRF would rank "both" first (agreement); interleave instead
        // seats each signal's #1 first, so a twin that tops one signal isn't buried.
        let dense = vec![scored("dense_top", None), scored("both", None)];
        let text = vec![scored("text_top", None), scored("both", None)];
        let out = interleave_fuse(&[dense, text], 10, 0);
        assert_eq!(out[0].memory.id, "dense_top");
        assert_eq!(out[1].memory.id, "text_top");
        assert_eq!(out[2].memory.id, "both");
        assert_eq!(out.len(), 3, "duplicates across lists are seated once");
        assert!(
            out[0].score > out[1].score,
            "score is a descending rank proxy"
        );
    }

    #[test]
    fn interleave_drops_expired_and_caps_at_k() {
        let dense = vec![scored("fresh", None), scored("stale", Some(1))];
        let out = interleave_fuse(&[dense], 10, 1000);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].memory.id, "fresh");
        // k cap.
        let many = vec![scored("a", None), scored("b", None), scored("c", None)];
        assert_eq!(interleave_fuse(&[many], 2, 0).len(), 2);
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

    #[test]
    fn proof_boost_rewards_corroboration() {
        let mut m = Memory::new("i", "c", MemoryKind::Fact, "t", Vec::new());
        m.proof_count = 1;
        assert!((proof_boost(&m) - 1.0).abs() < 1e-6, "neutral at one proof");
        m.proof_count = 8;
        assert!(proof_boost(&m) > 1.0, "corroboration boosts");
    }

    #[test]
    fn recency_boost_discounts_stale() {
        let now = 1_000_000_000;
        let mut fresh = Memory::new("a", "c", MemoryKind::Fact, "t", Vec::new());
        fresh.strength = 2.0;
        fresh.stability = 1.0;
        fresh.last_accessed_at = now;
        let mut stale = fresh.clone();
        stale.last_accessed_at = now - 86_400 * 30;
        assert!(
            recency_boost(&fresh, now) > recency_boost(&stale, now),
            "a long-idle memory is discounted relative to a fresh one"
        );
    }

    fn scored_c(id: &str, content: &str) -> ScoredMemory {
        ScoredMemory {
            memory: Memory::new(id, content, MemoryKind::Fact, "tag", Vec::new()),
            score: 0.0,
        }
    }

    #[test]
    fn fill_budget_caps_by_tokens_and_k() {
        // Each ~8-char content ≈ 3 estimated tokens.
        let items = vec![
            scored_c("a", "xxxxxxxx"),
            scored_c("b", "yyyyyyyy"),
            scored_c("c", "zzzzzzzz"),
        ];
        // Budget 5: take "a" (3), then "b" would reach 6 > 5 → stop at one.
        assert_eq!(fill_budget(items.clone(), 10, Some(5)).len(), 1);
        // No budget → plain top-k truncation.
        assert_eq!(fill_budget(items.clone(), 2, None).len(), 2);
        // k caps the count even with a generous budget.
        assert_eq!(fill_budget(items, 2, Some(10_000)).len(), 2);
    }

    #[test]
    fn temporal_boost_rewards_proximity_and_is_neutral_without_signal() {
        let now = 1_781_000_000;
        let day = 86_400;
        let window = Some((now - day, Some(now))); // ~"yesterday"
        let mut near = Memory::new("n", "c", MemoryKind::Episode, "t", Vec::new());
        near.occurred_start = Some(now - day);
        near.occurred_end = Some(now);
        let mut far = near.clone();
        far.occurred_start = Some(now - 60 * day);
        far.occurred_end = Some(now - 59 * day);
        assert!(temporal_boost(&near, window) > temporal_boost(&far, window));
        assert!(temporal_boost(&near, window) > 1.0, "a near match boosts");
        // No occurred-time, or no query window ⇒ neutral.
        let undated = Memory::new("u", "c", MemoryKind::Fact, "t", Vec::new());
        assert_eq!(temporal_boost(&undated, window), 1.0);
        assert_eq!(temporal_boost(&near, None), 1.0);
    }

    #[test]
    fn temporal_query_surfaces_time_matched_memory() {
        use crate::SqliteStore;
        let mut store = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        let now = now_unix();
        let day = 86_400;
        // Two memories with identical lexical + vector match; only one carries a time.
        let mut dated = Memory::new(
            "dated",
            "the meeting",
            MemoryKind::Episode,
            tag,
            vec![1.0, 0.0],
        );
        dated.occurred_start = Some(now - day);
        dated.occurred_end = Some(now);
        store.upsert(&dated).unwrap();
        store
            .upsert(&Memory::new(
                "undated",
                "the meeting",
                MemoryKind::Episode,
                tag,
                vec![1.0, 0.0],
            ))
            .unwrap();

        // A dated query ("yesterday") promotes the time-matched memory via the temporal
        // channel + proximity boost.
        let hits = search(
            &store,
            tag,
            &[1.0, 0.0],
            "the meeting yesterday",
            &SearchParams::default(),
        )
        .unwrap();
        assert_eq!(hits[0].memory.id, "dated");
    }
}
