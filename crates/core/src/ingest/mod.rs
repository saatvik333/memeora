//! Ingestion: the write path that turns text into stored memories.
//!
//! [`ingest`] ties the pieces together: extract candidates, embed each, then
//! either **reinforce** an existing near-duplicate (strengthen it instead of
//! storing a redundant copy) or **insert** a new memory and link it to its
//! moderately-similar neighbors with `extends` edges. This is the model-free MVP
//! write path; the daemon runs it on its writer thread.
//!
//! Memory ids are a content hash scoped to the container, so re-ingesting the
//! exact same statement is naturally idempotent (it resolves to reinforcement),
//! with no UUID dependency.
//!
//! Deferred (needs the gated NER/NLI stack): `updates` edges and
//! contradiction-based supersession.

use crate::container_tag::sha32;
use crate::embed::EmbeddingProvider;
use crate::error::{Error, Result};
use crate::extract::{Candidate, Extractor};
use crate::store::{EdgeKind, VectorStore};

/// A candidate paired with its document embedding, ready for the DB write path.
///
/// Produced by [`embed_candidates`] (which may run off the daemon's writer thread)
/// and consumed by [`ingest_prepared`] (which holds the writer).
pub type PreparedCandidate = (Candidate, Vec<f32>);

/// Tuning for [`ingest`].
///
/// Distance thresholds assume L2 distance over L2-normalized embeddings and are
/// **provisional** — tune against real data.
#[derive(Debug, Clone)]
pub struct IngestParams {
    /// A candidate within this KNN distance of an existing memory of the **same
    /// kind** reinforces it instead of being inserted (≈ cosine ≥ 0.98).
    pub dedup_max_distance: f32,
    /// Strength added to an existing memory when a near-duplicate reinforces it.
    pub reinforce_delta: f32,
    /// A newly inserted memory gets an `extends` edge to each neighbor within this
    /// distance (and beyond `dedup_max_distance`) — moderate relatedness.
    pub extends_max_distance: f32,
    /// Max `extends` edges to create per new memory.
    pub max_links: usize,
    /// KNN pool size considered for dedup + linking.
    pub link_candidates: usize,
}

impl Default for IngestParams {
    fn default() -> Self {
        IngestParams {
            dedup_max_distance: 0.2,
            reinforce_delta: 0.5,
            extends_max_distance: 0.6,
            max_links: 3,
            link_candidates: 5,
        }
    }
}

/// What [`ingest`] did, by memory id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IngestOutcome {
    /// Ids of newly inserted memories (includes the new version of a supersession).
    pub added: Vec<String>,
    /// Ids of existing memories that were reinforced by a near-duplicate.
    pub reinforced: Vec<String>,
    /// Ids of prior memories soft-superseded by a correcting statement (kept as history).
    pub superseded: Vec<String>,
    /// Number of `extends` edges created.
    pub edges_added: usize,
}

/// Explicit correction cues — the statement *replaces* a prior belief rather than
/// restating it. Deliberately tight: a false positive only supersedes when there is
/// also a same-topic memory to replace, and the prior version is preserved (never
/// hard-deleted), so a wrong call stays recoverable. The opt-in NLI tier (P6) is the
/// upgrade path for contradiction-driven supersession.
// ponytail: keyword floor; replace with NLI when the LLM tier lands.
const CORRECTION_CUES: &[&str] = &[
    "actually",
    "no longer",
    "not anymore",
    "correction:",
    "scratch that",
    "i changed",
    "we changed",
    "used to",
    "i now ",
    "we now ",
];

/// Whether `content` carries an explicit correction cue.
fn is_correction(content: &str) -> bool {
    let lower = content.to_lowercase();
    CORRECTION_CUES.iter().any(|cue| lower.contains(cue))
}

/// Deterministic, container-scoped id for a memory's content (128-bit, content-addressed).
fn content_id(container_tag: &str, content: &str) -> String {
    // NUL separator avoids tag/content boundary collisions.
    sha32(&format!("{container_tag}\u{0}{content}"))
}

/// Extract memories from `text`, embed them, and write them into `store` under
/// `container_tag` — reinforcing near-duplicates rather than duplicating them.
pub fn ingest(
    store: &mut dyn VectorStore,
    embedder: &dyn EmbeddingProvider,
    extractor: &dyn Extractor,
    container_tag: &str,
    text: &str,
    params: &IngestParams,
) -> Result<IngestOutcome> {
    let candidates = extractor.extract(text)?;
    ingest_candidates(store, embedder, container_tag, candidates, params)
}

/// Embed and write already-extracted `candidates` into `store` under
/// `container_tag`, applying the same dedup/reinforce + `extends`-linking logic as
/// [`ingest`]. Used directly when memories come from somewhere other than the
/// heuristic extractor (e.g. an explicit "add this").
pub fn ingest_candidates(
    store: &mut dyn VectorStore,
    embedder: &dyn EmbeddingProvider,
    container_tag: &str,
    candidates: Vec<Candidate>,
    params: &IngestParams,
) -> Result<IngestOutcome> {
    let prepared = embed_candidates(embedder, candidates)?;
    ingest_prepared(store, container_tag, None, prepared, params)
}

/// Embed each candidate's content as a document, pairing it with its vector.
///
/// Split out from the store write path so embedding (CPU-heavy, no DB access) can
/// run off the daemon's single writer thread. Errors if the provider returns the
/// wrong number of vectors (a contract violation) rather than silently inserting
/// an empty embedding.
pub fn embed_candidates(
    embedder: &dyn EmbeddingProvider,
    candidates: Vec<Candidate>,
) -> Result<Vec<PreparedCandidate>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let texts: Vec<&str> = candidates.iter().map(|c| c.content.as_str()).collect();
    let embeddings = embedder.embed_documents(&texts)?;
    if embeddings.len() != candidates.len() {
        return Err(Error::Embedding(format!(
            "embedder returned {} vectors for {} candidates",
            embeddings.len(),
            candidates.len()
        )));
    }
    Ok(candidates.into_iter().zip(embeddings).collect())
}

/// Write already-embedded candidates into `store`, deduping/reinforcing
/// near-duplicates and linking new memories to moderately-similar neighbors.
///
/// The DB-only half of ingestion (no embedding), so it can run alone on the
/// daemon's writer thread after [`embed_candidates`] has run elsewhere.
///
/// `source` identifies *who* is making these observations (an agent/session id), so
/// repeated corroboration from one source can't inflate `proof_count` — only distinct
/// sources raise it. When `None`, each statement's own content id stands in as the
/// source, so distinct restatements still count as independent evidence while exact
/// repeats (routed to reinforce) do not.
// ponytail: per-statement content-id fallback when no source is threaded — anonymous
// re-wordings still read as distinct evidence; thread a real `source` for true
// per-source dedup (the surface adapters that know the session do this later).
pub fn ingest_prepared(
    store: &mut dyn VectorStore,
    container_tag: &str,
    source: Option<&str>,
    prepared: Vec<PreparedCandidate>,
    params: &IngestParams,
) -> Result<IngestOutcome> {
    let mut outcome = IngestOutcome::default();

    for (candidate, embedding) in prepared {
        let id = content_id(container_tag, &candidate.content);
        // Canonical entities for this content — linked on every insert/resurrect so
        // memories about the same thing can be related (graph channel / consolidation).
        let entities = crate::entity::extract_entities(&candidate.content);
        // Evidence inputs for this observation, captured before `candidate` is moved.
        // Default source = this statement's content id (`id`); occurred = valid-time or now.
        let evidence_source = source.map(str::to_string).unwrap_or_else(|| id.clone());
        let quote = candidate.content.clone();
        let occurred_at = candidate
            .occurred_start
            .unwrap_or_else(crate::store::now_unix);

        // Exact re-ingest by content id, handled before the KNN heuristic (which can
        // miss the literal-same row).
        if let Some(existing) = store.get(&id)? {
            if existing.is_latest {
                // Live row: reinforce it (going through `upsert` would reset
                // strength/created_at).
                store.reinforce(&id, params.reinforce_delta)?;
                outcome.reinforced.push(id);
            } else {
                // The content was retired (`is_latest = 0`) — but plain forgetting and
                // supersession look identical at that level, and only the former may
                // resurrect. If this row's version chain has an active successor,
                // restating the old belief is corroborating history, not a revert:
                // resurrecting would put a second `is_latest` head on the chain,
                // returning both the retracted belief and its correction from recall.
                let root = existing.root_id.clone().unwrap_or_else(|| id.clone());
                let superseded = store
                    .history(&root)?
                    .iter()
                    .any(|m| m.is_latest && m.id != id);
                if superseded {
                    store.record_evidence(&id, &evidence_source, &quote, occurred_at)?;
                    outcome.reinforced.push(id);
                } else {
                    // Truly forgotten: re-stating it must *resurrect* it, not reinforce
                    // an invisible row: `upsert` restores `is_latest = 1` and re-inserts
                    // the vec + FTS rows (and preserves existing graph edges). Resurrect
                    // this exact id rather than falling through to the KNN path, which
                    // could reinforce a *different* neighbor instead of bringing this
                    // content back — and keep its original chain position rather than
                    // resetting lineage.
                    let mut memory = candidate.into_memory(id.clone(), container_tag, embedding);
                    memory.parent_id = existing.parent_id.clone();
                    memory.root_id = existing.root_id.clone();
                    store.upsert(&memory)?;
                    store.link_entities(&id, container_tag, &entities)?;
                    // `into_memory` reset proof_count to 1; recompute it from the evidence
                    // rows that survived the forget so prior corroboration isn't lost.
                    store.record_evidence(&id, &evidence_source, &quote, occurred_at)?;
                    outcome.added.push(id);
                }
            }
            continue;
        }

        // A correcting statement replaces the same-topic belief instead of adding to it.
        let correction = is_correction(&candidate.content);

        // One KNN lookup serves dedup, linking, and supersede targeting. Scope the
        // immutable borrow so it ends before the mutable writes below; carry only owned
        // decisions out.
        let (duplicate_id, link_targets, supersede_target) = {
            let neighbors = store.knn(container_tag, &embedding, params.link_candidates.max(1))?;
            // Reinforce the nearest same-kind near-duplicate (scan, don't just check
            // rank 1 — a closer different-kind neighbor must not shadow it).
            let duplicate_id = neighbors
                .iter()
                .find(|n| n.score <= params.dedup_max_distance && n.memory.kind == candidate.kind)
                .map(|n| n.memory.id.clone());
            // On a correction, the closest same-kind memory within the topic
            // neighbourhood (≤ extends distance) is the belief being corrected.
            let supersede_target = correction
                .then(|| {
                    neighbors
                        .iter()
                        .find(|n| {
                            n.memory.kind == candidate.kind
                                && n.score <= params.extends_max_distance
                        })
                        .map(|n| n.memory.id.clone())
                })
                .flatten();
            let link_targets: Vec<String> = if duplicate_id.is_some() || supersede_target.is_some()
            {
                Vec::new()
            } else {
                // `extends` links go to neighbors beyond the dedup window but still
                // moderately similar (matches the doc on `extends_max_distance`).
                neighbors
                    .iter()
                    .filter(|n| {
                        n.score > params.dedup_max_distance
                            && n.score <= params.extends_max_distance
                    })
                    .take(params.max_links)
                    .map(|n| n.memory.id.clone())
                    .collect()
            };
            (duplicate_id, link_targets, supersede_target)
        };

        // A correction supersedes its topic neighbour: the new statement becomes the
        // current version, the prior one is kept as history (never hard-deleted). This
        // takes precedence over corroboration — the cue says "replace", not "confirm".
        if let Some(old) = supersede_target {
            let memory = candidate.into_memory(id.clone(), container_tag, embedding);
            if store.supersede(&old, &memory)? {
                outcome.superseded.push(old);
            } else {
                // Target vanished mid-batch: insert the memory we built rather than
                // dropping the statement.
                store.upsert(&memory)?;
            }
            store.link_entities(&id, container_tag, &entities)?;
            store.record_evidence(&id, &evidence_source, &quote, occurred_at)?;
            outcome.added.push(id);
            continue;
        }

        if let Some(dup) = duplicate_id {
            // A distinct near-duplicate is independent corroboration: reinforce strength
            // and record it as evidence. proof_count grows only if this source is new to
            // the belief — recorded by `record_evidence`, not a blind counter bump.
            store.reinforce(&dup, params.reinforce_delta)?;
            store.record_evidence(&dup, &evidence_source, &quote, occurred_at)?;
            outcome.reinforced.push(dup);
            continue;
        }

        let memory = candidate.into_memory(id.clone(), container_tag, embedding);
        store.upsert(&memory)?;
        store.link_entities(&id, container_tag, &entities)?;
        // Record the originating observation so proof_count starts from a real source
        // set (and a later corroboration adds to it rather than replacing it).
        store.record_evidence(&id, &evidence_source, &quote, occurred_at)?;
        // Link the new memory to its moderately-similar neighbors.
        for target in &link_targets {
            store.add_edge(&id, target, EdgeKind::Extends)?;
            outcome.edges_added += 1;
        }
        outcome.added.push(id);
    }

    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SqliteStore;
    use crate::embed::EmbeddingSpace;
    use crate::extract::HeuristicExtractor;
    use std::collections::HashMap;

    /// Embedder that returns a prescribed vector per content (so tests control
    /// distances exactly); unknown content maps to a distinct fallback.
    struct MapEmbedder {
        space: EmbeddingSpace,
        map: HashMap<String, Vec<f32>>,
    }

    impl MapEmbedder {
        fn new(pairs: &[(&str, Vec<f32>)]) -> Self {
            MapEmbedder {
                space: EmbeddingSpace::new("mock", "map", 3),
                map: pairs
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect(),
            }
        }
    }

    impl EmbeddingProvider for MapEmbedder {
        fn space(&self) -> &EmbeddingSpace {
            &self.space
        }

        fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| self.map.get(*t).cloned().unwrap_or(vec![0.0, 0.0, 1.0]))
                .collect())
        }
    }

    #[test]
    fn inserts_new_then_reinforces_duplicate() {
        let extractor = HeuristicExtractor::default();
        let embedder =
            MapEmbedder::new(&[("I prefer dark mode in my editor", vec![1.0, 0.0, 0.0])]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let tag = "t";
        let text = "I prefer dark mode in my editor";

        let first = ingest(
            &mut store,
            &embedder,
            &extractor,
            tag,
            text,
            &IngestParams::default(),
        )
        .unwrap();
        assert_eq!(first.added.len(), 1);
        assert_eq!(first.reinforced.len(), 0);
        assert_eq!(store.count(tag).unwrap(), 1);

        // Same statement again: identical embedding (distance 0) → reinforce, no insert.
        let second = ingest(
            &mut store,
            &embedder,
            &extractor,
            tag,
            text,
            &IngestParams::default(),
        )
        .unwrap();
        assert_eq!(second.added.len(), 0);
        assert_eq!(second.reinforced.len(), 1);
        assert_eq!(store.count(tag).unwrap(), 1);
        assert!(store.get(&first.added[0]).unwrap().unwrap().strength > 1.0);
    }

    #[test]
    fn exact_reingest_reinforces_via_content_id_not_destructive_upsert() {
        // Even with KNN dedup effectively disabled, re-ingesting identical content
        // must reinforce the existing row (preserving/raising strength), never fall
        // through to a destructive upsert that resets strength.
        let extractor = HeuristicExtractor::default();
        let embedder =
            MapEmbedder::new(&[("I prefer dark mode in my editor", vec![1.0, 0.0, 0.0])]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let params = IngestParams {
            dedup_max_distance: -1.0, // KNN near-dup branch can never trigger
            ..IngestParams::default()
        };
        let tag = "t";
        let text = "I prefer dark mode in my editor";

        let first = ingest(&mut store, &embedder, &extractor, tag, text, &params).unwrap();
        assert_eq!(first.added.len(), 1);
        let strength_before = store.get(&first.added[0]).unwrap().unwrap().strength;

        let second = ingest(&mut store, &embedder, &extractor, tag, text, &params).unwrap();
        assert_eq!(second.added.len(), 0, "exact re-ingest must not insert");
        assert_eq!(second.reinforced.len(), 1);
        assert_eq!(store.count(tag).unwrap(), 1);
        let strength_after = store.get(&first.added[0]).unwrap().unwrap().strength;
        assert!(
            strength_after > strength_before,
            "strength must increase on re-ingest, not reset"
        );
    }

    #[test]
    fn reingesting_forgotten_content_resurrects_it() {
        // ingest → forget → re-ingest identical content must bring it back, not
        // silently reinforce the invisible (is_latest=0) row.
        let extractor = HeuristicExtractor::default();
        let embedder =
            MapEmbedder::new(&[("I prefer dark mode in my editor", vec![1.0, 0.0, 0.0])]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let tag = "t";
        let text = "I prefer dark mode in my editor";

        let first = ingest(
            &mut store,
            &embedder,
            &extractor,
            tag,
            text,
            &IngestParams::default(),
        )
        .unwrap();
        let id = first.added[0].clone();

        store.forget(&id).unwrap();
        assert_eq!(store.count(tag).unwrap(), 0, "forgotten → invisible");

        // Re-state the same content: it must reappear (counted as added, not reinforced).
        let again = ingest(
            &mut store,
            &embedder,
            &extractor,
            tag,
            text,
            &IngestParams::default(),
        )
        .unwrap();
        assert_eq!(again.added, vec![id.clone()], "resurrected, not reinforced");
        assert_eq!(again.reinforced.len(), 0);
        // Visible again across every active read path.
        assert_eq!(store.count(tag).unwrap(), 1);
        assert!(store.get(&id).unwrap().unwrap().is_latest);
        assert_eq!(store.knn(tag, &[1.0, 0.0, 0.0], 5).unwrap().len(), 1);
        assert_eq!(store.list_latest(tag, 5).unwrap().len(), 1);
    }

    #[test]
    fn distinct_statements_are_all_added() {
        let extractor = HeuristicExtractor::default();
        let embedder = MapEmbedder::new(&[
            ("I prefer dark mode", vec![1.0, 0.0, 0.0]),
            ("We use SQLite for storage", vec![0.0, 1.0, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let out = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "I prefer dark mode. We use SQLite for storage.",
            &IngestParams::default(),
        )
        .unwrap();
        assert_eq!(out.added.len(), 2);
        assert_eq!(store.count("t").unwrap(), 2);
    }

    #[test]
    fn near_duplicate_of_different_kind_is_not_merged() {
        // Same embedding, but one is a preference and one is a fact → kept separate.
        let extractor = HeuristicExtractor::default();
        let embedder = MapEmbedder::new(&[
            ("I prefer dark mode", vec![1.0, 0.0, 0.0]),
            ("We use dark mode", vec![1.0, 0.0, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let params = IngestParams::default();

        ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "I prefer dark mode",
            &params,
        )
        .unwrap();
        // "We use dark mode" classifies as Fact (we use…), same vector as the preference.
        let out = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "We use dark mode",
            &params,
        )
        .unwrap();
        assert_eq!(
            out.added.len(),
            1,
            "different kind should insert, not reinforce"
        );
        assert_eq!(store.count("t").unwrap(), 2);
    }

    #[test]
    fn links_new_memory_to_related_neighbor() {
        let extractor = HeuristicExtractor::default();
        let embedder = MapEmbedder::new(&[
            ("I prefer dark mode", vec![1.0, 0.0, 0.0]),
            ("I prefer light themes sometimes", vec![0.0, 1.0, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        // dedup tiny (nothing merges) + extends huge (any neighbor links).
        let params = IngestParams {
            dedup_max_distance: 0.001,
            extends_max_distance: 100.0,
            ..IngestParams::default()
        };

        let a = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "I prefer dark mode",
            &params,
        )
        .unwrap();
        let b = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "I prefer light themes sometimes",
            &params,
        )
        .unwrap();

        assert_eq!(b.added.len(), 1);
        assert_eq!(b.edges_added, 1);
        let edges = store.edges_from(&b.added[0]).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].to_id, a.added[0]);
        assert_eq!(edges[0].kind, crate::store::EdgeKind::Extends);
    }

    #[test]
    fn no_links_when_beyond_extends_distance() {
        let extractor = HeuristicExtractor::default();
        let embedder = MapEmbedder::new(&[
            ("I prefer dark mode", vec![1.0, 0.0, 0.0]),
            ("I prefer light themes sometimes", vec![0.0, 1.0, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let params = IngestParams {
            dedup_max_distance: 0.001,
            extends_max_distance: 0.0,
            ..IngestParams::default()
        };

        ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "I prefer dark mode",
            &params,
        )
        .unwrap();
        let b = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "I prefer light themes sometimes",
            &params,
        )
        .unwrap();
        assert_eq!(b.edges_added, 0);
    }

    #[test]
    fn empty_text_writes_nothing() {
        let extractor = HeuristicExtractor::default();
        let embedder = MapEmbedder::new(&[]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let out = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "how are you?",
            &IngestParams::default(),
        )
        .unwrap();
        assert_eq!(out, IngestOutcome::default());
        assert_eq!(store.count("t").unwrap(), 0);
    }

    #[test]
    fn batch_ingest_rolls_back_on_mid_batch_failure() {
        // A mid-batch failure (2nd candidate carries a wrong-dim embedding) must roll
        // the whole batch back through `SqliteStore::transaction` — the 1st candidate
        // must NOT be left committed (which a retry would then double-reinforce).
        use crate::store::MemoryKind;
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let tag = "t";
        let cand = |content: &str, kind| Candidate {
            content: content.to_string(),
            kind,
            expires_at: None,
            occurred_start: None,
            occurred_end: None,
            confidence: 1.0,
        };
        let good = (
            cand("I prefer dark mode", MemoryKind::Preference),
            vec![1.0, 0.0, 0.0],
        );
        let bad = (cand("We use SQLite", MemoryKind::Fact), vec![1.0, 0.0]); // dim 2 ≠ 3

        let params = IngestParams::default();
        let result = store.transaction(|s| ingest_prepared(s, tag, None, vec![good, bad], &params));

        assert!(result.is_err(), "a wrong-dim candidate must fail the batch");
        assert_eq!(
            store.count(tag).unwrap(),
            0,
            "partial batch must roll back — no memory left committed"
        );
    }

    #[test]
    fn ingest_links_shared_entities() {
        // Two memories mentioning the same code identifier become entity-linked.
        let extractor = HeuristicExtractor::default();
        let embedder = MapEmbedder::new(&[
            ("I prefer SqliteStore", vec![1.0, 0.0, 0.0]),
            ("We use SqliteStore", vec![0.0, 1.0, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let p = IngestParams::default();

        let a = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "I prefer SqliteStore",
            &p,
        )
        .unwrap();
        let b = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "We use SqliteStore",
            &p,
        )
        .unwrap();

        assert_eq!(a.added.len(), 1);
        assert_eq!(b.added.len(), 1);
        assert_eq!(
            store
                .graph_search("t", &[a.added[0].clone()], 10)
                .unwrap()
                .into_iter()
                .map(|h| h.memory.id)
                .collect::<Vec<_>>(),
            vec![b.added[0].clone()],
            "memories sharing the SqliteStore entity must resolve as related"
        );
    }

    #[test]
    fn near_duplicate_corroborates_proof_count() {
        // A distinct statement of the same belief (same kind, near in embedding)
        // reinforces AND corroborates — proof_count grows.
        let extractor = HeuristicExtractor::default();
        let embedder = MapEmbedder::new(&[
            ("I prefer dark mode", vec![1.0, 0.0, 0.0]),
            ("I like dark mode", vec![1.0, 0.0, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let p = IngestParams::default();

        let a = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "I prefer dark mode",
            &p,
        )
        .unwrap();
        assert_eq!(store.get(&a.added[0]).unwrap().unwrap().proof_count, 1);

        let b = ingest(
            &mut store,
            &embedder,
            &extractor,
            "t",
            "I like dark mode",
            &p,
        )
        .unwrap();
        assert_eq!(
            b.added.len(),
            0,
            "near-dup of same kind reinforces, not inserts"
        );
        assert_eq!(b.reinforced.len(), 1);
        assert_eq!(
            store.get(&a.added[0]).unwrap().unwrap().proof_count,
            2,
            "corroborated by a distinct statement"
        );
    }

    #[test]
    fn exact_reingest_does_not_inflate_proof_count() {
        // The SAME statement restated is repetition by one source, not new evidence.
        let extractor = HeuristicExtractor::default();
        let embedder = MapEmbedder::new(&[("I prefer dark mode", vec![1.0, 0.0, 0.0])]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let p = IngestParams::default();
        let text = "I prefer dark mode";

        let a = ingest(&mut store, &embedder, &extractor, "t", text, &p).unwrap();
        ingest(&mut store, &embedder, &extractor, "t", text, &p).unwrap();
        assert_eq!(
            store.get(&a.added[0]).unwrap().unwrap().proof_count,
            1,
            "exact re-ingest must not inflate proof_count"
        );
    }

    #[test]
    fn source_threading_dedups_corroboration_by_source() {
        // Two distinct restatements of one belief: from the SAME source they are one
        // observation (proof_count stays 1, set-union); from DIFFERENT sources they are
        // independent corroboration (proof_count 2). Same vector ⇒ the second near-dups
        // the first ⇒ the corroborate path records evidence under the threaded source.
        let embedder = MapEmbedder::new(&[
            ("I use Postgres", vec![1.0, 0.0, 0.0]),
            ("We run Postgres", vec![1.0, 0.0, 0.0]),
        ]);
        let p = IngestParams::default();
        let run = |src_a: &str, src_b: &str| {
            let mut store = SqliteStore::open_in_memory(3).unwrap();
            let a = ingest_prepared(
                &mut store,
                "t",
                Some(src_a),
                embed_candidates(&embedder, vec![fact("I use Postgres")]).unwrap(),
                &p,
            )
            .unwrap();
            ingest_prepared(
                &mut store,
                "t",
                Some(src_b),
                embed_candidates(&embedder, vec![fact("We run Postgres")]).unwrap(),
                &p,
            )
            .unwrap();
            store.get(&a.added[0]).unwrap().unwrap().proof_count
        };

        assert_eq!(
            run("agent-1", "agent-1"),
            1,
            "one source can't inflate proof"
        );
        assert_eq!(run("agent-1", "agent-2"), 2, "distinct sources corroborate");
    }

    fn fact(content: &str) -> Candidate {
        Candidate {
            content: content.to_string(),
            kind: crate::store::MemoryKind::Fact,
            expires_at: None,
            occurred_start: None,
            occurred_end: None,
            confidence: 1.0,
        }
    }

    #[test]
    fn correction_cue_supersedes_topical_neighbor() {
        // A statement with a correction cue, near (same topic, ≤ extends) a prior
        // same-kind belief, supersedes it: new current version, old kept as history.
        let correction = "Actually I no longer use MySQL, switching to Postgres";
        let embedder = MapEmbedder::new(&[
            ("I use MySQL", vec![1.0, 0.0, 0.0]),
            (correction, vec![0.9, 0.4, 0.0]), // ~0.41 from the first: topic, not dup
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let tag = "t";
        let p = IngestParams::default();

        let a =
            ingest_candidates(&mut store, &embedder, tag, vec![fact("I use MySQL")], &p).unwrap();
        assert_eq!(a.added.len(), 1);

        let b = ingest_candidates(&mut store, &embedder, tag, vec![fact(correction)], &p).unwrap();
        assert_eq!(
            b.added.len(),
            1,
            "the correction is the new current version"
        );
        assert_eq!(b.superseded, a.added, "and supersedes the prior belief");

        // Only the new version is active; the old one survives as history.
        assert_eq!(store.count(tag).unwrap(), 1);
        assert_eq!(store.list_latest(tag, 10).unwrap()[0].id, b.added[0]);
        assert!(!store.get(&a.added[0]).unwrap().unwrap().is_latest);
        // An `updates` edge records the supersession, and the lineage is retrievable.
        let edges = store.edges_from(&b.added[0]).unwrap();
        assert_eq!(
            (edges.len(), edges[0].kind, edges[0].to_id.as_str()),
            (1, EdgeKind::Updates, a.added[0].as_str())
        );
        assert_eq!(store.history(&a.added[0]).unwrap().len(), 2);
    }

    #[test]
    fn reingesting_superseded_content_does_not_resurrect_it() {
        // Supersession retires the old belief; restating the old content afterwards
        // is corroborating history, not a revert — it must not put a second
        // `is_latest` head on the chain (recall would then return the retracted
        // belief *and* its correction as both current).
        let correction = "Actually I no longer use MySQL, switching to Postgres";
        let embedder = MapEmbedder::new(&[
            ("I use MySQL", vec![1.0, 0.0, 0.0]),
            (correction, vec![0.9, 0.4, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let tag = "t";
        let p = IngestParams::default();

        let a =
            ingest_candidates(&mut store, &embedder, tag, vec![fact("I use MySQL")], &p).unwrap();
        let b = ingest_candidates(&mut store, &embedder, tag, vec![fact(correction)], &p).unwrap();
        assert_eq!(b.superseded, a.added);

        // Restate the superseded content verbatim.
        let c =
            ingest_candidates(&mut store, &embedder, tag, vec![fact("I use MySQL")], &p).unwrap();
        assert!(c.added.is_empty(), "must not resurrect a superseded belief");
        assert_eq!(c.reinforced, a.added, "recorded against the historical row");

        // Exactly one active head — the correction — and the old version stays
        // history with its lineage intact.
        assert_eq!(store.count(tag).unwrap(), 1);
        assert_eq!(store.list_latest(tag, 10).unwrap()[0].id, b.added[0]);
        let old = store.get(&a.added[0]).unwrap().unwrap();
        assert!(!old.is_latest);
        assert_eq!(
            store.get(&b.added[0]).unwrap().unwrap().parent_id,
            a.added.first().cloned()
        );
    }

    #[test]
    fn reingesting_a_forgotten_chain_head_resurrects_it_with_lineage() {
        // When the *head* of a chain is forgotten (no active successor), restating
        // its content resurrects it — keeping its chain position rather than
        // resetting parent/root and orphaning the lineage.
        let correction = "Actually I no longer use MySQL, switching to Postgres";
        let embedder = MapEmbedder::new(&[
            ("I use MySQL", vec![1.0, 0.0, 0.0]),
            (correction, vec![0.9, 0.4, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let tag = "t";
        let p = IngestParams::default();

        let a =
            ingest_candidates(&mut store, &embedder, tag, vec![fact("I use MySQL")], &p).unwrap();
        let b = ingest_candidates(&mut store, &embedder, tag, vec![fact(correction)], &p).unwrap();
        store.forget(&b.added[0]).unwrap();

        let again =
            ingest_candidates(&mut store, &embedder, tag, vec![fact(correction)], &p).unwrap();
        assert_eq!(again.added, b.added, "resurrected, not reinforced");
        let head = store.get(&b.added[0]).unwrap().unwrap();
        assert!(head.is_latest);
        assert_eq!(
            head.parent_id,
            a.added.first().cloned(),
            "lineage preserved"
        );
        assert_eq!(head.root_id, a.added.first().cloned());
        assert_eq!(store.count(tag).unwrap(), 1);
    }

    #[test]
    fn correction_without_topical_neighbor_just_inserts() {
        // A correction cue with no same-topic memory to replace must not supersede a
        // stranger — it's simply a new memory.
        let correction = "Actually I switched to Postgres";
        let embedder = MapEmbedder::new(&[
            ("I prefer dark mode", vec![1.0, 0.0, 0.0]),
            (correction, vec![0.0, 0.0, 1.0]), // far from the first
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        let tag = "t";
        let p = IngestParams::default();

        ingest_candidates(
            &mut store,
            &embedder,
            tag,
            vec![fact("I prefer dark mode")],
            &p,
        )
        .unwrap();
        let b = ingest_candidates(&mut store, &embedder, tag, vec![fact(correction)], &p).unwrap();
        assert_eq!(b.added.len(), 1);
        assert!(b.superseded.is_empty(), "no neighbour ⇒ no supersession");
        assert_eq!(store.count(tag).unwrap(), 2);
    }
}
