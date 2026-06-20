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
    /// Ids of newly inserted memories.
    pub added: Vec<String>,
    /// Ids of existing memories that were reinforced by a near-duplicate.
    pub reinforced: Vec<String>,
    /// Number of `extends` edges created.
    pub edges_added: usize,
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
    ingest_prepared(store, container_tag, prepared, params)
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
pub fn ingest_prepared(
    store: &mut dyn VectorStore,
    container_tag: &str,
    prepared: Vec<PreparedCandidate>,
    params: &IngestParams,
) -> Result<IngestOutcome> {
    let mut outcome = IngestOutcome::default();

    for (candidate, embedding) in prepared {
        let id = content_id(container_tag, &candidate.content);
        // Canonical entities for this content — linked on every insert/resurrect so
        // memories about the same thing can be related (graph channel / consolidation).
        let entities = crate::entity::extract_entities(&candidate.content);

        // Exact re-ingest by content id, handled before the KNN heuristic (which can
        // miss the literal-same row).
        if let Some(existing) = store.get(&id)? {
            if existing.is_latest {
                // Live row: reinforce it (going through `upsert` would reset
                // strength/created_at).
                store.reinforce(&id, params.reinforce_delta)?;
                outcome.reinforced.push(id);
            } else {
                // The content was forgotten (`is_latest = 0`, vec row removed).
                // Re-stating it must *resurrect* it, not reinforce an invisible row:
                // `upsert` restores `is_latest = 1` and re-inserts the vec + FTS rows
                // (and preserves existing graph edges). Resurrect this exact id rather
                // than falling through to the KNN path, which could reinforce a
                // *different* neighbor instead of bringing this content back.
                let memory = candidate.into_memory(id.clone(), container_tag, embedding);
                store.upsert(&memory)?;
                store.link_entities(&id, container_tag, &entities)?;
                outcome.added.push(id);
            }
            continue;
        }

        // One KNN lookup serves both dedup and linking. Scope the immutable borrow
        // so it ends before the mutable writes below; carry only owned decisions out.
        let (duplicate_id, link_targets) = {
            let neighbors = store.knn(container_tag, &embedding, params.link_candidates.max(1))?;
            // Reinforce the nearest same-kind near-duplicate (scan, don't just check
            // rank 1 — a closer different-kind neighbor must not shadow it).
            let duplicate_id = neighbors
                .iter()
                .find(|n| n.score <= params.dedup_max_distance && n.memory.kind == candidate.kind)
                .map(|n| n.memory.id.clone());
            let link_targets: Vec<String> = if duplicate_id.is_some() {
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
            (duplicate_id, link_targets)
        };

        if let Some(dup) = duplicate_id {
            // A distinct near-duplicate is independent corroboration, not mere
            // repetition: bump proof_count alongside strength (consolidation).
            store.corroborate(&dup, params.reinforce_delta)?;
            outcome.reinforced.push(dup);
            continue;
        }

        let memory = candidate.into_memory(id.clone(), container_tag, embedding);
        store.upsert(&memory)?;
        store.link_entities(&id, container_tag, &entities)?;
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
            confidence: 1.0,
        };
        let good = (
            cand("I prefer dark mode", MemoryKind::Preference),
            vec![1.0, 0.0, 0.0],
        );
        let bad = (cand("We use SQLite", MemoryKind::Fact), vec![1.0, 0.0]); // dim 2 ≠ 3

        let params = IngestParams::default();
        let result = store.transaction(|s| ingest_prepared(s, tag, vec![good, bad], &params));

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
            store.shared_entity_memory_ids(&a.added[0], 10).unwrap(),
            vec![(b.added[0].clone(), 1)],
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
}
