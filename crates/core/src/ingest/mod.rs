//! Ingestion: the write path that turns text into stored memories.
//!
//! [`ingest`] ties the pieces together: extract candidates, embed each, then
//! either **reinforce** an existing near-duplicate (strengthen it instead of
//! storing a redundant copy) or **insert** a new memory. This is the model-free
//! MVP write path; the daemon runs it on its writer thread.
//!
//! Memory ids are a content hash scoped to the container, so re-ingesting the
//! exact same statement is naturally idempotent (it resolves to reinforcement),
//! with no UUID dependency.
//!
//! Deferred (need the gated NER/NLI stack): `extends`/`updates` graph edges and
//! contradiction-based supersession.

use crate::Result;
use crate::container_tag::sha16;
use crate::embed::EmbeddingProvider;
use crate::extract::Extractor;
use crate::store::VectorStore;

/// Tuning for [`ingest`].
#[derive(Debug, Clone)]
pub struct IngestParams {
    /// A candidate within this KNN distance of an existing memory of the **same
    /// kind** reinforces it instead of being inserted. Provisional default —
    /// assumes L2 distance over L2-normalized embeddings (≈ cosine ≥ 0.98); tune
    /// against real data.
    pub dedup_max_distance: f32,
    /// Strength added to an existing memory when a near-duplicate reinforces it.
    pub reinforce_delta: f32,
}

impl Default for IngestParams {
    fn default() -> Self {
        IngestParams {
            dedup_max_distance: 0.2,
            reinforce_delta: 0.5,
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
}

/// Deterministic, container-scoped id for a memory's content.
fn content_id(container_tag: &str, content: &str) -> String {
    // NUL separator avoids tag/content boundary collisions.
    sha16(&format!("{container_tag}\u{0}{content}"))
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
    let mut outcome = IngestOutcome::default();

    for candidate in candidates {
        // Embed as a document (the form it will be stored and matched as).
        let embedding = embedder
            .embed_documents(&[candidate.content.as_str()])?
            .pop()
            .unwrap_or_default();

        // Find a near-duplicate of the same kind (scoped so the immutable borrow
        // ends before the mutable reinforce below).
        let duplicate_id = {
            let neighbors = store.knn(container_tag, &embedding, 1)?;
            neighbors
                .first()
                .filter(|top| {
                    top.score <= params.dedup_max_distance && top.memory.kind == candidate.kind
                })
                .map(|top| top.memory.id.clone())
        };
        if let Some(id) = duplicate_id {
            store.reinforce(&id, params.reinforce_delta)?;
            outcome.reinforced.push(id);
            continue;
        }

        let id = content_id(container_tag, &candidate.content);
        let memory = candidate.into_memory(id.clone(), container_tag, embedding);
        store.upsert(&memory)?;
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
}
