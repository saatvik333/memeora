//! Observation consolidation (Phase F): distil a scope's near-duplicate memories into
//! deduplicated canonical [`Observation`]s, each with a distinct-source proof count.
//!
//! This is the opt-in LLM tier's flagship, but the storage + clustering work with **no
//! LLM**: [`consolidate`] groups the scope's latest memories into near-duplicate clusters
//! using the store's vector KNN (exact/near-duplicate distillation — no model needed), then
//! writes one observation per cluster. The belief text for a cluster comes from an
//! [`ObservationSynthesizer`]; the default [`PassthroughSynthesizer`] picks the most
//! complete member verbatim (safe, offline), and an LLM synthesizer is an opt-in swap.
//!
//! Consolidation is **deterministic and idempotent**: the observation id is keyed on the
//! cluster's canonical (lexicographically smallest) member id, so re-running over unchanged
//! memories converges — the same observations are updated in place, and re-linking a known
//! source can't inflate `proof_count` (set-union via the composite PK, mirroring the
//! `record_evidence`/`proof_count` model).
//
// The LLM synthesizer tier now exists: [`llm::LlmSynthesizer`] (opt-in, fail-open).
// ponytail: nothing auto-triggers this — a daemon/CLI trigger is deliberately out of scope
// for the core crate. Wire a `consolidate(...)` call into the daemon (e.g. a periodic or
// on-demand "distil scope" job on the writer-actor, passing a `PassthroughSynthesizer` by
// default and a `LlmSynthesizer` only when `LlmConfig` is present and allowed).

pub mod llm;

pub use llm::LlmSynthesizer;

use std::collections::HashMap;

use crate::Result;
use crate::container_tag::sha32;
use crate::embed::EmbeddingProvider;
use crate::store::{Observation, VectorStore, now_unix};

/// Turns a cluster of near-duplicate member texts into one canonical belief sentence.
///
/// The default [`PassthroughSynthesizer`] needs no LLM. The opt-in [`llm::LlmSynthesizer`]
/// (reusing [`crate::extract::llm`]'s transport + consent gate) can be swapped in behind the
/// same trait so consolidation stays fully optional and tests use the passthrough.
///
/// `Send + Sync` so the daemon can hold one on its writer-actor thread (like `Reranker`).
pub trait ObservationSynthesizer: Send + Sync {
    /// Return one canonical belief sentence for a cluster's `members` (never empty).
    fn synthesize(&self, members: &[&str]) -> Result<String>;
}

/// The no-LLM default: pick the most complete member verbatim (longest by character
/// count, ties broken by the lexicographically smallest text). Deterministic — the same
/// cluster always yields the same canonical text, which keeps consolidation idempotent.
pub struct PassthroughSynthesizer;

impl ObservationSynthesizer for PassthroughSynthesizer {
    fn synthesize(&self, members: &[&str]) -> Result<String> {
        let canonical = members
            .iter()
            .copied()
            .reduce(|best, cur| {
                let (bl, cl) = (best.chars().count(), cur.chars().count());
                if cl > bl || (cl == bl && cur < best) {
                    cur
                } else {
                    best
                }
            })
            .unwrap_or("");
        Ok(canonical.to_string())
    }
}

/// Knobs for [`consolidate`]. Defaults match ingest's embedding-dedup convention.
#[derive(Debug, Clone)]
pub struct ConsolidationParams {
    /// Max vector distance for two memories to count as near-duplicates of the same
    /// cluster. The store's `knn` scores **L2 distance over L2-normalized embeddings**,
    /// lower = closer (see [`VectorStore::knn`] and ingest's `dedup_max_distance`). The
    /// `0.2` default matches ingest's dedup guard (≈ cosine ≥ 0.98) — the exact
    /// near-duplicate convention the rest of the engine uses (hindsight's cosine-≥-0.97
    /// guard is looser, ≈ L2 0.245; consolidation stays as tight as ingest dedup).
    pub dup_max_distance: f32,
    /// Max number of a scope's latest memories to consolidate in one pass.
    pub scan_limit: usize,
    /// KNN pool size fetched per seed when gathering a cluster's near-duplicates.
    pub knn_pool: usize,
}

impl Default for ConsolidationParams {
    fn default() -> Self {
        ConsolidationParams {
            dup_max_distance: 0.2,
            scan_limit: 500,
            knn_pool: 32,
        }
    }
}

/// What a [`consolidate`] pass produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsolidationOutcome {
    /// Observations written (one per cluster, including singletons); upserts count too.
    pub observations: usize,
    /// Total (observation, source-memory) links recorded across all clusters.
    pub sources_linked: usize,
}

/// Distil `scope`'s latest memories into observations. Groups near-duplicate memories into
/// clusters (greedy single-pass over the store's KNN at `params.dup_max_distance`), writes
/// one observation per cluster whose sources are the cluster's members (so
/// `proof_count` = distinct member count), and returns what it produced.
///
/// Deterministic and idempotent: cluster membership is a pure function of the current store
/// state, the observation id is keyed on the cluster's canonical member, and both the
/// observation upsert and source-linking are set-union writes — re-running over unchanged
/// memories converges without duplicating observations or inflating proof counts.
///
/// No LLM is required: pass a [`PassthroughSynthesizer`] for the offline default, or any
/// [`ObservationSynthesizer`] to synthesize richer belief text.
pub fn consolidate<S, E>(
    store: &mut S,
    embedder: &E,
    synth: &dyn ObservationSynthesizer,
    scope: &str,
    params: &ConsolidationParams,
) -> Result<ConsolidationOutcome>
where
    S: VectorStore + ?Sized,
    E: EmbeddingProvider + ?Sized,
{
    let memories = store.list_latest(scope, params.scan_limit)?;
    if memories.is_empty() {
        return Ok(ConsolidationOutcome::default());
    }

    // list_latest doesn't hydrate embeddings (they live in the vec0 index), so re-embed the
    // members to get the query vectors KNN needs — the same freshly-embedded-vector path
    // ingest uses. Batch once, aligned with `memories` by index.
    let texts: Vec<&str> = memories.iter().map(|m| m.content.as_str()).collect();
    let embeddings = embedder.embed_documents(&texts)?;
    let index_of: HashMap<&str, usize> = memories
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.as_str(), i))
        .collect();

    // Greedy single-pass clustering: for each not-yet-assigned memory, seed a cluster and
    // pull in its within-threshold KNN neighbours that are still in the scan set and
    // unassigned. `assigned` guarantees each memory lands in exactly one cluster, so the
    // partition is total and deterministic for a fixed store state.
    let mut assigned = vec![false; memories.len()];
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for i in 0..memories.len() {
        if assigned[i] {
            continue;
        }
        assigned[i] = true;
        let mut cluster = vec![i];
        let neighbors = store.knn(scope, &embeddings[i], params.knn_pool.max(1))?;
        for n in &neighbors {
            if n.score > params.dup_max_distance {
                continue; // KNN is distance-ordered; but keep scanning is cheap and clear.
            }
            if let Some(&j) = index_of.get(n.memory.id.as_str())
                && !assigned[j]
            {
                assigned[j] = true;
                cluster.push(j);
            }
        }
        clusters.push(cluster);
    }

    let now = now_unix();
    let mut outcome = ConsolidationOutcome::default();
    for cluster in &clusters {
        // Canonical member = lexicographically smallest id in the cluster. Stable regardless
        // of which member seeded the cluster, so the observation id is a pure function of the
        // membership set — the key to idempotent re-consolidation.
        let canonical = cluster
            .iter()
            .map(|&i| memories[i].id.as_str())
            .min()
            .expect("clusters are non-empty");
        let obs_id = sha32(&format!("{scope}\u{0}observation\u{0}{canonical}"));

        let members: Vec<&str> = cluster
            .iter()
            .map(|&i| memories[i].content.as_str())
            .collect();
        let content = synth.synthesize(&members)?;

        store.upsert_observation(&Observation {
            id: obs_id.clone(),
            container_tag: scope.to_string(),
            content,
            proof_count: cluster.len() as u32,
            created_at: now,
            updated_at: now,
        })?;
        for &i in cluster {
            store.add_observation_source(&obs_id, &memories[i].id)?;
            outcome.sources_linked += 1;
        }
        outcome.observations += 1;
    }

    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{EmbeddingProvider, EmbeddingSpace};
    use crate::store::sqlite::SqliteStore;
    use crate::store::{Memory, MemoryKind};
    use std::collections::HashMap;

    /// Map-backed embedder (mirrors the ingest tests): content → fixed vector, so tests
    /// control cosine geometry exactly. Unknown text maps to a distinct default vector.
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

    fn mem(id: &str, content: &str, tag: &str, emb: Vec<f32>) -> Memory {
        Memory::new(id, content, MemoryKind::Fact, tag, emb)
    }

    #[test]
    fn passthrough_picks_longest_then_lexicographic() {
        let s = PassthroughSynthesizer;
        // Longest by char count wins.
        assert_eq!(
            s.synthesize(&["dark mode", "user prefers dark mode", "dark"])
                .unwrap(),
            "user prefers dark mode"
        );
        // Equal length → lexicographically smallest.
        assert_eq!(s.synthesize(&["bbb", "aaa", "ccc"]).unwrap(), "aaa");
        // Order-independent (idempotency depends on this).
        assert_eq!(
            s.synthesize(&["ccc", "aaa", "bbb"]).unwrap(),
            s.synthesize(&["aaa", "bbb", "ccc"]).unwrap()
        );
    }

    #[test]
    fn near_duplicates_collapse_into_one_observation() {
        let tag = "t";
        // Two near-identical memories (near-parallel unit vectors → distance ≈ 0) plus one
        // orthogonal outlier.
        let embedder = MapEmbedder::new(&[
            ("user prefers dark mode", vec![1.0, 0.0, 0.0]),
            ("prefers dark mode", vec![0.999, 0.0447, 0.0]),
            ("deploys with docker", vec![0.0, 1.0, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        for (id, text, emb) in [
            ("m1", "user prefers dark mode", vec![1.0, 0.0, 0.0]),
            ("m2", "prefers dark mode", vec![0.999, 0.0447, 0.0]),
            ("m3", "deploys with docker", vec![0.0, 1.0, 0.0]),
        ] {
            store.upsert(&mem(id, text, tag, emb)).unwrap();
        }

        let outcome = consolidate(
            &mut store,
            &embedder,
            &PassthroughSynthesizer,
            tag,
            &ConsolidationParams::default(),
        )
        .unwrap();

        // m1+m2 collapse into one observation (proof 2); m3 is its own singleton (proof 1).
        assert_eq!(outcome.observations, 2);
        assert_eq!(outcome.sources_linked, 3);

        let obs = store.list_observations(tag, 10).unwrap();
        assert_eq!(obs.len(), 2);
        let merged = obs.iter().find(|o| o.proof_count == 2).unwrap();
        // Passthrough kept the most complete member verbatim.
        assert_eq!(merged.content, "user prefers dark mode");
        assert!(obs.iter().any(|o| o.proof_count == 1));
    }

    #[test]
    fn unrelated_memories_stay_separate() {
        let tag = "t";
        let embedder = MapEmbedder::new(&[
            ("a", vec![1.0, 0.0, 0.0]),
            ("b", vec![0.0, 1.0, 0.0]),
            ("c", vec![0.0, 0.0, 1.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        store
            .upsert(&mem("m1", "a", tag, vec![1.0, 0.0, 0.0]))
            .unwrap();
        store
            .upsert(&mem("m2", "b", tag, vec![0.0, 1.0, 0.0]))
            .unwrap();
        store
            .upsert(&mem("m3", "c", tag, vec![0.0, 0.0, 1.0]))
            .unwrap();

        let outcome = consolidate(
            &mut store,
            &embedder,
            &PassthroughSynthesizer,
            tag,
            &ConsolidationParams::default(),
        )
        .unwrap();

        // Three orthogonal memories → three singleton observations, each proof_count 1.
        assert_eq!(outcome.observations, 3);
        let obs = store.list_observations(tag, 10).unwrap();
        assert_eq!(obs.len(), 3);
        assert!(obs.iter().all(|o| o.proof_count == 1));
    }

    #[test]
    fn re_running_is_idempotent() {
        let tag = "t";
        let embedder = MapEmbedder::new(&[
            ("user prefers dark mode", vec![1.0, 0.0, 0.0]),
            ("prefers dark mode", vec![0.999, 0.0447, 0.0]),
        ]);
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        store
            .upsert(&mem(
                "m1",
                "user prefers dark mode",
                tag,
                vec![1.0, 0.0, 0.0],
            ))
            .unwrap();
        store
            .upsert(&mem(
                "m2",
                "prefers dark mode",
                tag,
                vec![0.999, 0.0447, 0.0],
            ))
            .unwrap();

        let run = |store: &mut SqliteStore| {
            consolidate(
                store,
                &embedder,
                &PassthroughSynthesizer,
                tag,
                &ConsolidationParams::default(),
            )
            .unwrap()
        };

        let first = run(&mut store);
        let ids_after_first: Vec<String> = store
            .list_observations(tag, 10)
            .unwrap()
            .into_iter()
            .map(|o| o.id)
            .collect();
        let second = run(&mut store);

        // Same partition both runs; no new observations, proof_count stable.
        assert_eq!(first, second);
        let obs = store.list_observations(tag, 10).unwrap();
        assert_eq!(obs.len(), 1, "re-running must not duplicate observations");
        assert_eq!(
            obs[0].proof_count, 2,
            "proof_count is stable across re-runs"
        );
        assert_eq!(
            ids_after_first,
            obs.iter().map(|o| o.id.clone()).collect::<Vec<_>>(),
            "observation id is stable across re-runs"
        );
    }

    #[test]
    fn observation_sources_round_trip_via_store() {
        // Directly exercise the additive store methods: upsert + source-union + proof recompute.
        let tag = "t";
        let mut store = SqliteStore::open_in_memory(3).unwrap();
        store
            .upsert(&mem("m1", "a", tag, vec![1.0, 0.0, 0.0]))
            .unwrap();
        store
            .upsert(&mem("m2", "b", tag, vec![0.0, 1.0, 0.0]))
            .unwrap();

        let now = now_unix();
        store
            .upsert_observation(&Observation {
                id: "obs1".into(),
                container_tag: tag.into(),
                content: "canonical belief".into(),
                proof_count: 0,
                created_at: now,
                updated_at: now,
            })
            .unwrap();
        store.add_observation_source("obs1", "m1").unwrap();
        assert_eq!(store.list_observations(tag, 10).unwrap()[0].proof_count, 1);

        // A distinct source raises proof_count...
        store.add_observation_source("obs1", "m2").unwrap();
        assert_eq!(store.list_observations(tag, 10).unwrap()[0].proof_count, 2);
        // ...but re-linking a known source is a set-union no-op.
        store.add_observation_source("obs1", "m1").unwrap();
        let obs = store.list_observations(tag, 10).unwrap();
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].proof_count, 2);
        assert_eq!(obs[0].content, "canonical belief");
    }
}
