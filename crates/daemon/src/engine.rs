//! The synchronous request handler behind the IPC protocol.
//!
//! Work is split so the CPU-heavy, DB-free part (extraction + embedding) runs off
//! the single writer thread:
//! - [`Preparer`] turns a [`Request`] into a [`Prepared`] request — embedding the
//!   query / candidates using a *shared* embedder + extractor. It runs on the
//!   per-connection threads, so model inference parallelizes across clients.
//! - [`Engine`] owns the store + profile cache and applies a [`Prepared`] request
//!   to the DB. It is intentionally sync (rusqlite is sync) and runs alone on the
//!   daemon's writer thread, so the DB stays single-writer.

use std::sync::Arc;

use memeora_core::{
    Candidate, ConsolidationParams, EmbeddingProvider, Extractor, IngestParams, Memory, MemoryKind,
    ObservationSynthesizer, PassthroughSynthesizer, PreparedCandidate, ProfileCache, Reranker,
    ScoredMemory, SearchParams, SqliteStore, VectorStore, consolidate, embed_candidates, freshness,
    ingest_prepared, now_unix, rerank_memories, sanitize, search,
};
use memeora_proto::{MemoryDto, PROTOCOL_VERSION, Request, Response};
use tokio::sync::broadcast;

/// A notification that a scope's memories changed, broadcast to the dashboard's
/// live (SSE) stream so connected browsers can refresh. Carries only the scope and
/// the kind of change — never memory content — so it stays cheap and leak-free.
#[derive(Clone, Debug)]
pub struct ChangeEvent {
    /// The container tag whose memories changed.
    pub scope: String,
    /// What happened: `"ingested"`, `"added"`, or `"forgotten"`.
    pub op: &'static str,
}

/// A request whose embedding/extraction has already been done, ready for the DB.
///
/// Built by [`Preparer::prepare`] off the writer thread; applied by
/// [`Engine::handle_prepared`] on it.
pub(crate) enum Prepared {
    Hello,
    Ingest {
        scope: String,
        candidates: Vec<PreparedCandidate>,
        source: Option<String>,
    },
    Add {
        scope: String,
        candidate: Candidate,
        embedding: Vec<f32>,
    },
    Recall {
        scope: String,
        query: String,
        query_embedding: Vec<f32>,
        k: usize,
        max_tokens: Option<usize>,
    },
    Context {
        scope: String,
    },
    Bundle {
        scope: String,
        query: String,
        query_embedding: Vec<f32>,
        k: usize,
        max_tokens: Option<usize>,
    },
    List {
        scope: String,
        limit: usize,
    },
    Forget {
        id: String,
    },
    Consolidate {
        scope: String,
    },
}

/// Turns a [`Request`] into a [`Prepared`] one by running extraction + embedding.
///
/// Cheaply cloneable (the embedder/extractor are shared `Arc`s) so each connection
/// thread holds its own handle and runs extraction/embedding off the writer thread.
/// (Extraction parallelizes; embedding serializes on the shared model's ONNX
/// session — the win is keeping inference off the single writer, not parallelism.)
#[derive(Clone)]
pub(crate) struct Preparer {
    embedder: Arc<dyn EmbeddingProvider>,
    extractor: Arc<dyn Extractor>,
}

impl Preparer {
    /// Extract + embed as needed, producing a DB-ready [`Prepared`] request.
    pub(crate) fn prepare(&self, request: Request) -> memeora_core::Result<Prepared> {
        Ok(match request {
            Request::Hello { .. } => Prepared::Hello,
            Request::Ingest {
                scope,
                text,
                source,
            } => {
                // Enforce the privacy invariant at the engine boundary: strip
                // <private>…</private> and redact secrets before extraction/embedding
                // so every write surface (MCP/IPC/hook) inherits it.
                let text = sanitize(&text);
                let candidates = self.extractor.extract(&text)?;
                let candidates = embed_candidates(self.embedder.as_ref(), candidates)?;
                Prepared::Ingest {
                    scope,
                    candidates,
                    source,
                }
            }
            Request::Add {
                scope,
                content,
                kind,
            } => {
                // Same engine-boundary sanitization as Ingest — an explicit `remember`
                // / `add` must not bypass the privacy invariant.
                let content = sanitize(&content);
                if content.trim().is_empty() {
                    return Err(memeora_core::Error::Invalid(
                        "memory content is empty after privacy filtering".into(),
                    ));
                }
                let kind = match kind.as_str() {
                    "fact" => MemoryKind::Fact,
                    "preference" => MemoryKind::Preference,
                    "episode" => MemoryKind::Episode,
                    _ => {
                        return Err(memeora_core::Error::Invalid(format!(
                            "invalid memory kind: {kind}"
                        )));
                    }
                };
                let candidate = Candidate {
                    content,
                    kind,
                    expires_at: None,
                    occurred_start: None,
                    occurred_end: None,
                    confidence: 1.0,
                };
                let mut prepared = embed_candidates(self.embedder.as_ref(), vec![candidate])?;
                let (candidate, embedding) = prepared
                    .pop()
                    .expect("one candidate embedded yields one prepared candidate");
                Prepared::Add {
                    scope,
                    candidate,
                    embedding,
                }
            }
            Request::Recall {
                scope,
                query,
                k,
                max_tokens,
            } => {
                let query_embedding = self.embedder.embed_query(&query)?;
                Prepared::Recall {
                    scope,
                    query,
                    query_embedding,
                    k,
                    max_tokens,
                }
            }
            Request::Context { scope } => Prepared::Context { scope },
            Request::Bundle {
                scope,
                query,
                k,
                max_tokens,
            } => {
                // Embed the query here (off the writer thread), mirroring Recall.
                let query_embedding = self.embedder.embed_query(&query)?;
                Prepared::Bundle {
                    scope,
                    query,
                    query_embedding,
                    k,
                    max_tokens,
                }
            }
            Request::List { scope, limit } => Prepared::List { scope, limit },
            Request::Forget { id } => Prepared::Forget { id },
            // Consolidation re-embeds cluster members itself on the write side (it needs
            // the store's KNN), so there's nothing to prepare off-thread here.
            Request::Consolidate { scope } => Prepared::Consolidate { scope },
        })
    }
}

/// Owns the store + profile cache and applies prepared requests to the DB.
pub struct Engine {
    store: SqliteStore,
    embedder: Arc<dyn EmbeddingProvider>,
    extractor: Arc<dyn Extractor>,
    profiles: ProfileCache,
    ingest_params: IngestParams,
    search_params: SearchParams,
    /// Optional cross-encoder reranker (opt-in). When present, recall over-fetches a
    /// larger candidate pool and re-scores it jointly against the query for a quality
    /// upgrade; when absent, recall is exactly the fused [`search`] result.
    reranker: Option<Box<dyn Reranker>>,
    /// Belief-text synthesizer for consolidation. Defaults to the no-LLM
    /// [`PassthroughSynthesizer`]; the daemon swaps in an LLM one when configured.
    synthesizer: Box<dyn ObservationSynthesizer>,
    /// Optional sink for [`ChangeEvent`]s, set by the daemon when the dashboard is
    /// enabled. `send` is best-effort: an error just means no live listeners.
    events: Option<broadcast::Sender<ChangeEvent>>,
    #[cfg(test)]
    panic_once: Option<&'static str>,
}

impl Engine {
    /// Build an engine over a store, an embedding backend, and an extractor.
    pub fn new(
        store: SqliteStore,
        embedder: Box<dyn EmbeddingProvider>,
        extractor: Box<dyn Extractor>,
    ) -> Self {
        Engine {
            store,
            embedder: Arc::from(embedder),
            extractor: Arc::from(extractor),
            profiles: ProfileCache::with_defaults(),
            ingest_params: IngestParams::default(),
            search_params: SearchParams::default(),
            reranker: None,
            synthesizer: Box::new(PassthroughSynthesizer),
            events: None,
            #[cfg(test)]
            panic_once: None,
        }
    }

    /// Attach a cross-encoder reranker so recall re-scores its fused candidates.
    /// Opt-in: without this, recall is byte-identical to the plain [`search`] path.
    pub fn with_reranker(mut self, reranker: Box<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Swap in a belief-text synthesizer for consolidation (e.g. the opt-in LLM one).
    /// Without this, consolidation uses the no-LLM [`PassthroughSynthesizer`].
    pub fn with_synthesizer(mut self, synthesizer: Box<dyn ObservationSynthesizer>) -> Self {
        self.synthesizer = synthesizer;
        self
    }

    /// Attach a [`ChangeEvent`] sink so mutations are broadcast to the dashboard's
    /// live stream. Without this, change events are simply not emitted.
    pub fn with_events(mut self, events: broadcast::Sender<ChangeEvent>) -> Self {
        self.events = Some(events);
        self
    }

    /// Best-effort broadcast of a change in `scope`. A send error means there are
    /// no live subscribers, which is fine — it is never fatal to a write.
    fn emit(&self, scope: &str, op: &'static str) {
        if let Some(tx) = &self.events {
            let _ = tx.send(ChangeEvent {
                scope: scope.to_string(),
                op,
            });
        }
    }

    #[cfg(test)]
    pub(crate) fn with_test_panic_once(mut self, message: &'static str) -> Self {
        self.panic_once = Some(message);
        self
    }

    /// A [`Preparer`] sharing this engine's embedder + extractor, for use on the
    /// connection threads (so embedding runs off the writer thread).
    pub(crate) fn preparer(&self) -> Preparer {
        Preparer {
            embedder: Arc::clone(&self.embedder),
            extractor: Arc::clone(&self.extractor),
        }
    }

    /// Handle one request end-to-end (prepare + apply). Convenience for callers
    /// that don't split the work across threads (tests, non-daemon embedders).
    #[cfg(test)]
    pub fn handle(&mut self, request: Request) -> Response {
        match self.preparer().prepare(request) {
            Ok(prepared) => self.handle_prepared(prepared),
            Err(err) => Response::Error {
                message: err.to_string(),
            },
        }
    }

    /// Apply an already-prepared request to the DB, converting any engine error
    /// into [`Response::Error`].
    pub(crate) fn handle_prepared(&mut self, prepared: Prepared) -> Response {
        #[cfg(test)]
        if !matches!(prepared, Prepared::Hello)
            && let Some(message) = self.panic_once.take()
        {
            panic!("{message}");
        }
        match self.dispatch(prepared) {
            Ok(response) => response,
            Err(err) => Response::Error {
                message: err.to_string(),
            },
        }
    }

    /// Run recall, applying the optional reranker. Reranking is CPU-bound and this
    /// runs on the engine's writer thread (a blocking context), matching where
    /// `search` already runs — so it never blocks a tokio worker.
    ///
    /// Without a reranker this is exactly `search` with the caller's `k`/`max_tokens`
    /// (byte-identical to the pre-rerank path). With one, it over-fetches a larger
    /// candidate pool (`k * candidate_multiplier`), re-scores it jointly against the
    /// query, and keeps the top `k`. The token budget still caps the over-fetched
    /// pool, so the reranked subset stays within budget.
    fn recall_hits(
        &self,
        scope: &str,
        query: &str,
        query_embedding: &[f32],
        k: usize,
        max_tokens: Option<usize>,
    ) -> memeora_core::Result<Vec<ScoredMemory>> {
        let Some(reranker) = &self.reranker else {
            let params = SearchParams {
                k,
                max_tokens,
                ..self.search_params.clone()
            };
            return search(&self.store, scope, query_embedding, query, &params);
        };
        let pool = k
            .saturating_mul(self.search_params.candidate_multiplier)
            .max(k);
        let params = SearchParams {
            k: pool,
            max_tokens,
            ..self.search_params.clone()
        };
        let candidates = search(&self.store, scope, query_embedding, query, &params)?;
        rerank_memories(reranker.as_ref(), query, &candidates, k)
    }

    fn dispatch(&mut self, prepared: Prepared) -> memeora_core::Result<Response> {
        Ok(match prepared {
            Prepared::Hello => Response::Hello {
                protocol_version: PROTOCOL_VERSION,
                server_version: env!("CARGO_PKG_VERSION").to_string(),
                capabilities: memeora_proto::capability::ALL
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            },

            Prepared::Ingest {
                scope,
                candidates,
                source,
            } => {
                // One transaction for the whole batch: a mid-batch failure rolls back
                // every candidate rather than leaving a partial write the client was
                // told failed (which a retry would then double-reinforce).
                let params = self.ingest_params.clone();
                let outcome = self.store.transaction(|s| {
                    ingest_prepared(s, &scope, source.as_deref(), candidates, &params)
                })?;
                self.profiles.invalidate(&scope);
                self.emit(&scope, "ingested");
                Response::Ingested {
                    added: outcome.added.len(),
                    reinforced: outcome.reinforced.len(),
                }
            }

            Prepared::Add {
                scope,
                candidate,
                embedding,
            } => {
                // Atomic insert + edge-linking (the new memory's `extends` edges are
                // separate writes) so an `add` is all-or-nothing too.
                let params = self.ingest_params.clone();
                let outcome = self.store.transaction(|s| {
                    ingest_prepared(s, &scope, None, vec![(candidate, embedding)], &params)
                })?;
                self.profiles.invalidate(&scope);
                self.emit(&scope, "added");
                // The single memory was either inserted or reinforced an existing one;
                // a missing id means it was dropped, which we surface rather than ack.
                match outcome.added.into_iter().chain(outcome.reinforced).next() {
                    Some(id) => Response::Added { id },
                    None => Response::Error {
                        message: "add stored no memory".to_string(),
                    },
                }
            }

            Prepared::Recall {
                scope,
                query,
                query_embedding,
                k,
                max_tokens,
            } => {
                let hits = self.recall_hits(&scope, &query, &query_embedding, k, max_tokens)?;
                let now = now_unix();
                Response::Memories {
                    memories: hits.iter().map(|h| scored_to_dto(h, now)).collect(),
                }
            }

            Prepared::Context { scope } => {
                let profile = self.profiles.get_or_build(&self.store, &scope)?;
                let now = now_unix();
                Response::Context {
                    statics: profile
                        .statics
                        .iter()
                        .map(|m| memory_to_dto(m, now))
                        .collect(),
                    dynamics: profile
                        .dynamics
                        .iter()
                        .map(|m| memory_to_dto(m, now))
                        .collect(),
                }
            }

            Prepared::Bundle {
                scope,
                query,
                query_embedding,
                k,
                max_tokens,
            } => {
                // Profile (cached) + recall in one round-trip. Both read the store; the
                // profile comes from the same cache Context uses, the recall from the
                // same `search` Recall uses.
                let profile = self.profiles.get_or_build(&self.store, &scope)?;
                let hits = self.recall_hits(&scope, &query, &query_embedding, k, max_tokens)?;
                let now = now_unix();

                // Dedup by id with priority static > dynamic > search: strip anything
                // already surfaced by a higher-priority section (the real case is a
                // recall hit that also lives in the profile).
                let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
                let statics: Vec<MemoryDto> = profile
                    .statics
                    .iter()
                    .inspect(|m| {
                        seen.insert(m.id.as_str());
                    })
                    .map(|m| memory_to_dto(m, now))
                    .collect();
                let dynamics: Vec<MemoryDto> = profile
                    .dynamics
                    .iter()
                    .filter(|m| seen.insert(m.id.as_str()))
                    .map(|m| memory_to_dto(m, now))
                    .collect();
                let memories: Vec<MemoryDto> = hits
                    .iter()
                    .filter(|h| seen.insert(h.memory.id.as_str()))
                    .map(|h| scored_to_dto(h, now))
                    .collect();
                Response::Bundle {
                    statics,
                    dynamics,
                    memories,
                }
            }

            Prepared::List { scope, limit } => {
                let memories = self.store.list_latest(&scope, limit)?;
                let now = now_unix();
                Response::Memories {
                    memories: memories.iter().map(|m| memory_to_dto(m, now)).collect(),
                }
            }

            Prepared::Forget { id } => {
                // Capture the scope before forgetting so we can invalidate its profile.
                let scope = self.store.get(&id)?.map(|m| m.container_tag);
                self.store.forget(&id)?;
                if let Some(scope) = scope {
                    self.profiles.invalidate(&scope);
                    self.emit(&scope, "forgotten");
                }
                Response::Forgotten
            }

            Prepared::Consolidate { scope } => {
                // Distil the scope's near-duplicate memories into observations. Idempotent
                // and per-observation atomic (no outer transaction needed); re-embeds
                // cluster members itself. Disjoint field borrows: store (mut) vs
                // embedder/synthesizer (shared).
                let params = ConsolidationParams::default();
                let outcome = consolidate(
                    &mut self.store,
                    self.embedder.as_ref(),
                    self.synthesizer.as_ref(),
                    &scope,
                    &params,
                )?;
                self.profiles.invalidate(&scope);
                Response::Consolidated {
                    observations: outcome.observations,
                    sources_linked: outcome.sources_linked,
                }
            }
        })
    }
}

/// Project a stored memory onto the wire DTO (no relevance score). `now` is the read
/// clock used to derive the freshness/decay trend label.
fn memory_to_dto(memory: &Memory, now: i64) -> MemoryDto {
    MemoryDto {
        id: memory.id.clone(),
        content: memory.content.clone(),
        kind: memory.kind.as_str().to_string(),
        strength: memory.strength,
        created_at: memory.created_at,
        score: None,
        freshness: Some(freshness(memory, now).to_string()),
    }
}

/// Project a scored search hit onto the wire DTO (carrying the relevance score).
fn scored_to_dto(scored: &ScoredMemory, now: i64) -> MemoryDto {
    MemoryDto {
        score: Some(scored.score),
        ..memory_to_dto(&scored.memory, now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memeora_core::{EmbeddingSpace, HeuristicExtractor, RerankHit};
    use std::collections::HashMap;

    /// Deterministic mock reranker: scores each candidate by its input position so the
    /// output is the exact reverse of the fused order. No model download — this proves
    /// the wiring (the engine adopts whatever order the reranker dictates).
    struct ReverseReranker;

    impl Reranker for ReverseReranker {
        fn rerank(
            &self,
            _query: &str,
            docs: &[&str],
            top_k: usize,
        ) -> memeora_core::Result<Vec<RerankHit>> {
            // Higher score = earlier: score by index so the LAST candidate ranks first.
            let mut hits: Vec<RerankHit> = docs
                .iter()
                .enumerate()
                .map(|(i, _)| RerankHit {
                    index: i,
                    score: i as f32,
                })
                .collect();
            hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            hits.truncate(top_k);
            Ok(hits)
        }
    }

    /// Deterministic embedder: prescribed vectors per text, distinct fallback.
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
        fn embed_documents(&self, texts: &[&str]) -> memeora_core::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| self.map.get(*t).cloned().unwrap_or(vec![0.0, 0.0, 1.0]))
                .collect())
        }
    }

    fn engine(pairs: &[(&str, Vec<f32>)]) -> Engine {
        Engine::new(
            SqliteStore::open_in_memory(3).unwrap(),
            Box::new(MapEmbedder::new(pairs)),
            Box::new(HeuristicExtractor::default()),
        )
    }

    #[test]
    fn hello_reports_versions() {
        let mut e = engine(&[]);
        match e.handle(Request::Hello {
            protocol_version: PROTOCOL_VERSION,
        }) {
            Response::Hello {
                protocol_version, ..
            } => assert_eq!(protocol_version, PROTOCOL_VERSION),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn add_then_recall_roundtrip() {
        let mut e = engine(&[("I prefer dark mode", vec![1.0, 0.0, 0.0])]);
        let scope = "memeora_user_x";

        let added = e.handle(Request::Add {
            scope: scope.into(),
            content: "I prefer dark mode".into(),
            kind: "preference".into(),
        });
        let id = match added {
            Response::Added { id } => id,
            other => panic!("unexpected: {other:?}"),
        };
        assert!(!id.is_empty());

        match e.handle(Request::Recall {
            scope: scope.into(),
            query: "I prefer dark mode".into(),
            k: 5,
            max_tokens: None,
        }) {
            Response::Memories { memories } => {
                assert_eq!(memories.len(), 1);
                assert_eq!(memories[0].id, id);
                assert!(memories[0].score.is_some());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn ingest_then_context_partitions() {
        let mut e = engine(&[
            ("I prefer dark mode", vec![1.0, 0.0, 0.0]),
            ("We use SQLite for storage", vec![0.0, 1.0, 0.0]),
        ]);
        let scope = "s";
        let out = e.handle(Request::Ingest {
            scope: scope.into(),
            text: "I prefer dark mode. We use SQLite for storage.".into(),
            source: None,
        });
        assert!(matches!(out, Response::Ingested { added: 2, .. }));

        match e.handle(Request::Context {
            scope: scope.into(),
        }) {
            Response::Context { statics, dynamics } => {
                assert_eq!(statics.len(), 2); // a preference + a fact
                assert_eq!(dynamics.len(), 0);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn bundle_returns_profile_and_deduped_recall() {
        let mut e = engine(&[
            ("I prefer dark mode", vec![1.0, 0.0, 0.0]),
            ("deployed the app today", vec![0.0, 1.0, 0.0]),
        ]);
        let scope = "s";
        let pref_id = match e.handle(Request::Add {
            scope: scope.into(),
            content: "I prefer dark mode".into(),
            kind: "preference".into(),
        }) {
            Response::Added { id } => id,
            other => panic!("unexpected: {other:?}"),
        };
        e.handle(Request::Add {
            scope: scope.into(),
            content: "deployed the app today".into(),
            kind: "episode".into(),
        });

        // Sanity: a plain recall DOES surface the preference — so the bundle's dedup
        // is what removes it from `memories`, not a missing hit.
        match e.handle(Request::Recall {
            scope: scope.into(),
            query: "I prefer dark mode".into(),
            k: 5,
            max_tokens: None,
        }) {
            Response::Memories { memories } => {
                assert!(memories.iter().any(|m| m.id == pref_id));
            }
            other => panic!("unexpected: {other:?}"),
        }

        match e.handle(Request::Bundle {
            scope: scope.into(),
            query: "I prefer dark mode".into(),
            k: 5,
            max_tokens: None,
        }) {
            Response::Bundle {
                statics,
                dynamics,
                memories,
            } => {
                // Profile partitions by kind: the preference is static, episode dynamic.
                assert_eq!(statics.len(), 1);
                assert_eq!(statics[0].id, pref_id);
                assert_eq!(dynamics.len(), 1);
                assert_eq!(dynamics[0].kind, "episode");
                // The preference lives in `statics`, so its recall hit is deduped out.
                assert!(
                    memories.iter().all(|m| m.id != pref_id),
                    "static leaked into memories: {memories:?}"
                );
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn consolidate_distils_near_duplicates_into_an_observation() {
        // Two cross-kind near-duplicates (same embedding, different kind — so ingest's
        // same-kind dedup does NOT merge them at Add time) plus an unrelated memory: the
        // consolidate op clusters the duplicates into one proof-counted observation and
        // leaves the singleton as its own. Uses the default PassthroughSynthesizer.
        let mut e = engine(&[
            ("the user prefers postgres", vec![1.0, 0.0, 0.0]),
            ("postgres is the chosen database", vec![1.0, 0.0, 0.0]),
            ("deploys with docker", vec![0.0, 1.0, 0.0]),
        ]);
        let scope = "s";
        for (content, kind) in [
            ("the user prefers postgres", "preference"),
            ("postgres is the chosen database", "fact"),
            ("deploys with docker", "fact"),
        ] {
            e.handle(Request::Add {
                scope: scope.into(),
                content: content.into(),
                kind: kind.into(),
            });
        }

        match e.handle(Request::Consolidate {
            scope: scope.into(),
        }) {
            Response::Consolidated {
                observations,
                sources_linked,
            } => {
                // Two clusters (the postgres pair + the docker singleton) → 2 observations,
                // 3 source links total.
                assert_eq!(
                    observations, 2,
                    "postgres pair collapses, docker stands alone"
                );
                assert_eq!(sources_linked, 3);
            }
            other => panic!("unexpected: {other:?}"),
        }

        // Idempotent: re-running converges (same observations, same links, no duplicates).
        match e.handle(Request::Consolidate {
            scope: scope.into(),
        }) {
            Response::Consolidated {
                observations,
                sources_linked,
            } => {
                assert_eq!(observations, 2);
                assert_eq!(sources_linked, 3);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn mutations_emit_change_events() {
        let (tx, mut rx) = broadcast::channel(16);
        let mut e = engine(&[("I prefer dark mode", vec![1.0, 0.0, 0.0])]).with_events(tx);
        let scope = "s";

        let id = match e.handle(Request::Add {
            scope: scope.into(),
            content: "I prefer dark mode".into(),
            kind: "preference".into(),
        }) {
            Response::Added { id } => id,
            other => panic!("unexpected: {other:?}"),
        };
        let ev = rx.try_recv().expect("add should emit an event");
        assert_eq!((ev.scope.as_str(), ev.op), (scope, "added"));

        e.handle(Request::Forget { id });
        let ev = rx.try_recv().expect("forget should emit an event");
        assert_eq!((ev.scope.as_str(), ev.op), (scope, "forgotten"));

        // A pure read (List) must NOT emit a change event.
        e.handle(Request::List {
            scope: scope.into(),
            limit: 10,
        });
        assert!(rx.try_recv().is_err(), "reads must not emit events");
    }

    #[test]
    fn forget_removes_from_list() {
        let mut e = engine(&[("I prefer dark mode", vec![1.0, 0.0, 0.0])]);
        let scope = "s";
        let id = match e.handle(Request::Add {
            scope: scope.into(),
            content: "I prefer dark mode".into(),
            kind: "preference".into(),
        }) {
            Response::Added { id } => id,
            other => panic!("unexpected: {other:?}"),
        };

        e.handle(Request::Forget { id });
        match e.handle(Request::List {
            scope: scope.into(),
            limit: 10,
        }) {
            Response::Memories { memories } => assert!(memories.is_empty()),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn add_sanitizes_private_and_secrets() {
        // The engine boundary must strip <private> spans and redact secrets on the
        // explicit-write path too (this is what the MCP `remember` tool hits).
        let mut e = engine(&[]);
        let scope = "s";
        let id = match e.handle(Request::Add {
            scope: scope.into(),
            content: "token sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 <private>my ssh key</private> ok"
                .into(),
            kind: "fact".into(),
        }) {
            Response::Added { id } => id,
            other => panic!("unexpected: {other:?}"),
        };

        match e.handle(Request::List {
            scope: scope.into(),
            limit: 10,
        }) {
            Response::Memories { memories } => {
                let stored = &memories.iter().find(|m| m.id == id).unwrap().content;
                assert!(!stored.contains("sk-ABCDEF"), "secret leaked: {stored:?}");
                assert!(
                    stored.contains("[REDACTED]"),
                    "secret not masked: {stored:?}"
                );
                assert!(
                    !stored.contains("my ssh key"),
                    "private span leaked: {stored:?}"
                );
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn reranker_dictates_recall_order() {
        // Two memories give a deterministic fused order. A reversing reranker must flip
        // it; an engine without one must return the untouched fused order.
        let pairs: &[(&str, Vec<f32>)] = &[
            ("alpha memory", vec![1.0, 0.0, 0.0]),
            ("beta memory", vec![0.0, 1.0, 0.0]),
        ];
        let scope = "s";

        // Seed identical data into an engine, then recall the ids in order.
        let seed_and_recall = |mut e: Engine| -> Vec<String> {
            for (content, _) in pairs {
                e.handle(Request::Add {
                    scope: scope.into(),
                    content: (*content).into(),
                    kind: "fact".into(),
                });
            }
            match e.handle(Request::Recall {
                scope: scope.into(),
                query: "alpha memory".into(),
                k: 5,
                max_tokens: None,
            }) {
                Response::Memories { memories } => memories.into_iter().map(|m| m.id).collect(),
                other => panic!("unexpected: {other:?}"),
            }
        };

        // Baseline (no reranker): the natural fused order.
        let base = seed_and_recall(engine(pairs));
        assert_eq!(base.len(), 2, "both memories recalled");

        // With the reversing reranker: the exact reverse of the baseline order.
        let reranked = seed_and_recall(engine(pairs).with_reranker(Box::new(ReverseReranker)));
        let mut expected = base.clone();
        expected.reverse();
        assert_eq!(
            reranked, expected,
            "recall must adopt the reranker's dictated order"
        );
    }
}
