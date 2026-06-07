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
    Candidate, EmbeddingProvider, Extractor, IngestParams, Memory, MemoryKind, PreparedCandidate,
    ProfileCache, ScoredMemory, SearchParams, SqliteStore, VectorStore, embed_candidates,
    ingest_prepared, search,
};
use memeora_proto::{MemoryDto, PROTOCOL_VERSION, Request, Response};

/// A request whose embedding/extraction has already been done, ready for the DB.
///
/// Built by [`Preparer::prepare`] off the writer thread; applied by
/// [`Engine::handle_prepared`] on it.
pub(crate) enum Prepared {
    Hello,
    Ingest {
        scope: String,
        candidates: Vec<PreparedCandidate>,
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
    },
    Context {
        scope: String,
    },
    List {
        scope: String,
        limit: usize,
    },
    Forget {
        id: String,
    },
}

/// Turns a [`Request`] into a [`Prepared`] one by running extraction + embedding.
///
/// Cheaply cloneable (the embedder/extractor are shared `Arc`s) so each connection
/// thread holds its own handle and inference happens in parallel, never on the
/// writer thread.
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
            Request::Ingest { scope, text } => {
                let candidates = self.extractor.extract(&text)?;
                let candidates = embed_candidates(self.embedder.as_ref(), candidates)?;
                Prepared::Ingest { scope, candidates }
            }
            Request::Add {
                scope,
                content,
                kind,
            } => {
                let candidate = Candidate {
                    content,
                    kind: MemoryKind::from_str_lossy(&kind),
                    expires_at: None,
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
            Request::Recall { scope, query, k } => {
                let query_embedding = self.embedder.embed_query(&query)?;
                Prepared::Recall {
                    scope,
                    query,
                    query_embedding,
                    k,
                }
            }
            Request::Context { scope } => Prepared::Context { scope },
            Request::List { scope, limit } => Prepared::List { scope, limit },
            Request::Forget { id } => Prepared::Forget { id },
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
        }
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
        match self.dispatch(prepared) {
            Ok(response) => response,
            Err(err) => Response::Error {
                message: err.to_string(),
            },
        }
    }

    fn dispatch(&mut self, prepared: Prepared) -> memeora_core::Result<Response> {
        Ok(match prepared {
            Prepared::Hello => Response::Hello {
                protocol_version: PROTOCOL_VERSION,
                server_version: env!("CARGO_PKG_VERSION").to_string(),
            },

            Prepared::Ingest { scope, candidates } => {
                let outcome =
                    ingest_prepared(&mut self.store, &scope, candidates, &self.ingest_params)?;
                self.profiles.invalidate(&scope);
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
                let outcome = ingest_prepared(
                    &mut self.store,
                    &scope,
                    vec![(candidate, embedding)],
                    &self.ingest_params,
                )?;
                self.profiles.invalidate(&scope);
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
            } => {
                let params = SearchParams {
                    k,
                    ..self.search_params.clone()
                };
                let hits = search(&self.store, &scope, &query_embedding, &query, &params)?;
                Response::Memories {
                    memories: hits.iter().map(scored_to_dto).collect(),
                }
            }

            Prepared::Context { scope } => {
                let profile = self.profiles.get_or_build(&self.store, &scope)?;
                Response::Context {
                    statics: profile.statics.iter().map(memory_to_dto).collect(),
                    dynamics: profile.dynamics.iter().map(memory_to_dto).collect(),
                }
            }

            Prepared::List { scope, limit } => {
                let memories = self.store.list_latest(&scope, limit)?;
                Response::Memories {
                    memories: memories.iter().map(memory_to_dto).collect(),
                }
            }

            Prepared::Forget { id } => {
                // Capture the scope before forgetting so we can invalidate its profile.
                let scope = self.store.get(&id)?.map(|m| m.container_tag);
                self.store.forget(&id)?;
                if let Some(scope) = scope {
                    self.profiles.invalidate(&scope);
                }
                Response::Forgotten
            }
        })
    }
}

/// Project a stored memory onto the wire DTO (no relevance score).
fn memory_to_dto(memory: &Memory) -> MemoryDto {
    MemoryDto {
        id: memory.id.clone(),
        content: memory.content.clone(),
        kind: memory.kind.as_str().to_string(),
        strength: memory.strength,
        created_at: memory.created_at,
        score: None,
    }
}

/// Project a scored search hit onto the wire DTO (carrying the relevance score).
fn scored_to_dto(scored: &ScoredMemory) -> MemoryDto {
    MemoryDto {
        score: Some(scored.score),
        ..memory_to_dto(&scored.memory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memeora_core::{EmbeddingSpace, HeuristicExtractor};
    use std::collections::HashMap;

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
}
