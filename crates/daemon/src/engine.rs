//! The synchronous request handler behind the IPC protocol.
//!
//! [`Engine`] holds the storage, embedding, extraction, and profile-cache pieces
//! and turns a [`Request`] into a [`Response`]. It is intentionally sync (rusqlite
//! is sync): the daemon runs one `Engine` on a dedicated writer thread, with the
//! tokio side forwarding requests to it over a channel.

use memeora_core::{
    Candidate, EmbeddingProvider, Extractor, IngestParams, Memory, MemoryKind, ProfileCache,
    ScoredMemory, SearchParams, SqliteStore, VectorStore, ingest, ingest_candidates, search,
};
use memeora_proto::{MemoryDto, PROTOCOL_VERSION, Request, Response};

/// Owns the engine and answers protocol requests.
pub struct Engine {
    store: SqliteStore,
    embedder: Box<dyn EmbeddingProvider>,
    extractor: Box<dyn Extractor>,
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
            embedder,
            extractor,
            profiles: ProfileCache::with_defaults(),
            ingest_params: IngestParams::default(),
            search_params: SearchParams::default(),
        }
    }

    /// Handle one request, converting any engine error into [`Response::Error`].
    pub fn handle(&mut self, request: Request) -> Response {
        match self.dispatch(request) {
            Ok(response) => response,
            Err(err) => Response::Error {
                message: err.to_string(),
            },
        }
    }

    fn dispatch(&mut self, request: Request) -> memeora_core::Result<Response> {
        Ok(match request {
            Request::Hello { .. } => Response::Hello {
                protocol_version: PROTOCOL_VERSION,
                server_version: env!("CARGO_PKG_VERSION").to_string(),
            },

            Request::Ingest { scope, text } => {
                let outcome = ingest(
                    &mut self.store,
                    self.embedder.as_ref(),
                    self.extractor.as_ref(),
                    &scope,
                    &text,
                    &self.ingest_params,
                )?;
                self.profiles.invalidate(&scope);
                Response::Ingested {
                    added: outcome.added.len(),
                    reinforced: outcome.reinforced.len(),
                }
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
                let outcome = ingest_candidates(
                    &mut self.store,
                    self.embedder.as_ref(),
                    &scope,
                    vec![candidate],
                    &self.ingest_params,
                )?;
                self.profiles.invalidate(&scope);
                // The single memory was either inserted or reinforced an existing one.
                let id = outcome
                    .added
                    .into_iter()
                    .chain(outcome.reinforced)
                    .next()
                    .unwrap_or_default();
                Response::Added { id }
            }

            Request::Recall { scope, query, k } => {
                let query_embedding = self.embedder.embed_query(&query)?;
                let params = SearchParams {
                    k,
                    ..self.search_params.clone()
                };
                let hits = search(&self.store, &scope, &query_embedding, &query, &params)?;
                Response::Memories {
                    memories: hits.iter().map(scored_to_dto).collect(),
                }
            }

            Request::Context { scope } => {
                let profile = self.profiles.get_or_build(&self.store, &scope)?;
                Response::Context {
                    statics: profile.statics.iter().map(memory_to_dto).collect(),
                    dynamics: profile.dynamics.iter().map(memory_to_dto).collect(),
                }
            }

            Request::List { scope, limit } => {
                let memories = self.store.list_latest(&scope, limit)?;
                Response::Memories {
                    memories: memories.iter().map(memory_to_dto).collect(),
                }
            }

            Request::Forget { id } => {
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
