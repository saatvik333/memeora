//! memeora IPC protocol: versioned message types and the capability handshake
//! shared between the daemon and all clients (MCP server, hook binary, CLI, SDKs).
//!
//! This is a public, semver'd contract — see `docs/ARCHITECTURE.md`. Types here
//! are deliberately decoupled from the engine's internal types ([`memeora-core`]):
//! the wire format can stay stable while internals evolve. The daemon maps
//! between the two.

use serde::{Deserialize, Serialize};

pub mod frame;

/// Wire protocol version. Bumped on breaking changes to the IPC contract.
pub const PROTOCOL_VERSION: u32 = 1;

/// A scope/container identifier (e.g. a `memeora_user_*` / `repo_*` tag).
pub type Scope = String;

/// A request from a client to the daemon.
///
/// Serialized with an `"op"` discriminator, e.g. `{"op":"recall","scope":"…",…}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Capability handshake — the client announces its protocol version.
    Hello {
        /// Client's [`PROTOCOL_VERSION`].
        protocol_version: u32,
    },
    /// Ingest raw text (extract → embed → dedup/reinforce → store). Async upstream.
    Ingest {
        /// Target scope.
        scope: Scope,
        /// Raw conversation text.
        text: String,
    },
    /// Add a single explicit memory.
    Add {
        /// Target scope.
        scope: Scope,
        /// Memory content.
        content: String,
        /// Memory kind (`fact` | `preference` | `episode`).
        kind: String,
    },
    /// Hybrid search within a scope.
    Recall {
        /// Scope to search.
        scope: Scope,
        /// Query text.
        query: String,
        /// Max results.
        k: usize,
    },
    /// Fetch the profile (static + dynamic) for a scope.
    Context {
        /// Scope to summarize.
        scope: Scope,
    },
    /// List the latest memories in a scope.
    List {
        /// Scope to list.
        scope: Scope,
        /// Max results.
        limit: usize,
    },
    /// Soft-forget a memory by id (never hard-deleted).
    Forget {
        /// Memory id.
        id: String,
    },
}

/// A response from the daemon to a client.
///
/// Serialized with a `"type"` discriminator, e.g. `{"type":"memories","memories":[…]}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    /// Handshake reply with the daemon's versions.
    Hello {
        /// Daemon's [`PROTOCOL_VERSION`].
        protocol_version: u32,
        /// Daemon crate version (semver).
        server_version: String,
    },
    /// Result of an [`Request::Ingest`].
    Ingested {
        /// Newly stored memories.
        added: usize,
        /// Existing memories reinforced by near-duplicates.
        reinforced: usize,
    },
    /// Result of an [`Request::Add`].
    Added {
        /// Id of the stored memory.
        id: String,
    },
    /// Result of an [`Request::Recall`] / [`Request::List`].
    Memories {
        /// Matched memories, most relevant / newest first.
        memories: Vec<MemoryDto>,
    },
    /// Result of an [`Request::Context`].
    Context {
        /// Stable facts and preferences.
        statics: Vec<MemoryDto>,
        /// Recent episodes.
        dynamics: Vec<MemoryDto>,
    },
    /// Acknowledgement for an [`Request::Forget`].
    Forgotten,
    /// The request failed.
    Error {
        /// Human-readable error message.
        message: String,
    },
}

/// A memory projected onto the wire (a subset of the engine's `Memory`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryDto {
    /// Stable id.
    pub id: String,
    /// The memory text.
    pub content: String,
    /// `fact` | `preference` | `episode`.
    pub kind: String,
    /// Reinforcement strength.
    pub strength: f32,
    /// Creation time (Unix seconds).
    pub created_at: i64,
    /// Relevance score when returned from a search (`None` for plain listings).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_request(req: &Request) -> Request {
        let json = serde_json::to_string(req).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    fn roundtrip_response(resp: &Response) -> Response {
        let json = serde_json::to_string(resp).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn request_roundtrips() {
        let reqs = [
            Request::Hello {
                protocol_version: PROTOCOL_VERSION,
            },
            Request::Ingest {
                scope: "memeora_user_abc".into(),
                text: "I prefer rust".into(),
            },
            Request::Recall {
                scope: "repo_memeora".into(),
                query: "language".into(),
                k: 5,
            },
            Request::Forget { id: "m1".into() },
        ];
        for req in &reqs {
            assert_eq!(&roundtrip_request(req), req);
        }
    }

    #[test]
    fn response_roundtrips() {
        let resp = Response::Memories {
            memories: vec![MemoryDto {
                id: "m1".into(),
                content: "I prefer rust".into(),
                kind: "preference".into(),
                strength: 1.0,
                created_at: 1_700_000_000,
                score: Some(0.42),
            }],
        };
        assert_eq!(roundtrip_response(&resp), resp);
    }

    #[test]
    fn request_uses_op_discriminator() {
        let json = serde_json::to_string(&Request::Context { scope: "s".into() }).unwrap();
        assert!(json.contains("\"op\":\"context\""), "got: {json}");
    }

    #[test]
    fn score_is_omitted_when_absent() {
        let dto = MemoryDto {
            id: "m1".into(),
            content: "x".into(),
            kind: "fact".into(),
            strength: 1.0,
            created_at: 1,
            score: None,
        };
        let json = serde_json::to_string(&dto).unwrap();
        assert!(!json.contains("score"), "score should be omitted: {json}");
    }
}
