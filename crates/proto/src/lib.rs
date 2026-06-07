//! memeora IPC protocol: versioned message types and the capability handshake
//! shared between the daemon and all clients (MCP server, hook binary, CLI, SDKs).
//!
//! This is a public, semver'd contract — see `docs/ARCHITECTURE.md`. Types here
//! are deliberately decoupled from the engine's internal types ([`memeora-core`]):
//! the wire format can stay stable while internals evolve. The daemon maps
//! between the two.

use interprocess::local_socket::{GenericFilePath, GenericNamespaced, Name, prelude::*};
use serde::{Deserialize, Serialize};

pub mod frame;

/// Wire protocol version. Bumped only on **breaking** changes to the IPC contract.
///
/// Additive changes (a new optional field, a new capability string, a new request
/// variant a server may ignore) do *not* bump this — see `docs/PROTOCOL.md` for the
/// stability policy. Clients gate optional behavior on [`capabilities`](Response::Hello)
/// rather than the version number.
pub const PROTOCOL_VERSION: u32 = 1;

/// Capability tokens a daemon advertises in its [`Response::Hello`], so clients can
/// negotiate optional features without bumping [`PROTOCOL_VERSION`]. The set is the
/// daemon's supported operations; future optional features append new tokens here.
pub mod capability {
    /// Ingest raw text ([`Request::Ingest`]).
    pub const INGEST: &str = "ingest";
    /// Add an explicit memory ([`Request::Add`]).
    pub const ADD: &str = "add";
    /// Hybrid recall ([`Request::Recall`]).
    pub const RECALL: &str = "recall";
    /// Profile/context ([`Request::Context`]).
    pub const CONTEXT: &str = "context";
    /// List memories ([`Request::List`]).
    pub const LIST: &str = "list";
    /// Soft-forget ([`Request::Forget`]).
    pub const FORGET: &str = "forget";

    /// The full set a current daemon supports. Returned by the daemon in its
    /// handshake; kept here so client and server agree on the canonical list.
    pub const ALL: &[&str] = &[INGEST, ADD, RECALL, CONTEXT, LIST, FORGET];
}

/// Build a local-socket [`Name`] from a string: a value containing a path
/// separator is a filesystem socket path; otherwise a namespaced name.
///
/// Shared by the daemon (listener) and every client (SDK, hook, CLI, MCP) so both
/// ends resolve the same socket string to the same endpoint — no drift.
pub fn build_name(socket: &str) -> std::io::Result<Name<'_>> {
    if socket.contains('/') || socket.contains('\\') {
        socket.to_fs_name::<GenericFilePath>()
    } else {
        socket.to_ns_name::<GenericNamespaced>()
    }
}

/// Default local-socket name the daemon listens on and clients connect to.
/// A bare name (no path separator) is treated as a namespaced socket
/// (Linux abstract namespace / Windows named pipe).
pub const DEFAULT_SOCKET: &str = "memeora-daemon.sock";

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
    /// Handshake reply with the daemon's versions and capabilities.
    Hello {
        /// Daemon's [`PROTOCOL_VERSION`].
        protocol_version: u32,
        /// Daemon crate version (semver).
        server_version: String,
        /// Operations/features this daemon supports (see [`capability`]). Defaults
        /// to empty when absent so a newer client still parses an older daemon's
        /// handshake — capability negotiation never breaks the wire format.
        #[serde(default)]
        capabilities: Vec<String>,
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
    fn hello_without_capabilities_is_back_compatible() {
        // An older daemon's handshake (no `capabilities` field) must still parse,
        // defaulting to an empty set — additive changes never break the wire format.
        let json = r#"{"type":"hello","protocol_version":1,"server_version":"0.0.0"}"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        match resp {
            Response::Hello {
                protocol_version,
                capabilities,
                ..
            } => {
                assert_eq!(protocol_version, 1);
                assert!(capabilities.is_empty());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn capability_set_is_nonempty_and_roundtrips() {
        assert!(!capability::ALL.is_empty());
        let resp = Response::Hello {
            protocol_version: PROTOCOL_VERSION,
            server_version: "0.0.0".into(),
            capabilities: capability::ALL.iter().map(|s| s.to_string()).collect(),
        };
        match roundtrip_response(&resp) {
            Response::Hello { capabilities, .. } => {
                assert!(capabilities.iter().any(|c| c == capability::RECALL));
            }
            other => panic!("unexpected: {other:?}"),
        }
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
