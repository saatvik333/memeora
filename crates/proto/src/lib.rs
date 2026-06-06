//! memeora IPC protocol: versioned message types and the capability handshake
//! shared between the daemon and all clients (MCP server, hook binary, CLI, SDKs).
//!
//! This is a public, semver'd contract — see `docs/ARCHITECTURE.md`.

/// Wire protocol version. Bumped on breaking changes to the IPC contract.
pub const PROTOCOL_VERSION: u32 = 1;
