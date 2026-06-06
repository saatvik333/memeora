//! Rust client SDK for the memeora daemon.
//!
//! Wraps the [`memeora_proto`] IPC contract with typed helpers so adapters and
//! third-party integrations can talk to the daemon without re-implementing the wire format.

/// Protocol version this client speaks.
pub const PROTOCOL_VERSION: u32 = memeora_proto::PROTOCOL_VERSION;
