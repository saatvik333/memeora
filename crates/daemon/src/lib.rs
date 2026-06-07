//! memeora daemon library: the engine that answers the IPC protocol.
//!
//! [`Engine`] is the synchronous request handler — it owns the store, models, and
//! profile cache and maps each [`memeora_proto`] request to a response. The binary
//! ([`main`](../main.rs)) wraps it in the tokio runtime, the single-writer thread,
//! and the IPC transport (added in later steps; see `docs/ARCHITECTURE.md`).

pub mod engine;

pub use engine::Engine;
