//! memeora daemon library: the engine that answers the IPC protocol.
//!
//! [`Engine`] is the synchronous request handler — it owns the store, models, and
//! profile cache and maps each [`memeora_proto`] request to a response. [`run`]
//! wraps it in the tokio runtime, the single-writer thread, and the IPC transport;
//! the shipped `memeora-daemon` binary is a thin wrapper around it.

pub mod dashboard;
pub mod engine;
pub mod run;
pub mod server;

pub use engine::{ChangeEvent, Engine};
pub use run::run;
pub use server::serve;
