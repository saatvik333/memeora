//! memeora daemon entrypoint (scaffold).
//!
//! Will host the tokio runtime, the embedding/extraction models, the single SQLite
//! writer, the async ingestion queue, and the IPC server. See `docs/ARCHITECTURE.md`.

fn main() {
    println!(
        "memeora-daemon {} (core {}, protocol v{})",
        env!("CARGO_PKG_VERSION"),
        memeora_core::version(),
        memeora_proto::PROTOCOL_VERSION,
    );
}
