//! memeora daemon entrypoint.
//!
//! Loads the local embedding model and SQLite store once, then serves the IPC
//! protocol over a local socket (the writer-actor lives in [`memeora_daemon::serve`]).
//! Storage lives under `~/.memeora` (override with `MEMEORA_HOME`); the socket name
//! defaults to [`memeora_proto::DEFAULT_SOCKET`] (override with `MEMEORA_SOCKET`).

use std::error::Error;
use std::path::PathBuf;

use memeora_core::embed::fastembed::FastEmbedder;
use memeora_core::{EmbeddingProvider, HeuristicExtractor, SqliteStore};
use memeora_daemon::{Engine, serve};
use memeora_proto::{DEFAULT_SOCKET, PROTOCOL_VERSION};

/// memeora's data directory: `$MEMEORA_HOME`, else `~/.memeora`.
fn data_dir() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(dir) = std::env::var_os("MEMEORA_HOME") {
        return Ok(PathBuf::from(dir));
    }
    let home = dirs::home_dir().ok_or("could not determine home directory")?;
    Ok(home.join(".memeora"))
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir = data_dir()?;
    std::fs::create_dir_all(&data_dir)?;
    let db_path = data_dir.join("memory.db");
    let model_cache = data_dir.join("models");

    eprintln!(
        "memeora-daemon {} (protocol v{}) — loading model…",
        env!("CARGO_PKG_VERSION"),
        PROTOCOL_VERSION,
    );

    // Local, no-API-key embedder (downloads weights to the cache on first run).
    let embedder = FastEmbedder::bge_small(Some(model_cache))?;
    let store = SqliteStore::open(&db_path, embedder.dim())?;
    let engine = Engine::new(
        store,
        Box::new(embedder),
        Box::new(HeuristicExtractor::default()),
    );

    let socket = std::env::var("MEMEORA_SOCKET").unwrap_or_else(|_| DEFAULT_SOCKET.to_string());
    eprintln!(
        "memeora-daemon ready — db {}, socket {socket}",
        db_path.display()
    );
    serve(engine, &socket)?;
    Ok(())
}
