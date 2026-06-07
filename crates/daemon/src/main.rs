//! memeora daemon entrypoint.
//!
//! Loads the local embedding model and SQLite store once, then serves the IPC
//! protocol over a local socket (the writer-actor lives in [`memeora_daemon::serve`]).
//! Storage lives under `~/.memeora` (override with `MEMEORA_HOME`); the socket name
//! defaults to [`memeora_proto::DEFAULT_SOCKET`] (override with `MEMEORA_SOCKET`).

use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;

use memeora_core::embed::fastembed::FastEmbedder;
use memeora_core::{EmbeddingProvider, HeuristicExtractor, SqliteStore};
use memeora_daemon::{Engine, dashboard, serve};
use memeora_proto::{DEFAULT_SOCKET, PROTOCOL_VERSION};
use tokio::sync::broadcast;

/// Default address the local dashboard binds (loopback only — no network exposure).
const DEFAULT_DASHBOARD_ADDR: &str = "127.0.0.1:7878";

/// memeora's data directory: `$MEMEORA_HOME`, else `~/.memeora`.
fn data_dir() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(dir) = std::env::var_os("MEMEORA_HOME") {
        return Ok(PathBuf::from(dir));
    }
    let home = dirs::home_dir().ok_or("could not determine home directory")?;
    Ok(home.join(".memeora"))
}

/// The dashboard's bind address: `$MEMEORA_DASHBOARD_ADDR` (default
/// [`DEFAULT_DASHBOARD_ADDR`]), or `None` if set to `off`/empty or unparseable.
fn dashboard_addr() -> Option<SocketAddr> {
    let raw = std::env::var("MEMEORA_DASHBOARD_ADDR")
        .unwrap_or_else(|_| DEFAULT_DASHBOARD_ADDR.to_string());
    if raw.is_empty() || raw.eq_ignore_ascii_case("off") {
        return None;
    }
    match raw.parse::<SocketAddr>() {
        // The dashboard has no auth — its whole security model is loopback-only.
        // Refuse a non-loopback bind (e.g. 0.0.0.0) rather than silently exposing
        // memory contents + destructive forget to the network.
        Ok(addr) if !addr.ip().is_loopback() => {
            eprintln!(
                "memeora-daemon: refusing non-loopback dashboard bind {addr} (the dashboard is unauthenticated); \
                 use a loopback address or MEMEORA_DASHBOARD_ADDR=off"
            );
            None
        }
        Ok(addr) => Some(addr),
        Err(e) => {
            eprintln!("memeora-daemon: invalid MEMEORA_DASHBOARD_ADDR {raw:?}: {e}; dashboard off");
            None
        }
    }
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
    let dim = embedder.dim();
    let store = SqliteStore::open(&db_path, dim)?;

    // Change broadcaster for the dashboard's live (SSE) stream. Cloned to the
    // engine (publisher) and the dashboard (subscriber); harmless if no dashboard.
    let (events_tx, _events_rx) = broadcast::channel(256);
    let engine = Engine::new(
        store,
        Box::new(embedder),
        Box::new(HeuristicExtractor::default()),
    )
    .with_events(events_tx.clone());

    let socket = std::env::var("MEMEORA_SOCKET").unwrap_or_else(|_| DEFAULT_SOCKET.to_string());

    // Start the local dashboard (optional) on its own thread + runtime, using a
    // second read-only connection so it never contends with the IPC writer. A
    // failure here is non-fatal: the daemon's core job is the IPC server.
    if let Some(addr) = dashboard_addr() {
        // A genuine read-only connection (the writer thread above already created +
        // migrated the DB), so the dashboard can never write — the daemon stays the
        // sole writer by construction, not by call-ordering.
        match SqliteStore::open_readonly(&db_path, dim) {
            Ok(read_store) => {
                let socket = socket.clone();
                let events = events_tx.clone();
                std::thread::spawn(move || {
                    let rt = match tokio::runtime::Runtime::new() {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("memeora-daemon: dashboard runtime failed: {e}");
                            return;
                        }
                    };
                    rt.block_on(async move {
                        eprintln!("memeora-daemon: dashboard on http://{addr}");
                        if let Err(e) = dashboard::serve(addr, read_store, socket, events).await {
                            eprintln!("memeora-daemon: dashboard stopped: {e}");
                        }
                    });
                });
            }
            Err(e) => {
                eprintln!("memeora-daemon: dashboard disabled (cannot open read store): {e}")
            }
        }
    }

    eprintln!(
        "memeora-daemon ready — db {}, socket {socket}",
        db_path.display()
    );
    // The IPC server is the daemon's lifetime; it owns the sole writer and blocks.
    serve(engine, &socket)?;
    Ok(())
}
