//! The `memeora-daemon` binary entrypoint, as a library function.
//!
//! Lives in the library (rather than a `main.rs`) so the single shipped `memeora`
//! package can expose every binary from one crate — `dist` bundles all binaries of
//! one package into a single installer, and it cannot merge separate packages (see
//! `docs/ARCHITECTURE.md`, Step 10). The thin `memeora-daemon` bin just calls
//! [`run`].

use std::error::Error;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use memeora_core::embed::fastembed::FastEmbedder;
use memeora_core::{EmbeddingProvider, HeuristicExtractor, SqliteStore};
use memeora_proto::{DEFAULT_SOCKET, PROTOCOL_VERSION};
use tokio::sync::broadcast;

use crate::{Engine, dashboard, serve};

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

/// Whether the model cache already holds weights (any entry present). A first-run
/// download is only attempted when this is false.
fn model_cache_populated(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false)
}

/// Explicit opt-in to a one-time first-run model download. Offline by default, so the
/// daemon never makes an unconsented network call to fetch weights ("no required
/// network / never a silent fallback").
fn model_download_allowed() -> bool {
    std::env::var("MEMEORA_ALLOW_MODEL_DOWNLOAD")
        .map(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

/// Load the model + store once, optionally start the dashboard, then serve IPC
/// (blocks for the process lifetime — the writer-actor owns the sole DB write conn).
pub fn run() -> Result<(), Box<dyn Error>> {
    let data_dir = data_dir()?;
    std::fs::create_dir_all(&data_dir)?;
    let db_path = data_dir.join("memory.db");
    // Honors MEMEORA_MODELS_DIR (offline bundle) → MEMEORA_HOME/models → ~/.memeora/models.
    let model_cache = memeora_core::models::resolve_dir();

    eprintln!(
        "memeora-daemon {} (protocol v{}) — loading model…",
        env!("CARGO_PKG_VERSION"),
        PROTOCOL_VERSION,
    );

    // If the model cache carries a SHA256SUMS manifest (an offline bundle, or one
    // stamped by `memeora models bundle`), verify integrity before loading — a
    // corrupt/tampered weight file should fail loudly, not silently mis-embed.
    if let Ok(Some(report)) = memeora_core::models::verify_dir(&model_cache)
        && !report.ok()
    {
        let (ok, mismatch, missing) = report.counts();
        eprintln!(
            "memeora-daemon: WARNING model integrity check failed in {} \
             ({ok} ok, {mismatch} mismatched, {missing} missing); re-download or re-bundle",
            model_cache.display()
        );
    }

    // Offline-first: don't silently reach out to HuggingFace on first run. If the
    // weights aren't already cached and the user hasn't opted into a one-time
    // download, refuse with an actionable message rather than making an unconsented
    // network call (the "no required network / never a silent fallback" invariant).
    if !model_cache_populated(&model_cache) && !model_download_allowed() {
        return Err(format!(
            "embedding model not found in {} and a first-run download is not enabled \
             (offline by default). Either set MEMEORA_ALLOW_MODEL_DOWNLOAD=1 to fetch it \
             once (~130 MB from HuggingFace), or provide an offline bundle there and point \
             MEMEORA_MODELS_DIR at it (see `memeora models bundle`).",
            model_cache.display()
        )
        .into());
    }

    // Local, no-API-key embedder (downloads weights to the cache on first run, only
    // when allowed by the consent check above).
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
                let dim = read_store.dim();
                let socket = socket.clone();
                let events = events_tx.clone();
                let db_path = db_path.clone();
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
                        if let Err(e) = dashboard::serve(addr, db_path, dim, socket, events).await {
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
