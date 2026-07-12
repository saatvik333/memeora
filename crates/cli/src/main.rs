//! memeora CLI — a thin client over the daemon for inspecting and editing memory.
//!
//! Talks to a running `memeora-daemon` over the local socket. (Lifecycle commands
//! like `serve`/`install` land in later steps; today this is the query/edit
//! surface plus `dashboard`, which opens the daemon's local graph UI.)

use std::error::Error;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use memeora_client::Client;
use memeora_core::container_tag::project_tag;
use memeora_core::models;
use memeora_proto::resolve_socket;

/// Default address the daemon serves the dashboard on (see `memeora-daemon`).
const DEFAULT_DASHBOARD_ADDR: &str = "127.0.0.1:7878";

#[derive(Parser)]
#[command(name = "memeora", version, about = "Local memory engine — CLI client")]
struct Cli {
    /// Daemon socket name/path (defaults to the built-in name).
    #[arg(long, global = true)]
    socket: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the engine daemon in the foreground (models + DB + IPC server).
    ///
    /// Equivalent to the `memeora-daemon` binary; provided so the single `memeora`
    /// command covers the whole lifecycle. Blocks until interrupted — background it
    /// with `&`. Honors `--socket` (forwarded as `MEMEORA_SOCKET`) and the daemon's
    /// env (`MEMEORA_HOME`, `MEMEORA_ALLOW_MODEL_DOWNLOAD`, …).
    Serve,
    /// Check that the daemon is reachable and report its protocol version.
    Doctor,
    /// Store a single explicit memory.
    Add {
        /// Scope/container tag.
        scope: String,
        /// Memory content.
        content: String,
        /// Kind: fact | preference | episode.
        #[arg(long, default_value = "fact")]
        kind: String,
    },
    /// Ingest raw text (the engine extracts memories from it).
    Ingest {
        /// Scope/container tag.
        scope: String,
        /// Text to ingest.
        text: String,
    },
    /// Search memories within a scope.
    Recall {
        /// Scope/container tag.
        scope: String,
        /// Query text.
        query: String,
        /// Max results.
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// Show the profile (stable facts/preferences + recent episodes) for a scope.
    Context {
        /// Scope/container tag.
        scope: String,
    },
    /// List the most recent memories in a scope.
    List {
        /// Scope/container tag.
        scope: String,
        /// Max results.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Soft-forget a memory by id.
    Forget {
        /// Memory id.
        id: String,
    },
    /// Consolidate a scope: distil near-duplicate memories into proof-counted observations.
    Consolidate {
        /// Scope/container tag.
        scope: String,
    },
    /// Print the project container tag for a path (defaults to the cwd).
    ///
    /// Daemon-free: lets adapters (e.g. the OpenCode shim) resolve the same
    /// scope the hook uses without reimplementing the hashing.
    Scope {
        /// Path to scope (defaults to the current directory).
        path: Option<String>,
    },
    /// Open the local dashboard (the graph UI served by the daemon) in a browser.
    Dashboard {
        /// Print the URL without launching a browser.
        #[arg(long)]
        no_open: bool,
    },
    /// Inspect and verify the local model cache (daemon-free).
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Print the resolved model cache directory.
    Dir,
    /// Verify model files against the cache's SHA256SUMS manifest.
    ///
    /// Exits non-zero if any file is mismatched or missing — usable in scripts to
    /// gate on an intact offline bundle before starting the daemon.
    Verify {
        /// Directory to verify (default: the resolved model cache).
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Stamp a SHA256SUMS manifest over the cache (for an offline model bundle).
    Bundle {
        /// Directory to stamp (default: the resolved model cache).
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Overwrite an existing manifest.
        #[arg(long)]
        force: bool,
    },
}

/// Handle the daemon-free `models` commands.
fn run_models(cmd: &ModelsCmd) -> Result<(), Box<dyn Error>> {
    match cmd {
        ModelsCmd::Dir => println!("{}", models::resolve_dir().display()),
        ModelsCmd::Verify { dir } => {
            let dir = dir.clone().unwrap_or_else(models::resolve_dir);
            match models::verify_dir(&dir)? {
                None => {
                    println!("no {} manifest in {}", models::MANIFEST_NAME, dir.display());
                    println!("(nothing to verify — run `memeora models bundle` to create one)");
                }
                Some(report) => {
                    for r in &report.results {
                        match &r.status {
                            models::AssetStatus::Ok => {}
                            models::AssetStatus::Mismatch { .. } => {
                                println!("MISMATCH  {}", r.path)
                            }
                            models::AssetStatus::Missing => println!("MISSING   {}", r.path),
                        }
                    }
                    let (ok, mismatch, missing) = report.counts();
                    println!(
                        "{ok} ok, {mismatch} mismatched, {missing} missing  ({})",
                        dir.display()
                    );
                    if !report.ok() {
                        return Err("model verification failed".into());
                    }
                }
            }
        }
        ModelsCmd::Bundle { dir, force } => {
            let dir = dir.clone().unwrap_or_else(models::resolve_dir);
            let manifest_path = dir.join(models::MANIFEST_NAME);
            if manifest_path.exists() && !force {
                return Err(format!(
                    "{} already exists; pass --force to overwrite",
                    manifest_path.display()
                )
                .into());
            }
            let manifest = models::generate_manifest(&dir)?;
            std::fs::write(&manifest_path, &manifest)?;
            println!(
                "wrote {} ({} files)",
                manifest_path.display(),
                manifest.lines().count()
            );
        }
    }
    Ok(())
}

/// The dashboard URL from `$MEMEORA_DASHBOARD_ADDR` (default
/// [`DEFAULT_DASHBOARD_ADDR`]), or `None` if the daemon won't serve it.
fn dashboard_url() -> Option<String> {
    let raw = std::env::var("MEMEORA_DASHBOARD_ADDR")
        .unwrap_or_else(|_| DEFAULT_DASHBOARD_ADDR.to_string());
    dashboard_url_from(&raw)
}

/// Mirrors the daemon's gating rule (`dashboard_addr` in `crates/daemon/src/run.rs`
/// — the source of truth): the daemon serves the dashboard only for a parseable,
/// loopback `SocketAddr`, refusing non-loopback binds (it is unauthenticated) and
/// disabling on `off`/empty/unparseable. The CLI must not print or open a URL the
/// daemon never serves.
fn dashboard_url_from(raw: &str) -> Option<String> {
    if raw.is_empty() || raw.eq_ignore_ascii_case("off") {
        return None;
    }
    let addr: std::net::SocketAddr = raw.parse().ok()?;
    if !addr.ip().is_loopback() {
        return None;
    }
    Some(format!("http://{addr}"))
}

/// Best-effort browser launch for the host platform.
fn open_browser(url: &str) -> std::io::Result<()> {
    use std::process::{Command, Stdio};
    let mut cmd;
    #[cfg(target_os = "windows")]
    {
        cmd = Command::new("cmd");
        cmd.args(["/C", "start", "", url]);
    }
    #[cfg(target_os = "macos")]
    {
        cmd = Command::new("open");
        cmd.arg(url);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        cmd = Command::new("xdg-open");
        cmd.arg(url);
    }
    cmd.stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|_| ())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // `serve` *is* the daemon — hand off before trying to connect as a client.
    if let Command::Serve = &cli.command {
        // The daemon resolves its socket from `$MEMEORA_SOCKET`; forward an explicit
        // `--socket` so `memeora --socket X serve` binds exactly where the CLI looks.
        if let Some(s) = &cli.socket {
            // SAFETY: single-threaded process startup, before the daemon spawns any
            // threads or reads the env — no concurrent access to the environment.
            unsafe { std::env::set_var("MEMEORA_SOCKET", s) };
        }
        return memeora_daemon::run();
    }

    // `scope` and `adapter` are pure local commands — handle them before the daemon.
    if let Command::Scope { path } = &cli.command {
        let path = match path {
            Some(p) => p.clone(),
            None => std::env::current_dir()?.display().to_string(),
        };
        println!("{}", project_tag(&path));
        return Ok(());
    }
    if let Command::Models { cmd } = &cli.command {
        return run_models(cmd);
    }

    let socket = resolve_socket(cli.socket);
    let mut client = Client::connect(&socket).map_err(|e| {
        format!("cannot reach the daemon at {socket}: {e}\nis `memeora-daemon` running?")
    })?;

    match cli.command {
        Command::Doctor => {
            let version = client.hello()?;
            println!("daemon ok — protocol v{version} (socket {socket})");
            println!("server version: {}", client.server_version());
            println!("capabilities: {}", client.capabilities().join(", "));
            let dir = models::resolve_dir();
            println!("model cache: {}", dir.display());
            match models::verify_dir(&dir) {
                Ok(Some(report)) => {
                    let (ok, mismatch, missing) = report.counts();
                    println!("model integrity: {ok} ok, {mismatch} mismatched, {missing} missing");
                    if !report.ok() {
                        return Err("model integrity check failed".into());
                    }
                }
                Ok(None) => println!("model integrity: no manifest (unverified)"),
                Err(e) => println!("model integrity: check error: {e}"),
            }
        }
        Command::Add {
            scope,
            content,
            kind,
        } => {
            let id = client.add(&scope, &content, &kind)?;
            println!("{id}");
        }
        Command::Ingest { scope, text } => {
            let (added, reinforced) = client.ingest(&scope, &text)?;
            println!("added {added}, reinforced {reinforced}");
        }
        Command::Recall { scope, query, k } => {
            for m in client.recall(&scope, &query, k)? {
                let score = m.score.map(|s| format!(" ({s:.3})")).unwrap_or_default();
                println!("[{}] {}{score}", m.kind, m.content);
            }
        }
        Command::Context { scope } => {
            let (statics, dynamics) = client.context(&scope)?;
            println!("# Stable");
            for m in &statics {
                println!("[{}] {}", m.kind, m.content);
            }
            println!("\n# Recent");
            for m in &dynamics {
                println!("[{}] {}", m.kind, m.content);
            }
        }
        Command::List { scope, limit } => {
            for m in client.list(&scope, limit)? {
                println!("{}  [{}] {}", m.id, m.kind, m.content);
            }
        }
        Command::Forget { id } => {
            client.forget(&id)?;
            println!("forgotten {id}");
        }
        Command::Consolidate { scope } => {
            let (observations, sources) = client.consolidate(&scope)?;
            println!("consolidated {scope}: {observations} observations, {sources} sources linked");
        }
        Command::Dashboard { no_open } => {
            // Reaching here means the daemon handshake above succeeded, so the
            // dashboard it serves should be live.
            match dashboard_url() {
                Some(url) => {
                    println!("dashboard: {url}");
                    if !no_open && let Err(e) = open_browser(&url) {
                        eprintln!("could not open a browser ({e}); visit {url} manually");
                    }
                }
                None => {
                    println!(
                        "the dashboard is disabled (MEMEORA_DASHBOARD_ADDR is off, invalid, or \
                         non-loopback — the daemon refuses to serve it)"
                    )
                }
            }
        }
        // Handled above before the daemon connection.
        Command::Serve | Command::Scope { .. } | Command::Models { .. } => {
            unreachable!("daemon-free commands are handled before connecting")
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        // clap's own lint: catches conflicting args, bad defaults, etc.
        Cli::command().debug_assert();
    }

    #[test]
    fn dashboard_url_matches_daemon_gating() {
        // Served: loopback addresses (v4 and v6).
        assert_eq!(
            dashboard_url_from("127.0.0.1:7878").as_deref(),
            Some("http://127.0.0.1:7878")
        );
        assert_eq!(
            dashboard_url_from("[::1]:7878").as_deref(),
            Some("http://[::1]:7878")
        );
        // Not served by the daemon → no URL: off/empty, non-loopback (refused),
        // and unparseable (hostname or bare port).
        assert_eq!(dashboard_url_from("off"), None);
        assert_eq!(dashboard_url_from("OFF"), None);
        assert_eq!(dashboard_url_from(""), None);
        assert_eq!(dashboard_url_from("0.0.0.0:7878"), None);
        assert_eq!(dashboard_url_from("192.168.1.5:7878"), None);
        assert_eq!(dashboard_url_from("localhost:7878"), None);
        assert_eq!(dashboard_url_from("7878"), None);
    }
}
