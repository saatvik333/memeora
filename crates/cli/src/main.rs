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
use memeora_proto::DEFAULT_SOCKET;

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
    /// Scaffold support for a new harness (daemon-free).
    Adapter {
        #[command(subcommand)]
        cmd: AdapterCmd,
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

#[derive(Subcommand)]
enum AdapterCmd {
    /// Create a starter host descriptor + README for a new command-hook harness.
    New {
        /// Harness name (e.g. `cursor`); used for the descriptor file + directory.
        name: String,
        /// Output directory (default: `adapters/<name>`).
        #[arg(long)]
        dir: Option<String>,
    },
}

/// Scaffold a new adapter: a starter host descriptor + a README of next steps.
/// Refuses to overwrite an existing descriptor so a re-run can't clobber edits.
fn scaffold_adapter(name: &str, dir: Option<&str>) -> Result<(), Box<dyn Error>> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(format!("invalid harness name {name:?}: use letters, digits, - or _").into());
    }
    let dir = PathBuf::from(
        dir.map(String::from)
            .unwrap_or_else(|| format!("adapters/{name}")),
    );
    let descriptor = dir.join(format!("{name}.toml"));
    if descriptor.exists() {
        return Err(format!(
            "{} already exists; refusing to overwrite",
            descriptor.display()
        )
        .into());
    }
    std::fs::create_dir_all(&dir)?;
    std::fs::write(&descriptor, descriptor_template(name))?;
    let readme = dir.join("README.md");
    if !readme.exists() {
        std::fs::write(&readme, adapter_readme(name))?;
    }
    println!("scaffolded adapter for {name:?}:");
    println!("  {}", descriptor.display());
    println!("  {}", readme.display());
    println!("next: edit the descriptor, wire {name}'s hooks to call");
    println!(
        "  memeora-hook --descriptor {} --event <session-start|stop|...>",
        descriptor.display()
    );
    println!(
        "then add fixtures to crates/hook/tests/fixtures/ and run `cargo test -p memeora-hook`."
    );
    println!("see docs/ADAPTERS.md");
    Ok(())
}

/// A commented starter host-descriptor for `name`.
fn descriptor_template(name: &str) -> String {
    format!(
        r#"# memeora host descriptor — {name}.
#
# Edit these fields to match how {name} delivers hook payloads, then wire its hooks
# to call: memeora-hook --descriptor <path-to-this-file> --event <event>
# Validate with the conformance kit (see docs/ADAPTERS.md).
name = "{name}"

# Payload field paths to read the project directory from (first hit wins; falls
# back to the process cwd). Dotted paths index objects; numeric segments index
# arrays (e.g. "workspacePaths.0").
scope_fields = ["cwd"]

# Payload field paths for the transcript file captured at Stop / PreCompact.
transcript_fields = ["transcript_path"]

# Injection render style:
#   additional_context -> {{"hookSpecificOutput":{{"hookEventName":<inject_event_name>,"additionalContext":<text>}}}}
#   inject_steps       -> {{"injectSteps":[{{"userMessage":<text>}}]}}
inject_style = "additional_context"
inject_event_name = "SessionStart"

# Raw JSON a capture event must print on stdout ("" = print nothing).
capture_ack = "{{}}"

# Uncomment to inject only on the first invocation (gate on a numeric field == 1):
# invocation_gate_field = "invocationNum"
"#
    )
}

/// A starter README for a new adapter.
fn adapter_readme(name: &str) -> String {
    format!(
        r#"# memeora adapter — {name}

Generated by `memeora adapter new {name}`.

## Steps

1. Edit `{name}.toml` to match this host's hook payload (scope/transcript field
   paths, injection style, capture ack, invocation gating). See `docs/ADAPTERS.md`.
2. Wire {name}'s hooks to invoke the memeora hook binary, e.g.:
   `memeora-hook --descriptor /abs/path/{name}.toml --event session-start`
   (and `--event stop` / `--event pre-compact` for capture).
3. Point {name} at the MCP server (`memeora-mcp`) for the recall/remember/context/
   list tools — most harnesses need only a config entry, no code.
4. Add fixtures under `crates/hook/tests/fixtures/{name}/` and run
   `cargo test -p memeora-hook` so the conformance kit validates this host.

All adapters assume `memeora-hook` / `memeora-mcp` are on `PATH` and a
`memeora-daemon` is running.
"#
    )
}

/// The dashboard URL from `$MEMEORA_DASHBOARD_ADDR` (default
/// [`DEFAULT_DASHBOARD_ADDR`]), or `None` if the dashboard is disabled (`off`).
fn dashboard_url() -> Option<String> {
    let raw = std::env::var("MEMEORA_DASHBOARD_ADDR")
        .unwrap_or_else(|_| DEFAULT_DASHBOARD_ADDR.to_string());
    if raw.is_empty() || raw.eq_ignore_ascii_case("off") {
        return None;
    }
    // A wildcard bind address isn't browsable — point the browser at loopback.
    let host = raw.replace("0.0.0.0", "127.0.0.1");
    Some(format!("http://{host}"))
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

    // `scope` and `adapter` are pure local commands — handle them before the daemon.
    if let Command::Scope { path } = &cli.command {
        let path = match path {
            Some(p) => p.clone(),
            None => std::env::current_dir()?.display().to_string(),
        };
        println!("{}", project_tag(&path));
        return Ok(());
    }
    if let Command::Adapter { cmd } = &cli.command {
        match cmd {
            AdapterCmd::New { name, dir } => scaffold_adapter(name, dir.as_deref())?,
        }
        return Ok(());
    }
    if let Command::Models { cmd } = &cli.command {
        return run_models(cmd);
    }

    let socket = cli.socket.unwrap_or_else(|| DEFAULT_SOCKET.to_string());
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
                    println!("the dashboard is disabled (MEMEORA_DASHBOARD_ADDR=off on the daemon)")
                }
            }
        }
        // Handled above before the daemon connection.
        Command::Scope { .. } | Command::Adapter { .. } | Command::Models { .. } => {
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
}
