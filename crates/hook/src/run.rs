//! The `memeora-hook` binary entrypoint, as a library function.
//!
//! Lives in the library (not a `main.rs`) so the single shipped `memeora` package
//! can expose every binary from one crate (see `docs/ARCHITECTURE.md`, Step 10).
//! Everything is best-effort: if the daemon is down the hook stays silent rather
//! than disrupting the host, and a stdin read failure never makes it exit non-zero.

use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use memeora_client::Client;
use memeora_proto::DEFAULT_SOCKET;
use serde_json::Value;

use crate::descriptor::{self, HostDescriptor};
use crate::{
    capture_ack, format_context, render_inject, resolve_scope, sanitize, session_capture,
    should_inject, transcript_path,
};

#[derive(Parser)]
#[command(name = "memeora-hook", version, about = "memeora command-hook adapter")]
struct Args {
    /// Built-in host (claude | codex | antigravity). Selects its descriptor.
    #[arg(long)]
    host: Option<String>,
    /// Path to a custom host-descriptor TOML (overrides `--host`).
    #[arg(long)]
    descriptor: Option<PathBuf>,
    /// Which lifecycle event this invocation handles.
    #[arg(long, value_enum)]
    event: Event,
    /// Daemon socket name/path (defaults to the built-in name).
    #[arg(long)]
    socket: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Event {
    /// Claude/Codex session start → inject profile.
    SessionStart,
    /// Antigravity per-invocation hook → inject profile on the first invocation.
    PreInvocation,
    /// Turn end → capture the transcript.
    Stop,
    /// Before context compaction → capture the transcript (last chance).
    PreCompact,
}

impl Event {
    /// Inject events fetch+render context; capture events read the transcript.
    fn is_inject(self) -> bool {
        matches!(self, Event::SessionStart | Event::PreInvocation)
    }
}

/// Resolve the descriptor from `--descriptor` (a file) or `--host` (a built-in).
fn resolve_descriptor(args: &Args) -> Result<HostDescriptor, Box<dyn Error>> {
    if let Some(path) = &args.descriptor {
        return Ok(descriptor::load(path)?);
    }
    if let Some(host) = &args.host {
        return descriptor::builtin(host).ok_or_else(|| {
            format!(
                "unknown built-in host {host:?} (known: {}); pass --descriptor <path> for a custom host",
                descriptor::BUILTIN_HOSTS.join(", ")
            )
            .into()
        });
    }
    Err("one of --host or --descriptor is required".into())
}

/// Parse args, read the stdin payload, and inject or capture per the descriptor.
pub fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let desc = resolve_descriptor(&args)?;

    // Read stdin best-effort: a failure (broken pipe, non-UTF-8) must not make the
    // hook exit non-zero, which some hosts treat as a hard error.
    let mut input = String::new();
    let _ = std::io::stdin().read_to_string(&mut input);
    let payload: Value = serde_json::from_str(&input).unwrap_or(Value::Null);

    let scope = resolve_scope(&desc, &payload);
    let socket = args.socket.unwrap_or_else(|| DEFAULT_SOCKET.to_string());

    if args.event.is_inject() {
        if should_inject(&desc, &payload)
            && let Some(context) = fetch_context(&socket, &scope)
        {
            print!("{}", render_inject(&desc, &context));
        }
    } else {
        capture(&socket, &scope, &desc, &payload);
        if let Some(ack) = capture_ack(&desc) {
            print!("{ack}");
        }
    }
    Ok(())
}

/// Fetch a scope's profile as injectable text, or `None` if empty/unreachable.
fn fetch_context(socket: &str, scope: &str) -> Option<String> {
    let mut client = Client::connect(socket).ok()?;
    let (statics, dynamics) = client.context(scope).ok()?;
    format_context(&statics, &dynamics)
}

/// Capture the host's transcript into memory (best-effort; daemon errors ignored).
fn capture(socket: &str, scope: &str, desc: &HostDescriptor, payload: &Value) {
    let Some(path) = transcript_path(desc, payload) else {
        return;
    };
    let Ok(jsonl) = std::fs::read_to_string(&path) else {
        return;
    };
    // Full sanitize (strip <private> + redact) here too, not just at the engine:
    // defense-in-depth, idempotent, and it keeps capture testable without a daemon.
    // session_capture derives file/command activity (not raw tool output), so the
    // session's work is understood without storing attacker-influenceable dumps.
    let text = sanitize(&session_capture(&jsonl, 40));
    if text.trim().is_empty() {
        return;
    }
    let _ = Client::connect(socket).and_then(|mut c| {
        c.ingest(scope, &text)?;
        Ok(())
    });
}
