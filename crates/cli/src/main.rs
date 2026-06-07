//! memeora CLI — a thin client over the daemon for inspecting and editing memory.
//!
//! Talks to a running `memeora-daemon` over the local socket. (Lifecycle commands
//! like `serve`/`install`/`dashboard` land in later steps; today this is the
//! query/edit surface.)

use std::error::Error;

use clap::{Parser, Subcommand};
use memeora_client::Client;
use memeora_core::container_tag::project_tag;
use memeora_proto::DEFAULT_SOCKET;

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
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // `scope` is a pure local computation — handle it before touching the daemon.
    if let Command::Scope { path } = &cli.command {
        let path = match path {
            Some(p) => p.clone(),
            None => std::env::current_dir()?.display().to_string(),
        };
        println!("{}", project_tag(&path));
        return Ok(());
    }

    let socket = cli.socket.unwrap_or_else(|| DEFAULT_SOCKET.to_string());
    let mut client = Client::connect(&socket).map_err(|e| {
        format!("cannot reach the daemon at {socket}: {e}\nis `memeora-daemon` running?")
    })?;

    match cli.command {
        Command::Doctor => {
            let version = client.hello()?;
            println!("daemon ok — protocol v{version} (socket {socket})");
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
        // Handled above before the daemon connection.
        Command::Scope { .. } => unreachable!("scope is resolved before connecting"),
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
