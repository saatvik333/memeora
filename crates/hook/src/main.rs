//! `memeora-hook` — one binary invoked by AI coding tools' command-hooks.
//!
//! `--host` selects the per-tool conventions; Claude Code and Codex share the
//! `SessionStart.additionalContext` injection format and a turn-end `Stop` event,
//! so they take the same path here. The hook reads the host's JSON payload on
//! stdin and:
//! - **session-start** → injects the scope's profile as `additionalContext`;
//! - **stop** → captures the transcript into memory (best-effort, async upstream).
//!
//! Everything here is best-effort: if the daemon is down, the hook stays silent
//! rather than disrupting the host. The transcript schema is parsed defensively
//! and should be validated against real per-host fixtures before relying on it.

use std::error::Error;
use std::io::Read;

use clap::{Parser, ValueEnum};
use memeora_client::Client;
use memeora_core::container_tag::project_tag;
use memeora_proto::DEFAULT_SOCKET;
use serde_json::Value;

#[derive(Parser)]
#[command(name = "memeora-hook", version, about = "memeora command-hook adapter")]
struct Args {
    /// Which host invoked the hook.
    #[arg(long, value_enum)]
    host: Host,
    /// Which lifecycle event this invocation handles.
    #[arg(long, value_enum)]
    event: Event,
    /// Daemon socket name/path (defaults to the built-in name).
    #[arg(long)]
    socket: Option<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Host {
    Claude,
    Codex,
}

#[derive(Clone, Copy, ValueEnum)]
enum Event {
    SessionStart,
    Stop,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let _ = args.host; // Claude & Codex share this path today; reserved for divergence.

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let payload: Value = serde_json::from_str(&input).unwrap_or(Value::Null);

    let scope = scope_from_payload(&payload);
    let socket = args.socket.unwrap_or_else(|| DEFAULT_SOCKET.to_string());

    match args.event {
        Event::SessionStart => {
            if let Some(context) = fetch_context(&socket, &scope) {
                print!("{}", render_session_start(&context));
            }
        }
        Event::Stop => {
            if let Some(path) = payload.get("transcript_path").and_then(Value::as_str)
                && let Ok(jsonl) = std::fs::read_to_string(path)
            {
                let text = transcript_to_text(&jsonl, 40);
                if !text.trim().is_empty() {
                    // Best-effort capture; ignore daemon errors.
                    let _ = Client::connect(&socket).and_then(|mut c| {
                        c.ingest(&scope, &text)?;
                        Ok(())
                    });
                }
            }
        }
    }
    Ok(())
}

/// Determine the project scope from the payload's `cwd` (or the process cwd).
fn scope_from_payload(payload: &Value) -> String {
    let cwd = payload
        .get("cwd")
        .and_then(Value::as_str)
        .map(String::from)
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|p| p.display().to_string())
        })
        .unwrap_or_default();
    project_tag(&cwd)
}

/// Fetch a scope's profile as injectable text, or `None` if empty/unreachable.
fn fetch_context(socket: &str, scope: &str) -> Option<String> {
    let mut client = Client::connect(socket).ok()?;
    let (statics, dynamics) = client.context(scope).ok()?;
    if statics.is_empty() && dynamics.is_empty() {
        return None;
    }
    let mut out = String::from(
        "The following is persistent memory about this user/project. Use it naturally; don't force it.\n",
    );
    for m in statics.iter().chain(dynamics.iter()) {
        out.push_str(&format!("- [{}] {}\n", m.kind, m.content));
    }
    Some(out)
}

/// Render the host's session-start context injection (Claude/Codex shared format).
fn render_session_start(context: &str) -> String {
    serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    })
    .to_string()
}

/// Extract the last `max_turns` user/assistant turns from a transcript JSONL into
/// compact `role: text` lines. Defensive: unknown lines/shapes are skipped.
fn transcript_to_text(jsonl: &str, max_turns: usize) -> String {
    let mut turns = Vec::new();
    for line in jsonl.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let role = value
            .get("message")
            .and_then(|m| m.get("role"))
            .and_then(Value::as_str)
            .or_else(|| value.get("role").and_then(Value::as_str));
        let Some(role) = role else { continue };
        if role != "user" && role != "assistant" {
            continue;
        }
        let text = extract_text(&value);
        if text.trim().is_empty() {
            continue;
        }
        turns.push(format!("{role}: {text}"));
    }
    let start = turns.len().saturating_sub(max_turns);
    turns[start..].join("\n")
}

/// Pull text from a transcript entry: `message.content` (or `content`) as a string
/// or an array of `{text: ...}` content blocks.
fn extract_text(value: &Value) -> String {
    let content = value
        .get("message")
        .and_then(|m| m.get("content"))
        .or_else(|| value.get("content"));
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_uses_cwd_from_payload() {
        let payload = serde_json::json!({ "cwd": "/home/u/proj" });
        assert_eq!(scope_from_payload(&payload), project_tag("/home/u/proj"));
    }

    #[test]
    fn session_start_injection_has_additional_context() {
        let out = render_session_start("hello memory");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "SessionStart");
        assert_eq!(v["hookSpecificOutput"]["additionalContext"], "hello memory");
    }

    #[test]
    fn transcript_extracts_roles_and_text() {
        let jsonl = [
            r#"{"message":{"role":"user","content":"I prefer rust"}}"#,
            r#"{"message":{"role":"assistant","content":[{"type":"text","text":"noted"}]}}"#,
            r#"{"type":"system","message":{"role":"system","content":"ignore me"}}"#,
            r#"not json"#,
        ]
        .join("\n");
        let text = transcript_to_text(&jsonl, 40);
        assert_eq!(text, "user: I prefer rust\nassistant: noted");
    }

    #[test]
    fn transcript_keeps_only_last_n_turns() {
        let lines: Vec<String> = (0..10)
            .map(|i| format!(r#"{{"role":"user","content":"m{i}"}}"#))
            .collect();
        let text = transcript_to_text(&lines.join("\n"), 3);
        assert_eq!(text, "user: m7\nuser: m8\nuser: m9");
    }
}
