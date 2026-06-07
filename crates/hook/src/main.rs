//! `memeora-hook` — one binary invoked by AI coding tools' command-hooks.
//!
//! `--host` selects per-tool conventions (a parser for the stdin payload and a
//! renderer for stdout), because the hosts do *not* share an event schema:
//!
//! - **claude / codex** share a snake_case payload, inject context at
//!   `SessionStart` via `hookSpecificOutput.additionalContext`, and capture the
//!   transcript at `Stop`/`PreCompact` (Codex requires JSON on stdout, so we ack
//!   with `{}`).
//! - **antigravity** uses its own camelCase schema: there is no session-start
//!   event, so context is injected at `PreInvocation` (gated to the first
//!   invocation) via `injectSteps`, scope comes from `workspacePaths`, the
//!   transcript path is `transcriptPath`, and `Stop` must return a `decision`.
//!
//! Everything is best-effort: if the daemon is down the hook stays silent rather
//! than disrupting the host. Payloads are parsed defensively and should be
//! validated against real per-host fixtures before relying on them.

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
    /// Which host invoked the hook (selects the payload/render conventions).
    #[arg(long, value_enum)]
    host: Host,
    /// Which lifecycle event this invocation handles.
    #[arg(long, value_enum)]
    event: Event,
    /// Daemon socket name/path (defaults to the built-in name).
    #[arg(long)]
    socket: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Host {
    /// Claude Code — snake_case payload, `hookSpecificOutput.additionalContext`.
    Claude,
    /// OpenAI Codex — same payload/inject format as Claude.
    Codex,
    /// Google Antigravity — camelCase payload, `injectSteps`, `decision`.
    Antigravity,
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

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let payload: Value = serde_json::from_str(&input).unwrap_or(Value::Null);

    let scope = scope_from_payload(args.host, &payload);
    let socket = args.socket.unwrap_or_else(|| DEFAULT_SOCKET.to_string());

    match args.event {
        Event::SessionStart | Event::PreInvocation => {
            if should_inject(args.event, &payload)
                && let Some(context) = fetch_context(&socket, &scope)
            {
                print!("{}", render_inject(args.host, &context));
            }
        }
        Event::Stop | Event::PreCompact => {
            capture(&socket, &scope, args.host, &payload);
            if let Some(ack) = capture_ack(args.host) {
                print!("{ack}");
            }
        }
    }
    Ok(())
}

/// Whether an inject event should actually inject. Antigravity's `PreInvocation`
/// fires before *every* model call, so we only inject on the first one; the
/// session-start events fire once and always inject.
fn should_inject(event: Event, payload: &Value) -> bool {
    match event {
        Event::PreInvocation => payload
            .get("invocationNum")
            .and_then(Value::as_u64)
            .map(|n| n == 1)
            .unwrap_or(true),
        _ => true,
    }
}

/// Determine the project scope from the payload, then the process cwd. Claude and
/// Codex carry `cwd`; Antigravity carries `workspacePaths` (first entry).
fn scope_from_payload(host: Host, payload: &Value) -> String {
    let from_host = match host {
        Host::Antigravity => payload
            .get("workspacePaths")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .map(String::from),
        Host::Claude | Host::Codex => payload.get("cwd").and_then(Value::as_str).map(String::from),
    };
    let cwd = from_host
        // Fall back to the other convention, then the process cwd.
        .or_else(|| payload.get("cwd").and_then(Value::as_str).map(String::from))
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

/// Render a context injection in the host's expected stdout format.
fn render_inject(host: Host, context: &str) -> String {
    match host {
        Host::Antigravity => serde_json::json!({
            "injectSteps": [ { "userMessage": context } ],
        })
        .to_string(),
        Host::Claude | Host::Codex => serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context,
            }
        })
        .to_string(),
    }
}

/// Capture the host's transcript into memory (best-effort; daemon errors ignored).
fn capture(socket: &str, scope: &str, host: Host, payload: &Value) {
    let Some(path) = transcript_path(host, payload) else {
        return;
    };
    let Ok(jsonl) = std::fs::read_to_string(&path) else {
        return;
    };
    let text = transcript_to_text(&jsonl, 40);
    if text.trim().is_empty() {
        return;
    }
    let _ = Client::connect(socket).and_then(|mut c| {
        c.ingest(scope, &text)?;
        Ok(())
    });
}

/// The transcript path field for the host (with a defensive fallback to the
/// other naming convention).
fn transcript_path(host: Host, payload: &Value) -> Option<String> {
    let primary = match host {
        Host::Antigravity => "transcriptPath",
        Host::Claude | Host::Codex => "transcript_path",
    };
    payload
        .get(primary)
        .and_then(Value::as_str)
        .or_else(|| payload.get("transcript_path").and_then(Value::as_str))
        .or_else(|| payload.get("transcriptPath").and_then(Value::as_str))
        .map(String::from)
}

/// Stdout a capture event must emit, if the host requires valid JSON even when
/// the hook only has side effects.
fn capture_ack(host: Host) -> Option<String> {
    match host {
        // Antigravity's `Stop` requires a `decision`; any value other than
        // "continue" lets the turn end normally. (Verify against a fixture.)
        Host::Antigravity => Some(r#"{"decision":"stop"}"#.to_string()),
        // Codex rejects non-JSON stdout from `Stop`; Claude tolerates `{}`.
        Host::Claude | Host::Codex => Some("{}".to_string()),
    }
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
    fn scope_uses_cwd_for_claude() {
        let payload = serde_json::json!({ "cwd": "/home/u/proj" });
        assert_eq!(
            scope_from_payload(Host::Claude, &payload),
            project_tag("/home/u/proj")
        );
    }

    #[test]
    fn scope_uses_workspace_paths_for_antigravity() {
        let payload = serde_json::json!({ "workspacePaths": ["/home/u/proj", "/tmp/x"] });
        assert_eq!(
            scope_from_payload(Host::Antigravity, &payload),
            project_tag("/home/u/proj")
        );
    }

    #[test]
    fn claude_injection_has_additional_context() {
        let out = render_inject(Host::Claude, "hello memory");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "SessionStart");
        assert_eq!(v["hookSpecificOutput"]["additionalContext"], "hello memory");
    }

    #[test]
    fn antigravity_injection_uses_inject_steps() {
        let out = render_inject(Host::Antigravity, "hello memory");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["injectSteps"][0]["userMessage"], "hello memory");
    }

    #[test]
    fn pre_invocation_injects_only_on_first() {
        assert!(should_inject(
            Event::PreInvocation,
            &serde_json::json!({ "invocationNum": 1 })
        ));
        assert!(!should_inject(
            Event::PreInvocation,
            &serde_json::json!({ "invocationNum": 2 })
        ));
        // Missing field → default to injecting.
        assert!(should_inject(Event::PreInvocation, &serde_json::json!({})));
        // Session-start always injects.
        assert!(should_inject(Event::SessionStart, &serde_json::json!({})));
    }

    #[test]
    fn antigravity_uses_camel_case_transcript_path() {
        let payload = serde_json::json!({ "transcriptPath": "/t/a.jsonl" });
        assert_eq!(
            transcript_path(Host::Antigravity, &payload).as_deref(),
            Some("/t/a.jsonl")
        );
    }

    #[test]
    fn capture_ack_is_valid_json_per_host() {
        let anti: Value = serde_json::from_str(&capture_ack(Host::Antigravity).unwrap()).unwrap();
        assert_eq!(anti["decision"], "stop");
        let codex: Value = serde_json::from_str(&capture_ack(Host::Codex).unwrap()).unwrap();
        assert!(codex.is_object());
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
