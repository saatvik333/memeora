//! `memeora-hook` library: the host-agnostic engine behind the binary.
//!
//! All host-specific behavior is expressed as data (a [`HostDescriptor`]); the
//! functions here read the stdin payload and render stdout *through* a descriptor,
//! so they're pure and fully testable without a daemon (see the conformance kit in
//! `tests/`). [`run`] wires these to stdin/stdout and the IPC client; the shipped
//! `memeora-hook` binary is a thin wrapper around it.

pub mod descriptor;
pub mod run;

use memeora_core::container_tag::project_tag;
use memeora_proto::MemoryDto;
use serde_json::Value;

pub use descriptor::{HostDescriptor, InjectStyle};
pub use run::run;

/// Resolve a dotted payload path: object keys, with numeric segments indexing
/// arrays (e.g. `workspacePaths.0`). Returns `None` if any segment is missing.
fn get_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for segment in path.split('.') {
        current = match segment.parse::<usize>() {
            Ok(index) => current.get(index)?,
            Err(_) => current.get(segment)?,
        };
    }
    Some(current)
}

/// First string value among `fields`, by dotted path.
fn first_field<'a>(payload: &'a Value, fields: &[String]) -> Option<&'a str> {
    fields
        .iter()
        .find_map(|f| get_path(payload, f).and_then(Value::as_str))
}

/// Determine the project scope tag from the payload per the descriptor, falling
/// back to the process cwd when no configured field resolves.
pub fn resolve_scope(desc: &HostDescriptor, payload: &Value) -> String {
    let dir = first_field(payload, &desc.scope_fields)
        .map(String::from)
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|p| p.display().to_string())
        })
        .unwrap_or_default();
    project_tag(&dir)
}

/// The transcript file path per the descriptor, if present.
pub fn transcript_path(desc: &HostDescriptor, payload: &Value) -> Option<String> {
    first_field(payload, &desc.transcript_fields).map(String::from)
}

/// Whether an inject event should actually inject. Hosts with an
/// `invocation_gate_field` (e.g. Antigravity's `invocationNum`) inject only on the
/// first invocation; others always inject. A missing field defaults to injecting.
pub fn should_inject(desc: &HostDescriptor, payload: &Value) -> bool {
    match &desc.invocation_gate_field {
        Some(field) => get_path(payload, field)
            .and_then(Value::as_u64)
            .map(|n| n == 1)
            .unwrap_or(true),
        None => true,
    }
}

/// Render a context injection in the descriptor's stdout style.
pub fn render_inject(desc: &HostDescriptor, context: &str) -> String {
    match desc.inject_style {
        InjectStyle::InjectSteps => serde_json::json!({
            "injectSteps": [ { "userMessage": context } ],
        })
        .to_string(),
        InjectStyle::AdditionalContext => serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": desc.inject_event_name.as_deref().unwrap_or("SessionStart"),
                "additionalContext": context,
            }
        })
        .to_string(),
    }
}

/// The stdout a capture event must emit for this host, or `None` if it needs none.
pub fn capture_ack(desc: &HostDescriptor) -> Option<String> {
    (!desc.capture_ack.trim().is_empty()).then(|| desc.capture_ack.clone())
}

/// Format a scope's profile into injectable text, or `None` if it's empty.
pub fn format_context(statics: &[MemoryDto], dynamics: &[MemoryDto]) -> Option<String> {
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

/// Extract the last `max_turns` user/assistant turns from a transcript JSONL into
/// compact `role: text` lines. Defensive: unknown lines/shapes are skipped.
pub fn transcript_to_text(jsonl: &str, max_turns: usize) -> String {
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
/// or an array of `{type:"text", text: ...}` blocks (tool/thinking blocks skipped).
pub fn extract_text(value: &Value) -> String {
    let content = value
        .get("message")
        .and_then(|m| m.get("content"))
        .or_else(|| value.get("content"));
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Best-effort redaction of obvious secrets before a transcript is persisted.
///
/// Conservative and heuristic (not a substitute for the user not pasting secrets):
/// masks known credential-prefixed tokens, `key=value`/`key: value` pairs with a
/// sensitive key, and long high-entropy blobs.
pub fn redact(text: &str) -> String {
    text.lines()
        .map(|line| {
            // Split on ALL whitespace (not just ' '): a tab-indented or
            // tab-separated token must still be inspected.
            line.split_whitespace()
                .map(redact_word)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Sensitive key substrings for `key=value` redaction.
const SENSITIVE_KEYS: &[&str] = &[
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "auth",
];

/// Known secret token prefixes (GitHub, OpenAI, AWS, Google, Slack, GitLab).
const SECRET_PREFIXES: &[&str] = &[
    "sk-",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "AKIA",
    "AIza",
    "glpat-",
];

/// Redact a single whitespace-delimited word, preserving non-secret text.
///
/// Handles tokens wrapped in punctuation (quotes, brackets, trailing commas) by
/// inspecting the alphanumeric core, so `"sk-…",` and a tab-indented `\tsk-…` are
/// caught — not just a bare space-delimited token.
fn redact_word(word: &str) -> String {
    // `key=value` / `key: value` with a sensitive key → mask only the value.
    if let Some(idx) = word.find(['=', ':']) {
        let (key, rest) = word.split_at(idx);
        let value = &rest[1..];
        let key_core = key
            .trim_matches(|c: char| !c.is_ascii_alphanumeric())
            .to_ascii_lowercase();
        if !value.is_empty() && SENSITIVE_KEYS.iter().any(|s| key_core.contains(s)) {
            // `rest` starts with the (1-byte ASCII) separator we matched.
            return format!("{key}{}[REDACTED]", &rest[..1]);
        }
    }
    // Strip surrounding non-alphanumerics before the prefix/entropy check, then
    // redact just the core so wrapping punctuation (and thus JSON shape) survives.
    let core = word.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    if !core.is_empty() && looks_secret(core) {
        return word.replace(core, "[REDACTED]");
    }
    word.to_string()
}

/// Whether a standalone token looks like a credential.
fn looks_secret(word: &str) -> bool {
    if SECRET_PREFIXES.iter().any(|p| word.starts_with(p)) {
        return true;
    }
    word.len() >= 32
        && word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_+/=.".contains(c))
        && word.chars().any(|c| c.is_ascii_alphabetic())
        && word.chars().any(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use descriptor::builtin;

    fn claude() -> HostDescriptor {
        builtin("claude").unwrap()
    }
    fn antigravity() -> HostDescriptor {
        builtin("antigravity").unwrap()
    }

    #[test]
    fn scope_uses_cwd_for_claude() {
        let payload = serde_json::json!({ "cwd": "/home/u/proj" });
        assert_eq!(
            resolve_scope(&claude(), &payload),
            project_tag("/home/u/proj")
        );
    }

    #[test]
    fn scope_uses_workspace_paths_for_antigravity() {
        let payload = serde_json::json!({ "workspacePaths": ["/home/u/proj", "/tmp/x"] });
        assert_eq!(
            resolve_scope(&antigravity(), &payload),
            project_tag("/home/u/proj")
        );
    }

    #[test]
    fn claude_injection_has_additional_context() {
        let out = render_inject(&claude(), "hello memory");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "SessionStart");
        assert_eq!(v["hookSpecificOutput"]["additionalContext"], "hello memory");
    }

    #[test]
    fn antigravity_injection_uses_inject_steps() {
        let out = render_inject(&antigravity(), "hello memory");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["injectSteps"][0]["userMessage"], "hello memory");
    }

    #[test]
    fn pre_invocation_injects_only_on_first() {
        let anti = antigravity();
        assert!(should_inject(
            &anti,
            &serde_json::json!({ "invocationNum": 1 })
        ));
        assert!(!should_inject(
            &anti,
            &serde_json::json!({ "invocationNum": 2 })
        ));
        // Missing gate field → default to injecting.
        assert!(should_inject(&anti, &serde_json::json!({})));
        // Claude has no gate → always injects.
        assert!(should_inject(&claude(), &serde_json::json!({})));
    }

    #[test]
    fn antigravity_uses_camel_case_transcript_path() {
        let payload = serde_json::json!({ "transcriptPath": "/t/a.jsonl" });
        assert_eq!(
            transcript_path(&antigravity(), &payload).as_deref(),
            Some("/t/a.jsonl")
        );
    }

    #[test]
    fn capture_ack_is_valid_json_per_host() {
        let anti: Value = serde_json::from_str(&capture_ack(&antigravity()).unwrap()).unwrap();
        assert_eq!(anti["decision"], "stop");
        let codex: Value =
            serde_json::from_str(&capture_ack(&builtin("codex").unwrap()).unwrap()).unwrap();
        assert!(codex.is_object());
    }

    #[test]
    fn nested_array_path_resolves() {
        let payload = serde_json::json!({ "a": { "b": ["x", "y"] } });
        assert_eq!(
            get_path(&payload, "a.b.1").and_then(Value::as_str),
            Some("y")
        );
        assert!(get_path(&payload, "a.b.9").is_none());
        assert!(get_path(&payload, "missing").is_none());
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
        assert_eq!(
            transcript_to_text(&jsonl, 40),
            "user: I prefer rust\nassistant: noted"
        );
    }

    #[test]
    fn redact_masks_secrets_but_keeps_prose() {
        let out = redact("deploy with key sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 today");
        assert!(out.contains("[REDACTED]"));
        assert!(out.contains("deploy with key"));
        assert!(out.contains("today"));
        assert!(!out.contains("sk-ABCDEFG"));

        let kv = redact("set password=hunter2supersecret in config");
        assert!(kv.contains("password=[REDACTED]"));
        assert!(kv.contains("in config"));

        assert_eq!(redact("I prefer dark mode"), "I prefer dark mode");
    }

    #[test]
    fn redact_catches_tokens_wrapped_in_punctuation_or_tabs() {
        // JSON-embedded (quotes + trailing comma) — the common real-world shape.
        let json = redact("\"token\": \"sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345\",");
        assert!(!json.contains("sk-ABCDEFG"), "quoted token leaked: {json}");
        assert!(json.contains("[REDACTED]"));

        // Tab-indented token (split_whitespace must see it).
        let tabbed = redact("\tghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
        assert!(
            !tabbed.contains("ghp_ABCDEFG"),
            "tabbed token leaked: {tabbed}"
        );
        assert!(tabbed.contains("[REDACTED]"));

        // High-entropy blob inside parentheses.
        let paren = redact("(AKIAIOSFODNN7EXAMPLEKEY1234567890)");
        assert!(
            paren.contains("[REDACTED]"),
            "paren-wrapped token leaked: {paren}"
        );

        // Ordinary punctuated prose is untouched (no false positives).
        assert_eq!(redact("done, thanks!"), "done, thanks!");
    }

    #[test]
    fn extract_text_skips_non_text_blocks() {
        let value = serde_json::json!({
            "message": { "role": "assistant", "content": [
                { "type": "text", "text": "hello" },
                { "type": "tool_use", "text": "secret tool input" },
                { "type": "tool_result", "text": "file dump" },
            ]}
        });
        assert_eq!(extract_text(&value), "hello");
    }

    #[test]
    fn transcript_keeps_only_last_n_turns() {
        let lines: Vec<String> = (0..10)
            .map(|i| format!(r#"{{"role":"user","content":"m{i}"}}"#))
            .collect();
        assert_eq!(
            transcript_to_text(&lines.join("\n"), 3),
            "user: m7\nuser: m8\nuser: m9"
        );
    }

    #[test]
    fn format_context_is_none_when_empty() {
        assert!(format_context(&[], &[]).is_none());
    }
}
