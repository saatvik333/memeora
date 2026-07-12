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
// Redaction is owned by the engine (core); re-export so the hook's capture path
// and the public API keep a single, canonical, whitespace-preserving implementation.
pub use memeora_core::privacy::{redact, sanitize};
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

/// The transcript file path per the descriptor, if present **and safe to read**.
///
/// The path is taken from the host payload using field names declared in a
/// (possibly untrusted, `--descriptor`-supplied) TOML, then `fs::read_to_string`d
/// in `capture`. Restrict it to a `.jsonl` file with no `..` traversal so a hostile
/// descriptor can't turn the capture hook into an arbitrary-file reader.
pub fn transcript_path(desc: &HostDescriptor, payload: &Value) -> Option<String> {
    let path = first_field(payload, &desc.transcript_fields)?;
    is_safe_transcript_path(path).then(|| path.to_string())
}

/// A transcript path is safe to read iff it is a `.jsonl` file with no parent-dir
/// traversal. Conservative on purpose: capture is best-effort, so an unsafe path
/// yields no capture rather than an error.
fn is_safe_transcript_path(path: &str) -> bool {
    !path.contains("..") && path.to_ascii_lowercase().ends_with(".jsonl")
}

/// Whether an inject event should actually inject. Hosts with an
/// `invocation_gate_field` (e.g. Antigravity's `invocationNum`) inject only on the
/// first invocation; others always inject. A missing field defaults to injecting.
pub fn should_inject(desc: &HostDescriptor, payload: &Value) -> bool {
    match &desc.invocation_gate_field {
        // Accept the gate value as an integer OR a float: some hosts emit
        // `invocationNum` as `1.0`, which `as_u64` alone would miss → the gate would
        // fail open and inject on every turn.
        Some(field) => get_path(payload, field)
            .map(|v| v.as_u64() == Some(1) || v.as_f64() == Some(1.0))
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

/// Marker line [`format_context`] opens every injected memory block with.
/// [`strip_injected_memory`] keys on this same constant, so the injector and the
/// stripper cannot drift apart.
pub const INJECT_PREAMBLE: &str =
    "The following is persistent memory about this user/project. Use it naturally; don't force it.";

/// Trailing line [`format_context`] appends when the wake-up profile overflows
/// [`WAKE_TOKEN_BUDGET`]. Like [`INJECT_PREAMBLE`], [`strip_injected_memory`] keys on
/// this same constant so the injector and stripper can't drift.
pub const INJECT_MORE_MARKER: &str = "…(more via recall)";

/// Token ceiling for the SessionStart wake-up injection. A profile can grow without
/// bound, but the injected block must stay small so it doesn't crowd the host's
/// context window — the rest stays reachable via the `recall` tool.
const WAKE_TOKEN_BUDGET: usize = 800;

/// Rough token estimate (~4 chars/token), mirroring the engine's budget heuristic
/// (`memeora_core::search::est_tokens`). Kept local so the hook stays daemon-free.
fn est_tokens(text: &str) -> usize {
    text.chars().count() / 4 + 1
}

/// Format a scope's profile into injectable text, or `None` if it's empty.
///
/// Bounded to [`WAKE_TOKEN_BUDGET`] tokens: statics before dynamics (importance-first,
/// as `build_profile` already orders them), stopping before the budget is exceeded and
/// appending an [`INJECT_MORE_MARKER`] line when anything was dropped.
pub fn format_context(statics: &[MemoryDto], dynamics: &[MemoryDto]) -> Option<String> {
    if statics.is_empty() && dynamics.is_empty() {
        return None;
    }
    let mut out = format!("{INJECT_PREAMBLE}\n");
    let mut tokens = est_tokens(&out);
    let mut truncated = false;
    for m in statics.iter().chain(dynamics.iter()) {
        let line = format!("- [{}] {}\n", m.kind, m.content);
        let line_tokens = est_tokens(&line);
        if tokens + line_tokens > WAKE_TOKEN_BUDGET {
            truncated = true;
            break;
        }
        out.push_str(&line);
        tokens += line_tokens;
    }
    if truncated {
        out.push_str(INJECT_MORE_MARKER);
        out.push('\n');
    }
    Some(out)
}

/// Extract the last `max_turns` user/assistant turns from a transcript JSONL into
/// compact `role: text` lines. Defensive: unknown lines/shapes are skipped.
///
/// ponytail: the whole capture pipeline (this, `extract_text`, `collect_activity`)
/// understands only Claude's Anthropic-Messages JSONL shape (`message.role` /
/// `message.content` blocks, tool names Edit/Write/Bash). Codex's descriptor claims
/// the same shape (see `adapters/_descriptors/codex.toml` and the codex conformance
/// fixture); a host whose transcripts nest differently gets a graceful empty capture
/// (pinned by `fixtures/antigravity/capture-unknown-shape.json`), never a crash.
/// Grow per-host transcript schemas only when a real host actually diverges.
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
        let text = strip_injected_memory(&extract_text(&value));
        if text.trim().is_empty() {
            continue;
        }
        turns.push(format!("{role}: {text}"));
    }
    let start = turns.len().saturating_sub(max_turns);
    turns[start..].join("\n")
}

/// Remove memeora's own injected memory block from captured text, so capture never
/// re-ingests what the hook itself injected (a feedback loop: past memories would be
/// extracted again as "new" and compound/echo across sessions).
///
/// Keys on the [`INJECT_PREAMBLE`] marker line emitted by [`format_context`] —
/// drops that line (injection can land mid-line when hosts join content blocks, so
/// any genuine prefix before the marker is kept) plus the immediately following run
/// of `- [kind] ...` bullet lines. Everything else, including genuine text that
/// merely mentions memory or contains bullets, passes through untouched.
pub fn strip_injected_memory(text: &str) -> String {
    if !text.contains(INJECT_PREAMBLE) {
        return text.to_string();
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        match line.find(INJECT_PREAMBLE) {
            Some(at) => {
                let prefix = line[..at].trim_end();
                if !prefix.is_empty() {
                    kept.push(prefix);
                }
                while lines
                    .next_if(|l| l.trim_start().starts_with("- ["))
                    .is_some()
                {}
                // Also drop the optional overflow marker that closes a bounded block,
                // so a truncated wake-up can't echo back through capture either.
                lines.next_if(|l| l.trim() == INJECT_MORE_MARKER);
            }
            None => kept.push(line),
        }
    }
    let mut out = kept.join("\n");
    if text.ends_with('\n') && !out.is_empty() {
        out.push('\n');
    }
    out
}

/// `message.content` (or top-level `content`) from a transcript entry, if present.
fn message_content(value: &Value) -> Option<&Value> {
    value
        .get("message")
        .and_then(|m| m.get("content"))
        .or_else(|| value.get("content"))
}

/// Pull text from a transcript entry: `message.content` (or `content`) as a string
/// or an array of `{type:"text", text: ...}` blocks (tool/thinking blocks skipped).
pub fn extract_text(value: &Value) -> String {
    let content = message_content(value);
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

/// Tools whose `file_path` input names a file the session created or edited.
const FILE_EDIT_TOOLS: &[&str] = &[
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    "Update",
    "Create",
];
/// Max files / commands surfaced in the activity summary, and per-command length.
const MAX_ACTIVITY: usize = 20;
const MAX_CMD_LEN: usize = 100;
/// Overall cap on captured text so a long session can't flood the embedder/store.
const MAX_CAPTURE_BYTES: usize = 8192;

/// Build the capture text for a session: a compact, derived summary of the *work*
/// (files edited, commands run) followed by the recent user/assistant turns.
///
/// Only **derived signals** are taken from tool calls — file paths from edit tools
/// and the command string from `Bash` — never tool *result* bodies. Raw command
/// output, fetched pages, and read file contents are attacker-influenceable, so
/// storing them verbatim would let a hostile repo seed "memory"; deriving the
/// structure of the work keeps what's captured to what the session actually did.
pub fn session_capture(jsonl: &str, max_turns: usize) -> String {
    let mut files: Vec<String> = Vec::new();
    let mut commands: Vec<String> = Vec::new();
    for line in jsonl.lines() {
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            collect_activity(&value, &mut files, &mut commands);
        }
    }
    dedup_in_place(&mut files);
    dedup_in_place(&mut commands);

    let mut sections: Vec<String> = Vec::new();
    if !files.is_empty() {
        let shown = files
            .iter()
            .take(MAX_ACTIVITY)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        sections.push(format!("edited files {shown}"));
    }
    for cmd in commands.iter().take(MAX_ACTIVITY) {
        sections.push(format!("ran command {cmd}"));
    }
    let turns = transcript_to_text(jsonl, max_turns);
    if !turns.is_empty() {
        sections.push(turns);
    }

    let mut out = sections.join("\n");
    if out.len() > MAX_CAPTURE_BYTES {
        let mut end = MAX_CAPTURE_BYTES;
        while !out.is_char_boundary(end) {
            end -= 1;
        }
        out.truncate(end);
    }
    out
}

/// Pull derived activity (edited files, run commands) from one transcript entry's
/// `tool_use` blocks. Unknown tools are ignored, so arbitrary tool payloads — which
/// can carry secrets or attacker-injected text — are never surfaced.
fn collect_activity(value: &Value, files: &mut Vec<String>, commands: &mut Vec<String>) {
    let Some(Value::Array(blocks)) = message_content(value) else {
        return;
    };
    for block in blocks {
        if block.get("type").and_then(Value::as_str) != Some("tool_use") {
            continue;
        }
        let name = block
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let input = block.get("input");
        if FILE_EDIT_TOOLS.contains(&name) {
            if let Some(path) = input
                .and_then(|i| i.get("file_path"))
                .and_then(Value::as_str)
            {
                files.push(path.to_string());
            }
        } else if name == "Bash"
            && let Some(cmd) = input.and_then(|i| i.get("command")).and_then(Value::as_str)
        {
            commands.push(truncate_chars(cmd, MAX_CMD_LEN));
        }
    }
}

/// Drop duplicate entries, keeping first-seen order.
fn dedup_in_place(items: &mut Vec<String>) {
    let mut seen = std::collections::HashSet::new();
    items.retain(|item| seen.insert(item.clone()));
}

/// Truncate `s` to at most `max` bytes on a char boundary, marking elision.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    // Reserve room for the 3-byte ellipsis so the result stays within `max` bytes.
    let mut end = max.saturating_sub("…".len());
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use descriptor::builtin;

    #[test]
    fn session_capture_surfaces_files_and_commands() {
        let jsonl = [
            r#"{"message":{"role":"user","content":"do the thing"}}"#,
            r#"{"message":{"role":"assistant","content":[{"type":"text","text":"ok"},{"type":"tool_use","name":"Edit","input":{"file_path":"src/a.rs"}},{"type":"tool_use","name":"Bash","input":{"command":"cargo build"}}]}}"#,
        ]
        .join("\n");
        let out = session_capture(&jsonl, 40);
        assert!(out.contains("edited files src/a.rs"), "{out}");
        assert!(out.contains("ran command cargo build"), "{out}");
        assert!(out.contains("user: do the thing"), "{out}");
    }

    #[test]
    fn session_capture_ignores_unknown_tool_payloads() {
        // A tool_use that isn't a known file/command tool must not surface its body.
        let jsonl = r#"{"message":{"role":"assistant","content":[{"type":"tool_use","name":"WebFetch","input":{"url":"http://x"},"text":"secret-ish"}]}}"#;
        let out = session_capture(jsonl, 40);
        assert!(!out.contains("secret-ish"), "{out}");
        assert!(!out.contains("WebFetch"), "{out}");
    }

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
        // A float-typed gate value (some hosts emit `1.0`) must still gate correctly.
        assert!(should_inject(
            &anti,
            &serde_json::json!({ "invocationNum": 1.0 })
        ));
        assert!(!should_inject(
            &anti,
            &serde_json::json!({ "invocationNum": 2.0 })
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
    fn transcript_path_rejects_unsafe_paths() {
        let anti = antigravity();
        // A non-.jsonl target and parent-dir traversal are refused, so a hostile
        // descriptor can't read arbitrary files via the capture hook.
        assert!(
            transcript_path(
                &anti,
                &serde_json::json!({ "transcriptPath": "/etc/passwd" })
            )
            .is_none()
        );
        assert!(
            transcript_path(
                &anti,
                &serde_json::json!({ "transcriptPath": "/t/../../etc/secrets.jsonl" })
            )
            .is_none()
        );
        // A normal .jsonl path is still accepted.
        assert!(
            transcript_path(
                &anti,
                &serde_json::json!({ "transcriptPath": "/t/a.jsonl" })
            )
            .is_some()
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

    #[test]
    fn format_context_caps_large_profile_and_marks_truncation() {
        // Far more than fits in the wake-up budget.
        let big: Vec<MemoryDto> = (0..500)
            .map(|i| {
                mem(
                    "fact",
                    &format!("memory number {i} with some descriptive content"),
                )
            })
            .collect();
        let out = format_context(&big, &[]).unwrap();
        // Truncated, and the overflow is explicitly marked.
        assert!(
            out.ends_with(&format!("{INJECT_MORE_MARKER}\n")),
            "overflow not marked: {out}"
        );
        // Bounded to ~the budget (plus small slack for the preamble/marker lines).
        assert!(
            est_tokens(&out) <= WAKE_TOKEN_BUDGET + est_tokens(INJECT_MORE_MARKER),
            "not capped: {} tokens",
            est_tokens(&out)
        );
        // The tail memories were dropped (only a prefix of the 500 made it in).
        assert!(
            !out.contains("memory number 499"),
            "tail not dropped: {out}"
        );
        // The marker is stripped back out on capture, so it can't echo into memory.
        assert_eq!(strip_injected_memory(&out), "");
    }

    fn mem(kind: &str, content: &str) -> MemoryDto {
        MemoryDto {
            id: "m1".into(),
            content: content.into(),
            kind: kind.into(),
            strength: 1.0,
            created_at: 0,
            score: None,
            freshness: None,
        }
    }

    /// The injected block (built via `format_context`, so this can't drift from the
    /// real injection text) must not survive capture — the anti-feedback-loop.
    #[test]
    fn capture_strips_injected_memory_block() {
        let injected = format_context(
            &[mem("preference", "prefers rust")],
            &[mem("fact", "works on memeora")],
        )
        .unwrap();
        // Injection prepended to a genuine user message in the same turn.
        let mixed = serde_json::json!({
            "message": { "role": "user", "content": format!("{injected}please fix the login bug") }
        })
        .to_string();
        // A turn that is *only* the injection must vanish entirely.
        let pure = serde_json::json!({
            "message": { "role": "user", "content": injected }
        })
        .to_string();
        let reply =
            r#"{"message":{"role":"assistant","content":[{"type":"text","text":"on it"}]}}"#;
        let jsonl = [pure, mixed, reply.to_string()].join("\n");
        assert_eq!(
            transcript_to_text(&jsonl, 40),
            "user: please fix the login bug\nassistant: on it"
        );
    }

    #[test]
    fn strip_injected_memory_keeps_prefix_before_mid_line_marker() {
        let injected = format_context(&[mem("fact", "likes fish")], &[]).unwrap();
        // Hosts that join content blocks with spaces can land the marker mid-line.
        let text = format!("real question first {injected}");
        assert_eq!(strip_injected_memory(&text), "real question first\n");
    }

    #[test]
    fn strip_injected_memory_leaves_ordinary_text_untouched() {
        // No marker → byte-identical, even when the text talks about memory or
        // happens to contain `- [kind]`-shaped bullets.
        let text =
            "user asked about persistent memory features\n- [preference] looks like a bullet\n";
        assert_eq!(strip_injected_memory(text), text);
        let jsonl = r#"{"message":{"role":"user","content":"tell me about memory"}}"#;
        assert_eq!(transcript_to_text(jsonl, 40), "user: tell me about memory");
    }
}
