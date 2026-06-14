//! Privacy enforcement at the engine write boundary.
//!
//! Every write surface — the MCP `remember` tool, the IPC client, the
//! auto-capture hook — funnels through the daemon's `Preparer`, which calls
//! [`sanitize`] before any text is extracted, embedded, or stored. Enforcing it
//! here, once, rather than per-surface keeps the privacy invariant true no matter
//! how a memory enters the engine (the vision treats agents as *surfaces*, not the
//! boundary).
//!
//! [`sanitize`] runs two passes, in order:
//! 1. [`strip_private`] removes `<private>…</private>` spans entirely.
//! 2. [`redact`] masks obvious secrets (credential-prefixed tokens,
//!    `key=value` / `key: value` pairs with a sensitive key, and long
//!    high-entropy blobs) in whatever survives.
//!
//! Both passes are heuristic and conservative — a safety net, not a licence to
//! paste secrets. Redaction preserves all surrounding whitespace, so stored
//! content stays verbatim apart from the masked spans.

/// Strip `<private>…</private>` spans, then redact secrets from the remainder.
///
/// The canonical write-path sanitizer. Deterministic, so a content-addressed id
/// computed over the sanitized text stays stable across re-ingest.
pub fn sanitize(text: &str) -> String {
    redact(&strip_private(text))
}

const PRIVATE_OPEN: &str = "<private>";
const PRIVATE_CLOSE: &str = "</private>";

/// Remove every `<private>…</private>` span (the tags matched case-insensitively).
///
/// An unterminated `<private>` drops everything from the tag to end-of-input —
/// failing **closed**, so a malformed fence can never leak the content it meant to
/// hide.
pub fn strip_private(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = find_ci(rest, PRIVATE_OPEN) {
        out.push_str(&rest[..open]);
        let after = &rest[open + PRIVATE_OPEN.len()..];
        match find_ci(after, PRIVATE_CLOSE) {
            Some(close) => rest = &after[close + PRIVATE_CLOSE.len()..],
            None => return out, // unterminated fence: drop the rest, fail closed
        }
    }
    out.push_str(rest);
    out
}

/// Case-insensitive ASCII substring search. `needle` must be lowercase ASCII.
///
/// `to_ascii_lowercase` is byte-length-preserving, so the byte offset it returns
/// is valid in the original `haystack`.
fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack.to_ascii_lowercase().find(needle)
}

/// Best-effort redaction of obvious secrets, preserving all non-secret text **and
/// whitespace** verbatim (tabs, runs of spaces, and newlines survive — unlike a
/// naive `split_whitespace().join(" ")`, which would corrupt code/TSV content).
///
/// Masks known credential-prefixed tokens, `key=value` / `key: value` pairs with a
/// sensitive key, and long high-entropy blobs.
pub fn redact(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !word.is_empty() {
                out.push_str(&redact_word(&word));
                word.clear();
            }
            out.push(ch); // preserve the exact whitespace char verbatim
        } else {
            word.push(ch);
        }
    }
    if !word.is_empty() {
        out.push_str(&redact_word(&word));
    }
    out
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

    #[test]
    fn strips_private_span() {
        assert_eq!(
            strip_private("before <private>secret</private> after"),
            "before  after"
        );
    }

    #[test]
    fn strips_private_case_insensitively_and_multiline() {
        assert_eq!(
            strip_private("a <PRIVATE>x\ny</Private> b"),
            "a  b",
            "tags match regardless of case, spans cross newlines"
        );
    }

    #[test]
    fn unterminated_private_fails_closed() {
        // No closing tag → everything from the open tag is dropped, never leaked.
        assert_eq!(strip_private("keep this <private>leaked?"), "keep this ");
    }

    #[test]
    fn redact_preserves_whitespace_verbatim() {
        // Tabs and runs of spaces must survive (the bug: split_whitespace collapse).
        let input = "a\tb   c\nd";
        assert_eq!(redact(input), input);
    }

    #[test]
    fn redact_masks_secret_keeping_surrounding_whitespace() {
        let out = redact("deploy\tkey sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345  now");
        assert!(out.contains("[REDACTED]"));
        assert!(out.contains("deploy\tkey"), "tab preserved: {out:?}");
        assert!(out.contains("  now"), "double space preserved: {out:?}");
        assert!(!out.contains("sk-ABCDEFG"));
    }

    #[test]
    fn sanitize_strips_then_redacts() {
        let out =
            sanitize("<private>topsecret</private> token=hunter2 and sk-ABCDEFGHIJKLMNOP012345");
        assert!(!out.contains("topsecret"), "private span survived: {out:?}");
        assert!(
            out.contains("token=[REDACTED]"),
            "kv secret survived: {out:?}"
        );
        assert!(
            !out.contains("sk-ABCDEF"),
            "prefixed token survived: {out:?}"
        );
    }

    #[test]
    fn sanitize_leaves_ordinary_prose_untouched() {
        assert_eq!(
            sanitize("I prefer dark mode in my editor"),
            "I prefer dark mode in my editor"
        );
    }
}
