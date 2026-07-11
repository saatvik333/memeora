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
pub(crate) fn strip_private(text: &str) -> String {
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
    // Whether the previous word was an auth scheme ("Bearer"/"Basic"), so the token
    // that follows it is redacted even without a recognizable prefix.
    let mut after_auth_scheme = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !word.is_empty() {
                out.push_str(&redact_word(&word, after_auth_scheme));
                after_auth_scheme = is_auth_scheme(&word);
                word.clear();
            }
            if ch == '\n' {
                // An auth-scheme cue never carries across lines: headers put the
                // credential on the same line, so a line ending in "Basic"/"Bearer"
                // must not mask the first long word of the next line.
                after_auth_scheme = false;
            }
            out.push(ch); // preserve the exact whitespace char verbatim
        } else {
            word.push(ch);
        }
    }
    if !word.is_empty() {
        out.push_str(&redact_word(&word, after_auth_scheme));
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
];

/// Sensitive keys matched as whole `_`/`-`-delimited segments, not substrings —
/// "auth" as a substring would also hit benign keys like "author".
const SENSITIVE_KEY_WORDS: &[&str] = &["auth", "oauth", "authorization"];

/// Whether a lowercased `key=value` key is sensitive: a substring hit for the
/// unambiguous keys, or a whole-segment hit for the auth family.
fn is_sensitive_key(key_core: &str) -> bool {
    SENSITIVE_KEYS.iter().any(|s| key_core.contains(s))
        || key_core
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|seg| SENSITIVE_KEY_WORDS.contains(&seg))
}

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
    // Stripe-style keys use an underscore, so the `sk-` prefix above does not cover them.
    "sk_live_",
    "sk_test_",
    "rk_live_",
    "rk_test_",
];

/// Redact a single whitespace-delimited word, preserving non-secret text.
///
/// Handles tokens wrapped in punctuation (quotes, brackets, trailing commas) by
/// inspecting the alphanumeric core, so `"sk-…",` and a tab-indented `\tsk-…` are
/// caught — not just a bare space-delimited token.
fn redact_word(word: &str, after_auth_scheme: bool) -> String {
    // A token right after an auth scheme ("Authorization: Bearer <tok>", "Basic
    // <tok>") is a credential even without a recognizable prefix. Gate on token
    // shape so an ordinary following word ("bearer of news") isn't masked.
    if after_auth_scheme {
        let core = word.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if looks_token(core) {
            return word.replace(core, "[REDACTED]");
        }
    }
    // Credential-in-URL: `scheme://user:pass@host` → mask just the userinfo, so a
    // captured connection string (postgres://…, redis://…) can't leak its password.
    if let Some(sep) = word.find("://") {
        let after = sep + 3;
        let rest = &word[after..];
        if let Some(at) = rest.find('@') {
            let host_start = rest.find('/').unwrap_or(rest.len());
            if at > 0 && at < host_start {
                return format!("{}[REDACTED]{}", &word[..after], &rest[at..]);
            }
        }
    }
    // `key=value` / `key: value` with a sensitive key → mask only the value.
    if let Some(idx) = word.find(['=', ':']) {
        let (key, rest) = word.split_at(idx);
        let value = &rest[1..];
        let key_core = key
            .trim_matches(|c: char| !c.is_ascii_alphanumeric())
            .to_ascii_lowercase();
        if !value.is_empty() && is_sensitive_key(&key_core) {
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

/// Whether a word is an auth scheme keyword whose following token is a credential.
fn is_auth_scheme(word: &str) -> bool {
    let core = word
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase();
    core == "bearer" || core == "basic"
}

/// Whether a bare token (no known prefix) is long/mixed enough to be a credential
/// rather than a prose word — used only after an auth-scheme cue, to bound false
/// positives on ordinary following words.
fn looks_token(core: &str) -> bool {
    // Length + charset alone would mask prose ("basic understanding" → the 13-char
    // all-lowercase "understanding"), so additionally require a non-prose signal:
    // a digit, a token separator, or interior uppercase (mixed case past the first
    // char, which prose words and Capitalized sentence starts don't have). JWTs and
    // base64 blobs always carry at least one of these.
    core.len() >= 12
        && core
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_+/=.".contains(c))
        && (core.chars().any(|c| c.is_ascii_digit())
            || core.chars().any(|c| "-_+/=.".contains(c))
            || core.chars().skip(1).any(|c| c.is_ascii_uppercase()))
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

    #[test]
    fn redacts_credential_in_url() {
        // Connection strings are the dominant secret shape in captured command args.
        let out = redact("ran psql postgres://admin:hunter2pw@db.internal/prod ok");
        assert!(
            out.contains("postgres://[REDACTED]@db.internal/prod"),
            "userinfo not masked: {out:?}"
        );
        assert!(!out.contains("hunter2pw"), "password leaked: {out:?}");
        assert!(out.contains("ran psql") && out.contains(" ok"));
        // A normal URL with no credentials must be left untouched.
        assert_eq!(
            redact("see https://example.com/docs"),
            "see https://example.com/docs"
        );
    }

    #[test]
    fn redacts_bearer_token_but_not_prose() {
        let out = redact("Authorization: Bearer abc123def456ghi789xyz");
        assert!(
            out.contains("[REDACTED]"),
            "bearer token not masked: {out:?}"
        );
        assert!(!out.contains("abc123def456"));
        // An all-alpha token (e.g. a JWT segment) after the scheme is still masked.
        let alpha = redact("Authorization: Bearer aAbBcCdDeEfFgGhHiIjJ");
        assert!(
            alpha.contains("[REDACTED]") && !alpha.contains("aAbBcCdD"),
            "{alpha:?}"
        );
        // The word after "bearer" in ordinary prose must NOT be redacted.
        assert_eq!(redact("the bearer of bad news"), "the bearer of bad news");
    }

    #[test]
    fn auth_scheme_words_in_prose_do_not_redact() {
        // "basic"/"bearer" as ordinary English must not mask the following word,
        // even a long one (all-lowercase / Capitalized words are prose, not tokens).
        for prose in [
            "a basic understanding of the problem",
            "basic configuration is required",
            "Basic Authentication is a scheme",
            "the bearer presentation went well",
        ] {
            assert_eq!(redact(prose), prose);
        }
        // The cue does not survive a line break: headers keep scheme + credential
        // on one line, so the next line's first long word stays untouched.
        let multiline = "the mode is Basic\nunderstanding follows tomorrow";
        assert_eq!(redact(multiline), multiline);
    }

    #[test]
    fn real_tokens_after_auth_scheme_still_redact() {
        // A JWT (dots + digits) after "Bearer" must always be masked.
        let jwt = redact("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig");
        assert!(jwt.contains("[REDACTED]"), "{jwt:?}");
        assert!(!jwt.contains("eyJhbGci"), "{jwt:?}");
        // Base64-ish Basic credentials (mixed case + digits) too.
        let basic = redact("Authorization: Basic dXNlcjpwYXNzd29yZDEyMw==");
        assert!(basic.contains("[REDACTED]"), "{basic:?}");
        assert!(!basic.contains("dXNlcjpw"), "{basic:?}");
    }

    #[test]
    fn auth_key_matches_whole_segment_not_substring() {
        // "author" must not trip the "auth" sensitive key.
        assert_eq!(redact("author=Jane Doe"), "author=Jane Doe");
        assert_eq!(redact("see ?author=jane&ref=x"), "see ?author=jane&ref=x");
        // Real auth-family keys are still masked.
        let auth = redact("auth=supersecretvalue");
        assert_eq!(auth, "auth=[REDACTED]");
        let hdr = redact("x-auth-token:abc123");
        assert!(hdr.contains("[REDACTED]"), "{hdr:?}");
        let oauth = redact("oauth:abc123");
        assert!(oauth.contains("[REDACTED]"), "{oauth:?}");
        let authz = redact("authorization=abc123");
        assert!(authz.contains("[REDACTED]"), "{authz:?}");
    }

    #[test]
    fn redacts_underscored_provider_key() {
        // `sk_live_…` is distinct from the hyphenated `sk-` prefix.
        // Bare token under 32 chars: only the prefix (not the high-entropy rule) catches it.
        let out = redact("billing key sk_live_51HxyzABCDEF here");
        assert!(out.contains("[REDACTED]"), "{out:?}");
        assert!(!out.contains("sk_live_51Hxyz"));
        assert!(out.contains("billing key") && out.contains("here"));
    }
}
