//! Tier-2/3 opt-in LLM extractor (VISION "Adapts" ladder).
//!
//! Talks to an **OpenAI-compatible** chat-completions endpoint to extract richer
//! memories than the heuristic floor. **Off by default** — enabled only by explicit
//! config (the daemon reads `MEMEORA_LLM_ENDPOINT`). Policy in one boolean: a
//! **local** endpoint (loopback) is allowed under local-first; a **remote** one
//! (Tier-3 BYOK) requires explicit consent ([`LlmConfig::allow_remote`]) and is
//! never used silently.
//!
//! It **never becomes a hard dependency**: any failure — disabled, disallowed,
//! network error, malformed response, or empty result — falls back to the
//! [`HeuristicExtractor`], so "no required LLM" stays literally true. LLM output is
//! graph-self-repaired (VISION "Heals"): every proposed candidate is trimmed,
//! dropped if empty, and its kind coerced to a valid value.
//!
//! The bundled [`HttpTransport`] is intentionally minimal — loopback plain HTTP with
//! `Connection: close` (read to EOF), no TLS or chunked decoding.
// ponytail: that transport covers exactly the local-first (loopback) case; for an
// external TLS endpoint, implement LlmTransport with a real HTTP client behind a feature.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::extract::{Candidate, Extractor, HeuristicExtractor};
use crate::store::MemoryKind;

const SYSTEM_PROMPT: &str = "You extract durable memories from text for a personal memory engine. \
Return ONLY a compact JSON array; each element is {\"content\": string, \"kind\": one of \"fact\", \
\"preference\", \"episode\"}. Capture stable facts, user preferences, and notable episodes; \
ignore chit-chat. No prose, no code fences.";

/// LLM-extracted confidence (above the heuristic's fact/preference score, below an
/// explicit user save cue).
const LLM_CONFIDENCE: f32 = 0.9;

/// Config for the opt-in LLM extractor tier.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// OpenAI-compatible base URL, e.g. `http://localhost:11434/v1`.
    pub endpoint: String,
    /// Model name, e.g. `llama3.1`.
    pub model: String,
    /// Explicit consent to use a NON-local (remote) endpoint. Local needs none.
    pub allow_remote: bool,
}

impl LlmConfig {
    /// Read config from the environment, or `None` when `MEMEORA_LLM_ENDPOINT` is
    /// unset/empty (the default — the tier is off). `MEMEORA_LLM_MODEL` defaults to a
    /// common local model; `MEMEORA_LLM_ALLOW_REMOTE=1` consents to a remote endpoint.
    pub fn from_env() -> Option<LlmConfig> {
        let endpoint = std::env::var("MEMEORA_LLM_ENDPOINT").ok()?;
        if endpoint.trim().is_empty() {
            return None;
        }
        Some(LlmConfig {
            endpoint,
            model: std::env::var("MEMEORA_LLM_MODEL").unwrap_or_else(|_| "llama3.1".to_string()),
            allow_remote: std::env::var("MEMEORA_LLM_ALLOW_REMOTE")
                .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
        })
    }

    /// Whether the endpoint is safe for the shipped transport. Remote LLMs remain
    /// disabled until the client has a TLS-validating transport; a consent flag alone
    /// cannot make raw TCP safe.
    pub fn is_allowed(&self) -> bool {
        host_is_local(&self.endpoint)
    }
}

/// The authority of `url`: the substring after `://` up to the first `/`, `?`, or
/// `#` — the three characters RFC 3986 §3.2 uses to terminate the authority. Treating
/// all three as terminators is what stops a spoofed `evil.com#@localhost` or
/// `evil.com?@localhost` from smuggling a fake host past the consent gate.
fn authority_of(url: &str) -> &str {
    let after_scheme = url.split("://").nth(1).unwrap_or(url);
    after_scheme.split(['/', '?', '#']).next().unwrap_or("")
}

/// Split an authority into its lowercased host and port (default 80), dropping any
/// `userinfo@` and `[ipv6]` brackets. ONE host parser, shared by the consent check
/// ([`host_is_local`]) and the actual connection ([`split_url`]), so the two can never
/// disagree on which host is in play — the property the egress gate depends on.
fn split_host_port(authority: &str) -> Result<(String, u16)> {
    // The host is what follows the last `@` (userinfo is `user:pass@`, not the host).
    let hostport = authority.rsplit('@').next().unwrap_or(authority);
    let (host, port_str) = if let Some(rest) = hostport.strip_prefix('[') {
        // IPv6 `[addr]` / `[addr]:port`: strip brackets so the bare address reaches
        // TcpStream::connect, and the inner colons don't confuse the port split.
        let (host, after) = rest.split_once(']').unwrap_or((rest, ""));
        (host, after.strip_prefix(':'))
    } else {
        match hostport.rsplit_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (hostport, None),
        }
    };
    let port = match port_str {
        Some(p) => p.parse::<u16>().map_err(|_| {
            Error::Llm(format!("invalid port {p:?} in URL authority {authority:?}"))
        })?,
        None => 80,
    };
    // Hostnames are case-insensitive, so `LOCALHOST` matches loopback too.
    Ok((host.to_ascii_lowercase(), port))
}

/// Whether `url`'s host is loopback (so it's allowed under local-first). A malformed
/// authority (e.g. a bad port) is treated as non-local — failing closed toward
/// requiring explicit consent.
fn host_is_local(url: &str) -> bool {
    let Ok((host, _)) = split_host_port(authority_of(url)) else {
        return false;
    };
    matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1") || host.ends_with(".localhost")
}

/// Sends a JSON POST to `url` and returns the response body. Abstracted so the
/// extractor logic is testable without a live server.
pub trait LlmTransport: Send + Sync {
    /// POST `body` (JSON) to `url`, returning the raw response body.
    fn post_json(&self, url: &str, body: &str) -> Result<String>;
}

/// Read/write timeout for the loopback transport (no setter — loopback is fast or absent).
const TRANSPORT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RESPONSE_BYTES: u64 = 4 * 1024 * 1024;

/// Minimal blocking HTTP/1.1 transport for a loopback OpenAI-compatible server.
pub struct HttpTransport;

impl LlmTransport for HttpTransport {
    fn post_json(&self, url: &str, body: &str) -> Result<String> {
        let (host, port, path) = split_url(url)?;
        let mut stream = TcpStream::connect((host.as_str(), port))
            .map_err(|e| Error::Llm(format!("connect {host}:{port}: {e}")))?;
        stream.set_read_timeout(Some(TRANSPORT_TIMEOUT)).ok();
        stream.set_write_timeout(Some(TRANSPORT_TIMEOUT)).ok();
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            host_header(&host, port),
            body.len()
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|e| Error::Llm(format!("write: {e}")))?;
        let mut raw = String::new();
        stream
            .take(MAX_RESPONSE_BYTES + 1)
            .read_to_string(&mut raw)
            .map_err(|e| Error::Llm(format!("read: {e}")))?;
        if raw.len() as u64 > MAX_RESPONSE_BYTES {
            return Err(Error::Llm(format!(
                "response exceeds {MAX_RESPONSE_BYTES} bytes"
            )));
        }
        // `Connection: close` ⇒ the server sent one non-chunked response then closed,
        // so the body is everything after the header terminator.
        raw.split_once("\r\n\r\n")
            .map(|(_, body)| body.to_string())
            .ok_or_else(|| Error::Llm("malformed HTTP response (no header/body split)".into()))
    }
}

/// `Host` header value for `host:port`. [`split_host_port`] strips the `[...]` from
/// an IPv6 literal for `TcpStream::connect`, but RFC 7230 §5.4 (via RFC 3986) requires
/// the brackets back in the header — `Host: [::1]:11434`, never `Host: ::1:11434`.
fn host_header(host: &str, port: u16) -> String {
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// Split `http://host[:port]/path` into `(host, port, path)`. Port defaults to 80.
/// Only `http://` (loopback) is supported — TLS is out of the minimal transport's
/// scope. Host/port come from [`split_host_port`] — the same parser the consent gate
/// uses — so the connected host always matches the classified one.
fn split_url(url: &str) -> Result<(String, u16, String)> {
    let rest = url.strip_prefix("http://").ok_or_else(|| {
        Error::Llm(format!(
            "only http:// (loopback) endpoints are supported by the built-in transport, got {url:?}"
        ))
    })?;
    // Authority ends at the first '/', '?', or '#'; whatever follows is the request path.
    let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (host, port) = split_host_port(&rest[..auth_end])?;
    let path = if rest[auth_end..].starts_with('/') {
        &rest[auth_end..]
    } else {
        "/"
    };
    Ok((host, port, path.to_string()))
}

/// The opt-in LLM extractor. Falls back to the heuristic floor on any failure.
pub struct LlmExtractor {
    config: LlmConfig,
    transport: Box<dyn LlmTransport>,
    fallback: HeuristicExtractor,
}

impl LlmExtractor {
    /// Build with the default loopback HTTP transport.
    pub fn new(config: LlmConfig) -> Self {
        LlmExtractor::with_transport(config, Box::new(HttpTransport))
    }

    /// Build with a custom transport (used in tests, or to support TLS/remote later).
    pub fn with_transport(config: LlmConfig, transport: Box<dyn LlmTransport>) -> Self {
        LlmExtractor {
            config,
            transport,
            fallback: HeuristicExtractor::default(),
        }
    }

    /// Attempt LLM extraction; returns an empty vec on any disallowed/failed path.
    fn try_llm(&self, text: &str) -> Vec<Candidate> {
        if !self.config.is_allowed() {
            return Vec::new(); // remote without consent → never call out
        }
        let url = format!(
            "{}/chat/completions",
            self.config.endpoint.trim_end_matches('/')
        );
        let body = build_chat_request(&self.config.model, text);
        match self.transport.post_json(&url, &body) {
            Ok(resp) => parse_content(&resp)
                .map(|c| parse_candidates(&c))
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }
}

impl Extractor for LlmExtractor {
    fn extract(&self, text: &str) -> Result<Vec<Candidate>> {
        let candidates = self.try_llm(text);
        if candidates.is_empty() {
            // Disabled, disallowed, network/parse failure, or nothing extracted: the
            // heuristic floor keeps "no required LLM" literally true.
            self.fallback.extract(text)
        } else {
            Ok(candidates)
        }
    }
}

/// Build the OpenAI-compatible chat-completions request body.
fn build_chat_request(model: &str, text: &str) -> String {
    serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "stream": false,
    })
    .to_string()
}

/// Pull `choices[0].message.content` out of a completion response, or `None` if the
/// shape is unexpected.
fn parse_content(response_body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(response_body).ok()?;
    v.get("choices")?
        .get(0)?
        .get("message")?
        .get("content")?
        .as_str()
        .map(str::to_string)
}

/// Raw element of the model's JSON array, before self-repair.
#[derive(Deserialize)]
struct RawCandidate {
    content: String,
    #[serde(default)]
    kind: String,
}

/// Parse the model's JSON-array output into self-repaired candidates. Returns an
/// empty vec (→ heuristic fallback) if the output isn't a JSON array at all; a
/// malformed *element* (e.g. `"kind": null`) is skipped individually, so one bad
/// candidate never discards the rest of the batch.
fn parse_candidates(content: &str) -> Vec<Candidate> {
    let cleaned = strip_code_fences(content);
    let Ok(items) = serde_json::from_str::<Vec<serde_json::Value>>(cleaned) else {
        return Vec::new();
    };
    items
        .into_iter()
        .filter_map(|v| serde_json::from_value::<RawCandidate>(v).ok())
        .filter_map(repair_candidate)
        .collect()
}

/// Graph self-repair for one proposed candidate: trim, drop-if-empty, coerce kind to
/// a valid [`MemoryKind`] (invalid → `Fact`).
fn repair_candidate(raw: RawCandidate) -> Option<Candidate> {
    let content = raw.content.trim().to_string();
    if content.is_empty() {
        return None;
    }
    Some(Candidate {
        content,
        kind: MemoryKind::from_str_lossy(&raw.kind.to_lowercase()),
        expires_at: None,
        // The opt-in LLM tier can populate occurred-time later; the floor leaves it unset.
        occurred_start: None,
        occurred_end: None,
        confidence: LLM_CONFIDENCE,
    })
}

/// Strip a leading ``` fence — with an optional language tag in any casing (`json`,
/// `JSON`, …) — and a trailing ``` the model may wrap JSON in. JSON itself starts
/// with `[`/`{`, so trimming leading ASCII letters can never eat the payload.
fn strip_code_fences(s: &str) -> &str {
    let mut t = s.trim();
    if let Some(rest) = t.strip_prefix("```") {
        t = rest
            .trim_start_matches(|c: char| c.is_ascii_alphabetic())
            .trim();
    }
    if let Some(rest) = t.strip_suffix("```") {
        t = rest.trim();
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTransport(String);
    impl LlmTransport for MockTransport {
        fn post_json(&self, _url: &str, _body: &str) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    struct FailTransport;
    impl LlmTransport for FailTransport {
        fn post_json(&self, _url: &str, _body: &str) -> Result<String> {
            Err(Error::Llm("boom".into()))
        }
    }

    fn cfg(endpoint: &str, allow_remote: bool) -> LlmConfig {
        LlmConfig {
            endpoint: endpoint.into(),
            model: "m".into(),
            allow_remote,
        }
    }

    fn completion(content: &str) -> String {
        serde_json::json!({ "choices": [{ "message": { "content": content } }] }).to_string()
    }

    #[test]
    fn only_loopback_endpoints_are_allowed() {
        assert!(cfg("http://localhost:11434/v1", false).is_allowed());
        assert!(cfg("http://127.0.0.1:1234", false).is_allowed());
        assert!(cfg("http://[::1]:8080/v1", false).is_allowed());
        assert!(!cfg("http://api.openai.com/v1", false).is_allowed());
        assert!(!cfg("http://api.openai.com/v1", true).is_allowed());
        assert!(!cfg("https://api.openai.com/v1", true).is_allowed());
        // A fragment-spoofed host must NOT be classified local (the consent-gate
        // bypass `http://evil.com#@localhost` reported in review).
        assert!(!cfg("http://evil.com:80#@localhost/v1", false).is_allowed());
        assert!(!cfg("http://evil.com#@localhost/", false).is_allowed());
        // Host comparison is case-insensitive: LOCALHOST is still loopback.
        assert!(cfg("http://LOCALHOST:11434/v1", false).is_allowed());
        // A query string before any path is also an authority terminator — a spoofed
        // `?@localhost` must NOT win (the parser-divergence hardening).
        assert!(!cfg("http://evil.com?@localhost", false).is_allowed());
        // userinfo is stripped: the host AFTER the last `@` is authoritative.
        assert!(!cfg("http://localhost@evil.com/v1", false).is_allowed());
        assert!(cfg("http://user:pass@127.0.0.1:11434/v1", false).is_allowed());
    }

    #[test]
    fn parses_and_repairs_llm_candidates() {
        let content = "```json\n[{\"content\":\"User prefers dark mode\",\"kind\":\"preference\"},\
                       {\"content\":\"   \",\"kind\":\"fact\"},\
                       {\"content\":\"Met Alex\",\"kind\":\"bogus\"}]\n```";
        let ex = LlmExtractor::with_transport(
            cfg("http://localhost:1/v1", false),
            Box::new(MockTransport(completion(content))),
        );
        let cands = ex.extract("whatever").unwrap();
        assert_eq!(cands.len(), 2, "empty-content candidate dropped");
        assert_eq!(cands[0].kind, MemoryKind::Preference);
        assert_eq!(
            cands[1].kind,
            MemoryKind::Fact,
            "invalid kind coerced to fact"
        );
    }

    #[test]
    fn malformed_element_does_not_discard_batch() {
        // `kind: null` fails RawCandidate deserialization (serde(default) only covers
        // an ABSENT field) — only that element is skipped, not the whole batch.
        let content = "[{\"content\":\"User prefers Rust\",\"kind\":\"preference\"},\
                       {\"content\":\"Deploys nightly\",\"kind\":null}]";
        let cands = parse_candidates(content);
        assert_eq!(cands.len(), 1, "good candidate survives the bad element");
        assert_eq!(cands[0].content, "User prefers Rust");
        assert_eq!(cands[0].kind, MemoryKind::Preference);
    }

    #[test]
    fn strips_fences_regardless_of_language_tag_casing() {
        assert_eq!(strip_code_fences("```json\n[1]\n```"), "[1]");
        assert_eq!(strip_code_fences("```JSON\n[1]\n```"), "[1]");
        assert_eq!(strip_code_fences("```Json\n[1]\n```"), "[1]");
        assert_eq!(strip_code_fences("```\n[1]\n```"), "[1]");
        assert_eq!(strip_code_fences("[1]"), "[1]");
    }

    #[test]
    fn host_header_brackets_ipv6_literals() {
        // RFC 7230: an IPv6 literal must keep its brackets in the Host header.
        assert_eq!(host_header("::1", 11434), "[::1]:11434");
        assert_eq!(host_header("localhost", 11434), "localhost:11434");
        assert_eq!(host_header("127.0.0.1", 80), "127.0.0.1:80");
    }

    #[test]
    fn falls_back_to_heuristic_on_transport_error() {
        let ex = LlmExtractor::with_transport(
            cfg("http://localhost:1/v1", false),
            Box::new(FailTransport),
        );
        let cands = ex.extract("I prefer dark mode in my editor").unwrap();
        assert_eq!(
            cands.len(),
            1,
            "heuristic floor still extracts the preference"
        );
        assert_eq!(cands[0].kind, MemoryKind::Preference);
    }

    #[test]
    fn disallowed_remote_never_calls_out() {
        struct PanicTransport;
        impl LlmTransport for PanicTransport {
            fn post_json(&self, _url: &str, _body: &str) -> Result<String> {
                panic!("transport must not be called for a disallowed endpoint")
            }
        }
        let ex = LlmExtractor::with_transport(
            cfg("http://api.openai.com/v1", false),
            Box::new(PanicTransport),
        );
        let cands = ex.extract("I prefer dark mode in my editor").unwrap();
        assert_eq!(cands.len(), 1, "heuristic floor applies, no network call");
    }

    #[test]
    fn split_url_parses_host_port_path() {
        assert_eq!(
            split_url("http://localhost:11434/v1/chat/completions").unwrap(),
            (
                "localhost".to_string(),
                11434,
                "/v1/chat/completions".to_string()
            )
        );
        assert_eq!(
            split_url("http://127.0.0.1/x").unwrap(),
            ("127.0.0.1".to_string(), 80, "/x".to_string())
        );
        assert!(split_url("https://x/y").is_err(), "TLS is out of scope");
        // IPv6 brackets are stripped so the bare address reaches TcpStream::connect.
        assert_eq!(
            split_url("http://[::1]:8080/v1").unwrap(),
            ("::1".to_string(), 8080, "/v1".to_string())
        );
        assert_eq!(
            split_url("http://[::1]/v1").unwrap(),
            ("::1".to_string(), 80, "/v1".to_string())
        );
        // A malformed port is a hard error, not a silent fallback to 80.
        assert!(split_url("http://localhost:notaport/v1").is_err());
        // userinfo is stripped, so the connected host matches the consent check.
        assert_eq!(
            split_url("http://user:pass@localhost:11434/v1").unwrap(),
            ("localhost".to_string(), 11434, "/v1".to_string())
        );
    }
}
