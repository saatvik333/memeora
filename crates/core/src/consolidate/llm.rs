//! Opt-in LLM [`ObservationSynthesizer`] (Phase F's LLM tier).
//!
//! Given a cluster's near-duplicate member texts, asks an **OpenAI-compatible** chat
//! endpoint to distil ONE canonical belief sentence capturing what the members agree on.
//! It reuses the extractor tier's consent gate and transport wholesale — [`LlmConfig`]
//! (loopback allowed under local-first; remote needs explicit consent) and the
//! [`LlmTransport`]/[`HttpTransport`] loopback client — so the two tiers can never
//! disagree on which endpoints are permitted.
//!
//! **Fail-open, like [`crate::extract::llm`]:** the LLM is never a hard dependency. If the
//! tier is disabled, the endpoint is disallowed, the network errors, or the response is
//! empty/malformed, [`synthesize`](LlmSynthesizer::synthesize) falls back to the
//! [`PassthroughSynthesizer`] (longest member verbatim) instead of erroring — so swapping
//! [`LlmSynthesizer`] in for the passthrough can never break or block consolidation, and
//! the fallback stays deterministic (idempotent re-consolidation).

use crate::Result;
use crate::consolidate::{ObservationSynthesizer, PassthroughSynthesizer};
use crate::extract::llm::{HttpTransport, LlmConfig, LlmTransport};

const SYSTEM_PROMPT: &str = "You merge several near-duplicate memory statements into ONE \
concise canonical statement of the shared fact. Output only the sentence, no prose, no quotes.";

/// The opt-in LLM synthesizer. Composes the [`PassthroughSynthesizer`] as its fallback and
/// falls back to it on any disallowed/failed/empty path — the LLM is never required.
pub struct LlmSynthesizer {
    config: LlmConfig,
    transport: Box<dyn LlmTransport>,
    fallback: PassthroughSynthesizer,
}

impl LlmSynthesizer {
    /// Build with the default loopback HTTP transport.
    pub fn new(config: LlmConfig) -> Self {
        LlmSynthesizer::with_transport(config, Box::new(HttpTransport))
    }

    /// Build with a custom transport (used in tests, or to support TLS/remote later).
    pub fn with_transport(config: LlmConfig, transport: Box<dyn LlmTransport>) -> Self {
        LlmSynthesizer {
            config,
            transport,
            fallback: PassthroughSynthesizer,
        }
    }

    /// Build from the environment, or `None` when the tier is unconfigured
    /// (`MEMEORA_LLM_ENDPOINT` unset/empty) — mirroring [`LlmConfig::from_env`], so an
    /// unconfigured caller simply keeps using the passthrough.
    pub fn from_env() -> Option<LlmSynthesizer> {
        LlmConfig::from_env().map(LlmSynthesizer::new)
    }

    /// Attempt LLM synthesis; `None` on any disallowed/failed/empty path so the caller
    /// falls back to the passthrough. Never calls out for a disallowed endpoint.
    fn try_llm(&self, members: &[&str]) -> Option<String> {
        if !self.config.is_allowed() {
            return None; // remote without consent → never call out
        }
        let url = format!(
            "{}/chat/completions",
            self.config.endpoint.trim_end_matches('/')
        );
        let body = build_chat_request(&self.config.model, members);
        let resp = self.transport.post_json(&url, &body).ok()?;
        let line = clean_line(&parse_content(&resp)?);
        if line.is_empty() { None } else { Some(line) }
    }
}

impl ObservationSynthesizer for LlmSynthesizer {
    fn synthesize(&self, members: &[&str]) -> Result<String> {
        match self.try_llm(members) {
            Some(line) => Ok(line),
            // Disabled, disallowed, network/parse failure, or empty output: the passthrough
            // floor keeps consolidation working with "no required LLM".
            None => self.fallback.synthesize(members),
        }
    }
}

/// Build the OpenAI-compatible chat-completions request body: the members are sent as a
/// bulleted list in the user message for the model to merge.
fn build_chat_request(model: &str, members: &[&str]) -> String {
    let user = members
        .iter()
        .map(|m| format!("- {m}"))
        .collect::<Vec<_>>()
        .join("\n");
    serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "stream": false,
    })
    .to_string()
}

/// Pull `choices[0].message.content` out of a completion response, or `None` if the shape
/// is unexpected.
fn parse_content(response_body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(response_body).ok()?;
    v.get("choices")?
        .get(0)?
        .get("message")?
        .get("content")?
        .as_str()
        .map(str::to_string)
}

/// Reduce the model's reply to one canonical line: strip a ``` code fence, take the first
/// non-empty line (the sentence), then peel one pair of wrapping quotes. Returns `""` (→
/// passthrough fallback) when nothing usable remains.
fn clean_line(content: &str) -> String {
    let unfenced = strip_code_fences(content);
    let line = unfenced
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    strip_wrapping_quotes(line).trim().to_string()
}

/// Strip a leading ``` fence (with an optional language tag on the same line) and a
/// trailing ``` the model may wrap its answer in.
fn strip_code_fences(s: &str) -> &str {
    let mut t = s.trim();
    if let Some(rest) = t.strip_prefix("```") {
        // Drop the fence's opening line (which may carry a language tag).
        t = rest.split_once('\n').map(|(_, r)| r).unwrap_or(rest).trim();
    }
    if let Some(rest) = t.strip_suffix("```") {
        t = rest.trim();
    }
    t
}

/// Peel a single pair of matching wrapping quotes (`"…"` or `'…'`) the model may add
/// despite the prompt; leaves unbalanced or inner quotes untouched.
fn strip_wrapping_quotes(s: &str) -> &str {
    for q in ['"', '\''] {
        if let Some(inner) = s
            .strip_prefix(q)
            .and_then(|t| t.strip_suffix(q))
            .filter(|inner| !inner.is_empty())
        {
            return inner.trim();
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

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

    /// Panics if called — proves a disallowed endpoint never reaches the transport.
    struct PanicTransport;
    impl LlmTransport for PanicTransport {
        fn post_json(&self, _url: &str, _body: &str) -> Result<String> {
            panic!("transport must not be called for a disallowed endpoint")
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

    /// The passthrough result for a set of members (longest, tie → lexicographically
    /// smallest) — the exact fallback the LLM synthesizer must reproduce.
    fn passthrough(members: &[&str]) -> String {
        PassthroughSynthesizer.synthesize(members).unwrap()
    }

    #[test]
    fn disallowed_remote_never_calls_out_and_falls_back() {
        let members = ["user prefers dark mode", "prefers dark mode"];
        let s = LlmSynthesizer::with_transport(
            cfg("http://api.openai.com/v1", false),
            Box::new(PanicTransport),
        );
        // No network call (PanicTransport untouched); result is the passthrough's.
        assert_eq!(s.synthesize(&members).unwrap(), passthrough(&members));
    }

    #[test]
    fn transport_error_falls_back_to_passthrough() {
        let members = ["dark mode", "user prefers dark mode"];
        let s = LlmSynthesizer::with_transport(
            cfg("http://localhost:1/v1", false),
            Box::new(FailTransport),
        );
        assert_eq!(s.synthesize(&members).unwrap(), passthrough(&members));
    }

    #[test]
    fn empty_response_content_falls_back_to_passthrough() {
        let members = ["a", "bbb", "cc"];
        let s = LlmSynthesizer::with_transport(
            cfg("http://localhost:1/v1", false),
            Box::new(MockTransport(completion("   \n  "))),
        );
        // Whitespace-only completion → clean_line empties → passthrough fallback.
        assert_eq!(s.synthesize(&members).unwrap(), passthrough(&members));
    }

    #[test]
    fn malformed_response_falls_back_to_passthrough() {
        let members = ["a", "bbb"];
        let s = LlmSynthesizer::with_transport(
            cfg("http://localhost:1/v1", false),
            Box::new(MockTransport("not json at all".into())),
        );
        assert_eq!(s.synthesize(&members).unwrap(), passthrough(&members));
    }

    #[test]
    fn successful_llm_returns_canonical_line() {
        let s = LlmSynthesizer::with_transport(
            cfg("http://localhost:1/v1", false),
            Box::new(MockTransport(completion(
                "```\n\"The user prefers dark mode.\"\n```",
            ))),
        );
        assert_eq!(
            s.synthesize(&["dark mode", "prefers dark"]).unwrap(),
            "The user prefers dark mode."
        );
    }

    #[test]
    fn clean_line_strips_fences_quotes_and_extra_lines() {
        assert_eq!(
            clean_line("The user prefers dark mode."),
            "The user prefers dark mode."
        );
        // Fence + surrounding double quotes.
        assert_eq!(clean_line("```\n\"merged fact\"\n```"), "merged fact");
        // Language-tagged fence.
        assert_eq!(clean_line("```text\nmerged fact\n```"), "merged fact");
        // Single quotes.
        assert_eq!(clean_line("'merged fact'"), "merged fact");
        // Only the first non-empty line survives.
        assert_eq!(clean_line("\n\nfirst line\nsecond line"), "first line");
        // Nothing usable → empty (drives the passthrough fallback).
        assert_eq!(clean_line("   \n  "), "");
        assert_eq!(clean_line("```\n\n```"), "");
        // An unbalanced quote is left intact (not a wrapping pair).
        assert_eq!(clean_line("\"half quoted"), "\"half quoted");
    }

    /// Real-endpoint smoke test — needs a live OpenAI-compatible server, so it's ignored
    /// in CI. Run with `MEMEORA_LLM_ENDPOINT=… cargo test -p memeora-core -- --ignored`.
    #[test]
    #[ignore = "requires a live LLM endpoint"]
    fn real_endpoint_synthesizes() {
        let Some(s) = LlmSynthesizer::from_env() else {
            return;
        };
        let out = s
            .synthesize(&["user prefers dark mode", "the user likes dark mode"])
            .unwrap();
        assert!(!out.is_empty());
    }
}
