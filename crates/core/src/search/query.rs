//! Defensive cleanup of agent-issued recall queries.
//!
//! Agents sometimes prepend system-prompt or context preamble to the actual
//! question, burying the signal both embeddings and BM25 rank on (MemPalace
//! measured an 89.8% → 1.0% R@10 cliff from exactly this). The cascade keeps
//! human-sized queries untouched and recovers the question from bloated ones.

/// Chars at or under which a query is passed through untouched.
const PASS_THROUGH_CHARS: usize = 200;
/// Fallback tail length when no sentence structure is recoverable.
const TAIL_CHARS: usize = 250;
/// A recovered sentence shorter than this is noise ("ok?"), not the question.
const MIN_SENTENCE_CHARS: usize = 8;

/// Sanitize a recall query: short queries pass through untouched; long ones fall
/// through a cascade — last question sentence, else last sentence, else raw tail.
pub fn sanitize_query(query: &str) -> &str {
    let trimmed = query.trim();
    if trimmed.chars().count() <= PASS_THROUGH_CHARS {
        return trimmed;
    }
    // Split into sentences once; the *last* question is the live ask (earlier text
    // is preamble), then fall back to the last substantive sentence, then the tail.
    // Whatever we pick is tail-bounded so a punctuation-free blob can't slip through.
    let sentences = split_sentences(trimmed);
    if let Some(q) = sentences
        .iter()
        .rev()
        .find(|s| s.ends_with('?') && s.chars().count() >= MIN_SENTENCE_CHARS)
    {
        return tail(q, TAIL_CHARS);
    }
    if let Some(s) = sentences
        .iter()
        .rev()
        .find(|s| s.chars().count() >= MIN_SENTENCE_CHARS)
    {
        return tail(s, TAIL_CHARS);
    }
    tail(trimmed, TAIL_CHARS)
}

/// Split into trimmed sentences on `.`/`!`/`?`/newline, keeping each terminator
/// with its sentence (so a recovered question keeps its `?`). Empty pieces dropped.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let end = i + ch.len_utf8();
            let piece = text[start..end].trim_matches(|c: char| c.is_whitespace() || c == '\n');
            if !piece.is_empty() {
                out.push(piece);
            }
            start = end;
        }
    }
    let last = text[start..].trim();
    if !last.is_empty() {
        out.push(last);
    }
    out
}

/// The final `n` chars of `text`, on a char boundary.
fn tail(text: &str, n: usize) -> &str {
    let start = text
        .char_indices()
        .rev()
        .nth(n.saturating_sub(1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    text[start..].trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_query_passes_through() {
        let q = "what database does the user prefer?";
        assert_eq!(sanitize_query(q), q);
        assert_eq!(sanitize_query("  trimmed  "), "trimmed");
    }

    #[test]
    fn recovers_trailing_question_from_preamble() {
        let bloated = format!(
            "{} Now answer the user. What port does the daemon bind?",
            "You are a helpful assistant. ".repeat(12)
        );
        assert_eq!(sanitize_query(&bloated), "What port does the daemon bind?");
    }

    #[test]
    fn prefers_the_last_question() {
        let bloated = format!(
            "{} Is it TCP? Actually which socket path is used?",
            "context preamble padding. ".repeat(12)
        );
        assert_eq!(
            sanitize_query(&bloated),
            "Actually which socket path is used?"
        );
    }

    #[test]
    fn falls_back_to_last_sentence_without_a_question() {
        let bloated = format!(
            "{} The user prefers postgres over mysql.",
            "system instructions here. ".repeat(12)
        );
        assert_eq!(
            sanitize_query(&bloated),
            "The user prefers postgres over mysql."
        );
    }

    #[test]
    fn falls_back_to_tail_when_no_sentence_structure() {
        let bloated = "x".repeat(400);
        let out = sanitize_query(&bloated);
        assert!(out.chars().count() <= TAIL_CHARS);
        assert!(!out.is_empty());
    }

    #[test]
    fn ignores_tiny_trailing_fragment() {
        // A long preamble followed by a real sentence then a noise fragment "ok?":
        // the noise is too short to win, so the real question is recovered.
        let bloated = format!(
            "{} Which embedding model is the default? ok?",
            "preamble. ".repeat(30)
        );
        assert_eq!(
            sanitize_query(&bloated),
            "Which embedding model is the default?"
        );
    }
}
