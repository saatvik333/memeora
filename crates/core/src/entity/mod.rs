//! Heuristic entity canonicalization (VISION "Adapts" — MemPalace `entity_registry`).
//!
//! Pure and offline: pull a conservative set of high-signal mentions from memory
//! text — file paths, code identifiers, and proper nouns — and canonicalize them
//! (lowercase) so memories about the same thing can be linked. A deliberately
//! conservative first cut: better to miss a fuzzy entity than to flood the graph
//! with noise. The consolidation (D) and graph-recall (F) layers treat shared
//! entities as an *additive* signal, so under-extraction degrades gracefully.
// ponytail: heuristic floor; the opt-in Tier-2 local-LLM NER (increment G) can
// supersede this for nuance, but this stays the no-LLM default.

use std::collections::HashSet;

/// Max entities surfaced per text, to bound noise and link fan-out.
const MAX_ENTITIES: usize = 12;

/// Capitalized words that merely start sentences or are too generic to be entities.
/// Compared case-insensitively, so a sentence-initial "The"/"Today" is filtered.
const STOPWORDS: &[&str] = &[
    "the",
    "this",
    "that",
    "these",
    "those",
    "there",
    "here",
    "it",
    "its",
    "we",
    "you",
    "they",
    "he",
    "she",
    "and",
    "but",
    "or",
    "if",
    "when",
    "while",
    "then",
    "also",
    "use",
    "using",
    "used",
    "set",
    "add",
    "added",
    "run",
    "ran",
    "get",
    "got",
    "new",
    "now",
    "not",
    "for",
    "with",
    "from",
    "into",
    "edited",
    "files",
    "file",
    "command",
    "let",
    "make",
    "just",
    "really",
    "today",
    "yesterday",
    "please",
    "remember",
    "note",
    "what",
    "why",
    "how",
    "where",
    "who",
    "our",
    "your",
    "their",
    "is",
    "are",
    "was",
    "were",
    "will",
    "should",
    "could",
    "can",
    "do",
    "does",
    "did",
];

/// Extract canonical entity names from `text` (deduped, capped, lowercased).
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw in text.split_whitespace() {
        let Some(canonical) = entity_of(trim_token(raw)) else {
            continue;
        };
        if seen.insert(canonical.clone()) {
            out.push(canonical);
            if out.len() >= MAX_ENTITIES {
                break;
            }
        }
    }
    out
}

/// Trim surrounding punctuation we never want inside an entity, keeping the
/// path/identifier characters (`/`, `.`, `_`, `-`) that are part of the token.
fn trim_token(raw: &str) -> &str {
    raw.trim_matches(|c: char| !(c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-')))
}

/// The canonical entity for a token, or `None` when it isn't a confident entity.
fn entity_of(token: &str) -> Option<String> {
    if token.len() < 3 {
        return None;
    }
    (is_path(token) || is_code_ident(token) || is_proper_noun(token))
        .then(|| token.to_ascii_lowercase())
}

/// A file-path-like token: slash-separated, every segment non-empty, has alphanumerics.
fn is_path(token: &str) -> bool {
    token.contains('/')
        && token.split('/').all(|seg| !seg.is_empty())
        && token.chars().any(|c| c.is_alphanumeric())
}

/// A code identifier: snake_case (underscored) or CamelCase (≥2 capital humps).
fn is_code_ident(token: &str) -> bool {
    let snake = token.contains('_')
        && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && token.chars().any(|c| c.is_ascii_alphabetic());
    let camel = token.chars().all(|c| c.is_ascii_alphanumeric())
        && token.chars().filter(|c| c.is_ascii_uppercase()).count() >= 2
        && token.starts_with(|c: char| c.is_ascii_uppercase());
    snake || camel
}

/// A proper noun: a capitalized, purely alphabetic word that isn't a stopword.
fn is_proper_noun(token: &str) -> bool {
    token.starts_with(|c: char| c.is_ascii_uppercase())
        && token.chars().all(|c| c.is_ascii_alphabetic())
        && !STOPWORDS.contains(&token.to_ascii_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_paths_idents_and_proper_nouns() {
        let e = extract_entities(
            "edited files crates/core/src/store/sqlite.rs using SqliteStore and proof_count for Postgres",
        );
        assert!(
            e.contains(&"crates/core/src/store/sqlite.rs".to_string()),
            "{e:?}"
        );
        assert!(e.contains(&"sqlitestore".to_string()), "{e:?}");
        assert!(e.contains(&"proof_count".to_string()), "{e:?}");
        assert!(e.contains(&"postgres".to_string()), "{e:?}");
    }

    #[test]
    fn skips_stopwords_and_short_tokens() {
        // Capitalized sentence-starters and short words are not entities.
        assert!(extract_entities("The we use it to go").is_empty());
    }

    #[test]
    fn dedups_and_lowercases() {
        assert_eq!(extract_entities("Rust Rust rust"), vec!["rust"]);
    }
}
