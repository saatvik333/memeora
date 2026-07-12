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
        .then(|| canonicalize(&token.to_ascii_lowercase()))
}

/// Curated allowlist of unambiguous, well-known dev synonyms (surface -> canonical).
/// This is an explicit, hand-verified lookup table — **not** a fuzzy matcher. Keep
/// it short: only add forms whose mapping is beyond doubt. Some surfaces here (e.g.
/// `"js"`, `"k8s"`) only ever reach [`canonicalize`] if they clear the extractor's
/// length/type gates; the table states policy, the extractor is the gate.
const ALIASES: &[(&str, &str)] = &[
    ("postgresql", "postgres"),
    ("js", "javascript"),
    ("ts", "typescript"),
    ("k8s", "kubernetes"),
];

/// Fold trivially-equivalent surface variants of an already-lowercased entity onto
/// one canonical form, doing only *safe* normalizations so the graph channel links
/// "Postgres"/"postgres"/"PostgreSQL" — without ever risking a false merge:
///
/// 1. A curated [`ALIASES`] allowlist of well-known dev synonyms.
/// 2. Naive singular/plural folding (strip a trailing "es"/"s"), applied **only**
///    to plain all-letter words. Anything carrying a path separator, dot,
///    underscore, or digit (paths, `snake_case`/dotted/versioned identifiers) keeps
///    its surface verbatim — a trailing "s" there is load-bearing (`sqlite.rs`,
///    `has_access`, a `docs/` segment must not change).
//
// ponytail: trigram / edit-distance resolution is deliberately omitted — that is
// where false merges live (a false merge links *unrelated* memories, which is worse
// than a miss). The plural rule + alias table capture the safe 80%; fuzzy matching
// could return later, gated by co-occurrence evidence.
fn canonicalize(token: &str) -> String {
    if let Some((_, canonical)) = ALIASES.iter().find(|(surface, _)| *surface == token) {
        return (*canonical).to_string();
    }
    // Only fold plurals on plain all-letter words. A token that is itself a
    // canonical alias target (e.g. "postgres", "kubernetes") is also left alone so
    // the naive rule can't mangle a non-plural word that merely ends in "s".
    if !token.bytes().all(|b| b.is_ascii_lowercase())
        || ALIASES.iter().any(|(_, canonical)| *canonical == token)
    {
        return token.to_string();
    }
    depluralize(token)
}

/// Naive singular/plural fold: strip a trailing "es"/"s". Only ever called for
/// plain all-letter words (see [`canonicalize`]), so it never touches paths or
/// identifiers.
fn depluralize(token: &str) -> String {
    // A "-es" plural (classes, boxes) — but keep a ≥3-char stem to avoid noise.
    if let Some(stem) = token.strip_suffix("es")
        && stem.len() >= 3
    {
        return stem.to_string();
    }
    // A "-s" plural — but words ending in a real double-s (class, access) are not
    // plurals of an "-s"-less stem, so leave those whole.
    if !token.ends_with("ss")
        && let Some(stem) = token.strip_suffix('s')
        && stem.len() >= 3
    {
        return stem.to_string();
    }
    token.to_string()
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
        && token.chars().any(|c| c.is_ascii_alphabetic())
        // Every `_`-delimited segment ≥2 chars: rejects noise like `a_b`, `_x`, `x__`.
        && token.split('_').all(|seg| seg.len() >= 2);
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

    #[test]
    fn skips_short_segment_snake_noise() {
        // `a_b` / `x_` are noise, not identifiers worth tracking; real idents survive.
        assert_eq!(
            extract_entities("touched a_b and x_ and proof_count"),
            vec!["proof_count"]
        );
    }

    #[test]
    fn folds_surface_variants_of_postgres() {
        // Casing + the alias table collapse every spelling onto one entity.
        assert_eq!(extract_entities("Postgres PostgreSQL"), vec!["postgres"]);
    }

    #[test]
    fn folds_singular_and_plural_words() {
        // "APIs" and "API" collapse to one canonical entity.
        assert_eq!(extract_entities("APIs and API"), vec!["api"]);
    }

    #[test]
    fn paths_and_snake_idents_keep_surface() {
        // A path segment or `snake_case` ident must NOT lose a trailing "s" or fold.
        let e =
            extract_entities("see crates/core/src/store/sqlite.rs and proof_count and has_access");
        assert!(
            e.contains(&"crates/core/src/store/sqlite.rs".to_string()),
            "{e:?}"
        );
        assert!(e.contains(&"proof_count".to_string()), "{e:?}");
        assert!(e.contains(&"has_access".to_string()), "{e:?}");
    }

    #[test]
    fn plain_proper_noun_still_works() {
        assert_eq!(extract_entities("Rust"), vec!["rust"]);
    }

    #[test]
    fn alias_table_maps_known_synonyms() {
        assert_eq!(canonicalize("postgresql"), "postgres");
        assert_eq!(canonicalize("js"), "javascript");
        assert_eq!(canonicalize("ts"), "typescript");
        assert_eq!(canonicalize("k8s"), "kubernetes");
    }

    #[test]
    fn keeps_double_s_and_canonical_words_whole() {
        // Real double-s words and canonical alias targets are never depluralized.
        assert_eq!(canonicalize("class"), "class");
        assert_eq!(canonicalize("classes"), "class");
        assert_eq!(canonicalize("access"), "access");
        assert_eq!(canonicalize("postgres"), "postgres");
        assert_eq!(canonicalize("kubernetes"), "kubernetes");
    }
}
