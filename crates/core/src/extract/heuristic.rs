//! Tier-0 heuristic extraction — the default, instant, model-free extractor.
//!
//! No ML, no network: split text into statements, keep the ones that carry a
//! "worth remembering" signal, and classify each as a fact, preference, or
//! episode. It is deliberately conservative and necessarily imperfect — that is
//! the point of Tier-0. A Tier-1 ONNX extractor will supersede it for nuance
//! once the `ort` version conflict is resolved.

use std::collections::HashSet;

use crate::Result;
use crate::extract::{Candidate, Extractor};
use crate::store::MemoryKind;

/// Explicit "save this" cues — the strongest signal a user can give.
const SAVE_SIGNALS: &[&str] = &[
    "remember",
    "save this",
    "note that",
    "make a note",
    "don't forget",
    "for future reference",
    "keep in mind",
];

/// First/second-person preference cues → [`MemoryKind::Preference`].
const PREFERENCE_MARKERS: &[&str] = &[
    "i prefer",
    "i like",
    "i love",
    "i hate",
    "i dislike",
    "i don't like",
    "i enjoy",
    "i can't stand",
    "i want",
    "i'd rather",
    "i would rather",
    "my favorite",
    "i always",
    "i never",
    "please always",
    "please don't",
];

/// Temporal / activity cues → [`MemoryKind::Episode`].
const EPISODE_MARKERS: &[&str] = &[
    "yesterday",
    "today",
    "this morning",
    "this afternoon",
    "tonight",
    "last night",
    "last week",
    "earlier",
    " ago",
    "i met",
    "we met",
    "i talked",
    "i spoke",
    "i went",
];

/// Stable-statement / decision cues → [`MemoryKind::Fact`].
const FACT_MARKERS: &[&str] = &[
    "i am",
    "i'm",
    "my name is",
    "i work",
    "i use",
    "we use",
    "i'm using",
    "we're using",
    "i decided",
    "we decided",
    "i chose",
    "we chose",
    "is built with",
    "the project",
    "the api",
    "the database",
    "the deadline",
    "works at",
    "lives in",
    "based in",
];

/// The default model-free [`Extractor`].
pub struct HeuristicExtractor {
    /// Drop candidates below this confidence.
    min_confidence: f32,
    /// Drop statements with fewer than this many words (too terse to be useful).
    min_words: usize,
}

impl Default for HeuristicExtractor {
    fn default() -> Self {
        HeuristicExtractor {
            min_confidence: 0.5,
            min_words: 3,
        }
    }
}

impl HeuristicExtractor {
    /// Build with explicit thresholds.
    pub fn new(min_confidence: f32, min_words: usize) -> Self {
        HeuristicExtractor {
            min_confidence,
            min_words,
        }
    }

    /// Classify a single statement, or `None` if it carries no memory signal.
    fn classify(&self, sentence: &str) -> Option<Candidate> {
        let lower = sentence.to_lowercase();
        let contains = |markers: &[&str]| markers.iter().any(|m| lower.contains(m));

        // Order matters: an explicit "remember" wins; a stated preference beats a
        // merely temporal framing; a bare fact is the weakest signal.
        let (kind, confidence) = if contains(SAVE_SIGNALS) {
            (classify_save_kind(&lower), 0.95)
        } else if contains(PREFERENCE_MARKERS) {
            (MemoryKind::Preference, 0.7)
        } else if contains(EPISODE_MARKERS) {
            (MemoryKind::Episode, 0.65)
        } else if contains(FACT_MARKERS) {
            (MemoryKind::Fact, 0.7)
        } else {
            return None;
        };

        Some(Candidate {
            content: sentence.to_string(),
            kind,
            expires_at: None,
            confidence,
        })
    }
}

impl Extractor for HeuristicExtractor {
    fn extract(&self, text: &str) -> Result<Vec<Candidate>> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = Vec::new();
        for sentence in segment(text) {
            if word_count(sentence) < self.min_words {
                continue;
            }
            let Some(candidate) = self.classify(sentence) else {
                continue;
            };
            if candidate.confidence < self.min_confidence {
                continue;
            }
            // Drop exact (case-insensitive) duplicates within this batch.
            if seen.insert(candidate.content.to_lowercase()) {
                out.push(candidate);
            }
        }
        Ok(out)
    }
}

/// Even under an explicit save cue, sub-classify by preference vs episode vs fact.
fn classify_save_kind(lower: &str) -> MemoryKind {
    let contains = |markers: &[&str]| markers.iter().any(|m| lower.contains(m));
    if contains(PREFERENCE_MARKERS) {
        MemoryKind::Preference
    } else if contains(EPISODE_MARKERS) {
        MemoryKind::Episode
    } else {
        MemoryKind::Fact
    }
}

/// Split text into trimmed, non-empty statements on sentence terminators and newlines.
fn segment(text: &str) -> Vec<&str> {
    text.split(['.', '!', '?', '\n', '\r'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(text: &str) -> Vec<Candidate> {
        HeuristicExtractor::default().extract(text).unwrap()
    }

    #[test]
    fn detects_preference() {
        let c = extract("I prefer dark mode in my editor");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].kind, MemoryKind::Preference);
    }

    #[test]
    fn detects_fact() {
        let c = extract("My name is Alex and I work at Stripe");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].kind, MemoryKind::Fact);
    }

    #[test]
    fn detects_episode() {
        let c = extract("I met Alex yesterday to discuss the design");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].kind, MemoryKind::Episode);
    }

    #[test]
    fn explicit_save_is_high_confidence() {
        let c = extract("Remember to always run the formatter before committing");
        assert_eq!(c.len(), 1);
        assert!(c[0].confidence > 0.9);
    }

    #[test]
    fn preference_beats_temporal_framing() {
        // Has the episode cue "today" but the durable signal is the preference.
        let c = extract("Today I realized I prefer tabs over spaces");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].kind, MemoryKind::Preference);
    }

    #[test]
    fn ignores_non_memory_and_short_text() {
        assert!(extract("How are you doing?").is_empty());
        assert!(extract("ok").is_empty());
        assert!(extract("I am").is_empty()); // under min_words
    }

    #[test]
    fn extracts_multiple_sentences() {
        let c = extract("I prefer Rust. We decided to use SQLite for storage.");
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].kind, MemoryKind::Preference);
        assert_eq!(c[1].kind, MemoryKind::Fact);
    }

    #[test]
    fn deduplicates_within_batch() {
        let c = extract("I prefer dark mode\nI prefer dark mode");
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn confidence_threshold_filters() {
        // Raise the bar above the fact/preference score (0.7) but below save (0.95).
        let strict = HeuristicExtractor::new(0.9, 3);
        let facts = strict.extract("I work at Stripe").unwrap();
        assert!(facts.is_empty());
        let saved = strict.extract("Remember that I work at Stripe").unwrap();
        assert_eq!(saved.len(), 1);
    }

    #[test]
    fn into_memory_carries_fields() {
        let c = extract("I prefer dark mode in my editor").pop().unwrap();
        let m = c.clone().into_memory("id1", "tag", vec![0.0, 1.0]);
        assert_eq!(m.id, "id1");
        assert_eq!(m.content, c.content);
        assert_eq!(m.kind, MemoryKind::Preference);
        assert_eq!(m.container_tag, "tag");
    }
}
