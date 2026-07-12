//! LoCoMo loader (snap-research/locomo, `data/locomo10.json`).
//!
//! Each sample is one very long two-speaker conversation split into sessions
//! (`conversation.session_<n>` arrays, with `session_<n>_date_time` strings)
//! plus a shared `qa` list whose `evidence` entries are dialog-turn ids like
//! `"D1:3"` (session 1, turn 3). The harness scores retrieval at *session*
//! granularity (mirroring LongMemEval), so evidence ids are mapped to their
//! containing session: `"D1:3"` → `session_1`. One [`Bank`] per conversation,
//! shared by all of its QA items; adversarial questions with no evidence get an
//! empty gold set and are skipped by the harness.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde_json::Value;

use super::{Bank, BenchQuestion, Doc};
use crate::BoxError;

/// Load a LoCoMo JSON file into one [`Bank`] per conversation.
pub fn load(path: &Path) -> Result<Vec<Bank>, BoxError> {
    parse(&fs::read_to_string(path)?)
}

/// Parse LoCoMo JSON (a top-level array of conversation samples).
pub fn parse(json: &str) -> Result<Vec<Bank>, BoxError> {
    let samples: Vec<Value> = serde_json::from_str(json)?;
    samples
        .iter()
        .enumerate()
        .map(|(i, sample)| bank_of(i, sample))
        .collect()
}

fn bank_of(index: usize, sample: &Value) -> Result<Bank, BoxError> {
    let sample_id = sample
        .get("sample_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("sample_{index}"));

    let conv = sample
        .get("conversation")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{sample_id}: missing `conversation` object"))?;

    // Session keys are `session_<n>`; `session_<n>_date_time` fails the numeric
    // parse and is looked up separately per session.
    let mut nums: Vec<u64> = conv
        .keys()
        .filter_map(|k| k.strip_prefix("session_")?.parse().ok())
        .collect();
    nums.sort_unstable();
    nums.dedup();

    let mut docs = Vec::new();
    for n in nums {
        let key = format!("session_{n}");
        let Some(turns) = conv.get(&key).and_then(Value::as_array) else {
            continue;
        };
        let mut text = String::new();
        if let Some(date) = conv
            .get(&format!("{key}_date_time"))
            .and_then(Value::as_str)
        {
            text.push_str(date);
            text.push('\n');
        }
        for turn in turns {
            let speaker = turn.get("speaker").and_then(Value::as_str).unwrap_or("");
            let line = turn.get("text").and_then(Value::as_str).unwrap_or("");
            text.push_str(speaker);
            text.push_str(": ");
            text.push_str(line);
            // Image turns carry a BLIP caption; keep it so image-grounded
            // questions have something lexical to match.
            if let Some(caption) = turn.get("blip_caption").and_then(Value::as_str) {
                text.push_str(" [shared image: ");
                text.push_str(caption);
                text.push(']');
            }
            text.push('\n');
        }
        docs.push(Doc { id: key, text });
    }

    let qa = sample
        .get("qa")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{sample_id}: missing `qa` array"))?;
    let questions = qa
        .iter()
        .enumerate()
        .map(|(qi, item)| BenchQuestion {
            id: format!("{sample_id}:qa{qi}"),
            qtype: match item.get("category").and_then(Value::as_i64) {
                Some(c) => format!("category-{c}"),
                None => "category-unknown".to_owned(),
            },
            question: item
                .get("question")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            gold: evidence_sessions(item.get("evidence")),
        })
        .collect();

    Ok(Bank { docs, questions })
}

/// Map evidence dialog-turn ids (`"D<session>:<turn>"`, possibly nested in
/// arrays) to their containing session document ids, deduplicated.
fn evidence_sessions(evidence: Option<&Value>) -> Vec<String> {
    let mut sessions = BTreeSet::new();
    collect_evidence(evidence, &mut sessions);
    sessions.into_iter().collect()
}

fn collect_evidence(value: Option<&Value>, out: &mut BTreeSet<String>) {
    match value {
        Some(Value::String(s)) => {
            if let Some(n) = parse_dia_session(s) {
                out.insert(format!("session_{n}"));
            }
        }
        Some(Value::Array(items)) => {
            for item in items {
                collect_evidence(Some(item), out);
            }
        }
        _ => {}
    }
}

/// `"D1:3"` → `Some(1)`; anything not of that shape → `None`.
fn parse_dia_session(s: &str) -> Option<u64> {
    s.strip_prefix('D')?.split(':').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_documented_sample_shape() {
        let json = r#"[{
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "Ana",
                "speaker_b": "Ben",
                "session_1_date_time": "1:00 pm on 8 May, 2023",
                "session_1": [
                    {"speaker": "Ana", "dia_id": "D1:1", "text": "I got a dog"},
                    {"speaker": "Ben", "dia_id": "D1:2", "text": "look!",
                     "blip_caption": "a golden retriever"}
                ],
                "session_2_date_time": "2:00 pm on 9 May, 2023",
                "session_2": [
                    {"speaker": "Ana", "dia_id": "D2:1", "text": "went hiking"}
                ]
            },
            "qa": [
                {"question": "what pet does Ana have?", "answer": "a dog",
                 "evidence": ["D1:1"], "category": 4},
                {"question": "unanswerable?", "adversarial_answer": "n/a",
                 "evidence": [], "category": 5}
            ]
        }]"#;
        let banks = parse(json).unwrap();
        assert_eq!(banks.len(), 1);
        let bank = &banks[0];
        assert_eq!(bank.docs.len(), 2);
        assert_eq!(bank.docs[0].id, "session_1");
        assert!(bank.docs[0].text.contains("Ana: I got a dog"));
        assert!(
            bank.docs[0]
                .text
                .contains("[shared image: a golden retriever]")
        );
        assert!(bank.docs[0].text.starts_with("1:00 pm on 8 May, 2023"));

        assert_eq!(bank.questions.len(), 2);
        let q0 = &bank.questions[0];
        assert_eq!(q0.id, "conv-1:qa0");
        assert_eq!(q0.qtype, "category-4");
        assert_eq!(q0.gold, vec!["session_1".to_string()]);
        // Adversarial question: no evidence → empty gold (harness skips it).
        assert!(bank.questions[1].gold.is_empty());
    }

    #[test]
    fn evidence_ids_dedupe_to_sessions_and_ignore_junk() {
        let ev = serde_json::json!(["D1:1", "D1:9", ["D2:3"], "not-an-id", 7]);
        assert_eq!(
            evidence_sessions(Some(&ev)),
            vec!["session_1".to_string(), "session_2".to_string()]
        );
        assert!(evidence_sessions(None).is_empty());
    }
}
