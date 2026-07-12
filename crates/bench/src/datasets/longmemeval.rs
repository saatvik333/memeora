//! LongMemEval loader (HF `xiaowu0162/longmemeval-cleaned`).
//!
//! Each item carries its own haystack of chat sessions plus the ids of the
//! sessions that contain the answer evidence — so each question becomes one
//! [`Bank`]: one document per haystack session (turns joined), gold =
//! `answer_session_ids`. Abstention questions (`*_abs`) have no evidence
//! sessions; the harness skips them (their empty gold set makes recall
//! degenerate).

use std::fs;
use std::path::Path;

use serde::Deserialize;

use super::{Bank, BenchQuestion, Doc};
use crate::BoxError;

#[derive(Deserialize)]
struct RawItem {
    question_id: String,
    question: String,
    #[serde(default)]
    question_type: String,
    #[serde(default)]
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<RawTurn>>,
    #[serde(default)]
    answer_session_ids: Vec<String>,
}

#[derive(Deserialize)]
struct RawTurn {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: String,
}

/// Load a LongMemEval JSON file into one [`Bank`] per question.
pub fn load(path: &Path) -> Result<Vec<Bank>, BoxError> {
    parse(&fs::read_to_string(path)?)
}

/// Parse LongMemEval JSON (a top-level array of question items).
pub fn parse(json: &str) -> Result<Vec<Bank>, BoxError> {
    let items: Vec<RawItem> = serde_json::from_str(json)?;
    items.into_iter().map(bank_of).collect()
}

fn bank_of(item: RawItem) -> Result<Bank, BoxError> {
    if item.haystack_session_ids.len() != item.haystack_sessions.len() {
        return Err(format!(
            "{}: {} haystack_session_ids but {} haystack_sessions",
            item.question_id,
            item.haystack_session_ids.len(),
            item.haystack_sessions.len()
        )
        .into());
    }
    let docs = item
        .haystack_session_ids
        .into_iter()
        .zip(item.haystack_sessions)
        .map(|(id, turns)| {
            let mut text = String::new();
            for turn in turns {
                text.push_str(&turn.role);
                text.push_str(": ");
                text.push_str(&turn.content);
                text.push('\n');
            }
            Doc { id, text }
        })
        .collect();
    Ok(Bank {
        docs,
        questions: vec![BenchQuestion {
            id: item.question_id,
            qtype: item.question_type,
            question: item.question,
            gold: item.answer_session_ids,
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_documented_item_shape() {
        let json = r#"[{
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "what did the user adopt?",
            "answer": "a puppy",
            "question_date": "2023/05/20 (Sat) 02:21",
            "haystack_dates": ["2023/05/01", "2023/05/02"],
            "haystack_session_ids": ["s1", "s2"],
            "haystack_sessions": [
                [{"role": "user", "content": "I adopted a puppy"},
                 {"role": "assistant", "content": "congrats!"}],
                [{"role": "user", "content": "deploying with docker"}]
            ],
            "answer_session_ids": ["s1"]
        }]"#;
        let banks = parse(json).unwrap();
        assert_eq!(banks.len(), 1);
        let bank = &banks[0];
        assert_eq!(bank.docs.len(), 2);
        assert_eq!(bank.docs[0].id, "s1");
        assert!(bank.docs[0].text.contains("user: I adopted a puppy"));
        assert!(bank.docs[0].text.contains("assistant: congrats!"));
        let q = &bank.questions[0];
        assert_eq!(q.id, "q1");
        assert_eq!(q.qtype, "single-session-user");
        assert_eq!(q.gold, vec!["s1".to_string()]);
    }

    #[test]
    fn rejects_mismatched_haystack_lengths() {
        let json = r#"[{
            "question_id": "q1",
            "question": "?",
            "haystack_session_ids": ["s1"],
            "haystack_sessions": [],
            "answer_session_ids": []
        }]"#;
        let err = parse(json).unwrap_err().to_string();
        assert!(err.contains("q1"), "error should name the question: {err}");
    }
}
