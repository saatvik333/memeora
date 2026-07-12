//! The benchmark loop: ingest each bank into a fresh in-memory store built from
//! memeora's real engine, recall with each question, and score the retrieval.

use std::collections::HashSet;

use memeora_core::{
    EmbeddingProvider, Memory, MemoryKind, SearchParams, SqliteStore, VectorStore, search,
};
use serde::Serialize;

use crate::BoxError;
use crate::datasets::{Bank, BenchQuestion};
use crate::metrics;
use crate::split::{self, SplitChoice};

/// Retrieval depth per query; the `--k` metric cutoff is applied on this list.
pub const RETRIEVE_DEPTH: usize = 50;

/// NDCG is always reported at this fixed cutoff, independent of `--k`.
pub const NDCG_K: usize = 10;

/// Container tag every bench memory is scoped under.
const TAG: &str = "bench";

/// Run parameters (from the CLI).
pub struct RunConfig {
    /// Cutoff for recall_any@k / recall_all@k.
    pub k: usize,
    /// Which seed-42 partition of the questions to evaluate.
    pub split: SplitChoice,
    /// Evaluate at most this many questions (after split filtering).
    pub limit: Option<usize>,
}

/// One retrieved document in a per-question result row.
#[derive(Debug, Serialize)]
pub struct RetrievedDoc {
    /// Document (session) id.
    pub id: String,
    /// RRF relevance score from [`search`] (higher is more relevant).
    pub score: f32,
    /// Whether this document is in the question's gold set.
    pub hit: bool,
}

/// Per-question result, serialized as one JSONL row when `--out` is given.
#[derive(Debug, Serialize)]
pub struct QuestionResult {
    /// Question id.
    pub question_id: String,
    /// Question category, for per-type aggregates.
    pub question_type: String,
    /// Gold evidence document ids.
    pub gold: Vec<String>,
    /// Retrieved documents, best first, up to [`RETRIEVE_DEPTH`].
    pub retrieved: Vec<RetrievedDoc>,
    /// recall_any@k (1.0 = hit, 0.0 = miss).
    pub recall_any: f64,
    /// recall_all@k.
    pub recall_all: f64,
    /// NDCG@[`NDCG_K`].
    pub ndcg: f64,
}

/// The full run: per-question results plus how many questions were skipped
/// because they carry no gold evidence ids (e.g. abstention/adversarial ones).
pub struct RunOutput {
    /// One row per evaluated question.
    pub results: Vec<QuestionResult>,
    /// Questions in the requested split that had an empty gold set.
    pub skipped_no_gold: usize,
}

/// Evaluate `banks` with `embedder` under `cfg`.
///
/// Per bank: embed all documents in one batch, upsert them into a fresh
/// [`SqliteStore::open_in_memory`] (one memory per session, id = session id),
/// then run memeora's hybrid [`search`] for each question and score the ranked
/// session ids against the gold set.
pub fn run(
    banks: &[Bank],
    embedder: &dyn EmbeddingProvider,
    cfg: &RunConfig,
) -> Result<RunOutput, BoxError> {
    let dev = split::dev_ids(
        banks
            .iter()
            .flat_map(|b| &b.questions)
            .map(|q| q.id.as_str()),
    );
    let depth = RETRIEVE_DEPTH.max(cfg.k);
    let params = SearchParams {
        k: depth,
        ..SearchParams::default()
    };
    let limit = cfg.limit.unwrap_or(usize::MAX);

    let mut results = Vec::new();
    let mut skipped_no_gold = 0usize;
    'banks: for bank in banks {
        if results.len() >= limit {
            break;
        }
        let in_split: Vec<&BenchQuestion> = bank
            .questions
            .iter()
            .filter(|q| split::keep(cfg.split, &dev, &q.id))
            .collect();
        let (ask, no_gold): (Vec<_>, Vec<_>) =
            in_split.into_iter().partition(|q| !q.gold.is_empty());
        skipped_no_gold += no_gold.len();
        if ask.is_empty() {
            continue;
        }

        // Fresh, isolated store per bank — the engine's real ingestion surface.
        let texts: Vec<&str> = bank.docs.iter().map(|d| d.text.as_str()).collect();
        let embeddings = embedder.embed_documents(&texts)?;
        let mut store = SqliteStore::open_in_memory(embedder.dim())?;
        for (doc, embedding) in bank.docs.iter().zip(embeddings) {
            store.upsert(&Memory::new(
                &doc.id,
                &doc.text,
                MemoryKind::Episode,
                TAG,
                embedding,
            ))?;
        }

        for q in ask {
            if results.len() >= limit {
                break 'banks;
            }
            let query_embedding = embedder.embed_query(&q.question)?;
            let hits = search(&store, TAG, &query_embedding, &q.question, &params)?;
            results.push(score_question(q, &hits, cfg.k));
            if results.len() % 50 == 0 {
                eprint!("\r{} questions evaluated", results.len());
            }
        }
    }
    if results.len() >= 50 {
        eprintln!("\r{} questions evaluated", results.len());
    }
    Ok(RunOutput {
        results,
        skipped_no_gold,
    })
}

fn score_question(
    q: &BenchQuestion,
    hits: &[memeora_core::ScoredMemory],
    k: usize,
) -> QuestionResult {
    let gold: HashSet<String> = q.gold.iter().cloned().collect();
    let ids: Vec<String> = hits.iter().map(|h| h.memory.id.clone()).collect();
    let retrieved = hits
        .iter()
        .map(|h| RetrievedDoc {
            id: h.memory.id.clone(),
            score: h.score,
            hit: gold.contains(&h.memory.id),
        })
        .collect();
    QuestionResult {
        question_id: q.id.clone(),
        question_type: q.qtype.clone(),
        gold: q.gold.clone(),
        retrieved,
        recall_any: metrics::recall_any_at_k(&ids, &gold, k),
        recall_all: metrics::recall_all_at_k(&ids, &gold, k),
        ndcg: metrics::ndcg_at_k(&ids, &gold, NDCG_K),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::Doc;
    use crate::embedder::HashedBowEmbedder;

    fn bank() -> Bank {
        Bank {
            docs: vec![
                Doc {
                    id: "s1".into(),
                    text: "user: I adopted a golden retriever puppy last spring\n".into(),
                },
                Doc {
                    id: "s2".into(),
                    text: "user: we deployed the api server with docker compose\n".into(),
                },
            ],
            questions: vec![BenchQuestion {
                id: "q1".into(),
                qtype: "single-session-user".into(),
                question: "which golden retriever puppy did the user adopt?".into(),
                gold: vec!["s1".into()],
            }],
        }
    }

    #[test]
    fn end_to_end_recall_over_the_real_engine() {
        let embedder = HashedBowEmbedder::new(HashedBowEmbedder::DEFAULT_DIM);
        let cfg = RunConfig {
            k: 10,
            split: SplitChoice::All,
            limit: None,
        };
        let out = run(&[bank()], &embedder, &cfg).unwrap();
        assert_eq!(out.results.len(), 1);
        assert_eq!(out.skipped_no_gold, 0);
        let r = &out.results[0];
        assert_eq!(r.question_id, "q1");
        assert_eq!(r.recall_any, 1.0, "gold session must be in the top-10");
        assert_eq!(r.recall_all, 1.0);
        assert_eq!(
            r.retrieved[0].id, "s1",
            "lexical overlap should rank s1 first"
        );
        assert!(r.retrieved[0].hit);
        assert!(r.ndcg > 0.9);
    }

    #[test]
    fn empty_gold_questions_are_skipped_not_scored() {
        let mut b = bank();
        b.questions.push(BenchQuestion {
            id: "q-abs".into(),
            qtype: "abstention".into(),
            question: "unanswerable".into(),
            gold: vec![],
        });
        let embedder = HashedBowEmbedder::new(64);
        let cfg = RunConfig {
            k: 10,
            split: SplitChoice::All,
            limit: None,
        };
        let out = run(&[b], &embedder, &cfg).unwrap();
        assert_eq!(out.results.len(), 1);
        assert_eq!(out.skipped_no_gold, 1);
    }

    #[test]
    fn limit_caps_the_number_of_evaluated_questions() {
        let banks = vec![bank(), bank()];
        let embedder = HashedBowEmbedder::new(64);
        let cfg = RunConfig {
            k: 10,
            split: SplitChoice::All,
            limit: Some(1),
        };
        let out = run(&banks, &embedder, &cfg).unwrap();
        assert_eq!(out.results.len(), 1);
    }
}
