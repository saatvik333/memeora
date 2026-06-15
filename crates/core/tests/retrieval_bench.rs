//! Offline retrieval-quality regression guard — a tiny LongMemEval/LoCoMo-shaped
//! harness. Hand-placed vectors + real BM25 over an in-memory store, no ONNX, so it
//! runs in CI and pins the vision's headline retrieval claim: **hybrid (RRF) recall
//! is at least as good as either single signal, and strictly better than
//! vector-only** on a set where dense ranks a distractor on top.
//!
//! Unit tests prove the fusion *mechanics* on 2–4 synthetic rows; this proves the
//! *outcome* with a recall/MRR metric over a small corpus, so a change to `rrf_fuse`,
//! `rrf_k`, the candidate pool, or a future channel that silently degrades recall
//! trips here while the unit tests stay green. It is also the seed the step-11
//! benchmark (LoCoMo/LongMemEval scored vs Hindsight) grows from.

use memeora_core::SqliteStore;
use memeora_core::search::{SearchMode, SearchParams, search};
use memeora_core::store::{Memory, MemoryKind, VectorStore};

const TAG: &str = "bench";

struct Query {
    text: &'static str,
    vec: [f32; 3],
    gold: &'static str,
}

/// A small corpus across three topics (rust / docker / coffee). Each topic carries
/// the gold memory plus a distractor that is *closer in vector space* (a dense
/// distractor) or shares query terms but sits far in vector space.
fn corpus() -> SqliteStore {
    let mut s = SqliteStore::open_in_memory(3).unwrap();
    let docs: &[(&str, &str, [f32; 3])] = &[
        (
            "g_rust",
            "rust is my favorite programming language",
            [0.80, 0.20, 0.00],
        ),
        // Dense distractor for the rust query: closer vector, no shared query terms.
        (
            "go",
            "golang is great for building systems",
            [0.97, 0.03, 0.00],
        ),
        // Lexical distractor: shares "rust" but far in vector space.
        (
            "rust_safety",
            "rust prevents memory safety bugs",
            [0.10, 0.10, 0.80],
        ),
        (
            "g_docker",
            "deploy services with docker compose",
            [0.20, 0.80, 0.00],
        ),
        // Dense distractor for the docker query.
        (
            "k8s",
            "kubernetes orchestrates container clusters",
            [0.05, 0.95, 0.00],
        ),
        (
            "g_coffee",
            "i drink coffee every morning",
            [0.00, 0.10, 0.90],
        ),
        ("tea", "tea is a calming beverage", [0.00, 0.05, 0.95]),
    ];
    for (id, content, v) in docs {
        s.upsert(&Memory::new(
            *id,
            *content,
            MemoryKind::Fact,
            TAG,
            v.to_vec(),
        ))
        .unwrap();
    }
    s
}

fn queries() -> Vec<Query> {
    vec![
        // Dense ranks `go` above `g_rust`; BM25 ranks `g_rust` top → fusion recovers it.
        Query {
            text: "rust programming language",
            vec: [0.90, 0.10, 0.00],
            gold: "g_rust",
        },
        // Dense ranks `k8s` above `g_docker`; BM25 ranks `g_docker` top.
        Query {
            text: "docker compose deploy",
            vec: [0.10, 0.90, 0.00],
            gold: "g_docker",
        },
        // Easy case: both signals agree on the gold.
        Query {
            text: "coffee morning",
            vec: [0.00, 0.10, 0.90],
            gold: "g_coffee",
        },
    ]
}

/// Mean reciprocal rank of the gold memory across the query set, in `mode`.
fn mrr(store: &SqliteStore, qs: &[Query], mode: SearchMode) -> f32 {
    let mut sum = 0.0;
    for q in qs {
        let params = SearchParams {
            k: 5,
            mode,
            ..SearchParams::default()
        };
        let hits = search(store, TAG, &q.vec, q.text, &params).unwrap();
        if let Some(pos) = hits.iter().position(|h| h.memory.id == q.gold) {
            sum += 1.0 / (pos as f32 + 1.0);
        }
    }
    sum / qs.len() as f32
}

/// Fraction of queries whose gold memory is within the top `k` (in `mode`).
fn recall_at_k(store: &SqliteStore, qs: &[Query], mode: SearchMode, k: usize) -> f32 {
    let mut hit = 0.0;
    for q in qs {
        let params = SearchParams {
            k,
            mode,
            ..SearchParams::default()
        };
        let hits = search(store, TAG, &q.vec, q.text, &params).unwrap();
        if hits.iter().take(k).any(|h| h.memory.id == q.gold) {
            hit += 1.0;
        }
    }
    hit / qs.len() as f32
}

#[test]
fn hybrid_recall_beats_single_signals_and_meets_floor() {
    let store = corpus();
    let qs = queries();

    let hybrid = mrr(&store, &qs, SearchMode::Hybrid);
    let vector = mrr(&store, &qs, SearchMode::Vector);
    let text = mrr(&store, &qs, SearchMode::Text);

    // The corpus is built so dense alone ranks a distractor above the gold on two of
    // three queries — so vector-only must trail. This guards against the assertion
    // becoming vacuous if the fixture drifts.
    assert!(
        vector < 1.0,
        "fixture no longer challenges vector-only (MRR {vector})"
    );

    // Headline claim: hybrid (RRF) is at least as good as either single signal, and
    // strictly better than vector-only here.
    assert!(
        hybrid >= vector,
        "hybrid MRR {hybrid} < vector-only {vector}"
    );
    assert!(hybrid >= text, "hybrid MRR {hybrid} < BM25-only {text}");
    assert!(
        hybrid > vector,
        "hybrid MRR {hybrid} should strictly beat vector-only {vector}"
    );

    // Absolute floors — a fusion/RRF/pool regression that drops the gold out of the
    // top results trips these even when the comparative checks still hold.
    assert!(hybrid >= 0.8, "hybrid MRR regressed below floor: {hybrid}");
    assert!(
        recall_at_k(&store, &qs, SearchMode::Hybrid, 3) >= 0.999,
        "hybrid recall@3 regressed below 1.0"
    );
}
