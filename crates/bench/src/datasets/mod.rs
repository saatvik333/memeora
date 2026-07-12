//! Dataset loaders. Each loader normalises its source into [`Bank`]s: documents
//! (one per conversation session) plus the questions asked over them, with gold
//! evidence expressed as *document ids* so the harness stays dataset-agnostic.

pub mod locomo;
pub mod longmemeval;

/// One retrievable document: a conversation session, its turns joined to text.
#[derive(Debug, Clone)]
pub struct Doc {
    /// Stable id, used both as the store memory id and as the gold-evidence key.
    pub id: String,
    /// The session text ingested (and embedded) as one memory.
    pub text: String,
}

/// One benchmark question with its gold evidence document ids.
#[derive(Debug, Clone)]
pub struct BenchQuestion {
    /// Unique question id (drives the dev/held-out partition).
    pub id: String,
    /// Question category (e.g. LongMemEval `question_type`), for per-type aggregates.
    pub qtype: String,
    /// The query text posed to the engine.
    pub question: String,
    /// Ids of the documents that contain the evidence for the answer.
    pub gold: Vec<String>,
}

/// An isolated memory bank: the documents to ingest into one fresh store, plus
/// the questions recalled against it. LongMemEval yields one bank per question
/// (per-question haystack); LoCoMo yields one bank per conversation shared by
/// that conversation's QA items (recall is read-only, so sharing is safe).
#[derive(Debug, Clone)]
pub struct Bank {
    /// Documents ingested into the bank's store.
    pub docs: Vec<Doc>,
    /// Questions evaluated against the bank.
    pub questions: Vec<BenchQuestion>,
}
