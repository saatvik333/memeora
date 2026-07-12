//! Aggregation and output: per-type mean metrics, a plain-text table, and the
//! per-question JSONL file.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::harness::{NDCG_K, QuestionResult};

/// One aggregate row: a question type (or `overall`) with mean metrics.
#[derive(Debug)]
pub struct AggRow {
    /// Question type, or `"overall"` for the all-questions row.
    pub label: String,
    /// Number of questions in the group.
    pub n: usize,
    /// Mean recall_any@k.
    pub recall_any: f64,
    /// Mean recall_all@k.
    pub recall_all: f64,
    /// Mean NDCG@10.
    pub ndcg: f64,
}

/// Group results by `question_type` (sorted) and append an `overall` row.
pub fn aggregate(results: &[QuestionResult]) -> Vec<AggRow> {
    if results.is_empty() {
        return Vec::new();
    }
    let mut groups: BTreeMap<&str, Vec<&QuestionResult>> = BTreeMap::new();
    for r in results {
        groups.entry(&r.question_type).or_default().push(r);
    }
    let mut rows: Vec<AggRow> = groups
        .into_iter()
        .map(|(label, group)| mean_row(label, &group))
        .collect();
    rows.push(mean_row("overall", &results.iter().collect::<Vec<_>>()));
    rows
}

fn mean_row(label: &str, group: &[&QuestionResult]) -> AggRow {
    let n = group.len();
    let mean = |f: fn(&QuestionResult) -> f64| group.iter().map(|r| f(r)).sum::<f64>() / n as f64;
    AggRow {
        label: label.to_owned(),
        n,
        recall_any: mean(|r| r.recall_any),
        recall_all: mean(|r| r.recall_all),
        ndcg: mean(|r| r.ndcg),
    }
}

/// Render the aggregate rows as an aligned plain-text table.
pub fn render_table(rows: &[AggRow], k: usize) -> String {
    if rows.is_empty() {
        return "no questions evaluated\n".to_owned();
    }
    let any = format!("recall_any@{k}");
    let all = format!("recall_all@{k}");
    let ndcg = format!("ndcg@{NDCG_K}");
    let width = rows
        .iter()
        .map(|r| r.label.len())
        .chain(["question_type".len()])
        .max()
        .unwrap_or(0);
    let mut out = format!(
        "{:<width$}  {:>6}  {any:>14}  {all:>14}  {ndcg:>8}\n",
        "question_type", "n"
    );
    for r in rows {
        out.push_str(&format!(
            "{:<width$}  {:>6}  {:>14.4}  {:>14.4}  {:>8.4}\n",
            r.label, r.n, r.recall_any, r.recall_all, r.ndcg
        ));
    }
    out
}

/// Write one JSON object per question to `path` (JSONL).
pub fn write_jsonl(path: &Path, results: &[QuestionResult]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for r in results {
        serde_json::to_writer(&mut w, r).map_err(std::io::Error::other)?;
        w.write_all(b"\n")?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(qtype: &str, recall_any: f64, recall_all: f64, ndcg: f64) -> QuestionResult {
        QuestionResult {
            question_id: "q".into(),
            question_type: qtype.into(),
            gold: vec![],
            retrieved: vec![],
            recall_any,
            recall_all,
            ndcg,
        }
    }

    #[test]
    fn aggregates_per_type_plus_overall() {
        let results = vec![
            result("a", 1.0, 1.0, 1.0),
            result("a", 0.0, 0.0, 0.5),
            result("b", 1.0, 0.0, 0.25),
        ];
        let rows = aggregate(&results);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].label, "a");
        assert_eq!(rows[0].n, 2);
        assert!((rows[0].recall_any - 0.5).abs() < 1e-12);
        assert!((rows[0].ndcg - 0.75).abs() < 1e-12);
        let overall = rows.last().unwrap();
        assert_eq!(overall.label, "overall");
        assert_eq!(overall.n, 3);
        assert!((overall.recall_any - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn empty_results_render_a_notice() {
        assert!(aggregate(&[]).is_empty());
        assert_eq!(render_table(&[], 10), "no questions evaluated\n");
    }

    #[test]
    fn table_contains_headers_and_values() {
        let rows = aggregate(&[result("temporal-reasoning", 1.0, 1.0, 1.0)]);
        let table = render_table(&rows, 10);
        assert!(table.contains("recall_any@10"));
        assert!(table.contains("ndcg@10"));
        assert!(table.contains("temporal-reasoning"));
        assert!(table.contains("overall"));
    }
}
