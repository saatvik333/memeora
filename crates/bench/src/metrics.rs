//! Retrieval metrics: recall_any@k, recall_all@k, NDCG@k.
//!
//! All metrics are per-question over a ranked list of retrieved document ids and
//! a gold set of evidence ids; the harness averages them across questions.
//! Questions with an empty gold set are skipped by the harness (the metrics are
//! degenerate there), but the functions stay total: empty gold yields 0.0 for
//! `recall_any`/`ndcg` and (vacuously) 1.0 for `recall_all`.

use std::collections::HashSet;

/// 1.0 if *any* gold id appears in the top `k` retrieved ids, else 0.0.
pub fn recall_any_at_k(retrieved: &[String], gold: &HashSet<String>, k: usize) -> f64 {
    if retrieved.iter().take(k).any(|id| gold.contains(id)) {
        1.0
    } else {
        0.0
    }
}

/// 1.0 if *every* gold id appears in the top `k` retrieved ids, else 0.0.
pub fn recall_all_at_k(retrieved: &[String], gold: &HashSet<String>, k: usize) -> f64 {
    let top: HashSet<&String> = retrieved.iter().take(k).collect();
    if gold.iter().all(|id| top.contains(id)) {
        1.0
    } else {
        0.0
    }
}

/// NDCG@k with binary relevance: DCG counts each gold id found at (1-based)
/// rank `r` as `1/log2(r+1)`; the ideal DCG packs `min(|gold|, k)` hits at the
/// top. 0.0 when the gold set is empty.
pub fn ndcg_at_k(retrieved: &[String], gold: &HashSet<String>, k: usize) -> f64 {
    if gold.is_empty() {
        return 0.0;
    }
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, id)| gold.contains(*id))
        .map(|(i, _)| 1.0 / ((i + 2) as f64).log2())
        .sum();
    let ideal_hits = gold.len().min(k);
    let idcg: f64 = (0..ideal_hits).map(|i| 1.0 / ((i + 2) as f64).log2()).sum();
    dcg / idcg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn gold(v: &[&str]) -> HashSet<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn recall_any_respects_the_cutoff() {
        let retrieved = ids(&["x", "a"]);
        let g = gold(&["a"]);
        assert_eq!(recall_any_at_k(&retrieved, &g, 1), 0.0);
        assert_eq!(recall_any_at_k(&retrieved, &g, 2), 1.0);
        assert_eq!(recall_any_at_k(&[], &g, 10), 0.0);
    }

    #[test]
    fn recall_all_requires_every_gold_id_in_top_k() {
        let retrieved = ids(&["a", "x", "b"]);
        let g = gold(&["a", "b"]);
        assert_eq!(recall_all_at_k(&retrieved, &g, 2), 0.0);
        assert_eq!(recall_all_at_k(&retrieved, &g, 3), 1.0);
        // Vacuously satisfied with no gold ids (harness skips these anyway).
        assert_eq!(recall_all_at_k(&retrieved, &gold(&[]), 3), 1.0);
    }

    #[test]
    fn ndcg_hand_computed() {
        // gold {a,b}, retrieved [a, x, b] @3:
        //   DCG  = 1/log2(2) + 1/log2(4)         = 1.0 + 0.5
        //   IDCG = 1/log2(2) + 1/log2(3)
        let retrieved = ids(&["a", "x", "b"]);
        let g = gold(&["a", "b"]);
        let expected = 1.5 / (1.0 + 1.0 / 3f64.log2());
        assert!((ndcg_at_k(&retrieved, &g, 3) - expected).abs() < 1e-12);
    }

    #[test]
    fn ndcg_extremes() {
        let g = gold(&["a", "b"]);
        // Perfect ranking → 1.0.
        assert!((ndcg_at_k(&ids(&["a", "b", "x"]), &g, 10) - 1.0).abs() < 1e-12);
        // No hits → 0.0; hit below the cutoff → 0.0; empty gold → 0.0.
        assert_eq!(ndcg_at_k(&ids(&["x", "y"]), &g, 10), 0.0);
        assert_eq!(ndcg_at_k(&ids(&["x", "a"]), &g, 1), 0.0);
        assert_eq!(ndcg_at_k(&ids(&["a"]), &gold(&[]), 10), 0.0);
    }
}
