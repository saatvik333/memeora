//! Deterministic dev/held-out partition of the benchmark questions.
//!
//! Tuning must never report on the data it was tuned against. The partition is
//! a seed-42 FNV-1a hash over each question id: the [`DEV_SIZE`] ids with the
//! lowest hashes form the **dev** split (tune against this), everything else is
//! **held-out** (report on this). Because the hash is content-derived and the
//! seed fixed, the partition is identical on every machine and every run — no
//! RNG state, no ordering dependence.

use std::collections::HashSet;

use clap::ValueEnum;

use crate::hash::fnv1a_seeded;

/// Number of questions in the dev split (all of them, if fewer exist).
pub const DEV_SIZE: usize = 50;

/// Seed for the partition hash. Fixed forever: changing it silently re-splits.
pub const SPLIT_SEED: u64 = 42;

/// Which partition of the questions to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SplitChoice {
    /// The 50-question tuning split.
    Dev,
    /// Everything not in dev — report final numbers here.
    HeldOut,
    /// No filtering.
    All,
}

impl SplitChoice {
    /// CLI-facing name (matches the clap value names).
    pub fn as_str(self) -> &'static str {
        match self {
            SplitChoice::Dev => "dev",
            SplitChoice::HeldOut => "held-out",
            SplitChoice::All => "all",
        }
    }
}

/// The dev split: the [`DEV_SIZE`] ids with the lowest seed-42 hashes.
/// Ties (identical hashes) break lexicographically by id, so the result is
/// fully deterministic regardless of input order.
pub fn dev_ids<'a, I>(ids: I) -> HashSet<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut ranked: Vec<(u64, &str)> = ids
        .into_iter()
        .map(|id| (fnv1a_seeded(SPLIT_SEED, id.as_bytes()), id))
        .collect();
    ranked.sort_unstable();
    ranked.truncate(DEV_SIZE);
    ranked.into_iter().map(|(_, id)| id.to_owned()).collect()
}

/// Whether a question id belongs to the requested split.
pub fn keep(choice: SplitChoice, dev: &HashSet<String>, id: &str) -> bool {
    match choice {
        SplitChoice::All => true,
        SplitChoice::Dev => dev.contains(id),
        SplitChoice::HeldOut => !dev.contains(id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question_ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("q{i}")).collect()
    }

    #[test]
    fn dev_is_exactly_fifty_and_deterministic() {
        let ids = question_ids(120);
        let a = dev_ids(ids.iter().map(String::as_str));
        let b = dev_ids(ids.iter().rev().map(String::as_str)); // order must not matter
        assert_eq!(a.len(), DEV_SIZE);
        assert_eq!(a, b);
        assert!(a.iter().all(|id| ids.contains(id)));
    }

    #[test]
    fn small_datasets_are_entirely_dev() {
        let ids = question_ids(30);
        let dev = dev_ids(ids.iter().map(String::as_str));
        assert_eq!(dev.len(), 30);
    }

    #[test]
    fn splits_partition_the_ids() {
        let ids = question_ids(120);
        let dev = dev_ids(ids.iter().map(String::as_str));
        let (d, h): (Vec<_>, Vec<_>) = ids.iter().partition(|id| keep(SplitChoice::Dev, &dev, id));
        assert_eq!(d.len(), DEV_SIZE);
        assert_eq!(h.len(), 120 - DEV_SIZE);
        // held-out is the complement, `all` keeps everything.
        assert!(h.iter().all(|id| keep(SplitChoice::HeldOut, &dev, id)));
        assert!(ids.iter().all(|id| keep(SplitChoice::All, &dev, id)));
    }
}
