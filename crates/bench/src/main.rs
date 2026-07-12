//! `memeora-bench` — offline retrieval benchmarks for the memeora engine.
//!
//! See `crates/bench/README.md` for dataset sources, commands, and what the
//! numbers mean (retrieval recall, not QA accuracy).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use memeora_bench::BoxError;
use memeora_bench::datasets::{Bank, locomo, longmemeval};
use memeora_bench::embedder::HashedBowEmbedder;
use memeora_bench::harness::{self, RunConfig};
use memeora_bench::report;
use memeora_bench::split::SplitChoice;
use memeora_core::EmbeddingProvider;

#[derive(Parser)]
#[command(
    name = "memeora-bench",
    version,
    about = "Offline retrieval benchmarks for the memeora engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// LongMemEval: per-question haystack sessions; gold = answer_session_ids.
    Longmemeval(RunArgs),
    /// LoCoMo: per-conversation sessions; gold = sessions named by QA evidence ids.
    Locomo(RunArgs),
}

#[derive(Args)]
struct RunArgs {
    /// Path to the dataset JSON file.
    #[arg(long)]
    data: PathBuf,
    /// Metric cutoff for recall_any@k / recall_all@k (NDCG stays @10).
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Evaluate at most N questions (applied after split filtering).
    #[arg(long)]
    limit: Option<usize>,
    /// Write per-question results as JSONL to this path.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Which seed-42 partition to evaluate: tune on dev, report on held-out.
    #[arg(long, value_enum, default_value_t = SplitChoice::All)]
    split: SplitChoice,
    /// Embed with the real fastembed model instead of the offline hashed
    /// embedder (requires building with `--features real-embeddings`;
    /// downloads model weights on first use).
    #[arg(long)]
    real_embeddings: bool,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type Loader = fn(&Path) -> Result<Vec<Bank>, BoxError>;

fn run(cli: Cli) -> Result<(), BoxError> {
    let (name, loader, args): (&str, Loader, RunArgs) = match cli.command {
        Command::Longmemeval(args) => ("LongMemEval", longmemeval::load, args),
        Command::Locomo(args) => ("LoCoMo", locomo::load, args),
    };
    let banks = loader(&args.data)?;
    let embedder = make_embedder(args.real_embeddings)?;
    let cfg = RunConfig {
        k: args.k,
        split: args.split,
        limit: args.limit,
    };
    let output = harness::run(&banks, embedder.as_ref(), &cfg)?;

    println!(
        "{name} — split={}, k={}, embedder={}, evaluated={} (skipped {} with no gold ids)",
        args.split.as_str(),
        args.k,
        embedder.space().namespace(),
        output.results.len(),
        output.skipped_no_gold
    );
    print!(
        "{}",
        report::render_table(&report::aggregate(&output.results), args.k)
    );

    if let Some(path) = &args.out {
        report::write_jsonl(path, &output.results)?;
        eprintln!("wrote {} rows to {}", output.results.len(), path.display());
    }
    Ok(())
}

fn make_embedder(real: bool) -> Result<Box<dyn EmbeddingProvider>, BoxError> {
    if real {
        #[cfg(feature = "real-embeddings")]
        {
            let embedder = memeora_core::embed::fastembed::FastEmbedder::bge_small(None)?;
            return Ok(Box::new(embedder));
        }
        #[cfg(not(feature = "real-embeddings"))]
        return Err(
            "--real-embeddings requires building with `--features real-embeddings` \
             (compiles the ONNX stack and downloads model weights on first use)"
                .into(),
        );
    }
    Ok(Box::new(HashedBowEmbedder::new(
        HashedBowEmbedder::DEFAULT_DIM,
    )))
}
