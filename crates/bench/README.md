# memeora-bench

Offline **retrieval benchmark harness** for the memeora engine. It exercises the
real retrieval pipeline — `SqliteStore` (sqlite-vec + FTS5) plus the hybrid
`memeora_core::search` (RRF fusion, graph/temporal channels, boosts) — over
public long-term-memory datasets, and reports recall metrics. Every engine
tuning change gets measured against this harness.

## What the numbers mean (read this first)

- **These are retrieval metrics, not QA accuracy.** A "hit" means the engine
  surfaced the session(s) containing the answer evidence in its top-k — it says
  nothing about whether a downstream model would answer correctly. Retrieval
  recall is an *upper bound* on end-to-end QA quality.
- **The default embedder is deliberately dumb but deterministic**: a hashed
  bag-of-words (signed feature hashing, sqrt-damped term counts, L2-normalised,
  512-d). It needs no network and produces identical numbers on every machine,
  so it is a *stable relative signal* for tuning the engine (fusion, boosts,
  channels) — not an absolute quality ceiling. It captures lexical overlap
  only; use `--real-embeddings` for model-grade semantic vectors.
- **recall_any@k** — fraction of questions where *at least one* gold session is
  in the top-k. **recall_all@k** — fraction where *all* gold sessions are
  (matters for multi-session questions). **NDCG@10** — rank-sensitive: rewards
  putting gold sessions near the top. Retrieval depth is 50 regardless of `--k`.

## Datasets

Neither dataset ships with the repo; download once, then everything is offline.

| Benchmark | Source | File |
| --- | --- | --- |
| LongMemEval | Hugging Face: [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | the `longmemeval_s*` JSON (~500 questions, ~50 sessions each); `oracle` is the evidence-only variant |
| LoCoMo | GitHub: [`snap-research/locomo`](https://github.com/snap-research/locomo) | `data/locomo10.json` (10 conversations, ~200 QA each) |

Example fetch:

```sh
hf download xiaowu0162/longmemeval-cleaned --repo-type dataset --local-dir data/longmemeval
curl -LO https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## Commands

```sh
# Offline, deterministic (default hashed embedder):
cargo run -p memeora-bench --release -- \
    longmemeval --data data/longmemeval/longmemeval_s_cleaned.json \
    --k 10 --split held-out --out results/lme.jsonl

cargo run -p memeora-bench --release -- \
    locomo --data data/locomo10.json --k 10 --out results/locomo.jsonl

# Quick smoke run:
cargo run -p memeora-bench -- longmemeval --data ... --limit 20

# Opt-in real embeddings (compiles ONNX, downloads BGE-small weights once):
cargo run -p memeora-bench --release --features real-embeddings -- \
    longmemeval --data ... --real-embeddings
```

Flags: `--data <path.json>` (required), `--k` metric cutoff (default 10),
`--limit N` cap on evaluated questions, `--out <path.jsonl>` per-question rows,
`--split dev|held-out|all` (default `all`), `--real-embeddings`.

## Harness design

- **Isolation:** each LongMemEval question gets a *fresh* in-memory store
  ingesting only its own haystack (one memory per session, turns joined,
  memory id = session id). LoCoMo builds one store per conversation, shared by
  that conversation's QA items — recall is read-only, so sharing is safe.
- **Gold mapping:** LongMemEval gold = `answer_session_ids` directly. LoCoMo
  QA `evidence` ids are dialog turns (`"D1:3"` = session 1, turn 3) and are
  mapped to their containing session (`session_1`), so both benchmarks score at
  session granularity.
- **Skipped questions:** abstention (LongMemEval `*_abs`) and adversarial
  (LoCoMo category 5) questions carry no evidence ids; recall is undefined
  there, so they are skipped and reported as a count.

## Dev / held-out split

`--split` partitions questions deterministically: each question id is hashed
with 64-bit FNV-1a seeded with **42**, and the **50** lowest-hashed ids form
`dev`; the rest are `held-out`. The partition depends only on question ids —
not on file order, RNG state, machine, or run — so it never drifts.

**Protocol:** tune engine parameters against `--split dev`; report results from
`--split held-out`. Never tune on held-out numbers.

## Output

Stdout: an aggregate table (overall + per `question_type` / LoCoMo
`category-N`). With `--out`, one JSON object per question:

```json
{"question_id": "...", "question_type": "...", "gold": ["..."],
 "retrieved": [{"id": "...", "score": 0.031, "hit": true}, ...],
 "recall_any": 1.0, "recall_all": 1.0, "ndcg": 0.63}
```

`cargo test -p memeora-bench` needs no datasets: dataset-dependent runs are
CLI-only; tests cover the metric math, the split partition, the embedder, the
loaders (inline fixtures), and an end-to-end pass over the real engine.
