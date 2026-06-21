# memeora — agent guide

Self-hosted, **Rust** memory engine + universal connector giving persistent memory to
Claude Code, Codex, Antigravity & OpenCode. Local-first, no required LLM, no API key.

## Conventions
- **Commits:** Conventional Commits, subject < 200 chars (e.g. `feat(core): add VectorStore trait`).
- **Think before implementing**; divide work hierarchically (tasks → subtasks).
- **Keep docs updated** with code: `docs/ARCHITECTURE.md` is the source of truth (plan).
- Format on save (PostToolUse `rustfmt` hook); `cargo clippy -D warnings` must pass.

## Layout (Cargo workspace)
- `crates/core` — engine: `EmbeddingProvider` / `Extractor` / `VectorStore` traits, graph, search, profiles.
- `crates/proto` — versioned IPC contract (public, semver'd).
- `crates/client` — Rust client SDK over the IPC.
- `crates/daemon` — tokio process; holds models + DB; **sole DB writer**; async queue.
- `crates/mcp` — `rmcp` MCP server (memory/recall/context/list).
- `crates/hook` — `memeora-hook`: multi-host command-hook binary (`--host`).
- `crates/cli` — `memeora` CLI (install/serve/doctor/index/dashboard).
- `adapters/` — per-tool packaging; `dashboard/` — Svelte+Sigma.js UI (minimal styling).

## Key decisions (see docs/ARCHITECTURE.md for full rationale)
- **Rust-first**; only OpenCode's in-process plugin is a thin TS shim.
- **MVP = `fastembed` embeddings only + Tier-0 heuristic extraction.** `gline-rs`/`gliclass-rs`
  NER are **deferred behind a feature** due to an `ort` version conflict (fastembed's `ort` rc —
  currently rc.12 — vs gline-rs's rc.9; `links` rule forbids two versions). `fastembed` is an
  **opt-in feature of `memeora-core`** (off by default so core tests stay offline; binaries enable it).
- Storage: `rusqlite` + **statically-registered** `sqlite-vec` + FTS5 + RRF hybrid search.
- Daemon is the sole DB writer (writer-actor thread + WAL; never block tokio with sync SQLite).
- MVP engine: add/recall/hybrid/profiles/dedup/conservative-updates. Defer `derives` + auto-forgetting.
- Toolchain pinned in `rust-toolchain.toml` (1.95, edition 2024). License: MIT OR Apache-2.0.

## Commands
- Build/check: `cargo build --workspace` · `cargo check --workspace`
- Lint/format: `cargo clippy --workspace --all-targets -- -D warnings` · `cargo fmt --all`
- Test: `cargo test --workspace`
- Deps audit: `cargo deny check` (config in `deny.toml`)
- **Full local gate (mirrors CI):** `scripts/check.sh` — run before committing/pushing.
  `scripts/check.sh --fast` skips the `--all-features`/deny passes for a quick loop.

## One-time setup
- `git config core.hooksPath .githooks` — enables the pre-commit hook (runs `scripts/check.sh`).
- `cargo install cargo-deny --locked` — so the gate runs license/advisory checks locally.

## Workflow (per build-order step) — use `/ship` to run it
implement → `scripts/check.sh` (fix all failures) → self-review diff → update
`README.md`/`docs/ARCHITECTURE.md` → update the `project-memeora` native memory →
conventional commit. Don't start the next step unless asked.

## Gotchas (learned the hard way — don't repeat)
- **CI runs everything with `--all-features`** (clippy + test) plus `cargo-deny`. A plain
  `cargo test`/`cargo check` does NOT compile the `fastembed`/ONNX stack — only the
  `--all-features` pass does. Always run `scripts/check.sh` before pushing.
- **cargo-deny is a real gate, not advisory.** Adding a native/ML dep drags in transitive
  licenses the allowlist may lack (so far: `NCSA` via libfuzzer-sys→rav1e→image,
  `CDLA-Permissive-2.0` via webpki-roots→hf-hub). Add them to `deny.toml` with a comment.
  Transitive *unmaintained* advisories (e.g. `paste`) are tolerated via `unmaintained = "workspace"`.
- **`cargo tree -i <crate>` filters by host platform/features** — it can hide deps that
  cargo-deny (all targets) still sees. Trust `cargo deny check`, not just `cargo tree`.
- **Library/CLI APIs: verify via Context7 first** (fastembed, cargo-deny, …) — don't guess.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
