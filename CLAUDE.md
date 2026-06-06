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
