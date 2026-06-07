# memeora

Self-hosted memory engine + universal connector for Claude Code, Codex, Antigravity & OpenCode.

memeora gives your AI coding tools **persistent memory** — it learns facts from your sessions,
builds a knowledge graph, and recalls the right context at the right time. It's a free,
**local-first**, open alternative to hosted memory APIs: **no required LLM, no API key, works offline.**

> **Status:** Steps 1–7 implemented — the engine, its surfaces, and per-tool
> packaging are end-to-end:
> - **Engine (`crates/core`):** SQLite + statically-linked `sqlite-vec` KNN + FTS5 behind the
>   `VectorStore` trait (container-tag scoping, soft-forget); `EmbeddingProvider` with a
>   content-hash cache and a local `fastembed` backend; hybrid retrieval (dense + BM25 fused
>   with **RRF**, expiry filtering, optional cross-encoder rerank); cached per-tag **profiles**;
>   Tier-0 **extraction**; and an **ingest** path that dedups/reinforces and links memories with
>   `extends` edges in a SQLite knowledge graph.
> - **Daemon + IPC:** versioned contract (`crates/proto`) with length-delimited framing; a
>   blocking **writer-actor** server over `interprocess`; the daemon binary loads the model + DB
>   and serves it.
> - **Surfaces:** `memeora-client` (typed Rust SDK), `memeora-mcp` (rmcp MCP server — recall/
>   remember/context/list over stdio, **scope defaults to the current project**), the `memeora`
>   **CLI**, and `memeora-hook` (multi-host command-hook: session-start/`PreInvocation` injection
>   + `Stop`/`PreCompact` capture for Claude, Codex & Antigravity).
> - **Adapters ([`adapters/`](adapters/)):** ready-to-install plugin bundles for **Claude Code**
>   (plugin marketplace), **Codex** (`config.toml` + hooks), **Antigravity** (plugin bundle, own
>   camelCase schema), and **OpenCode** (thin TS shim — the only non-Rust adapter). All four
>   share one daemon/DB, so memory is cross-tool by construction.
>
> Next: dashboard, ecosystem, release. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Highlights
- **Rust** engine + daemon + MCP server + hook binary + CLI → one self-contained distributable.
- **Universal:** one MCP server + one multi-host hook binary serve all four tools; any
  MCP-capable harness works with zero extra code.
- **Local & private:** `fastembed` embeddings + `sqlite-vec` hybrid search (dense + BM25 + RRF),
  heuristic fact extraction — no cloud, no keys.
- **Extensible:** versioned IPC + data-driven host descriptors so the community can add new
  harnesses without forking.

## Workspace
| Crate | Role |
|-------|------|
| `crates/core` | engine: storage, embeddings, extraction, graph, hybrid search, profiles |
| `crates/proto` | versioned IPC contract (public) |
| `crates/client` | Rust client SDK |
| `crates/daemon` | blocking writer-actor daemon: holds models + DB, sole writer; embeds off the writer thread |
| `crates/mcp` | `rmcp` MCP server (recall / remember / context / list) |
| `crates/hook` | `memeora-hook` multi-host command-hook binary |
| `crates/cli` | `memeora` CLI (doctor / add / ingest / recall / context / list / forget / scope) |

## Build
```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
Toolchain is pinned in `rust-toolchain.toml` (Rust 1.95, edition 2024).

> **Note:** `cargo test --workspace` compiles the `fastembed`/ONNX stack (feature
> unification pulls it in via the daemon), so the first run downloads/builds it. For
> the fast, fully-offline core loop use `cargo test -p memeora-core`. The full gate
> (`scripts/check.sh`) mirrors CI and runs everything `--all-features`.

## Contributing
See [`CONTRIBUTING.md`](CONTRIBUTING.md). Adding support for a new harness is designed to be easy —
often just a host-descriptor file.

## License
Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).
