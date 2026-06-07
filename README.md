# memeora

Self-hosted memory engine + universal connector for Claude Code, Codex, Antigravity & OpenCode.

memeora gives your AI coding tools **persistent memory** — it learns facts from your sessions,
builds a knowledge graph, and recalls the right context at the right time. It's a free,
**local-first**, open alternative to hosted memory APIs: **no required LLM, no API key, works offline.**

> **Status:** Steps 1–3 implemented — SQLite + statically-linked `sqlite-vec` KNN + FTS5
> behind the `VectorStore` trait (container-tag scoping); the `EmbeddingProvider` trait with
> a content-hash cache and a local `fastembed` backend; hybrid retrieval — dense + BM25 fused
> with **RRF**, expiry filtering, and an optional cross-encoder reranker (opt-in ONNX behind
> the `fastembed` feature); cached per-tag **profiles** (static facts/prefs + dynamic
> episodes); Tier-0 **extraction** — a model-free `Extractor` that classifies text into
> fact/preference/episode candidates; and an **ingest** write path that embeds candidates,
> reinforces near-duplicates instead of storing copies, and links related memories with
> `extends` edges in a SQLite-backed knowledge graph. The daemon + MCP server are next.
> See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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
| `crates/daemon` | tokio daemon: holds models + DB, sole writer, async queue |
| `crates/mcp` | `rmcp` MCP server (memory / recall / context / list) |
| `crates/hook` | `memeora-hook` multi-host command-hook binary |
| `crates/cli` | `memeora` CLI (install / serve / doctor / index / dashboard) |

## Build
```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
Toolchain is pinned in `rust-toolchain.toml` (Rust 1.95, edition 2024).

## Contributing
See [`CONTRIBUTING.md`](CONTRIBUTING.md). Adding support for a new harness is designed to be easy —
often just a host-descriptor file.

## License
Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).
