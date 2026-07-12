# memeora

Self-hosted memory engine + universal connector for Claude Code, Codex, Antigravity & OpenCode.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6b61963c-47f4-40b4-accd-7c4397603839" />

memeora gives your AI coding tools **persistent memory** — it learns facts from your sessions,
builds a knowledge graph, and recalls the right context at the right time. It's a free,
**local-first**, open alternative to hosted memory APIs: **no required LLM, no API key, works offline.**

> **Status:** Steps 1–10 implemented (v0.1.0), plus the **step-11 engine-evolution core** and
> the **VISION-gap engine program (Phases 1–3)** — session-aware capture, entity
> canonicalization, consolidation (`proof_count`), the Ebbinghaus/Hebbian/Cepeda forgetting
> engine, the opt-in local-LLM extractor (off by default), bi-temporal valid-time +
> version-chain supersession, 4-channel recall (dense + BM25 + graph + temporal) with
> token-budget fill, and the distinct-source evidence/freshness model. The engine, its
> surfaces, per-tool packaging, the local dashboard, the extensibility/ecosystem layer, and
> the cross-platform **release pipeline** are end-to-end:
> - **Engine (`crates/core`):** SQLite + statically-linked `sqlite-vec` KNN + FTS5 behind the
>   `VectorStore` trait (container-tag scoping, soft-forget); `EmbeddingProvider` with a
>   content-hash cache and a local `fastembed` backend; **4-channel hybrid retrieval** (dense +
>   BM25 + graph + temporal fused with **RRF**, token-budget fill, expiry filtering, optional
>   cross-encoder rerank); cached per-tag **profiles**; Tier-0 **extraction**; and an
>   **ingest** path that dedups/reinforces, links memories with `extends`/`updates` edges, and
>   tracks bi-temporal valid-time + version-chain supersession in a SQLite knowledge graph.
> - **Daemon + IPC:** versioned contract (`crates/proto`) with length-delimited framing; a
>   blocking **writer-actor** server over `interprocess`; the daemon binary loads the model + DB
>   and serves it.
> - **Surfaces:** `memeora-client` (typed Rust SDK), `memeora-mcp` (rmcp MCP server — recall/
>   remember/context/list/forget over stdio, **scope defaults to the current project**), the `memeora`
>   **CLI**, and `memeora-hook` (multi-host command-hook: session-start/`PreInvocation` injection
>   + `Stop`/`PreCompact` capture for Claude, Codex & Antigravity).
> - **Adapters ([`adapters/`](adapters/)):** ready-to-install plugin bundles for **Claude Code**
>   (plugin marketplace), **Codex** (`config.toml` + hooks), **Antigravity** (plugin bundle, own
>   camelCase schema), and **OpenCode** (thin TS shim — the only non-Rust adapter). All four
>   share one daemon/DB, so memory is cross-tool by construction.
> - **Dashboard ([`dashboard/`](dashboard/)):** a local graph UI served *by the daemon* — an
>   `axum` JSON API + SSE live stream + an embedded **Svelte 5 + Sigma.js** (WebGL) graph,
>   on `127.0.0.1` only. Sole-writer stays intact: reads use a second read-only SQLite
>   connection, while `search`/`forget` route back through the daemon's own IPC. Open it with
>   `memeora dashboard`.
> - **Ecosystem:** the IPC protocol is **versioned with a capability handshake**
>   ([`docs/PROTOCOL.md`](docs/PROTOCOL.md)); command-hook hosts are **data-driven host
>   descriptors** ([`adapters/_descriptors/`](adapters/_descriptors/)) so adding a harness is
>   a TOML file, not Rust — copy a first-party descriptor as a starting point, validate it with the
>   **conformance kit** (`crates/hook/tests/`). Client SDKs ship for **Rust** (`memeora-client`)
>   and **TypeScript** ([`@memeora/client`](sdk/ts/)). See [`docs/ADAPTERS.md`](docs/ADAPTERS.md).
> - **Release:** [`dist`](https://opensource.axo.dev/cargo-dist/) cross-compiles all four
>   binaries — `memeora`, `memeora-daemon`, `memeora-hook`, `memeora-mcp` — as **one app**
>   for Linux (x86_64 + arm64), macOS (Intel + Apple Silicon) & Windows, and publishes a
>   GitHub Release with **shell / PowerShell / Homebrew / npm** installers and per-artifact
>   **SHA-256** checksums (`.github/workflows/release.yml`, triggered by a `v*` tag). Models
>   stay out of the binary (Risk F): they download on first run or ship as an offline bundle,
>   integrity-checked via a `SHA256SUMS` manifest (`memeora models verify` / `bundle`).
>
> Next (optional/scale): Tier-1 NER, LanceDB, a benchmark harness. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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
| `crates/daemon` | blocking writer-actor daemon: holds models + DB, sole writer; embeds off the writer thread; serves the dashboard (axum + SSE) |
| `crates/mcp` | `rmcp` MCP server (recall / remember / context / list / forget) |
| `crates/hook` | `memeora-hook` descriptor-driven command-hook library (its binary ships from `crates/cli`) |
| `crates/cli` | `memeora` CLI (doctor / add / ingest / recall / context / list / forget / scope / dashboard / models) — also the package that ships all four binaries |
| `crates/bench` | `memeora-bench` — offline LongMemEval/LoCoMo retrieval harness (recall@k / NDCG@10, seed-42 dev split) |
| `dashboard/` | Svelte 5 + Vite + Sigma.js graph UI, embedded into the daemon via `rust-embed` |
| `sdk/ts/` | `@memeora/client` — TypeScript client over the IPC protocol |
| `adapters/_descriptors/` | data-driven host descriptors (claude / codex / antigravity) |

## Install

### Guided setup (recommended)

One interactive command installs the binaries, handles the embedding model, wires
memeora into your coding tools (Claude Code / Codex / Antigravity / OpenCode), and
optionally starts the daemon — you choose each step. It reads prompts from your
terminal even under `curl | sh`, never uses `sudo`, backs up any file it edits, and
is safe to re-run:

```sh
curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/saatvik333/memeora/main/scripts/install.sh | sh
```

For CI / non-interactive use, pass flags or `MEMEORA_*` env through the pipe, e.g.
`… | sh -s -- --yes --adapters=claude,codex --offline --no-daemon`. Run with
`--help` (or read [`scripts/install.sh`](scripts/install.sh)) for every option.

### Just the binaries

Released builds ship every binary (`memeora`, `memeora-daemon`, `memeora-hook`,
`memeora-mcp`) in a single installer. Pick one:

```sh
# Linux / macOS — shell installer (curl | sh)
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/saatvik333/memeora/releases/latest/download/memeora-installer.sh | sh

# macOS / Linux — Homebrew
brew install saatvik333/tap/memeora

# any platform — published npm wrapper (use bun; npm/pnpm also work)
bun add -g @memeora/memeora
```
```powershell
# Windows — PowerShell installer
powershell -ExecutionPolicy ByPass -c "irm https://github.com/saatvik333/memeora/releases/latest/download/memeora-installer.ps1 | iex"
```

Or grab a prebuilt archive (with its `.sha256`) from the [releases page](https://github.com/saatvik333/memeora/releases).
On first run the daemon needs the embedding model. It is **offline by default and will
not download silently**: set `MEMEORA_ALLOW_MODEL_DOWNLOAD=1` to fetch it once
(~130 MB from HuggingFace) into `~/.memeora/models`, or — for an **air-gapped** install —
drop the model files there (or point `MEMEORA_MODELS_DIR` at a bundle) and verify
integrity with `memeora models verify`.

## Build (from source)
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
often just a host-descriptor file: see [`docs/ADAPTERS.md`](docs/ADAPTERS.md). Start by copying a
first-party descriptor from [`adapters/_descriptors/`](adapters/_descriptors/).

## License
Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).
