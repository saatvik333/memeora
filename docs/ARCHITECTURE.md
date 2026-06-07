# memeora — a Rust, self-hosted memory engine + universal connector for Claude Code, Codex, Antigravity & OpenCode

## Context

`supermemory` is two things: (1) a **proprietary cloud engine** (LLM fact extraction, a "facts-on-facts" knowledge graph, automatic forgetting, user profiles, hybrid RAG+memory search) and (2) **open-source thin clients** (plugins for Claude Code / OpenCode, an MCP server) that just call its **paid** API. We are building our own connector for all four tools **and** owning the engine, so there's no paid backend and no vendor lock-in.

**Decisions locked with the user:**
- Own engine, local-first, modeled on supermemory ("the best"), no paid backend.
- **No required LLM + no required API key** — works fully offline by default; hosted APIs are optional quality upgrades.
- Fast / efficient / performant + robust → **implemented in Rust as much as possible** (single static binary, no Node/Python runtime).
- **Both** a universal MCP server **and** native auto-capture hooks.

## Why Rust (and where it isn't)

The performance-critical stack has **first-class Rust libraries** (all Context7-verified, see end): MCP SDK `rmcp`, embeddings+rerank `fastembed`, SQLite `rusqlite`+`rusqlite_migration` with **statically-registered** `sqlite-vec` (C), ONNX inference `ort`. So the engine, daemon, MCP server, hook binary, and CLI are **all Rust**. Benefits: a **self-contained distributable** per platform (portable bundle first; fully-static later — ONNX Runtime makes true-static nontrivial, see Risk F), no Node/Python runtime, no native-module prebuild hell, lower latency/RAM, memory-safety.

**The only forced non-Rust piece** is OpenCode's plugin, which is an *in-process* JS/TS module (`@opencode-ai/plugin`). It becomes a ~50-line **TS shim** that forwards to the Rust daemon. Everything else stays Rust because Claude/Codex/Antigravity hooks are `type:"command"` shell hooks that can exec a Rust binary directly.

## Feasibility of the four targets (research-confirmed)

| Tool | Memory tools | Auto-capture | Config location | How memeora plugs in |
|------|-------------|--------------|-----------------|----------------------|
| **Claude Code** | MCP + plugin | Hooks: `SessionStart`, `Stop`, `PreCompact`; slash commands; Skills | `.claude-plugin/`, `.mcp.json` | command-hook → `memeora-hook` (Rust) + `rmcp` server |
| **Codex (OpenAI)** | MCP (`[mcp_servers]`) | Hooks (`hooks.json`/`[hooks]`): `SessionStart`,`Stop`,`PreCompact`; `.codex-plugin/plugin.json`. **Sets `CLAUDE_PLUGIN_ROOT`** | `~/.codex/config.toml`, `hooks.json` | reuses the **same** `memeora-hook` binary |
| **Antigravity** | MCP + plugin | Has hooks + plugin bundles, but **its own schema** (current docs indicate `PreToolUse`/`PostToolUse`/`PreInvocation`/`PostInvocation`/`Stop`, camelCase + `injectSteps` — **NOT** the Gemini-CLI event names; **verify against live docs**) | `~/.gemini/antigravity-cli/…`, `…/plugins/<name>/` | `memeora-hook --host antigravity` (own parser/renderer) |
| **OpenCode** | MCP + in-process tool | TS plugin (**current** API: `session.*`, `tool.execute.*`, `message.updated`) | `~/.config/opencode/opencode.jsonc` | thin **TS shim** → daemon (only non-Rust adapter) |

**Key insight (corrected):** all four expose a `type:"command"` hook model + MCP, so **one Rust `memeora-hook` binary serves all of them — but via per-host parsers/renderers (`--host claude|codex|antigravity`), not a single event-map.** Verified overlap: Claude & Codex both have `SessionStart` with `hookSpecificOutput.additionalContext` and a turn-end `Stop`, so context-injection + capture reuse cleanly there; **Antigravity uses a different event schema** (own adapter logic). So: **one Rust engine/daemon → one `rmcp` MCP server + one multi-host `memeora-hook` binary → thin per-tool packaging.** Build real stdin-payload E2E fixtures per host before shipping.

---

## The engine (`crates/core`) — local, free, Rust, modeled on supermemory

### 1. Embeddings — local, quantized, no API key
- **Default: `fastembed-rs`** with a small contextual quantized ONNX model (`bge-small-en-v1.5`/`gte-small`, 384d). CPU, few-ms, **no GPU/API/LLM**, real attention for nuanced recall. Context7-verified model list incl. quantized variants + reranking + sparse/ColBERT. (Alternative: **`gte-rs`** — embeddings+rerank from the same `orp`/`ort` family as `gline-rs`, which guarantees a single shared `ort` version; pick one to minimize the `ort`-rc pin surface.)
- **Speed/offline option:** Model2Vec static embeddings (~100–500× faster, uncontextualized) for bulk indexing.
- **Max-quality option:** hosted free tier (Gemini `text-embedding-004`, Voyage, Jina) via a trait impl.
- **Matryoshka:** truncate dims (e.g. 256) for smaller/faster index. **`EmbeddingProvider` trait**; content-hash **embedding cache**; **vectors namespaced by (provider, model, dims)** so switching triggers a scoped re-embed, never silent corruption.

### 2. Storage + retrieval — `rusqlite` + `sqlite-vec` + FTS5 + RRF (benchmarked fastest at our scale)
- SQLite via **`rusqlite`** (bundled). **Register `sqlite-vec` *statically*** via the crate's `sqlite3_auto_extension`/init entrypoint (NOT runtime `load_extension`, which needs a dynamic `.so/.dylib/.dll` and breaks the single-distributable goal; reserve `load_extension` for `doctor`/dev only). Schema migrations via **`rusqlite_migration`** (`user_version`). `vec0` supports `float[N]`/`int8`/`bit[N]`, **KNN + metadata filters in one WHERE**, and `vec_quantize_binary()` — with limits to design around: **max 16 metadata columns** and restricted operators; vec0 is **brute-force, pre-v1**.
- **Hybrid search:** dense (sqlite-vec) + lexical **BM25 (FTS5)** fused with **Reciprocal Rank Fusion (RRF)** → optional **cross-encoder rerank** (`fastembed-rs` `TextRerank`). Modes `hybrid` (default)/`memories`/`documents`. Benchmarks: SQLite+FTS5+sqlite-vec is fastest (~0.1–1ms) and ≥ others once reranked; 100% recall (brute force) well within personal-store size.
- **`VectorStore` trait** → SQLite is the **default for personal-memory scale** (not framed as "fastest benchmarked" — the cited bench is small-N); a switch to **LanceDB** (Rust-native) for large/codebase-scale is **benchmark-driven**, and on *filtered* workloads LanceDB's own docs suggest `IVF_PQ`/`IVF_RQ` over `IVF_HNSW_SQ`.
- DB at `~/.memeora/memory.db`. Tables: `memories`(content, embedding, fts, type, container_tag, is_latest, strength, created/last_accessed, expires_at, metadata), `relationships`(from_id,to_id,kind∈`updates|extends|derives`), `documents`, `profiles`.

### 3. Extraction — tiered, **no LLM required** (Queued→Extract→Chunk→Embed→Index→Done)
- **Tier 0 — heuristic (default, instant, pure Rust):** segment turns/sentences; signal-filter candidate facts (first/second-person, preference verbs, decision/architecture keywords, explicit `remember`/`save this`); classify type (preference/fact/episode).
- **Tier 1 — local ONNX models, *non-generative*, no LLM (recommended quality):** **`gline-rs`** (v1.0.1, production) does **NER *and* zero-shot relation extraction** in one crate → entities + structured facts; **`gliclass-rs`** for zero-shot memory-type classification (fact/preference/episode); **NLI cross-encoder** via `ort` (entailment/neutral/**contradiction**) for robust `updates` detection ("moved to SF" supersedes "lives in NYC"). All local, no API. (Same-author `orp`/`ort` stack → shared runtime.)
- **Tier 2 — optional LLM extractor** (`derives` + nuanced facts), behind the same `Extractor` trait, off by default.
- **Graph edges:** `updates` (NLI contradiction + high similarity → flip `is_latest`, **never hard-delete**), `extends` (moderate similarity, same entity), `derives` (Tier 2). **Dedup/reinforce:** near-duplicate strengthens an existing memory. **Forgetting:** recency-weighted decay on episodes, `expires_at` from temporal phrases, contradiction resolution via `updates`.

### 4. Profiles & scoping
- **Profiles** = `static` (stable facts/preferences) + `dynamic` (recent episodes), maintained **incrementally** (cache per tag, invalidate on write) → ~50ms reads.
- **Three-tier scoping** (matches supermemory; `sha256(x)[..16]`): `memeora_user_{sha256(git email)}` (cross-project) · `memeora_project_{sha256(gitRoot|cwd)}` (private) · `repo_{sanitize(repoName)}` (**team-shareable**, name-based). Overridable via project config.

---

## Performance & robustness architecture

- **Long-lived Rust daemon + thin clients (essential).** supermemory cold-starts a 200KB JS bundle every turn — tolerable only because it just makes a network call. Our engine loads embedding/NER models, so `memeora serve` (tokio) holds models + DB **once**; the `rmcp` MCP server, the `memeora-hook` binary, the CLI, and the OpenCode shim are all thin clients over **cross-platform IPC** (`interprocess`: Unix domain socket / Windows named pipe). 
- **DB concurrency pattern (required — `rusqlite` is sync; do NOT block tokio):** a **dedicated writer-actor thread** owns the single write connection (WAL mode); writes are queued to it; heavy embed/index runs on a bounded `spawn_blocking` pool; read-only pooled connections only if needed. The **daemon is the sole DB writer.**
- **Daemon lifecycle:** clients (esp. stdio-MCP for Antigravity) **auto-spawn the daemon if absent**, guarded by a **lockfile + PID check + version handshake**; stale-socket cleanup; crash recovery; explicit **one-global-daemon vs per-project** policy.
- **Async ingestion queue:** `memory.add` enqueues and returns immediately; a background task does segment→extract→embed→index. Agent never blocks on writes.
- **Incremental capture:** per-session `lastCapturedUuid` tracker (`~/.memeora/trackers/{id}`) → only new turns parsed/stored; optional "signal mode" (only turns near keyword triggers).
- Content-hash embedding cache; batched embeds for bulk `index`.

---

## Repo layout (Cargo workspace)

```
memeora/
├── crates/
│   ├── core/    engine: VectorStore + EmbeddingProvider + Extractor traits, graph, search(RRF+rerank), profiles, forgetting
│   ├── daemon/  tokio process holding models+DB; IPC server (interprocess); sole DB writer; async queue
│   ├── mcp/     rmcp server (stdio + streamable HTTP) → daemon client; tools below
│   ├── hook/    `memeora-hook` binary for Claude/Codex/Antigravity command-hooks → daemon client
│   └── cli/     `memeora` (clap): install/serve/doctor/index → daemon client
├── adapters/
│   ├── claude-code/  .claude-plugin/{plugin.json,hooks.json,commands/*.md,skills/*/SKILL.md,.mcp.json}
│   ├── codex/        .codex-plugin/plugin.json + hooks/hooks.json + config.toml snippet
│   ├── antigravity/  plugins/memeora/{plugin.json,hooks.json,mcp_config.json,skills/}
│   └── opencode/      thin TS plugin (@opencode-ai/plugin) → daemon   ← only non-Rust
├── dashboard/   Vite web app (graph UI); built assets embedded into the daemon via rust-embed
├── models/      bundled/first-run ONNX (embedder, NER, NLI, reranker)
└── .github/workflows/  CI (fmt/clippy/test/audit) + release (dist)
```

Verified crates: `rmcp = "1"` (1.7.0; `#[tool]`/`#[tool_router]`, stdio + `StreamableHttpService`), `fastembed` (MVP embedder/reranker), `rusqlite`(+bundled sqlite, **static** `sqlite-vec`)+`rusqlite_migration`, `tokio`, `clap`, `interprocess` (local socket / Windows named pipe), `serde`. **Deferred behind feature** (ort-alignment): `gline-rs`/`gliclass-rs` (or all-`orp` family `gte-rs`+`gline-rs`+`gliclass-rs` as the alternative ML stack). TS shim only: `@opencode-ai/plugin`.

### Universal MCP tools (identical in all four)
| Tool | Purpose |
|------|---------|
| `memory` (add/forget) | Save/forget; agent calls when something's worth remembering |
| `recall` (search) | Hybrid search → memories + profile summary |
| `context` | Inject full profile (static + dynamic) at session start |
| `list` | List by scope (`user`/`project`) |

### Adapters (auto-capture). One multi-host `memeora-hook` binary with **per-host parser/renderer** (`--host`), mapping each tool's events to capture / inject / compaction:
Claude `Stop` / `SessionStart` / `PreCompact` · Codex `Stop` / `SessionStart` / `PreCompact` · Antigravity (own schema — verify live) `PostInvocation|Stop` / `PreToolUse|injectSteps` / its compaction event · OpenCode (TS shim, **current** API) `session.idle`+`message.updated` / `session.created` / `session.compacting` (+ custom `tool()`).
- **Context injection format:** `User Profile (Persistent)` (static) + `Recent Context` (dynamic) + `Relevant Memories` with **relative time** ("3hrs ago") + **similarity %**, wrapped with a short intro + "use naturally, don't force it" disclaimer.
- **Keyword nudge over silent auto-save:** inject a synthetic message telling the agent to call `memory` with judgment (scope/type).
- **Privacy:** strip `<private>…</private>` before storing; refuse fully-private content. **No auth subsystem** (local) — supermemory needs OAuth; we don't.

---

## Adapter patterns adopted from supermemory's source

(From reading `claude-supermemory` + `opencode-supermemory`.)
- **Incremental transcript capture** via `lastCapturedUuid`; turn grouping; "signal mode".
- **Transcript→memory formatting:** strip `thinking`/`system-reminder` tags, truncate tool results (~500c) and inputs (~100c), emit compact `<|start|>role<|message|>…<|end|>` turns. The **adapter** produces this blob; the **daemon's local extractor** turns it into memories — the exact seam where our no-LLM engine replaces their server-side LLM `entityContext`.
- **One engine, many frontends** (vs supermemory's separate per-tool codebases + separate hosted MCP): consistent behavior + cross-tool shared memory by construction.

---

---

## Dashboard — local graph UI (basic, function-first), like supermemory

**Served by the daemon, zero extra processes.** The daemon already holds the DB; add an **`axum`** (tokio-native, Context7-verified) HTTP server exposing a JSON API + **SSE** stream, with the built web assets **embedded into the binary via `rust-embed`**. `memeora dashboard` opens `localhost`. No separate deploy, still one binary, works offline.

- **Frontend stack (chosen): Svelte 5 (runes) + Vite + TypeScript + Tailwind**, static SPA build (no SSR needed — served locally), embedded via `rust-embed`. Smaller bundle + no virtual-DOM overhead than React.
- **Styling: deliberately minimal — no design system.** Native HTML controls + a small amount of plain CSS (or a thin Tailwind layer). `bits-ui` only where a primitive genuinely needs accessibility (dialog, popover) — otherwise plain elements. Function over form; the **graph is the centerpiece**, everything else is barebones. `lucide-svelte` icons optional. No bespoke tokens, no elaborate theming, no motion work.
- **Graph rendering:** **Sigma.js + graphology** (WebGL, thousands of nodes; `graphology-layout-forceatlas2`), used **directly** in a Svelte component (`onMount`/action — Sigma is framework-agnostic, so no React wrapper needed). Nodes = memories (color by type: fact / preference / episode; size by strength·recency; dimmed when `is_latest=false`); edges = `updates`/`extends`/`derives`. Sigma (WebGL) is chosen over SVG diagramming (Svelte/React Flow) because the memory graph grows past what SVG handles.
- **Core views:**
  - **Graph canvas** — pan/zoom/cluster; click a node → inspector panel (content, type, metadata, relative time, similarity, relationships, version history via `updates`).
  - **Search** — hybrid search box highlights/filters matching nodes live.
  - **Spaces switcher** — `user` / `project` / `repo` container tags.
  - **Profile** — static + dynamic facts.
  - **Timeline / decay** — recency + forgetting visualization.
  - **CRUD** — forget / pin / edit, with "never hard-delete" honored (soft-archive).
  - **Live mode** — SSE pushes new memories as the daemon extracts them → watch memory form in real time (supermemory-like delight).
- **Look:** plain and basic — system font, default-ish controls, monochrome, no chartjunk. Readable and functional, not designed. (Polish can come later if ever needed.)
- **Auth/exposure:** binds to `127.0.0.1` only, no network exposure (it's a local tool); optional token if bound elsewhere.

## CI/CD & release

- **CI (GitHub Actions, on PR/push):** `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test` (or `nextest`), **`cargo-deny`/`cargo-audit`** (advisories **+ license checks** — also satisfies the "verify model/dep license redistribution" risk), MSRV check, `Swatinem/rust-cache`. Dashboard job: `pnpm typecheck`/`build`. TS shim job: typecheck + build.
- **Release (tag-triggered): `dist` (formerly cargo-dist)** — Context7-verified; plans the release, **cross-compiles the static binary** for the matrix (Linux x86_64+aarch64 *musl* static, macOS x86_64+arm64, Windows x86_64), and auto-generates **installers** (`curl|sh`, PowerShell, Homebrew, npm wrapper) + GitHub Release with checksums. Optional `release-plz` for workspace versioning + changelog + crates.io publish.
- **Models:** publish ONNX models as release assets with **SHA-256 checksums**; binary either bundles them (`include_bytes!`/a `models` crate) or fetches on first run and verifies the checksum, with an offline-bundle fallback.
- **Adapter publishing:** OpenCode TS shim → npm; Claude Code plugin → marketplace; the dashboard ships inside the binary (no separate hosting). The build pipeline embeds the dashboard assets into the daemon before `dist` packages the binary.
- **Reproducibility:** pinned `rust-toolchain.toml`, `Cargo.lock` committed, deterministic builds where feasible.

---

---

## Plugin ecosystem & extensibility (community plugins for any harness)

memeora must be **easy for the open-source community to extend to new harnesses** (Cursor, Windsurf, Zed, Aider, Cline, Jules, …) **without forking the core**. The design makes the engine a stable backend with public, versioned contracts:

- **Two public, semver'd contracts:** (1) the **MCP server** (`rmcp`) — *any* MCP-capable harness gets memory with **zero new code**, just a config entry (this is the universal baseline); (2) the **daemon IPC protocol** (JSON-RPC over the local socket) with a small verb set — `capture(turns, scope)`, `inject(scope)→context`, `search`, `add`, `profile`, `list`, `forget` — plus a **capability/version handshake**. Core logic lives behind these; adapters never touch engine internals.
- **Data-driven host descriptors (no Rust needed for most harnesses):** the multi-host `memeora-hook` binary reads a **host descriptor** (TOML/JSON: event-name map + stdin field mapping + stdout renderer). Adding a command-hook harness = contributing a descriptor file, not code. In-process-only harnesses (OpenCode-style) need a thin language-native shim that calls the IPC.
- **Client SDKs** so plugin authors work in their language: a Rust crate `memeora-client` and a TS/npm `@memeora/client`, both wrapping the IPC protocol with typed helpers.
- **Contributor tooling:** `memeora adapter new <harness>` scaffold; a **conformance test kit** (replay real stdin fixtures → assert correct IPC calls) so third-party adapters self-verify; "add your harness in N steps" docs; adapters may live in-repo (`adapters/`) or in separate community repos against the stable contract.
- **memeora ships *as* a plugin** to plugin-capable harnesses (Claude marketplace, Antigravity plugin store, OpenCode registry, npm).
- **Stability policy:** versioned protocol + capability negotiation + documented backward-compat guarantees — the foundation an ecosystem can rely on.

## Scalability (designed in via traits, not bolted on)

- **Storage scale:** `VectorStore` trait → SQLite/sqlite-vec (brute-force, personal scale) **swaps to LanceDB ANN** for large/codebase scale, benchmark-driven; **int8/binary quantization** to shrink memory; **container-tag partitioning**; pruning/decay/archival to bound the active working set.
- **Remote/team/multi-tenant (future-proofing):** keep storage behind a `VectorStore` + **sync/backend trait** so the *same* engine, extraction, and adapters run **local-only OR against a shared server** (hosted memeora, libsql/Turso, or Postgres+pgvector) for team-wide `repo_<name>` memory — added later **without touching adapters or extraction**.
- **Concurrency:** writer-actor thread + WAL + read pool + async ingestion queue; one daemon serves **many projects/editors** concurrently over IPC.
- **Multi-project:** one global daemon keyed by container tags (or per-project daemons) — explicit, documented policy.
- **Throughput:** batched embeds, content-hash embedding cache, incremental (new-turns-only) capture, lazy per-capability model loading; optional GPU/NPU via `ort` execution providers; tiered model sizes (static → small ONNX → hosted API) to match the host's resource budget.
- **Principle:** every heavy/variable concern is a **trait** (`EmbeddingProvider`, `Extractor`, `VectorStore`, sync backend) → scale up or swap implementations without rewrites or adapter churn.

---

## Build order

1. **`core` storage** — rusqlite + static sqlite-vec + FTS5, `rusqlite_migration` schema, `VectorStore` trait (SQLite impl), container-tag hashing.
2. **`EmbeddingProvider`** — fastembed-rs default; static + hosted impls; embedding cache.
3. **Retrieval** — dense + BM25 + **RRF** + optional rerank; profiles (incremental); forgetting.
4. **`Extractor`** — Tier 0 heuristic (default); Tier 1 (gline-rs/ort NER + NLI) behind trait; graph edges + dedup/reinforce. Test heavily.
5. **`daemon`** (tokio + interprocess IPC, async queue) + **`mcp`** (rmcp) over it.
6. **`cli`** + **`hook`** binaries (install/serve/doctor/index; hook event handlers).
7. **Adapters** by leverage: Antigravity (plugin bundle) → Claude Code → Codex (reuse hook binary) → OpenCode (TS shim). **DONE — see `adapters/`.** `memeora-hook` now has a per-host parser/renderer (`--host claude|codex|antigravity`, events `session-start`/`pre-invocation`/`stop`/`pre-compact`); MCP tool `scope` is optional and defaults to `project_tag(cwd)`; `memeora scope [path]` resolves that tag daemon-free for the OpenCode shim. All four bundles assume the binaries are on `PATH` + the daemon is running (first-class installer is step 10).
8. **Dashboard:** `axum` JSON+SSE API in the daemon → Svelte 5 + Vite + Sigma.js graph, minimal/basic styling (native controls, plain CSS, `bits-ui` only where needed); embed assets via `rust-embed`; `memeora dashboard`. **DONE — see `crates/daemon/src/dashboard.rs` + `dashboard/`.** The daemon serves the UI + a read-mostly JSON API (`/api/scopes|graph|list|context|search|forget`) + an SSE live stream (`/api/events`) on `127.0.0.1:7878` (`MEMEORA_DASHBOARD_ADDR`, `off` to disable). Sole-writer is preserved: reads use a **second read-only SQLite connection** (WAL concurrent readers); `search` (needs the embedder) and `forget` (needs the writer) go back through the daemon's own IPC socket as a normal client; the engine broadcasts `ChangeEvent`s the SSE stream forwards. `memeora dashboard [--no-open]` opens the URL. A `build.rs` writes a placeholder `dashboard/dist/index.html` so the daemon (and CI) builds without a Node toolchain; `pnpm --dir dashboard build` produces the real embedded UI.
9. **Ecosystem:** freeze + version the **IPC protocol** (capability handshake); publish `memeora-client` (Rust) + `@memeora/client` (TS); **host-descriptor** format + `memeora adapter new` scaffold + **conformance kit**; "add your harness" docs. **DONE.** Protocol: `Response::Hello` now carries `capabilities` (serde-default, back-compat); `Client` exposes `server_version`/`capabilities`/`supports`; stability policy in `docs/PROTOCOL.md` (additive changes don't bump `PROTOCOL_VERSION`, negotiate features via capabilities). The hook is **data-driven**: `crates/hook` is now lib+bin, a `HostDescriptor` (TOML) encodes scope/transcript field paths + inject style + ack + invocation gating; the three first-party descriptors live in `adapters/_descriptors/*.toml` (embedded via `include_str!` so built-ins == shipped files), and `--descriptor <path>` loads a community host. Conformance kit (`crates/hook/tests/conformance.rs` + `fixtures/<host>/*.json`) replays real payloads → asserts scope/inject/ack. `memeora adapter new <harness>` scaffolds a descriptor + README (daemon-free). TS SDK in `sdk/ts/` (`@memeora/client`, Node `net` + the same framing). "Add your harness" guide: `docs/ADAPTERS.md`.
10. **Release:** `dist` cross-platform installers (portable+assets tier first); model assets + checksums.
11. **Optional/scale:** Tier 1 NER/relations (once `ort` aligned) → Tier 2 LLM extractor; **LanceDB** store + remote/sync backend; **benchmark harness** (LongMemEval/LoCoMo-style) vs supermemory/mem0.

**Post-review hardening (applied after a full codebase review):** `upsert` is now an
edge-preserving in-place UPDATE (delete-then-insert previously cascade-deleted a node's
graph edges via the `relationships` FK); exact re-ingest reinforces via the content id
(no destructive strength reset); FTS5 `MATCH` input is sanitized (raw user text no longer
errors the whole recall); `forget` drops the vec row so soft-forgotten memories can't
starve KNN top-k; the store persists its embedding dim and rejects a dim-mismatched reopen;
`embed_query` applies the BGE query instruction. **Daemon:** the writer-actor wraps each
request in `catch_unwind` (a panic no longer zombies the process); **embedding/extraction
run on the connection threads** (a shared `Arc` embedder/extractor via a `Preparer`), so the
single writer only does the fast DB critical section — `Ingest`/`Recall` no longer serialize
all clients behind inference; the job channel is bounded (backpressure) and connections are
capped; a startup probe refuses to start a second daemon on a live socket (cross-process
sole-writer). **Protocol:** `Client::connect` performs the version handshake and errors on
mismatch. **CI:** `deny.toml` gained `[graph] all-features` + tier-1 `targets` so the ML
stack's licenses/advisories are actually checked. (Deferred: trimming fastembed's
`image-models` to drop the NCSA exception; bounded LRU on the unused `CachingEmbedder`.)

*(CI from day one, in parallel with step 1: `fmt`/`clippy`/`test`/`cargo-deny` workflow, `rust-toolchain.toml`, committed `Cargo.lock`.)*

## Critical files (representative)
- `crates/core/src/{db/schema.rs, store/sqlite.rs, embeddings/{mod,fastembed,static,hosted}.rs, extract/{heuristic,ner,nli}.rs, search.rs, profile.rs, forget.rs, container_tag.rs}`
- `crates/daemon/src/{server.rs, ipc.rs, queue.rs, writer.rs}`, `crates/mcp/src/server.rs`, `crates/hook/src/main.rs` (+ `hosts/*.toml` descriptors), `crates/cli/src/main.rs`
- **Ecosystem:** `crates/proto` (versioned IPC types + handshake), `crates/client` (`memeora-client`), `sdk/ts` (`@memeora/client`), `adapters/_descriptors/<harness>.toml`, conformance fixtures
- `adapters/claude-code/.claude-plugin/{plugin.json,hooks.json,...}`, `adapters/{codex,antigravity}/…`, `adapters/opencode/src/index.ts`

## Verification
- **Engine unit tests:** add→recall; contradiction→old `is_latest=false`, latest returned (NLI path); episode decays; near-dup reinforces; scoping isolates user/project; **hybrid(RRF) beats vector-only & BM25-only**; reranker lifts NDCG.
- **Perf:** model load once (daemon); `recall` P95 sub-5ms on ~10k memories; `memory.add` returns before extraction completes.
- **MCP:** `memeora serve`; exercise `memory`/`recall`/`context`/`list` via MCP inspector / `codex mcp`.
- **Per-tool E2E:** Antigravity (tools in `/mcp`, hooks inject+capture), Claude Code (`SessionStart` inject, `Stop` capture, `/memeora:context`), Codex (`codex mcp` lists, `/hooks` trust, hooks fire), OpenCode (tool appears, keyword nudge, compaction preserves memory).
- **Cross-tool persistence:** save in one tool → recall in the others (same DB + tag).
- `memeora doctor` passes; **fully offline, no API key** (fastembed + Tier 0). **Self-contained distributable runs with no Node/Python installed.**
- **Ecosystem:** IPC protocol versioned + capability handshake; `memeora adapter new` scaffolds a working adapter; conformance kit validates a third-party host descriptor against real fixtures.

## Open questions & risks (audit targets)

**A. ML-in-Rust — ⛔ `ort` CONFLICT CONFIRMED (was the plan's biggest hidden bug).** `fastembed` 5.13.0 pins `ort =2.0.0-rc.11`; `gline-rs` 1.0.1 + `gliclass-rs` pin `ort =2.0.0-rc.9`. `ort`/`ort-sys` link the native ONNX runtime, and **Cargo's `links` rule forbids two versions linking the same native lib → they cannot share one workspace.** Decision (must prove `cargo build --locked` clean before committing): **MVP = `fastembed`-only** (embeddings + rerank + sparse) with **Tier-0 heuristic extraction**; **Tier-1 NER/relations (`gline-rs`/`gliclass-rs`) is deferred behind a feature flag**, unlocked only when either (a) `fastembed`'s `ort` and the fbilhaut family converge, or (b) we adopt the **all-`orp` family** instead (swap embeddings to `gte-rs` so `gte-rs`+`gline-rs`+`gliclass-rs` all sit on one `ort` rc, dropping `fastembed`). NLI cross-encoder shares whichever `ort` we standardize on. Daemon RAM with models loaded (~0.5–1.5GB) → lazy-load per capability.
**B. Engine scope — MVP DECIDED (avoid overpromising "supermemory-quality").** v1 ships: `add`, `recall`, **hybrid search**, **profiles**, **dedup/reinforce**, and **conservative `updates`** (NLI-gated, embedding pre-filter to top-k to bound O(n), high thresholds, **never hard-delete** — only flip `is_latest`/soft-archive). **Deferred:** `derives` (not credible without a generative model) and **automatic forgetting** (v1 = TTL/`expires_at` + manual archive only). This keeps quality honest while the no-LLM path proves out via the benchmark harness.
**C. Storage/runtime.** `sqlite-vec` is pre-1.0 (KNN+metadata+quantization verified, but confirm stability at scale; cited bench is small-N). Decide SQLite→LanceDB threshold. `rusqlite` is sync → run heavy embed/index on a worker task so the daemon stays responsive.
**D. Daemon lifecycle & IPC.** Cross-platform via `interprocess` (Unix socket / Windows named pipe). Define auto-spawn, stale-socket cleanup, crash recovery, one-global-daemon vs per-project, and **sole-writer** enforcement.
**E. Target-tool assumptions (post-audit status; re-verified for step 7).** *Codex:* docs **confirm** `CLAUDE_PLUGIN_ROOT`/`CLAUDE_PLUGIN_DATA`, `.codex-plugin/plugin.json`, auto-discovered `hooks/hooks.json`, `SessionStart.additionalContext`, `Stop`, `[mcp_servers]` (underscore!). **Confirmed instability:** plugin-bundled hooks are gated behind a `plugin_hooks` feature flag and OpenAI's own `plugin-creator` validator has *rejected* a manifest `hooks` field — so the Codex adapter **leads with the stable `~/.codex/{config.toml,hooks.json}` path** and ships the plugin manifest as experimental. Codex's `Stop` **requires JSON on stdout** (hook acks `{}`); new/changed hooks need `/hooks` trust. *Antigravity:* **confirmed** its own camelCase schema — events `PreToolUse/PostToolUse/PreInvocation/PostInvocation/Stop` (no `SessionStart`), context injected via `PreInvocation`→`injectSteps[].userMessage` (gate on `invocationNum==1`), scope from `workspacePaths`, transcript at `transcriptPath`, `Stop` returns `decision` (non-`"continue"`). `mcp_config.json` uses `mcpServers`; **stdio `env` passthrough reportedly broken in early 2.x** and **IDE vs CLI paths diverge** — flagged in the adapter README. *OpenCode:* plugin API is `tool()` + `experimental.session.compacting` (+ bus `event`); the shim shells the `memeora` CLI (no IPC reimplemented) and resolves scope via `memeora scope`; native MCP config is the zero-code alternative. *Claude+Codex* genuinely share `SessionStart.additionalContext`+`Stop`. Installers must **merge, never overwrite**, and per-host payload shapes still need real captured fixtures before auto-capture is trusted.
**F. Distribution (reality-checked).** "Single *fully-static* binary" is **not** trivial: ONNX Runtime via `ort` defaults to downloading/copying a dynamic lib, and true static linking needs a custom ORT build or `ORT_LIB_LOCATION` (and musl adds friction). **Release tiers:** (1) *portable + assets* (self-contained dir / installer, dynamically-linked ORT) **first**; (2) fully-static later. Models: **first-run download with SHA-256 + an offline bundle tarball** — do **not** embed large ONNX models in the executable by default. `dist` builds installers across the matrix; `cargo-deny` enforces advisory + license (incl. model redistribution) checks. Still far simpler than the Node path (no `better-sqlite3`/`onnxruntime-node` prebuilds, no JS runtime).
**G. Dashboard UI — DECIDED: Svelte 5 + Vite + Sigma.js, deliberately minimal styling (no design system).** Native controls + plain CSS; `bits-ui` only where a11y needs it. Sigma.js (WebGL) chosen over SVG diagramming because the memory graph grows large. Function-first; the graph is the centerpiece.

## Docs verified via Context7
`rmcp` `/websites/rs_rmcp_rmcp` · `fastembed-rs` `/anush008/fastembed-rs` · `gline-rs` v1.0.1 (NER+relations, crates.io/docs.rs, prod-ready) + `orp`/`ort` family (`gte-rs`, `gliclass-rs`) · `sqlite-vec` `/asg017/sqlite-vec` · `rusqlite` `/websites/rs_rusqlite_rusqlite` + `rusqlite_migration` · `axum` `/tokio-rs/axum` (dashboard server) · `dist`/cargo-dist `/axodotdev/cargo-dist` (release) · `sigma.js` `/jacomyal/sigma.js` (graph UI) · `svelte` `/sveltejs/svelte` (v5) + `bits-ui` `/huntabyte/bits-ui` (only where a11y needed; styling kept minimal) · MCP TS SDK `/modelcontextprotocol/typescript-sdk` (OpenCode-side reference) · `transformers.js` `/huggingface/transformers.js` (Node fallback) · OpenCode plugins `/websites/opencode_ai_plugins`. Reference impl worth studying: GraphRAG-with-sqlite-vec (TS) `/khaentertainment/graphrag-with-sqlite_vec-ts-vercel-ai-sdk`.
