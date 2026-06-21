# memeora — the unified memory brain (vision)

> Distilled from a source audit of the strongest memory systems in the open:
> **supermemory**, **Hindsight** (Vectorize), **MemPalace**, and **Understand-Anything**.
> This doc is the north star; `docs/ARCHITECTURE.md` remains the technical plan.

## The shift

memeora began as *"a universal connector giving persistent memory to Claude Code, Codex,
Antigravity & OpenCode."* That framing is too small. The four coding agents are the **first
surfaces, not the boundary.**

memeora is **a local memory brain that learns, adapts, heals, and evolves — that any tool can
plug into.** The engine is the product; every agent, framework, voice app, browser, and
automation runner is just a thin surface over the same brain.

## Design invariants (non-negotiable — inherited, reaffirmed)

These filter everything we borrow. An idea that violates one is declined or demoted to an
opt-in tier.

- **Rust, single static binary, sole-writer daemon.** No Node/Python runtime in the core.
- **Local-first. No *required* LLM, no *required* API key. Fully offline by default.**
- **Every quality tier above the heuristic floor is the user's explicit choice.** Detection
  ≠ activation. A localhost model is part of the user's machine; an external API is a
  deliberate opt-in, never a silent fallback.
- **Never hard-delete. Verbatim content preserved.** History is traversable; salience decays,
  data does not vanish.
- **Privacy by architecture.** The brain physically cannot leak what it never sends.

## Provenance & honesty

- **Hindsight** and **MemPalace** are open — their *algorithms* are portable, and the
  constants below are read from their source.
- **supermemory**'s retrieval engine is **closed** (the OSS repo is clients + SDKs over a
  hosted API). Its contribution is the **memory-record contract** and the **surface-reach map**,
  not algorithms.
- **Understand-Anything** is a *code-comprehension* tool, not a memory engine; its one
  transferable gift is a graph self-repair pattern.
- We take the best of each, **through the invariants above** — and we say plainly what we
  decline and why (table at the end).

---

## The four pillars (each a concrete mechanism, not a slogan)

### 🧠 Learns — turns raw turns into durable, consolidated understanding
- **Observation network** *(Hindsight)* — a consolidated-belief layer over raw memories.
  The bookkeeping is **heuristic and LLM-free**: `proof_count = |distinct source ids|`,
  evidence carries exact source quotes + timestamps, merge is **set-union over source ids**
  (lineage only ever grows), and updates **never overwrite** (history preserved). The one
  LLM-needing step — phrasing the merged belief — is the **opt-in local-LLM tier**; with no
  model it degrades to embedding-kNN clustering keyed by canonical entity.
- **Entity-first canonicalization** *(MemPalace `entity_registry`)* — unify mentions, aliases,
  and spellings into one canonical entity (alias map + context disambiguation for names that
  are also common words). Pure heuristic; the only network call (Wikipedia) is off by default.

### 🔧 Adapts — meets the user's hardware, privacy posture, and tools where they are
- **The user-chosen extractor ladder** (the heart of "adapts"):
  - **Tier 0 — heuristic** (default, offline, zero-dep, instant). The floor that keeps
    "no required LLM" literally true.
  - **Tier 1 — local embeddings/rerank** (`fastembed`, in-process ONNX).
  - **Tier 2 — local LLM** (Ollama / LM Studio / llama.cpp / vLLM on `localhost`, **opt-in**).
    Spoken to over **OpenAI-compatible HTTP** — zero native deps, and it **sidesteps the
    `ort`/`gline` version conflict** that has blocked Tier-1 NER for the whole project.
  - **Tier 3 — external BYOK** (Anthropic/OpenAI/Gemini). Deliberate opt-in, never silent.
  - Policy in one boolean: `endpoint_is_local` → allowed under local-first; else → consented.
- **Multi-format ingestion** *(MemPalace `normalize`)* — beyond agent hooks: Claude Code /
  Codex / Gemini CLI transcripts **and** ChatGPT, Claude.ai, and Slack exports → canonical
  `(role, text)`. Heuristic, no network.
- **Intent-specific context** *(Understand-Anything builders)* — tailor the injected slice by
  intent (session-start vs recall vs explain vs diff), not one fixed profile dump.

### 🩹 Heals — keeps its own graph clean, especially under LLM input
- **Graph self-repair** *(Understand-Anything `schema.ts`)* — the safety layer that makes the
  opt-in local-LLM tier trustworthy. Every emitted node/edge runs
  `lowercase → alias-map → default → coerce → clamp → drop-invalid-node → **drop-dangling-edge**
  → fatal-only-if-zero-nodes`, and **every mutation is logged** as a `GraphIssue`
  (`auto-corrected | dropped | fatal`) audit trail. Invariant: *validate nodes first, then drop
  any edge whose endpoints aren't in the surviving node set* — applied at write and again after
  any merge.
- **Dedup / repair / contradiction** *(MemPalace + Hindsight + memeora's existing hardening)* —
  near-duplicates reinforce; contradictions flip `is_latest` (never delete); poisoned-sequence
  detection guards ingestion.

### 🌱 Evolves — gets sharper with time instead of bloating
- **Forgetting & reinforcement engine** *(MemPalace `dynamics.py` — the keystone)*. This
  un-defers the "automatic forgetting" memeora explicitly postponed (Risk B), with a
  neuroscience-grounded, **pure-heuristic** model applied to both memories and graph edges:
  - **Hebbian potentiation** on co-access: `strength = min(MAX, strength + δ)`.
  - **Ebbinghaus decay**: `strength = max(FLOOR, strength · exp(−Δdays / stability))` — higher
    stability ⇒ slower forgetting; `FLOOR ≈ 0.05` so nothing is ever truly lost, salience just
    drops.
  - **Cepeda spacing effect**: stability (durability) grows **only on spaced** repetition —
    rapid bursts don't build lasting memory.
- **Bi-temporal memory** *(Hindsight + MemPalace converge)* — **valid-time** (`occurred_start/
  end`: when it was true) separate from **transaction-time** (`created_at`: when we learned it)
  and **mention-time**. Query = interval-overlap on valid-time. Answers "what did I decide in
  2024?" *and* recency ranking without one clobbering the other. Both projects independently
  built this in SQLite — strong signal it's right.
- **Version chain** *(supermemory contract)* — explicit `root / parent / next` + `is_latest`;
  history is a traversable chain, never destroyed.
- **Freshness trends** *(Hindsight, LLM-free)* — each observation carries
  `stable | strengthening | weakening | new | stale`, computed from an evidence-density ratio
  (recent vs older windows). Pure arithmetic.

---

## The recall pipeline (one design, runs with zero required LLM)

memeora has 2 of 4 channels today (dense + BM25 + RRF + optional rerank). The distilled
pipeline — Hindsight's shape with its real constants as tunable starting points:

1. **Four channels in parallel:** dense (sqlite-vec) · BM25 (FTS5) · **graph** · **temporal**.
   - *Graph activation* (fed into fusion as a ranked list):
     `score = tanh(shared_entities × 0.5) + semantic_link[0.7,1.0] + causal_link(+1.0)`, summed
     so independent evidence channels each contribute.
   - *Temporal*: rule-based date parsing ("last spring" → `[τ_start, τ_end]`; **no** model
     fallback — degrade gracefully) → proximity around the window midpoint.
2. **Reciprocal Rank Fusion** — `k = 60`, rank-based, **equal-weight** (importance from rank,
   not per-channel multipliers); per-channel cap before fusing.
3. **Rerank** — cross-encoder *(fastembed)* **or passthrough** when no model: `CE = 1 − 0.9·rank/(n−1)`,
   so the whole pipeline runs with **only the embedder**, no second model required.
4. **Multiplicative boosts** — `final = base × recency × temporal_proximity × proof_count`,
   bounded (α = 0.2 / 0.2 / 0.1 → ±10% / ±10% / ±5%); a neutral signal collapses its boost to
   1.0 so secondaries never overpower relevance.
5. **Token-budget fill, not top-k** — agents think in tokens: fill a `max_tokens` budget
   top-down by `final` score. Directly serves memeora's "same answer, fewer tokens" value.

---

## Beyond the four agents — the surface reach

supermemory proves the model commercially: **one engine, thin per-surface adapters, ~25
surfaces.** memeora already has the *seam* for this — `rmcp` MCP server + **data-driven host
descriptors** + client SDKs (`memeora-client`, `@memeora/client`) + the conformance kit — so
breadth is adapters, **not** a re-architecture.

| Tier | Surfaces | Mechanism | Status |
|------|----------|-----------|--------|
| **A — any MCP host** | Cursor, Windsurf, Zed, Cline, Copilot, Gemini CLI, … | the universal `rmcp` server — **zero new code**, one config entry | free **today** |
| **B — coding agents** | Claude Code, Codex, Antigravity, OpenCode | command-hook + descriptor | **have** |
| **C — agent frameworks** | LangChain, LangGraph, CrewAI, Mastra, Agno, AI-SDK, OpenAI-Agents, VoltAgent | thin SDK middleware over IPC/MCP | new adapters |
| **D — voice agents** | Pipecat, Cartesia | context-provider + capture | new adapters |
| **E — no-code / automation** | n8n, Zapier, viasocket | HTTP nodes over the local API | new adapters |
| **F — personal surfaces** | browser extension, Raycast/Alfred, desktop quick-capture | local API + small UIs | new adapters |

All share the same engine, scoping, extraction, graph, and dynamics — added **without touching
the core**, because the trait + descriptor + SDK seams already exist.

---

## What we deliberately decline (and why)

| Declined | Source | Why |
|----------|--------|-----|
| `reflect`/Cara reasoning, **opinion network**, disposition knobs (skepticism/literalism/empathy) | Hindsight | an LLM-per-query *reasoning* layer, not a *memory* layer; at most opt-in Tier-3 |
| **AAAK** emotion/flag taxonomy | MemPalace | lossy heuristic their own docs disown for the headline benchmark; embeddings approximate the index better. (We keep the *idea* — a cheap pointer layer over verbatim — not the codes.) |
| Postgres/pgvector · ChromaDB · Qdrant default | Hindsight / MemPalace | conflicts with single-binary, sqlite-vec, local-first (kept behind the `VectorStore` trait for scale, benchmark-driven) |
| tree-sitter code analysis, React/d3 dashboard internals | Understand-Anything | different domain (code comprehension), and memeora's host-descriptor design is already cleaner than per-host plugin dirs |
| **LLM-required** extraction/consolidation *by default* | all | violates "no required LLM" — we make it the **opt-in local tier** with a heuristic fallback (exactly Hindsight's own `provider=none` escape hatch, but as a first-class floor) |
| hosted engine, OAuth/auth subsystem | supermemory | we are local; there is nothing to authenticate against |

---

## North star, in one sentence

A memory that is **yours**, runs on **your machine**, costs **nothing**, never forgets your
words — and gets *smarter* (consolidates), *sharper* (decays noise), *self-heals* (repairs its
own graph), and *follows you* across **every tool you use** through one local brain.
