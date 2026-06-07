# memeora dashboard

A local, function-first graph UI for the memeora memory engine — **Svelte 5 +
Vite + Sigma.js** (WebGL). Styling is deliberately minimal; the graph is the
centerpiece.

The built SPA is **embedded into the daemon binary** (via `rust-embed`) and
served by the daemon's HTTP server — no separate process, works offline. Open it
with `memeora dashboard`.

## Develop

```sh
pnpm install
pnpm dev          # Vite dev server; proxies /api to a running daemon
```

`pnpm dev` proxies `/api/*` to `http://127.0.0.1:7878` (the daemon's default
dashboard address — start `memeora-daemon` first). Override the proxy target in
`vite.config.ts` if you set `MEMEORA_DASHBOARD_ADDR`.

## Build (embed into the binary)

```sh
pnpm install
pnpm build        # → dist/, which the daemon embeds via rust-embed
cargo build --release -p memeora-daemon
```

If `dist/` is absent at compile time the daemon's `build.rs` writes a placeholder
page, so the Rust build (and CI) never depends on a Node toolchain. Building the
real UI simply replaces that placeholder.

## What it shows

- **Graph canvas** — nodes are memories (color by kind: fact / preference /
  episode; size by strength; superseded/forgotten nodes dimmed), edges are
  `extends` / `updates` / `derives` relationships. Click a node to inspect it.
- **Spaces switcher** — every scope (`user` / `project` / `repo` container tag).
- **Search** — hybrid search within the current space.
- **Inspector** — content, kind, strength, relative timestamps; soft-forget.
- **Live mode** — an SSE stream (`/api/events`) refreshes the graph as the daemon
  ingests or forgets memories.

## API (served by the daemon)

`GET /api/scopes` · `GET /api/graph?scope=` · `GET /api/list?scope=&limit=` ·
`GET /api/context?scope=` · `GET /api/search?scope=&q=&k=` ·
`POST /api/forget {id}` · `GET /api/events` (SSE) · `GET /api/health`.
