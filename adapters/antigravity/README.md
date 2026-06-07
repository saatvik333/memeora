# memeora — Antigravity adapter

Gives Google Antigravity persistent memory. Antigravity uses its **own** hook
schema (camelCase events and payloads), so the hook runs under `--host
antigravity`, which differs from the Claude/Codex path:

- **No session-start event** → context is injected on `PreInvocation`, gated to
  the first invocation (`invocationNum == 1`), via `injectSteps[].userMessage`.
- **Scope** comes from `workspacePaths[0]`; the transcript path is `transcriptPath`.
- **`Stop`** captures the transcript and returns a `decision` (non-`"continue"`
  so the turn ends normally).

## Prerequisites

`memeora-mcp` and `memeora-hook` on `PATH`, and the daemon running:

```sh
cargo install --path crates/mcp --path crates/hook --path crates/daemon
memeora-daemon &
```

## Install

Copy the plugin bundle into Antigravity's plugins directory:

```sh
# CLI:
cp -r plugins/memeora ~/.gemini/antigravity-cli/plugins/
# or shared (all Antigravity surfaces):
cp -r plugins/memeora ~/.gemini/config/plugins/
# or per-workspace:
cp -r plugins/memeora <workspace>/.agents/plugins/
```

The bundle provides:

- **`plugin.json`** — the plugin marker.
- **`mcp_config.json`** — registers the `memeora-mcp` stdio server.
- **`hooks.json`** — `PreInvocation` (inject) + `Stop` (capture).
- **`skills/remember/`** — guidance for proactively storing memories.

The IDE build may read a separate path (`~/.gemini/antigravity-ide/`) — check
your install.

## Caveats (verify against a real fixture)

Antigravity's 2.x config layout is still settling, so confirm on your build:

- **`env` passthrough for stdio MCP servers** has been reported broken in early
  2.x builds. memeora-mcp needs no env by default, but note it if you customize.
- **Hook I/O shapes** (`injectSteps`, `decision`, `transcriptPath`,
  `workspacePaths`) are parsed defensively; validate with a captured payload.
- **IDE vs CLI config paths** diverge post-2.x; the IDE may not read the shared
  layout.
