# memeora — OpenCode adapter

The only non-Rust adapter: a thin TypeScript plugin that forwards to the
`memeora` CLI (a client over the daemon), so no IPC is reimplemented in TS.

## Prerequisites

The `memeora` CLI (and the daemon) must be installed and running:

```sh
cargo install --path crates/cli   # builds memeora, memeora-mcp, memeora-hook, memeora-daemon
memeora-daemon &
```

## Option A — MCP only (zero code)

OpenCode speaks MCP natively. For just the memory tools, add `memeora-mcp` to
`~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "memeora": {
      "type": "local",
      "command": ["memeora-mcp"],
      "enabled": true
    }
  }
}
```

The tools default to the current project scope.

## Option B — this plugin

For the convenience tools (`memeora_recall` / `memeora_remember` /
`memeora_context`) plus profile injection into the compaction prompt, load the
plugin. Either publish it (`bun publish`) and reference it:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@memeora/opencode"]
}
```

…or drop [`src/index.ts`](./src/index.ts) into your project's `.opencode/plugin/`
directory.

## Scoping

The plugin resolves the project scope via `memeora scope <dir>` (daemon-free), so
its tools and the captured memory share the same container tag as the other
hosts.

## Status

The custom tools and compaction injection are wired and correct against the
documented OpenCode plugin API (`tool()`, `experimental.session.compacting`).
Auto-capture on `session.idle`/`message.updated` requires fetching session
messages via the OpenCode SDK; that shape should be validated against your
installed `@opencode-ai/sdk` version before wiring it — it's intentionally left
out of this thin shim for now (use `memeora_remember` or the Stop-hook hosts for
capture).
