# memeora — Codex adapter

Gives OpenAI Codex persistent memory via the same `memeora-mcp` server and
`memeora-hook` binary used by the other hosts. Codex shares Claude Code's
`SessionStart.additionalContext` injection format and `Stop` capture, so the
`--host codex` path mirrors `--host claude`.

## Prerequisites

`memeora-mcp` and `memeora-hook` on `PATH`, and the daemon running:

```sh
cargo install --path crates/mcp --path crates/hook --path crates/daemon
memeora-daemon &
```

## Install — stable path (recommended)

Codex's plugin loader is in flux (see *Caveats*), so wire memeora through the
documented global config, which is stable:

1. **MCP tools** — merge [`config.toml`](./config.toml) into `~/.codex/config.toml`:

   ```toml
   [mcp_servers.memeora]
   command = "memeora-mcp"
   enabled = true
   ```

2. **Hooks** — copy [`hooks/hooks.json`](./hooks/hooks.json) to `~/.codex/hooks.json`
   (or `<project>/.codex/hooks.json`). On first run, **trust** the hooks via the
   `/hooks` command in Codex — untrusted command hooks are skipped until reviewed.

Verify with `codex mcp` (lists `memeora`) and `/hooks` (shows the three hooks).

## Install — as a plugin (experimental)

This directory is also a Codex plugin: [`.codex-plugin/plugin.json`](./.codex-plugin/plugin.json)
references `./.mcp.json` and `./hooks/hooks.json` (auto-discovered). Codex sets
`CLAUDE_PLUGIN_ROOT`/`CLAUDE_PLUGIN_DATA` for plugin hooks.

## Caveats (verify against your Codex build)

- **Manifest `hooks` may be rejected.** Plugin-bundled hooks are gated behind the
  `plugin_hooks` feature flag, and OpenAI's own `plugin-creator` validator has
  rejected a `hooks` manifest field in some builds. Prefer the global
  `~/.codex/hooks.json` path above until you've confirmed manifest hooks work.
- **Hook trust gate.** New/changed command hooks are skipped until trusted via
  `/hooks` (trust is keyed to the hook's hash). For automation only:
  `--dangerously-bypass-hook-trust`.
- **`Stop` requires JSON on stdout** when it exits 0 — the hook emits `{}`.
