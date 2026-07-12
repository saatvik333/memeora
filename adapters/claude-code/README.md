# memeora — Claude Code adapter

Gives Claude Code persistent memory: MCP tools (`recall` / `remember` / `context`
/ `list`), auto-injected project context at session start, and auto-capture of
each turn into the memeora knowledge graph.

## Prerequisites

The `memeora-mcp` and `memeora-hook` binaries must be on your `PATH`, and the
daemon must be running:

```sh
cargo install --path crates/cli   # builds memeora, memeora-mcp, memeora-hook, memeora-daemon
memeora serve &         # loads the model + DB once, serves the local socket
```

(`memeora serve` is the daemon; `memeora-daemon` is the same thing. A first-class
installer lands with the release step; for now `cargo install` puts the binaries
on `PATH`.)

## Install (plugin marketplace)

```
/plugin marketplace add saatvik333/memeora
/plugin install memeora@memeora
```

This wires up, from this directory:

- **`.mcp.json`** — registers the `memeora-mcp` stdio server (the memory tools).
- **`hooks/hooks.json`** — `SessionStart` injects the project profile; `Stop` and
  `PreCompact` capture the transcript.
- **`commands/context.md`** — `/memeora:context` to pull the profile on demand.
- **`skills/remember/`** — guidance for proactively storing memories.

## Scoping

Tools and hooks default to the **current project** scope (`project_tag(cwd)`), so
captured memory and recalled memory line up automatically. Pass an explicit
`scope` to a tool to target a different container (e.g. a shared `repo_*` tag).
