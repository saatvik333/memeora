# memeora adapters

Per-tool packaging that plugs the memeora engine into each AI coding host. All
four are thin wrappers over the **same** backend — one daemon, one `memeora-mcp`
server, one multi-host `memeora-hook` binary — so memory is shared across tools
by construction (same DB, same scope tags).

| Host | MCP tools | Auto-capture | Inject | Notes |
|------|-----------|--------------|--------|-------|
| [Claude Code](./claude-code) | `.mcp.json` → `memeora-mcp` | `Stop` / `PreCompact` hook | `SessionStart` → `additionalContext` | installs via plugin marketplace |
| [Codex](./codex) | `[mcp_servers]` in `config.toml` | `Stop` / `PreCompact` hook | `SessionStart` → `additionalContext` | same hook path as Claude |
| [Antigravity](./antigravity) | `mcp_config.json` → `memeora-mcp` | `Stop` hook | `PreInvocation` → `injectSteps` | own camelCase schema |
| [OpenCode](./opencode) | native MCP config | (via tools / SDK) | compaction-prompt injection | thin TS shim (only non-Rust) |

## How it fits together

```
host event ──► memeora-hook --host <h> ──► daemon (capture / fetch profile)
host MCP   ──► memeora-mcp            ──► daemon (recall/remember/context/list)
                                            │
                                  ~/.memeora/memory.db  (one store, scoped by tag)
```

- **`memeora-hook`** parses each host's payload (`--host claude|codex|antigravity`)
  and renders the host's expected stdout (`additionalContext` vs `injectSteps`),
  capturing transcripts at turn-end and injecting the project profile at start.
- **`memeora-mcp`** exposes `recall` / `remember` / `context` / `list`. Scope is
  optional and **defaults to the current project** (`project_tag(cwd)`), so MCP
  tools and hook capture converge on the same container automatically.
- **`memeora scope [path]`** prints that project tag (daemon-free) — the seam the
  OpenCode shim uses to stay coherent with the Rust hosts.

## Prerequisites (all hosts)

The binaries must be on `PATH` and the daemon running:

```sh
cargo install --path crates/cli --path crates/mcp --path crates/hook --path crates/daemon
memeora-daemon &
```

A first-class installer (`memeora install <host>`) and prebuilt binaries arrive
with the release step. Each host directory has its own README with exact install
steps and host-specific caveats.

> Hook payload schemas evolve per host; they're parsed defensively but should be
> validated against real captured fixtures before relying on auto-capture.
