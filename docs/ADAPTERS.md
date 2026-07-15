# Add your harness

memeora is designed so the community can add support for new AI tools (Cursor,
Windsurf, Zed, Aider, Cline, …) **without forking the core**. The engine is a
stable backend behind two public, versioned contracts — the MCP server and the
daemon IPC protocol ([`PROTOCOL.md`](PROTOCOL.md)) — and adapters never touch
engine internals.

There are three ways to integrate, easiest first.

## 1. MCP — zero code

Any MCP-capable harness gets memory by pointing it at the `memeora-mcp` server
(stdio). Tools: `recall`, `remember`, `context`, `list`, `forget`. This is the universal
baseline — just a config entry. The scope defaults to the current project
(`MEMEORA_PROJECT_ROOT` → cwd), or pass an explicit `scope`. The MCP server talks
to the daemon over `$MEMEORA_SOCKET` (or `memeora-daemon.sock`), so hosts should
set `MEMEORA_SOCKET` when they run the daemon on a filesystem socket.

## 2. Command-hook host — a descriptor file (no Rust)

If your harness can run a shell command on lifecycle events (session start, turn
end, pre-compaction), it can use `memeora-hook` for **auto-capture and context
injection**. Everything host-specific is data — a [`HostDescriptor`](../crates/hook/src/descriptor.rs):

```toml
name = "yourtool"
scope_fields       = ["cwd"]                 # where the project dir is in the payload
transcript_fields  = ["transcript_path"]     # where the transcript file path is
inject_style       = "additional_context"    # or "inject_steps"
inject_event_name  = "SessionStart"          # for additional_context
capture_ack        = "{}"                    # JSON the capture event must print ("" = none)
# invocation_gate_field = "invocationNum"     # inject only on the first invocation
```

Field paths are dotted; numeric segments index arrays (`workspacePaths.0`). Scope
resolution falls back to the process cwd; transcript/scope try each field in order.

### Start from a first-party descriptor

```sh
cp adapters/_descriptors/claude.toml adapters/yourtool.toml   # then edit for your host
```

Edit the descriptor, then wire your harness's hooks to call:

```sh
memeora-hook --descriptor /abs/path/yourtool.toml --event session-start   # inject
memeora-hook --descriptor /abs/path/yourtool.toml --event stop            # capture
memeora-hook --descriptor /abs/path/yourtool.toml --event pre-compact     # capture
```

The four canonical `--event` values (`session-start`, `pre-invocation`, `stop`,
`pre-compact`) cover all hosts; map your harness's events onto them in its hook
config. The first-party descriptors in [`adapters/_descriptors/`](../adapters/_descriptors/)
are working references.

### Prove it conforms

Drop a few real stdin payloads into `crates/hook/tests/fixtures/yourtool/*.json`
with the expected outcomes, then run the conformance kit:

```sh
cargo test -p memeora-hook --test conformance
```

```jsonc
// crates/hook/tests/fixtures/yourtool/session-start.json
{
  "host": "yourtool",
  "event": "session-start",
  "payload": { "cwd": "/home/u/proj" },
  "expect": { "scope_dir": "/home/u/proj", "should_inject": true,
              "inject_contains": "additionalContext" }
}
```

A built-in descriptor is required for `host` in fixtures; a custom descriptor used
only via `--descriptor` is validated by running the hook against your fixtures
directly. (Each `expect` field is optional — assert only what matters.)

## 3. In-process plugin — a thin shim

Harnesses that only support in-process plugins (e.g. OpenCode) need a small
language-native shim that calls the daemon. Use a client SDK:

- **Rust:** [`memeora-client`](../crates/client) — typed methods over the IPC.
- **TypeScript:** [`@memeora/client`](../sdk/ts) — same protocol from Node.

Both perform the version + capability handshake and expose the supported operations
(`ingest`/`add`/`recall`/`context`/`bundle`/`list`/`forget`/`consolidate`). For
cross-language use, run the daemon on a filesystem socket
(`MEMEORA_SOCKET=/path/to.sock`).

## Stability

The IPC protocol is versioned and negotiates optional features via capabilities,
so an adapter pinned to a major version keeps working as the engine evolves — see
[`PROTOCOL.md`](PROTOCOL.md). Contribute adapters in-repo under `adapters/` or in
your own repo against the published contract.
