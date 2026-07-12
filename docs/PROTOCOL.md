# memeora IPC protocol

The daemon and every client (the Rust SDK `memeora-client`, the TS SDK
`@memeora/client`, the MCP server, the hook, the CLI) talk over a single
**versioned, public contract**: `memeora-proto`. This document is the stability
policy the ecosystem can build on.

## Transport

- A local socket (Unix domain socket / Windows named pipe) via `interprocess`.
  The name resolves from a string: one containing a path separator is a
  filesystem socket path, otherwise a namespaced name (`DEFAULT_SOCKET =
  "memeora-daemon.sock"`, overridable with `MEMEORA_SOCKET`). On macOS/BSD, use
  a filesystem socket path because bare namespaced names cannot be matched by the
  Rust/Node clients.
- **Framing:** each message is a `u32` big-endian length prefix followed by that
  many bytes of UTF-8 JSON. Messages larger than `MAX_MESSAGE_BYTES` (16 MiB) are
  rejected. A clean EOF between frames is a normal close.

## Messages

Requests are a serde-tagged enum on `"op"`; responses on `"type"`:

| Request (`op`) | Response (`type`) | Capability |
|----------------|-------------------|------------|
| `hello` | `hello` | — (always) |
| `ingest` | `ingested` | `ingest` |
| `add` | `added` | `add` |
| `recall` | `memories` | `recall` |
| `context` | `context` | `context` |
| `bundle` | `bundle` | `bundle` |
| `list` | `memories` | `list` |
| `forget` | `forgotten` | `forget` |

Any request can also come back as `{"type":"error","message":"…"}`.

`bundle` is a one-round-trip convenience: `{"op":"bundle","scope":"…","query":"…","k":10,"max_tokens":null}`
returns `{"type":"bundle","statics":[…],"dynamics":[…],"memories":[…]}` — the scope
profile (statics + dynamics, as `context`) plus the query's recall hits (as `recall`),
with any recall hit already in the profile removed (priority static > dynamic > search).
Additive + capability-gated (`bundle`); older daemons simply don't advertise it.

### Optional, capability-gated fields

Beyond the required fields implied by each op, three fields are additive and
`#[serde(default)]`, so older peers that omit them still round-trip; each is
gated on a capability token rather than the version:

| Field | On | Type | Gated by |
|-------|----|------|----------|
| `source` | `ingest` request | `string \| null` | `evidence` |
| `max_tokens` | `recall` request | `number \| null` | `token_budget` |
| `freshness` | `MemoryDto` (in `memories`/`context` responses) | `string \| null` | `evidence` |

- `ingest.source` attributes ingested text to an observer (agent/session id).
  Repeated corroboration from the *same* source can't inflate a memory's
  `proof_count` — only distinct sources raise it. When omitted, each distinct
  statement stands in as its own source.
- `recall.max_tokens` fills results best-first up to that many estimated tokens
  instead of a fixed `k` (`k` still caps the count).
- `MemoryDto.freshness` is a coarse decay × distinct-source-proof trend label:
  `new` / `strengthening` / `stable` / `weakening` / `stale`.

```jsonc
// client → daemon
{ "op": "ingest", "scope": "repo_memeora", "text": "I prefer rust", "source": "agent-x" }
{ "op": "recall", "scope": "repo_memeora", "query": "language", "k": 5, "max_tokens": 2000 }
// daemon → client (a MemoryDto inside "memories"/"context")
{ "id": "m1", "content": "I prefer rust", "kind": "preference", "strength": 1.0,
  "created_at": 1700000000, "score": 0.42, "freshness": "stable" }
```

## Handshake & capability negotiation

The first message on a connection is `hello`:

```jsonc
// client → daemon
{ "op": "hello", "protocol_version": 1 }
// daemon → client
{ "type": "hello", "protocol_version": 1, "server_version": "0.1.0",
  "capabilities": ["ingest","add","recall","context","list","forget","token_budget","evidence"] }
```

- `protocol_version` — the **wire** version. A client must refuse a daemon whose
  `protocol_version` differs from the one it was built against (`memeora-client`
  returns `io::ErrorKind::Unsupported`).
- `capabilities` — the optional features/operations this daemon supports. Clients
  gate optional behavior on these tokens (`Client::supports(cap)`), **not** on the
  version number. Two tokens gate fields rather than whole operations:
  `token_budget` gates `max_tokens` on `recall`, and `evidence` gates `source` on
  `ingest` plus `freshness` on returned memories (see below).

## Versioning policy

`PROTOCOL_VERSION` is bumped **only for breaking changes** — anything that would
make an existing client misread a message:

- removing or renaming a request/response variant or field,
- changing a field's type or the meaning of an existing value,
- changing the framing.

**Additive changes do not bump the version**, because they're backward
compatible:

- a **new capability token** (advertise it in `capabilities`; old clients ignore
  unknown tokens),
- a **new optional field** (must be `#[serde(default)]` so older peers that omit
  it still parse — e.g. `capabilities` itself was added this way and an older
  daemon's `hello` without it still deserializes to an empty set),
- a **new request variant** a daemon may answer with `error` if it predates it.

This lets the engine evolve while adapters and SDKs pinned to a major version keep
working: negotiate features through `capabilities`, reserve version bumps for true
breaks.

## Adding a harness

You usually need **no protocol code**: any MCP-capable tool gets memory via the
MCP server with a config entry, and command-hook tools are described by a data
file — see [`ADAPTERS.md`](ADAPTERS.md).
