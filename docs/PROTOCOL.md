# memeora IPC protocol

The daemon and every client (the Rust SDK `memeora-client`, the TS SDK
`@memeora/client`, the MCP server, the hook, the CLI) talk over a single
**versioned, public contract**: `memeora-proto`. This document is the stability
policy the ecosystem can build on.

## Transport

- A local socket (Unix domain socket / Windows named pipe) via `interprocess`.
  The name resolves from a string: one containing a path separator is a
  filesystem socket path, otherwise a namespaced name (`DEFAULT_SOCKET =
  "memeora-daemon.sock"`, overridable with `MEMEORA_SOCKET`).
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
| `list` | `memories` | `list` |
| `forget` | `forgotten` | `forget` |

Any request can also come back as `{"type":"error","message":"…"}`.

## Handshake & capability negotiation

The first message on a connection is `hello`:

```jsonc
// client → daemon
{ "op": "hello", "protocol_version": 1 }
// daemon → client
{ "type": "hello", "protocol_version": 1, "server_version": "0.1.0",
  "capabilities": ["ingest","add","recall","context","list","forget"] }
```

- `protocol_version` — the **wire** version. A client must refuse a daemon whose
  `protocol_version` differs from the one it was built against (`memeora-client`
  returns `io::ErrorKind::Unsupported`).
- `capabilities` — the optional features/operations this daemon supports. Clients
  gate optional behavior on these tokens (`Client::supports(cap)`), **not** on the
  version number.

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
