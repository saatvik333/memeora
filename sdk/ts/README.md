# @memeora/client

TypeScript client for the [memeora](https://github.com/saatvik333/memeora) daemon
— persistent memory for AI tools, local-first, no API key. It speaks the same
versioned IPC protocol as the Rust SDK (`memeora-client`); see
[`docs/PROTOCOL.md`](../../docs/PROTOCOL.md).

## Install

```sh
bun add @memeora/client
```

Requires a running `memeora-daemon`.

## Usage

```ts
import { Client } from "@memeora/client";

// Recommended for cross-language use: run the daemon with a filesystem socket,
// e.g. MEMEORA_SOCKET=/tmp/memeora.sock, and pass that path here.
const client = await Client.connect("/tmp/memeora.sock");

await client.add("repo_myproject", "We use SQLite + sqlite-vec", "fact");
const hits = await client.recall("repo_myproject", "storage engine", 5);
const { statics, dynamics } = await client.context("repo_myproject");

console.log(client.getServerVersion(), client.getCapabilities());
client.close();
```

### Capability-gated params

Some params only take effect on a daemon that advertises the matching
capability (check with `client.supports("...")`); they're silently ignored by
older daemons rather than erroring:

- `ingest(scope, text, source?)` — `source` attributes the text to an observer
  (an agent/session id) so repeated corroboration from the same source can't
  inflate a memory's proof. Gate on `"evidence"`.
- `recall(scope, query, k?, maxTokens?)` — when set, the daemon fills results
  best-first up to `maxTokens` estimated tokens instead of a fixed `k` (which
  still caps the count). Gate on `"token_budget"`.
- `MemoryDto.freshness` — a coarse trend label (`new` / `strengthening` /
  `stable` / `weakening` / `stale`) from decay × distinct-source proof, present
  on results from a daemon with `"evidence"`; `null`/absent otherwise.

## Sockets

`Client.connect(socket)` accepts:

- a **filesystem path** (contains `/` or `\`) — used verbatim; the portable choice
  for cross-language use (start the daemon with `MEMEORA_SOCKET=/path/to.sock`);
- a **bare name** (the default `memeora-daemon.sock`) — mapped to the platform's
  namespaced form (Windows named pipe, Linux abstract namespace). On macOS/BSD,
  prefer a filesystem path.

## API

`connect` · `ingest` · `add` · `recall` · `context` · `list` · `forget` ·
`getServerVersion` · `getCapabilities` · `supports` · `close`. Each maps 1:1 to a
protocol verb and throws on a daemon `error` response or a protocol-version
mismatch.

## License

MIT OR Apache-2.0.
