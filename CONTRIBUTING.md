# Contributing to memeora

Thanks for helping build memeora! This guide covers conventions and local setup.

## Development setup
- Rust toolchain is pinned via `rust-toolchain.toml` (1.95, edition 2024) — `rustup` installs it
  automatically (incl. `rustfmt`, `clippy`, `rust-analyzer`).
- TS tooling (dashboard, OpenCode shim) uses **bun**.

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo deny check        # advisories + license policy (see deny.toml)
```

## Commit conventions
- **[Conventional Commits](https://www.conventionalcommits.org/)**, subject **< 200 characters**.
- Format: `type(scope): summary` — e.g. `feat(core): add hybrid RRF search`, `fix(hook): handle empty stdin`.
- Common scopes: `core`, `proto`, `client`, `daemon`, `mcp`, `hook`, `cli`, `adapters`, `dashboard`, `ci`, `docs`.

## Workflow
- **Think before implementing**; break work into a task hierarchy.
- **Keep docs updated** alongside code — `docs/ARCHITECTURE.md` is the source of truth.
- CI must pass: `fmt --check`, `clippy -D warnings`, `test`, and `cargo-deny`.
- New code matches surrounding style; format runs automatically on save.

## Adding a new harness (plugin ecosystem)
memeora is built so you can support a new tool **without forking the core**:
1. If the harness speaks **MCP**, it already works — just point it at the memeora MCP server.
2. For auto-capture via command-hooks (Claude/Codex/Antigravity style), add a **host descriptor**
   (`adapters/_descriptors/<harness>.toml`: event-name map + stdin parsing + stdout rendering) — no Rust needed.
3. For in-process plugins (OpenCode style), write a thin shim against `@memeora/client` (TS) or
   `memeora-client` (Rust) that forwards to the daemon.
4. Validate with the conformance kit before submitting.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and rationale.
