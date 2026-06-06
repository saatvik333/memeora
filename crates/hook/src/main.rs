//! memeora-hook (scaffold).
//!
//! One binary invoked by Claude Code / Codex / Antigravity command-hooks. Selects a
//! per-host parser/renderer via `--host` (driven by `hosts/*.toml` descriptors) and
//! forwards capture/inject calls to the daemon over IPC.

fn main() {
    println!(
        "memeora-hook {} (client protocol v{})",
        env!("CARGO_PKG_VERSION"),
        memeora_client::PROTOCOL_VERSION,
    );
}
