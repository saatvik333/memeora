//! `memeora-mcp` — the MCP server binary (memory tools over stdio).
//!
//! A thin wrapper: all logic lives in [`memeora_mcp::run`] so every memeora
//! binary ships from the one `memeora` package (a single `dist` installer).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    memeora_mcp::run()
}
