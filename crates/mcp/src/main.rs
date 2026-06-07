//! memeora MCP server binary — serves the memory tools over stdio.
//!
//! Point an MCP-capable client (Claude Code, Codex, …) at this binary. It talks
//! to the daemon at `$MEMEORA_SOCKET` (or [`memeora_proto::DEFAULT_SOCKET`]).

use memeora_mcp::MemoryServer;
use rmcp::ServiceExt;
use rmcp::transport::stdio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket = std::env::var("MEMEORA_SOCKET")
        .unwrap_or_else(|_| memeora_proto::DEFAULT_SOCKET.to_string());
    let service = MemoryServer::new(socket).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
