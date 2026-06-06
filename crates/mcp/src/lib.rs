//! memeora MCP server.
//!
//! Built on `rmcp` (stdio + streamable HTTP), exposing the universal tools
//! `memory`, `recall`, `context`, and `list` as a thin client over the daemon.

/// Tool names exposed over MCP. Kept stable as part of the public surface.
pub const TOOLS: &[&str] = &["memory", "recall", "context", "list"];
