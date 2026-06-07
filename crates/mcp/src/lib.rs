//! memeora MCP server (`rmcp`): exposes the memory engine as MCP tools so any
//! MCP-capable agent gets persistent memory with zero custom code.
//!
//! Each tool is a thin wrapper over [`memeora_client`]: it opens a short-lived
//! connection to the daemon on a blocking thread (the client is sync) and renders
//! the result as text. The socket defaults to [`memeora_proto::DEFAULT_SOCKET`].

use memeora_client::Client;
use memeora_proto::MemoryDto;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content};
use rmcp::{ErrorData, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

/// An MCP server backed by a memeora daemon.
#[derive(Clone)]
pub struct MemoryServer {
    socket: String,
}

/// Arguments for `recall`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallArgs {
    /// Scope/container tag to search within.
    pub scope: String,
    /// Natural-language query.
    pub query: String,
    /// Maximum number of results (default 10).
    pub k: Option<usize>,
}

/// Arguments for `remember`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RememberArgs {
    /// Scope/container tag to store under.
    pub scope: String,
    /// The memory content to store.
    pub content: String,
    /// Kind: `fact`, `preference`, or `episode` (default `fact`).
    pub kind: Option<String>,
}

/// Arguments for `context` and `list`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScopeArgs {
    /// Scope/container tag.
    pub scope: String,
    /// Max results, for `list` (default 20).
    pub limit: Option<usize>,
}

#[tool_router]
impl MemoryServer {
    /// Build a server that talks to the daemon at `socket`.
    pub fn new(socket: String) -> Self {
        Self { socket }
    }

    #[tool(description = "Search stored memories within a scope (hybrid dense + keyword search).")]
    async fn recall(
        &self,
        Parameters(args): Parameters<RecallArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let socket = self.socket.clone();
        let memories = blocking(move || {
            Client::connect(&socket)?.recall(&args.scope, &args.query, args.k.unwrap_or(10))
        })
        .await?;
        Ok(CallToolResult::success(vec![Content::text(render(
            &memories,
        ))]))
    }

    #[tool(description = "Store a memory (fact, preference, or episode) in a scope.")]
    async fn remember(
        &self,
        Parameters(args): Parameters<RememberArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let socket = self.socket.clone();
        let kind = args.kind.unwrap_or_else(|| "fact".to_string());
        let id = blocking(move || Client::connect(&socket)?.add(&args.scope, &args.content, &kind))
            .await?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "stored memory {id}"
        ))]))
    }

    #[tool(
        description = "Get the profile (stable facts/preferences + recent episodes) for a scope."
    )]
    async fn context(
        &self,
        Parameters(args): Parameters<ScopeArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let socket = self.socket.clone();
        let (statics, dynamics) =
            blocking(move || Client::connect(&socket)?.context(&args.scope)).await?;
        let text = format!(
            "## Stable\n{}\n\n## Recent\n{}",
            render(&statics),
            render(&dynamics)
        );
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(description = "List the most recent memories in a scope.")]
    async fn list(
        &self,
        Parameters(args): Parameters<ScopeArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let socket = self.socket.clone();
        let memories =
            blocking(move || Client::connect(&socket)?.list(&args.scope, args.limit.unwrap_or(20)))
                .await?;
        Ok(CallToolResult::success(vec![Content::text(render(
            &memories,
        ))]))
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {}

/// Run a sync client call on a blocking thread, mapping failures to MCP errors.
async fn blocking<T, F>(f: F) -> Result<T, ErrorData>
where
    F: FnOnce() -> std::io::Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))
}

/// Render memories as compact text for an agent to read.
fn render(memories: &[MemoryDto]) -> String {
    if memories.is_empty() {
        return "(none)".to_string();
    }
    memories
        .iter()
        .map(|m| format!("- [{}] {}", m.kind, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}
