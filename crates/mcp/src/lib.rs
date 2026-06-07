//! memeora MCP server (`rmcp`): exposes the memory engine as MCP tools so any
//! MCP-capable agent gets persistent memory with zero custom code.
//!
//! Each tool is a thin wrapper over [`memeora_client`]: it opens a short-lived
//! connection to the daemon on a blocking thread (the client is sync) and renders
//! the result as text. The socket defaults to [`memeora_proto::DEFAULT_SOCKET`].

use memeora_client::Client;
use memeora_core::container_tag::project_tag;
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
    /// Scope/container tag to search within (defaults to the current project).
    pub scope: Option<String>,
    /// Natural-language query.
    pub query: String,
    /// Maximum number of results (default 10).
    pub k: Option<usize>,
}

/// Arguments for `remember`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RememberArgs {
    /// Scope/container tag to store under (defaults to the current project).
    pub scope: Option<String>,
    /// The memory content to store.
    pub content: String,
    /// Kind: `fact`, `preference`, or `episode` (default `fact`).
    pub kind: Option<String>,
}

/// Arguments for `context` and `list`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScopeArgs {
    /// Scope/container tag (defaults to the current project).
    pub scope: Option<String>,
    /// Max results, for `list` (default 20).
    pub limit: Option<usize>,
}

/// Resolve a caller-supplied scope to a concrete container tag. An empty or
/// missing scope defaults to the project tag for the server's working directory,
/// so MCP tools and the `memeora-hook` capture path agree on the same scope.
fn resolve_scope(scope: Option<String>) -> String {
    match scope {
        Some(s) if !s.trim().is_empty() => s,
        _ => {
            let cwd = std::env::current_dir()
                .ok()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            project_tag(&cwd)
        }
    }
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
        let scope = resolve_scope(args.scope);
        let memories = blocking(move || {
            Client::connect(&socket)?.recall(&scope, &args.query, args.k.unwrap_or(10))
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
        let scope = resolve_scope(args.scope);
        let kind = args.kind.unwrap_or_else(|| "fact".to_string());
        let id =
            blocking(move || Client::connect(&socket)?.add(&scope, &args.content, &kind)).await?;
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
        let scope = resolve_scope(args.scope);
        let (statics, dynamics) =
            blocking(move || Client::connect(&socket)?.context(&scope)).await?;
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
        let scope = resolve_scope(args.scope);
        let memories =
            blocking(move || Client::connect(&socket)?.list(&scope, args.limit.unwrap_or(20)))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_scope_is_passed_through() {
        assert_eq!(
            resolve_scope(Some("repo_memeora".into())),
            "repo_memeora".to_string()
        );
    }

    #[test]
    fn missing_or_blank_scope_falls_back_to_project_tag() {
        let default = resolve_scope(None);
        assert!(default.starts_with("memeora_project_"));
        // Blank strings resolve the same way as a missing scope.
        assert_eq!(resolve_scope(Some("   ".into())), default);
    }
}
