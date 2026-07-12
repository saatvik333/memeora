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
use rmcp::transport::stdio;
use rmcp::{ErrorData, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Run the MCP server over stdio until the client disconnects.
///
/// Lives in the library (not a `main.rs`) so the single shipped `memeora` package
/// can expose every binary from one crate (see `docs/ARCHITECTURE.md`, Step 10).
/// Talks to the daemon at `$MEMEORA_SOCKET` (or [`memeora_proto::DEFAULT_SOCKET`]).
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let socket = std::env::var("MEMEORA_SOCKET")
            .unwrap_or_else(|_| memeora_proto::DEFAULT_SOCKET.to_string());
        let service = MemoryServer::new(socket).serve(stdio()).await?;
        service.waiting().await?;
        Ok(())
    })
}

/// An MCP server backed by a memeora daemon.
#[derive(Clone)]
pub struct MemoryServer {
    socket: String,
    /// Scope used when a tool call omits one. Resolved once at startup (see
    /// [`default_scope`]) because an MCP stdio server's process cwd is fixed at
    /// launch and isn't a reliable per-call project signal.
    default_scope: String,
    /// Per-turn de-dup cache for the read tools (see [`RecallCache`]); shared
    /// across clones so every request handler sees the same entries.
    cache: Arc<RecallCache>,
}

const CALL_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a cached recall/context result may be served.
const CACHE_TTL: Duration = Duration::from_secs(30);

/// Max cached results; the oldest entry is evicted first.
const CACHE_CAP: usize = 64;

/// Key for one read-tool invocation: the tool, its normalized inputs, and the
/// cache generation observed when the key was built. A write bumps the
/// generation, so keys built afterwards can never match pre-write entries.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheKey {
    tool: &'static str,
    generation: u64,
    scope: String,
    query: String,
    k: usize,
    max_tokens: Option<usize>,
}

/// A short-TTL, size-bounded cache that de-duplicates identical `recall`/`context`
/// calls within one agent turn (tool loops often repeat the same lookup), so a
/// burst hits the daemon once. It is *not* a correctness cache: entries expire
/// after [`CACHE_TTL`], at most [`CACHE_CAP`] live at once, and any mutation in
/// this process ([`Self::invalidate`]) orphans everything cached before it.
struct RecallCache {
    ttl: Duration,
    cap: usize,
    /// Bumped on every mutating tool call; part of every [`CacheKey`].
    generation: AtomicU64,
    /// Oldest-first; at most `cap` entries, so a linear scan is fine.
    entries: Mutex<VecDeque<(CacheKey, Instant, String)>>,
}

impl RecallCache {
    fn new(ttl: Duration, cap: usize) -> Self {
        Self {
            ttl,
            cap,
            generation: AtomicU64::new(0),
            entries: Mutex::new(VecDeque::new()),
        }
    }

    /// Build the key for a call, capturing the current generation and
    /// normalizing the query (trim, collapse whitespace, lowercase).
    fn key(
        &self,
        tool: &'static str,
        scope: &str,
        query: &str,
        k: usize,
        max_tokens: Option<usize>,
    ) -> CacheKey {
        CacheKey {
            tool,
            generation: self.generation.load(Ordering::Relaxed),
            scope: scope.to_string(),
            query: normalize_query(query),
            k,
            max_tokens,
        }
    }

    /// The cached rendered result for `key`, if present and fresh.
    fn get(&self, key: &CacheKey) -> Option<String> {
        let now = Instant::now();
        let entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        entries
            .iter()
            .find(|(k, at, _)| k == key && now.duration_since(*at) < self.ttl)
            .map(|(_, _, text)| text.clone())
    }

    /// Cache the rendered result for `key`, evicting the oldest entry at capacity.
    fn insert(&self, key: CacheKey, text: String) {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        entries.retain(|(k, _, _)| k != &key);
        if entries.len() >= self.cap {
            entries.pop_front();
        }
        entries.push_back((key, Instant::now(), text));
    }

    /// Called before every mutating tool call: bumps the generation so all
    /// existing entries (keyed under older generations) stop matching.
    fn invalidate(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
}

/// Normalize a query for cache-key purposes: trim, collapse internal
/// whitespace, lowercase.
fn normalize_query(query: &str) -> String {
    query
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
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

/// Arguments for `list`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScopeArgs {
    /// Scope/container tag (defaults to the current project).
    pub scope: Option<String>,
    /// Max results (default 20).
    pub limit: Option<usize>,
}

/// Arguments for `context` — scope only (no `limit`, which `context` ignores, so it
/// must not appear in the tool's JSON schema and mislead callers).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ContextArgs {
    /// Scope/container tag (defaults to the current project).
    pub scope: Option<String>,
}

/// Arguments for `forget`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ForgetArgs {
    /// The id of the memory to forget (soft-delete — history is preserved).
    pub id: String,
}

/// Resolve a caller-supplied scope to a concrete container tag, falling back to
/// the server's `default` when the caller omits one (or passes blank).
fn resolve_scope(scope: Option<String>, default: &str) -> String {
    match scope {
        Some(s) if !s.trim().is_empty() => s,
        _ => default.to_string(),
    }
}

/// The scope used when a tool call omits one.
///
/// Prefers `MEMEORA_PROJECT_ROOT` (the host can set this to the actual project
/// dir, since the MCP server's process cwd is fixed at launch and unreliable),
/// then the process cwd, then a stable named fallback (never an empty bucket).
fn default_scope() -> String {
    let root = std::env::var("MEMEORA_PROJECT_ROOT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|p| p.display().to_string())
                .filter(|s| !s.is_empty())
        });
    match root {
        Some(path) => project_tag(&path),
        None => "memeora_project_unknown".to_string(),
    }
}

#[tool_router]
impl MemoryServer {
    /// Build a server that talks to the daemon at `socket`.
    pub fn new(socket: String) -> Self {
        Self {
            socket,
            default_scope: default_scope(),
            cache: Arc::new(RecallCache::new(CACHE_TTL, CACHE_CAP)),
        }
    }

    #[tool(description = "Search stored memories within a scope (hybrid dense + keyword search).")]
    async fn recall(
        &self,
        Parameters(args): Parameters<RecallArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let scope = resolve_scope(args.scope, &self.default_scope);
        let k = args.k.unwrap_or(10);
        let key = self.cache.key("recall", &scope, &args.query, k, None);
        if let Some(text) = self.cache.get(&key) {
            return Ok(CallToolResult::success(vec![Content::text(text)]));
        }
        let socket = self.socket.clone();
        let memories =
            blocking(move || Client::connect(&socket)?.recall(&scope, &args.query, k)).await?;
        let text = render(&memories);
        self.cache.insert(key, text.clone());
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(description = "Store a memory (fact, preference, or episode) in a scope.")]
    async fn remember(
        &self,
        Parameters(args): Parameters<RememberArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let socket = self.socket.clone();
        let scope = resolve_scope(args.scope, &self.default_scope);
        let kind = args.kind.unwrap_or_else(|| "fact".to_string());
        // Invalidate before the write so no lookup issued after this point can be
        // served a pre-write result (even if the write itself outlives a timeout).
        self.cache.invalidate();
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
        Parameters(args): Parameters<ContextArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let scope = resolve_scope(args.scope, &self.default_scope);
        let key = self.cache.key("context", &scope, "", 0, None);
        if let Some(text) = self.cache.get(&key) {
            return Ok(CallToolResult::success(vec![Content::text(text)]));
        }
        let socket = self.socket.clone();
        let (statics, dynamics) =
            blocking(move || Client::connect(&socket)?.context(&scope)).await?;
        let text = format!(
            "## Stable\n{}\n\n## Recent\n{}",
            render(&statics),
            render(&dynamics)
        );
        self.cache.insert(key, text.clone());
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(description = "List the most recent memories in a scope.")]
    async fn list(
        &self,
        Parameters(args): Parameters<ScopeArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let socket = self.socket.clone();
        let scope = resolve_scope(args.scope, &self.default_scope);
        let memories =
            blocking(move || Client::connect(&socket)?.list(&scope, args.limit.unwrap_or(20)))
                .await?;
        Ok(CallToolResult::success(vec![Content::text(render(
            &memories,
        ))]))
    }

    #[tool(
        description = "Forget (soft-delete) a memory by id. History is preserved; it just stops surfacing in recall/list/context."
    )]
    async fn forget(
        &self,
        Parameters(args): Parameters<ForgetArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let socket = self.socket.clone();
        let id = args.id;
        let reply = format!("forgot memory {id}");
        // Invalidate before the write — see `remember`.
        self.cache.invalidate();
        blocking(move || Client::connect(&socket)?.forget(&id)).await?;
        Ok(CallToolResult::success(vec![Content::text(reply)]))
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
    blocking_with_timeout(f, CALL_TIMEOUT).await
}

/// Cap on concurrently-running daemon calls (see [`CALL_PERMITS`]).
const MAX_INFLIGHT_CALLS: usize = 16;

/// Bounds the threads a wedged daemon can leak. The sync client has no read
/// deadline and tokio cannot cancel a blocking task, so a timed-out call's thread
/// keeps running until the daemon call actually returns. Each call holds a permit
/// *inside* the blocking closure — released when the thread truly finishes, not at
/// timeout — so at most [`MAX_INFLIGHT_CALLS`] threads can be stuck at once; further
/// calls then fail fast at the timeout instead of draining tokio's blocking pool
/// (default 512) and starving unrelated work.
static CALL_PERMITS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(MAX_INFLIGHT_CALLS);

async fn blocking_with_timeout<T, F>(f: F, timeout: Duration) -> Result<T, ErrorData>
where
    F: FnOnce() -> std::io::Result<T> + Send + 'static,
    T: Send + 'static,
{
    let call = async {
        let permit = CALL_PERMITS
            .acquire()
            .await
            .expect("CALL_PERMITS is never closed");
        tokio::task::spawn_blocking(move || {
            let out = f();
            drop(permit); // the thread is done — only now is the slot free again
            out
        })
        .await
    };
    tokio::time::timeout(timeout, call)
        .await
        .map_err(|_| {
            ErrorData::internal_error(
                format!("memeora daemon call timed out after {timeout:?}"),
                None,
            )
        })?
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?
        .map_err(map_io_err)
}

/// Map a client I/O error to an MCP error, giving an actionable hint when the
/// daemon is simply unreachable rather than a generic internal error.
fn map_io_err(e: std::io::Error) -> ErrorData {
    use std::io::ErrorKind::*;
    match e.kind() {
        ConnectionRefused | NotFound | ConnectionReset | BrokenPipe => ErrorData::internal_error(
            format!(
                "memeora daemon unreachable ({e}); is `memeora-daemon` running and does `MEMEORA_SOCKET` point at its socket?"
            ),
            None,
        ),
        _ => ErrorData::internal_error(e.to_string(), None),
    }
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
            resolve_scope(Some("repo_memeora".into()), "default_tag"),
            "repo_memeora".to_string()
        );
    }

    #[test]
    fn missing_or_blank_scope_uses_default() {
        assert_eq!(resolve_scope(None, "default_tag"), "default_tag");
        // Blank strings resolve the same way as a missing scope.
        assert_eq!(
            resolve_scope(Some("   ".into()), "default_tag"),
            "default_tag"
        );
    }

    #[test]
    fn default_scope_is_never_an_empty_bucket() {
        // Whatever the environment, the default is a concrete, non-empty tag.
        let s = default_scope();
        assert!(!s.is_empty());
        assert!(s.starts_with("memeora_project_"));
    }

    #[test]
    fn cache_dedups_identical_recalls_and_refetches_after_a_write() {
        let cache = RecallCache::new(Duration::from_secs(30), 64);
        let daemon_calls = std::cell::Cell::new(0);
        // Mirrors the `recall` tool's wiring: cached text or fetch-and-insert.
        let lookup = |query: &str| {
            let key = cache.key("recall", "s", query, 10, None);
            cache.get(&key).unwrap_or_else(|| {
                daemon_calls.set(daemon_calls.get() + 1);
                let text = "- [fact] we use sqlite".to_string();
                cache.insert(key, text.clone());
                text
            })
        };

        lookup("What DB do we use?");
        // Identical modulo whitespace/case: served from cache, daemon untouched.
        lookup("  what   DB do we USE?  ");
        assert_eq!(daemon_calls.get(), 1);

        // A `remember`/`forget` invalidates: the same recall must re-fetch.
        cache.invalidate();
        lookup("What DB do we use?");
        assert_eq!(daemon_calls.get(), 2);
    }

    #[test]
    fn cache_entries_expire_and_the_oldest_is_evicted() {
        // TTL zero: nothing is ever fresh enough to serve.
        let cache = RecallCache::new(Duration::ZERO, 64);
        let key = cache.key("recall", "s", "q", 10, None);
        cache.insert(key.clone(), "text".into());
        assert_eq!(cache.get(&key), None);

        // Capacity bound: the oldest entry goes first.
        let cache = RecallCache::new(Duration::from_secs(30), 2);
        let k1 = cache.key("recall", "s", "q1", 10, None);
        let k2 = cache.key("recall", "s", "q2", 10, None);
        let k3 = cache.key("recall", "s", "q3", 10, None);
        cache.insert(k1.clone(), "1".into());
        cache.insert(k2.clone(), "2".into());
        cache.insert(k3.clone(), "3".into());
        assert_eq!(cache.get(&k1), None);
        assert_eq!(cache.get(&k2), Some("2".into()));
        assert_eq!(cache.get(&k3), Some("3".into()));
    }

    #[test]
    fn cache_keys_distinguish_tool_scope_and_params() {
        let cache = RecallCache::new(Duration::from_secs(30), 64);
        let base = cache.key("recall", "s", "q", 10, None);
        assert_ne!(base, cache.key("context", "s", "q", 10, None));
        assert_ne!(base, cache.key("recall", "other", "q", 10, None));
        assert_ne!(base, cache.key("recall", "s", "q2", 10, None));
        assert_ne!(base, cache.key("recall", "s", "q", 5, None));
        assert_ne!(base, cache.key("recall", "s", "q", 10, Some(100)));
        // Normalization folds whitespace/case variants into the same key.
        assert_eq!(base, cache.key("recall", "s", " Q  ", 10, None));
    }

    #[tokio::test]
    async fn blocking_call_times_out() {
        let err = blocking_with_timeout(
            || {
                std::thread::sleep(Duration::from_millis(50));
                Ok::<(), std::io::Error>(())
            },
            Duration::from_millis(1),
        )
        .await
        .unwrap_err();
        assert!(format!("{err:?}").contains("timed out"));
    }

    #[tokio::test]
    async fn timed_out_call_releases_its_permit_when_the_thread_finishes() {
        // A timed-out call keeps its blocking thread (and permit) only until the
        // underlying call returns — the leak is bounded, not permanent.
        let _ = blocking_with_timeout(
            || {
                std::thread::sleep(Duration::from_millis(30));
                Ok::<(), std::io::Error>(())
            },
            Duration::from_millis(1),
        )
        .await;
        // Fast calls still succeed while the abandoned thread runs (permits remain).
        blocking_with_timeout(|| Ok::<(), std::io::Error>(()), Duration::from_secs(5))
            .await
            .unwrap();
        // Once the abandoned thread finishes, all permits are back (poll: other
        // tests in this process may briefly hold permits too).
        for _ in 0..200 {
            if CALL_PERMITS.available_permits() == MAX_INFLIGHT_CALLS {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("permit was never released after the blocking call finished");
    }
}
