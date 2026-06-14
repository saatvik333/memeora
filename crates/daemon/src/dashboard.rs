//! The local dashboard: an `axum` HTTP server, served by the daemon, exposing a
//! read-mostly JSON API + an SSE live stream + the embedded Svelte/Sigma.js UI.
//!
//! Access split (so the daemon stays the sole DB writer):
//! - **Reads** (`scopes`, `graph`, `list`, `context`) go through a *second*,
//!   read-only [`SqliteStore`] connection on the same WAL database — concurrent
//!   readers never block the writer.
//! - **Search** and **forget** go back through the daemon's own IPC socket as an
//!   ordinary [`memeora_client::Client`]: search needs the daemon's embedder, and
//!   forget must go through the single writer (and its profile invalidation).
//! - **Live mode** subscribes to the engine's [`ChangeEvent`] broadcast and
//!   forwards lightweight `{scope, op}` events over SSE; the browser refetches.
//!
//! Binds to `127.0.0.1` only — it's a local tool with no auth and no network
//! exposure (see `docs/ARCHITECTURE.md`).

use std::convert::Infallible;
use std::fmt::Display;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::Router;
use axum::extract::{Query, State};
use axum::http::{StatusCode, Uri, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use memeora_core::{Memory, ProfileParams, ScopeInfo, SqliteStore, VectorStore, build_profile};
use memeora_proto::MemoryDto;
use rust_embed::Embed;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::engine::ChangeEvent;

/// Embedded dashboard UI (built Svelte app). See `build.rs` — the folder always
/// exists at compile time, so this builds even when the frontend wasn't built.
#[derive(Embed)]
#[folder = "../../dashboard/dist"]
struct Assets;

/// Default graph node cap, so a huge scope can't render the browser unusable.
const DEFAULT_GRAPH_CAP: usize = 2000;

/// Shared dashboard state. Cheap to clone (everything is `Arc`/`Sender`).
#[derive(Clone)]
struct AppState {
    /// Shared read-only store — opened once at startup so we don't pay the
    /// connection + migration overhead on every request.  `Mutex` is needed
    /// because `rusqlite::Connection` is `Send` but not `Sync`; WAL mode
    /// allows this reader to run concurrently with the daemon's writer.
    store: Arc<Mutex<SqliteStore>>,
    /// The daemon's IPC socket — for search (needs the embedder) and forget.
    socket: Arc<str>,
    /// Source of [`ChangeEvent`]s for the SSE live stream.
    events: broadcast::Sender<ChangeEvent>,
    /// Max nodes returned by the graph endpoint.
    graph_cap: usize,
}

/// Build the dashboard router over the given state pieces. Split from [`serve`]
/// so it can be exercised in tests without binding a socket.
///
/// Returns `Err` if the read-only store cannot be opened (e.g. DB not yet
/// created by the writer).
fn build_router(
    db_path: PathBuf,
    dim: usize,
    socket: String,
    events: broadcast::Sender<ChangeEvent>,
    graph_cap: usize,
) -> Result<Router, memeora_core::Error> {
    let store = SqliteStore::open_readonly(&db_path, dim)?;
    let state = AppState {
        store: Arc::new(Mutex::new(store)),
        socket: Arc::from(socket),
        events,
        graph_cap,
    };
    Ok(Router::new()
        .route("/api/health", get(health))
        .route("/api/scopes", get(scopes))
        .route("/api/graph", get(graph))
        .route("/api/list", get(list))
        .route("/api/context", get(context))
        .route("/api/search", get(search))
        .route("/api/forget", post(forget))
        .route("/api/events", get(events_stream))
        // Everything else: embedded UI assets, with SPA fallback to index.html.
        .fallback(static_asset)
        .with_state(state))
}

/// Serve the dashboard on `addr` until the process exits. `db_path` is the WAL
/// database path; `socket` is the daemon's IPC socket and `events` is the
/// engine's change broadcaster.
pub async fn serve(
    addr: SocketAddr,
    db_path: PathBuf,
    dim: usize,
    socket: String,
    events: broadcast::Sender<ChangeEvent>,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = build_router(db_path, dim, socket, events, DEFAULT_GRAPH_CAP)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ---- handlers ---------------------------------------------------------------

async fn health() -> Json<Health> {
    Json(Health {
        ok: true,
        server_version: env!("CARGO_PKG_VERSION"),
        protocol_version: memeora_proto::PROTOCOL_VERSION,
    })
}

async fn scopes(State(st): State<AppState>) -> Result<Json<Vec<ScopeDto>>, ApiError> {
    let scopes = read(&st, |s| s.list_scopes()).await?;
    Ok(Json(scopes.iter().map(ScopeDto::from).collect()))
}

async fn graph(
    State(st): State<AppState>,
    Query(q): Query<ScopeOnly>,
) -> Result<Json<GraphDto>, ApiError> {
    let cap = q.cap.unwrap_or(st.graph_cap).min(st.graph_cap);
    let scope = q.scope.clone();
    let g = read(&st, move |s| s.graph(&scope, cap)).await?;
    Ok(Json(GraphDto {
        scope: q.scope,
        nodes: g.nodes.iter().map(NodeDto::from).collect(),
        edges: g
            .edges
            .iter()
            .map(|e| EdgeDto {
                source: e.from_id.clone(),
                target: e.to_id.clone(),
                kind: e.kind.as_str(),
                created_at: e.created_at,
            })
            .collect(),
    }))
}

async fn list(
    State(st): State<AppState>,
    Query(q): Query<ListQuery>,
) -> Result<Json<Vec<MemDto>>, ApiError> {
    let scope = q.scope;
    let limit = q.limit;
    let memories = read(&st, move |s| s.list_latest(&scope, limit)).await?;
    Ok(Json(memories.iter().map(MemDto::from_memory).collect()))
}

async fn context(
    State(st): State<AppState>,
    Query(q): Query<ScopeOnly>,
) -> Result<Json<ContextDto>, ApiError> {
    let scope = q.scope;
    let profile = read(&st, move |s| {
        build_profile(s as &dyn VectorStore, &scope, &ProfileParams::default())
    })
    .await?;
    Ok(Json(ContextDto {
        statics: profile.statics.iter().map(MemDto::from_memory).collect(),
        dynamics: profile.dynamics.iter().map(MemDto::from_memory).collect(),
    }))
}

async fn search(
    State(st): State<AppState>,
    Query(q): Query<SearchQuery>,
) -> Result<Json<Vec<MemDto>>, ApiError> {
    // Search needs the daemon's embedder → go back through the IPC socket.
    let socket = st.socket.clone();
    let hits = tokio::task::spawn_blocking(move || {
        let mut client = memeora_client::Client::connect(&socket)?;
        client.recall(&q.scope, &q.q, q.k)
    })
    .await
    .map_err(ApiError::internal)?
    .map_err(ApiError::upstream)?;
    Ok(Json(hits.iter().map(MemDto::from_dto).collect()))
}

async fn forget(
    State(st): State<AppState>,
    Json(body): Json<ForgetBody>,
) -> Result<Json<Ack>, ApiError> {
    // Forget must go through the single writer → IPC socket, not the read conn.
    let socket = st.socket.clone();
    tokio::task::spawn_blocking(move || {
        let mut client = memeora_client::Client::connect(&socket)?;
        client.forget(&body.id)
    })
    .await
    .map_err(ApiError::internal)?
    .map_err(ApiError::upstream)?;
    Ok(Json(Ack { ok: true }))
}

/// Live stream: forwards each [`ChangeEvent`] as an SSE `change` event. Lagged
/// receivers (a slow browser) just drop the missed events rather than error out.
async fn events_stream(
    State(st): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = st.events.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| {
        let ev = res.ok()?;
        let dto = ChangeDto {
            scope: ev.scope,
            op: ev.op,
        };
        let event = Event::default()
            .event("change")
            .json_data(dto)
            .unwrap_or_else(|_| Event::default());
        Some(Ok(event))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Serve an embedded UI asset, falling back to `index.html` for unknown paths so
/// the single-page app's client-side routing works.
async fn static_asset(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    if let Some(file) = Assets::get(path) {
        let mime = file.metadata.mimetype().to_string();
        return ([(header::CONTENT_TYPE, mime)], file.data.into_owned()).into_response();
    }
    match Assets::get("index.html") {
        Some(file) => (
            [(header::CONTENT_TYPE, "text/html".to_string())],
            file.data.into_owned(),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "dashboard UI not built").into_response(),
    }
}

/// Run a read-only store query on a blocking thread, reusing the shared store
/// opened at startup so we don't pay per-request connection + migration cost.
async fn read<T, F>(st: &AppState, f: F) -> Result<T, ApiError>
where
    F: FnOnce(&SqliteStore) -> memeora_core::Result<T> + Send + 'static,
    T: Send + 'static,
{
    let store = Arc::clone(&st.store);
    tokio::task::spawn_blocking(move || {
        let guard = store
            .lock()
            .map_err(|_| ApiError::internal("store lock poisoned"))?;
        f(&guard).map_err(ApiError::internal)
    })
    .await
    .map_err(ApiError::internal)?
}

// ---- DTOs -------------------------------------------------------------------

#[derive(Serialize)]
struct Health {
    ok: bool,
    server_version: &'static str,
    protocol_version: u32,
}

#[derive(Serialize)]
struct ScopeDto {
    tag: String,
    latest: usize,
    total: usize,
}

impl From<&ScopeInfo> for ScopeDto {
    fn from(s: &ScopeInfo) -> Self {
        ScopeDto {
            tag: s.tag.clone(),
            latest: s.latest,
            total: s.total,
        }
    }
}

#[derive(Serialize)]
struct GraphDto {
    scope: String,
    nodes: Vec<NodeDto>,
    edges: Vec<EdgeDto>,
}

#[derive(Serialize)]
struct NodeDto {
    id: String,
    content: String,
    kind: &'static str,
    strength: f32,
    is_latest: bool,
    created_at: i64,
    last_accessed_at: i64,
    expires_at: Option<i64>,
}

impl From<&Memory> for NodeDto {
    fn from(m: &Memory) -> Self {
        NodeDto {
            id: m.id.clone(),
            content: m.content.clone(),
            kind: m.kind.as_str(),
            strength: m.strength,
            is_latest: m.is_latest,
            created_at: m.created_at,
            last_accessed_at: m.last_accessed_at,
            expires_at: m.expires_at,
        }
    }
}

#[derive(Serialize)]
struct EdgeDto {
    source: String,
    target: String,
    kind: &'static str,
    created_at: i64,
}

/// A memory in a flat list/search/context response (with an optional relevance score).
#[derive(Serialize)]
struct MemDto {
    id: String,
    content: String,
    kind: String,
    strength: f32,
    created_at: i64,
    score: Option<f32>,
}

impl MemDto {
    fn from_memory(m: &Memory) -> Self {
        MemDto {
            id: m.id.clone(),
            content: m.content.clone(),
            kind: m.kind.as_str().to_string(),
            strength: m.strength,
            created_at: m.created_at,
            score: None,
        }
    }

    fn from_dto(m: &MemoryDto) -> Self {
        MemDto {
            id: m.id.clone(),
            content: m.content.clone(),
            kind: m.kind.clone(),
            strength: m.strength,
            created_at: m.created_at,
            score: m.score,
        }
    }
}

#[derive(Serialize)]
struct ContextDto {
    statics: Vec<MemDto>,
    dynamics: Vec<MemDto>,
}

#[derive(Serialize)]
struct ChangeDto {
    scope: String,
    op: &'static str,
}

#[derive(Serialize)]
struct Ack {
    ok: bool,
}

// ---- query / body params ----------------------------------------------------

#[derive(Deserialize)]
struct ScopeOnly {
    scope: String,
    #[serde(default)]
    cap: Option<usize>,
}

#[derive(Deserialize)]
struct ListQuery {
    scope: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

#[derive(Deserialize)]
struct SearchQuery {
    scope: String,
    q: String,
    #[serde(default = "default_k")]
    k: usize,
}

#[derive(Deserialize)]
struct ForgetBody {
    id: String,
}

fn default_limit() -> usize {
    50
}

fn default_k() -> usize {
    10
}

// ---- errors -----------------------------------------------------------------

/// An API error rendered as an HTTP status + plain-text message.
struct ApiError(StatusCode, String);

impl ApiError {
    /// An internal failure (a panicked task, a DB error).
    ///
    /// The full error is logged to stderr; only a generic message is sent to
    /// the client so internal details (file paths, SQL, stack hints) are not
    /// exposed over HTTP.
    fn internal(e: impl Display) -> Self {
        eprintln!("memeora-dashboard: internal error: {e}");
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal error".to_string(),
        )
    }

    /// A failure talking to the daemon over IPC (e.g. it isn't running).
    ///
    /// As above: log the detail, send a generic body.
    fn upstream(e: impl Display) -> Self {
        eprintln!("memeora-dashboard: upstream error: {e}");
        ApiError(StatusCode::BAD_GATEWAY, "upstream error".to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use memeora_core::{EdgeKind, MemoryKind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt; // for `oneshot`

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn mem(id: &str, content: &str, tag: &str) -> Memory {
        Memory::new(id, content, MemoryKind::Fact, tag, vec![1.0, 0.0])
    }

    fn test_store() -> (PathBuf, SqliteStore) {
        let counter = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "memeora-dashboard-test-{}-{}.db",
            std::process::id(),
            counter
        ));
        let mut store = SqliteStore::open(&path, 2).unwrap();
        store.upsert(&mem("a", "alpha", "tag_a")).unwrap();
        store.upsert(&mem("b", "beta", "tag_a")).unwrap();
        store.add_edge("a", "b", EdgeKind::Extends).unwrap();
        store.upsert(&mem("c", "gamma", "tag_b")).unwrap();
        (path, store)
    }

    fn test_app() -> (Router, PathBuf) {
        let (path, store) = test_store();
        let dim = store.dim();
        // Drop the write store before opening the read-only one in build_router.
        drop(store);
        let (tx, _) = broadcast::channel(16);
        (
            build_router(
                path.clone(),
                dim,
                "unused.sock".to_string(),
                tx,
                DEFAULT_GRAPH_CAP,
            )
            .unwrap(),
            path,
        )
    }

    async fn body_string(resp: Response) -> String {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn scopes_endpoint_lists_tags_with_counts() {
        let (app, _db_path) = test_app();
        let resp = app
            .oneshot(Request::get("/api/scopes").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_string(resp).await;
        assert!(body.contains("tag_a"), "got: {body}");
        assert!(body.contains("tag_b"), "got: {body}");
    }

    #[tokio::test]
    async fn graph_endpoint_returns_nodes_and_edges() {
        let (app, _db_path) = test_app();
        let resp = app
            .oneshot(
                Request::get("/api/graph?scope=tag_a")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let g: GraphDtoOwned = serde_json::from_str(&body_string(resp).await).unwrap();
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.edges[0].source, "a");
        assert_eq!(g.edges[0].target, "b");
    }

    #[tokio::test]
    async fn unknown_route_falls_back_to_index_html() {
        // The build.rs placeholder guarantees index.html exists, so an unknown SPA
        // route serves HTML (200) rather than 404.
        let (app, _db_path) = test_app();
        let resp = app
            .oneshot(Request::get("/some/spa/route").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // Owned mirrors of the response DTOs so tests can deserialize them.
    #[derive(Deserialize)]
    struct GraphDtoOwned {
        nodes: Vec<serde_json::Value>,
        edges: Vec<EdgeOwned>,
    }
    #[derive(Deserialize)]
    struct EdgeOwned {
        source: String,
        target: String,
    }
}
