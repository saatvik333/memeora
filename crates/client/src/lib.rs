//! Rust client SDK for the memeora daemon.
//!
//! [`Client`] opens a local-socket connection and exposes one typed method per
//! IPC verb. Each call frames a [`Request`], reads the framed [`Response`], and
//! maps [`Response::Error`] / unexpected variants to an [`io::Error`]. The wire
//! framing and message types come from [`memeora_proto`].

use std::io::{self, BufReader};
use std::time::Duration;

use interprocess::ConnectWaitMode;
use interprocess::local_socket::{ConnectOptions, Stream, prelude::*};
use memeora_proto::{MemoryDto, PROTOCOL_VERSION, Request, Response, build_name, frame};

/// How long [`Client::connect`] waits for the daemon to accept the connection.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Per-operation socket deadline for reads and writes. A wedged daemon surfaces
/// as a prompt [`io::ErrorKind::TimedOut`] error instead of blocking the caller
/// forever (fail-open: the caller can degrade to "no memory" and move on).
const IO_TIMEOUT: Duration = Duration::from_secs(30);

/// A connected client to a memeora daemon.
pub struct Client {
    conn: BufReader<Stream>,
    /// Daemon crate version, captured during the connect handshake.
    server_version: String,
    /// Capabilities the daemon advertised at connect (see [`memeora_proto::capability`]).
    capabilities: Vec<String>,
}

impl Client {
    /// Connect to the default per-user daemon socket.
    pub fn connect_default() -> io::Result<Self> {
        Self::connect(&memeora_proto::resolve_socket(None))
    }

    /// Connect to a daemon on a specific socket name/path.
    ///
    /// Performs the protocol handshake and fails with [`io::ErrorKind::Unsupported`]
    /// if the daemon speaks a different [`PROTOCOL_VERSION`], so a version skew is a
    /// clear error here rather than an opaque deserialization failure later.
    ///
    /// Connecting is bounded by [`CONNECT_TIMEOUT`] and every subsequent read/write
    /// (including the handshake) by [`IO_TIMEOUT`]; a hung daemon yields a
    /// [`io::ErrorKind::TimedOut`] error rather than blocking forever.
    pub fn connect(socket: &str) -> io::Result<Self> {
        Self::connect_with_deadlines(socket, CONNECT_TIMEOUT, IO_TIMEOUT)
    }

    /// [`Self::connect`] with explicit deadlines (both must be nonzero).
    fn connect_with_deadlines(
        socket: &str,
        connect_timeout: Duration,
        io_timeout: Duration,
    ) -> io::Result<Self> {
        let stream: Stream = ConnectOptions::new()
            .name(build_name(socket)?)
            .wait_mode(ConnectWaitMode::Timeout(connect_timeout))
            .connect_sync()?;
        stream.set_recv_timeout(Some(io_timeout))?;
        stream.set_send_timeout(Some(io_timeout))?;
        let mut client = Client {
            conn: BufReader::new(stream),
            server_version: String::new(),
            capabilities: Vec::new(),
        };
        let (daemon_version, server_version, capabilities) = client.handshake()?;
        if daemon_version != PROTOCOL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "protocol version mismatch: client speaks v{PROTOCOL_VERSION}, daemon speaks v{daemon_version}"
                ),
            ));
        }
        client.server_version = server_version;
        client.capabilities = capabilities;
        Ok(client)
    }

    /// Send one request and read its response.
    ///
    /// A [`TimedOut`](io::ErrorKind::TimedOut) error means the daemon missed the
    /// I/O deadline; the stream may hold a partial frame afterwards, so drop this
    /// client and reconnect rather than retrying the call on it.
    fn call(&mut self, request: &Request) -> io::Result<Response> {
        frame::write_message(self.conn.get_mut(), request).map_err(deadline_err)?;
        frame::read_message(&mut self.conn)
            .map_err(deadline_err)?
            .ok_or_else(|| io::Error::other("daemon closed the connection"))
    }

    /// Perform the handshake, returning `(protocol_version, server_version, capabilities)`.
    fn handshake(&mut self) -> io::Result<(u32, String, Vec<String>)> {
        match self.call(&Request::Hello {
            protocol_version: PROTOCOL_VERSION,
        })? {
            Response::Hello {
                protocol_version,
                server_version,
                capabilities,
            } => Ok((protocol_version, server_version, capabilities)),
            other => Err(unexpected(other)),
        }
    }

    /// Handshake; returns the daemon's protocol version.
    pub fn hello(&mut self) -> io::Result<u32> {
        Ok(self.handshake()?.0)
    }

    /// The daemon's crate version, captured at connect.
    pub fn server_version(&self) -> &str {
        &self.server_version
    }

    /// Capabilities the daemon advertised at connect (see [`memeora_proto::capability`]).
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    /// Whether the connected daemon advertised support for `capability`.
    pub fn supports(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }

    fn require_capability(&self, capability: &str) -> io::Result<()> {
        self.supports(capability).then_some(()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                format!("daemon does not support capability {capability}"),
            )
        })
    }

    /// Ingest raw text; returns `(added, reinforced)` counts.
    pub fn ingest(&mut self, scope: &str, text: &str) -> io::Result<(usize, usize)> {
        self.ingest_from(scope, text, None)
    }

    /// Ingest raw text attributed to `source` (an agent/session id), so repeated
    /// corroboration from the same source can't inflate a memory's proof. Gate on the
    /// `evidence` capability; pass `None` for the unattributed default.
    pub fn ingest_from(
        &mut self,
        scope: &str,
        text: &str,
        source: Option<&str>,
    ) -> io::Result<(usize, usize)> {
        match self.call(&Request::Ingest {
            scope: scope.to_string(),
            text: text.to_string(),
            source: source.map(str::to_string),
        })? {
            Response::Ingested { added, reinforced } => Ok((added, reinforced)),
            other => Err(unexpected(other)),
        }
    }

    /// Add a single explicit memory; returns its id.
    pub fn add(&mut self, scope: &str, content: &str, kind: &str) -> io::Result<String> {
        match self.call(&Request::Add {
            scope: scope.to_string(),
            content: content.to_string(),
            kind: kind.to_string(),
        })? {
            Response::Added { id } => Ok(id),
            other => Err(unexpected(other)),
        }
    }

    /// Hybrid search within a scope (plain top-`k`, no token budget).
    pub fn recall(&mut self, scope: &str, query: &str, k: usize) -> io::Result<Vec<MemoryDto>> {
        self.recall_within(scope, query, k, None)
    }

    /// Hybrid search with an optional token budget: when `max_tokens` is set, the daemon
    /// fills results best-first up to that many estimated tokens (still capped at `k`).
    pub fn recall_within(
        &mut self,
        scope: &str,
        query: &str,
        k: usize,
        max_tokens: Option<usize>,
    ) -> io::Result<Vec<MemoryDto>> {
        match self.call(&Request::Recall {
            scope: scope.to_string(),
            query: query.to_string(),
            k,
            max_tokens,
        })? {
            Response::Memories { memories } => Ok(memories),
            other => Err(unexpected(other)),
        }
    }

    /// Single-call context bundle: the scope's profile (statics, dynamics) **and** the
    /// query's recall hits, in one round-trip. Returns `(statics, dynamics, memories)`,
    /// deduped by id (a profile memory never reappears in `memories`). Gate on the
    /// `bundle` capability; `max_tokens` budgets the recall portion like [`Self::recall_within`].
    pub fn bundle(
        &mut self,
        scope: &str,
        query: &str,
        k: usize,
        max_tokens: Option<usize>,
    ) -> io::Result<(Vec<MemoryDto>, Vec<MemoryDto>, Vec<MemoryDto>)> {
        self.require_capability(memeora_proto::capability::BUNDLE)?;
        match self.call(&Request::Bundle {
            scope: scope.to_string(),
            query: query.to_string(),
            k,
            max_tokens,
        })? {
            Response::Bundle {
                statics,
                dynamics,
                memories,
            } => Ok((statics, dynamics, memories)),
            other => Err(unexpected(other)),
        }
    }

    /// Fetch the profile (static facts/prefs, dynamic episodes) for a scope.
    pub fn context(&mut self, scope: &str) -> io::Result<(Vec<MemoryDto>, Vec<MemoryDto>)> {
        match self.call(&Request::Context {
            scope: scope.to_string(),
        })? {
            Response::Context { statics, dynamics } => Ok((statics, dynamics)),
            other => Err(unexpected(other)),
        }
    }

    /// List the latest memories in a scope.
    pub fn list(&mut self, scope: &str, limit: usize) -> io::Result<Vec<MemoryDto>> {
        match self.call(&Request::List {
            scope: scope.to_string(),
            limit,
        })? {
            Response::Memories { memories } => Ok(memories),
            other => Err(unexpected(other)),
        }
    }

    /// Soft-forget a memory by id.
    pub fn forget(&mut self, id: &str) -> io::Result<()> {
        match self.call(&Request::Forget { id: id.to_string() })? {
            Response::Forgotten => Ok(()),
            other => Err(unexpected(other)),
        }
    }

    /// Consolidate a scope: distil its near-duplicate memories into distinct-source-proofed
    /// observations. Returns `(observations, sources_linked)`. Idempotent — re-running
    /// converges. Gate on the `consolidate` capability.
    pub fn consolidate(&mut self, scope: &str) -> io::Result<(usize, usize)> {
        self.require_capability(memeora_proto::capability::CONSOLIDATE)?;
        match self.call(&Request::Consolidate {
            scope: scope.to_string(),
        })? {
            Response::Consolidated {
                observations,
                sources_linked,
            } => Ok((observations, sources_linked)),
            other => Err(unexpected(other)),
        }
    }
}

/// Normalize a socket-deadline expiry into a clear [`io::ErrorKind::TimedOut`]
/// error. On Unix, `SO_RCVTIMEO`/`SO_SNDTIMEO` expiry surfaces as `WouldBlock`
/// (EAGAIN); Windows named pipes report `TimedOut` — callers see one kind.
fn deadline_err(e: io::Error) -> io::Error {
    match e.kind() {
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => io::Error::new(
            io::ErrorKind::TimedOut,
            format!("memeora daemon did not respond within the I/O deadline: {e}"),
        ),
        _ => e,
    }
}

/// Map an error / unexpected response to an [`io::Error`].
fn unexpected(response: Response) -> io::Error {
    match response {
        Response::Error { message } => io::Error::other(message),
        other => io::Error::other(format!("unexpected daemon response: {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memeora_core::{EmbeddingProvider, EmbeddingSpace, HeuristicExtractor, SqliteStore};
    use memeora_daemon::{Engine, bind, serve};
    use std::thread;
    use std::time::Duration;

    struct LenEmbedder(EmbeddingSpace);
    impl EmbeddingProvider for LenEmbedder {
        fn space(&self) -> &EmbeddingSpace {
            &self.0
        }
        fn embed_documents(&self, texts: &[&str]) -> memeora_core::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32, 0.0, 1.0])
                .collect())
        }
    }

    fn start_server(socket: &'static str) {
        let engine = Engine::new(
            SqliteStore::open_in_memory(3).unwrap(),
            Box::new(LenEmbedder(EmbeddingSpace::new("mock", "len", 3))),
            Box::new(HeuristicExtractor::default()),
        );
        thread::spawn(move || serve(engine, bind(socket).unwrap()).unwrap());
    }

    fn connect_retry(socket: &str) -> Client {
        for _ in 0..200 {
            if let Ok(c) = Client::connect(socket) {
                return c;
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("could not connect");
    }

    #[test]
    fn typed_client_roundtrip() {
        let socket = "memeora-test-client-roundtrip.sock";
        start_server(socket);
        let mut client = connect_retry(socket);

        assert_eq!(client.hello().unwrap(), PROTOCOL_VERSION);
        // The connect handshake captured the daemon's capabilities.
        assert!(client.supports(memeora_proto::capability::RECALL));
        assert!(!client.supports("nonexistent-capability"));
        assert!(!client.server_version().is_empty());

        let id = client.add("s", "I prefer dark mode", "preference").unwrap();
        assert!(!id.is_empty());

        let hits = client.recall("s", "I prefer dark mode", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id);

        let (statics, dynamics) = client.context("s").unwrap();
        assert_eq!(statics.len(), 1);
        assert_eq!(dynamics.len(), 0);

        client.forget(&id).unwrap();
        assert!(client.list("s", 10).unwrap().is_empty());
    }

    #[test]
    fn ingest_counts_returned() {
        let socket = "memeora-test-client-ingest.sock";
        start_server(socket);
        let mut client = connect_retry(socket);
        let (added, reinforced) = client.ingest("s", "I prefer rust. We use SQLite.").unwrap();
        assert_eq!(added, 2);
        assert_eq!(reinforced, 0);
    }

    #[test]
    fn bundle_returns_profile_and_recall() {
        let socket = "memeora-test-client-bundle.sock";
        start_server(socket);
        let mut client = connect_retry(socket);
        // The daemon advertises the bundle capability.
        assert!(client.supports(memeora_proto::capability::BUNDLE));

        let pref = client.add("s", "I prefer dark mode", "preference").unwrap();
        client
            .add("s", "deployed the app today", "episode")
            .unwrap();

        let (statics, dynamics, memories) = client.bundle("s", "dark mode", 5, None).unwrap();
        assert_eq!(statics.len(), 1);
        assert_eq!(statics[0].id, pref);
        assert_eq!(dynamics.len(), 1);
        assert_eq!(dynamics[0].kind, "episode");
        // Whatever recall surfaces, the profiled preference is never duplicated.
        assert!(memories.iter().all(|m| m.id != pref));
    }

    #[test]
    fn connect_to_missing_socket_fails_promptly() {
        let start = std::time::Instant::now();
        Client::connect("/tmp/memeora-test-no-such-daemon.sock")
            .err()
            .expect("connecting without a daemon must fail");
        // No daemon means a prompt error, never a hang on the connect deadline.
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn read_deadline_fires_on_a_silent_daemon() {
        use interprocess::local_socket::{ListenerOptions, traits::Listener as _};

        let socket = "memeora-test-client-silent.sock";
        let listener = ListenerOptions::new()
            .name(memeora_proto::build_name(socket).unwrap())
            .create_sync()
            .unwrap();
        // Accept the connection and hold it open without ever replying, so the
        // handshake read can only end via the deadline (a drop would be a clean
        // EOF and a different error).
        thread::spawn(move || {
            let conn = listener.accept();
            thread::sleep(Duration::from_secs(3));
            drop(conn);
        });

        let start = std::time::Instant::now();
        let err = Client::connect_with_deadlines(
            socket,
            Duration::from_secs(5),
            Duration::from_millis(100),
        )
        .err()
        .expect("handshake against a silent daemon must fail");
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "handshake read did not respect the I/O deadline"
        );
    }
}
