//! Rust client SDK for the memeora daemon.
//!
//! [`Client`] opens a local-socket connection and exposes one typed method per
//! IPC verb. Each call frames a [`Request`], reads the framed [`Response`], and
//! maps [`Response::Error`] / unexpected variants to an [`io::Error`]. The wire
//! framing and message types come from [`memeora_proto`].

use std::io::{self, BufReader};

use interprocess::local_socket::{Stream, prelude::*};
use memeora_proto::{
    DEFAULT_SOCKET, MemoryDto, PROTOCOL_VERSION, Request, Response, build_name, frame,
};

/// A connected client to a memeora daemon.
pub struct Client {
    conn: BufReader<Stream>,
}

impl Client {
    /// Connect to the default daemon socket ([`DEFAULT_SOCKET`]).
    pub fn connect_default() -> io::Result<Self> {
        Self::connect(DEFAULT_SOCKET)
    }

    /// Connect to a daemon on a specific socket name/path.
    ///
    /// Performs the protocol handshake and fails with [`io::ErrorKind::Unsupported`]
    /// if the daemon speaks a different [`PROTOCOL_VERSION`], so a version skew is a
    /// clear error here rather than an opaque deserialization failure later.
    pub fn connect(socket: &str) -> io::Result<Self> {
        let stream = Stream::connect(build_name(socket)?)?;
        let mut client = Client {
            conn: BufReader::new(stream),
        };
        let daemon_version = client.hello()?;
        if daemon_version != PROTOCOL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "protocol version mismatch: client speaks v{PROTOCOL_VERSION}, daemon speaks v{daemon_version}"
                ),
            ));
        }
        Ok(client)
    }

    /// Send one request and read its response.
    fn call(&mut self, request: &Request) -> io::Result<Response> {
        frame::write_message(self.conn.get_mut(), request)?;
        frame::read_message(&mut self.conn)?
            .ok_or_else(|| io::Error::other("daemon closed the connection"))
    }

    /// Handshake; returns the daemon's protocol version.
    pub fn hello(&mut self) -> io::Result<u32> {
        match self.call(&Request::Hello {
            protocol_version: PROTOCOL_VERSION,
        })? {
            Response::Hello {
                protocol_version, ..
            } => Ok(protocol_version),
            other => Err(unexpected(other)),
        }
    }

    /// Ingest raw text; returns `(added, reinforced)` counts.
    pub fn ingest(&mut self, scope: &str, text: &str) -> io::Result<(usize, usize)> {
        match self.call(&Request::Ingest {
            scope: scope.to_string(),
            text: text.to_string(),
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

    /// Hybrid search within a scope.
    pub fn recall(&mut self, scope: &str, query: &str, k: usize) -> io::Result<Vec<MemoryDto>> {
        match self.call(&Request::Recall {
            scope: scope.to_string(),
            query: query.to_string(),
            k,
        })? {
            Response::Memories { memories } => Ok(memories),
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
    use memeora_daemon::{Engine, serve};
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
        thread::spawn(move || serve(engine, socket).unwrap());
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
}
