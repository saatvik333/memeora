//! The blocking IPC server: a writer-actor thread owns the [`Engine`], and a
//! thread-per-connection accept loop frames requests to it.
//!
//! Why blocking threads rather than tokio: the [`Engine`] and its rusqlite store
//! are synchronous, and [`memeora_proto::frame`] is sync `std::io`. A single
//! writer thread owning the `Engine` is the natural "sole DB writer" and needs no
//! locks; connection threads parse/frame off the writer and forward each request to
//! the writer over a channel. This sidesteps blocking a tokio runtime with sync
//! SQLite. (tokio enters later only for the MCP HTTP transport and the dashboard.)
//!
//! Note on parallelism: connection threads run the DB-free framing/extraction
//! concurrently, but embedding serializes through the shared embedder's single ONNX
//! session (see [`memeora_core::embed::fastembed`]). Moving inference off the writer
//! still matters (the writer isn't blocked on it); it just isn't cross-client parallel.

use std::io::{self, BufReader};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender, SyncSender};
use std::thread;
use std::time::Duration;

use interprocess::local_socket::{Listener, ListenerOptions, Stream, prelude::*};
use memeora_proto::{Request, Response, build_name, frame};

use crate::Engine;
use crate::engine::{Prepared, Preparer};

/// Bound on queued-but-unhandled jobs. A full queue applies backpressure
/// (`SyncSender::send` blocks the connection thread) instead of growing unbounded.
const JOB_QUEUE_DEPTH: usize = 1024;

/// Bound on concurrent connection threads, so a flood of clients can't exhaust
/// threads/FDs. Excess connections are dropped (the client can retry).
const MAX_CONNECTIONS: usize = 256;
/// A client must identify itself promptly; this prevents idle local connections
/// from consuming every connection slot indefinitely.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// A framed request cannot hold a server connection forever.
const IO_TIMEOUT: Duration = Duration::from_secs(30);
/// Bound client-controlled result counts before they reach SQLite/KNN queries.
const MAX_RESULTS: usize = 1_000;
const MAX_TOKEN_BUDGET: usize = 100_000;

/// A prepared request plus the channel its response goes back on.
struct Job {
    prepared: Prepared,
    reply: Sender<Response>,
}

/// RAII guard for the live-connection count: decrements on drop so a connection
/// slot is released even if its thread unwinds (see the accept loop).
struct ActiveSlot(Arc<AtomicUsize>);

impl Drop for ActiveSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Spawn the writer-actor: it owns the `Engine` and applies jobs serially.
///
/// Each job is handled under [`catch_unwind`] so a panic in one request degrades
/// to a `Response::Error` for that client instead of killing the writer thread
/// (which would leave the daemon a zombie that accepts but never answers).
fn spawn_writer(mut engine: Engine) -> SyncSender<Job> {
    let (tx, rx) = mpsc::sync_channel::<Job>(JOB_QUEUE_DEPTH);
    thread::spawn(move || {
        for job in rx {
            let response = catch_unwind(AssertUnwindSafe(|| engine.handle_prepared(job.prepared)))
                .unwrap_or_else(|_| Response::Error {
                    message: "internal error: the request handler panicked".to_string(),
                });
            // If the connection is gone, drop the response silently.
            let _ = job.reply.send(response);
        }
    });
    tx
}

/// Acquire the daemon's singleton socket before opening any mutable state.
pub fn bind(socket: &str) -> io::Result<Listener> {
    let already_listening = || {
        io::Error::new(
            io::ErrorKind::AddrInUse,
            format!("a memeora-daemon is already listening on {socket}"),
        )
    };
    // Sole-writer guard: if a daemon already answers on this socket, refuse to
    // start rather than overwrite it and end up with two writers on one DB.
    let live = Stream::connect(build_name(socket)?).is_ok();
    if live {
        return Err(already_listening());
    }

    // Reclaim a dead filesystem socket while preserving an active listener: the
    // interprocess overwrite operation distinguishes stale names from live peers.
    match ListenerOptions::new()
        .name(build_name(socket)?)
        .try_overwrite(true)
        .create_sync()
    {
        Ok(listener) => Ok(listener),
        Err(e) if e.kind() == io::ErrorKind::AddrInUse => Err(already_listening()),
        Err(e) => Err(e),
    }
}

/// Serve requests on an already-bound singleton listener. `engine` moves onto the
/// dedicated writer thread; embedding and extraction run on connection threads.
pub fn serve(engine: Engine, listener: Listener) -> io::Result<()> {
    let preparer = engine.preparer();
    let writer = spawn_writer(engine);
    let active = Arc::new(AtomicUsize::new(0));

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                if active.load(Ordering::Relaxed) >= MAX_CONNECTIONS {
                    eprintln!("memeora-daemon: connection limit reached, dropping connection");
                    drop(stream);
                    continue;
                }
                active.fetch_add(1, Ordering::Relaxed);
                let writer = writer.clone();
                let preparer = preparer.clone();
                let active = Arc::clone(&active);
                thread::spawn(move || {
                    // RAII so the slot is freed on thread exit *including a panic* in
                    // prepare/handle_conn — `handle_conn` runs embedding/extraction on
                    // this thread, outside the writer's `catch_unwind`, so a bare
                    // `fetch_sub` after the call would be skipped on unwind and leak
                    // the slot until the daemon stops accepting (a zombie at 256).
                    let _slot = ActiveSlot(active);
                    // Isolate panics in embedding/extraction so a single bad request
                    // cannot tear down the whole connection thread and kill the slot
                    // counter.  The writer thread already has its own catch_unwind;
                    // this mirrors that guard on the connection side.
                    if catch_unwind(AssertUnwindSafe(|| {
                        handle_conn(stream, &writer, &preparer);
                    }))
                    .is_err()
                    {
                        eprintln!("memeora-daemon: connection handler panicked; connection closed");
                    }
                });
            }
            Err(e) => eprintln!("memeora-daemon: accept error: {e}"),
        }
    }
    Ok(())
}

/// Serve one connection: read framed requests, prepare (embed/extract) them on
/// this thread, forward to the writer, frame back the responses, until the peer
/// closes or an I/O error occurs.
fn handle_conn(stream: Stream, writer: &SyncSender<Job>, preparer: &Preparer) {
    if stream.set_recv_timeout(Some(HANDSHAKE_TIMEOUT)).is_err()
        || stream.set_send_timeout(Some(IO_TIMEOUT)).is_err()
    {
        return;
    }
    let mut reader = BufReader::new(stream);
    let hello = matches!(
        frame::read_message::<_, Request>(&mut reader),
        Ok(Some(Request::Hello { .. }))
    );
    if !hello {
        return;
    }
    if reader.get_mut().set_recv_timeout(Some(IO_TIMEOUT)).is_err() {
        return;
    }

    // The handshake is also processed by the writer so clients receive the same
    // version/capability response as every other transport.
    let mut next_request = Some(Request::Hello {
        protocol_version: memeora_proto::PROTOCOL_VERSION,
    });
    while let Some(request) = next_request.take() {
        if let Err(err) = validate_request_limits(&request) {
            let _ = frame::write_message(
                reader.get_mut(),
                &Response::Error {
                    message: err.to_string(),
                },
            );
            break;
        }

        // Embedding/extraction happens here, off the writer thread (extraction is
        // parallel; embedding serializes on the shared model — see the module docs).
        let response = match preparer.prepare(request) {
            Ok(prepared) => {
                let (reply_tx, reply_rx) = mpsc::channel();
                if writer
                    .send(Job {
                        prepared,
                        reply: reply_tx,
                    })
                    .is_err()
                {
                    break; // writer thread gone
                }
                let Ok(response) = reply_rx.recv() else { break };
                response
            }
            Err(err) => Response::Error {
                message: err.to_string(),
            },
        };

        if frame::write_message(reader.get_mut(), &response).is_err() {
            break;
        }
        next_request = match frame::read_message::<_, Request>(&mut reader) {
            Ok(Some(Request::Hello { .. })) => Some(Request::Hello {
                protocol_version: memeora_proto::PROTOCOL_VERSION,
            }),
            Ok(Some(next)) => Some(next),
            Ok(None) | Err(_) => None,
        };
    }
}

fn validate_request_limits(request: &Request) -> memeora_core::Result<()> {
    let (count, tokens) = match request {
        Request::Recall { k, max_tokens, .. } | Request::Bundle { k, max_tokens, .. } => {
            (*k, *max_tokens)
        }
        Request::List { limit, .. } => (*limit, None),
        _ => return Ok(()),
    };
    if count > MAX_RESULTS {
        return Err(memeora_core::Error::Invalid(format!(
            "result count exceeds {MAX_RESULTS}"
        )));
    }
    if tokens.is_some_and(|value| value > MAX_TOKEN_BUDGET) {
        return Err(memeora_core::Error::Invalid(format!(
            "token budget exceeds {MAX_TOKEN_BUDGET}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;
    use memeora_core::{EmbeddingProvider, EmbeddingSpace, HeuristicExtractor, SqliteStore};
    use memeora_proto::PROTOCOL_VERSION;
    use std::time::Duration;

    struct ZeroEmbedder(EmbeddingSpace);
    impl EmbeddingProvider for ZeroEmbedder {
        fn space(&self) -> &EmbeddingSpace {
            &self.0
        }
        fn embed_documents(&self, texts: &[&str]) -> memeora_core::Result<Vec<Vec<f32>>> {
            // Distinct-ish vectors by length so unrelated texts don't dedup.
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32, 0.0, 1.0])
                .collect())
        }
    }

    fn test_engine() -> Engine {
        Engine::new(
            SqliteStore::open_in_memory(3).unwrap(),
            Box::new(ZeroEmbedder(EmbeddingSpace::new("mock", "zero", 3))),
            Box::new(HeuristicExtractor::default()),
        )
    }

    fn test_engine_with_panic_once() -> Engine {
        test_engine().with_test_panic_once("test writer panic")
    }

    fn connect(socket: &str) -> BufReader<Stream> {
        // The server binds before accepting; retry until it's up.
        for _ in 0..200 {
            if let Ok(stream) = Stream::connect(build_name(socket).unwrap()) {
                let mut reader = BufReader::new(stream);
                frame::write_message(
                    reader.get_mut(),
                    &Request::Hello {
                        protocol_version: PROTOCOL_VERSION,
                    },
                )
                .unwrap();
                let response: Response = frame::read_message(&mut reader).unwrap().unwrap();
                assert!(matches!(response, Response::Hello { .. }));
                return reader;
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("could not connect to test server");
    }

    #[test]
    fn server_handles_framed_requests() {
        let socket = "memeora-test-server-roundtrip.sock";
        thread::spawn(move || serve(test_engine(), bind(socket).unwrap()).unwrap());

        let mut conn = connect(socket);

        // Handshake.
        frame::write_message(
            conn.get_mut(),
            &Request::Hello {
                protocol_version: PROTOCOL_VERSION,
            },
        )
        .unwrap();
        let resp: Response = frame::read_message(&mut conn).unwrap().unwrap();
        assert!(
            matches!(resp, Response::Hello { protocol_version, .. } if protocol_version == PROTOCOL_VERSION)
        );

        // Add then recall over the same connection.
        frame::write_message(
            conn.get_mut(),
            &Request::Add {
                scope: "s".into(),
                content: "I prefer dark mode".into(),
                kind: "preference".into(),
            },
        )
        .unwrap();
        let added: Response = frame::read_message(&mut conn).unwrap().unwrap();
        assert!(matches!(added, Response::Added { .. }));

        frame::write_message(
            conn.get_mut(),
            &Request::Recall {
                scope: "s".into(),
                query: "I prefer dark mode".into(),
                k: 5,
                max_tokens: None,
            },
        )
        .unwrap();
        match frame::read_message::<_, Response>(&mut conn)
            .unwrap()
            .unwrap()
        {
            Response::Memories { memories } => assert_eq!(memories.len(), 1),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn concurrent_clients_share_writer() {
        let socket = "memeora-test-server-concurrent.sock";
        thread::spawn(move || serve(test_engine(), bind(socket).unwrap()).unwrap());

        let mut handles = Vec::new();
        for i in 0..8 {
            let socket = socket.to_string();
            handles.push(thread::spawn(move || {
                let mut conn = connect(&socket);
                let content = format!("concurrent memory {i}");

                frame::write_message(
                    conn.get_mut(),
                    &Request::Add {
                        scope: "s".into(),
                        content: content.clone(),
                        kind: "fact".into(),
                    },
                )
                .unwrap();
                match frame::read_message::<_, Response>(&mut conn)
                    .unwrap()
                    .unwrap()
                {
                    Response::Added { id } => assert!(!id.is_empty()),
                    other => panic!("unexpected add response: {other:?}"),
                }

                frame::write_message(
                    conn.get_mut(),
                    &Request::Recall {
                        scope: "s".into(),
                        query: content,
                        k: 1,
                        max_tokens: None,
                    },
                )
                .unwrap();
                match frame::read_message::<_, Response>(&mut conn)
                    .unwrap()
                    .unwrap()
                {
                    Response::Memories { memories } => assert_eq!(memories.len(), 1),
                    other => panic!("unexpected recall response: {other:?}"),
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn writer_panic_recovers_and_keeps_serving() {
        let socket = "memeora-test-server-panic-recovery.sock";
        thread::spawn(move || serve(test_engine_with_panic_once(), bind(socket).unwrap()).unwrap());

        let mut conn = connect(socket);

        frame::write_message(
            conn.get_mut(),
            &Request::Add {
                scope: "s".into(),
                content: "panic first".into(),
                kind: "fact".into(),
            },
        )
        .unwrap();
        match frame::read_message::<_, Response>(&mut conn)
            .unwrap()
            .unwrap()
        {
            Response::Error { message } => {
                assert_eq!(message, "internal error: the request handler panicked")
            }
            other => panic!("unexpected first response: {other:?}"),
        }

        frame::write_message(
            conn.get_mut(),
            &Request::Add {
                scope: "s".into(),
                content: "still alive".into(),
                kind: "fact".into(),
            },
        )
        .unwrap();
        match frame::read_message::<_, Response>(&mut conn)
            .unwrap()
            .unwrap()
        {
            Response::Added { id } => assert!(!id.is_empty()),
            other => panic!("unexpected recovery response: {other:?}"),
        }
    }

    #[test]
    fn second_daemon_on_same_socket_is_refused() {
        let socket = "memeora-test-singleton.sock";
        thread::spawn(move || {
            let _ = serve(test_engine(), bind(socket).unwrap());
        });
        // Wait until the first daemon is accepting connections.
        let _conn = connect(socket);
        // A second daemon on the same socket must refuse rather than hijack it.
        let err = bind(socket).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AddrInUse);
    }
}
