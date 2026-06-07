//! The blocking IPC server: a writer-actor thread owns the [`Engine`], and a
//! thread-per-connection accept loop frames requests to it.
//!
//! Why blocking threads rather than tokio: the [`Engine`] and its rusqlite store
//! are synchronous, and [`memeora_proto::frame`] is sync `std::io`. A single
//! writer thread owning the `Engine` is the natural "sole DB writer" and needs no
//! locks; connection threads parse/frame in parallel and forward each request to
//! the writer over a channel. This sidesteps blocking a tokio runtime with sync
//! SQLite. (tokio enters later only for the MCP HTTP transport and the dashboard.)

use std::io::{self, BufReader};
use std::sync::mpsc::{self, Sender};
use std::thread;

use interprocess::local_socket::{
    GenericFilePath, GenericNamespaced, ListenerOptions, Name, Stream, prelude::*,
};
use memeora_proto::{Request, Response, frame};

use crate::Engine;

/// A request plus the channel its response goes back on.
struct Job {
    request: Request,
    reply: Sender<Response>,
}

/// Build a local-socket [`Name`] from a string: a value containing a path
/// separator is a filesystem socket path; otherwise a namespaced name.
pub fn build_name(socket: &str) -> io::Result<Name<'_>> {
    if socket.contains('/') || socket.contains('\\') {
        socket.to_fs_name::<GenericFilePath>()
    } else {
        socket.to_ns_name::<GenericNamespaced>()
    }
}

/// Spawn the writer-actor: it owns the `Engine` and handles jobs serially.
fn spawn_writer(mut engine: Engine) -> Sender<Job> {
    let (tx, rx) = mpsc::channel::<Job>();
    thread::spawn(move || {
        for job in rx {
            let response = engine.handle(job.request);
            // If the connection is gone, drop the response silently.
            let _ = job.reply.send(response);
        }
    });
    tx
}

/// Serve requests on `socket` until the listener errors fatally. Blocks the
/// calling thread. `engine` moves onto the dedicated writer thread.
pub fn serve(engine: Engine, socket: &str) -> io::Result<()> {
    let name = build_name(socket)?;
    let listener = ListenerOptions::new()
        .name(name)
        .try_overwrite(true)
        .create_sync()?;
    let writer = spawn_writer(engine);

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let writer = writer.clone();
                thread::spawn(move || handle_conn(stream, &writer));
            }
            Err(e) => eprintln!("memeora-daemon: accept error: {e}"),
        }
    }
    Ok(())
}

/// Serve one connection: read framed requests, forward to the writer, frame back
/// the responses, until the peer closes or an I/O error occurs.
fn handle_conn(stream: Stream, writer: &Sender<Job>) {
    let mut reader = BufReader::new(stream);
    loop {
        let request = match frame::read_message::<_, Request>(&mut reader) {
            Ok(Some(request)) => request,
            Ok(None) => break, // peer closed cleanly
            Err(_) => break,   // truncated / bad frame
        };

        let (reply_tx, reply_rx) = mpsc::channel();
        if writer
            .send(Job {
                request,
                reply: reply_tx,
            })
            .is_err()
        {
            break; // writer thread gone
        }
        let Ok(response) = reply_rx.recv() else { break };
        if frame::write_message(reader.get_mut(), &response).is_err() {
            break;
        }
    }
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

    fn connect(socket: &str) -> BufReader<Stream> {
        // The server binds before accepting; retry until it's up.
        for _ in 0..200 {
            if let Ok(stream) = Stream::connect(build_name(socket).unwrap()) {
                return BufReader::new(stream);
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("could not connect to test server");
    }

    #[test]
    fn server_handles_framed_requests() {
        let socket = "memeora-test-server-roundtrip.sock";
        thread::spawn(move || serve(test_engine(), socket).unwrap());

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
}
