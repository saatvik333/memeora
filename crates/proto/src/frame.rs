//! Length-delimited JSON framing for the IPC stream.
//!
//! Each message is a big-endian `u32` byte length followed by that many bytes of
//! JSON. Both the daemon and clients use this identical framing, so it lives in
//! the shared contract. The functions are generic over [`std::io`] `Read`/`Write`,
//! so they work over Unix sockets, named pipes, or test buffers, and from blocking
//! contexts (the daemon frames on its writer thread).

use std::io::{self, Read, Write};

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Reject frames larger than this (16 MiB) to bound allocation on bad input.
pub const MAX_MESSAGE_BYTES: u32 = 16 * 1024 * 1024;

/// Write one length-prefixed JSON message and flush.
pub fn write_message<W: Write, T: Serialize>(writer: &mut W, message: &T) -> io::Result<()> {
    let bytes = serde_json::to_vec(message).map_err(io::Error::other)?;
    let len = u32::try_from(bytes.len())
        .ok()
        .filter(|&n| n <= MAX_MESSAGE_BYTES)
        .ok_or_else(|| io::Error::other("message exceeds MAX_MESSAGE_BYTES"))?;
    writer.write_all(&len.to_be_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()
}

/// Read one length-prefixed JSON message.
///
/// Returns `Ok(None)` on a clean EOF at a frame boundary (the peer closed the
/// connection), and an error on a truncated frame or oversize length.
pub fn read_message<R: Read, T: DeserializeOwned>(reader: &mut R) -> io::Result<Option<T>> {
    let mut len_buf = [0u8; 4];
    // A clean EOF is valid only before a frame starts. Once even one header byte
    // arrives, EOF means the peer sent a malformed/truncated frame.
    match reader.read(&mut len_buf[..1]) {
        Ok(0) => return Ok(None),
        Ok(1) => {}
        Ok(_) => unreachable!("single-byte buffer cannot read more than one byte"),
        Err(e) => return Err(e),
    }
    reader.read_exact(&mut len_buf[1..])?;

    let len = u32::from_be_bytes(len_buf);
    if len > MAX_MESSAGE_BYTES {
        return Err(io::Error::other("incoming frame exceeds MAX_MESSAGE_BYTES"));
    }

    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    let message = serde_json::from_slice(&buf).map_err(io::Error::other)?;
    Ok(Some(message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Request, Response};
    use std::io::Cursor;

    #[test]
    fn roundtrips_a_message() {
        let req = Request::Recall {
            scope: "s".into(),
            query: "q".into(),
            k: 3,
            max_tokens: None,
        };
        let mut buf = Vec::new();
        write_message(&mut buf, &req).unwrap();

        let mut cursor = Cursor::new(buf);
        let got: Option<Request> = read_message(&mut cursor).unwrap();
        assert_eq!(got, Some(req));
    }

    #[test]
    fn reads_multiple_framed_messages_in_order() {
        let msgs = [Response::Forgotten, Response::Added { id: "m1".into() }];
        let mut buf = Vec::new();
        for m in &msgs {
            write_message(&mut buf, m).unwrap();
        }

        let mut cursor = Cursor::new(buf);
        for m in &msgs {
            let got: Option<Response> = read_message(&mut cursor).unwrap();
            assert_eq!(got.as_ref(), Some(m));
        }
        // Clean EOF after the last frame.
        let end: Option<Response> = read_message(&mut cursor).unwrap();
        assert_eq!(end, None);
    }

    #[test]
    fn clean_eof_on_empty_stream() {
        let mut cursor = Cursor::new(Vec::new());
        let got: Option<Request> = read_message(&mut cursor).unwrap();
        assert_eq!(got, None);
    }

    #[test]
    fn truncated_frame_is_an_error() {
        // Declares 10 bytes but provides 2.
        let mut buf = 10u32.to_be_bytes().to_vec();
        buf.extend_from_slice(b"hi");
        let mut cursor = Cursor::new(buf);
        let result: io::Result<Option<Request>> = read_message(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn oversize_length_is_rejected() {
        let buf = (MAX_MESSAGE_BYTES + 1).to_be_bytes().to_vec();
        let mut cursor = Cursor::new(buf);
        let result: io::Result<Option<Request>> = read_message(&mut cursor);
        assert!(result.is_err());
    }
}
