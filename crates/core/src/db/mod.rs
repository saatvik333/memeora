//! Database connection management and `sqlite-vec` registration.

pub mod schema;

use std::path::Path;
use std::sync::Once;

use rusqlite::Connection;

use crate::Result;

static VEC_INIT: Once = Once::new();

/// Register the statically-linked `sqlite-vec` extension via SQLite's auto-extension
/// hook so every connection opened afterwards has `vec0` available. Runs exactly once.
fn register_sqlite_vec() {
    VEC_INIT.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is the C entrypoint for the statically-linked
        // sqlite-vec extension; its signature matches what `sqlite3_auto_extension`
        // expects. The transmute reconciles the two crates' bindgen fn-pointer types.
        // Registered once via `Once`.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

fn configure(conn: &Connection, wal: bool) -> Result<()> {
    if wal {
        conn.pragma_update(None, "journal_mode", "WAL")?;
    }
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    Ok(())
}

/// Open (or create) a database at `path`, register sqlite-vec, set pragmas, and migrate.
pub fn open(path: impl AsRef<Path>) -> Result<Connection> {
    register_sqlite_vec();
    let mut conn = Connection::open(path)?;
    configure(&conn, true)?;
    schema::migrate(&mut conn)?;
    Ok(conn)
}

/// Open an in-memory database (used by tests). WAL is skipped (irrelevant for `:memory:`).
pub fn open_in_memory() -> Result<Connection> {
    register_sqlite_vec();
    let mut conn = Connection::open_in_memory()?;
    configure(&conn, false)?;
    schema::migrate(&mut conn)?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqlite_vec_is_registered_and_vec0_works() {
        // Proves static sqlite-vec linking: create a vec0 table and confirm vec_version().
        let conn = open_in_memory().unwrap();
        let version: String = conn
            .query_row("SELECT vec_version()", [], |r| r.get(0))
            .unwrap();
        assert!(
            version.starts_with('v'),
            "unexpected vec_version: {version}"
        );
        conn.execute_batch("CREATE VIRTUAL TABLE t USING vec0(embedding FLOAT[3]);")
            .unwrap();
    }
}
