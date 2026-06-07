//! Schema migrations (the `vec0` virtual table is created per-store in `store::sqlite`,
//! since its dimensionality is configured at runtime).

use rusqlite::Connection;
use rusqlite_migration::{M, Migrations};

use crate::Result;

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![
        M::up(
            "CREATE TABLE memories (
            rowid            INTEGER PRIMARY KEY,
            id               TEXT NOT NULL UNIQUE,
            content          TEXT NOT NULL,
            kind             TEXT NOT NULL,
            container_tag    TEXT NOT NULL,
            is_latest        INTEGER NOT NULL DEFAULT 1,
            strength         REAL NOT NULL DEFAULT 1.0,
            created_at       INTEGER NOT NULL,
            last_accessed_at INTEGER NOT NULL,
            expires_at       INTEGER,
            metadata         TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX idx_memories_container ON memories(container_tag, is_latest);
        CREATE VIRTUAL TABLE fts_memories USING fts5(memory_id UNINDEXED, content);",
        ),
        // Knowledge-graph edges between memories (updates | extends | derives).
        M::up(
            "CREATE TABLE relationships (
            from_id    TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            to_id      TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            kind       TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (from_id, to_id, kind)
        );
        CREATE INDEX idx_relationships_from ON relationships(from_id);
        CREATE INDEX idx_relationships_to ON relationships(to_id);",
        ),
    ])
}

/// Apply all pending migrations to bring `conn` to the latest schema.
pub fn migrate(conn: &mut Connection) -> Result<()> {
    migrations().to_latest(conn)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_are_valid() {
        // rusqlite_migration validates that every migration parses and applies cleanly.
        assert!(migrations().validate().is_ok());
    }
}
