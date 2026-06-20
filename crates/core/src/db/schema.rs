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
        // Vision-readiness columns (additive, for step 11). Widening the schema now —
        // while the store is pre-1.0 — keeps these a clean ALTER instead of a data
        // migration over real memories later: version chain (parent_id/root_id),
        // bi-temporal valid-time (occurred_start/end), observation corroboration
        // (proof_count), and forgetting-engine bookkeeping (stability/access_count).
        M::up(
            "ALTER TABLE memories ADD COLUMN parent_id TEXT;
             ALTER TABLE memories ADD COLUMN root_id TEXT;
             ALTER TABLE memories ADD COLUMN occurred_start INTEGER;
             ALTER TABLE memories ADD COLUMN occurred_end INTEGER;
             ALTER TABLE memories ADD COLUMN proof_count INTEGER NOT NULL DEFAULT 1;
             ALTER TABLE memories ADD COLUMN stability REAL NOT NULL DEFAULT 1.0;
             ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;
             CREATE INDEX idx_memories_root ON memories(root_id);",
        ),
        // Entity layer (increment C): canonical entities + memory↔entity links,
        // scoped per container. Powers entity-keyed consolidation (D) and the graph
        // recall channel (F). Entity `id` is sha32(container_tag\0canonical), so
        // INSERT OR IGNORE makes (re)linking idempotent.
        M::up(
            "CREATE TABLE entities (
            id            TEXT NOT NULL PRIMARY KEY,
            canonical     TEXT NOT NULL,
            container_tag TEXT NOT NULL,
            created_at    INTEGER NOT NULL
        );
        CREATE TABLE memory_entities (
            memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            PRIMARY KEY (memory_id, entity_id)
        );
        CREATE INDEX idx_memory_entities_entity ON memory_entities(entity_id);",
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
