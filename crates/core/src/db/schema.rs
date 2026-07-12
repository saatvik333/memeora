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
        // Observation/evidence layer (P3): one row per (memory, distinct source) that
        // corroborates the belief. `proof_count` on `memories` becomes a denormalized
        // cache of `COUNT(DISTINCT source_id)` here (read on every recall), refreshed
        // by `record_evidence`. The composite PK makes re-recording the same source a
        // set-union no-op, so one source restating a belief can't inflate proof_count.
        M::up(
            "CREATE TABLE evidence (
            memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            source_id   TEXT NOT NULL,
            quote       TEXT,
            occurred_at INTEGER,
            PRIMARY KEY (memory_id, source_id)
        );
        CREATE INDEX idx_evidence_memory ON evidence(memory_id);",
        ),
        // Edge dynamics (Phase E): Hebbian potentiation + Ebbinghaus/Cepeda decay applied
        // to graph edges, mirroring the per-memory strength model. `strength` is the
        // potentiation level at last co-activation; `stability` is Cepeda durability (grows
        // only on spaced activations, slows decay); `last_activated` is the last
        // co-activation time (nullable — `graph_search` COALESCEs to `created_at`);
        // `activation_count` is the Hebbian activation tally. Additive ALTERs (defaults
        // back-fill existing rows in place); the UPDATE seeds `last_activated` for edges
        // that predate this migration so decay measures idle time from creation.
        M::up(
            "ALTER TABLE relationships ADD COLUMN strength REAL NOT NULL DEFAULT 1.0;
             ALTER TABLE relationships ADD COLUMN stability REAL NOT NULL DEFAULT 1.0;
             ALTER TABLE relationships ADD COLUMN last_activated INTEGER;
             ALTER TABLE relationships ADD COLUMN activation_count INTEGER NOT NULL DEFAULT 0;
             UPDATE relationships SET last_activated = created_at WHERE last_activated IS NULL;",
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

    #[test]
    fn relationships_gain_edge_dynamics_columns() {
        // The Phase-E migration widens `relationships` with edge-dynamics columns whose
        // defaults back-fill any row inserted without them (the sole-writer daemon never
        // sets strength/stability/activation_count on the insert path).
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        conn.execute_batch(
            "INSERT INTO memories (id, content, kind, container_tag, created_at, last_accessed_at)
               VALUES ('a','x','fact','t',100,100),('b','y','fact','t',100,100);
             INSERT INTO relationships (from_id, to_id, kind, created_at)
               VALUES ('a','b','extends',150);",
        )
        .unwrap();
        let (strength, stability, count): (f64, f64, i64) = conn
            .query_row(
                "SELECT strength, stability, activation_count FROM relationships WHERE from_id='a'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!((strength, stability, count), (1.0, 1.0, 0));
        // `last_activated` is nullable; rows inserted without it (bypassing `add_edge`) stay
        // NULL — `graph_search` COALESCEs to `created_at`, so decay still has an anchor.
        let last_activated: Option<i64> = conn
            .query_row(
                "SELECT last_activated FROM relationships WHERE from_id='a'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(last_activated.is_none());
    }
}
