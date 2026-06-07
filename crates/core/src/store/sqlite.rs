//! SQLite-backed [`VectorStore`]: `sqlite-vec` for KNN + FTS5 for lexical search.

use std::path::Path;

use rusqlite::{Connection, OptionalExtension, Row, params};

use crate::db;
use crate::error::{Error, Result};
use crate::store::{Memory, MemoryKind, ScoredMemory, VectorStore};

/// SQLite store. Owns one connection (the daemon keeps a single writer; see ARCHITECTURE.md).
pub struct SqliteStore {
    conn: Connection,
    dim: usize,
}

impl SqliteStore {
    /// Open (or create) a store at `path` with embedding dimensionality `dim`.
    pub fn open(path: impl AsRef<Path>, dim: usize) -> Result<Self> {
        Self::init(db::open(path)?, dim)
    }

    /// Open an in-memory store (tests) with embedding dimensionality `dim`.
    pub fn open_in_memory(dim: usize) -> Result<Self> {
        Self::init(db::open_in_memory()?, dim)
    }

    /// Embedding dimensionality this store was created with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn init(conn: Connection, dim: usize) -> Result<Self> {
        // The vec0 table is dimensionality-dependent, so it is created here rather than
        // in the static migrations. `container_tag` is a metadata column for KNN filtering.
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                memory_rowid INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}],
                container_tag TEXT
            );"
        ))?;
        Ok(SqliteStore { conn, dim })
    }
}

/// Serialize an f32 slice to the little-endian byte blob `vec0` expects.
fn vec_blob(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// Hydrate a [`Memory`] from a row that selected the `memories` columns by name.
/// The embedding is not read back (it lives in the vec0 index).
fn row_to_memory(row: &Row) -> rusqlite::Result<Memory> {
    Ok(Memory {
        id: row.get("id")?,
        content: row.get("content")?,
        kind: MemoryKind::from_str_lossy(&row.get::<_, String>("kind")?),
        container_tag: row.get("container_tag")?,
        embedding: Vec::new(),
        is_latest: row.get::<_, i64>("is_latest")? != 0,
        strength: row.get::<_, f64>("strength")? as f32,
        created_at: row.get("created_at")?,
        last_accessed_at: row.get("last_accessed_at")?,
        expires_at: row.get("expires_at")?,
        metadata: row.get("metadata")?,
    })
}

const MEMORY_COLS: &str = "m.id, m.content, m.kind, m.container_tag, m.is_latest, m.strength, \
     m.created_at, m.last_accessed_at, m.expires_at, m.metadata";

impl VectorStore for SqliteStore {
    fn upsert(&mut self, memory: &Memory) -> Result<()> {
        if memory.embedding.len() != self.dim {
            return Err(Error::DimMismatch {
                expected: self.dim,
                got: memory.embedding.len(),
            });
        }
        let tx = self.conn.transaction()?;
        // Replace any prior row with this id across all three tables.
        if let Some(rowid) = tx
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![memory.id],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
        {
            tx.execute(
                "DELETE FROM vec_memories WHERE memory_rowid = ?1",
                params![rowid],
            )?;
        }
        tx.execute(
            "DELETE FROM fts_memories WHERE memory_id = ?1",
            params![memory.id],
        )?;
        tx.execute("DELETE FROM memories WHERE id = ?1", params![memory.id])?;

        tx.execute(
            "INSERT INTO memories
                (id, content, kind, container_tag, is_latest, strength,
                 created_at, last_accessed_at, expires_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                memory.id,
                memory.content,
                memory.kind.as_str(),
                memory.container_tag,
                memory.is_latest as i64,
                memory.strength as f64,
                memory.created_at,
                memory.last_accessed_at,
                memory.expires_at,
                memory.metadata,
            ],
        )?;
        let rowid = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO vec_memories (memory_rowid, embedding, container_tag)
             VALUES (?1, ?2, ?3)",
            params![rowid, vec_blob(&memory.embedding), memory.container_tag],
        )?;
        tx.execute(
            "INSERT INTO fts_memories (memory_id, content) VALUES (?1, ?2)",
            params![memory.id, memory.content],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn knn(&self, container_tag: &str, query: &[f32], k: usize) -> Result<Vec<ScoredMemory>> {
        if query.len() != self.dim {
            return Err(Error::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        // `k` is a store-controlled integer; inlining it sidesteps vec0's binding rules
        // for the `k = N` KNN constraint. The query vector is bound as a blob param.
        let sql = format!(
            "WITH knn AS (
                SELECT memory_rowid, distance FROM vec_memories
                WHERE embedding MATCH ?1 AND k = {k} AND container_tag = ?2
                ORDER BY distance
             )
             SELECT {MEMORY_COLS}, knn.distance AS distance
             FROM knn JOIN memories m ON m.rowid = knn.memory_rowid
             ORDER BY knn.distance"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![vec_blob(query), container_tag], |row| {
            Ok(ScoredMemory {
                memory: row_to_memory(row)?,
                score: row.get::<_, f64>("distance")? as f32,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    fn text_search(&self, container_tag: &str, query: &str, k: usize) -> Result<Vec<ScoredMemory>> {
        let sql = format!(
            "SELECT {MEMORY_COLS}, bm25(fts_memories) AS distance
             FROM fts_memories f JOIN memories m ON m.id = f.memory_id
             WHERE fts_memories MATCH ?1 AND m.container_tag = ?2
             ORDER BY distance
             LIMIT ?3"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![query, container_tag, k as i64], |row| {
            Ok(ScoredMemory {
                memory: row_to_memory(row)?,
                score: row.get::<_, f64>("distance")? as f32,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    fn get(&self, id: &str) -> Result<Option<Memory>> {
        let sql = format!("SELECT {MEMORY_COLS} FROM memories m WHERE m.id = ?1");
        let memory = self
            .conn
            .query_row(&sql, params![id], row_to_memory)
            .optional()?;
        Ok(memory)
    }

    fn count(&self, container_tag: &str) -> Result<usize> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE container_tag = ?1",
            params![container_tag],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    fn list_latest(&self, container_tag: &str, limit: usize) -> Result<Vec<Memory>> {
        let sql = format!(
            "SELECT {MEMORY_COLS} FROM memories m
             WHERE m.container_tag = ?1 AND m.is_latest = 1
             ORDER BY m.created_at DESC, m.rowid DESC
             LIMIT ?2"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![container_tag, limit as i64], row_to_memory)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: &str, content: &str, tag: &str, emb: Vec<f32>) -> Memory {
        Memory::new(id, content, MemoryKind::Fact, tag, emb)
    }

    #[test]
    fn knn_round_trip_orders_by_distance() {
        let mut s = SqliteStore::open_in_memory(4).unwrap();
        let tag = "memeora_user_test";
        s.upsert(&mem("m1", "alpha", tag, vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        s.upsert(&mem("m2", "beta", tag, vec![0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        s.upsert(&mem("m3", "gamma", tag, vec![0.0, 0.0, 1.0, 0.0]))
            .unwrap();

        let hits = s.knn(tag, &[0.9, 0.1, 0.0, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].memory.id, "m1");
        assert!(hits[0].score <= hits[1].score);
    }

    #[test]
    fn container_scope_is_isolated() {
        let mut s = SqliteStore::open_in_memory(3).unwrap();
        s.upsert(&mem("a", "in a", "tag_a", vec![1.0, 0.0, 0.0]))
            .unwrap();
        s.upsert(&mem("b", "in b", "tag_b", vec![1.0, 0.0, 0.0]))
            .unwrap();

        let hits = s.knn("tag_a", &[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.id, "a");
        assert_eq!(s.count("tag_a").unwrap(), 1);
        assert_eq!(s.count("tag_b").unwrap(), 1);
    }

    #[test]
    fn text_search_matches_content() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        s.upsert(&mem("m1", "the user prefers tailwind", tag, vec![1.0, 0.0]))
            .unwrap();
        s.upsert(&mem(
            "m2",
            "deploy with docker compose",
            tag,
            vec![0.0, 1.0],
        ))
        .unwrap();

        let hits = s.text_search(tag, "tailwind", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.id, "m1");
    }

    #[test]
    fn upsert_replaces_existing_id() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        s.upsert(&mem("m1", "first", tag, vec![1.0, 0.0])).unwrap();
        s.upsert(&mem("m1", "second", tag, vec![0.0, 1.0])).unwrap();
        assert_eq!(s.count(tag).unwrap(), 1);
        assert_eq!(s.get("m1").unwrap().unwrap().content, "second");
    }

    #[test]
    fn dim_mismatch_is_rejected() {
        let mut s = SqliteStore::open_in_memory(4).unwrap();
        let err = s.upsert(&mem("m1", "x", "t", vec![1.0, 0.0])).unwrap_err();
        assert!(matches!(
            err,
            Error::DimMismatch {
                expected: 4,
                got: 2
            }
        ));
        assert!(s.knn("t", &[1.0, 0.0], 1).is_err());
    }

    #[test]
    fn get_missing_returns_none() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        assert!(s.get("nope").unwrap().is_none());
    }
}
