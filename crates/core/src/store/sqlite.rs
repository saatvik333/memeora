//! SQLite-backed [`VectorStore`]: `sqlite-vec` for KNN + FTS5 for lexical search.

use std::path::Path;

use rusqlite::{Connection, OptionalExtension, Row, params};

use crate::db;
use crate::dynamics;
use crate::error::{Error, Result};
use crate::store::{
    EdgeKind, GraphData, Memory, MemoryKind, Relationship, ScopeInfo, ScoredMemory, VectorStore,
    now_unix,
};

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

    /// Open an existing store as a **read-only** reader (no migrate, no table
    /// creation, writes refused at the SQL layer). For the dashboard's second
    /// connection so the daemon keeps a single writer. Validates the stored
    /// embedding dim matches `dim` but never writes it.
    pub fn open_readonly(path: impl AsRef<Path>, dim: usize) -> Result<Self> {
        let conn = db::open_reader(path)?;
        if let Some(prev) = stored_dim(&conn)?
            && prev != dim
        {
            return Err(Error::DimMismatch {
                expected: prev,
                got: dim,
            });
        }
        Ok(SqliteStore { conn, dim })
    }

    /// Embedding dimensionality this store was created with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn init(conn: Connection, dim: usize) -> Result<Self> {
        // Persist the embedding dim so reopening with a different model (which would
        // silently leave the old-dimension vec0 table in place) is caught loudly.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            [],
        )?;
        match stored_dim(&conn)? {
            Some(prev) if prev != dim => {
                // Reusing DimMismatch: the store was built for `prev`, opened for `dim`.
                return Err(Error::DimMismatch {
                    expected: prev,
                    got: dim,
                });
            }
            Some(_) => {}
            None => {
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('embedding_dim', ?1)",
                    params![dim.to_string()],
                )?;
            }
        }

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

    /// List every scope that holds memories, with its latest and total counts.
    ///
    /// Read-only and not part of [`VectorStore`]: it powers the dashboard's spaces
    /// switcher (a whole-DB scan across container tags), which the scoped trait
    /// methods deliberately don't expose.
    pub fn list_scopes(&self) -> Result<Vec<ScopeInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT container_tag,
                    COALESCE(SUM(is_latest), 0) AS latest,
                    COUNT(*) AS total
             FROM memories
             GROUP BY container_tag
             ORDER BY latest DESC, total DESC, container_tag",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ScopeInfo {
                tag: row.get("container_tag")?,
                latest: row.get::<_, i64>("latest")? as usize,
                total: row.get::<_, i64>("total")? as usize,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Fetch a scope's graph for visualization: up to `cap` nodes (all versions,
    /// newest first, *including* soft-forgotten ones so the UI can dim them) plus
    /// the edges whose endpoints are both among the returned nodes.
    ///
    /// Read-only and not part of [`VectorStore`]: unlike [`list_latest`], this
    /// intentionally returns non-latest memories so the graph shows version history.
    ///
    /// [`list_latest`]: VectorStore::list_latest
    pub fn graph(&self, container_tag: &str, cap: usize) -> Result<GraphData> {
        let node_sql = format!(
            "SELECT {MEMORY_COLS} FROM memories m
             WHERE m.container_tag = ?1
             ORDER BY m.created_at DESC, m.rowid DESC
             LIMIT ?2"
        );
        let mut stmt = self.conn.prepare(&node_sql)?;
        let nodes: Vec<Memory> = stmt
            .query_map(params![container_tag, cap as i64], row_to_memory)?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        // Only keep edges whose endpoints both survived the node cap, so the UI
        // never references a missing node.
        let ids: std::collections::HashSet<&str> = nodes.iter().map(|m| m.id.as_str()).collect();
        let mut edge_stmt = self.conn.prepare(
            "SELECT r.from_id, r.to_id, r.kind, r.created_at
             FROM relationships r
             JOIN memories mf ON mf.id = r.from_id
             JOIN memories mt ON mt.id = r.to_id
             WHERE mf.container_tag = ?1 AND mt.container_tag = ?1
             ORDER BY r.created_at",
        )?;
        let edges: Vec<Relationship> = edge_stmt
            .query_map(params![container_tag], |row| {
                Ok(Relationship {
                    from_id: row.get(0)?,
                    to_id: row.get(1)?,
                    kind: EdgeKind::from_str_lossy(&row.get::<_, String>(2)?),
                    created_at: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .filter(|e| ids.contains(e.from_id.as_str()) && ids.contains(e.to_id.as_str()))
            .collect();

        Ok(GraphData { nodes, edges })
    }

    /// Run `f` inside a single SQLite transaction so a multi-write batch is
    /// all-or-nothing: any error rolls the whole batch back, leaving no partial
    /// writes. The store's own write methods (`upsert`/`forget`) use SAVEPOINTs,
    /// which nest correctly inside this outer transaction and also work standalone,
    /// so calling them through `f` keeps the batch atomic.
    pub fn transaction<R>(&mut self, f: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        // Raw BEGIN/COMMIT rather than a held `Transaction` guard, so `f` can still
        // take `&mut self` and call the normal write methods. IMMEDIATE takes the
        // write lock up front — right for the sole-writer daemon (no mid-batch lock
        // upgrade that could fail with SQLITE_BUSY).
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        match f(self) {
            Ok(value) => {
                self.conn.execute_batch("COMMIT")?;
                Ok(value)
            }
            Err(err) => {
                // Best-effort rollback; surface the original error, not a rollback one.
                let _ = self.conn.execute_batch("ROLLBACK");
                Err(err)
            }
        }
    }
}

/// Build a safe FTS5 `MATCH` expression from arbitrary user text.
///
/// FTS5 `MATCH` is a query language, so passing raw text (with `:`, `-`, `"`, `*`,
/// `NEAR`, …) risks a syntax error that would fail the whole search. We extract
/// alphanumeric tokens and quote each as a phrase (implicit AND), which can never
/// be malformed. Returns `None` when there are no usable tokens.
fn fts5_match(query: &str) -> Option<String> {
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        // Tokens are alphanumeric, so there are no quotes to escape.
        .map(|t| format!("\"{t}\""))
        .collect();
    (!tokens.is_empty()).then(|| tokens.join(" "))
}

/// The persisted embedding dim (`meta.embedding_dim`), or `None` before first init.
fn stored_dim(conn: &Connection) -> Result<Option<usize>> {
    let raw: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'embedding_dim'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    Ok(raw.map(|s| s.parse().unwrap_or(0)))
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
        parent_id: row.get("parent_id")?,
        root_id: row.get("root_id")?,
        occurred_start: row.get("occurred_start")?,
        occurred_end: row.get("occurred_end")?,
        proof_count: row.get::<_, i64>("proof_count")? as u32,
        stability: row.get::<_, f64>("stability")? as f32,
        access_count: row.get::<_, i64>("access_count")? as u32,
    })
}

const MEMORY_COLS: &str = "m.id, m.content, m.kind, m.container_tag, m.is_latest, m.strength, \
     m.created_at, m.last_accessed_at, m.expires_at, m.metadata, m.parent_id, m.root_id, \
     m.occurred_start, m.occurred_end, m.proof_count, m.stability, m.access_count";

impl VectorStore for SqliteStore {
    fn upsert(&mut self, memory: &Memory) -> Result<()> {
        if memory.embedding.len() != self.dim {
            return Err(Error::DimMismatch {
                expected: self.dim,
                got: memory.embedding.len(),
            });
        }
        // SAVEPOINT (not BEGIN) so this nests inside `SqliteStore::transaction`'s
        // batch when present, and acts as its own transaction when called alone.
        let tx = self.conn.savepoint()?;
        let existing_rowid = tx
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![memory.id],
                |r| r.get::<_, i64>(0),
            )
            .optional()?;

        // When the id already exists, UPDATE the row in place rather than
        // delete-then-insert: the `relationships` FK is `ON DELETE CASCADE`, so
        // deleting the row would silently wipe this memory's graph edges.
        let rowid = if let Some(rowid) = existing_rowid {
            tx.execute(
                "UPDATE memories SET
                    content = ?2, kind = ?3, container_tag = ?4, is_latest = ?5,
                    strength = ?6, created_at = ?7, last_accessed_at = ?8,
                    expires_at = ?9, metadata = ?10, parent_id = ?11, root_id = ?12,
                    occurred_start = ?13, occurred_end = ?14, proof_count = ?15,
                    stability = ?16, access_count = ?17
                 WHERE id = ?1",
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
                    memory.parent_id,
                    memory.root_id,
                    memory.occurred_start,
                    memory.occurred_end,
                    memory.proof_count as i64,
                    memory.stability as f64,
                    memory.access_count as i64,
                ],
            )?;
            // Refresh the vec row in place (same rowid; vec0 has no inbound FKs).
            tx.execute(
                "DELETE FROM vec_memories WHERE memory_rowid = ?1",
                params![rowid],
            )?;
            rowid
        } else {
            tx.execute(
                "INSERT INTO memories
                    (id, content, kind, container_tag, is_latest, strength,
                     created_at, last_accessed_at, expires_at, metadata,
                     parent_id, root_id, occurred_start, occurred_end, proof_count,
                     stability, access_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
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
                    memory.parent_id,
                    memory.root_id,
                    memory.occurred_start,
                    memory.occurred_end,
                    memory.proof_count as i64,
                    memory.stability as f64,
                    memory.access_count as i64,
                ],
            )?;
            tx.last_insert_rowid()
        };

        tx.execute(
            "INSERT INTO vec_memories (memory_rowid, embedding, container_tag)
             VALUES (?1, ?2, ?3)",
            params![rowid, vec_blob(&memory.embedding), memory.container_tag],
        )?;
        // FTS row has no inbound FK; replace it to reflect updated content.
        tx.execute(
            "DELETE FROM fts_memories WHERE memory_id = ?1",
            params![memory.id],
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
             WHERE m.is_latest = 1
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
        // Sanitize arbitrary user text into a valid FTS5 MATCH; no tokens → no hits.
        let Some(match_query) = fts5_match(query) else {
            return Ok(Vec::new());
        };
        let sql = format!(
            "SELECT {MEMORY_COLS}, bm25(fts_memories) AS distance
             FROM fts_memories f JOIN memories m ON m.id = f.memory_id
             WHERE fts_memories MATCH ?1 AND m.container_tag = ?2 AND m.is_latest = 1
             ORDER BY distance
             LIMIT ?3"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![match_query, container_tag, k as i64], |row| {
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
            "SELECT COUNT(*) FROM memories WHERE container_tag = ?1 AND is_latest = 1",
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

    fn reinforce(&mut self, id: &str, delta: f32) -> Result<()> {
        // Hebbian potentiation capped at STRENGTH_MAX + Cepeda spacing: stability grows
        // only when this access is ≥ SPACING_SECS after the last (the CASE reads the
        // pre-update last_accessed_at). Decay itself is applied lazily at read time.
        self.conn.execute(
            "UPDATE memories SET
                strength = MIN(?1, strength + ?2),
                stability = stability + CASE WHEN (?3 - last_accessed_at) >= ?4 THEN ?5 ELSE 0 END,
                last_accessed_at = ?3,
                access_count = access_count + 1
             WHERE id = ?6",
            params![
                dynamics::STRENGTH_MAX as f64,
                delta as f64,
                now_unix(),
                dynamics::SPACING_SECS,
                dynamics::STABILITY_DELTA as f64,
                id,
            ],
        )?;
        Ok(())
    }

    fn record_evidence(
        &mut self,
        memory_id: &str,
        source_id: &str,
        quote: &str,
        occurred_at: i64,
    ) -> Result<()> {
        // Two writes as a unit (nests in the daemon batch like `link_entities`): add the
        // observation (set-union via the composite PK — a repeated source is ignored),
        // then refresh proof_count to the distinct-source count. Recompute (not +1) is
        // what makes one source restating idempotent. Assumes originating evidence was
        // recorded at insert time, so the count never drops below the real source set.
        let tx = self.conn.savepoint()?;
        tx.execute(
            "INSERT OR IGNORE INTO evidence (memory_id, source_id, quote, occurred_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![memory_id, source_id, quote, occurred_at],
        )?;
        tx.execute(
            "UPDATE memories SET proof_count =
                (SELECT COUNT(DISTINCT source_id) FROM evidence WHERE memory_id = ?1)
             WHERE id = ?1",
            params![memory_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn add_edge(&mut self, from_id: &str, to_id: &str, kind: EdgeKind) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO relationships (from_id, to_id, kind, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![from_id, to_id, kind.as_str(), now_unix()],
        )?;
        Ok(())
    }

    fn edges_from(&self, id: &str) -> Result<Vec<Relationship>> {
        let mut stmt = self.conn.prepare(
            "SELECT from_id, to_id, kind, created_at FROM relationships WHERE from_id = ?1
             ORDER BY created_at",
        )?;
        let rows = stmt.query_map(params![id], |row| {
            Ok(Relationship {
                from_id: row.get(0)?,
                to_id: row.get(1)?,
                kind: EdgeKind::from_str_lossy(&row.get::<_, String>(2)?),
                created_at: row.get(3)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    fn forget(&mut self, id: &str) -> Result<()> {
        // SAVEPOINT (not BEGIN) so this nests inside `SqliteStore::transaction`'s
        // batch when present, and acts as its own transaction when called alone.
        let tx = self.conn.savepoint()?;
        // Remove the vector so a forgotten memory can't occupy a KNN top-k slot
        // (vec0 applies its `k` limit before the outer `is_latest` filter). The row
        // is kept (is_latest = 0) so `get` still resolves it and the edges survive.
        if let Some(rowid) = tx
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![id],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
        {
            tx.execute(
                "DELETE FROM vec_memories WHERE memory_rowid = ?1",
                params![rowid],
            )?;
        }
        // Drop the FTS row too, so a forgotten memory doesn't keep contributing to
        // BM25 corpus stats (IDF/avgdl) and skew ranking of live results. `upsert`
        // re-inserts it if the content is later resurrected.
        tx.execute("DELETE FROM fts_memories WHERE memory_id = ?1", params![id])?;
        tx.execute(
            "UPDATE memories SET is_latest = 0 WHERE id = ?1",
            params![id],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn history(&self, root_id: &str) -> Result<Vec<Memory>> {
        // The root row carries `id = root_id` (its own `root_id` is NULL); every
        // superseding version carries `root_id = root_id`. Newest first so the current
        // version (if still on this lineage) leads.
        let sql = format!(
            "SELECT {MEMORY_COLS} FROM memories m
             WHERE m.root_id = ?1 OR m.id = ?1
             ORDER BY m.created_at DESC, m.rowid DESC"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![root_id], row_to_memory)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    fn link_entities(
        &mut self,
        memory_id: &str,
        container_tag: &str,
        entities: &[String],
    ) -> Result<()> {
        // SAVEPOINT so the whole link set is atomic: it nests inside the daemon's batch
        // transaction, and is self-contained if a caller links outside one (no partial
        // entity writes on mid-list failure).
        let tx = self.conn.savepoint()?;
        for canonical in entities {
            // Deterministic, container-scoped id so re-linking the same entity is a
            // no-op (INSERT OR IGNORE) rather than a duplicate.
            let entity_id =
                crate::container_tag::sha32(&format!("{container_tag}\u{0}{canonical}"));
            tx.execute(
                "INSERT OR IGNORE INTO entities (id, canonical, container_tag, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![entity_id, canonical, container_tag, now_unix()],
            )?;
            tx.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?1, ?2)",
                params![memory_id, entity_id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    fn graph_search(
        &self,
        container_tag: &str,
        seed_ids: &[String],
        k: usize,
    ) -> Result<Vec<ScoredMemory>> {
        if seed_ids.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        // One global aggregation across all seeds in a single query. A per-seed LIMIT
        // would silently drop a neighbor weakly shared with many seeds but strong
        // overall, corrupting the top-k. Activation = a bounded saturating shared-entity
        // term + the strongest graph-edge bonus to any seed, so a directly-linked
        // neighbour outranks one merely sharing an entity. The MIN(...,5)/5 saturation is
        // a SQLite-native stand-in for VISION's `tanh(shared·0.5)` (same ranking; RRF
        // fuses by rank, not magnitude; avoids relying on the optional SQL math ext).
        // Seed ids are store-controlled and bound as parameters (4 groups: two edge
        // sides, the seed set, and the exclusion set).
        let ph = vec!["?"; seed_ids.len()].join(",");
        let sql = format!(
            "SELECT {MEMORY_COLS},
                (MIN(COUNT(DISTINCT me2.entity_id), 5) / 5.0)
                + MAX(CASE WHEN r.kind = 'extends' THEN 0.85
                           WHEN r.kind IS NOT NULL THEN 1.0
                           ELSE 0.0 END) AS score
             FROM memory_entities me1
             JOIN memory_entities me2 ON me1.entity_id = me2.entity_id
             JOIN memories m ON m.id = me2.memory_id
             LEFT JOIN relationships r
               ON (r.from_id = m.id AND r.to_id IN ({ph}))
               OR (r.to_id = m.id AND r.from_id IN ({ph}))
             WHERE me1.memory_id IN ({ph}) AND me2.memory_id NOT IN ({ph})
               AND m.container_tag = ? AND m.is_latest = 1
             GROUP BY m.id
             ORDER BY score DESC, m.id
             LIMIT ?"
        );
        let mut sql_params: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(seed_ids.len() * 4 + 2);
        for _ in 0..4 {
            sql_params.extend(seed_ids.iter().map(|s| s as &dyn rusqlite::ToSql));
        }
        sql_params.push(&container_tag);
        let k = k as i64;
        sql_params.push(&k);

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(sql_params.as_slice(), |row| {
            Ok(ScoredMemory {
                memory: row_to_memory(row)?,
                score: row.get::<_, f64>("score")? as f32,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    fn temporal_search(
        &self,
        container_tag: &str,
        window: (i64, Option<i64>),
        k: usize,
    ) -> Result<Vec<ScoredMemory>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let (w_start, w_end) = (window.0, window.1.unwrap_or(window.0));
        let w_mid = (w_start + w_end) / 2;
        // Interval-overlap: a memory's [occurred_start, occurred_end||start] must
        // intersect the query window [w_start, w_end]. Score = |midpoint distance|
        // (lower = nearer), so the closest occurrences fuse highest.
        let sql = format!(
            "SELECT {MEMORY_COLS},
                ABS(((m.occurred_start + COALESCE(m.occurred_end, m.occurred_start)) / 2) - ?4)
                  AS distance
             FROM memories m
             WHERE m.container_tag = ?1 AND m.is_latest = 1 AND m.occurred_start IS NOT NULL
               AND m.occurred_start <= ?3
               AND COALESCE(m.occurred_end, m.occurred_start) >= ?2
             ORDER BY distance ASC, m.id
             LIMIT ?5"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(
            params![container_tag, w_start, w_end, w_mid, k as i64],
            |row| {
                Ok(ScoredMemory {
                    memory: row_to_memory(row)?,
                    score: row.get::<_, f64>("distance")? as f32,
                })
            },
        )?;
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

    #[test]
    fn reinforce_increases_strength() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        s.upsert(&mem("m1", "x", "t", vec![1.0, 0.0])).unwrap();
        assert_eq!(s.get("m1").unwrap().unwrap().strength, 1.0);
        s.reinforce("m1", 0.5).unwrap();
        assert_eq!(s.get("m1").unwrap().unwrap().strength, 1.5);
        // Unknown id is a no-op, not an error.
        s.reinforce("nope", 1.0).unwrap();
    }

    #[test]
    fn forget_hides_from_retrieval_but_keeps_row() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        s.upsert(&mem("m1", "the user prefers tailwind", tag, vec![1.0, 0.0]))
            .unwrap();
        assert_eq!(s.count(tag).unwrap(), 1);

        s.forget("m1").unwrap();

        // Gone from every active read path...
        assert_eq!(s.count(tag).unwrap(), 0);
        assert!(s.knn(tag, &[1.0, 0.0], 5).unwrap().is_empty());
        assert!(s.text_search(tag, "tailwind", 5).unwrap().is_empty());
        assert!(s.list_latest(tag, 5).unwrap().is_empty());
        // ...but never hard-deleted.
        assert!(s.get("m1").unwrap().is_some());
    }

    #[test]
    fn edges_roundtrip_and_are_idempotent() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        s.upsert(&mem("a", "a", "t", vec![1.0, 0.0])).unwrap();
        s.upsert(&mem("b", "b", "t", vec![0.0, 1.0])).unwrap();
        s.add_edge("a", "b", EdgeKind::Extends).unwrap();
        s.add_edge("a", "b", EdgeKind::Extends).unwrap(); // duplicate ignored

        let edges = s.edges_from("a").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].to_id, "b");
        assert_eq!(edges[0].kind, EdgeKind::Extends);
        assert!(s.edges_from("b").unwrap().is_empty());
    }

    #[test]
    fn list_latest_orders_newest_first() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        let mut a = mem("a", "first", tag, vec![1.0, 0.0]);
        a.created_at = 100;
        let mut b = mem("b", "second", tag, vec![0.0, 1.0]);
        b.created_at = 200;
        s.upsert(&a).unwrap();
        s.upsert(&b).unwrap();
        let latest = s.list_latest(tag, 10).unwrap();
        assert_eq!(latest.len(), 2);
        assert_eq!(latest[0].id, "b");
        assert_eq!(latest[1].id, "a");
    }

    #[test]
    fn upsert_update_preserves_graph_edges() {
        // Re-upserting an existing node must NOT cascade-delete its relationships.
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        s.upsert(&mem("a", "a", "t", vec![1.0, 0.0])).unwrap();
        s.upsert(&mem("b", "b", "t", vec![0.0, 1.0])).unwrap();
        s.add_edge("a", "b", EdgeKind::Extends).unwrap();

        // Update both endpoints' content + embedding.
        s.upsert(&mem("a", "a-updated", "t", vec![0.5, 0.5]))
            .unwrap();
        s.upsert(&mem("b", "b-updated", "t", vec![0.2, 0.8]))
            .unwrap();

        let edges = s.edges_from("a").unwrap();
        assert_eq!(edges.len(), 1, "edge must survive upsert of its endpoints");
        assert_eq!(edges[0].to_id, "b");
        assert_eq!(s.get("a").unwrap().unwrap().content, "a-updated");
    }

    #[test]
    fn vision_columns_default_and_reinforce_bumps_access_count() {
        // The step-11 readiness columns round-trip with their defaults, and reinforce
        // accumulates real Hebbian access_count from day one.
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        s.upsert(&mem("m1", "x", "t", vec![1.0, 0.0])).unwrap();
        let m = s.get("m1").unwrap().unwrap();
        assert_eq!((m.proof_count, m.access_count), (1, 0));
        assert_eq!(m.stability, 1.0);
        assert!(m.parent_id.is_none() && m.root_id.is_none());
        assert!(m.occurred_start.is_none() && m.occurred_end.is_none());

        s.reinforce("m1", 0.5).unwrap();
        let m = s.get("m1").unwrap().unwrap();
        assert_eq!(m.access_count, 1, "reinforce bumps Hebbian access_count");
        assert!(m.strength > 1.0);
    }

    #[test]
    fn forget_does_not_starve_knn_top_k() {
        // The forgotten (nearest) memory must not occupy a KNN slot and crowd out a
        // still-latest neighbor.
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        s.upsert(&mem("near", "near", tag, vec![1.0, 0.0])).unwrap();
        s.upsert(&mem("mid", "mid", tag, vec![0.8, 0.2])).unwrap();
        s.forget("near").unwrap();

        // k = 1 against the query closest to the forgotten "near" still returns "mid".
        let hits = s.knn(tag, &[1.0, 0.0], 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.id, "mid");
    }

    #[test]
    fn text_search_tolerates_fts5_operators() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        s.upsert(&mem(
            "m1",
            "the user prefers rust over python",
            tag,
            vec![1.0, 0.0],
        ))
        .unwrap();
        // Queries that are invalid raw FTS5 (colon, leading dash, stray quote) must
        // not error; they sanitize to token phrases.
        assert_eq!(s.text_search(tag, "rust: -python", 5).unwrap().len(), 1);
        assert_eq!(s.text_search(tag, "\"rust", 5).unwrap().len(), 1);
        // No usable tokens → empty, not an error.
        assert!(s.text_search(tag, "   :-\"  ", 5).unwrap().is_empty());
        assert!(s.text_search(tag, "", 5).unwrap().is_empty());
    }

    #[test]
    fn list_scopes_reports_latest_and_total_counts() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        s.upsert(&mem("a1", "a one", "tag_a", vec![1.0, 0.0]))
            .unwrap();
        s.upsert(&mem("a2", "a two", "tag_a", vec![0.0, 1.0]))
            .unwrap();
        s.upsert(&mem("b1", "b one", "tag_b", vec![1.0, 0.0]))
            .unwrap();
        // Forgetting keeps the row (total) but drops it from latest.
        s.forget("a2").unwrap();

        let scopes = s.list_scopes().unwrap();
        assert_eq!(scopes.len(), 2);
        // tag_a: 1 latest, 2 total; tag_b: 1 latest, 1 total. Ordered by latest desc,
        // then total desc — so tag_a (more total) comes first.
        let a = scopes.iter().find(|s| s.tag == "tag_a").unwrap();
        assert_eq!((a.latest, a.total), (1, 2));
        let b = scopes.iter().find(|s| s.tag == "tag_b").unwrap();
        assert_eq!((b.latest, b.total), (1, 1));
        assert_eq!(scopes[0].tag, "tag_a");
    }

    #[test]
    fn graph_returns_all_versions_and_scoped_edges() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        s.upsert(&mem("a", "a", tag, vec![1.0, 0.0])).unwrap();
        s.upsert(&mem("b", "b", tag, vec![0.0, 1.0])).unwrap();
        // A node in another scope, plus an edge that must NOT appear in `t`'s graph.
        s.upsert(&mem("x", "x", "other", vec![1.0, 1.0])).unwrap();
        s.add_edge("a", "b", EdgeKind::Extends).unwrap();
        s.forget("b").unwrap(); // still a node (dimmed), not dropped

        let g = s.graph(tag, 100).unwrap();
        // Both nodes returned despite one being soft-forgotten.
        assert_eq!(g.nodes.len(), 2);
        assert!(g.nodes.iter().any(|m| m.id == "b" && !m.is_latest));
        // The one in-scope edge is returned; cross-scope nodes/edges are excluded.
        assert_eq!(g.edges.len(), 1);
        assert_eq!(
            (g.edges[0].from_id.as_str(), g.edges[0].to_id.as_str()),
            ("a", "b")
        );
    }

    #[test]
    fn graph_drops_edges_to_capped_out_nodes() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        // Three nodes with increasing created_at so the cap keeps the newest.
        let mut a = mem("a", "a", tag, vec![1.0, 0.0]);
        a.created_at = 100;
        let mut b = mem("b", "b", tag, vec![0.0, 1.0]);
        b.created_at = 200;
        s.upsert(&a).unwrap();
        s.upsert(&b).unwrap();
        s.add_edge("b", "a", EdgeKind::Extends).unwrap();

        // cap = 1 keeps only "b"; the b→a edge references a missing node and is dropped.
        let g = s.graph(tag, 1).unwrap();
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].id, "b");
        assert!(
            g.edges.is_empty(),
            "edge to a capped-out node must be dropped"
        );
    }

    #[test]
    fn readonly_store_reads_but_refuses_writes() {
        let mut path = std::env::temp_dir();
        path.push("memeora-readonly-test.db");
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
        {
            let mut s = SqliteStore::open(&path, 2).unwrap();
            s.upsert(&mem("m1", "the user prefers tailwind", "t", vec![1.0, 0.0]))
                .unwrap();
        }
        let reader = SqliteStore::open_readonly(&path, 2).unwrap();
        // Reads work through the read-only connection.
        assert_eq!(reader.count("t").unwrap(), 1);
        assert_eq!(reader.list_latest("t", 5).unwrap().len(), 1);
        assert_eq!(reader.list_scopes().unwrap().len(), 1);
        // Writes are refused (query_only), so the dashboard connection can't write.
        let mut reader = reader;
        assert!(reader.upsert(&mem("m2", "x", "t", vec![0.0, 1.0])).is_err());
        // A dim mismatch on reopen is caught without writing.
        assert!(matches!(
            SqliteStore::open_readonly(&path, 9),
            Err(Error::DimMismatch {
                expected: 2,
                got: 9
            })
        ));
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    #[test]
    fn reopening_with_a_different_dim_is_rejected() {
        let mut path = std::env::temp_dir();
        path.push("memeora-dim-reopen-test.db");
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }

        {
            let mut s = SqliteStore::open(&path, 3).unwrap();
            s.upsert(&mem("m1", "x", "t", vec![1.0, 0.0, 0.0])).unwrap();
        }
        // Reopening with the same dim is fine.
        assert!(SqliteStore::open(&path, 3).is_ok());
        // Reopening with a different dim is a loud error, not silent corruption.
        // (`SqliteStore` isn't `Debug`, so match the Result rather than `unwrap_err`.)
        assert!(matches!(
            SqliteStore::open(&path, 5),
            Err(Error::DimMismatch {
                expected: 3,
                got: 5
            })
        ));

        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    #[test]
    fn entities_link_and_resolve_shared() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        s.upsert(&mem("a", "about SqliteStore", tag, vec![1.0, 0.0]))
            .unwrap();
        s.upsert(&mem("b", "more SqliteStore design", tag, vec![0.0, 1.0]))
            .unwrap();
        s.upsert(&mem("c", "unrelated", tag, vec![1.0, 1.0]))
            .unwrap();
        s.link_entities("a", tag, &["sqlitestore".into(), "proof_count".into()])
            .unwrap();
        s.link_entities("b", tag, &["sqlitestore".into()]).unwrap();

        // "a" and "b" share one entity; "c" shares none — graph recall surfaces "b".
        let neighbors = |s: &SqliteStore, seed: &str| -> Vec<String> {
            s.graph_search(tag, &[seed.to_string()], 10)
                .unwrap()
                .into_iter()
                .map(|h| h.memory.id)
                .collect()
        };
        assert_eq!(neighbors(&s, "a"), vec!["b".to_string()]);
        assert!(neighbors(&s, "c").is_empty());

        // Linking is idempotent — re-linking doesn't duplicate the neighbor.
        s.link_entities("b", tag, &["sqlitestore".into()]).unwrap();
        assert_eq!(neighbors(&s, "a"), vec!["b".to_string()]);

        // A forgotten memory drops out (is_latest filter).
        s.forget("b").unwrap();
        assert!(neighbors(&s, "a").is_empty());
    }

    #[test]
    fn record_evidence_counts_distinct_sources_only() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        s.upsert(&mem("m1", "x", "t", vec![1.0, 0.0])).unwrap();
        // Originating observation (recorded by the ingest insert path).
        s.record_evidence("m1", "src-a", "x", 100).unwrap();
        assert_eq!(s.get("m1").unwrap().unwrap().proof_count, 1);

        // A distinct source corroborating raises the distinct-source count.
        s.record_evidence("m1", "src-b", "x restated", 200).unwrap();
        assert_eq!(
            s.get("m1").unwrap().unwrap().proof_count,
            2,
            "a distinct source raises proof_count"
        );

        // The same source again is a set-union no-op — one source can't inflate proof.
        s.record_evidence("m1", "src-a", "x again", 300).unwrap();
        assert_eq!(
            s.get("m1").unwrap().unwrap().proof_count,
            2,
            "re-recording a known source does not inflate proof_count"
        );
    }

    #[test]
    fn reinforce_caps_strength() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        s.upsert(&mem("m1", "x", "t", vec![1.0, 0.0])).unwrap();
        for _ in 0..100 {
            s.reinforce("m1", 1.0).unwrap();
        }
        assert!(
            s.get("m1").unwrap().unwrap().strength <= dynamics::STRENGTH_MAX,
            "Hebbian cap must bound runaway strength"
        );
    }

    #[test]
    fn reinforce_builds_stability_only_when_spaced() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let now = now_unix();
        let mut m = mem("m1", "x", "t", vec![1.0, 0.0]);
        m.last_accessed_at = now - dynamics::SPACING_SECS - 10; // last access > spacing ago
        s.upsert(&m).unwrap();
        let before = s.get("m1").unwrap().unwrap().stability;

        s.reinforce("m1", 0.5).unwrap(); // spaced → stability grows
        let spaced = s.get("m1").unwrap().unwrap().stability;
        assert!(spaced > before, "spaced reinforcement builds durability");

        s.reinforce("m1", 0.5).unwrap(); // immediate burst → no growth
        assert_eq!(
            s.get("m1").unwrap().unwrap().stability,
            spaced,
            "a rapid burst must not build stability"
        );
    }

    #[test]
    fn graph_search_finds_entity_neighbors() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        s.upsert(&mem("a", "x", tag, vec![1.0, 0.0])).unwrap();
        s.upsert(&mem("b", "x", tag, vec![0.0, 1.0])).unwrap();
        s.upsert(&mem("c", "x", tag, vec![1.0, 1.0])).unwrap();
        s.link_entities("a", tag, &["sqlitestore".into()]).unwrap();
        s.link_entities("b", tag, &["sqlitestore".into()]).unwrap();

        // Seed "a": "b" shares the entity (and isn't a seed); "c" shares nothing.
        let g = s.graph_search(tag, &["a".to_string()], 10).unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].memory.id, "b");
        // One shared entity, no edge: saturating term MIN(1,5)/5 = 0.2, bonus 0.
        assert!((g[0].score - 0.2).abs() < 1e-6, "{}", g[0].score);

        // Seeds are excluded from their own neighbor results.
        assert!(
            s.graph_search(tag, &["a".to_string(), "b".to_string()], 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn graph_search_aggregates_shared_count_across_seeds() {
        // A neighbor weakly linked to several seeds must accumulate its global score
        // (the bug a per-seed LIMIT would hide).
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        for id in ["s1", "s2", "n", "other"] {
            s.upsert(&mem(id, "x", tag, vec![1.0, 0.0])).unwrap();
        }
        s.link_entities("s1", tag, &["e1".into()]).unwrap();
        s.link_entities("s2", tag, &["e2".into()]).unwrap();
        s.link_entities("n", tag, &["e1".into(), "e2".into()])
            .unwrap(); // shares 1 with each seed

        let g = s
            .graph_search(tag, &["s1".to_string(), "s2".to_string()], 10)
            .unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].memory.id, "n");
        // Two distinct shared entities (one per seed): MIN(2,5)/5 = 0.4, no edge bonus.
        assert!(
            (g[0].score - 0.4).abs() < 1e-6,
            "shared entities aggregate across both seeds: {}",
            g[0].score
        );
    }

    #[test]
    fn supersede_links_chain_and_preserves_history() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        s.upsert(&mem("old", "I use MySQL", tag, vec![1.0, 0.0]))
            .unwrap();
        let new = mem("new", "I use Postgres now", tag, vec![0.0, 1.0]);
        assert!(s.supersede("old", &new).unwrap());

        // New is the current version, linked to old as both parent and lineage root.
        let n = s.get("new").unwrap().unwrap();
        assert!(n.is_latest);
        assert_eq!(n.parent_id.as_deref(), Some("old"));
        assert_eq!(n.root_id.as_deref(), Some("old"));
        // Old is soft-forgotten but preserved — never hard-deleted.
        assert!(!s.get("old").unwrap().unwrap().is_latest);
        // A new --updates--> old edge records the supersession.
        let edges = s.edges_from("new").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(
            (edges[0].to_id.as_str(), edges[0].kind),
            ("old", EdgeKind::Updates)
        );
        // Active retrieval sees only the current version...
        assert_eq!(s.count(tag).unwrap(), 1);
        assert_eq!(s.list_latest(tag, 10).unwrap()[0].id, "new");
        // ...but history shows the full lineage, newest first.
        assert_eq!(
            s.history("old")
                .unwrap()
                .iter()
                .map(|m| m.id.clone())
                .collect::<Vec<_>>(),
            vec!["new", "old"]
        );

        // A further correction keeps the original lineage root.
        assert!(
            s.supersede(
                "new",
                &mem("newer", "I use SQLite now", tag, vec![0.0, 0.0])
            )
            .unwrap()
        );
        let nw = s.get("newer").unwrap().unwrap();
        assert_eq!(nw.parent_id.as_deref(), Some("new"));
        assert_eq!(nw.root_id.as_deref(), Some("old"), "root stays the origin");
        assert_eq!(s.history("old").unwrap().len(), 3);

        // Unknown old id and self-supersession are both no-ops.
        assert!(
            !s.supersede("nope", &mem("z", "z", tag, vec![1.0, 0.0]))
                .unwrap()
        );
        assert!(!s.supersede("newer", &nw).unwrap());
    }

    #[test]
    fn graph_search_edge_link_outranks_entity_only() {
        // A neighbour directly graph-linked to a seed outranks one that merely shares
        // more entities — the edge bonus dominates the saturating shared term.
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        for id in ["a", "b", "c"] {
            s.upsert(&mem(id, "x", tag, vec![1.0, 0.0])).unwrap();
        }
        s.link_entities("a", tag, &["e1".into(), "e2".into()])
            .unwrap();
        s.link_entities("b", tag, &["e1".into()]).unwrap(); // shares 1 with the seed
        s.link_entities("c", tag, &["e1".into(), "e2".into()])
            .unwrap(); // shares 2
        s.add_edge("a", "b", EdgeKind::Extends).unwrap(); // but b is directly linked

        let g = s.graph_search(tag, &["a".to_string()], 10).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(
            g[0].memory.id, "b",
            "edge-linked neighbour ranks above entity-only"
        );
        assert!(g[0].score > g[1].score);
        assert_eq!(g[1].memory.id, "c");
    }

    #[test]
    fn temporal_search_overlaps_window_nearest_first() {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        let tag = "t";
        let day = 86_400;
        let base = 1_781_000_000;
        let dated = |id: &str, start: i64, end: i64| {
            let mut m = mem(id, "x", tag, vec![1.0, 0.0]);
            m.occurred_start = Some(start);
            m.occurred_end = Some(end);
            m
        };
        s.upsert(&dated("in", base, base + day)).unwrap(); // overlaps, midpoint at window mid
        s.upsert(&dated("near", base + day, base + 2 * day))
            .unwrap(); // overlaps at boundary
        s.upsert(&dated("out", base + 10 * day, base + 11 * day))
            .unwrap(); // no overlap
        s.upsert(&mem("undated", "x", tag, vec![1.0, 0.0])).unwrap(); // no occurred-time

        let g = s
            .temporal_search(tag, (base, Some(base + day)), 10)
            .unwrap();
        let ids: Vec<&str> = g.iter().map(|m| m.memory.id.as_str()).collect();
        assert_eq!(ids, vec!["in", "near"], "overlapping only, nearest-first");
    }
}
