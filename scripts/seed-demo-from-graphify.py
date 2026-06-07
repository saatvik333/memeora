#!/usr/bin/env python3
"""Seed a rich demo graph into memeora from a graphify knowledge graph.

Loads ``graphify-out/graph.json`` and writes its nodes + edges into the memeora
SQLite DB under the ``graphify`` container tag, so the dashboard has a real,
large network to show off (instead of a handful of hand-typed memories).

It writes straight to the `memories` + `relationships` tables (for the graph
view) and the `fts_memories` FTS5 index (so dashboard keyword search works). It
skips the `vec_memories` vector table — that needs the statically-linked
sqlite-vec extension which Python can't load — so these demo memories show up in
the graph and in lexical search, but not in semantic/vector recall, which is
fine for a visual demo.

Stop the daemon before running (it is the sole writer), then restart it:

    pkill -f memeora-daemon
    python3 scripts/seed-demo-from-graphify.py
    ./target/debug/memeora-daemon &
"""

import json
import os
import sqlite3
import sys
import time

TAG = "graphify"
# Color by community: the dashboard palette has three kinds, so communities cycle
# through them — clusters end up color-coded, which reads well under force layout.
KINDS = ["fact", "preference", "episode"]


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_path = os.path.join(root, "graphify-out", "graph.json")
    db_path = os.path.expanduser(os.environ.get("MEMEORA_DB", "~/.memeora/memory.db"))

    if not os.path.exists(graph_path):
        print(f"no graphify graph at {graph_path}; run /graphify first", file=sys.stderr)
        return 1
    if not os.path.exists(db_path):
        print(f"no memeora DB at {db_path}; start the daemon once first", file=sys.stderr)
        return 1

    g = json.load(open(graph_path, encoding="utf-8"))
    now = int(time.time())

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    cur = con.cursor()

    # Idempotent: drop a previous import. `fts_memories` is a virtual table with
    # no FK, so clear its rows for this tag *before* deleting the memories they
    # reference; deleting memories then cascades to relationships.
    cur.execute(
        "DELETE FROM fts_memories WHERE memory_id IN "
        "(SELECT id FROM memories WHERE container_tag = ?)",
        (TAG,),
    )
    cur.execute("DELETE FROM memories WHERE container_tag = ?", (TAG,))

    ids: set[str] = set()
    node_rows = []
    fts_rows = []
    for n in g["nodes"]:
        nid = n["id"]
        if nid in ids:
            continue
        ids.add(nid)
        content = n.get("label") or nid
        kind = KINDS[(n.get("community") or 0) % len(KINDS)]
        node_rows.append((nid, content, kind, TAG, 1, 1.0, now, now, None, "{}"))
        fts_rows.append((nid, content))

    cur.executemany(
        """INSERT INTO memories
           (id, content, kind, container_tag, is_latest, strength,
            created_at, last_accessed_at, expires_at, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        node_rows,
    )

    # Populate the FTS5 index so keyword search works in the dashboard. (The
    # `vec_memories` vector table still needs the native sqlite-vec extension,
    # which Python can't load, so semantic/vector recall stays empty for the demo
    # — but lexical search is enough to exercise the search UI.)
    cur.executemany(
        "INSERT INTO fts_memories (memory_id, content) VALUES (?, ?)",
        fts_rows,
    )

    # One undirected edge per connected pair (collapse direction + relation kinds
    # so the viz shows a single link between two nodes).
    seen: set[tuple[str, str]] = set()
    edge_rows = []
    for link in g["links"]:
        s, t = link["source"], link["target"]
        if s not in ids or t not in ids or s == t:
            continue
        key = (s, t) if s < t else (t, s)
        if key in seen:
            continue
        seen.add(key)
        edge_rows.append((key[0], key[1], "extends", now))

    cur.executemany(
        """INSERT OR IGNORE INTO relationships (from_id, to_id, kind, created_at)
           VALUES (?, ?, ?, ?)""",
        edge_rows,
    )

    con.commit()
    con.close()
    print(f"seeded scope '{TAG}': {len(node_rows)} nodes, {len(edge_rows)} edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
