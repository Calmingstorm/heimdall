"""Session archive vector store — semantic search over archived conversations.

Uses SQLite + sqlite-vec for vector storage and FTS5 for keyword search.
Archives are indexed when sessions are compacted, enabling cross-session search.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging import get_logger
from .hybrid import reciprocal_rank_fusion
from .sqlite_vec import load_extension, serialize_vector

if TYPE_CHECKING:
    from .embedder import LocalEmbedder
    from .fts import FullTextIndex

log = get_logger("search.vectorstore")

MAX_MESSAGES_PER_DOC = 20
MAX_MSG_CHARS = 500
VECTOR_DIM = 384  # must match LocalEmbedder.DIMENSIONS


class SessionVectorStore:
    """Semantic + FTS search over archived session conversations."""

    def __init__(self, db_path: str, fts_index: FullTextIndex | None = None) -> None:
        self._conn: sqlite3.Connection | None = None
        self._has_vec = False
        self._fts = fts_index
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            self._has_vec = load_extension(conn)
            if not self._has_vec:
                log.warning("sqlite-vec not available — semantic session search disabled, FTS-only mode")
            # Metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_archives (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    channel_id TEXT NOT NULL DEFAULT '',
                    last_active REAL NOT NULL DEFAULT 0,
                    message_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_channel ON session_archives(channel_id)"
            )
            # Vector table (only if sqlite-vec loaded)
            if self._has_vec:
                conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS session_vec USING vec0(
                        doc_id TEXT PRIMARY KEY,
                        embedding float[{VECTOR_DIM}] distance_metric=cosine
                    )
                """)
            conn.commit()
            self._conn = conn
            log.info("Session vector store initialized at %s (vec=%s)", db_path, self._has_vec)
        except Exception as e:
            log.error("Session vector store init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._conn is not None

    async def index_session(self, archive_path: Path, embedder: LocalEmbedder) -> bool:
        """Index a single archived session JSON. Returns True on success."""
        if not self.available:
            return False
        try:
            data = json.loads(archive_path.read_text())
        except Exception as e:
            log.error("Failed to read archive %s: %s", archive_path, e)
            return False

        doc_id = archive_path.stem  # e.g. "channelid_timestamp"
        doc_text = self._build_document_text(data)
        if not doc_text:
            return False

        channel_id = str(data.get("channel_id", ""))
        last_active = float(data.get("last_active", 0))
        message_count = len(data.get("messages", []))

        try:
            # Always write metadata
            self._conn.execute(
                "INSERT OR REPLACE INTO session_archives "
                "(doc_id, content, channel_id, last_active, message_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, doc_text, channel_id, last_active, message_count),
            )
            # Always write to FTS5 (decoupled from embedding success)
            if self._fts:
                self._fts.index_session(doc_id, doc_text, channel_id, last_active)

            # Try vector embedding (optional — FTS still works without it)
            if self._has_vec and embedder:
                vector = await embedder.embed(doc_text)
                if vector is not None:
                    vec_bytes = serialize_vector(vector)
                    self._conn.execute(
                        "INSERT OR REPLACE INTO session_vec (doc_id, embedding) VALUES (?, ?)",
                        (doc_id, vec_bytes),
                    )
                else:
                    log.warning("Failed to embed archive %s", archive_path.name)

            self._conn.commit()
            log.info("Indexed session %s for search", doc_id)
            return True
        except Exception as e:
            log.error("Session index failed for %s: %s", doc_id, e)
            return False

    async def search(self, query: str, embedder: LocalEmbedder, limit: int = 10) -> list[dict]:
        """Semantic search across archived sessions."""
        if not self.available or not self._has_vec:
            return []

        vector = await embedder.embed(query)
        if vector is None:
            return []

        try:
            vec_bytes = serialize_vector(vector)
            rows = self._conn.execute(
                """
                SELECT v.doc_id, v.distance, a.content, a.channel_id, a.last_active
                FROM session_vec v
                JOIN session_archives a ON a.doc_id = v.doc_id
                WHERE v.embedding MATCH ?
                AND k = ?
                ORDER BY v.distance
                """,
                (vec_bytes, limit),
            ).fetchall()
        except Exception as e:
            log.warning("Session vector search failed: %s", e)
            return []

        out = []
        for row in rows:
            distance = row[1]
            # Cosine distance: 0 = identical, higher = more different. Skip poor matches.
            if distance > 0.8:
                continue
            out.append({
                "type": "semantic",
                "content": row[2][:500],
                "timestamp": float(row[4]),
                "channel_id": str(row[3]),
            })

        return out

    async def backfill(self, archive_dir: Path, embedder: LocalEmbedder) -> int:
        """Index all archive JSONs not yet indexed. Returns count of newly indexed."""
        if not self.available:
            return 0
        if not archive_dir.exists():
            return 0

        # Get already-indexed IDs
        try:
            existing = {
                r[0] for r in
                self._conn.execute("SELECT doc_id FROM session_archives").fetchall()
            }
        except Exception:
            existing = set()

        count = 0
        for path in sorted(archive_dir.glob("*.json")):
            doc_id = path.stem
            if doc_id in existing:
                continue
            if await self.index_session(path, embedder):
                count += 1

        # Backfill FTS5 for any sessions in SQLite but not in FTS
        if self._fts:
            for path in sorted(archive_dir.glob("*.json")):
                doc_id = path.stem
                if self._fts.has_session(doc_id):
                    continue
                try:
                    data = json.loads(path.read_text())
                    doc_text = self._build_document_text(data)
                    if doc_text:
                        self._fts.index_session(
                            doc_id, doc_text,
                            str(data.get("channel_id", "")),
                            float(data.get("last_active", 0)),
                        )
                except Exception:
                    continue

        return count

    async def search_hybrid(
        self, query: str, embedder: LocalEmbedder | None, limit: int = 10,
    ) -> list[dict]:
        """Combined FTS5 + semantic search with Reciprocal Rank Fusion.

        Works in FTS-only mode when embedder is None or vector search unavailable.
        """
        semantic_results = []
        if embedder and self._has_vec:
            semantic_results = await self.search(query, embedder, limit=limit * 2)

        fts_results = []
        if self._fts:
            fts_results = self._fts.search_sessions(query, limit=limit * 2)

        if not semantic_results and not fts_results:
            return []

        # Normalize both to use doc_id as the merge key
        for r in semantic_results:
            if "doc_id" not in r:
                r["doc_id"] = f"{r.get('channel_id', '')}_{r.get('timestamp', 0)}"

        return reciprocal_rank_fusion(
            semantic_results, fts_results, id_key="doc_id", limit=limit,
        )

    @staticmethod
    def _build_document_text(data: dict) -> str:
        """Build embeddable text from archive data."""
        parts = []

        summary = data.get("summary", "")
        if summary:
            parts.append(f"Summary: {summary}")

        messages = data.get("messages", [])
        for msg in messages[:MAX_MESSAGES_PER_DOC]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:MAX_MSG_CHARS]
            parts.append(f"{role}: {content}")

        return "\n".join(parts)
