"""Knowledge base store — semantic search over ingested documents.

Uses SQLite + sqlite-vec for vector storage and FTS5 for keyword search.
Documents are chunked into ~500-token segments with overlap for better retrieval.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from ..logging import get_logger
from ..search.hybrid import reciprocal_rank_fusion
from ..search.sqlite_vec import load_extension, serialize_vector

if TYPE_CHECKING:
    from ..search.embedder import LocalEmbedder
    from ..search.fts import FullTextIndex

log = get_logger("knowledge")

CHUNK_SIZE = 1500  # chars per chunk (~375 tokens)
CHUNK_OVERLAP = 200  # overlap between chunks
VECTOR_DIM = 384  # must match LocalEmbedder.DIMENSIONS


class KnowledgeStore:
    """Semantic search over ingested documents (runbooks, configs, READMEs, etc.)."""

    def __init__(self, db_path: str, fts_index: FullTextIndex | None = None) -> None:
        self._conn: sqlite3.Connection | None = None
        self._has_vec = False
        self._fts = fts_index
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            self._has_vec = load_extension(conn)
            if not self._has_vec:
                log.warning("sqlite-vec not available — vector search disabled, FTS-only mode")
            # Metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    uploader TEXT NOT NULL DEFAULT 'system',
                    ingested_at TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_chunks(source)"
            )
            # Vector table (only if sqlite-vec loaded)
            if self._has_vec:
                conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vec USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding float[{VECTOR_DIM}] distance_metric=cosine
                    )
                """)
            conn.commit()
            self._conn = conn
            count = self.count()
            log.info("Knowledge base initialized (%d chunks indexed)", count)
        except Exception as e:
            log.error("Knowledge base init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._conn is not None

    def count(self) -> int:
        if not self._conn:
            return 0
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    async def ingest(
        self,
        content: str,
        source: str,
        embedder: LocalEmbedder | None = None,
        *,
        uploader: str = "system",
    ) -> int:
        """Ingest a document by chunking and embedding it.

        Returns the number of chunks indexed.
        """
        if not self.available:
            return 0

        chunks = self._chunk_text(content)
        if not chunks:
            return 0

        # Generate a stable document ID from source name
        doc_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        now = datetime.now().isoformat()

        # Remove any existing chunks for this source (blocking → offload)
        await asyncio.to_thread(self.delete_source, source)

        # Embed all chunks first (async, non-blocking)
        vectors: list[list[float] | None] = []
        for chunk in chunks:
            if self._has_vec and embedder:
                vec = await embedder.embed(chunk)
                if vec is None:
                    log.warning("Failed to embed chunk %d of '%s'", len(vectors), source)
                vectors.append(vec)
            else:
                vectors.append(None)

        # Batch write metadata + vectors to DB (blocking → offload)
        indexed = await asyncio.to_thread(
            self._write_chunks_sync, chunks, vectors, doc_hash, source, now, uploader,
        )
        log.info("Ingested '%s': %d/%d chunks indexed", source, indexed, len(chunks))
        return indexed

    def _write_chunks_sync(
        self,
        chunks: list[str],
        vectors: list[list[float] | None],
        doc_hash: str,
        source: str,
        now: str,
        uploader: str,
    ) -> int:
        """Write chunk metadata, FTS entries, and vectors to database (sync)."""
        indexed = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_hash}_{i}"
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO knowledge_chunks "
                    "(chunk_id, content, source, chunk_index, total_chunks, uploader, ingested_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, chunk, source, i, len(chunks), uploader, now),
                )
                if self._fts:
                    self._fts.index_knowledge_chunk(chunk_id, chunk, source, i)
                if vectors[i] is not None:
                    vec_bytes = serialize_vector(vectors[i])
                    self._conn.execute(
                        "INSERT OR REPLACE INTO knowledge_vec (chunk_id, embedding) VALUES (?, ?)",
                        (chunk_id, vec_bytes),
                    )
                indexed += 1
            except Exception as e:
                log.error("Failed to index chunk %d of '%s': %s", i, source, e)
        self._conn.commit()
        return indexed

    async def search(
        self,
        query: str,
        embedder: LocalEmbedder | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Semantic search across the knowledge base.

        Returns list of dicts with: content, source, score, chunk_index.
        """
        if not self.available or not self._has_vec or not embedder:
            return []

        vector = await embedder.embed(query)
        if vector is None:
            return []

        try:
            vec_bytes = serialize_vector(vector)
            rows = await asyncio.to_thread(self._search_vec_sync, vec_bytes, limit)
        except Exception as e:
            log.warning("Knowledge search failed: %s", e)
            return []

        out = []
        for row in rows:
            distance = row[1]
            # Cosine distance: 0 = identical, higher = more different. Skip poor matches.
            if distance > 0.8:
                continue
            out.append({
                "content": row[2],
                "source": row[3],
                "score": round(1 - distance, 3),  # Convert to similarity
                "chunk_index": row[4],
            })

        return out

    def _search_vec_sync(self, vec_bytes: bytes, limit: int) -> list:
        """Execute vector similarity search (sync, for use with asyncio.to_thread)."""
        return self._conn.execute(
            """
            SELECT v.chunk_id, v.distance, c.content, c.source, c.chunk_index
            FROM knowledge_vec v
            JOIN knowledge_chunks c ON c.chunk_id = v.chunk_id
            WHERE v.embedding MATCH ?
            AND k = ?
            ORDER BY v.distance
            """,
            (vec_bytes, limit),
        ).fetchall()

    def list_sources(self) -> list[dict]:
        """List all ingested document sources with metadata."""
        if not self.available:
            return []

        try:
            rows = self._conn.execute(
                """
                SELECT source, COUNT(*) as chunks, uploader, MAX(ingested_at) as ingested_at
                FROM knowledge_chunks
                GROUP BY source
                ORDER BY source
                """
            ).fetchall()
        except Exception:
            return []

        results = []
        for r in rows:
            entry: dict = {
                "source": r[0],
                "chunks": r[1],
                "uploader": r[2],
                "ingested_at": r[3],
            }
            # Add preview from first chunk
            try:
                first = self._conn.execute(
                    "SELECT content FROM knowledge_chunks WHERE source = ? ORDER BY chunk_index LIMIT 1",
                    (r[0],),
                ).fetchone()
                if first and first[0]:
                    text = first[0][:200]
                    if len(first[0]) > 200:
                        text += "..."
                    entry["preview"] = text
            except Exception:
                pass
            results.append(entry)
        return results

    def get_source_chunks(self, source: str) -> list[dict]:
        """Get all chunks for a source with metadata for the chunk browser."""
        if not self.available:
            return []
        try:
            rows = self._conn.execute(
                "SELECT chunk_id, content, chunk_index, total_chunks, ingested_at "
                "FROM knowledge_chunks WHERE source = ? ORDER BY chunk_index",
                (source,),
            ).fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "content": r[1],
                    "chunk_index": r[2],
                    "total_chunks": r[3],
                    "ingested_at": r[4],
                    "char_count": len(r[1]) if r[1] else 0,
                }
                for r in rows
            ]
        except Exception:
            return []

    def get_source_content(self, source: str) -> str | None:
        """Get the full concatenated content of a source (for re-ingest)."""
        if not self.available:
            return None
        try:
            rows = self._conn.execute(
                "SELECT content FROM knowledge_chunks WHERE source = ? ORDER BY chunk_index",
                (source,),
            ).fetchall()
            if not rows:
                return None
            return "\n\n".join(r[0] for r in rows)
        except Exception:
            return None

    def delete_source(self, source: str) -> int:
        """Delete all chunks for a document source. Returns count deleted."""
        if not self.available:
            return 0

        try:
            # Get chunk IDs for this source
            ids = [
                r[0] for r in
                self._conn.execute(
                    "SELECT chunk_id FROM knowledge_chunks WHERE source = ?", (source,)
                ).fetchall()
            ]
            if not ids:
                return 0

            # Delete from vector table
            if self._has_vec:
                for chunk_id in ids:
                    self._conn.execute(
                        "DELETE FROM knowledge_vec WHERE chunk_id = ?", (chunk_id,)
                    )
            # Delete from chunks table
            self._conn.execute(
                "DELETE FROM knowledge_chunks WHERE source = ?", (source,)
            )
            self._conn.commit()
            # Delete from FTS
            if self._fts:
                self._fts.delete_knowledge_source(source)
            log.info("Deleted %d chunks for source '%s'", len(ids), source)
            return len(ids)
        except Exception as e:
            log.error("Failed to delete source '%s': %s", source, e)
        return 0

    async def search_hybrid(
        self, query: str, embedder: LocalEmbedder | None = None, limit: int = 5,
    ) -> list[dict]:
        """Combined FTS5 + semantic search with Reciprocal Rank Fusion.

        Works in FTS-only mode when embedder is None or vector search unavailable.
        """
        semantic_results = []
        if embedder:
            semantic_results = await self.search(query, embedder, limit=limit * 2)
        fts_results = []
        if self._fts:
            fts_results = await asyncio.to_thread(
                self._fts.search_knowledge, query, limit * 2,
            )

        if not semantic_results and not fts_results:
            return []

        # Normalize semantic results to use chunk_id
        for r in semantic_results:
            if "chunk_id" not in r:
                r["chunk_id"] = f"{r.get('source', '')}_{r.get('chunk_index', 0)}"

        return reciprocal_rank_fusion(
            semantic_results, fts_results, id_key="chunk_id", limit=limit,
        )

    def backfill_fts(self) -> int:
        """Index existing knowledge chunks into FTS5. Returns count indexed."""
        if not self._fts or not self.available:
            return 0
        try:
            rows = self._conn.execute(
                "SELECT chunk_id, content, source, chunk_index FROM knowledge_chunks"
            ).fetchall()
        except Exception:
            return 0

        count = 0
        for row in rows:
            chunk_id, content, source, chunk_index = row
            if self._fts.has_knowledge_chunk(chunk_id):
                continue
            if content:
                self._fts.index_knowledge_chunk(chunk_id, content, source, chunk_index)
                count += 1
        return count

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """Split text into overlapping chunks for embedding."""
        text = text.strip()
        if not text:
            return []

        # If short enough, return as single chunk
        if len(text) <= CHUNK_SIZE:
            return [text]

        chunks = []
        # Try to split on paragraph boundaries first
        paragraphs = re.split(r"\n\n+", text)

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= CHUNK_SIZE:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If a single paragraph is longer than chunk size, split it
                if len(para) > CHUNK_SIZE:
                    words = para.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= CHUNK_SIZE:
                            current_chunk = f"{current_chunk} {word}" if current_chunk else word
                        else:
                            chunks.append(current_chunk.strip())
                            # Overlap: keep last portion
                            overlap_start = max(0, len(current_chunk) - CHUNK_OVERLAP)
                            current_chunk = current_chunk[overlap_start:] + " " + word
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks
