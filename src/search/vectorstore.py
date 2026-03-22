from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging import get_logger
from .hybrid import reciprocal_rank_fusion

if TYPE_CHECKING:
    from .embedder import OllamaEmbedder
    from .fts import FullTextIndex

log = get_logger("search.vectorstore")

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

COLLECTION_NAME = "session_archives"
MAX_MESSAGES_PER_DOC = 20
MAX_MSG_CHARS = 500


class SessionVectorStore:
    def __init__(self, chromadb_path: str, fts_index: FullTextIndex | None = None) -> None:
        self._client = None
        self._collection = None
        self._fts = fts_index
        if not HAS_CHROMADB:
            log.warning("chromadb not installed — semantic search disabled")
            return
        try:
            self._client = chromadb.PersistentClient(path=chromadb_path)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            log.info("ChromaDB initialized at %s", chromadb_path)
        except Exception as e:
            log.error("ChromaDB init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._collection is not None

    async def index_session(self, archive_path: Path, embedder: OllamaEmbedder) -> bool:
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

        vector = await embedder.embed(doc_text)
        if vector is None:
            log.warning("Failed to embed archive %s", archive_path.name)
            return False

        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[doc_text],
                embeddings=[vector],
                metadatas=[{
                    "channel_id": str(data.get("channel_id", "")),
                    "last_active": float(data.get("last_active", 0)),
                    "message_count": len(data.get("messages", [])),
                }],
            )
            # Dual-write to FTS5
            if self._fts:
                self._fts.index_session(
                    doc_id, doc_text,
                    str(data.get("channel_id", "")),
                    float(data.get("last_active", 0)),
                )
            log.info("Indexed session %s for semantic search", doc_id)
            return True
        except Exception as e:
            log.error("ChromaDB upsert failed for %s: %s", doc_id, e)
            return False

    async def search(self, query: str, embedder: OllamaEmbedder, limit: int = 10) -> list[dict]:
        """Semantic search across archived sessions. Returns results in search_history format."""
        if not self.available:
            return []

        vector = await embedder.embed(query)
        if vector is None:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[vector],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.warning("ChromaDB query failed: %s", e)
            return []

        out = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return out

        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            doc = results["documents"][0][i] if results.get("documents") else ""
            distance = results["distances"][0][i] if results.get("distances") else 1.0

            # Cosine distance: 0 = identical, 2 = opposite. Skip poor matches.
            if distance > 1.0:
                continue

            out.append({
                "type": "semantic",
                "content": doc[:500],
                "timestamp": float(meta.get("last_active", 0)),
                "channel_id": str(meta.get("channel_id", "unknown")),
            })

        return out

    async def backfill(self, archive_dir: Path, embedder: OllamaEmbedder) -> int:
        """Index all archive JSONs not yet in ChromaDB. Returns count of newly indexed."""
        if not self.available:
            return 0
        if not archive_dir.exists():
            return 0

        # Get already-indexed IDs
        try:
            existing = set(self._collection.get()["ids"])
        except Exception:
            existing = set()

        count = 0
        for path in sorted(archive_dir.glob("*.json")):
            doc_id = path.stem
            if doc_id in existing:
                continue
            if await self.index_session(path, embedder):
                count += 1

        # Backfill FTS5 for any sessions already in ChromaDB but not in FTS
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
        self, query: str, embedder: OllamaEmbedder, limit: int = 10,
    ) -> list[dict]:
        """Combined FTS5 + semantic search with Reciprocal Rank Fusion."""
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
