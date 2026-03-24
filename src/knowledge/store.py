"""Knowledge base store — semantic search over ingested documents.

Uses ChromaDB with Ollama embeddings (same stack as session archive search).
Documents are chunked into ~500-token segments with overlap for better retrieval.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import TYPE_CHECKING

from ..logging import get_logger
from ..search.hybrid import reciprocal_rank_fusion

if TYPE_CHECKING:
    from ..search.embedder import LocalEmbedder
    from ..search.fts import FullTextIndex

log = get_logger("knowledge")

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

COLLECTION_NAME = "knowledge_docs"
CHUNK_SIZE = 1500  # chars per chunk (~375 tokens)
CHUNK_OVERLAP = 200  # overlap between chunks


class KnowledgeStore:
    """Semantic search over ingested documents (runbooks, configs, READMEs, etc.)."""

    def __init__(self, chromadb_path: str, fts_index: FullTextIndex | None = None) -> None:
        self._client = None
        self._collection = None
        self._fts = fts_index
        if not HAS_CHROMADB:
            log.warning("chromadb not installed — knowledge base disabled")
            return
        try:
            self._client = chromadb.PersistentClient(path=chromadb_path)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            count = self._collection.count()
            log.info("Knowledge base initialized (%d chunks indexed)", count)
        except Exception as e:
            log.error("Knowledge base init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._collection is not None

    def count(self) -> int:
        if not self.available:
            return 0
        return self._collection.count()

    async def ingest(
        self,
        content: str,
        source: str,
        embedder: LocalEmbedder,
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

        # Remove any existing chunks for this source (re-ingest replaces)
        try:
            existing = self._collection.get(
                where={"source": source},
            )
            if existing and existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                log.info("Removed %d old chunks for source '%s'", len(existing["ids"]), source)
        except Exception:
            pass  # Collection may not support where-delete on first use

        # Also clear FTS5
        if self._fts:
            self._fts.delete_knowledge_source(source)

        indexed = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_hash}_{i}"
            vector = await embedder.embed(chunk)
            if vector is None:
                log.warning("Failed to embed chunk %d of '%s'", i, source)
                continue

            try:
                self._collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=[vector],
                    metadatas=[{
                        "source": source,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "uploader": uploader,
                        "ingested_at": now,
                    }],
                )
                # Dual-write to FTS5
                if self._fts:
                    self._fts.index_knowledge_chunk(chunk_id, chunk, source, i)
                indexed += 1
            except Exception as e:
                log.error("Failed to index chunk %d of '%s': %s", i, source, e)

        log.info("Ingested '%s': %d/%d chunks indexed", source, indexed, len(chunks))
        return indexed

    async def search(
        self,
        query: str,
        embedder: LocalEmbedder,
        limit: int = 5,
    ) -> list[dict]:
        """Semantic search across the knowledge base.

        Returns list of dicts with: content, source, score, chunk_index.
        """
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
            log.warning("Knowledge search failed: %s", e)
            return []

        out = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return out

        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            # Cosine distance: 0 = identical, 2 = opposite. Skip poor matches.
            if distance > 0.8:
                continue

            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            doc = results["documents"][0][i] if results.get("documents") else ""

            out.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "score": round(1 - distance, 3),  # Convert to similarity (higher = better)
                "chunk_index": meta.get("chunk_index", 0),
            })

        return out

    def list_sources(self) -> list[dict]:
        """List all ingested document sources with metadata."""
        if not self.available:
            return []

        try:
            all_docs = self._collection.get(
                include=["metadatas"],
            )
        except Exception:
            return []

        # Group by source
        sources: dict[str, dict] = {}
        for meta in all_docs.get("metadatas", []):
            source = meta.get("source", "unknown")
            if source not in sources:
                sources[source] = {
                    "source": source,
                    "chunks": 0,
                    "uploader": meta.get("uploader", "unknown"),
                    "ingested_at": meta.get("ingested_at", ""),
                }
            sources[source]["chunks"] += 1

        return sorted(sources.values(), key=lambda s: s["source"])

    def delete_source(self, source: str) -> int:
        """Delete all chunks for a document source. Returns count deleted."""
        if not self.available:
            return 0

        try:
            existing = self._collection.get(
                where={"source": source},
            )
            if existing and existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                if self._fts:
                    self._fts.delete_knowledge_source(source)
                log.info("Deleted %d chunks for source '%s'", len(existing["ids"]), source)
                return len(existing["ids"])
        except Exception as e:
            log.error("Failed to delete source '%s': %s", source, e)
        return 0

    async def search_hybrid(
        self, query: str, embedder: LocalEmbedder, limit: int = 5,
    ) -> list[dict]:
        """Combined FTS5 + semantic search with Reciprocal Rank Fusion."""
        semantic_results = await self.search(query, embedder, limit=limit * 2)
        fts_results = []
        if self._fts:
            fts_results = self._fts.search_knowledge(query, limit=limit * 2)

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
        """Index existing ChromaDB knowledge chunks into FTS5. Returns count indexed."""
        if not self._fts or not self.available:
            return 0
        try:
            all_docs = self._collection.get(include=["documents", "metadatas"])
        except Exception:
            return 0

        count = 0
        for i, doc_id in enumerate(all_docs.get("ids", [])):
            if self._fts.has_knowledge_chunk(doc_id):
                continue
            meta = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
            doc = all_docs["documents"][i] if all_docs.get("documents") else ""
            if doc:
                self._fts.index_knowledge_chunk(
                    doc_id, doc, meta.get("source", ""), meta.get("chunk_index", 0),
                )
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
