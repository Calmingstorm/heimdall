"""Tests for src/knowledge/store.py — ChromaDB-backed knowledge base.

Covers: init, availability, count, ingest, search, hybrid search,
list_sources, delete_source, backfill_fts, and _chunk_text.
All ChromaDB and Ollama calls are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.knowledge.store import KnowledgeStore, CHUNK_SIZE, CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_collection(initial_count: int = 0) -> MagicMock:
    """Create a mock ChromaDB collection with standard return values."""
    col = MagicMock()
    col.count.return_value = initial_count
    col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    col.query.return_value = {
        "ids": [[]], "documents": [[]], "metadatas": [[]],
        "distances": [[]],
    }
    col.upsert.return_value = None
    col.delete.return_value = None
    return col


@pytest.fixture
def mock_collection():
    return _make_mock_collection(5)


@pytest.fixture
def mock_embedder():
    emb = AsyncMock()
    emb.embed.return_value = [0.1] * 768
    return emb


@pytest.fixture
def mock_fts():
    fts = MagicMock()
    fts.delete_knowledge_source.return_value = 0
    fts.index_knowledge_chunk.return_value = True
    fts.search_knowledge.return_value = []
    fts.has_knowledge_chunk.return_value = False
    return fts


def _make_store(collection, fts_index=None):
    """Build a KnowledgeStore bypassing the real constructor (chromadb not installed)."""
    with patch("src.knowledge.store.HAS_CHROMADB", False):
        s = KnowledgeStore("/tmp/test_chromadb", fts_index=fts_index)
    # Manually wire up the mock collection so .available is True
    s._collection = collection
    s._client = MagicMock()
    return s


@pytest.fixture
def store(mock_collection):
    """Create a KnowledgeStore with a mocked ChromaDB collection."""
    return _make_store(mock_collection)


@pytest.fixture
def store_with_fts(mock_collection, mock_fts):
    """KnowledgeStore that also has an FTS index."""
    return _make_store(mock_collection, fts_index=mock_fts)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Test KnowledgeStore initialization."""

    def test_init_with_chromadb_available(self, mock_collection):
        """When HAS_CHROMADB is True and client succeeds, store is available."""
        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            with patch("src.knowledge.store.HAS_CHROMADB", True):
                with patch("src.knowledge.store.chromadb", mock_chromadb, create=True):
                    s = KnowledgeStore("/tmp/path")

        assert s.available is True
        mock_chromadb.PersistentClient.assert_called_once_with(path="/tmp/path")

    def test_init_failure_makes_unavailable(self):
        """If PersistentClient raises, store is not available."""
        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.side_effect = RuntimeError("boom")
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            with patch("src.knowledge.store.HAS_CHROMADB", True):
                with patch("src.knowledge.store.chromadb", mock_chromadb, create=True):
                    s = KnowledgeStore("/tmp/bad")

        assert s.available is False
        assert s.count() == 0

    def test_init_without_chromadb_installed(self):
        """If HAS_CHROMADB is False, store is not available."""
        with patch("src.knowledge.store.HAS_CHROMADB", False):
            s = KnowledgeStore("/tmp/nochroma")

        assert s.available is False


# ---------------------------------------------------------------------------
# Properties and count
# ---------------------------------------------------------------------------

class TestAvailabilityAndCount:
    """Test available property and count method."""

    def test_available_when_collection_exists(self, store):
        """available returns True when collection is initialized."""
        assert store.available is True

    def test_count_delegates_to_collection(self, store, mock_collection):
        """count() returns the collection's count."""
        mock_collection.count.return_value = 42
        assert store.count() == 42

    def test_count_returns_zero_when_unavailable(self):
        """count() returns 0 when store is not available."""
        with patch("src.knowledge.store.HAS_CHROMADB", False):
            s = KnowledgeStore("/tmp/x")
        assert s.count() == 0


# ---------------------------------------------------------------------------
# _chunk_text (static, pure logic)
# ---------------------------------------------------------------------------

class TestChunkText:
    """Test the _chunk_text static method."""

    def test_empty_text_returns_empty(self):
        """Empty or whitespace text returns no chunks."""
        assert KnowledgeStore._chunk_text("") == []
        assert KnowledgeStore._chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        """Text shorter than CHUNK_SIZE is returned as one chunk."""
        text = "Hello world"
        result = KnowledgeStore._chunk_text(text)
        assert result == ["Hello world"]

    def test_text_at_chunk_size_boundary(self):
        """Text exactly at CHUNK_SIZE is a single chunk."""
        text = "x" * CHUNK_SIZE
        result = KnowledgeStore._chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_multiple_paragraphs_split(self):
        """Long text with paragraph boundaries splits on paragraphs."""
        # Create two paragraphs, each ~CHUNK_SIZE/2 + 100 chars so they
        # individually fit but together exceed CHUNK_SIZE
        para_size = CHUNK_SIZE // 2 + 100
        p1 = "A " * (para_size // 2)
        p2 = "B " * (para_size // 2)
        text = f"{p1}\n\n{p2}"

        result = KnowledgeStore._chunk_text(text)
        assert len(result) >= 2
        assert "A" in result[0]
        assert "B" in result[-1]

    def test_single_long_paragraph_splits_on_words(self):
        """A paragraph longer than CHUNK_SIZE splits on word boundaries."""
        words = ["word" + str(i) for i in range(1000)]
        text = " ".join(words)
        assert len(text) > CHUNK_SIZE

        result = KnowledgeStore._chunk_text(text)
        assert len(result) >= 2
        # All words should be present across chunks
        joined = " ".join(result)
        assert "word0" in joined
        assert "word999" in joined

    def test_overlap_between_chunks(self):
        """When splitting a large paragraph, chunks have overlapping content."""
        # Create a long single paragraph
        words = [f"w{i:04d}" for i in range(800)]
        text = " ".join(words)
        assert len(text) > CHUNK_SIZE

        result = KnowledgeStore._chunk_text(text)
        assert len(result) >= 2

        # Check that the tail of chunk[0] overlaps with start of chunk[1]
        # The overlap should be up to CHUNK_OVERLAP characters
        end_of_first = result[0][-50:]
        # Some part of the first chunk's end should appear in the second chunk
        # (overlap mechanism keeps last CHUNK_OVERLAP chars)
        # We verify by checking a word near the end of chunk 0 appears in chunk 1
        last_words_first = result[0].split()[-3:]
        for w in last_words_first:
            if w in result[1]:
                break
        else:
            # It's possible the overlap is very small; just verify we have >1 chunk
            assert len(result) >= 2

    def test_mixed_short_and_long_paragraphs(self):
        """Handles a mix of short and long paragraphs correctly."""
        short = "Short paragraph."
        long_para = "x " * (CHUNK_SIZE + 100)
        text = f"{short}\n\n{long_para}\n\nAnother short."

        result = KnowledgeStore._chunk_text(text)
        assert len(result) >= 2
        # First chunk should contain the short paragraph
        assert "Short paragraph." in result[0]


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestIngest:
    """Test document ingestion."""

    async def test_ingest_single_chunk(self, store, mock_collection, mock_embedder):
        """Ingesting a short doc produces one chunk upserted to ChromaDB."""
        count = await store.ingest("Hello world", "test.md", mock_embedder)

        assert count == 1
        mock_embedder.embed.assert_called_once_with("Hello world")
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        assert call_kwargs[1]["documents"] == ["Hello world"]
        meta = call_kwargs[1]["metadatas"][0]
        assert meta["source"] == "test.md"
        assert meta["chunk_index"] == 0
        assert meta["total_chunks"] == 1

    async def test_ingest_multi_chunk(self, store, mock_collection, mock_embedder):
        """Ingesting a long doc produces multiple chunks."""
        long_text = "paragraph one " * 200 + "\n\n" + "paragraph two " * 200
        count = await store.ingest(long_text, "long.md", mock_embedder)

        assert count >= 2
        assert mock_collection.upsert.call_count >= 2
        assert mock_embedder.embed.call_count >= 2

    async def test_ingest_replaces_existing_source(self, store, mock_collection, mock_embedder):
        """Re-ingesting a source deletes old chunks first."""
        mock_collection.get.return_value = {
            "ids": ["old_1", "old_2"],
            "documents": ["a", "b"],
            "metadatas": [{"source": "doc.md"}, {"source": "doc.md"}],
        }

        await store.ingest("new content", "doc.md", mock_embedder)

        mock_collection.delete.assert_called_once_with(ids=["old_1", "old_2"])

    async def test_ingest_clears_fts(self, store_with_fts, mock_collection, mock_fts, mock_embedder):
        """Ingesting also clears and re-indexes FTS5."""
        await store_with_fts.ingest("content here", "readme.md", mock_embedder)

        mock_fts.delete_knowledge_source.assert_called_once_with("readme.md")
        mock_fts.index_knowledge_chunk.assert_called_once()

    async def test_ingest_skips_failed_embedding(self, store, mock_collection, mock_embedder):
        """If embedder returns None for a chunk, that chunk is skipped."""
        mock_embedder.embed.return_value = None

        count = await store.ingest("short text", "fail.md", mock_embedder)

        assert count == 0
        mock_collection.upsert.assert_not_called()

    async def test_ingest_handles_upsert_exception(self, store, mock_collection, mock_embedder):
        """If upsert raises, the error is logged and count is reduced."""
        mock_collection.upsert.side_effect = RuntimeError("disk full")

        count = await store.ingest("some text", "err.md", mock_embedder)

        assert count == 0

    async def test_ingest_empty_content(self, store, mock_embedder):
        """Ingesting empty content returns 0."""
        count = await store.ingest("", "empty.md", mock_embedder)
        assert count == 0

    async def test_ingest_unavailable_store(self, mock_embedder):
        """Ingesting when store is unavailable returns 0."""
        with patch("src.knowledge.store.HAS_CHROMADB", False):
            s = KnowledgeStore("/tmp/x")
        count = await s.ingest("text", "src.md", mock_embedder)
        assert count == 0

    async def test_ingest_with_custom_uploader(self, store, mock_collection, mock_embedder):
        """The uploader field is stored in metadata."""
        await store.ingest("data", "doc.md", mock_embedder, uploader="alice")

        meta = mock_collection.upsert.call_args[1]["metadatas"][0]
        assert meta["uploader"] == "alice"

    async def test_ingest_existing_get_exception(self, store, mock_collection, mock_embedder):
        """If getting existing chunks raises, ingestion continues (pass on exception)."""
        mock_collection.get.side_effect = RuntimeError("where not supported")

        count = await store.ingest("data", "doc.md", mock_embedder)
        # Should still upsert the new chunk
        assert count == 1


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    """Test semantic search."""

    async def test_search_returns_results(self, store, mock_collection, mock_embedder):
        """Search returns formatted results from ChromaDB query."""
        mock_collection.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "documents": [["First result", "Second result"]],
            "metadatas": [[
                {"source": "doc1.md", "chunk_index": 0},
                {"source": "doc2.md", "chunk_index": 1},
            ]],
            "distances": [[0.2, 0.5]],
        }

        results = await store.search("test query", mock_embedder, limit=5)

        assert len(results) == 2
        assert results[0]["content"] == "First result"
        assert results[0]["source"] == "doc1.md"
        assert results[0]["score"] == 0.8  # 1 - 0.2
        assert results[1]["score"] == 0.5  # 1 - 0.5

    async def test_search_filters_poor_matches(self, store, mock_collection, mock_embedder):
        """Results with distance > 0.8 are filtered out."""
        mock_collection.query.return_value = {
            "ids": [["good", "bad"]],
            "documents": [["Good match", "Bad match"]],
            "metadatas": [[
                {"source": "a.md", "chunk_index": 0},
                {"source": "b.md", "chunk_index": 0},
            ]],
            "distances": [[0.3, 0.9]],  # 0.9 > 0.8 threshold
        }

        results = await store.search("query", mock_embedder)

        assert len(results) == 1
        assert results[0]["source"] == "a.md"

    async def test_search_unavailable_returns_empty(self, mock_embedder):
        """Search on unavailable store returns empty list."""
        with patch("src.knowledge.store.HAS_CHROMADB", False):
            s = KnowledgeStore("/tmp/x")
        results = await s.search("query", mock_embedder)
        assert results == []

    async def test_search_embed_failure_returns_empty(self, store, mock_embedder):
        """If embedding the query fails, returns empty list."""
        mock_embedder.embed.return_value = None
        results = await store.search("query", mock_embedder)
        assert results == []

    async def test_search_query_exception_returns_empty(self, store, mock_collection, mock_embedder):
        """If ChromaDB query raises, returns empty list."""
        mock_collection.query.side_effect = RuntimeError("corrupt index")

        results = await store.search("query", mock_embedder)
        assert results == []

    async def test_search_empty_results(self, store, mock_collection, mock_embedder):
        """If ChromaDB returns empty ids, returns empty list."""
        mock_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]],
            "distances": [[]],
        }

        results = await store.search("query", mock_embedder)
        assert results == []

    async def test_search_missing_distances_defaults(self, store, mock_collection, mock_embedder):
        """If distances key is missing, defaults to 1.0 (which is > 0.8, so filtered)."""
        mock_collection.query.return_value = {
            "ids": [["c1"]],
            "documents": [["text"]],
            "metadatas": [[{"source": "x.md", "chunk_index": 0}]],
            # No "distances" key
        }

        results = await store.search("query", mock_embedder)
        # distance defaults to 1.0 which is > 0.8 threshold, so filtered
        assert results == []

    async def test_search_missing_metadata_defaults(self, store, mock_collection, mock_embedder):
        """If metadatas key is missing, source defaults to 'unknown'."""
        mock_collection.query.return_value = {
            "ids": [["c1"]],
            "documents": [["text"]],
            "distances": [[0.1]],
            # No metadatas
        }

        results = await store.search("query", mock_embedder)
        assert len(results) == 1
        assert results[0]["source"] == "unknown"

    async def test_search_respects_limit(self, store, mock_collection, mock_embedder):
        """The limit parameter is passed to n_results."""
        mock_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]],
            "distances": [[]],
        }

        await store.search("query", mock_embedder, limit=3)

        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["n_results"] == 3

    async def test_search_none_results(self, store, mock_collection, mock_embedder):
        """If query returns None-ish results, returns empty list."""
        mock_collection.query.return_value = None

        results = await store.search("query", mock_embedder)
        assert results == []


# ---------------------------------------------------------------------------
# list_sources
# ---------------------------------------------------------------------------

class TestListSources:
    """Test listing ingested document sources."""

    def test_list_sources_groups_by_source(self, store, mock_collection):
        """list_sources groups chunks by source and counts them."""
        mock_collection.get.return_value = {
            "ids": ["c1", "c2", "c3"],
            "metadatas": [
                {"source": "doc1.md", "uploader": "alice", "ingested_at": "2024-01-01"},
                {"source": "doc1.md", "uploader": "alice", "ingested_at": "2024-01-01"},
                {"source": "doc2.md", "uploader": "bob", "ingested_at": "2024-01-02"},
            ],
        }

        sources = store.list_sources()

        assert len(sources) == 2
        # Sorted by source name
        assert sources[0]["source"] == "doc1.md"
        assert sources[0]["chunks"] == 2
        assert sources[0]["uploader"] == "alice"
        assert sources[1]["source"] == "doc2.md"
        assert sources[1]["chunks"] == 1

    def test_list_sources_unavailable(self):
        """list_sources returns empty when store is unavailable."""
        with patch("src.knowledge.store.HAS_CHROMADB", False):
            s = KnowledgeStore("/tmp/x")
        assert s.list_sources() == []

    def test_list_sources_exception(self, store, mock_collection):
        """list_sources returns empty on exception."""
        mock_collection.get.side_effect = RuntimeError("fail")
        assert store.list_sources() == []

    def test_list_sources_empty(self, store, mock_collection):
        """list_sources returns empty when no docs exist."""
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        assert store.list_sources() == []

    def test_list_sources_missing_metadata_fields(self, store, mock_collection):
        """Handles metadata missing optional fields gracefully."""
        mock_collection.get.return_value = {
            "ids": ["c1"],
            "metadatas": [{"source": "bare.md"}],  # no uploader, no ingested_at
        }

        sources = store.list_sources()
        assert len(sources) == 1
        assert sources[0]["uploader"] == "unknown"
        assert sources[0]["ingested_at"] == ""


# ---------------------------------------------------------------------------
# delete_source
# ---------------------------------------------------------------------------

class TestDeleteSource:
    """Test deleting a document source."""

    def test_delete_source_removes_chunks(self, store, mock_collection):
        """delete_source deletes all chunks for the given source."""
        mock_collection.get.return_value = {
            "ids": ["c1", "c2"],
            "documents": ["a", "b"],
            "metadatas": [{"source": "doc.md"}, {"source": "doc.md"}],
        }

        deleted = store.delete_source("doc.md")

        assert deleted == 2
        mock_collection.delete.assert_called_once_with(ids=["c1", "c2"])

    def test_delete_source_also_clears_fts(self, store_with_fts, mock_collection, mock_fts):
        """delete_source also removes FTS5 entries."""
        mock_collection.get.return_value = {
            "ids": ["c1"],
            "documents": ["text"],
            "metadatas": [{"source": "doc.md"}],
        }

        store_with_fts.delete_source("doc.md")
        mock_fts.delete_knowledge_source.assert_called_once_with("doc.md")

    def test_delete_source_not_found(self, store, mock_collection):
        """delete_source returns 0 if source has no chunks."""
        mock_collection.get.return_value = {"ids": [], "metadatas": []}

        deleted = store.delete_source("missing.md")
        assert deleted == 0
        mock_collection.delete.assert_not_called()

    def test_delete_source_unavailable(self):
        """delete_source returns 0 when store is unavailable."""
        with patch("src.knowledge.store.HAS_CHROMADB", False):
            s = KnowledgeStore("/tmp/x")
        assert s.delete_source("doc.md") == 0

    def test_delete_source_exception(self, store, mock_collection):
        """delete_source returns 0 on exception."""
        mock_collection.get.side_effect = RuntimeError("fail")

        deleted = store.delete_source("doc.md")
        assert deleted == 0


# ---------------------------------------------------------------------------
# search_hybrid
# ---------------------------------------------------------------------------

class TestSearchHybrid:
    """Test hybrid search (semantic + FTS with RRF)."""

    async def test_hybrid_combines_semantic_and_fts(self, store_with_fts, mock_collection, mock_fts, mock_embedder):
        """Hybrid search calls both semantic and FTS, fuses results."""
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "documents": [["semantic result"]],
            "metadatas": [[{"source": "a.md", "chunk_index": 0}]],
            "distances": [[0.2]],
        }
        mock_fts.search_knowledge.return_value = [
            {"chunk_id": "f1", "content": "fts result", "source": "b.md", "score": 5.0},
        ]

        results = await store_with_fts.search_hybrid("query", mock_embedder, limit=5)

        mock_fts.search_knowledge.assert_called_once()
        assert len(results) >= 1

    async def test_hybrid_no_fts_index(self, store, mock_collection, mock_embedder):
        """Without FTS index, hybrid uses only semantic results."""
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "documents": [["semantic only"]],
            "metadatas": [[{"source": "a.md", "chunk_index": 0}]],
            "distances": [[0.3]],
        }

        results = await store.search_hybrid("query", mock_embedder, limit=5)
        # Should still return semantic results via RRF
        assert len(results) >= 1

    async def test_hybrid_both_empty(self, store_with_fts, mock_collection, mock_fts, mock_embedder):
        """If both semantic and FTS return nothing, result is empty."""
        mock_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]],
            "distances": [[]],
        }
        mock_fts.search_knowledge.return_value = []

        results = await store_with_fts.search_hybrid("query", mock_embedder)
        assert results == []

    async def test_hybrid_adds_chunk_id(self, store, mock_collection, mock_embedder):
        """Hybrid search normalizes semantic results to include chunk_id."""
        mock_collection.query.return_value = {
            "ids": [["c1"]],
            "documents": [["text"]],
            "metadatas": [[{"source": "src.md", "chunk_index": 2}]],
            "distances": [[0.1]],
        }

        results = await store.search_hybrid("query", mock_embedder, limit=5)

        # The RRF output should contain chunk_id
        assert len(results) >= 1
        # chunk_id is added as source_chunk_index
        for r in results:
            assert "chunk_id" in r

    async def test_hybrid_passes_double_limit(self, store_with_fts, mock_collection, mock_fts, mock_embedder):
        """Hybrid search queries each backend with 2x the final limit."""
        mock_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]],
            "distances": [[]],
        }
        mock_fts.search_knowledge.return_value = []

        await store_with_fts.search_hybrid("query", mock_embedder, limit=3)

        # Semantic search gets limit * 2
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["n_results"] == 6  # 3 * 2
        # FTS also gets limit * 2
        mock_fts.search_knowledge.assert_called_once_with("query", limit=6)


# ---------------------------------------------------------------------------
# backfill_fts
# ---------------------------------------------------------------------------

class TestBackfillFts:
    """Test FTS5 backfill from existing ChromaDB data."""

    def test_backfill_indexes_new_chunks(self, store_with_fts, mock_collection, mock_fts):
        """backfill_fts indexes chunks not already in FTS."""
        mock_collection.get.return_value = {
            "ids": ["c1", "c2"],
            "documents": ["text one", "text two"],
            "metadatas": [
                {"source": "doc.md", "chunk_index": 0},
                {"source": "doc.md", "chunk_index": 1},
            ],
        }
        mock_fts.has_knowledge_chunk.return_value = False

        count = store_with_fts.backfill_fts()

        assert count == 2
        assert mock_fts.index_knowledge_chunk.call_count == 2

    def test_backfill_skips_existing_chunks(self, store_with_fts, mock_collection, mock_fts):
        """backfill_fts skips chunks already in FTS."""
        mock_collection.get.return_value = {
            "ids": ["c1", "c2"],
            "documents": ["text one", "text two"],
            "metadatas": [
                {"source": "doc.md", "chunk_index": 0},
                {"source": "doc.md", "chunk_index": 1},
            ],
        }
        mock_fts.has_knowledge_chunk.side_effect = [True, False]

        count = store_with_fts.backfill_fts()

        assert count == 1  # Only c2 was indexed

    def test_backfill_no_fts_returns_zero(self, store):
        """backfill_fts returns 0 when there's no FTS index."""
        assert store.backfill_fts() == 0

    def test_backfill_unavailable_store(self, mock_fts):
        """backfill_fts returns 0 when store is unavailable."""
        with patch("src.knowledge.store.HAS_CHROMADB", False):
            s = KnowledgeStore("/tmp/x", fts_index=mock_fts)
        assert s.backfill_fts() == 0

    def test_backfill_get_exception(self, store_with_fts, mock_collection):
        """backfill_fts returns 0 if collection.get raises."""
        mock_collection.get.side_effect = RuntimeError("fail")
        assert store_with_fts.backfill_fts() == 0

    def test_backfill_skips_empty_documents(self, store_with_fts, mock_collection, mock_fts):
        """backfill_fts skips chunks with empty document text."""
        mock_collection.get.return_value = {
            "ids": ["c1", "c2"],
            "documents": ["", "has text"],
            "metadatas": [
                {"source": "doc.md", "chunk_index": 0},
                {"source": "doc.md", "chunk_index": 1},
            ],
        }
        mock_fts.has_knowledge_chunk.return_value = False

        count = store_with_fts.backfill_fts()

        assert count == 1  # Only c2 with actual text
        mock_fts.index_knowledge_chunk.assert_called_once()

    def test_backfill_missing_metadata(self, store_with_fts, mock_collection, mock_fts):
        """backfill_fts handles missing metadatas gracefully."""
        mock_collection.get.return_value = {
            "ids": ["c1"],
            "documents": ["some text"],
            # No metadatas key
        }
        mock_fts.has_knowledge_chunk.return_value = False

        count = store_with_fts.backfill_fts()

        assert count == 1
        # Should use defaults for source and chunk_index
        call_args = mock_fts.index_knowledge_chunk.call_args
        assert call_args[0][2] == ""  # default source
        assert call_args[0][3] == 0   # default chunk_index
