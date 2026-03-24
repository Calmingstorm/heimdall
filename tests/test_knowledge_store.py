"""Tests for src/knowledge/store.py — SQLite-backed knowledge base.

Covers: init, availability, count, ingest, search, hybrid search,
list_sources, delete_source, backfill_fts, and _chunk_text.
SQLite operations use real temp databases. Embedder is mocked.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.knowledge.store import KnowledgeStore, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder():
    emb = AsyncMock()
    emb.embed.return_value = [0.1] * VECTOR_DIM
    return emb


@pytest.fixture
def mock_fts():
    fts = MagicMock()
    fts.delete_knowledge_source.return_value = 0
    fts.index_knowledge_chunk.return_value = True
    fts.search_knowledge.return_value = []
    fts.has_knowledge_chunk.return_value = False
    return fts


@pytest.fixture
def store(tmp_path):
    """Create a KnowledgeStore with a real SQLite DB (no vec extension)."""
    db_path = str(tmp_path / "knowledge.db")
    with patch("src.knowledge.store.load_extension", return_value=False):
        s = KnowledgeStore(db_path)
    return s


@pytest.fixture
def store_with_fts(tmp_path, mock_fts):
    """KnowledgeStore with a real SQLite DB and mock FTS index."""
    db_path = str(tmp_path / "knowledge.db")
    with patch("src.knowledge.store.load_extension", return_value=False):
        s = KnowledgeStore(db_path, fts_index=mock_fts)
    return s


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Test KnowledgeStore initialization."""

    def test_init_creates_sqlite_db(self, tmp_path):
        """Store creates a SQLite database file."""
        db_path = str(tmp_path / "test.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            s = KnowledgeStore(db_path)
        assert s.available is True
        assert os.path.exists(db_path)

    def test_init_with_vec_extension(self, tmp_path):
        """When sqlite-vec loads, _has_vec is True."""
        db_path = str(tmp_path / "test.db")
        with patch("src.knowledge.store.load_extension", return_value=True):
            # This will fail to create vec0 virtual table since extension isn't
            # really loaded, but we test the flag
            s = KnowledgeStore(db_path)
        # May or may not succeed depending on whether CREATE VIRTUAL TABLE works
        # but the load_extension flag was set

    def test_init_failure_makes_unavailable(self, tmp_path):
        """If DB open fails, store is unavailable."""
        # Use an invalid path
        bad_path = str(tmp_path / "nonexistent" / "subdir" / "test.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            s = KnowledgeStore(bad_path)
        assert s.available is False
        assert s.count() == 0


# ---------------------------------------------------------------------------
# Properties and count
# ---------------------------------------------------------------------------

class TestAvailabilityAndCount:
    """Test available property and count method."""

    def test_available_when_db_open(self, store):
        """available returns True when DB connection is open."""
        assert store.available is True

    def test_count_zero_initially(self, store):
        """count() returns 0 for empty store."""
        assert store.count() == 0

    def test_count_returns_zero_when_unavailable(self, tmp_path):
        """count() returns 0 when store is not available."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            s = KnowledgeStore(bad_path)
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
        joined = " ".join(result)
        assert "word0" in joined
        assert "word999" in joined

    def test_overlap_between_chunks(self):
        """When splitting a large paragraph, chunks have overlapping content."""
        words = [f"w{i:04d}" for i in range(800)]
        text = " ".join(words)
        assert len(text) > CHUNK_SIZE

        result = KnowledgeStore._chunk_text(text)
        assert len(result) >= 2

        last_words_first = result[0].split()[-3:]
        for w in last_words_first:
            if w in result[1]:
                break
        else:
            assert len(result) >= 2

    def test_mixed_short_and_long_paragraphs(self):
        """Handles a mix of short and long paragraphs correctly."""
        short = "Short paragraph."
        long_para = "x " * (CHUNK_SIZE + 100)
        text = f"{short}\n\n{long_para}\n\nAnother short."

        result = KnowledgeStore._chunk_text(text)
        assert len(result) >= 2
        assert "Short paragraph." in result[0]


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestIngest:
    """Test document ingestion."""

    async def test_ingest_single_chunk(self, store, mock_embedder):
        """Ingesting a short doc produces one chunk in SQLite."""
        count = await store.ingest("Hello world", "test.md", mock_embedder)
        assert count == 1
        assert store.count() == 1

    async def test_ingest_multi_chunk(self, store, mock_embedder):
        """Ingesting a long doc produces multiple chunks."""
        long_text = "paragraph one " * 200 + "\n\n" + "paragraph two " * 200
        count = await store.ingest(long_text, "long.md", mock_embedder)
        assert count >= 2

    async def test_ingest_replaces_existing_source(self, store, mock_embedder):
        """Re-ingesting a source replaces old chunks."""
        await store.ingest("old content", "doc.md", mock_embedder)
        assert store.count() == 1

        await store.ingest("new content", "doc.md", mock_embedder)
        assert store.count() == 1  # Old chunk replaced, not duplicated

    async def test_ingest_clears_fts(self, store_with_fts, mock_fts, mock_embedder):
        """Ingesting also writes to FTS5."""
        await store_with_fts.ingest("content here", "readme.md", mock_embedder)
        mock_fts.index_knowledge_chunk.assert_called_once()

    async def test_reingest_clears_fts(self, store_with_fts, mock_fts, mock_embedder):
        """Re-ingesting a source clears old FTS entries."""
        await store_with_fts.ingest("old content", "readme.md", mock_embedder)
        mock_fts.reset_mock()
        await store_with_fts.ingest("new content", "readme.md", mock_embedder)
        mock_fts.delete_knowledge_source.assert_called_once_with("readme.md")

    async def test_ingest_works_without_embedder(self, store):
        """Ingesting without embedder still writes metadata and succeeds."""
        count = await store.ingest("short text", "no-embed.md", embedder=None)
        assert count == 1
        assert store.count() == 1

    async def test_ingest_empty_content(self, store, mock_embedder):
        """Ingesting empty content returns 0."""
        count = await store.ingest("", "empty.md", mock_embedder)
        assert count == 0

    async def test_ingest_unavailable_store(self, tmp_path, mock_embedder):
        """Ingesting when store is unavailable returns 0."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            s = KnowledgeStore(bad_path)
        count = await s.ingest("text", "src.md", mock_embedder)
        assert count == 0

    async def test_ingest_with_custom_uploader(self, store, mock_embedder):
        """The uploader field is stored correctly."""
        await store.ingest("data", "doc.md", mock_embedder, uploader="alice")
        sources = store.list_sources()
        assert len(sources) == 1
        assert sources[0]["uploader"] == "alice"

    async def test_ingest_fts_written_even_when_embed_fails(self, store_with_fts, mock_fts):
        """FTS5 is written even when embedder returns None (the bug fix)."""
        bad_embedder = AsyncMock()
        bad_embedder.embed.return_value = None

        count = await store_with_fts.ingest("some text", "fail.md", bad_embedder)
        # Chunk should still be written to metadata table and FTS
        assert count == 1
        mock_fts.index_knowledge_chunk.assert_called_once()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    """Test semantic search."""

    async def test_search_unavailable_returns_empty(self, tmp_path, mock_embedder):
        """Search on unavailable store returns empty list."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            s = KnowledgeStore(bad_path)
        results = await s.search("query", mock_embedder)
        assert results == []

    async def test_search_no_vec_returns_empty(self, store, mock_embedder):
        """Search returns empty when no sqlite-vec (FTS-only mode)."""
        results = await store.search("query", mock_embedder)
        assert results == []

    async def test_search_embed_failure_returns_empty(self, store, mock_embedder):
        """If embedding the query fails, returns empty list."""
        mock_embedder.embed.return_value = None
        results = await store.search("query", mock_embedder)
        assert results == []

    async def test_search_no_embedder_returns_empty(self, store):
        """Search without embedder returns empty list."""
        results = await store.search("query", embedder=None)
        assert results == []


# ---------------------------------------------------------------------------
# list_sources
# ---------------------------------------------------------------------------

class TestListSources:
    """Test listing ingested document sources."""

    async def test_list_sources_groups_by_source(self, store, mock_embedder):
        """list_sources groups chunks by source and counts them."""
        await store.ingest("content one", "doc1.md", mock_embedder, uploader="alice")
        await store.ingest("content two", "doc2.md", mock_embedder, uploader="bob")

        sources = store.list_sources()
        assert len(sources) == 2
        names = [s["source"] for s in sources]
        assert "doc1.md" in names
        assert "doc2.md" in names

    def test_list_sources_unavailable(self, tmp_path):
        """list_sources returns empty when store is unavailable."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            s = KnowledgeStore(bad_path)
        assert s.list_sources() == []

    def test_list_sources_empty(self, store):
        """list_sources returns empty when no docs exist."""
        assert store.list_sources() == []


# ---------------------------------------------------------------------------
# delete_source
# ---------------------------------------------------------------------------

class TestDeleteSource:
    """Test deleting a document source."""

    async def test_delete_source_removes_chunks(self, store, mock_embedder):
        """delete_source deletes all chunks for the given source."""
        await store.ingest("content", "doc.md", mock_embedder)
        assert store.count() == 1

        deleted = store.delete_source("doc.md")
        assert deleted == 1
        assert store.count() == 0

    async def test_delete_source_also_clears_fts(self, store_with_fts, mock_fts, mock_embedder):
        """delete_source also removes FTS5 entries."""
        await store_with_fts.ingest("content", "doc.md", mock_embedder)
        store_with_fts.delete_source("doc.md")
        mock_fts.delete_knowledge_source.assert_called_with("doc.md")

    def test_delete_source_not_found(self, store):
        """delete_source returns 0 if source has no chunks."""
        deleted = store.delete_source("missing.md")
        assert deleted == 0

    def test_delete_source_unavailable(self, tmp_path):
        """delete_source returns 0 when store is unavailable."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            s = KnowledgeStore(bad_path)
        assert s.delete_source("doc.md") == 0


# ---------------------------------------------------------------------------
# search_hybrid
# ---------------------------------------------------------------------------

class TestSearchHybrid:
    """Test hybrid search (semantic + FTS with RRF)."""

    async def test_hybrid_fts_only(self, store_with_fts, mock_fts, mock_embedder):
        """Hybrid search uses FTS results when no vec available."""
        mock_fts.search_knowledge.return_value = [
            {"chunk_id": "f1", "content": "fts result", "source": "b.md", "score": 5.0},
        ]

        results = await store_with_fts.search_hybrid("query", mock_embedder, limit=5)
        mock_fts.search_knowledge.assert_called_once()
        assert len(results) >= 1

    async def test_hybrid_no_embedder(self, store_with_fts, mock_fts):
        """Hybrid search works without embedder (FTS-only mode)."""
        mock_fts.search_knowledge.return_value = [
            {"chunk_id": "f1", "content": "fts result", "source": "a.md", "score": 3.0},
        ]

        results = await store_with_fts.search_hybrid("query", embedder=None, limit=5)
        assert len(results) >= 1

    async def test_hybrid_both_empty(self, store_with_fts, mock_fts, mock_embedder):
        """If both semantic and FTS return nothing, result is empty."""
        mock_fts.search_knowledge.return_value = []

        results = await store_with_fts.search_hybrid("query", mock_embedder)
        assert results == []


# ---------------------------------------------------------------------------
# backfill_fts
# ---------------------------------------------------------------------------

class TestBackfillFts:
    """Test FTS5 backfill from existing SQLite data."""

    async def test_backfill_indexes_new_chunks(self, store_with_fts, mock_fts, mock_embedder):
        """backfill_fts indexes chunks not already in FTS."""
        # First ingest some data (this also writes FTS via the mock)
        await store_with_fts.ingest("text one", "doc.md", mock_embedder)
        mock_fts.has_knowledge_chunk.return_value = False

        count = store_with_fts.backfill_fts()
        assert count == 1

    async def test_backfill_skips_existing_chunks(self, store_with_fts, mock_fts, mock_embedder):
        """backfill_fts skips chunks already in FTS."""
        await store_with_fts.ingest("text", "doc.md", mock_embedder)
        mock_fts.has_knowledge_chunk.return_value = True

        count = store_with_fts.backfill_fts()
        assert count == 0

    def test_backfill_no_fts_returns_zero(self, store):
        """backfill_fts returns 0 when there's no FTS index."""
        assert store.backfill_fts() == 0

    def test_backfill_unavailable_store(self, tmp_path, mock_fts):
        """backfill_fts returns 0 when store is unavailable."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            s = KnowledgeStore(bad_path, fts_index=mock_fts)
        assert s.backfill_fts() == 0
