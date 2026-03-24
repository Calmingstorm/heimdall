"""Tests for SQLite-backed KnowledgeStore and SessionVectorStore.

Covers: ingest + search, FTS-only fallback, delete, list sources,
session indexing + search, FTS write even when embed fails.
Uses real SQLite databases (temp), mocks sqlite-vec extension and embedder.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.knowledge.store import KnowledgeStore, CHUNK_SIZE, VECTOR_DIM
from src.search.vectorstore import SessionVectorStore, VECTOR_DIM as SV_VECTOR_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns a fixed vector."""
    emb = AsyncMock()
    emb.embed.return_value = [0.1] * VECTOR_DIM
    return emb


@pytest.fixture
def failing_embedder():
    """Mock embedder that always returns None (embed failure)."""
    emb = AsyncMock()
    emb.embed.return_value = None
    return emb


@pytest.fixture
def mock_fts():
    """Mock FTS index for knowledge store."""
    fts = MagicMock()
    fts.index_knowledge_chunk.return_value = True
    fts.search_knowledge.return_value = []
    fts.delete_knowledge_source.return_value = 0
    fts.has_knowledge_chunk.return_value = False
    return fts


@pytest.fixture
def mock_session_fts():
    """Mock FTS index for session store."""
    fts = MagicMock()
    fts.index_session.return_value = True
    fts.search_sessions.return_value = []
    fts.has_session.return_value = False
    return fts


@pytest.fixture
def knowledge_store(tmp_path):
    """KnowledgeStore with real SQLite, no vec extension."""
    db_path = str(tmp_path / "knowledge.db")
    with patch("src.knowledge.store.load_extension", return_value=False):
        return KnowledgeStore(db_path)


@pytest.fixture
def knowledge_store_fts(tmp_path, mock_fts):
    """KnowledgeStore with real SQLite and mock FTS."""
    db_path = str(tmp_path / "knowledge_fts.db")
    with patch("src.knowledge.store.load_extension", return_value=False):
        return KnowledgeStore(db_path, fts_index=mock_fts)


@pytest.fixture
def session_store(tmp_path):
    """SessionVectorStore with real SQLite, no vec extension."""
    db_path = str(tmp_path / "sessions.db")
    with patch("src.search.vectorstore.load_extension", return_value=False):
        return SessionVectorStore(db_path)


@pytest.fixture
def session_store_fts(tmp_path, mock_session_fts):
    """SessionVectorStore with real SQLite and mock FTS."""
    db_path = str(tmp_path / "sessions_fts.db")
    with patch("src.search.vectorstore.load_extension", return_value=False):
        return SessionVectorStore(db_path, fts_index=mock_session_fts)


# ---------------------------------------------------------------------------
# Knowledge Store — Ingest and Search
# ---------------------------------------------------------------------------


class TestKnowledgeIngestAndSearch:
    """Test ingesting documents and searching them."""

    async def test_ingest_short_document(self, knowledge_store, mock_embedder):
        """Ingesting a short doc creates exactly one chunk."""
        count = await knowledge_store.ingest(
            "This is a test document about servers.", "test-doc", mock_embedder
        )
        assert count == 1

    async def test_ingest_long_document_creates_chunks(self, knowledge_store, mock_embedder):
        """Ingesting a document longer than CHUNK_SIZE creates multiple chunks."""
        long_text = "word " * (CHUNK_SIZE // 2)  # many words, well beyond limit
        count = await knowledge_store.ingest(long_text, "long-doc", mock_embedder)
        assert count > 1

    async def test_ingest_updates_count(self, knowledge_store, mock_embedder):
        """Count reflects ingested chunks."""
        assert knowledge_store.count() == 0
        await knowledge_store.ingest("Test content.", "doc1", mock_embedder)
        assert knowledge_store.count() > 0

    async def test_reingest_replaces_chunks(self, knowledge_store, mock_embedder):
        """Re-ingesting same source replaces previous chunks."""
        await knowledge_store.ingest("Version 1", "my-doc", mock_embedder)
        count1 = knowledge_store.count()
        await knowledge_store.ingest("Version 2", "my-doc", mock_embedder)
        count2 = knowledge_store.count()
        assert count1 == count2  # replaced, not appended

    async def test_ingest_without_embedder(self, knowledge_store):
        """Ingest works without embedder (FTS-only mode)."""
        count = await knowledge_store.ingest(
            "Document text here.", "no-embed-doc"
        )
        assert count == 1
        assert knowledge_store.count() == 1

    async def test_search_without_vec_returns_empty(self, knowledge_store, mock_embedder):
        """Semantic search returns empty when vec extension not loaded."""
        await knowledge_store.ingest("Some content", "doc", mock_embedder)
        results = await knowledge_store.search("content", mock_embedder)
        assert results == []  # No vec extension → no vector search


# ---------------------------------------------------------------------------
# Knowledge Store — FTS Fallback (no embedder)
# ---------------------------------------------------------------------------


class TestKnowledgeFtsFallbackNoEmbedder:
    """Knowledge store works in FTS-only mode when no embedder available."""

    async def test_ingest_with_fts_only(self, knowledge_store_fts, mock_fts):
        """Ingest writes to FTS even without embedder."""
        count = await knowledge_store_fts.ingest(
            "FTS only document", "fts-doc", embedder=None
        )
        assert count == 1
        mock_fts.index_knowledge_chunk.assert_called_once()

    async def test_hybrid_search_fts_only(self, knowledge_store_fts, mock_fts):
        """Hybrid search works with FTS results only (no embedder)."""
        mock_fts.search_knowledge.return_value = [
            {"chunk_id": "abc_0", "content": "test result", "source": "doc1", "score": 0.5}
        ]
        results = await knowledge_store_fts.search_hybrid("test", embedder=None)
        assert len(results) >= 1
        mock_fts.search_knowledge.assert_called_once()

    async def test_hybrid_search_no_fts_no_embedder(self, knowledge_store):
        """Hybrid search returns empty when neither FTS nor embedder available."""
        results = await knowledge_store.search_hybrid("test", embedder=None)
        assert results == []


# ---------------------------------------------------------------------------
# Knowledge Store — Delete Source
# ---------------------------------------------------------------------------


class TestKnowledgeDeleteSource:
    """Test deleting ingested document sources."""

    async def test_delete_existing_source(self, knowledge_store, mock_embedder):
        """Deleting a source removes its chunks."""
        await knowledge_store.ingest("Content to delete.", "deleteme", mock_embedder)
        assert knowledge_store.count() > 0
        deleted = knowledge_store.delete_source("deleteme")
        assert deleted > 0
        assert knowledge_store.count() == 0

    async def test_delete_nonexistent_source(self, knowledge_store):
        """Deleting a non-existent source returns 0."""
        deleted = knowledge_store.delete_source("doesnt-exist")
        assert deleted == 0

    async def test_delete_one_source_preserves_others(self, knowledge_store, mock_embedder):
        """Deleting one source doesn't affect other sources."""
        await knowledge_store.ingest("Doc A", "source-a", mock_embedder)
        await knowledge_store.ingest("Doc B", "source-b", mock_embedder)
        assert knowledge_store.count() == 2
        knowledge_store.delete_source("source-a")
        assert knowledge_store.count() == 1

    async def test_delete_with_fts(self, knowledge_store_fts, mock_embedder, mock_fts):
        """Delete also cleans up FTS index."""
        await knowledge_store_fts.ingest("FTS doc", "fts-source", mock_embedder)
        knowledge_store_fts.delete_source("fts-source")
        mock_fts.delete_knowledge_source.assert_called_with("fts-source")


# ---------------------------------------------------------------------------
# Knowledge Store — List Sources
# ---------------------------------------------------------------------------


class TestKnowledgeListSources:
    """Test listing ingested document sources."""

    async def test_list_sources_empty(self, knowledge_store):
        """Empty store returns empty list."""
        assert knowledge_store.list_sources() == []

    async def test_list_sources_after_ingest(self, knowledge_store, mock_embedder):
        """After ingest, source appears in list."""
        await knowledge_store.ingest("Content", "my-source", mock_embedder)
        sources = knowledge_store.list_sources()
        assert len(sources) == 1
        assert sources[0]["source"] == "my-source"
        assert sources[0]["chunks"] >= 1
        assert "ingested_at" in sources[0]

    async def test_list_multiple_sources(self, knowledge_store, mock_embedder):
        """Multiple sources appear in list."""
        await knowledge_store.ingest("A", "source-a", mock_embedder)
        await knowledge_store.ingest("B", "source-b", mock_embedder)
        sources = knowledge_store.list_sources()
        names = {s["source"] for s in sources}
        assert names == {"source-a", "source-b"}


# ---------------------------------------------------------------------------
# Session Store — Index and Search
# ---------------------------------------------------------------------------


def _make_archive(summary="Test summary", channel="123", last_active=1700000000.0):
    """Create test archive JSON data."""
    return {
        "summary": summary,
        "messages": [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi!"},
        ],
        "channel_id": channel,
        "last_active": last_active,
    }


class TestSessionIndexAndSearch:
    """Test session archive indexing and search."""

    async def test_index_session(self, session_store, mock_embedder, tmp_path):
        """Indexing a session archive creates a database entry."""
        archive = _make_archive()
        path = tmp_path / "test_session.json"
        path.write_text(json.dumps(archive))

        result = await session_store.index_session(path, mock_embedder)
        assert result is True

    async def test_index_session_metadata(self, session_store, mock_embedder, tmp_path):
        """Indexed session stores correct metadata."""
        archive = _make_archive(channel="456", last_active=1700000001.0)
        path = tmp_path / "chan456_12345.json"
        path.write_text(json.dumps(archive))

        await session_store.index_session(path, mock_embedder)

        # Verify metadata in database
        row = session_store._conn.execute(
            "SELECT doc_id, channel_id, last_active, message_count FROM session_archives"
        ).fetchone()
        assert row is not None
        assert row[0] == "chan456_12345"  # doc_id = stem
        assert row[1] == "456"
        assert row[3] == 2  # 2 messages

    async def test_search_without_vec_returns_empty(self, session_store, mock_embedder, tmp_path):
        """Semantic search returns empty without vec extension."""
        archive = _make_archive()
        path = tmp_path / "session.json"
        path.write_text(json.dumps(archive))
        await session_store.index_session(path, mock_embedder)

        results = await session_store.search("Hello", mock_embedder)
        assert results == []  # No vec extension

    async def test_index_invalid_json_returns_false(self, session_store, mock_embedder, tmp_path):
        """Invalid JSON archive returns False."""
        path = tmp_path / "bad.json"
        path.write_text("not valid json{{{")
        result = await session_store.index_session(path, mock_embedder)
        assert result is False

    async def test_index_empty_archive_returns_false(self, session_store, mock_embedder, tmp_path):
        """Empty messages + no summary returns False."""
        archive = {"summary": "", "messages": [], "channel_id": "1", "last_active": 0}
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(archive))
        result = await session_store.index_session(path, mock_embedder)
        assert result is False


# ---------------------------------------------------------------------------
# Session Store — FTS Fallback
# ---------------------------------------------------------------------------


class TestSessionFtsFallback:
    """Session search works in FTS-only mode."""

    async def test_hybrid_search_fts_only(self, session_store_fts, mock_session_fts):
        """Hybrid search uses FTS when no embedder."""
        mock_session_fts.search_sessions.return_value = [
            {"doc_id": "sess_1", "content": "test", "channel_id": "1", "timestamp": 1700000000.0}
        ]
        results = await session_store_fts.search_hybrid("test", embedder=None)
        assert len(results) >= 1
        mock_session_fts.search_sessions.assert_called_once()

    async def test_hybrid_search_no_fts_no_embedder(self, session_store):
        """Hybrid search returns empty without both FTS and embedder."""
        results = await session_store.search_hybrid("test", embedder=None)
        assert results == []

    async def test_index_session_writes_fts(
        self, session_store_fts, mock_embedder, mock_session_fts, tmp_path
    ):
        """Indexing a session always writes to FTS regardless of vec."""
        archive = _make_archive()
        path = tmp_path / "fts_session.json"
        path.write_text(json.dumps(archive))

        await session_store_fts.index_session(path, mock_embedder)
        mock_session_fts.index_session.assert_called_once()


# ---------------------------------------------------------------------------
# FTS write even when embed fails (THE BUG FIX)
# ---------------------------------------------------------------------------


class TestFtsWriteEvenWhenEmbedFails:
    """Verify FTS write is decoupled from embedding success.

    This was a bug where if embedder.embed() returned None, the FTS
    write was also skipped. Now FTS always writes independently.
    """

    async def test_knowledge_fts_write_on_embed_failure(
        self, knowledge_store_fts, failing_embedder, mock_fts
    ):
        """FTS index_knowledge_chunk is called even when embedder returns None."""
        count = await knowledge_store_fts.ingest(
            "Content that fails to embed.", "embed-fail-doc", failing_embedder
        )
        assert count == 1
        # FTS was still written
        mock_fts.index_knowledge_chunk.assert_called_once()

    async def test_knowledge_fts_write_without_embedder(
        self, knowledge_store_fts, mock_fts
    ):
        """FTS index_knowledge_chunk is called even with embedder=None."""
        count = await knowledge_store_fts.ingest(
            "No embedder content.", "no-embed-doc", embedder=None
        )
        assert count == 1
        mock_fts.index_knowledge_chunk.assert_called_once()

    async def test_knowledge_chunk_metadata_written_on_embed_failure(
        self, knowledge_store_fts, failing_embedder
    ):
        """Chunk metadata is always written regardless of embed outcome."""
        await knowledge_store_fts.ingest(
            "Metadata test content.", "metadata-doc", failing_embedder
        )
        count = knowledge_store_fts.count()
        assert count == 1

    async def test_session_fts_write_on_embed_failure(
        self, session_store_fts, failing_embedder, mock_session_fts, tmp_path
    ):
        """Session FTS write succeeds even when embedder returns None."""
        archive = _make_archive()
        path = tmp_path / "fail_embed.json"
        path.write_text(json.dumps(archive))

        result = await session_store_fts.index_session(path, failing_embedder)
        assert result is True
        # FTS was still written
        mock_session_fts.index_session.assert_called_once()

    async def test_session_metadata_written_on_embed_failure(
        self, session_store_fts, failing_embedder, mock_session_fts, tmp_path
    ):
        """Session metadata is always written regardless of embed outcome."""
        archive = _make_archive()
        path = tmp_path / "meta_fail.json"
        path.write_text(json.dumps(archive))

        await session_store_fts.index_session(path, failing_embedder)

        # Metadata row exists
        row = session_store_fts._conn.execute(
            "SELECT doc_id FROM session_archives"
        ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# Knowledge Store — Unavailable store
# ---------------------------------------------------------------------------


class TestUnavailableStore:
    """Operations on an unavailable store return safe defaults."""

    def test_unavailable_count(self):
        """count() returns 0 when store is unavailable."""
        with patch("src.knowledge.store.load_extension", return_value=False):
            store = KnowledgeStore("/nonexistent/path/db.db")
        assert store.count() == 0

    async def test_unavailable_ingest(self):
        """ingest() returns 0 when store is unavailable."""
        with patch("src.knowledge.store.load_extension", return_value=False):
            store = KnowledgeStore("/nonexistent/path/db.db")
        result = await store.ingest("content", "source")
        assert result == 0

    def test_unavailable_list_sources(self):
        """list_sources() returns [] when store is unavailable."""
        with patch("src.knowledge.store.load_extension", return_value=False):
            store = KnowledgeStore("/nonexistent/path/db.db")
        assert store.list_sources() == []

    def test_unavailable_delete(self):
        """delete_source() returns 0 when store is unavailable."""
        with patch("src.knowledge.store.load_extension", return_value=False):
            store = KnowledgeStore("/nonexistent/path/db.db")
        assert store.delete_source("anything") == 0


# ---------------------------------------------------------------------------
# Knowledge Store — Chunk text
# ---------------------------------------------------------------------------


class TestChunkText:
    """Test the _chunk_text static method."""

    def test_empty_text(self):
        assert KnowledgeStore._chunk_text("") == []

    def test_whitespace_only(self):
        assert KnowledgeStore._chunk_text("   \n  ") == []

    def test_short_text_single_chunk(self):
        chunks = KnowledgeStore._chunk_text("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_long_text_multiple_chunks(self):
        long_text = "word " * (CHUNK_SIZE // 2)
        chunks = KnowledgeStore._chunk_text(long_text)
        assert len(chunks) > 1

    def test_paragraph_boundary_splitting(self):
        """Prefers splitting on paragraph boundaries."""
        para1 = "A" * (CHUNK_SIZE - 100)
        para2 = "B" * (CHUNK_SIZE - 100)
        text = f"{para1}\n\n{para2}"
        chunks = KnowledgeStore._chunk_text(text)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# sqlite_vec helpers
# ---------------------------------------------------------------------------


class TestSqliteVecHelpers:
    """Test the low-level sqlite_vec helper functions."""

    def test_serialize_deserialize_roundtrip(self):
        from src.search.sqlite_vec import serialize_vector, deserialize_vector

        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        serialized = serialize_vector(original)
        assert isinstance(serialized, bytes)

        deserialized = deserialize_vector(serialized, len(original))
        for a, b in zip(original, deserialized):
            assert abs(a - b) < 1e-6

    def test_serialize_384_dim(self):
        from src.search.sqlite_vec import serialize_vector

        vec = [float(i) / 384 for i in range(384)]
        data = serialize_vector(vec)
        assert len(data) == 384 * 4  # 4 bytes per float32

    def test_load_extension_failure(self):
        """load_extension returns False when sqlite_vec is not available."""
        import sqlite3
        from src.search.sqlite_vec import load_extension

        conn = sqlite3.connect(":memory:")
        # Mock import failure by making sqlite_vec import raise
        import importlib
        with patch.dict("sys.modules", {"sqlite_vec": None}):
            result = load_extension(conn)
        # Should return False on failure (import raises, caught by except)
        assert result is False
        conn.close()
