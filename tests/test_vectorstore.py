"""Tests for src/search/vectorstore.py — SQLite session archive search.

Covers: init, availability, index_session, search, backfill,
search_hybrid, and _build_document_text.
SQLite operations use real temp databases. Embedder is mocked.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.search.vectorstore import SessionVectorStore, MAX_MESSAGES_PER_DOC, MAX_MSG_CHARS, VECTOR_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_archive_data(summary="Test summary", messages=None, channel_id="123", last_active=1700000000.0):
    """Create test archive data dict."""
    if messages is None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    return {
        "summary": summary,
        "messages": messages,
        "channel_id": channel_id,
        "last_active": last_active,
    }


@pytest.fixture
def mock_embedder():
    emb = AsyncMock()
    emb.embed.return_value = [0.1] * VECTOR_DIM
    return emb


@pytest.fixture
def mock_fts():
    fts = MagicMock()
    fts.index_session.return_value = True
    fts.search_sessions.return_value = []
    fts.has_session.return_value = False
    return fts


@pytest.fixture
def store(tmp_path):
    """Create a SessionVectorStore with a real SQLite DB (no vec extension)."""
    db_path = str(tmp_path / "sessions.db")
    with patch("src.search.vectorstore.load_extension", return_value=False):
        s = SessionVectorStore(db_path)
    return s


@pytest.fixture
def store_with_fts(tmp_path, mock_fts):
    """SessionVectorStore with a real SQLite DB and mock FTS index."""
    db_path = str(tmp_path / "sessions.db")
    with patch("src.search.vectorstore.load_extension", return_value=False):
        s = SessionVectorStore(db_path, fts_index=mock_fts)
    return s


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Test SessionVectorStore initialization."""

    def test_init_creates_sqlite_db(self, tmp_path):
        """Store creates a SQLite database file."""
        db_path = str(tmp_path / "test.db")
        with patch("src.search.vectorstore.load_extension", return_value=False):
            s = SessionVectorStore(db_path)
        assert s.available is True
        assert os.path.exists(db_path)

    def test_init_failure_makes_unavailable(self, tmp_path):
        """If DB open fails, store is unavailable."""
        bad_path = str(tmp_path / "nonexistent" / "subdir" / "test.db")
        with patch("src.search.vectorstore.load_extension", return_value=False):
            s = SessionVectorStore(bad_path)
        assert s.available is False


# ---------------------------------------------------------------------------
# _build_document_text (static, pure logic)
# ---------------------------------------------------------------------------

class TestBuildDocumentText:
    """Test the _build_document_text helper."""

    def test_with_summary_and_messages(self):
        """Builds text with summary and formatted messages."""
        data = _make_archive_data()
        text = SessionVectorStore._build_document_text(data)
        assert "Summary: Test summary" in text
        assert "user: Hello" in text
        assert "assistant: Hi there!" in text

    def test_without_summary(self):
        """Works without a summary."""
        data = _make_archive_data(summary="")
        text = SessionVectorStore._build_document_text(data)
        assert "Summary:" not in text
        assert "user: Hello" in text

    def test_empty_messages(self):
        """Returns summary only when no messages."""
        data = _make_archive_data(messages=[])
        text = SessionVectorStore._build_document_text(data)
        assert text == "Summary: Test summary"

    def test_empty_data(self):
        """Returns empty string for empty data."""
        text = SessionVectorStore._build_document_text({})
        assert text == ""

    def test_truncates_messages(self):
        """Only includes MAX_MESSAGES_PER_DOC messages."""
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(30)]
        data = _make_archive_data(summary="", messages=msgs)
        text = SessionVectorStore._build_document_text(data)
        lines = text.strip().split("\n")
        assert len(lines) == MAX_MESSAGES_PER_DOC

    def test_truncates_long_content(self):
        """Message content is truncated to MAX_MSG_CHARS."""
        long_content = "x" * (MAX_MSG_CHARS + 100)
        data = _make_archive_data(summary="", messages=[{"role": "user", "content": long_content}])
        text = SessionVectorStore._build_document_text(data)
        msg_line = text.strip()
        content_part = msg_line.split(": ", 1)[1]
        assert len(content_part) <= MAX_MSG_CHARS


# ---------------------------------------------------------------------------
# index_session
# ---------------------------------------------------------------------------

class TestIndexSession:
    """Test indexing a session archive."""

    async def test_index_success(self, store, mock_embedder, tmp_path):
        """Successfully indexes an archive file."""
        archive = tmp_path / "session_123.json"
        archive.write_text(json.dumps(_make_archive_data()))

        result = await store.index_session(archive, mock_embedder)
        assert result is True

    async def test_index_with_fts(self, store_with_fts, mock_fts, mock_embedder, tmp_path):
        """Indexing also writes to FTS5."""
        archive = tmp_path / "session_456.json"
        archive.write_text(json.dumps(_make_archive_data()))

        await store_with_fts.index_session(archive, mock_embedder)
        mock_fts.index_session.assert_called_once()

    async def test_index_unavailable(self, tmp_path, mock_embedder):
        """Returns False when store is unavailable."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.search.vectorstore.load_extension", return_value=False):
            s = SessionVectorStore(bad_path)
        archive = tmp_path / "test.json"
        archive.write_text("{}")
        assert await s.index_session(archive, mock_embedder) is False

    async def test_index_bad_json(self, store, mock_embedder, tmp_path):
        """Returns False for invalid JSON."""
        archive = tmp_path / "bad.json"
        archive.write_text("not json")
        assert await store.index_session(archive, mock_embedder) is False

    async def test_index_empty_document_text(self, store, mock_embedder, tmp_path):
        """Returns False when archive produces no document text."""
        archive = tmp_path / "empty.json"
        archive.write_text(json.dumps({}))
        assert await store.index_session(archive, mock_embedder) is False

    async def test_index_writes_metadata(self, store, mock_embedder, tmp_path):
        """Indexing writes channel_id and message_count to metadata."""
        archive = tmp_path / "session_789.json"
        archive.write_text(json.dumps(_make_archive_data(channel_id="789")))

        await store.index_session(archive, mock_embedder)

        # Verify data in SQLite
        row = store._conn.execute(
            "SELECT channel_id, message_count FROM session_archives WHERE doc_id = 'session_789'"
        ).fetchone()
        assert row is not None
        assert row[0] == "789"
        assert row[1] == 2  # Two messages in test data


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    """Test semantic search across archives."""

    async def test_search_no_vec_returns_empty(self, store, mock_embedder):
        """Returns empty when no sqlite-vec available (FTS-only mode)."""
        results = await store.search("test", mock_embedder, limit=5)
        assert results == []

    async def test_search_unavailable(self, tmp_path, mock_embedder):
        """Returns empty when store is unavailable."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.search.vectorstore.load_extension", return_value=False):
            s = SessionVectorStore(bad_path)
        assert await s.search("q", mock_embedder) == []

    async def test_search_embed_failure(self, store, mock_embedder):
        """Returns empty when embedding fails."""
        mock_embedder.embed.return_value = None
        assert await store.search("q", mock_embedder) == []


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------

class TestBackfill:
    """Test backfilling archives into SQLite."""

    async def test_backfill_indexes_new_archives(self, store, mock_embedder, tmp_path):
        """Backfill indexes archives not already in SQLite."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "sess1.json").write_text(json.dumps(_make_archive_data()))
        (archive_dir / "sess2.json").write_text(json.dumps(_make_archive_data()))

        count = await store.backfill(archive_dir, mock_embedder)
        assert count == 2

    async def test_backfill_skips_existing(self, store, mock_embedder, tmp_path):
        """Backfill skips archives already indexed."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        archive = archive_dir / "sess1.json"
        archive.write_text(json.dumps(_make_archive_data()))

        # Index once
        await store.index_session(archive, mock_embedder)
        # Backfill should skip it
        count = await store.backfill(archive_dir, mock_embedder)
        assert count == 0

    async def test_backfill_unavailable(self, tmp_path, mock_embedder):
        """Returns 0 when store is unavailable."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.search.vectorstore.load_extension", return_value=False):
            s = SessionVectorStore(bad_path)
        assert await s.backfill(tmp_path, mock_embedder) == 0

    async def test_backfill_missing_dir(self, store, mock_embedder, tmp_path):
        """Returns 0 when archive dir doesn't exist."""
        count = await store.backfill(tmp_path / "nonexistent", mock_embedder)
        assert count == 0

    async def test_backfill_with_fts(self, store_with_fts, mock_fts, mock_embedder, tmp_path):
        """Backfill also indexes into FTS5."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "sess1.json").write_text(json.dumps(_make_archive_data()))

        await store_with_fts.backfill(archive_dir, mock_embedder)
        assert mock_fts.index_session.call_count >= 1

    async def test_backfill_fts_skips_existing(self, store_with_fts, mock_fts, mock_embedder, tmp_path):
        """FTS backfill skips sessions already in FTS."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "sess1.json").write_text(json.dumps(_make_archive_data()))

        # Pre-index the session
        await store_with_fts.index_session(archive_dir / "sess1.json", mock_embedder)
        mock_fts.reset_mock()

        # Mark as already in FTS
        mock_fts.has_session.return_value = True

        await store_with_fts.backfill(archive_dir, mock_embedder)
        # FTS backfill phase should not call index_session again
        fts_calls = [c for c in mock_fts.index_session.call_args_list]
        assert len(fts_calls) == 0

    async def test_backfill_fts_handles_bad_json(self, store_with_fts, mock_fts, mock_embedder, tmp_path):
        """FTS backfill continues if a JSON file is invalid."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "bad.json").write_text("not json")

        # Pre-index to SQLite so main loop skips it
        store_with_fts._conn.execute(
            "INSERT INTO session_archives (doc_id, content, channel_id, last_active, message_count) "
            "VALUES ('bad', 'text', '', 0, 0)"
        )
        store_with_fts._conn.commit()
        mock_fts.has_session.return_value = False

        count = await store_with_fts.backfill(archive_dir, mock_embedder)
        assert count == 0


# ---------------------------------------------------------------------------
# search_hybrid
# ---------------------------------------------------------------------------

class TestSearchHybrid:
    """Test hybrid search combining semantic + FTS."""

    async def test_hybrid_fts_only(self, store_with_fts, mock_fts, mock_embedder):
        """Hybrid search uses FTS results when no vec available."""
        mock_fts.search_sessions.return_value = [
            {"doc_id": "f1", "content": "fts text", "channel_id": "2", "timestamp": 2.0},
        ]

        results = await store_with_fts.search_hybrid("query", mock_embedder, limit=5)
        assert len(results) >= 1

    async def test_hybrid_no_embedder(self, store_with_fts, mock_fts):
        """Hybrid works without embedder."""
        mock_fts.search_sessions.return_value = [
            {"doc_id": "f1", "content": "fts text", "channel_id": "2", "timestamp": 2.0},
        ]

        results = await store_with_fts.search_hybrid("query", embedder=None, limit=5)
        assert len(results) >= 1

    async def test_hybrid_both_empty(self, store_with_fts, mock_fts, mock_embedder):
        """Returns empty when both backends return nothing."""
        mock_fts.search_sessions.return_value = []

        results = await store_with_fts.search_hybrid("query", mock_embedder)
        assert results == []
