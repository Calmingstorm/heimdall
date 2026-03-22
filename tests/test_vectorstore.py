"""Tests for src/search/vectorstore.py — ChromaDB session archive search.

Covers: init, availability, index_session, search, backfill,
search_hybrid, and _build_document_text.
All ChromaDB and Ollama calls are mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.search.vectorstore import SessionVectorStore, MAX_MESSAGES_PER_DOC, MAX_MSG_CHARS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_collection(initial_ids=None) -> MagicMock:
    """Create a mock ChromaDB collection."""
    col = MagicMock()
    col.count.return_value = len(initial_ids) if initial_ids else 0
    col.get.return_value = {"ids": initial_ids or [], "documents": [], "metadatas": []}
    col.query.return_value = {
        "ids": [[]], "documents": [[]], "metadatas": [[]],
        "distances": [[]],
    }
    col.upsert.return_value = None
    col.delete.return_value = None
    return col


def _make_store(collection, fts_index=None):
    """Build a SessionVectorStore bypassing the real constructor."""
    with patch("src.search.vectorstore.HAS_CHROMADB", False):
        s = SessionVectorStore("/tmp/test_chromadb", fts_index=fts_index)
    s._collection = collection
    s._client = MagicMock()
    return s


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
def mock_collection():
    return _make_mock_collection()


@pytest.fixture
def mock_embedder():
    emb = AsyncMock()
    emb.embed.return_value = [0.1] * 768
    return emb


@pytest.fixture
def mock_fts():
    fts = MagicMock()
    fts.index_session.return_value = True
    fts.search_sessions.return_value = []
    fts.has_session.return_value = False
    return fts


@pytest.fixture
def store(mock_collection):
    return _make_store(mock_collection)


@pytest.fixture
def store_with_fts(mock_collection, mock_fts):
    return _make_store(mock_collection, fts_index=mock_fts)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Test SessionVectorStore initialization."""

    def test_init_with_chromadb(self):
        """When chromadb is available and succeeds, store is available."""
        mock_chromadb = MagicMock()
        col = _make_mock_collection()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = col
        with patch("src.search.vectorstore.HAS_CHROMADB", True):
            with patch("src.search.vectorstore.chromadb", mock_chromadb, create=True):
                s = SessionVectorStore("/tmp/path")
        assert s.available is True

    def test_init_failure(self):
        """If PersistentClient raises, store is unavailable."""
        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.side_effect = RuntimeError("fail")
        with patch("src.search.vectorstore.HAS_CHROMADB", True):
            with patch("src.search.vectorstore.chromadb", mock_chromadb, create=True):
                s = SessionVectorStore("/tmp/bad")
        assert s.available is False

    def test_init_no_chromadb(self):
        """If HAS_CHROMADB is False, store is unavailable."""
        with patch("src.search.vectorstore.HAS_CHROMADB", False):
            s = SessionVectorStore("/tmp/x")
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
        # "user: <content>" — the content portion after "user: " should be at most MAX_MSG_CHARS
        msg_line = text.strip()
        content_part = msg_line.split(": ", 1)[1]
        assert len(content_part) <= MAX_MSG_CHARS


# ---------------------------------------------------------------------------
# index_session
# ---------------------------------------------------------------------------

class TestIndexSession:
    """Test indexing a session archive."""

    async def test_index_success(self, store, mock_collection, mock_embedder, tmp_path):
        """Successfully indexes an archive file."""
        archive = tmp_path / "session_123.json"
        archive.write_text(json.dumps(_make_archive_data()))

        result = await store.index_session(archive, mock_embedder)

        assert result is True
        mock_embedder.embed.assert_called_once()
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["ids"] == ["session_123"]
        meta = call_kwargs["metadatas"][0]
        assert meta["channel_id"] == "123"

    async def test_index_with_fts(self, store_with_fts, mock_collection, mock_fts, mock_embedder, tmp_path):
        """Indexing also writes to FTS5."""
        archive = tmp_path / "session_456.json"
        archive.write_text(json.dumps(_make_archive_data()))

        await store_with_fts.index_session(archive, mock_embedder)

        mock_fts.index_session.assert_called_once()

    async def test_index_unavailable(self, mock_embedder, tmp_path):
        """Returns False when store is unavailable."""
        with patch("src.search.vectorstore.HAS_CHROMADB", False):
            s = SessionVectorStore("/tmp/x")
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

    async def test_index_embed_failure(self, store, mock_embedder, tmp_path):
        """Returns False when embedder returns None."""
        mock_embedder.embed.return_value = None
        archive = tmp_path / "test.json"
        archive.write_text(json.dumps(_make_archive_data()))
        assert await store.index_session(archive, mock_embedder) is False

    async def test_index_upsert_failure(self, store, mock_collection, mock_embedder, tmp_path):
        """Returns False when ChromaDB upsert raises."""
        mock_collection.upsert.side_effect = RuntimeError("disk full")
        archive = tmp_path / "test.json"
        archive.write_text(json.dumps(_make_archive_data()))
        assert await store.index_session(archive, mock_embedder) is False


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    """Test semantic search across archives."""

    async def test_search_returns_results(self, store, mock_collection, mock_embedder):
        """Search returns formatted results from ChromaDB."""
        mock_collection.query.return_value = {
            "ids": [["sess1"]],
            "documents": [["user: Hello\nassistant: Hi"]],
            "metadatas": [[{"channel_id": "123", "last_active": 1700000000.0}]],
            "distances": [[0.3]],
        }

        results = await store.search("test", mock_embedder, limit=5)

        assert len(results) == 1
        assert results[0]["type"] == "semantic"
        assert results[0]["channel_id"] == "123"
        assert results[0]["timestamp"] == 1700000000.0
        assert len(results[0]["content"]) <= 500

    async def test_search_filters_poor_matches(self, store, mock_collection, mock_embedder):
        """Results with distance > 1.0 are filtered out."""
        mock_collection.query.return_value = {
            "ids": [["good", "bad"]],
            "documents": [["good match", "bad match"]],
            "metadatas": [[
                {"channel_id": "1", "last_active": 1.0},
                {"channel_id": "2", "last_active": 2.0},
            ]],
            "distances": [[0.5, 1.5]],
        }

        results = await store.search("q", mock_embedder)
        assert len(results) == 1
        assert results[0]["channel_id"] == "1"

    async def test_search_unavailable(self, mock_embedder):
        """Returns empty when store is unavailable."""
        with patch("src.search.vectorstore.HAS_CHROMADB", False):
            s = SessionVectorStore("/tmp/x")
        assert await s.search("q", mock_embedder) == []

    async def test_search_embed_failure(self, store, mock_embedder):
        """Returns empty when embedding fails."""
        mock_embedder.embed.return_value = None
        assert await store.search("q", mock_embedder) == []

    async def test_search_query_exception(self, store, mock_collection, mock_embedder):
        """Returns empty when ChromaDB query raises."""
        mock_collection.query.side_effect = RuntimeError("fail")
        assert await store.search("q", mock_embedder) == []

    async def test_search_empty_results(self, store, mock_collection, mock_embedder):
        """Returns empty for no matches."""
        mock_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]],
            "distances": [[]],
        }
        assert await store.search("q", mock_embedder) == []

    async def test_search_missing_metadata_defaults(self, store, mock_collection, mock_embedder):
        """Missing metadata fields use defaults."""
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "documents": [["text"]],
            "distances": [[0.2]],
        }

        results = await store.search("q", mock_embedder)
        assert len(results) == 1
        assert results[0]["channel_id"] == "unknown"
        assert results[0]["timestamp"] == 0

    async def test_search_missing_distances_defaults(self, store, mock_collection, mock_embedder):
        """Missing distances default to 1.0 (exact threshold, not filtered)."""
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "documents": [["text"]],
            "metadatas": [[{"channel_id": "1", "last_active": 1.0}]],
            # No distances key
        }

        results = await store.search("q", mock_embedder)
        assert len(results) == 1  # distance 1.0 is exactly at threshold (not >1.0)

    async def test_search_truncates_content(self, store, mock_collection, mock_embedder):
        """Content is truncated to 500 chars."""
        long_doc = "x" * 1000
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "documents": [[long_doc]],
            "metadatas": [[{"channel_id": "1", "last_active": 1.0}]],
            "distances": [[0.1]],
        }

        results = await store.search("q", mock_embedder)
        assert len(results[0]["content"]) == 500

    async def test_search_none_results(self, store, mock_collection, mock_embedder):
        """Returns empty if query returns None."""
        mock_collection.query.return_value = None
        assert await store.search("q", mock_embedder) == []


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------

class TestBackfill:
    """Test backfilling archives into ChromaDB."""

    async def test_backfill_indexes_new_archives(self, store, mock_collection, mock_embedder, tmp_path):
        """Backfill indexes archives not already in ChromaDB."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "sess1.json").write_text(json.dumps(_make_archive_data()))
        (archive_dir / "sess2.json").write_text(json.dumps(_make_archive_data()))
        mock_collection.get.return_value = {"ids": []}

        count = await store.backfill(archive_dir, mock_embedder)

        assert count == 2
        assert mock_collection.upsert.call_count == 2

    async def test_backfill_skips_existing(self, store, mock_collection, mock_embedder, tmp_path):
        """Backfill skips archives already indexed."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "sess1.json").write_text(json.dumps(_make_archive_data()))
        (archive_dir / "sess2.json").write_text(json.dumps(_make_archive_data()))
        mock_collection.get.return_value = {"ids": ["sess1"]}

        count = await store.backfill(archive_dir, mock_embedder)

        assert count == 1  # Only sess2

    async def test_backfill_unavailable(self, mock_embedder, tmp_path):
        """Returns 0 when store is unavailable."""
        with patch("src.search.vectorstore.HAS_CHROMADB", False):
            s = SessionVectorStore("/tmp/x")
        assert await s.backfill(tmp_path, mock_embedder) == 0

    async def test_backfill_missing_dir(self, store, mock_embedder, tmp_path):
        """Returns 0 when archive dir doesn't exist."""
        count = await store.backfill(tmp_path / "nonexistent", mock_embedder)
        assert count == 0

    async def test_backfill_get_exception(self, store, mock_collection, mock_embedder, tmp_path):
        """If getting existing IDs raises, treats all as new."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "sess1.json").write_text(json.dumps(_make_archive_data()))
        mock_collection.get.side_effect = RuntimeError("fail")

        count = await store.backfill(archive_dir, mock_embedder)
        assert count == 1

    async def test_backfill_with_fts(self, store_with_fts, mock_collection, mock_fts, mock_embedder, tmp_path):
        """Backfill also indexes into FTS5."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "sess1.json").write_text(json.dumps(_make_archive_data()))
        mock_collection.get.return_value = {"ids": []}

        await store_with_fts.backfill(archive_dir, mock_embedder)

        # FTS index_session called for main index + backfill
        assert mock_fts.index_session.call_count >= 1

    async def test_backfill_fts_skips_existing(self, store_with_fts, mock_collection, mock_fts, mock_embedder, tmp_path):
        """FTS backfill skips sessions already in FTS."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "sess1.json").write_text(json.dumps(_make_archive_data()))
        mock_collection.get.return_value = {"ids": ["sess1"]}  # already in ChromaDB
        mock_fts.has_session.return_value = True  # already in FTS

        await store_with_fts.backfill(archive_dir, mock_embedder)

        # index_session on FTS should not be called during backfill phase
        # (only during the main indexing loop, which is skipped since sess1 is existing)
        # The FTS backfill section should also skip it since has_session returns True
        fts_calls = [c for c in mock_fts.index_session.call_args_list]
        assert len(fts_calls) == 0

    async def test_backfill_fts_handles_bad_json(self, store_with_fts, mock_collection, mock_fts, mock_embedder, tmp_path):
        """FTS backfill continues if a JSON file is invalid."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        (archive_dir / "bad.json").write_text("not json")
        mock_collection.get.return_value = {"ids": ["bad"]}  # in chromadb but not fts
        mock_fts.has_session.return_value = False

        count = await store_with_fts.backfill(archive_dir, mock_embedder)
        assert count == 0  # Bad JSON can't be indexed


# ---------------------------------------------------------------------------
# search_hybrid
# ---------------------------------------------------------------------------

class TestSearchHybrid:
    """Test hybrid search combining semantic + FTS."""

    async def test_hybrid_combines_results(self, store_with_fts, mock_collection, mock_fts, mock_embedder):
        """Hybrid search combines semantic and FTS results via RRF."""
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "documents": [["semantic text"]],
            "metadatas": [[{"channel_id": "1", "last_active": 1.0}]],
            "distances": [[0.3]],
        }
        mock_fts.search_sessions.return_value = [
            {"doc_id": "f1", "content": "fts text", "channel_id": "2", "timestamp": 2.0},
        ]

        results = await store_with_fts.search_hybrid("query", mock_embedder, limit=5)
        assert len(results) >= 1

    async def test_hybrid_no_fts(self, store, mock_collection, mock_embedder):
        """Without FTS, hybrid uses only semantic results."""
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "documents": [["text"]],
            "metadatas": [[{"channel_id": "1", "last_active": 1.0}]],
            "distances": [[0.3]],
        }

        results = await store.search_hybrid("query", mock_embedder, limit=5)
        assert len(results) >= 1

    async def test_hybrid_both_empty(self, store_with_fts, mock_collection, mock_fts, mock_embedder):
        """Returns empty when both backends return nothing."""
        mock_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]],
            "distances": [[]],
        }
        mock_fts.search_sessions.return_value = []

        results = await store_with_fts.search_hybrid("query", mock_embedder)
        assert results == []

    async def test_hybrid_adds_doc_id(self, store, mock_collection, mock_embedder):
        """Hybrid normalizes semantic results to include doc_id."""
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "documents": [["text"]],
            "metadatas": [[{"channel_id": "42", "last_active": 100.0}]],
            "distances": [[0.2]],
        }

        results = await store.search_hybrid("query", mock_embedder, limit=5)
        assert len(results) >= 1
        for r in results:
            assert "doc_id" in r

    async def test_hybrid_passes_double_limit(self, store_with_fts, mock_collection, mock_fts, mock_embedder):
        """Hybrid queries each backend with 2x the final limit."""
        mock_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]],
            "distances": [[]],
        }
        mock_fts.search_sessions.return_value = []

        await store_with_fts.search_hybrid("query", mock_embedder, limit=3)

        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["n_results"] == 6
        mock_fts.search_sessions.assert_called_once_with("query", limit=6)
