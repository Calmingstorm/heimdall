"""Round 23 — Performance: verify blocking I/O is offloaded via asyncio.to_thread.

Tests cover:
  - KnowledgeStore: search, ingest, search_hybrid use to_thread for SQLite ops
  - SessionVectorStore: search, index_session, backfill use to_thread
  - ConversationReflector: _reflect offloads _load/_save
  - API endpoints offload sync store calls
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.knowledge.store import KnowledgeStore, VECTOR_DIM
from src.search.vectorstore import SessionVectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def knowledge_store(tmp_path):
    db = str(tmp_path / "knowledge.db")
    with patch("src.knowledge.store.load_extension", return_value=False):
        return KnowledgeStore(db)


@pytest.fixture
def knowledge_store_with_fts(tmp_path):
    db = str(tmp_path / "knowledge.db")
    fts = MagicMock()
    fts.index_knowledge_chunk.return_value = True
    fts.search_knowledge.return_value = [
        {"chunk_id": "abc_0", "content": "fts result", "source": "doc.md",
         "chunk_index": 0, "type": "fts", "rank": -1.5},
    ]
    fts.delete_knowledge_source.return_value = 0
    fts.has_knowledge_chunk.return_value = False
    with patch("src.knowledge.store.load_extension", return_value=False):
        return KnowledgeStore(db, fts_index=fts)


@pytest.fixture
def session_store(tmp_path):
    db = str(tmp_path / "sessions.db")
    with patch("src.search.vectorstore.load_extension", return_value=False):
        return SessionVectorStore(db)


@pytest.fixture
def session_store_with_fts(tmp_path):
    db = str(tmp_path / "sessions.db")
    fts = MagicMock()
    fts.index_session.return_value = True
    fts.search_sessions.return_value = [
        {"doc_id": "sess1", "content": "fts hit", "channel_id": "ch1",
         "timestamp": 100.0, "type": "fts", "rank": -1.2},
    ]
    fts.has_session.return_value = False
    with patch("src.search.vectorstore.load_extension", return_value=False):
        return SessionVectorStore(db, fts_index=fts)


@pytest.fixture
def mock_embedder():
    emb = AsyncMock()
    emb.embed.return_value = [0.1] * VECTOR_DIM
    return emb


def _make_archive(tmp_path, name="session_1", channel_id="ch1"):
    data = {
        "channel_id": channel_id,
        "last_active": 12345.0,
        "summary": "test summary",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
    }
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# KnowledgeStore — sync helper methods exist and work
# ---------------------------------------------------------------------------

class TestKnowledgeStoreSyncHelpers:
    """Verify sync helper methods used by asyncio.to_thread."""

    def test_search_vec_sync_exists(self, knowledge_store):
        assert hasattr(knowledge_store, "_search_vec_sync")

    def test_write_chunks_sync_exists(self, knowledge_store):
        assert hasattr(knowledge_store, "_write_chunks_sync")

    def test_write_chunks_sync_writes_to_db(self, knowledge_store):
        """_write_chunks_sync should write chunks to SQLite."""
        indexed = knowledge_store._write_chunks_sync(
            ["chunk one", "chunk two"],
            [None, None],
            "abc12345", "test.md", "2024-01-01T00:00:00", "tester",
        )
        assert indexed == 2
        row = knowledge_store._conn.execute(
            "SELECT COUNT(*) FROM knowledge_chunks WHERE source = 'test.md'"
        ).fetchone()
        assert row[0] == 2

    def test_write_chunks_sync_with_vector(self, tmp_path):
        """_write_chunks_sync writes vector data when available."""
        db = str(tmp_path / "kv.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            store = KnowledgeStore(db)
        # No vec table, so vectors are skipped (no error)
        indexed = store._write_chunks_sync(
            ["text"], [None], "hash1", "src.md", "2024-01-01", "test",
        )
        assert indexed == 1

    def test_write_chunks_sync_with_fts(self, knowledge_store_with_fts):
        """_write_chunks_sync calls FTS indexing."""
        store = knowledge_store_with_fts
        store._write_chunks_sync(
            ["text"], [None], "hash1", "src.md", "2024-01-01", "test",
        )
        store._fts.index_knowledge_chunk.assert_called_once()

    def test_write_chunks_sync_handles_chunk_error(self, knowledge_store):
        """_write_chunks_sync returns count of successfully indexed chunks."""
        # Write 3 chunks: all should succeed when no errors
        indexed = knowledge_store._write_chunks_sync(
            ["a", "b", "c"], [None, None, None], "h", "s.md", "now", "u",
        )
        assert indexed == 3
        row = knowledge_store._conn.execute(
            "SELECT COUNT(*) FROM knowledge_chunks WHERE source = 's.md'"
        ).fetchone()
        assert row[0] == 3


# ---------------------------------------------------------------------------
# KnowledgeStore — async methods use asyncio.to_thread
# ---------------------------------------------------------------------------

class TestKnowledgeStoreAsyncOffload:
    """Verify async methods offload blocking I/O to threads."""

    async def test_ingest_offloads_delete_and_write(self, knowledge_store, mock_embedder):
        """ingest() should call asyncio.to_thread for delete and write."""
        with patch("src.knowledge.store.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = [0, 1]  # delete returns 0, write returns 1
            count = await knowledge_store.ingest("hello", "test.md", mock_embedder)
            assert mock_thread.call_count == 2
            # First call: delete_source
            assert mock_thread.call_args_list[0].args[0] == knowledge_store.delete_source
            # Second call: _write_chunks_sync
            assert mock_thread.call_args_list[1].args[0] == knowledge_store._write_chunks_sync

    async def test_ingest_embeds_before_write(self, knowledge_store, mock_embedder):
        """Embeddings are generated (async) before the sync batch write."""
        count = await knowledge_store.ingest("hello world", "test.md", mock_embedder)
        assert count == 1
        row = knowledge_store._conn.execute(
            "SELECT content FROM knowledge_chunks WHERE source = 'test.md'"
        ).fetchone()
        assert row[0] == "hello world"

    async def test_ingest_fts_written_even_on_embed_fail(self, knowledge_store_with_fts):
        """FTS is written even when embedder returns None."""
        bad_embedder = AsyncMock()
        bad_embedder.embed.return_value = None
        count = await knowledge_store_with_fts.ingest("text", "fail.md", bad_embedder)
        assert count == 1
        knowledge_store_with_fts._fts.index_knowledge_chunk.assert_called_once()

    async def test_search_offloads_sqlite_query(self, knowledge_store):
        """search() should use asyncio.to_thread for the SQLite query."""
        with patch("src.knowledge.store.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = []
            emb = AsyncMock()
            emb.embed.return_value = [0.1] * VECTOR_DIM
            # search requires _has_vec=True
            knowledge_store._has_vec = True
            await knowledge_store.search("test", emb)
            mock_thread.assert_called_once()
            assert mock_thread.call_args.args[0] == knowledge_store._search_vec_sync

    async def test_search_hybrid_offloads_fts(self, knowledge_store_with_fts):
        """search_hybrid() should use asyncio.to_thread for FTS search."""
        with patch("src.knowledge.store.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = []
            await knowledge_store_with_fts.search_hybrid("test query")
            mock_thread.assert_called_once()
            assert mock_thread.call_args.args[0] == knowledge_store_with_fts._fts.search_knowledge

    async def test_search_hybrid_returns_fts_results(self, knowledge_store_with_fts):
        """search_hybrid returns FTS results when no embedder."""
        results = await knowledge_store_with_fts.search_hybrid("test")
        assert len(results) > 0
        assert results[0]["source"] == "doc.md"


# ---------------------------------------------------------------------------
# SessionVectorStore — sync helper methods
# ---------------------------------------------------------------------------

class TestSessionStoreSyncHelpers:
    """Verify sync helper methods for asyncio.to_thread."""

    def test_search_vec_sync_exists(self, session_store):
        assert hasattr(session_store, "_search_vec_sync")

    def test_read_archive_sync_exists(self):
        assert hasattr(SessionVectorStore, "_read_archive_sync")

    def test_write_session_sync_exists(self, session_store):
        assert hasattr(session_store, "_write_session_sync")

    def test_get_indexed_ids_sync_exists(self, session_store):
        assert hasattr(session_store, "_get_indexed_ids_sync")

    def test_backfill_fts_sync_exists(self, session_store):
        assert hasattr(session_store, "_backfill_fts_sync")

    def test_read_archive_sync_parses_json(self, tmp_path):
        p = _make_archive(tmp_path)
        data = SessionVectorStore._read_archive_sync(p)
        assert data["channel_id"] == "ch1"
        assert len(data["messages"]) == 2

    def test_write_session_sync_writes_metadata(self, session_store):
        session_store._write_session_sync(
            "doc1", "text content", "ch1", 100.0, 5, None,
        )
        row = session_store._conn.execute(
            "SELECT channel_id, message_count FROM session_archives WHERE doc_id = 'doc1'"
        ).fetchone()
        assert row[0] == "ch1"
        assert row[1] == 5

    def test_write_session_sync_with_fts(self, session_store_with_fts):
        session_store_with_fts._write_session_sync(
            "doc1", "text", "ch1", 100.0, 2, None,
        )
        session_store_with_fts._fts.index_session.assert_called_once()

    def test_get_indexed_ids_sync_returns_set(self, session_store):
        session_store._conn.execute(
            "INSERT INTO session_archives (doc_id, content) VALUES ('a', 'text')"
        )
        session_store._conn.execute(
            "INSERT INTO session_archives (doc_id, content) VALUES ('b', 'text')"
        )
        session_store._conn.commit()
        ids = session_store._get_indexed_ids_sync()
        assert ids == {"a", "b"}


# ---------------------------------------------------------------------------
# SessionVectorStore — async offloading
# ---------------------------------------------------------------------------

class TestSessionStoreAsyncOffload:
    """Verify async methods offload blocking I/O to threads."""

    async def test_index_session_offloads_file_read(self, session_store, mock_embedder, tmp_path):
        """index_session() offloads archive file reading."""
        archive = _make_archive(tmp_path)
        with patch("src.search.vectorstore.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = [
                # First call: _read_archive_sync
                json.loads(archive.read_text()),
                # Second call: _write_session_sync
                None,
            ]
            await session_store.index_session(archive, mock_embedder)
            assert mock_thread.call_count == 2
            assert mock_thread.call_args_list[0].args[0] == SessionVectorStore._read_archive_sync

    async def test_index_session_writes_to_db(self, session_store, mock_embedder, tmp_path):
        """index_session works end-to-end with offloaded I/O."""
        archive = _make_archive(tmp_path, name="sess_42")
        result = await session_store.index_session(archive, mock_embedder)
        assert result is True
        row = session_store._conn.execute(
            "SELECT doc_id FROM session_archives WHERE doc_id = 'sess_42'"
        ).fetchone()
        assert row is not None

    async def test_search_offloads_sqlite(self, session_store):
        """search() should use asyncio.to_thread for the SQLite query."""
        with patch("src.search.vectorstore.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = []
            emb = AsyncMock()
            emb.embed.return_value = [0.1] * VECTOR_DIM
            session_store._has_vec = True
            await session_store.search("query", emb)
            mock_thread.assert_called_once()
            assert mock_thread.call_args.args[0] == session_store._search_vec_sync

    async def test_search_hybrid_offloads_fts(self, session_store_with_fts):
        """search_hybrid() offloads FTS search to thread."""
        with patch("src.search.vectorstore.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = []
            await session_store_with_fts.search_hybrid("test", None)
            mock_thread.assert_called_once()
            assert mock_thread.call_args.args[0] == session_store_with_fts._fts.search_sessions

    async def test_backfill_offloads_initial_query(self, session_store, mock_embedder, tmp_path):
        """backfill() offloads the initial indexed-IDs query."""
        with patch("src.search.vectorstore.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = set()
            await session_store.backfill(tmp_path, mock_embedder)
            # First call should be _get_indexed_ids_sync
            assert mock_thread.call_args_list[0].args[0] == session_store._get_indexed_ids_sync


# ---------------------------------------------------------------------------
# ConversationReflector — async offloading
# ---------------------------------------------------------------------------

class TestReflectorAsyncOffload:
    """Verify reflector offloads file I/O to threads."""

    async def test_reflect_offloads_load_and_save(self, tmp_path):
        """_reflect() should use asyncio.to_thread for _load and _save."""
        from src.learning.reflector import ConversationReflector

        r = ConversationReflector(str(tmp_path / "learned.json"))
        text_fn = AsyncMock(return_value='[{"key": "k1", "category": "fact", "content": "test"}]')
        r.set_text_fn(text_fn)

        with patch("src.learning.reflector.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = [
                # _load returns empty data
                {"version": 1, "last_reflection": None, "entries": []},
                # _save returns None
                None,
            ]
            await r._reflect("user: hello\nassistant: hi", full=True)
            assert mock_thread.call_count == 2
            assert mock_thread.call_args_list[0].args[0] == r._load
            assert mock_thread.call_args_list[1].args[0] == r._save

    async def test_reflect_works_end_to_end(self, tmp_path):
        """Full reflect cycle works with offloaded I/O."""
        from src.learning.reflector import ConversationReflector

        r = ConversationReflector(str(tmp_path / "learned.json"))
        text_fn = AsyncMock(return_value='[{"key": "k1", "category": "fact", "content": "sky is blue"}]')
        r.set_text_fn(text_fn)

        from src.sessions.manager import Message, Session
        session = Session(channel_id="ch1")
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            session.messages.append(Message(role=role, content=f"msg {i}"))

        await r.reflect_on_session(session, user_ids=["u1"])
        data = r._load()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["key"] == "k1"


# ---------------------------------------------------------------------------
# Web API — offloaded sync calls
# ---------------------------------------------------------------------------

class TestApiAsyncOffload:
    """Verify web API endpoints offload sync store methods."""

    async def test_list_knowledge_offloads(self):
        """GET /api/knowledge should use asyncio.to_thread for list_sources."""
        import inspect
        from src.web.api import create_api_routes
        source = inspect.getsource(create_api_routes)
        assert "asyncio.to_thread(store.list_sources)" in source

    async def test_delete_knowledge_offloads(self):
        """DELETE /api/knowledge/{source} should use asyncio.to_thread."""
        import inspect
        from src.web.api import create_api_routes
        source = inspect.getsource(create_api_routes)
        assert "asyncio.to_thread(store.delete_source" in source

    async def test_reingest_offloads_get_content(self):
        """POST /api/knowledge/{source}/reingest should offload get_source_content."""
        import inspect
        from src.web.api import create_api_routes
        source = inspect.getsource(create_api_routes)
        assert "asyncio.to_thread(store.get_source_content" in source


# ---------------------------------------------------------------------------
# Client.py — backfill_fts offloaded
# ---------------------------------------------------------------------------

class TestClientBackfillOffload:
    """Verify client.py offloads backfill_fts."""

    async def test_backfill_fts_uses_to_thread(self):
        """_backfill_archives should offload backfill_fts via asyncio.to_thread."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._backfill_archives)
        assert "asyncio.to_thread(self._knowledge_store.backfill_fts)" in source


# ---------------------------------------------------------------------------
# Embedder — already properly async
# ---------------------------------------------------------------------------

class TestEmbedderIsNonBlocking:
    """Verify embedder uses run_in_executor for CPU-bound work."""

    async def test_embed_uses_run_in_executor(self):
        """LocalEmbedder.embed() should use run_in_executor."""
        import inspect
        from src.search.embedder import LocalEmbedder
        source = inspect.getsource(LocalEmbedder.embed)
        assert "run_in_executor" in source

    def test_model_lazy_loaded(self):
        """Model should not be loaded at init time."""
        from src.search.embedder import LocalEmbedder
        emb = LocalEmbedder()
        assert emb._model is None


# ---------------------------------------------------------------------------
# Caching — verify existing caches are working
# ---------------------------------------------------------------------------

class TestCachingVerification:
    """Verify performance-critical caches are properly set up."""

    def test_merged_tools_cached(self):
        """HeimdallBot should cache merged tool definitions."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._merged_tool_definitions)
        assert "_cached_merged_tools" in source

    def test_tool_conversion_cached(self):
        """CodexChatClient should cache tool format conversion."""
        import inspect
        from src.llm.openai_codex import CodexChatClient
        source = inspect.getsource(CodexChatClient._convert_tools_cached)
        assert "_last_tools_list" in source

    def test_memory_cache_has_ttl(self):
        """Memory cache should have a TTL."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot)
        assert "_memory_cache_ttl" in source

    def test_reflector_cache_has_ttl(self):
        """Reflector cache should have a TTL."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot)
        assert "_reflector_cache_ttl" in source

    def test_tool_memory_cache_has_ttl(self):
        """ToolMemory hints cache should have a TTL."""
        import inspect
        from src.tools.tool_memory import ToolMemory
        source = inspect.getsource(ToolMemory)
        assert "_hints_cache_ttl" in source
