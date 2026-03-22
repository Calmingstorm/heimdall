"""Tests for sessions/manager.py — covering compaction, archive search, reflection,
indexing, continuity carry-forward, and error handling."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.sessions.manager import (
    COMPACTION_THRESHOLD,
    CONTINUITY_MAX_AGE,
    Message,
    Session,
    SessionManager,
)


@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=10,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


# ---------------------------------------------------------------------------
# get_history_with_compaction + _compact (lines 136-211)
# ---------------------------------------------------------------------------

class TestCompaction:
    @pytest.mark.asyncio
    async def test_compaction_triggers_above_threshold(self, tmp_dir):
        """When message count exceeds COMPACTION_THRESHOLD, _compact is called."""
        # max_history must be small enough so keep_count (max_history // 2)
        # is less than the total message count, otherwise nothing gets summarized.
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "test_ch"
        session = mgr.get_or_create(channel)

        # Add messages above threshold
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        assert len(session.messages) > COMPACTION_THRESHOLD

        compact_fn = AsyncMock(return_value="Summarized conversation about servers.")
        mgr.set_compaction_fn(compact_fn)
        history = await mgr.get_history_with_compaction(channel)

        # Compaction should have been triggered via the compaction fn
        compact_fn.assert_awaited_once()
        # Messages should be trimmed (kept = max_history // 2 = 15)
        assert len(session.messages) == 15
        # Summary should be set
        assert session.summary
        # History should include the summary preamble
        assert any("Previous conversation summary" in m["content"] for m in history)

    @pytest.mark.asyncio
    async def test_no_compaction_below_threshold(self, session_mgr):
        """When message count is below threshold, no compaction occurs."""
        channel = "test_ch"
        for i in range(5):
            session_mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        history = await session_mgr.get_history_with_compaction(channel)

        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_compaction_merges_existing_summary(self, tmp_dir):
        """When session already has a summary, compaction includes it for merging."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "test_ch"
        session = mgr.get_or_create(channel)
        session.summary = "We previously discussed DNS setup."

        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        captured_content = []

        async def capture_fn(messages, system):
            captured_content.append(messages[0]["content"])
            return "Summarized conversation about servers."

        mgr.set_compaction_fn(capture_fn)
        await mgr.get_history_with_compaction(channel)

        # The summarization call should include the previous summary
        convo_text = captured_content[0]
        assert "Previous summary" in convo_text
        assert "DNS setup" in convo_text

    @pytest.mark.asyncio
    async def test_compaction_triggers_reflection(self, tmp_dir):
        """Compaction triggers reflection when reflector is set and enough messages discarded."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_compacted = AsyncMock()

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
            reflector=mock_reflector,
        )
        channel = "test_ch"
        for i in range(COMPACTION_THRESHOLD + 10):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        mgr.set_compaction_fn(AsyncMock(return_value="Summarized conversation."))
        await mgr.get_history_with_compaction(channel)

        # Wait for background reflection task
        await asyncio.sleep(0.1)
        mock_reflector.reflect_on_compacted.assert_called_once()

    @pytest.mark.asyncio
    async def test_compaction_failure_fallback(self, tmp_dir):
        """When compaction API call fails, fallback trims without summary."""
        mgr = SessionManager(
            max_history=20, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "test_ch"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        mgr.set_compaction_fn(AsyncMock(side_effect=RuntimeError("API down")))

        await mgr.get_history_with_compaction(channel)

        session = mgr.get_or_create(channel)
        # Should have been trimmed to max_history
        assert len(session.messages) <= 20


# ---------------------------------------------------------------------------
# _archive_session with reflector and vector store (lines 236-262)
# ---------------------------------------------------------------------------

class TestArchiveSession:
    @pytest.mark.asyncio
    async def test_archive_triggers_reflection(self, tmp_dir):
        """Archiving a session with 3+ messages triggers reflection."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_session = AsyncMock()

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
            reflector=mock_reflector,
        )
        channel = "test_ch"
        for i in range(5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        session = mgr.get_or_create(channel)
        session.last_active = time.time() - 7200
        mgr.prune()

        await asyncio.sleep(0.1)
        mock_reflector.reflect_on_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_triggers_vector_indexing(self, tmp_dir):
        """Archiving a session with vector store triggers indexing."""
        mock_vs = MagicMock()
        mock_vs.available = True
        mock_vs.index_session = AsyncMock()
        mock_embedder = MagicMock()

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
            vector_store=mock_vs,
            embedder=mock_embedder,
        )
        channel = "test_ch"
        for i in range(3):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        session = mgr.get_or_create(channel)
        session.last_active = time.time() - 7200
        mgr.prune()

        await asyncio.sleep(0.1)
        mock_vs.index_session.assert_called_once()

    def test_archive_empty_session_skips(self, tmp_dir):
        """Archiving a session with no messages doesn't create archive file."""
        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "test_ch"
        mgr.get_or_create(channel)
        session = mgr.get_or_create(channel)
        session.last_active = time.time() - 7200
        mgr.prune()

        archive_dir = Path(mgr.persist_dir) / "archive"
        if archive_dir.exists():
            assert list(archive_dir.glob("*.json")) == []


# ---------------------------------------------------------------------------
# _safe_index error handling (lines 264-269)
# ---------------------------------------------------------------------------

class TestSafeIndex:
    @pytest.mark.asyncio
    async def test_safe_index_catches_errors(self, tmp_dir):
        """_safe_index catches and logs indexing errors without crashing."""
        mock_vs = MagicMock()
        mock_vs.available = True
        mock_vs.index_session = AsyncMock(side_effect=RuntimeError("ChromaDB down"))
        mock_embedder = MagicMock()

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
            vector_store=mock_vs,
            embedder=mock_embedder,
        )
        channel = "test_ch"
        for i in range(3):
            mgr.add_message(channel, "user", f"msg {i}")

        session = mgr.get_or_create(channel)
        session.last_active = time.time() - 7200
        # Should not raise
        mgr.prune()
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# _safe_reflect and _safe_reflect_compacted error handling (lines 362-381)
# ---------------------------------------------------------------------------

class TestSafeReflect:
    @pytest.mark.asyncio
    async def test_safe_reflect_catches_errors(self, tmp_dir):
        """_safe_reflect catches reflection errors without crashing."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_session = AsyncMock(
            side_effect=RuntimeError("Reflection API failed"),
        )

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
            reflector=mock_reflector,
        )
        channel = "test_ch"
        for i in range(5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        session = mgr.get_or_create(channel)
        session.last_active = time.time() - 7200
        mgr.prune()
        await asyncio.sleep(0.1)
        # Should have been called despite the error
        mock_reflector.reflect_on_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_safe_reflect_compacted_catches_errors(self, tmp_dir):
        """_safe_reflect_compacted catches errors without crashing."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_compacted = AsyncMock(
            side_effect=RuntimeError("Reflection failed"),
        )

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
            reflector=mock_reflector,
        )
        channel = "test_ch"
        for i in range(COMPACTION_THRESHOLD + 10):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        mgr.set_compaction_fn(AsyncMock(return_value="Summarized conversation."))
        await mgr.get_history_with_compaction(channel)
        await asyncio.sleep(0.1)
        # Should have been called
        mock_reflector.reflect_on_compacted.assert_called_once()


# ---------------------------------------------------------------------------
# search_history — archive search, summary match, hybrid search (lines 271-341)
# ---------------------------------------------------------------------------

class TestSearchHistoryAdvanced:
    @pytest.mark.asyncio
    async def test_search_finds_summary_match(self, session_mgr):
        """search_history matches on session summaries."""
        session = session_mgr.get_or_create("ch1")
        session.summary = "Discussed the DNS server configuration"
        results = await session_mgr.search_history("DNS")
        assert len(results) >= 1
        assert any(r["type"] == "summary" for r in results)

    @pytest.mark.asyncio
    async def test_search_archives(self, session_mgr):
        """search_history searches archived sessions."""
        # Create and archive a session
        session_mgr.add_message("ch1", "user", "deploy the new nginx config")
        session = session_mgr.get_or_create("ch1")
        session.last_active = time.time() - 7200
        session_mgr.prune()

        results = await session_mgr.search_history("nginx")
        assert len(results) >= 1
        assert any("nginx" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_archive_summary(self, session_mgr):
        """search_history finds matches in archived summaries."""
        session_mgr.add_message("ch1", "user", "hello")
        session = session_mgr.get_or_create("ch1")
        session.summary = "Configured the Prometheus alerting rules"
        session.last_active = time.time() - 7200
        session_mgr.prune()

        results = await session_mgr.search_history("Prometheus alerting")
        assert len(results) >= 1
        assert any(r["type"] == "summary" for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, session_mgr):
        """search_history stops at limit."""
        for i in range(20):
            session_mgr.add_message("ch1", "user", f"deploy item {i}")

        results = await session_mgr.search_history("deploy", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_with_hybrid_fallback(self, tmp_dir):
        """search_history uses hybrid search when available and results < limit."""
        mock_vs = MagicMock()
        mock_vs.available = True
        mock_vs.search_hybrid = AsyncMock(return_value=[
            {
                "type": "user",
                "content": "semantic match for servers",
                "timestamp": 1234567890.0,
                "channel_id": "ch_hybrid",
            },
        ])
        mock_embedder = MagicMock()

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
            vector_store=mock_vs,
            embedder=mock_embedder,
        )
        results = await mgr.search_history("servers", limit=10)
        assert len(results) >= 1
        mock_vs.search_hybrid.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_hybrid_deduplicates(self, tmp_dir):
        """search_history deduplicates results between keyword and hybrid search."""
        mock_vs = MagicMock()
        mock_vs.available = True
        mock_vs.search_hybrid = AsyncMock(return_value=[
            {
                "type": "user",
                "content": "deploy the server",
                "timestamp": 100.0,
                "channel_id": "ch1",
            },
        ])
        mock_embedder = MagicMock()

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
            vector_store=mock_vs,
            embedder=mock_embedder,
        )
        # Add the same message that hybrid search will return
        mgr.add_message("ch1", "user", "deploy the server")
        session = mgr.get_or_create("ch1")
        # Force same timestamp for dedup
        session.messages[0].timestamp = 100.0

        results = await mgr.search_history("deploy", limit=10)
        # Should not have duplicate
        deploy_results = [r for r in results if "deploy" in r["content"]]
        channels_timestamps = [(r["channel_id"], r["timestamp"]) for r in deploy_results]
        assert len(channels_timestamps) == len(set(channels_timestamps))

    @pytest.mark.asyncio
    async def test_search_hybrid_error_returns_keyword_results(self, tmp_dir):
        """search_history returns keyword results even if hybrid search fails."""
        mock_vs = MagicMock()
        mock_vs.available = True
        mock_vs.search_hybrid = AsyncMock(side_effect=RuntimeError("ChromaDB error"))
        mock_embedder = MagicMock()

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
            vector_store=mock_vs,
            embedder=mock_embedder,
        )
        mgr.add_message("ch1", "user", "deploy the server")
        results = await mgr.search_history("deploy", limit=10)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Continuity carry-forward from _find_recent_summary (lines 75-93)
# ---------------------------------------------------------------------------

class TestContinuityCarryForward:
    def test_carries_forward_recent_summary(self, session_mgr):
        """New session gets summary from a recent archive."""
        # Create archive with summary
        archive_dir = Path(session_mgr.persist_dir) / "archive"
        archive_dir.mkdir(parents=True)
        archive_data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": time.time() - 3600,  # 1 hour ago (within 48h window)
            "summary": "Discussed server monitoring setup",
        }
        archive_path = archive_dir / f"ch1_{int(time.time() - 3600)}.json"
        archive_path.write_text(json.dumps(archive_data))

        session = session_mgr.get_or_create("ch1")
        assert "Continuing from previous conversation" in session.summary
        assert "monitoring" in session.summary

    def test_ignores_old_summary(self, session_mgr):
        """Archives older than CONTINUITY_MAX_AGE are not carried forward."""
        archive_dir = Path(session_mgr.persist_dir) / "archive"
        archive_dir.mkdir(parents=True)
        old_time = time.time() - CONTINUITY_MAX_AGE - 3600
        archive_data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": old_time,
            "summary": "Very old discussion",
        }
        archive_path = archive_dir / f"ch1_{int(old_time)}.json"
        archive_path.write_text(json.dumps(archive_data))

        session = session_mgr.get_or_create("ch1")
        assert session.summary == ""

    def test_picks_most_recent_summary(self, session_mgr):
        """When multiple archives exist, picks the most recent one."""
        archive_dir = Path(session_mgr.persist_dir) / "archive"
        archive_dir.mkdir(parents=True)

        # Older archive
        t1 = time.time() - 7200
        d1 = {"channel_id": "ch1", "messages": [], "last_active": t1, "summary": "Old topic"}
        (archive_dir / f"ch1_{int(t1)}.json").write_text(json.dumps(d1))

        # Newer archive
        t2 = time.time() - 1800
        d2 = {"channel_id": "ch1", "messages": [], "last_active": t2, "summary": "Recent topic"}
        (archive_dir / f"ch1_{int(t2)}.json").write_text(json.dumps(d2))

        session = session_mgr.get_or_create("ch1")
        assert "Recent topic" in session.summary


# ---------------------------------------------------------------------------
# scrub_secrets edge case (line 348)
# ---------------------------------------------------------------------------

class TestScrubSecretsEdge:
    def test_scrub_nonexistent_session(self, session_mgr):
        """scrub_secrets on nonexistent session returns False."""
        result = session_mgr.scrub_secrets("nonexistent_ch", "secret content")
        assert result is False


# ---------------------------------------------------------------------------
# remove_last_message (lines 104-117)
# ---------------------------------------------------------------------------

class TestRemoveLastMessage:
    def test_remove_matching_role(self, session_mgr):
        """remove_last_message removes last message if role matches."""
        session_mgr.add_message("ch1", "user", "hello")
        session_mgr.add_message("ch1", "assistant", "hi there")
        removed = session_mgr.remove_last_message("ch1", "assistant")
        assert removed is True
        session = session_mgr.get_or_create("ch1")
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"

    def test_remove_non_matching_role(self, session_mgr):
        """remove_last_message returns False if role doesn't match."""
        session_mgr.add_message("ch1", "user", "hello")
        removed = session_mgr.remove_last_message("ch1", "assistant")
        assert removed is False
        session = session_mgr.get_or_create("ch1")
        assert len(session.messages) == 1

    def test_remove_from_empty_session(self, session_mgr):
        """remove_last_message on empty session returns False."""
        session_mgr.get_or_create("ch1")
        removed = session_mgr.remove_last_message("ch1", "user")
        assert removed is False

    def test_remove_from_nonexistent_session(self, session_mgr):
        """remove_last_message on nonexistent session returns False."""
        removed = session_mgr.remove_last_message("nonexistent", "user")
        assert removed is False


# ---------------------------------------------------------------------------
# add_message with user_id (lines 95-102)
# ---------------------------------------------------------------------------

class TestAddMessageUserId:
    def test_user_id_stored_on_user_message(self, session_mgr):
        """add_message stores user_id when role is 'user'."""
        session_mgr.add_message("ch1", "user", "hello", user_id="user123")
        session = session_mgr.get_or_create("ch1")
        assert session.last_user_id == "user123"

    def test_user_id_not_stored_on_assistant_message(self, session_mgr):
        """add_message does not update last_user_id for assistant messages."""
        session_mgr.add_message("ch1", "user", "hello", user_id="user123")
        session_mgr.add_message("ch1", "assistant", "hi")
        session = session_mgr.get_or_create("ch1")
        assert session.last_user_id == "user123"


# ---------------------------------------------------------------------------
# load with corrupted file (lines 402-403)
# ---------------------------------------------------------------------------

class TestLoadCorruptedFile:
    def test_load_corrupted_json(self, tmp_dir):
        """load() handles corrupted session files gracefully."""
        persist = tmp_dir / "sessions"
        persist.mkdir()
        # Write corrupted JSON
        (persist / "bad_channel.json").write_text("{invalid json!!!")

        mgr = SessionManager(max_history=10, max_age_hours=1, persist_dir=str(persist))
        mgr.load()  # Should not raise
        # Should have no sessions loaded
        assert len(mgr._sessions) == 0

    def test_load_valid_and_corrupted(self, tmp_dir):
        """load() loads valid files even when some are corrupted."""
        persist = tmp_dir / "sessions"
        persist.mkdir()

        # Valid session
        valid = {
            "channel_id": "good_ch",
            "messages": [{"role": "user", "content": "hello", "timestamp": 1.0}],
            "created_at": 1.0,
            "last_active": 1.0,
            "summary": "",
        }
        (persist / "good_ch.json").write_text(json.dumps(valid))

        # Corrupted session
        (persist / "bad_ch.json").write_text("not valid json at all")

        mgr = SessionManager(max_history=10, max_age_hours=1, persist_dir=str(persist))
        mgr.load()
        assert "good_ch" in mgr._sessions
        assert "bad_ch" not in mgr._sessions


# ---------------------------------------------------------------------------
# prune deletes session file (lines 228-230)
# ---------------------------------------------------------------------------

class TestPruneDeletesSessionFile:
    def test_prune_removes_session_file(self, session_mgr):
        """Prune removes the session JSON file after archiving."""
        session_mgr.add_message("ch1", "user", "test msg")
        session_mgr.save()

        session_file = Path(session_mgr.persist_dir) / "ch1.json"
        assert session_file.exists()

        session = session_mgr.get_or_create("ch1")
        session.last_active = time.time() - 7200
        session_mgr.prune()

        assert not session_file.exists()
