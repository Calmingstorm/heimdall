"""Tests for sessions/manager.py."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.sessions.manager import SessionManager, Message, Session


@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=10,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


class TestGetOrCreate:
    def test_creates_new_session(self, session_mgr: SessionManager):
        session = session_mgr.get_or_create("ch1")
        assert session.channel_id == "ch1"
        assert session.messages == []

    def test_returns_existing(self, session_mgr: SessionManager):
        s1 = session_mgr.get_or_create("ch1")
        s2 = session_mgr.get_or_create("ch1")
        assert s1 is s2

    def test_updates_last_active(self, session_mgr: SessionManager):
        s1 = session_mgr.get_or_create("ch1")
        t1 = s1.last_active
        time.sleep(0.01)
        s2 = session_mgr.get_or_create("ch1")
        assert s2.last_active >= t1


class TestAddMessage:
    def test_adds_message(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "hello")
        session = session_mgr.get_or_create("ch1")
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "hello"

    def test_multiple_messages(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "hello")
        session_mgr.add_message("ch1", "assistant", "hi")
        history = session_mgr.get_history("ch1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_stores_user_id_on_message(self, session_mgr: SessionManager):
        """add_message should store user_id on user messages."""
        session_mgr.add_message("ch1", "user", "hello", user_id="42")
        session = session_mgr.get_or_create("ch1")
        assert session.messages[0].user_id == "42"

    def test_assistant_message_no_user_id(self, session_mgr: SessionManager):
        """Assistant messages should not get a user_id."""
        session_mgr.add_message("ch1", "assistant", "hi")
        session = session_mgr.get_or_create("ch1")
        assert session.messages[0].user_id is None

    def test_user_id_persists_through_save_load(self, session_mgr: SessionManager):
        """user_id on messages should survive save/load round-trip."""
        session_mgr.add_message("ch1", "user", "hello", user_id="42")
        session_mgr.add_message("ch1", "assistant", "hi")
        session_mgr.save()

        mgr2 = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=session_mgr.persist_dir,
        )
        mgr2.load()
        session = mgr2.get_or_create("ch1")
        assert session.messages[0].user_id == "42"
        assert session.messages[1].user_id is None


class TestGetHistory:
    def test_empty_history(self, session_mgr: SessionManager):
        history = session_mgr.get_history("ch1")
        assert history == []

    def test_with_summary(self, session_mgr: SessionManager):
        session = session_mgr.get_or_create("ch1")
        session.summary = "Previously discussed servers."
        session_mgr.add_message("ch1", "user", "what next?")
        history = session_mgr.get_history("ch1")
        assert len(history) == 3  # summary pair + new message
        assert "Previous conversation summary" in history[0]["content"]
        assert history[1]["role"] == "assistant"


class TestPrune:
    def test_prunes_expired(self, session_mgr: SessionManager):
        session = session_mgr.get_or_create("ch1")
        session_mgr.add_message("ch1", "user", "test")
        # Force expiry
        session.last_active = time.time() - 7200  # 2 hours ago
        pruned = session_mgr.prune()
        assert pruned == 1

    def test_keeps_active(self, session_mgr: SessionManager):
        session_mgr.get_or_create("ch1")
        pruned = session_mgr.prune()
        assert pruned == 0

    def test_archives_on_prune(self, session_mgr: SessionManager):
        session = session_mgr.get_or_create("ch1")
        session_mgr.add_message("ch1", "user", "remember this")
        session.last_active = time.time() - 7200
        session_mgr.prune()
        archive_dir = Path(session_mgr.persist_dir) / "archive"
        assert archive_dir.exists()
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) == 1
        data = json.loads(archives[0].read_text())
        assert data["channel_id"] == "ch1"


class TestReset:
    def test_reset_removes_session(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "hello")
        session_mgr.reset("ch1")
        history = session_mgr.get_history("ch1")
        assert history == []

    def test_reset_nonexistent(self, session_mgr: SessionManager):
        # Should not raise
        session_mgr.reset("nonexistent")


class TestScrubSecrets:
    def test_scrubs_matching(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "my password is hunter2")
        session_mgr.add_message("ch1", "user", "harmless message")
        removed = session_mgr.scrub_secrets("ch1", "my password is hunter2")
        assert removed is True
        session = session_mgr.get_or_create("ch1")
        assert len(session.messages) == 1

    def test_scrub_no_match(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "hello")
        removed = session_mgr.scrub_secrets("ch1", "not in history")
        assert removed is False


class TestPersistence:
    def test_save_and_load(self, tmp_dir: Path):
        mgr1 = SessionManager(max_history=10, max_age_hours=1, persist_dir=str(tmp_dir / "sessions"))
        mgr1.add_message("ch1", "user", "hello")
        mgr1.add_message("ch1", "assistant", "hi")
        mgr1.save()

        mgr2 = SessionManager(max_history=10, max_age_hours=1, persist_dir=str(tmp_dir / "sessions"))
        mgr2.load()
        history = mgr2.get_history("ch1")
        assert len(history) == 2
        assert history[0]["content"] == "hello"


class TestSearchHistory:
    @pytest.mark.asyncio
    async def test_keyword_search_current(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "deploy the server")
        session_mgr.add_message("ch1", "assistant", "deploying now")
        results = await session_mgr.search_history("deploy")
        assert len(results) >= 1
        assert any("deploy" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_no_results(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "hello world")
        results = await session_mgr.search_history("nonexistent_term_xyz")
        assert results == []
