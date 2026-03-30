"""Tests for session 4 audit fixes:
- #13: Compaction fallback clears stale summary
- #14: save() only writes dirty sessions
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.sessions.manager import (
    COMPACTION_THRESHOLD,
    Message,
    Session,
    SessionManager,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=10,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


# ── #13: Compaction fallback preserves existing summary ──────────────


class TestCompactionFallbackPreservesSummary:
    """When compaction fails, the fallback should preserve the existing summary."""

    @pytest.mark.asyncio
    async def test_fallback_preserves_summary(self, tmp_dir):
        mgr = SessionManager(
            max_history=30,
            max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "test_ch"
        session = mgr.get_or_create(channel)
        session.summary = "Summary from a previous compaction."

        # Add enough messages to trigger compaction
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        # Make the compaction fn raise an error to trigger fallback
        mgr.set_compaction_fn(AsyncMock(side_effect=RuntimeError("API down")))

        await mgr.get_history_with_compaction(channel)

        # Summary should be preserved, not cleared (Round 21 fix)
        assert session.summary == "Summary from a previous compaction."
        # Messages should be trimmed to max_history
        assert len(session.messages) == 30


# ── #14: save() only writes dirty sessions ──────────────────────────


class TestDirtySave:
    """save() only persists sessions that changed since the last save."""

    def test_save_only_writes_dirty(self, session_mgr: SessionManager):
        """Only changed sessions are written to disk."""
        session_mgr.add_message("ch1", "user", "hello")
        session_mgr.add_message("ch2", "user", "hi there")
        session_mgr.save()

        # Both files should exist now
        assert (session_mgr.persist_dir / "ch1.json").exists()
        assert (session_mgr.persist_dir / "ch2.json").exists()

        # Record modification times
        mtime1 = (session_mgr.persist_dir / "ch1.json").stat().st_mtime_ns
        mtime2 = (session_mgr.persist_dir / "ch2.json").stat().st_mtime_ns

        # Only modify ch1
        session_mgr.add_message("ch1", "assistant", "response")
        session_mgr.save()

        # ch1 should be rewritten, ch2 should not
        new_mtime1 = (session_mgr.persist_dir / "ch1.json").stat().st_mtime_ns
        new_mtime2 = (session_mgr.persist_dir / "ch2.json").stat().st_mtime_ns
        assert new_mtime1 > mtime1  # ch1 was rewritten
        assert new_mtime2 == mtime2  # ch2 was not touched

    def test_save_clears_dirty_set(self, session_mgr: SessionManager):
        """After save(), the dirty set should be empty."""
        session_mgr.add_message("ch1", "user", "msg")
        assert "ch1" in session_mgr._dirty
        session_mgr.save()
        assert len(session_mgr._dirty) == 0

    def test_no_dirty_no_write(self, session_mgr: SessionManager):
        """Calling save() with no dirty sessions writes nothing."""
        session_mgr.add_message("ch1", "user", "msg")
        session_mgr.save()
        mtime = (session_mgr.persist_dir / "ch1.json").stat().st_mtime_ns

        # Save again with no changes
        session_mgr.save()
        assert (session_mgr.persist_dir / "ch1.json").stat().st_mtime_ns == mtime

    def test_save_all_writes_everything(self, session_mgr: SessionManager):
        """save_all() writes all sessions regardless of dirty state."""
        session_mgr.add_message("ch1", "user", "hello")
        session_mgr.add_message("ch2", "user", "world")
        session_mgr.save()

        # Modify file content directly so we can detect rewrite
        path2 = session_mgr.persist_dir / "ch2.json"
        path2.write_text("{}")

        # save() won't rewrite ch2 (not dirty)
        session_mgr.save()
        assert json.loads(path2.read_text()) == {}

        # save_all() will rewrite everything
        session_mgr.save_all()
        data = json.loads(path2.read_text())
        assert data["channel_id"] == "ch2"

    def test_add_message_marks_dirty(self, session_mgr: SessionManager):
        """add_message marks the channel dirty."""
        session_mgr.add_message("ch1", "user", "test")
        assert "ch1" in session_mgr._dirty

    def test_remove_last_message_marks_dirty(self, session_mgr: SessionManager):
        """remove_last_message marks the channel dirty."""
        session_mgr.add_message("ch1", "user", "test")
        session_mgr._dirty.clear()
        session_mgr.remove_last_message("ch1", "user")
        assert "ch1" in session_mgr._dirty

    def test_get_or_create_new_marks_dirty(self, session_mgr: SessionManager):
        """Creating a new session marks it dirty."""
        session_mgr.get_or_create("new_ch")
        assert "new_ch" in session_mgr._dirty

    def test_get_or_create_existing_not_dirty(self, session_mgr: SessionManager):
        """Accessing an existing session does not mark it dirty."""
        session_mgr.get_or_create("ch1")
        session_mgr._dirty.clear()
        session_mgr.get_or_create("ch1")
        assert "ch1" not in session_mgr._dirty

    def test_scrub_secrets_marks_dirty(self, session_mgr: SessionManager):
        """scrub_secrets marks the channel dirty when messages are removed."""
        session_mgr.add_message("ch1", "user", "my password is hunter2")
        session_mgr._dirty.clear()
        session_mgr.scrub_secrets("ch1", "hunter2")
        assert "ch1" in session_mgr._dirty

    def test_scrub_secrets_no_match_not_dirty(self, session_mgr: SessionManager):
        """scrub_secrets does not mark dirty when nothing is removed."""
        session_mgr.add_message("ch1", "user", "normal message")
        session_mgr._dirty.clear()
        session_mgr.scrub_secrets("ch1", "nonexistent")
        assert "ch1" not in session_mgr._dirty
