"""Tests for cross-session conversation continuity.

When a session expires and a new one starts for the same channel, the bot
carries forward the archived session's summary (if < 48 hours old) so the
LLM retains context from the previous conversation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.sessions.manager import (
    CONTINUITY_MAX_AGE,
    SessionManager,
    Session,
)


@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=10,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


# ---------------------------------------------------------------------------
# _find_recent_summary
# ---------------------------------------------------------------------------

class TestFindRecentSummary:
    def test_no_archive_dir(self, session_mgr: SessionManager):
        """Returns empty when archive directory doesn't exist."""
        result = session_mgr._find_recent_summary("ch1")
        assert result == ""

    def test_no_matching_archives(self, session_mgr: SessionManager):
        """Returns empty when no archives match the channel."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        data = {
            "channel_id": "other_ch",
            "messages": [],
            "last_active": time.time(),
            "summary": "Some summary",
        }
        (archive_dir / "other_ch_12345.json").write_text(json.dumps(data))
        result = session_mgr._find_recent_summary("ch1")
        assert result == ""

    def test_finds_recent_summary(self, session_mgr: SessionManager):
        """Returns summary from a recent archive."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": time.time() - 3600,  # 1 hour ago
            "summary": "Discussed server deployment",
        }
        ts = int(time.time() - 3600)
        (archive_dir / f"ch1_{ts}.json").write_text(json.dumps(data))
        result = session_mgr._find_recent_summary("ch1")
        assert result == "Discussed server deployment"

    def test_ignores_old_archive(self, session_mgr: SessionManager):
        """Archives older than 48 hours are ignored."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        old_time = time.time() - CONTINUITY_MAX_AGE - 3600  # 49 hours ago
        data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": old_time,
            "summary": "Old summary",
        }
        (archive_dir / f"ch1_{int(old_time)}.json").write_text(json.dumps(data))
        result = session_mgr._find_recent_summary("ch1")
        assert result == ""

    def test_ignores_archive_without_summary(self, session_mgr: SessionManager):
        """Archives with empty summary are ignored."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        data = {
            "channel_id": "ch1",
            "messages": [{"role": "user", "content": "hi"}],
            "last_active": time.time() - 600,
            "summary": "",
        }
        ts = int(time.time() - 600)
        (archive_dir / f"ch1_{ts}.json").write_text(json.dumps(data))
        result = session_mgr._find_recent_summary("ch1")
        assert result == ""

    def test_picks_most_recent(self, session_mgr: SessionManager):
        """When multiple archives exist, picks the most recent one."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        # Older archive
        old_ts = int(time.time() - 7200)  # 2 hours ago
        old_data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": time.time() - 7200,
            "summary": "Older summary",
        }
        (archive_dir / f"ch1_{old_ts}.json").write_text(json.dumps(old_data))
        # Newer archive
        new_ts = int(time.time() - 1800)  # 30 min ago
        new_data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": time.time() - 1800,
            "summary": "Newer summary",
        }
        (archive_dir / f"ch1_{new_ts}.json").write_text(json.dumps(new_data))
        result = session_mgr._find_recent_summary("ch1")
        assert result == "Newer summary"

    def test_corrupt_archive_skipped(self, session_mgr: SessionManager):
        """Corrupt JSON files are skipped without error."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        ts = int(time.time() - 600)
        (archive_dir / f"ch1_{ts}.json").write_text("not valid json{{{")
        result = session_mgr._find_recent_summary("ch1")
        assert result == ""

    def test_missing_summary_key(self, session_mgr: SessionManager):
        """Archive without 'summary' key is treated as empty summary."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": time.time() - 600,
        }
        ts = int(time.time() - 600)
        (archive_dir / f"ch1_{ts}.json").write_text(json.dumps(data))
        result = session_mgr._find_recent_summary("ch1")
        assert result == ""

    def test_missing_last_active_treated_as_old(self, session_mgr: SessionManager):
        """Archive without 'last_active' defaults to 0 (very old)."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        data = {
            "channel_id": "ch1",
            "messages": [],
            "summary": "Should be ignored",
        }
        ts = int(time.time() - 600)
        (archive_dir / f"ch1_{ts}.json").write_text(json.dumps(data))
        result = session_mgr._find_recent_summary("ch1")
        assert result == ""

    def test_boundary_exactly_48_hours(self, session_mgr: SessionManager):
        """Archive exactly at the boundary (48h) is excluded (strict <)."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        boundary_time = time.time() - CONTINUITY_MAX_AGE
        data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": boundary_time,
            "summary": "Edge case summary",
        }
        (archive_dir / f"ch1_{int(boundary_time)}.json").write_text(json.dumps(data))
        result = session_mgr._find_recent_summary("ch1")
        assert result == ""


# ---------------------------------------------------------------------------
# get_or_create with continuity
# ---------------------------------------------------------------------------

class TestGetOrCreateContinuity:
    def test_new_session_no_archive(self, session_mgr: SessionManager):
        """New session without any archive has no summary."""
        session = session_mgr.get_or_create("ch1")
        assert session.summary == ""

    def test_new_session_carries_forward_summary(self, session_mgr: SessionManager):
        """New session picks up summary from a recent archive."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": time.time() - 1800,
            "summary": "Talked about disk usage",
        }
        ts = int(time.time() - 1800)
        (archive_dir / f"ch1_{ts}.json").write_text(json.dumps(data))
        session = session_mgr.get_or_create("ch1")
        assert session.summary.startswith("[Continuing from previous conversation]")
        assert "Talked about disk usage" in session.summary

    def test_prefix_format(self, session_mgr: SessionManager):
        """The carried-forward summary has the exact expected prefix."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": time.time() - 600,
            "summary": "DNS config changes",
        }
        ts = int(time.time() - 600)
        (archive_dir / f"ch1_{ts}.json").write_text(json.dumps(data))
        session = session_mgr.get_or_create("ch1")
        assert session.summary == "[Continuing from previous conversation] DNS config changes"

    def test_existing_session_not_affected(self, session_mgr: SessionManager):
        """Returning an existing session does not re-check archives."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": time.time() - 600,
            "summary": "Archive summary",
        }
        ts = int(time.time() - 600)
        (archive_dir / f"ch1_{ts}.json").write_text(json.dumps(data))
        # First call picks up continuity
        s1 = session_mgr.get_or_create("ch1")
        assert "Archive summary" in s1.summary
        # Manually clear summary
        s1.summary = ""
        # Second call should return same session, not re-read archive
        s2 = session_mgr.get_or_create("ch1")
        assert s2 is s1
        assert s2.summary == ""

    def test_old_archive_not_carried_forward(self, session_mgr: SessionManager):
        """Archives older than 48h do not affect new sessions."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        old_time = time.time() - CONTINUITY_MAX_AGE - 7200
        data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": old_time,
            "summary": "Very old summary",
        }
        (archive_dir / f"ch1_{int(old_time)}.json").write_text(json.dumps(data))
        session = session_mgr.get_or_create("ch1")
        assert session.summary == ""

    def test_history_includes_carried_summary(self, session_mgr: SessionManager):
        """get_history() properly includes the carried-forward summary."""
        archive_dir = session_mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        data = {
            "channel_id": "ch1",
            "messages": [],
            "last_active": time.time() - 600,
            "summary": "Previous context",
        }
        ts = int(time.time() - 600)
        (archive_dir / f"ch1_{ts}.json").write_text(json.dumps(data))
        session_mgr.get_or_create("ch1")
        session_mgr.add_message("ch1", "user", "what were we doing?")
        history = session_mgr.get_history("ch1")
        # summary pair + user message = 3
        assert len(history) == 3
        assert "[Continuing from previous conversation]" in history[0]["content"]
        assert "Previous context" in history[0]["content"]


# ---------------------------------------------------------------------------
# End-to-end: prune → new session → continuity
# ---------------------------------------------------------------------------

class TestPruneThenContinuity:
    def test_prune_then_new_session_carries_summary(self, tmp_dir: Path):
        """Full cycle: session expires, gets pruned/archived, new session picks up summary."""
        persist = str(tmp_dir / "sessions")
        mgr = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)

        # Create a session with a summary (simulating post-compaction state)
        session = mgr.get_or_create("ch1")
        session.summary = "Discussed Docker container restarts"
        mgr.add_message("ch1", "user", "thanks")
        mgr.add_message("ch1", "assistant", "you're welcome")

        # Expire and prune
        session.last_active = time.time() - 7200
        mgr.prune()
        assert "ch1" not in mgr._sessions

        # New session should carry forward the summary
        new_session = mgr.get_or_create("ch1")
        assert new_session.summary.startswith("[Continuing from previous conversation]")
        assert "Discussed Docker container restarts" in new_session.summary

    def test_prune_no_summary_no_continuity(self, tmp_dir: Path):
        """If the archived session had no summary, the new session starts clean."""
        persist = str(tmp_dir / "sessions")
        mgr = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)

        session = mgr.get_or_create("ch1")
        mgr.add_message("ch1", "user", "short chat")
        # No summary set

        session.last_active = time.time() - 7200
        mgr.prune()

        new_session = mgr.get_or_create("ch1")
        assert new_session.summary == ""

    def test_different_channels_independent(self, tmp_dir: Path):
        """Continuity only applies to the same channel."""
        persist = str(tmp_dir / "sessions")
        mgr = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)

        # Archive for ch1 with summary
        s1 = mgr.get_or_create("ch1")
        s1.summary = "Ch1 context"
        mgr.add_message("ch1", "user", "msg")
        s1.last_active = time.time() - 7200
        mgr.prune()

        # New session for ch2 should not get ch1's summary
        new_s2 = mgr.get_or_create("ch2")
        assert new_s2.summary == ""

        # New session for ch1 should get it
        new_s1 = mgr.get_or_create("ch1")
        assert "Ch1 context" in new_s1.summary

    def test_multiple_archives_picks_latest(self, tmp_dir: Path):
        """Multiple archived sessions for a channel — picks the most recent."""
        persist = str(tmp_dir / "sessions")
        mgr = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)

        # First session
        s1 = mgr.get_or_create("ch1")
        s1.summary = "First conversation"
        mgr.add_message("ch1", "user", "msg1")
        s1.last_active = time.time() - 7200
        mgr.prune()

        # Second session (slightly more recent archive)
        s2 = mgr.get_or_create("ch1")
        # Clear the carried-forward summary and set a new one
        s2.summary = "Second conversation"
        mgr.add_message("ch1", "user", "msg2")
        s2.last_active = time.time() - 3700  # just over 1 hour
        mgr.prune()

        # Third session should pick up "Second conversation"
        s3 = mgr.get_or_create("ch1")
        assert "Second conversation" in s3.summary


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_continuity_max_age_48_hours(self):
        assert CONTINUITY_MAX_AGE == 48 * 3600

    def test_continuity_max_age_seconds(self):
        assert CONTINUITY_MAX_AGE == 172800
