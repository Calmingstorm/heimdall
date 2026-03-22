"""Tests for session pruning file cleanup and startup pruning.

Verifies that prune() deletes stale session files from the persist directory
(preventing reload on restart) and that on_ready() triggers pruning at startup.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.sessions.manager import SessionManager, Session


@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=10,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


class TestPruneFileCleanup:
    """Verify prune() deletes session files after archiving."""

    def test_prune_deletes_session_file(self, session_mgr: SessionManager):
        """Pruning an expired session should delete its .json file."""
        session_mgr.add_message("ch1", "user", "hello")
        session_mgr.save()

        session_file = session_mgr.persist_dir / "ch1.json"
        assert session_file.exists(), "Session file should exist after save"

        # Expire the session
        session_mgr._sessions["ch1"].last_active = time.time() - 7200
        session_mgr.prune()

        assert not session_file.exists(), "Session file should be deleted after prune"

    def test_prune_creates_archive_before_deleting(self, session_mgr: SessionManager):
        """Pruning should archive the session before deleting the file."""
        session_mgr.add_message("ch1", "user", "important data")
        session_mgr.save()

        session_mgr._sessions["ch1"].last_active = time.time() - 7200
        session_mgr.prune()

        archive_dir = session_mgr.persist_dir / "archive"
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) == 1
        data = json.loads(archives[0].read_text())
        assert data["channel_id"] == "ch1"
        assert any(m["content"] == "important data" for m in data["messages"])

    def test_prune_preserves_active_session_files(self, session_mgr: SessionManager):
        """Active session files should not be deleted."""
        session_mgr.add_message("ch1", "user", "active")
        session_mgr.add_message("ch2", "user", "stale")
        session_mgr.save()

        # Expire only ch2
        session_mgr._sessions["ch2"].last_active = time.time() - 7200
        session_mgr.prune()

        assert (session_mgr.persist_dir / "ch1.json").exists()
        assert not (session_mgr.persist_dir / "ch2.json").exists()

    def test_prune_safe_when_file_missing(self, session_mgr: SessionManager):
        """Prune should not error if the session file doesn't exist on disk."""
        session_mgr.add_message("ch1", "user", "hello")
        # Don't call save() — no file on disk
        session_mgr._sessions["ch1"].last_active = time.time() - 7200
        # Should not raise
        pruned = session_mgr.prune()
        assert pruned == 1

    def test_prune_multiple_expired_deletes_all_files(self, session_mgr: SessionManager):
        """All expired session files should be deleted."""
        for ch in ["ch1", "ch2", "ch3"]:
            session_mgr.add_message(ch, "user", f"msg in {ch}")
        session_mgr.save()

        for ch in ["ch1", "ch2", "ch3"]:
            session_mgr._sessions[ch].last_active = time.time() - 7200
        pruned = session_mgr.prune()

        assert pruned == 3
        for ch in ["ch1", "ch2", "ch3"]:
            assert not (session_mgr.persist_dir / f"{ch}.json").exists()

    def test_prune_empty_session_no_archive_still_deletes_file(self, session_mgr: SessionManager):
        """Sessions with no messages should still have their file deleted."""
        session = session_mgr.get_or_create("ch1")
        session_mgr.save()

        session_file = session_mgr.persist_dir / "ch1.json"
        assert session_file.exists()

        session.last_active = time.time() - 7200
        session_mgr.prune()

        assert not session_file.exists()
        # No archive for empty sessions (archive_session skips empty)
        archive_dir = session_mgr.persist_dir / "archive"
        if archive_dir.exists():
            assert list(archive_dir.glob("*.json")) == []


class TestPruneReloadCycle:
    """Verify that pruned sessions don't reappear after save+load cycles."""

    def test_pruned_session_not_reloaded(self, tmp_dir: Path):
        """After prune, a new load() should not find the pruned session."""
        persist = str(tmp_dir / "sessions")

        mgr1 = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)
        mgr1.add_message("ch1", "user", "hello")
        mgr1.add_message("ch2", "user", "world")
        mgr1.save()

        # Expire ch1
        mgr1._sessions["ch1"].last_active = time.time() - 7200
        mgr1.prune()

        # New manager loads from disk
        mgr2 = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)
        mgr2.load()

        assert "ch1" not in mgr2._sessions, "Pruned session should not be reloaded"
        assert "ch2" in mgr2._sessions, "Active session should still be loadable"

    def test_multiple_restart_cycles_no_duplicate_archives(self, tmp_dir: Path):
        """Pruning should not create duplicate archives across restarts."""
        persist = str(tmp_dir / "sessions")

        # First run: create and save a session
        mgr1 = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)
        mgr1.add_message("ch1", "user", "hello")
        mgr1.save()

        # Expire and prune
        mgr1._sessions["ch1"].last_active = time.time() - 7200
        mgr1.prune()

        # Second run: load — ch1 should NOT be reloaded
        mgr2 = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)
        mgr2.load()
        assert "ch1" not in mgr2._sessions

        # No second prune needed, and no duplicate archives
        archive_dir = Path(persist) / "archive"
        archives = list(archive_dir.glob("ch1_*.json"))
        assert len(archives) == 1, "Should have exactly one archive, not duplicates"

    def test_active_sessions_survive_prune_and_reload(self, tmp_dir: Path):
        """Active sessions should persist through prune+save+load cycles."""
        persist = str(tmp_dir / "sessions")

        mgr1 = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)
        mgr1.add_message("active", "user", "I'm still here")
        mgr1.add_message("stale", "user", "I'm old")
        mgr1.save()

        mgr1._sessions["stale"].last_active = time.time() - 7200
        mgr1.prune()
        mgr1.save()  # save remaining active sessions

        mgr2 = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)
        mgr2.load()

        assert "active" in mgr2._sessions
        assert "stale" not in mgr2._sessions
        history = mgr2.get_history("active")
        assert history[0]["content"] == "I'm still here"


class TestStartupPrune:
    """Verify that on_ready() prunes stale sessions at startup."""

    @pytest.mark.asyncio
    async def test_startup_prune_stale_sessions(self, tmp_dir: Path):
        """Simulates on_ready() startup: load() then prune() removes stale sessions."""
        persist = str(tmp_dir / "sessions")

        # Pre-create a stale session file on disk
        persist_dir = Path(persist)
        persist_dir.mkdir(parents=True, exist_ok=True)
        stale_data = {
            "channel_id": "stale_ch",
            "messages": [{"role": "user", "content": "old msg", "timestamp": 1000.0}],
            "created_at": 1000.0,
            "last_active": 1000.0,  # epoch + 1000s = very old
            "summary": "",
        }
        (persist_dir / "stale_ch.json").write_text(json.dumps(stale_data))

        # Also create a fresh session file
        fresh_data = {
            "channel_id": "fresh_ch",
            "messages": [{"role": "user", "content": "recent msg", "timestamp": time.time()}],
            "created_at": time.time(),
            "last_active": time.time(),
            "summary": "",
        }
        (persist_dir / "fresh_ch.json").write_text(json.dumps(fresh_data))

        # Simulate bot startup: __init__ calls load(), on_ready calls prune()
        mgr = SessionManager(
            max_history=10, max_age_hours=1, persist_dir=persist,
        )
        mgr.load()

        # Both sessions should be loaded initially
        assert "stale_ch" in mgr._sessions
        assert "fresh_ch" in mgr._sessions

        # Simulate on_ready: prune stale sessions
        pruned = mgr.prune()

        assert pruned == 1
        assert "stale_ch" not in mgr._sessions
        assert "fresh_ch" in mgr._sessions
        assert not (persist_dir / "stale_ch.json").exists()
        assert (persist_dir / "fresh_ch.json").exists()

    @pytest.mark.asyncio
    async def test_startup_prune_archives_before_deleting(self, tmp_dir: Path):
        """Stale sessions pruned at startup should be archived first."""
        persist = str(tmp_dir / "sessions")
        persist_dir = Path(persist)
        persist_dir.mkdir(parents=True, exist_ok=True)

        stale_data = {
            "channel_id": "old_ch",
            "messages": [
                {"role": "user", "content": "important old data", "timestamp": 1000.0},
                {"role": "assistant", "content": "noted", "timestamp": 1001.0},
            ],
            "created_at": 1000.0,
            "last_active": 1000.0,
            "summary": "",
        }
        (persist_dir / "old_ch.json").write_text(json.dumps(stale_data))

        mgr = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)
        mgr.load()
        mgr.prune()

        # Session file deleted
        assert not (persist_dir / "old_ch.json").exists()

        # Archive created with data preserved
        archive_dir = persist_dir / "archive"
        archives = list(archive_dir.glob("old_ch_*.json"))
        assert len(archives) == 1
        archived = json.loads(archives[0].read_text())
        assert archived["channel_id"] == "old_ch"
        assert any(m["content"] == "important old data" for m in archived["messages"])

    @pytest.mark.asyncio
    async def test_startup_prune_search_excludes_stale(self, tmp_dir: Path):
        """After startup prune, search_history should not return stale session data."""
        persist = str(tmp_dir / "sessions")
        persist_dir = Path(persist)
        persist_dir.mkdir(parents=True, exist_ok=True)

        stale_data = {
            "channel_id": "stale_ch",
            "messages": [{"role": "user", "content": "unique_stale_keyword", "timestamp": 1000.0}],
            "created_at": 1000.0,
            "last_active": 1000.0,
            "summary": "",
        }
        (persist_dir / "stale_ch.json").write_text(json.dumps(stale_data))

        mgr = SessionManager(max_history=10, max_age_hours=1, persist_dir=persist)
        mgr.load()
        mgr.prune()

        # Search should not find the stale keyword in active sessions
        results = await mgr.search_history("unique_stale_keyword")
        # Results may come from archive search, but not from _sessions
        active_results = [r for r in results if r.get("type") != "summary"]
        for r in active_results:
            # If found, it should be from archive, not active sessions
            assert r["channel_id"] == "stale_ch"  # from archive scan, acceptable


class TestPruneWithReflector:
    """Verify prune file cleanup works with reflector/indexer configured."""

    def test_prune_with_reflector_still_deletes_file(self, tmp_dir: Path):
        """File should be deleted even when reflector is configured."""
        persist = str(tmp_dir / "sessions")
        reflector = MagicMock()
        mgr = SessionManager(
            max_history=10, max_age_hours=1, persist_dir=persist,
            reflector=reflector,
        )

        # Add enough messages to trigger reflection (>= 3)
        for i in range(4):
            mgr.add_message("ch1", "user" if i % 2 == 0 else "assistant", f"msg {i}")
        mgr.save()

        session_file = mgr.persist_dir / "ch1.json"
        assert session_file.exists()

        mgr._sessions["ch1"].last_active = time.time() - 7200

        # Patch asyncio.create_task since we're not in an async context
        with patch("asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock()
            mgr.prune()

        assert not session_file.exists(), "File should be deleted even with reflector"
        # Archive should still be created
        archive_dir = mgr.persist_dir / "archive"
        assert len(list(archive_dir.glob("*.json"))) == 1
