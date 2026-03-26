"""Tests for session 4 audit fixes:
- #8: Incus instance name validation
- #12: find_patterns Jaccard fallback when cosine below threshold
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
from src.tools.executor import ToolExecutor


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=10,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


@pytest.fixture
def executor(tools_config, tmp_dir):
    return ToolExecutor(tools_config, memory_path=str(tmp_dir / "memory.json"))


# ── #8: Incus instance name validation ───────────────────────────────


class TestIncusNameValidation:
    """incus handlers reject names with shell metacharacters."""

    @pytest.mark.asyncio
    async def test_valid_instance_name(self, executor):
        """Normal alphanumeric-hyphen names pass validation."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "root")
            result = await executor.execute("incus_exec", {
                "instance": "my-vm-01",
                "command": "whoami",
            })
        assert "root" in result

    @pytest.mark.asyncio
    async def test_shell_metachar_instance_rejected(self, executor):
        """Instance name with shell metacharacters is rejected."""
        result = await executor.execute("incus_exec", {
            "instance": "myvm; rm -rf /",
            "command": "whoami",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_semicolon_in_instance_rejected(self, executor):
        """Semicolon in instance name is rejected."""
        result = await executor.execute("incus_info", {
            "instance": "test;whoami",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_backtick_in_instance_rejected(self, executor):
        """Backticks in instance name are rejected."""
        result = await executor.execute("incus_start", {
            "instance": "`id`",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_pipe_in_instance_rejected(self, executor):
        """Pipe in instance name is rejected."""
        result = await executor.execute("incus_stop", {
            "instance": "test|cat",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_empty_instance_rejected(self, executor):
        """Empty instance name is rejected."""
        result = await executor.execute("incus_exec", {
            "instance": "",
            "command": "ls",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_too_long_instance_rejected(self, executor):
        """Instance name > 63 chars is rejected."""
        result = await executor.execute("incus_exec", {
            "instance": "a" * 64,
            "command": "ls",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_63_char_instance_accepted(self, executor):
        """Instance name of exactly 63 chars is accepted."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            result = await executor.execute("incus_exec", {
                "instance": "a" * 63,
                "command": "ls",
            })
        assert "Invalid" not in result

    @pytest.mark.asyncio
    async def test_hyphen_start_rejected(self, executor):
        """Instance name starting with hyphen is rejected."""
        result = await executor.execute("incus_exec", {
            "instance": "-badname",
            "command": "ls",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_incus_delete_validates(self, executor):
        """incus_delete also validates instance name."""
        result = await executor.execute("incus_delete", {
            "instance": "test$(evil)",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_incus_logs_validates(self, executor):
        """incus_logs also validates instance name."""
        result = await executor.execute("incus_logs", {
            "instance": "test&bg",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_incus_restart_validates(self, executor):
        """incus_restart also validates instance name."""
        result = await executor.execute("incus_restart", {
            "instance": "bad name",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_incus_snapshot_validates_instance(self, executor):
        """incus_snapshot validates instance name."""
        result = await executor.execute("incus_snapshot", {
            "instance": "bad;name",
            "action": "create",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_incus_snapshot_validates_snapshot_name(self, executor):
        """incus_snapshot validates snapshot name too."""
        result = await executor.execute("incus_snapshot", {
            "instance": "good-name",
            "action": "create",
            "snapshot": "bad;snap",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_incus_launch_validates_name(self, executor):
        """incus_launch validates the 'name' parameter."""
        result = await executor.execute("incus_launch", {
            "image": "ubuntu:22.04",
            "name": "test$(evil)",
        })
        assert "Invalid Incus name" in result

    @pytest.mark.asyncio
    async def test_incus_snapshot_list_validates(self, executor):
        """incus_snapshot_list validates instance name."""
        result = await executor.execute("incus_snapshot_list", {
            "instance": "test>evil",
        })
        assert "Invalid Incus name" in result


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
