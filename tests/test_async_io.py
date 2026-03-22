"""Tests verifying sync I/O is wrapped in asyncio.to_thread to avoid blocking the event loop.

Audit finding #11: Multiple places do synchronous file reads/writes in async handlers.
These tests verify that the hot-path async methods use asyncio.to_thread for file I/O.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.tools.executor import ToolExecutor  # noqa: E402
from src.scheduler.scheduler import Scheduler  # noqa: E402
from src.sessions.manager import SessionManager  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_executor(tmp_path: Path) -> ToolExecutor:
    mem_file = tmp_path / "memory.json"
    mem_file.write_text("{}")
    return ToolExecutor(MagicMock(), memory_path=str(mem_file))


# ===========================================================================
# executor.py: _handle_memory_manage uses asyncio.to_thread
# ===========================================================================

class TestExecutorMemoryAsyncIO:
    """Memory tool handler offloads file I/O to a thread."""

    @pytest.mark.asyncio
    async def test_memory_list_uses_thread(self, tmp_path):
        executor = make_executor(tmp_path)
        with patch("src.tools.executor.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = {"global": {"key1": "val1"}}
            result = await executor._handle_memory_manage(
                {"action": "list"}, user_id="u1",
            )
            # Should have called to_thread with _load_all_memory
            mock_thread.assert_called_once_with(executor._load_all_memory)

    @pytest.mark.asyncio
    async def test_memory_save_uses_thread(self, tmp_path):
        executor = make_executor(tmp_path)
        with patch("src.tools.executor.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            # First call: load, second call: save
            mock_thread.side_effect = [{"global": {}}, None]
            await executor._handle_memory_manage(
                {"action": "save", "key": "k", "value": "v"}, user_id="u1",
            )
            assert mock_thread.call_count == 2
            # Second call should be _save_all_memory
            save_call = mock_thread.call_args_list[1]
            assert save_call[0][0] == executor._save_all_memory

    @pytest.mark.asyncio
    async def test_memory_delete_uses_thread(self, tmp_path):
        executor = make_executor(tmp_path)
        with patch("src.tools.executor.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = [{"global": {"k": "v"}}, None]
            await executor._handle_memory_manage(
                {"action": "delete", "key": "k"}, user_id="u1",
            )
            assert mock_thread.call_count == 2

    @pytest.mark.asyncio
    async def test_memory_operations_still_work_end_to_end(self, tmp_path):
        """End-to-end: memory save/load works correctly through asyncio.to_thread."""
        executor = make_executor(tmp_path)
        result = await executor._handle_memory_manage(
            {"action": "save", "key": "server", "value": "10.0.0.99"},
            user_id="u1",
        )
        assert "Saved" in result

        result = await executor._handle_memory_manage(
            {"action": "list"}, user_id="u1",
        )
        assert "10.0.0.99" in result

        result = await executor._handle_memory_manage(
            {"action": "delete", "key": "server"}, user_id="u1",
        )
        assert "Deleted" in result


# ===========================================================================
# executor.py: _handle_manage_list uses asyncio.to_thread
# ===========================================================================

class TestExecutorListAsyncIO:
    """List tool handler offloads file I/O to a thread."""

    @pytest.mark.asyncio
    async def test_list_add_uses_thread(self, tmp_path):
        executor = make_executor(tmp_path)
        with patch("src.tools.executor.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = [{}, None]  # load returns empty, save returns None
            await executor._handle_manage_list(
                {"action": "add", "list_name": "grocery", "items": ["milk"]},
                user_id="u1",
            )
            assert mock_thread.call_count == 2
            # First call: load, second call: save
            assert mock_thread.call_args_list[0][0][0] == executor._load_lists
            assert mock_thread.call_args_list[1][0][0] == executor._save_lists

    @pytest.mark.asyncio
    async def test_list_operations_end_to_end(self, tmp_path):
        """End-to-end: list add/show works through asyncio.to_thread."""
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list(
            {"action": "add", "list_name": "todo", "items": ["write tests"]},
            user_id="u1",
        )
        assert "Added" in result

        result = await executor._handle_manage_list(
            {"action": "show", "list_name": "todo"}, user_id="u1",
        )
        assert "write tests" in result


# ===========================================================================
# scheduler.py: _tick and fire_triggers use asyncio.to_thread for _save
# ===========================================================================

class TestSchedulerAsyncIO:
    """Scheduler tick and fire_triggers offload _save to a thread."""

    @pytest.mark.asyncio
    async def test_tick_save_uses_thread(self, tmp_path):
        from datetime import datetime, timedelta

        scheduler = Scheduler(data_path=str(tmp_path / "schedules.json"))
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Due",
            action="reminder",
            channel_id="ch1",
            run_at=(datetime.now() - timedelta(minutes=1)).isoformat(),
            message="Fire now",
        )

        with patch("src.scheduler.scheduler.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await scheduler._tick()
            mock_thread.assert_called_once_with(scheduler._save)

    @pytest.mark.asyncio
    async def test_fire_triggers_save_uses_thread(self, tmp_path):
        scheduler = Scheduler(data_path=str(tmp_path / "schedules.json"))
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Trigger test",
            action="reminder",
            channel_id="ch1",
            message="hi",
            trigger={"source": "gitea"},
        )

        with patch("src.scheduler.scheduler.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await scheduler.fire_triggers("gitea", {"event": "push"})
            mock_thread.assert_called_once_with(scheduler._save)

    @pytest.mark.asyncio
    async def test_tick_still_saves_end_to_end(self, tmp_path):
        """End-to-end: tick fires and persists via asyncio.to_thread."""
        from datetime import datetime, timedelta

        scheduler = Scheduler(data_path=str(tmp_path / "schedules.json"))
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Past",
            action="reminder",
            channel_id="ch1",
            run_at=(datetime.now() - timedelta(minutes=1)).isoformat(),
            message="Fire",
        )

        await scheduler._tick()
        callback.assert_called_once()
        # One-time schedule should be removed
        assert len(scheduler.list_all()) == 0


# ===========================================================================
# sessions/manager.py: search_history uses asyncio.to_thread for archives
# ===========================================================================

class TestSessionSearchAsyncIO:
    """search_history offloads archive file reads to a thread."""

    @pytest.mark.asyncio
    async def test_search_history_uses_thread_for_archives(self, tmp_path):
        """Archive search is delegated to _search_archives via asyncio.to_thread."""
        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_path / "sessions"),
        )
        with patch.object(mgr, "_search_archives", return_value=[]) as mock_search:
            with patch("src.sessions.manager.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = []
                await mgr.search_history("test")
                mock_thread.assert_any_call(mock_search, "test", 10)

    @pytest.mark.asyncio
    async def test_search_archives_reads_files(self, tmp_path):
        """_search_archives correctly reads archive JSON files."""
        persist_dir = tmp_path / "sessions"
        persist_dir.mkdir(parents=True)
        archive_dir = persist_dir / "archive"
        archive_dir.mkdir()

        # Write a test archive
        archive_data = {
            "channel_id": "ch1",
            "last_active": 1000,
            "summary": "Deployed nginx",
            "messages": [
                {"role": "user", "content": "deploy nginx to server", "timestamp": 999},
            ],
        }
        (archive_dir / "ch1_1000.json").write_text(json.dumps(archive_data))

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(persist_dir),
        )

        results = mgr._search_archives("nginx", limit=10)
        assert len(results) >= 1
        assert any("nginx" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_history_end_to_end_with_archives(self, tmp_path):
        """End-to-end: search_history finds results in archives via thread."""
        persist_dir = tmp_path / "sessions"
        persist_dir.mkdir(parents=True)
        archive_dir = persist_dir / "archive"
        archive_dir.mkdir()

        archive_data = {
            "channel_id": "ch1",
            "last_active": 1000,
            "summary": "",
            "messages": [
                {"role": "user", "content": "check disk space on server", "timestamp": 999},
            ],
        }
        (archive_dir / "ch1_1000.json").write_text(json.dumps(archive_data))

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(persist_dir),
        )

        results = await mgr.search_history("disk space")
        assert len(results) >= 1
        assert any("disk space" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_archives_respects_limit(self, tmp_path):
        """_search_archives stops after reaching limit."""
        persist_dir = tmp_path / "sessions"
        persist_dir.mkdir(parents=True)
        archive_dir = persist_dir / "archive"
        archive_dir.mkdir()

        archive_data = {
            "channel_id": "ch1",
            "last_active": 1000,
            "summary": "",
            "messages": [
                {"role": "user", "content": "test msg 1", "timestamp": 1},
                {"role": "user", "content": "test msg 2", "timestamp": 2},
                {"role": "user", "content": "test msg 3", "timestamp": 3},
            ],
        }
        (archive_dir / "ch1_1000.json").write_text(json.dumps(archive_data))

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(persist_dir),
        )

        results = mgr._search_archives("test", limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_archives_no_archive_dir(self, tmp_path):
        """_search_archives returns empty when archive dir doesn't exist."""
        persist_dir = tmp_path / "sessions"
        persist_dir.mkdir(parents=True)

        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(persist_dir),
        )

        results = mgr._search_archives("test", limit=10)
        assert results == []
