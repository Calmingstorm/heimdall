"""Tests for scheduler/scheduler.py."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.scheduler.scheduler import Scheduler, ALLOWED_CHECK_TOOLS


@pytest.fixture
def scheduler(tmp_dir: Path) -> Scheduler:
    return Scheduler(data_path=str(tmp_dir / "schedules.json"))


class TestAdd:
    def test_add_cron(self, scheduler: Scheduler):
        schedule = scheduler.add(
            description="Test",
            action="reminder",
            channel_id="ch1",
            cron="0 9 * * *",
            message="Good morning",
        )
        assert schedule["id"]
        assert schedule["cron"] == "0 9 * * *"
        assert schedule["one_time"] is False
        assert "next_run" in schedule

    def test_add_one_time(self, scheduler: Scheduler):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        schedule = scheduler.add(
            description="Reminder",
            action="reminder",
            channel_id="ch1",
            run_at=run_at,
            message="Don't forget",
        )
        assert schedule["one_time"] is True

    def test_add_check_with_valid_tool(self, scheduler: Scheduler):
        schedule = scheduler.add(
            description="Disk check",
            action="check",
            channel_id="ch1",
            cron="0 */6 * * *",
            tool_name="check_disk",
            tool_input={"host": "server"},
        )
        assert schedule["tool_name"] == "check_disk"

    def test_add_check_with_invalid_tool(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="not allowed"):
            scheduler.add(
                description="Bad check",
                action="check",
                channel_id="ch1",
                cron="0 * * * *",
                tool_name="run_command",
            )

    def test_add_check_without_tool_name(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="tool_name is required"):
            scheduler.add(
                description="Missing tool",
                action="check",
                channel_id="ch1",
                cron="0 * * * *",
            )

    def test_invalid_cron(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="Invalid cron"):
            scheduler.add(
                description="Bad cron",
                action="reminder",
                channel_id="ch1",
                cron="not a cron expression",
            )

    def test_no_cron_or_run_at(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="Either"):
            scheduler.add(
                description="No time",
                action="reminder",
                channel_id="ch1",
            )

    def test_add_invalid_run_at(self, scheduler: Scheduler):
        """Invalid ISO datetime for run_at should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ISO datetime"):
            scheduler.add(
                description="Bad time",
                action="reminder",
                channel_id="ch1",
                run_at="tomorrow at 9am",
                message="test",
            )

    def test_add_malformed_run_at(self, scheduler: Scheduler):
        """Malformed date string for run_at should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ISO datetime"):
            scheduler.add(
                description="Bad date",
                action="reminder",
                channel_id="ch1",
                run_at="2026-13-45T99:99",
                message="test",
            )

    def test_add_valid_run_at_accepted(self, scheduler: Scheduler):
        """Valid ISO datetime for run_at should be accepted."""
        run_at = (datetime.now() + timedelta(hours=2)).isoformat()
        schedule = scheduler.add(
            description="Valid time",
            action="reminder",
            channel_id="ch1",
            run_at=run_at,
            message="test",
        )
        assert schedule["run_at"] == run_at


class TestListAndDelete:
    def test_list_all(self, scheduler: Scheduler):
        scheduler.add(
            description="A", action="reminder", channel_id="ch1",
            cron="0 9 * * *", message="hi",
        )
        scheduler.add(
            description="B", action="reminder", channel_id="ch1",
            cron="0 18 * * *", message="bye",
        )
        assert len(scheduler.list_all()) == 2

    def test_delete(self, scheduler: Scheduler):
        schedule = scheduler.add(
            description="Delete me", action="reminder", channel_id="ch1",
            cron="0 9 * * *", message="hi",
        )
        assert scheduler.delete(schedule["id"]) is True
        assert len(scheduler.list_all()) == 0

    def test_delete_nonexistent(self, scheduler: Scheduler):
        assert scheduler.delete("nonexistent") is False


class TestPersistence:
    def test_survives_reload(self, tmp_dir: Path):
        path = str(tmp_dir / "schedules.json")
        s1 = Scheduler(data_path=path)
        s1.add(
            description="Persistent", action="reminder", channel_id="ch1",
            cron="0 9 * * *", message="hi",
        )

        s2 = Scheduler(data_path=path)
        assert len(s2.list_all()) == 1
        assert s2.list_all()[0]["description"] == "Persistent"


class TestTick:
    @pytest.mark.asyncio
    async def test_fires_due_schedule(self, scheduler: Scheduler):
        callback = AsyncMock()
        scheduler.start(callback)

        # Add a schedule that's already past due
        schedule = scheduler.add(
            description="Due", action="reminder", channel_id="ch1",
            run_at=(datetime.now() - timedelta(minutes=1)).isoformat(),
            message="Fire now",
        )

        await scheduler._tick()
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_removes_one_time_after_fire(self, scheduler: Scheduler):
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Once", action="reminder", channel_id="ch1",
            run_at=(datetime.now() - timedelta(minutes=1)).isoformat(),
            message="Fire once",
        )

        await scheduler._tick()
        assert len(scheduler.list_all()) == 0

    @pytest.mark.asyncio
    async def test_does_not_fire_future(self, scheduler: Scheduler):
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Future", action="reminder", channel_id="ch1",
            run_at=(datetime.now() + timedelta(hours=1)).isoformat(),
            message="Not yet",
        )

        await scheduler._tick()
        callback.assert_not_called()
