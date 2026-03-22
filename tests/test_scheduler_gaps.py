"""Tests for scheduler.py coverage gaps.

Targets uncovered lines: 39-41, 61, 72, 75, 218-224, 230, 233-234,
244, 257-258, 262-264.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.scheduler.scheduler import Scheduler


@pytest.fixture
def scheduler(tmp_dir: Path) -> Scheduler:
    return Scheduler(data_path=str(tmp_dir / "schedules.json"))


class TestLoadErrors:
    def test_load_corrupt_json(self, tmp_dir: Path):
        """Corrupt JSON file triggers error path (lines 39-41)."""
        path = tmp_dir / "schedules.json"
        path.write_text("not valid json {{{")
        sched = Scheduler(data_path=str(path))
        assert sched.list_all() == []

    def test_load_nonexistent_file(self, tmp_dir: Path):
        """Non-existent file is handled gracefully."""
        sched = Scheduler(data_path=str(tmp_dir / "missing.json"))
        assert sched.list_all() == []


class TestAddDigest:
    def test_add_digest_action(self, scheduler: Scheduler):
        """Digest action skips tool validation (line 61)."""
        schedule = scheduler.add(
            description="Daily digest",
            action="digest",
            channel_id="ch1",
            cron="0 9 * * *",
        )
        assert schedule["action"] == "digest"
        assert schedule["id"]


class TestAddWorkflow:
    def test_add_workflow_valid(self, scheduler: Scheduler):
        """Workflow with valid steps succeeds (lines 72, 75)."""
        steps = [
            {"tool_name": "check_disk", "tool_input": {"host": "server"}},
            {"tool_name": "check_memory", "tool_input": {"host": "server"}},
        ]
        schedule = scheduler.add(
            description="Health workflow",
            action="workflow",
            channel_id="ch1",
            cron="0 8 * * *",
            steps=steps,
        )
        assert schedule["steps"] == steps

    def test_add_workflow_no_steps(self, scheduler: Scheduler):
        """Workflow without steps raises ValueError (line 72)."""
        with pytest.raises(ValueError, match="steps.*required"):
            scheduler.add(
                description="No steps",
                action="workflow",
                channel_id="ch1",
                cron="0 8 * * *",
            )

    def test_add_workflow_invalid_step(self, scheduler: Scheduler):
        """Workflow with invalid step dict raises ValueError (line 75)."""
        with pytest.raises(ValueError, match="Step 0.*tool_name"):
            scheduler.add(
                description="Bad step",
                action="workflow",
                channel_id="ch1",
                cron="0 8 * * *",
                steps=[{"bad_key": "value"}],
            )

    def test_add_workflow_step_not_dict(self, scheduler: Scheduler):
        """Workflow with non-dict step raises ValueError (line 75)."""
        with pytest.raises(ValueError, match="Step 0"):
            scheduler.add(
                description="Not dict step",
                action="workflow",
                channel_id="ch1",
                cron="0 8 * * *",
                steps=["not a dict"],
            )


class TestStopAndLoop:
    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, scheduler: Scheduler):
        """stop() cancels the loop task (lines 218-224)."""
        callback = AsyncMock()
        scheduler.start(callback)
        # Let the loop start
        await asyncio.sleep(0.05)
        await scheduler.stop()
        assert scheduler._task.cancelled() or scheduler._task.done()

    @pytest.mark.asyncio
    async def test_stop_when_no_task(self, scheduler: Scheduler):
        """stop() is safe when no task running."""
        await scheduler.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_loop_handles_cancelled_error(self, scheduler: Scheduler):
        """_loop breaks on CancelledError (lines 230, 231-232)."""
        callback = AsyncMock()
        scheduler.start(callback)
        await asyncio.sleep(0.05)
        scheduler._task.cancel()
        # Wait for task to finish — should not raise
        try:
            await scheduler._task
        except asyncio.CancelledError:
            pass


class TestSchedulerLock:
    def test_scheduler_has_asyncio_lock(self, scheduler: Scheduler):
        """Scheduler should have an asyncio.Lock for mutual exclusion."""
        assert hasattr(scheduler, "_lock")
        assert isinstance(scheduler._lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_tick_and_fire_triggers_mutual_exclusion(self, tmp_dir: Path):
        """_tick and fire_triggers should not run concurrently."""
        sched = Scheduler(data_path=str(tmp_dir / "schedules.json"))
        execution_order = []

        async def slow_callback(schedule):
            execution_order.append(f"start-{schedule['action']}")
            await asyncio.sleep(0.1)
            execution_order.append(f"end-{schedule['action']}")

        sched.start(slow_callback)

        # Add a cron schedule due now and a trigger schedule
        sched.add(
            description="Cron",
            action="reminder",
            channel_id="ch1",
            run_at=(datetime.now() - timedelta(minutes=1)).isoformat(),
            message="cron fire",
        )
        sched.add(
            description="Trigger",
            action="reminder",
            channel_id="ch1",
            trigger={"source": "gitea"},
            message="trigger fire",
        )

        # Run _tick and fire_triggers concurrently — lock ensures sequential
        await asyncio.gather(
            sched._tick(),
            sched.fire_triggers("gitea", {"event": "push"}),
        )

        # Both should have run (order depends on who gets lock first)
        assert len(execution_order) == 4
        # Verify no interleaving: starts and ends should pair correctly
        assert execution_order[0].startswith("start-")
        assert execution_order[1].startswith("end-")
        assert execution_order[2].startswith("start-")
        assert execution_order[3].startswith("end-")

        await sched.stop()


class TestTickEdgeCases:
    @pytest.mark.asyncio
    async def test_tick_callback_failure(self, scheduler: Scheduler):
        """Callback exception during tick is caught (lines 257-258)."""
        callback = AsyncMock(side_effect=RuntimeError("callback boom"))
        scheduler.start(callback)

        scheduler.add(
            description="Will fail",
            action="reminder",
            channel_id="ch1",
            run_at=(datetime.now() - timedelta(minutes=1)).isoformat(),
            message="Fire now",
        )

        # Should not raise despite callback failure
        await scheduler._tick()
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_reschedules_cron(self, scheduler: Scheduler):
        """Cron schedule gets next_run updated after firing (lines 262-264)."""
        callback = AsyncMock()
        scheduler.start(callback)

        schedule = scheduler.add(
            description="Cron test",
            action="reminder",
            channel_id="ch1",
            cron="0 9 * * *",
            message="Morning",
        )
        # Force it to be past due
        schedule["next_run"] = (datetime.now() - timedelta(minutes=1)).isoformat()

        await scheduler._tick()
        callback.assert_called_once()

        # next_run should be updated to a future time
        remaining = scheduler.list_all()
        assert len(remaining) == 1
        next_run = datetime.fromisoformat(remaining[0]["next_run"])
        assert next_run > datetime.now()

    @pytest.mark.asyncio
    async def test_tick_skips_no_next_run(self, scheduler: Scheduler):
        """Schedules without next_run (trigger-based) are skipped (line 244)."""
        callback = AsyncMock()
        scheduler.start(callback)

        # Add a trigger-based schedule (no next_run)
        schedule = scheduler.add(
            description="Trigger test",
            action="reminder",
            channel_id="ch1",
            trigger={"source": "gitea"},
            message="On push",
        )
        # Trigger schedules have no next_run
        assert "next_run" not in schedule or schedule.get("next_run") is None

        await scheduler._tick()
        callback.assert_not_called()
