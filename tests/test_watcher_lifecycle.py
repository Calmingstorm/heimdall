"""Tests for monitoring/watcher.py — stagger, lifecycle, and loop behavior."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from src.config.schema import MonitoringConfig, MonitorCheck, ToolsConfig, ToolHost
from src.monitoring.watcher import InfraWatcher, AlertState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def executor(tools_config: ToolsConfig) -> MagicMock:
    from src.tools.executor import ToolExecutor
    return ToolExecutor(tools_config)


@pytest.fixture
def alert_callback() -> AsyncMock:
    return AsyncMock()


def _make_checks(n: int) -> list[MonitorCheck]:
    """Create *n* disk checks with unique names for testing."""
    return [
        MonitorCheck(
            name=f"check_{i}",
            type="disk",
            hosts=["server"],
            threshold=90,
            interval_minutes=30,
        )
        for i in range(n)
    ]


def _make_config(checks: list[MonitorCheck], *, enabled: bool = True) -> MonitoringConfig:
    return MonitoringConfig(
        enabled=enabled,
        alert_channel_id="12345",
        cooldown_minutes=60,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# STAGGER_INTERVAL constant
# ---------------------------------------------------------------------------

class TestStaggerInterval:
    def test_stagger_interval_is_positive(self):
        assert InfraWatcher.STAGGER_INTERVAL > 0

    def test_stagger_interval_reasonable(self):
        """Stagger should be between 1 and 30 seconds — enough to spread
        SSH connections without delaying first check excessively."""
        assert 1 <= InfraWatcher.STAGGER_INTERVAL <= 30


# ---------------------------------------------------------------------------
# start() stagger calculation
# ---------------------------------------------------------------------------

class TestStartStagger:
    @pytest.mark.asyncio
    async def test_start_passes_different_stagger_per_check(
        self, executor, alert_callback,
    ):
        """Each check must receive a different stagger delay so they don't
        all fire at the same time."""
        checks = _make_checks(4)
        config = _make_config(checks)
        watcher = InfraWatcher(config, executor, alert_callback)

        stagger_values: list[int] = []

        original_check_loop = watcher._check_loop

        async def capture_stagger(check, stagger_seconds=5):
            stagger_values.append(stagger_seconds)
            # Don't actually run the loop
            await asyncio.sleep(0)

        with patch.object(watcher, "_check_loop", side_effect=capture_stagger):
            watcher.start()
            # Let the tasks start
            await asyncio.sleep(0.05)

        assert len(stagger_values) == 4
        # Each stagger should be unique
        assert len(set(stagger_values)) == 4
        # First check: STAGGER_INTERVAL * 1, second: * 2, etc.
        interval = InfraWatcher.STAGGER_INTERVAL
        assert stagger_values == [interval * 1, interval * 2, interval * 3, interval * 4]

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_start_single_check_stagger(self, executor, alert_callback):
        """A single check should get stagger = STAGGER_INTERVAL * 1."""
        checks = _make_checks(1)
        config = _make_config(checks)
        watcher = InfraWatcher(config, executor, alert_callback)

        stagger_values = []

        async def capture_stagger(check, stagger_seconds=5):
            stagger_values.append(stagger_seconds)
            await asyncio.sleep(0)

        with patch.object(watcher, "_check_loop", side_effect=capture_stagger):
            watcher.start()
            await asyncio.sleep(0.05)

        assert stagger_values == [InfraWatcher.STAGGER_INTERVAL]
        await watcher.stop()

    @pytest.mark.asyncio
    async def test_stagger_spreads_over_time(self, executor, alert_callback):
        """With N checks, the last check starts after STAGGER_INTERVAL * N
        seconds — verify the maximum delay is predictable."""
        checks = _make_checks(5)
        config = _make_config(checks)
        watcher = InfraWatcher(config, executor, alert_callback)

        stagger_values = []

        async def capture_stagger(check, stagger_seconds=5):
            stagger_values.append(stagger_seconds)
            await asyncio.sleep(0)

        with patch.object(watcher, "_check_loop", side_effect=capture_stagger):
            watcher.start()
            await asyncio.sleep(0.05)

        max_stagger = max(stagger_values)
        assert max_stagger == InfraWatcher.STAGGER_INTERVAL * 5
        await watcher.stop()


# ---------------------------------------------------------------------------
# start() / stop() lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_tasks(self, executor, alert_callback):
        """start() should create one asyncio task per check."""
        checks = _make_checks(3)
        config = _make_config(checks)
        watcher = InfraWatcher(config, executor, alert_callback)

        # Prevent actual loop execution
        with patch.object(watcher, "_check_loop", new_callable=AsyncMock):
            watcher.start()
            assert len(watcher._check_tasks) == 3
            assert set(watcher._check_tasks.keys()) == {"check_0", "check_1", "check_2"}
            await watcher.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_all_tasks(self, executor, alert_callback):
        """stop() should cancel every running task and clear the dict."""
        checks = _make_checks(2)
        config = _make_config(checks)
        watcher = InfraWatcher(config, executor, alert_callback)

        with patch.object(watcher, "_check_loop", new_callable=AsyncMock):
            watcher.start()
            assert len(watcher._check_tasks) == 2
            await watcher.stop()
            assert len(watcher._check_tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, executor, alert_callback):
        """Calling stop() when nothing is running should not error."""
        config = _make_config([])
        watcher = InfraWatcher(config, executor, alert_callback)
        await watcher.stop()  # no-op, should not raise
        assert len(watcher._check_tasks) == 0

    def test_start_disabled(self, executor, alert_callback):
        """start() with enabled=False should create no tasks."""
        config = _make_config(_make_checks(3), enabled=False)
        watcher = InfraWatcher(config, executor, alert_callback)
        watcher.start()
        assert len(watcher._check_tasks) == 0

    def test_start_no_checks(self, executor, alert_callback):
        """start() with an empty checks list should create no tasks."""
        config = _make_config([])
        watcher = InfraWatcher(config, executor, alert_callback)
        watcher.start()
        assert len(watcher._check_tasks) == 0


# ---------------------------------------------------------------------------
# _check_loop behavior
# ---------------------------------------------------------------------------

class TestCheckLoop:
    @pytest.mark.asyncio
    async def test_loop_uses_stagger_delay(self, executor, alert_callback):
        """_check_loop should sleep for the stagger_seconds on startup."""
        config = _make_config(_make_checks(1))
        watcher = InfraWatcher(config, executor, alert_callback)
        check = config.checks[0]

        sleep_calls: list[float] = []

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)
            # Only actually sleep briefly for the stagger;
            # cancel before the interval sleep
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        with patch("src.monitoring.watcher.asyncio.sleep", side_effect=mock_sleep), \
             patch.object(watcher, "_run_check", new_callable=AsyncMock):
            try:
                await watcher._check_loop(check, stagger_seconds=15)
            except asyncio.CancelledError:
                pass

        # First sleep call should be the stagger
        assert sleep_calls[0] == 15

    @pytest.mark.asyncio
    async def test_loop_default_stagger(self, executor, alert_callback):
        """_check_loop without explicit stagger should default to 5."""
        config = _make_config(_make_checks(1))
        watcher = InfraWatcher(config, executor, alert_callback)
        check = config.checks[0]

        sleep_calls: list[float] = []

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        with patch("src.monitoring.watcher.asyncio.sleep", side_effect=mock_sleep), \
             patch.object(watcher, "_run_check", new_callable=AsyncMock):
            try:
                await watcher._check_loop(check)
            except asyncio.CancelledError:
                pass

        assert sleep_calls[0] == 5

    @pytest.mark.asyncio
    async def test_loop_interval_sleep_after_check(self, executor, alert_callback):
        """After the stagger, the loop should sleep for interval_minutes * 60."""
        check = MonitorCheck(
            name="test",
            type="disk",
            hosts=["server"],
            threshold=90,
            interval_minutes=10,
        )
        config = _make_config([check])
        watcher = InfraWatcher(config, executor, alert_callback)

        sleep_calls: list[float] = []

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        with patch("src.monitoring.watcher.asyncio.sleep", side_effect=mock_sleep), \
             patch.object(watcher, "_run_check", new_callable=AsyncMock):
            try:
                await watcher._check_loop(check, stagger_seconds=5)
            except asyncio.CancelledError:
                pass

        # Second sleep should be the interval (10 minutes = 600 seconds)
        assert len(sleep_calls) >= 2
        assert sleep_calls[1] == 600

    @pytest.mark.asyncio
    async def test_loop_continues_after_check_error(self, executor, alert_callback):
        """If _run_check raises, the loop should log the error and continue."""
        config = _make_config(_make_checks(1))
        watcher = InfraWatcher(config, executor, alert_callback)
        check = config.checks[0]

        call_count = 0
        sleep_calls: list[float] = []

        async def failing_check(c):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("SSH connection refused")
            # Third call succeeds

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)
            # Cancel after the second check to avoid infinite loop
            if call_count >= 2:
                raise asyncio.CancelledError

        with patch("src.monitoring.watcher.asyncio.sleep", side_effect=mock_sleep), \
             patch.object(watcher, "_run_check", side_effect=failing_check):
            try:
                await watcher._check_loop(check, stagger_seconds=1)
            except asyncio.CancelledError:
                pass

        # _run_check should have been called at least twice (loop continued)
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_loop_stops_on_cancel(self, executor, alert_callback):
        """CancelledError during _run_check should break the loop cleanly."""
        config = _make_config(_make_checks(1))
        watcher = InfraWatcher(config, executor, alert_callback)
        check = config.checks[0]

        async def cancelling_check(c):
            raise asyncio.CancelledError

        with patch("src.monitoring.watcher.asyncio.sleep", new_callable=AsyncMock), \
             patch.object(watcher, "_run_check", side_effect=cancelling_check):
            # _check_loop should exit cleanly (break) on CancelledError
            await watcher._check_loop(check, stagger_seconds=0)

    @pytest.mark.asyncio
    async def test_loop_runs_check_on_first_iteration(self, executor, alert_callback):
        """After the stagger sleep, _run_check should be called immediately."""
        config = _make_config(_make_checks(1))
        watcher = InfraWatcher(config, executor, alert_callback)
        check = config.checks[0]

        check_called = False

        async def track_check(c):
            nonlocal check_called
            check_called = True

        async def mock_sleep(seconds):
            if check_called:
                raise asyncio.CancelledError

        with patch("src.monitoring.watcher.asyncio.sleep", side_effect=mock_sleep), \
             patch.object(watcher, "_run_check", side_effect=track_check):
            try:
                await watcher._check_loop(check, stagger_seconds=0)
            except asyncio.CancelledError:
                pass

        assert check_called


# ---------------------------------------------------------------------------
# _run_check dispatch
# ---------------------------------------------------------------------------

class TestRunCheckDispatch:
    @pytest.mark.asyncio
    async def test_unknown_type_logged(self, executor, alert_callback):
        """Unknown check type should not raise, just log a warning."""
        check = MonitorCheck(
            name="mystery",
            type="quantum",
            hosts=["server"],
            threshold=50,
            interval_minutes=5,
        )
        config = _make_config([check])
        watcher = InfraWatcher(config, executor, alert_callback)

        # Should not raise
        await watcher._run_check(check)
        alert_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatches_to_correct_handler(self, executor, alert_callback):
        """Each check type should call the corresponding handler."""
        config = _make_config(_make_checks(1))
        watcher = InfraWatcher(config, executor, alert_callback)

        for check_type, handler_name in [
            ("disk", "_check_disk"),
            ("memory", "_check_memory"),
            ("service", "_check_service"),
            ("promql", "_check_promql"),
        ]:
            kwargs = dict(
                name=f"test_{check_type}",
                type=check_type,
                hosts=["server"],
                threshold=90,
                interval_minutes=5,
            )
            if check_type == "promql":
                kwargs["query"] = "up"
            if check_type == "service":
                kwargs["services"] = ["test"]
            check = MonitorCheck(**kwargs)
            with patch.object(watcher, handler_name, new_callable=AsyncMock) as mock_handler:
                await watcher._run_check(check)
                mock_handler.assert_called_once_with(check)


# ---------------------------------------------------------------------------
# get_status reflects running state
# ---------------------------------------------------------------------------

class TestGetStatusLifecycle:
    @pytest.mark.asyncio
    async def test_status_shows_running_count(self, executor, alert_callback):
        """get_status should reflect how many checks are currently running."""
        checks = _make_checks(3)
        config = _make_config(checks)
        watcher = InfraWatcher(config, executor, alert_callback)

        assert watcher.get_status()["running"] == 0

        with patch.object(watcher, "_check_loop", new_callable=AsyncMock):
            watcher.start()
            assert watcher.get_status()["running"] == 3
            await watcher.stop()

        assert watcher.get_status()["running"] == 0
