"""Tests for the ProcessRegistry (process_manager.py) and manage_process handler.

Covers: start, poll, write, kill, list, max concurrent limit, cleanup,
lifetime enforcement, and executor dispatch.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.process_manager import (
    MAX_CONCURRENT,
    MAX_LIFETIME_SECONDS,
    OUTPUT_BUFFER_LINES,
    ProcessInfo,
    ProcessRegistry,
)
from src.tools.executor import ToolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ProcessRegistry:
    return ProcessRegistry()


def _make_fake_process(pid: int = 1234, returncode: int = 0) -> MagicMock:
    """Create a mock asyncio.subprocess.Process."""
    proc = MagicMock()
    proc.pid = pid
    proc.returncode = returncode
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(return_value=b"")
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    return proc


def _inject_process(
    registry: ProcessRegistry,
    pid: int = 1234,
    command: str = "echo hello",
    host: str = "localhost",
    status: str = "running",
    start_time: float | None = None,
    output_lines: list[str] | None = None,
    process: MagicMock | None = None,
) -> ProcessInfo:
    """Manually inject a ProcessInfo into the registry."""
    info = ProcessInfo(
        pid=pid,
        command=command,
        host=host,
        start_time=start_time or time.time(),
        status=status,
        process=process or _make_fake_process(pid),
    )
    if output_lines:
        for line in output_lines:
            info.output_buffer.append(line)
    registry._processes[pid] = info
    return info


# ---------------------------------------------------------------------------
# ProcessRegistry.start
# ---------------------------------------------------------------------------

class TestStartProcess:
    """Test ProcessRegistry.start()."""

    @pytest.mark.asyncio
    async def test_start_process_returns_pid(self):
        """start() returns confirmation string with PID."""
        registry = _make_registry()
        fake_proc = _make_fake_process(pid=42)

        with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = fake_proc
            result = await registry.start("localhost", "sleep 10")

        assert "42" in result
        assert "Process started" in result
        assert 42 in registry._processes

    @pytest.mark.asyncio
    async def test_start_process_failure(self):
        """start() returns error if subprocess creation fails."""
        registry = _make_registry()

        with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock,
                    side_effect=OSError("No such command")):
            result = await registry.start("localhost", "nonexistent_cmd")

        assert "Failed to start" in result
        assert len(registry._processes) == 0


# ---------------------------------------------------------------------------
# ProcessRegistry.poll
# ---------------------------------------------------------------------------

class TestPollProcess:
    """Test ProcessRegistry.poll()."""

    def test_poll_process_with_output(self):
        """poll() returns recent output lines and status."""
        registry = _make_registry()
        _inject_process(registry, pid=100, output_lines=["line1\n", "line2\n", "line3\n"])

        result = registry.poll(100)
        assert "PID 100" in result
        assert "running" in result
        assert "line1" in result
        assert "line3" in result

    def test_poll_process_no_output(self):
        """poll() with no output yet shows '(no output yet)'."""
        registry = _make_registry()
        _inject_process(registry, pid=200)

        result = registry.poll(200)
        assert "no output yet" in result

    def test_poll_unknown_pid(self):
        """poll() for unknown PID returns error."""
        registry = _make_registry()
        result = registry.poll(9999)
        assert "No process with PID 9999" in result

    def test_poll_shows_exit_code(self):
        """poll() shows exit_code when process has completed."""
        registry = _make_registry()
        info = _inject_process(registry, pid=300, status="completed")
        info.exit_code = 0

        result = registry.poll(300)
        assert "exit_code=0" in result

    def test_poll_shows_uptime(self):
        """poll() shows uptime in the status line."""
        registry = _make_registry()
        _inject_process(registry, pid=400, start_time=time.time() - 120)

        result = registry.poll(400)
        assert "uptime=" in result


# ---------------------------------------------------------------------------
# ProcessRegistry.write
# ---------------------------------------------------------------------------

class TestWriteStdin:
    """Test ProcessRegistry.write()."""

    @pytest.mark.asyncio
    async def test_write_stdin_success(self):
        """write() sends text to process stdin."""
        registry = _make_registry()
        proc = _make_fake_process(pid=500)
        _inject_process(registry, pid=500, process=proc)

        result = await registry.write(500, "hello\n")

        assert "Wrote 6 bytes" in result
        proc.stdin.write.assert_called_once_with(b"hello\n")
        proc.stdin.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_unknown_pid(self):
        """write() to unknown PID returns error."""
        registry = _make_registry()
        result = await registry.write(9999, "data")
        assert "No process with PID 9999" in result

    @pytest.mark.asyncio
    async def test_write_to_completed_process(self):
        """write() to a completed process returns error."""
        registry = _make_registry()
        _inject_process(registry, pid=600, status="completed")

        result = await registry.write(600, "data")
        assert "not running" in result

    @pytest.mark.asyncio
    async def test_write_no_stdin(self):
        """write() when process has no stdin returns error."""
        registry = _make_registry()
        proc = _make_fake_process(pid=700)
        proc.stdin = None
        _inject_process(registry, pid=700, process=proc)

        result = await registry.write(700, "data")
        assert "no stdin" in result

    @pytest.mark.asyncio
    async def test_write_drain_error(self):
        """write() returns error on drain failure."""
        registry = _make_registry()
        proc = _make_fake_process(pid=800)
        proc.stdin.drain = AsyncMock(side_effect=BrokenPipeError("pipe broke"))
        _inject_process(registry, pid=800, process=proc)

        result = await registry.write(800, "data")
        assert "Failed to write" in result


# ---------------------------------------------------------------------------
# ProcessRegistry.kill
# ---------------------------------------------------------------------------

class TestKillProcess:
    """Test ProcessRegistry.kill()."""

    @pytest.mark.asyncio
    async def test_kill_running_process(self):
        """kill() terminates a running process."""
        registry = _make_registry()
        proc = _make_fake_process(pid=900)
        _inject_process(registry, pid=900, process=proc)

        result = await registry.kill(900)

        assert "killed" in result.lower()
        proc.terminate.assert_called_once()
        assert registry._processes[900].status == "failed"
        assert registry._processes[900].exit_code == -9

    @pytest.mark.asyncio
    async def test_kill_unknown_pid(self):
        """kill() for unknown PID returns error."""
        registry = _make_registry()
        result = await registry.kill(9999)
        assert "No process with PID 9999" in result

    @pytest.mark.asyncio
    async def test_kill_already_completed(self):
        """kill() on already completed process returns status."""
        registry = _make_registry()
        _inject_process(registry, pid=1000, status="completed")

        result = await registry.kill(1000)
        assert "already completed" in result

    @pytest.mark.asyncio
    async def test_kill_with_timeout_escalates_to_kill(self):
        """kill() escalates to kill() if terminate doesn't stop it in 5s."""
        registry = _make_registry()
        proc = _make_fake_process(pid=1100)
        proc.wait = AsyncMock(side_effect=asyncio.TimeoutError)
        _inject_process(registry, pid=1100, process=proc)

        result = await registry.kill(1100)

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert "killed" in result.lower()


# ---------------------------------------------------------------------------
# ProcessRegistry.list_all
# ---------------------------------------------------------------------------

class TestListProcesses:
    """Test ProcessRegistry.list_all()."""

    def test_list_empty(self):
        """list_all() with no processes returns message."""
        registry = _make_registry()
        result = registry.list_all()
        assert "No processes tracked" in result

    def test_list_with_processes(self):
        """list_all() shows all tracked processes in a table."""
        registry = _make_registry()
        _inject_process(registry, pid=10, command="echo hello", status="running")
        _inject_process(registry, pid=20, command="sleep 1000", status="completed")

        result = registry.list_all()
        assert "10" in result
        assert "20" in result
        assert "echo hello" in result
        assert "sleep 1000" in result
        assert "running" in result
        assert "completed" in result

    def test_list_shows_uptime_formatting(self):
        """list_all() formats uptime as seconds, minutes, or hours."""
        registry = _make_registry()
        # 30 seconds ago
        _inject_process(registry, pid=1, start_time=time.time() - 30, command="cmd1")
        # 5 minutes ago
        _inject_process(registry, pid=2, start_time=time.time() - 300, command="cmd2")
        # 2 hours ago
        _inject_process(registry, pid=3, start_time=time.time() - 7200, command="cmd3")

        result = registry.list_all()
        # Should have 's', 'm', 'h' formats
        assert "s" in result
        assert "m" in result
        assert "h" in result


# ---------------------------------------------------------------------------
# ProcessRegistry.cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """Test ProcessRegistry.cleanup()."""

    def test_cleanup_removes_old_dead_processes(self):
        """cleanup() removes dead processes older than MAX_LIFETIME_SECONDS."""
        registry = _make_registry()
        old_time = time.time() - MAX_LIFETIME_SECONDS - 100
        _inject_process(registry, pid=1, status="completed", start_time=old_time)
        _inject_process(registry, pid=2, status="failed", start_time=old_time)
        _inject_process(registry, pid=3, status="running")  # Should be kept

        removed = registry.cleanup()

        assert removed == 2
        assert 1 not in registry._processes
        assert 2 not in registry._processes
        assert 3 in registry._processes

    def test_cleanup_preserves_recent_dead(self):
        """cleanup() keeps recently-dead processes."""
        registry = _make_registry()
        _inject_process(registry, pid=1, status="completed", start_time=time.time())

        removed = registry.cleanup()
        assert removed == 0
        assert 1 in registry._processes

    def test_cleanup_cancels_reader_task(self):
        """cleanup() cancels pending reader tasks on removed processes."""
        registry = _make_registry()
        old_time = time.time() - MAX_LIFETIME_SECONDS - 100
        info = _inject_process(registry, pid=1, status="completed", start_time=old_time)
        mock_task = MagicMock()
        mock_task.done.return_value = False
        info._reader_task = mock_task

        registry.cleanup()

        mock_task.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# Max concurrent limit
# ---------------------------------------------------------------------------

class TestMaxConcurrentLimit:
    """Test enforcement of MAX_CONCURRENT process limit."""

    @pytest.mark.asyncio
    async def test_max_concurrent_blocks_start(self):
        """start() refuses when MAX_CONCURRENT running processes exist."""
        registry = _make_registry()
        # Fill up with MAX_CONCURRENT running processes
        for i in range(MAX_CONCURRENT):
            _inject_process(registry, pid=i + 1000, status="running")

        result = await registry.start("localhost", "one_more")

        assert "Cannot start" in result
        assert str(MAX_CONCURRENT) in result

    @pytest.mark.asyncio
    async def test_completed_dont_count_toward_limit(self):
        """Completed processes don't count toward the concurrent limit."""
        registry = _make_registry()
        # Fill with completed processes
        for i in range(MAX_CONCURRENT):
            _inject_process(registry, pid=i + 2000, status="completed")

        fake_proc = _make_fake_process(pid=3000)
        with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = fake_proc
            result = await registry.start("localhost", "echo ok")

        assert "Process started" in result


# ---------------------------------------------------------------------------
# Max lifetime enforcement
# ---------------------------------------------------------------------------

class TestMaxLifetimeEnforcement:
    """Test auto-kill after MAX_LIFETIME_SECONDS."""

    @pytest.mark.asyncio
    async def test_enforce_lifetime_kills_running_process(self):
        """_enforce_lifetime kills a running process after the timeout."""
        registry = _make_registry()
        proc = _make_fake_process(pid=4000)
        _inject_process(registry, pid=4000, process=proc)

        # Call _enforce_lifetime with a very short timeout
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await registry._enforce_lifetime(4000, 0)

        assert registry._processes[4000].status == "failed"

    @pytest.mark.asyncio
    async def test_enforce_lifetime_skips_already_completed(self):
        """_enforce_lifetime does nothing for already-completed processes."""
        registry = _make_registry()
        _inject_process(registry, pid=5000, status="completed")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await registry._enforce_lifetime(5000, 0)

        # Status unchanged
        assert registry._processes[5000].status == "completed"

    @pytest.mark.asyncio
    async def test_enforce_lifetime_skips_removed_process(self):
        """_enforce_lifetime does nothing if PID no longer exists."""
        registry = _make_registry()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Should not raise
            await registry._enforce_lifetime(99999, 0)


# ---------------------------------------------------------------------------
# ProcessInfo dataclass
# ---------------------------------------------------------------------------

class TestProcessInfo:
    """Test ProcessInfo dataclass defaults."""

    def test_default_status(self):
        info = ProcessInfo(pid=1, command="cmd", host="h", start_time=0.0)
        assert info.status == "running"
        assert info.exit_code is None
        assert isinstance(info.output_buffer, deque)
        assert info.output_buffer.maxlen == OUTPUT_BUFFER_LINES

    def test_output_buffer_maxlen(self):
        """Output buffer respects maxlen."""
        info = ProcessInfo(pid=1, command="cmd", host="h", start_time=0.0)
        for i in range(OUTPUT_BUFFER_LINES + 100):
            info.output_buffer.append(f"line{i}\n")
        assert len(info.output_buffer) == OUTPUT_BUFFER_LINES


# ---------------------------------------------------------------------------
# Executor dispatch — _handle_manage_process
# ---------------------------------------------------------------------------

class TestManageProcessHandler:
    """Test _handle_manage_process routing in executor.py."""

    @pytest.mark.asyncio
    async def test_dispatch_start(self):
        """action=start routes to registry.start()."""
        executor = ToolExecutor(MagicMock())
        registry_mock = MagicMock()
        registry_mock.start = AsyncMock(return_value="Process started (PID 42): echo hi")
        registry_mock.cleanup = MagicMock(return_value=0)
        executor._process_registry = registry_mock

        result = await executor._handle_manage_process({
            "action": "start", "host": "localhost", "command": "echo hi"
        })

        assert "Process started" in result
        registry_mock.start.assert_awaited_once_with("localhost", "echo hi")

    @pytest.mark.asyncio
    async def test_dispatch_poll(self):
        """action=poll routes to registry.poll()."""
        executor = ToolExecutor(MagicMock())
        registry_mock = MagicMock()
        registry_mock.poll = MagicMock(return_value="[PID 42] status=running")
        executor._process_registry = registry_mock

        result = await executor._handle_manage_process({"action": "poll", "pid": 42})

        assert "PID 42" in result

    @pytest.mark.asyncio
    async def test_dispatch_write(self):
        """action=write routes to registry.write()."""
        executor = ToolExecutor(MagicMock())
        registry_mock = MagicMock()
        registry_mock.write = AsyncMock(return_value="Wrote 5 bytes to PID 42.")
        executor._process_registry = registry_mock

        result = await executor._handle_manage_process({
            "action": "write", "pid": 42, "input_text": "hello"
        })

        assert "Wrote 5 bytes" in result

    @pytest.mark.asyncio
    async def test_dispatch_kill(self):
        """action=kill routes to registry.kill()."""
        executor = ToolExecutor(MagicMock())
        registry_mock = MagicMock()
        registry_mock.kill = AsyncMock(return_value="Process 42 killed.")
        executor._process_registry = registry_mock

        result = await executor._handle_manage_process({"action": "kill", "pid": 42})

        assert "killed" in result.lower()

    @pytest.mark.asyncio
    async def test_dispatch_list(self):
        """action=list routes to registry.list_all()."""
        executor = ToolExecutor(MagicMock())
        registry_mock = MagicMock()
        registry_mock.list_all = MagicMock(return_value="No processes tracked.")
        executor._process_registry = registry_mock

        result = await executor._handle_manage_process({"action": "list"})

        assert "No processes" in result

    @pytest.mark.asyncio
    async def test_dispatch_default_is_list(self):
        """Missing action defaults to list."""
        executor = ToolExecutor(MagicMock())
        registry_mock = MagicMock()
        registry_mock.list_all = MagicMock(return_value="No processes tracked.")
        executor._process_registry = registry_mock

        result = await executor._handle_manage_process({})

        assert "No processes" in result

    @pytest.mark.asyncio
    async def test_dispatch_unknown_action(self):
        """Unknown action returns error."""
        executor = ToolExecutor(MagicMock())
        executor._process_registry = MagicMock()

        result = await executor._handle_manage_process({"action": "explode"})

        assert "Unknown action" in result

    @pytest.mark.asyncio
    async def test_start_missing_command(self):
        """start without command returns error."""
        executor = ToolExecutor(MagicMock())
        executor._process_registry = MagicMock()

        result = await executor._handle_manage_process({"action": "start", "host": "h"})
        assert "command is required" in result

    @pytest.mark.asyncio
    async def test_start_missing_host(self):
        """start without host returns error."""
        executor = ToolExecutor(MagicMock())
        executor._process_registry = MagicMock()

        result = await executor._handle_manage_process({"action": "start", "command": "c"})
        assert "host is required" in result

    @pytest.mark.asyncio
    async def test_poll_missing_pid(self):
        """poll without pid returns error."""
        executor = ToolExecutor(MagicMock())
        executor._process_registry = MagicMock()

        result = await executor._handle_manage_process({"action": "poll"})
        assert "pid is required" in result

    @pytest.mark.asyncio
    async def test_write_missing_pid(self):
        """write without pid returns error."""
        executor = ToolExecutor(MagicMock())
        executor._process_registry = MagicMock()

        result = await executor._handle_manage_process({"action": "write", "input_text": "x"})
        assert "pid is required" in result

    @pytest.mark.asyncio
    async def test_write_missing_input_text(self):
        """write without input_text returns error."""
        executor = ToolExecutor(MagicMock())
        executor._process_registry = MagicMock()

        result = await executor._handle_manage_process({"action": "write", "pid": 1})
        assert "input_text is required" in result

    @pytest.mark.asyncio
    async def test_kill_missing_pid(self):
        """kill without pid returns error."""
        executor = ToolExecutor(MagicMock())
        executor._process_registry = MagicMock()

        result = await executor._handle_manage_process({"action": "kill"})
        assert "pid is required" in result

    @pytest.mark.asyncio
    async def test_lazy_init_creates_registry(self):
        """First call to _handle_manage_process creates ProcessRegistry."""
        executor = ToolExecutor(MagicMock())
        assert not hasattr(executor, "_process_registry")

        result = await executor._handle_manage_process({"action": "list"})

        assert hasattr(executor, "_process_registry")
        assert isinstance(executor._process_registry, ProcessRegistry)


# ---------------------------------------------------------------------------
# _read_output internal
# ---------------------------------------------------------------------------

class TestReadOutput:
    """Test the background output reader."""

    @pytest.mark.asyncio
    async def test_read_output_fills_buffer(self):
        """_read_output reads lines into the output buffer."""
        registry = _make_registry()
        proc = _make_fake_process(pid=10)
        info = _inject_process(registry, pid=10, process=proc)

        lines = [b"line1\n", b"line2\n", b""]
        call_count = 0

        async def fake_readline():
            nonlocal call_count
            if call_count < len(lines):
                result = lines[call_count]
                call_count += 1
                return result
            return b""

        proc.stdout.readline = fake_readline

        await registry._read_output(info)

        assert "line1\n" in info.output_buffer
        assert "line2\n" in info.output_buffer
