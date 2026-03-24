"""Background process lifecycle management.

Provides start/poll/write/kill/list operations for long-running processes
spawned locally or on remote hosts. Each process gets a ring buffer of
output lines (max 500) and is auto-killed after 1 hour.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from ..logging import get_logger

log = get_logger("process_manager")

MAX_CONCURRENT = 20
MAX_LIFETIME_SECONDS = 3600  # 1 hour
OUTPUT_BUFFER_LINES = 500


@dataclass
class ProcessInfo:
    """Metadata and handles for a managed process."""

    pid: int
    command: str
    host: str
    start_time: float
    status: str = "running"  # running | completed | failed
    output_buffer: deque = field(default_factory=lambda: deque(maxlen=OUTPUT_BUFFER_LINES))
    process: asyncio.subprocess.Process | None = None
    _reader_task: asyncio.Task | None = field(default=None, repr=False)
    exit_code: int | None = None


class ProcessRegistry:
    """Registry for background processes with full lifecycle management."""

    def __init__(self) -> None:
        self._processes: dict[int, ProcessInfo] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, host: str, command: str, timeout: int = 300) -> str:
        """Start a background process. Returns confirmation with PID."""
        # Enforce concurrency limit (only count running)
        running = sum(1 for p in self._processes.values() if p.status == "running")
        if running >= MAX_CONCURRENT:
            return f"Cannot start: {running} processes already running (max {MAX_CONCURRENT})."

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            return f"Failed to start process: {e}"

        pid = proc.pid
        info = ProcessInfo(
            pid=pid,
            command=command,
            host=host,
            start_time=time.time(),
            process=proc,
        )
        self._processes[pid] = info

        # Background reader task to drain stdout into the ring buffer
        info._reader_task = asyncio.create_task(self._read_output(info))

        # Auto-kill after max lifetime
        asyncio.create_task(self._enforce_lifetime(pid, MAX_LIFETIME_SECONDS))

        log.info("Started process PID %d: %s", pid, command[:80])
        return f"Process started (PID {pid}): {command[:120]}"

    def poll(self, pid: int) -> str:
        """Return recent output lines from a process."""
        info = self._processes.get(pid)
        if not info:
            return f"No process with PID {pid}."

        lines = list(info.output_buffer)
        status_line = f"[PID {pid}] status={info.status}"
        if info.exit_code is not None:
            status_line += f" exit_code={info.exit_code}"
        elapsed = time.time() - info.start_time
        status_line += f" uptime={elapsed:.0f}s"

        if not lines:
            return f"{status_line}\n(no output yet)"
        # Show last 50 lines by default
        recent = lines[-50:]
        return f"{status_line}\n" + "".join(recent)

    async def write(self, pid: int, text: str) -> str:
        """Write text to a process's stdin."""
        info = self._processes.get(pid)
        if not info:
            return f"No process with PID {pid}."
        if info.status != "running":
            return f"Process {pid} is not running (status: {info.status})."
        if not info.process or not info.process.stdin:
            return f"Process {pid} has no stdin."

        try:
            info.process.stdin.write(text.encode())
            await info.process.stdin.drain()
            return f"Wrote {len(text)} bytes to PID {pid}."
        except Exception as e:
            return f"Failed to write to PID {pid}: {e}"

    async def kill(self, pid: int) -> str:
        """Kill a running process."""
        info = self._processes.get(pid)
        if not info:
            return f"No process with PID {pid}."
        if info.status != "running":
            return f"Process {pid} already {info.status}."

        try:
            if info.process:
                info.process.terminate()
                # Give it a moment to exit gracefully
                try:
                    await asyncio.wait_for(info.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    info.process.kill()
            info.status = "failed"
            info.exit_code = -9
            log.info("Killed process PID %d", pid)
            return f"Process {pid} killed."
        except Exception as e:
            return f"Failed to kill PID {pid}: {e}"

    def list_all(self) -> str:
        """Return a formatted table of all tracked processes."""
        if not self._processes:
            return "No processes tracked."

        lines = [f"{'PID':<8} {'STATUS':<12} {'UPTIME':<10} {'COMMAND'}"]
        lines.append("-" * 60)
        now = time.time()
        for pid, info in sorted(self._processes.items()):
            elapsed = now - info.start_time
            if elapsed < 60:
                uptime = f"{elapsed:.0f}s"
            elif elapsed < 3600:
                uptime = f"{elapsed / 60:.1f}m"
            else:
                uptime = f"{elapsed / 3600:.1f}h"
            cmd_short = info.command[:40]
            lines.append(f"{pid:<8} {info.status:<12} {uptime:<10} {cmd_short}")
        return "\n".join(lines)

    def cleanup(self) -> int:
        """Remove dead processes older than 1 hour. Returns count removed."""
        now = time.time()
        to_remove = [
            pid for pid, info in self._processes.items()
            if info.status != "running"
            and (now - info.start_time) > MAX_LIFETIME_SECONDS
        ]
        for pid in to_remove:
            info = self._processes.pop(pid)
            if info._reader_task and not info._reader_task.done():
                info._reader_task.cancel()
        return len(to_remove)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _read_output(self, info: ProcessInfo) -> None:
        """Continuously read stdout and append to the output buffer."""
        try:
            while info.process and info.process.stdout:
                line = await info.process.stdout.readline()
                if not line:
                    break
                info.output_buffer.append(line.decode("utf-8", errors="replace"))
        except Exception:
            pass

        # Process finished — update status
        if info.process:
            try:
                await info.process.wait()
                info.exit_code = info.process.returncode
                info.status = "completed" if info.exit_code == 0 else "failed"
            except Exception:
                info.status = "failed"

    async def _enforce_lifetime(self, pid: int, max_seconds: int) -> None:
        """Auto-kill process after max lifetime."""
        await asyncio.sleep(max_seconds)
        info = self._processes.get(pid)
        if info and info.status == "running":
            log.warning("Auto-killing PID %d after %ds lifetime limit", pid, max_seconds)
            await self.kill(pid)
