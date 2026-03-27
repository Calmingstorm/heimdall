"""Autonomous loop system — LLM-intelligent recurring tasks.

Each loop iteration triggers a full LLM reasoning cycle with tool access.
The LLM decides what to check, how to interpret results, and what to report.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..llm.secret_scrubber import scrub_output_secrets
from ..logging import get_logger

log = get_logger("autonomous_loop")

# Type for the LLM iteration callback:
# Takes (goal_prompt, channel, iteration_context) -> response text
# The callback should run the full Codex + tool loop internally.
LoopIterationCallback = Callable[
    [str, Any, str | None],  # (prompt, channel, previous_context)
    Awaitable[str],
]

MAX_CONCURRENT_LOOPS = 10
MAX_LOOP_LIFETIME_SECONDS = 4 * 3600  # 4 hours
MIN_INTERVAL_SECONDS = 10
DEFAULT_INTERVAL_SECONDS = 60
DEFAULT_MAX_ITERATIONS = 50
MAX_CONTEXT_HISTORY = 3  # Keep last N iteration summaries for context
MAX_CONSECUTIVE_ERRORS = 5  # Stop loop after this many consecutive failures
MAX_BACKOFF_SECONDS = 300  # Cap exponential backoff at 5 minutes
RUNAWAY_THRESHOLD = 3  # Identical outputs before interval increase

LOOP_STOP_SENTINEL = "LOOP_STOP"


@dataclass
class LoopInfo:
    """Metadata for an active autonomous loop."""
    id: str
    goal: str
    mode: str  # "notify", "act", "silent"
    interval_seconds: int
    stop_condition: str | None
    max_iterations: int
    channel_id: str
    requester_id: str
    requester_name: str
    iteration_count: int = 0
    last_trigger: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "running"  # running, stopped, completed, error
    _task: asyncio.Task | None = field(default=None, repr=False)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    _iteration_history: list[str] = field(default_factory=list)


class LoopManager:
    """Manages autonomous loops — LLM-driven recurring tasks."""

    def __init__(self, agents_enabled: bool = False) -> None:
        self._loops: dict[str, LoopInfo] = {}
        self._agents_enabled = agents_enabled

    @property
    def active_count(self) -> int:
        return sum(1 for loop in self._loops.values() if loop.status == "running")

    def start_loop(
        self,
        goal: str,
        channel: Any,  # discord.abc.Messageable
        requester_id: str,
        requester_name: str,
        iteration_callback: LoopIterationCallback,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        mode: str = "notify",
        stop_condition: str | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> str:
        """Start a new autonomous loop. Returns the loop ID or an error string."""
        if self.active_count >= MAX_CONCURRENT_LOOPS:
            return f"Error: Maximum concurrent loops ({MAX_CONCURRENT_LOOPS}) reached. Stop a loop first."

        if mode not in ("notify", "act", "silent"):
            mode = "notify"
        interval_seconds = max(MIN_INTERVAL_SECONDS, interval_seconds)
        max_iterations = max(1, min(max_iterations, 1000))

        loop_id = uuid.uuid4().hex[:8]
        info = LoopInfo(
            id=loop_id,
            goal=goal,
            mode=mode,
            interval_seconds=interval_seconds,
            stop_condition=stop_condition,
            max_iterations=max_iterations,
            channel_id=str(getattr(channel, "id", "")),
            requester_id=requester_id,
            requester_name=requester_name,
        )
        self._loops[loop_id] = info

        info._task = asyncio.create_task(
            self._run_loop(info, channel, iteration_callback)
        )

        log.info(
            "Loop %s started: goal=%r interval=%ds mode=%s max=%d",
            loop_id, goal, interval_seconds, mode, max_iterations,
        )
        return loop_id

    def stop_loop(self, loop_id: str) -> str:
        """Stop a loop by ID. Use 'all' to stop all loops."""
        if loop_id == "all":
            stopped = []
            for lid, info in list(self._loops.items()):
                if info.status == "running":
                    info._cancel_event.set()
                    info.status = "stopped"
                    stopped.append(lid)
            if not stopped:
                return "No active loops to stop."
            return f"Stopped {len(stopped)} loop(s): {', '.join(stopped)}"

        info = self._loops.get(loop_id)
        if not info:
            return f"No loop found with ID `{loop_id}`."
        if info.status != "running":
            return f"Loop `{loop_id}` is not running (status: {info.status})."
        info._cancel_event.set()
        info.status = "stopped"
        return f"Loop `{loop_id}` stopped."

    def list_loops(self) -> str:
        """Return a formatted list of all loops."""
        if not self._loops:
            return "No autonomous loops."

        lines = []
        for lid, info in self._loops.items():
            elapsed = ""
            if info.last_trigger:
                ago = int(time.monotonic() - info.last_trigger)
                elapsed = f", last ran {ago}s ago"
            lines.append(
                f"- `{lid}` [{info.status}] **{info.goal[:80]}** "
                f"(every {info.interval_seconds}s, mode={info.mode}, "
                f"iter {info.iteration_count}/{info.max_iterations}{elapsed})"
            )
        return "\n".join(lines)

    def cleanup_finished(self) -> None:
        """Remove loops that have been finished for a while."""
        now = time.monotonic()
        to_remove = []
        for lid, info in self._loops.items():
            if info.status != "running" and info.last_trigger:
                if now - info.last_trigger > 3600:  # 1 hour after finish
                    to_remove.append(lid)
        for lid in to_remove:
            del self._loops[lid]

    async def _run_loop(
        self,
        info: LoopInfo,
        channel: Any,
        iteration_callback: LoopIterationCallback,
    ) -> None:
        """Main loop coroutine — runs iterations until stopped."""
        start_time = time.monotonic()
        consecutive_identical = 0
        consecutive_errors = 0
        last_output = ""

        try:
            while info.iteration_count < info.max_iterations:
                # Check cancellation
                if info._cancel_event.is_set():
                    break

                # Check lifetime limit
                if time.monotonic() - start_time > MAX_LOOP_LIFETIME_SECONDS:
                    info.status = "completed"
                    try:
                        await channel.send(
                            f"Loop `{info.id}` reached maximum lifetime (4 hours). Stopped."
                        )
                    except Exception:
                        pass
                    break

                # Calculate wait time: normal interval + exponential backoff on errors
                wait_seconds = info.interval_seconds
                if consecutive_errors > 0:
                    backoff = min(
                        info.interval_seconds * (2 ** consecutive_errors),
                        MAX_BACKOFF_SECONDS,
                    )
                    wait_seconds = backoff
                    log.info(
                        "Loop %s: backing off %ds after %d consecutive errors",
                        info.id, wait_seconds, consecutive_errors,
                    )

                # Wait for interval (interruptible by cancel)
                try:
                    await asyncio.wait_for(
                        info._cancel_event.wait(),
                        timeout=wait_seconds,
                    )
                    # If we get here, cancel was set during the wait
                    break
                except asyncio.TimeoutError:
                    pass  # Normal — interval elapsed, proceed with iteration

                # Check cancellation again after wait
                if info._cancel_event.is_set():
                    break

                info.iteration_count += 1
                info.last_trigger = time.monotonic()

                # Build iteration prompt
                prompt = self._build_iteration_prompt(info)

                # Build previous context from iteration history
                prev_context = None
                if info._iteration_history:
                    prev_context = "\n---\n".join(info._iteration_history[-MAX_CONTEXT_HISTORY:])

                # Run the LLM iteration
                try:
                    response = await iteration_callback(prompt, channel, prev_context)
                    response = scrub_output_secrets(response.strip()) if response else ""
                    consecutive_errors = 0  # Reset on success
                except Exception as e:
                    consecutive_errors += 1
                    log.warning(
                        "Loop %s iteration %d failed (%d consecutive): %s",
                        info.id, info.iteration_count, consecutive_errors, e,
                    )
                    # Store error in history but don't crash the loop
                    info._iteration_history.append(
                        f"Iteration {info.iteration_count}: ERROR - {str(e)[:200]}"
                    )
                    # Stop loop after too many consecutive errors
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        info.status = "error"
                        try:
                            await channel.send(
                                f"Loop `{info.id}` stopped after {MAX_CONSECUTIVE_ERRORS} "
                                f"consecutive errors. Last error: {str(e)[:200]}"
                            )
                        except Exception:
                            pass
                        break
                    continue

                # Store iteration result (truncated) in history
                summary = response[:500] if response else "(no output)"
                info._iteration_history.append(
                    f"Iteration {info.iteration_count}: {summary}"
                )
                # Trim history to prevent unbounded growth
                if len(info._iteration_history) > MAX_CONTEXT_HISTORY * 2:
                    info._iteration_history = info._iteration_history[-MAX_CONTEXT_HISTORY:]

                # Check for LOOP_STOP sentinel
                if LOOP_STOP_SENTINEL in response:
                    info.status = "completed"
                    log.info("Loop %s stopped by LLM (LOOP_STOP)", info.id)
                    break

                # Runaway detection: identical consecutive outputs
                if response == last_output and response:
                    consecutive_identical += 1
                    if consecutive_identical >= RUNAWAY_THRESHOLD:
                        old_interval = info.interval_seconds
                        info.interval_seconds = min(
                            info.interval_seconds * 2, 3600,
                        )
                        log.warning(
                            "Loop %s: %d identical outputs, interval %ds -> %ds",
                            info.id, RUNAWAY_THRESHOLD,
                            old_interval, info.interval_seconds,
                        )
                        try:
                            await channel.send(
                                f"Loop `{info.id}`: {RUNAWAY_THRESHOLD} identical "
                                f"outputs detected — increasing interval from "
                                f"{old_interval}s to {info.interval_seconds}s."
                            )
                        except Exception:
                            pass
                        consecutive_identical = 0
                else:
                    consecutive_identical = 0
                last_output = response

                # Post response to channel based on mode
                if response and info.status == "running":
                    await self._post_response(info, channel, response)

            # Loop ended normally (max iterations reached)
            if info.status == "running":
                info.status = "completed"
                try:
                    await channel.send(
                        f"Loop `{info.id}` completed after {info.iteration_count} iterations."
                    )
                except Exception:
                    pass

        except asyncio.CancelledError:
            info.status = "stopped"
        except Exception as e:
            info.status = "error"
            log.error("Loop %s crashed: %s", info.id, e, exc_info=True)
            try:
                await channel.send(
                    f"Loop `{info.id}` encountered an error and stopped: {str(e)[:200]}"
                )
            except Exception:
                pass

    def _build_iteration_prompt(self, info: LoopInfo) -> str:
        """Build the prompt for a single loop iteration."""
        parts = [
            f"AUTONOMOUS LOOP (iteration {info.iteration_count} of {info.max_iterations})",
            f"Goal: {info.goal}",
            f"Mode: {info.mode}",
        ]
        if info.stop_condition:
            parts.append(f"Stop condition: {info.stop_condition}")

        parts.append("")
        parts.append(
            "You are in an autonomous loop. Execute the goal above using tools. When done:"
        )
        if info.mode in ("notify", "act"):
            parts.append("- Post a concise update to the channel.")
        elif info.mode == "silent":
            parts.append(
                "- Only respond if something notable or urgent happened. "
                "If you need to report something, include [NOTIFY] at the start "
                "of your response. For critical issues, use [ALERT] instead. "
                "Responses without these markers will be suppressed."
            )

        if info.stop_condition:
            parts.append(
                f'- If the stop condition is met ("{info.stop_condition}"), '
                f'include the exact text "LOOP_STOP" in your response.'
            )

        # Agent awareness: tell the LLM it can spawn agents for parallel sub-tasks
        if self._agents_enabled:
            parts.append("")
            parts.append(
                "AGENTS: You can spawn agents (spawn_agent) for parallel sub-tasks. "
                "Use wait_for_agents to collect results. Good for: investigating "
                "multiple hosts, running parallel checks, delegating fixes."
            )

        return "\n".join(parts)

    async def _post_response(
        self, info: LoopInfo, channel: Any, response: str,
    ) -> None:
        """Post the loop iteration response to the channel, respecting mode.

        - notify/act: always post the response.
        - silent: only post if the response contains [NOTIFY] or [ALERT].
        """
        if info.mode == "silent":
            if "[NOTIFY]" not in response and "[ALERT]" not in response:
                log.debug(
                    "Loop %s: silent mode suppressed output (%d chars)",
                    info.id, len(response),
                )
                return

        # Post to channel (truncate for Discord limit)
        try:
            text = response
            if len(text) > 2000:
                text = text[:1950] + "\n... (truncated)"
            await channel.send(text)
        except Exception as e:
            log.warning("Loop %s: failed to post response: %s", info.id, e)
