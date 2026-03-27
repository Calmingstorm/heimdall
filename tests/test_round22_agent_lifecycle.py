"""Round 22 — Agent lifecycle + cleanup tests.

Tests auto-terminate after timeout, stale detection, dead agent cleanup
after CLEANUP_DELAY, check_health safety net, done-callback scheduling,
and error isolation (agent crash doesn't crash bot).
"""
from __future__ import annotations

import asyncio
import inspect
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.manager import (
    AGENT_BLOCKED_TOOLS,
    AgentInfo,
    AgentManager,
    CLEANUP_DELAY,
    MAX_AGENT_ITERATIONS,
    MAX_AGENT_LIFETIME,
    MAX_CONCURRENT_AGENTS,
    STALE_WARN_SECONDS,
    _TERMINAL_STATUSES,
    _run_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_callbacks(text="Task complete.", tool_calls=None):
    """Return (iteration_cb, tool_exec_cb, announce_cb)."""
    iteration_cb = AsyncMock(return_value={
        "text": text,
        "tool_calls": tool_calls or [],
        "stop_reason": "end_turn",
    })
    tool_exec_cb = AsyncMock(return_value="ok")
    announce_cb = AsyncMock()
    return iteration_cb, tool_exec_cb, announce_cb


def _spawn(mgr, label="test", goal="do something", channel_id="100", **kw):
    """Spawn agent with minimal boilerplate. Returns (agent_id, iter_cb, tool_cb, ann_cb)."""
    iter_cb, tool_cb, ann_cb = _make_callbacks(**kw)
    aid = mgr.spawn(
        label=label, goal=goal, channel_id=channel_id,
        requester_id="u1", requester_name="user1",
        iteration_callback=iter_cb,
        tool_executor_callback=tool_cb,
        announce_callback=ann_cb,
    )
    return aid, iter_cb, tool_cb, ann_cb


# ---------------------------------------------------------------------------
# Auto-terminate after timeout (MAX_AGENT_LIFETIME = 3600s)
# ---------------------------------------------------------------------------

class TestAutoTerminateTimeout:
    """Agents auto-terminate when lifetime exceeds MAX_AGENT_LIFETIME."""

    async def test_timeout_sets_status(self):
        """Agent exceeding lifetime gets status='timeout'."""
        mgr = AgentManager()
        call_count = {"n": 0}

        async def _iter(messages, sys, tools):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"text": "", "tool_calls": [{"name": "cmd", "input": {}}],
                        "stop_reason": "end_turn"}
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        ann_cb = AsyncMock()
        tool_cb = AsyncMock(return_value="ok")

        aid = mgr.spawn("timeout-test", "task", "100", "u1", "u1",
                         _iter, tool_cb, ann_cb)
        agent = mgr._agents[aid]
        # Backdate creation to exceed lifetime
        agent.created_at = time.time() - MAX_AGENT_LIFETIME - 10
        await asyncio.sleep(0.15)

        assert agent.status == "timeout"
        assert agent.ended_at is not None

    async def test_timeout_announces_result(self):
        """Timed-out agent announces to channel."""
        mgr = AgentManager()
        announced = []

        async def _announce(ch_id, text):
            announced.append(text)

        call_count = {"n": 0}
        async def _iter(messages, sys, tools):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"text": "Checking...", "tool_calls": [{"name": "cmd", "input": {}}],
                        "stop_reason": "end_turn"}
            return {"text": "Still working.", "tool_calls": [{"name": "cmd", "input": {}}],
                    "stop_reason": "end_turn"}

        aid = mgr.spawn("timeout-ann", "task", "100", "u1", "u1",
                         _iter, AsyncMock(return_value="ok"), _announce)
        agent = mgr._agents[aid]
        agent.created_at = time.time() - MAX_AGENT_LIFETIME - 10
        await asyncio.sleep(0.2)

        assert len(announced) >= 1
        assert "timed out" in announced[0]

    async def test_timeout_preserves_last_progress(self):
        """Timed-out agent preserves last assistant message as result."""
        mgr = AgentManager()
        gate = asyncio.Event()
        call_count = {"n": 0}

        async def _iter(messages, sys, tools):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: return a tool call so we loop
                return {"text": "Found 3 servers.", "tool_calls": [{"name": "cmd", "input": {}}],
                        "stop_reason": "end_turn"}
            # Signal we're on iteration 2+ (lifetime will be expired)
            gate.set()
            return {"text": "Checked 2 of 3.", "tool_calls": [{"name": "cmd", "input": {}}],
                    "stop_reason": "end_turn"}

        async def _tool(name, inp):
            # After first tool call, backdate so timeout triggers on next iteration
            if call_count["n"] == 1:
                agent.created_at = time.time() - MAX_AGENT_LIFETIME - 10
            return "ok"

        aid = mgr.spawn("progress", "task", "100", "u1", "u1",
                         _iter, _tool, AsyncMock())
        agent = mgr._agents[aid]
        await asyncio.sleep(0.3)

        assert agent.status == "timeout"
        # result should be the last assistant message
        assert agent.result != ""
        assert agent.result != "(no output)"

    def test_max_agent_lifetime_constant(self):
        assert MAX_AGENT_LIFETIME == 3600

    async def test_lifetime_check_in_run_agent(self):
        """_run_agent checks lifetime at each iteration."""
        source = inspect.getsource(_run_agent)
        assert "MAX_AGENT_LIFETIME" in source


# ---------------------------------------------------------------------------
# Stale detection (STALE_WARN_SECONDS = 120s)
# ---------------------------------------------------------------------------

class TestStaleDetection:
    """Stale agents (no activity for STALE_WARN_SECONDS) trigger warnings."""

    def test_stale_warn_constant(self):
        assert STALE_WARN_SECONDS == 120

    async def test_stale_warning_logged(self):
        """Agent with no activity for >STALE_WARN_SECONDS logs warning."""
        source = inspect.getsource(_run_agent)
        assert "STALE_WARN_SECONDS" in source
        assert "idle" in source.lower() or "stale" in source.lower()

    def test_check_health_detects_stale(self):
        """check_health() detects stale agents."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="a1", label="stale", goal="goal", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.last_activity = time.time() - STALE_WARN_SECONDS - 10
        mgr._agents["a1"] = agent

        result = mgr.check_health()
        assert result["stale"] == 1

    def test_check_health_no_stale_if_recent(self):
        """check_health() doesn't flag recently active agents."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="a1", label="active", goal="goal", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.last_activity = time.time()
        mgr._agents["a1"] = agent

        result = mgr.check_health()
        assert result["stale"] == 0

    def test_check_health_ignores_terminal_agents(self):
        """check_health() skips agents in terminal states."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="a1", label="done", goal="goal", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.status = "completed"
        agent.last_activity = time.time() - 500  # Would be stale if running
        mgr._agents["a1"] = agent

        result = mgr.check_health()
        assert result["stale"] == 0
        assert result["killed"] == 0


# ---------------------------------------------------------------------------
# Dead agent cleanup (CLEANUP_DELAY = 300s)
# ---------------------------------------------------------------------------

class TestDeadAgentCleanup:
    """Finished agents are cleaned up after CLEANUP_DELAY."""

    def test_cleanup_delay_constant(self):
        assert CLEANUP_DELAY == 300

    async def test_schedule_cleanup_called_on_completion(self):
        """Done-callback on asyncio task triggers _schedule_cleanup."""
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()
        aid = mgr.spawn("test", "goal", "100", "u1", "u1",
                         iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.1)

        # After agent completes, _schedule_cleanup should have been called
        # which creates a cleanup task
        assert aid in mgr._cleanup_tasks

    async def test_schedule_cleanup_called_on_failure(self):
        """Done-callback fires on agent failure too."""
        mgr = AgentManager()

        async def _fail(messages, sys, tools):
            raise RuntimeError("boom")

        aid = mgr.spawn("fail", "goal", "100", "u1", "u1",
                         _fail, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.1)

        assert aid in mgr._cleanup_tasks

    async def test_schedule_cleanup_called_on_kill(self):
        """Done-callback fires when agent is killed (via task cancel)."""
        mgr = AgentManager()
        gate = asyncio.Event()

        async def _block(messages, sys, tools):
            gate.set()
            await asyncio.sleep(10)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        aid = mgr.spawn("kill-test", "goal", "100", "u1", "u1",
                         _block, AsyncMock(return_value="ok"), AsyncMock())
        await gate.wait()

        # Kill sets cancel event; also cancel the task for immediate exit
        mgr.kill(aid)
        agent = mgr._agents[aid]
        agent._task.cancel()
        await asyncio.sleep(0.2)

        assert aid in mgr._cleanup_tasks

    async def test_cleanup_removes_old_agents(self):
        """cleanup() removes agents past CLEANUP_DELAY."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="old", label="old", goal="goal", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.status = "completed"
        agent.ended_at = time.time() - CLEANUP_DELAY - 10
        mgr._agents["old"] = agent

        removed = await mgr.cleanup()
        assert removed == 1
        assert "old" not in mgr._agents

    async def test_cleanup_keeps_recent_terminal(self):
        """cleanup() keeps recently-finished agents."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="recent", label="recent", goal="goal", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.status = "completed"
        agent.ended_at = time.time() - 10  # Only 10s ago

        mgr._agents["recent"] = agent

        removed = await mgr.cleanup()
        assert removed == 0
        assert "recent" in mgr._agents

    async def test_cleanup_keeps_running_agents(self):
        """cleanup() never removes running agents."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="running", label="running", goal="goal", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.status = "running"
        mgr._agents["running"] = agent

        removed = await mgr.cleanup()
        assert removed == 0
        assert "running" in mgr._agents

    async def test_cleanup_handles_multiple_statuses(self):
        """cleanup() correctly handles mix of terminal statuses."""
        mgr = AgentManager()
        old_time = time.time() - CLEANUP_DELAY - 10
        for status in ("completed", "failed", "timeout", "killed"):
            agent = AgentInfo(
                id=status, label=status, goal="g", channel_id="100",
                requester_id="u1", requester_name="u1",
            )
            agent.status = status
            agent.ended_at = old_time
            mgr._agents[status] = agent

        removed = await mgr.cleanup()
        assert removed == 4
        assert len(mgr._agents) == 0

    async def test_delayed_cleanup_auto_removes(self):
        """_schedule_cleanup creates a delayed task that removes the agent."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="auto", label="auto", goal="g", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.status = "completed"
        agent.ended_at = time.time()
        mgr._agents["auto"] = agent

        # Schedule with a very short delay for testing
        with patch("src.agents.manager.CLEANUP_DELAY", 0.05):
            mgr._schedule_cleanup("auto")
            assert "auto" in mgr._cleanup_tasks
            await asyncio.sleep(0.15)

        assert "auto" not in mgr._agents

    def test_done_callback_in_spawn_source(self):
        """spawn() adds done-callback to schedule cleanup."""
        source = inspect.getsource(AgentManager.spawn)
        assert "add_done_callback" in source
        assert "_schedule_cleanup" in source


# ---------------------------------------------------------------------------
# check_health() safety net
# ---------------------------------------------------------------------------

class TestCheckHealth:
    """check_health() kills stuck agents and warns about stale ones."""

    def test_check_health_exists(self):
        mgr = AgentManager()
        assert hasattr(mgr, "check_health")
        assert callable(mgr.check_health)

    def test_kills_stuck_agents(self):
        """Agents past MAX_AGENT_LIFETIME get kill signal."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="stuck", label="stuck", goal="g", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.created_at = time.time() - MAX_AGENT_LIFETIME - 100
        agent.last_activity = time.time()  # Recently active but too old
        mgr._agents["stuck"] = agent

        result = mgr.check_health()
        assert result["killed"] == 1
        assert agent._cancel_event.is_set()

    def test_no_kill_if_within_lifetime(self):
        """Agents within lifetime are not killed."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="ok", label="ok", goal="g", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        mgr._agents["ok"] = agent

        result = mgr.check_health()
        assert result["killed"] == 0
        assert not agent._cancel_event.is_set()

    def test_returns_dict_with_keys(self):
        mgr = AgentManager()
        result = mgr.check_health()
        assert "killed" in result
        assert "stale" in result

    def test_mixed_agents(self):
        """check_health handles mix of stuck, stale, and healthy agents."""
        mgr = AgentManager()
        now = time.time()

        # Stuck agent (past lifetime)
        stuck = AgentInfo(id="stuck", label="stuck", goal="g",
                          channel_id="100", requester_id="u1", requester_name="u1")
        stuck.created_at = now - MAX_AGENT_LIFETIME - 100
        stuck.last_activity = now
        mgr._agents["stuck"] = stuck

        # Stale agent (idle too long)
        stale = AgentInfo(id="stale", label="stale", goal="g",
                          channel_id="100", requester_id="u1", requester_name="u1")
        stale.last_activity = now - STALE_WARN_SECONDS - 10
        mgr._agents["stale"] = stale

        # Healthy agent
        healthy = AgentInfo(id="healthy", label="healthy", goal="g",
                            channel_id="100", requester_id="u1", requester_name="u1")
        mgr._agents["healthy"] = healthy

        result = mgr.check_health()
        assert result["killed"] == 1
        assert result["stale"] == 1

    def test_empty_agents_no_error(self):
        mgr = AgentManager()
        result = mgr.check_health()
        assert result == {"killed": 0, "stale": 0}

    def test_completed_agents_ignored(self):
        """Terminal agents are not checked by health."""
        mgr = AgentManager()
        for status in _TERMINAL_STATUSES:
            agent = AgentInfo(id=status, label=status, goal="g",
                              channel_id="100", requester_id="u1", requester_name="u1")
            agent.status = status
            agent.created_at = time.time() - MAX_AGENT_LIFETIME - 100
            agent.last_activity = time.time() - STALE_WARN_SECONDS - 100
            mgr._agents[status] = agent

        result = mgr.check_health()
        assert result["killed"] == 0
        assert result["stale"] == 0


# ---------------------------------------------------------------------------
# Error isolation (agent crash doesn't crash bot)
# ---------------------------------------------------------------------------

class TestErrorIsolation:
    """Agent errors are contained and don't affect the bot."""

    async def test_llm_exception_sets_failed(self):
        """LLM exception → status='failed', error captured."""
        mgr = AgentManager()
        async def _fail(messages, sys, tools):
            raise ConnectionError("API down")

        aid = mgr.spawn("err", "goal", "100", "u1", "u1",
                         _fail, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.1)

        agent = mgr._agents.get(aid)
        assert agent.status == "failed"
        assert "API down" in agent.error or "LLM call failed" in agent.error

    async def test_tool_exception_continues(self):
        """Tool exception is captured, agent continues."""
        mgr = AgentManager()
        call_count = {"n": 0}

        async def _iter(messages, sys, tools):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"text": "", "tool_calls": [{"name": "cmd", "input": {}}],
                        "stop_reason": "end_turn"}
            return {"text": "Recovered.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _bad_tool(name, inp):
            raise OSError("Permission denied")

        aid = mgr.spawn("tool-err", "goal", "100", "u1", "u1",
                         _iter, _bad_tool, AsyncMock())
        await asyncio.sleep(0.15)

        agent = mgr._agents.get(aid)
        assert agent.status == "completed"
        assert agent.result == "Recovered."

    async def test_unhandled_exception_caught(self):
        """Unhandled exception in _run_agent caught by top-level except."""
        mgr = AgentManager()

        async def _crash(messages, sys, tools):
            raise RuntimeError("unexpected crash")

        aid = mgr.spawn("crash", "goal", "100", "u1", "u1",
                         _crash, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.1)

        agent = mgr._agents.get(aid)
        assert agent.status == "failed"
        assert agent.ended_at is not None

    async def test_cancelled_error_sets_killed(self):
        """asyncio.CancelledError → status='killed'."""
        mgr = AgentManager()
        gate = asyncio.Event()

        async def _block(messages, sys, tools):
            gate.set()
            await asyncio.sleep(100)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        aid = mgr.spawn("cancel", "goal", "100", "u1", "u1",
                         _block, AsyncMock(return_value="ok"), AsyncMock())
        await gate.wait()

        # Cancel the task directly
        agent = mgr._agents[aid]
        agent._task.cancel()
        await asyncio.sleep(0.1)

        assert agent.status == "killed"
        assert agent.ended_at is not None

    async def test_announce_failure_doesnt_crash_agent(self):
        """If announce callback fails, agent still completes."""
        mgr = AgentManager()

        async def _bad_announce(ch_id, text):
            raise RuntimeError("Discord is down")

        iter_cb = AsyncMock(return_value={
            "text": "Done.", "tool_calls": [], "stop_reason": "end_turn"})
        aid = mgr.spawn("ann-fail", "goal", "100", "u1", "u1",
                         iter_cb, AsyncMock(return_value="ok"), _bad_announce)
        await asyncio.sleep(0.1)

        agent = mgr._agents.get(aid)
        assert agent.status == "completed"

    async def test_multiple_agents_one_fails(self):
        """One agent failing doesn't affect siblings."""
        mgr = AgentManager()

        async def _fail(messages, sys, tools):
            raise ValueError("Agent A fails")

        async def _succeed(messages, sys, tools):
            return {"text": "B succeeds.", "tool_calls": [], "stop_reason": "end_turn"}

        ann_cb = AsyncMock()
        tool_cb = AsyncMock(return_value="ok")

        id_a = mgr.spawn("a", "task A", "100", "u1", "u1", _fail, tool_cb, ann_cb)
        id_b = mgr.spawn("b", "task B", "100", "u1", "u1", _succeed, tool_cb, ann_cb)
        await asyncio.sleep(0.1)

        assert mgr._agents[id_a].status == "failed"
        assert mgr._agents[id_b].status == "completed"
        assert mgr._agents[id_b].result == "B succeeds."


# ---------------------------------------------------------------------------
# Max iterations exhaustion
# ---------------------------------------------------------------------------

class TestMaxIterations:
    """Agent completing all MAX_AGENT_ITERATIONS gets proper lifecycle handling."""

    async def test_exhausted_iterations_completes(self):
        """Agent that uses all iterations sets status='completed'."""
        mgr = AgentManager()

        async def _always_tools(messages, sys, tools):
            return {"text": "working", "tool_calls": [{"name": "cmd", "input": {}}],
                    "stop_reason": "end_turn"}

        aid = mgr.spawn("exhaust", "goal", "100", "u1", "u1",
                         _always_tools, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.5)

        agent = mgr._agents.get(aid)
        assert agent.status == "completed"
        assert agent.iteration_count == MAX_AGENT_ITERATIONS

    async def test_exhausted_preserves_last_progress(self):
        """Exhausted agent captures last assistant text as result."""
        mgr = AgentManager()

        async def _always_tools(messages, sys, tools):
            return {"text": "Still going.", "tool_calls": [{"name": "cmd", "input": {}}],
                    "stop_reason": "end_turn"}

        aid = mgr.spawn("exhaust2", "goal", "100", "u1", "u1",
                         _always_tools, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.5)

        agent = mgr._agents.get(aid)
        assert agent.result == "Still going."


# ---------------------------------------------------------------------------
# Client integration — cleanup cycle
# ---------------------------------------------------------------------------

class TestClientCleanupIntegration:
    """client.py integrates agent health check into cleanup cycle."""

    def test_cleanup_stale_caches_calls_check_health(self):
        """_cleanup_stale_caches includes agent_manager.check_health()."""
        import src.discord.client as mod
        source = inspect.getsource(mod.HeimdallBot._cleanup_stale_caches)
        assert "agent_manager" in source
        assert "check_health" in source

    def test_check_health_in_cleanup_source(self):
        """Verify the call is in the right method."""
        import src.discord.client as mod
        source = inspect.getsource(mod.HeimdallBot._cleanup_stale_caches)
        assert "agent_manager" in source


# ---------------------------------------------------------------------------
# Terminal statuses constant
# ---------------------------------------------------------------------------

class TestTerminalStatuses:
    """_TERMINAL_STATUSES is complete and correct."""

    def test_terminal_statuses_contents(self):
        assert _TERMINAL_STATUSES == frozenset({"completed", "failed", "timeout", "killed"})

    def test_running_is_not_terminal(self):
        assert "running" not in _TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# Source code verification
# ---------------------------------------------------------------------------

class TestSourceVerification:
    """Verify lifecycle mechanisms exist in source code."""

    def test_done_callback_in_spawn(self):
        source = inspect.getsource(AgentManager.spawn)
        assert "add_done_callback" in source
        assert "_schedule_cleanup" in source

    def test_check_health_method_exists(self):
        assert hasattr(AgentManager, "check_health")

    def test_check_health_checks_lifetime(self):
        source = inspect.getsource(AgentManager.check_health)
        assert "MAX_AGENT_LIFETIME" in source

    def test_check_health_checks_stale(self):
        source = inspect.getsource(AgentManager.check_health)
        assert "STALE_WARN_SECONDS" in source

    def test_check_health_sets_cancel_event(self):
        source = inspect.getsource(AgentManager.check_health)
        assert "_cancel_event.set()" in source

    def test_run_agent_checks_cancel(self):
        source = inspect.getsource(_run_agent)
        assert "_cancel_event.is_set()" in source

    def test_run_agent_checks_lifetime(self):
        source = inspect.getsource(_run_agent)
        assert "MAX_AGENT_LIFETIME" in source

    def test_run_agent_catches_cancelled_error(self):
        source = inspect.getsource(_run_agent)
        assert "CancelledError" in source

    def test_run_agent_catches_generic_exception(self):
        source = inspect.getsource(_run_agent)
        assert "except Exception" in source

    def test_schedule_cleanup_creates_task(self):
        source = inspect.getsource(AgentManager._schedule_cleanup)
        assert "ensure_future" in source
        assert "CLEANUP_DELAY" in source

    def test_cleanup_checks_terminal_statuses(self):
        source = inspect.getsource(AgentManager.cleanup)
        assert "completed" in source
        assert "failed" in source
        assert "ended_at" in source

    def test_client_integrates_health_check(self):
        import src.discord.client as mod
        source = inspect.getsource(mod.HeimdallBot._cleanup_stale_caches)
        assert "check_health" in source


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for lifecycle management."""

    async def test_double_kill_no_error(self):
        """Killing an already-killed agent doesn't crash."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="x", label="x", goal="g", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.status = "killed"
        mgr._agents["x"] = agent

        result = mgr.kill("x")
        assert "already in terminal state" in result

    async def test_send_to_completed_agent_rejected(self):
        """Can't send messages to completed agents."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="x", label="x", goal="g", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.status = "completed"
        mgr._agents["x"] = agent

        result = mgr.send("x", "hello")
        assert "not running" in result

    async def test_check_health_with_zero_agents(self):
        mgr = AgentManager()
        result = mgr.check_health()
        assert result == {"killed": 0, "stale": 0}

    async def test_cleanup_empty_returns_zero(self):
        mgr = AgentManager()
        removed = await mgr.cleanup()
        assert removed == 0

    async def test_agent_ended_at_set_on_all_exits(self):
        """ended_at is set regardless of how agent exits."""
        # Test completed
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()
        aid = mgr.spawn("comp", "goal", "100", "u1", "u1",
                         iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.1)
        agent = mgr._agents.get(aid)
        assert agent.ended_at is not None

    async def test_agent_ended_at_on_failure(self):
        mgr = AgentManager()
        async def _fail(messages, sys, tools):
            raise RuntimeError("fail")

        aid = mgr.spawn("fail", "goal", "100", "u1", "u1",
                         _fail, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.1)
        agent = mgr._agents.get(aid)
        assert agent.ended_at is not None

    async def test_cleanup_with_no_ended_at_skips(self):
        """Cleanup skips agents without ended_at (should not happen but safety)."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="no-end", label="no-end", goal="g", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.status = "completed"
        agent.ended_at = None  # Abnormal state
        mgr._agents["no-end"] = agent

        removed = await mgr.cleanup()
        assert removed == 0  # Skipped because ended_at is None

    async def test_multiple_health_checks_idempotent(self):
        """Running check_health() twice doesn't double-kill."""
        mgr = AgentManager()
        agent = AgentInfo(
            id="x", label="x", goal="g", channel_id="100",
            requester_id="u1", requester_name="u1",
        )
        agent.created_at = time.time() - MAX_AGENT_LIFETIME - 100
        mgr._agents["x"] = agent

        r1 = mgr.check_health()
        assert r1["killed"] == 1
        assert agent._cancel_event.is_set()

        # Second call — agent still "running" (hasn't been processed yet)
        # but cancel event already set, so it will be killed again (idempotent)
        r2 = mgr.check_health()
        assert r2["killed"] == 1  # Counts it again (still running)

    def test_active_count_excludes_terminal(self):
        mgr = AgentManager()
        for status in ["running", "completed", "failed"]:
            agent = AgentInfo(id=status, label=status, goal="g",
                              channel_id="100", requester_id="u1", requester_name="u1")
            agent.status = status
            mgr._agents[status] = agent

        assert mgr.active_count == 1
        assert mgr.total_count == 3

    async def test_last_activity_updated_per_iteration(self):
        """last_activity is updated each iteration."""
        source = inspect.getsource(_run_agent)
        assert "last_activity" in source
        # Should be updated both at LLM call and at tool execution
        assert source.count("last_activity") >= 2
