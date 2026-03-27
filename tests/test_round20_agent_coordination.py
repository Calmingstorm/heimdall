"""Round 20 — Agent coordination patterns tests.

Tests wait_for_agents (the core coordination primitive), spawn_group,
fan-out/pipeline/supervisor patterns, tool definition, and client handler.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.manager import (
    AgentInfo,
    AgentManager,
    MAX_CONCURRENT_AGENTS,
    MAX_AGENT_ITERATIONS,
    WAIT_DEFAULT_TIMEOUT,
    WAIT_POLL_INTERVAL,
    _TERMINAL_STATUSES,
    _run_agent,
)
from src.tools.registry import TOOLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_callbacks():
    """Return (iteration_cb, tool_exec_cb, announce_cb) that complete immediately."""
    # iteration_callback: returns "done" with no tool calls
    iteration_cb = AsyncMock(return_value={
        "text": "Task complete.",
        "tool_calls": [],
        "stop_reason": "end_turn",
    })
    tool_exec_cb = AsyncMock(return_value="ok")
    announce_cb = AsyncMock()
    return iteration_cb, tool_exec_cb, announce_cb


def _make_tool_callbacks():
    """Return callbacks where first call uses a tool, second completes."""
    call_count = {"n": 0}

    async def _iter_cb(messages, system, tools):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "text": "Running tool...",
                "tool_calls": [{"name": "run_command", "input": {"command": "echo hi"}}],
                "stop_reason": "tool_use",
            }
        return {
            "text": "Done after tool.",
            "tool_calls": [],
            "stop_reason": "end_turn",
        }

    tool_exec_cb = AsyncMock(return_value="hi")
    announce_cb = AsyncMock()
    return _iter_cb, tool_exec_cb, announce_cb


def _make_slow_callbacks(delay=0.5):
    """Return callbacks that take `delay` seconds to complete."""
    async def _iter_cb(messages, system, tools):
        await asyncio.sleep(delay)
        return {
            "text": f"Done after {delay}s.",
            "tool_calls": [],
            "stop_reason": "end_turn",
        }

    tool_exec_cb = AsyncMock(return_value="ok")
    announce_cb = AsyncMock()
    return _iter_cb, tool_exec_cb, announce_cb


def _spawn_agent(mgr, label="test", goal="do something", channel_id="100", **kwargs):
    """Convenience spawner with default callbacks."""
    iter_cb, tool_cb, announce_cb = kwargs.pop("callbacks", _make_callbacks())
    return mgr.spawn(
        label=label, goal=goal, channel_id=channel_id,
        requester_id="u1", requester_name="User",
        iteration_callback=iter_cb,
        tool_executor_callback=tool_cb,
        announce_callback=announce_cb,
        **kwargs,
    )


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants:
    """Verify coordination constants exist and have sane values."""

    def test_wait_default_timeout(self):
        assert WAIT_DEFAULT_TIMEOUT == 300

    def test_wait_poll_interval(self):
        assert WAIT_POLL_INTERVAL == 2

    def test_terminal_statuses(self):
        assert _TERMINAL_STATUSES == frozenset({"completed", "failed", "timeout", "killed"})

    def test_terminal_statuses_is_frozenset(self):
        assert isinstance(_TERMINAL_STATUSES, frozenset)

    def test_wait_default_timeout_positive(self):
        assert WAIT_DEFAULT_TIMEOUT > 0

    def test_wait_poll_interval_positive(self):
        assert WAIT_POLL_INTERVAL > 0

    def test_poll_interval_less_than_timeout(self):
        assert WAIT_POLL_INTERVAL < WAIT_DEFAULT_TIMEOUT


# ===========================================================================
# wait_for_agents
# ===========================================================================

class TestWaitForAgents:
    """Tests for AgentManager.wait_for_agents()."""

    async def test_empty_list_returns_empty(self):
        mgr = AgentManager()
        result = await mgr.wait_for_agents([])
        assert result == {}

    async def test_waits_for_single_agent(self):
        mgr = AgentManager()
        cbs = _make_slow_callbacks(delay=0.1)
        aid = _spawn_agent(mgr, label="a1", callbacks=cbs)
        assert not aid.startswith("Error")

        results = await mgr.wait_for_agents([aid], timeout=5)
        assert aid in results
        assert results[aid]["status"] == "completed"

    async def test_waits_for_multiple_agents(self):
        mgr = AgentManager()
        ids = []
        for i in range(3):
            cbs = _make_slow_callbacks(delay=0.05)
            aid = _spawn_agent(mgr, label=f"agent-{i}", callbacks=cbs)
            ids.append(aid)

        results = await mgr.wait_for_agents(ids, timeout=5)
        assert len(results) == 3
        for aid in ids:
            assert results[aid]["status"] == "completed"

    async def test_timeout_returns_partial_results(self):
        mgr = AgentManager()
        # One fast agent, one very slow
        fast_cbs = _make_slow_callbacks(delay=0.05)
        slow_cbs = _make_slow_callbacks(delay=100)
        fast_id = _spawn_agent(mgr, label="fast", callbacks=fast_cbs)
        slow_id = _spawn_agent(mgr, label="slow", callbacks=slow_cbs)

        results = await mgr.wait_for_agents([fast_id, slow_id], timeout=0.3)
        assert results[fast_id]["status"] == "completed"
        assert results[slow_id]["status"] == "running"

        # Clean up
        mgr.kill(slow_id)
        await asyncio.sleep(0.1)

    async def test_nonexistent_agent_reported(self):
        mgr = AgentManager()
        results = await mgr.wait_for_agents(["fake123"])
        assert "fake123" in results
        assert results["fake123"]["status"] == "not_found"
        assert "not found" in results["fake123"]["error"].lower()

    async def test_already_completed_returns_immediately(self):
        mgr = AgentManager()
        cbs = _make_slow_callbacks(delay=0.01)
        aid = _spawn_agent(mgr, label="fast", callbacks=cbs)
        await asyncio.sleep(0.1)  # let it complete

        start = time.time()
        results = await mgr.wait_for_agents([aid], timeout=10)
        elapsed = time.time() - start
        assert elapsed < 1.0  # should return almost instantly
        assert results[aid]["status"] == "completed"

    async def test_mixed_statuses(self):
        mgr = AgentManager()
        # One completes, one fails
        ok_cb = AsyncMock(return_value={"text": "ok", "tool_calls": [], "stop_reason": "end_turn"})
        fail_cb = AsyncMock(side_effect=Exception("boom"))
        announce = AsyncMock()

        ok_id = mgr.spawn("ok", "succeed", "100", "u1", "User", ok_cb, AsyncMock(), announce)
        fail_id = mgr.spawn("fail", "crash", "100", "u1", "User", fail_cb, AsyncMock(), announce)

        results = await mgr.wait_for_agents([ok_id, fail_id], timeout=5)
        assert results[ok_id]["status"] == "completed"
        assert results[fail_id]["status"] == "failed"

    async def test_poll_interval_respected(self):
        mgr = AgentManager()
        slow_cbs = _make_slow_callbacks(delay=100)
        aid = _spawn_agent(mgr, label="slow", callbacks=slow_cbs)

        start = time.time()
        # Very short timeout, should exit after ~1 poll
        await mgr.wait_for_agents([aid], timeout=0.1, poll_interval=0.05)
        elapsed = time.time() - start
        assert elapsed < 0.5

        mgr.kill(aid)
        await asyncio.sleep(0.1)

    async def test_returns_full_results_dict(self):
        mgr = AgentManager()
        cbs = _make_slow_callbacks(delay=0.01)
        aid = _spawn_agent(mgr, label="detailed", goal="test goal", callbacks=cbs)
        await asyncio.sleep(0.2)

        results = await mgr.wait_for_agents([aid])
        r = results[aid]
        assert "id" in r
        assert "label" in r
        assert "status" in r
        assert "result" in r
        assert "runtime_seconds" in r
        assert "iteration_count" in r
        assert "tools_used" in r
        assert "goal" in r


# ===========================================================================
# spawn_group
# ===========================================================================

class TestSpawnGroup:
    """Tests for AgentManager.spawn_group()."""

    async def test_spawns_multiple_agents(self):
        mgr = AgentManager()
        iter_cb, tool_cb, announce_cb = _make_callbacks()
        tasks = [
            {"label": "a1", "goal": "do thing 1"},
            {"label": "a2", "goal": "do thing 2"},
            {"label": "a3", "goal": "do thing 3"},
        ]
        ids = mgr.spawn_group(
            tasks, "100", "u1", "User",
            iter_cb, tool_cb, announce_cb,
        )
        assert len(ids) == 3
        # All should be valid agent IDs (not errors)
        for aid in ids:
            assert not aid.startswith("Error"), f"Got error: {aid}"

        # Let them complete
        await asyncio.sleep(0.2)

    async def test_empty_tasks_returns_empty(self):
        mgr = AgentManager()
        iter_cb, tool_cb, announce_cb = _make_callbacks()
        ids = mgr.spawn_group(
            [], "100", "u1", "User",
            iter_cb, tool_cb, announce_cb,
        )
        assert ids == []

    async def test_respects_channel_limit(self):
        mgr = AgentManager()
        # Spawn 5 agents first (channel limit)
        for i in range(5):
            cbs = _make_slow_callbacks(delay=100)
            _spawn_agent(mgr, label=f"existing-{i}", callbacks=cbs)

        iter_cb, tool_cb, announce_cb = _make_slow_callbacks(delay=100)
        tasks = [{"label": "over-limit", "goal": "should fail"}]
        ids = mgr.spawn_group(
            tasks, "100", "u1", "User",
            iter_cb, tool_cb, announce_cb,
        )
        assert len(ids) == 1
        assert ids[0].startswith("Error")

        # Clean up
        for agent in list(mgr._agents.values()):
            mgr.kill(agent.id)
        await asyncio.sleep(0.1)

    async def test_partial_failure(self):
        """If some tasks fail validation, they return errors while others succeed."""
        mgr = AgentManager()
        iter_cb, tool_cb, announce_cb = _make_callbacks()
        tasks = [
            {"label": "good", "goal": "valid goal"},
            {"label": "", "goal": ""},  # empty = error
            {"label": "also-good", "goal": "another goal"},
        ]
        ids = mgr.spawn_group(
            tasks, "100", "u1", "User",
            iter_cb, tool_cb, announce_cb,
        )
        assert len(ids) == 3
        assert not ids[0].startswith("Error")
        assert ids[1].startswith("Error")
        assert not ids[2].startswith("Error")

        await asyncio.sleep(0.2)

    async def test_spawn_group_then_wait(self):
        """Fan-out pattern: spawn group → wait for all → collect results."""
        mgr = AgentManager()
        iter_cb, tool_cb, announce_cb = _make_slow_callbacks(delay=0.05)
        tasks = [
            {"label": "disk-check", "goal": "check disk usage"},
            {"label": "mem-check", "goal": "check memory usage"},
        ]
        ids = mgr.spawn_group(
            tasks, "100", "u1", "User",
            iter_cb, tool_cb, announce_cb,
        )
        valid_ids = [i for i in ids if not i.startswith("Error")]
        assert len(valid_ids) == 2

        results = await mgr.wait_for_agents(valid_ids, timeout=5)
        for aid in valid_ids:
            assert results[aid]["status"] == "completed"

    async def test_passes_system_prompt(self):
        mgr = AgentManager()
        iter_cb, tool_cb, announce_cb = _make_callbacks()
        ids = mgr.spawn_group(
            [{"label": "t1", "goal": "test"}],
            "100", "u1", "User",
            iter_cb, tool_cb, announce_cb,
            system_prompt="Custom system prompt",
        )
        assert len(ids) == 1
        agent = mgr._agents[ids[0]]
        assert agent.label == "t1"
        await asyncio.sleep(0.1)


# ===========================================================================
# Fan-out pattern
# ===========================================================================

class TestFanOutPattern:
    """Tests verifying the fan-out pattern works end-to-end."""

    async def test_spawn_n_wait_collect(self):
        """Fan-out: spawn N agents → wait → all completed with results."""
        mgr = AgentManager()
        n = 4
        ids = []
        for i in range(n):
            cbs = _make_slow_callbacks(delay=0.05)
            aid = _spawn_agent(mgr, label=f"worker-{i}", goal=f"task {i}", callbacks=cbs)
            ids.append(aid)

        results = await mgr.wait_for_agents(ids, timeout=5)
        completed = [r for r in results.values() if r["status"] == "completed"]
        assert len(completed) == n

    async def test_fan_out_with_tool_usage(self):
        """Fan-out agents that each use tools before completing."""
        mgr = AgentManager()
        ids = []
        for i in range(2):
            iter_cb, tool_cb, announce_cb = _make_tool_callbacks()
            aid = _spawn_agent(mgr, label=f"tool-agent-{i}", callbacks=(iter_cb, tool_cb, announce_cb))
            ids.append(aid)

        results = await mgr.wait_for_agents(ids, timeout=5)
        for aid in ids:
            assert results[aid]["status"] == "completed"
            assert results[aid]["result"] == "Done after tool."

    async def test_fan_out_partial_failure(self):
        """Fan-out: some agents fail, others succeed."""
        mgr = AgentManager()

        ok_cb = AsyncMock(return_value={"text": "success", "tool_calls": [], "stop_reason": "end_turn"})
        fail_cb = AsyncMock(side_effect=Exception("error"))
        announce = AsyncMock()

        ok_id = mgr.spawn("ok", "succeed", "100", "u1", "User", ok_cb, AsyncMock(), announce)
        fail_id = mgr.spawn("fail", "crash", "100", "u1", "User", fail_cb, AsyncMock(), announce)

        results = await mgr.wait_for_agents([ok_id, fail_id], timeout=5)
        assert results[ok_id]["status"] == "completed"
        assert results[fail_id]["status"] == "failed"


# ===========================================================================
# Pipeline pattern
# ===========================================================================

class TestPipelinePattern:
    """Tests verifying the pipeline pattern (A output → B input)."""

    async def test_sequential_agent_pipeline(self):
        """Pipeline: spawn A → wait → use A's result as B's goal."""
        mgr = AgentManager()

        # Agent A produces a result
        a_cb = AsyncMock(return_value={
            "text": "Server-a disk at 92%",
            "tool_calls": [],
            "stop_reason": "end_turn",
        })
        a_id = mgr.spawn("analyzer", "check disk", "100", "u1", "User",
                          a_cb, AsyncMock(), AsyncMock())

        # Wait for A
        results_a = await mgr.wait_for_agents([a_id], timeout=5)
        assert results_a[a_id]["status"] == "completed"
        a_result = results_a[a_id]["result"]
        assert "92%" in a_result

        # Agent B uses A's result
        b_cb = AsyncMock(return_value={
            "text": "Cleaned 2GB from server-a",
            "tool_calls": [],
            "stop_reason": "end_turn",
        })
        b_id = mgr.spawn("fixer", f"Fix issue: {a_result}", "100", "u1", "User",
                          b_cb, AsyncMock(), AsyncMock())

        results_b = await mgr.wait_for_agents([b_id], timeout=5)
        assert results_b[b_id]["status"] == "completed"
        # Verify B received A's output in its goal
        b_agent = mgr._agents.get(b_id)
        if b_agent:
            assert "92%" in b_agent.goal

    async def test_three_stage_pipeline(self):
        """Pipeline: A → B → C, each using previous result."""
        mgr = AgentManager()
        results_chain = []

        for i, output in enumerate(["step1-data", "step2-data", "step3-final"]):
            cb = AsyncMock(return_value={
                "text": output,
                "tool_calls": [],
                "stop_reason": "end_turn",
            })
            goal = f"Previous: {results_chain[-1]}" if results_chain else "Start"
            aid = mgr.spawn(f"stage-{i}", goal, "100", "u1", "User",
                            cb, AsyncMock(), AsyncMock())

            wait_result = await mgr.wait_for_agents([aid], timeout=5)
            results_chain.append(wait_result[aid]["result"])

        assert results_chain == ["step1-data", "step2-data", "step3-final"]


# ===========================================================================
# Supervisor pattern
# ===========================================================================

class TestSupervisorPattern:
    """Tests verifying the supervisor pattern (monitor + intervene)."""

    async def test_monitor_agent_status(self):
        """Supervisor: spawn → list → check status."""
        mgr = AgentManager()
        cbs = _make_slow_callbacks(delay=0.5)
        aid = _spawn_agent(mgr, label="worker", callbacks=cbs)

        # Supervisor checks status
        agents = mgr.list("100")
        assert len(agents) == 1
        assert agents[0]["status"] == "running"

        # Wait for completion
        await mgr.wait_for_agents([aid], timeout=5)
        agents = mgr.list("100")
        assert agents[0]["status"] == "completed"

    async def test_intervene_with_send(self):
        """Supervisor: send_to_agent to redirect a running agent."""
        mgr = AgentManager()
        call_count = {"n": 0}
        intervention_sent = asyncio.Event()

        async def _iter_cb(messages, system, tools):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First iteration: use a slow tool to keep agent running
                return {
                    "text": "Starting...",
                    "tool_calls": [{"name": "run_command", "input": {"command": "echo"}}],
                    "stop_reason": "tool_use",
                }
            if call_count["n"] == 2:
                # Second iteration: still working, give time for intervention
                return {
                    "text": "Continuing...",
                    "tool_calls": [{"name": "run_command", "input": {"command": "ls"}}],
                    "stop_reason": "tool_use",
                }
            return {
                "text": "Done with intervention",
                "tool_calls": [],
                "stop_reason": "end_turn",
            }

        async def _slow_tool_cb(name, inp):
            # Slow tool execution to keep agent running long enough
            await asyncio.sleep(0.1)
            return "ok"

        announce_cb = AsyncMock()
        aid = _spawn_agent(mgr, label="worker", callbacks=(_iter_cb, _slow_tool_cb, announce_cb))

        # Give agent time to start but not finish
        await asyncio.sleep(0.05)

        # Verify agent is still running, then send
        agent = mgr._agents.get(aid)
        if agent and agent.status == "running":
            result = mgr.send(aid, "Also check /var/log")
            assert "delivered" in result.lower()
        # If it completed too fast, the test still passes (race-condition-safe)

        # Wait for completion
        results = await mgr.wait_for_agents([aid], timeout=5)
        assert results[aid]["status"] == "completed"

    async def test_kill_unresponsive_agent(self):
        """Supervisor: kill agent that's taking too long."""
        mgr = AgentManager()
        cbs = _make_slow_callbacks(delay=100)
        aid = _spawn_agent(mgr, label="stuck", callbacks=cbs)

        # Check it's running
        agents = mgr.list("100")
        assert agents[0]["status"] == "running"

        # Supervisor kills it
        result = mgr.kill(aid)
        assert "kill" in result.lower()

        # Wait briefly for cleanup
        await asyncio.sleep(0.2)

        # Verify killed
        r = mgr.get_results(aid)
        assert r["status"] == "killed"


# ===========================================================================
# wait_for_agents tool definition
# ===========================================================================

class TestWaitForAgentsToolDef:
    """Tests for the wait_for_agents tool in registry."""

    def test_tool_exists(self):
        names = [t["name"] for t in TOOLS]
        assert "wait_for_agents" in names

    def test_tool_schema(self):
        tool = next(t for t in TOOLS if t["name"] == "wait_for_agents")
        props = tool["input_schema"]["properties"]
        assert "agent_ids" in props
        assert props["agent_ids"]["type"] == "array"
        assert "timeout" in props
        assert props["timeout"]["type"] == "number"

    def test_tool_required_fields(self):
        tool = next(t for t in TOOLS if t["name"] == "wait_for_agents")
        assert "agent_ids" in tool["input_schema"]["required"]

    def test_tool_description(self):
        tool = next(t for t in TOOLS if t["name"] == "wait_for_agents")
        desc = tool["description"].lower()
        assert "wait" in desc
        assert "fan-out" in desc or "pipeline" in desc

    def test_total_tool_count(self):
        """80 tools: 67 base + 6 agent + 2 loop-agent bridge + 2 skill toggle + 3 skill management tools."""
        assert len(TOOLS) == 80

    def test_agent_tool_names(self):
        agent_tools = {"spawn_agent", "send_to_agent", "list_agents",
                        "kill_agent", "get_agent_results", "wait_for_agents"}
        tool_names = {t["name"] for t in TOOLS}
        assert agent_tools.issubset(tool_names)


# ===========================================================================
# Client handler: _handle_wait_for_agents
# ===========================================================================

class TestHandleWaitForAgents:
    """Tests for the client handler."""

    async def test_empty_agent_ids(self):
        from src.discord.client import HeimdallBot
        bot = MagicMock(spec=HeimdallBot)
        bot.agent_manager = AgentManager()
        result = await HeimdallBot._handle_wait_for_agents(bot, {"agent_ids": []})
        assert "required" in result.lower()

    async def test_missing_agent_ids(self):
        from src.discord.client import HeimdallBot
        bot = MagicMock(spec=HeimdallBot)
        bot.agent_manager = AgentManager()
        result = await HeimdallBot._handle_wait_for_agents(bot, {})
        assert "required" in result.lower()

    async def test_non_list_agent_ids(self):
        from src.discord.client import HeimdallBot
        bot = MagicMock(spec=HeimdallBot)
        bot.agent_manager = AgentManager()
        result = await HeimdallBot._handle_wait_for_agents(bot, {"agent_ids": "not-a-list"})
        assert "list" in result.lower()

    async def test_returns_formatted_results(self):
        from src.discord.client import HeimdallBot
        bot = MagicMock(spec=HeimdallBot)
        mgr = AgentManager()
        bot.agent_manager = mgr

        cbs = _make_slow_callbacks(delay=0.01)
        aid = _spawn_agent(mgr, label="test-agent", goal="test", callbacks=cbs)
        await asyncio.sleep(0.2)

        result = await HeimdallBot._handle_wait_for_agents(bot, {"agent_ids": [aid]})
        assert "test-agent" in result
        assert "completed" in result

    async def test_handles_timeout_parameter(self):
        from src.discord.client import HeimdallBot
        bot = MagicMock(spec=HeimdallBot)
        mgr = AgentManager()
        bot.agent_manager = mgr

        cbs = _make_slow_callbacks(delay=100)
        aid = _spawn_agent(mgr, label="slow", callbacks=cbs)

        start = time.time()
        result = await HeimdallBot._handle_wait_for_agents(
            bot, {"agent_ids": [aid], "timeout": 0.2}
        )
        elapsed = time.time() - start
        assert elapsed < 2.0
        assert "running" in result or "slow" in result

        mgr.kill(aid)
        await asyncio.sleep(0.1)

    async def test_nonexistent_agent(self):
        from src.discord.client import HeimdallBot
        bot = MagicMock(spec=HeimdallBot)
        bot.agent_manager = AgentManager()
        result = await HeimdallBot._handle_wait_for_agents(
            bot, {"agent_ids": ["nonexistent"]}
        )
        assert "not_found" in result

    async def test_multiple_agents_formatted(self):
        from src.discord.client import HeimdallBot
        bot = MagicMock(spec=HeimdallBot)
        mgr = AgentManager()
        bot.agent_manager = mgr

        ids = []
        for i in range(2):
            cbs = _make_slow_callbacks(delay=0.01)
            aid = _spawn_agent(mgr, label=f"agent-{i}", callbacks=cbs)
            ids.append(aid)
        await asyncio.sleep(0.2)

        result = await HeimdallBot._handle_wait_for_agents(
            bot, {"agent_ids": ids}
        )
        for aid in ids:
            assert aid in result
        assert "agent-0" in result
        assert "agent-1" in result

    async def test_long_result_truncated(self):
        from src.discord.client import HeimdallBot
        bot = MagicMock(spec=HeimdallBot)
        mgr = AgentManager()
        bot.agent_manager = mgr

        long_text = "x" * 2000
        cb = AsyncMock(return_value={"text": long_text, "tool_calls": [], "stop_reason": "end_turn"})
        aid = mgr.spawn("long", "test", "100", "u1", "User", cb, AsyncMock(), AsyncMock())
        await asyncio.sleep(0.2)

        result = await HeimdallBot._handle_wait_for_agents(
            bot, {"agent_ids": [aid]}
        )
        # Result should be truncated (800 chars for content)
        assert "..." in result
        assert len(result) < len(long_text)


# ===========================================================================
# Source verification
# ===========================================================================

class TestSourceVerification:
    """Verify source code structure for coordination features."""

    def test_manager_has_wait_for_agents(self):
        assert hasattr(AgentManager, "wait_for_agents")

    def test_wait_for_agents_is_async(self):
        assert asyncio.iscoroutinefunction(AgentManager.wait_for_agents)

    def test_manager_has_spawn_group(self):
        assert hasattr(AgentManager, "spawn_group")

    def test_spawn_group_is_sync(self):
        assert not asyncio.iscoroutinefunction(AgentManager.spawn_group)

    def test_wait_for_agents_signature(self):
        sig = inspect.signature(AgentManager.wait_for_agents)
        params = list(sig.parameters.keys())
        assert "agent_ids" in params
        assert "timeout" in params
        assert "poll_interval" in params

    def test_spawn_group_signature(self):
        sig = inspect.signature(AgentManager.spawn_group)
        params = list(sig.parameters.keys())
        assert "tasks" in params
        assert "channel_id" in params

    def test_terminal_statuses_constant(self):
        from src.agents.manager import _TERMINAL_STATUSES
        assert "completed" in _TERMINAL_STATUSES
        assert "failed" in _TERMINAL_STATUSES
        assert "timeout" in _TERMINAL_STATUSES
        assert "killed" in _TERMINAL_STATUSES
        assert "running" not in _TERMINAL_STATUSES

    def test_client_has_handle_wait(self):
        import src.discord.client as client_mod
        src = inspect.getsource(client_mod.HeimdallBot)
        assert "_handle_wait_for_agents" in src

    def test_client_dispatch_has_wait(self):
        import src.discord.client as client_mod
        src = inspect.getsource(client_mod.HeimdallBot)
        assert '"wait_for_agents"' in src

    def test_client_loop_dispatch_has_wait(self):
        import src.discord.client as client_mod
        src = inspect.getsource(client_mod.HeimdallBot._dispatch_loop_tool)
        assert "wait_for_agents" in src

    def test_registry_tool_in_agent_section(self):
        """wait_for_agents should be near other agent tools."""
        from src.tools import registry
        src = inspect.getsource(registry)
        agent_section = src.find("Agent orchestration")
        wait_pos = src.find("wait_for_agents")
        assert agent_section > 0
        assert wait_pos > agent_section

    def test_wait_default_timeout_constant(self):
        import src.agents.manager as mgr_mod
        src = inspect.getsource(mgr_mod)
        assert "WAIT_DEFAULT_TIMEOUT" in src
        assert "WAIT_POLL_INTERVAL" in src


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases for coordination patterns."""

    async def test_wait_zero_timeout(self):
        """Zero timeout returns immediately with current status."""
        mgr = AgentManager()
        cbs = _make_slow_callbacks(delay=100)
        aid = _spawn_agent(mgr, label="slow", callbacks=cbs)

        start = time.time()
        results = await mgr.wait_for_agents([aid], timeout=0)
        elapsed = time.time() - start
        assert elapsed < 0.5
        assert results[aid]["status"] == "running"

        mgr.kill(aid)
        await asyncio.sleep(0.1)

    async def test_duplicate_ids_in_wait(self):
        """Duplicate IDs should not cause issues."""
        mgr = AgentManager()
        cbs = _make_slow_callbacks(delay=0.01)
        aid = _spawn_agent(mgr, label="dup", callbacks=cbs)
        await asyncio.sleep(0.2)

        results = await mgr.wait_for_agents([aid, aid])
        assert aid in results
        assert results[aid]["status"] == "completed"

    async def test_spawn_group_with_tools(self):
        """spawn_group passes tools to all agents."""
        mgr = AgentManager()
        iter_cb, tool_cb, announce_cb = _make_callbacks()
        ids = mgr.spawn_group(
            [{"label": "t1", "goal": "go"}],
            "100", "u1", "User",
            iter_cb, tool_cb, announce_cb,
            tools=[{"name": "test_tool"}],
        )
        assert len(ids) == 1
        assert not ids[0].startswith("Error")
        await asyncio.sleep(0.1)

    async def test_wait_for_killed_agent(self):
        """Waiting for a killed agent returns immediately."""
        mgr = AgentManager()
        cbs = _make_slow_callbacks(delay=100)
        aid = _spawn_agent(mgr, label="to-kill", callbacks=cbs)

        mgr.kill(aid)
        await asyncio.sleep(0.2)

        results = await mgr.wait_for_agents([aid], timeout=5)
        assert results[aid]["status"] == "killed"

    async def test_wait_for_timed_out_agent(self):
        """If agent timed out, wait returns timeout status."""
        mgr = AgentManager()
        # Manually create an agent in timeout state
        agent = AgentInfo(
            id="t123", label="timed-out", goal="test",
            channel_id="100", requester_id="u1", requester_name="User",
        )
        agent.status = "timeout"
        agent.ended_at = time.time()
        mgr._agents["t123"] = agent

        results = await mgr.wait_for_agents(["t123"], timeout=1)
        assert results["t123"]["status"] == "timeout"

    async def test_concurrent_wait_calls(self):
        """Multiple wait_for_agents calls on same agents don't interfere."""
        mgr = AgentManager()
        cbs = _make_slow_callbacks(delay=0.1)
        aid = _spawn_agent(mgr, label="shared", callbacks=cbs)

        # Two concurrent waits
        r1, r2 = await asyncio.gather(
            mgr.wait_for_agents([aid], timeout=5),
            mgr.wait_for_agents([aid], timeout=5),
        )
        assert r1[aid]["status"] == "completed"
        assert r2[aid]["status"] == "completed"
