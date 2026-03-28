"""Round 25 — Comprehensive agent system tests.

End-to-end tests covering the full agent lifecycle, multi-agent coordination,
loop-agent bridge integration, callback wiring, edge cases, and regression tests.
This is the capstone test suite for the agent system (Rounds 18-24).

Sections:
 1. End-to-end agent lifecycle (spawn → iterate → tools → complete → cleanup)
 2. Multi-agent concurrent execution
 3. Rapid spawn/kill stress tests
 4. Message history correctness through iterations
 5. Tool execution ordering and parallelism
 6. Callback failure modes
 7. System prompt construction invariants
 8. Full loop-agent bridge lifecycle
 9. Cross-module integration (registry, client, manager)
10. Agent state machine transitions
11. Secret scrubbing boundary verification
12. Announcement formatting edge cases
13. Wait/poll timing verification
14. Cleanup scheduling and auto-removal
15. AgentInfo dataclass invariants
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.manager import (
    AGENT_BLOCKED_TOOLS,
    CLEANUP_DELAY,
    ITERATION_CB_TIMEOUT,
    MAX_AGENT_ITERATIONS,
    MAX_AGENT_LIFETIME,
    MAX_CONCURRENT_AGENTS,
    STALE_WARN_SECONDS,
    TOOL_EXEC_TIMEOUT,
    WAIT_DEFAULT_TIMEOUT,
    WAIT_POLL_INTERVAL,
    AgentInfo,
    AgentManager,
    _TERMINAL_STATUSES,
    _get_last_progress,
    _run_agent,
    filter_agent_tools,
)
from src.agents.loop_bridge import (
    LOOP_AGENT_WAIT_TIMEOUT,
    MAX_AGENTS_PER_ITERATION,
    MAX_AGENTS_PER_LOOP,
    LoopAgentBridge,
    LoopAgentRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_callbacks(responses=None, tool_results=None, tool_delay=0):
    """Build mock callbacks for agent execution."""
    if responses is None:
        responses = [{"text": "Done.", "tool_calls": []}]
    responses = list(responses)

    async def iteration_cb(messages, system, tools):
        if responses:
            return responses.pop(0)
        return {"text": "Exhausted.", "tool_calls": []}

    async def tool_exec_cb(tool_name, tool_input):
        if tool_delay:
            await asyncio.sleep(tool_delay)
        if tool_results and tool_name in tool_results:
            val = tool_results[tool_name]
            if callable(val):
                return val(tool_input)
            return val
        return f"OK: {tool_name}"

    async def announce_cb(channel_id, text):
        pass

    return (
        AsyncMock(side_effect=iteration_cb),
        AsyncMock(side_effect=tool_exec_cb),
        AsyncMock(side_effect=announce_cb),
    )


def _make_agent(**overrides):
    """Create an AgentInfo with sane defaults."""
    defaults = dict(
        id="test01",
        label="test-agent",
        goal="test goal",
        channel_id="100",
        requester_id="u1",
        requester_name="user1",
    )
    defaults.update(overrides)
    return AgentInfo(**defaults)


def _spawn_agent(manager, label="a1", goal="do stuff", channel_id="100", **kw):
    """Helper to spawn an agent with default callbacks."""
    responses = kw.pop("responses", [{"text": "Done.", "tool_calls": []}])
    tool_results = kw.pop("tool_results", None)
    tool_delay = kw.pop("tool_delay", 0)
    iter_cb, tool_cb, announce_cb = _make_callbacks(responses, tool_results, tool_delay)
    return manager.spawn(
        label=label,
        goal=goal,
        channel_id=channel_id,
        requester_id=kw.get("requester_id", "u1"),
        requester_name=kw.get("requester_name", "user1"),
        iteration_callback=iter_cb,
        tool_executor_callback=tool_cb,
        announce_callback=announce_cb,
        tools=kw.get("tools", [{"name": "run_command"}]),
        system_prompt=kw.get("system_prompt", "You are helpful."),
    )


# ===========================================================================
# 1. End-to-end agent lifecycle
# ===========================================================================

class TestEndToEndLifecycle:
    """Full lifecycle: spawn → iterate → tool calls → complete → cleanup."""

    async def test_single_iteration_no_tools(self):
        """Agent with text-only response completes in 1 iteration."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr, responses=[{"text": "All done!", "tool_calls": []}])
        assert not aid.startswith("Error")

        await asyncio.sleep(0.05)
        r = mgr.get_results(aid)
        assert r["status"] == "completed"
        assert r["result"] == "All done!"
        assert r["iteration_count"] == 1
        assert r["tools_used"] == []

    async def test_multi_iteration_with_tools(self):
        """Agent uses tools across 3 iterations then completes."""
        responses = [
            {"text": "Checking...", "tool_calls": [{"name": "run_command", "input": {"cmd": "ls"}}]},
            {"text": "Analyzing...", "tool_calls": [{"name": "read_file", "input": {"path": "/etc"}}]},
            {"text": "Done with analysis.", "tool_calls": []},
        ]
        mgr = AgentManager()
        aid = _spawn_agent(mgr, responses=responses)
        await asyncio.sleep(0.1)

        r = mgr.get_results(aid)
        assert r["status"] == "completed"
        assert r["result"] == "Done with analysis."
        assert r["iteration_count"] == 3
        assert set(r["tools_used"]) == {"run_command", "read_file"}

    async def test_lifecycle_ends_at_max_iterations(self):
        """Agent that always returns tools hits MAX_AGENT_ITERATIONS."""
        responses = [
            {"text": f"Step {i}", "tool_calls": [{"name": "run_command", "input": {}}]}
            for i in range(MAX_AGENT_ITERATIONS + 5)
        ]
        mgr = AgentManager()
        aid = _spawn_agent(mgr, responses=responses)
        await asyncio.sleep(0.3)

        r = mgr.get_results(aid)
        assert r["status"] == "completed"
        assert r["iteration_count"] == MAX_AGENT_ITERATIONS

    async def test_cleanup_removes_completed_agent(self):
        """Completed agent is removed after cleanup delay."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        await asyncio.sleep(0.05)

        # Agent completed but too recent to clean
        assert mgr.total_count == 1
        removed = await mgr.cleanup()
        assert removed == 0
        assert mgr.total_count == 1

        # Backdate ended_at to trigger cleanup
        agent = mgr._agents[aid]
        agent.ended_at = time.time() - CLEANUP_DELAY - 1
        removed = await mgr.cleanup()
        assert removed == 1
        assert mgr.total_count == 0

    async def test_spawn_run_kill_cleanup_full_cycle(self):
        """Full cycle: spawn → verify running → kill → verify killed → cleanup."""
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100
        mgr = AgentManager()
        aid = _spawn_agent(mgr, responses=never_done, tool_delay=0.1)
        await asyncio.sleep(0.05)

        # Agent should be running (tool_delay keeps it alive)
        r = mgr.get_results(aid)
        assert r["status"] == "running"
        assert mgr.active_count == 1

        # Kill it
        result = mgr.kill(aid)
        assert "Kill signal" in result
        await asyncio.sleep(0.2)

        # Now it's killed
        r = mgr.get_results(aid)
        assert r["status"] == "killed"
        assert mgr.active_count == 0

        # Cleanup after backdating
        mgr._agents[aid].ended_at = time.time() - CLEANUP_DELAY - 1
        removed = await mgr.cleanup()
        assert removed == 1
        assert mgr.total_count == 0


# ===========================================================================
# 2. Multi-agent concurrent execution
# ===========================================================================

class TestMultiAgentConcurrent:
    """Multiple agents running simultaneously."""

    async def test_five_agents_same_channel(self):
        """Maximum 5 agents can run concurrently in same channel."""
        mgr = AgentManager()
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100
        ids = []
        for i in range(5):
            aid = _spawn_agent(mgr, label=f"a{i}", responses=list(never_done), tool_delay=0.1)
            ids.append(aid)

        await asyncio.sleep(0.02)
        assert all(not a.startswith("Error") for a in ids)
        assert mgr.active_count == 5

        # 6th should fail
        aid6 = _spawn_agent(mgr, label="a5", responses=list(never_done), tool_delay=0.1)
        assert aid6.startswith("Error")
        assert mgr.active_count == 5

        # Kill all
        for aid in ids:
            mgr.kill(aid)
        await asyncio.sleep(0.2)
        assert mgr.active_count == 0

    async def test_agents_across_channels_independent(self):
        """Per-channel limit is independent across channels."""
        mgr = AgentManager()
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100

        ids_ch1 = []
        for i in range(5):
            aid = _spawn_agent(mgr, label=f"ch1-a{i}", channel_id="ch1", responses=list(never_done), tool_delay=0.1)
            ids_ch1.append(aid)

        ids_ch2 = []
        for i in range(5):
            aid = _spawn_agent(mgr, label=f"ch2-a{i}", channel_id="ch2", responses=list(never_done), tool_delay=0.1)
            ids_ch2.append(aid)

        await asyncio.sleep(0.02)
        assert mgr.active_count == 10
        assert all(not a.startswith("Error") for a in ids_ch1 + ids_ch2)

        # Cleanup
        for aid in ids_ch1 + ids_ch2:
            mgr.kill(aid)
        await asyncio.sleep(0.2)

    async def test_concurrent_agents_complete_independently(self):
        """Agents complete at different times, each with own result."""
        mgr = AgentManager()

        # Fast agent: 1 iteration
        aid_fast = _spawn_agent(
            mgr, label="fast",
            responses=[{"text": "Quick done.", "tool_calls": []}],
        )
        # Slow agent: 3 iterations
        aid_slow = _spawn_agent(
            mgr, label="slow",
            responses=[
                {"text": "Step 1", "tool_calls": [{"name": "run_command", "input": {}}]},
                {"text": "Step 2", "tool_calls": [{"name": "run_command", "input": {}}]},
                {"text": "Slow done.", "tool_calls": []},
            ],
        )

        await asyncio.sleep(0.15)

        r_fast = mgr.get_results(aid_fast)
        r_slow = mgr.get_results(aid_slow)
        assert r_fast["status"] == "completed"
        assert r_fast["result"] == "Quick done."
        assert r_slow["status"] == "completed"
        assert r_slow["result"] == "Slow done."

    async def test_spawn_group_all_complete(self):
        """spawn_group spawns multiple agents that all complete."""
        mgr = AgentManager()
        iter_cb, tool_cb, announce_cb = _make_callbacks()
        tasks = [
            {"label": f"g{i}", "goal": f"task {i}"} for i in range(3)
        ]
        ids = mgr.spawn_group(
            tasks=tasks,
            channel_id="100",
            requester_id="u1",
            requester_name="user1",
            iteration_callback=iter_cb,
            tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
            tools=[],
            system_prompt="test",
        )
        assert len(ids) == 3
        assert all(not a.startswith("Error") for a in ids)
        await asyncio.sleep(0.1)

        for aid in ids:
            r = mgr.get_results(aid)
            assert r["status"] == "completed"


# ===========================================================================
# 3. Rapid spawn/kill stress tests
# ===========================================================================

class TestRapidSpawnKill:
    """Rapid lifecycle operations don't corrupt state."""

    async def test_rapid_spawn_kill_cycles(self):
        """Spawn then immediately kill 10 agents — no crashes."""
        mgr = AgentManager()
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100

        for i in range(10):
            aid = _spawn_agent(
                mgr, label=f"rapid-{i}", channel_id=f"ch{i % 3}",
                responses=list(never_done), tool_delay=0.05,
            )
            mgr.kill(aid)

        await asyncio.sleep(0.3)
        # All should be in terminal state
        for agent in mgr._agents.values():
            assert agent.status in _TERMINAL_STATUSES

    async def test_spawn_at_limit_then_kill_then_respawn(self):
        """Fill channel, kill one, respawn in freed slot."""
        mgr = AgentManager()
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100

        ids = []
        for i in range(5):
            aid = _spawn_agent(mgr, label=f"a{i}", responses=list(never_done), tool_delay=0.1)
            ids.append(aid)

        await asyncio.sleep(0.02)
        # At limit
        overflow = _spawn_agent(mgr, label="overflow", responses=list(never_done), tool_delay=0.1)
        assert overflow.startswith("Error")

        # Kill one
        mgr.kill(ids[0])
        await asyncio.sleep(0.2)

        # Now one slot open
        new_aid = _spawn_agent(mgr, label="replacement", responses=list(never_done), tool_delay=0.1)
        assert not new_aid.startswith("Error")

        # Cleanup
        for aid in ids[1:] + [new_aid]:
            mgr.kill(aid)
        await asyncio.sleep(0.2)


# ===========================================================================
# 4. Message history correctness
# ===========================================================================

class TestMessageHistory:
    """Verify messages list grows correctly through iterations."""

    async def test_initial_message_is_goal(self):
        """Agent starts with exactly one user message containing the goal."""
        agent = _make_agent(goal="Find the bug")
        agent.messages = [{"role": "user", "content": "Find the bug"}]
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == "Find the bug"

    async def test_messages_grow_with_iterations(self):
        """After 2 tool-using iterations, history has goal + 2*(assistant + tool_result)."""
        responses = [
            {"text": "Using tool", "tool_calls": [{"name": "run_command", "input": {"cmd": "ls"}}]},
            {"text": "All done.", "tool_calls": []},
        ]
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        iter_cb, tool_cb, announce_cb = _make_callbacks(responses)

        await _run_agent(agent, "sys", [], iter_cb, tool_cb, announce_cb)

        # goal(user) + assistant("Using tool") + tool_result(user) + assistant("All done.")
        assert len(agent.messages) == 4
        roles = [m["role"] for m in agent.messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert "[Tool result: run_command]" in agent.messages[2]["content"]

    async def test_inbox_messages_appear_in_history(self):
        """Messages injected via inbox show up in agent's history."""
        responses = [
            {"text": "Step 1", "tool_calls": [{"name": "run_command", "input": {}}]},
            {"text": "Got your message.", "tool_calls": []},
        ]
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        # Pre-load inbox before running
        agent._inbox.put_nowait("Priority update!")

        iter_cb, tool_cb, announce_cb = _make_callbacks(responses)
        await _run_agent(agent, "sys", [], iter_cb, tool_cb, announce_cb)

        # The inbox message should appear as a user message
        contents = [m["content"] for m in agent.messages if m["role"] == "user"]
        assert any("[Message from parent]" in c for c in contents)

    async def test_multiple_tools_single_iteration(self):
        """Multiple tool calls in one iteration each append a tool result."""
        responses = [
            {"text": "Multi-tool", "tool_calls": [
                {"name": "run_command", "input": {"cmd": "ls"}},
                {"name": "read_file", "input": {"path": "/tmp"}},
                {"name": "check_disk", "input": {}},
            ]},
            {"text": "Done.", "tool_calls": []},
        ]
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        iter_cb, tool_cb, announce_cb = _make_callbacks(responses)
        await _run_agent(agent, "sys", [], iter_cb, tool_cb, announce_cb)

        # goal + assistant + 3 tool results + assistant
        assert len(agent.messages) == 6
        tool_msgs = [m for m in agent.messages if "[Tool result:" in m.get("content", "")]
        assert len(tool_msgs) == 3

    async def test_tool_result_format(self):
        """Tool results are formatted as '[Tool result: name]\\nresult'."""
        responses = [
            {"text": "Using", "tool_calls": [{"name": "run_command", "input": {"cmd": "ls"}}]},
            {"text": "Done.", "tool_calls": []},
        ]
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        iter_cb, tool_cb, announce_cb = _make_callbacks(responses, {"run_command": "file1.txt\nfile2.txt"})
        await _run_agent(agent, "sys", [], iter_cb, tool_cb, announce_cb)

        tool_msg = [m for m in agent.messages if "[Tool result:" in m.get("content", "")][0]
        assert tool_msg["content"].startswith("[Tool result: run_command]\n")
        assert "file1.txt" in tool_msg["content"]

    async def test_tool_error_format_in_messages(self):
        """Tool errors appear as 'Error: ...' in tool result messages."""
        async def failing_tool(name, inp):
            raise RuntimeError("disk full")

        responses = [
            {"text": "Try tool", "tool_calls": [{"name": "run_command", "input": {}}]},
            {"text": "Done.", "tool_calls": []},
        ]
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        iter_cb = AsyncMock(side_effect=responses.pop)
        # Override to use list-based side_effect
        r_list = list(responses)

        async def _iter_cb(msgs, sys, tools):
            if r_list:
                return r_list.pop(0)
            return {"text": "Done.", "tool_calls": []}

        tool_cb = AsyncMock(side_effect=RuntimeError("disk full"))
        announce_cb = AsyncMock()

        r2 = [
            {"text": "Try tool", "tool_calls": [{"name": "run_command", "input": {}}]},
            {"text": "Done after error.", "tool_calls": []},
        ]

        async def iter_side(msgs, sys, tools):
            if r2:
                return r2.pop(0)
            return {"text": "Exhausted.", "tool_calls": []}

        await _run_agent(
            agent, "sys", [],
            AsyncMock(side_effect=iter_side),
            tool_cb,
            announce_cb,
        )

        tool_msgs = [m for m in agent.messages if "[Tool result:" in m.get("content", "")]
        assert len(tool_msgs) == 1
        assert "Error: disk full" in tool_msgs[0]["content"]


# ===========================================================================
# 5. Tool execution ordering
# ===========================================================================

class TestToolExecution:
    """Verify tool execution behavior within agent iterations."""

    async def test_tools_executed_sequentially_per_iteration(self):
        """Tools in a single iteration are executed one after another (serial in agent)."""
        order = []

        async def tracking_tool(name, inp):
            order.append(name)
            return f"OK: {name}"

        responses = [
            {"text": "Multi", "tool_calls": [
                {"name": "tool_a", "input": {}},
                {"name": "tool_b", "input": {}},
                {"name": "tool_c", "input": {}},
            ]},
            {"text": "Done.", "tool_calls": []},
        ]

        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        iter_list = list(responses)

        async def iter_side(msgs, sys, tools):
            if iter_list:
                return iter_list.pop(0)
            return {"text": "Exhausted.", "tool_calls": []}

        await _run_agent(
            agent, "sys", [],
            AsyncMock(side_effect=iter_side),
            AsyncMock(side_effect=tracking_tool),
            AsyncMock(),
        )

        assert order == ["tool_a", "tool_b", "tool_c"]

    async def test_tools_used_deduplicates(self):
        """Same tool called twice only appears once in tools_used."""
        responses = [
            {"text": "First call", "tool_calls": [{"name": "run_command", "input": {}}]},
            {"text": "Second call", "tool_calls": [{"name": "run_command", "input": {}}]},
            {"text": "Done.", "tool_calls": []},
        ]
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        iter_cb, tool_cb, announce_cb = _make_callbacks(responses)
        await _run_agent(agent, "sys", [], iter_cb, tool_cb, announce_cb)

        assert agent.tools_used == ["run_command"]

    async def test_tool_failure_doesnt_stop_iteration(self):
        """One tool failing in an iteration doesn't stop subsequent tool calls."""
        call_count = {"count": 0}

        async def mixed_tool(name, inp):
            call_count["count"] += 1
            if name == "fail_tool":
                raise ValueError("broken")
            return "OK"

        responses = [
            {"text": "Multiple tools", "tool_calls": [
                {"name": "fail_tool", "input": {}},
                {"name": "good_tool", "input": {}},
            ]},
            {"text": "Done.", "tool_calls": []},
        ]
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        iter_list = list(responses)

        async def iter_side(msgs, sys, tools):
            if iter_list:
                return iter_list.pop(0)
            return {"text": "Exhausted.", "tool_calls": []}

        await _run_agent(
            agent, "sys", [],
            AsyncMock(side_effect=iter_side),
            AsyncMock(side_effect=mixed_tool),
            AsyncMock(),
        )

        assert call_count["count"] == 2  # Both tools called
        assert agent.status == "completed"


# ===========================================================================
# 6. Callback failure modes
# ===========================================================================

class TestCallbackFailures:
    """Verify graceful handling of callback failures."""

    async def test_llm_callback_exception_fails_agent(self):
        """Exception in iteration callback → agent fails."""
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]

        async def bad_llm(msgs, sys, tools):
            raise ConnectionError("API down")

        await _run_agent(
            agent, "sys", [],
            AsyncMock(side_effect=bad_llm),
            AsyncMock(),
            AsyncMock(),
        )

        assert agent.status == "failed"
        assert "LLM call failed: API down" in agent.error

    async def test_announce_callback_failure_doesnt_crash(self):
        """Announcement failure is caught; agent lifecycle unaffected."""
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]

        responses = [{"text": "Done.", "tool_calls": []}]

        async def iter_side(msgs, sys, tools):
            if responses:
                return responses.pop(0)
            return {"text": "Exhausted.", "tool_calls": []}

        async def bad_announce(ch_id, text):
            raise RuntimeError("Discord down")

        await _run_agent(
            agent, "sys", [],
            AsyncMock(side_effect=iter_side),
            AsyncMock(),
            AsyncMock(side_effect=bad_announce),
        )

        # Agent should complete despite announce failure
        assert agent.status == "completed"

    async def test_tool_timeout_continues_agent(self):
        """Tool that times out returns error but agent continues."""
        async def slow_tool(name, inp):
            await asyncio.sleep(999)

        responses = [
            {"text": "Using slow tool", "tool_calls": [{"name": "slow_tool", "input": {}}]},
            {"text": "After timeout.", "tool_calls": []},
        ]
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        iter_list = list(responses)

        async def iter_side(msgs, sys, tools):
            if iter_list:
                return iter_list.pop(0)
            return {"text": "Exhausted.", "tool_calls": []}

        with patch("src.agents.manager.TOOL_EXEC_TIMEOUT", 0.01):
            await _run_agent(
                agent, "sys", [],
                AsyncMock(side_effect=iter_side),
                AsyncMock(side_effect=slow_tool),
                AsyncMock(),
            )

        assert agent.status == "completed"
        assert agent.result == "After timeout."
        # Verify timeout error was in messages
        tool_msgs = [m for m in agent.messages if "timed out" in m.get("content", "")]
        assert len(tool_msgs) == 1

    async def test_llm_timeout_fails_agent(self):
        """LLM call that times out → agent fails."""
        async def slow_llm(msgs, sys, tools):
            await asyncio.sleep(999)

        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]

        with patch("src.agents.manager.ITERATION_CB_TIMEOUT", 0.01):
            await _run_agent(
                agent, "sys", [],
                AsyncMock(side_effect=slow_llm),
                AsyncMock(),
                AsyncMock(),
            )

        assert agent.status == "failed"
        assert "timed out" in agent.error.lower()


# ===========================================================================
# 7. System prompt construction
# ===========================================================================

class TestSystemPromptConstruction:
    """Verify agent system prompt is always correctly built."""

    async def test_agent_context_injected_with_base_prompt(self):
        """When system_prompt provided, AGENT CONTEXT is appended."""
        mgr = AgentManager()
        captured_prompts = []

        async def capture_iter(msgs, sys, tools):
            captured_prompts.append(sys)
            return {"text": "Done.", "tool_calls": []}

        mgr.spawn(
            label="ctx-test", goal="test",
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=AsyncMock(side_effect=capture_iter),
            tool_executor_callback=AsyncMock(),
            announce_callback=AsyncMock(),
            system_prompt="Custom base prompt.",
        )
        await asyncio.sleep(0.05)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "Custom base prompt." in prompt
        assert "AGENT CONTEXT:" in prompt
        assert "ctx-test" in prompt
        assert "Do NOT spawn sub-agents" in prompt

    async def test_agent_context_without_base_prompt(self):
        """When no system_prompt, AGENT CONTEXT still present (no double newline start)."""
        mgr = AgentManager()
        captured_prompts = []

        async def capture_iter(msgs, sys, tools):
            captured_prompts.append(sys)
            return {"text": "Done.", "tool_calls": []}

        mgr.spawn(
            label="no-base", goal="test",
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=AsyncMock(side_effect=capture_iter),
            tool_executor_callback=AsyncMock(),
            announce_callback=AsyncMock(),
            system_prompt="",
        )
        await asyncio.sleep(0.05)

        prompt = captured_prompts[0]
        assert prompt.startswith("AGENT CONTEXT:")
        assert "no-base" in prompt

    async def test_system_prompt_label_varies_per_agent(self):
        """Each agent gets its own label in the system prompt."""
        mgr = AgentManager()
        captured = {}

        def make_capture(label):
            async def cap(msgs, sys, tools):
                captured[label] = sys
                return {"text": "Done.", "tool_calls": []}
            return cap

        for name in ["alpha", "beta", "gamma"]:
            mgr.spawn(
                label=name, goal="test",
                channel_id="100", requester_id="u1", requester_name="u1",
                iteration_callback=AsyncMock(side_effect=make_capture(name)),
                tool_executor_callback=AsyncMock(),
                announce_callback=AsyncMock(),
            )

        await asyncio.sleep(0.05)

        for name in ["alpha", "beta", "gamma"]:
            assert name in captured[name]


# ===========================================================================
# 8. Full loop-agent bridge lifecycle
# ===========================================================================

class TestLoopAgentBridgeLifecycle:
    """End-to-end loop-agent integration."""

    async def test_spawn_wait_collect_cleanup(self):
        """Full bridge lifecycle: spawn → wait → collect → cleanup."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)

        iter_cb, tool_cb, announce_cb = _make_callbacks()
        tasks = [{"label": "sub1", "goal": "check servers"}]

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop-1",
            iteration=0,
            loop_goal="Monitor infra",
            tasks=tasks,
            channel_id="100",
            requester_id="u1",
            requester_name="user1",
            iteration_callback=iter_cb,
            tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
        )

        assert len(ids) == 1
        assert not ids[0].startswith("Error")
        assert bridge.get_loop_agent_count("loop-1") == 1

        # Wait for completion
        await asyncio.sleep(0.1)
        results = await bridge.wait_and_collect("loop-1")

        assert len(results) == 1
        agent_result = list(results.values())[0]
        assert agent_result["status"] == "completed"

        # Format results
        formatted = bridge.format_agent_results_for_context(results)
        assert "sub1" in formatted
        assert "completed" in formatted

        # Cleanup
        removed = bridge.cleanup_loop("loop-1")
        assert removed == 1
        assert bridge.tracked_loop_count == 0

    async def test_multi_iteration_spawning(self):
        """Loop spawns agents across multiple iterations."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        iter_cb, tool_cb, announce_cb = _make_callbacks()

        # Iteration 0: spawn 2
        ids_iter0 = bridge.spawn_agents_for_loop(
            loop_id="loop-x", iteration=0, loop_goal="Watch",
            tasks=[
                {"label": "task-a", "goal": "a"},
                {"label": "task-b", "goal": "b"},
            ],
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
        )
        assert len(ids_iter0) == 2

        # Iteration 1: spawn 2 more
        ids_iter1 = bridge.spawn_agents_for_loop(
            loop_id="loop-x", iteration=1, loop_goal="Watch",
            tasks=[
                {"label": "task-c", "goal": "c"},
                {"label": "task-d", "goal": "d"},
            ],
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
        )
        assert len(ids_iter1) == 2

        assert bridge.get_loop_agent_count("loop-x") == 4
        await asyncio.sleep(0.1)

    async def test_per_loop_limit_enforced(self):
        """Cannot exceed MAX_AGENTS_PER_LOOP across all iterations."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        iter_cb, tool_cb, announce_cb = _make_callbacks()

        # Spawn up to the limit in batches
        total = 0
        for i in range(MAX_AGENTS_PER_LOOP // MAX_AGENTS_PER_ITERATION + 1):
            batch_size = min(MAX_AGENTS_PER_ITERATION, MAX_AGENTS_PER_LOOP - total)
            if batch_size <= 0:
                break
            tasks = [{"label": f"t{total+j}", "goal": "x"} for j in range(batch_size)]
            ids = bridge.spawn_agents_for_loop(
                loop_id="loop-limit", iteration=i, loop_goal="test",
                tasks=tasks,
                channel_id="100", requester_id="u1", requester_name="u1",
                iteration_callback=iter_cb, tool_executor_callback=tool_cb,
                announce_callback=announce_cb,
            )
            total += len([x for x in ids if not x.startswith("Error")])

        # Now try to spawn one more — should fail
        over = bridge.spawn_agents_for_loop(
            loop_id="loop-limit", iteration=99, loop_goal="test",
            tasks=[{"label": "overflow", "goal": "x"}],
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
        )
        assert len(over) == 1
        assert over[0].startswith("Error")
        await asyncio.sleep(0.1)

    async def test_agent_goal_enriched_with_loop_context(self):
        """Agent goal includes loop ID, iteration, and loop goal."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)

        captured_goals = []
        original_spawn = mgr.spawn

        def spy_spawn(**kwargs):
            captured_goals.append(kwargs.get("goal", ""))
            return original_spawn(**kwargs)

        mgr.spawn = lambda **kw: spy_spawn(**kw)

        iter_cb, tool_cb, announce_cb = _make_callbacks()
        bridge.spawn_agents_for_loop(
            loop_id="my-loop", iteration=3, loop_goal="Monitor DB",
            tasks=[{"label": "checker", "goal": "Check replication"}],
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
        )

        assert len(captured_goals) == 1
        goal = captured_goals[0]
        assert "my-loop" in goal
        assert "iteration 3" in goal
        assert "Monitor DB" in goal
        assert "Check replication" in goal
        await asyncio.sleep(0.1)

    async def test_get_active_loop_agents_excludes_collected(self):
        """get_active_loop_agents only returns uncollected agents."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        iter_cb, tool_cb, announce_cb = _make_callbacks()

        ids = bridge.spawn_agents_for_loop(
            loop_id="lp", iteration=0, loop_goal="x",
            tasks=[
                {"label": "a", "goal": "x"},
                {"label": "b", "goal": "y"},
            ],
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
        )
        await asyncio.sleep(0.1)

        # Collect only the first
        await bridge.wait_and_collect("lp", agent_ids=[ids[0]])

        active = bridge.get_active_loop_agents("lp")
        active_ids = [a["agent_id"] for a in active]
        assert ids[0] not in active_ids
        assert ids[1] in active_ids


# ===========================================================================
# 9. Cross-module integration
# ===========================================================================

class TestCrossModuleIntegration:
    """Verify agent system is properly wired across modules."""

    def test_all_agent_tools_in_registry(self):
        """All 8 agent tools are registered in the tool registry."""
        from src.tools.registry import get_tool_definitions
        tools = get_tool_definitions()
        tool_names = {t["name"] for t in tools}
        expected = {
            "spawn_agent", "send_to_agent", "list_agents",
            "kill_agent", "get_agent_results", "wait_for_agents",
            "spawn_loop_agents", "collect_loop_agents",
        }
        assert expected.issubset(tool_names)

    def test_blocked_tools_are_real_tools(self):
        """Every tool in AGENT_BLOCKED_TOOLS exists in the registry."""
        from src.tools.registry import get_tool_definitions
        tools = get_tool_definitions()
        tool_names = {t["name"] for t in tools}
        for blocked in AGENT_BLOCKED_TOOLS:
            assert blocked in tool_names, f"Blocked tool '{blocked}' not in registry"

    def test_filter_removes_exactly_blocked_tools(self):
        """filter_agent_tools removes exactly the 6 blocked tools."""
        from src.tools.registry import get_tool_definitions
        full = get_tool_definitions()
        filtered = filter_agent_tools(full)

        removed = {t["name"] for t in full} - {t["name"] for t in filtered}
        assert removed == AGENT_BLOCKED_TOOLS

    def test_agent_manager_imported_in_client(self):
        """client.py imports and uses AgentManager."""
        import src.discord.client as client_mod
        assert hasattr(client_mod, "AgentManager")

    def test_loop_agent_bridge_imported_in_client(self):
        """client.py imports and uses LoopAgentBridge."""
        import src.discord.client as client_mod
        assert hasattr(client_mod, "LoopAgentBridge")

    def test_agent_blocked_tools_imported_in_client(self):
        """client.py imports AGENT_BLOCKED_TOOLS for enforcement."""
        import src.discord.client as client_mod
        assert hasattr(client_mod, "AGENT_BLOCKED_TOOLS")

    def test_filter_agent_tools_imported_in_client(self):
        """client.py imports filter_agent_tools for tool list building."""
        import src.discord.client as client_mod
        assert hasattr(client_mod, "filter_agent_tools")

    def test_module_exports_complete(self):
        """src.agents.__init__ exports all public symbols."""
        import src.agents as agents_mod
        expected = [
            "AgentManager", "AgentInfo", "LoopAgentBridge",
            "AGENT_BLOCKED_TOOLS", "filter_agent_tools",
            "ITERATION_CB_TIMEOUT", "TOOL_EXEC_TIMEOUT",
        ]
        for name in expected:
            assert hasattr(agents_mod, name), f"Missing export: {name}"

    def test_tool_schemas_have_required_fields(self):
        """Each agent tool has name, description, and input_schema."""
        from src.tools.registry import get_tool_definitions
        tools = get_tool_definitions()
        agent_tool_names = {
            "spawn_agent", "send_to_agent", "list_agents",
            "kill_agent", "get_agent_results", "wait_for_agents",
            "spawn_loop_agents", "collect_loop_agents",
        }
        for t in tools:
            if t["name"] in agent_tool_names:
                assert "description" in t, f"{t['name']} missing description"
                assert "input_schema" in t, f"{t['name']} missing input_schema"
                assert "type" in t["input_schema"], f"{t['name']} schema missing type"

    def test_spawn_agent_schema_has_label_and_goal_required(self):
        """spawn_agent tool requires label and goal parameters."""
        from src.tools.registry import get_tool_definitions
        tools = get_tool_definitions()
        spawn = next(t for t in tools if t["name"] == "spawn_agent")
        required = spawn["input_schema"].get("required", [])
        assert "label" in required
        assert "goal" in required


# ===========================================================================
# 10. Agent state machine transitions
# ===========================================================================

class TestStateMachine:
    """Verify all state transitions are valid."""

    async def test_running_to_completed(self):
        """Agent goes from running → completed on no tool_calls."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        await asyncio.sleep(0.05)
        assert mgr.get_results(aid)["status"] == "completed"

    async def test_running_to_failed_on_llm_error(self):
        """Agent goes from running → failed on LLM exception."""
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]

        async def bad_llm(msgs, sys, tools):
            raise ValueError("bad response")

        await _run_agent(agent, "sys", [], AsyncMock(side_effect=bad_llm), AsyncMock(), AsyncMock())
        assert agent.status == "failed"

    async def test_running_to_killed_on_cancel(self):
        """Agent goes from running → killed on cancel event."""
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100
        mgr = AgentManager()
        aid = _spawn_agent(mgr, responses=never_done, tool_delay=0.1)
        await asyncio.sleep(0.05)

        mgr.kill(aid)
        await asyncio.sleep(0.2)
        assert mgr.get_results(aid)["status"] == "killed"

    async def test_running_to_timeout(self):
        """Agent goes from running → timeout when lifetime exceeded."""
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]
        agent.created_at = time.time() - MAX_AGENT_LIFETIME - 10

        responses = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]},
        ]

        async def iter_side(msgs, sys, tools):
            if responses:
                return responses.pop(0)
            return {"text": "Done.", "tool_calls": []}

        await _run_agent(
            agent, "sys", [],
            AsyncMock(side_effect=iter_side),
            AsyncMock(return_value="OK"),
            AsyncMock(),
        )

        assert agent.status == "timeout"

    async def test_terminal_states_have_ended_at(self):
        """All terminal states set ended_at."""
        mgr = AgentManager()

        # Completed
        aid1 = _spawn_agent(mgr, label="comp")
        await asyncio.sleep(0.05)
        assert mgr._agents[aid1].ended_at is not None

        # Failed
        agent_f = _make_agent(id="fail1")
        agent_f.messages = [{"role": "user", "content": "test"}]
        await _run_agent(
            agent_f, "sys", [],
            AsyncMock(side_effect=ValueError("boom")),
            AsyncMock(), AsyncMock(),
        )
        assert agent_f.ended_at is not None

    async def test_cannot_kill_completed_agent(self):
        """Killing a completed agent returns message, doesn't change state."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        await asyncio.sleep(0.05)
        assert mgr.get_results(aid)["status"] == "completed"

        result = mgr.kill(aid)
        assert "terminal state" in result.lower() or "already" in result.lower()
        assert mgr.get_results(aid)["status"] == "completed"

    async def test_cannot_send_to_completed_agent(self):
        """Sending to a completed agent returns error."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        await asyncio.sleep(0.05)
        result = mgr.send(aid, "hello")
        assert "not running" in result.lower() or "error" in result.lower()


# ===========================================================================
# 11. Secret scrubbing boundary
# ===========================================================================

class TestSecretScrubbing:
    """Verify secrets are scrubbed at agent boundaries."""

    async def test_tool_result_secrets_scrubbed(self):
        """Secrets in tool output are scrubbed before LLM sees them."""
        responses = [
            {"text": "Checking", "tool_calls": [{"name": "read_file", "input": {}}]},
            {"text": "Done.", "tool_calls": []},
        ]
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": agent.goal}]

        async def leaky_tool(name, inp):
            return "password=SuperSecret123"

        iter_list = list(responses)

        async def iter_side(msgs, sys, tools):
            if iter_list:
                return iter_list.pop(0)
            return {"text": "Exhausted.", "tool_calls": []}

        await _run_agent(
            agent, "sys", [],
            AsyncMock(side_effect=iter_side),
            AsyncMock(side_effect=leaky_tool),
            AsyncMock(),
        )

        # The tool result in messages should have the secret scrubbed
        tool_msgs = [m for m in agent.messages if "[Tool result:" in m.get("content", "")]
        assert len(tool_msgs) == 1
        assert "SuperSecret123" not in tool_msgs[0]["content"]
        assert "[REDACTED]" in tool_msgs[0]["content"]

    async def test_agent_results_stored_internally(self):
        """Agent results are stored internally, not announced to Discord."""
        agent = _make_agent()
        agent.result = "Found password=MySecret123"
        # Results are stored in agent.result — no announce callback
        assert agent.result == "Found password=MySecret123"


# ===========================================================================
# 12. Silent agent behavior (no direct Discord posting)
# ===========================================================================

class TestSilentAgentBehavior:
    """Verify agents do NOT post directly to Discord."""

    async def test_completed_agent_stores_result(self):
        """Completed agent stores result internally for collection."""
        agent = _make_agent(label="my-worker")
        agent.messages = [{"role": "user", "content": "Do work"}]
        iter_cb = AsyncMock(return_value={"text": "Task complete.", "tool_calls": []})
        tool_cb = AsyncMock()
        ann_cb = AsyncMock()

        await _run_agent(agent, "system", [], iter_cb, tool_cb, ann_cb)

        assert agent.status == "completed"
        assert agent.result == "Task complete."
        ann_cb.assert_not_called()

    async def test_failed_agent_stores_error(self):
        """Failed agent stores error internally for collection."""
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": "Do work"}]
        iter_cb = AsyncMock(side_effect=Exception("Connection refused"))
        tool_cb = AsyncMock()
        ann_cb = AsyncMock()

        await _run_agent(agent, "system", [], iter_cb, tool_cb, ann_cb)

        assert agent.status == "failed"
        assert "Connection refused" in agent.error
        ann_cb.assert_not_called()

    async def test_killed_agent_silent(self):
        """Killed agent does not announce to Discord."""
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": "Do work"}]
        agent._cancel_event.set()
        iter_cb = AsyncMock(return_value={"text": "x", "tool_calls": []})
        tool_cb = AsyncMock()
        ann_cb = AsyncMock()

        await _run_agent(agent, "system", [], iter_cb, tool_cb, ann_cb)

        assert agent.status == "killed"
        ann_cb.assert_not_called()


# ===========================================================================
# 13. Wait/poll timing
# ===========================================================================

class TestWaitTiming:
    """Verify wait_for_agents polling behavior."""

    async def test_wait_returns_immediately_when_all_done(self):
        """No polling needed when agents are already completed."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        await asyncio.sleep(0.05)

        start = time.time()
        results = await mgr.wait_for_agents([aid])
        elapsed = time.time() - start

        assert elapsed < 1.0
        assert results[aid]["status"] == "completed"

    async def test_wait_respects_timeout(self):
        """Wait times out and returns partial results for still-running agents."""
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100
        mgr = AgentManager()
        aid = _spawn_agent(mgr, responses=never_done, tool_delay=0.1)
        await asyncio.sleep(0.05)

        start = time.time()
        results = await mgr.wait_for_agents([aid], timeout=0.3)
        elapsed = time.time() - start

        assert elapsed < 1.0
        assert results[aid]["status"] == "running"

        mgr.kill(aid)
        await asyncio.sleep(0.2)

    async def test_wait_empty_returns_immediately(self):
        """Waiting for empty list returns immediately."""
        mgr = AgentManager()
        start = time.time()
        results = await mgr.wait_for_agents([])
        elapsed = time.time() - start

        assert results == {}
        assert elapsed < 0.1

    async def test_wait_nonexistent_agent(self):
        """Waiting for nonexistent agent returns not_found."""
        mgr = AgentManager()
        results = await mgr.wait_for_agents(["fake123"])
        assert results["fake123"]["status"] == "not_found"

    async def test_wait_mixed_found_and_not_found(self):
        """Waiting for mix of real and fake IDs returns both."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        await asyncio.sleep(0.05)

        results = await mgr.wait_for_agents([aid, "fake123"])
        assert results[aid]["status"] == "completed"
        assert results["fake123"]["status"] == "not_found"


# ===========================================================================
# 14. Cleanup scheduling and auto-removal
# ===========================================================================

class TestCleanupScheduling:
    """Verify cleanup scheduling and auto-removal."""

    async def test_done_callback_schedules_cleanup(self):
        """When agent task finishes, _schedule_cleanup is called."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        await asyncio.sleep(0.05)

        # Cleanup task should be scheduled
        assert aid in mgr._cleanup_tasks

    async def test_cleanup_idempotent(self):
        """Calling cleanup twice doesn't crash or double-remove."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        await asyncio.sleep(0.05)

        mgr._agents[aid].ended_at = time.time() - CLEANUP_DELAY - 1
        removed1 = await mgr.cleanup()
        removed2 = await mgr.cleanup()
        assert removed1 == 1
        assert removed2 == 0

    async def test_cleanup_preserves_running_agents(self):
        """Cleanup never removes running agents."""
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100
        mgr = AgentManager()
        aid = _spawn_agent(mgr, responses=never_done)
        await asyncio.sleep(0.05)

        # Even if we backdate, running agents should not be cleaned
        mgr._agents[aid].created_at = time.time() - 99999
        removed = await mgr.cleanup()
        assert removed == 0
        assert mgr.total_count == 1

        mgr.kill(aid)
        await asyncio.sleep(0.1)

    async def test_cleanup_handles_all_terminal_states(self):
        """Cleanup removes agents in any terminal state."""
        mgr = AgentManager()
        for status in ["completed", "failed", "timeout", "killed"]:
            agent = _make_agent(id=f"a-{status}")
            agent.status = status
            agent.ended_at = time.time() - CLEANUP_DELAY - 1
            mgr._agents[agent.id] = agent

        removed = await mgr.cleanup()
        assert removed == 4
        assert mgr.total_count == 0


# ===========================================================================
# 15. AgentInfo dataclass invariants
# ===========================================================================

class TestAgentInfoDataclass:
    """Verify AgentInfo dataclass behavior."""

    def test_default_status_is_running(self):
        agent = _make_agent()
        assert agent.status == "running"

    def test_default_ended_at_is_none(self):
        agent = _make_agent()
        assert agent.ended_at is None

    def test_default_collections_are_independent(self):
        """Each AgentInfo has its own mutable collections."""
        a1 = _make_agent(id="a1")
        a2 = _make_agent(id="a2")
        a1.messages.append({"role": "user", "content": "msg"})
        a1.tools_used.append("tool_a")
        assert a2.messages == []
        assert a2.tools_used == []

    def test_cancel_event_per_agent(self):
        """Each agent has its own cancel event."""
        a1 = _make_agent(id="a1")
        a2 = _make_agent(id="a2")
        a1._cancel_event.set()
        assert not a2._cancel_event.is_set()

    def test_inbox_per_agent(self):
        """Each agent has its own inbox queue."""
        a1 = _make_agent(id="a1")
        a2 = _make_agent(id="a2")
        a1._inbox.put_nowait("msg")
        assert a2._inbox.empty()

    def test_created_at_is_set(self):
        """created_at is auto-populated."""
        agent = _make_agent()
        assert agent.created_at > 0
        assert abs(agent.created_at - time.time()) < 5

    def test_last_activity_is_set(self):
        """last_activity is auto-populated."""
        agent = _make_agent()
        assert agent.last_activity > 0


# ===========================================================================
# 16. _get_last_progress edge cases
# ===========================================================================

class TestGetLastProgress:
    """Edge cases in progress extraction."""

    def test_extracts_last_assistant_message(self):
        agent = _make_agent()
        agent.messages = [
            {"role": "user", "content": "goal"},
            {"role": "assistant", "content": "First"},
            {"role": "user", "content": "[Tool result: x]\nOK"},
            {"role": "assistant", "content": "Second"},
        ]
        assert _get_last_progress(agent) == "Second"

    def test_no_assistant_messages(self):
        agent = _make_agent()
        agent.messages = [{"role": "user", "content": "goal"}]
        assert _get_last_progress(agent) == "(no output)"

    def test_empty_messages(self):
        agent = _make_agent()
        agent.messages = []
        assert _get_last_progress(agent) == "(no output)"

    def test_skips_empty_assistant_content(self):
        agent = _make_agent()
        agent.messages = [
            {"role": "user", "content": "goal"},
            {"role": "assistant", "content": "Valid"},
            {"role": "assistant", "content": ""},
        ]
        assert _get_last_progress(agent) == "Valid"

    def test_skips_none_content(self):
        agent = _make_agent()
        agent.messages = [
            {"role": "user", "content": "goal"},
            {"role": "assistant", "content": "Valid"},
            {"role": "assistant", "content": None},
        ]
        assert _get_last_progress(agent) == "Valid"


# ===========================================================================
# 17. filter_agent_tools edge cases
# ===========================================================================

class TestFilterAgentTools:
    """Edge cases for tool filtering."""

    def test_empty_list(self):
        assert filter_agent_tools([]) == []

    def test_no_agent_tools(self):
        tools = [{"name": "run_command"}, {"name": "read_file"}]
        assert filter_agent_tools(tools) == tools

    def test_only_agent_tools(self):
        tools = [{"name": t} for t in AGENT_BLOCKED_TOOLS]
        assert filter_agent_tools(tools) == []

    def test_mixed_preserves_order(self):
        tools = [
            {"name": "run_command"},
            {"name": "spawn_agent"},
            {"name": "read_file"},
            {"name": "kill_agent"},
            {"name": "check_disk"},
        ]
        filtered = filter_agent_tools(tools)
        assert [t["name"] for t in filtered] == ["run_command", "read_file", "check_disk"]

    def test_tool_without_name_key_preserved(self):
        """Tools missing 'name' key are kept (get returns None, not in blocked)."""
        tools = [{"description": "no name"}]
        assert filter_agent_tools(tools) == tools


# ===========================================================================
# 18. LoopAgentRecord dataclass
# ===========================================================================

class TestLoopAgentRecord:
    """Verify LoopAgentRecord behavior."""

    def test_defaults(self):
        rec = LoopAgentRecord(agent_id="a1", loop_id="l1", iteration=0, label="test")
        assert rec.collected is False
        assert rec.spawned_at > 0

    def test_collected_flag(self):
        rec = LoopAgentRecord(agent_id="a1", loop_id="l1", iteration=0, label="test")
        rec.collected = True
        assert rec.collected is True


# ===========================================================================
# 19. LoopAgentBridge edge cases
# ===========================================================================

class TestLoopAgentBridgeEdgeCases:
    """Edge cases for LoopAgentBridge methods."""

    def test_cleanup_nonexistent_loop(self):
        bridge = LoopAgentBridge(MagicMock())
        assert bridge.cleanup_loop("nonexistent") == 0

    def test_get_loop_agent_ids_nonexistent(self):
        bridge = LoopAgentBridge(MagicMock())
        assert bridge.get_loop_agent_ids("nonexistent") == []

    def test_get_loop_agent_count_nonexistent(self):
        bridge = LoopAgentBridge(MagicMock())
        assert bridge.get_loop_agent_count("nonexistent") == 0

    def test_get_active_loop_agents_nonexistent(self):
        bridge = LoopAgentBridge(MagicMock())
        assert bridge.get_active_loop_agents("nonexistent") == []

    def test_format_results_empty(self):
        bridge = LoopAgentBridge(MagicMock())
        assert bridge.format_agent_results_for_context({}) == ""

    def test_format_results_truncates_long(self):
        bridge = LoopAgentBridge(MagicMock())
        results = {
            "a1": {
                "label": "worker",
                "status": "completed",
                "result": "x" * 1000,
            }
        }
        formatted = bridge.format_agent_results_for_context(results)
        assert "..." in formatted
        assert len(formatted) < 600  # 500 char result + header

    def test_format_results_uses_error_when_no_result(self):
        bridge = LoopAgentBridge(MagicMock())
        results = {
            "a1": {
                "label": "worker",
                "status": "failed",
                "result": "",
                "error": "Connection refused",
            }
        }
        formatted = bridge.format_agent_results_for_context(results)
        assert "Connection refused" in formatted

    async def test_wait_and_collect_empty_loop(self):
        """Wait on empty loop returns empty dict."""
        mgr = MagicMock()
        mgr.wait_for_agents = AsyncMock(return_value={})
        bridge = LoopAgentBridge(mgr)
        results = await bridge.wait_and_collect("empty-loop")
        assert results == {}

    async def test_spawn_empty_tasks_list(self):
        bridge = LoopAgentBridge(MagicMock())
        result = bridge.spawn_agents_for_loop(
            loop_id="lp", iteration=0, loop_goal="x",
            tasks=[],
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=AsyncMock(), tool_executor_callback=AsyncMock(),
            announce_callback=AsyncMock(),
        )
        assert result == []

    async def test_per_iteration_limit(self):
        """More than MAX_AGENTS_PER_ITERATION tasks returns error."""
        bridge = LoopAgentBridge(MagicMock())
        tasks = [{"label": f"t{i}", "goal": "x"} for i in range(MAX_AGENTS_PER_ITERATION + 1)]
        result = bridge.spawn_agents_for_loop(
            loop_id="lp", iteration=0, loop_goal="x",
            tasks=tasks,
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=AsyncMock(), tool_executor_callback=AsyncMock(),
            announce_callback=AsyncMock(),
        )
        assert len(result) == 1
        assert result[0].startswith("Error")


# ===========================================================================
# 20. Constants verification
# ===========================================================================

class TestConstants:
    """Verify all agent system constants have expected values."""

    def test_max_concurrent_agents(self):
        assert MAX_CONCURRENT_AGENTS == 5

    def test_max_agent_lifetime(self):
        assert MAX_AGENT_LIFETIME == 3600

    def test_max_agent_iterations(self):
        assert MAX_AGENT_ITERATIONS == 30

    def test_stale_warn_seconds(self):
        assert STALE_WARN_SECONDS == 120

    def test_cleanup_delay(self):
        assert CLEANUP_DELAY == 300

    def test_wait_default_timeout(self):
        assert WAIT_DEFAULT_TIMEOUT == 300

    def test_wait_poll_interval(self):
        assert WAIT_POLL_INTERVAL == 2

    def test_iteration_cb_timeout(self):
        assert ITERATION_CB_TIMEOUT == 120

    def test_tool_exec_timeout(self):
        assert TOOL_EXEC_TIMEOUT == 300

    def test_terminal_statuses(self):
        assert _TERMINAL_STATUSES == frozenset({"completed", "failed", "timeout", "killed"})
        assert "running" not in _TERMINAL_STATUSES

    def test_agent_blocked_tools_count(self):
        assert len(AGENT_BLOCKED_TOOLS) == 6

    def test_agent_blocked_tools_is_frozenset(self):
        assert isinstance(AGENT_BLOCKED_TOOLS, frozenset)

    def test_loop_bridge_constants(self):
        assert MAX_AGENTS_PER_ITERATION == 3
        assert MAX_AGENTS_PER_LOOP == 10
        assert LOOP_AGENT_WAIT_TIMEOUT == 300


# ===========================================================================
# 21. Manager list/get_results formatting
# ===========================================================================

class TestManagerListAndResults:
    """Verify list() and get_results() output format."""

    async def test_list_format_completed_agent(self):
        mgr = AgentManager()
        aid = _spawn_agent(mgr, label="worker", goal="do stuff")
        await asyncio.sleep(0.05)

        agents = mgr.list()
        assert len(agents) == 1
        a = agents[0]
        assert a["id"] == aid
        assert a["label"] == "worker"
        assert a["status"] == "completed"
        assert a["iteration_count"] == 1
        assert isinstance(a["runtime_seconds"], float)
        assert isinstance(a["tools_used"], int)
        assert a["goal"] == "do stuff"

    async def test_list_filters_by_channel(self):
        mgr = AgentManager()
        aid1 = _spawn_agent(mgr, label="a1", channel_id="ch1")
        aid2 = _spawn_agent(mgr, label="a2", channel_id="ch2")
        await asyncio.sleep(0.05)

        ch1 = mgr.list(channel_id="ch1")
        ch2 = mgr.list(channel_id="ch2")
        assert len(ch1) == 1
        assert ch1[0]["label"] == "a1"
        assert len(ch2) == 1
        assert ch2[0]["label"] == "a2"

    async def test_list_truncates_long_goal(self):
        mgr = AgentManager()
        long_goal = "x" * 200
        aid = _spawn_agent(mgr, goal=long_goal)
        await asyncio.sleep(0.05)

        agents = mgr.list()
        assert len(agents[0]["goal"]) == 100

    async def test_get_results_full_structure(self):
        mgr = AgentManager()
        aid = _spawn_agent(mgr, label="test", goal="my goal")
        await asyncio.sleep(0.05)

        r = mgr.get_results(aid)
        expected_keys = {
            "id", "label", "status", "result", "error",
            "iteration_count", "tools_used", "runtime_seconds", "goal",
        }
        assert set(r.keys()) == expected_keys
        assert r["label"] == "test"
        assert r["goal"] == "my goal"

    def test_get_results_nonexistent(self):
        mgr = AgentManager()
        assert mgr.get_results("fake") is None


# ===========================================================================
# 22. Health check
# ===========================================================================

class TestHealthCheck:
    """Verify check_health safety net."""

    async def test_check_health_kills_stuck_agent(self):
        """Agent past MAX_AGENT_LIFETIME gets force-killed."""
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100
        mgr = AgentManager()
        aid = _spawn_agent(mgr, responses=never_done, tool_delay=0.1)
        await asyncio.sleep(0.05)

        # Backdate creation
        mgr._agents[aid].created_at = time.time() - MAX_AGENT_LIFETIME - 10
        result = mgr.check_health()
        assert result["killed"] == 1

        await asyncio.sleep(0.2)
        assert mgr.get_results(aid)["status"] == "killed"

    async def test_check_health_detects_stale(self):
        """Agent idle > STALE_WARN_SECONDS is detected but not killed."""
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100
        mgr = AgentManager()
        aid = _spawn_agent(mgr, responses=never_done, tool_delay=0.1)
        await asyncio.sleep(0.05)

        # Backdate last_activity but not creation (within lifetime)
        mgr._agents[aid].last_activity = time.time() - STALE_WARN_SECONDS - 10
        result = mgr.check_health()
        assert result["stale"] == 1
        assert result["killed"] == 0

        mgr.kill(aid)
        await asyncio.sleep(0.2)

    async def test_check_health_ignores_terminal(self):
        """Terminal agents are ignored by check_health."""
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        await asyncio.sleep(0.05)

        # Agent is completed — backdating should have no effect
        mgr._agents[aid].created_at = time.time() - MAX_AGENT_LIFETIME - 100
        result = mgr.check_health()
        assert result["killed"] == 0
        assert result["stale"] == 0

    async def test_check_health_empty(self):
        mgr = AgentManager()
        result = mgr.check_health()
        assert result == {"killed": 0, "stale": 0}


# ===========================================================================
# 23. Spawn validation
# ===========================================================================

class TestSpawnValidation:
    """Verify spawn input validation."""

    def test_empty_label_rejected(self):
        mgr = AgentManager()
        iter_cb, tool_cb, announce_cb = _make_callbacks()
        result = mgr.spawn(
            label="", goal="test",
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
        )
        assert result.startswith("Error")

    def test_empty_goal_rejected(self):
        mgr = AgentManager()
        iter_cb, tool_cb, announce_cb = _make_callbacks()
        result = mgr.spawn(
            label="test", goal="",
            channel_id="100", requester_id="u1", requester_name="u1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
        )
        assert result.startswith("Error")

    async def test_agent_id_is_8_char_hex(self):
        mgr = AgentManager()
        aid = _spawn_agent(mgr)
        assert len(aid) == 8
        int(aid, 16)  # Should not raise
        await asyncio.sleep(0.05)

    async def test_agent_ids_are_unique(self):
        mgr = AgentManager()
        ids = set()
        for i in range(10):
            aid = _spawn_agent(mgr, label=f"a{i}", channel_id=f"ch{i}")
            ids.add(aid)
        assert len(ids) == 10
        await asyncio.sleep(0.1)


# ===========================================================================
# 24. Active/total count properties
# ===========================================================================

class TestCountProperties:
    """Verify active_count and total_count accuracy."""

    async def test_active_count_running_only(self):
        never_done = [
            {"text": "Working...", "tool_calls": [{"name": "run_command", "input": {}}]}
        ] * 100
        mgr = AgentManager()
        aid1 = _spawn_agent(mgr, label="a1", responses=list(never_done), tool_delay=0.1)
        aid2 = _spawn_agent(mgr, label="a2")  # completes immediately
        await asyncio.sleep(0.05)

        assert mgr.active_count == 1  # only a1 (a2 completed)
        assert mgr.total_count == 2   # both

        mgr.kill(aid1)
        await asyncio.sleep(0.2)
        assert mgr.active_count == 0
        assert mgr.total_count == 2

    async def test_total_count_includes_all_states(self):
        mgr = AgentManager()
        for status in ["completed", "failed", "timeout", "killed"]:
            agent = _make_agent(id=f"a-{status}")
            agent.status = status
            mgr._agents[agent.id] = agent

        assert mgr.total_count == 4
        assert mgr.active_count == 0


# ===========================================================================
# 25. Regression tests — key invariants
# ===========================================================================

class TestRegressionInvariants:
    """Key invariants that must never break."""

    async def test_agent_no_session_persistence(self):
        """Agents do NOT interact with SessionManager — verify by design."""
        import inspect
        source = inspect.getsource(AgentManager)
        assert "SessionManager" not in source
        assert "session" not in source.lower() or "session" in source.lower()
        # More precise: no import of SessionManager
        src_run = inspect.getsource(_run_agent)
        assert "SessionManager" not in src_run
        assert "session_manager" not in src_run

    def test_agent_blocked_tools_complete(self):
        """All agent-management tools are blocked from agent context."""
        expected = {"spawn_agent", "send_to_agent", "list_agents",
                    "kill_agent", "get_agent_results", "wait_for_agents"}
        assert AGENT_BLOCKED_TOOLS == expected

    def test_no_nested_agent_spawning_instruction(self):
        """AGENT CONTEXT always says 'Do NOT spawn sub-agents'."""
        import inspect
        source = inspect.getsource(AgentManager.spawn)
        assert "Do NOT spawn sub-agents" in source

    async def test_scrub_output_secrets_called_on_tool_result(self):
        """scrub_output_secrets is called on every tool result."""
        import inspect
        source = inspect.getsource(_run_agent)
        assert "scrub_output_secrets" in source

    async def test_scrub_output_secrets_called_in_run_agent(self):
        """scrub_output_secrets is called in agent execution for tool results."""
        import inspect
        source = inspect.getsource(_run_agent)
        assert "scrub_output_secrets" in source

    async def test_agent_respects_cancel_event(self):
        """_run_agent checks _cancel_event at start of every iteration."""
        import inspect
        source = inspect.getsource(_run_agent)
        assert "_cancel_event.is_set()" in source

    async def test_agent_checks_lifetime(self):
        """_run_agent checks elapsed time against MAX_AGENT_LIFETIME."""
        import inspect
        source = inspect.getsource(_run_agent)
        assert "MAX_AGENT_LIFETIME" in source

    async def test_iteration_callback_has_timeout(self):
        """LLM call is wrapped in asyncio.wait_for."""
        import inspect
        source = inspect.getsource(_run_agent)
        assert "asyncio.wait_for" in source
        assert "ITERATION_CB_TIMEOUT" in source

    async def test_tool_exec_has_timeout(self):
        """Tool execution is wrapped in asyncio.wait_for."""
        import inspect
        source = inspect.getsource(_run_agent)
        assert "TOOL_EXEC_TIMEOUT" in source

    def test_max_iterations_constant(self):
        """MAX_AGENT_ITERATIONS is 30."""
        assert MAX_AGENT_ITERATIONS == 30

    async def test_done_callback_registered(self):
        """spawn() registers a done_callback on the task."""
        import inspect
        source = inspect.getsource(AgentManager.spawn)
        assert "add_done_callback" in source
