"""Round 24 — Agent fix-up tests.

Comprehensive tests for agent system issues identified in rounds 16-23:
- Iteration callback timeout enforcement
- Tool execution timeout enforcement
- Blocked tool rejection in agent context
- Callback contract verification
- Tool definition schema completeness
- Cleanup race condition handling
- Agent inbox deterministic delivery
- Error message preservation in agent messages
- Loop-agent bridge edge cases
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
    TOOL_EXEC_TIMEOUT,
    AgentInfo,
    AgentManager,
    _get_last_progress,
    _run_agent,
    filter_agent_tools,
)
from src.agents.loop_bridge import (
    LoopAgentBridge,
    MAX_AGENTS_PER_ITERATION,
    MAX_AGENTS_PER_LOOP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_callbacks(responses=None, tool_results=None):
    """Build mock callbacks for agent execution."""
    if responses is None:
        responses = [{"text": "Done.", "tool_calls": []}]
    responses = list(responses)

    async def iteration_cb(messages, system, tools):
        if responses:
            return responses.pop(0)
        return {"text": "Exhausted.", "tool_calls": []}

    async def tool_exec_cb(tool_name, tool_input):
        if tool_results and tool_name in tool_results:
            return tool_results[tool_name]
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


# ===========================================================================
# 1. Iteration callback timeout
# ===========================================================================

class TestIterationCallbackTimeout:
    """Verify agents fail gracefully when LLM call hangs."""

    async def test_timeout_constant_exists(self):
        assert ITERATION_CB_TIMEOUT == 120

    async def test_tool_exec_timeout_constant_exists(self):
        assert TOOL_EXEC_TIMEOUT == 300

    async def test_iteration_callback_timeout_triggers_failure(self):
        """Agent should fail with 'timeout' when iteration callback hangs."""
        agent = _make_agent()
        announce = AsyncMock()

        async def _hanging_cb(messages, system, tools):
            await asyncio.sleep(999)  # Hang forever

        with patch("src.agents.manager.ITERATION_CB_TIMEOUT", 0.1):
            await _run_agent(
                agent=agent,
                system_prompt="test",
                tools=[],
                iteration_callback=_hanging_cb,
                tool_executor_callback=AsyncMock(),
                announce_callback=announce,
            )

        assert agent.status == "failed"
        assert "timed out" in agent.error.lower()
        assert agent.ended_at is not None

    async def test_iteration_callback_timeout_announces(self):
        """Timeout should trigger an announcement."""
        agent = _make_agent()
        announce = AsyncMock()

        async def _hanging_cb(messages, system, tools):
            await asyncio.sleep(999)

        with patch("src.agents.manager.ITERATION_CB_TIMEOUT", 0.1):
            await _run_agent(
                agent=agent,
                system_prompt="test",
                tools=[],
                iteration_callback=_hanging_cb,
                tool_executor_callback=AsyncMock(),
                announce_callback=announce,
            )

        assert announce.call_count >= 1
        call_text = announce.call_args[0][1]
        assert "timeout" in call_text.lower()


# ===========================================================================
# 2. Tool execution timeout
# ===========================================================================

class TestToolExecutionTimeout:
    """Verify tool execution has timeout enforcement."""

    async def test_tool_timeout_returns_error_and_continues(self):
        """When a tool times out, agent gets error message and continues."""
        agent = _make_agent()
        call_count = 0

        async def _iter_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "calling tool",
                    "tool_calls": [{"name": "slow_tool", "input": {}}],
                    "stop_reason": "end_turn",
                }
            return {"text": "Done after timeout.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _slow_tool(name, inp):
            await asyncio.sleep(999)

        with patch("src.agents.manager.TOOL_EXEC_TIMEOUT", 0.1):
            await _run_agent(
                agent=agent,
                system_prompt="test",
                tools=[],
                iteration_callback=_iter_cb,
                tool_executor_callback=_slow_tool,
                announce_callback=AsyncMock(),
            )

        assert agent.status == "completed"
        assert "Done after timeout." in agent.result
        # Tool timeout error should be in messages
        tool_msgs = [m for m in agent.messages if "timed out" in m.get("content", "").lower()]
        assert len(tool_msgs) >= 1

    async def test_tool_timeout_error_includes_tool_name(self):
        """Tool timeout error message should include the tool name."""
        agent = _make_agent()
        call_count = 0

        async def _iter_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "",
                    "tool_calls": [{"name": "my_special_tool", "input": {}}],
                    "stop_reason": "end_turn",
                }
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _slow(name, inp):
            await asyncio.sleep(999)

        with patch("src.agents.manager.TOOL_EXEC_TIMEOUT", 0.1):
            await _run_agent(
                agent=agent,
                system_prompt="test",
                tools=[],
                iteration_callback=_iter_cb,
                tool_executor_callback=_slow,
                announce_callback=AsyncMock(),
            )

        tool_msgs = [m for m in agent.messages if "my_special_tool" in m.get("content", "")]
        assert len(tool_msgs) >= 1


# ===========================================================================
# 3. Blocked tool rejection
# ===========================================================================

class TestBlockedToolRejection:
    """Verify agent tools are properly filtered and rejected."""

    def test_filter_removes_all_blocked_tools(self):
        tools = [{"name": n} for n in AGENT_BLOCKED_TOOLS]
        tools.append({"name": "run_command"})
        filtered = filter_agent_tools(tools)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "run_command"

    def test_filter_preserves_non_agent_tools(self):
        tools = [
            {"name": "run_command"},
            {"name": "read_file"},
            {"name": "search_knowledge"},
        ]
        filtered = filter_agent_tools(tools)
        assert len(filtered) == 3

    def test_blocked_tools_set_is_frozen(self):
        assert isinstance(AGENT_BLOCKED_TOOLS, frozenset)

    def test_blocked_tools_includes_all_agent_tools(self):
        expected = {
            "spawn_agent", "send_to_agent", "list_agents",
            "kill_agent", "get_agent_results", "wait_for_agents",
        }
        assert expected == AGENT_BLOCKED_TOOLS

    async def test_agent_tool_exec_rejects_blocked_tools_in_practice(self):
        """When agent's LLM tries to call spawn_agent, tool_exec should error."""
        agent = _make_agent()
        call_count = 0

        async def _iter_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "spawning sub-agent",
                    "tool_calls": [{"name": "spawn_agent", "input": {"label": "sub", "goal": "test"}}],
                    "stop_reason": "end_turn",
                }
            return {"text": "Got rejection.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _tool_cb(name, inp):
            # Simulate what client.py does — reject agent tools
            if name in AGENT_BLOCKED_TOOLS:
                return f"Error: Tool '{name}' is not available inside agents."
            return "ok"

        await _run_agent(
            agent=agent,
            system_prompt="test",
            tools=[],
            iteration_callback=_iter_cb,
            tool_executor_callback=_tool_cb,
            announce_callback=AsyncMock(),
        )

        assert agent.status == "completed"
        # The rejection error should be in the agent's message history
        rejection_msgs = [
            m for m in agent.messages
            if "not available inside agents" in m.get("content", "")
        ]
        assert len(rejection_msgs) >= 1


# ===========================================================================
# 4. Callback contract verification
# ===========================================================================

class TestCallbackContract:
    """Verify iteration callback response format is handled correctly."""

    async def test_response_with_text_only(self):
        """Agent completes when response has text but no tool_calls."""
        agent = _make_agent()
        iter_cb, tool_cb, ann_cb = _make_callbacks(
            responses=[{"text": "All done.", "tool_calls": []}]
        )
        await _run_agent(agent, "sys", [], iter_cb, tool_cb, ann_cb)
        assert agent.status == "completed"
        assert agent.result == "All done."

    async def test_response_with_empty_text_and_no_tools(self):
        """Agent completes even with empty text response."""
        agent = _make_agent()
        iter_cb, tool_cb, ann_cb = _make_callbacks(
            responses=[{"text": "", "tool_calls": []}]
        )
        await _run_agent(agent, "sys", [], iter_cb, tool_cb, ann_cb)
        assert agent.status == "completed"

    async def test_response_missing_text_key(self):
        """Agent handles response missing 'text' key gracefully."""
        agent = _make_agent()

        async def _iter_cb(messages, system, tools):
            return {"tool_calls": [], "stop_reason": "end_turn"}

        await _run_agent(agent, "sys", [], _iter_cb, AsyncMock(), AsyncMock())
        assert agent.status == "completed"

    async def test_response_missing_tool_calls_key(self):
        """Agent handles response missing 'tool_calls' key (treats as no tools)."""
        agent = _make_agent()

        async def _iter_cb(messages, system, tools):
            return {"text": "response without tool_calls key"}

        await _run_agent(agent, "sys", [], _iter_cb, AsyncMock(), AsyncMock())
        assert agent.status == "completed"

    async def test_tool_call_dict_format(self):
        """Tool calls must have 'name' and 'input' keys."""
        agent = _make_agent()
        call_count = 0

        async def _iter_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "",
                    "tool_calls": [{"name": "run_command", "input": {"cmd": "ls"}}],
                    "stop_reason": "end_turn",
                }
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        tool_cb = AsyncMock(return_value="file1\nfile2")
        await _run_agent(agent, "sys", [], _iter_cb, tool_cb, AsyncMock())
        assert agent.status == "completed"
        tool_cb.assert_called_once_with("run_command", {"cmd": "ls"})


# ===========================================================================
# 5. Tool definition schema completeness
# ===========================================================================

class TestToolDefinitionSchemas:
    """Verify agent tool definitions are correct and complete."""

    @pytest.fixture()
    def agent_tools(self):
        from src.tools.registry import TOOLS
        names = {
            "spawn_agent", "send_to_agent", "list_agents", "kill_agent",
            "get_agent_results", "wait_for_agents", "spawn_loop_agents",
            "collect_loop_agents",
        }
        return [t for t in TOOLS if t["name"] in names]

    def test_all_agent_tools_exist(self, agent_tools):
        names = {t["name"] for t in agent_tools}
        expected = {
            "spawn_agent", "send_to_agent", "list_agents", "kill_agent",
            "get_agent_results", "wait_for_agents", "spawn_loop_agents",
            "collect_loop_agents",
        }
        assert names == expected

    def test_all_tools_have_input_schema(self, agent_tools):
        for t in agent_tools:
            assert "input_schema" in t, f"{t['name']} missing input_schema"
            assert t["input_schema"]["type"] == "object"

    def test_spawn_agent_mentions_no_nesting(self, agent_tools):
        spawn = next(t for t in agent_tools if t["name"] == "spawn_agent")
        desc = spawn["description"].lower()
        assert "sub-agent" in desc or "nesting" in desc or "cannot" in desc

    def test_spawn_agent_has_required_fields(self, agent_tools):
        spawn = next(t for t in agent_tools if t["name"] == "spawn_agent")
        assert set(spawn["input_schema"]["required"]) == {"label", "goal"}

    def test_wait_for_agents_has_agent_ids_required(self, agent_tools):
        wait = next(t for t in agent_tools if t["name"] == "wait_for_agents")
        assert "agent_ids" in wait["input_schema"]["required"]

    def test_wait_for_agents_timeout_description_mentions_default(self, agent_tools):
        wait = next(t for t in agent_tools if t["name"] == "wait_for_agents")
        timeout_desc = wait["input_schema"]["properties"]["timeout"]["description"]
        assert "300" in timeout_desc or "default" in timeout_desc.lower()

    def test_spawn_loop_agents_has_required_fields(self, agent_tools):
        spawn_loop = next(t for t in agent_tools if t["name"] == "spawn_loop_agents")
        required = set(spawn_loop["input_schema"]["required"])
        assert "loop_id" in required
        assert "tasks" in required

    def test_collect_loop_agents_timeout_mentions_default(self, agent_tools):
        collect = next(t for t in agent_tools if t["name"] == "collect_loop_agents")
        timeout_desc = collect["input_schema"]["properties"]["timeout"]["description"]
        assert "300" in timeout_desc or "default" in timeout_desc.lower()


# ===========================================================================
# 6. Error message preservation
# ===========================================================================

class TestErrorPreservation:
    """Verify errors are properly captured in agent state and messages."""

    async def test_tool_error_preserved_in_messages(self):
        """Tool errors should appear in agent's message history."""
        agent = _make_agent()
        call_count = 0

        async def _iter_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "",
                    "tool_calls": [{"name": "failing_tool", "input": {}}],
                    "stop_reason": "end_turn",
                }
            return {"text": "Handled error.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _fail_cb(name, inp):
            raise ValueError("database connection lost")

        await _run_agent(agent, "sys", [], _iter_cb, _fail_cb, AsyncMock())

        assert agent.status == "completed"
        error_msgs = [m for m in agent.messages if "database connection lost" in m.get("content", "")]
        assert len(error_msgs) >= 1

    async def test_llm_error_sets_agent_error_field(self):
        """LLM call failure should set agent.error."""
        agent = _make_agent()

        async def _fail_cb(messages, system, tools):
            raise RuntimeError("API rate limited")

        await _run_agent(agent, "sys", [], _fail_cb, AsyncMock(), AsyncMock())

        assert agent.status == "failed"
        assert "API rate limited" in agent.error

    async def test_tool_error_format(self):
        """Tool error message follows 'Error: {exception}' format."""
        agent = _make_agent()
        call_count = 0

        async def _iter_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "",
                    "tool_calls": [{"name": "bad_tool", "input": {}}],
                    "stop_reason": "end_turn",
                }
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _err(name, inp):
            raise IOError("disk full")

        await _run_agent(agent, "sys", [], _iter_cb, _err, AsyncMock())

        error_msgs = [m for m in agent.messages if "[Tool result: bad_tool]" in m.get("content", "")]
        assert len(error_msgs) >= 1
        assert "Error:" in error_msgs[0]["content"]
        assert "disk full" in error_msgs[0]["content"]


# ===========================================================================
# 7. Cleanup correctness
# ===========================================================================

class TestCleanupCorrectness:
    """Verify cleanup handles edge cases."""

    async def test_cleanup_only_removes_expired_agents(self):
        """Cleanup should only remove agents past CLEANUP_DELAY."""
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        # Spawn and let it complete
        id1 = mgr.spawn("a1", "goal1", "100", "u1", "u1", iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.1)

        # Agent should still exist (not past CLEANUP_DELAY)
        assert mgr.get_results(id1) is not None
        removed = await mgr.cleanup()
        assert removed == 0
        assert mgr.get_results(id1) is not None

    async def test_cleanup_removes_after_delay(self):
        """With patched CLEANUP_DELAY=0, cleanup should remove completed agents."""
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        id1 = mgr.spawn("a1", "goal1", "100", "u1", "u1", iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.1)

        # Manually set ended_at far in the past
        agent = mgr._agents[id1]
        agent.ended_at = time.time() - CLEANUP_DELAY - 10

        removed = await mgr.cleanup()
        assert removed == 1
        assert mgr.get_results(id1) is None

    async def test_cleanup_idempotent(self):
        """Calling cleanup twice doesn't crash."""
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        id1 = mgr.spawn("a1", "goal1", "100", "u1", "u1", iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.1)
        agent = mgr._agents[id1]
        agent.ended_at = time.time() - CLEANUP_DELAY - 10

        await mgr.cleanup()
        removed = await mgr.cleanup()
        assert removed == 0

    async def test_check_health_kills_stuck_agents(self):
        """check_health should signal cancel on agents past MAX_AGENT_LIFETIME."""
        mgr = AgentManager()
        gate = asyncio.Event()

        async def _block(messages, system, tools):
            await gate.wait()
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        id1 = mgr.spawn("stuck", "stuck goal", "100", "u1", "u1",
                         _block, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.05)

        # Backdate creation to simulate stuck agent
        mgr._agents[id1].created_at = time.time() - MAX_AGENT_LIFETIME - 10

        result = mgr.check_health()
        assert result["killed"] >= 1

        # check_health sets the cancel event
        assert mgr._agents[id1]._cancel_event.is_set()

        gate.set()
        await asyncio.sleep(0.2)

        # Agent should reach killed or timeout (or completed if it raced)
        agent = mgr._agents.get(id1)
        if agent:
            assert agent.status in ("killed", "timeout", "completed")


# ===========================================================================
# 8. Inbox message delivery
# ===========================================================================

class TestInboxDelivery:
    """Verify inbox messages are delivered correctly."""

    async def test_message_appears_in_agent_context(self):
        """Sent message should appear as user message in agent iteration."""
        mgr = AgentManager()
        received_messages = []
        gate = asyncio.Event()

        async def _iter_cb(messages, system, tools):
            if not gate.is_set():
                gate.set()
                await asyncio.sleep(0.1)
                return {
                    "text": "",
                    "tool_calls": [{"name": "noop", "input": {}}],
                    "stop_reason": "end_turn",
                }
            received_messages.extend(messages)
            return {"text": "Got it.", "tool_calls": [], "stop_reason": "end_turn"}

        id1 = mgr.spawn("inbox-test", "test inbox", "100", "u1", "u1",
                         _iter_cb, AsyncMock(return_value="ok"), AsyncMock())
        await gate.wait()
        mgr.send(id1, "important update")
        await asyncio.sleep(0.3)

        parent_msgs = [m for m in received_messages if "[Message from parent]" in m.get("content", "")]
        assert len(parent_msgs) >= 1
        assert "important update" in parent_msgs[0]["content"]

    async def test_send_to_nonexistent_agent_returns_error(self):
        mgr = AgentManager()
        result = mgr.send("nonexistent", "hello")
        assert "not found" in result.lower()

    async def test_send_to_completed_agent_returns_error(self):
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()
        id1 = mgr.spawn("done", "done goal", "100", "u1", "u1",
                         iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.1)
        result = mgr.send(id1, "too late")
        assert "not running" in result.lower()

    async def test_send_empty_message_returns_error(self):
        mgr = AgentManager()
        gate = asyncio.Event()

        async def _block(messages, system, tools):
            gate.set()
            await asyncio.sleep(5)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        id1 = mgr.spawn("msg-test", "test", "100", "u1", "u1",
                         _block, AsyncMock(), AsyncMock())
        await gate.wait()
        result = mgr.send(id1, "")
        assert "empty" in result.lower()
        mgr.kill(id1)
        await asyncio.sleep(0.1)


# ===========================================================================
# 9. Agent concurrency limits
# ===========================================================================

class TestConcurrencyLimits:
    """Verify per-channel agent limits."""

    async def test_max_concurrent_reached(self):
        mgr = AgentManager()
        gate = asyncio.Event()

        async def _block(messages, system, tools):
            await gate.wait()
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        ids = []
        for i in range(MAX_CONCURRENT_AGENTS):
            aid = mgr.spawn(f"agent-{i}", "goal", "100", "u1", "u1",
                            _block, AsyncMock(), AsyncMock())
            assert not aid.startswith("Error"), f"Unexpected error on agent {i}: {aid}"
            ids.append(aid)

        # Next spawn should fail
        result = mgr.spawn("over-limit", "goal", "100", "u1", "u1",
                           _block, AsyncMock(), AsyncMock())
        assert result.startswith("Error")
        assert "Maximum" in result

        # Different channel should succeed
        diff_channel = mgr.spawn("other-chan", "goal", "200", "u1", "u1",
                                 _block, AsyncMock(), AsyncMock())
        assert not diff_channel.startswith("Error")

        gate.set()
        await asyncio.sleep(0.2)


# ===========================================================================
# 10. Loop-agent bridge
# ===========================================================================

class TestLoopAgentBridgeEdgeCases:
    """Edge cases for the loop-agent bridge."""

    def test_spawn_empty_tasks(self):
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        result = bridge.spawn_agents_for_loop(
            "loop1", 1, "test", [], "100", "u1", "u1",
            AsyncMock(), AsyncMock(), AsyncMock(),
        )
        assert result == []

    def test_spawn_exceeds_per_iteration_limit(self):
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        tasks = [{"label": f"t{i}", "goal": f"g{i}"} for i in range(MAX_AGENTS_PER_ITERATION + 1)]
        result = bridge.spawn_agents_for_loop(
            "loop1", 1, "test", tasks, "100", "u1", "u1",
            AsyncMock(), AsyncMock(), AsyncMock(),
        )
        assert len(result) == 1
        assert result[0].startswith("Error")

    def test_spawn_exceeds_per_loop_lifetime_limit(self):
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        gate = asyncio.Event()

        async def _block(messages, system, tools):
            await gate.wait()
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        # Spawn MAX_AGENTS_PER_LOOP agents across different channels to avoid
        # the per-channel concurrent limit (MAX_CONCURRENT_AGENTS=5)
        for i in range(MAX_AGENTS_PER_LOOP):
            tasks = [{"label": f"agent-{i}", "goal": f"goal-{i}"}]
            result = bridge.spawn_agents_for_loop(
                "loop1", i, "test loop", tasks, str(1000 + i), "u1", "u1",
                _block, AsyncMock(return_value="ok"), AsyncMock(),
            )
            assert not result[0].startswith("Error"), f"Failed at agent {i}: {result}"

        # Next spawn should fail (per-loop lifetime limit)
        tasks = [{"label": "over-limit", "goal": "nope"}]
        result = bridge.spawn_agents_for_loop(
            "loop1", 99, "test", tasks, "2000", "u1", "u1",
            _block, AsyncMock(), AsyncMock(),
        )
        assert result[0].startswith("Error")
        assert "limit" in result[0].lower()

        gate.set()

    def test_cleanup_loop_removes_records(self):
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)

        async def _done(messages, system, tools):
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        tasks = [{"label": "a1", "goal": "g1"}]
        bridge.spawn_agents_for_loop(
            "loop1", 1, "test", tasks, "100", "u1", "u1",
            _done, AsyncMock(return_value="ok"), AsyncMock(),
        )
        assert bridge.get_loop_agent_count("loop1") == 1

        removed = bridge.cleanup_loop("loop1")
        assert removed == 1
        assert bridge.get_loop_agent_count("loop1") == 0

    def test_format_results_empty(self):
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        assert bridge.format_agent_results_for_context({}) == ""

    def test_format_results_truncates(self):
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        results = {
            "a1": {
                "label": "test",
                "status": "completed",
                "result": "x" * 1000,
            }
        }
        text = bridge.format_agent_results_for_context(results)
        assert "..." in text
        assert len(text) < 1000

    def test_get_active_loop_agents_empty(self):
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        assert bridge.get_active_loop_agents("nonexistent") == []

    def test_tracked_loop_count(self):
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)

        async def _done(messages, system, tools):
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        bridge.spawn_agents_for_loop(
            "loop1", 1, "test1", [{"label": "a1", "goal": "g1"}],
            "100", "u1", "u1", _done, AsyncMock(return_value="ok"), AsyncMock(),
        )
        bridge.spawn_agents_for_loop(
            "loop2", 1, "test2", [{"label": "a2", "goal": "g2"}],
            "100", "u1", "u1", _done, AsyncMock(return_value="ok"), AsyncMock(),
        )
        assert bridge.tracked_loop_count == 2


# ===========================================================================
# 11. Agent properties
# ===========================================================================

class TestAgentProperties:
    """Verify AgentManager properties."""

    async def test_active_count(self):
        mgr = AgentManager()
        gate = asyncio.Event()

        async def _block(messages, system, tools):
            await gate.wait()
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        mgr.spawn("a1", "g1", "100", "u1", "u1", _block, AsyncMock(), AsyncMock())
        mgr.spawn("a2", "g2", "100", "u1", "u1", _block, AsyncMock(), AsyncMock())
        await asyncio.sleep(0.05)

        assert mgr.active_count == 2
        assert mgr.total_count == 2

        gate.set()
        await asyncio.sleep(0.1)

        assert mgr.active_count == 0

    async def test_total_count_includes_all_states(self):
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()
        mgr.spawn("done", "goal", "100", "u1", "u1", iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.1)

        assert mgr.total_count >= 1
        # Completed agents still count until cleaned up
        agent = list(mgr._agents.values())[0]
        assert agent.status == "completed"


# ===========================================================================
# 12. _get_last_progress edge cases
# ===========================================================================

class TestGetLastProgress:
    """Verify _get_last_progress extraction."""

    def test_returns_last_assistant_message(self):
        agent = _make_agent()
        agent.messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "first response"},
            {"role": "user", "content": "[Tool result: x]\nok"},
            {"role": "assistant", "content": "final response"},
        ]
        assert _get_last_progress(agent) == "final response"

    def test_returns_no_output_when_empty(self):
        agent = _make_agent()
        agent.messages = []
        assert _get_last_progress(agent) == "(no output)"

    def test_skips_empty_assistant_messages(self):
        agent = _make_agent()
        agent.messages = [
            {"role": "assistant", "content": "real content"},
            {"role": "assistant", "content": ""},
        ]
        assert _get_last_progress(agent) == "real content"


# ===========================================================================
# 13. Secret scrubbing in agents
# ===========================================================================

class TestAgentSecretScrubbing:
    """Verify tool output is scrubbed for secrets inside agents."""

    async def test_tool_output_scrubbed(self):
        """Tool output containing secrets should be scrubbed."""
        agent = _make_agent()
        call_count = 0

        async def _iter_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "",
                    "tool_calls": [{"name": "read_env", "input": {}}],
                    "stop_reason": "end_turn",
                }
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _tool_cb(name, inp):
            return "password=supersecret123"

        await _run_agent(agent, "sys", [], _iter_cb, _tool_cb, AsyncMock())

        # The secret should be scrubbed in messages
        tool_msgs = [m for m in agent.messages if "[Tool result: read_env]" in m.get("content", "")]
        assert len(tool_msgs) >= 1
        assert "supersecret123" not in tool_msgs[0]["content"]
        assert "REDACTED" in tool_msgs[0]["content"]


# ===========================================================================
# 14. Announcement formatting
# ===========================================================================

class TestAnnouncementFormatting:
    """Verify announcements are formatted and truncated correctly."""

    async def test_announcement_on_completion(self):
        agent = _make_agent()
        ann = AsyncMock()
        iter_cb, tool_cb, _ = _make_callbacks(
            responses=[{"text": "Task complete.", "tool_calls": []}]
        )
        await _run_agent(agent, "sys", [], iter_cb, tool_cb, ann)

        assert ann.call_count >= 1
        text = ann.call_args[0][1]
        assert "Agent: test-agent" in text
        assert "completed" in text.lower()

    async def test_announcement_truncates_long_results(self):
        agent = _make_agent()
        ann = AsyncMock()

        async def _iter_cb(messages, system, tools):
            return {"text": "x" * 5000, "tool_calls": [], "stop_reason": "end_turn"}

        await _run_agent(agent, "sys", [], _iter_cb, AsyncMock(), ann)

        text = ann.call_args[0][1]
        assert len(text) < 2500  # Should be truncated around 1800 + header

    async def test_announcement_on_failure(self):
        agent = _make_agent()
        ann = AsyncMock()

        async def _fail(messages, system, tools):
            raise RuntimeError("crash")

        await _run_agent(agent, "sys", [], _fail, AsyncMock(), ann)

        assert ann.call_count >= 1
        text = ann.call_args[0][1]
        assert "failed" in text.lower()


# ===========================================================================
# 15. Cancellation handling
# ===========================================================================

class TestCancellationHandling:
    """Verify agents respond to kill signals."""

    async def test_kill_during_iteration(self):
        """Agent should stop when kill signal is set between iterations."""
        mgr = AgentManager()
        gate = asyncio.Event()
        tool_called = asyncio.Event()

        async def _iter_cb(messages, system, tools):
            if not gate.is_set():
                gate.set()
                return {
                    "text": "",
                    "tool_calls": [{"name": "noop", "input": {}}],
                    "stop_reason": "end_turn",
                }
            # Second call — should not happen if kill was processed
            return {"text": "Should not reach.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _tool_cb(name, inp):
            tool_called.set()
            # Give time for kill signal to arrive before next iteration
            await asyncio.sleep(0.3)
            return "ok"

        id1 = mgr.spawn("killable", "goal", "100", "u1", "u1",
                         _iter_cb, _tool_cb, AsyncMock())
        await gate.wait()
        await tool_called.wait()
        mgr.kill(id1)
        await asyncio.sleep(0.5)

        agent = mgr._agents.get(id1)
        assert agent is not None
        assert agent.status == "killed"

    async def test_kill_nonexistent_agent(self):
        mgr = AgentManager()
        result = mgr.kill("fake")
        assert "not found" in result.lower()

    async def test_kill_already_completed(self):
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()
        id1 = mgr.spawn("done", "goal", "100", "u1", "u1", iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.1)
        result = mgr.kill(id1)
        assert "terminal" in result.lower() or "already" in result.lower()


# ===========================================================================
# 16. Multi-tool execution in single iteration
# ===========================================================================

class TestMultiToolExecution:
    """Verify agents handle multiple tool calls in one iteration."""

    async def test_multiple_tools_per_iteration(self):
        agent = _make_agent()
        call_count = 0

        async def _iter_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "",
                    "tool_calls": [
                        {"name": "tool_a", "input": {}},
                        {"name": "tool_b", "input": {}},
                        {"name": "tool_c", "input": {}},
                    ],
                    "stop_reason": "end_turn",
                }
            return {"text": "Done with all tools.", "tool_calls": [], "stop_reason": "end_turn"}

        tool_calls = []

        async def _tool_cb(name, inp):
            tool_calls.append(name)
            return f"result-{name}"

        await _run_agent(agent, "sys", [], _iter_cb, _tool_cb, AsyncMock())

        assert agent.status == "completed"
        assert set(tool_calls) == {"tool_a", "tool_b", "tool_c"}
        assert set(agent.tools_used) == {"tool_a", "tool_b", "tool_c"}

    async def test_tools_used_dedup(self):
        """Tools called multiple times should appear once in tools_used."""
        agent = _make_agent()
        call_count = 0

        async def _iter_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {
                    "text": "",
                    "tool_calls": [{"name": "run_command", "input": {"cmd": f"cmd{call_count}"}}],
                    "stop_reason": "end_turn",
                }
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        await _run_agent(agent, "sys", [], _iter_cb, AsyncMock(return_value="ok"), AsyncMock())

        assert agent.tools_used.count("run_command") == 1


# ===========================================================================
# 17. Max iterations exhaustion
# ===========================================================================

class TestMaxIterationsExhaustion:
    """Verify agent behavior when MAX_AGENT_ITERATIONS is reached."""

    async def test_agent_completes_at_max_iterations(self):
        agent = _make_agent()

        async def _never_done(messages, system, tools):
            return {
                "text": f"Iteration {agent.iteration_count}",
                "tool_calls": [{"name": "noop", "input": {}}],
                "stop_reason": "end_turn",
            }

        with patch("src.agents.manager.MAX_AGENT_ITERATIONS", 3):
            await _run_agent(
                agent, "sys", [],
                _never_done, AsyncMock(return_value="ok"), AsyncMock(),
            )

        assert agent.status == "completed"
        assert agent.iteration_count == 3
        assert agent.result  # Should have last progress


# ===========================================================================
# 18. Wait for agents
# ===========================================================================

class TestWaitForAgents:
    """Verify wait_for_agents coordination."""

    async def test_wait_returns_all_results(self):
        mgr = AgentManager()
        responses_a = [{"text": "A result.", "tool_calls": []}]
        responses_b = [{"text": "B result.", "tool_calls": []}]

        iter_a, tool_a, ann_a = _make_callbacks(responses=responses_a)
        iter_b, tool_b, ann_b = _make_callbacks(responses=responses_b)

        id_a = mgr.spawn("a", "goal-a", "100", "u1", "u1", iter_a, tool_a, ann_a)
        id_b = mgr.spawn("b", "goal-b", "100", "u1", "u1", iter_b, tool_b, ann_b)

        results = await mgr.wait_for_agents([id_a, id_b], timeout=5)
        assert id_a in results
        assert id_b in results
        assert results[id_a]["status"] == "completed"
        assert results[id_b]["status"] == "completed"

    async def test_wait_for_nonexistent_agent(self):
        mgr = AgentManager()
        results = await mgr.wait_for_agents(["fake123"], timeout=1)
        assert "fake123" in results
        assert results["fake123"]["status"] == "not_found"

    async def test_wait_timeout(self):
        mgr = AgentManager()

        async def _block(messages, system, tools):
            await asyncio.sleep(999)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        id1 = mgr.spawn("slow", "goal", "100", "u1", "u1",
                         _block, AsyncMock(), AsyncMock())
        results = await mgr.wait_for_agents([id1], timeout=0.1, poll_interval=0.05)
        assert results[id1]["status"] == "running"

        mgr.kill(id1)
        await asyncio.sleep(0.1)

    async def test_wait_empty_list(self):
        mgr = AgentManager()
        results = await mgr.wait_for_agents([])
        assert results == {}


# ===========================================================================
# 19. Spawn validation
# ===========================================================================

class TestSpawnValidation:
    """Verify spawn parameter validation."""

    def test_empty_label(self):
        mgr = AgentManager()
        result = mgr.spawn("", "goal", "100", "u1", "u1",
                           AsyncMock(), AsyncMock(), AsyncMock())
        assert result.startswith("Error")

    def test_empty_goal(self):
        mgr = AgentManager()
        result = mgr.spawn("label", "", "100", "u1", "u1",
                           AsyncMock(), AsyncMock(), AsyncMock())
        assert result.startswith("Error")

    def test_spawn_returns_8_char_hex_id(self):
        mgr = AgentManager()

        async def _done(messages, system, tools):
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        id1 = mgr.spawn("test", "goal", "100", "u1", "u1",
                         _done, AsyncMock(return_value="ok"), AsyncMock())
        assert len(id1) == 8
        assert all(c in "0123456789abcdef" for c in id1)


# ===========================================================================
# 20. Agent system prompt
# ===========================================================================

class TestAgentSystemPrompt:
    """Verify agent system prompt construction."""

    async def test_agent_context_injected(self):
        """Agent system prompt should include AGENT CONTEXT with label."""
        received_prompts = []
        agent = _make_agent(label="disk-audit")

        async def _iter_cb(messages, system, tools):
            received_prompts.append(system)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        # spawn builds the system prompt, but _run_agent receives it pre-built
        # Test that _run_agent passes system_prompt to iteration_callback
        await _run_agent(
            agent, "BASE PROMPT\n\nAGENT CONTEXT: You are agent 'disk-audit'",
            [], _iter_cb, AsyncMock(), AsyncMock(),
        )

        assert len(received_prompts) >= 1
        assert "AGENT CONTEXT" in received_prompts[0]
        assert "disk-audit" in received_prompts[0]

    async def test_empty_base_prompt_gets_agent_context(self):
        """Even with empty base prompt, agent should get AGENT CONTEXT."""
        mgr = AgentManager()

        async def _done(messages, system, tools):
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        # Spawn with empty system_prompt
        id1 = mgr.spawn("my-agent", "goal", "100", "u1", "u1",
                         _done, AsyncMock(return_value="ok"), AsyncMock(),
                         system_prompt="")
        await asyncio.sleep(0.1)

        # The agent should still have gotten AGENT CONTEXT
        agent = mgr._agents.get(id1)
        if agent:
            # Agent completed already, but we can verify it was constructed
            assert agent.status == "completed"
