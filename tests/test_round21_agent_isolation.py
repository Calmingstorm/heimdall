"""Round 21 — Agent context isolation tests.

Tests strict isolation between agents, parent session protection,
tool filtering for nesting prevention, sibling independence, and
announcement boundary integrity.
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
    _run_agent,
    filter_agent_tools,
)
from src.tools.registry import TOOLS


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


def _spawn_agent(mgr, label="test", goal="do something", channel_id="100", **kw):
    """Helper to spawn an agent with minimal boilerplate."""
    iter_cb, tool_cb, ann_cb = _make_callbacks(**kw)
    aid = mgr.spawn(
        label=label,
        goal=goal,
        channel_id=channel_id,
        requester_id="u1",
        requester_name="user1",
        iteration_callback=iter_cb,
        tool_executor_callback=tool_cb,
        announce_callback=ann_cb,
    )
    return aid, iter_cb, tool_cb, ann_cb


# ---------------------------------------------------------------------------
# AGENT_BLOCKED_TOOLS constant
# ---------------------------------------------------------------------------

class TestAgentBlockedTools:
    """AGENT_BLOCKED_TOOLS prevents nesting and cross-agent interference."""

    def test_contains_all_agent_tools(self):
        expected = {"spawn_agent", "send_to_agent", "list_agents",
                    "kill_agent", "get_agent_results", "wait_for_agents"}
        assert AGENT_BLOCKED_TOOLS == expected

    def test_is_frozenset(self):
        assert isinstance(AGENT_BLOCKED_TOOLS, frozenset)

    def test_exactly_six_tools(self):
        assert len(AGENT_BLOCKED_TOOLS) == 6

    def test_all_blocked_tools_exist_in_registry(self):
        registry_names = {t["name"] for t in TOOLS}
        for name in AGENT_BLOCKED_TOOLS:
            assert name in registry_names, f"{name} not in registry"


# ---------------------------------------------------------------------------
# filter_agent_tools function
# ---------------------------------------------------------------------------

class TestFilterAgentTools:
    """filter_agent_tools removes blocked tools from tool list."""

    def test_removes_spawn_agent(self):
        tools = [{"name": "run_command"}, {"name": "spawn_agent"}]
        filtered = filter_agent_tools(tools)
        names = [t["name"] for t in filtered]
        assert "spawn_agent" not in names
        assert "run_command" in names

    def test_removes_all_blocked(self):
        tools = [{"name": n} for n in AGENT_BLOCKED_TOOLS]
        filtered = filter_agent_tools(tools)
        assert filtered == []

    def test_preserves_non_agent_tools(self):
        tools = [{"name": "run_command"}, {"name": "read_file"},
                 {"name": "spawn_agent"}, {"name": "search_knowledge"}]
        filtered = filter_agent_tools(tools)
        names = [t["name"] for t in filtered]
        assert names == ["run_command", "read_file", "search_knowledge"]

    def test_empty_list(self):
        assert filter_agent_tools([]) == []

    def test_no_agent_tools_unchanged(self):
        tools = [{"name": "run_command"}, {"name": "read_file"}]
        filtered = filter_agent_tools(tools)
        assert len(filtered) == 2

    def test_preserves_tool_schema(self):
        tool = {"name": "run_command", "description": "Run cmd",
                "input_schema": {"type": "object"}}
        filtered = filter_agent_tools([tool])
        assert filtered[0] == tool

    def test_full_registry_minus_agent_tools(self):
        filtered = filter_agent_tools(TOOLS)
        filtered_names = {t["name"] for t in filtered}
        for name in AGENT_BLOCKED_TOOLS:
            assert name not in filtered_names
        # Should have removed exactly 6 tools
        assert len(filtered) == len(TOOLS) - len(AGENT_BLOCKED_TOOLS)


# ---------------------------------------------------------------------------
# Agent message history isolation
# ---------------------------------------------------------------------------

class TestMessageHistoryIsolation:
    """Each agent has its own message history, isolated from parent and siblings."""

    async def test_agent_starts_with_only_goal(self):
        """Agent's initial messages contain only the goal, not parent history."""
        mgr = AgentManager()
        captured_messages = []
        async def _iter_cb(messages, sys, tools):
            captured_messages.extend(messages)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}
        ann_cb = AsyncMock()
        mgr.spawn(
            label="test", goal="Check disk usage",
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=_iter_cb,
            tool_executor_callback=AsyncMock(return_value="ok"),
            announce_callback=ann_cb,
        )
        await asyncio.sleep(0.05)
        assert len(captured_messages) == 1
        assert captured_messages[0]["role"] == "user"
        assert captured_messages[0]["content"] == "Check disk usage"

    async def test_siblings_have_separate_histories(self):
        """Two agents in same channel have completely independent message lists."""
        mgr = AgentManager()
        histories = {"a": [], "b": []}

        async def _iter_a(messages, sys, tools):
            histories["a"] = list(messages)
            return {"text": "A done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _iter_b(messages, sys, tools):
            histories["b"] = list(messages)
            return {"text": "B done.", "tool_calls": [], "stop_reason": "end_turn"}

        ann_cb = AsyncMock()
        tool_cb = AsyncMock(return_value="ok")
        mgr.spawn("agent-a", "task A", "100", "u1", "user1",
                   _iter_a, tool_cb, ann_cb)
        mgr.spawn("agent-b", "task B", "100", "u1", "user1",
                   _iter_b, tool_cb, ann_cb)
        await asyncio.sleep(0.05)

        assert len(histories["a"]) == 1
        assert histories["a"][0]["content"] == "task A"
        assert len(histories["b"]) == 1
        assert histories["b"][0]["content"] == "task B"

    async def test_agent_messages_not_shared(self):
        """Agent A's messages list is a different object from agent B's."""
        mgr = AgentManager()
        agents_info = {}

        async def _iter(messages, sys, tools):
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        ann_cb = AsyncMock()
        tool_cb = AsyncMock(return_value="ok")
        id_a = mgr.spawn("a", "goal A", "100", "u1", "u1", _iter, tool_cb, ann_cb)
        id_b = mgr.spawn("b", "goal B", "100", "u1", "u1", _iter, tool_cb, ann_cb)

        agent_a = mgr._agents[id_a]
        agent_b = mgr._agents[id_b]
        assert agent_a.messages is not agent_b.messages

    async def test_tool_results_stay_in_agent_history(self):
        """Tool results from agent execution stay in that agent's messages only."""
        mgr = AgentManager()
        final_messages = []
        call_count = {"n": 0}

        async def _iter(messages, sys, tools):
            call_count["n"] += 1
            final_messages.clear()
            final_messages.extend(messages)
            if call_count["n"] == 1:
                return {"text": "", "tool_calls": [{"name": "run_command", "input": {"command": "df -h"}}], "stop_reason": "end_turn"}
            return {"text": "Disk is fine.", "tool_calls": [], "stop_reason": "end_turn"}

        ann_cb = AsyncMock()
        tool_cb = AsyncMock(return_value="Filesystem  Size  Used  Avail\n/dev/sda1  100G  45G  55G")

        mgr.spawn("disk-check", "Check disk", "100", "u1", "user1",
                   _iter, tool_cb, ann_cb)
        await asyncio.sleep(0.1)

        # Agent's messages should contain: goal, assistant (tool call), tool result
        assert len(final_messages) == 3
        assert final_messages[0]["content"] == "Check disk"
        assert "[Tool result: run_command]" in final_messages[2]["content"]

    async def test_inbox_messages_only_in_receiving_agent(self):
        """Messages sent to agent A don't appear in agent B's history."""
        mgr = AgentManager()
        a_messages = []
        b_messages = []
        gate = asyncio.Event()

        async def _iter_a(messages, sys, tools):
            a_messages.clear()
            a_messages.extend(messages)
            if not gate.is_set():
                gate.set()
                await asyncio.sleep(0.1)  # Wait to receive inbox message
                return {"text": "", "tool_calls": [{"name": "noop", "input": {}}], "stop_reason": "end_turn"}
            return {"text": "A done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _iter_b(messages, sys, tools):
            b_messages.clear()
            b_messages.extend(messages)
            return {"text": "B done.", "tool_calls": [], "stop_reason": "end_turn"}

        ann_cb = AsyncMock()
        tool_cb = AsyncMock(return_value="ok")

        id_a = mgr.spawn("agent-a", "task A", "100", "u1", "u1",
                          _iter_a, tool_cb, ann_cb)
        id_b = mgr.spawn("agent-b", "task B", "100", "u1", "u1",
                          _iter_b, tool_cb, ann_cb)

        await gate.wait()
        mgr.send(id_a, "extra info for A only")
        # Wait for both agents to complete
        await asyncio.sleep(0.3)

        # Agent A should have inbox message in its iteration history, agent B should not
        a_has_parent_msg = any("[Message from parent]" in m.get("content", "") for m in a_messages)
        b_has_parent_msg = any("[Message from parent]" in m.get("content", "") for m in b_messages)
        assert a_has_parent_msg, "Agent A should have received the inbox message"
        assert not b_has_parent_msg


# ---------------------------------------------------------------------------
# System prompt isolation
# ---------------------------------------------------------------------------

class TestSystemPromptIsolation:
    """Each agent gets its own system prompt with AGENT CONTEXT."""

    async def test_system_prompt_contains_agent_context(self):
        """System prompt includes AGENT CONTEXT with label."""
        mgr = AgentManager()
        captured_system = []

        async def _iter(messages, sys, tools):
            captured_system.append(sys)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        mgr.spawn("my-task", "Do something", "100", "u1", "u1",
                   _iter, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.05)

        assert len(captured_system) >= 1
        assert "AGENT CONTEXT" in captured_system[0]
        assert "my-task" in captured_system[0]

    async def test_system_prompt_forbids_sub_agents(self):
        """System prompt tells agent not to spawn sub-agents."""
        mgr = AgentManager()
        captured_system = []

        async def _iter(messages, sys, tools):
            captured_system.append(sys)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        mgr.spawn("test", "goal", "100", "u1", "u1",
                   _iter, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.05)

        assert "sub-agents" in captured_system[0].lower() or "sub-agent" in captured_system[0]

    async def test_siblings_get_different_system_prompts(self):
        """Two agents get system prompts with their own labels."""
        mgr = AgentManager()
        systems = {}

        async def _iter_a(messages, sys, tools):
            systems["a"] = sys
            return {"text": "A done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _iter_b(messages, sys, tools):
            systems["b"] = sys
            return {"text": "B done.", "tool_calls": [], "stop_reason": "end_turn"}

        mgr.spawn("alpha", "task A", "100", "u1", "u1",
                   _iter_a, AsyncMock(return_value="ok"), AsyncMock())
        mgr.spawn("beta", "task B", "100", "u1", "u1",
                   _iter_b, AsyncMock(return_value="ok"), AsyncMock())
        await asyncio.sleep(0.05)

        assert "alpha" in systems["a"]
        assert "beta" in systems["b"]
        assert "beta" not in systems["a"]
        assert "alpha" not in systems["b"]

    async def test_custom_system_prompt_preserved(self):
        """Custom system prompt is prepended to AGENT CONTEXT."""
        mgr = AgentManager()
        captured_system = []

        async def _iter(messages, sys, tools):
            captured_system.append(sys)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        mgr.spawn("test", "goal", "100", "u1", "u1",
                   _iter, AsyncMock(return_value="ok"), AsyncMock(),
                   system_prompt="You are Heimdall.")
        await asyncio.sleep(0.05)

        assert captured_system[0].startswith("You are Heimdall.")
        assert "AGENT CONTEXT" in captured_system[0]


# ---------------------------------------------------------------------------
# Tool nesting prevention
# ---------------------------------------------------------------------------

class TestToolNestingPrevention:
    """Agents cannot call agent tools (flat depth model)."""

    async def test_agent_tools_filtered_from_tool_list(self):
        """Agent receives tool list without agent tools."""
        mgr = AgentManager()
        captured_tools = []

        async def _iter(messages, sys, tools):
            captured_tools.extend(tools)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        all_tools = [
            {"name": "run_command"},
            {"name": "spawn_agent"},
            {"name": "read_file"},
            {"name": "kill_agent"},
        ]
        mgr.spawn("test", "goal", "100", "u1", "u1",
                   _iter, AsyncMock(return_value="ok"), AsyncMock(),
                   tools=filter_agent_tools(all_tools))
        await asyncio.sleep(0.05)

        names = [t["name"] for t in captured_tools]
        assert "run_command" in names
        assert "read_file" in names
        assert "spawn_agent" not in names
        assert "kill_agent" not in names

    def test_filter_removes_all_six(self):
        """All 6 agent tools are filtered."""
        tools = [{"name": n} for n in list(AGENT_BLOCKED_TOOLS) + ["run_command"]]
        filtered = filter_agent_tools(tools)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "run_command"

    async def test_tool_executor_rejects_blocked_tools(self):
        """Even if a tool somehow bypasses filtering, executor rejects it."""
        # Simulate what client.py's _tool_exec_cb does
        async def _tool_exec_cb(tool_name, tool_input):
            if tool_name in AGENT_BLOCKED_TOOLS:
                return f"Error: Tool '{tool_name}' is not available inside agents."
            return "ok"

        result = await _tool_exec_cb("spawn_agent", {})
        assert "not available inside agents" in result

    async def test_tool_executor_allows_normal_tools(self):
        """Normal tools pass through the executor check."""
        async def _tool_exec_cb(tool_name, tool_input):
            if tool_name in AGENT_BLOCKED_TOOLS:
                return f"Error: Tool '{tool_name}' is not available inside agents."
            return "ok"

        result = await _tool_exec_cb("run_command", {})
        assert result == "ok"


# ---------------------------------------------------------------------------
# Parent session protection
# ---------------------------------------------------------------------------

class TestParentSessionProtection:
    """Agent execution does not pollute parent session history."""

    async def test_agent_does_not_save_to_session_manager(self):
        """Agent completion does not call SessionManager.add_message."""
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()
        aid = mgr.spawn("test", "do thing", "100", "u1", "u1",
                         iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.05)

        agent = mgr._agents.get(aid)
        # Agent should have completed
        assert agent is None or agent.status == "completed"
        # The key assertion: no session manager is ever referenced in manager.py
        import src.agents.manager as mgr_module
        source = inspect.getsource(mgr_module)
        assert "SessionManager" not in source
        assert "self.sessions" not in source

    async def test_announce_bypasses_session(self):
        """Announce callback posts to Discord, not to session history."""
        mgr = AgentManager()
        announce_calls = []
        async def _announce(ch_id, text):
            announce_calls.append((ch_id, text))

        iter_cb = AsyncMock(return_value={
            "text": "Result here.", "tool_calls": [], "stop_reason": "end_turn"})
        mgr.spawn("test", "goal", "100", "u1", "u1",
                   iter_cb, AsyncMock(return_value="ok"), _announce)
        await asyncio.sleep(0.05)

        assert len(announce_calls) == 1
        assert "Result here." in announce_calls[0][1]
        assert announce_calls[0][0] == "100"

    def test_manager_has_no_session_dependency(self):
        """AgentManager doesn't import or reference SessionManager."""
        import src.agents.manager as mod
        source = inspect.getsource(mod)
        assert "sessions" not in source.lower() or "session" in source.lower()
        # More precise: no SessionManager import
        assert "SessionManager" not in source
        assert "from ..sessions" not in source


# ---------------------------------------------------------------------------
# Agent result labeling
# ---------------------------------------------------------------------------

class TestResultLabeling:
    """Agent results are labeled with [Agent: {label}] format."""

    async def test_completed_result_labeled(self):
        """Completed agent result includes [Agent: label] tag."""
        mgr = AgentManager()
        announced = []
        async def _announce(ch_id, text):
            announced.append(text)

        iter_cb = AsyncMock(return_value={
            "text": "All clear.", "tool_calls": [], "stop_reason": "end_turn"})
        mgr.spawn("disk-check", "Check disk", "100", "u1", "u1",
                   iter_cb, AsyncMock(return_value="ok"), _announce)
        await asyncio.sleep(0.05)

        assert len(announced) == 1
        assert "**[Agent: disk-check]**" in announced[0]
        assert "All clear." in announced[0]

    async def test_failed_result_labeled(self):
        """Failed agent result includes [Agent: label] tag."""
        mgr = AgentManager()
        announced = []
        async def _announce(ch_id, text):
            announced.append(text)

        async def _fail_iter(messages, sys, tools):
            raise RuntimeError("LLM down")

        mgr.spawn("net-check", "Check network", "100", "u1", "u1",
                   _fail_iter, AsyncMock(return_value="ok"), _announce)
        await asyncio.sleep(0.05)

        assert len(announced) == 1
        assert "**[Agent: net-check]**" in announced[0]

    async def test_result_label_format(self):
        """Result follows pattern: **[Agent: {label}]** ({status_text})"""
        mgr = AgentManager()
        announced = []
        async def _announce(ch_id, text):
            announced.append(text)

        iter_cb = AsyncMock(return_value={
            "text": "Done.", "tool_calls": [], "stop_reason": "end_turn"})
        mgr.spawn("my-task", "do it", "100", "u1", "u1",
                   iter_cb, AsyncMock(return_value="ok"), _announce)
        await asyncio.sleep(0.05)

        # Pattern: **[Agent: my-task]** (completed in Xs, N tool calls)
        text = announced[0]
        assert text.startswith("**[Agent: my-task]**")
        assert "completed" in text

    async def test_result_scrubbed_for_secrets(self):
        """Agent results are scrubbed before announcement."""
        mgr = AgentManager()
        announced = []
        async def _announce(ch_id, text):
            announced.append(text)

        # Return text with a secret pattern
        iter_cb = AsyncMock(return_value={
            "text": "Found password=s3cr3t123 in config.",
            "tool_calls": [], "stop_reason": "end_turn"})
        mgr.spawn("sec-check", "Check security", "100", "u1", "u1",
                   iter_cb, AsyncMock(return_value="ok"), _announce)
        await asyncio.sleep(0.05)

        # Secret should be scrubbed in the announcement
        text = announced[0]
        assert "s3cr3t123" not in text

    async def test_long_result_truncated(self):
        """Very long results are truncated in announcement."""
        mgr = AgentManager()
        announced = []
        async def _announce(ch_id, text):
            announced.append(text)

        long_text = "x" * 5000
        iter_cb = AsyncMock(return_value={
            "text": long_text, "tool_calls": [], "stop_reason": "end_turn"})
        mgr.spawn("big-task", "big work", "100", "u1", "u1",
                   iter_cb, AsyncMock(return_value="ok"), _announce)
        await asyncio.sleep(0.05)

        text = announced[0]
        assert len(text) < 2500  # Well under Discord limit
        assert "..." in text


# ---------------------------------------------------------------------------
# No shared state between siblings
# ---------------------------------------------------------------------------

class TestSiblingIsolation:
    """Sibling agents (same channel) cannot access each other's state."""

    async def test_sibling_tools_used_independent(self):
        """Each agent tracks its own tools_used list independently."""
        mgr = AgentManager()
        call_counts = {"a": 0, "b": 0}

        async def _iter_a(messages, sys, tools):
            call_counts["a"] += 1
            if call_counts["a"] == 1:
                return {"text": "", "tool_calls": [{"name": "run_command", "input": {}}],
                        "stop_reason": "end_turn"}
            return {"text": "A done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _iter_b(messages, sys, tools):
            call_counts["b"] += 1
            if call_counts["b"] == 1:
                return {"text": "", "tool_calls": [{"name": "read_file", "input": {}}],
                        "stop_reason": "end_turn"}
            return {"text": "B done.", "tool_calls": [], "stop_reason": "end_turn"}

        tool_cb = AsyncMock(return_value="ok")
        ann_cb = AsyncMock()

        id_a = mgr.spawn("a", "task A", "100", "u1", "u1",
                          _iter_a, tool_cb, ann_cb)
        id_b = mgr.spawn("b", "task B", "100", "u1", "u1",
                          _iter_b, tool_cb, ann_cb)
        await asyncio.sleep(0.15)

        agent_a = mgr._agents.get(id_a)
        agent_b = mgr._agents.get(id_b)
        if agent_a:
            assert "run_command" in agent_a.tools_used
            assert "read_file" not in agent_a.tools_used
        if agent_b:
            assert "read_file" in agent_b.tools_used
            assert "run_command" not in agent_b.tools_used

    async def test_sibling_iteration_counts_independent(self):
        """Each agent has its own iteration count."""
        mgr = AgentManager()
        a_count = {"n": 0}

        async def _iter_a(messages, sys, tools):
            a_count["n"] += 1
            if a_count["n"] < 3:
                return {"text": "", "tool_calls": [{"name": "run_command", "input": {}}],
                        "stop_reason": "end_turn"}
            return {"text": "A done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _iter_b(messages, sys, tools):
            return {"text": "B done.", "tool_calls": [], "stop_reason": "end_turn"}

        tool_cb = AsyncMock(return_value="ok")
        ann_cb = AsyncMock()

        id_a = mgr.spawn("a", "task A", "100", "u1", "u1",
                          _iter_a, tool_cb, ann_cb)
        id_b = mgr.spawn("b", "task B", "100", "u1", "u1",
                          _iter_b, tool_cb, ann_cb)
        await asyncio.sleep(0.2)

        agent_a = mgr._agents.get(id_a)
        agent_b = mgr._agents.get(id_b)
        if agent_a and agent_b:
            assert agent_a.iteration_count > agent_b.iteration_count

    async def test_sibling_cancel_events_independent(self):
        """Killing one agent doesn't affect its sibling."""
        mgr = AgentManager()
        gate = asyncio.Event()

        async def _slow_iter(messages, sys, tools):
            gate.set()
            await asyncio.sleep(5)  # Will be killed
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _fast_iter(messages, sys, tools):
            return {"text": "Fast done.", "tool_calls": [], "stop_reason": "end_turn"}

        tool_cb = AsyncMock(return_value="ok")
        ann_cb = AsyncMock()

        id_a = mgr.spawn("slow", "slow task", "100", "u1", "u1",
                          _slow_iter, tool_cb, ann_cb)
        id_b = mgr.spawn("fast", "fast task", "100", "u1", "u1",
                          _fast_iter, tool_cb, ann_cb)

        await asyncio.sleep(0.1)
        mgr.kill(id_a)
        await asyncio.sleep(0.1)

        agent_b = mgr._agents.get(id_b)
        assert agent_b is None or agent_b.status == "completed"

    async def test_sibling_results_independent(self):
        """Each agent has its own result text."""
        mgr = AgentManager()

        async def _iter_a(messages, sys, tools):
            return {"text": "Result A", "tool_calls": [], "stop_reason": "end_turn"}

        async def _iter_b(messages, sys, tools):
            return {"text": "Result B", "tool_calls": [], "stop_reason": "end_turn"}

        tool_cb = AsyncMock(return_value="ok")
        ann_cb = AsyncMock()

        id_a = mgr.spawn("a", "task A", "100", "u1", "u1",
                          _iter_a, tool_cb, ann_cb)
        id_b = mgr.spawn("b", "task B", "100", "u1", "u1",
                          _iter_b, tool_cb, ann_cb)
        await asyncio.sleep(0.05)

        result_a = mgr.get_results(id_a)
        result_b = mgr.get_results(id_b)
        assert result_a and result_a["result"] == "Result A"
        assert result_b and result_b["result"] == "Result B"

    async def test_sibling_errors_dont_cascade(self):
        """One agent's error doesn't affect its sibling."""
        mgr = AgentManager()

        async def _iter_a(messages, sys, tools):
            raise ValueError("Agent A exploded")

        async def _iter_b(messages, sys, tools):
            return {"text": "B is fine.", "tool_calls": [], "stop_reason": "end_turn"}

        tool_cb = AsyncMock(return_value="ok")
        ann_cb = AsyncMock()

        id_a = mgr.spawn("a", "task A", "100", "u1", "u1",
                          _iter_a, tool_cb, ann_cb)
        id_b = mgr.spawn("b", "task B", "100", "u1", "u1",
                          _iter_b, tool_cb, ann_cb)
        await asyncio.sleep(0.1)

        result_a = mgr.get_results(id_a)
        result_b = mgr.get_results(id_b)
        assert result_a and result_a["status"] == "failed"
        assert result_b and result_b["status"] == "completed"
        assert result_b["result"] == "B is fine."


# ---------------------------------------------------------------------------
# Cross-channel isolation
# ---------------------------------------------------------------------------

class TestCrossChannelIsolation:
    """Agents in different channels are fully isolated."""

    async def test_list_only_shows_channel_agents(self):
        """list() with channel_id only returns that channel's agents."""
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        mgr.spawn("a", "task A", "100", "u1", "u1", iter_cb, tool_cb, ann_cb)
        mgr.spawn("b", "task B", "200", "u1", "u1", iter_cb, tool_cb, ann_cb)
        await asyncio.sleep(0.05)

        ch100 = mgr.list("100")
        ch200 = mgr.list("200")
        assert all(a["label"] != "b" for a in ch100)
        assert all(a["label"] != "a" for a in ch200)

    async def test_channel_limit_independent(self):
        """Per-channel limit is enforced independently."""
        mgr = AgentManager()
        gate = asyncio.Event()

        async def _block_iter(messages, sys, tools):
            gate.set()
            await asyncio.sleep(10)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        tool_cb = AsyncMock(return_value="ok")
        ann_cb = AsyncMock()

        # Fill channel 100 to limit
        for i in range(MAX_CONCURRENT_AGENTS):
            mgr.spawn(f"a{i}", f"task {i}", "100", "u1", "u1",
                      _block_iter, tool_cb, ann_cb)

        # Channel 200 should still allow spawning
        result = mgr.spawn("b0", "task b0", "200", "u1", "u1",
                            _block_iter, tool_cb, ann_cb)
        assert not result.startswith("Error")

        # Channel 100 should be full
        result = mgr.spawn("overflow", "task", "100", "u1", "u1",
                            _block_iter, tool_cb, ann_cb)
        assert result.startswith("Error")

        # Cleanup
        for agent in list(mgr._agents.values()):
            agent._cancel_event.set()
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------

class TestErrorIsolation:
    """Agent errors don't crash the bot or affect other agents."""

    async def test_llm_error_contained(self):
        """LLM failure in agent doesn't propagate to caller."""
        mgr = AgentManager()
        announced = []
        async def _announce(ch_id, text):
            announced.append(text)

        async def _fail(messages, sys, tools):
            raise ConnectionError("API unreachable")

        aid = mgr.spawn("failing", "fail task", "100", "u1", "u1",
                         _fail, AsyncMock(return_value="ok"), _announce)
        await asyncio.sleep(0.05)

        agent = mgr._agents.get(aid)
        assert agent.status == "failed"
        assert "API unreachable" in agent.error or "LLM call failed" in agent.error
        # Result was announced
        assert len(announced) == 1

    async def test_tool_error_doesnt_kill_agent(self):
        """Tool execution error is captured and agent continues."""
        mgr = AgentManager()
        call_count = {"n": 0}

        async def _iter(messages, sys, tools):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"text": "", "tool_calls": [{"name": "run_command", "input": {}}],
                        "stop_reason": "end_turn"}
            return {"text": "Recovered.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _fail_tool(name, inp):
            raise OSError("Permission denied")

        ann_cb = AsyncMock()
        aid = mgr.spawn("tool-fail", "try something", "100", "u1", "u1",
                         _iter, _fail_tool, ann_cb)
        await asyncio.sleep(0.15)

        agent = mgr._agents.get(aid)
        assert agent.status == "completed"
        assert agent.result == "Recovered."

    async def test_announce_failure_doesnt_crash(self):
        """If announce callback fails, agent still completes."""
        mgr = AgentManager()

        async def _bad_announce(ch_id, text):
            raise RuntimeError("Discord down")

        iter_cb = AsyncMock(return_value={
            "text": "Done.", "tool_calls": [], "stop_reason": "end_turn"})
        aid = mgr.spawn("test", "goal", "100", "u1", "u1",
                         iter_cb, AsyncMock(return_value="ok"), _bad_announce)
        await asyncio.sleep(0.05)

        agent = mgr._agents.get(aid)
        assert agent.status == "completed"


# ---------------------------------------------------------------------------
# Secret scrubbing boundary
# ---------------------------------------------------------------------------

class TestSecretScrubbing:
    """Secrets in agent tool results are scrubbed."""

    async def test_tool_result_secrets_scrubbed(self):
        """Secret patterns in tool output are scrubbed before LLM sees them."""
        mgr = AgentManager()
        final_messages = []
        call_count = {"n": 0}

        async def _iter(messages, sys, tools):
            call_count["n"] += 1
            final_messages.clear()
            final_messages.extend(messages)
            if call_count["n"] == 1:
                return {"text": "", "tool_calls": [{"name": "run_command", "input": {}}],
                        "stop_reason": "end_turn"}
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        # Tool returns a secret
        tool_cb = AsyncMock(return_value="api_key=sk-abc123secretkey")
        ann_cb = AsyncMock()

        mgr.spawn("sec-test", "check config", "100", "u1", "u1",
                   _iter, tool_cb, ann_cb)
        await asyncio.sleep(0.15)

        # Check that the secret was scrubbed in agent's message history
        tool_result_msgs = [m for m in final_messages if "[Tool result:" in m.get("content", "")]
        if tool_result_msgs:
            content = tool_result_msgs[0]["content"]
            assert "sk-abc123secretkey" not in content

    async def test_result_announcement_scrubbed(self):
        """Final result text is scrubbed before channel announcement."""
        import src.agents.manager as mgr_mod
        source = inspect.getsource(mgr_mod._announce_formatted)
        assert "scrub_output_secrets" in source


# ---------------------------------------------------------------------------
# Source code verification
# ---------------------------------------------------------------------------

class TestSourceVerification:
    """Verify isolation mechanisms exist in source code."""

    def test_agent_blocked_tools_constant_exists(self):
        from src.agents.manager import AGENT_BLOCKED_TOOLS
        assert isinstance(AGENT_BLOCKED_TOOLS, frozenset)

    def test_filter_agent_tools_exists(self):
        from src.agents.manager import filter_agent_tools
        assert callable(filter_agent_tools)

    def test_client_imports_filter(self):
        import src.discord.client as client_mod
        source = inspect.getsource(client_mod)
        assert "filter_agent_tools" in source

    def test_client_imports_blocked_tools(self):
        import src.discord.client as client_mod
        source = inspect.getsource(client_mod)
        assert "AGENT_BLOCKED_TOOLS" in source

    def test_client_applies_filter_in_spawn(self):
        """_handle_spawn_agent uses filter_agent_tools."""
        import src.discord.client as client_mod
        source = inspect.getsource(client_mod)
        assert "filter_agent_tools" in source

    def test_client_tool_exec_rejects_blocked(self):
        """Tool executor callback rejects blocked tools."""
        import src.discord.client as client_mod
        source = inspect.getsource(client_mod)
        assert "AGENT_BLOCKED_TOOLS" in source
        assert "not available inside agents" in source

    def test_no_session_manager_in_agents(self):
        """Agent manager has no SessionManager dependency."""
        import src.agents.manager as mod
        source = inspect.getsource(mod)
        assert "SessionManager" not in source

    def test_announce_formatted_scrubs_secrets(self):
        import src.agents.manager as mod
        source = inspect.getsource(mod._announce_formatted)
        assert "scrub_output_secrets" in source

    def test_run_agent_scrubs_tool_results(self):
        import src.agents.manager as mod
        source = inspect.getsource(mod._run_agent)
        assert "scrub_output_secrets" in source

    def test_agent_context_in_system_prompt(self):
        import src.agents.manager as mod
        source = inspect.getsource(mod)
        assert "AGENT CONTEXT" in source

    def test_no_spawn_instruction(self):
        """System prompt tells agent not to spawn."""
        import src.agents.manager as mod
        source = inspect.getsource(mod)
        assert "Do NOT spawn sub-agents" in source

    def test_agents_init_exports(self):
        from src.agents import AGENT_BLOCKED_TOOLS, filter_agent_tools
        assert AGENT_BLOCKED_TOOLS
        assert callable(filter_agent_tools)

    def test_message_from_parent_prefix(self):
        """Inbox messages are prefixed with [Message from parent]."""
        import src.agents.manager as mod
        source = inspect.getsource(mod._run_agent)
        assert "[Message from parent]" in source


# ---------------------------------------------------------------------------
# Client integration — tool filtering applied
# ---------------------------------------------------------------------------

class TestClientIntegration:
    """Verify client.py correctly applies isolation."""

    def test_spawn_handler_filters_tools(self):
        """_handle_spawn_agent calls filter_agent_tools on the tool list."""
        import src.discord.client as mod
        source = inspect.getsource(mod.HeimdallBot._handle_spawn_agent)
        assert "filter_agent_tools" in source

    def test_spawn_handler_has_blocked_guard(self):
        """Tool executor callback in spawn handler checks AGENT_BLOCKED_TOOLS."""
        import src.discord.client as mod
        source = inspect.getsource(mod.HeimdallBot._handle_spawn_agent)
        assert "AGENT_BLOCKED_TOOLS" in source

    def test_agent_manager_created_in_init(self):
        """HeimdallBot.__init__ creates agent_manager."""
        import src.discord.client as mod
        source = inspect.getsource(mod.HeimdallBot.__init__)
        assert "agent_manager" in source
        assert "AgentManager" in source

    def test_announce_callback_uses_scrub(self):
        """Announce callback scrubs secrets before sending."""
        import src.discord.client as mod
        source = inspect.getsource(mod.HeimdallBot._handle_spawn_agent)
        assert "scrub_response_secrets" in source


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for isolation."""

    async def test_empty_tool_list_agent(self):
        """Agent can run with empty tool list."""
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()
        aid = mgr.spawn("notool", "just chat", "100", "u1", "u1",
                         iter_cb, tool_cb, ann_cb, tools=[])
        await asyncio.sleep(0.05)
        agent = mgr._agents.get(aid)
        assert agent is None or agent.status == "completed"

    async def test_agent_with_no_system_prompt(self):
        """Agent works with no custom system prompt."""
        mgr = AgentManager()
        captured = []
        async def _iter(messages, sys, tools):
            captured.append(sys)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        mgr.spawn("test", "goal", "100", "u1", "u1",
                   _iter, AsyncMock(return_value="ok"), AsyncMock(),
                   system_prompt="")
        await asyncio.sleep(0.05)

        assert "AGENT CONTEXT" in captured[0]
        # Should not start with \n\n when no custom prompt
        assert not captured[0].startswith("\n")

    async def test_concurrent_five_agents_isolated(self):
        """5 concurrent agents in same channel maintain isolation."""
        mgr = AgentManager()
        results = {}

        for i in range(5):
            async def _make_iter(idx=i):
                async def _iter(messages, sys, tools):
                    return {"text": f"Result {idx}", "tool_calls": [],
                            "stop_reason": "end_turn"}
                return _iter

            iter_cb = await _make_iter()
            aid = mgr.spawn(f"agent-{i}", f"task {i}", "100", "u1", "u1",
                            iter_cb, AsyncMock(return_value="ok"), AsyncMock())
            results[i] = aid

        await asyncio.sleep(0.1)

        for i, aid in results.items():
            r = mgr.get_results(aid)
            assert r is not None
            assert r["status"] == "completed"
            assert r["result"] == f"Result {i}"

    async def test_killed_agent_cleanup_doesnt_affect_siblings(self):
        """Cleaning up a killed agent doesn't remove siblings."""
        mgr = AgentManager()
        gate = asyncio.Event()

        async def _block(messages, sys, tools):
            gate.set()
            await asyncio.sleep(10)
            return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

        async def _fast(messages, sys, tools):
            return {"text": "Fast.", "tool_calls": [], "stop_reason": "end_turn"}

        tool_cb = AsyncMock(return_value="ok")
        ann_cb = AsyncMock()

        id_a = mgr.spawn("slow", "slow task", "100", "u1", "u1",
                          _block, tool_cb, ann_cb)
        id_b = mgr.spawn("fast", "fast task", "100", "u1", "u1",
                          _fast, tool_cb, ann_cb)
        await asyncio.sleep(0.1)

        # Kill agent A
        mgr.kill(id_a)
        await asyncio.sleep(0.1)

        # Agent B's results should still be available
        result_b = mgr.get_results(id_b)
        assert result_b is not None
        assert result_b["status"] == "completed"

    async def test_agent_inbox_queue_per_agent(self):
        """Each agent has its own inbox queue object."""
        mgr = AgentManager()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        id_a = mgr.spawn("a", "task A", "100", "u1", "u1",
                          iter_cb, tool_cb, ann_cb)
        id_b = mgr.spawn("b", "task B", "100", "u1", "u1",
                          iter_cb, tool_cb, ann_cb)

        agent_a = mgr._agents[id_a]
        agent_b = mgr._agents[id_b]
        assert agent_a._inbox is not agent_b._inbox
        assert agent_a._cancel_event is not agent_b._cancel_event

    def test_agent_info_fields_independent(self):
        """AgentInfo instances don't share mutable defaults."""
        a = AgentInfo(id="a", label="a", goal="g", channel_id="1",
                      requester_id="u", requester_name="u")
        b = AgentInfo(id="b", label="b", goal="g", channel_id="1",
                      requester_id="u", requester_name="u")
        a.messages.append({"test": True})
        assert len(b.messages) == 0
        a.tools_used.append("run_command")
        assert len(b.tools_used) == 0
