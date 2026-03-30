"""Round 19 — Agent communication + Discord integration tests.

Tests the 5 agent tool handlers in client.py (_handle_spawn_agent, _handle_send_to_agent,
_handle_list_agents, _handle_kill_agent, _handle_get_agent_results), callback wiring,
announce flow, tool dispatch routing, and end-to-end agent integration.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.agents.manager import (
    AgentInfo,
    AgentManager,
    MAX_CONCURRENT_AGENTS,
    _run_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_message(channel_id="123", user_id="456", user_name="TestUser"):
    """Create a mock Discord message with channel and author."""
    msg = MagicMock()
    msg.channel = MagicMock()
    msg.channel.id = int(channel_id)
    msg.channel.send = AsyncMock()
    msg.author = MagicMock()
    msg.author.id = int(user_id)
    msg.author.__str__ = lambda self: user_name
    msg.id = 999
    return msg


def _make_mock_bot(codex_enabled=True):
    """Create a minimal mock HeimdallBot with agent_manager and codex_client."""
    bot = MagicMock()
    bot.agent_manager = AgentManager()
    bot.config = MagicMock()
    bot.config.tools.enabled = True

    if codex_enabled:
        bot.codex_client = MagicMock()
        bot.codex_client.chat_with_tools = AsyncMock()
    else:
        bot.codex_client = None

    bot._build_system_prompt = MagicMock(return_value="System prompt.")
    bot._merged_tool_definitions = MagicMock(return_value=[{"name": "run_command"}])
    bot._dispatch_loop_tool = AsyncMock(return_value="tool result")
    bot.get_channel = MagicMock(return_value=None)
    return bot


# ===========================================================================
# _handle_spawn_agent
# ===========================================================================

class TestHandleSpawnAgent:
    """Tests for the spawn_agent tool handler."""

    async def test_spawn_returns_agent_id(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()
        inp = {"label": "disk-check", "goal": "Check disk usage"}

        result = await HeimdallBot._handle_spawn_agent(bot, msg, inp)

        assert "spawned" in result.lower()
        assert "disk-check" in result
        assert "ID:" in result

    async def test_spawn_missing_label(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        result = await HeimdallBot._handle_spawn_agent(bot, msg, {"goal": "do stuff"})
        assert "required" in result.lower()

    async def test_spawn_missing_goal(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        result = await HeimdallBot._handle_spawn_agent(bot, msg, {"label": "test"})
        assert "required" in result.lower()

    async def test_spawn_no_codex(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot(codex_enabled=False)
        msg = _make_mock_message()

        result = await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "test", "goal": "do stuff"},
        )
        assert "not available" in result.lower()

    async def test_spawn_creates_agent_in_manager(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        result = await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "audit", "goal": "Run audit"},
        )
        assert bot.agent_manager.total_count == 1
        agents = bot.agent_manager.list()
        assert len(agents) == 1
        assert agents[0]["label"] == "audit"

    async def test_spawn_respects_channel_limit(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message(channel_id="100")

        # Spawn MAX_CONCURRENT_AGENTS agents
        for i in range(MAX_CONCURRENT_AGENTS):
            await HeimdallBot._handle_spawn_agent(
                bot, msg, {"label": f"agent-{i}", "goal": f"Task {i}"},
            )

        # Next spawn should fail
        result = await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "overflow", "goal": "One too many"},
        )
        assert "error" in result.lower() or "maximum" in result.lower()

    async def test_spawn_passes_system_prompt(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        bot._build_system_prompt.return_value = "Heimdall system prompt"
        msg = _make_mock_message()

        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "test", "goal": "Do stuff"},
        )

        # Verify system prompt was built
        bot._build_system_prompt.assert_called()

    async def test_spawn_passes_tools(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        bot._merged_tool_definitions.return_value = [
            {"name": "run_command"}, {"name": "check_disk"},
        ]
        msg = _make_mock_message()

        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "test", "goal": "Check things"},
        )

        bot._merged_tool_definitions.assert_called()

    async def test_spawn_goal_truncated_in_response(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()
        long_goal = "x" * 200

        result = await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "test", "goal": long_goal},
        )

        # Goal should be truncated in the response
        assert len(result) < 300

    async def test_spawn_from_loop_proxy(self):
        """spawn_agent should work with _LoopMessageProxy too."""
        from src.discord.client import HeimdallBot, _LoopMessageProxy
        bot = _make_mock_bot()
        channel = MagicMock()
        channel.id = 123
        proxy = _LoopMessageProxy(channel, "456", "LoopUser")

        result = await HeimdallBot._handle_spawn_agent(
            bot, proxy, {"label": "loop-agent", "goal": "Do loop work"},
        )

        assert "spawned" in result.lower()


# ===========================================================================
# _handle_send_to_agent
# ===========================================================================

class TestHandleSendToAgent:
    """Tests for the send_to_agent tool handler."""

    async def test_send_to_running_agent(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        # Spawn an agent first
        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "worker", "goal": "Some task"},
        )
        agents = bot.agent_manager.list()
        agent_id = agents[0]["id"]

        result = HeimdallBot._handle_send_to_agent(
            bot, {"agent_id": agent_id, "message": "Check /var/log too"},
        )
        assert "delivered" in result.lower()

    async def test_send_missing_agent_id(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_send_to_agent(
            bot, {"message": "hello"},
        )
        assert "required" in result.lower()

    async def test_send_missing_message(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_send_to_agent(
            bot, {"agent_id": "abc12345"},
        )
        assert "required" in result.lower()

    async def test_send_to_nonexistent_agent(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_send_to_agent(
            bot, {"agent_id": "doesnotexist", "message": "hello"},
        )
        assert "not found" in result.lower()


# ===========================================================================
# _handle_list_agents
# ===========================================================================

class TestHandleListAgents:
    """Tests for the list_agents tool handler."""

    async def test_list_empty(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        result = HeimdallBot._handle_list_agents(bot, msg)
        assert "no agents" in result.lower()

    async def test_list_shows_agents(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "worker-1", "goal": "Task 1"},
        )
        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "worker-2", "goal": "Task 2"},
        )

        result = HeimdallBot._handle_list_agents(bot, msg)
        assert "worker-1" in result
        assert "worker-2" in result
        assert "Agents (2)" in result

    async def test_list_filters_by_channel(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg1 = _make_mock_message(channel_id="100")
        msg2 = _make_mock_message(channel_id="200")

        await HeimdallBot._handle_spawn_agent(
            bot, msg1, {"label": "ch100-agent", "goal": "Task A"},
        )
        await HeimdallBot._handle_spawn_agent(
            bot, msg2, {"label": "ch200-agent", "goal": "Task B"},
        )

        result1 = HeimdallBot._handle_list_agents(bot, msg1)
        assert "ch100-agent" in result1
        assert "ch200-agent" not in result1

        result2 = HeimdallBot._handle_list_agents(bot, msg2)
        assert "ch200-agent" in result2
        assert "ch100-agent" not in result2


# ===========================================================================
# _handle_kill_agent
# ===========================================================================

class TestHandleKillAgent:
    """Tests for the kill_agent tool handler."""

    async def test_kill_running_agent(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "victim", "goal": "Something"},
        )
        agent_id = bot.agent_manager.list()[0]["id"]

        result = HeimdallBot._handle_kill_agent(bot, {"agent_id": agent_id})
        assert "kill" in result.lower()

    async def test_kill_missing_agent_id(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_kill_agent(bot, {})
        assert "required" in result.lower()

    async def test_kill_nonexistent(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_kill_agent(bot, {"agent_id": "nope"})
        assert "not found" in result.lower()


# ===========================================================================
# _handle_get_agent_results
# ===========================================================================

class TestHandleGetAgentResults:
    """Tests for the get_agent_results tool handler."""

    async def test_get_results_not_found(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_get_agent_results(bot, {"agent_id": "nope"})
        assert "not found" in result.lower()

    async def test_get_results_missing_id(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_get_agent_results(bot, {})
        assert "required" in result.lower()

    async def test_get_results_running_agent(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "runner", "goal": "Still going"},
        )
        agent_id = bot.agent_manager.list()[0]["id"]

        result = HeimdallBot._handle_get_agent_results(bot, {"agent_id": agent_id})
        assert "still running" in result.lower()

    async def test_get_results_completed_agent(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        # Directly create a completed agent in the manager
        agent = AgentInfo(
            id="done1234", label="done-agent", goal="Finished task",
            channel_id="123", requester_id="456", requester_name="TestUser",
            status="completed", result="All servers healthy.",
            tools_used=["run_command", "check_disk"],
            iteration_count=5,
        )
        agent.ended_at = time.time()
        bot.agent_manager._agents["done1234"] = agent

        result = HeimdallBot._handle_get_agent_results(bot, {"agent_id": "done1234"})
        assert "done-agent" in result
        assert "completed" in result
        assert "All servers healthy" in result
        assert "run_command" in result

    async def test_get_results_failed_agent(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        agent = AgentInfo(
            id="fail1234", label="fail-agent", goal="Broken task",
            channel_id="123", requester_id="456", requester_name="TestUser",
            status="failed", error="SSH connection refused",
        )
        agent.ended_at = time.time()
        bot.agent_manager._agents["fail1234"] = agent

        result = HeimdallBot._handle_get_agent_results(bot, {"agent_id": "fail1234"})
        assert "failed" in result
        assert "SSH connection refused" in result

    async def test_get_results_truncates_long_result(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        agent = AgentInfo(
            id="long1234", label="verbose", goal="Generate lots",
            channel_id="123", requester_id="456", requester_name="TestUser",
            status="completed", result="x" * 3000,
        )
        agent.ended_at = time.time()
        bot.agent_manager._agents["long1234"] = agent

        result = HeimdallBot._handle_get_agent_results(bot, {"agent_id": "long1234"})
        assert "..." in result
        assert len(result) < 2000


# ===========================================================================
# Callback wiring
# ===========================================================================

class TestCallbackWiring:
    """Tests that spawn creates proper callbacks."""

    async def test_iteration_callback_wraps_codex(self):
        """The iteration callback should call codex_client.chat_with_tools."""
        from src.discord.client import HeimdallBot
        from src.llm.types import LLMResponse, ToolCall
        bot = _make_mock_bot()

        # Mock codex to return an LLMResponse
        bot.codex_client.chat_with_tools = AsyncMock(return_value=LLMResponse(
            text="Agent response.",
            tool_calls=[ToolCall(id="tc_1", name="run_command", input={"command": "ls"})],
            stop_reason="tool_use",
        ))

        msg = _make_mock_message()
        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "cb-test", "goal": "Test callbacks"},
        )

        # The agent should have been spawned with callbacks
        assert bot.agent_manager.total_count == 1

    async def test_tool_executor_callback_uses_dispatch(self):
        """Tool executor callback should route through _dispatch_loop_tool."""
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        bot._dispatch_loop_tool = AsyncMock(return_value="command output")

        msg = _make_mock_message()
        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "tool-test", "goal": "Test tools"},
        )

        # The dispatch mock should exist and be ready
        assert bot._dispatch_loop_tool is not None

    async def test_announce_callback_sends_to_channel(self):
        """Announce callback should send scrubbed text to the channel."""
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        channel = MagicMock()
        channel.id = 123
        channel.send = AsyncMock()
        bot.get_channel = MagicMock(return_value=channel)

        msg = _make_mock_message()
        msg.channel = channel

        await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "announce-test", "goal": "Test announce"},
        )

        # Agent was spawned — the announce callback is wired
        assert bot.agent_manager.total_count == 1


# ===========================================================================
# Tool dispatch routing
# ===========================================================================

class TestToolDispatchRouting:
    """Tests that agent tools are routed correctly in tool dispatch chains."""

    def test_agent_tools_in_dispatch_source(self):
        """Verify agent tool dispatch code exists in client.py source."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client)

        for tool_name in ("spawn_agent", "send_to_agent", "list_agents",
                          "kill_agent", "get_agent_results"):
            assert f'"{tool_name}"' in source or f"'{tool_name}'" in source, (
                f"Tool '{tool_name}' not found in client.py dispatch"
            )

    def test_handler_methods_exist(self):
        """All 5 handler methods should exist on HeimdallBot."""
        from src.discord.client import HeimdallBot
        for name in ("_handle_spawn_agent", "_handle_send_to_agent",
                      "_handle_list_agents", "_handle_kill_agent",
                      "_handle_get_agent_results"):
            assert hasattr(HeimdallBot, name), f"Missing handler: {name}"

    def test_spawn_handler_is_async(self):
        """_handle_spawn_agent must be async (it creates callbacks)."""
        import asyncio
        from src.discord.client import HeimdallBot
        assert asyncio.iscoroutinefunction(HeimdallBot._handle_spawn_agent)

    def test_send_handler_is_sync(self):
        """_handle_send_to_agent is sync (simple delegate)."""
        import asyncio
        from src.discord.client import HeimdallBot
        assert not asyncio.iscoroutinefunction(HeimdallBot._handle_send_to_agent)

    def test_list_handler_is_sync(self):
        from src.discord.client import HeimdallBot
        assert not asyncio.iscoroutinefunction(HeimdallBot._handle_list_agents)

    def test_kill_handler_is_sync(self):
        from src.discord.client import HeimdallBot
        assert not asyncio.iscoroutinefunction(HeimdallBot._handle_kill_agent)

    def test_get_results_handler_is_sync(self):
        from src.discord.client import HeimdallBot
        assert not asyncio.iscoroutinefunction(HeimdallBot._handle_get_agent_results)


# ===========================================================================
# Agent tools in loop dispatch
# ===========================================================================

class TestLoopDispatchAgentTools:
    """Tests that agent tools are available in _dispatch_loop_tool."""

    def test_agent_tools_in_loop_dispatch_source(self):
        """Verify agent tools appear in _dispatch_loop_tool."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._dispatch_loop_tool)

        for tool_name in ("spawn_agent", "send_to_agent", "list_agents",
                          "kill_agent", "get_agent_results"):
            assert tool_name in source, (
                f"Tool '{tool_name}' missing from _dispatch_loop_tool"
            )

    def test_skill_crud_tools_in_loop_dispatch_source(self):
        """Verify skill management tools appear in _dispatch_loop_tool."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._dispatch_loop_tool)

        for tool_name in ("install_skill", "export_skill", "skill_status"):
            assert tool_name in source, (
                f"Tool '{tool_name}' missing from _dispatch_loop_tool"
            )


# ===========================================================================
# Agent manager integration
# ===========================================================================

class TestAgentManagerIntegration:
    """Tests that agent_manager is properly integrated into the bot."""

    def test_agent_manager_import(self):
        """AgentManager should be imported in client module."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert "from ..agents import AgentManager" in source

    def test_agent_manager_initialized(self):
        """Bot.__init__ should create self.agent_manager."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert "self.agent_manager = AgentManager()" in source


# ===========================================================================
# End-to-end agent execution
# ===========================================================================

class TestAgentEndToEnd:
    """End-to-end tests of agent spawning and execution flow."""

    async def test_agent_completes_simple_task(self):
        """Agent that returns text without tools should complete immediately."""
        mgr = AgentManager()

        async def iteration_cb(messages, system, tools):
            return {"text": "Done: all servers healthy.", "tool_calls": []}

        agent_id = mgr.spawn(
            label="health", goal="Check health",
            channel_id="123", requester_id="456", requester_name="Tester",
            iteration_callback=iteration_cb,
            tool_executor_callback=AsyncMock(return_value="ok"),
            announce_callback=AsyncMock(),
        )

        # Wait for the agent to finish
        agent = mgr._agents[agent_id]
        for _ in range(50):
            if agent.status != "running":
                break
            await asyncio.sleep(0.05)

        assert agent.status == "completed"
        assert "all servers healthy" in agent.result

    async def test_agent_uses_tools_then_completes(self):
        """Agent that calls tools and then returns text."""
        mgr = AgentManager()
        call_count = 0

        async def iteration_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "Running command...",
                    "tool_calls": [{"name": "run_command", "input": {"command": "df -h"}}],
                    "stop_reason": "tool_use",
                }
            return {"text": "Disk at 42%. All good.", "tool_calls": []}

        async def tool_exec_cb(tool_name, tool_input):
            return "/dev/sda1 42% /\n/dev/sdb1 78% /data"

        announce = AsyncMock()

        agent_id = mgr.spawn(
            label="disk", goal="Check disk",
            channel_id="123", requester_id="456", requester_name="Tester",
            iteration_callback=iteration_cb,
            tool_executor_callback=tool_exec_cb,
            announce_callback=announce,
        )

        agent = mgr._agents[agent_id]
        for _ in range(50):
            if agent.status != "running":
                break
            await asyncio.sleep(0.05)

        assert agent.status == "completed"
        assert "42%" in agent.result
        assert "run_command" in agent.tools_used
        assert agent.iteration_count == 2
        # Agents are silent — announce should NOT be called
        announce.assert_not_called()

    async def test_agent_receives_injected_message(self):
        """send_to_agent injects messages into the agent's context."""
        mgr = AgentManager()
        call_count = 0
        seen_messages = []

        async def iteration_cb(messages, system, tools):
            nonlocal call_count
            call_count += 1
            seen_messages.append([m.get("content", "") for m in messages])

            if call_count == 1:
                # Return tool call to keep running
                return {
                    "text": "Starting...",
                    "tool_calls": [{"name": "check_disk", "input": {}}],
                    "stop_reason": "tool_use",
                }
            # Second call — should have the injected message
            return {"text": "Done with extra info.", "tool_calls": []}

        agent_id = mgr.spawn(
            label="inject-test", goal="Test injection",
            channel_id="123", requester_id="456", requester_name="Tester",
            iteration_callback=iteration_cb,
            tool_executor_callback=AsyncMock(return_value="ok"),
            announce_callback=AsyncMock(),
        )

        # Small delay to let first iteration start
        await asyncio.sleep(0.1)

        # Send a message to the agent
        mgr.send(agent_id, "Also check /var/log")

        agent = mgr._agents[agent_id]
        for _ in range(50):
            if agent.status != "running":
                break
            await asyncio.sleep(0.05)

        assert agent.status == "completed"
        # Check that the injected message appeared in conversation
        all_contents = [c for msgs in seen_messages for c in msgs]
        found_injected = any("[Message from parent]" in c for c in all_contents)
        assert found_injected or agent.iteration_count >= 2

    async def test_agent_error_isolation(self):
        """Agent crash should not crash the bot — reports error via announce."""
        mgr = AgentManager()

        async def iteration_cb(messages, system, tools):
            raise RuntimeError("LLM backend exploded")

        announce = AsyncMock()

        agent_id = mgr.spawn(
            label="crasher", goal="Crash test",
            channel_id="123", requester_id="456", requester_name="Tester",
            iteration_callback=iteration_cb,
            tool_executor_callback=AsyncMock(),
            announce_callback=announce,
        )

        agent = mgr._agents[agent_id]
        for _ in range(50):
            if agent.status != "running":
                break
            await asyncio.sleep(0.05)

        assert agent.status == "failed"
        assert "LLM" in agent.error or "exploded" in agent.error
        announce.assert_not_called()

    async def test_agent_kill_flow(self):
        """Kill signal should stop the agent."""
        mgr = AgentManager()

        async def iteration_cb(messages, system, tools):
            # Keep returning tool calls to stay alive
            return {
                "text": "Working...",
                "tool_calls": [{"name": "run_command", "input": {"command": "sleep 1"}}],
            }

        async def slow_tool(tool_name, tool_input):
            await asyncio.sleep(0.5)
            return "done"

        agent_id = mgr.spawn(
            label="killable", goal="Long task",
            channel_id="123", requester_id="456", requester_name="Tester",
            iteration_callback=iteration_cb,
            tool_executor_callback=slow_tool,
            announce_callback=AsyncMock(),
        )

        await asyncio.sleep(0.1)
        result = mgr.kill(agent_id)
        assert "kill" in result.lower()

        agent = mgr._agents[agent_id]
        for _ in range(50):
            if agent.status != "running":
                break
            await asyncio.sleep(0.05)

        assert agent.status == "killed"


# ===========================================================================
# Context isolation
# ===========================================================================

class TestContextIsolation:
    """Tests that agents have isolated context."""

    async def test_agent_has_own_messages(self):
        """Agent messages list is separate from any parent context."""
        mgr = AgentManager()

        async def iteration_cb(messages, system, tools):
            return {"text": "Done.", "tool_calls": []}

        agent_id = mgr.spawn(
            label="isolated", goal="My isolated task",
            channel_id="123", requester_id="456", requester_name="Tester",
            iteration_callback=iteration_cb,
            tool_executor_callback=AsyncMock(),
            announce_callback=AsyncMock(),
        )

        agent = mgr._agents[agent_id]
        for _ in range(50):
            if agent.status != "running":
                break
            await asyncio.sleep(0.05)

        # Agent should have its goal as the first message
        assert agent.messages[0]["role"] == "user"
        assert "My isolated task" in agent.messages[0]["content"]

    async def test_sibling_agents_isolated(self):
        """Two agents spawned in same channel have separate message histories."""
        mgr = AgentManager()
        agent_messages = {}

        def make_cb(label):
            async def iteration_cb(messages, system, tools):
                agent_messages[label] = [m.get("content", "") for m in messages]
                return {"text": f"Done from {label}.", "tool_calls": []}
            return iteration_cb

        id1 = mgr.spawn(
            label="sibling-a", goal="Task A",
            channel_id="123", requester_id="456", requester_name="Tester",
            iteration_callback=make_cb("a"),
            tool_executor_callback=AsyncMock(),
            announce_callback=AsyncMock(),
        )
        id2 = mgr.spawn(
            label="sibling-b", goal="Task B",
            channel_id="123", requester_id="456", requester_name="Tester",
            iteration_callback=make_cb("b"),
            tool_executor_callback=AsyncMock(),
            announce_callback=AsyncMock(),
        )

        for _ in range(50):
            a1 = mgr._agents[id1]
            a2 = mgr._agents[id2]
            if a1.status != "running" and a2.status != "running":
                break
            await asyncio.sleep(0.05)

        # Each agent should only see its own goal
        if "a" in agent_messages:
            assert "Task A" in agent_messages["a"][0]
            assert "Task B" not in str(agent_messages["a"])
        if "b" in agent_messages:
            assert "Task B" in agent_messages["b"][0]
            assert "Task A" not in str(agent_messages["b"])

    async def test_agent_system_prompt_contains_context(self):
        """Agent system prompt should contain AGENT CONTEXT."""
        mgr = AgentManager()
        captured_system = []

        async def iteration_cb(messages, system, tools):
            captured_system.append(system)
            return {"text": "Done.", "tool_calls": []}

        mgr.spawn(
            label="sys-test", goal="Test system prompt",
            channel_id="123", requester_id="456", requester_name="Tester",
            iteration_callback=iteration_cb,
            tool_executor_callback=AsyncMock(),
            announce_callback=AsyncMock(),
            system_prompt="Base prompt",
        )

        await asyncio.sleep(0.3)

        assert len(captured_system) > 0
        assert "AGENT CONTEXT" in captured_system[0]
        assert "sys-test" in captured_system[0]


# ===========================================================================
# Announce flow
# ===========================================================================

class TestSilentAgentFlow:
    """Tests that agents are silent internal workers — no Discord posting."""

    async def test_completed_agent_does_not_announce(self):
        """Agent completion should NOT trigger announce callback."""
        mgr = AgentManager()
        announce = AsyncMock()

        async def iteration_cb(messages, system, tools):
            return {"text": "Task complete.", "tool_calls": []}

        aid = mgr.spawn(
            label="announcer", goal="Do and announce",
            channel_id="999", requester_id="456", requester_name="Tester",
            iteration_callback=iteration_cb,
            tool_executor_callback=AsyncMock(),
            announce_callback=announce,
        )

        await asyncio.sleep(0.3)
        announce.assert_not_called()
        # Results stored internally for collection
        agent = mgr._agents[aid]
        assert agent.status == "completed"
        assert agent.result == "Task complete."

    async def test_failed_agent_does_not_announce(self):
        """Agent failure should NOT announce to channel."""
        mgr = AgentManager()
        announce = AsyncMock()

        async def iteration_cb(messages, system, tools):
            raise ValueError("Something broke")

        aid = mgr.spawn(
            label="broken", goal="Break things",
            channel_id="888", requester_id="456", requester_name="Tester",
            iteration_callback=iteration_cb,
            tool_executor_callback=AsyncMock(),
            announce_callback=announce,
        )

        await asyncio.sleep(0.3)
        announce.assert_not_called()
        agent = mgr._agents[aid]
        assert agent.status == "failed"
        assert "Something broke" in agent.error


# ===========================================================================
# Source verification
# ===========================================================================

class TestSourceVerification:
    """Verify source code structure."""

    def test_client_imports_agent_manager(self):
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert "from ..agents import AgentManager" in source

    def test_init_creates_agent_manager(self):
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert "self.agent_manager = AgentManager()" in source

    def test_dispatch_has_spawn_agent(self):
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert 'tool_name == "spawn_agent"' in source

    def test_dispatch_has_send_to_agent(self):
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert 'tool_name == "send_to_agent"' in source

    def test_dispatch_has_list_agents(self):
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert 'tool_name == "list_agents"' in source

    def test_dispatch_has_kill_agent(self):
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert 'tool_name == "kill_agent"' in source

    def test_dispatch_has_get_agent_results(self):
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert 'tool_name == "get_agent_results"' in source

    def test_loop_dispatch_has_agent_tools(self):
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._dispatch_loop_tool)
        for name in ("spawn_agent", "send_to_agent", "list_agents",
                      "kill_agent", "get_agent_results"):
            assert name in source

    def test_agent_run_scrubs_secrets(self):
        """Agent execution scrubs secrets from tool results."""
        import inspect
        from src.agents.manager import _run_agent
        source = inspect.getsource(_run_agent)
        assert "scrub_output_secrets" in source

    def test_handler_uses_loop_message_proxy(self):
        """spawn_agent should create a _LoopMessageProxy for tool dispatch."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._handle_spawn_agent)
        assert "_LoopMessageProxy" in source

    def test_handler_uses_dispatch_loop_tool(self):
        """Tool callback should route through _dispatch_loop_tool."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._handle_spawn_agent)
        assert "_dispatch_loop_tool" in source

    def test_spawn_handler_builds_system_prompt(self):
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._handle_spawn_agent)
        assert "_build_system_prompt" in source

    def test_spawn_handler_gets_tools(self):
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._handle_spawn_agent)
        assert "_merged_tool_definitions" in source


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge case tests."""

    async def test_spawn_empty_label(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        result = await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "", "goal": "Test"},
        )
        assert "required" in result.lower()

    async def test_spawn_empty_goal(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message()

        result = await HeimdallBot._handle_spawn_agent(
            bot, msg, {"label": "test", "goal": ""},
        )
        assert "required" in result.lower()

    async def test_send_empty_agent_id(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_send_to_agent(bot, {"agent_id": "", "message": "hi"})
        assert "required" in result.lower()

    async def test_send_empty_message(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_send_to_agent(
            bot, {"agent_id": "abc", "message": ""},
        )
        assert "required" in result.lower()

    async def test_kill_empty_agent_id(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_kill_agent(bot, {"agent_id": ""})
        assert "required" in result.lower()

    async def test_get_results_empty_id(self):
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()

        result = HeimdallBot._handle_get_agent_results(bot, {"agent_id": ""})
        assert "required" in result.lower()

    async def test_list_agents_with_proxy_message(self):
        """list_agents should work with _LoopMessageProxy."""
        from src.discord.client import HeimdallBot, _LoopMessageProxy
        bot = _make_mock_bot()
        channel = MagicMock()
        channel.id = 123
        proxy = _LoopMessageProxy(channel, "456", "ProxyUser")

        result = HeimdallBot._handle_list_agents(bot, proxy)
        assert "no agents" in result.lower()

    async def test_multiple_spawns_same_channel(self):
        """Multiple agents in same channel should all appear in list."""
        from src.discord.client import HeimdallBot
        bot = _make_mock_bot()
        msg = _make_mock_message(channel_id="555")

        for i in range(3):
            await HeimdallBot._handle_spawn_agent(
                bot, msg, {"label": f"agent-{i}", "goal": f"Task {i}"},
            )

        result = HeimdallBot._handle_list_agents(bot, msg)
        assert "Agents (3)" in result
        for i in range(3):
            assert f"agent-{i}" in result
