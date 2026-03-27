"""Round 23 — Agent + Loop Integration tests.

Tests LoopAgentBridge, spawn_loop_agents/collect_loop_agents tool handlers,
loop iteration prompt agent awareness, and cleanup integration.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.loop_bridge import (
    LOOP_AGENT_WAIT_TIMEOUT,
    MAX_AGENTS_PER_ITERATION,
    MAX_AGENTS_PER_LOOP,
    LoopAgentBridge,
    LoopAgentRecord,
)
from src.agents.manager import AgentManager, MAX_CONCURRENT_AGENTS
from src.tools.autonomous_loop import LoopManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_manager():
    """Create an AgentManager."""
    return AgentManager()


def _make_callbacks(text="Done.", tool_calls=None):
    """Return (iteration_cb, tool_exec_cb, announce_cb)."""
    return (
        AsyncMock(return_value={
            "text": text, "tool_calls": tool_calls or [],
            "stop_reason": "end_turn",
        }),
        AsyncMock(return_value="ok"),
        AsyncMock(),
    )


def _make_bridge():
    """Create AgentManager + LoopAgentBridge pair."""
    mgr = _make_agent_manager()
    bridge = LoopAgentBridge(mgr)
    return mgr, bridge


# ---------------------------------------------------------------------------
# LoopAgentBridge — spawn_agents_for_loop
# ---------------------------------------------------------------------------

class TestSpawnAgentsForLoop:
    """Test spawning agents from loop iterations via the bridge."""

    async def test_spawn_single_agent(self):
        """Bridge spawns one agent with loop context in the goal."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=3, loop_goal="Monitor disk usage",
            tasks=[{"label": "check-host-a", "goal": "Check /dev/sda on host-a"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        assert len(ids) == 1
        assert not ids[0].startswith("Error")
        # Agent should exist in manager
        agent = mgr._agents.get(ids[0])
        assert agent is not None
        assert "loop1" in agent.goal
        assert "Monitor disk usage" in agent.goal
        assert "Check /dev/sda on host-a" in agent.goal

    async def test_spawn_multiple_agents(self):
        """Bridge spawns multiple agents in one iteration."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        tasks = [
            {"label": f"agent-{i}", "goal": f"Task {i}"}
            for i in range(3)
        ]
        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Multi-check",
            tasks=tasks, channel_id="100", requester_id="u1",
            requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        assert len(ids) == 3
        assert all(not aid.startswith("Error") for aid in ids)
        assert bridge.get_loop_agent_count("loop1") == 3

    async def test_per_iteration_limit(self):
        """Cannot spawn more than MAX_AGENTS_PER_ITERATION in one call."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        tasks = [
            {"label": f"agent-{i}", "goal": f"Task {i}"}
            for i in range(MAX_AGENTS_PER_ITERATION + 1)
        ]
        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Overflow",
            tasks=tasks, channel_id="100", requester_id="u1",
            requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        assert len(ids) == 1
        assert ids[0].startswith("Error")
        assert str(MAX_AGENTS_PER_ITERATION) in ids[0]

    async def test_per_loop_lifetime_limit(self):
        """Cannot exceed MAX_AGENTS_PER_LOOP across the loop's lifetime."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        # Spawn MAX_AGENTS_PER_LOOP agents across iterations
        # Use different channel_ids to avoid hitting per-channel concurrent limit
        for i in range(MAX_AGENTS_PER_LOOP):
            ids = bridge.spawn_agents_for_loop(
                loop_id="loop1", iteration=i, loop_goal="Batch",
                tasks=[{"label": f"a-{i}", "goal": f"T-{i}"}],
                channel_id=str(100 + i), requester_id="u1", requester_name="user1",
                iteration_callback=iter_cb, tool_executor_callback=tool_cb,
                announce_callback=ann_cb,
            )
            assert not ids[0].startswith("Error"), f"Failed at iteration {i}: {ids[0]}"

        # Next one should fail due to per-loop lifetime limit
        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=MAX_AGENTS_PER_LOOP, loop_goal="Batch",
            tasks=[{"label": "overflow", "goal": "Over"}],
            channel_id="200", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )
        assert ids[0].startswith("Error")
        assert str(MAX_AGENTS_PER_LOOP) in ids[0]

    async def test_empty_tasks_returns_empty(self):
        """Empty task list returns empty list."""
        mgr, bridge = _make_bridge()
        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Empty",
            tasks=[], channel_id="100", requester_id="u1",
            requester_name="user1",
            iteration_callback=AsyncMock(), tool_executor_callback=AsyncMock(),
            announce_callback=AsyncMock(),
        )
        assert ids == []

    async def test_agent_manager_error_propagated(self):
        """AgentManager.spawn error strings are passed through."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        # Fill channel to capacity
        for i in range(MAX_CONCURRENT_AGENTS):
            mgr.spawn(
                label=f"fill-{i}", goal="fill", channel_id="100",
                requester_id="u1", requester_name="user1",
                iteration_callback=iter_cb, tool_executor_callback=tool_cb,
                announce_callback=ann_cb,
            )

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Overflow",
            tasks=[{"label": "extra", "goal": "extra"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )
        assert ids[0].startswith("Error")
        # Should NOT be tracked in bridge records
        assert bridge.get_loop_agent_count("loop1") == 0


# ---------------------------------------------------------------------------
# LoopAgentBridge — wait_and_collect
# ---------------------------------------------------------------------------

class TestWaitAndCollect:
    """Test collecting agent results from loop-spawned agents."""

    async def test_collect_all_agents(self):
        """Collect all agents spawned by a loop."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Collect test",
            tasks=[
                {"label": "a1", "goal": "Task 1"},
                {"label": "a2", "goal": "Task 2"},
            ],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        # Let agents complete
        await asyncio.sleep(0.1)

        results = await bridge.wait_and_collect("loop1")
        assert len(results) == 2
        for aid in ids:
            assert aid in results
            assert results[aid]["status"] in ("completed", "running")

    async def test_collect_specific_agents(self):
        """Collect only specified agent IDs."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Selective",
            tasks=[
                {"label": "a1", "goal": "Task 1"},
                {"label": "a2", "goal": "Task 2"},
            ],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        await asyncio.sleep(0.1)

        # Collect only first agent
        results = await bridge.wait_and_collect("loop1", agent_ids=[ids[0]])
        assert len(results) == 1
        assert ids[0] in results

    async def test_collect_marks_collected(self):
        """Collected agents are marked so they aren't collected again."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Mark test",
            tasks=[{"label": "a1", "goal": "Task 1"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        await asyncio.sleep(0.1)

        # First collect
        results1 = await bridge.wait_and_collect("loop1")
        assert len(results1) == 1

        # Second collect: no uncollected agents
        results2 = await bridge.wait_and_collect("loop1")
        assert len(results2) == 0

    async def test_collect_empty_loop(self):
        """Collecting from a loop with no agents returns empty."""
        _, bridge = _make_bridge()
        results = await bridge.wait_and_collect("nonexistent")
        assert results == {}


# ---------------------------------------------------------------------------
# LoopAgentBridge — format_agent_results_for_context
# ---------------------------------------------------------------------------

class TestFormatResults:
    """Test formatting agent results for loop iteration context."""

    def test_format_completed_results(self):
        """Completed agent results format correctly."""
        _, bridge = _make_bridge()
        results = {
            "abc123": {
                "label": "disk-check",
                "status": "completed",
                "result": "Disk usage at 45%.",
                "error": "",
            },
            "def456": {
                "label": "cpu-check",
                "status": "completed",
                "result": "CPU idle at 90%.",
                "error": "",
            },
        }
        text = bridge.format_agent_results_for_context(results)
        assert "Agent results:" in text
        assert "disk-check" in text
        assert "cpu-check" in text
        assert "completed" in text

    def test_format_truncates_long_results(self):
        """Long result text is truncated."""
        _, bridge = _make_bridge()
        results = {
            "abc": {
                "label": "verbose",
                "status": "completed",
                "result": "x" * 1000,
                "error": "",
            }
        }
        text = bridge.format_agent_results_for_context(results)
        assert len(text) < 800  # Significantly less than 1000 chars
        assert "..." in text

    def test_format_empty_results(self):
        """Empty results return empty string."""
        _, bridge = _make_bridge()
        assert bridge.format_agent_results_for_context({}) == ""

    def test_format_error_result(self):
        """Failed agent shows error instead of result."""
        _, bridge = _make_bridge()
        results = {
            "fail1": {
                "label": "broken",
                "status": "failed",
                "result": "",
                "error": "Connection refused",
            }
        }
        text = bridge.format_agent_results_for_context(results)
        assert "Connection refused" in text
        assert "failed" in text


# ---------------------------------------------------------------------------
# LoopAgentBridge — cleanup_loop
# ---------------------------------------------------------------------------

class TestCleanupLoop:
    """Test cleanup of loop-agent records."""

    async def test_cleanup_removes_records(self):
        """Cleanup removes all records for a loop."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Cleanup test",
            tasks=[{"label": "a1", "goal": "T1"}, {"label": "a2", "goal": "T2"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        assert bridge.get_loop_agent_count("loop1") == 2
        removed = bridge.cleanup_loop("loop1")
        assert removed == 2
        assert bridge.get_loop_agent_count("loop1") == 0

    def test_cleanup_nonexistent_loop(self):
        """Cleanup of nonexistent loop returns 0."""
        _, bridge = _make_bridge()
        assert bridge.cleanup_loop("nonexistent") == 0


# ---------------------------------------------------------------------------
# LoopAgentBridge — get_active_loop_agents
# ---------------------------------------------------------------------------

class TestGetActiveLoopAgents:
    """Test listing active agents for a loop."""

    async def test_active_agents_listed(self):
        """Active (uncollected) agents are returned."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Active test",
            tasks=[{"label": "a1", "goal": "T1"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        await asyncio.sleep(0.1)

        active = bridge.get_active_loop_agents("loop1")
        assert len(active) == 1
        assert active[0]["agent_id"] == ids[0]
        assert active[0]["label"] == "a1"

    async def test_collected_agents_excluded(self):
        """Collected agents are not returned as active."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Collect test",
            tasks=[{"label": "a1", "goal": "T1"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        await asyncio.sleep(0.1)
        await bridge.wait_and_collect("loop1")

        active = bridge.get_active_loop_agents("loop1")
        assert len(active) == 0


# ---------------------------------------------------------------------------
# LoopAgentBridge — tracked_loop_count property
# ---------------------------------------------------------------------------

class TestTrackedLoopCount:
    """Test the tracked_loop_count property."""

    async def test_count_reflects_tracked_loops(self):
        """tracked_loop_count reflects loops with agent records."""
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        assert bridge.tracked_loop_count == 0

        bridge.spawn_agents_for_loop(
            loop_id="loop1", iteration=1, loop_goal="Count test",
            tasks=[{"label": "a1", "goal": "T1"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )
        assert bridge.tracked_loop_count == 1

        bridge.spawn_agents_for_loop(
            loop_id="loop2", iteration=1, loop_goal="Count test 2",
            tasks=[{"label": "a2", "goal": "T2"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )
        assert bridge.tracked_loop_count == 2

        bridge.cleanup_loop("loop1")
        assert bridge.tracked_loop_count == 1


# ---------------------------------------------------------------------------
# LoopManager — agents_enabled flag
# ---------------------------------------------------------------------------

class TestLoopManagerAgentsEnabled:
    """Test LoopManager agents_enabled flag affects iteration prompts."""

    def test_agents_enabled_true(self):
        """When agents_enabled, iteration prompt includes agent instructions."""
        mgr = LoopManager(agents_enabled=True)
        from src.tools.autonomous_loop import LoopInfo
        info = LoopInfo(
            id="test", goal="Monitor servers", mode="act",
            interval_seconds=60, stop_condition=None, max_iterations=10,
            channel_id="100", requester_id="u1", requester_name="user1",
        )
        info.iteration_count = 1
        prompt = mgr._build_iteration_prompt(info)
        assert "spawn_agent" in prompt
        assert "AGENTS:" in prompt

    def test_agents_enabled_false(self):
        """When agents_enabled=False, prompt has no agent instructions."""
        mgr = LoopManager(agents_enabled=False)
        from src.tools.autonomous_loop import LoopInfo
        info = LoopInfo(
            id="test", goal="Monitor servers", mode="act",
            interval_seconds=60, stop_condition=None, max_iterations=10,
            channel_id="100", requester_id="u1", requester_name="user1",
        )
        info.iteration_count = 1
        prompt = mgr._build_iteration_prompt(info)
        assert "spawn_agent" not in prompt
        assert "AGENTS:" not in prompt

    def test_default_agents_disabled(self):
        """Default LoopManager has agents_enabled=False."""
        mgr = LoopManager()
        assert mgr._agents_enabled is False


# ---------------------------------------------------------------------------
# LoopManager iteration prompt — all modes with agents
# ---------------------------------------------------------------------------

class TestLoopPromptWithAgents:
    """Test iteration prompt formatting with agent awareness across modes."""

    def _make_info(self, mode="notify"):
        from src.tools.autonomous_loop import LoopInfo
        return LoopInfo(
            id="lp1", goal="Watch servers", mode=mode,
            interval_seconds=30, stop_condition=None, max_iterations=5,
            channel_id="100", requester_id="u1", requester_name="user1",
        )

    def test_notify_mode_with_agents(self):
        mgr = LoopManager(agents_enabled=True)
        info = self._make_info("notify")
        info.iteration_count = 2
        prompt = mgr._build_iteration_prompt(info)
        assert "AGENTS:" in prompt
        assert "notify" not in prompt.split("AGENTS:")[1]  # Agent section separate

    def test_silent_mode_with_agents(self):
        mgr = LoopManager(agents_enabled=True)
        info = self._make_info("silent")
        info.iteration_count = 1
        prompt = mgr._build_iteration_prompt(info)
        assert "AGENTS:" in prompt
        assert "[NOTIFY]" in prompt

    def test_stop_condition_with_agents(self):
        from src.tools.autonomous_loop import LoopInfo
        mgr = LoopManager(agents_enabled=True)
        info = LoopInfo(
            id="lp1", goal="Watch", mode="act",
            interval_seconds=30, stop_condition="disk > 90%",
            max_iterations=5, channel_id="100",
            requester_id="u1", requester_name="user1",
        )
        info.iteration_count = 1
        prompt = mgr._build_iteration_prompt(info)
        assert "LOOP_STOP" in prompt
        assert "AGENTS:" in prompt


# ---------------------------------------------------------------------------
# LoopAgentRecord dataclass
# ---------------------------------------------------------------------------

class TestLoopAgentRecord:
    """Test LoopAgentRecord dataclass."""

    def test_defaults(self):
        r = LoopAgentRecord(
            agent_id="abc", loop_id="loop1",
            iteration=3, label="test-agent",
        )
        assert r.agent_id == "abc"
        assert r.loop_id == "loop1"
        assert r.iteration == 3
        assert r.collected is False
        assert r.spawned_at > 0


# ---------------------------------------------------------------------------
# Tool handler integration — spawn_loop_agents
# ---------------------------------------------------------------------------

class TestHandleSpawnLoopAgents:
    """Test _handle_spawn_loop_agents tool handler on HeimdallBot."""

    def _make_bot(self):
        """Create a minimal mock bot with the needed attributes."""
        bot = MagicMock()
        bot.loop_manager = LoopManager(agents_enabled=True)
        bot.agent_manager = AgentManager()
        bot.loop_agent_bridge = LoopAgentBridge(bot.agent_manager)
        bot.codex_client = MagicMock()
        bot.codex_client.chat_with_tools = AsyncMock(return_value=MagicMock(
            text="Done.", tool_calls=[], stop_reason="end_turn",
        ))
        bot.config = MagicMock()
        bot.config.tools.enabled = True
        bot._merged_tool_definitions = MagicMock(return_value=[])
        bot._build_system_prompt = MagicMock(return_value="sys prompt")
        bot._dispatch_loop_tool = AsyncMock(return_value="ok")
        bot.audit = MagicMock()
        bot.audit.log_execution = AsyncMock()
        return bot

    async def test_missing_loop_id(self):
        """Returns error when loop_id missing."""
        from src.discord.client import HeimdallBot
        bot = self._make_bot()
        result = await HeimdallBot._handle_spawn_loop_agents(bot, MagicMock(), {"tasks": []})
        assert "loop_id" in result.lower()

    async def test_missing_tasks(self):
        """Returns error when tasks missing."""
        from src.discord.client import HeimdallBot
        bot = self._make_bot()
        result = await HeimdallBot._handle_spawn_loop_agents(
            bot, MagicMock(), {"loop_id": "abc", "tasks": []},
        )
        assert "tasks" in result.lower()

    async def test_nonexistent_loop(self):
        """Returns error when loop doesn't exist."""
        from src.discord.client import HeimdallBot
        bot = self._make_bot()
        result = await HeimdallBot._handle_spawn_loop_agents(
            bot, MagicMock(), {"loop_id": "nope", "tasks": [{"label": "a", "goal": "b"}]},
        )
        assert "not found" in result.lower()

    async def test_successful_spawn(self):
        """Successfully spawns agents through the bridge."""
        from src.discord.client import HeimdallBot
        from src.tools.autonomous_loop import LoopInfo

        bot = self._make_bot()
        # Add a running loop
        info = LoopInfo(
            id="lp1", goal="Monitor", mode="act",
            interval_seconds=60, stop_condition=None, max_iterations=10,
            channel_id="100", requester_id="u1", requester_name="user1",
        )
        info.iteration_count = 2
        bot.loop_manager._loops["lp1"] = info

        msg = MagicMock()
        msg.channel = MagicMock()
        msg.channel.id = 100

        result = await HeimdallBot._handle_spawn_loop_agents(
            bot, msg, {
                "loop_id": "lp1",
                "tasks": [{"label": "check-a", "goal": "Check host A"}],
            },
        )
        assert "Spawned 1 agent" in result
        assert bot.loop_agent_bridge.get_loop_agent_count("lp1") == 1

    async def test_stopped_loop_rejected(self):
        """Cannot spawn agents for a stopped loop."""
        from src.discord.client import HeimdallBot
        from src.tools.autonomous_loop import LoopInfo

        bot = self._make_bot()
        info = LoopInfo(
            id="lp2", goal="Done", mode="act",
            interval_seconds=60, stop_condition=None, max_iterations=10,
            channel_id="100", requester_id="u1", requester_name="user1",
        )
        info.status = "stopped"
        bot.loop_manager._loops["lp2"] = info

        result = await HeimdallBot._handle_spawn_loop_agents(
            bot, MagicMock(), {
                "loop_id": "lp2",
                "tasks": [{"label": "a", "goal": "b"}],
            },
        )
        assert "not running" in result.lower()


# ---------------------------------------------------------------------------
# Tool handler integration — collect_loop_agents
# ---------------------------------------------------------------------------

class TestHandleCollectLoopAgents:
    """Test _handle_collect_loop_agents tool handler on HeimdallBot."""

    def _make_bot(self):
        bot = MagicMock()
        bot.loop_manager = LoopManager(agents_enabled=True)
        bot.agent_manager = AgentManager()
        bot.loop_agent_bridge = LoopAgentBridge(bot.agent_manager)
        return bot

    async def test_missing_loop_id(self):
        from src.discord.client import HeimdallBot
        bot = self._make_bot()
        result = await HeimdallBot._handle_collect_loop_agents(bot, {})
        assert "loop_id" in result.lower()

    async def test_nonexistent_loop(self):
        from src.discord.client import HeimdallBot
        bot = self._make_bot()
        result = await HeimdallBot._handle_collect_loop_agents(
            bot, {"loop_id": "nonexistent"},
        )
        assert "not found" in result.lower()

    async def test_no_agents_to_collect(self):
        from src.discord.client import HeimdallBot
        from src.tools.autonomous_loop import LoopInfo

        bot = self._make_bot()
        info = LoopInfo(
            id="lp1", goal="Empty", mode="act",
            interval_seconds=60, stop_condition=None, max_iterations=10,
            channel_id="100", requester_id="u1", requester_name="user1",
        )
        bot.loop_manager._loops["lp1"] = info

        result = await HeimdallBot._handle_collect_loop_agents(
            bot, {"loop_id": "lp1"},
        )
        assert "no agents" in result.lower()


# ---------------------------------------------------------------------------
# Tool definition registration
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    """Test that spawn_loop_agents and collect_loop_agents are in the registry."""

    def test_spawn_loop_agents_defined(self):
        from src.tools.registry import TOOLS
        names = {t["name"] for t in TOOLS}
        assert "spawn_loop_agents" in names

    def test_collect_loop_agents_defined(self):
        from src.tools.registry import TOOLS
        names = {t["name"] for t in TOOLS}
        assert "collect_loop_agents" in names

    def test_spawn_loop_agents_schema(self):
        from src.tools.registry import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "spawn_loop_agents")
        assert "loop_id" in tool["input_schema"]["properties"]
        assert "tasks" in tool["input_schema"]["properties"]
        assert "loop_id" in tool["input_schema"]["required"]
        assert "tasks" in tool["input_schema"]["required"]

    def test_collect_loop_agents_schema(self):
        from src.tools.registry import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "collect_loop_agents")
        assert "loop_id" in tool["input_schema"]["properties"]
        assert "loop_id" in tool["input_schema"]["required"]


# ---------------------------------------------------------------------------
# Cleanup integration in cache cleanup
# ---------------------------------------------------------------------------

class TestCleanupIntegration:
    """Test that loop-agent bridge cleanup integrates with periodic cleanup."""

    def test_bridge_cleanup_on_finished_loop(self):
        """Bridge records for finished loops are cleaned up."""
        from src.tools.autonomous_loop import LoopInfo

        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        bridge.spawn_agents_for_loop(
            loop_id="lp1", iteration=1, loop_goal="Finish test",
            tasks=[{"label": "a1", "goal": "T1"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )
        assert bridge.tracked_loop_count == 1

        # Simulate cleanup: loop is now stopped
        loop_manager = LoopManager(agents_enabled=True)
        info = LoopInfo(
            id="lp1", goal="Done", mode="act",
            interval_seconds=60, stop_condition=None, max_iterations=10,
            channel_id="100", requester_id="u1", requester_name="user1",
        )
        info.status = "stopped"
        loop_manager._loops["lp1"] = info

        # Simulate the cleanup logic from _cleanup_stale_caches
        for loop_id in list(bridge._loop_agents):
            linfo = loop_manager._loops.get(loop_id)
            if not linfo or linfo.status != "running":
                bridge.cleanup_loop(loop_id)

        assert bridge.tracked_loop_count == 0

    def test_bridge_cleanup_preserves_running_loops(self):
        """Bridge records for running loops are not cleaned up."""
        from src.tools.autonomous_loop import LoopInfo

        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        bridge.spawn_agents_for_loop(
            loop_id="lp1", iteration=1, loop_goal="Running test",
            tasks=[{"label": "a1", "goal": "T1"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        loop_manager = LoopManager(agents_enabled=True)
        info = LoopInfo(
            id="lp1", goal="Running", mode="act",
            interval_seconds=60, stop_condition=None, max_iterations=10,
            channel_id="100", requester_id="u1", requester_name="user1",
        )
        info.status = "running"
        loop_manager._loops["lp1"] = info

        # Cleanup logic should preserve running loops
        for loop_id in list(bridge._loop_agents):
            linfo = loop_manager._loops.get(loop_id)
            if not linfo or linfo.status != "running":
                bridge.cleanup_loop(loop_id)

        assert bridge.tracked_loop_count == 1


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------

class TestExports:
    """Test that LoopAgentBridge is exported from agents package."""

    def test_loop_agent_bridge_exported(self):
        from src.agents import LoopAgentBridge
        assert LoopAgentBridge is not None

    def test_all_exports(self):
        import src.agents
        assert "LoopAgentBridge" in src.agents.__all__


# ---------------------------------------------------------------------------
# get_loop_agent_ids
# ---------------------------------------------------------------------------

class TestGetLoopAgentIds:
    """Test get_loop_agent_ids helper."""

    async def test_returns_agent_ids(self):
        mgr, bridge = _make_bridge()
        iter_cb, tool_cb, ann_cb = _make_callbacks()

        ids = bridge.spawn_agents_for_loop(
            loop_id="lp1", iteration=1, loop_goal="ID test",
            tasks=[{"label": "a1", "goal": "T1"}, {"label": "a2", "goal": "T2"}],
            channel_id="100", requester_id="u1", requester_name="user1",
            iteration_callback=iter_cb, tool_executor_callback=tool_cb,
            announce_callback=ann_cb,
        )

        agent_ids = bridge.get_loop_agent_ids("lp1")
        assert set(agent_ids) == set(ids)

    def test_empty_for_unknown_loop(self):
        _, bridge = _make_bridge()
        assert bridge.get_loop_agent_ids("unknown") == []
