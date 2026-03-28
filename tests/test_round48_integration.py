"""Round 48 — Integration Testing.

End-to-end tests covering cross-system integration:
  - Agent full lifecycle (spawn → iterate → tool calls → completion → results)
  - Agent group spawn + wait + cleanup
  - Loop-Agent bridge (spawn for loop → wait and collect → format results)
  - Skill loading → tool registration → execution → resource tracking
  - Session management → compaction → topic detection → relevance filtering
  - Web chat → _process_with_tools → session persistence
  - Detection systems in tool-loop context (fabrication, hedging, premature failure,
    code hedging, tool unavailable)
  - Agent health check integration
  - Session continuity across archive/load cycle
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.loop_bridge import (
    LoopAgentBridge,
    MAX_AGENTS_PER_ITERATION,
    MAX_AGENTS_PER_LOOP,
)
from src.agents.manager import (
    AgentInfo,
    AgentManager,
    MAX_AGENT_ITERATIONS,
    MAX_AGENT_LIFETIME,
    MAX_CONCURRENT_AGENTS,
    _run_agent,
    filter_agent_tools,
    AGENT_BLOCKED_TOOLS,
)
from src.discord.client import (
    detect_code_hedging,
    detect_fabrication,
    detect_hedging,
    detect_premature_failure,
    detect_tool_unavailable,
)
from src.llm.secret_scrubber import scrub_output_secrets
from src.sessions.manager import (
    COMPACTION_THRESHOLD,
    CONTEXT_TOKEN_BUDGET,
    RELEVANCE_KEEP_RECENT,
    RELEVANCE_MIN_SCORE,
    SessionManager,
    apply_token_budget,
    estimate_tokens,
    score_relevance,
    summarize_tool_response,
)
from src.web.chat import WebMessage, process_web_chat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(text="", tool_calls=None, stop_reason="end_turn"):
    """Build an LLM response dict suitable for agent iteration callback."""
    return {
        "text": text,
        "tool_calls": tool_calls or [],
        "stop_reason": stop_reason,
    }


def _make_iteration_cb(responses):
    """Build an async iteration callback that yields responses in order."""
    call_count = {"n": 0}

    async def cb(messages, system_prompt, tools):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    return cb


def _make_tool_exec_cb(results=None):
    """Build an async tool executor callback with optional canned results."""
    _results = results or {}

    async def cb(tool_name, tool_input):
        return _results.get(tool_name, f"Result of {tool_name}")

    return cb


def _make_announce_cb():
    """Build an async announce callback that records calls."""
    calls = []

    async def cb(channel_id, text):
        calls.append({"channel_id": channel_id, "text": text})

    cb.calls = calls
    return cb


def _make_mock_bot(tmp_path, codex_responses=None):
    """Create a mock HeimdallBot with session manager and codex client."""
    bot = MagicMock()
    bot.sessions = SessionManager(
        max_history=50, max_age_hours=24,
        persist_dir=str(tmp_path / "sessions"),
    )
    bot.agent_manager = AgentManager()

    bot.config = MagicMock()
    bot.config.tools.enabled = True
    bot.config.tools.tool_packs = []

    if codex_responses is not None:
        bot.codex_client = MagicMock()
        # Will be configured per-test
    else:
        bot.codex_client = None

    bot._build_system_prompt = MagicMock(return_value="System prompt.")
    bot._inject_tool_hints = AsyncMock(side_effect=lambda sp, q, uid: sp)
    bot._merged_tool_definitions = MagicMock(return_value=[{"name": "run_command"}])
    bot.get_channel = MagicMock(return_value=None)
    return bot


# ===========================================================================
# 1. Agent Full Lifecycle Integration
# ===========================================================================

class TestAgentFullLifecycle:
    """Test agent from spawn through tool execution to completion."""

    async def test_agent_spawn_iterate_tools_complete(self):
        """Agent spawns, calls tools, completes, and results are available."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        # Agent will: 1) call run_command tool, 2) return final text
        responses = [
            _make_llm_response(
                text="Running check...",
                tool_calls=[{"name": "run_command", "input": {"command": "df -h"}}],
            ),
            _make_llm_response(text="Disk usage is 45% on /dev/sda1."),
        ]

        agent_id = mgr.spawn(
            label="disk-check",
            goal="Check disk usage on server",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="TestUser",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(
                {"run_command": "Filesystem  Use%  Mounted\n/dev/sda1  45%  /"}
            ),
            announce_callback=announce,
            tools=[{"name": "run_command"}],
        )

        assert not agent_id.startswith("Error")

        # Wait for agent to complete
        results = await mgr.wait_for_agents([agent_id], timeout=10)

        assert agent_id in results
        r = results[agent_id]
        assert r["status"] == "completed"
        assert "Disk usage" in r["result"]
        assert "run_command" in r["tools_used"]
        assert r["iteration_count"] == 2
        assert r["runtime_seconds"] >= 0

        # Agents are silent — no announce callback
        assert len(announce.calls) == 0

    async def test_agent_no_tools_completes_immediately(self):
        """Agent that returns text without tool calls completes in 1 iteration."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        responses = [_make_llm_response(text="Hello, I'm ready to help.")]

        agent_id = mgr.spawn(
            label="greeter",
            goal="Greet the user",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="TestUser",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        results = await mgr.wait_for_agents([agent_id], timeout=10)
        assert results[agent_id]["status"] == "completed"
        assert results[agent_id]["iteration_count"] == 1
        assert results[agent_id]["tools_used"] == []

    async def test_agent_multi_tool_calls_per_iteration(self):
        """Agent can call multiple tools in a single iteration."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        responses = [
            _make_llm_response(
                text="Checking multiple things...",
                tool_calls=[
                    {"name": "run_command", "input": {"command": "df -h"}},
                    {"name": "check_service", "input": {"service": "nginx"}},
                ],
            ),
            _make_llm_response(text="Disk is fine and nginx is running."),
        ]

        agent_id = mgr.spawn(
            label="multi-check",
            goal="Check disk and service",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="TestUser",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb({
                "run_command": "50% used",
                "check_service": "active (running)",
            }),
            announce_callback=announce,
        )

        results = await mgr.wait_for_agents([agent_id], timeout=10)
        r = results[agent_id]
        assert r["status"] == "completed"
        assert "run_command" in r["tools_used"]
        assert "check_service" in r["tools_used"]

    async def test_agent_tool_error_continues(self):
        """Agent continues iterating even when a tool raises an error."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        async def failing_tool(name, inp):
            if name == "bad_tool":
                raise RuntimeError("Connection refused")
            return "ok"

        responses = [
            _make_llm_response(
                tool_calls=[{"name": "bad_tool", "input": {}}],
            ),
            _make_llm_response(
                tool_calls=[{"name": "good_tool", "input": {}}],
            ),
            _make_llm_response(text="Recovered after error."),
        ]

        agent_id = mgr.spawn(
            label="resilient",
            goal="Handle errors gracefully",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="TestUser",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=failing_tool,
            announce_callback=announce,
        )

        results = await mgr.wait_for_agents([agent_id], timeout=10)
        r = results[agent_id]
        assert r["status"] == "completed"
        assert "bad_tool" in r["tools_used"]
        assert "good_tool" in r["tools_used"]

    async def test_agent_kill_terminates(self):
        """Killing a running agent transitions it to killed status."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        cancel_check = asyncio.Event()

        # Agent iterates with short tool calls, checking cancel between them
        async def iterable_iteration(messages, system, tools):
            cancel_check.set()
            # Return a tool call so the agent continues iterating
            return _make_llm_response(
                tool_calls=[{"name": "noop", "input": {}}],
            )

        async def slow_tool(name, inp):
            # Wait a bit to give the test time to send kill signal
            await asyncio.sleep(0.5)
            return "ok"

        agent_id = mgr.spawn(
            label="slow-agent",
            goal="Do something slowly",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="TestUser",
            iteration_callback=iterable_iteration,
            tool_executor_callback=slow_tool,
            announce_callback=announce,
        )

        # Wait for agent to start iterating
        await cancel_check.wait()

        result = mgr.kill(agent_id)
        assert "Kill signal" in result

        # Agent should detect cancel_event at start of next iteration
        results = await mgr.wait_for_agents([agent_id], timeout=10)
        assert results[agent_id]["status"] == "killed"

    async def test_agent_concurrent_limit(self):
        """Cannot exceed MAX_CONCURRENT_AGENTS per channel."""
        mgr = AgentManager()

        async def hang_cb(messages, system, tools):
            await asyncio.sleep(30)
            return _make_llm_response()

        announce = _make_announce_cb()

        # Spawn up to the limit
        ids = []
        for i in range(MAX_CONCURRENT_AGENTS):
            aid = mgr.spawn(
                label=f"agent-{i}",
                goal="hang",
                channel_id="chan-1",
                requester_id="user-1",
                requester_name="Test",
                iteration_callback=hang_cb,
                tool_executor_callback=_make_tool_exec_cb(),
                announce_callback=announce,
            )
            assert not aid.startswith("Error"), f"Failed to spawn agent {i}: {aid}"
            ids.append(aid)

        # Next spawn should fail
        result = mgr.spawn(
            label="over-limit",
            goal="should fail",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=hang_cb,
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )
        assert result.startswith("Error")
        assert str(MAX_CONCURRENT_AGENTS) in result

        # Different channel should work
        diff_id = mgr.spawn(
            label="other-channel",
            goal="should work",
            channel_id="chan-2",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=hang_cb,
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )
        assert not diff_id.startswith("Error")

        # Cleanup
        for aid in ids + [diff_id]:
            mgr.kill(aid)

    async def test_agent_send_message_to_running(self):
        """Messages sent to a running agent are delivered via inbox."""
        mgr = AgentManager()
        announce = _make_announce_cb()
        received_messages = []

        call_count = {"n": 0}

        async def tracking_iteration(messages, system, tools):
            call_count["n"] += 1
            received_messages.extend(messages)
            if call_count["n"] == 1:
                # First iteration: wait for message injection
                await asyncio.sleep(0.2)
                return _make_llm_response(
                    tool_calls=[{"name": "noop", "input": {}}]
                )
            return _make_llm_response(text="Done with injected context.")

        agent_id = mgr.spawn(
            label="receiver",
            goal="Wait for messages",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=tracking_iteration,
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        # Send a message while agent is running
        await asyncio.sleep(0.05)
        result = mgr.send(agent_id, "Extra context: disk is on /dev/sdb")
        assert "delivered" in result.lower()

        results = await mgr.wait_for_agents([agent_id], timeout=10)
        assert results[agent_id]["status"] == "completed"

        # Verify the inbox message was incorporated
        inbox_msgs = [m for m in received_messages if "Extra context" in m.get("content", "")]
        assert len(inbox_msgs) > 0

    async def test_agent_results_include_secret_scrubbing(self):
        """Tool results containing secrets are scrubbed before storage."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        async def leaky_tool(name, inp):
            return "password=SuperSecret123 api_key=sk-abcdef123"

        responses = [
            _make_llm_response(
                tool_calls=[{"name": "get_config", "input": {}}],
            ),
            _make_llm_response(text="Config retrieved."),
        ]

        agent_id = mgr.spawn(
            label="secret-test",
            goal="Get config",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=leaky_tool,
            announce_callback=announce,
        )

        results = await mgr.wait_for_agents([agent_id], timeout=10)
        r = results[agent_id]
        assert r["status"] == "completed"

        # Check that secrets were scrubbed from agent messages
        agent = mgr._agents[agent_id]
        tool_result_msgs = [
            m for m in agent.messages
            if "Tool result" in m.get("content", "")
        ]
        for msg in tool_result_msgs:
            assert "SuperSecret123" not in msg["content"]
            assert "sk-abcdef123" not in msg["content"]


# ===========================================================================
# 2. Agent Group + Wait Integration
# ===========================================================================

class TestAgentGroupIntegration:
    """Test group spawn, parallel execution, and wait."""

    async def test_spawn_group_and_wait_all(self):
        """spawn_group creates multiple agents that can be awaited together."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        responses = [_make_llm_response(text="Task done.")]

        tasks = [
            {"label": "task-a", "goal": "Do A"},
            {"label": "task-b", "goal": "Do B"},
            {"label": "task-c", "goal": "Do C"},
        ]

        ids = mgr.spawn_group(
            tasks=tasks,
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        assert len(ids) == 3
        assert not any(i.startswith("Error") for i in ids)

        results = await mgr.wait_for_agents(ids, timeout=10)
        assert all(r["status"] == "completed" for r in results.values())

    async def test_wait_for_agents_timeout(self):
        """wait_for_agents returns running agents after timeout."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        async def hang(messages, system, tools):
            await asyncio.sleep(60)
            return _make_llm_response()

        agent_id = mgr.spawn(
            label="hang",
            goal="hang forever",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=hang,
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        results = await mgr.wait_for_agents([agent_id], timeout=0.5)
        assert results[agent_id]["status"] == "running"

        mgr.kill(agent_id)

    async def test_wait_for_nonexistent_agent(self):
        """wait_for_agents handles missing agent IDs gracefully."""
        mgr = AgentManager()
        results = await mgr.wait_for_agents(["nonexistent-id"], timeout=1)
        assert results["nonexistent-id"]["status"] == "not_found"


# ===========================================================================
# 3. Loop-Agent Bridge Integration
# ===========================================================================

class TestLoopAgentBridgeIntegration:
    """Test the bridge between autonomous loops and agent spawning."""

    async def test_bridge_spawn_and_collect(self):
        """Bridge spawns agents for a loop and collects results."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        announce = _make_announce_cb()

        responses = [_make_llm_response(text="Sub-task completed.")]

        tasks = [
            {"label": "monitor-cpu", "goal": "Check CPU"},
            {"label": "monitor-mem", "goal": "Check memory"},
        ]

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop-1",
            iteration=1,
            loop_goal="Monitor system health",
            tasks=tasks,
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        assert len(ids) == 2
        assert not any(i.startswith("Error") for i in ids)

        # Verify agent IDs tracked
        assert bridge.get_loop_agent_count("loop-1") == 2
        assert set(bridge.get_loop_agent_ids("loop-1")) == set(ids)

        # Wait and collect
        results = await bridge.wait_and_collect("loop-1", timeout=10)
        assert len(results) == 2
        assert all(r["status"] == "completed" for r in results.values())

        # Format results for context
        context = bridge.format_agent_results_for_context(results)
        assert "Agent results:" in context
        assert "monitor-cpu" in context
        assert "monitor-mem" in context
        assert "completed" in context

    async def test_bridge_per_iteration_limit(self):
        """Bridge enforces MAX_AGENTS_PER_ITERATION."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)

        tasks = [{"label": f"t-{i}", "goal": "x"} for i in range(MAX_AGENTS_PER_ITERATION + 1)]

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop-2",
            iteration=1,
            loop_goal="test",
            tasks=tasks,
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb([]),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=_make_announce_cb(),
        )

        assert len(ids) == 1
        assert ids[0].startswith("Error")

    async def test_bridge_per_loop_lifetime_limit(self):
        """Bridge enforces MAX_AGENTS_PER_LOOP across iterations."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        announce = _make_announce_cb()
        responses = [_make_llm_response(text="done")]

        # Spawn MAX_AGENTS_PER_LOOP over multiple iterations.
        # Use different channel IDs per batch to avoid per-channel concurrent limit.
        total_spawned = 0
        for iteration in range(MAX_AGENTS_PER_LOOP):
            # Wait for previous agents to finish to not hit concurrent limit
            if iteration > 0 and iteration % MAX_CONCURRENT_AGENTS == 0:
                await asyncio.sleep(0.2)

            ids = bridge.spawn_agents_for_loop(
                loop_id="loop-3",
                iteration=iteration,
                loop_goal="test",
                tasks=[{"label": f"a-{iteration}", "goal": "test"}],
                channel_id=f"chan-{iteration % MAX_CONCURRENT_AGENTS}",
                requester_id="user-1",
                requester_name="Test",
                iteration_callback=_make_iteration_cb(responses),
                tool_executor_callback=_make_tool_exec_cb(),
                announce_callback=announce,
            )
            if ids and not ids[0].startswith("Error"):
                total_spawned += 1

        assert total_spawned == MAX_AGENTS_PER_LOOP

        # Next spawn should be blocked by loop lifetime limit
        ids = bridge.spawn_agents_for_loop(
            loop_id="loop-3",
            iteration=MAX_AGENTS_PER_LOOP + 1,
            loop_goal="test",
            tasks=[{"label": "over-limit", "goal": "test"}],
            channel_id="chan-99",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )
        assert ids[0].startswith("Error")

    async def test_bridge_cleanup_loop(self):
        """cleanup_loop removes agent records for a finished loop."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        announce = _make_announce_cb()
        responses = [_make_llm_response(text="done")]

        bridge.spawn_agents_for_loop(
            loop_id="loop-cleanup",
            iteration=1,
            loop_goal="test",
            tasks=[{"label": "cleanup-test", "goal": "test"}],
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        assert bridge.get_loop_agent_count("loop-cleanup") == 1
        removed = bridge.cleanup_loop("loop-cleanup")
        assert removed == 1
        assert bridge.get_loop_agent_count("loop-cleanup") == 0

    async def test_bridge_enriches_goal_with_loop_context(self):
        """Agents spawned by bridge have loop context in their goal."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        announce = _make_announce_cb()
        responses = [_make_llm_response(text="done")]

        ids = bridge.spawn_agents_for_loop(
            loop_id="loop-ctx",
            iteration=3,
            loop_goal="Monitor servers",
            tasks=[{"label": "ctx-test", "goal": "Check CPU"}],
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        agent = mgr._agents[ids[0]]
        assert "loop-ctx" in agent.goal
        assert "iteration 3" in agent.goal
        assert "Monitor servers" in agent.goal
        assert "Check CPU" in agent.goal

    def test_bridge_format_empty_results(self):
        """format_agent_results_for_context handles empty results."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        assert bridge.format_agent_results_for_context({}) == ""

    def test_bridge_format_truncates_long_results(self):
        """Agent results are truncated to 500 chars in formatted output."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)

        results = {
            "agent-1": {
                "label": "verbose",
                "status": "completed",
                "result": "x" * 1000,
            }
        }

        formatted = bridge.format_agent_results_for_context(results)
        assert "..." in formatted
        # Each result line should be bounded
        for line in formatted.split("\n"):
            if line.startswith("- "):
                assert len(line) < 600  # label + status + 500 + ...


# ===========================================================================
# 4. Agent Tool Filtering Integration
# ===========================================================================

class TestAgentToolFiltering:
    """Test that agent tools are properly filtered."""

    def test_filter_removes_agent_tools(self):
        """filter_agent_tools removes all agent management tools."""
        tools = [
            {"name": "run_command"},
            {"name": "spawn_agent"},
            {"name": "send_to_agent"},
            {"name": "list_agents"},
            {"name": "kill_agent"},
            {"name": "get_agent_results"},
            {"name": "wait_for_agents"},
            {"name": "read_file"},
        ]

        filtered = filter_agent_tools(tools)
        names = {t["name"] for t in filtered}
        assert names == {"run_command", "read_file"}
        assert not names & AGENT_BLOCKED_TOOLS

    def test_filter_preserves_all_non_agent_tools(self):
        """filter_agent_tools keeps all non-agent tools intact."""
        tools = [{"name": f"tool_{i}"} for i in range(50)]
        filtered = filter_agent_tools(tools)
        assert len(filtered) == 50


# ===========================================================================
# 5. Agent Health Check Integration
# ===========================================================================

class TestAgentHealthCheckIntegration:
    """Test agent health check detects and kills stuck agents."""

    async def test_health_check_kills_expired_agents(self):
        """Health check force-kills agents that exceed MAX_AGENT_LIFETIME."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        async def hang(messages, system, tools):
            await asyncio.sleep(60)
            return _make_llm_response()

        agent_id = mgr.spawn(
            label="old-agent",
            goal="be old",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=hang,
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        # Artificially age the agent
        mgr._agents[agent_id].created_at = time.time() - MAX_AGENT_LIFETIME - 10

        result = mgr.check_health()
        assert result["killed"] == 1

        # Agent should be transitioning to killed
        await asyncio.sleep(0.1)
        agent = mgr._agents[agent_id]
        assert agent.status in ("killed", "timeout")

    async def test_health_check_detects_stale_agents(self):
        """Health check logs warnings for idle agents."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        async def hang(messages, system, tools):
            await asyncio.sleep(60)
            return _make_llm_response()

        agent_id = mgr.spawn(
            label="stale-agent",
            goal="be stale",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=hang,
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        # Make agent appear idle but not expired
        mgr._agents[agent_id].last_activity = time.time() - 200

        result = mgr.check_health()
        assert result["stale"] == 1
        assert result["killed"] == 0

        mgr.kill(agent_id)

    async def test_cleanup_removes_old_terminal_agents(self):
        """cleanup() removes agents that finished more than CLEANUP_DELAY ago."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        responses = [_make_llm_response(text="done")]
        agent_id = mgr.spawn(
            label="finished",
            goal="finish fast",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        await mgr.wait_for_agents([agent_id], timeout=10)

        # Artificially age the ended_at timestamp
        mgr._agents[agent_id].ended_at = time.time() - 400

        removed = await mgr.cleanup()
        assert removed == 1
        assert agent_id not in mgr._agents


# ===========================================================================
# 6. Session Management Integration
# ===========================================================================

class TestSessionIntegration:
    """Test session management end-to-end: add, compact, topic detect, relevance."""

    async def test_session_add_and_retrieve(self, tmp_path):
        """Messages are added and retrieved in order."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))
        sm.add_message("chan-1", "user", "Hello", user_id="u1")
        sm.add_message("chan-1", "assistant", "Hi there")
        sm.add_message("chan-1", "user", "How's disk?", user_id="u1")

        history = sm.get_history("chan-1")
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[2]["content"] == "How's disk?"

    async def test_session_compaction_triggers_at_threshold(self, tmp_path):
        """Compaction triggers when messages exceed COMPACTION_THRESHOLD."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))

        compaction_called = {"count": 0}

        async def mock_compact(messages, system):
            compaction_called["count"] += 1
            return "[Topics: testing]\n- Compacted history"

        sm.set_compaction_fn(mock_compact)

        # Add messages beyond threshold
        for i in range(COMPACTION_THRESHOLD + 5):
            sm.add_message("chan-1", "user" if i % 2 == 0 else "assistant",
                           f"Message {i}", user_id="u1")

        # get_task_history should trigger compaction
        history = await sm.get_task_history("chan-1")
        assert compaction_called["count"] == 1

        # Session should have fewer messages now
        session = sm._sessions["chan-1"]
        assert len(session.messages) < COMPACTION_THRESHOLD
        assert session.summary  # Summary should be set

    async def test_session_compaction_preserves_summary(self, tmp_path):
        """Compaction merges existing summary with new messages."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))

        summaries_received = []

        async def tracking_compact(messages, system):
            summaries_received.append(messages[0]["content"])
            return "[Topics: merged]\n- Merged summary"

        sm.set_compaction_fn(tracking_compact)

        # Set initial summary
        session = sm.get_or_create("chan-1")
        session.summary = "Previous context about nginx"

        # Add enough messages to trigger compaction
        for i in range(COMPACTION_THRESHOLD + 5):
            sm.add_message("chan-1", "user" if i % 2 == 0 else "assistant",
                           f"Message {i}", user_id="u1")

        await sm.get_task_history("chan-1")

        # Verify the compaction prompt included the existing summary
        assert len(summaries_received) == 1
        assert "Previous context about nginx" in summaries_received[0]

    def test_topic_change_detection(self, tmp_path):
        """Topic change detected when query has no overlap with recent messages."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))

        # Add messages about nginx
        sm.add_message("chan-1", "user", "Check nginx status on server-a")
        sm.add_message("chan-1", "assistant", "nginx is running on server-a")
        sm.add_message("chan-1", "user", "Restart nginx on server-a")

        # Query about completely different topic
        result = sm.detect_topic_change("chan-1", "What's the weather in Tokyo?")
        assert result["is_topic_change"] is True
        assert result["max_overlap"] < 0.05

    def test_topic_continuity_detection(self, tmp_path):
        """Same topic is NOT detected as a topic change."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))

        sm.add_message("chan-1", "user", "Check nginx status on server-a")
        sm.add_message("chan-1", "assistant", "nginx is running on server-a")

        result = sm.detect_topic_change("chan-1", "Restart nginx on server-a")
        assert result["is_topic_change"] is False
        assert result["max_overlap"] > 0.05

    async def test_relevance_filtering_drops_stale_messages(self, tmp_path):
        """Relevance filter drops older messages with no overlap to current query."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))

        # Old messages about unrelated topic
        sm.add_message("chan-1", "user", "Configure DNS resolver for office network")
        sm.add_message("chan-1", "assistant", "DNS configured to 8.8.8.8")
        sm.add_message("chan-1", "user", "Set up DHCP scope for floor 3")
        sm.add_message("chan-1", "assistant", "DHCP scope configured")

        # Recent messages about docker
        sm.add_message("chan-1", "user", "List docker containers on server-b")
        sm.add_message("chan-1", "assistant", "Found 5 containers running")
        sm.add_message("chan-1", "user", "Restart the nginx container")

        history = await sm.get_task_history(
            "chan-1", max_messages=20,
            current_query="Stop the nginx docker container on server-b",
        )

        # Recent messages should be present
        content_joined = " ".join(h["content"] for h in history)
        assert "docker" in content_joined.lower()
        assert "nginx container" in content_joined.lower()

    async def test_topic_change_shrinks_history(self, tmp_path):
        """Topic change reduces history to just the most recent message."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))

        for i in range(10):
            sm.add_message("chan-1", "user" if i % 2 == 0 else "assistant",
                           f"Old message {i}")

        history = await sm.get_task_history(
            "chan-1", max_messages=20, topic_change=True,
        )

        # Should have only the last message (+ summary pair if summary exists)
        msg_contents = [h["content"] for h in history]
        assert any("Old message 9" in c for c in msg_contents)
        # Should NOT have all old messages
        assert not any("Old message 0" in c for c in msg_contents)

    def test_score_relevance_high_overlap(self):
        """High keyword overlap yields high score."""
        score = score_relevance(
            "check nginx status on server-a",
            "nginx is running on server-a port 80",
        )
        assert score > 0.3

    def test_score_relevance_no_overlap(self):
        """No keyword overlap yields zero score."""
        score = score_relevance(
            "check nginx status",
            "The weather is sunny today",
        )
        assert score == 0.0

    def test_token_budget_enforcement(self):
        """apply_token_budget trims messages to fit within budget."""
        messages = [
            {"role": "user", "content": "x" * 4000}  # ~1000 tokens each
            for _ in range(20)
        ]
        trimmed, dropped = apply_token_budget(messages, budget=5000)
        assert dropped > 0
        total_tokens = sum(estimate_tokens(m["content"]) for m in trimmed)
        assert total_tokens <= 5000

    def test_token_budget_preserves_recent(self):
        """apply_token_budget always keeps the most recent messages."""
        messages = [
            {"role": "user", "content": f"msg-{i} " + "x" * 2000}
            for i in range(10)
        ]
        trimmed, dropped = apply_token_budget(messages, budget=3000)
        # Last 3 should always be preserved
        last_contents = [m["content"] for m in trimmed[-3:]]
        assert any("msg-9" in c for c in last_contents)
        assert any("msg-8" in c for c in last_contents)
        assert any("msg-7" in c for c in last_contents)


# ===========================================================================
# 7. Session Persistence Integration
# ===========================================================================

class TestSessionPersistenceIntegration:
    """Test session save/load cycle preserves data."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Sessions survive a save → new manager → load cycle."""
        persist_dir = str(tmp_path / "sessions")

        # Create and populate
        sm1 = SessionManager(50, 24, persist_dir)
        sm1.add_message("chan-1", "user", "Hello from user", user_id="u1")
        sm1.add_message("chan-1", "assistant", "Hello back")
        session1 = sm1.get_or_create("chan-1")
        session1.summary = "Previous conversation about servers"
        sm1._dirty.add("chan-1")
        sm1.save()

        # Load into new manager
        sm2 = SessionManager(50, 24, persist_dir)
        sm2.load()

        history = sm2.get_history("chan-1")
        # Summary pair + 2 messages = 4
        assert len(history) == 4
        assert "Previous conversation" in history[0]["content"]
        assert history[2]["content"] == "Hello from user"
        assert history[3]["content"] == "Hello back"

    def test_prune_archives_expired_sessions(self, tmp_path):
        """Pruning moves expired sessions to archive."""
        persist_dir = str(tmp_path / "sessions")
        sm = SessionManager(50, 1, persist_dir)  # 1 hour max age

        sm.add_message("chan-1", "user", "Old message")
        session = sm.get_or_create("chan-1")
        session.last_active = time.time() - 7200  # 2 hours ago

        pruned = sm.prune()
        assert pruned == 1
        assert "chan-1" not in sm._sessions

        # Verify archive exists
        archive_dir = Path(persist_dir) / "archive"
        assert archive_dir.exists()
        archives = list(archive_dir.glob("chan-1_*.json"))
        assert len(archives) == 1

    def test_continuity_from_archive(self, tmp_path):
        """New session carries forward summary from recent archive."""
        persist_dir = str(tmp_path / "sessions")

        # Create archive manually
        archive_dir = Path(persist_dir) / "archive"
        archive_dir.mkdir(parents=True)
        archive_data = {
            "channel_id": "chan-1",
            "messages": [],
            "last_active": time.time() - 3600,  # 1 hour ago (within 48hr window)
            "summary": "Previous: nginx configured on server-a",
        }
        (archive_dir / "chan-1_12345.json").write_text(json.dumps(archive_data))

        sm = SessionManager(50, 24, persist_dir)
        session = sm.get_or_create("chan-1")
        assert "nginx configured" in session.summary
        assert "Continuing from previous" in session.summary


# ===========================================================================
# 8. Tool Response Summarization Integration
# ===========================================================================

class TestToolResponseSummarization:
    """Test tool response summarization for history storage."""

    def test_short_response_not_summarized(self):
        """Responses under threshold are returned unchanged."""
        result = summarize_tool_response(
            "Disk is at 45%",
            ["run_command", "check_disk"],
        )
        assert result == "Disk is at 45%"

    def test_long_response_with_many_tools_summarized(self):
        """Long responses with many tool calls are summarized."""
        long_text = "Step 1: checked disk\n\n" * 50 + "Final result: All good."
        tools = [f"tool_{i}" for i in range(15)]

        result = summarize_tool_response(long_text, tools)
        assert "[Task used" in result
        assert len(result) <= 600  # generous bound
        assert "All good" in result

    def test_summarization_preserves_outcome(self):
        """Summarization keeps the final paragraph."""
        text = (
            "I ran several commands to check the system.\n\n"
            "First I checked CPU which was at 20%.\n\n"
            "Then I checked memory which was at 60%.\n\n"
            "Final status: All systems operational, CPU 20%, RAM 60%."
        )
        tools = [f"t{i}" for i in range(12)]

        result = summarize_tool_response(text, tools)
        assert "All systems operational" in result

    def test_summarization_deduplicates_tools(self):
        """Repeated tool names are deduplicated in the summary."""
        tools = ["run_command"] * 15
        text = "a" * 600
        result = summarize_tool_response(text, tools)
        assert "15 tool calls" in result
        # Should list run_command only once
        assert result.count("run_command") == 1


# ===========================================================================
# 9. Web Chat Integration
# ===========================================================================

class TestWebChatIntegration:
    """Test web chat processing end-to-end."""

    async def test_web_chat_basic_flow(self, tmp_path):
        """Web chat processes a message and returns response."""
        from src.llm.types import LLMResponse

        bot = _make_mock_bot(tmp_path, codex_responses=True)
        bot._process_with_tools = AsyncMock(
            return_value=("Hello from Heimdall.", False, False, [], False)
        )

        result = await process_web_chat(
            bot, "Hello", channel_id="web-chan-1",
            user_id="web-user", username="WebUser",
        )

        assert result["response"] == "Hello from Heimdall."
        assert result["is_error"] is False
        assert result["tools_used"] == []

    async def test_web_chat_with_tools(self, tmp_path):
        """Web chat response with tools is saved to session."""
        bot = _make_mock_bot(tmp_path, codex_responses=True)
        bot._process_with_tools = AsyncMock(
            return_value=("Disk at 45%.", False, False, ["run_command"], False)
        )

        result = await process_web_chat(
            bot, "Check disk", channel_id="web-chan-2",
            user_id="web-user", username="WebUser",
        )

        assert result["tools_used"] == ["run_command"]

        # Verify session was updated
        history = bot.sessions.get_history("web-chan-2")
        assert any("Check disk" in h["content"] for h in history)
        assert any("Disk at 45%" in h["content"] for h in history)

    async def test_web_chat_no_codex_returns_error(self, tmp_path):
        """Web chat without codex client returns error."""
        bot = _make_mock_bot(tmp_path)
        assert bot.codex_client is None

        result = await process_web_chat(
            bot, "Hello", channel_id="web-chan-3",
        )

        assert result["is_error"] is True
        assert "No LLM backend" in result["response"]

    async def test_web_chat_error_saves_sanitized_message(self, tmp_path):
        """On error, a sanitized message is saved to session."""
        bot = _make_mock_bot(tmp_path, codex_responses=True)
        bot._process_with_tools = AsyncMock(
            return_value=("Error occurred.", False, True, ["run_command"], False)
        )

        result = await process_web_chat(
            bot, "Do something", channel_id="web-chan-4",
        )

        assert result["is_error"] is True

        # Verify sanitized error message in session
        history = bot.sessions.get_history("web-chan-4")
        error_msgs = [h for h in history if "encountered an error" in h["content"]]
        assert len(error_msgs) == 1

    async def test_web_chat_scrubs_secrets(self, tmp_path):
        """Web chat scrubs secrets from response."""
        bot = _make_mock_bot(tmp_path, codex_responses=True)
        bot._process_with_tools = AsyncMock(
            return_value=(
                "Found password=SuperSecret api_key=sk-abc123",
                False, False, [], False,
            )
        )

        result = await process_web_chat(
            bot, "Get config", channel_id="web-chan-5",
        )

        assert "SuperSecret" not in result["response"]
        assert "sk-abc123" not in result["response"]

    async def test_web_chat_exception_returns_error(self, tmp_path):
        """Unhandled exception in processing returns error dict."""
        bot = _make_mock_bot(tmp_path, codex_responses=True)
        bot._process_with_tools = AsyncMock(side_effect=RuntimeError("boom"))

        result = await process_web_chat(
            bot, "Crash", channel_id="web-chan-6",
        )

        assert result["is_error"] is True
        assert "Error processing" in result["response"]

    async def test_web_chat_content_limit(self, tmp_path):
        """Web chat uses virtual message objects with proper structure."""
        msg = WebMessage(channel_id="ch-1", user_id="u-1", username="Bob")
        assert msg.channel.id == "ch-1"
        assert msg.author.id == "u-1"
        assert str(msg.author) == "Bob"
        assert msg.author.bot is False
        assert msg.webhook_id is None
        assert msg.attachments == []

        # Channel typing is a no-op context manager
        async with msg.channel.typing():
            pass

        # Channel send returns a no-op message
        sent = await msg.channel.send("test")
        await sent.edit(content="new")  # Should not raise

    async def test_web_chat_tool_only_response_saved(self, tmp_path):
        """Chat-only response (no tools, no handoff) is NOT saved to session."""
        bot = _make_mock_bot(tmp_path, codex_responses=True)
        bot._process_with_tools = AsyncMock(
            return_value=("Just chatting.", False, False, [], False)  # no tools, no handoff
        )

        await process_web_chat(
            bot, "Hi there", channel_id="web-chan-7",
        )

        # Should have the user message but NOT the assistant response
        history = bot.sessions.get_history("web-chan-7")
        assistant_msgs = [h for h in history if h["role"] == "assistant"]
        assert not any("Just chatting" in m["content"] for m in assistant_msgs)

    async def test_web_chat_handoff_response_saved(self, tmp_path):
        """Handoff response is saved to session even without tools."""
        bot = _make_mock_bot(tmp_path, codex_responses=True)
        bot._process_with_tools = AsyncMock(
            return_value=("Skill result.", False, False, [], True)  # handoff=True
        )

        await process_web_chat(
            bot, "Run my skill", channel_id="web-chan-8",
        )

        history = bot.sessions.get_history("web-chan-8")
        assert any("Skill result" in h["content"] for h in history)


# ===========================================================================
# 10. Detection Systems Integration
# ===========================================================================

class TestDetectionSystemsIntegration:
    """Test all 5 detection systems with realistic responses."""

    # --- Fabrication Detection ---

    def test_fabrication_detects_claimed_execution(self):
        """Fabrication detected when LLM claims to have run commands."""
        assert detect_fabrication(
            "I ran the command and checked disk usage. The output shows 45% used.",
            [],
        )

    def test_fabrication_not_triggered_with_tools(self):
        """Fabrication NOT detected when tools were actually used."""
        assert not detect_fabrication(
            "I ran the command and checked disk usage. The output shows 45% used.",
            ["run_command"],
        )

    def test_fabrication_detects_fake_output_block(self):
        """Fabrication detected with fake terminal output."""
        text = (
            "Here are the results:\n"
            "```bash\n"
            "$ df -h\n"
            "Filesystem  Size  Use%  Mounted\n"
            "/dev/sda1   50G   45%   /\n"
            "```"
        )
        assert detect_fabrication(text, [])

    def test_fabrication_ignores_short_text(self):
        """Very short text is not flagged as fabrication."""
        assert not detect_fabrication("Ok.", [])

    def test_fabrication_detects_action_claims(self):
        """Fabrication detected when claiming completed actions."""
        assert detect_fabrication(
            "I generated and posted the image file to the channel.",
            [],
        )

    # --- Tool Unavailable Detection ---

    def test_tool_unavail_detects_false_claims(self):
        """Tool-unavailability detected when claiming tool is disabled."""
        assert detect_tool_unavailable(
            "The image generation tool is not enabled in this instance.",
            [],
        )

    def test_tool_unavail_not_triggered_with_tools(self):
        """Tool-unavailability NOT triggered when tools were called."""
        assert not detect_tool_unavailable(
            "The tool is not available.",
            ["generate_image"],
        )

    def test_tool_unavail_detects_capability_claims(self):
        """Detects claims of lacking access."""
        assert detect_tool_unavailable(
            "I don't have access to generate images in this environment.",
            [],
        )

    # --- Hedging Detection ---

    def test_hedging_detects_permission_asking(self):
        """Hedging detected when asking for permission."""
        assert detect_hedging(
            "Would you like me to check the disk usage on server-a?",
            [],
        )

    def test_hedging_detects_plan_proposal(self):
        """Hedging detected when proposing a plan instead of executing."""
        assert detect_hedging(
            "Here's a plan: First, I'll check the DNS settings, then review the firewall.",
            [],
        )

    def test_hedging_not_triggered_with_tools(self):
        """Hedging NOT triggered when tools were used."""
        assert not detect_hedging(
            "Shall I also check the memory?",
            ["run_command"],
        )

    def test_hedging_detects_awaiting_confirmation(self):
        """Hedging detected when awaiting confirmation."""
        assert detect_hedging(
            "I'll wait for your go-ahead before making changes.",
            [],
        )

    # --- Code Hedging Detection ---

    def test_code_hedging_detects_bash_block(self):
        """Code hedging detected when showing bash code block."""
        text = (
            "You can run this command:\n"
            "```bash\n"
            "systemctl restart nginx\n"
            "```"
        )
        assert detect_code_hedging(text, [])

    def test_code_hedging_not_triggered_with_tools(self):
        """Code hedging NOT triggered when tools were used."""
        text = "```bash\ndf -h\n```"
        assert not detect_code_hedging(text, ["run_command"])

    def test_code_hedging_ignores_non_bash_blocks(self):
        """Non-bash code blocks don't trigger code hedging."""
        text = "Here's the config:\n```json\n{\"key\": \"value\"}\n```"
        assert not detect_code_hedging(text, [])

    def test_code_hedging_detects_sh_block(self):
        """sh code blocks also trigger code hedging."""
        text = "```sh\nls -la\n```"
        assert detect_code_hedging(text, [])

    # --- Premature Failure Detection ---

    def test_premature_failure_detects_giving_up(self):
        """Premature failure detected when giving up after tools were used."""
        assert detect_premature_failure(
            "I couldn't get the DNS records. The server might be down.",
            ["run_command"],
        )

    def test_premature_failure_requires_tools(self):
        """Premature failure NOT triggered without tool usage."""
        assert not detect_premature_failure(
            "Couldn't find the file.",
            [],
        )

    def test_premature_failure_detects_workaround_suggestions(self):
        """Premature failure detected when suggesting workarounds."""
        assert detect_premature_failure(
            "The API returned an error. Try this workaround instead: use curl directly.",
            ["run_command"],
        )

    def test_premature_failure_detects_connection_issues(self):
        """Premature failure detected for connection failures."""
        assert detect_premature_failure(
            "The service doesn't seem to be working. Connection refused on port 443.",
            ["run_command"],
        )

    def test_premature_failure_ignores_short_text(self):
        """Very short text is not flagged as premature failure."""
        assert not detect_premature_failure("Error", ["run_command"])

    # --- Detection Interactions ---

    def test_fabrication_and_hedging_mutual_exclusion(self):
        """Text that hedges but doesn't fabricate should only trigger hedging."""
        text = "Would you like me to check the system status?"
        assert detect_hedging(text, [])
        # This text doesn't claim to have done anything
        assert not detect_fabrication(text, [])

    def test_all_detections_safe_with_empty_text(self):
        """All detectors handle empty/None text gracefully."""
        for fn in [detect_fabrication, detect_hedging, detect_code_hedging,
                    detect_tool_unavailable]:
            assert not fn("", [])
            assert not fn(None, [])  # type: ignore

    def test_premature_failure_safe_with_empty(self):
        """Premature failure detector handles empty text."""
        assert not detect_premature_failure("", ["tool"])
        assert not detect_premature_failure(None, ["tool"])  # type: ignore


# ===========================================================================
# 11. Secret Scrubbing Integration
# ===========================================================================

class TestSecretScrubbingIntegration:
    """Test that secrets are scrubbed across different data flows."""

    def test_scrub_passwords(self):
        """Password patterns are scrubbed."""
        text = "Connected with password=MySecret123"
        scrubbed = scrub_output_secrets(text)
        assert "MySecret123" not in scrubbed

    def test_scrub_api_keys(self):
        """API key patterns are scrubbed."""
        text = "Using api_key=sk-1234567890abcdef"
        scrubbed = scrub_output_secrets(text)
        assert "sk-1234567890abcdef" not in scrubbed

    def test_scrub_openai_keys(self):
        """OpenAI sk- keys are scrubbed."""
        text = "key: sk-abcdefghijklmnopqrstuvwxyz12345"
        scrubbed = scrub_output_secrets(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz12345" not in scrubbed

    def test_scrub_github_tokens(self):
        """GitHub tokens are scrubbed."""
        text = "token=ghp_abcdefghijklmnopqrstuvwxyz123456"
        scrubbed = scrub_output_secrets(text)
        assert "ghp_abcdefghijklmnopqrstuvwxyz" not in scrubbed

    def test_scrub_aws_keys(self):
        """AWS access keys are scrubbed."""
        text = "AWS key: AKIAIOSFODNN7EXAMPLE"
        scrubbed = scrub_output_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed

    def test_scrub_preserves_safe_text(self):
        """Non-secret text is preserved intact."""
        text = "Disk usage is 45% on /dev/sda1"
        assert scrub_output_secrets(text) == text

    def test_scrub_multiple_secrets(self):
        """Multiple different secret types are all scrubbed."""
        text = (
            "Config:\n"
            "  password=Secret123\n"
            "  api_key=sk-abcdefghijklmnopqrstuvwxyz\n"
            "  AKIAIOSFODNN7EXAMPLE\n"
            "Status: ok"
        )
        scrubbed = scrub_output_secrets(text)
        assert "Secret123" not in scrubbed
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in scrubbed
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
        assert "Status: ok" in scrubbed


# ===========================================================================
# 12. Skill Manager Integration
# ===========================================================================

class TestSkillManagerIntegration:
    """Test skill loading, registration, and execution."""

    def test_skill_create_and_list(self, tmp_path):
        """Creating a skill makes it available in list."""
        from src.tools.skill_manager import SkillManager

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        sm = SkillManager(str(skills_dir), tool_executor=MagicMock())
        result = sm.create_skill("test_skill", '''
SKILL_DEFINITION = {
    "name": "test_skill",
    "description": "A test skill",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(tool_input, context):
    return "Hello from test skill"
''')

        assert "created" in result.lower() or "loaded" in result.lower() or "success" in result.lower() or "test_skill" in result.lower()

        skills = sm.list_skills()
        names = [s["name"] for s in skills]
        assert "test_skill" in names

    def test_skill_tool_definitions(self, tmp_path):
        """Created skills generate tool definitions."""
        from src.tools.skill_manager import SkillManager

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        sm = SkillManager(str(skills_dir), tool_executor=MagicMock())
        sm.create_skill("my_tool", '''
SKILL_DEFINITION = {
    "name": "my_tool",
    "description": "Does something useful",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}

async def execute(tool_input, context):
    return f"Result for: {tool_input.get('query', '')}"
''')

        defs = sm.get_tool_definitions()
        assert any(d["name"] == "my_tool" for d in defs)

    def test_skill_disable_enable(self, tmp_path):
        """Disabled skills are excluded from tool definitions."""
        from src.tools.skill_manager import SkillManager

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        sm = SkillManager(str(skills_dir), tool_executor=MagicMock())
        sm.create_skill("toggle_skill", '''
SKILL_DEFINITION = {
    "name": "toggle_skill",
    "description": "Can be toggled",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(tool_input, context):
    return "toggled"
''')

        assert sm.is_enabled("toggle_skill")
        assert any(d["name"] == "toggle_skill" for d in sm.get_tool_definitions())

        sm.disable_skill("toggle_skill")
        assert not sm.is_enabled("toggle_skill")
        assert not any(d["name"] == "toggle_skill" for d in sm.get_tool_definitions())

        sm.enable_skill("toggle_skill")
        assert sm.is_enabled("toggle_skill")
        assert any(d["name"] == "toggle_skill" for d in sm.get_tool_definitions())

    def test_skill_delete(self, tmp_path):
        """Deleting a skill removes it from list and definitions."""
        from src.tools.skill_manager import SkillManager

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        sm = SkillManager(str(skills_dir), tool_executor=MagicMock())
        sm.create_skill("deleteme", '''
SKILL_DEFINITION = {
    "name": "deleteme",
    "description": "Will be deleted",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(tool_input, context):
    return "x"
''')

        assert sm.has_skill("deleteme")
        sm.delete_skill("deleteme")
        assert not sm.has_skill("deleteme")

    async def test_skill_execution(self, tmp_path):
        """Skills can be executed and return results."""
        from src.tools.skill_manager import SkillManager

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        sm = SkillManager(str(skills_dir), tool_executor=MagicMock())
        sm.create_skill("exec_skill", '''
SKILL_DEFINITION = {
    "name": "exec_skill",
    "description": "Executable skill",
    "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
}

async def execute(tool_input, context):
    name = tool_input.get("name", "World")
    return f"Hello, {name}!"
''')

        result = await sm.execute("exec_skill", {"name": "Heimdall"})
        assert "Hello, Heimdall!" in result

    def test_skill_validation(self, tmp_path):
        """Invalid skill code is rejected on validation."""
        from src.tools.skill_manager import SkillManager

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        sm = SkillManager(str(skills_dir), tool_executor=MagicMock())

        # Missing SKILL_DEFINITION
        result = sm.validate_skill_code("def foo(): pass", "test.py")
        assert not result["valid"] or "SKILL_DEFINITION" in str(result.get("errors", []))

    def test_skill_edit_updates_code(self, tmp_path):
        """Editing a skill replaces its code file and reloads."""
        from src.tools.skill_manager import SkillManager

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        sm = SkillManager(str(skills_dir), tool_executor=MagicMock())
        sm.create_skill("editable", '''
SKILL_DEFINITION = {
    "name": "editable",
    "description": "Version 1",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(tool_input, context):
    return "v1"
''')

        info1 = sm.get_skill_info("editable")
        assert "Version 1" in info1["description"]

        result = sm.edit_skill("editable", '''
SKILL_DEFINITION = {
    "name": "editable",
    "description": "Version 2",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(tool_input, context):
    return "v2"
''')

        # Verify the file was updated on disk
        skill_file = skills_dir / "editable.py"
        assert "Version 2" in skill_file.read_text()

        # The edit_skill result should indicate success
        assert "error" not in result.lower() or "updated" in result.lower() or "success" in result.lower()

    def test_skill_metadata_extraction(self, tmp_path):
        """Skill metadata (version, author, tags) is extracted."""
        from src.tools.skill_manager import SkillManager

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        sm = SkillManager(str(skills_dir), tool_executor=MagicMock())
        sm.create_skill("meta_skill", '''
SKILL_DEFINITION = {
    "name": "meta_skill",
    "description": "Skill with metadata",
    "input_schema": {"type": "object", "properties": {}},
    "version": "1.2.3",
    "author": "TestAuthor",
    "tags": ["monitoring", "alerts"],
}

async def execute(tool_input, context):
    return "meta"
''')

        info = sm.get_skill_info("meta_skill")
        assert info is not None
        assert info.get("metadata", {}).get("version") == "1.2.3"
        assert info.get("metadata", {}).get("author") == "TestAuthor"


# ===========================================================================
# 13. Skill Context Integration
# ===========================================================================

class TestSkillContextIntegration:
    """Test SkillContext API surface."""

    def test_skill_safe_tools_list(self):
        """SKILL_SAFE_TOOLS contains expected read-only tools."""
        from src.tools.skill_context import SKILL_SAFE_TOOLS

        # Should include read-only tools
        assert "read_file" in SKILL_SAFE_TOOLS
        assert "check_disk" in SKILL_SAFE_TOOLS
        assert "search_knowledge" in SKILL_SAFE_TOOLS

        # Should NOT include destructive tools
        assert "run_command" not in SKILL_SAFE_TOOLS
        assert "write_file" not in SKILL_SAFE_TOOLS
        assert "run_script" not in SKILL_SAFE_TOOLS

    def test_skill_context_path_denial(self):
        """SkillContext blocks access to sensitive paths."""
        from src.tools.skill_context import is_path_denied

        # Check that sensitive paths are denied
        denied_paths = [".env", "/etc/shadow"]
        for path in denied_paths:
            assert is_path_denied(path), f"Expected {path} to be denied"

        # Safe paths should be allowed
        assert not is_path_denied("/var/log/syslog")

    def test_skill_context_remember_recall(self, tmp_path):
        """SkillContext memory (remember/recall) persists across calls."""
        from src.tools.skill_context import SkillContext

        memory_path = str(tmp_path / "skill_memory.json")
        ctx = SkillContext(
            tool_executor=MagicMock(),
            skill_name="test",
            memory_path=memory_path,
        )

        ctx.remember("server_ip", "10.0.0.1")
        value = ctx.recall("server_ip")
        assert value == "10.0.0.1"

        # Non-existent key returns None
        assert ctx.recall("nonexistent") is None

    def test_skill_context_resource_limits(self):
        """SkillContext tracks and enforces resource limits."""
        from src.tools.skill_context import ResourceTracker

        tracker = ResourceTracker()
        assert tracker.tool_calls == 0
        assert tracker.http_requests == 0
        assert tracker.messages_sent == 0
        assert tracker.files_sent == 0


# ===========================================================================
# 14. Config Integration
# ===========================================================================

class TestConfigIntegration:
    """Test configuration model integration."""

    def test_config_roundtrip(self, config):
        """Config model can be serialized and deserialized."""
        data = config.model_dump()
        from src.config.schema import Config
        restored = Config(**data)
        assert restored.discord.token == config.discord.token
        assert len(restored.tools.hosts) == len(config.tools.hosts)

    def test_config_tool_packs_default_empty(self, config):
        """Empty tool_packs means all tools loaded."""
        assert config.tools.tool_packs == []

    def test_config_hosts_accessible(self, config):
        """Tool hosts are accessible by alias."""
        assert "server" in config.tools.hosts
        assert config.tools.hosts["server"].address == "10.0.0.1"

    def test_config_ssh_paths(self, config):
        """SSH paths are set correctly."""
        assert config.tools.ssh_key_path is not None
        assert config.tools.ssh_known_hosts_path is not None


# ===========================================================================
# 15. Tool Registry Integration
# ===========================================================================

class TestToolRegistryIntegration:
    """Test tool registry and pack system."""

    def test_tool_definitions_loaded(self):
        """Tool definitions are loaded and have required fields."""
        from src.tools.registry import get_tool_definitions

        tools = get_tool_definitions()
        assert len(tools) > 0

        for tool in tools:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing 'description'"
            assert "input_schema" in tool, f"Tool {tool.get('name')} missing 'input_schema'"

    def test_tool_packs_filtering(self):
        """enabled_packs parameter filters tool definitions."""
        from src.tools.registry import get_tool_definitions, get_pack_tool_names

        all_tools = get_tool_definitions()
        systemd_tools = get_pack_tool_names("systemd")

        if systemd_tools:
            # With only systemd pack, should have core + systemd tools
            filtered = get_tool_definitions(enabled_packs=["systemd"])
            filtered_names = {t["name"] for t in filtered}

            # Systemd tools should be present
            for name in systemd_tools:
                assert name in filtered_names

    def test_no_duplicate_tool_names(self):
        """All tool definitions have unique names."""
        from src.tools.registry import get_tool_definitions

        tools = get_tool_definitions()
        names = [t["name"] for t in tools]
        assert len(names) == len(set(names)), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"

    def test_tool_defs_cache_invalidation(self):
        """Cache invalidation produces fresh results."""
        from src.tools.registry import get_tool_definitions, invalidate_tool_defs_cache

        tools1 = get_tool_definitions()
        invalidate_tool_defs_cache()
        tools2 = get_tool_definitions()

        # Both should have the same tools
        assert len(tools1) == len(tools2)


# ===========================================================================
# 16. Cross-System: Web Chat + Session + Detection
# ===========================================================================

class TestWebChatSessionDetectionIntegration:
    """Test that web chat, sessions, and detection work together."""

    async def test_web_chat_preserves_session_across_messages(self, tmp_path):
        """Multiple web chat messages build up session history."""
        bot = _make_mock_bot(tmp_path, codex_responses=True)
        bot._process_with_tools = AsyncMock(
            return_value=("Response 1.", False, False, ["tool_a"], False)
        )

        await process_web_chat(bot, "First message", channel_id="multi-1")

        bot._process_with_tools = AsyncMock(
            return_value=("Response 2.", False, False, ["tool_b"], False)
        )
        await process_web_chat(bot, "Second message", channel_id="multi-1")

        history = bot.sessions.get_history("multi-1")
        contents = [h["content"] for h in history]
        assert any("First message" in c for c in contents)
        assert any("Second message" in c for c in contents)
        assert any("Response 1" in c for c in contents)
        assert any("Response 2" in c for c in contents)

    async def test_detection_patterns_comprehensive_coverage(self):
        """All detection patterns fire on their intended inputs."""
        # Fabrication
        fabrication_texts = [
            "I checked the disk and found 45% usage.",
            "I ran the command and here is the output:",
            "I verified that nginx is running correctly.",
        ]
        for text in fabrication_texts:
            assert detect_fabrication(text, []), f"Should detect fabrication: {text}"

        # Hedging
        hedging_texts = [
            "Shall I proceed with the disk check?",
            "Would you like me to restart the service?",
            "Let me know if you want me to continue.",
        ]
        for text in hedging_texts:
            assert detect_hedging(text, []), f"Should detect hedging: {text}"

        # Code hedging
        code_hedging_texts = [
            "You can use:\n```bash\ndf -h\n```",
            "Try running:\n```sh\nsystemctl status nginx\n```",
        ]
        for text in code_hedging_texts:
            assert detect_code_hedging(text, []), f"Should detect code hedging: {text}"

        # Premature failure
        failure_texts = [
            "I couldn't resolve the hostname. The DNS server might be down.",
            "Failed to connect to the API. Try this workaround instead.",
            "The connection timed out when reaching the service.",
        ]
        for text in failure_texts:
            assert detect_premature_failure(text, ["run_command"]), (
                f"Should detect premature failure: {text}"
            )

    async def test_detection_safe_texts_not_flagged(self):
        """Normal operational text should not trigger any detector."""
        safe_texts = [
            "The disk usage on server-a is 45%.",
            "Nginx is running on port 80.",
            "Container 'redis' was restarted successfully.",
            "All 5 services are healthy.",
        ]
        for text in safe_texts:
            assert not detect_fabrication(text, ["run_command"]), (
                f"Should not detect fabrication with tools: {text}"
            )
            assert not detect_hedging(text, ["run_command"]), (
                f"Should not detect hedging with tools: {text}"
            )
            assert not detect_code_hedging(text, ["run_command"]), (
                f"Should not detect code hedging with tools: {text}"
            )


# ===========================================================================
# 17. Agent + Secret Scrubbing + Announce Integration
# ===========================================================================

class TestAgentSilentBehaviorIntegration:
    """Test agents are silent internal workers — no Discord posting."""

    async def test_completed_agent_stores_result_no_announce(self):
        """Completed agent stores result internally, does not announce."""
        mgr = AgentManager()
        announce = _make_announce_cb()

        responses = [_make_llm_response(text="All servers healthy. CPU at 15%.")]

        agent_id = mgr.spawn(
            label="health-report",
            goal="Check server health",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        await mgr.wait_for_agents([agent_id], timeout=10)

        # Agent should NOT post to Discord
        assert len(announce.calls) == 0
        # Result stored internally
        results = mgr.get_results(agent_id)
        assert results["status"] == "completed"
        assert "All servers healthy" in results["result"]

    async def test_results_collected_via_wait_for_agents(self):
        """Parent collects agent results via wait_for_agents."""
        mgr = AgentManager()

        responses = [
            _make_llm_response(text="Disk at 42%, all good."),
        ]

        agent_id = mgr.spawn(
            label="disk-check",
            goal="Check disk",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
        )

        results = await mgr.wait_for_agents([agent_id], timeout=10)
        assert agent_id in results
        assert results[agent_id]["status"] == "completed"
        assert "42%" in results[agent_id]["result"]

    async def test_long_result_stored_in_full(self):
        """Long results are stored in full internally (no truncation)."""
        mgr = AgentManager()

        responses = [_make_llm_response(text="x" * 3000)]

        agent_id = mgr.spawn(
            label="verbose",
            goal="test",
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
        )

        results = await mgr.wait_for_agents([agent_id], timeout=10)
        assert len(results[agent_id]["result"]) == 3000


# ===========================================================================
# 18. Session + Compaction + Reflection Integration
# ===========================================================================

class TestSessionCompactionReflectionIntegration:
    """Test that compaction triggers reflection on discarded messages."""

    async def test_compaction_triggers_reflection(self, tmp_path):
        """Compaction calls reflector for discarded messages."""
        reflector = MagicMock()
        reflector.reflect_on_compacted = AsyncMock()

        sm = SessionManager(
            50, 24, str(tmp_path / "sessions"),
            reflector=reflector,
        )

        async def mock_compact(messages, system):
            return "[Topics: test]\n- Compacted"

        sm.set_compaction_fn(mock_compact)

        # Add enough messages to trigger compaction
        for i in range(COMPACTION_THRESHOLD + 10):
            sm.add_message("chan-1", "user" if i % 2 == 0 else "assistant",
                           f"Message {i}", user_id="u1")

        await sm.get_task_history("chan-1")

        # Give the reflection task time to fire
        await asyncio.sleep(0.1)

        reflector.reflect_on_compacted.assert_called_once()

    async def test_compaction_without_reflector_still_works(self, tmp_path):
        """Compaction works even without a reflector."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))

        async def mock_compact(messages, system):
            return "[Topics: test]\n- Compacted"

        sm.set_compaction_fn(mock_compact)

        for i in range(COMPACTION_THRESHOLD + 10):
            sm.add_message("chan-1", "user" if i % 2 == 0 else "assistant",
                           f"Msg {i}", user_id="u1")

        history = await sm.get_task_history("chan-1")
        assert len(history) > 0

        session = sm._sessions["chan-1"]
        assert session.summary  # Should have a summary
        assert len(session.messages) < COMPACTION_THRESHOLD

    async def test_compaction_without_fn_falls_back(self, tmp_path):
        """Without compaction_fn, fallback trimming preserves existing summary."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))
        # No compaction_fn set

        session = sm.get_or_create("chan-1")
        session.summary = "Existing summary about nginx"

        for i in range(COMPACTION_THRESHOLD + 10):
            sm.add_message("chan-1", "user" if i % 2 == 0 else "assistant",
                           f"Msg {i}", user_id="u1")

        await sm.get_task_history("chan-1")

        # Summary should be preserved despite failed compaction
        assert "nginx" in session.summary


# ===========================================================================
# 19. Session Remove + Scrub Integration
# ===========================================================================

class TestSessionRemoveAndScrub:
    """Test message removal and secret scrubbing in sessions."""

    def test_remove_last_message(self, tmp_path):
        """remove_last_message removes the most recent matching message."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))
        sm.add_message("chan-1", "user", "Hello")
        sm.add_message("chan-1", "assistant", "Hi")

        assert sm.remove_last_message("chan-1", "assistant")
        history = sm.get_history("chan-1")
        assert len(history) == 1
        assert history[0]["content"] == "Hello"

    def test_remove_last_message_wrong_role(self, tmp_path):
        """remove_last_message returns False if role doesn't match."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))
        sm.add_message("chan-1", "user", "Hello")
        sm.add_message("chan-1", "assistant", "Hi")

        assert not sm.remove_last_message("chan-1", "user")

    def test_scrub_secrets_from_session(self, tmp_path):
        """scrub_secrets removes messages containing secrets."""
        sm = SessionManager(50, 24, str(tmp_path / "sessions"))
        sm.add_message("chan-1", "user", "password=MySecret123")
        sm.add_message("chan-1", "assistant", "Got it")
        sm.add_message("chan-1", "user", "Normal message")

        removed = sm.scrub_secrets("chan-1", "password=MySecret123")
        assert removed

        history = sm.get_history("chan-1")
        assert not any("MySecret123" in h["content"] for h in history)
        assert any("Normal message" in h["content"] for h in history)


# ===========================================================================
# 20. End-to-End: Agent Spawned from Web Chat Context
# ===========================================================================

class TestAgentWebChatEndToEnd:
    """Test agent spawning in a web chat context."""

    async def test_agent_shares_session_context(self, tmp_path):
        """Agents spawned from web chat use the same session manager."""
        bot = _make_mock_bot(tmp_path, codex_responses=True)
        bot._process_with_tools = AsyncMock(
            return_value=("Task done.", False, False, ["spawn_agent"], False)
        )

        # Process a web chat message
        await process_web_chat(bot, "Deploy to server-a", channel_id="agent-web-1")

        # Verify session was updated
        history = bot.sessions.get_history("agent-web-1")
        assert len(history) > 0

    async def test_agent_manager_list_filtering(self):
        """Agent listing filters by channel correctly."""
        mgr = AgentManager()
        announce = _make_announce_cb()
        responses = [_make_llm_response(text="done")]

        mgr.spawn(
            label="chan1-agent", goal="test", channel_id="100",
            requester_id="u1", requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )
        mgr.spawn(
            label="chan2-agent", goal="test", channel_id="200",
            requester_id="u1", requester_name="Test",
            iteration_callback=_make_iteration_cb(responses),
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        # Wait for both to complete
        all_agents = mgr.list()
        assert len(all_agents) == 2

        chan1_agents = mgr.list(channel_id="100")
        assert len(chan1_agents) == 1
        assert chan1_agents[0]["label"] == "chan1-agent"

        chan2_agents = mgr.list(channel_id="200")
        assert len(chan2_agents) == 1
        assert chan2_agents[0]["label"] == "chan2-agent"


# ===========================================================================
# 21. Bridge Active Agents Integration
# ===========================================================================

class TestBridgeActiveAgentsIntegration:
    """Test loop bridge active agent tracking."""

    async def test_get_active_loop_agents(self):
        """get_active_loop_agents returns uncollected agents with status."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)
        announce = _make_announce_cb()

        async def hang(messages, system, tools):
            await asyncio.sleep(30)
            return _make_llm_response()

        ids = bridge.spawn_agents_for_loop(
            loop_id="active-test",
            iteration=1,
            loop_goal="test",
            tasks=[
                {"label": "a1", "goal": "test 1"},
                {"label": "a2", "goal": "test 2"},
            ],
            channel_id="chan-1",
            requester_id="user-1",
            requester_name="Test",
            iteration_callback=hang,
            tool_executor_callback=_make_tool_exec_cb(),
            announce_callback=announce,
        )

        await asyncio.sleep(0.05)

        active = bridge.get_active_loop_agents("active-test")
        assert len(active) == 2
        assert all(a["status"] == "running" for a in active)

        # Cleanup
        for aid in ids:
            mgr.kill(aid)

    def test_tracked_loop_count(self):
        """tracked_loop_count reflects number of loops with agents."""
        mgr = AgentManager()
        bridge = LoopAgentBridge(mgr)

        assert bridge.tracked_loop_count == 0

        # Spawn for two different loops
        for loop_id in ["loop-x", "loop-y"]:
            bridge.spawn_agents_for_loop(
                loop_id=loop_id,
                iteration=1,
                loop_goal="test",
                tasks=[{"label": "a", "goal": "test"}],
                channel_id="chan-1",
                requester_id="user-1",
                requester_name="Test",
                iteration_callback=_make_iteration_cb([_make_llm_response(text="done")]),
                tool_executor_callback=_make_tool_exec_cb(),
                announce_callback=_make_announce_cb(),
            )

        assert bridge.tracked_loop_count == 2

        bridge.cleanup_loop("loop-x")
        assert bridge.tracked_loop_count == 1
