"""Tests for partial completion reporting in the tool loop.

Round 13: When the tool loop fails (API error, max iterations), include a
human-readable summary of what was already accomplished before the failure.
This uses the progress_steps list to build a report showing completed steps
with tool names and elapsed times.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot, MAX_TOOL_ITERATIONS  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal LokiBot stub for partial completion tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test prompt"
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.codex_client = MagicMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub._send_with_retry = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run command", "input_schema": {"type": "object", "properties": {}}},
        {"name": "check_disk", "description": "Check disk", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed
    stub._build_partial_completion_report = LokiBot._build_partial_completion_report
    return stub


def _make_message():
    """Create a mock Discord message."""
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = "test-chan"
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None),
    ))
    msg.channel.send = AsyncMock(return_value=AsyncMock())
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    return msg


# ---------------------------------------------------------------------------
# Unit tests: _build_partial_completion_report
# ---------------------------------------------------------------------------

class TestBuildPartialCompletionReport:
    """Unit tests for the static _build_partial_completion_report method."""

    def test_empty_steps_returns_empty(self):
        result = LokiBot._build_partial_completion_report([])
        assert result == ""

    def test_no_done_steps_returns_empty(self):
        steps = [
            {"tools": ["run_command"], "status": "running", "reasoning": None},
        ]
        result = LokiBot._build_partial_completion_report(steps)
        assert result == ""

    def test_one_done_step(self):
        steps = [
            {"tools": ["check_disk"], "status": "done", "elapsed_ms": 1200, "reasoning": None},
        ]
        result = LokiBot._build_partial_completion_report(steps)
        assert "**Partial completion (1/1 steps):**" in result
        assert "`check_disk`" in result
        assert "1.2s" in result

    def test_multiple_done_steps(self):
        steps = [
            {"tools": ["check_disk"], "status": "done", "elapsed_ms": 1200, "reasoning": None},
            {"tools": ["run_command"], "status": "done", "elapsed_ms": 3400, "reasoning": None},
            {"tools": ["restart_service"], "status": "running", "reasoning": None},
        ]
        result = LokiBot._build_partial_completion_report(steps)
        assert "**Partial completion (2/3 steps):**" in result
        assert "`check_disk`" in result
        assert "1.2s" in result
        assert "`run_command`" in result
        assert "3.4s" in result
        # Running step NOT included
        assert "restart_service" not in result

    def test_multiple_tools_in_one_step(self):
        steps = [
            {"tools": ["check_disk", "check_memory"], "status": "done", "elapsed_ms": 2000, "reasoning": None},
        ]
        result = LokiBot._build_partial_completion_report(steps)
        assert "`check_disk`" in result
        assert "`check_memory`" in result

    def test_missing_elapsed_ms_defaults_to_zero(self):
        steps = [
            {"tools": ["check_disk"], "status": "done", "reasoning": None},
        ]
        result = LokiBot._build_partial_completion_report(steps)
        assert "0.0s" in result

    def test_checkmark_in_report(self):
        steps = [
            {"tools": ["run_command"], "status": "done", "elapsed_ms": 500, "reasoning": None},
        ]
        result = LokiBot._build_partial_completion_report(steps)
        assert "\u2713" in result  # ✓ checkmark


# ---------------------------------------------------------------------------
# Integration tests: API error with partial completion
# ---------------------------------------------------------------------------

class TestPartialCompletionOnApiError:
    """Test that API errors mid-loop include partial completion report."""

    @pytest.mark.asyncio
    async def test_api_error_after_one_step_includes_report(self):
        """When chat_with_tools fails on iteration 2, report shows step 1."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Iteration 1: tool call succeeds
        # Iteration 2: API error
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Checking disk", tool_calls=[
                ToolCall(id="t1", name="check_disk", input={}),
            ]),
            ConnectionError("API unreachable"),
        ])

        result, _, is_error, tools_used, _ = await LokiBot._process_with_tools(
            stub, msg, [],
        )

        assert is_error is True
        assert "**Partial completion (1/1 steps):**" in result
        assert "`check_disk`" in result
        assert "LLM API error: API unreachable" in result

    @pytest.mark.asyncio
    async def test_api_error_on_first_call_no_report(self):
        """When chat_with_tools fails on first call, no partial report (nothing done)."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=ConnectionError("API unreachable"),
        )

        result, _, is_error, _, _ = await LokiBot._process_with_tools(
            stub, msg, [],
        )

        assert is_error is True
        assert "LLM API error: API unreachable" in result
        # No partial report since no steps were done
        assert "Partial completion" not in result

    @pytest.mark.asyncio
    async def test_api_error_embed_updated_to_red(self):
        """When API fails mid-loop, progress embed is updated to error (red)."""
        stub = _make_bot_stub()
        msg = _make_message()
        embed_msg = AsyncMock()
        msg.channel.send = AsyncMock(return_value=embed_msg)

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[
                ToolCall(id="t1", name="check_disk", input={}),
            ]),
            RuntimeError("Server error"),
        ])

        result, _, is_error, _, _ = await LokiBot._process_with_tools(
            stub, msg, [],
        )

        assert is_error is True
        # The embed should have been edited to error state
        assert embed_msg.edit.called
        last_edit = embed_msg.edit.call_args_list[-1]
        embed = last_edit[1]["embed"]
        from discord import Color
        assert embed.color == Color.red()


# ---------------------------------------------------------------------------
# Integration tests: max iterations with partial completion
# ---------------------------------------------------------------------------

class TestPartialCompletionOnMaxIterations:
    """Test that max iteration errors include partial completion report."""

    @pytest.mark.asyncio
    async def test_max_iterations_includes_report(self):
        """When tool loop hits MAX_TOOL_ITERATIONS, report shows completed steps."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Return tool calls for every iteration (never return text-only)
        stub.codex_client.chat_with_tools = AsyncMock(return_value=LLMResponse(
            text="", tool_calls=[ToolCall(id="t1", name="run_command", input={})],
        ))

        result, _, is_error, tools_used, _ = await LokiBot._process_with_tools(
            stub, msg, [],
        )

        assert is_error is True
        assert "Too many tool calls" in result
        # All MAX_TOOL_ITERATIONS steps completed
        assert f"**Partial completion ({MAX_TOOL_ITERATIONS}/{MAX_TOOL_ITERATIONS} steps):**" in result
        assert "`run_command`" in result

    @pytest.mark.asyncio
    async def test_max_iterations_with_mixed_tools(self):
        """Max iterations report shows different tool names per step."""
        stub = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def alternating_tools(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return LLMResponse(text="", tool_calls=[
                    ToolCall(id=f"t{call_count}", name="check_disk", input={}),
                ])
            else:
                return LLMResponse(text="", tool_calls=[
                    ToolCall(id=f"t{call_count}", name="run_command", input={}),
                ])

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=alternating_tools)

        result, _, is_error, _, _ = await LokiBot._process_with_tools(
            stub, msg, [],
        )

        assert is_error is True
        assert "`check_disk`" in result
        assert "`run_command`" in result


# ---------------------------------------------------------------------------
# Integration test: normal completion has no partial report
# ---------------------------------------------------------------------------

class TestNoReportOnSuccess:
    """Verify that successful completion does NOT include partial report text."""

    @pytest.mark.asyncio
    async def test_success_no_partial_report(self):
        """Normal tool loop completion returns clean text, no report."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Checking...", tool_calls=[
                ToolCall(id="t1", name="check_disk", input={}),
            ]),
            LLMResponse(text="All disks are healthy!", tool_calls=[]),
        ])

        result, _, is_error, _, _ = await LokiBot._process_with_tools(
            stub, msg, [],
        )

        assert is_error is False
        assert result == "All disks are healthy!"
        assert "Partial completion" not in result
