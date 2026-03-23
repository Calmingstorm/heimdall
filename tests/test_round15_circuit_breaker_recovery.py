"""Tests for circuit breaker recovery in the tool loop.

Round 15: When the circuit breaker opens mid-task, instead of failing
immediately, wait for the recovery timeout and retry once. If the retry
succeeds, the tool loop continues normally. If it fails, the loop reports
partial completion and exits with an error.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot, MAX_TOOL_ITERATIONS  # noqa: E402
from src.llm.circuit_breaker import CircuitOpenError  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal LokiBot stub for circuit breaker recovery tests."""
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
# Unit tests: _build_tool_progress_embed footer
# ---------------------------------------------------------------------------

class TestProgressEmbedFooter:
    """Test the footer parameter in _build_tool_progress_embed."""

    def test_footer_appears_in_embed(self):
        steps = [
            {"tools": ["check_disk"], "status": "done", "elapsed_ms": 1200, "reasoning": None},
        ]
        embed = LokiBot._build_tool_progress_embed(steps, "running", footer="Waiting for recovery...")
        assert "Waiting for recovery..." in embed.description

    def test_no_footer_by_default(self):
        steps = [
            {"tools": ["check_disk"], "status": "done", "elapsed_ms": 1200, "reasoning": None},
        ]
        embed = LokiBot._build_tool_progress_embed(steps, "running")
        assert "Waiting" not in embed.description

    def test_footer_with_none(self):
        steps = [
            {"tools": ["check_disk"], "status": "done", "elapsed_ms": 1200, "reasoning": None},
        ]
        embed = LokiBot._build_tool_progress_embed(steps, "running", footer=None)
        # Should be same as no footer
        assert "recovery" not in embed.description.lower()


# ---------------------------------------------------------------------------
# Integration tests: circuit breaker recovery in tool loop
# ---------------------------------------------------------------------------

class TestCircuitBreakerRecoverySuccess:
    """Test that the tool loop recovers when circuit breaker opens then clears."""

    @pytest.mark.asyncio
    async def test_circuit_open_then_recovery_succeeds(self):
        """Circuit breaker opens, waits, retries — retry succeeds, loop continues."""
        stub = _make_bot_stub()
        msg = _make_message()

        # First call: raises CircuitOpenError
        # Second call (retry after wait): succeeds with text-only response
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                CircuitOpenError("codex_api", retry_after=0.01),  # tiny wait for test speed
                LLMResponse(text="All done!", tool_calls=[], stop_reason="end_turn"),
            ]
        )

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check disk"}]
            )

        assert not is_error
        assert text == "All done!"
        # Should have slept for the retry_after duration (capped)
        mock_sleep.assert_awaited_once()
        wait_arg = mock_sleep.call_args[0][0]
        assert wait_arg == pytest.approx(0.01, abs=0.001)

    @pytest.mark.asyncio
    async def test_circuit_open_recovery_continues_tool_loop(self):
        """After recovery, the tool loop continues with tool calls normally."""
        stub = _make_bot_stub()
        msg = _make_message()

        # First call: circuit breaker opens
        # Second call (retry): tool call
        # Third call: text response (end)
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                CircuitOpenError("codex_api", retry_after=0.01),
                LLMResponse(
                    text="Checking disk now",
                    tool_calls=[ToolCall(id="tc1", name="check_disk", input={})],
                    stop_reason="tool_use",
                ),
                LLMResponse(text="Disk is fine!", tool_calls=[], stop_reason="end_turn"),
            ]
        )

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock):
            text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check disk"}]
            )

        assert not is_error
        assert text == "Disk is fine!"
        assert "check_disk" in tools_used


class TestCircuitBreakerRecoveryFailure:
    """Test that the tool loop reports partial completion when recovery fails."""

    @pytest.mark.asyncio
    async def test_circuit_open_retry_also_fails(self):
        """Circuit breaker opens, waits, retries — retry also fails."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                CircuitOpenError("codex_api", retry_after=0.01),
                RuntimeError("Still down"),
            ]
        )

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock):
            text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check disk"}]
            )

        assert is_error
        assert "circuit breaker recovery failed" in text
        assert "Still down" in text

    @pytest.mark.asyncio
    async def test_circuit_open_after_completed_steps_includes_report(self):
        """Circuit breaker opens mid-task — partial report shows completed steps."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Step 1: tool call succeeds
        # Step 2 (LLM call): circuit breaker opens
        # Retry: also fails
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                LLMResponse(
                    text="Running check",
                    tool_calls=[ToolCall(id="tc1", name="check_disk", input={})],
                    stop_reason="tool_use",
                ),
                CircuitOpenError("codex_api", retry_after=0.01),
                RuntimeError("API still down"),
            ]
        )

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock):
            text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check all disks"}]
            )

        assert is_error
        assert "circuit breaker recovery failed" in text
        # Partial completion report should show the first completed step
        assert "Partial completion" in text
        assert "`check_disk`" in text

    @pytest.mark.asyncio
    async def test_circuit_open_retry_circuit_open_again(self):
        """Circuit breaker opens, waits, retries — breaker opens again on retry."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                CircuitOpenError("codex_api", retry_after=0.01),
                CircuitOpenError("codex_api", retry_after=30.0),  # still open
            ]
        )

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock):
            text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check disk"}]
            )

        assert is_error
        assert "circuit breaker recovery failed" in text


class TestCircuitBreakerWaitCap:
    """Test that the wait time is capped at 90 seconds."""

    @pytest.mark.asyncio
    async def test_retry_after_capped_at_90(self):
        """retry_after > 90s is capped to 90s to avoid very long waits."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                CircuitOpenError("codex_api", retry_after=300.0),  # 5 minutes
                LLMResponse(text="Recovered!", tool_calls=[], stop_reason="end_turn"),
            ]
        )

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check disk"}]
            )

        assert not is_error
        wait_arg = mock_sleep.call_args[0][0]
        assert wait_arg == 90.0  # capped, not 300


class TestCircuitBreakerEmbedUpdates:
    """Test that the progress embed is updated during circuit breaker recovery."""

    @pytest.mark.asyncio
    async def test_embed_shows_recovery_message(self):
        """Progress embed is updated with recovery wait message."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Step 1: tool call (embed is created)
        # Step 2: circuit breaker opens (embed should show recovery)
        # Retry: succeeds
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                LLMResponse(
                    text="Checking",
                    tool_calls=[ToolCall(id="tc1", name="check_disk", input={})],
                    stop_reason="tool_use",
                ),
                CircuitOpenError("codex_api", retry_after=0.01),
                LLMResponse(text="Done!", tool_calls=[], stop_reason="end_turn"),
            ]
        )

        embed_msg = AsyncMock()
        msg.channel.send = AsyncMock(return_value=embed_msg)

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock):
            text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check disk"}]
            )

        assert not is_error
        assert text == "Done!"
        # Check that embed was edited with recovery message
        edit_calls = embed_msg.edit.call_args_list
        # At least one edit should contain the recovery footer
        recovery_edits = [
            c for c in edit_calls
            if "recovering" in str(c.kwargs.get("embed", {}).description).lower()
            or "retrying" in str(c.kwargs.get("embed", {}).description).lower()
        ]
        assert len(recovery_edits) >= 1

    @pytest.mark.asyncio
    async def test_embed_turns_red_on_recovery_failure(self):
        """Progress embed turns red when recovery fails."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Step 1: tool call (embed is created)
        # Step 2: circuit breaker opens
        # Retry: fails
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                LLMResponse(
                    text="Checking",
                    tool_calls=[ToolCall(id="tc1", name="check_disk", input={})],
                    stop_reason="tool_use",
                ),
                CircuitOpenError("codex_api", retry_after=0.01),
                RuntimeError("Still down"),
            ]
        )

        embed_msg = AsyncMock()
        msg.channel.send = AsyncMock(return_value=embed_msg)

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock):
            text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check disk"}]
            )

        assert is_error
        # Last embed edit should be error/red
        last_edit = embed_msg.edit.call_args_list[-1]
        last_embed = last_edit.kwargs.get("embed")
        if last_embed:
            import discord
            assert last_embed.color == discord.Color.red()


class TestCircuitBreakerNoRegression:
    """Ensure regular API errors still work as before (no regression)."""

    @pytest.mark.asyncio
    async def test_regular_exception_not_affected(self):
        """Non-CircuitOpenError exceptions still handled by generic except."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=RuntimeError("Connection refused")
        )

        text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "check disk"}]
        )

        assert is_error
        assert "LLM API error:" in text
        assert "Connection refused" in text
        # Should NOT contain "circuit breaker" — this is a regular error
        assert "circuit breaker" not in text
