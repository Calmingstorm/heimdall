"""Cross-feature integration tests — Round 18 full review.

These tests verify interactions between features added in different rounds:
- Circuit breaker recovery + cancel button (R15 + R17)
- Per-tool timeout + progress embed step status (R12 + R9)
- Error checkpoint-save continuity (R14 + R13)
- All 6 terminal paths of _process_with_tools (comprehensive)
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    HeimdallBot,
    MAX_TOOL_ITERATIONS,
    ToolLoopCancelView,
)
from src.llm.circuit_breaker import CircuitOpenError  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal HeimdallBot stub for integration tests."""
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
        {"name": "run_command", "description": "Run", "input_schema": {"type": "object", "properties": {}}},
        {"name": "check_disk", "description": "Disk", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._build_tool_progress_embed = HeimdallBot._build_tool_progress_embed
    stub._build_partial_completion_report = HeimdallBot._build_partial_completion_report
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


def _tool_call(name="run_command", id_="tc1"):
    return ToolCall(id=id_, name=name, input={"command": "test"})


def _tool_resp(text="", tool_calls=None):
    """Build an LLMResponse. is_tool_use is a property derived from tool_calls."""
    return LLMResponse(text=text, tool_calls=tool_calls or [])


# ---------------------------------------------------------------------------
# Cross-feature: Circuit Breaker + Cancel Button (R15 + R17)
# ---------------------------------------------------------------------------

class TestCircuitBreakerWithCancel:
    """Verify cancel button works during/after circuit breaker recovery."""

    async def test_cancel_during_circuit_breaker_sleep(self):
        """When user presses Cancel while circuit breaker is sleeping,
        the sleep completes, retry is attempted, and then cancel is
        detected at the NEXT iteration start."""
        stub = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def fake_chat_with_tools(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("I'll check it.", [_tool_call("check_disk", "tc1")])
            elif call_count == 2:
                raise CircuitOpenError("codex", 0.01)
            elif call_count == 3:
                # Retry after recovery — returns another tool call
                return _tool_resp("Continuing.", [_tool_call("check_disk", "tc2")])
            else:
                return _tool_resp("Done.")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat_with_tools)

        original_send = msg.channel.send

        async def intercept_send(**kwargs):
            result = await original_send(**kwargs)
            if "view" in kwargs:
                view = kwargs["view"]
                if isinstance(view, ToolLoopCancelView):
                    async def delayed_cancel():
                        await asyncio.sleep(0.005)
                        view._cancel_event.set()
                    asyncio.create_task(delayed_cancel())
            return result

        msg.channel.send = AsyncMock(side_effect=intercept_send)

        text, _, is_error, tools_used, _ = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert is_error is True
        assert "cancelled" in text.lower()
        assert "check_disk" in " ".join(tools_used)

    async def test_circuit_breaker_recovery_then_normal_completion(self):
        """Circuit breaker opens, recovery succeeds, loop completes normally."""
        stub = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("Step 1", [_tool_call("check_disk", "tc1")])
            elif call_count == 2:
                raise CircuitOpenError("codex", 0.01)
            elif call_count == 3:
                return _tool_resp("Recovered", [_tool_call("check_disk", "tc2")])
            else:
                return _tool_resp("Done")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)

        text, _, is_error, tools_used, _ = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert is_error is False
        assert text == "Done"
        assert tools_used == ["check_disk", "check_disk"]


# ---------------------------------------------------------------------------
# Cross-feature: Per-Tool Timeout + Progress Embed (R12 + R9)
# ---------------------------------------------------------------------------

class TestTimeoutWithProgressEmbed:
    """Verify that tool timeouts produce correct progress embed updates."""

    async def test_timeout_step_still_marked_done(self):
        """When a tool times out, the step is still marked 'done' in progress
        (the timeout error is returned as a tool result, not an exception)."""
        stub = _make_bot_stub()
        stub.config.tools.tool_timeout_seconds = 0.01
        msg = _make_message()

        async def slow_execute(*a, **kw):
            await asyncio.sleep(10)
            return "should not reach"

        stub.tool_executor.execute = AsyncMock(side_effect=slow_execute)

        call_count = 0

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("Checking.", [_tool_call("check_disk", "tc1")])
            else:
                return _tool_resp("The check timed out.")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)

        text, _, is_error, tools_used, _ = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert is_error is False
        assert "timed out" in text.lower()
        assert tools_used == ["check_disk"]

        # Verify embed was updated
        embed_msg = msg.channel.send.return_value
        assert embed_msg.edit.called


# ---------------------------------------------------------------------------
# Cross-feature: Error → Checkpoint → Continuity (R14 + R13)
# ---------------------------------------------------------------------------

class TestCheckpointContinuity:
    """Verify that error responses saved to history enable continuation."""

    async def test_error_response_includes_partial_report(self):
        """When tool loop fails after 2 steps, the error includes
        the partial completion report."""
        stub = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("Step 1", [_tool_call("check_disk", "tc1")])
            elif call_count == 2:
                return _tool_resp("Step 2", [_tool_call("run_command", "tc2")])
            else:
                raise RuntimeError("API crashed")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)

        text, _, is_error, _, _ = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert is_error is True
        assert "Partial completion" in text
        assert "check_disk" in text
        assert "run_command" in text
        assert "API crashed" in text


# ---------------------------------------------------------------------------
# All 6 Terminal Paths (Comprehensive)
# ---------------------------------------------------------------------------

class TestAllTerminalPaths:
    """Verify all 6 terminal paths of _process_with_tools return correctly."""

    async def test_path_1_text_only_response(self):
        """LLM returns text without tool calls → normal completion."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_resp("Hello!"),
        )

        text, already_sent, is_error, tools_used, handoff = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert text == "Hello!"
        assert already_sent is False
        assert is_error is False
        assert tools_used == []
        assert handoff is False

    async def test_path_2_tool_then_text(self):
        """LLM calls tool, then returns text → normal completion with green embed."""
        stub = _make_bot_stub()
        msg = _make_message()
        call_count = 0

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("Checking", [_tool_call("check_disk", "tc1")])
            return _tool_resp("All good.")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)

        text, _, is_error, tools_used, handoff = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert text == "All good."
        assert is_error is False
        assert tools_used == ["check_disk"]
        assert handoff is False

    async def test_path_3_api_error(self):
        """LLM API raises exception → error."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=RuntimeError("server down"),
        )

        text, _, is_error, _, handoff = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert is_error is True
        assert "server down" in text
        assert handoff is False

    async def test_path_4_circuit_breaker_recovery_failure(self):
        """Circuit breaker opens, retry also fails → error."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=CircuitOpenError("codex", 0.01),
        )

        text, _, is_error, _, handoff = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert is_error is True
        assert "circuit breaker" in text.lower()
        assert handoff is False

    async def test_path_5_cancelled_by_user(self):
        """User presses cancel button → error with partial report."""
        stub = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("Step 1", [_tool_call("check_disk", "tc1")])
            return _tool_resp("Done.")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)

        original_send = msg.channel.send

        async def intercept_send(**kwargs):
            result = await original_send(**kwargs)
            if "view" in kwargs:
                view = kwargs["view"]
                if isinstance(view, ToolLoopCancelView):
                    view._cancel_event.set()
            return result

        msg.channel.send = AsyncMock(side_effect=intercept_send)

        text, _, is_error, _, handoff = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert is_error is True
        assert "cancelled" in text.lower()
        assert handoff is False

    async def test_path_6_max_iterations(self):
        """LLM keeps calling tools until MAX_TOOL_ITERATIONS → error."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_resp("Another step", [_tool_call("check_disk", "tc1")]),
        )

        text, _, is_error, tools_used, handoff = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert is_error is True
        assert "too many tool calls" in text.lower()
        assert len(tools_used) == MAX_TOOL_ITERATIONS
        assert handoff is False

    async def test_path_skill_handoff(self):
        """All tools are skills that want Codex handoff → handoff=True."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.skill_manager.has_skill = MagicMock(return_value=True)
        stub.skill_manager.execute = AsyncMock(return_value="skill result")
        stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=True)

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_resp("Using skill", [_tool_call("my_skill", "tc1")]),
        )

        text, _, is_error, tools_used, handoff = await HeimdallBot._process_with_tools(
            stub, msg, [], system_prompt_override="test",
        )

        assert is_error is False
        assert handoff is True
        assert tools_used == ["my_skill"]
