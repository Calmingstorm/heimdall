"""Tests for typing indicator during Codex tool loop LLM calls.

Round 7: The tool loop in _process_with_tools now shows "Bot is typing..."
while waiting for the LLM (chat_with_tools) to respond. This eliminates the
"dead zone" where nothing appears to happen during LLM thinking time.

The typing indicator is shown ONLY during chat_with_tools calls, NOT during
tool execution (which already has its own "Running: tool_name..." messages).
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal HeimdallBot stub for typing indicator tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test prompt"
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.max_tool_iterations_chat = 30
    stub.config.tools.max_tool_iterations_loop = 100
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
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
        {"name": "check_disk", "description": "Check disk", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._classify_completion = AsyncMock(return_value=(True, ""))
    return stub


def _make_message():
    """Create a mock Discord message with proper typing() setup."""
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = "test-chan"
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None),
    ))
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    msg.attachments = []
    return msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTypingDuringLLMCall:
    """Typing indicator should be active during chat_with_tools calls."""

    async def test_typing_called_on_text_response(self):
        """Typing should be shown even for simple text-only LLM responses."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Disk is 42% full.", tool_calls=[], stop_reason="end_turn")
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "Disk is 42% full."
        # typing() was called once (one LLM call, no tool loop)
        msg.channel.typing.assert_called_once()

    async def test_typing_called_each_iteration(self):
        """Typing should be shown on EACH iteration of the tool loop."""
        stub = _make_bot_stub()
        msg = _make_message()

        # First call: LLM returns a tool call
        # Second call: LLM returns text (loop ends)
        tool_response = LLMResponse(
            text="I'll check the disk.",
            tool_calls=[ToolCall(id="tc-1", name="check_disk", input={"host": "server"})],
            stop_reason="tool_use",
        )
        text_response = LLMResponse(
            text="Disk is fine.", tool_calls=[], stop_reason="end_turn",
        )
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[tool_response, text_response]
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "Disk is fine."
        assert tools_used == ["check_disk"]
        # typing() called 3 times: LLM call iter 1, tool execution, LLM call iter 2
        assert msg.channel.typing.call_count == 3

    async def test_typing_called_three_iterations(self):
        """For a 3-iteration loop, typing should be called 3 times."""
        stub = _make_bot_stub()
        msg = _make_message()

        tool_call_1 = LLMResponse(
            text="Step 1", stop_reason="tool_use",
            tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})],
        )
        tool_call_2 = LLMResponse(
            text="Step 2", stop_reason="tool_use",
            tool_calls=[ToolCall(id="tc-2", name="check_disk", input={})],
        )
        final = LLMResponse(text="All done.", tool_calls=[], stop_reason="end_turn")
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[tool_call_1, tool_call_2, final]
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "All done."
        assert len(tools_used) == 2
        # typing() called 5 times: LLM call + tool exec for iters 1-2, LLM call for iter 3
        assert msg.channel.typing.call_count == 5

    async def test_typing_context_manager_entered_and_exited(self):
        """The typing context manager should be properly entered and exited."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        # The context manager was entered and exited
        typing_cm = msg.channel.typing.return_value
        typing_cm.__aenter__.assert_called_once()
        typing_cm.__aexit__.assert_called_once()

    async def test_typing_active_during_tool_execution(self):
        """Typing should be active during both LLM calls and tool execution.

        The tool loop structure is:
        1. async with typing: chat_with_tools()  ← typing active
        2. async with typing: execute tools       ← typing active
        3. back to step 1 for next iteration
        """
        stub = _make_bot_stub()
        msg = _make_message()

        # Track the order of operations
        call_order = []

        # typing enter/exit tracking
        typing_cm = msg.channel.typing.return_value
        typing_cm.__aenter__ = AsyncMock(
            side_effect=lambda: call_order.append("typing_enter")
        )
        typing_cm.__aexit__ = AsyncMock(
            side_effect=lambda *a: call_order.append("typing_exit")
        )

        # chat_with_tools tracking
        tool_response = LLMResponse(
            text="Checking disk.",
            tool_calls=[ToolCall(id="tc-1", name="run_command", input={})],
            stop_reason="tool_use",
        )
        final_response = LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")

        async def mock_chat_with_tools(**kwargs):
            call_order.append("llm_call")
            if len([c for c in call_order if c == "llm_call"]) == 1:
                return tool_response
            return final_response

        stub.codex_client.chat_with_tools = mock_chat_with_tools

        # tool execution tracking
        original_execute = stub.tool_executor.execute
        async def mock_execute(*args, **kwargs):
            call_order.append("tool_execute")
            return "OK"
        stub.tool_executor.execute = mock_execute

        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        # Verify: typing wraps both LLM call and tool execution
        assert call_order == [
            "typing_enter",   # 1. typing starts for LLM call
            "llm_call",       # 2. LLM thinks (returns tool call)
            "typing_exit",    # 3. typing ends after LLM
            "typing_enter",   # 4. typing starts for tool execution
            "tool_execute",   # 5. tool executes (typing active)
            "typing_exit",    # 6. typing ends after tools
            "typing_enter",   # 7. typing starts for next LLM call
            "llm_call",       # 8. LLM thinks (returns final text)
            "typing_exit",    # 9. typing ends
        ]

    async def test_typing_works_with_no_tools_configured(self):
        """Typing should still work when tools are disabled."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.config.tools.enabled = False
        stub.config.tools.tool_timeout_seconds = 300
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="No tools.", tool_calls=[], stop_reason="end_turn")
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "No tools."
        msg.channel.typing.assert_called_once()

