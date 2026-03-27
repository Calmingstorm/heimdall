"""Tests for surfacing LLM reasoning text in tool loop progress embeds.

Round 8: When Codex returns text alongside tool calls (e.g., "I'll check the
disk first"), the reasoning text is now included in the progress embed sent
to Discord.

Round 9 updated these tests: progress is now shown via a single editable
Discord embed instead of scattered text messages.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal HeimdallBot stub for reasoning text tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test prompt"
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
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
    stub._build_tool_progress_embed = HeimdallBot._build_tool_progress_embed
    return stub


def _make_message():
    """Create a mock Discord message with embed support."""
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = "test-chan"
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None),
    ))
    # channel.send returns a mock message that supports .edit()
    embed_msg = AsyncMock()
    msg.channel.send = AsyncMock(return_value=embed_msg)
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    msg.attachments = []
    return msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReasoningTextInProgress:
    """LLM reasoning text should appear in the progress embed."""

    async def test_reasoning_text_shown_in_embed(self):
        """When LLM returns text + tool call, text appears in embed description."""
        stub = _make_bot_stub()
        msg = _make_message()

        tool_response = LLMResponse(
            text="I'll check the disk usage first.",
            tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})],
            stop_reason="tool_use",
        )
        final_response = LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        # The embed should include reasoning text
        msg.channel.send.assert_called()
        embed = msg.channel.send.call_args[1]["embed"]
        assert "I'll check the disk usage first." in embed.description
        assert "check_disk" in embed.description

    async def test_no_reasoning_when_text_empty(self):
        """When LLM returns no text with tool call, no reasoning in embed."""
        stub = _make_bot_stub()
        msg = _make_message()

        tool_response = LLMResponse(
            text="",
            tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})],
            stop_reason="tool_use",
        )
        final_response = LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        embed = msg.channel.send.call_args[1]["embed"]
        # Should have tool name but no italic reasoning text
        assert "check_disk" in embed.description
        assert "*" not in embed.description  # no italic reasoning

    async def test_long_reasoning_truncated(self):
        """Reasoning text longer than 200 chars is truncated with '...'."""
        stub = _make_bot_stub()
        msg = _make_message()

        long_text = "A" * 250  # 250 chars, over the 200 limit
        tool_response = LLMResponse(
            text=long_text,
            tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})],
            stop_reason="tool_use",
        )
        final_response = LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        embed = msg.channel.send.call_args[1]["embed"]
        # Should be truncated: 200 chars + "..."
        assert "A" * 200 + "..." in embed.description
        # Should NOT contain the full 250-char string
        assert "A" * 250 not in embed.description

    async def test_reasoning_exactly_200_not_truncated(self):
        """Reasoning text exactly 200 chars is NOT truncated."""
        stub = _make_bot_stub()
        msg = _make_message()

        exact_text = "B" * 200
        tool_response = LLMResponse(
            text=exact_text,
            tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})],
            stop_reason="tool_use",
        )
        final_response = LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        embed = msg.channel.send.call_args[1]["embed"]
        # Full text included, no truncation suffix
        assert exact_text in embed.description
        assert exact_text + "..." not in embed.description

    async def test_reasoning_with_multiple_tools(self):
        """Reasoning text works with multiple tool calls in one iteration."""
        stub = _make_bot_stub()
        msg = _make_message()

        tool_response = LLMResponse(
            text="Let me check both hosts.",
            tool_calls=[
                ToolCall(id="tc-1", name="check_disk", input={"host": "server"}),
                ToolCall(id="tc-2", name="check_disk", input={"host": "desktop"}),
            ],
            stop_reason="tool_use",
        )
        final_response = LLMResponse(text="Both fine.", tool_calls=[], stop_reason="end_turn")
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        embed = msg.channel.send.call_args[1]["embed"]
        assert "Let me check both hosts." in embed.description
        assert "`check_disk`" in embed.description

    async def test_reasoning_per_iteration(self):
        """Each iteration's embed update includes the latest reasoning."""
        stub = _make_bot_stub()
        msg = _make_message()

        resp1 = LLMResponse(
            text="First, checking disk.",
            tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})],
            stop_reason="tool_use",
        )
        resp2 = LLMResponse(
            text="Now restarting the service.",
            tool_calls=[ToolCall(id="tc-2", name="check_disk", input={})],
            stop_reason="tool_use",
        )
        final = LLMResponse(text="All done.", tool_calls=[], stop_reason="end_turn")
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[resp1, resp2, final]
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        # First embed (channel.send) has first reasoning
        first_embed = msg.channel.send.call_args[1]["embed"]
        assert "First, checking disk." in first_embed.description

        # Subsequent updates go through embed_msg.edit — the second step's
        # reasoning should appear in one of the edit calls
        embed_msg = msg.channel.send.return_value
        edit_calls = embed_msg.edit.call_args_list
        # Find the edit call that shows the second running step
        second_reasoning_found = False
        for call in edit_calls:
            embed = call[1]["embed"]
            if "Now restarting the service." in embed.description:
                second_reasoning_found = True
                break
        assert second_reasoning_found, "Second step reasoning not found in edit calls"

    async def test_reasoning_in_italic_format(self):
        """Reasoning text is shown in italic (*text*) in the embed."""
        stub = _make_bot_stub()
        msg = _make_message()

        tool_response = LLMResponse(
            text="Checking things.",
            tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})],
            stop_reason="tool_use",
        )
        final_response = LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        embed = msg.channel.send.call_args[1]["embed"]
        assert "*Checking things.*" in embed.description
