"""Tests for Codex tool calling integration in the bot tool loop.

Verifies that:
1. _process_with_tools calls codex_client.chat_with_tools()
2. Codex tool loop iteration works (tool calls → execution → next iteration)
3. Task route uses _process_with_tools when Codex is available
4. When codex_client is None, task route returns "No tool backend" error
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal HeimdallBot stub."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test system prompt"
    stub._channel_locks = {}
    stub._processed_messages = MagicMock()
    stub._processed_messages_max = 100
    stub._background_tasks = {}
    stub._background_tasks_max = 20
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["12345"]
    stub.config.discord.channels = ["67890"]
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.config.monitoring.alert_channel_id = "67890"
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.reset = MagicMock()
    stub.sessions.search_history = AsyncMock(return_value=[])
    stub.sessions.get_or_create = MagicMock()
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex chat response")
    stub.codex_client.chat_with_tools = AsyncMock(
        return_value=LLMResponse(text="Codex tool response", tool_calls=[], stop_reason="end_turn")
    )
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub.tool_executor.set_user_context = MagicMock()
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._process_with_tools = AsyncMock(
        return_value=("response", False, False, [], False)
    )
    stub._merged_tool_definitions = MagicMock(return_value=[
        {"name": "check_disk", "description": "Check disk usage", "input_schema": {"type": "object", "properties": {"host": {"type": "string"}}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.voice_manager = None
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._pending_files = {}
    stub._track_recent_action = MagicMock()
    return stub


def _make_message(channel_id="chan-1", author_id="12345"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    return msg


# ---------------------------------------------------------------------------
# _process_with_tools uses codex_client.chat_with_tools
# ---------------------------------------------------------------------------

class TestProcessWithToolsCodex:
    """Tests for _process_with_tools — always uses codex_client.chat_with_tools."""

    async def test_codex_text_response(self):
        """Codex returns text-only response — no tools used."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="The disk is 42% full.", tool_calls=[], stop_reason="end_turn")
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        text, already_sent, is_error, tools_used, handoff = await stub._process_with_tools(
            msg, [], system_prompt_override="test prompt",
        )

        assert text == "The disk is 42% full."
        assert already_sent is False
        assert is_error is False
        assert tools_used == []
        assert handoff is False
        stub.codex_client.chat_with_tools.assert_called_once()

    async def test_codex_tool_call_execution(self):
        """Codex returns a tool call, tool is executed, then text response."""
        stub = _make_bot_stub()
        msg = _make_message()

        # First call: returns a tool call
        # Second call: returns text response
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="Let me check.",
                tool_calls=[ToolCall(id="call_1", name="check_disk", input={"host": "server"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="The disk is 42% full on server.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub.tool_executor.execute = AsyncMock(return_value="Filesystem  Size  Used\n/  50G  21G")
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, already_sent, is_error, tools_used, handoff = await stub._process_with_tools(
                msg, [], system_prompt_override="test prompt",
            )

        assert text == "The disk is 42% full on server."
        assert tools_used == ["check_disk"]
        assert stub.codex_client.chat_with_tools.call_count == 2
        stub.tool_executor.execute.assert_called_once_with("check_disk", {"host": "server"}, user_id="12345")

    async def test_codex_multiple_tool_calls(self):
        """Codex returns multiple tool calls in one response."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="call_1", name="check_disk", input={"host": "server"}),
                    ToolCall(id="call_2", name="check_disk", input={"host": "desktop"}),
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Both disks are fine.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub.tool_executor.execute = AsyncMock(return_value="ok")
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [], system_prompt_override="test",
            )

        assert tools_used == ["check_disk", "check_disk"]
        assert stub.tool_executor.execute.call_count == 2

    async def test_codex_message_history_format(self):
        """After tool execution, messages should be in internal format for re-conversion."""
        stub = _make_bot_stub()
        msg = _make_message()

        captured_messages = []

        async def capture_chat_with_tools(messages, system, tools):
            captured_messages.append(list(messages))
            if len(captured_messages) == 1:
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="call_1", name="check_disk", input={"host": "server"})],
                    stop_reason="tool_use",
                )
            return LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=capture_chat_with_tools)
        stub.tool_executor.execute = AsyncMock(return_value="disk ok")
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [], system_prompt_override="test")

        # Second call should have tool_use and tool_result in messages
        second_call_msgs = captured_messages[1]
        # Should have assistant message with tool_use and user message with tool_result
        assistant_msg = second_call_msgs[-2]
        assert assistant_msg["role"] == "assistant"
        assert any(b["type"] == "tool_use" for b in assistant_msg["content"])

        user_msg = second_call_msgs[-1]
        assert user_msg["role"] == "user"
        assert any(b["type"] == "tool_result" for b in user_msg["content"])

    async def test_codex_no_tools_disabled(self):
        """When tools are disabled, _process_with_tools should still work (empty tools list)."""
        stub = _make_bot_stub()
        stub.config.tools.enabled = False
        stub.config.tools.tool_timeout_seconds = 300
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Just chatting.", tool_calls=[], stop_reason="end_turn")
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        text, _, _, _, _ = await stub._process_with_tools(
            msg, [], system_prompt_override="test",
        )
        assert text == "Just chatting."


# ---------------------------------------------------------------------------
# Task route — Codex with tools
# ---------------------------------------------------------------------------

class TestTaskRouteCodex:
    """Tests for _handle_message_inner task route using Codex."""

    async def test_task_uses_codex_with_tools(self):
        """Task route should call _process_with_tools with system_prompt_override."""
        stub = _make_bot_stub()
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk on server", "chan-1")

        stub._process_with_tools.assert_called_once()
        call_kwargs = stub._process_with_tools.call_args[1]
        # system_prompt_override is passed; use_codex is no longer a parameter
        assert "system_prompt_override" in call_kwargs

    async def test_task_keyword_uses_codex(self):
        """Keyword-matched tasks should also call _process_with_tools."""
        stub = _make_bot_stub()
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._process_with_tools.assert_called_once()
        call_kwargs = stub._process_with_tools.call_args[1]
        assert "system_prompt_override" in call_kwargs

    async def test_task_process_with_tools_exception_sends_error(self):
        """When _process_with_tools raises, task route catches and sends error."""
        stub = _make_bot_stub()
        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("Codex API down"))
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Error is caught, response is sent via _send_chunked (not _send_with_retry)
        stub._send_chunked.assert_called_once()

    async def test_task_no_codex_returns_error(self):
        """When codex_client is None, task route returns 'No tool backend' error."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._process_with_tools.assert_not_called()
        stub._send_with_retry.assert_called_once()
        call_text = stub._send_with_retry.call_args[0][1]
        assert "no tool backend" in call_text.lower()


# ---------------------------------------------------------------------------
# No budget concept — Codex is always free (subscription)
# ---------------------------------------------------------------------------

class TestBudgetWithCodex:
    """Tests confirming the task route uses Codex regardless of any budget state."""

    async def test_task_with_codex_always_works(self):
        """Task route always calls _process_with_tools when codex_client is set."""
        stub = _make_bot_stub()
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Should call _process_with_tools — no budget gate
        stub._process_with_tools.assert_called_once()

    async def test_task_no_codex_sends_no_backend_message(self):
        """Task without codex_client sends 'No tool backend' instead of blocking on budget."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._process_with_tools.assert_not_called()
        stub._send_with_retry.assert_called_once()
        call_text = stub._send_with_retry.call_args[0][1]
        assert "no tool backend" in call_text.lower()
