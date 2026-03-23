"""Tests for error response handling in conversation history.

Round 14 (checkpoint-save): error responses ARE saved to conversation history
so the user can reference what was done and continue with "keep going" or
"finish the task".  The partial completion report from Round 13 is included
in the error response, giving the LLM context about what was already
accomplished.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal LokiBot stub with the fields _process_with_tools
    and _handle_message_inner need."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._last_tool_use = {}
    stub._system_prompt = "You are a bot."
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.classifier.classify = AsyncMock(return_value="task")
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex chat response")
    stub.codex_client.chat_with_tools = AsyncMock(
        return_value=LLMResponse(text="Hello!", stop_reason="end_turn")
    )
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._pending_files = {}
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.tool_executor = MagicMock()
    stub.tool_executor.set_user_context = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="OK")
    return stub


def _make_message(channel_id="chan-1", author_id="user-1"):
    """Create a minimal discord.Message mock."""
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.reply = AsyncMock()
    return msg


# ---------------------------------------------------------------------------
# _process_with_tools return value tests
# ---------------------------------------------------------------------------

class TestProcessWithToolsErrorFlag:
    """_process_with_tools must return (text, already_sent, is_error, tools_used)."""

    async def test_codex_error_returns_is_error_true(self):
        """When codex_client.chat_with_tools raises, _process_with_tools catches it
        and returns an error message with is_error=True (Round 13: partial completion)."""
        stub = _make_bot_stub()
        stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed
        stub._build_partial_completion_report = LokiBot._build_partial_completion_report
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=RuntimeError("The AI service is temporarily overloaded.")
        )
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        # Round 13 changed _process_with_tools to catch API errors and return
        # them as error messages instead of re-raising.
        text, already_sent, is_error, tools_used, handoff = await stub._process_with_tools(msg, [])
        assert is_error is True
        assert "overloaded" in text
        assert "LLM API error" in text

    async def test_codex_success_returns_is_error_false(self):
        """A successful Codex response should return is_error=False."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Hello! How can I help?", stop_reason="end_turn")
        )
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        text, already_sent, is_error, _tools, _handoff = await stub._process_with_tools(msg, [])
        assert is_error is False
        assert text == "Hello! How can I help?"
        assert already_sent is False  # Codex path doesn't pre-stream to Discord

    async def test_max_iterations_returns_is_error_true(self):
        """When the tool loop exhausts MAX_TOOL_ITERATIONS, is_error=True."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.config.tools.enabled = True
        stub.config.tools.tool_timeout_seconds = 300
        stub._merged_tool_definitions = MagicMock(return_value=[{"name": "test"}])

        # Make every chat_with_tools return a tool call to exhaust the loop
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tool-1", name="test_tool", input={})],
                stop_reason="tool_use",
            )
        )

        # Mock tool execution
        stub.tool_executor.execute = AsyncMock(return_value="OK")

        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)
        stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, already_sent, is_error, _tools, _handoff = await stub._process_with_tools(msg, [])
        assert is_error is True
        assert "Too many tool calls" in text

    async def test_codex_error_propagates_to_handle_message(self):
        """When Codex raises in task route, _handle_message_inner catches it as is_error."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Task route: _process_with_tools raises → caught by inner except
        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("Rate limited"))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", str(msg.channel.id))

        # Error should be sent via _send_chunked
        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "tool execution failed" in sent_text.lower() or "rate limited" in sent_text.lower()


# ---------------------------------------------------------------------------
# History save behavior
# ---------------------------------------------------------------------------

class TestHistorySaveOnError:
    """Error responses are saved to history for checkpoint-save (Round 14)."""

    async def _run_handle_message(self, process_return):
        """Helper: runs _handle_message_inner with mocked _process_with_tools
        returning the given 3-tuple."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Force task routing via keyword match
        stub.classifier.classify = AsyncMock(return_value="task")

        stub._process_with_tools = AsyncMock(return_value=process_return)
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(
                msg, "check disk", str(msg.channel.id),
            )
        return stub

    async def test_normal_response_saved_to_history(self):
        """A normal (non-error) response should be saved to session history."""
        stub = await self._run_handle_message(
            ("Disk usage is 42%.", False, False, ["check_disk"], False)
        )
        calls = stub.sessions.add_message.call_args_list
        # First call is user message (line 756), second is assistant response
        assert len(calls) == 2
        assert calls[1][0] == ("chan-1", "assistant", "Disk usage is 42%.")

    async def test_error_response_saved_to_history(self):
        """An error response (is_error=True) should be saved to history
        as a sanitized marker (not raw error text) to prevent context poisoning."""
        stub = await self._run_handle_message(
            ("The AI service is temporarily overloaded.", False, True, [], False)
        )
        calls = stub.sessions.add_message.call_args_list
        # Both user message and sanitized error marker should be saved
        assert len(calls) == 2
        assert calls[0][0][1] == "user"
        # Error with no tools_used saves a sanitized marker
        assert calls[1][0] == ("chan-1", "assistant", "[Previous request encountered an error before tool execution.]")

    async def test_error_still_sent_to_user(self):
        """Even when is_error=True, the error message should still be shown to the user."""
        stub = await self._run_handle_message(
            ("Something went wrong processing that request.", False, True, [], False)
        )
        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "Something went wrong" in sent_text

    async def test_error_triggers_prune_and_save(self):
        """When is_error=True, prune() and save() should be called
        to persist the checkpoint (Round 14)."""
        stub = await self._run_handle_message(
            ("Rate limited", False, True, [], False)
        )
        stub.sessions.prune.assert_called()
        stub.sessions.save.assert_called()

    async def test_normal_response_triggers_prune_and_save(self):
        """When is_error=False, prune() and save() should be called."""
        stub = await self._run_handle_message(
            ("Here are the results.", False, False, ["check_disk"], False)
        )
        stub.sessions.prune.assert_called_once()
        stub.sessions.save.assert_called_once()

    async def test_already_sent_error_still_saved_to_history(self):
        """If error was already streamed (already_sent=True), it should still
        be saved to history as a sanitized marker for checkpoint-save (Round 14)."""
        stub = await self._run_handle_message(
            ("Partial error text", True, True, [], False)
        )
        calls = stub.sessions.add_message.call_args_list
        # Both user message and sanitized error marker
        assert len(calls) == 2
        assert calls[1][0] == ("chan-1", "assistant", "[Previous request encountered an error before tool execution.]")

    async def test_voice_callback_still_called_on_error(self):
        """Voice callback should still be called even on error responses,
        so the user hears the error message."""
        stub = _make_bot_stub()
        msg = _make_message()
        voice_cb = AsyncMock()

        stub.classifier.classify = AsyncMock(return_value="task")
        stub._process_with_tools = AsyncMock(
            return_value=("Service overloaded.", False, True, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(
                msg, "check server", str(msg.channel.id),
                voice_callback=voice_cb,
            )
        voice_cb.assert_called_once_with("Service overloaded.")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestErrorHistoryEdgeCases:
    """Edge cases for the error-in-history fix."""

    async def test_empty_error_text_saved_for_checkpoint(self):
        """Even empty/fallback error text is saved as a sanitized marker
        for checkpoint-save (Round 14)."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.classifier.classify = AsyncMock(return_value="task")
        stub._process_with_tools = AsyncMock(
            return_value=("(no response)", False, True, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(
                msg, "test", str(msg.channel.id),
            )
        calls = stub.sessions.add_message.call_args_list
        assert len(calls) == 2  # user message + sanitized error marker
        assert calls[0][0][1] == "user"
        assert calls[1][0] == (str(msg.channel.id), "assistant", "[Previous request encountered an error before tool execution.]")

    async def test_codex_fallback_error_saved_for_checkpoint(self):
        """When Codex chat fails, the error is saved for checkpoint-save (Round 14)."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(side_effect=Exception("Codex down"))
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "hey whats up", str(msg.channel.id),
            )
        calls = stub.sessions.add_message.call_args_list
        assert len(calls) == 2  # user message + error response
        assert calls[0][0][1] == "user"
        assert calls[1][0][1] == "assistant"
