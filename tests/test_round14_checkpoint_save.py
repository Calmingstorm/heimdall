"""Tests for Round 14: Checkpoint-Save on Failure.

When the tool loop fails (API error, max iterations, exception), the error
response (including partial completion report from Round 13) is saved to
session history.  This lets the user reference what was done and continue
with "keep going" or "finish the task".

Previously, errors triggered remove_last_message (user message deleted,
no assistant message saved).  Now both the user message and the error
response are preserved in history.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal LokiBot stub."""
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
    stub.config.tools.approval_timeout_seconds = 30
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
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
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    return stub


def _make_message(channel_id="chan-1", author_id="user-1"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock(return_value=AsyncMock())
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.reply = AsyncMock()
    return msg


# ---------------------------------------------------------------------------
# Core checkpoint-save tests
# ---------------------------------------------------------------------------

class TestCheckpointSaveOnToolLoopError:
    """When _process_with_tools returns is_error=True, the error response
    (with partial completion report) is saved to session history."""

    async def test_error_response_saved_as_assistant_message(self):
        """Error response becomes a sanitized assistant message in history."""
        stub = _make_bot_stub()
        msg = _make_message()

        error_text = (
            "**Partial completion (2/3 steps):**\n"
            "\u2713 Step 1: `check_disk` (1.2s)\n"
            "\u2713 Step 2: `run_command` (2.5s)\n\n"
            "LLM API error: API unreachable"
        )
        stub._process_with_tools = AsyncMock(
            return_value=(error_text, False, True, ["check_disk", "run_command"], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check all hosts", "chan-1")

        # Error should be saved as sanitized marker (tools were used)
        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1
        saved_text = assistant_saves[0][0][2]
        # Sanitized marker includes tools used and "error"
        assert "check_disk" in saved_text
        assert "run_command" in saved_text
        assert "error" in saved_text.lower()

    async def test_user_message_preserved(self):
        """User's original message is NOT removed from history."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("LLM API error: timeout", False, True, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "deploy app", "chan-1")

        # remove_last_message should NOT be called
        stub.sessions.remove_last_message.assert_not_called()

    async def test_session_persisted_to_disk(self):
        """Error checkpoint triggers prune() and save() to persist state."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("Error occurred", False, True, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub.sessions.prune.assert_called()
        stub.sessions.save.assert_called()

    async def test_error_still_sent_to_discord(self):
        """Error response is still sent to the user on Discord."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("LLM API error: rate limited", False, True, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "rate limited" in sent_text


class TestCheckpointSaveOnException:
    """When _process_with_tools raises an exception, the error is still
    saved to history via the inner except handler."""

    async def test_exception_error_saved(self):
        """Exception caught by inner except saves sanitized error to history."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            side_effect=RuntimeError("Connection reset")
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "deploy app", "chan-1")

        # Error saved as sanitized marker
        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1
        # Sanitized: no tools used -> "[Previous request encountered an error before tool execution.]"
        assert "error" in assistant_saves[0][0][2].lower()
        # User message preserved
        stub.sessions.remove_last_message.assert_not_called()


class TestCheckpointSaveOnChatError:
    """Chat path errors also save to history for continuity."""

    async def test_chat_error_saved(self):
        """When Codex chat fails, error is saved as sanitized marker."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client.chat = AsyncMock(side_effect=Exception("API down"))
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1")

        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1
        # Chat errors are now saved as sanitized markers
        assert "error" in assistant_saves[0][0][2].lower()


class TestCheckpointSaveSuccessUnchanged:
    """Verify that successful responses are handled identically to before."""

    async def test_success_saves_normally(self):
        """Successful tool loop response saved as assistant message."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("Disk is 42% full.", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1
        assert "42%" in assistant_saves[0][0][2]
        stub.sessions.remove_last_message.assert_not_called()

    async def test_success_records_tool_memory(self):
        """Tool memory is still recorded on success (not on error)."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("Done", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub.tool_memory.record.assert_called()

    async def test_error_skips_tool_memory(self):
        """Tool memory is NOT recorded on error."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("Error", False, True, ["check_disk"], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub.tool_memory.record.assert_not_called()


class TestNoCodexEarlyReturnUnchanged:
    """The 'no codex backend' early-return path still removes user message.
    This path returns BEFORE the is_error check, so it's unaffected by Round 14."""

    async def test_no_codex_still_removes_user_message(self):
        stub = _make_bot_stub()
        stub.codex_client = None
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        # This path still removes user message (early return at line 1061)
        stub.sessions.remove_last_message.assert_called_with("chan-1", "user")
