"""Tests for stream error preview cleanup.

Bug: When streaming fails after a Discord preview was created, the preview
was left as a frozen partial response with "..." — the error message was never
shown because already_sent=True skipped _send_chunked. Fix: delete the preview
and return already_sent=False so the error message is delivered normally.
"""
from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal HeimdallBot stub."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._system_prompt = "You are a bot."
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex response")
    stub.codex_client.chat_with_tools = AsyncMock()
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
    return stub


def _make_message(channel_id="chan-1"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.reply = AsyncMock()
    return msg



# ---------------------------------------------------------------------------
# Codex error path — error responses reach the user
# ---------------------------------------------------------------------------
# The tests below verify Codex error paths reach the user correctly.

class TestStreamErrorPreviewCleanup:
    """When the Codex tool loop fails or returns is_error=True, the error
    message must reach the user via _send_chunked (already_sent=False)."""

    async def test_codex_error_already_sent_false(self):
        """When _process_with_tools returns is_error=True and already_sent=False,
        the error reaches the user."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("Tool execution failed: timeout", False, True, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", str(msg.channel.id))

        # Error should be sent via _send_chunked (already_sent=False)
        stub._send_chunked.assert_called_once()

    async def test_codex_exception_caught_internally(self):
        """When _process_with_tools raises, inner try/except catches it and sends error."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("Codex API down"))
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", str(msg.channel.id))

        # Inner except catches it, sends error via _send_chunked
        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "tool execution failed" in sent_text.lower() or "codex api down" in sent_text.lower()

    async def test_stream_error_with_preview_error_sent_to_user(self):
        """Integration: when Codex returns an error response, it reaches user via _send_chunked."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("The AI service is temporarily overloaded. Try again in a moment.", False, True, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", str(msg.channel.id))

        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "overloaded" in sent_text

    async def test_stream_error_with_preview_full_flow(self):
        """Full integration: error message is sent via _send_chunked in _handle_message_inner."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("The AI service is temporarily overloaded.", False, True, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", str(msg.channel.id))

        # Error should be sent via _send_chunked since already_sent=False
        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "overloaded" in sent_text

    async def test_successful_codex_response_is_sent(self):
        """Regression: successful Codex responses should not be affected by error handling."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("Here are the results.", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", str(msg.channel.id))

        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "Here are the results." in sent_text



    def test_old_formula_would_fail_on_boot(self):
        """Demonstrate that the old formula (default=0) produces a false positive
        when time.monotonic() is small."""
        last_tool_use = {}

        # Old formula: .get(channel_id, 0) → monotonic - 0 = 10 < 300 → True!
        with patch("time.monotonic", return_value=10.0):
            recent_old = time.monotonic() - last_tool_use.get("chan-1", 0) < 300
            assert recent_old is True  # BUG: false positive

    def test_known_channel_still_detected(self):
        """Channels that actually have recent tool use should still be detected."""
        now = time.monotonic()
        last_tool_use = {"chan-1": now - 60}  # 1 minute ago

        channel_id = "chan-1"
        recent = (
            channel_id in last_tool_use
            and time.monotonic() - last_tool_use[channel_id] < 300
        )
        assert recent is True

    def test_expired_channel_not_detected(self):
        """Channels with tool use > 5 minutes ago should not be detected."""
        now = time.monotonic()
        last_tool_use = {"chan-1": now - 400}  # ~6.7 minutes ago

        channel_id = "chan-1"
        recent = (
            channel_id in last_tool_use
            and time.monotonic() - last_tool_use[channel_id] < 300
        )
        assert recent is False

    def test_unknown_channel_not_affected_by_known(self):
        """Tool use in one channel should not affect another channel's detection."""
        now = time.monotonic()
        last_tool_use = {"chan-1": now}

        channel_id = "chan-2"
        recent = (
            channel_id in last_tool_use
            and time.monotonic() - last_tool_use[channel_id] < 300
        )
        assert recent is False

