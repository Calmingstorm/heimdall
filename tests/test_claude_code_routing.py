"""Tests for message routing after classifier removal.

All messages now route to "task" (Codex with tools) by default.
The Haiku classifier has been removed; msg_type is hardcoded to "task".

Tests cover:
- Task routing via _process_with_tools when Codex is available
- Error when no Codex backend and message classified as task
- Keyword bypass still routes to task (not claude_code)
- Image messages force task route
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal HeimdallBot stub for _handle_message_inner tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._system_prompt = "initial system prompt"
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.max_tool_iterations_chat = 30
    stub.config.tools.max_tool_iterations_loop = 100
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex fallback response")
    stub.codex_client.chat_with_tools = AsyncMock(return_value=MagicMock())
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor._handle_claude_code = AsyncMock(
        return_value="The check_service function does X, Y, Z..."
    )
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._process_with_tools = AsyncMock(
        return_value=("Codex task response", False, False, [], False)
    )
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.voice_manager = None
    stub._pending_files = {}
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
# Routing: all messages go to task (Codex with tools)
# ---------------------------------------------------------------------------

class TestTaskRouting:
    """All non-guest messages now route to task (Codex with tools)."""

    async def test_task_with_codex_uses_process_with_tools(self):
        """When Codex is available, messages route to _process_with_tools."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Chat response")
        stub._process_with_tools = AsyncMock(return_value=("Tool result", False, False, ["check_disk"], False))
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check the server metrics", "chan-1")

        # Should route to Codex tool calling
        stub._process_with_tools.assert_called_once()
        # system_prompt_override should be passed
        call_kwargs = stub._process_with_tools.call_args[1]
        assert "system_prompt_override" in call_kwargs

    async def test_task_no_codex_returns_error(self):
        """When no Codex configured, should return error."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = None
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "restart apache", "chan-1")

        # Should send "no tool backend" message
        stub._send_with_retry.assert_called_once()
        call_args = stub._send_with_retry.call_args[0]
        assert "no tool backend" in call_args[1].lower()

    async def test_plain_message_routes_to_task(self):
        """A plain message (no keywords, no images) should route to task."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "hello there", "chan-1")

        # Should go to _process_with_tools (task route), not _handle_claude_code
        stub._process_with_tools.assert_called_once()
        stub.tool_executor._handle_claude_code.assert_not_called()


# ---------------------------------------------------------------------------
# Routing: keyword bypass still routes to task
# ---------------------------------------------------------------------------

class TestKeywordBypassUnchanged:
    """Keyword-matched messages should still go directly to 'task'."""

    async def test_keyword_match_goes_to_task(self):
        """Messages with infra keywords should go to task."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._process_with_tools = AsyncMock(
            return_value=("Done", False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "deploy the latest code", "chan-1")

        # Keyword match -> task -> Codex tool loop
        stub.tool_executor._handle_claude_code.assert_not_called()
        stub._process_with_tools.assert_called_once()

    async def test_image_message_goes_to_task(self):
        """Image messages should force task route."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._process_with_tools = AsyncMock(
            return_value=("I see the image", False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(
            msg, "what is this?", "chan-1",
            image_blocks=[{"type": "image", "source": {"data": "base64data"}}],
        )

        stub.tool_executor._handle_claude_code.assert_not_called()
        stub._process_with_tools.assert_called_once()
