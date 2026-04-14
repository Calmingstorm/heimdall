"""Round 4 verification: confirm routing works.

Tests verify:
1. All messages (human + bot) route to _process_with_tools (Codex with tools)
2. Single routing path (no 3-way branching)

Note: Removal checks (approval, classifier, schedule guard, claude_code registry)
are consolidated into test_round10_verification.py.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

from src.discord.client import HeimdallBot  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(respond_to_bots=False):
    """Minimal HeimdallBot stub for routing verification."""
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
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.respond_to_bots = respond_to_bots
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="chat response")
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="result")
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._process_with_tools = AsyncMock(
        return_value=("Executed successfully.", False, False, ["run_command"], False)
    )
    stub._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run command", "input_schema": {"type": "object", "properties": {}}},
        {"name": "claude_code", "description": "Coding agent", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._track_recent_action = MagicMock()
    return stub


def _make_message(is_bot=False):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = "chan-1"
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.display_name = "Bot" if is_bot else "User"
    msg.author.name = "bot" if is_bot else "user"
    msg.author.bot = is_bot
    msg.reply = AsyncMock()
    return msg


# ===========================================================================
# 1. Human message routing
# ===========================================================================

class TestHumanMessageRouting:
    """Human (non-guest) messages route to _process_with_tools."""

    async def test_human_message_uses_tools(self):
        """A normal user message routes through _process_with_tools."""
        stub = _make_bot_stub()
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message(is_bot=False)
        await stub._handle_message_inner(msg, "check disk on server", "chan-1")

        stub._process_with_tools.assert_called_once()
        stub._send_chunked.assert_called()

    async def test_human_casual_message_still_uses_tools(self):
        """Even casual messages ('hello') go through tools path."""
        stub = _make_bot_stub()
        stub._process_with_tools = AsyncMock(
            return_value=("Hey there!", False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message(is_bot=False)
        await stub._handle_message_inner(msg, "hello", "chan-1")

        stub._process_with_tools.assert_called_once()
        # Chat route (codex_client.chat) should NOT be called for non-guest
        stub.codex_client.chat.assert_not_called()

    async def test_guest_uses_chat_not_tools(self):
        """Guest users get chat route (no tools)."""
        stub = _make_bot_stub()
        stub.permissions.is_guest = MagicMock(return_value=True)
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message(is_bot=False)
        await stub._handle_message_inner(msg, "hello", "chan-1")

        stub._process_with_tools.assert_not_called()
        stub.codex_client.chat.assert_called_once()


# ===========================================================================
# 2. Bot message routing
# ===========================================================================

class TestBotMessageRouting:
    """Bot messages (when respond_to_bots=True) route to _process_with_tools."""

    async def test_bot_message_uses_tools(self):
        """Bot message routes through _process_with_tools, not chat."""
        stub = _make_bot_stub(respond_to_bots=True)
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message(is_bot=True)
        await stub._handle_message_inner(msg, "deploy latest build", "chan-1")

        stub._process_with_tools.assert_called_once()
        stub.codex_client.chat.assert_not_called()
        stub._send_chunked.assert_called()

    async def test_bot_message_result_sent(self):
        """Bot message executes tools and response is sent back."""
        stub = _make_bot_stub(respond_to_bots=True)
        stub._process_with_tools = AsyncMock(
            return_value=("Build deployed to production.", False, False, ["run_command"], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message(is_bot=True)
        await stub._handle_message_inner(msg, "deploy latest build", "chan-1")

        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "deployed" in sent_text.lower()


# ===========================================================================
# 3. Single routing path (no branching)
# ===========================================================================

class TestSingleRoutingPath:
    """Verify there's only one routing path: guest -> chat, everyone else -> tools."""

    async def test_no_msg_type_variable(self):
        """_handle_message_inner should not set msg_type (removed in Round 3)."""
        import inspect
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        assert 'msg_type = "chat"' not in source
        assert 'msg_type = "claude_code"' not in source

    async def test_no_keyword_detection(self):
        """No keyword detection in _handle_message_inner."""
        import inspect
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        assert "is_task_by_keyword" not in source

    async def test_no_classifier_call(self):
        """No classifier call in _handle_message_inner."""
        import inspect
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        assert "classifier" not in source.lower() or "no classifier" in source.lower()
