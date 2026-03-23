"""Round 8: End-to-end integration tests for Issues 1-5.

These tests exercise _handle_message_inner through the full routing path,
verifying that the fixes from Rounds 5-6 work end-to-end:

- Issue 3:   Task route text-only response → _last_tool_use set for channel
- Issue 3:   Task route exception → _last_tool_use still set
- Issue 5:   Task route empty → friendly fallback (not "(no response)")
- Handoff:   Codex handoff returns empty → skill response preserved
- Issue 2:   No "REQUIRES APPROVAL" in any tool description sent to Codex

Note: Chat route tests removed — all messages now route to "task" (no classifier).
"""
from __future__ import annotations

import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot, _EMPTY_RESPONSE_FALLBACK  # noqa: E402
from src.llm.types import LLMResponse  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(**overrides):
    """Create a minimal LokiBot stub for _handle_message_inner tests.

    Uses the same pattern as test_chat_path_optimization.py but adds the
    extra attributes needed for task route and handoff paths.
    """
    stub = MagicMock()

    # State tracking
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._last_tool_use = {}
    stub._pending_files = {}

    # Config
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False

    # Sessions
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.get_or_create = MagicMock()

    # Codex client
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex chat response")
    stub.codex_client.chat_with_tools = AsyncMock(
        return_value=LLMResponse(text="Done", tool_calls=[], stop_reason="end_turn"),
    )

    # Skill manager
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])

    # Audit
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()

    # Prompt building
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)

    # Discord sending
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()

    # Permissions
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)

    # Tool memory
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.tool_memory.suggest = AsyncMock(return_value=[])

    # Reflector
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")

    # Voice
    stub.voice_manager = None

    # Bind the real method under test
    stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

    for k, v in overrides.items():
        setattr(stub, k, v)
    return stub


def _make_message(channel_id="chan-1", content="test"):
    """Create a mock Discord message."""
    msg = AsyncMock()
    msg.content = content
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock()
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(), __aexit__=AsyncMock(),
    ))
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.bot = False
    msg.author.display_name = "TestUser"
    msg.author.name = "TestUser"
    msg.reply = AsyncMock()
    msg.attachments = []
    return msg


# ---------------------------------------------------------------------------
# Issue 3: Task route text-only → _last_tool_use set
# ---------------------------------------------------------------------------

class TestIssue3TaskRouteTracksActivity:
    """After task route completes (even text-only), _last_tool_use is set."""

    async def test_task_text_only_sets_last_tool_use(self):
        """Codex returns text-only (no tool calls) on task route.
        _last_tool_use[channel_id] should be set so follow-ups route to task."""
        stub = _make_bot_stub()
        msg = _make_message(channel_id="chan-42")

        # _process_with_tools returns text-only (no tools, no handoff)
        stub._process_with_tools = AsyncMock(
            return_value=("I can help with that. Shall I proceed?", False, False, [], False)
        )

        before = time.monotonic()
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "restart nginx", "chan-42")
        after = time.monotonic()

        # _last_tool_use should be set even though no tool was actually called
        assert "chan-42" in stub._last_tool_use
        assert stub._last_tool_use["chan-42"] >= before
        assert stub._last_tool_use["chan-42"] <= after

    async def test_task_with_tools_sets_last_tool_use(self):
        """Codex calls tools on task route. _last_tool_use set after completion."""
        stub = _make_bot_stub()
        msg = _make_message(channel_id="chan-42")

        stub._process_with_tools = AsyncMock(
            return_value=("Disk is 42% full.", False, False, ["check_disk"], False)
        )

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk on server", "chan-42")

        assert "chan-42" in stub._last_tool_use

    async def test_task_exception_still_sets_last_tool_use(self):
        """Even if _process_with_tools raises and is caught by inner try/except,
        _last_tool_use should still be set (user might say 'try again')."""
        stub = _make_bot_stub()
        msg = _make_message(channel_id="chan-42")

        stub._process_with_tools = AsyncMock(
            side_effect=RuntimeError("Codex timeout")
        )

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "restart nginx", "chan-42")

        # Should be set despite exception — the inner except catches it
        assert "chan-42" in stub._last_tool_use



# ---------------------------------------------------------------------------
# Issue 5: Task route empty → friendly fallback (not "(no response)")
# ---------------------------------------------------------------------------

class TestIssue5TaskRouteEmptyFallback:
    """Task route should use friendly fallback instead of '(no response)'."""

    async def test_task_empty_response_sends_fallback(self):
        """Full end-to-end: Codex returns empty LLMResponse on task route,
        _process_with_tools applies fallback, _handle_message_inner sends it."""
        stub = _make_bot_stub()
        msg = _make_message(channel_id="chan-1")

        # Mock Codex to return empty LLMResponse (the real _process_with_tools runs)
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="", tool_calls=[], stop_reason="end_turn"),
        )
        stub._cancelled_tasks = set()
        # Bind the REAL _process_with_tools so fallback logic runs
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "(no response)" not in sent_text
        assert "try again" in sent_text.lower()
        assert sent_text == _EMPTY_RESPONSE_FALLBACK

    async def test_task_empty_response_from_process_with_tools(self):
        """Test the _process_with_tools method directly: empty LLMResponse
        should yield the fallback constant."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="", tool_calls=[], stop_reason="end_turn"),
        )
        stub._cancelled_tasks = set()
        # Bind the real _process_with_tools
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        text, already_sent, is_error, tools_used, handoff = await stub._process_with_tools(
            msg, [{"role": "user", "content": "hello"}], system_prompt_override="test",
        )

        assert text == _EMPTY_RESPONSE_FALLBACK
        assert "(no response)" not in text


# ---------------------------------------------------------------------------
# Handoff: Empty Codex chat → skill response preserved
# ---------------------------------------------------------------------------

class TestHandoffPreservesSkillResponseIntegration:
    """When skill handoff Codex chat returns empty, original skill output is used."""

    async def test_handoff_empty_preserves_skill_response(self):
        """Full _handle_message_inner path: skill produces output, handoff chat()
        returns empty → original skill output should be sent to Discord."""
        stub = _make_bot_stub()
        msg = _make_message(channel_id="chan-1")

        # _process_with_tools returns skill output with handoff=True
        skill_output = "Nginx restarted successfully on server."
        stub._process_with_tools = AsyncMock(
            return_value=(skill_output, False, False, ["restart_service"], True)
        )
        # Codex handoff chat returns empty
        stub.codex_client.chat = AsyncMock(return_value="")

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "restart nginx", "chan-1")

        # The skill output should be preserved and sent (not the empty Codex response)
        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "Nginx restarted" in sent_text

    async def test_handoff_exception_preserves_skill_response(self):
        """When handoff chat() raises, original skill output should be sent."""
        stub = _make_bot_stub()
        msg = _make_message(channel_id="chan-1")

        skill_output = "Disk usage: 42% on server."
        stub._process_with_tools = AsyncMock(
            return_value=(skill_output, False, False, ["check_disk"], True)
        )
        # Codex handoff raises
        stub.codex_client.chat = AsyncMock(side_effect=RuntimeError("Codex down"))

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "42%" in sent_text

    async def test_handoff_success_uses_codex_response(self):
        """When handoff chat() succeeds, its response should be sent."""
        stub = _make_bot_stub()
        msg = _make_message(channel_id="chan-1")

        stub._process_with_tools = AsyncMock(
            return_value=("Raw tool output", False, False, ["check_disk"], True)
        )
        stub.codex_client.chat = AsyncMock(
            return_value="The disk on server is 42% full. Looking good!"
        )

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "42% full" in sent_text
        assert "Looking good" in sent_text


# ---------------------------------------------------------------------------
# Issue 2: Tool descriptions sent to Codex are clean
# ---------------------------------------------------------------------------

class TestIssue2ToolDescriptionsClean:
    """No tool description sent to Codex should contain 'REQUIRES APPROVAL'."""

    def test_converted_tools_have_no_approval_text(self):
        """The OpenAI-format tool definitions produced by _convert_tools()
        should not contain 'REQUIRES APPROVAL' in any description."""
        from src.tools.registry import TOOLS
        from src.llm.openai_codex import CodexChatClient

        converted = CodexChatClient._convert_tools(TOOLS)
        for tool in converted:
            func = tool.get("function", tool)
            desc = func.get("description", "")
            assert "REQUIRES APPROVAL" not in desc, (
                f"Tool {func.get('name', '?')} description still contains "
                f"'REQUIRES APPROVAL': {desc[:100]}"
            )

    def test_no_requires_approval_flags(self):
        """The requires_approval field should no longer exist on tools."""
        from src.tools.registry import TOOLS
        for tool in TOOLS:
            assert "requires_approval" not in tool, (
                f"Tool {tool['name']} should not have requires_approval field"
            )
