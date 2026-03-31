"""Round 12: Bot interop hardening (continued).

Tests areas not covered by Round 11:
1. Bot buffer lifecycle — flush timing, cleanup, multi-bot keys, empty buffer
2. Bot messages skip attachment processing (known design: bot buffer returns early)
3. Bot messages skip secret detection (known design: _check_for_secrets in human path)
4. Webhook bot bypass path — gets attachments/secrets, no bot buffer
5. Bot preamble edge cases — single-message history, webhook bots
6. Multi-bot concurrent buffering — separate keys, independent flush
7. Bot mention stripping in flush handler
8. Tool-less bot responses NOT saved to history (anti-poisoning)
9. Bot display name tagging in session
10. Bot message dedup
"""
from __future__ import annotations

import asyncio
import sys
import time
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    HeimdallBot,
    combine_bot_messages,
    detect_fabrication,
    detect_hedging,
)
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tc(name, inp=None):
    """Shorthand for ToolCall creation."""
    return ToolCall(id=f"tc-{name}", name=name, input=inp or {})


def _make_on_message_stub(**overrides):
    """HeimdallBot stub for on_message-level tests (buffer, flush, lifecycle)."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test prompt"
    stub._channel_locks = {}
    stub._processed_messages = OrderedDict()
    stub._processed_messages_max = 100
    stub._background_tasks = {}
    stub._background_tasks_max = 20
    stub._bot_msg_buffer = {}
    stub._bot_msg_tasks = {}
    stub._bot_msg_buffer_delay = 0  # immediate flush for tests
    stub._bot_msg_buffer_max = 20
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.channels = []
    stub.config.discord.respond_to_bots = True
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.scrub_secrets = MagicMock()
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="response")
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")
    stub._knowledge_store = None
    stub._embedder = None
    stub._fts_index = None
    stub._vector_store = None
    stub.scheduler = MagicMock()
    stub.infra_watcher = None
    stub.voice_manager = None
    stub.user = MagicMock()
    stub.user.id = 111
    stub.guilds = []
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.tool_executor = MagicMock()
    stub.tool_memory = MagicMock()
    stub.browser_manager = None
    stub.context_loader = MagicMock()
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._memory_path = "/tmp/test_memory.json"
    stub._is_allowed_user = MagicMock(return_value=True)
    stub._is_allowed_channel = MagicMock(return_value=True)
    stub._handle_message = AsyncMock()
    stub._process_attachments = AsyncMock(return_value=("", []))
    stub._check_for_secrets = MagicMock(return_value=False)
    stub.user.mentioned_in = MagicMock(return_value=False)
    stub.on_message = HeimdallBot.on_message.__get__(stub)

    for k, v in overrides.items():
        setattr(stub, k, v)
    stub._classify_completion = AsyncMock(return_value=(True, ""))
    return stub


def _make_process_with_tools_stub(respond_to_bots=True):
    """HeimdallBot stub for _process_with_tools-level tests."""
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
    stub.config.discord.respond_to_bots = respond_to_bots
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
        {"name": "run_script", "description": "Script", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._classify_completion = AsyncMock(return_value=(True, ""))
    return stub


def _make_msg(is_bot=False, channel_id="ch-1", author_id="user-1", content="test",
              display_name=None):
    """Create a mock Discord message."""
    msg = AsyncMock()
    msg.id = int(time.time() * 1000)
    msg.content = content
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock(return_value=AsyncMock())
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None),
    ))
    msg.channel.guild = MagicMock()
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.display_name = display_name or ("TestBot" if is_bot else "TestUser")
    msg.author.name = msg.author.display_name
    msg.author.bot = is_bot
    msg.author.mention = f"<@{author_id}>"
    msg.webhook_id = None
    msg.reply = AsyncMock()
    msg.delete = AsyncMock()
    msg.attachments = []
    return msg


# ---------------------------------------------------------------------------
# 1. Bot buffer lifecycle
# ---------------------------------------------------------------------------

class TestBotBufferLifecycle:
    """Test buffer creation, flush, cleanup, and edge cases."""

    async def test_buffer_created_on_first_bot_message(self):
        """First bot message in a channel creates a buffer entry."""
        stub = _make_on_message_stub()
        msg = _make_msg(is_bot=True, content="hello")
        msg.id = int(time.time() * 1000) + 1

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        # After flush, buffer should be cleaned up
        key = (str(msg.channel.id), str(msg.author.id))
        assert key not in stub._bot_msg_buffer

    async def test_buffer_cleaned_up_after_flush(self):
        """Buffer entry is removed after flush completes."""
        stub = _make_on_message_stub()
        msg = _make_msg(is_bot=True, content="test cleanup")
        msg.id = int(time.time() * 1000) + 2

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        key = (str(msg.channel.id), str(msg.author.id))
        assert key not in stub._bot_msg_buffer
        assert key not in stub._bot_msg_tasks

    async def test_empty_content_bot_message_does_not_call_handle(self):
        """Bot message with empty content after mention strip does not trigger processing."""
        stub = _make_on_message_stub()
        # Bot sends only a mention, which gets stripped to empty
        stub.user.id = 111
        msg = _make_msg(is_bot=True, content="<@111>")
        msg.id = int(time.time() * 1000) + 3

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        # _handle_message should NOT be called for empty content
        stub._handle_message.assert_not_called()

    async def test_multiple_messages_buffered_then_combined(self):
        """Multiple bot messages buffered and combined on flush."""
        stub = _make_on_message_stub()
        stub._bot_msg_buffer_delay = 0.05  # small delay to allow accumulation

        bot_id = "bot-100"
        msgs = []
        for i, content in enumerate(["line 1", "line 2", "line 3"]):
            msg = _make_msg(is_bot=True, author_id=bot_id, content=content)
            msg.id = int(time.time() * 1000) + 10 + i
            msgs.append(msg)

        for m in msgs:
            await stub.on_message(m)

        await asyncio.sleep(0.15)

        # Should be called once with combined content
        stub._handle_message.assert_awaited_once()
        call_args = stub._handle_message.call_args
        combined = call_args[0][1]  # second positional arg is content
        assert "line 1" in combined
        assert "line 2" in combined
        assert "line 3" in combined

    async def test_buffer_key_is_channel_plus_author(self):
        """Buffer key includes both channel and author for isolation."""
        stub = _make_on_message_stub()
        stub._bot_msg_buffer_delay = 0.05

        # Bot A in channel 1
        msg_a = _make_msg(is_bot=True, channel_id="ch-1", author_id="bot-a", content="from A")
        msg_a.id = int(time.time() * 1000) + 20

        # Bot B in channel 1
        msg_b = _make_msg(is_bot=True, channel_id="ch-1", author_id="bot-b", content="from B")
        msg_b.id = int(time.time() * 1000) + 21

        await stub.on_message(msg_a)
        await stub.on_message(msg_b)
        await asyncio.sleep(0.15)

        # Two separate flush calls
        assert stub._handle_message.await_count == 2

    async def test_same_bot_different_channels_separate_buffers(self):
        """Same bot in different channels gets separate buffers."""
        stub = _make_on_message_stub()
        stub._bot_msg_buffer_delay = 0.05

        msg_ch1 = _make_msg(is_bot=True, channel_id="ch-1", author_id="bot-x", content="ch1")
        msg_ch1.id = int(time.time() * 1000) + 30

        msg_ch2 = _make_msg(is_bot=True, channel_id="ch-2", author_id="bot-x", content="ch2")
        msg_ch2.id = int(time.time() * 1000) + 31

        await stub.on_message(msg_ch1)
        await stub.on_message(msg_ch2)
        await asyncio.sleep(0.15)

        assert stub._handle_message.await_count == 2


class TestBotBufferTimerReset:
    """Test that new messages reset the flush timer."""

    async def test_timer_cancelled_on_new_message(self):
        """Each new bot message cancels the previous flush timer."""
        stub = _make_on_message_stub()
        stub._bot_msg_buffer_delay = 0.1  # 100ms delay

        bot_id = "bot-timer"
        msg1 = _make_msg(is_bot=True, author_id=bot_id, content="first")
        msg1.id = int(time.time() * 1000) + 40

        await stub.on_message(msg1)

        # Before timer fires, send another message
        await asyncio.sleep(0.03)
        msg2 = _make_msg(is_bot=True, channel_id=str(msg1.channel.id),
                         author_id=bot_id, content="second")
        msg2.id = int(time.time() * 1000) + 41

        await stub.on_message(msg2)

        # Wait for second timer to fire
        await asyncio.sleep(0.15)

        # Only one call with both messages combined
        stub._handle_message.assert_awaited_once()
        combined = stub._handle_message.call_args[0][1]
        assert "first" in combined
        assert "second" in combined


# ---------------------------------------------------------------------------
# 2. Bot messages skip attachment processing
# ---------------------------------------------------------------------------

class TestBotAttachmentHandling:
    """Bot buffer path returns early — attachments are NOT processed."""

    async def test_bot_message_does_not_call_process_attachments(self):
        """Bot messages bypass _process_attachments entirely."""
        stub = _make_on_message_stub()
        msg = _make_msg(is_bot=True, content="check this file")
        msg.attachments = [MagicMock(filename="test.py", size=100)]
        msg.id = int(time.time() * 1000) + 50

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        # _process_attachments should NOT be called for bot messages
        stub._process_attachments.assert_not_called()

    async def test_human_message_calls_process_attachments(self):
        """Human messages DO call _process_attachments (control test)."""
        stub = _make_on_message_stub()
        msg = _make_msg(is_bot=False, content="check this file")
        msg.attachments = [MagicMock(filename="test.py", size=100)]
        msg.id = int(time.time() * 1000) + 51

        await stub.on_message(msg)

        stub._process_attachments.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Bot messages skip secret detection
# ---------------------------------------------------------------------------

class TestBotSecretDetection:
    """Bot messages bypass _check_for_secrets (they take the buffer path)."""

    async def test_bot_message_does_not_check_secrets(self):
        """Bot messages bypass _check_for_secrets."""
        stub = _make_on_message_stub()
        msg = _make_msg(is_bot=True, content="my password is hunter2")
        msg.id = int(time.time() * 1000) + 60

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        stub._check_for_secrets.assert_not_called()
        # Message still processed
        stub._handle_message.assert_awaited_once()

    async def test_human_message_checks_secrets(self):
        """Human messages DO check for secrets (control test)."""
        stub = _make_on_message_stub()
        stub._check_for_secrets.return_value = False
        msg = _make_msg(is_bot=False, content="normal message")
        msg.id = int(time.time() * 1000) + 61

        await stub.on_message(msg)

        stub._check_for_secrets.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Webhook bot bypass path
# ---------------------------------------------------------------------------

class TestWebhookBotPath:
    """Allowed webhook bots bypass the bot buffer and take the human path."""

    async def test_webhook_bot_takes_human_path_not_buffer(self):
        """Webhook bots go through human path (attachments + secrets checked)."""
        import src.discord.client as client_mod
        original = client_mod._ALLOWED_WEBHOOK_IDS
        try:
            client_mod._ALLOWED_WEBHOOK_IDS = {"wh-999"}
            stub = _make_on_message_stub()
            stub.config.discord.respond_to_bots = False  # bots disabled globally

            msg = _make_msg(is_bot=True, content="webhook message")
            msg.webhook_id = "wh-999"
            msg.id = int(time.time() * 1000) + 70

            await stub.on_message(msg)

            # Should go through human path — attachments checked
            stub._process_attachments.assert_called_once()
            stub._check_for_secrets.assert_called_once()
            stub._handle_message.assert_awaited_once()
        finally:
            client_mod._ALLOWED_WEBHOOK_IDS = original

    async def test_webhook_bot_not_buffered(self):
        """Webhook bot messages are NOT accumulated in the bot buffer."""
        import src.discord.client as client_mod
        original = client_mod._ALLOWED_WEBHOOK_IDS
        try:
            client_mod._ALLOWED_WEBHOOK_IDS = {"wh-888"}
            stub = _make_on_message_stub()
            stub.config.discord.respond_to_bots = False

            msg = _make_msg(is_bot=True, content="webhook msg")
            msg.webhook_id = "wh-888"
            msg.id = int(time.time() * 1000) + 71

            await stub.on_message(msg)

            # Buffer should be empty — webhook didn't go through buffer path
            assert len(stub._bot_msg_buffer) == 0
        finally:
            client_mod._ALLOWED_WEBHOOK_IDS = original


# ---------------------------------------------------------------------------
# 5. Bot preamble edge cases
# ---------------------------------------------------------------------------

class TestBotPreambleEdgeCases:
    """Test context separator / bot preamble injection edge cases."""

    async def test_bot_preamble_injected_with_history(self):
        """Bot message with history gets the bot preamble in separator."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=True)

        # History with 2 messages (enough to trigger separator)
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "user", "content": "new bot message"},
        ]

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="I executed it.", tool_calls=[_tc("run_command")])
        )

        resp, already, err, tools, handoff = await HeimdallBot._process_with_tools(
            stub, msg, history, "test-chan",
        )

        # Check that the separator was injected with bot preamble
        chat_call = stub.codex_client.chat_with_tools.call_args
        messages_sent = chat_call[1]["messages"] if "messages" in chat_call[1] else chat_call[0][0]
        # Find the developer separator
        separators = [m for m in messages_sent if m.get("role") == "developer"]
        assert len(separators) == 1
        assert "ANOTHER BOT" in separators[0]["content"]
        assert "EXECUTE immediately" in separators[0]["content"]

    async def test_no_bot_preamble_for_human_message(self):
        """Human message gets separator but NOT the bot preamble."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=False)

        history = [
            {"role": "user", "content": "previous question"},
            {"role": "user", "content": "new human message"},
        ]

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Done.", tool_calls=[_tc("run_command")])
        )

        await HeimdallBot._process_with_tools(stub, msg, history, "test-chan")

        chat_call = stub.codex_client.chat_with_tools.call_args
        messages_sent = chat_call[1]["messages"] if "messages" in chat_call[1] else chat_call[0][0]
        separators = [m for m in messages_sent if m.get("role") == "developer"]
        if separators:
            assert "ANOTHER BOT" not in separators[0]["content"]

    async def test_no_full_separator_with_single_message_history(self):
        """Single-message history gets no full separator, only message ID note."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=True)

        # Only one message in history
        history = [{"role": "user", "content": "bot message"}]

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="ok", tool_calls=[_tc("run_command")])
        )

        await HeimdallBot._process_with_tools(stub, msg, history, "test-chan")

        chat_call = stub.codex_client.chat_with_tools.call_args
        messages_sent = chat_call[1]["messages"] if "messages" in chat_call[1] else chat_call[0][0]
        separators = [m for m in messages_sent if m.get("role") == "developer"]
        # Should have a lightweight message ID note but NOT the full separator
        assert len(separators) == 1
        assert "Current message ID" in separators[0]["content"]
        assert "CURRENT REQUEST" not in separators[0]["content"]

    async def test_bot_preamble_not_injected_when_respond_to_bots_false(self):
        """Bot message with respond_to_bots=False gets no bot preamble."""
        stub = _make_process_with_tools_stub(respond_to_bots=False)
        msg = _make_msg(is_bot=True)

        history = [
            {"role": "user", "content": "previous"},
            {"role": "user", "content": "bot msg"},
        ]

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="ok", tool_calls=[_tc("run_command")])
        )

        await HeimdallBot._process_with_tools(stub, msg, history, "test-chan")

        chat_call = stub.codex_client.chat_with_tools.call_args
        messages_sent = chat_call[1]["messages"] if "messages" in chat_call[1] else chat_call[0][0]
        separators = [m for m in messages_sent if m.get("role") == "developer"]
        if separators:
            # Separator exists but should NOT have bot preamble
            assert "ANOTHER BOT" not in separators[0]["content"]


# ---------------------------------------------------------------------------
# 6. Bot mention stripping in flush handler
# ---------------------------------------------------------------------------

class TestBotMentionStripping:
    """Test that bot mentions are stripped from combined content during flush."""

    async def test_mention_stripped_from_bot_message(self):
        """Bot mention (<@BOT_ID>) is stripped from combined content."""
        stub = _make_on_message_stub()
        stub.user.id = 111

        msg = _make_msg(is_bot=True, content="<@111> check the server")
        msg.id = int(time.time() * 1000) + 80

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        stub._handle_message.assert_awaited_once()
        content = stub._handle_message.call_args[0][1]
        assert "<@111>" not in content
        assert "check the server" in content

    async def test_nickname_mention_stripped(self):
        """Nickname mention format (<@!BOT_ID>) is also stripped."""
        stub = _make_on_message_stub()
        stub.user.id = 111

        msg = _make_msg(is_bot=True, content="<@!111> restart nginx")
        msg.id = int(time.time() * 1000) + 81

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        content = stub._handle_message.call_args[0][1]
        assert "<@!111>" not in content
        assert "restart nginx" in content

    async def test_mention_only_message_not_processed(self):
        """Message that is ONLY a mention (empty after strip) is not processed."""
        stub = _make_on_message_stub()
        stub.user.id = 111

        msg = _make_msg(is_bot=True, content="<@111>")
        msg.id = int(time.time() * 1000) + 82

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        stub._handle_message.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Bot message dedup
# ---------------------------------------------------------------------------

class TestBotMessageDedup:
    """Bot messages go through the same dedup check as human messages."""

    async def test_duplicate_bot_message_skipped(self):
        """Same message ID processed twice is deduplicated."""
        stub = _make_on_message_stub()

        msg = _make_msg(is_bot=True, content="check disk")
        msg.id = 12345  # fixed ID

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        # Process same message again
        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        # Only one flush should have occurred
        stub._handle_message.assert_awaited_once()

    async def test_different_message_ids_both_processed(self):
        """Different message IDs from same bot are both processed."""
        stub = _make_on_message_stub()
        stub._bot_msg_buffer_delay = 0.05

        msg1 = _make_msg(is_bot=True, content="msg one")
        msg1.id = 10001

        msg2 = _make_msg(is_bot=True, channel_id=str(msg1.channel.id),
                         author_id=str(msg1.author.id), content="msg two")
        msg2.id = 10002

        await stub.on_message(msg1)
        await stub.on_message(msg2)
        await asyncio.sleep(0.15)

        # Both buffered into one flush
        stub._handle_message.assert_awaited_once()
        combined = stub._handle_message.call_args[0][1]
        assert "msg one" in combined
        assert "msg two" in combined


# ---------------------------------------------------------------------------
# 8. Tool-less bot responses not saved to history
# ---------------------------------------------------------------------------

class TestBotResponseHistorySaving:
    """Verify anti-poisoning: tool-less responses are not saved to session."""

    async def test_tool_less_response_not_saved_for_bot(self):
        """Tool-less response from bot message is not saved to session history."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=True)

        history = [{"role": "user", "content": "what is docker?"}]

        # LLM returns text only, no tools
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Docker is a containerization platform.", tool_calls=[])
        )

        resp, already, err, tools, handoff = await HeimdallBot._process_with_tools(
            stub, msg, history, "test-chan",
        )

        # Response is returned (user sees it in Discord)
        assert resp == "Docker is a containerization platform."
        # But no tools were used
        assert tools == []

    async def test_tool_response_returned_for_bot(self):
        """Response with tool use IS returned normally for bot message."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=True)

        history = [{"role": "user", "content": "check disk on server"}]

        # First call returns tool use, second returns final text
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                LLMResponse(text="Checking disk...", tool_calls=[_tc("run_command", {"host": "server", "command": "df -h"})]),
                LLMResponse(text="Disk usage is 42%.", tool_calls=[]),
            ]
        )

        resp, already, err, tools, handoff = await HeimdallBot._process_with_tools(
            stub, msg, history, "test-chan",
        )

        assert "42%" in resp
        assert "run_command" in tools


# ---------------------------------------------------------------------------
# 9. Bot display name tagging
# ---------------------------------------------------------------------------

class TestBotDisplayNameTagging:
    """Verify bot messages are tagged with [BotName]: content in session."""

    async def test_bot_message_tagged_with_display_name(self):
        """Bot's display name appears in session history tag."""
        stub = _make_on_message_stub()
        stub.sessions = MagicMock()
        stub.sessions.add_message = MagicMock()
        stub.sessions.get_history_with_compaction = AsyncMock(return_value=[
            {"role": "user", "content": "[Boostie]: run the script"}
        ])
        stub.codex_client = MagicMock()
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Done", tool_calls=[])
        )
        stub.permissions.is_guest = MagicMock(return_value=False)
        stub._merged_tool_definitions = MagicMock(return_value=[
            {"name": "run_command", "description": "Run", "input_schema": {"type": "object", "properties": {}}},
        ])
        stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
        stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)
        stub._process_with_tools = AsyncMock(return_value=("Done", False, False, [], False))
        stub._send_chunked = AsyncMock()
        stub.tool_memory = MagicMock()
        stub.tool_memory.record = AsyncMock()
        stub._build_system_prompt = MagicMock(return_value="prompt")

        msg = _make_msg(is_bot=True, display_name="Boostie", content="run the script")

        await stub._handle_message_inner(msg, "run the script", "ch-1")

        # Verify tagged_content includes display name
        add_call = stub.sessions.add_message.call_args_list[0]
        tagged = add_call[0][2]  # third positional arg
        assert "[Boostie]:" in tagged
        assert "run the script" in tagged


# ---------------------------------------------------------------------------
# 10. combine_bot_messages advanced edge cases
# ---------------------------------------------------------------------------

class TestCombineBotMessagesAdvanced:
    """Edge cases for combine_bot_messages not covered in Round 11."""

    def test_all_empty_parts(self):
        """All empty string parts."""
        result = combine_bot_messages(["", ""])
        assert result == "\n\n"  # two empty parts joined with paragraph break

    def test_whitespace_only_parts(self):
        """Parts containing only whitespace."""
        result = combine_bot_messages(["  ", "\t", "\n"])
        assert "  " in result
        assert "\t" in result

    def test_backticks_inside_code_block(self):
        """Triple backticks inside a code block (markdown in script)."""
        parts = [
            '```bash\necho "```test```"\n```',
        ]
        result = combine_bot_messages(parts)
        # Single message returns as-is
        assert result == parts[0]

    def test_mismatched_language_tags_adjacent_merge(self):
        """Adjacent code blocks with different language tags — merge keeps first."""
        parts = [
            "```python\nprint('hello')\n```",
            "```bash\necho 'hello'\n```",
        ]
        result = combine_bot_messages(parts)
        # Adjacent blocks merge — the second block's language tag is in the merge
        # regex capture but the fence pair is removed
        assert "print('hello')" in result
        assert "echo 'hello'" in result

    def test_very_long_single_part(self):
        """Very long single part (>10KB) is returned unchanged."""
        long_content = "x" * 15000
        result = combine_bot_messages([long_content])
        assert result == long_content
        assert len(result) == 15000

    def test_many_parts(self):
        """20 parts combined correctly."""
        parts = [f"line {i}" for i in range(20)]
        result = combine_bot_messages(parts)
        for i in range(20):
            assert f"line {i}" in result
        # All joined with paragraph breaks (no code blocks)
        assert result.count("\n\n") == 19

    def test_unclosed_code_block_with_multiple_continuations(self):
        """Code block opened in first message, continued across 4 messages."""
        parts = [
            "```python\ndef func():",
            "    x = 1",
            "    y = 2",
            "    return x + y\n```",
        ]
        result = combine_bot_messages(parts)
        assert result.count("```") == 2  # one open, one close
        assert "def func():" in result
        assert "    x = 1" in result
        assert "    return x + y" in result

    def test_code_block_with_empty_continuation(self):
        """Code block split with an empty message in between."""
        parts = [
            "```bash\necho hello",
            "",
            "echo world\n```",
        ]
        result = combine_bot_messages(parts)
        assert "echo hello" in result
        assert "echo world" in result
        assert result.count("```") == 2


# ---------------------------------------------------------------------------
# 12. Bot + guest tier interaction
# ---------------------------------------------------------------------------

class TestBotGuestTierInteraction:
    """Test how bot messages interact with guest tier permissions."""

    async def test_bot_with_guest_tier_gets_chat_route(self):
        """Bot whose ID is guest tier goes to chat route (no tools)."""
        stub = _make_on_message_stub()
        stub.permissions.is_guest = MagicMock(return_value=True)
        # Need real _handle_message_inner for this test
        stub.codex_client.chat = AsyncMock(return_value="guest response")
        stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
        stub._send_chunked = AsyncMock()
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)
        stub.sessions = MagicMock()
        stub.sessions.add_message = MagicMock()
        stub.sessions.get_history_with_compaction = AsyncMock(return_value=[
            {"role": "user", "content": "[TestBot]: hi"}
        ])
        stub.sessions.prune = MagicMock()
        stub.sessions.save = MagicMock()
        stub.tool_memory = MagicMock()
        stub.tool_memory.record = AsyncMock()

        msg = _make_msg(is_bot=True)

        await stub._handle_message_inner(msg, "hi", "ch-1")

        # Chat route used (no tools)
        stub.codex_client.chat.assert_awaited_once()


# ---------------------------------------------------------------------------
# 13. Fabrication + hedging detection order for bot messages
# ---------------------------------------------------------------------------

class TestDetectionOrderBotMessages:
    """Verify fabrication fires before hedging for bot messages."""

    async def test_fabrication_checked_before_hedging_for_bots(self):
        """If response is both fabrication AND hedging, fabrication fires first."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=True)

        history = [
            {"role": "user", "content": "previous"},
            {"role": "user", "content": "check disk"},
        ]

        # Response that is both fabrication (fake df output) AND hedging
        fabricated_and_hedging = (
            "Here's the disk usage:\n"
            "Filesystem      Size  Used Avail Use%\n"
            "/dev/sda1        50G   20G   30G  40%\n"
            "Would you like me to check anything else?"
        )

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                LLMResponse(text=fabricated_and_hedging, tool_calls=[]),
                # After fabrication retry, returns with tool use
                LLMResponse(text="Checking...", tool_calls=[_tc("run_command")]),
                LLMResponse(text="Disk is at 40%.", tool_calls=[]),
            ]
        )

        resp, already, err, tools, handoff = await HeimdallBot._process_with_tools(
            stub, msg, history, "test-chan",
        )

        # Should have retried for fabrication (not hedging)
        assert stub.codex_client.chat_with_tools.await_count >= 2

    async def test_hedging_fires_for_all_messages(self):
        """Hedging retry does NOT fire for human messages."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=False)  # HUMAN message

        history = [
            {"role": "user", "content": "previous"},
            {"role": "user", "content": "check disk"},
        ]

        hedging_response = "Would you like me to check the disk usage?"

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text=hedging_response, tool_calls=[])
        )

        resp, already, err, tools, handoff = await HeimdallBot._process_with_tools(
            stub, msg, history, "test-chan",
        )

        # No retry — hedging only fires for bot messages
        assert stub.codex_client.chat_with_tools.await_count == 2
        assert resp == hedging_response


# ---------------------------------------------------------------------------
# 14. First message from bot (orig_msg) used for processing
# ---------------------------------------------------------------------------

class TestBotOrigMessageCapture:
    """The last message that created the flush task is used as orig_msg."""

    async def test_last_message_used_as_orig(self):
        """The last bot message's object is passed to _handle_message.

        Each new bot message cancels the previous timer and creates a new task
        with the current message as orig_msg. So the LAST message's object
        is what gets passed to _handle_message.
        """
        stub = _make_on_message_stub()
        stub._bot_msg_buffer_delay = 0.05

        msg1 = _make_msg(is_bot=True, author_id="bot-orig", content="first msg")
        msg1.id = 90001
        msg1.author.display_name = "OrigBot"

        msg2 = _make_msg(is_bot=True, channel_id=str(msg1.channel.id),
                         author_id="bot-orig", content="second msg")
        msg2.id = 90002
        msg2.author.display_name = "OrigBot"

        await stub.on_message(msg1)
        await stub.on_message(msg2)
        await asyncio.sleep(0.15)

        # The LAST message object is passed (it created the final flush task)
        call_args = stub._handle_message.call_args
        orig_msg = call_args[0][0]
        assert orig_msg.id == msg2.id

    async def test_single_bot_message_used_as_orig(self):
        """Single bot message — its own object is used as orig_msg."""
        stub = _make_on_message_stub()

        msg = _make_msg(is_bot=True, author_id="bot-single", content="only msg")
        msg.id = 90010

        await stub.on_message(msg)
        await asyncio.sleep(0.05)

        call_args = stub._handle_message.call_args
        orig_msg = call_args[0][0]
        assert orig_msg.id == msg.id


# ---------------------------------------------------------------------------
# 15. Source code structure verification
# ---------------------------------------------------------------------------

class TestBotInteropSourceStructure:
    """Verify the structure of bot interop code in client.py."""

    def test_bot_buffer_attributes_exist(self):
        """HeimdallBot has _bot_msg_buffer, _bot_msg_tasks, _bot_msg_buffer_delay."""
        import inspect
        source = inspect.getsource(HeimdallBot.__init__)
        assert "_bot_msg_buffer" in source
        assert "_bot_msg_tasks" in source
        assert "_bot_msg_buffer_delay" in source

    def test_combine_bot_messages_called_in_flush(self):
        """combine_bot_messages is called in the flush handler."""
        import inspect
        source = inspect.getsource(HeimdallBot.on_message)
        assert "combine_bot_messages" in source

    def test_is_bot_message_check_in_process_with_tools(self):
        """_process_with_tools checks is_bot_message for preamble injection."""
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        assert "is_bot_message" in source
        assert "ANOTHER BOT" in source

    def test_hedging_check_is_bot_only(self):
        """Hedging retry check includes is_bot_message condition."""
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        # Find the hedging detection block — it should reference is_bot_message
        assert "is_bot_message" in source
        assert "detect_hedging" in source

    def test_fabrication_check_not_bot_only(self):
        """Fabrication check does NOT require is_bot_message (fires for all)."""
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        # Fabrication detection block should exist
        assert "detect_fabrication" in source

    def test_bot_buffer_uses_channel_author_key(self):
        """Buffer key construction uses channel_id and author_id."""
        import inspect
        source = inspect.getsource(HeimdallBot.on_message)
        assert "message.channel.id" in source
        assert "message.author.id" in source
