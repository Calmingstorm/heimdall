"""Tests for respond_to_bots and require_mention config features (Round 3)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.schema import DiscordConfig
from src.discord.client import LokiBot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(**overrides):
    """Minimal LokiBot stub for on_message tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._last_tool_use = {}
    stub._system_prompt = "test system prompt"
    stub._channel_locks = {}
    stub._processed_messages = {}
    stub._processed_messages_max = 100
    stub._background_tasks = {}
    stub._background_tasks_max = 20
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.channels = []
    stub.config.discord.respond_to_bots = False
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
    stub._pending_files = {}
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._memory_path = "/tmp/test_memory.json"

    for k, v in overrides.items():
        setattr(stub, k, v)
    return stub


def _make_message(channel_id="67890", author_id="12345", content="test"):
    """Create a mock Discord message."""
    msg = AsyncMock()
    msg.id = int(time.time() * 1000)
    msg.content = content
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock()
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(), __aexit__=AsyncMock(),
    ))
    msg.channel.guild = MagicMock()  # not a DM
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.bot = False
    msg.author.mention = f"<@{author_id}>"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    msg.delete = AsyncMock()
    msg.attachments = []
    msg.webhook_id = None
    return msg


# ---------------------------------------------------------------------------
# Config schema tests
# ---------------------------------------------------------------------------

class TestDiscordConfigDefaults:
    """Test that DiscordConfig has correct defaults for new fields."""

    def test_respond_to_bots_default_false(self):
        cfg = DiscordConfig(token="test")
        assert cfg.respond_to_bots is False

    def test_require_mention_default_false(self):
        cfg = DiscordConfig(token="test")
        assert cfg.require_mention is False

    def test_respond_to_bots_set_true(self):
        cfg = DiscordConfig(token="test", respond_to_bots=True)
        assert cfg.respond_to_bots is True

    def test_require_mention_set_true(self):
        cfg = DiscordConfig(token="test", require_mention=True)
        assert cfg.require_mention is True


# ---------------------------------------------------------------------------
# respond_to_bots tests
# ---------------------------------------------------------------------------

class TestRespondToBots:
    """Tests for the respond_to_bots config toggle."""

    async def test_ignores_bots_when_disabled(self):
        """With respond_to_bots=False, bot messages are ignored."""
        stub = _make_bot_stub()
        stub.config.discord.respond_to_bots = False
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._handle_message = AsyncMock()
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message()
        msg.author.bot = True
        msg.id = int(time.time() * 1000) + 100

        await stub.on_message(msg)
        stub._handle_message.assert_not_called()

    async def test_processes_bots_when_enabled(self):
        """With respond_to_bots=True, bot messages are buffered then processed."""
        stub = _make_bot_stub()
        stub.config.discord.respond_to_bots = True
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._handle_message = AsyncMock()
        stub.user.mentioned_in = MagicMock(return_value=False)
        stub._bot_msg_buffer = {}
        stub._bot_msg_tasks = {}
        stub._bot_msg_buffer_delay = 0
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="hello from another bot")
        msg.author.bot = True
        msg.id = int(time.time() * 1000) + 101

        await stub.on_message(msg)
        # Buffer flushes asynchronously — give event loop a tick
        await asyncio.sleep(0.1)
        stub._handle_message.assert_awaited_once()

    async def test_never_responds_to_self(self):
        """Even with respond_to_bots=True, bot ignores its own messages."""
        stub = _make_bot_stub()
        stub.config.discord.respond_to_bots = True
        stub._handle_message = AsyncMock()
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message()
        msg.author = stub.user  # same object = self
        msg.id = int(time.time() * 1000) + 102

        await stub.on_message(msg)
        stub._handle_message.assert_not_called()

    async def test_allowed_webhook_always_passes(self):
        """Allowed webhooks bypass bot check regardless of respond_to_bots."""
        import src.discord.client as client_mod
        original = client_mod._ALLOWED_WEBHOOK_IDS
        try:
            client_mod._ALLOWED_WEBHOOK_IDS = {"webhook-123"}

            stub = _make_bot_stub()
            stub.config.discord.respond_to_bots = False
            stub._is_allowed_user = MagicMock(return_value=True)
            stub._is_allowed_channel = MagicMock(return_value=True)
            stub._process_attachments = AsyncMock(return_value=("", []))
            stub._check_for_secrets = MagicMock(return_value=False)
            stub._handle_message = AsyncMock()
            stub.user.mentioned_in = MagicMock(return_value=False)
            stub.on_message = LokiBot.on_message.__get__(stub)

            msg = _make_message(content="webhook message")
            msg.author.bot = True
            msg.webhook_id = "webhook-123"
            msg.id = int(time.time() * 1000) + 103

            await stub.on_message(msg)
            stub._handle_message.assert_awaited_once()
        finally:
            client_mod._ALLOWED_WEBHOOK_IDS = original

    async def test_human_messages_unaffected(self):
        """respond_to_bots=False doesn't affect human messages."""
        stub = _make_bot_stub()
        stub.config.discord.respond_to_bots = False
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._handle_message = AsyncMock()
        stub.user.mentioned_in = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="hello")
        msg.author.bot = False
        msg.id = int(time.time() * 1000) + 104

        await stub.on_message(msg)
        stub._handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# require_mention tests
# ---------------------------------------------------------------------------

class TestRequireMention:
    """Tests for the require_mention config toggle."""

    async def test_ignores_non_mentioned_when_enabled(self):
        """With require_mention=True, messages without @mention are ignored."""
        stub = _make_bot_stub()
        stub.config.discord.require_mention = True
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._handle_message = AsyncMock()
        stub.user.mentioned_in = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="hello there")
        msg.id = int(time.time() * 1000) + 200

        await stub.on_message(msg)
        stub._handle_message.assert_not_called()

    async def test_processes_mentioned_when_enabled(self):
        """With require_mention=True, @mentioned messages are processed."""
        stub = _make_bot_stub()
        stub.config.discord.require_mention = True
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._handle_message = AsyncMock()
        stub.user.mentioned_in = MagicMock(return_value=True)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content=f"<@111> check disk")
        msg.id = int(time.time() * 1000) + 201

        await stub.on_message(msg)
        stub._handle_message.assert_awaited_once()

    async def test_all_messages_processed_when_disabled(self):
        """With require_mention=False (default), all messages are processed."""
        stub = _make_bot_stub()
        stub.config.discord.require_mention = False
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._handle_message = AsyncMock()
        stub.user.mentioned_in = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="hello")
        msg.id = int(time.time() * 1000) + 202

        await stub.on_message(msg)
        stub._handle_message.assert_awaited_once()

    async def test_dm_bypasses_require_mention(self):
        """DMs should always be processed even with require_mention=True."""
        stub = _make_bot_stub()
        stub.config.discord.require_mention = True
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._handle_message = AsyncMock()
        stub.user.mentioned_in = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="hello in DM")
        msg.channel.guild = None  # DM channel has no guild
        msg.id = int(time.time() * 1000) + 203

        await stub.on_message(msg)
        stub._handle_message.assert_awaited_once()

    async def test_require_mention_with_respond_to_bots(self):
        """Both flags together: bot @mentions the bot, should be buffered then processed."""
        stub = _make_bot_stub()
        stub.config.discord.respond_to_bots = True
        stub.config.discord.require_mention = True
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._handle_message = AsyncMock()
        stub.user.mentioned_in = MagicMock(return_value=True)
        stub._bot_msg_buffer = {}
        stub._bot_msg_tasks = {}
        stub._bot_msg_buffer_delay = 0
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content=f"<@111> hello from bot")
        msg.author.bot = True
        msg.id = int(time.time() * 1000) + 204

        await stub.on_message(msg)
        await asyncio.sleep(0.1)
        stub._handle_message.assert_awaited_once()

    async def test_require_mention_bot_no_mention_ignored(self):
        """Both flags: bot message without @mention should be ignored."""
        stub = _make_bot_stub()
        stub.config.discord.respond_to_bots = True
        stub.config.discord.require_mention = True
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._handle_message = AsyncMock()
        stub.user.mentioned_in = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="hello from bot")
        msg.author.bot = True
        msg.id = int(time.time() * 1000) + 205

        await stub.on_message(msg)
        stub._handle_message.assert_not_called()


class TestNoApprovalConfig:
    """Approval system has been removed — verify config doesn't have approval fields."""

    def test_no_auto_approve_field(self):
        from src.config.schema import ToolsConfig
        cfg = ToolsConfig()
        assert not hasattr(cfg, "auto_approve")
