"""Tests targeting remaining coverage gaps in src/discord/client.py.

Covers:
- __init__ with various config combinations (lines 100-259)
- on_ready lifecycle (lines 538-555)
- _backfill_archives (lines 559-572)
- on_voice_state_update (lines 581-598)
- _on_voice_transcription (lines 840-858)
- Tool dispatch via _process_with_tools (lines 1286-1346)
- Streaming HTTPException edge cases (lines 1434-1444, 1491-1513)
- Miscellaneous gaps: dedup overflow, attachment merge, parse_time error,
  background task pruning/crash, list_tasks truncation, workflow post failure,
  post_file edge cases, thread no-summary
"""
from __future__ import annotations

import asyncio
import collections
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402
import discord  # noqa: E402

from src.config.schema import (  # noqa: E402
    Config, DiscordConfig, SessionsConfig, UsageConfig,
    ContextConfig, SearchConfig, VoiceConfig, BrowserConfig, OpenAICodexConfig,
    MonitoringConfig, MonitorCheck, ToolsConfig, PermissionsConfig, LearningConfig,
    WebhookConfig,
)
from src.discord.client import (  # noqa: E402
    LokiBot,
    DISCORD_MAX_LEN,
    MAX_TOOL_ITERATIONS,
    scrub_response_secrets,
    truncate_tool_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config(tmp_path, **overrides):
    """Create a Config with sensible test defaults."""
    defaults = dict(
        discord=DiscordConfig(token="test-token", allowed_users=["12345"]),
        sessions=SessionsConfig(persist_directory=str(tmp_path / "sessions")),
        usage=UsageConfig(directory=str(tmp_path / "usage")),
        context=ContextConfig(directory=str(tmp_path / "context")),
        search=SearchConfig(enabled=False, search_db_path=str(tmp_path / "search")),
        voice=VoiceConfig(enabled=False),
        browser=BrowserConfig(enabled=False),
        openai_codex=OpenAICodexConfig(enabled=False),
        monitoring=MonitoringConfig(enabled=False),
        learning=LearningConfig(enabled=True),
        permissions=PermissionsConfig(
            tiers={"12345": "admin"},
            overrides_path=str(tmp_path / "perm_overrides.json"),
        ),
        webhook=WebhookConfig(enabled=False),
    )
    defaults.update(overrides)
    return Config(**defaults)


# Names of service classes imported at the top of src/discord/client.py
_CORE_PATCHES = [
    "ContextLoader",
    "ConversationReflector",
    "SessionManager",
    "ToolExecutor",
    "SkillManager",
    "Scheduler",
    "AuditLogger",
    "PermissionManager",
    "ToolMemory",
    "LocalEmbedder",
    "SessionVectorStore",
    "KnowledgeStore",
    "VoiceManager",
    "InfraWatcher",
    "CodexAuth",
    "CodexAuthPool",
    "CodexChatClient",
]


def _construct_bot(config, mock_overrides=None):
    """Construct LokiBot with all external deps mocked.

    Returns (bot, mocks_dict) where mocks_dict maps class name to its mock.
    """
    with ExitStack() as stack:
        # Patch discord.Client.__init__ to avoid actual Discord setup
        stack.enter_context(
            patch.object(discord.Client, "__init__", lambda self, **kw: None)
        )
        # Patch CommandTree to avoid real tree setup
        stack.enter_context(patch("src.discord.client.app_commands.CommandTree"))
        # Patch build_system_prompt to return a simple string
        stack.enter_context(
            patch("src.discord.client.build_system_prompt", return_value="system prompt")
        )

        mocks = {}
        for name in _CORE_PATCHES:
            m = stack.enter_context(patch(f"src.discord.client.{name}"))
            mocks[name] = m

        # Also patch locally-imported FullTextIndex (imported inside __init__)
        m_fts = stack.enter_context(patch("src.search.fts.FullTextIndex"))
        mocks["FullTextIndex"] = m_fts

        # Also patch locally-imported BrowserManager
        m_bm = stack.enter_context(patch("src.tools.browser.BrowserManager"))
        mocks["BrowserManager"] = m_bm

        # Apply overrides (e.g., set return_value.available = True)
        if mock_overrides:
            for name, attrs in mock_overrides.items():
                if name in mocks:
                    for attr, val in attrs.items():
                        # Set on the return_value (instance) of the mock class
                        setattr(mocks[name].return_value, attr, val)

        bot = LokiBot(config)
        return bot, mocks


def _make_bot_stub(**overrides):
    """Create a minimal LokiBot stub for method-level tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test system prompt"
    stub._channel_locks = {}
    stub._processed_messages = collections.OrderedDict()
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
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock(return_value=0)
    stub.sessions.save = MagicMock()
    stub.sessions.reset = MagicMock()
    stub.sessions.search_history = AsyncMock(return_value=[])
    stub.sessions.get_or_create = MagicMock()
    stub.sessions._sessions = {}
    stub.sessions.persist_dir = Path("/tmp/test_sessions")
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="response")
    stub.codex_client.chat_with_tools = AsyncMock(
        return_value=MagicMock(text="Done", tool_calls=[], is_tool_use=False, stop_reason="end_turn")
    )
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])
    stub.skill_manager.create_skill = MagicMock(return_value="Skill created")
    stub.skill_manager.edit_skill = MagicMock(return_value="Skill updated")
    stub.skill_manager.delete_skill = MagicMock(return_value="Skill deleted")
    stub.skill_manager.execute = AsyncMock(return_value="Skill executed")
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.audit.search = AsyncMock(return_value=[])
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[{"name": "test_tool"}])
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._pending_files = {}
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.permissions.is_admin = MagicMock(return_value=True)
    stub.voice_manager = None
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub.tool_executor._resolve_host = MagicMock(
        return_value=("10.0.0.1", "root", "linux")
    )
    stub.tool_executor._load_memory_for_user = MagicMock(return_value={})
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.browser_manager = None
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")
    stub._knowledge_store = None
    stub._embedder = None
    stub._fts_index = None
    stub._vector_store = None
    stub._memory_path = "/tmp/test_memory.json"
    stub.scheduler = MagicMock()
    stub.scheduler.add = MagicMock(return_value={
        "id": "sched-1", "description": "Test", "next_run": "2026-03-19T08:00:00",
    })
    stub.scheduler.list_all = MagicMock(return_value=[])
    stub.scheduler.delete = MagicMock(return_value=True)
    stub.infra_watcher = None
    stub.context_loader = MagicMock()
    stub.tree = MagicMock()
    stub.user = MagicMock()
    stub.user.id = 111
    stub.guilds = []
    stub.get_channel = MagicMock(return_value=None)

    # Handler methods used in tool dispatch (async ones must be AsyncMock)
    stub._handle_purge = AsyncMock(return_value="Purged messages")
    stub._handle_browser_screenshot = AsyncMock(return_value="Screenshot taken")
    stub._handle_generate_file = AsyncMock(return_value="File generated")
    stub._handle_post_file = AsyncMock(return_value="File posted")
    stub._handle_schedule_task = MagicMock(return_value="Task scheduled")
    stub._handle_list_schedules = MagicMock(return_value="No schedules")
    stub._handle_delete_schedule = MagicMock(return_value="Schedule deleted")
    stub._handle_parse_time = MagicMock(return_value="Parsed: 2026-03-19")
    stub._handle_search_history = AsyncMock(return_value="Search results")
    stub._handle_delegate_task = AsyncMock(return_value="Task delegated")
    stub._handle_list_tasks = MagicMock(return_value="No tasks")
    stub._handle_cancel_task = MagicMock(return_value="Task cancelled")
    stub._handle_search_knowledge = AsyncMock(return_value="Knowledge results")
    stub._handle_ingest_document = AsyncMock(return_value="Document ingested")
    stub._handle_list_knowledge = MagicMock(return_value="No knowledge")
    stub._handle_delete_knowledge = MagicMock(return_value="Knowledge deleted")
    stub._handle_set_permission = MagicMock(return_value="Permission set")
    stub._handle_search_audit = AsyncMock(return_value="Audit results")
    stub._handle_create_digest = MagicMock(return_value="Digest created")
    stub._handle_message = AsyncMock()
    stub._is_allowed_user = MagicMock(return_value=True)
    stub._track_recent_action = MagicMock()

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
    msg.channel.purge = AsyncMock(return_value=[1, 2, 3])
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(), __aexit__=AsyncMock(),
    ))
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.bot = False
    msg.author.mention = f"<@{author_id}>"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    msg.delete = AsyncMock()
    msg.edit = AsyncMock()
    msg.attachments = []
    return msg


def _make_codex_mock(tool_name=None, tool_input=None, text="Done"):
    """Create a mock codex_client.chat_with_tools that returns a tool call first, then text.

    If tool_name is None, returns text on the first call.
    """
    from src.llm.types import LLMResponse, ToolCall

    call_count = [0]

    async def fake_chat_with_tools(messages, system, tools):
        call_count[0] += 1
        if tool_name and call_count[0] == 1:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id=f"tool_{tool_name}", name=tool_name, input=tool_input or {})],
                stop_reason="tool_use",
            )
        else:
            return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn")

    return fake_chat_with_tools


# ---------------------------------------------------------------------------
# __init__ tests (lines 100-259)
# ---------------------------------------------------------------------------

class TestLokiBotInit:
    """Test bot constructor with various config combinations."""

    def test_init_minimal_config(self, tmp_path):
        """Construct bot with minimal config — no optional services."""
        config = _minimal_config(tmp_path)
        bot, mocks = _construct_bot(config)

        assert bot.config == config
        mocks["ContextLoader"].assert_called_once()
        mocks["ConversationReflector"].assert_called_once()
        mocks["SessionManager"].assert_called_once()
        mocks["ToolExecutor"].assert_called_once()
        mocks["SkillManager"].assert_called_once()
        mocks["Scheduler"].assert_called_once()
        mocks["AuditLogger"].assert_called_once()
        mocks["PermissionManager"].assert_called_once()
        mocks["ToolMemory"].assert_called_once()
        # Optional services should not be created
        assert bot.voice_manager is None
        assert bot.browser_manager is None
        assert bot.codex_client is None
        assert bot.infra_watcher is None
        assert bot._vector_store is None
        assert bot._embedder is None
        assert bot._knowledge_store is None
        assert bot._fts_index is None

    def test_init_search_enabled(self, tmp_path):
        """Construct bot with search enabled — embedder, vector store, knowledge store, FTS."""
        config = _minimal_config(
            tmp_path,
            search=SearchConfig(
                enabled=True,
                search_db_path=str(tmp_path / "search"),
            ),
        )
        overrides = {
            "SessionVectorStore": {"available": True},
            "KnowledgeStore": {"available": True},
            "FullTextIndex": {"available": True},
        }
        bot, mocks = _construct_bot(config, mock_overrides=overrides)

        mocks["LocalEmbedder"].assert_called_once()
        assert bot._embedder is not None
        assert bot._vector_store is not None
        assert bot._knowledge_store is not None
        assert bot._fts_index is not None

    def test_init_search_stores_unavailable(self, tmp_path):
        """When vector store or knowledge store is unavailable, they're set to None."""
        config = _minimal_config(
            tmp_path,
            search=SearchConfig(
                enabled=True,
                search_db_path=str(tmp_path / "search"),
            ),
        )
        overrides = {
            "SessionVectorStore": {"available": False},
            "KnowledgeStore": {"available": False},
            "FullTextIndex": {"available": False},
        }
        bot, mocks = _construct_bot(config, mock_overrides=overrides)

        assert bot._vector_store is None
        assert bot._knowledge_store is None
        assert bot._fts_index is None

    def test_init_browser_enabled(self, tmp_path):
        """Construct bot with browser automation enabled."""
        config = _minimal_config(
            tmp_path,
            browser=BrowserConfig(
                enabled=True,
                cdp_url="ws://localhost:3000",
            ),
        )
        bot, mocks = _construct_bot(config)

        assert bot.browser_manager is not None
        mocks["BrowserManager"].assert_called_once()

    def test_init_codex_enabled_configured(self, tmp_path):
        """Construct bot with Codex enabled and credentials available."""
        config = _minimal_config(
            tmp_path,
            openai_codex=OpenAICodexConfig(
                enabled=True,
                credentials_path=str(tmp_path / "codex_auth.json"),
            ),
        )
        overrides = {"CodexAuthPool": {"is_configured": MagicMock(return_value=True)}}
        bot, mocks = _construct_bot(config, mock_overrides=overrides)

        assert bot.codex_client is not None
        mocks["CodexAuthPool"].assert_called_once()
        mocks["CodexChatClient"].assert_called_once()

    def test_init_codex_enabled_not_configured(self, tmp_path):
        """Construct bot with Codex enabled but no credentials."""
        config = _minimal_config(
            tmp_path,
            openai_codex=OpenAICodexConfig(
                enabled=True,
                credentials_path=str(tmp_path / "codex_auth.json"),
            ),
        )
        overrides = {"CodexAuthPool": {"is_configured": MagicMock(return_value=False)}}
        bot, mocks = _construct_bot(config, mock_overrides=overrides)

        assert bot.codex_client is None

    def test_init_voice_enabled(self, tmp_path):
        """Construct bot with voice support enabled."""
        config = _minimal_config(
            tmp_path,
            voice=VoiceConfig(enabled=True, transcript_channel_id="99999"),
        )
        bot, mocks = _construct_bot(config)

        assert bot.voice_manager is not None
        mocks["VoiceManager"].assert_called_once()

    def test_init_monitoring_enabled(self, tmp_path):
        """Construct bot with proactive monitoring enabled."""
        config = _minimal_config(
            tmp_path,
            monitoring=MonitoringConfig(
                enabled=True,
                checks=[MonitorCheck(name="disk", type="disk", hosts=["server"])],
                alert_channel_id="67890",
            ),
        )
        bot, mocks = _construct_bot(config)

        assert bot.infra_watcher is not None
        mocks["InfraWatcher"].assert_called_once()


# ---------------------------------------------------------------------------
# on_ready tests (lines 538-555)
# ---------------------------------------------------------------------------

class TestOnReady:
    """Test on_ready lifecycle."""

    async def test_on_ready_syncs_guilds(self):
        """on_ready should sync slash commands to each guild."""
        stub = _make_bot_stub()
        stub.on_ready = LokiBot.on_ready.__get__(stub)

        guild1 = MagicMock()
        guild1.name = "TestGuild"
        stub.guilds = [guild1]
        stub.tree.sync = AsyncMock()
        stub._vector_store = None
        stub._embedder = None

        await stub.on_ready()

        stub.sessions.prune.assert_called_once()
        stub.tree.copy_global_to.assert_called_once_with(guild=guild1)
        stub.tree.sync.assert_called_once_with(guild=guild1)
        stub.scheduler.start.assert_called_once()

    async def test_on_ready_prunes_stale_sessions(self):
        """on_ready should prune stale sessions and log if any were pruned."""
        stub = _make_bot_stub()
        stub.on_ready = LokiBot.on_ready.__get__(stub)
        stub.sessions.prune = MagicMock(return_value=5)
        stub.guilds = []
        stub._vector_store = None
        stub._embedder = None

        await stub.on_ready()

        stub.sessions.prune.assert_called_once()
        stub.scheduler.start.assert_called_once()

    async def test_on_ready_starts_backfill_when_search_enabled(self):
        """on_ready should start backfill task when vector store and embedder are set."""
        stub = _make_bot_stub()
        stub.on_ready = LokiBot.on_ready.__get__(stub)
        stub.guilds = []
        stub._vector_store = MagicMock()
        stub._embedder = MagicMock()

        calls = []
        def _close_and_track(coro):
            calls.append(coro)
            coro.close()
            return MagicMock()

        with patch("asyncio.create_task", side_effect=_close_and_track):
            await stub.on_ready()
            assert len(calls) == 1

    async def test_on_ready_starts_infra_watcher(self):
        """on_ready should start infrastructure watcher if configured."""
        stub = _make_bot_stub()
        stub.on_ready = LokiBot.on_ready.__get__(stub)
        stub.guilds = []
        stub._vector_store = None
        stub._embedder = None
        stub.infra_watcher = MagicMock()

        await stub.on_ready()

        stub.infra_watcher.start.assert_called_once()


# ---------------------------------------------------------------------------
# _backfill_archives tests (lines 559-572)
# ---------------------------------------------------------------------------

class TestBackfillArchives:
    """Test _backfill_archives."""

    async def test_backfill_with_counts(self):
        """Should log when sessions are backfilled."""
        stub = _make_bot_stub()
        stub._backfill_archives = LokiBot._backfill_archives.__get__(stub)
        stub._vector_store = AsyncMock()
        stub._vector_store.backfill = AsyncMock(return_value=5)
        stub._embedder = MagicMock()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.backfill_fts = MagicMock(return_value=3)
        stub._fts_index = MagicMock()

        await stub._backfill_archives()

        stub._vector_store.backfill.assert_called_once()

    async def test_backfill_zero_count(self):
        """Should handle zero backfill count (vector store up to date)."""
        stub = _make_bot_stub()
        stub._backfill_archives = LokiBot._backfill_archives.__get__(stub)
        stub._vector_store = AsyncMock()
        stub._vector_store.backfill = AsyncMock(return_value=0)
        stub._embedder = MagicMock()
        stub._knowledge_store = None
        stub._fts_index = None

        await stub._backfill_archives()

        stub._vector_store.backfill.assert_called_once()

    async def test_backfill_exception(self):
        """Should catch and log exceptions during backfill."""
        stub = _make_bot_stub()
        stub._backfill_archives = LokiBot._backfill_archives.__get__(stub)
        stub._vector_store = AsyncMock()
        stub._vector_store.backfill = AsyncMock(side_effect=RuntimeError("store down"))
        stub._embedder = MagicMock()

        # Should not raise
        await stub._backfill_archives()


# ---------------------------------------------------------------------------
# on_voice_state_update tests (lines 581-598)
# ---------------------------------------------------------------------------

class TestOnVoiceStateUpdate:
    """Test on_voice_state_update auto-join/leave logic."""

    async def test_no_voice_manager(self):
        """Should return immediately if voice_manager is None."""
        stub = _make_bot_stub()
        stub.on_voice_state_update = LokiBot.on_voice_state_update.__get__(stub)
        stub.voice_manager = None
        stub.config.voice.auto_join = True

        member = MagicMock()
        before = MagicMock()
        after = MagicMock()

        await stub.on_voice_state_update(member, before, after)
        # No crash, no action

    async def test_auto_join_disabled(self):
        """Should return immediately if auto_join is False."""
        stub = _make_bot_stub()
        stub.on_voice_state_update = LokiBot.on_voice_state_update.__get__(stub)
        stub.voice_manager = MagicMock()
        stub.config.voice.auto_join = False

        member = MagicMock()
        before = MagicMock()
        after = MagicMock()

        await stub.on_voice_state_update(member, before, after)

    async def test_ignores_bot_users(self):
        """Should ignore bot users."""
        stub = _make_bot_stub()
        stub.on_voice_state_update = LokiBot.on_voice_state_update.__get__(stub)
        stub.voice_manager = MagicMock()
        stub.config.voice.auto_join = True

        member = MagicMock()
        member.bot = True

        await stub.on_voice_state_update(member, MagicMock(), MagicMock())

    async def test_ignores_disallowed_users(self):
        """Should ignore users not in allowed_users."""
        stub = _make_bot_stub()
        stub.on_voice_state_update = LokiBot.on_voice_state_update.__get__(stub)
        stub.voice_manager = MagicMock()
        stub.config.voice.auto_join = True
        stub._is_allowed_user = MagicMock(return_value=False)

        member = MagicMock()
        member.bot = False

        await stub.on_voice_state_update(member, MagicMock(), MagicMock())

    async def test_auto_join_user_joins_channel(self):
        """Should auto-join when an allowed user joins a voice channel."""
        stub = _make_bot_stub()
        stub.on_voice_state_update = LokiBot.on_voice_state_update.__get__(stub)
        stub.voice_manager = MagicMock()
        stub.voice_manager.is_connected = False
        stub.voice_manager.join_channel = AsyncMock()
        stub.config.voice.auto_join = True

        member = MagicMock()
        member.bot = False

        before = MagicMock()
        before.channel = None  # Was not in a channel
        after = MagicMock()
        after.channel = MagicMock()
        after.channel.name = "General"

        await stub.on_voice_state_update(member, before, after)

        stub.voice_manager.join_channel.assert_called_once_with(after.channel)

    async def test_auto_leave_empty_channel(self):
        """Should auto-leave when all humans leave the voice channel."""
        stub = _make_bot_stub()
        stub.on_voice_state_update = LokiBot.on_voice_state_update.__get__(stub)
        stub.voice_manager = MagicMock()
        stub.voice_manager.is_connected = True
        stub.voice_manager.leave_channel = AsyncMock()
        stub.config.voice.auto_join = True

        member = MagicMock()
        member.bot = False

        # Bot is the only one left in the channel
        bot_member = MagicMock()
        bot_member.bot = True
        before = MagicMock()
        before.channel = MagicMock()
        before.channel.members = [bot_member]
        stub.voice_manager.current_channel = before.channel
        after = MagicMock()
        after.channel = None  # User left

        await stub.on_voice_state_update(member, before, after)

        stub.voice_manager.leave_channel.assert_called_once()

    async def test_no_auto_leave_when_humans_remain(self):
        """Should not leave when other humans are still in the channel."""
        stub = _make_bot_stub()
        stub.on_voice_state_update = LokiBot.on_voice_state_update.__get__(stub)
        stub.voice_manager = MagicMock()
        stub.voice_manager.is_connected = True
        stub.voice_manager.leave_channel = AsyncMock()
        stub.config.voice.auto_join = True

        member = MagicMock()
        member.bot = False

        # Another human is still in the channel
        other_human = MagicMock()
        other_human.bot = False
        before = MagicMock()
        before.channel = MagicMock()
        before.channel.members = [other_human]
        stub.voice_manager.current_channel = before.channel
        after = MagicMock()
        after.channel = None

        await stub.on_voice_state_update(member, before, after)

        stub.voice_manager.leave_channel.assert_not_called()


# ---------------------------------------------------------------------------
# _on_voice_transcription tests (lines 840-858)
# ---------------------------------------------------------------------------

class TestOnVoiceTranscription:
    """Test _on_voice_transcription handler."""

    async def test_routes_transcription_to_message_pipeline(self):
        """Should post transcription and route through _handle_message."""
        stub = _make_bot_stub()
        stub._on_voice_transcription = LokiBot._on_voice_transcription.__get__(stub)
        stub.voice_manager = MagicMock()
        stub.voice_manager.speak = AsyncMock()

        member = MagicMock()
        member.display_name = "TestUser"
        member.guild = MagicMock()

        channel = AsyncMock()
        channel.send = AsyncMock()

        await stub._on_voice_transcription("hello world", member, channel)

        # Should post the transcription to the channel
        channel.send.assert_called_once()
        assert "TestUser" in channel.send.call_args[0][0]
        assert "hello world" in channel.send.call_args[0][0]

        # Should call _handle_message with a proxy message
        stub._handle_message.assert_called_once()
        call_args = stub._handle_message.call_args
        assert call_args[0][1] == "hello world"  # text argument
        # Should include a voice_callback
        assert call_args[1].get("voice_callback") is not None

    async def test_voice_callback_calls_speak(self):
        """The voice_callback should call voice_manager.speak."""
        stub = _make_bot_stub()
        stub._on_voice_transcription = LokiBot._on_voice_transcription.__get__(stub)
        stub.voice_manager = MagicMock()
        stub.voice_manager.speak = AsyncMock()

        member = MagicMock()
        member.display_name = "TestUser"
        member.guild = MagicMock()
        channel = AsyncMock()
        channel.send = AsyncMock()

        await stub._on_voice_transcription("test", member, channel)

        # Extract the voice_callback that was passed to _handle_message
        call_kwargs = stub._handle_message.call_args[1]
        voice_callback = call_kwargs["voice_callback"]

        # Call it and verify it speaks
        await voice_callback("response text")
        stub.voice_manager.speak.assert_called_once_with("response text")


# ---------------------------------------------------------------------------
# Tool dispatch tests (lines 1286-1346)
# ---------------------------------------------------------------------------

class TestToolDispatchBranches:
    """Test that _process_with_tools dispatches to correct handler for each tool name."""

    async def _run_dispatch(self, stub, tool_name, tool_input=None):
        """Helper: run _process_with_tools with a single tool call."""
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)
        stub.codex_client.chat_with_tools = _make_codex_mock(
            tool_name=tool_name,
            tool_input=tool_input or {},
        )
        msg = _make_message()
        return await stub._process_with_tools(
            msg, [{"role": "user", "content": "do something"}],
        )

    @pytest.mark.parametrize("tool_name,handler_attr", [
        ("purge_messages", "_handle_purge"),
        ("browser_screenshot", "_handle_browser_screenshot"),
        ("generate_file", "_handle_generate_file"),
        ("post_file", "_handle_post_file"),
    ])
    async def test_async_dispatch_msg_and_input(self, tool_name, handler_attr):
        """Tool dispatch for async handlers that take (message, tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, tool_name, {"key": "value"})
        getattr(stub, handler_attr).assert_called_once()

    async def test_dispatch_schedule_task(self):
        """schedule_task dispatch calls _handle_schedule_task(message, tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "schedule_task", {"description": "test"})
        stub._handle_schedule_task.assert_called_once()

    async def test_dispatch_list_schedules(self):
        """list_schedules dispatch calls _handle_list_schedules()."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "list_schedules")
        stub._handle_list_schedules.assert_called_once()

    async def test_dispatch_delete_schedule(self):
        """delete_schedule dispatch calls _handle_delete_schedule(tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "delete_schedule", {"id": "sched-1"})
        stub._handle_delete_schedule.assert_called_once()

    async def test_dispatch_parse_time(self):
        """parse_time dispatch calls _handle_parse_time(tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "parse_time", {"expression": "tomorrow"})
        stub._handle_parse_time.assert_called_once()

    async def test_dispatch_search_history(self):
        """search_history dispatch calls _handle_search_history(tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "search_history", {"query": "test"})
        stub._handle_search_history.assert_called_once()

    async def test_dispatch_delegate_task(self):
        """delegate_task dispatch calls _handle_delegate_task(message, tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "delegate_task", {"description": "test"})
        stub._handle_delegate_task.assert_called_once()

    async def test_dispatch_list_tasks(self):
        """list_tasks dispatch calls _handle_list_tasks(tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "list_tasks")
        stub._handle_list_tasks.assert_called_once()

    async def test_dispatch_cancel_task(self):
        """cancel_task dispatch calls _handle_cancel_task(tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "cancel_task", {"task_id": "t1"})
        stub._handle_cancel_task.assert_called_once()

    async def test_dispatch_search_knowledge(self):
        """search_knowledge dispatch calls _handle_search_knowledge(tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "search_knowledge", {"query": "test"})
        stub._handle_search_knowledge.assert_called_once()

    async def test_dispatch_ingest_document(self):
        """ingest_document dispatch calls _handle_ingest_document(tool_input, author)."""
        stub = _make_bot_stub()
        await self._run_dispatch(
            stub, "ingest_document", {"title": "doc", "content": "text"},
        )
        stub._handle_ingest_document.assert_called_once()

    async def test_dispatch_list_knowledge(self):
        """list_knowledge dispatch calls _handle_list_knowledge()."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "list_knowledge")
        stub._handle_list_knowledge.assert_called_once()

    async def test_dispatch_delete_knowledge(self):
        """delete_knowledge dispatch calls _handle_delete_knowledge(tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "delete_knowledge", {"id": "doc1"})
        stub._handle_delete_knowledge.assert_called_once()

    async def test_dispatch_set_permission(self):
        """set_permission dispatch calls _handle_set_permission(user_id, tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(
            stub, "set_permission", {"user_id": "12345", "tier": "admin"},
        )
        stub._handle_set_permission.assert_called_once()

    async def test_dispatch_search_audit(self):
        """search_audit dispatch calls _handle_search_audit(tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "search_audit", {"query": "test"})
        stub._handle_search_audit.assert_called_once()

    async def test_dispatch_create_digest(self):
        """create_digest dispatch calls _handle_create_digest(message, tool_input)."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "create_digest", {"name": "test"})
        stub._handle_create_digest.assert_called_once()

    async def test_dispatch_create_skill(self):
        """create_skill dispatch calls skill_manager.create_skill and rebuilds prompt."""
        stub = _make_bot_stub()
        await self._run_dispatch(
            stub, "create_skill", {"name": "my_skill", "code": "def execute(): pass"},
        )
        stub.skill_manager.create_skill.assert_called_once_with(
            "my_skill", "def execute(): pass",
        )
        stub._build_system_prompt.assert_called()

    async def test_dispatch_edit_skill(self):
        """edit_skill dispatch calls skill_manager.edit_skill and rebuilds prompt."""
        stub = _make_bot_stub()
        await self._run_dispatch(
            stub, "edit_skill", {"name": "my_skill", "code": "new code"},
        )
        stub.skill_manager.edit_skill.assert_called_once_with("my_skill", "new code")
        stub._build_system_prompt.assert_called()

    async def test_dispatch_delete_skill(self):
        """delete_skill dispatch calls skill_manager.delete_skill and rebuilds prompt."""
        stub = _make_bot_stub()
        await self._run_dispatch(stub, "delete_skill", {"name": "my_skill"})
        stub.skill_manager.delete_skill.assert_called_once_with("my_skill")
        stub._build_system_prompt.assert_called()

    async def test_dispatch_list_skills_empty(self):
        """list_skills with no skills returns 'No user-created skills.'"""
        stub = _make_bot_stub()
        stub.skill_manager.list_skills = MagicMock(return_value=[])
        text, _, _, _, _ = await self._run_dispatch(stub, "list_skills")
        # The handler returns "No user-created skills."
        # which becomes part of the tool result
        stub.skill_manager.list_skills.assert_called()

    async def test_dispatch_list_skills_with_skills(self):
        """list_skills with skills formats a list."""
        stub = _make_bot_stub()
        stub.skill_manager.list_skills = MagicMock(return_value=[
            {"name": "skill1", "description": "First skill"},
            {"name": "skill2", "description": "Dangerous skill"},
        ])
        await self._run_dispatch(stub, "list_skills")
        stub.skill_manager.list_skills.assert_called()

    async def test_dispatch_custom_skill(self):
        """Custom skill dispatch calls skill_manager.execute when has_skill is True."""
        stub = _make_bot_stub()
        stub.skill_manager.has_skill = MagicMock(return_value=True)
        stub.skill_manager.execute = AsyncMock(return_value="Custom result")
        await self._run_dispatch(
            stub, "my_custom_skill", {"param": "value"},
        )
        stub.skill_manager.execute.assert_called_once()

    async def test_dispatch_unknown_tool_to_executor(self):
        """Unknown tools fall through to tool_executor.execute."""
        stub = _make_bot_stub()
        stub.skill_manager.has_skill = MagicMock(return_value=False)
        await self._run_dispatch(stub, "check_disk", {"host": "server"})
        stub.tool_executor.execute.assert_called_once_with("check_disk", {"host": "server"}, user_id="12345")


# ---------------------------------------------------------------------------
# Streaming edge cases (lines 1434-1444, 1491-1513)
# ---------------------------------------------------------------------------
# (TestStreamHTTPExceptions removed — _stream_iteration no longer exists)


# ---------------------------------------------------------------------------
# Miscellaneous coverage gaps
# ---------------------------------------------------------------------------

class TestMiscCoverageGaps:
    """Test various scattered uncovered lines."""

    async def test_dedup_overflow_popitem(self):
        """Should pop oldest entries when processed_messages exceeds max (line 620)."""
        stub = _make_bot_stub()
        stub.on_message = LokiBot.on_message.__get__(stub)
        stub._processed_messages = collections.OrderedDict()
        stub._processed_messages_max = 2

        # Pre-fill with 2 entries
        stub._processed_messages[1001] = None
        stub._processed_messages[1002] = None

        # Process a new message — should evict 1001
        msg = _make_message()
        msg.id = 1003
        msg.author.bot = False
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub.user = MagicMock()
        stub.user.id = 111
        stub.user.mentioned_in = MagicMock(return_value=False)

        await stub.on_message(msg)

        assert 1001 not in stub._processed_messages
        assert 1003 in stub._processed_messages

    async def test_attachment_text_merged_with_content(self):
        """Should merge attachment text with message content (line 631)."""
        stub = _make_bot_stub()
        stub.on_message = LokiBot.on_message.__get__(stub)
        stub._processed_messages = collections.OrderedDict()
        stub._processed_messages_max = 100
        stub._process_attachments = AsyncMock(
            return_value=("**Attached file: test.py**\n```\ncode\n```", []),
        )
        stub._check_for_secrets = MagicMock(return_value=False)
        stub.user = MagicMock()
        stub.user.id = 111
        stub.user.mentioned_in = MagicMock(return_value=False)
        stub._is_allowed_channel = MagicMock(return_value=True)

        msg = _make_message(content="Check this file")
        msg.id = 2001
        msg.author.bot = False

        await stub.on_message(msg)

        # _handle_message should be called with merged content
        call_args = stub._handle_message.call_args
        content = call_args[0][1]
        assert "Check this file" in content
        assert "Attached file: test.py" in content

    async def test_voice_callback_set_when_connected(self):
        """Should set vc_callback when voice_manager is connected (lines 683-684)."""
        stub = _make_bot_stub()
        stub.on_message = LokiBot.on_message.__get__(stub)
        stub._processed_messages = collections.OrderedDict()
        stub._processed_messages_max = 100
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub.user = MagicMock()
        stub.user.id = 111
        stub.user.mentioned_in = MagicMock(return_value=False)
        stub._is_allowed_channel = MagicMock(return_value=True)

        # Set up voice manager as connected
        stub.voice_manager = MagicMock()
        stub.voice_manager.is_connected = True
        stub.voice_manager.speak = AsyncMock()

        msg = _make_message(content="hello")
        msg.id = 3001
        msg.author.bot = False

        await stub.on_message(msg)

        # _handle_message should be called with a voice_callback
        call_kwargs = stub._handle_message.call_args[1]
        vc_callback = call_kwargs.get("voice_callback")
        assert vc_callback is not None

        # Call the callback to cover the closure body (line 684)
        await vc_callback("test response")
        stub.voice_manager.speak.assert_called_once_with("test response")

    async def test_attachment_read_error(self):
        """Should handle attachment read errors gracefully (lines 771-772)."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)

        att = MagicMock()
        att.content_type = "text/plain"
        att.filename = "test.txt"
        att.size = 100
        att.read = AsyncMock(side_effect=Exception("download failed"))

        msg = _make_message()
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)

        assert "failed to read" in text

    async def test_thread_no_summary_inherits_context(self):
        """Thread with no existing summary gets parent context (line 888)."""
        stub = _make_bot_stub()
        stub._handle_message = LokiBot._handle_message.__get__(stub)
        stub._handle_message_inner = AsyncMock()

        thread = MagicMock()
        thread.id = 55555
        thread.parent = MagicMock()
        thread.parent.id = 67890

        # Thread session with no messages
        thread_session = MagicMock()
        thread_session.messages = []
        thread_session.summary = None

        # Parent session with messages and no summary (triggers else branch)
        parent_msg = MagicMock()
        parent_msg.role = "user"
        parent_msg.content = "Hello from parent"
        parent_session = MagicMock()
        parent_session.messages = [parent_msg]
        parent_session.summary = None  # So `parent_session.summary or ""` → ""

        # get_or_create returns thread_session for thread channel, parent_session for parent
        def _get_or_create(channel_id):
            if channel_id == "55555":
                return thread_session
            return parent_session
        stub.sessions.get_or_create = MagicMock(side_effect=_get_or_create)

        msg = _make_message()
        msg.channel = thread

        # Patch discord.Thread so isinstance(MagicMock, MagicMock) is True
        with patch.object(discord, "Thread", MagicMock):
            await stub._handle_message(msg, "thread message")

        # Thread session should now have parent context in its summary
        assert thread_session.summary is not None
        assert "Parent channel context" in thread_session.summary

    async def test_thread_context_uses_channel_lock(self):
        """Thread context inheritance should run inside the channel lock."""
        stub = _make_bot_stub()
        stub._handle_message = LokiBot._handle_message.__get__(stub)
        stub._handle_message_inner = AsyncMock()

        thread = MagicMock()
        thread.id = 55555
        thread.parent = MagicMock()
        thread.parent.id = 67890

        thread_session = MagicMock()
        thread_session.messages = []
        thread_session.summary = None

        parent_session = MagicMock()
        parent_session.messages = [MagicMock(role="user", content="Hello parent")]
        parent_session.summary = "Parent summary"

        def _get_or_create(channel_id):
            if channel_id == "55555":
                return thread_session
            return parent_session
        stub.sessions.get_or_create = MagicMock(side_effect=_get_or_create)

        msg = _make_message()
        msg.channel = thread

        with patch.object(discord, "Thread", MagicMock):
            await stub._handle_message(msg, "thread message")

        # The lock should have been created via setdefault
        assert "55555" in stub._channel_locks
        # And _handle_message_inner should have been called (inside the lock)
        stub._handle_message_inner.assert_awaited_once()

    async def test_channel_lock_uses_setdefault(self):
        """Channel lock should use setdefault for atomic creation."""
        stub = _make_bot_stub()
        stub._handle_message = LokiBot._handle_message.__get__(stub)
        stub._handle_message_inner = AsyncMock()

        msg = _make_message()

        await stub._handle_message(msg, "hello")

        # Lock should exist for the channel
        channel_id = str(msg.channel.id)
        assert channel_id in stub._channel_locks

    async def test_thread_parent_no_messages_no_summary_skips(self):
        """Thread should not inherit from parent with no messages and no summary."""
        stub = _make_bot_stub()
        stub._handle_message = LokiBot._handle_message.__get__(stub)
        stub._handle_message_inner = AsyncMock()

        thread = MagicMock()
        thread.id = 55555
        thread.parent = MagicMock()
        thread.parent.id = 67890

        thread_session = MagicMock()
        thread_session.messages = []
        thread_session.summary = None

        # Parent with no messages and no summary
        parent_session = MagicMock()
        parent_session.messages = []
        parent_session.summary = None

        def _get_or_create(channel_id):
            if channel_id == "55555":
                return thread_session
            return parent_session
        stub.sessions.get_or_create = MagicMock(side_effect=_get_or_create)

        msg = _make_message()
        msg.channel = thread

        with patch.object(discord, "Thread", MagicMock):
            await stub._handle_message(msg, "thread message")

        # Thread session should NOT have inherited anything
        assert thread_session.summary is None

    async def test_parse_time_value_error(self):
        """parse_time handler should return error on ValueError (lines 1703-1704)."""
        stub = _make_bot_stub()
        stub._handle_parse_time = LokiBot._handle_parse_time.__get__(stub)

        with patch("src.tools.time_parser.parse_time", side_effect=ValueError("bad expr")):
            result = stub._handle_parse_time({"expression": "gibberish"})

        assert "Error" in result
        assert "bad expr" in result

    async def test_background_task_pruning(self):
        """Should prune old completed tasks when over limit (lines 1997-1998)."""
        stub = _make_bot_stub()
        stub._handle_delegate_task = LokiBot._handle_delegate_task.__get__(stub)
        stub._background_tasks_max = 2

        # Pre-fill with completed tasks beyond the max
        for i in range(5):
            task = MagicMock()
            task.task_id = f"old-{i}"
            task.status = "completed"
            task.results = []
            stub._background_tasks[f"old-{i}"] = task

        def _close_coro(coro):
            """Mock create_task that closes the coroutine to avoid warnings."""
            coro.close()
            return MagicMock()

        with patch("src.discord.client.BackgroundTask") as MockTask, \
             patch("src.discord.client.create_task_id", return_value="new-1"), \
             patch("src.discord.client.run_background_task", new_callable=AsyncMock), \
             patch("asyncio.create_task", side_effect=_close_coro):
            MockTask.return_value = MagicMock()
            MockTask.return_value.task_id = "new-1"

            msg = _make_message()
            result = await stub._handle_delegate_task(msg, {
                "description": "new task",
                "steps": [{"description": "step1", "tool_name": "check_disk", "tool_input": {}}],
            })

        assert "new-1" in result

    async def test_list_tasks_truncation(self):
        """Should truncate detailed task view when over 3800 chars (line 2045)."""
        stub = _make_bot_stub()
        stub._handle_list_tasks = LokiBot._handle_list_tasks.__get__(stub)

        # Create a task with many results that produce long output
        task = MagicMock()
        task.task_id = "t1"
        task.description = "Long task"
        task.status = "completed"
        result_mock = MagicMock()
        result_mock.status = "ok"
        result_mock.index = 0
        result_mock.description = "step"
        result_mock.elapsed_ms = 100
        result_mock.output = "X" * 500
        task.results = [result_mock] * 20
        task.steps = [{"desc": "s"}] * 20
        stub._background_tasks = {"t1": task}

        result = stub._handle_list_tasks({"task_id": "t1"})

        assert "truncated" in result

    async def test_workflow_post_failure(self):
        """Should catch exception when posting workflow results (lines 2124-2125)."""
        stub = _make_bot_stub()
        stub._run_scheduled_workflow = LokiBot._run_scheduled_workflow.__get__(stub)
        stub.skill_manager.has_skill = MagicMock(return_value=False)

        channel = AsyncMock()
        channel.send = AsyncMock(side_effect=Exception("Discord down"))

        # Should not raise — exception is caught and logged
        await stub._run_scheduled_workflow(channel, {
            "description": "test workflow",
            "steps": [
                {"tool_name": "check_disk", "tool_input": {"host": "server"}},
            ],
        })

    async def test_post_file_generic_exception(self):
        """post_file should handle generic exceptions during SSH (lines 1610-1611)."""
        stub = _make_bot_stub()
        stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)

        msg = _make_message()

        with patch("asyncio.create_subprocess_exec", side_effect=Exception("SSH broke")):
            result = await stub._handle_post_file(msg, {
                "host": "server",
                "path": "/tmp/test.txt",
            })

        assert "Failed to fetch file" in result

    async def test_post_file_empty_file(self):
        """post_file should handle empty files (line 1614)."""
        stub = _make_bot_stub()
        stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)
        stub.tool_executor._resolve_host = MagicMock(
            return_value=("10.0.0.1", "root", "linux"),
        )
        stub.config.tools.ssh_key_path = "/app/.ssh/id"
        stub.config.tools.ssh_known_hosts_path = "/app/.ssh/known_hosts"

        import base64
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(base64.b64encode(b""), b""))
        proc.returncode = 0

        msg = _make_message()
        proc_mock = AsyncMock()
        proc_mock.returncode = 0
        proc_mock.communicate = AsyncMock(
            return_value=(base64.b64encode(b""), b""),
        )
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc_mock)):
            result = await stub._handle_post_file(msg, {
                "host": "server",
                "path": "/tmp/test.txt",
            })

        assert "not found or empty" in result or "Posted" in result

    async def test_post_file_too_large(self):
        """post_file should reject files over 25MB (line 1618)."""
        stub = _make_bot_stub()
        stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)
        stub.tool_executor._resolve_host = MagicMock(
            return_value=("10.0.0.1", "root", "linux"),
        )
        stub.config.tools.ssh_key_path = "/app/.ssh/id"
        stub.config.tools.ssh_known_hosts_path = "/app/.ssh/known_hosts"

        import base64
        # 30MB file
        large_data = b"X" * (30 * 1024 * 1024)
        encoded = base64.b64encode(large_data)

        proc_mock = AsyncMock()
        proc_mock.returncode = 0
        proc_mock.communicate = AsyncMock(return_value=(encoded, b""))

        msg = _make_message()
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc_mock)):
            result = await stub._handle_post_file(msg, {
                "host": "server",
                "path": "/tmp/huge.bin",
            })

        assert "too large" in result.lower() or "25 MB" in result

    async def test_post_file_discord_upload_failure(self):
        """post_file should handle Discord upload failure (lines 1625-1626)."""
        stub = _make_bot_stub()
        stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)
        stub.tool_executor._resolve_host = MagicMock(
            return_value=("10.0.0.1", "root", "linux"),
        )
        stub.config.tools.ssh_key_path = "/app/.ssh/id"
        stub.config.tools.ssh_known_hosts_path = "/app/.ssh/known_hosts"

        import base64
        file_data = b"Hello, world!"
        encoded = base64.b64encode(file_data)

        proc_mock = AsyncMock()
        proc_mock.returncode = 0
        proc_mock.communicate = AsyncMock(return_value=(encoded, b""))

        msg = _make_message()
        msg.channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(), "upload failed"),
        )

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc_mock)):
            result = await stub._handle_post_file(msg, {
                "host": "server",
                "path": "/tmp/test.txt",
            })

        assert "Failed to upload" in result

    async def test_background_task_crash(self):
        """Background task crash should be caught and status set to failed (lines 2010-2012)."""
        stub = _make_bot_stub()
        stub._handle_delegate_task = LokiBot._handle_delegate_task.__get__(stub)

        task_mock = MagicMock()
        task_mock.task_id = "crash-1"
        task_mock.status = "pending"

        async def _crash(*args, **kwargs):
            raise RuntimeError("task exploded")

        with patch("src.discord.client.BackgroundTask", return_value=task_mock), \
             patch("src.discord.client.create_task_id", return_value="crash-1"), \
             patch("src.discord.client.run_background_task", new=_crash):
            msg = _make_message()
            result = await stub._handle_delegate_task(msg, {
                "description": "crashy task",
                "steps": [
                    {"description": "s1", "tool_name": "check_disk", "tool_input": {}},
                ],
            })

            # Wait for the background task to run
            assert "crash-1" in result
            # Let the created task run
            await asyncio.sleep(0.05)

        assert task_mock.status == "failed"

    async def test_empty_tools_after_permission_filter(self):
        """When permission filter returns empty list, tools should become None (line 1224)."""
        from src.llm.types import LLMResponse
        stub = _make_bot_stub()
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        # filter_tools returns empty list (guest with no allowed tools)
        stub.permissions.filter_tools = MagicMock(return_value=[])

        # Mock codex_client to return a simple text response
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Simple response", tool_calls=[], stop_reason="end_turn")
        )

        msg = _make_message()
        text, already_sent, is_error, tools, _handoff = await stub._process_with_tools(
            msg, [{"role": "user", "content": "hi"}],
        )

        assert text == "Simple response"
