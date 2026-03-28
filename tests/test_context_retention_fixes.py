"""Tests for context retention fixes (Rounds 1-5).

Covers:
1. Text-only (no-tool) assistant responses ARE saved to history
2. Image markers are descriptive ("User shared image")
3. Updated constant values (RELEVANCE_KEEP_RECENT=5, CONTEXT_TOKEN_BUDGET=16000, etc.)
4. Token budget protects 5 recent messages
5. CHAT_RESPONSE_MAX_CHARS truncation for text-only responses
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402
from src.sessions.manager import (  # noqa: E402
    BUDGET_KEEP_RECENT,
    CHAT_RESPONSE_MAX_CHARS,
    COMPACTION_MAX_CHARS,
    CONTEXT_TOKEN_BUDGET,
    RELEVANCE_KEEP_RECENT,
    RELEVANCE_MAX_OLDER,
    RELEVANCE_MIN_SCORE,
    SessionManager,
    apply_token_budget,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors test_round25_integration.py pattern)
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal HeimdallBot stub with all required attributes."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "You are Heimdall."
    stub._pending_files = {}
    stub._cancelled_tasks = set()
    stub._embedder = None
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_history = MagicMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.get_or_create = MagicMock()
    stub.sessions.detect_topic_change = MagicMock(return_value={
        "is_topic_change": False, "time_gap": 0.0,
        "has_time_gap": False, "max_overlap": 1.0,
    })
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Chat response")
    stub.codex_client.chat_with_tools = AsyncMock(
        return_value=LLMResponse(text="Done", tool_calls=[], stop_reason="end_turn"),
    )
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run a command",
         "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._build_tool_progress_embed = HeimdallBot._build_tool_progress_embed
    stub._build_partial_completion_report = HeimdallBot._build_partial_completion_report
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._maybe_cleanup_caches = MagicMock()
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.tool_memory.suggest = AsyncMock(return_value=[])
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")
    stub.voice_manager = None
    return stub


def _make_message(*, content="hello", channel_id="chan-1", author_id="user-1"):
    msg = AsyncMock()
    msg.content = content
    msg.id = hash(content) % 2**32
    msg.webhook_id = None
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.bot = False
    msg.author.display_name = "TestUser"
    msg.attachments = []
    msg.reference = None
    msg.guild = MagicMock()
    msg.guild.me = MagicMock()
    msg.guild.me.display_name = "Heimdall"
    return msg


# ===========================================================================
# 1. Text-only responses ARE saved to history
# ===========================================================================

class TestTextOnlyResponsesSaved:
    """Verify that assistant responses without tool usage are saved."""

    async def test_text_only_response_saved_to_session(self):
        """When _process_with_tools returns no tools, response is still saved."""
        stub = _make_bot_stub()
        msg = _make_message(content="what's up?")
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)
        stub._process_with_tools = AsyncMock(
            return_value=("Just chatting!", False, False, [], False),
        )

        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "what's up?", "chan-1")

        # Both user AND assistant messages should be saved
        user_saves = [c for c in stub.sessions.add_message.call_args_list if c[0][1] == "user"]
        assistant_saves = [c for c in stub.sessions.add_message.call_args_list if c[0][1] == "assistant"]
        assert len(user_saves) == 1
        assert len(assistant_saves) == 1
        assert assistant_saves[0][0][2] == "Just chatting!"

    async def test_text_only_response_truncated_at_limit(self):
        """Long text-only responses are truncated to CHAT_RESPONSE_MAX_CHARS."""
        stub = _make_bot_stub()
        long_response = "x" * (CHAT_RESPONSE_MAX_CHARS + 500)
        msg = _make_message(content="tell me a story")
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)
        stub._process_with_tools = AsyncMock(
            return_value=(long_response, False, False, [], False),
        )

        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "tell me a story", "chan-1")

        assistant_saves = [c for c in stub.sessions.add_message.call_args_list if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1
        saved_content = assistant_saves[0][0][2]
        assert len(saved_content) == CHAT_RESPONSE_MAX_CHARS

    async def test_short_text_only_response_not_truncated(self):
        """Short text-only responses are saved in full."""
        stub = _make_bot_stub()
        short_response = "I'm fine."
        msg = _make_message(content="how are you?")
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)
        stub._process_with_tools = AsyncMock(
            return_value=(short_response, False, False, [], False),
        )

        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "how are you?", "chan-1")

        assistant_saves = [c for c in stub.sessions.add_message.call_args_list if c[0][1] == "assistant"]
        saved_content = assistant_saves[0][0][2]
        assert saved_content == short_response

    async def test_tool_bearing_response_still_saved(self):
        """Responses with tools are still saved (regression check)."""
        stub = _make_bot_stub()
        msg = _make_message(content="check disk")
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)
        stub._process_with_tools = AsyncMock(
            return_value=("Disk: 45%", False, False, ["run_command"], False),
        )

        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        assistant_saves = [c for c in stub.sessions.add_message.call_args_list if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1

    async def test_error_response_saves_sanitized_marker(self):
        """Error responses save a sanitized marker, not the raw error."""
        stub = _make_bot_stub()
        msg = _make_message(content="do thing")
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)
        stub._process_with_tools = AsyncMock(
            return_value=("Something failed", False, True, ["run_command"], False),
        )

        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "do thing", "chan-1")

        # Error responses save a sanitized marker (not the raw error)
        assistant_saves = [c for c in stub.sessions.add_message.call_args_list if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1
        saved = assistant_saves[0][0][2]
        assert "error" in saved.lower()
        assert "Something failed" not in saved  # raw error not persisted


# ===========================================================================
# 2. Image markers are descriptive
# ===========================================================================

class TestImageMarkers:
    """Verify image attachment markers use descriptive text."""

    async def test_image_marker_says_user_shared(self):
        """Image markers should say 'User shared image' not 'Image attached'."""
        stub = _make_bot_stub()
        stub._process_attachments = HeimdallBot._process_attachments.__get__(stub)
        stub._detect_image_type = HeimdallBot._detect_image_type

        att = AsyncMock()
        att.filename = "photo.png"
        att.content_type = "image/png"
        att.size = 1024
        att.read = AsyncMock(return_value=b"\x89PNG" + b"\x00" * 100)

        msg = MagicMock(spec=discord.Message)
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)
        assert "User shared image" in text
        assert "photo.png" in text
        assert "Image attached" not in text


# ===========================================================================
# 3. Updated constant values
# ===========================================================================

class TestUpdatedConstants:
    """Verify all context retention constants have their new values."""

    def test_relevance_keep_recent(self):
        assert RELEVANCE_KEEP_RECENT == 5

    def test_relevance_min_score(self):
        assert RELEVANCE_MIN_SCORE == 0.10

    def test_relevance_max_older(self):
        assert RELEVANCE_MAX_OLDER == 10

    def test_context_token_budget(self):
        assert CONTEXT_TOKEN_BUDGET == 16000

    def test_budget_keep_recent(self):
        assert BUDGET_KEEP_RECENT == 5

    def test_compaction_max_chars(self):
        assert COMPACTION_MAX_CHARS == 800

    def test_chat_response_max_chars(self):
        assert CHAT_RESPONSE_MAX_CHARS == 1500

    def test_budget_and_relevance_keep_recent_aligned(self):
        """BUDGET_KEEP_RECENT and RELEVANCE_KEEP_RECENT must be equal."""
        assert BUDGET_KEEP_RECENT == RELEVANCE_KEEP_RECENT

    def test_chat_response_max_chars_importable(self):
        """CHAT_RESPONSE_MAX_CHARS can be imported from sessions.manager."""
        from src.sessions.manager import CHAT_RESPONSE_MAX_CHARS as imported
        assert imported == 1500


# ===========================================================================
# 4. Token budget protects 5 recent messages
# ===========================================================================

class TestBudgetProtectsRecent5:
    """Verify token budget protects 5 most recent messages."""

    def test_recent_5_protected_even_over_budget(self):
        """5 recent messages are kept even if they exceed the budget."""
        msgs = [{"role": "user", "content": "a" * 8000} for _ in range(7)]
        result, dropped = apply_token_budget(msgs, budget=2000)
        # Recent 5 are protected, older 2 dropped
        assert dropped == 2
        assert result == msgs[-5:]

    def test_4_messages_all_recent(self):
        """With fewer messages than BUDGET_KEEP_RECENT, all are kept."""
        msgs = [{"role": "user", "content": "a" * 8000} for _ in range(4)]
        result, dropped = apply_token_budget(msgs, budget=100)
        assert dropped == 0
        assert result == msgs

    def test_5_messages_all_recent(self):
        """Exactly BUDGET_KEEP_RECENT messages — all are 'recent', none dropped."""
        msgs = [{"role": "user", "content": "a" * 8000} for _ in range(5)]
        result, dropped = apply_token_budget(msgs, budget=100)
        assert dropped == 0
        assert result == msgs

    def test_6_messages_drops_oldest(self):
        """6 messages: oldest 1 is 'older' and gets dropped when over budget."""
        msgs = [{"role": "user", "content": "a" * 8000} for _ in range(6)]
        result, dropped = apply_token_budget(msgs, budget=100)
        assert dropped == 1
        assert result == msgs[-5:]

    async def test_relevance_keeps_5_recent(self, tmp_path):
        """Relevance scoring always keeps the 5 most recent messages."""
        sm = SessionManager(max_history=100, max_age_hours=24, persist_dir=str(tmp_path))
        # Add 8 messages: 3 old irrelevant + 5 recent
        for i in range(3):
            sm.add_message("ch1", "user", f"completely unrelated topic {i}")
        for i in range(5):
            sm.add_message("ch1", "user", f"nginx config step {i}")

        result = await sm.get_task_history("ch1", max_messages=10, current_query="nginx server")
        contents = [m["content"] for m in result]
        # All 5 recent messages should be present
        for i in range(5):
            assert f"nginx config step {i}" in contents


# ===========================================================================
# 5. Compaction prompt uses dynamic character limit
# ===========================================================================

class TestCompactionPromptDynamic:
    """Verify compaction prompt references COMPACTION_MAX_CHARS dynamically."""

    async def test_compaction_prompt_references_800(self, tmp_path):
        """Compaction prompt should say '800 characters' (from COMPACTION_MAX_CHARS)."""
        from src.sessions.manager import COMPACTION_THRESHOLD

        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "[Topics: test]\n- stuff"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_path / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        for i in range(COMPACTION_THRESHOLD + 5):
            role = "user" if i % 2 == 0 else "assistant"
            mgr.add_message("ch1", role, f"message {i}")
        await mgr.get_history_with_compaction("ch1")

        assert f"{COMPACTION_MAX_CHARS} characters" in captured["system"]
