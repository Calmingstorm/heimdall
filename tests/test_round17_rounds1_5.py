"""Round 17: Tests for changes made in Rounds 1-5.

Covers:
- System prompt: chat-without-tools guidance (Round 1)
- System prompt: tool availability rule — Rule 11 (Round 1)
- System prompt: self-awareness directive — Rule 12 (Round 2)
- System prompt: still under 5000 chars
- Context separator: includes message ID (Round 3)
- add_reaction: handles "current"/"this"/"self"/empty message_id (Round 3)
- architecture.md: contains "Responding Without Tools" section (Round 1)
- Skill context: http_get/http_post custom headers (Round 5)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest

import discord

from src.llm.system_prompt import (
    SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
)
from src.discord.client import LokiBot, DISCORD_MAX_LEN
from src.tools.skill_context import SkillContext

_ARCH_PATH = Path(__file__).parent.parent / "data" / "context" / "architecture.md"
_ARCH_CONTEXT = _ARCH_PATH.read_text()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(**overrides):
    """Minimal LokiBot stub for method-level tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test system prompt"
    stub._channel_locks = {}
    stub._processed_messages = MagicMock()
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
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.codex_client = MagicMock()
    stub.codex_client.chat_with_tools = AsyncMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._pending_files = {}
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.browser_manager = None
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")
    stub._knowledge_store = None
    stub._embedder = None
    stub._fts_index = None
    stub._vector_store = None
    stub.scheduler = MagicMock()
    stub.voice_manager = None
    stub.context_loader = MagicMock()
    stub.loop_manager = MagicMock()
    for k, v in overrides.items():
        setattr(stub, k, v)
    return stub


def _make_message(channel_id="67890", author_id="12345", content="test", msg_id=None):
    """Create a mock Discord message."""
    msg = AsyncMock()
    msg.id = msg_id or int(time.time() * 1000)
    msg.content = content
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock()
    msg.channel.fetch_message = AsyncMock()
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(), __aexit__=AsyncMock(),
    ))
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.bot = False
    msg.author.mention = f"<@{author_id}>"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    msg.attachments = []
    msg.webhook_id = None
    return msg


def _build_prompt(**kwargs):
    """Build a system prompt with defaults for brevity."""
    defaults = dict(context="", hosts={}, services=[], playbooks=[])
    defaults.update(kwargs)
    return build_system_prompt(**defaults)


# ===========================================================================
# Round 1: Chat-without-tools guidance
# ===========================================================================

class TestChatWithoutToolsGuidance:
    """System prompt contains guidance that plain text responses are valid."""

    def test_prompt_mentions_plain_text_chat(self):
        prompt = _build_prompt()
        assert "plain text" in prompt.lower()

    def test_prompt_says_tools_for_actions_not_prerequisite(self):
        prompt = _build_prompt()
        assert "Tools are for ACTIONS" in prompt
        assert "not a prerequisite" in prompt

    def test_chat_guidance_near_executor_line(self):
        """Chat guidance should appear in the CORE BEHAVIOR section."""
        prompt = _build_prompt()
        executor_idx = prompt.index("EXECUTOR")
        chat_idx = prompt.index("plain text for chat")
        # Chat guidance comes after EXECUTOR but before the Rules section
        rules_idx = prompt.index("## Rules")
        assert executor_idx < chat_idx < rules_idx


# ===========================================================================
# Round 1: Tool availability rule (Rule 11)
# ===========================================================================

class TestToolAvailabilityRule:
    """System prompt Rule 11: never claim a tool is unavailable without calling it."""

    def test_rule_11_exists(self):
        prompt = _build_prompt()
        assert "11." in prompt

    def test_rule_11_forbids_claiming_unavailable(self):
        prompt = _build_prompt()
        assert "NEVER claim a tool is unavailable" in prompt

    def test_rule_11_mentions_disabled_not_enabled(self):
        prompt = _build_prompt()
        # Must mention common false-claim patterns
        assert "disabled" in prompt.lower() or "not enabled" in prompt.lower()

    def test_rule_11_says_report_actual_error(self):
        prompt = _build_prompt()
        assert "actual error" in prompt.lower()

    def test_rule_11_says_call_first(self):
        prompt = _build_prompt()
        assert "calling it first" in prompt


# ===========================================================================
# Round 2: Self-awareness directive (Rule 12)
# ===========================================================================

class TestSelfAwarenessDirective:
    """System prompt Rule 12: context separation for external projects."""

    def test_rule_12_exists(self):
        prompt = _build_prompt()
        assert "12." in prompt

    def test_rule_12_mentions_source_code_location(self):
        prompt = _build_prompt()
        assert "/opt/loki" in prompt

    def test_rule_12_warns_against_searching_own_code(self):
        prompt = _build_prompt()
        assert "do NOT search" in prompt or "do NOT\nsearch" in prompt

    def test_rule_12_allows_self_modification(self):
        """Rule 12 must explicitly allow modifying own source when asked."""
        prompt = _build_prompt()
        assert "modify your own source" in prompt or "work on yourself" in prompt

    def test_rule_12_uses_claude_code_dir_variable(self):
        """Rule 12 should use the claude_code_dir parameter, not hardcode."""
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
            claude_code_dir="/custom/loki/path",
        )
        assert "/custom/loki/path" in prompt

    def test_rule_12_default_claude_code_dir(self):
        """Default claude_code_dir is /opt/loki."""
        prompt = _build_prompt()
        assert "/opt/loki" in prompt


# ===========================================================================
# System prompt size limit
# ===========================================================================

class TestSystemPromptSize:
    """System prompt must stay under 5000 chars."""

    def test_template_under_5000_chars(self):
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000, (
            f"Template is {len(SYSTEM_PROMPT_TEMPLATE)} chars, must be under 5000"
        )

    def test_built_prompt_under_5000_chars(self):
        """Even with architecture context, the template portion stays under 5000."""
        # The template itself (before variable substitution) is what matters
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000

    def test_built_prompt_reasonable_size(self):
        """Full prompt with minimal config should be reasonable."""
        prompt = _build_prompt()
        # With empty context, should be well under 5000
        assert len(prompt) < 5000


# ===========================================================================
# Round 3: Context separator includes message ID
# ===========================================================================

class TestContextSeparatorMessageID:
    """The context separator injected by _process_with_tools includes the message ID."""

    @pytest.mark.asyncio
    async def test_message_id_in_separator_multi_message_history(self):
        """With multiple messages in history, separator contains message ID."""
        stub = _make_bot_stub()
        msg = _make_message(msg_id=123456789)

        # History with multiple messages
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "new question"},
        ]

        # Make chat_with_tools return a simple text response (no tools)
        from src.llm.types import LLMResponse
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="test response", tool_calls=[])
        )

        # Call _process_with_tools
        result = await LokiBot._process_with_tools(stub, msg, history)

        # Verify chat_with_tools was called
        assert stub.codex_client.chat_with_tools.called
        call_args = stub.codex_client.chat_with_tools.call_args

        # Find the messages arg (first positional or keyword)
        messages = call_args[1].get("messages") or call_args[0][0]

        # Look for the developer message containing the message ID
        developer_msgs = [m for m in messages if m.get("role") == "developer"]
        assert any("123456789" in m["content"] for m in developer_msgs), (
            f"Message ID 123456789 not found in developer messages: {developer_msgs}"
        )

    @pytest.mark.asyncio
    async def test_message_id_in_single_message_history(self):
        """With a single message, a lightweight developer note has the message ID."""
        stub = _make_bot_stub()
        msg = _make_message(msg_id=987654321)

        # Single message history
        history = [{"role": "user", "content": "hello"}]

        from src.llm.types import LLMResponse
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="hi", tool_calls=[])
        )

        result = await LokiBot._process_with_tools(stub, msg, history)

        call_args = stub.codex_client.chat_with_tools.call_args
        messages = call_args[1].get("messages") or call_args[0][0]

        developer_msgs = [m for m in messages if m.get("role") == "developer"]
        assert any("987654321" in m["content"] for m in developer_msgs), (
            f"Message ID 987654321 not found in developer messages: {developer_msgs}"
        )

    @pytest.mark.asyncio
    async def test_separator_contains_current_request_marker(self):
        """Multi-message separator should contain CURRENT REQUEST marker."""
        stub = _make_bot_stub()
        msg = _make_message()

        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "new"},
        ]

        from src.llm.types import LLMResponse
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="ok", tool_calls=[])
        )

        await LokiBot._process_with_tools(stub, msg, history)

        call_args = stub.codex_client.chat_with_tools.call_args
        messages = call_args[1].get("messages") or call_args[0][0]
        developer_msgs = [m for m in messages if m.get("role") == "developer"]
        assert any("CURRENT REQUEST" in m["content"] for m in developer_msgs)


# ===========================================================================
# Round 3: add_reaction handles "current"/"this"/"self"/empty message_id
# ===========================================================================

class TestAddReactionMessageIDResolution:
    """_handle_add_reaction resolves special message_id values to triggering message."""

    @pytest.mark.asyncio
    async def test_reaction_with_this(self):
        """message_id='this' resolves to the triggering message ID."""
        stub = _make_bot_stub()
        msg = _make_message(msg_id=111222333)
        fetched_msg = AsyncMock()
        fetched_msg.add_reaction = AsyncMock()
        msg.channel.fetch_message = AsyncMock(return_value=fetched_msg)

        result = await LokiBot._handle_add_reaction(
            stub, msg, {"message_id": "this", "emoji": "thumbsup"}
        )

        msg.channel.fetch_message.assert_called_once_with(111222333)
        assert "added" in result.lower() or "reaction" in result.lower()

    @pytest.mark.asyncio
    async def test_reaction_with_current(self):
        """message_id='current' resolves to the triggering message ID."""
        stub = _make_bot_stub()
        msg = _make_message(msg_id=444555666)
        fetched_msg = AsyncMock()
        fetched_msg.add_reaction = AsyncMock()
        msg.channel.fetch_message = AsyncMock(return_value=fetched_msg)

        result = await LokiBot._handle_add_reaction(
            stub, msg, {"message_id": "current", "emoji": "fire"}
        )

        msg.channel.fetch_message.assert_called_once_with(444555666)

    @pytest.mark.asyncio
    async def test_reaction_with_self(self):
        """message_id='self' resolves to the triggering message ID."""
        stub = _make_bot_stub()
        msg = _make_message(msg_id=777888999)
        fetched_msg = AsyncMock()
        fetched_msg.add_reaction = AsyncMock()
        msg.channel.fetch_message = AsyncMock(return_value=fetched_msg)

        result = await LokiBot._handle_add_reaction(
            stub, msg, {"message_id": "self", "emoji": "heart"}
        )

        msg.channel.fetch_message.assert_called_once_with(777888999)

    @pytest.mark.asyncio
    async def test_reaction_with_empty_message_id(self):
        """Empty message_id resolves to the triggering message ID."""
        stub = _make_bot_stub()
        msg = _make_message(msg_id=101010101)
        fetched_msg = AsyncMock()
        fetched_msg.add_reaction = AsyncMock()
        msg.channel.fetch_message = AsyncMock(return_value=fetched_msg)

        result = await LokiBot._handle_add_reaction(
            stub, msg, {"message_id": "", "emoji": "wave"}
        )

        msg.channel.fetch_message.assert_called_once_with(101010101)

    @pytest.mark.asyncio
    async def test_reaction_with_none_message_id(self):
        """None message_id resolves to the triggering message ID."""
        stub = _make_bot_stub()
        msg = _make_message(msg_id=202020202)
        fetched_msg = AsyncMock()
        fetched_msg.add_reaction = AsyncMock()
        msg.channel.fetch_message = AsyncMock(return_value=fetched_msg)

        result = await LokiBot._handle_add_reaction(
            stub, msg, {"emoji": "check"}
        )

        msg.channel.fetch_message.assert_called_once_with(202020202)

    @pytest.mark.asyncio
    async def test_reaction_with_explicit_message_id(self):
        """Explicit numeric message_id is used directly (not overridden)."""
        stub = _make_bot_stub()
        msg = _make_message(msg_id=111)
        fetched_msg = AsyncMock()
        fetched_msg.add_reaction = AsyncMock()
        msg.channel.fetch_message = AsyncMock(return_value=fetched_msg)

        result = await LokiBot._handle_add_reaction(
            stub, msg, {"message_id": "999888777", "emoji": "star"}
        )

        # Should use the explicit ID, not the triggering message ID
        msg.channel.fetch_message.assert_called_once_with(999888777)

    @pytest.mark.asyncio
    async def test_reaction_requires_emoji(self):
        """Missing emoji returns an error."""
        stub = _make_bot_stub()
        msg = _make_message()

        result = await LokiBot._handle_add_reaction(stub, msg, {"message_id": "123"})
        assert "required" in result.lower() or "emoji" in result.lower()


# ===========================================================================
# Round 1: architecture.md contains "Responding Without Tools" section
# ===========================================================================

class TestArchitectureDoc:
    """architecture.md must contain the Round 1 additions."""

    def test_responding_without_tools_section_exists(self):
        assert "## Responding Without Tools" in _ARCH_CONTEXT

    def test_responding_without_tools_mentions_plain_text(self):
        assert "plain text" in _ARCH_CONTEXT

    def test_responding_without_tools_mentions_chat(self):
        assert "Chat" in _ARCH_CONTEXT or "chat" in _ARCH_CONTEXT

    def test_responding_without_tools_mentions_creative_writing(self):
        assert "creative writing" in _ARCH_CONTEXT

    def test_responding_without_tools_says_tools_for_actions(self):
        assert "Tools are for actions" in _ARCH_CONTEXT

    def test_architecture_loads_as_context(self):
        """architecture.md can be loaded and injected into the system prompt."""
        prompt = build_system_prompt(
            context=_ARCH_CONTEXT, hosts={}, services=[], playbooks=[],
        )
        assert "Responding Without Tools" in prompt


# ===========================================================================
# Round 5: Skill context http_get/http_post custom headers
# ===========================================================================

class TestSkillContextHTTPHeaders:
    """http_get and http_post accept and use custom headers."""

    @pytest.fixture
    def ctx(self):
        executor = MagicMock()
        return SkillContext(executor, "test_skill")

    @pytest.mark.asyncio
    async def test_http_get_default_accept_json(self, ctx):
        """http_get includes Accept: application/json by default."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.content_type = "application/json"
            mock_resp.json = AsyncMock(return_value={"ok": True})
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock()

            mock_session = AsyncMock()
            mock_get = MagicMock(return_value=mock_resp)
            mock_session.get = mock_get
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_session_cls.return_value = mock_session

            await ctx.http_get("https://example.com/api")

            # Check headers passed to session.get
            call_kwargs = mock_get.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert headers is not None
            assert headers.get("Accept") == "application/json"

    @pytest.mark.asyncio
    async def test_http_get_custom_headers_override(self, ctx):
        """Custom headers override defaults in http_get."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.content_type = "text/plain"
            mock_resp.text = AsyncMock(return_value="plain text")
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock()

            mock_session = AsyncMock()
            mock_get = MagicMock(return_value=mock_resp)
            mock_session.get = mock_get
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_session_cls.return_value = mock_session

            await ctx.http_get(
                "https://example.com/api",
                headers={"Accept": "text/html", "X-Custom": "value"},
            )

            call_kwargs = mock_get.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            # Custom Accept overrides default
            assert headers["Accept"] == "text/html"
            assert headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_http_post_custom_headers(self, ctx):
        """http_post passes custom headers through."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.content_type = "application/json"
            mock_resp.json = AsyncMock(return_value={"created": True})
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock()

            mock_session = AsyncMock()
            mock_post = MagicMock(return_value=mock_resp)
            mock_session.post = mock_post
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_session_cls.return_value = mock_session

            await ctx.http_post(
                "https://example.com/api",
                json={"key": "val"},
                headers={"Authorization": "Bearer tok123"},
            )

            call_kwargs = mock_post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert headers["Authorization"] == "Bearer tok123"

    @pytest.mark.asyncio
    async def test_http_post_no_headers(self, ctx):
        """http_post without headers passes None (or empty dict)."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.content_type = "application/json"
            mock_resp.json = AsyncMock(return_value={})
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock()

            mock_session = AsyncMock()
            mock_post = MagicMock(return_value=mock_resp)
            mock_session.post = mock_post
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_session_cls.return_value = mock_session

            await ctx.http_post("https://example.com/api", json={"x": 1})

            call_kwargs = mock_post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            # No custom headers → should be None or empty
            assert headers is None or headers == {}
