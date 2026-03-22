"""Tests for chat-path optimization: deferred system prompt build and Codex cleanup.

Session 15 deferred the full system prompt build (`_build_system_prompt`) from
running on EVERY message to only running on task-classified messages.  Chat
messages use `_build_chat_system_prompt` instead, avoiding the full prompt's
disk I/O (memory.json + learned.json reads) and string formatting (host lists,
services, playbooks).

Also verifies that the Codex client's `instructions` field no longer duplicates
the emoji rule already present in the chat system prompt template.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402
from src.llm.openai_codex import CodexChatClient  # noqa: E402
from src.llm.system_prompt import CHAT_SYSTEM_PROMPT_TEMPLATE  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal LokiBot stub for _handle_message_inner tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._last_tool_use = {}
    stub._system_prompt = "initial system prompt"
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
    stub.classifier.classify = AsyncMock(return_value="chat")
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex chat response")
    stub.codex_client.chat_with_tools = AsyncMock(return_value=MagicMock())
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.voice_manager = None
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
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
# Deferred system prompt build — chat paths
# ---------------------------------------------------------------------------

class TestChatPathSkipsFullPromptBuild:
    """Chat-routed messages should NOT call _build_system_prompt."""

    async def test_codex_chat_does_not_build_full_prompt(self):
        """When Codex handles a chat message, _build_system_prompt should
        not be called (only _build_chat_system_prompt)."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Hey!")
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hey whats up", "chan-1")

        stub._build_system_prompt.assert_not_called()
        stub._build_chat_system_prompt.assert_called_once()

    async def test_no_codex_chat_returns_error(self):
        """When no Codex is configured and chat is selected, should return
        error message without building the full system prompt."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = None
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hey whats up", "chan-1")

        stub._build_system_prompt.assert_not_called()
        # Error message sent to user
        stub._send_chunked.assert_called_once()
        sent = stub._send_chunked.call_args[0][1]
        assert "not configured" in sent.lower()

    async def test_codex_failure_fallback_does_not_build_full_prompt(self):
        """When Codex fails and falls back to _process_with_tools,
        _build_system_prompt should still not be called."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(side_effect=Exception("Codex down"))
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._process_with_tools = AsyncMock(
            return_value=("Fallback response", False, False, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hey whats up", "chan-1")

        stub._build_system_prompt.assert_not_called()

    async def test_chat_does_not_build_full_prompt(self):
        """When message is routed to Codex chat,
        _build_system_prompt should not be called."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Chat response")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hey whats up", "chan-1")

        stub._build_system_prompt.assert_not_called()


# ---------------------------------------------------------------------------
# Deferred system prompt build — task paths
# ---------------------------------------------------------------------------

class TestTaskPathBuildsFullPrompt:
    """Task-routed messages MUST call _build_system_prompt."""

    async def test_task_builds_full_prompt(self):
        """When a message is classified as 'task', _build_system_prompt
        should be called before _process_with_tools."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._process_with_tools = AsyncMock(
            return_value=("Disk is 42% full.", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._build_system_prompt.assert_called_once_with(channel=msg.channel, user_id=str(msg.author.id), query="check disk")

    async def test_task_keyword_builds_full_prompt(self):
        """Keyword-matched task messages should build the full system prompt."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._process_with_tools = AsyncMock(
            return_value=("Server is up.", False, False, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check the server", "chan-1")

        stub._build_system_prompt.assert_called_once()

    async def test_no_codex_task_does_not_build_full_prompt(self):
        """When no Codex client and task, the full prompt should NOT
        be built (we return early with error)."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = None
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._build_system_prompt.assert_not_called()

    async def test_task_prompt_passed_to_process_with_tools(self):
        """The freshly-built system prompt should be passed via system_prompt_override."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._build_system_prompt = MagicMock(return_value="fresh task prompt")

        # Capture what _process_with_tools receives as system_prompt_override
        captured_prompt = []

        async def capture_process(*args, **kwargs):
            captured_prompt.append(kwargs.get("system_prompt_override"))
            return ("result", False, False, [], False)

        stub._process_with_tools = capture_process
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        assert captured_prompt == ["fresh task prompt"]


# ---------------------------------------------------------------------------
# Deferred system prompt build — image path
# ---------------------------------------------------------------------------

class TestImagePathBuildsFullPrompt:
    """Images force the task route, so _build_system_prompt should be called."""

    async def test_image_message_builds_full_prompt(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub._process_with_tools = AsyncMock(
            return_value=("I see a cat!", False, False, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(
            msg, "what is this?", "chan-1",
            image_blocks=[{"type": "image", "source": {"type": "base64", "data": "..."}}],
        )

        stub._build_system_prompt.assert_called_once_with(channel=msg.channel, user_id=str(msg.author.id), query="what is this?")


# ---------------------------------------------------------------------------
# Codex emoji duplication fix
# ---------------------------------------------------------------------------

class TestCodexInstructionsCleanup:
    """The Codex client should not duplicate instructions already in the prompt."""

    async def test_codex_instructions_equal_system_param(self):
        """The instructions field sent to the Codex API should be exactly
        the system parameter, with no extra appended text."""
        auth = MagicMock()
        auth.get_access_token = AsyncMock(return_value="token")
        auth.get_account_id = MagicMock(return_value="acct")

        client = CodexChatClient(auth=auth, model="gpt-4", max_tokens=1000)

        captured_body = {}

        async def mock_stream_request(headers, body):
            captured_body.update(body)
            return "response text"

        client._stream_request = mock_stream_request

        system = "You are a bot. NEVER use emojis."
        await client.chat(messages=[], system=system)

        # Instructions should be exactly the system prompt, no duplication
        assert captured_body["instructions"] == system

    def test_codex_instructions_no_duplicate_emoji_rule(self):
        """Verify the Codex client code does not append emoji rules."""
        import inspect
        source = inspect.getsource(CodexChatClient.chat)
        # Should NOT contain the old hardcoded style append
        assert "Style: Keep responses concise" not in source
        assert "No smiley faces" not in source

    def test_chat_template_has_emoji_rule(self):
        """The chat system prompt template must contain the emoji rule,
        confirming it's the canonical source (not Codex's hardcoded append)."""
        assert "NEVER use emojis" in CHAT_SYSTEM_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Chat prompt called with correct channel
# ---------------------------------------------------------------------------

class TestChatPromptChannelPassthrough:
    """_build_chat_system_prompt should receive the message's channel."""

    async def test_codex_path_passes_channel(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Hi!")
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1")

        stub._build_chat_system_prompt.assert_called_with(channel=msg.channel, user_id=str(msg.author.id))

    async def test_no_codex_chat_returns_error_with_channel(self):
        """When no Codex configured and chat is selected, should return error."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = None
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1")

        stub._send_chunked.assert_called_once()
        sent = stub._send_chunked.call_args[0][1]
        assert "not configured" in sent.lower()


# ---------------------------------------------------------------------------
# Chat prompt override correctness
# ---------------------------------------------------------------------------

class TestChatPromptOverride:
    """Chat paths must pass the chat prompt as system_prompt_override
    to _process_with_tools rather than relying on _system_prompt."""

    async def test_no_codex_chat_returns_error_message(self):
        """When no Codex is configured and chat selected, returns error without
        calling _process_with_tools."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = None
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1")

        stub._process_with_tools.assert_not_called()
        stub._send_chunked.assert_called_once()

    async def test_codex_failure_returns_error_message(self):
        """When Codex fails on chat route, returns error message."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(side_effect=Exception("fail"))
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1")

        # Should NOT fall back to _process_with_tools
        stub._process_with_tools.assert_not_called()
        # Error message sent to user
        stub._send_chunked.assert_called_once()
        sent = stub._send_chunked.call_args[0][1]
        assert "temporarily unavailable" in sent.lower()
