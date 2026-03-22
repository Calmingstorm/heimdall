"""Tests for 3-way message classification and claude_code routing.

Tests cover:
- Classifier returns "claude_code" as a valid classification
- Classifier prompt includes the claude_code category
- Routing: claude_code messages go to _handle_claude_code
- Routing: fallback when claude -p fails
- No full system prompt build for claude_code path
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402
from src.discord.routing import CLAUDE_CODE_DEFAULTS  # noqa: E402


@pytest.fixture(autouse=True)
def _setup_routing_defaults():
    """Set up test routing defaults before each test, restore after."""
    old = dict(CLAUDE_CODE_DEFAULTS)
    CLAUDE_CODE_DEFAULTS["primary"] = ("desktop", "/root/project")
    CLAUDE_CODE_DEFAULTS["secondary"] = ("server", "/opt/project")
    yield
    CLAUDE_CODE_DEFAULTS.clear()
    CLAUDE_CODE_DEFAULTS.update(old)


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
    stub.classifier.classify = AsyncMock(return_value="claude_code")
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
# Routing: claude_code path
# ---------------------------------------------------------------------------

class TestClaudeCodeRouting:
    """Messages classified as 'claude_code' should route to claude -p CLI."""

    async def test_claude_code_calls_handle_claude_code(self):
        """claude_code messages should invoke tool_executor._handle_claude_code."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "what does check_service do?", "chan-1")

        stub.tool_executor._handle_claude_code.assert_called_once()
        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["host"] == "desktop"
        assert call_args["working_directory"] == "/root/project"
        assert call_args["prompt"] == "what does check_service do?"
        assert call_args["allow_edits"] is False

    async def test_claude_code_response_sent_to_user(self):
        """The response from claude -p should be sent to the user."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="The function checks if a systemd service is running."
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "what does check_service do?", "chan-1")

        stub._send_chunked.assert_called_once_with(
            msg, "The function checks if a systemd service is running."
        )

    async def test_claude_code_response_saved_to_history(self):
        """Successful claude_code responses should be saved to session history."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain the code", "chan-1")

        stub.sessions.add_message.assert_any_call(
            "chan-1", "assistant", "The check_service function does X, Y, Z..."
        )
        stub.sessions.save.assert_called_once()

    async def test_claude_code_does_not_build_full_prompt(self):
        """claude_code path should NOT call _build_system_prompt."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review this code", "chan-1")

        stub._build_system_prompt.assert_not_called()
        stub._build_chat_system_prompt.assert_not_called()

    async def test_claude_code_does_not_call_process_with_tools(self):
        """claude_code path should NOT use the Codex tool loop."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain the function", "chan-1")

        stub._process_with_tools.assert_not_called()


# ---------------------------------------------------------------------------
# Routing: claude_code fallback to Codex
# ---------------------------------------------------------------------------

class TestClaudeCodeFallback:
    """When claude -p fails, the bot should fall back to Codex chat."""

    async def test_fallback_on_error_response(self):
        """If _handle_claude_code returns an error string, fall back to Codex chat."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Codex fallback response")
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Claude Code failed (exit 1):\ncommand not found"
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # Should fall back to Codex chat
        stub.codex_client.chat.assert_called_once()
        stub._build_chat_system_prompt.assert_called()

    async def test_fallback_on_unknown_host(self):
        """If _handle_claude_code returns 'Unknown or disallowed host', fall back to Codex."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Codex fallback")
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Unknown or disallowed host: desktop"
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        stub.codex_client.chat.assert_called_once()

    async def test_fallback_on_exception(self):
        """If _handle_claude_code raises an exception, fall back to Codex chat."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Codex fallback")
        stub.tool_executor._handle_claude_code = AsyncMock(
            side_effect=Exception("SSH connection refused")
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # Should fall back to Codex chat
        stub.codex_client.chat.assert_called_once()
        stub._build_chat_system_prompt.assert_called()

    async def test_no_codex_fallback_on_error_response(self):
        """When claude -p fails and no Codex available, error is returned."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = None
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Claude Code failed (exit 1):\nerror"
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # No Codex fallback — error response is used as-is
        stub._process_with_tools.assert_not_called()

    async def test_no_codex_fallback_on_exception(self):
        """When claude -p raises and no Codex available, send error to user."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = None
        stub.tool_executor._handle_claude_code = AsyncMock(
            side_effect=Exception("SSH timeout")
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        stub._process_with_tools.assert_not_called()
        # Error saved to history for checkpoint-save (Round 14)
        stub.sessions.remove_last_message.assert_not_called()
        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1


# ---------------------------------------------------------------------------
# Routing: classification-driven routing
# ---------------------------------------------------------------------------

class TestClassificationRouting:
    """Messages are routed correctly based on classification."""

    async def test_claude_code_routes_to_claude_p(self):
        """claude_code classification should route to claude -p."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain the code", "chan-1")

        # Should route to claude -p, not block
        stub.tool_executor._handle_claude_code.assert_called_once()
        stub._send_chunked.assert_called_once()

    async def test_task_with_codex_uses_process_with_tools(self):
        """When Codex is available and message classified as task, uses _process_with_tools."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Chat response")
        stub.classifier.classify = AsyncMock(return_value="task")
        stub._process_with_tools = AsyncMock(return_value=("Tool result", False, False, ["check_disk"], False))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "check the server metrics", "chan-1")

        # Should route to Codex tool calling
        stub._process_with_tools.assert_called_once()
        # system_prompt_override should be passed (not use_codex, which no longer exists)
        call_kwargs = stub._process_with_tools.call_args[1]
        assert "system_prompt_override" in call_kwargs

    async def test_task_no_codex_returns_error(self):
        """When no Codex and classified as 'task', should return error."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = None
        stub.classifier.classify = AsyncMock(return_value="task")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "restart apache", "chan-1")

        # Should send "no tool backend" message
        stub._send_with_retry.assert_called_once()
        call_args = stub._send_with_retry.call_args[0]
        assert "no tool backend" in call_args[1].lower()

    async def test_classifier_always_called(self):
        """Classifier should be called for non-keyword messages."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1")

        # Classifier should still be called
        stub.classifier.classify.assert_called_once()


# ---------------------------------------------------------------------------
# Routing: keyword bypass still routes to task
# ---------------------------------------------------------------------------

class TestKeywordBypassUnchanged:
    """Keyword-matched messages should still go directly to 'task'."""

    async def test_keyword_match_goes_to_task_not_claude_code(self):
        """Even code-related messages with infra keywords should go to task."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub._process_with_tools = AsyncMock(
            return_value=("Done", False, False, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "deploy the latest code", "chan-1")

        # Keyword match → task → Codex tool loop, classifier not called
        stub.classifier.classify.assert_not_called()
        stub.tool_executor._handle_claude_code.assert_not_called()
        stub._process_with_tools.assert_called_once()

    async def test_image_message_goes_to_task_not_claude_code(self):
        """Image messages should still force task route."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._process_with_tools = AsyncMock(
            return_value=("I see the image", False, False, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "what is this?", "chan-1",
                image_blocks=[{"type": "image", "source": {"data": "base64data"}}],
            )

        stub.classifier.classify.assert_not_called()
        stub.tool_executor._handle_claude_code.assert_not_called()
        stub._process_with_tools.assert_called_once()


# ---------------------------------------------------------------------------
# Routing: claude_code conversation context
# ---------------------------------------------------------------------------

class TestClaudeCodeConversationContext:
    """claude_code prompts should include recent conversation context for follow-ups."""

    async def test_first_message_no_context_prefix(self):
        """First message (single item in history) should have no context prefix."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.sessions.get_history_with_compaction = AsyncMock(
            return_value=[{"role": "user", "content": "explain the function"}]
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain the function", "chan-1")

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["prompt"] == "explain the function"

    async def test_empty_history_no_context_prefix(self):
        """Empty history (e.g. mocked) should have no context prefix."""
        stub = _make_bot_stub()
        msg = _make_message()
        # Default mock returns []
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "what does check_service do?", "chan-1")

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["prompt"] == "what does check_service do?"

    async def test_followup_includes_prior_exchange(self):
        """Follow-up messages should include previous conversation context."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.sessions.get_history_with_compaction = AsyncMock(
            return_value=[
                {"role": "user", "content": "what does check_service do?"},
                {"role": "assistant", "content": "It checks if a systemd service is running via SSH."},
                {"role": "user", "content": "what about error handling?"},
            ]
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "what about error handling?", "chan-1")

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        prompt = call_args["prompt"]
        assert "Previous conversation" in prompt
        assert "check_service" in prompt
        assert "systemd service" in prompt
        assert "Current request:" in prompt
        assert prompt.endswith("what about error handling?")

    async def test_context_has_role_labels(self):
        """Context messages should be labeled with User/Assistant roles."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.sessions.get_history_with_compaction = AsyncMock(
            return_value=[
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "follow-up"},
            ]
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "follow-up", "chan-1")

        prompt = stub.tool_executor._handle_claude_code.call_args[0][0]["prompt"]
        assert "User: first question" in prompt
        assert "Assistant: first answer" in prompt

    async def test_context_truncates_long_messages(self):
        """Long messages in context should be truncated to 500 chars."""
        stub = _make_bot_stub()
        msg = _make_message()
        long_response = "A" * 800
        stub.sessions.get_history_with_compaction = AsyncMock(
            return_value=[
                {"role": "user", "content": "explain everything"},
                {"role": "assistant", "content": long_response},
                {"role": "user", "content": "summarize that"},
            ]
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "summarize that", "chan-1")

        prompt = stub.tool_executor._handle_claude_code.call_args[0][0]["prompt"]
        # The 800-char response should be truncated to 500 + "..."
        assert "..." in prompt
        # Full 800-char string should NOT appear
        assert long_response not in prompt

    async def test_context_limits_to_six_recent_messages(self):
        """Context should include at most 6 preceding messages (3 exchanges)."""
        stub = _make_bot_stub()
        msg = _make_message()
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"question_{i}"})
            history.append({"role": "assistant", "content": f"answer_{i}"})
        history.append({"role": "user", "content": "latest question"})
        stub.sessions.get_history_with_compaction = AsyncMock(return_value=history)
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "latest question", "chan-1")

        prompt = stub.tool_executor._handle_claude_code.call_args[0][0]["prompt"]
        # Early messages should NOT appear (only last 6 before current)
        assert "question_0" not in prompt
        assert "answer_0" not in prompt
        assert "question_5" not in prompt
        assert "answer_5" not in prompt
        # Recent messages SHOULD appear (messages 7-9 = last 6: q7,a7,q8,a8,q9,a9)
        assert "question_7" in prompt
        assert "answer_9" in prompt

    async def test_context_handles_multimodal_content(self):
        """Non-string message content (e.g. multimodal) should be stringified."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.sessions.get_history_with_compaction = AsyncMock(
            return_value=[
                {"role": "user", "content": [
                    {"type": "image", "source": {"data": "base64"}},
                    {"type": "text", "text": "what is this image?"},
                ]},
                {"role": "assistant", "content": "It shows a network diagram."},
                {"role": "user", "content": "explain the topology"},
            ]
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain the topology", "chan-1")

        prompt = stub.tool_executor._handle_claude_code.call_args[0][0]["prompt"]
        # Should not crash on non-string content
        assert "Previous conversation" in prompt
        assert "network diagram" in prompt

    async def test_context_not_in_codex_fallback_history(self):
        """When claude -p fails and falls back to Codex chat, the original history
        (not the enriched prompt) should be passed to Codex."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Codex fallback response")
        stub.sessions.get_history_with_compaction = AsyncMock(
            return_value=[
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "review the code"},
            ]
        )
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Claude Code failed (exit 1):\ncommand not found"
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # Codex fallback should use original history, not enriched prompt
        stub.codex_client.chat.assert_called_once()
        passed_history = stub.codex_client.chat.call_args[1].get("messages") or stub.codex_client.chat.call_args[0][0]
        last_user = [m for m in passed_history if m["role"] == "user"][-1]
        assert last_user["content"] == "review the code"
        assert "Previous conversation" not in str(last_user["content"])

    async def test_boundary_two_messages(self):
        """With exactly 2 messages in history (1 prior + current), context is included."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.sessions.get_history_with_compaction = AsyncMock(
            return_value=[
                {"role": "assistant", "content": "Welcome! How can I help?"},
                {"role": "user", "content": "explain the router"},
            ]
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain the router", "chan-1")

        prompt = stub.tool_executor._handle_claude_code.call_args[0][0]["prompt"]
        assert "Previous conversation" in prompt
        assert "Welcome" in prompt

    async def test_response_saved_is_output_not_prompt(self):
        """The response saved to history should be claude -p output, not the enriched prompt."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.sessions.get_history_with_compaction = AsyncMock(
            return_value=[
                {"role": "user", "content": "what does classify_message do?"},
                {"role": "assistant", "content": "It classifies messages."},
                {"role": "user", "content": "how accurate is it?"},
            ]
        )
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="The classifier is quite accurate."
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "how accurate is it?", "chan-1")

        # Prompt to claude -p should have context
        prompt = stub.tool_executor._handle_claude_code.call_args[0][0]["prompt"]
        assert "Previous conversation" in prompt

        # But history should have the raw output, not the enriched prompt
        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1
        assert assistant_saves[0][0][2] == "The classifier is quite accurate."
