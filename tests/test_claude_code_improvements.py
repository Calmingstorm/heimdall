"""Tests for claude_code routing improvements.

Round 2, Session 7: Three improvements to the claude_code routing path:

1. Fix is_error bug: When claude_code returns an error AND budget is exceeded,
   is_error was not set to True, causing the error response to be saved to
   session history as a valid assistant message.

2. Configurable output limit: _handle_claude_code now accepts max_output_chars
   parameter (default 3000 for tool loop). The routing path passes 8000
   since there's no tool-loop amplification.

3. Typing indicator: The routing path wraps _handle_claude_code in
   message.channel.typing() so users see "Bot is typing..." while waiting.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402


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
    stub.claude = MagicMock()
    stub.classifier.classify = AsyncMock(return_value="claude_code")
    stub.codex_client = None
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor._handle_claude_code = AsyncMock(
        return_value="Code analysis result."
    )
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._process_with_tools = AsyncMock(
        return_value=("Fallback response", False, False, [], False)
    )
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
# 1. is_error bug fix: error responses
# ---------------------------------------------------------------------------

class TestClaudeCodeIsError:
    """When claude_code returns an error, is_error should be True so the
    error doesn't pollute session history."""

    async def test_error_response_sets_is_error(self):
        """Error response → saved to history for checkpoint-save (Round 14)."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Claude Code failed (exit 1):\ncommand not found"
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # is_error=True saves error to history (checkpoint-save, Round 14)
        stub.sessions.remove_last_message.assert_not_called()
        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1
        assert "[Previous request encountered an error before tool execution.]" in assistant_saves[0][0][2]

    async def test_error_response_still_sends_to_user(self):
        """Error response should still be sent to the user, even with is_error."""
        stub = _make_bot_stub()
        msg = _make_message()
        error_text = "Claude Code failed (exit 1):\ncommand not found"
        stub.tool_executor._handle_claude_code = AsyncMock(return_value=error_text)
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # The error should be sent to the user via _send_chunked
        stub._send_chunked.assert_called_once()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "Claude Code failed" in sent_text

    async def test_error_response_triggers_save(self):
        """When is_error=True, sessions.save() IS called for checkpoint-save (Round 14)."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Claude Code failed (exit 1):\nerror"
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review code", "chan-1")

        stub.sessions.save.assert_called()
        stub.sessions.prune.assert_called()

    async def test_unknown_host_sets_is_error(self):
        """'Unknown or disallowed host' error → is_error=True, saved for checkpoint."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Unknown or disallowed host: desktop"
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review code", "chan-1")

        # Error saved to history for checkpoint-save (Round 14)
        stub.sessions.remove_last_message.assert_not_called()
        stub.sessions.save.assert_called()

    async def test_success_saves_normally(self):
        """Successful claude_code response should save normally."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="The function processes user input and validates it."
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain the function", "chan-1")

        # Successful response should be saved
        stub.sessions.add_message.assert_any_call(
            "chan-1", "assistant",
            "The function processes user input and validates it.",
        )
        stub.sessions.save.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Configurable max_output_chars
# ---------------------------------------------------------------------------

class TestClaudeCodeMaxOutputChars:
    """_handle_claude_code should respect the max_output_chars parameter."""

    @pytest.fixture
    def executor(self):
        """Create a minimal ToolExecutor-like object for testing."""
        from src.tools.executor import ToolExecutor
        config = MagicMock()
        config.hosts = {"desktop": MagicMock(address="10.0.0.2", ssh_user="root", os="linux")}
        config.ssh_key_path = "/app/.ssh/id_ed25519"
        config.ssh_known_hosts_path = "/app/.ssh/known_hosts"
        config.command_timeout_seconds = 30
        config.memory_path = None
        return ToolExecutor(config)

    async def test_default_truncation_at_3000(self, executor):
        """Without max_output_chars, output should be truncated at 3000 chars."""
        long_output = "x" * 5000
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        assert len(result) < 5000
        assert "[... truncated ...]" in result
        # Default 3000: first 1500 + truncation notice + last 1500
        assert result.startswith("x" * 1500)
        assert result.endswith("x" * 1500)

    async def test_custom_max_output_chars_8000(self, executor):
        """With max_output_chars=8000, more output should be preserved."""
        long_output = "y" * 10000
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
                "max_output_chars": 8000,
            })
        assert "[... truncated ...]" in result
        # 8000: first 4000 + truncation notice + last 4000
        assert result.startswith("y" * 4000)
        assert result.endswith("y" * 4000)

    async def test_short_output_not_truncated_default(self, executor):
        """Short output should pass through unchanged regardless of limit."""
        short_output = "The function does X."
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, short_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        assert result == short_output

    async def test_short_output_not_truncated_custom(self, executor):
        """Short output should pass through unchanged even with custom limit."""
        short_output = "The function does X."
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, short_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
                "max_output_chars": 8000,
            })
        assert result == short_output

    async def test_output_at_limit_not_truncated(self, executor):
        """Output exactly at max_output_chars should NOT be truncated."""
        exact_output = "z" * 3000
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, exact_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        assert result == exact_output
        assert "[... truncated ...]" not in result

    async def test_output_one_over_limit_truncated(self, executor):
        """Output one char over max_output_chars should be truncated."""
        over_output = "a" * 3001
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, over_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        assert "[... truncated ...]" in result

    async def test_routing_passes_higher_limit(self):
        """The routing path should pass max_output_chars=8000 to _handle_claude_code."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain this code", "chan-1")

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["max_output_chars"] == 8000

    async def test_error_output_not_truncated(self, executor):
        """Failed command output (exit != 0) has its own truncation, not max_output_chars."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "error " * 1000)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        # Error path uses output[-2000:], not max_output_chars
        assert result.startswith("Claude Code failed")


# ---------------------------------------------------------------------------
# 3. Typing indicator
# ---------------------------------------------------------------------------

class TestClaudeCodeTypingIndicator:
    """The claude_code routing path should show a typing indicator."""

    async def test_typing_called_during_claude_code(self):
        """message.channel.typing() should be called during _handle_claude_code."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain this code", "chan-1")

        # typing() should have been called
        msg.channel.typing.assert_called_once()

    async def test_typing_not_called_for_chat(self):
        """Chat path should NOT use typing (streaming handles its own UX)."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Hello!")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1")

        msg.channel.typing.assert_not_called()

    async def test_typing_not_called_for_task(self):
        """Task path should NOT use typing (streaming handles its own UX)."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.classifier.classify = AsyncMock(return_value="task")
        stub._process_with_tools = AsyncMock(
            return_value=("Done", False, False, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        msg.channel.typing.assert_not_called()

    async def test_typing_exits_before_fallback_to_codex(self):
        """When claude -p fails, typing should end before Codex fallback starts."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Codex fallback")
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Claude Code failed (exit 1):\nerror"
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "review code", "chan-1")

        # Typing was used during claude -p call
        msg.channel.typing.assert_called_once()
        # Codex fallback ran after typing ended
        stub.codex_client.chat.assert_called_once()

    async def test_typing_on_keyword_bypass_route(self):
        """Keyword-bypassed claude_code messages should also show typing."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        # "refactor" keyword bypasses classifier to claude_code
        await stub._handle_message_inner(msg, "refactor the routing module", "chan-1")

        msg.channel.typing.assert_called_once()
        stub.tool_executor._handle_claude_code.assert_called_once()
