"""Tests for LLM response secret scrubbing before Discord delivery.

Session R2-S2 identified a security gap: tool output was scrubbed before reaching
the LLM (scrub_output_secrets in _run_tool), but LLM responses were sent to Discord
without any scrubbing.  If the LLM echoed, reconstructed, or hallucinated a secret
in its response text, that secret would go straight to Discord.

scrub_response_secrets() closes this gap by applying the same tool-output patterns
plus additional natural-language patterns before any Discord API call.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    scrub_response_secrets,
    _RESPONSE_EXTRA_PATTERNS,
    HeimdallBot,
)
from src.llm.secret_scrubber import OUTPUT_SECRET_PATTERNS  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests for scrub_response_secrets()
# ---------------------------------------------------------------------------

class TestScrubResponseSecrets:
    """Verify scrub_response_secrets catches all known secret patterns."""

    # -- Tool-output patterns (inherited from scrub_output_secrets) --

    def test_scrubs_api_key_equals(self):
        text = "The config has api_key=sk-ant-abc123xyz456def789ghi"
        result = scrub_response_secrets(text)
        assert "sk-ant-abc" not in result
        assert "[REDACTED]" in result

    def test_scrubs_password_field(self):
        text = "Found password: hunter2intheconfig"
        result = scrub_response_secrets(text)
        assert "hunter2" not in result

    def test_scrubs_openai_key(self):
        text = "The key is sk-abcdefghijklmnopqrstuvwxyz1234567890"
        result = scrub_response_secrets(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result

    def test_scrubs_private_key_header(self):
        text = "I found this:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEpA..."
        result = scrub_response_secrets(text)
        assert "BEGIN RSA PRIVATE KEY" not in result

    def test_scrubs_database_url(self):
        text = "Database is at postgres://admin:supersecret@db.host:5432/prod"
        result = scrub_response_secrets(text)
        assert "supersecret" not in result

    def test_scrubs_mongodb_url(self):
        text = "mongodb+srv://user:pass123@cluster.example.net/db"
        result = scrub_response_secrets(text)
        assert "pass123" not in result

    def test_scrubs_token_field(self):
        text = "auth_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.longtoken"
        result = scrub_response_secrets(text)
        assert "eyJhbGci" not in result

    # -- Extra response patterns (natural language + Slack) --

    def test_scrubs_slack_bot_token(self):
        text = "The Slack bot token is xoxb-123456789012-abcdefgh"
        result = scrub_response_secrets(text)
        assert "xoxb-123456789012" not in result

    def test_scrubs_slack_user_token(self):
        text = "Use this token: xoxp-98765432-abcdefghij"
        result = scrub_response_secrets(text)
        assert "xoxp-98765432" not in result

    def test_scrubs_natural_language_password(self):
        text = "The server password is mySecretPass123"
        result = scrub_response_secrets(text)
        assert "mySecretPass123" not in result

    def test_scrubs_my_password_is(self):
        text = "my password for the server is hunter2abc"
        result = scrub_response_secrets(text)
        assert "hunter2abc" not in result

    def test_scrubs_password_was(self):
        text = "The old passwd was OldSecret99"
        result = scrub_response_secrets(text)
        assert "OldSecret99" not in result

    # -- Safe text preservation --

    def test_leaves_normal_text_alone(self):
        text = "The server is running normally. CPU usage is at 42%."
        assert scrub_response_secrets(text) == text

    def test_leaves_code_blocks_alone(self):
        text = "```python\nprint('hello world')\n```"
        assert scrub_response_secrets(text) == text

    def test_leaves_short_password_word(self):
        """The word 'password' in normal context should not trigger scrubbing."""
        text = "You should change your password regularly."
        assert scrub_response_secrets(text) == text

    def test_leaves_empty_string(self):
        assert scrub_response_secrets("") == ""

    def test_scrubs_multiple_secrets_in_one_text(self):
        text = (
            "Found these:\n"
            "password: secretvalue123\n"
            "api_key=sk-abcdefghijklmnopqrstuvwxyz1234567890\n"
            "The database is postgres://admin:dbpass@host/db"
        )
        result = scrub_response_secrets(text)
        assert "secretvalue123" not in result
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result
        assert "dbpass" not in result
        assert "Found these:" in result  # non-secret text preserved

    def test_preserves_surrounding_text(self):
        text = "Before secret. password: hunter22 After secret."
        result = scrub_response_secrets(text)
        assert "hunter22" not in result
        assert "Before secret." in result
        assert "After secret." in result


# ---------------------------------------------------------------------------
# Pattern coverage tests
# ---------------------------------------------------------------------------

class TestPatternCoverage:
    """Ensure scrub_response_secrets covers all known pattern sets."""

    def test_response_extra_patterns_exist(self):
        assert len(_RESPONSE_EXTRA_PATTERNS) >= 2  # Slack + natural language

    def test_output_patterns_applied(self):
        """scrub_response_secrets should apply all OUTPUT_SECRET_PATTERNS."""
        # Test with a pattern unique to OUTPUT_SECRET_PATTERNS (private key header)
        text = "BEGIN OPENSSH PRIVATE KEY"
        result = scrub_response_secrets(text)
        assert "OPENSSH PRIVATE KEY" not in result

    def test_extra_patterns_applied(self):
        """scrub_response_secrets should apply _RESPONSE_EXTRA_PATTERNS."""
        # Slack token (only in extra patterns, not in OUTPUT_SECRET_PATTERNS)
        text = "xoxs-abc123-def456"
        result = scrub_response_secrets(text)
        assert "xoxs-abc123" not in result


# ---------------------------------------------------------------------------
# Integration: scrubbing in _handle_message_inner
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal HeimdallBot stub for routing tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._system_prompt = "system prompt"
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.max_tool_iterations_chat = 30
    stub.config.tools.max_tool_iterations_loop = 100
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex chat response")
    stub.codex_client.chat_with_tools = AsyncMock(return_value=MagicMock())
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
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


class TestHandleMessageInnerScrubbing:
    """Verify _handle_message_inner scrubs responses before Discord delivery.

    All messages now route to "task" (no classifier). Tests mock _process_with_tools
    to return responses and verify scrubbing happens before Discord delivery.
    """

    async def test_codex_response_scrubbed_before_send(self):
        """Task response containing a secret should be scrubbed."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._process_with_tools = AsyncMock(
            return_value=("The password: supersecretvalue99", False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "what's the server password?", "chan-1")

        # The text sent to Discord should be scrubbed
        sent_text = stub._send_chunked.call_args[0][1]
        assert "supersecretvalue99" not in sent_text
        assert "[REDACTED]" in sent_text

    async def test_codex_response_scrubbed_in_history(self):
        """Scrubbed text should also be saved to session history."""
        stub = _make_bot_stub()
        msg = _make_message()
        # Include tools_used so the response gets saved to history
        # (task route skips saving tool-less responses to prevent poisoning)
        stub._process_with_tools = AsyncMock(
            return_value=("Found api_key=sk-ant-abc123xyz456def789ghi in config", False, False, ["read_file"], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check the config", "chan-1")

        # History should contain the scrubbed version
        assistant_saves = [c for c in stub.sessions.add_message.call_args_list if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1
        saved_text = assistant_saves[0][0][2]
        assert "sk-ant-abc123" not in saved_text

    async def test_safe_response_unchanged(self):
        """Normal responses without secrets should pass through unchanged."""
        stub = _make_bot_stub()
        msg = _make_message()
        original = "The server is running fine. CPU at 23%, disk at 45%."
        stub._process_with_tools = AsyncMock(
            return_value=(original, False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "how's the server?", "chan-1")

        sent_text = stub._send_chunked.call_args[0][1]
        assert sent_text == original

    async def test_process_with_tools_response_scrubbed(self):
        """Task response containing a secret should be scrubbed."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._process_with_tools = AsyncMock(
            return_value=("The database URL is postgres://admin:secret@host/db", False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "show me the database config", "chan-1")

        sent_text = stub._send_chunked.call_args[0][1]
        assert "secret@host" not in sent_text

    async def test_error_response_saved_to_history(self):
        """Error responses are saved as sanitized markers for checkpoint-save."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._process_with_tools = AsyncMock(
            return_value=("API overloaded", False, True, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Error saved as sanitized marker (not raw error text)
        stub.sessions.remove_last_message.assert_not_called()
        add_calls = [c for c in stub.sessions.add_message.call_args_list if c[0][1] == "assistant"]
        assert len(add_calls) == 1
        # Sanitized marker: no tools used -> "[Previous request encountered an error before tool execution.]"
        assert "error" in add_calls[0][0][2].lower()


# ---------------------------------------------------------------------------
# Integration: scrubbing in _process_with_tools (Codex path)
# ---------------------------------------------------------------------------

class TestStreamIterationScrubbing:
    """Verify Codex task responses are scrubbed before Discord delivery.

    These tests verify Codex path secret scrubbing via _handle_message_inner.
    """

    async def test_codex_task_response_with_secret_scrubbed(self):
        """When Codex task response contains a secret, it should be scrubbed
        before being sent to Discord via _send_chunked."""
        stub = _make_bot_stub()
        msg = _make_message()

        secret_text = "Here is the token: sk-abcdefghijklmnopqrstuvwxyz1234567890"
        stub._process_with_tools = AsyncMock(
            return_value=(secret_text, False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check the API key", "chan-1")

        sent_text = stub._send_chunked.call_args[0][1]
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in sent_text
        assert "[REDACTED]" in sent_text

    async def test_codex_task_response_with_password_scrubbed(self):
        """Codex task response with password pattern should be scrubbed."""
        stub = _make_bot_stub()
        msg = _make_message()

        secret_text = "x" * 40 + " password: supersecretvalue99 " + "y" * 40
        stub._process_with_tools = AsyncMock(
            return_value=(secret_text, False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "show config", "chan-1")

        sent_text = stub._send_chunked.call_args[0][1]
        assert "supersecretvalue99" not in sent_text
        assert "[REDACTED]" in sent_text
