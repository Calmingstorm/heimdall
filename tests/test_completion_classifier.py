"""Tests for the completion classifier (Rounds 1-2).

Round 1: Removed old regex continuation, added stub, wired call site, fixed max_tokens.
Round 2: Implemented real LLM classifier call with parsing, error handling, logging.

Tests cover:
1. Removed items are gone (Round 1)
2. Kept items still present (Round 1)
3. max_tokens forwarding (Round 1)
4. Response parsing (COMPLETE, INCOMPLETE, reason extraction, edge cases)
5. Full _classify_completion method (API call, timeout, errors, CircuitOpenError)
6. Classifier system prompt and user message format
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    HeimdallBot,
    _CONTINUATION_MSG,
)


# ---------------------------------------------------------------------------
# 1. Removed items are gone (Round 1)
# ---------------------------------------------------------------------------


class TestRemovedItems:
    """Verify old regex-based continuation items were deleted."""

    def test_no_checkpoint_patterns(self):
        import src.discord.client as mod
        assert not hasattr(mod, "_CHECKPOINT_PATTERNS")

    def test_no_continuation_max_chars(self):
        import src.discord.client as mod
        assert not hasattr(mod, "_CONTINUATION_MAX_CHARS")

    def test_no_is_mid_task_checkpoint(self):
        import src.discord.client as mod
        assert not hasattr(mod, "_is_mid_task_checkpoint")

    def test_no_should_continue_task(self):
        import src.discord.client as mod
        assert not hasattr(mod, "_should_continue_task")


# ---------------------------------------------------------------------------
# 2. Kept items still present (Round 1)
# ---------------------------------------------------------------------------


class TestKeptItems:
    """Items that must survive the removal."""

    def test_continuation_msg_exists(self):
        assert _CONTINUATION_MSG["role"] == "developer"
        assert "continue" in _CONTINUATION_MSG["content"].lower()

    def test_classify_completion_is_method(self):
        assert hasattr(HeimdallBot, "_classify_completion")

    def test_classifier_system_prompt_exists(self):
        assert hasattr(HeimdallBot, "_CLASSIFIER_SYSTEM_PROMPT")
        prompt = HeimdallBot._CLASSIFIER_SYSTEM_PROMPT
        assert "COMPLETE" in prompt
        assert "INCOMPLETE" in prompt
        assert "completion judge" in prompt.lower()

    def test_parse_classifier_response_is_static(self):
        assert hasattr(HeimdallBot, "_parse_classifier_response")


# ---------------------------------------------------------------------------
# 3. max_tokens forwarding in CodexChatClient.chat() (Round 1)
# ---------------------------------------------------------------------------


class TestMaxTokensForwarding:
    """Verify that chat() passes max_tokens to the API body."""

    @pytest.mark.asyncio
    async def test_max_tokens_added_to_body(self):
        from src.llm.openai_codex import CodexChatClient

        client = MagicMock(spec=CodexChatClient)
        client.model = "gpt-4"
        client.max_tokens = None
        client.auth = MagicMock()
        client.auth.get_access_token = AsyncMock(return_value="tok")
        client.auth.get_account_id = MagicMock(return_value=None)
        client._convert_messages = MagicMock(return_value=[])

        captured_body = {}

        async def fake_stream(headers, body):
            captured_body.update(body)
            return "ok"

        client._stream_request = fake_stream

        result = await CodexChatClient.chat(
            client, messages=[], system="sys", max_tokens=10,
        )
        assert result == "ok"
        assert captured_body.get("max_output_tokens") == 10

    @pytest.mark.asyncio
    async def test_no_max_tokens_when_none(self):
        from src.llm.openai_codex import CodexChatClient

        client = MagicMock(spec=CodexChatClient)
        client.model = "gpt-4"
        client.max_tokens = None
        client.auth = MagicMock()
        client.auth.get_access_token = AsyncMock(return_value="tok")
        client.auth.get_account_id = MagicMock(return_value=None)
        client._convert_messages = MagicMock(return_value=[])

        captured_body = {}

        async def fake_stream(headers, body):
            captured_body.update(body)
            return "ok"

        client._stream_request = fake_stream

        result = await CodexChatClient.chat(
            client, messages=[], system="sys",
        )
        assert "max_output_tokens" not in captured_body

    @pytest.mark.asyncio
    async def test_instance_max_tokens_used_as_fallback(self):
        from src.llm.openai_codex import CodexChatClient

        client = MagicMock(spec=CodexChatClient)
        client.model = "gpt-4"
        client.max_tokens = 500
        client.auth = MagicMock()
        client.auth.get_access_token = AsyncMock(return_value="tok")
        client.auth.get_account_id = MagicMock(return_value=None)
        client._convert_messages = MagicMock(return_value=[])

        captured_body = {}

        async def fake_stream(headers, body):
            captured_body.update(body)
            return "ok"

        client._stream_request = fake_stream

        result = await CodexChatClient.chat(
            client, messages=[], system="sys",
        )
        assert captured_body.get("max_output_tokens") == 500


# ---------------------------------------------------------------------------
# 4. Response parsing (_parse_classifier_response)
# ---------------------------------------------------------------------------


class TestParseClassifierResponse:
    """Test the static response parser in isolation."""

    def test_complete_uppercase(self):
        is_complete, reason = HeimdallBot._parse_classifier_response("COMPLETE")
        assert is_complete is True
        assert reason == ""

    def test_complete_lowercase(self):
        is_complete, reason = HeimdallBot._parse_classifier_response("complete")
        assert is_complete is True
        assert reason == ""

    def test_complete_mixed_case(self):
        is_complete, reason = HeimdallBot._parse_classifier_response("Complete")
        assert is_complete is True
        assert reason == ""

    def test_complete_with_trailing_text(self):
        """'COMPLETELY done' starts with COMPLETE → treated as COMPLETE."""
        is_complete, reason = HeimdallBot._parse_classifier_response("COMPLETELY done")
        assert is_complete is True
        assert reason == ""

    def test_incomplete_bare(self):
        is_complete, reason = HeimdallBot._parse_classifier_response("INCOMPLETE")
        assert is_complete is False
        assert reason == ""

    def test_incomplete_with_colon_reason(self):
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "INCOMPLETE: deployment not performed"
        )
        assert is_complete is False
        assert reason == "deployment not performed"

    def test_incomplete_with_dash_reason(self):
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "INCOMPLETE - not verified"
        )
        assert is_complete is False
        assert reason == "not verified"

    def test_incomplete_with_em_dash_reason(self):
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "INCOMPLETE — verification step missing"
        )
        assert is_complete is False
        assert reason == "verification step missing"

    def test_incomplete_with_em_dash_no_spaces(self):
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "INCOMPLETE—no spaces around em dash"
        )
        assert is_complete is False
        assert reason == "no spaces around em dash"

    def test_incomplete_lowercase(self):
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "incomplete: partial"
        )
        assert is_complete is False
        assert reason == "partial"

    def test_incomplete_mixed_case(self):
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "Incomplete: needs more work"
        )
        assert is_complete is False
        assert reason == "needs more work"

    def test_empty_string_fail_open(self):
        is_complete, reason = HeimdallBot._parse_classifier_response("")
        assert is_complete is True
        assert reason == ""

    def test_none_fail_open(self):
        is_complete, reason = HeimdallBot._parse_classifier_response(None)
        assert is_complete is True
        assert reason == ""

    def test_gibberish_fail_open(self):
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "I think the response is mostly done"
        )
        assert is_complete is True
        assert reason == ""

    def test_whitespace_only_fail_open(self):
        is_complete, reason = HeimdallBot._parse_classifier_response("   \n  ")
        assert is_complete is True
        assert reason == ""

    def test_leading_whitespace_stripped(self):
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "  INCOMPLETE: trailing spaces too  "
        )
        assert is_complete is False
        assert reason == "trailing spaces too"

    def test_incomplete_colon_preferred_over_dash(self):
        """When both colon and dash are present, colon (first checked) wins."""
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "INCOMPLETE: first - second"
        )
        assert is_complete is False
        # Colon comes first in the separator list, so everything after colon
        assert reason == "first - second"

    def test_incomplete_no_separator_empty_reason(self):
        """INCOMPLETE with no separator → empty reason."""
        is_complete, reason = HeimdallBot._parse_classifier_response(
            "INCOMPLETE the deploy was skipped"
        )
        assert is_complete is False
        assert reason == ""


# ---------------------------------------------------------------------------
# 5. Full _classify_completion method tests
# ---------------------------------------------------------------------------


def _make_classifier_bot(chat_return="COMPLETE", chat_side_effect=None):
    """Helper: create a MagicMock(spec=HeimdallBot) wired for classifier tests.

    Sets codex_client.chat, the classifier system prompt, and the static parser
    so that _classify_completion works correctly through MagicMock.
    """
    bot = MagicMock(spec=HeimdallBot)
    bot.codex_client = AsyncMock()
    if chat_side_effect is not None:
        bot.codex_client.chat = AsyncMock(side_effect=chat_side_effect)
    else:
        bot.codex_client.chat = AsyncMock(return_value=chat_return)
    bot._CLASSIFIER_SYSTEM_PROMPT = HeimdallBot._CLASSIFIER_SYSTEM_PROMPT
    # MagicMock(spec=...) intercepts staticmethod lookups — bind the real one
    bot._parse_classifier_response = HeimdallBot._parse_classifier_response
    return bot


class TestClassifyCompletion:
    """Test the full async method with mocked codex_client."""

    @pytest.mark.asyncio
    async def test_returns_complete_on_complete_response(self):
        bot = _make_classifier_bot("COMPLETE")
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "deploy the app", "App deployed successfully.", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_returns_incomplete_with_reason(self):
        bot = _make_classifier_bot("INCOMPLETE: deployment not performed")
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "build and deploy", "Built the app!", ["run_command"],
        )
        assert is_complete is False
        assert reason == "deployment not performed"

    @pytest.mark.asyncio
    async def test_returns_incomplete_bare(self):
        bot = _make_classifier_bot("INCOMPLETE")
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "check status", "Checking...", ["run_command"],
        )
        assert is_complete is False
        assert reason == ""

    @pytest.mark.asyncio
    async def test_fail_open_on_timeout(self):
        bot = MagicMock(spec=HeimdallBot)
        bot.codex_client = AsyncMock()

        async def slow_chat(**kwargs):
            await asyncio.sleep(60)
            return "INCOMPLETE"

        bot.codex_client.chat = slow_chat
        bot._CLASSIFIER_SYSTEM_PROMPT = HeimdallBot._CLASSIFIER_SYSTEM_PROMPT
        bot._parse_classifier_response = HeimdallBot._parse_classifier_response

        # The method wraps in wait_for(timeout=10), so this should timeout
        # and return COMPLETE (fail-open)
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "task", "response", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_fail_open_on_generic_exception(self):
        bot = _make_classifier_bot(chat_side_effect=RuntimeError("API down"))
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "task", "response", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_fail_open_on_circuit_open_error(self):
        from src.llm.circuit_breaker import CircuitOpenError

        bot = _make_classifier_bot(
            chat_side_effect=CircuitOpenError("codex_api", 30.0)
        )
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "task", "response", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_fail_open_on_connection_error(self):
        bot = _make_classifier_bot(chat_side_effect=ConnectionError("refused"))
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "task", "response", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_no_codex_client_returns_complete(self):
        """When codex_client is None, classifier skips and returns COMPLETE."""
        bot = MagicMock(spec=HeimdallBot)
        bot.codex_client = None

        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "task", "response", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_chat_called_with_correct_params(self):
        """Verify the classifier passes the right system prompt, user msg, and max_tokens."""
        bot = _make_classifier_bot("COMPLETE")

        await HeimdallBot._classify_completion(
            bot, "deploy app", "Deployed!", ["run_command", "read_file"],
        )

        bot.codex_client.chat.assert_called_once()
        call_kwargs = bot.codex_client.chat.call_args
        # Check system prompt
        assert call_kwargs.kwargs["system"] == HeimdallBot._CLASSIFIER_SYSTEM_PROMPT
        # Check max_tokens
        assert call_kwargs.kwargs["max_tokens"] == 50
        # Check user message contains task, tools, and response
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "deploy app" in user_msg
        assert "run_command, read_file" in user_msg
        assert "Deployed!" in user_msg

    @pytest.mark.asyncio
    async def test_user_message_format(self):
        """Verify the exact structure of the user message sent to the classifier."""
        bot = _make_classifier_bot("COMPLETE")

        await HeimdallBot._classify_completion(
            bot, "check disk usage", "Disk is 80% full.", ["run_command"],
        )

        call_kwargs = bot.codex_client.chat.call_args
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert user_msg.startswith("User's task: check disk usage")
        assert "\n\nTools called: run_command\n\n" in user_msg
        assert user_msg.endswith("Assistant's response: Disk is 80% full.")

    @pytest.mark.asyncio
    async def test_fail_open_on_empty_response(self):
        """Empty API response → fail-open as COMPLETE."""
        bot = _make_classifier_bot("")
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "task", "response", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_fail_open_on_gibberish_response(self):
        """Gibberish API response → fail-open as COMPLETE."""
        bot = _make_classifier_bot(
            "Well, I think the task is mostly done but maybe not"
        )
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "task", "response", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_incomplete_dash_separator(self):
        """INCOMPLETE with ' - ' separator extracts reason correctly."""
        bot = _make_classifier_bot("INCOMPLETE - not verified")
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "task", "response", ["run_command"],
        )
        assert is_complete is False
        assert reason == "not verified"

    @pytest.mark.asyncio
    async def test_fail_open_on_os_error(self):
        """OSError (e.g., network issue) → fail-open as COMPLETE."""
        bot = _make_classifier_bot(chat_side_effect=OSError("network unreachable"))
        is_complete, reason = await HeimdallBot._classify_completion(
            bot, "task", "response", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_multiple_tools_in_message(self):
        """Verify multiple tool names are comma-separated in the user message."""
        bot = _make_classifier_bot("COMPLETE")

        await HeimdallBot._classify_completion(
            bot, "deploy", "Done.",
            ["run_command", "read_file", "write_file", "run_script"],
        )

        call_kwargs = bot.codex_client.chat.call_args
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "run_command, read_file, write_file, run_script" in user_msg


# ---------------------------------------------------------------------------
# 6. Classifier prompt validation
# ---------------------------------------------------------------------------


class TestClassifierPrompt:
    """Verify the classifier system prompt has the expected content."""

    def test_prompt_mentions_complete_and_incomplete(self):
        prompt = HeimdallBot._CLASSIFIER_SYSTEM_PROMPT
        assert "COMPLETE" in prompt
        assert "INCOMPLETE" in prompt

    def test_prompt_mentions_failure_report(self):
        """Failure report after genuinely trying should count as COMPLETE."""
        prompt = HeimdallBot._CLASSIFIER_SYSTEM_PROMPT
        assert "failure report" in prompt.lower()

    def test_prompt_mentions_partial_completion(self):
        """Prompt explicitly mentions partial completion as INCOMPLETE."""
        prompt = HeimdallBot._CLASSIFIER_SYSTEM_PROMPT
        assert "part of what was asked" in prompt.lower() or "part of it" in prompt.lower()

    def test_prompt_not_too_long(self):
        """Classifier prompt should be concise — under 1000 chars."""
        assert len(HeimdallBot._CLASSIFIER_SYSTEM_PROMPT) < 1000
