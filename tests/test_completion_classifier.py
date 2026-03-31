"""Tests for the completion classifier (Rounds 1-4).

Round 1: Removed old regex continuation, added stub, wired call site, fixed max_tokens.
Round 2: Implemented real LLM classifier call with parsing, error handling, logging.
Round 3: Edge case tests for tier interaction, max_continuations, premature failure
         ordering, promise detection for first-response, continuation message format.
Round 4: Integration-style tests — full _process_with_tools flow with real
         _classify_completion bound, circuit breaker interaction, 3-continuation
         cycles with interleaved tool calls, premature failure + classifier ordering.

Tests cover:
1. Removed items are gone (Round 1)
2. Kept items still present (Round 1)
3. max_tokens forwarding (Round 1)
4. Response parsing (COMPLETE, INCOMPLETE, reason extraction, edge cases)
5. Full _classify_completion method (API call, timeout, errors, CircuitOpenError)
6. Classifier system prompt and user message format
7. Tier interaction edge cases (Round 3)
8. Integration tests with real classifier bound (Round 4)
"""
from __future__ import annotations

import asyncio
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    HeimdallBot,
    _CONTINUATION_MSG,
    _PROMISE_PATTERNS,
    _PROMISE_CHAT_EXEMPTIONS,
    detect_promise_without_action,
    detect_premature_failure,
)
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


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


# ---------------------------------------------------------------------------
# 7. Edge case tests for tier interaction (Round 3)
# ---------------------------------------------------------------------------

# Helpers for tool-loop edge case tests.  These create a minimal bot stub
# wired with the REAL _process_with_tools so we can exercise the full
# tool-loop flow (tier 1 → tier 2 → tier 3) without mocking the loop itself.


def _make_loop_bot_stub():
    """Minimal HeimdallBot stub with real _process_with_tools bound."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "You are a bot."
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.codex_client = MagicMock()
    stub.codex_client.chat_with_tools = AsyncMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="OK")
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[{"name": "test"}])
    stub._build_system_prompt = MagicMock(return_value="You are a bot.")
    stub._pending_files = {}
    # Bind real methods
    stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)
    stub._track_recent_action = HeimdallBot._track_recent_action.__get__(stub)
    # classifier is an AsyncMock so tests can control it
    stub._classify_completion = AsyncMock(return_value=(True, ""))
    return stub


def _make_loop_message(content="deploy the app"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = "chan-edge"
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None),
    ))
    msg.channel.send = AsyncMock(return_value=AsyncMock())
    msg.author = MagicMock()
    msg.author.id = "12345"
    msg.content = content
    msg.reply = AsyncMock()
    msg.webhook_id = None
    return msg


def _tool_resp(tool_calls, text=""):
    return LLMResponse(text=text, tool_calls=tool_calls, stop_reason="tool_use")


def _text_resp(text="Done."):
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn")


class TestClassifierOnlyWhenToolsUsed:
    """Tier 3 classifier must NOT fire when no tools were used (first-response)."""

    @pytest.mark.asyncio
    async def test_classifier_not_called_for_pure_text_response(self):
        """When LLM returns text without using any tools, classifier is skipped."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message()
        # LLM returns text on first call — no tools used
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_text_resp("Hello! How can I help?"),
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "Hello! How can I help?"
        assert tools_used == []
        # Classifier should NOT have been called — no tools were used
        stub._classify_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_classifier_called_when_tools_were_used(self):
        """When tools were used and LLM returns text, classifier fires."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message()
        tc = ToolCall(id="t1", name="run_command", input={"command": "ls"})
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Listed files."),
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "Listed files."
        assert tools_used == ["run_command"]
        # Classifier SHOULD have been called — tools were used
        stub._classify_completion.assert_called_once()
        call_args = stub._classify_completion.call_args
        assert call_args[0][0] == "deploy the app"  # user_message
        assert call_args[0][1] == "Listed files."     # response_text
        assert call_args[0][2] == ["run_command"]      # tools_used


class TestMaxContinuationsRespected:
    """Classifier should stop injecting continuations after max_continuations (3)."""

    @pytest.mark.asyncio
    async def test_max_three_continuations(self):
        """After 3 INCOMPLETE judgments, the 4th text response is returned as-is."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("build, test, and deploy")

        tc = ToolCall(id="t1", name="run_command", input={"command": "build"})

        # Sequence: tool call, text (INCOMPLETE), text (INCOMPLETE),
        # text (INCOMPLETE), text (classifier not called — cap reached)
        # Each INCOMPLETE triggers a continuation, so the loop retries.
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),                   # iter 0: tool call
            _text_resp("Built."),               # iter 1: text → classifier → INCOMPLETE
            _text_resp("Tested."),              # iter 2: text → classifier → INCOMPLETE
            _text_resp("Deployed."),            # iter 3: text → classifier → INCOMPLETE
            _text_resp("All done finally."),    # iter 4: text → classifier NOT called (cap=3)
        ])
        # Classifier always says INCOMPLETE
        stub._classify_completion = AsyncMock(return_value=(False, "more work needed"))

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        # After 3 continuations, the 4th text response is returned
        assert text == "All done finally."
        # Classifier was called exactly 3 times (the max_continuations cap)
        assert stub._classify_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_on_first_check_returns_immediately(self):
        """If classifier says COMPLETE on first check, no continuation injected."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("check disk")
        tc = ToolCall(id="t1", name="check_disk", input={})
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Disk is 42% full."),
        ])
        stub._classify_completion = AsyncMock(return_value=(True, ""))

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "Disk is 42% full."
        stub._classify_completion.assert_called_once()


class TestPrematureFailureBeforeClassifier:
    """Tier 2 (premature failure) must fire before Tier 3 (classifier)."""

    @pytest.mark.asyncio
    async def test_premature_failure_fires_before_classifier(self):
        """When premature failure is detected, it retries BEFORE classifier runs."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("deploy the app to production")
        tc = ToolCall(id="t1", name="run_command", input={"command": "deploy"})

        # The text contains a premature failure pattern (error + gave up)
        # "couldn't complete" matches _FAILURE_PATTERNS[0]
        failure_text = (
            "I tried to deploy but couldn't complete the deployment. "
            "The server appears to be having issues."
        )

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),           # iter 0: tool call
            _text_resp(failure_text),   # iter 1: premature failure text
            _text_resp("Deployed!"),    # iter 2: success after retry
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        # Premature failure fired first — the classifier only ran on the
        # second text response ("Deployed!")
        assert text == "Deployed!"
        # Classifier was called once (for "Deployed!" after premature failure retry)
        stub._classify_completion.assert_called_once()
        # The classifier saw the successful response, not the failure
        call_args = stub._classify_completion.call_args
        assert call_args[0][1] == "Deployed!"

    @pytest.mark.asyncio
    async def test_premature_failure_only_fires_once(self):
        """Premature failure detection fires once per request (flag-based)."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("fix the server")
        tc = ToolCall(id="t1", name="run_command", input={"command": "fix"})

        failure_text = (
            "I couldn't complete the task due to network issues. "
            "The server doesn't seem to be responding properly."
        )

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),           # iter 0: tool call
            _text_resp(failure_text),   # iter 1: premature failure → retry
            _text_resp(failure_text),   # iter 2: same failure again, but flag is set
        ])
        # Classifier says COMPLETE so loop exits
        stub._classify_completion = AsyncMock(return_value=(True, ""))

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        # Second failure text was returned (premature failure only fires once,
        # then classifier sees it and says COMPLETE)
        assert text == failure_text


class TestPromiseDetectionFirstResponse:
    """Tier 1 (promise detection) still fires for first-response promises with no tools."""

    @pytest.mark.asyncio
    async def test_promise_detector_fires_for_first_response(self):
        """When LLM promises action without calling tools, promise detector fires."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("check the server")

        # First response: "I'll check the server now" — promise without tools
        # Second response after retry: uses a tool
        tc = ToolCall(id="t1", name="run_command", input={"command": "uptime"})
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _text_resp("I'll check the server now and get back to you."),  # promise, no tools
            _tool_resp([tc]),                                              # retry → tool call
            _text_resp("Server uptime is 42 days."),                       # final text
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        # Promise detector fired, causing retry. Tool was then called.
        assert text == "Server uptime is 42 days."
        assert "run_command" in tools_used
        # Classifier was called for the final text (tools were used)
        stub._classify_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_promise_detector_does_not_fire_after_tools(self):
        """Promise detection only fires when tools_used_in_loop is empty (Tier 1)."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("deploy it")
        tc = ToolCall(id="t1", name="run_command", input={"command": "deploy"})

        # After tool use, a promise-like text should NOT trigger promise detector.
        # It should go to the classifier instead.
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("I'll continue deploying the rest of the servers."),
        ])
        # Classifier says COMPLETE — so the promise-like text is returned
        stub._classify_completion = AsyncMock(return_value=(True, ""))

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        # Text returned as-is — promise detector did NOT fire (tools were used)
        assert "I'll continue" in text
        # Classifier was called (it's the tier that handles post-tool responses)
        stub._classify_completion.assert_called_once()


class TestContinuationMessageFormat:
    """Verify the continuation message is injected correctly (not as assistant text)."""

    @pytest.mark.asyncio
    async def test_incomplete_with_reason_injects_targeted_message(self):
        """INCOMPLETE with reason → dynamic developer message injected."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("build and deploy")
        tc = ToolCall(id="t1", name="run_command", input={"command": "build"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Built the app!"),     # INCOMPLETE → continuation
            _text_resp("Deployed the app!"),  # COMPLETE
        ])
        stub._classify_completion = AsyncMock(side_effect=[
            (False, "deployment not performed"),  # first: INCOMPLETE
            (True, ""),                           # second: COMPLETE
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "Deployed the app!"
        # Verify the continuation message was injected with reason
        # The chat_with_tools second call should have the developer message
        second_call_messages = stub.codex_client.chat_with_tools.call_args_list[1]
        messages_arg = second_call_messages.kwargs.get("messages", second_call_messages.args[0] if second_call_messages.args else [])
        # Find the developer continuation message
        dev_msgs = [m for m in messages_arg if m.get("role") == "developer"
                    and "You are not done" in m.get("content", "")]
        assert len(dev_msgs) == 1
        assert "deployment not performed" in dev_msgs[0]["content"]
        assert "Continue with tool calls now" in dev_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_incomplete_without_reason_uses_default_continuation(self):
        """INCOMPLETE without reason → uses default _CONTINUATION_MSG."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("check everything")
        tc = ToolCall(id="t1", name="run_command", input={"command": "check"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Started checking."),  # INCOMPLETE, no reason
            _text_resp("All checks passed."),  # COMPLETE
        ])
        stub._classify_completion = AsyncMock(side_effect=[
            (False, ""),    # first: INCOMPLETE, no reason
            (True, ""),     # second: COMPLETE
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "All checks passed."
        # Verify default continuation was injected
        second_call_messages = stub.codex_client.chat_with_tools.call_args_list[1]
        messages_arg = second_call_messages.kwargs.get("messages", second_call_messages.args[0] if second_call_messages.args else [])
        # Should contain _CONTINUATION_MSG content
        cont_msgs = [m for m in messages_arg if m.get("role") == "developer"
                     and "continue executing" in m.get("content", "").lower()]
        assert len(cont_msgs) >= 1

    @pytest.mark.asyncio
    async def test_assistant_text_not_appended_on_continuation(self):
        """When classifier says INCOMPLETE, the assistant's text must NOT be saved
        to the messages list as an assistant message."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("deploy everything")
        tc = ToolCall(id="t1", name="run_command", input={"command": "build"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Built it."),            # INCOMPLETE → continuation
            _text_resp("Now deployed."),         # COMPLETE
        ])
        stub._classify_completion = AsyncMock(side_effect=[
            (False, "deployment missing"),
            (True, ""),
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "Now deployed."
        # Check that "Built it." was NOT appended as an assistant message
        # in the third call to chat_with_tools
        third_call_messages = stub.codex_client.chat_with_tools.call_args_list[2]
        messages_arg = third_call_messages.kwargs.get("messages", third_call_messages.args[0] if third_call_messages.args else [])
        assistant_msgs = [m for m in messages_arg
                          if isinstance(m.get("content"), str)
                          and m.get("role") == "assistant"
                          and m.get("content") == "Built it."]
        assert len(assistant_msgs) == 0, "INCOMPLETE text should NOT be saved as assistant message"


class TestPromisePatternsAndExemptionsIntact:
    """Verify _PROMISE_PATTERNS and _PROMISE_CHAT_EXEMPTIONS are intact and
    used by detect_promise_without_action()."""

    def test_promise_patterns_exist_and_nonempty(self):
        assert len(_PROMISE_PATTERNS) >= 6, "Expected at least 6 promise patterns"
        for p in _PROMISE_PATTERNS:
            assert isinstance(p, re.Pattern)

    def test_promise_chat_exemptions_exist_and_nonempty(self):
        assert len(_PROMISE_CHAT_EXEMPTIONS) >= 4, "Expected at least 4 exemptions"
        for p in _PROMISE_CHAT_EXEMPTIONS:
            assert isinstance(p, re.Pattern)

    def test_promise_patterns_catch_ill_do(self):
        """'I'll check the server' should match promise patterns."""
        assert detect_promise_without_action("I'll check the server now", []) is True

    def test_promise_patterns_catch_im_doing(self):
        """'I'm deploying the app' should match promise patterns."""
        assert detect_promise_without_action("I'm deploying the app now for you", []) is True

    def test_promise_exemption_thinking(self):
        """'I'm thinking about it' should be exempt (chat, not a promise)."""
        assert detect_promise_without_action("I'm thinking about the best approach", []) is False

    def test_promise_exemption_refusal(self):
        """'I can't do that' should be exempt."""
        assert detect_promise_without_action("I can't do that because it's not allowed", []) is False

    def test_promise_detector_returns_false_when_tools_used(self):
        """Promise detector immediately returns False when tools_used is truthy."""
        assert detect_promise_without_action("I'll deploy the app now", ["run_command"]) is False

    def test_promise_detector_returns_false_for_short_text(self):
        """Very short text is not checked for promises."""
        assert detect_promise_without_action("I'll do it", []) is False  # < 15 chars


class TestDetectPrematureFailureIntact:
    """Verify detect_premature_failure() still works correctly as Tier 2."""

    def test_premature_failure_requires_tools_used(self):
        """Returns False when no tools were used (Tier 1 handles that)."""
        assert detect_premature_failure(
            "The connection was refused and I couldn't complete the task.",
            [],
        ) is False

    def test_premature_failure_catches_connection_refused(self):
        """Catches 'connection refused' when tools were used."""
        assert detect_premature_failure(
            "I tried running the command but the connection refused. I can't proceed.",
            ["run_command"],
        ) is True

    def test_premature_failure_ignores_short_text(self):
        """Very short text is not checked."""
        assert detect_premature_failure("Failed.", ["run_command"]) is False


class TestClassifierWithMultipleToolCalls:
    """Test classifier behavior when multiple tools are used across iterations."""

    @pytest.mark.asyncio
    async def test_all_tool_names_passed_to_classifier(self):
        """Classifier receives the accumulated list of ALL tools used."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("check disk and memory")
        tc1 = ToolCall(id="t1", name="check_disk", input={})
        tc2 = ToolCall(id="t2", name="check_memory", input={})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc1]),                          # iter 0: check_disk
            _tool_resp([tc2]),                          # iter 1: check_memory
            _text_resp("Disk 42%, Memory 8GB free."),   # iter 2: final text
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert tools_used == ["check_disk", "check_memory"]
        stub._classify_completion.assert_called_once()
        call_args = stub._classify_completion.call_args
        assert call_args[0][2] == ["check_disk", "check_memory"]

    @pytest.mark.asyncio
    async def test_classifier_not_called_during_tool_use_iterations(self):
        """Classifier only fires on text-only responses, not during tool iterations."""
        stub = _make_loop_bot_stub()
        msg = _make_loop_message("run three commands")
        tc1 = ToolCall(id="t1", name="cmd1", input={})
        tc2 = ToolCall(id="t2", name="cmd2", input={})
        tc3 = ToolCall(id="t3", name="cmd3", input={})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc1]),
            _tool_resp([tc2]),
            _tool_resp([tc3]),
            _text_resp("All three commands done."),
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        # Classifier only called once — at the end, not during tool iterations
        stub._classify_completion.assert_called_once()


# ---------------------------------------------------------------------------
# 8. Integration tests with real _classify_completion bound (Round 4)
#
# These tests bind the REAL _classify_completion + _parse_classifier_response
# onto the bot stub so the full chain runs: tool loop → classifier API call
# (mocked codex_client.chat) → parse → continuation decision.  This catches
# wiring bugs between the call site and the method implementation.
# ---------------------------------------------------------------------------


def _make_integration_bot_stub():
    """Bot stub with REAL _classify_completion + _parse_classifier_response bound.

    Unlike _make_loop_bot_stub() (which mocks _classify_completion as AsyncMock),
    this helper binds the real implementations so the full classifier flow runs:
    codex_client.chat_with_tools (tool loop) + codex_client.chat (classifier).

    codex_client.chat is an AsyncMock whose return_value/side_effect controls
    what the classifier sees.  codex_client.chat_with_tools drives the tool loop.
    """
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "You are a bot."
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.codex_client = MagicMock()
    stub.codex_client.chat_with_tools = AsyncMock()
    # chat is used by the classifier — leave as AsyncMock for test control
    stub.codex_client.chat = AsyncMock(return_value="COMPLETE")
    stub.skill_manager = MagicMock()
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="OK")
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[{"name": "test"}])
    stub._build_system_prompt = MagicMock(return_value="You are a bot.")
    stub._pending_files = {}
    # Bind REAL methods so the full classifier path executes
    stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)
    stub._track_recent_action = HeimdallBot._track_recent_action.__get__(stub)
    stub._classify_completion = HeimdallBot._classify_completion.__get__(stub)
    stub._parse_classifier_response = HeimdallBot._parse_classifier_response
    stub._CLASSIFIER_SYSTEM_PROMPT = HeimdallBot._CLASSIFIER_SYSTEM_PROMPT
    return stub


class TestIntegrationClassifierComplete:
    """Integration: tool loop → real classifier call → COMPLETE → response returned."""

    @pytest.mark.asyncio
    async def test_tools_then_complete_response(self):
        """Basic flow: tool call → text → classifier says COMPLETE → return."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("check disk usage")
        tc = ToolCall(id="t1", name="run_command", input={"command": "df -h"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Disk is 42% full."),
        ])
        stub.codex_client.chat = AsyncMock(return_value="COMPLETE")

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "Disk is 42% full."
        assert tools_used == ["run_command"]
        # Classifier API was called once via codex_client.chat
        stub.codex_client.chat.assert_called_once()
        call_kw = stub.codex_client.chat.call_args.kwargs
        assert call_kw["max_tokens"] == 50
        assert "check disk usage" in call_kw["messages"][0]["content"]
        assert "run_command" in call_kw["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_tools_then_complete(self):
        """Multiple tool iterations → text → classifier COMPLETE → return."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("check disk and memory")
        tc1 = ToolCall(id="t1", name="check_disk", input={})
        tc2 = ToolCall(id="t2", name="check_memory", input={})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc1]),
            _tool_resp([tc2]),
            _text_resp("Disk 42%, Memory 8GB free."),
        ])
        stub.codex_client.chat = AsyncMock(return_value="COMPLETE")

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "Disk 42%, Memory 8GB free."
        assert tools_used == ["check_disk", "check_memory"]
        # Classifier user message should include both tool names
        user_msg = stub.codex_client.chat.call_args.kwargs["messages"][0]["content"]
        assert "check_disk, check_memory" in user_msg


class TestIntegrationClassifierIncomplete:
    """Integration: classifier says INCOMPLETE → continuation injected → loop continues."""

    @pytest.mark.asyncio
    async def test_incomplete_then_complete(self):
        """Classifier says INCOMPLETE on first text → continuation → then COMPLETE."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("build and deploy the app")
        tc = ToolCall(id="t1", name="run_command", input={"command": "make build"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Built the app successfully!"),
            _text_resp("App deployed to production."),
        ])
        # Classifier: first call INCOMPLETE, second call COMPLETE
        stub.codex_client.chat = AsyncMock(side_effect=[
            "INCOMPLETE: deployment not performed",
            "COMPLETE",
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "App deployed to production."
        # Classifier called twice (once per text response)
        assert stub.codex_client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_incomplete_reason_in_continuation_message(self):
        """The classifier reason appears in the continuation developer message."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("build, test, and deploy")
        tc = ToolCall(id="t1", name="run_command", input={"command": "build"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Built!"),
            _text_resp("All done."),
        ])
        stub.codex_client.chat = AsyncMock(side_effect=[
            "INCOMPLETE: testing and deployment not performed",
            "COMPLETE",
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "All done."
        # Verify the continuation message was injected into messages
        second_cwt_call = stub.codex_client.chat_with_tools.call_args_list[2]
        msgs = second_cwt_call.kwargs.get("messages", second_cwt_call.args[0] if second_cwt_call.args else [])
        dev_msgs = [m for m in msgs if m.get("role") == "developer"
                    and "You are not done" in m.get("content", "")]
        assert len(dev_msgs) == 1
        assert "testing and deployment not performed" in dev_msgs[0]["content"]
        assert "Continue with tool calls now." in dev_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_incomplete_with_interleaved_tool_calls(self):
        """INCOMPLETE → continuation → model calls more tools → final text."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("build and deploy")
        tc1 = ToolCall(id="t1", name="run_command", input={"command": "make"})
        tc2 = ToolCall(id="t2", name="run_command", input={"command": "deploy"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc1]),                    # tool: build
            _text_resp("Built!"),                 # text → classifier: INCOMPLETE
            _tool_resp([tc2]),                    # tool: deploy (after continuation)
            _text_resp("Built and deployed!"),    # text → classifier: COMPLETE
        ])
        stub.codex_client.chat = AsyncMock(side_effect=[
            "INCOMPLETE: deployment not performed",
            "COMPLETE",
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "Built and deployed!"
        # Both tool calls were tracked
        assert tools_used == ["run_command", "run_command"]
        # The second classifier call should see both tool calls
        second_chat_call = stub.codex_client.chat.call_args_list[1]
        user_msg = second_chat_call.kwargs["messages"][0]["content"]
        assert "run_command, run_command" in user_msg


class TestIntegrationFullContinuationCycle:
    """Integration: full 3-continuation cycle with tool calls between."""

    @pytest.mark.asyncio
    async def test_three_incomplete_then_cap_reached(self):
        """3 INCOMPLETE judgments → 4th text returned without classifier call."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("full deploy pipeline")
        tc = ToolCall(id="t1", name="run_command", input={"command": "step"})

        # Pattern: tool → text(INCOMPLETE) → text(INCOMPLETE) → text(INCOMPLETE) → text(returned)
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Step 1 done."),
            _text_resp("Step 2 done."),
            _text_resp("Step 3 done."),
            _text_resp("Step 4 done."),
        ])
        stub.codex_client.chat = AsyncMock(side_effect=[
            "INCOMPLETE: step 2 not done",
            "INCOMPLETE: step 3 not done",
            "INCOMPLETE: step 4 not done",
            # No 4th call — cap reached
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        # 4th text is returned because continuation_count (3) >= max_continuations (3)
        assert text == "Step 4 done."
        # Classifier called exactly 3 times
        assert stub.codex_client.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_three_continuations_with_tools_between(self):
        """3 INCOMPLETE with new tool calls between each → full pipeline execution."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("build, test, deploy, verify")
        tc_build = ToolCall(id="t1", name="run_command", input={"command": "build"})
        tc_test = ToolCall(id="t2", name="run_command", input={"command": "test"})
        tc_deploy = ToolCall(id="t3", name="run_command", input={"command": "deploy"})
        tc_verify = ToolCall(id="t4", name="run_command", input={"command": "verify"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc_build]),                       # tool: build
            _text_resp("Built!"),                         # INCOMPLETE
            _tool_resp([tc_test]),                        # tool: test
            _text_resp("Built and tested!"),              # INCOMPLETE
            _tool_resp([tc_deploy]),                      # tool: deploy
            _text_resp("Built, tested, deployed!"),       # INCOMPLETE
            _tool_resp([tc_verify]),                      # tool: verify (after cap)
            _text_resp("All 4 steps complete."),          # cap reached → returned
        ])
        stub.codex_client.chat = AsyncMock(side_effect=[
            "INCOMPLETE: testing, deployment, verification missing",
            "INCOMPLETE: deployment, verification missing",
            "INCOMPLETE: verification missing",
            # No 4th call — cap reached
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "All 4 steps complete."
        assert tools_used == ["run_command", "run_command", "run_command", "run_command"]
        assert stub.codex_client.chat.call_count == 3


class TestIntegrationFailOpen:
    """Integration: classifier errors → fail-open → response returned as-is."""

    @pytest.mark.asyncio
    async def test_classifier_api_error_returns_response(self):
        """When classifier API call raises, response is returned (fail-open)."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("check status")
        tc = ToolCall(id="t1", name="run_command", input={"command": "status"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Status looks good."),
        ])
        # Classifier API error
        stub.codex_client.chat = AsyncMock(side_effect=RuntimeError("API down"))

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "Status looks good."
        assert is_error is False  # Not a tool loop error — classifier failure is transparent
        assert tools_used == ["run_command"]

    @pytest.mark.asyncio
    async def test_classifier_timeout_returns_response(self):
        """When classifier times out, response is returned (fail-open)."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("run diagnostics")
        tc = ToolCall(id="t1", name="run_command", input={"command": "diag"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Diagnostics complete."),
        ])

        # Slow classifier that will timeout
        async def slow_chat(**kwargs):
            await asyncio.sleep(60)
            return "INCOMPLETE"
        stub.codex_client.chat = slow_chat

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "Diagnostics complete."

    @pytest.mark.asyncio
    async def test_classifier_circuit_breaker_open(self):
        """When CircuitOpenError is raised by classifier, fail-open to COMPLETE."""
        from src.llm.circuit_breaker import CircuitOpenError

        stub = _make_integration_bot_stub()
        msg = _make_loop_message("deploy app")
        tc = ToolCall(id="t1", name="run_command", input={"command": "deploy"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Deployed successfully."),
        ])
        # Circuit breaker open
        stub.codex_client.chat = AsyncMock(
            side_effect=CircuitOpenError("codex_api", 30.0)
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "Deployed successfully."

    @pytest.mark.asyncio
    async def test_classifier_connection_error_returns_response(self):
        """When classifier raises ConnectionError, fail-open."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("run checks")
        tc = ToolCall(id="t1", name="run_command", input={"command": "check"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("All checks passed."),
        ])
        stub.codex_client.chat = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "All checks passed."

    @pytest.mark.asyncio
    async def test_classifier_gibberish_response_treated_as_complete(self):
        """When classifier returns gibberish, fail-open as COMPLETE."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("run report")
        tc = ToolCall(id="t1", name="run_command", input={"command": "report"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Report generated."),
        ])
        stub.codex_client.chat = AsyncMock(
            return_value="Well, I think it might be partially done but I'm not sure"
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "Report generated."

    @pytest.mark.asyncio
    async def test_no_codex_client_skips_classifier(self):
        """When codex_client is None, classifier is skipped entirely."""
        stub = _make_integration_bot_stub()
        stub.codex_client = None
        msg = _make_loop_message("test")

        # We can't use chat_with_tools when codex_client is None, but
        # _classify_completion is called AFTER tool loop — the point is that
        # _classify_completion handles codex_client=None gracefully.
        # Use a separate test of the method directly.
        is_complete, reason = await HeimdallBot._classify_completion(
            stub, "test", "response", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""


class TestIntegrationPrematureFailureBeforeClassifier:
    """Integration: premature failure (Tier 2) fires before classifier (Tier 3)."""

    @pytest.mark.asyncio
    async def test_premature_failure_retries_then_classifier_judges(self):
        """Premature failure triggers retry → retry text goes to classifier."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("restart the service")
        tc = ToolCall(id="t1", name="run_command", input={"command": "restart"})

        failure_text = (
            "I tried to restart the service but couldn't complete the operation. "
            "The process seems to be locked."
        )
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),            # tool call
            _text_resp(failure_text),    # premature failure → retry
            _text_resp("Service restarted successfully."),  # after retry
        ])
        stub.codex_client.chat = AsyncMock(return_value="COMPLETE")

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "Service restarted successfully."
        # Classifier saw the successful response, not the failure
        assert stub.codex_client.chat.call_count == 1
        user_msg = stub.codex_client.chat.call_args.kwargs["messages"][0]["content"]
        assert "Service restarted successfully." in user_msg

    @pytest.mark.asyncio
    async def test_premature_failure_once_then_classifier_incomplete(self):
        """Premature failure fires once (flag-based), then classifier takes over."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("deploy and verify")
        tc = ToolCall(id="t1", name="run_command", input={"command": "deploy"})

        failure_text = (
            "I couldn't complete the deployment due to a timeout. "
            "The server isn't responding."
        )
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),            # tool call
            _text_resp(failure_text),    # premature failure → retry (flag set)
            _text_resp(failure_text),    # same failure again — flag already set, goes to classifier
        ])
        # Classifier says COMPLETE on the second failure (genuine failure report)
        stub.codex_client.chat = AsyncMock(return_value="COMPLETE")

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        # Second failure text returned — premature failure only fires once
        assert text == failure_text
        # Classifier was called once (on the second failure text)
        assert stub.codex_client.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_premature_failure_then_incomplete_then_complete(self):
        """Full chain: premature failure → retry → classifier INCOMPLETE → continuation → COMPLETE."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("deploy and verify")
        tc = ToolCall(id="t1", name="run_command", input={"command": "deploy"})

        failure_text = (
            "I tried but couldn't complete the deployment. "
            "Network issues encountered."
        )
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),                              # tool call
            _text_resp(failure_text),                      # premature failure → retry
            _text_resp("Deployed!"),                       # classifier → INCOMPLETE
            _text_resp("Deployed and verified!"),          # classifier → COMPLETE
        ])
        stub.codex_client.chat = AsyncMock(side_effect=[
            "INCOMPLETE: verification not performed",
            "COMPLETE",
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "Deployed and verified!"
        assert stub.codex_client.chat.call_count == 2


class TestIntegrationSystemPromptOverride:
    """Integration: system_prompt_override parameter works with classifier."""

    @pytest.mark.asyncio
    async def test_system_prompt_override_does_not_affect_classifier(self):
        """Classifier uses its own system prompt regardless of override."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("check server")
        tc = ToolCall(id="t1", name="run_command", input={"command": "uptime"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Server up 42 days."),
        ])
        stub.codex_client.chat = AsyncMock(return_value="COMPLETE")

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(
                msg, [], system_prompt_override="Custom override prompt",
            )

        assert text == "Server up 42 days."
        # Classifier used its own system prompt, not the override
        classifier_system = stub.codex_client.chat.call_args.kwargs["system"]
        assert classifier_system == HeimdallBot._CLASSIFIER_SYSTEM_PROMPT
        assert "Custom override prompt" not in classifier_system


class TestIntegrationChatOnlyNoClassifier:
    """Integration: pure chat (no tools) → classifier NOT called."""

    @pytest.mark.asyncio
    async def test_chat_only_skips_classifier(self):
        """When LLM returns text without any tool calls, classifier is not invoked."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("hello, how are you?")

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_text_resp("I'm doing well, thanks!"),
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert text == "I'm doing well, thanks!"
        assert tools_used == []
        # codex_client.chat should NOT have been called (classifier skipped)
        stub.codex_client.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_latency_for_chat(self):
        """Chat-only responses have zero classifier overhead (chat never called)."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("what time is it?")

        # Slow classifier that would timeout if called
        async def slow_chat(**kwargs):
            await asyncio.sleep(60)
            return "INCOMPLETE"
        stub.codex_client.chat = slow_chat

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_text_resp("I don't have a clock, but I can check for you."),
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            # This should return immediately — classifier not called for chat
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert "don't have a clock" in text


class TestIntegrationClassifierMessageContent:
    """Integration: verify exact content passed to classifier API call."""

    @pytest.mark.asyncio
    async def test_classifier_receives_correct_user_task(self):
        """Classifier user message contains the original Discord message content."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("deploy app to prod-server-3 with rolling update")
        tc = ToolCall(id="t1", name="run_command", input={"command": "deploy"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Deployed."),
        ])
        stub.codex_client.chat = AsyncMock(return_value="COMPLETE")

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        user_msg = stub.codex_client.chat.call_args.kwargs["messages"][0]["content"]
        assert "deploy app to prod-server-3 with rolling update" in user_msg
        assert "User's task:" in user_msg
        assert "Tools called: run_command" in user_msg
        assert "Assistant's response: Deployed." in user_msg

    @pytest.mark.asyncio
    async def test_classifier_sees_accumulated_tools(self):
        """After multiple tool iterations, classifier sees ALL tool names."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("run full pipeline")
        tc1 = ToolCall(id="t1", name="build_app", input={})
        tc2 = ToolCall(id="t2", name="run_tests", input={})
        tc3 = ToolCall(id="t3", name="deploy_prod", input={})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc1]),
            _tool_resp([tc2]),
            _tool_resp([tc3]),
            _text_resp("Pipeline complete."),
        ])
        stub.codex_client.chat = AsyncMock(return_value="COMPLETE")

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert tools_used == ["build_app", "run_tests", "deploy_prod"]
        user_msg = stub.codex_client.chat.call_args.kwargs["messages"][0]["content"]
        assert "build_app, run_tests, deploy_prod" in user_msg

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls_all_tracked(self):
        """When LLM returns multiple tool calls in one iteration, all are tracked."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("check all servers")
        tc1 = ToolCall(id="t1", name="check_server_a", input={})
        tc2 = ToolCall(id="t2", name="check_server_b", input={})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc1, tc2]),  # concurrent tool calls
            _text_resp("Both servers healthy."),
        ])
        stub.codex_client.chat = AsyncMock(return_value="COMPLETE")

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        assert tools_used == ["check_server_a", "check_server_b"]
        user_msg = stub.codex_client.chat.call_args.kwargs["messages"][0]["content"]
        assert "check_server_a, check_server_b" in user_msg


class TestIntegrationEdgeCases:
    """Integration edge cases: empty responses, None text, etc."""

    @pytest.mark.asyncio
    async def test_empty_llm_response_text(self):
        """When LLM returns empty text, classifier still works (empty string passed)."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("do something")
        tc = ToolCall(id="t1", name="run_command", input={"command": "echo"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            LLMResponse(text="", tool_calls=[], stop_reason="end_turn"),
        ])
        stub.codex_client.chat = AsyncMock(return_value="COMPLETE")

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        # Empty text → fallback response, but classifier was called
        assert stub.codex_client.chat.call_count == 1
        # The classifier received "" for response_text
        user_msg = stub.codex_client.chat.call_args.kwargs["messages"][0]["content"]
        assert "Assistant's response: " in user_msg

    @pytest.mark.asyncio
    async def test_classifier_error_on_first_then_success_on_second(self):
        """First INCOMPLETE via classifier → continuation → classifier error on 2nd → fail-open."""
        stub = _make_integration_bot_stub()
        msg = _make_loop_message("build and deploy")
        tc = ToolCall(id="t1", name="run_command", input={"command": "build"})

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_resp([tc]),
            _text_resp("Built!"),               # classifier: INCOMPLETE
            _text_resp("Built and deployed!"),   # classifier: error → fail-open
        ])
        stub.codex_client.chat = AsyncMock(side_effect=[
            "INCOMPLETE: deployment missing",
            RuntimeError("API crashed"),
        ])

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        # Second response returned because classifier errored → fail-open
        assert text == "Built and deployed!"
        assert stub.codex_client.chat.call_count == 2
