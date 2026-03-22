"""Tests for llm/haiku_classifier.py — HaikuClassifier."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.llm.haiku_classifier import HaikuClassifier, _RetryableError
from src.llm.circuit_breaker import CircuitBreaker, CircuitOpenError


@pytest.fixture
def classifier():
    return HaikuClassifier(
        api_key="test-api-key",
        model="claude-haiku-4-5-20251001",
    )


def _make_haiku_response(text: str = "task", status: int = 200, *, json_data: dict | None = None):
    """Create a mock aiohttp session that returns a Haiku Messages API response.

    The session can be injected into classifier._session directly.
    """
    mock_resp = AsyncMock()
    mock_resp.status = status
    if json_data is not None:
        mock_resp.json = AsyncMock(return_value=json_data)
    else:
        mock_resp.json = AsyncMock(return_value={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
    mock_resp.text = AsyncMock(return_value=f"HTTP {status} error")

    mock_post_ctx = AsyncMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_ctx)
    mock_session.closed = False

    return mock_session


def _inject_session(classifier, mock_session):
    """Inject a mock session so _get_session() returns it directly."""
    classifier._session = mock_session


class TestClassifyCategories:
    """Test that all three valid categories are returned correctly."""

    async def test_classify_chat(self, classifier):
        _inject_session(classifier, _make_haiku_response("chat"))
        result = await classifier.classify("hello how are you?")
        assert result == "chat"

    async def test_classify_task(self, classifier):
        _inject_session(classifier, _make_haiku_response("task"))
        result = await classifier.classify("check disk usage on the server")
        assert result == "task"

    async def test_classify_claude_code(self, classifier):
        _inject_session(classifier, _make_haiku_response("claude_code"))
        result = await classifier.classify("review this Python function")
        assert result == "claude_code"

    async def test_classify_strips_whitespace(self, classifier):
        _inject_session(classifier, _make_haiku_response("  task\n "))
        result = await classifier.classify("restart the service")
        assert result == "task"

    async def test_classify_case_insensitive(self, classifier):
        _inject_session(classifier, _make_haiku_response("CHAT"))
        result = await classifier.classify("hey there")
        assert result == "chat"


class TestUnknownResponse:
    """Test that unknown/unexpected responses default to 'task'."""

    async def test_unknown_response_defaults_to_task(self, classifier):
        _inject_session(classifier, _make_haiku_response("question"))
        result = await classifier.classify("what's this?")
        assert result == "task"

    async def test_empty_response_defaults_to_task(self, classifier):
        _inject_session(classifier, _make_haiku_response(""))
        result = await classifier.classify("something")
        assert result == "task"

    async def test_multiword_response_defaults_to_task(self, classifier):
        _inject_session(classifier, _make_haiku_response("this is a task"))
        result = await classifier.classify("check the server")
        assert result == "task"


class TestSystemPrompt:
    """Test that the system prompt includes context-dependent sections."""

    def test_base_prompt_has_all_categories(self, classifier):
        prompt = classifier._build_system_prompt()
        assert "'chat'" in prompt
        assert "'claude_code'" in prompt
        assert "'task'" in prompt
        assert "ONLY one word" in prompt

    def test_skill_hints_injected(self, classifier):
        prompt = classifier._build_system_prompt(skill_hints="weather, image_gen")
        assert "weather, image_gen" in prompt
        assert "tools available" in prompt

    def test_skill_hints_empty_not_injected(self, classifier):
        prompt = classifier._build_system_prompt(skill_hints="")
        assert "tools available" not in prompt

    def test_recent_tool_use_context(self, classifier):
        prompt = classifier._build_system_prompt(has_recent_tool_use=True)
        assert "recently ran tool commands" in prompt
        assert "'and the desktop?'" in prompt

    def test_no_recent_tool_use_context(self, classifier):
        prompt = classifier._build_system_prompt(has_recent_tool_use=False)
        assert "recently ran tool commands" not in prompt

    def test_both_hints_and_tool_use(self, classifier):
        prompt = classifier._build_system_prompt(
            has_recent_tool_use=True,
            skill_hints="web_search, check_disk",
        )
        assert "web_search, check_disk" in prompt
        assert "recently ran tool commands" in prompt


class TestErrorHandling:
    """Test error handling and fallback behavior."""

    async def test_non_retryable_http_error_defaults_to_task(self, classifier):
        """Non-retryable errors (e.g. 400, 401) fail without retry."""
        _inject_session(classifier, _make_haiku_response("", status=400))
        result = await classifier.classify("check disk")
        assert result == "task"

    async def test_connection_error_defaults_to_task(self, classifier):
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))
        _inject_session(classifier, mock_session)
        result = await classifier.classify("check disk")
        assert result == "task"

    async def test_timeout_defaults_to_task(self, classifier):
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=asyncio.TimeoutError("Request timed out"))
        _inject_session(classifier, mock_session)
        result = await classifier.classify("check disk")
        assert result == "task"


class TestCircuitBreaker:
    """Test circuit breaker integration."""

    async def test_circuit_open_defaults_to_chat(self, classifier):
        """When Haiku is known down, route to chat (free Codex backend)."""
        classifier.breaker.check = MagicMock(
            side_effect=CircuitOpenError("haiku_classify", 45.0)
        )
        result = await classifier.classify("check disk on the server")
        assert result == "chat"

    async def test_circuit_records_success(self, classifier):
        classifier.breaker.record_success = MagicMock()
        _inject_session(classifier, _make_haiku_response("task"))
        await classifier.classify("check disk")
        classifier.breaker.record_success.assert_called_once()

    async def test_circuit_records_failure_on_error(self, classifier):
        classifier.breaker.record_failure = MagicMock()
        mock_session = MagicMock()
        mock_session.closed = False
        # Fails both attempts (original + retry), then records failure
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("fail"))
        _inject_session(classifier, mock_session)
        await classifier.classify("check disk")
        classifier.breaker.record_failure.assert_called_once()

    async def test_circuit_records_failure_on_http_error(self, classifier):
        """Retryable status (503) fails both attempts -> records failure."""
        classifier.breaker.record_failure = MagicMock()
        _inject_session(classifier, _make_haiku_response("", status=503))
        await classifier.classify("check disk")
        classifier.breaker.record_failure.assert_called_once()

    async def test_circuit_not_called_on_circuit_open(self, classifier):
        """When circuit is already open, don't record another failure."""
        classifier.breaker.check = MagicMock(
            side_effect=CircuitOpenError("haiku_classify", 30.0)
        )
        classifier.breaker.record_failure = MagicMock()
        await classifier.classify("anything")
        classifier.breaker.record_failure.assert_not_called()


class TestRequestFormat:
    """Test that the correct request is sent to the Anthropic Messages API."""

    async def test_request_body_format(self, classifier):
        captured_kwargs = {}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "task"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        def capture_post(url, **kwargs):
            captured_kwargs.update(kwargs)
            captured_kwargs["url"] = url
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session.closed = False
        _inject_session(classifier, mock_session)

        await classifier.classify("check disk")

        assert captured_kwargs["url"] == "https://api.anthropic.com/v1/messages"
        body = captured_kwargs["json"]
        assert body["model"] == "claude-haiku-4-5-20251001"
        assert body["max_tokens"] == 5
        assert body["temperature"] == 0.0
        assert body["messages"] == [{"role": "user", "content": "check disk"}]
        assert "system" in body

        headers = captured_kwargs["headers"]
        assert headers["x-api-key"] == "test-api-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["Content-Type"] == "application/json"

    async def test_default_model(self):
        c = HaikuClassifier(api_key="key")
        assert c.model == "claude-haiku-4-5-20251001"

    async def test_custom_model(self):
        c = HaikuClassifier(api_key="key", model="claude-haiku-4-5")
        assert c.model == "claude-haiku-4-5"


class TestConfigIntegration:
    """Test that AnthropicConfig.model default matches HaikuClassifier."""

    def test_anthropic_config_has_haiku_model_default(self):
        from src.config.schema import AnthropicConfig
        cfg = AnthropicConfig()
        assert cfg.model == "claude-haiku-4-5-20251001"

    def test_anthropic_config_custom_model(self):
        from src.config.schema import AnthropicConfig
        cfg = AnthropicConfig(model="claude-haiku-4-5")
        assert cfg.model == "claude-haiku-4-5"

    def test_anthropic_config_api_key(self):
        from src.config.schema import AnthropicConfig
        cfg = AnthropicConfig(api_key="sk-ant-test")
        assert cfg.api_key == "sk-ant-test"


class TestRetryLogic:
    """Test retry behavior for transient failures."""

    async def test_retries_on_503_then_succeeds(self, classifier):
        """503 triggers retry; second attempt succeeds."""
        # First call: 503, second call: 200
        resp_503 = AsyncMock()
        resp_503.status = 503
        resp_503.text = AsyncMock(return_value="overloaded")
        ctx_503 = AsyncMock()
        ctx_503.__aenter__ = AsyncMock(return_value=resp_503)
        ctx_503.__aexit__ = AsyncMock(return_value=False)

        resp_ok = AsyncMock()
        resp_ok.status = 200
        resp_ok.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "task"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
        ctx_ok = AsyncMock()
        ctx_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        ctx_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[ctx_503, ctx_ok])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("check disk")
        assert result == "task"
        assert mock_session.post.call_count == 2

    async def test_retries_on_connection_error_then_succeeds(self, classifier):
        """Connection error triggers retry; second attempt succeeds."""
        resp_ok = AsyncMock()
        resp_ok.status = 200
        resp_ok.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "chat"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
        ctx_ok = AsyncMock()
        ctx_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        ctx_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[
            aiohttp.ClientError("Connection refused"),
            ctx_ok,
        ])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("hello")
        assert result == "chat"
        assert mock_session.post.call_count == 2

    async def test_retries_on_timeout_then_succeeds(self, classifier):
        """Timeout triggers retry; second attempt succeeds."""
        resp_ok = AsyncMock()
        resp_ok.status = 200
        resp_ok.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "claude_code"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
        ctx_ok = AsyncMock()
        ctx_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        ctx_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[
            asyncio.TimeoutError(),
            ctx_ok,
        ])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("review code")
        assert result == "claude_code"

    async def test_both_attempts_fail_defaults_to_task(self, classifier):
        """When both attempts fail, records failure and returns 'task'."""
        classifier.breaker.record_failure = MagicMock()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("down"))
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("check disk")
        assert result == "task"
        classifier.breaker.record_failure.assert_called_once()

    async def test_no_retry_on_non_retryable_http(self, classifier):
        """Non-retryable HTTP errors (401, 400) are NOT retried."""
        _inject_session(classifier, _make_haiku_response("", status=401))
        result = await classifier.classify("check disk")
        assert result == "task"
        # Only 1 call — no retry for 401
        classifier._session.post.assert_called_once()

    async def test_retry_sleeps_1_second(self, classifier):
        """Retry delay is 1 second."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("down"))
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await classifier.classify("check disk")
        mock_sleep.assert_called_once_with(1)


class TestSessionReuse:
    """Test reusable aiohttp session behavior."""

    async def test_session_created_on_first_call(self, classifier):
        """_get_session() creates a new session when none exists."""
        assert classifier._session is None
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_cls.return_value = mock_instance
            session = await classifier._get_session()
            assert session is mock_instance
            mock_cls.assert_called_once()

    async def test_session_reused_on_second_call(self, classifier):
        """_get_session() returns the existing session if not closed."""
        mock_session = MagicMock()
        mock_session.closed = False
        classifier._session = mock_session

        with patch("aiohttp.ClientSession") as mock_cls:
            session = await classifier._get_session()
            assert session is mock_session
            mock_cls.assert_not_called()

    async def test_session_recreated_if_closed(self, classifier):
        """_get_session() creates a new session if the existing one is closed."""
        old_session = MagicMock()
        old_session.closed = True
        classifier._session = old_session

        with patch("aiohttp.ClientSession") as mock_cls:
            new_session = MagicMock()
            new_session.closed = False
            mock_cls.return_value = new_session
            session = await classifier._get_session()
            assert session is new_session

    async def test_close_closes_session(self, classifier):
        """close() closes the underlying session."""
        mock_session = AsyncMock()
        mock_session.closed = False
        classifier._session = mock_session
        await classifier.close()
        mock_session.close.assert_called_once()

    async def test_close_noop_when_no_session(self, classifier):
        """close() is a no-op when no session exists."""
        await classifier.close()  # Should not raise

    async def test_close_noop_when_already_closed(self, classifier):
        """close() is a no-op when session is already closed."""
        mock_session = AsyncMock()
        mock_session.closed = True
        classifier._session = mock_session
        await classifier.close()
        mock_session.close.assert_not_called()

    async def test_multiple_classifies_reuse_session(self, classifier):
        """Multiple classify calls reuse the same session."""
        mock_session = _make_haiku_response("task")
        _inject_session(classifier, mock_session)

        await classifier.classify("first")
        await classifier.classify("second")
        await classifier.classify("third")

        # All three calls should hit the same session
        assert mock_session.post.call_count == 3


class TestMissingResponseKey:
    """Test handling of Anthropic responses without expected keys."""

    async def test_missing_content_key_defaults_to_task(self, classifier):
        """When Anthropic returns error JSON without 'content' key."""
        _inject_session(classifier, _make_haiku_response(
            "", json_data={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
        ))
        result = await classifier.classify("check disk")
        assert result == "task"

    async def test_empty_content_array_defaults_to_task(self, classifier):
        """When Anthropic returns an empty content array."""
        _inject_session(classifier, _make_haiku_response(
            "", json_data={
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [],
                "model": "claude-haiku-4-5-20251001",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 0},
            }
        ))
        result = await classifier.classify("check disk")
        assert result == "task"


class TestEmptyApiKey:
    """Test behavior when API key is empty."""

    async def test_empty_api_key_still_attempts_request(self):
        """With empty API key, request proceeds but API returns 401 -> fallback to 'task'."""
        c = HaikuClassifier(api_key="", model="claude-haiku-4-5-20251001")
        _inject_session(c, _make_haiku_response("", status=401))
        result = await c.classify("check disk")
        assert result == "task"


class TestSystemPromptRegression:
    """Regression guard — verify the classification prompt content hasn't drifted."""

    # Hardcoded expected base prompt (no skill_hints, no has_recent_tool_use).
    # If you intentionally change the prompt, update this string.
    EXPECTED_BASE = (
        "Classify the user message as 'chat', 'claude_code', or 'task'. "
        "Reply with ONLY one word.\n"
        "- 'chat' = casual conversation, greetings, opinions, advice, "
        "simple questions answerable from general knowledge\n"
        "- 'claude_code' = code analysis, code review, explaining code or functions, "
        "debugging code, summarizing code changes, reading or "
        "searching through source files — ONLY when the task is pure code analysis "
        "with no need to post files, deploy, or interact with Discord/infrastructure\n"
        "- 'task' = ALWAYS 'task' if the message involves ANY of: "
        "git operations (commits, diffs, logs, reviews of specific commits), "
        "running commands on remote hosts, checking system status (disk/memory/CPU/services), "
        "deployments, Docker operations, Prometheus queries, restarting services, "
        "news, headlines, current events, what's happening, what's in the news, "
        "remembering/recalling/forgetting things, saving notes, anything needing "
        "SSH access, real-time monitoring data, or external APIs, "
        "generating/writing code AND posting/attaching/deploying it\n"
        "- Action directives like 'try again', 'go ahead', 'do it', 'proceed', "
        "'retry', 'run it', 'yes do that' = ALWAYS 'task' "
        "(they imply re-executing a previous action)"
    )

    EXPECTED_SKILL_SUFFIX = (
        "\n- The bot has these tools available: weather, image_gen\n"
        "If the user is REQUESTING information these tools provide (e.g. 'what's the weather', "
        "'pronounce this word', 'any news about X', 'generate an image of'), classify as 'task'.\n"
        "If the user is just TALKING ABOUT the topic casually (e.g. 'the weather sucks', "
        "'I was practicing pronunciation'), classify as 'chat'."
    )

    EXPECTED_TOOL_USE_SUFFIX = (
        "\n- CONTEXT: The user recently ran tool commands in this conversation. "
        "Short follow-ups like 'and the desktop?', 'what about X?', "
        "'same for Y', 'now check Z', 'how about memory?' are 'task' "
        "(they refer to the previous action). Only pure pleasantries "
        "like 'thanks', 'cool', 'ok' are 'chat'."
    )

    def test_base_prompt(self):
        c = HaikuClassifier(api_key="test")
        assert c._build_system_prompt() == self.EXPECTED_BASE

    def test_with_skill_hints(self):
        c = HaikuClassifier(api_key="test")
        expected = self.EXPECTED_BASE + self.EXPECTED_SKILL_SUFFIX
        assert c._build_system_prompt(skill_hints="weather, image_gen") == expected

    def test_with_recent_tool_use(self):
        c = HaikuClassifier(api_key="test")
        expected = self.EXPECTED_BASE + self.EXPECTED_TOOL_USE_SUFFIX
        assert c._build_system_prompt(has_recent_tool_use=True) == expected

    def test_with_both(self):
        c = HaikuClassifier(api_key="test")
        expected = self.EXPECTED_BASE + self.EXPECTED_SKILL_SUFFIX + self.EXPECTED_TOOL_USE_SUFFIX
        assert c._build_system_prompt(
            has_recent_tool_use=True, skill_hints="weather, image_gen",
        ) == expected


# ---------------------------------------------------------------------------
# Edge Case Tests — Round 8 hardening
# ---------------------------------------------------------------------------


class TestEdgeCaseMessages:
    """Test classifier with unusual message content."""

    async def test_empty_content_string(self, classifier):
        """Empty string should still be classified (API decides)."""
        _inject_session(classifier, _make_haiku_response("chat"))
        result = await classifier.classify("")
        assert result == "chat"

    async def test_whitespace_only_content(self, classifier):
        """Whitespace-only content should still be sent to API."""
        _inject_session(classifier, _make_haiku_response("chat"))
        result = await classifier.classify("   \n\t  ")
        assert result == "chat"

    async def test_very_long_message(self, classifier):
        """Messages >4000 chars should pass through without truncation."""
        long_msg = "check disk usage " * 300  # ~5100 chars
        captured = {}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "task"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1500, "output_tokens": 1},
        })
        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        def capture_post(url, **kwargs):
            captured.update(kwargs)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session.closed = False
        _inject_session(classifier, mock_session)

        result = await classifier.classify(long_msg)
        assert result == "task"
        # Verify full message was sent without truncation
        assert captured["json"]["messages"][0]["content"] == long_msg

    async def test_unicode_emoji_content(self, classifier):
        """Unicode and emoji in messages should pass through correctly."""
        _inject_session(classifier, _make_haiku_response("chat"))
        result = await classifier.classify("こんにちは 👋 how are you? 🤖")
        assert result == "chat"

    async def test_code_block_content(self, classifier):
        """Messages containing markdown code blocks."""
        msg = "```python\ndef hello():\n    print('hi')\n```\nReview this code"
        _inject_session(classifier, _make_haiku_response("claude_code"))
        result = await classifier.classify(msg)
        assert result == "claude_code"

    async def test_multiline_content(self, classifier):
        """Multi-line messages should be sent as-is."""
        msg = "Line 1\nLine 2\nLine 3\nCheck all servers"
        _inject_session(classifier, _make_haiku_response("task"))
        result = await classifier.classify(msg)
        assert result == "task"


class TestAllRetryableStatuses:
    """Verify ALL retryable HTTP status codes trigger retry."""

    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
    async def test_retryable_status_triggers_retry(self, classifier, status):
        """Status {status} should trigger one retry."""
        # First call: retryable error, second call: success
        resp_err = AsyncMock()
        resp_err.status = status
        resp_err.text = AsyncMock(return_value=f"HTTP {status}")
        ctx_err = AsyncMock()
        ctx_err.__aenter__ = AsyncMock(return_value=resp_err)
        ctx_err.__aexit__ = AsyncMock(return_value=False)

        resp_ok = AsyncMock()
        resp_ok.status = 200
        resp_ok.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "task"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
        ctx_ok = AsyncMock()
        ctx_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        ctx_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[ctx_err, ctx_ok])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("test")
        assert result == "task"
        assert mock_session.post.call_count == 2

    @pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 422])
    async def test_non_retryable_status_no_retry(self, classifier, status):
        """Status {status} should NOT trigger retry — fail immediately."""
        _inject_session(classifier, _make_haiku_response("", status=status))
        result = await classifier.classify("test")
        assert result == "task"
        classifier._session.post.assert_called_once()


class TestConcurrentClassification:
    """Test concurrent classify calls share the session safely."""

    async def test_concurrent_classifies_all_succeed(self, classifier):
        """Multiple concurrent classify() calls via asyncio.gather should all succeed."""
        _inject_session(classifier, _make_haiku_response("task"))

        results = await asyncio.gather(
            classifier.classify("msg 1"),
            classifier.classify("msg 2"),
            classifier.classify("msg 3"),
            classifier.classify("msg 4"),
            classifier.classify("msg 5"),
        )
        assert all(r == "task" for r in results)
        assert classifier._session.post.call_count == 5

    async def test_concurrent_mixed_results(self, classifier):
        """Concurrent calls can return different categories."""
        # Each call gets its own response via side_effect
        responses = []
        for text in ["chat", "task", "claude_code", "task", "chat"]:
            resp = AsyncMock()
            resp.status = 200
            resp.json = AsyncMock(return_value={
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": "claude-haiku-4-5-20251001",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 1},
            })
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            responses.append(ctx)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=responses)
        _inject_session(classifier, mock_session)

        results = await asyncio.gather(
            classifier.classify("hello"),
            classifier.classify("check disk"),
            classifier.classify("review code"),
            classifier.classify("deploy"),
            classifier.classify("thanks"),
        )
        assert results == ["chat", "task", "claude_code", "task", "chat"]


class TestCircuitBreakerFullCycle:
    """Test the complete circuit breaker lifecycle: closed→open→half_open→closed."""

    async def test_full_cycle_failure_to_recovery(self):
        """3 failures → circuit opens → timeout → half_open → success → closes."""
        classifier = HaikuClassifier(
            api_key="test-key", model="claude-haiku-4-5-20251001",
        )
        # Use a very short recovery timeout for testing
        classifier.breaker = CircuitBreaker(
            "haiku_classify", failure_threshold=3, recovery_timeout=0.1,
        )

        # --- Phase 1: 3 consecutive failures → circuit opens ---
        fail_session = MagicMock()
        fail_session.closed = False
        fail_session.post = MagicMock(side_effect=aiohttp.ClientError("down"))
        _inject_session(classifier, fail_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            for i in range(3):
                result = await classifier.classify(f"msg {i}")
                assert result == "task", f"Failure {i+1} should return 'task'"

        assert classifier.breaker.state == "open"

        # --- Phase 2: circuit is open → returns "chat" immediately ---
        result = await classifier.classify("should be blocked")
        assert result == "chat"

        # --- Phase 3: wait for recovery timeout → half_open ---
        await asyncio.sleep(0.15)  # slightly longer than 0.1s recovery
        assert classifier.breaker.state == "half_open"

        # --- Phase 4: successful probe → circuit closes ---
        _inject_session(classifier, _make_haiku_response("task"))
        result = await classifier.classify("probe message")
        assert result == "task"
        assert classifier.breaker.state == "closed"

        # --- Phase 5: subsequent calls work normally ---
        _inject_session(classifier, _make_haiku_response("chat"))
        result = await classifier.classify("hello again")
        assert result == "chat"

    async def test_half_open_probe_failure_reopens_circuit(self):
        """If the half_open probe fails, circuit reopens."""
        classifier = HaikuClassifier(api_key="test-key")
        classifier.breaker = CircuitBreaker(
            "haiku_classify", failure_threshold=3, recovery_timeout=0.1,
        )

        # 3 failures → open
        fail_session = MagicMock()
        fail_session.closed = False
        fail_session.post = MagicMock(side_effect=aiohttp.ClientError("down"))
        _inject_session(classifier, fail_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            for _ in range(3):
                await classifier.classify("fail")

        assert classifier.breaker.state == "open"

        # Wait for recovery
        await asyncio.sleep(0.15)
        assert classifier.breaker.state == "half_open"

        # Probe also fails → back to open
        fail_session2 = MagicMock()
        fail_session2.closed = False
        fail_session2.post = MagicMock(side_effect=aiohttp.ClientError("still down"))
        _inject_session(classifier, fail_session2)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("probe")
        assert result == "task"  # probe failed → "task"

        # Circuit should be open again (failure recorded bumps count to 4 ≥ 3)
        result = await classifier.classify("should be blocked again")
        assert result == "chat"


# ---------------------------------------------------------------------------
# Logging Verification Tests — Round 9 hardening
# ---------------------------------------------------------------------------


class TestLoggingBehavior:
    """Verify correct log messages and levels for each code path."""

    async def test_success_logs_debug(self, classifier):
        """Successful classification logs at INFO level with content preview."""
        _inject_session(classifier, _make_haiku_response("task"))
        with patch("src.llm.haiku_classifier.log") as mock_log:
            await classifier.classify("check disk on server")
        mock_log.info.assert_any_call(
            "Classified %r as %r via Haiku (skills=%s, recent_tools=%s)",
            "check disk on server", "task", False, False,
        )

    async def test_success_truncates_long_content_in_log(self, classifier):
        """Log preview of content is truncated to 80 chars."""
        long_msg = "a" * 100
        _inject_session(classifier, _make_haiku_response("chat"))
        with patch("src.llm.haiku_classifier.log") as mock_log:
            await classifier.classify(long_msg)
        mock_log.info.assert_any_call(
            "Classified %r as %r via Haiku (skills=%s, recent_tools=%s)",
            long_msg[:80], "chat", False, False,
        )

    async def test_retry_logs_info(self, classifier):
        """First transient failure logs at INFO level before retry."""
        resp_503 = AsyncMock()
        resp_503.status = 503
        resp_503.text = AsyncMock(return_value="overloaded")
        ctx_503 = AsyncMock()
        ctx_503.__aenter__ = AsyncMock(return_value=resp_503)
        ctx_503.__aexit__ = AsyncMock(return_value=False)

        resp_ok = AsyncMock()
        resp_ok.status = 200
        resp_ok.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "task"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
        ctx_ok = AsyncMock()
        ctx_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        ctx_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[ctx_503, ctx_ok])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock), \
             patch("src.llm.haiku_classifier.log") as mock_log:
            await classifier.classify("test")
        # info called twice: once for retry message, once for success
        assert mock_log.info.call_count == 2
        retry_call = mock_log.info.call_args_list[0][0]
        assert "attempt 1 failed" in retry_call[0]
        assert "retrying" in retry_call[0]

    async def test_circuit_open_logs_debug(self, classifier):
        """Circuit open fallback logs at DEBUG level."""
        classifier.breaker.check = MagicMock(
            side_effect=CircuitOpenError("haiku_classify", 30.0)
        )
        with patch("src.llm.haiku_classifier.log") as mock_log:
            await classifier.classify("anything")
        mock_log.debug.assert_any_call("Haiku circuit open — defaulting to 'chat'")

    async def test_failure_logs_warning_with_exc_info(self, classifier):
        """Permanent failure logs at WARNING with exc_info=True."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("down"))
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock), \
             patch("src.llm.haiku_classifier.log") as mock_log:
            await classifier.classify("check disk")
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args
        assert "failed" in call_args[0][0].lower()
        assert "task" in call_args[0][0].lower()
        assert call_args[1] == {"exc_info": True}

    async def test_invalid_category_no_warning_returns_task(self, classifier):
        """Unknown category from API returns 'task' without logging a warning."""
        _inject_session(classifier, _make_haiku_response("unknown_category"))
        with patch("src.llm.haiku_classifier.log") as mock_log:
            result = await classifier.classify("something")
        assert result == "task"
        # Should log info (success path) but NOT warning (not an error)
        mock_log.warning.assert_not_called()
        mock_log.info.assert_any_call(
            "Classified %r as %r via Haiku (skills=%s, recent_tools=%s)",
            "something", "unknown_category", False, False,
        )


# ---------------------------------------------------------------------------
# Keyword Bypass + Classifier Interaction — Round 9 hardening
# ---------------------------------------------------------------------------


class TestKeywordBypassInteraction:
    """Verify that keyword bypass correctly gates classifier calls.

    These tests exercise the routing logic from client.py:940-963 to ensure
    keyword-matched messages never reach the classifier, while non-keyword
    messages always go through classification.
    """

    async def test_keyword_match_skips_classify(self):
        """When is_task_by_keyword returns True, classifier.classify is NOT called."""
        from src.discord.routing import is_task_by_keyword

        classifier = HaikuClassifier(api_key="test-key")
        classifier.classify = AsyncMock(return_value="chat")

        # These should all match keywords
        keyword_msgs = [
            "restart nginx",
            "deploy to production",
            "check disk on server",
            "docker logs nginx",
            "try again",
            "go ahead",
        ]
        for msg in keyword_msgs:
            assert is_task_by_keyword(msg) is True, f"Expected keyword match for: {msg!r}"

    async def test_ambiguous_messages_need_classifier(self):
        """Ambiguous messages that DON'T match keywords need classification."""
        from src.discord.routing import is_task_by_keyword

        ambiguous_msgs = [
            "hello how are you?",
            "what's the weather like?",
            "tell me a joke",
            "how's memory looking?",
            "check on things",
            "what about the logs?",
        ]
        for msg in ambiguous_msgs:
            assert is_task_by_keyword(msg) is False, f"Unexpected keyword match for: {msg!r}"


# ---------------------------------------------------------------------------
# Timeout Edge Cases — Round 9 hardening
# ---------------------------------------------------------------------------


class TestTimeoutBehavior:
    """Verify timeout configuration and behavior."""

    async def test_timeout_value_is_10_seconds(self, classifier):
        """The timeout passed to aiohttp should be 10 seconds."""
        captured = {}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "task"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        def capture_post(url, **kwargs):
            captured.update(kwargs)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session.closed = False
        _inject_session(classifier, mock_session)

        await classifier.classify("test")
        assert captured["timeout"].total == 10

    async def test_timeout_on_first_attempt_retries(self, classifier):
        """Timeout on first attempt triggers retry, second attempt works."""
        resp_ok = AsyncMock()
        resp_ok.status = 200
        resp_ok.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "task"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
        ctx_ok = AsyncMock()
        ctx_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        ctx_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[
            asyncio.TimeoutError(),
            ctx_ok,
        ])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("test")
        assert result == "task"
        assert mock_session.post.call_count == 2

    async def test_timeout_on_both_attempts_defaults_to_task(self, classifier):
        """Timeout on both attempts records failure, returns 'task'."""
        classifier.breaker.record_failure = MagicMock()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("test")
        assert result == "task"
        classifier.breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# Anthropic-specific Response Edge Cases — Round 9 hardening
# ---------------------------------------------------------------------------


class TestAnthropicResponseEdgeCases:
    """Test edge cases specific to the Anthropic Messages API response format."""

    async def test_overloaded_error_529_is_retryable(self, classifier):
        """Anthropic overloaded_error (529) is retryable — retries once, then succeeds."""
        resp_529 = AsyncMock()
        resp_529.status = 529
        resp_529.text = AsyncMock(return_value="overloaded")
        ctx_529 = AsyncMock()
        ctx_529.__aenter__ = AsyncMock(return_value=resp_529)
        ctx_529.__aexit__ = AsyncMock(return_value=False)

        resp_ok = AsyncMock()
        resp_ok.status = 200
        resp_ok.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "task"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
        ctx_ok = AsyncMock()
        ctx_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        ctx_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[ctx_529, ctx_ok])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("test")
        assert result == "task"
        assert mock_session.post.call_count == 2  # retried once

    async def test_response_with_multiple_content_blocks(self, classifier):
        """Response with multiple content blocks uses the first one."""
        _inject_session(classifier, _make_haiku_response(
            "", json_data={
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [
                    {"type": "text", "text": "task"},
                    {"type": "text", "text": "extra content"},
                ],
                "model": "claude-haiku-4-5-20251001",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 2},
            }
        ))
        result = await classifier.classify("check disk")
        assert result == "task"

    async def test_response_with_leading_trailing_newlines(self, classifier):
        """Response text with newlines is stripped properly."""
        _inject_session(classifier, _make_haiku_response("\n\n chat \n"))
        result = await classifier.classify("hello")
        assert result == "chat"

    async def test_response_with_mixed_case_and_whitespace(self, classifier):
        """Response with mixed case + whitespace normalizes correctly."""
        _inject_session(classifier, _make_haiku_response("  Claude_Code  "))
        result = await classifier.classify("review this function")
        assert result == "claude_code"


# ---------------------------------------------------------------------------
# Anthropic-specific Error Codes — Round 10 hardening
# ---------------------------------------------------------------------------


class TestAnthropicSpecificErrors:
    """Test Anthropic-specific HTTP error codes from the official API docs.

    Anthropic returns these non-standard codes:
    - 402: billing_error (non-retryable)
    - 413: request_too_large (non-retryable)
    - 529: overloaded_error (retryable — added to _RETRYABLE_STATUSES in Round 10)
    """

    async def test_billing_error_402_no_retry(self, classifier):
        """402 billing_error is non-retryable — fails immediately."""
        _inject_session(classifier, _make_haiku_response("", status=402))
        result = await classifier.classify("test")
        assert result == "task"
        # Only 1 call — no retry for billing errors
        classifier._session.post.assert_called_once()

    async def test_request_too_large_413_no_retry(self, classifier):
        """413 request_too_large is non-retryable — fails immediately."""
        _inject_session(classifier, _make_haiku_response("", status=413))
        result = await classifier.classify("test")
        assert result == "task"
        classifier._session.post.assert_called_once()

    async def test_529_both_attempts_fail_defaults_to_task(self, classifier):
        """529 overloaded on both attempts → records failure, returns 'task'."""
        classifier.breaker.record_failure = MagicMock()
        _inject_session(classifier, _make_haiku_response("", status=529))
        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("test")
        assert result == "task"
        classifier.breaker.record_failure.assert_called_once()


class TestServerDisconnectedError:
    """Verify ServerDisconnectedError (subclass of ClientError) triggers retry."""

    async def test_server_disconnected_triggers_retry(self, classifier):
        """aiohttp.ServerDisconnectedError triggers retry as a ClientError subclass."""
        resp_ok = AsyncMock()
        resp_ok.status = 200
        resp_ok.json = AsyncMock(return_value={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "task"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 1},
        })
        ctx_ok = AsyncMock()
        ctx_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        ctx_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[
            aiohttp.ServerDisconnectedError("Server disconnected"),
            ctx_ok,
        ])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("test")
        assert result == "task"
        assert mock_session.post.call_count == 2  # retried once

    async def test_server_disconnected_both_fail_defaults_to_task(self, classifier):
        """ServerDisconnectedError on both attempts → records failure, returns 'task'."""
        classifier.breaker.record_failure = MagicMock()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            side_effect=aiohttp.ServerDisconnectedError("Server disconnected"),
        )
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("test")
        assert result == "task"
        classifier.breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# Response Parsing Edge Cases — Round 13 hardening
# ---------------------------------------------------------------------------


class TestResponseParsingEdgeCases:
    """Test edge cases in parsing the Anthropic Messages API response JSON."""

    async def test_malformed_json_response_defaults_to_task(self, classifier):
        """When API returns non-JSON (e.g. HTML error page), falls back to 'task'.

        aiohttp's resp.json() raises ContentTypeError for non-JSON content types.
        This should be caught by the outer except and default to 'task'.
        """
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            side_effect=aiohttp.ContentTypeError(
                MagicMock(), MagicMock(), message="Attempt to decode JSON with unexpected mimetype: text/html"
            )
        )
        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_post_ctx)
        _inject_session(classifier, mock_session)

        result = await classifier.classify("check disk")
        assert result == "task"

    async def test_stop_reason_max_tokens_still_returns_category(self, classifier):
        """When stop_reason is 'max_tokens' (truncated), we still read content[0].text.

        With max_tokens=5 and typical 1-token responses, this shouldn't happen in
        practice, but if it does, we should still extract whatever text was returned.
        """
        _inject_session(classifier, _make_haiku_response(
            "", json_data={
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "task"}],
                "model": "claude-haiku-4-5-20251001",
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 200, "output_tokens": 5},
            }
        ))
        result = await classifier.classify("check disk")
        assert result == "task"

    async def test_stop_reason_max_tokens_partial_word_defaults_to_task(self, classifier):
        """If max_tokens truncates mid-word, the result won't match a valid category.

        For example, "clau" (truncated "claude_code") → not in _VALID_CATEGORIES → "task".
        """
        _inject_session(classifier, _make_haiku_response(
            "", json_data={
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "clau"}],
                "model": "claude-haiku-4-5-20251001",
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 200, "output_tokens": 5},
            }
        ))
        result = await classifier.classify("review code")
        assert result == "task"  # "clau" not in valid categories

    async def test_non_text_content_block_defaults_to_task(self, classifier):
        """If content[0] is not a text block (e.g. tool_use), KeyError on 'text' → 'task'.

        This is hypothetical — Haiku shouldn't return tool_use for a classification
        prompt — but the error path should be safe.
        """
        _inject_session(classifier, _make_haiku_response(
            "", json_data={
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_123", "name": "test", "input": {}}],
                "model": "claude-haiku-4-5-20251001",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 200, "output_tokens": 10},
            }
        ))
        result = await classifier.classify("check disk")
        assert result == "task"  # KeyError on ["text"] → caught → "task"

    async def test_null_text_in_content_block_defaults_to_task(self, classifier):
        """If text field is None, .strip() raises AttributeError → caught → 'task'."""
        _inject_session(classifier, _make_haiku_response(
            "", json_data={
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": None}],
                "model": "claude-haiku-4-5-20251001",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 0},
            }
        ))
        result = await classifier.classify("check disk")
        assert result == "task"  # AttributeError on None.strip() → caught → "task"

    async def test_retry_different_error_types(self, classifier):
        """First attempt fails with timeout, retry fails with connection error.

        Both error types are retryable but different — the outer handler
        catches the second error and records failure.
        """
        classifier.breaker.record_failure = MagicMock()

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[
            asyncio.TimeoutError(),
            aiohttp.ClientError("Connection refused"),
        ])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("check disk")
        assert result == "task"
        assert mock_session.post.call_count == 2
        classifier.breaker.record_failure.assert_called_once()

    async def test_first_retryable_http_second_non_retryable(self, classifier):
        """First attempt gets 503 (retryable), retry gets 401 (non-retryable).

        The 401 on retry is a RuntimeError (not _RetryableError), which falls through
        to the outer except → record_failure → "task".
        """
        classifier.breaker.record_failure = MagicMock()

        resp_503 = AsyncMock()
        resp_503.status = 503
        resp_503.text = AsyncMock(return_value="overloaded")
        ctx_503 = AsyncMock()
        ctx_503.__aenter__ = AsyncMock(return_value=resp_503)
        ctx_503.__aexit__ = AsyncMock(return_value=False)

        resp_401 = AsyncMock()
        resp_401.status = 401
        resp_401.text = AsyncMock(return_value="unauthorized")
        ctx_401 = AsyncMock()
        ctx_401.__aenter__ = AsyncMock(return_value=resp_401)
        ctx_401.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[ctx_503, ctx_401])
        _inject_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("check disk")
        assert result == "task"
        assert mock_session.post.call_count == 2
        classifier.breaker.record_failure.assert_called_once()
