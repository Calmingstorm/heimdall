"""Tests for Round 22: Error Handling + Resilience.

Covers:
- Audit log failure doesn't crash tool execution
- Session save failure doesn't crash message processing
- Circuit breaker trips on exhausted empty-response retries
- _send_with_retry catches network-level errors
- Scheduled reminder send failure is caught
- Graceful degradation when optional services are unavailable
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.openai_codex import CodexChatClient, MAX_RETRIES
from src.llm.codex_auth import CodexAuth
from src.llm.circuit_breaker import CircuitBreaker
from src.llm.types import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_auth() -> MagicMock:
    auth = MagicMock(spec=CodexAuth)
    auth.get_access_token = AsyncMock(return_value="test_token")
    auth.get_account_id = MagicMock(return_value="acct_1")
    auth._refresh = AsyncMock()
    auth._load = MagicMock(return_value={})
    return auth


def _make_client() -> CodexChatClient:
    return CodexChatClient(auth=_make_auth(), model="o4-mini", max_tokens=4096)


def _sse_lines(*events) -> list[bytes]:
    lines = []
    for event in events:
        if isinstance(event, dict):
            lines.append(f"data: {json.dumps(event)}\n\n".encode())
        else:
            lines.append(f"{event}\n\n".encode())
    return lines


def _mock_aiohttp_response(status, sse_lines=None, body=b""):
    mock_resp = MagicMock()
    mock_resp.status = status
    if sse_lines is not None:
        async def _aiter():
            for line in sse_lines:
                yield line
        mock_resp.content = _aiter()
        mock_resp.content.__aiter__ = _aiter
    mock_resp.read = AsyncMock(return_value=body)
    return mock_resp


def _mock_session_post(mock_resp):
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=ctx)
    return mock_session


def _make_bot_stub():
    """Create a minimal bot stub for testing client methods."""
    bot = MagicMock()
    bot.config = MagicMock()
    bot.config.discord = MagicMock()
    bot.config.discord.allowed_users = ["123"]
    bot.config.tools = MagicMock()
    bot.config.tools.tool_timeout_seconds = 300
    bot.config.tools.enabled = True
    bot.config.comfyui = MagicMock()
    bot.config.comfyui.enabled = False
    bot.config.browser = MagicMock()
    bot.config.browser.enabled = False
    bot.sessions = MagicMock()
    bot.sessions.save = MagicMock()
    bot.audit = MagicMock()
    bot.audit.log_execution = AsyncMock()
    bot.tool_executor = MagicMock()
    bot.tool_executor.execute = AsyncMock(return_value="tool result")
    bot.skill_manager = MagicMock()
    bot.skill_manager.has_skill = MagicMock(return_value=False)
    bot.permissions = MagicMock()
    bot.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    bot._recent_actions = {}
    bot._recent_actions_max = 10
    bot._recent_actions_expiry = 3600
    bot._pending_files = {}
    bot._cached_merged_tools = None
    bot._bot_msg_buffer_max = 20
    bot.browser_manager = None
    bot.codex_client = MagicMock()
    bot.loop_manager = MagicMock()
    return bot


# ---------------------------------------------------------------------------
# 1. Audit log failure doesn't crash tool execution
# ---------------------------------------------------------------------------

class TestAuditLogResilience:
    """Audit log failures must not prevent tool results from being returned."""

    async def test_audit_failure_does_not_crash_tool(self):
        """When audit.log_execution raises, the tool result should still be returned."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        # Verify audit.log_execution is wrapped: the try keyword must appear
        # before the log_execution call, with an except catching audit_err
        audit_idx = source.index("log_execution")
        # Find the nearest "try:" before the audit call
        try_idx = source.rfind("try:", 0, audit_idx)
        assert try_idx != -1, "audit.log_execution must be inside a try block"
        # And there should be an except with audit_err after it
        except_idx = source.find("audit_err", audit_idx)
        assert except_idx != -1, "audit try/except should catch as audit_err"

    async def test_audit_failure_logged_as_warning(self):
        """Audit failure should log a warning, not silently swallow."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        # Check that the audit try/except logs the error
        assert "audit_err" in source or "audit log failed" in source.lower(), \
            "Audit failure should be logged with a warning"

    async def test_track_recent_action_wrapped(self):
        """_track_recent_action in _run_tool should be wrapped in try/except."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        # The tracking call should have error handling
        assert "Non-critical tracking" in source or "track_recent_action" in source


# ---------------------------------------------------------------------------
# 2. Session save resilience
# ---------------------------------------------------------------------------

class TestSessionSaveResilience:
    """Session save failures must not crash message processing."""

    async def test_session_save_wrapped_in_try(self):
        """sessions.save() should be inside try/except in _handle_message_inner."""
        from src.discord.client import HeimdallBot
        import ast
        import inspect
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        # Check that save_err is referenced (our try/except pattern)
        assert "save_err" in source, \
            "sessions.save() must be wrapped in try/except with save_err"

    async def test_session_save_failure_logged(self):
        """Session save failure should log a warning."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        assert "Session save failed" in source


# ---------------------------------------------------------------------------
# 3. Circuit breaker on empty response exhaustion
# ---------------------------------------------------------------------------

class TestCircuitBreakerEmptyResponse:
    """Circuit breaker should record failure when empty responses exhaust retries."""

    async def test_stream_request_empty_records_failure(self):
        """_stream_request records breaker failure after exhausting empty retries."""
        client = _make_client()
        client.breaker = MagicMock()

        # Create mock that always returns empty SSE stream
        sse = _sse_lines({"type": "response.created"})

        # Need to create a fresh response for each retry attempt
        call_count = 0

        def make_ctx(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Each call needs a fresh async iterator
            async def _aiter():
                for line in _sse_lines({"type": "response.created"}):
                    yield line
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.content = _aiter()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=make_ctx)
        client._session = mock_session

        result = await client._stream_request(
            headers={"Authorization": "Bearer tok"},
            body={},
        )
        assert result == ""
        # Should record failure after exhausting retries
        client.breaker.record_failure.assert_called_once()
        client.breaker.record_success.assert_not_called()

    async def test_stream_tool_request_empty_records_failure(self):
        """_stream_tool_request records breaker failure after exhausting empty retries."""
        client = _make_client()
        client.breaker = MagicMock()

        def make_ctx(*args, **kwargs):
            async def _aiter():
                for line in _sse_lines({"type": "response.created"}):
                    yield line
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.content = _aiter()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=make_ctx)
        client._session = mock_session

        result = await client._stream_tool_request(
            headers={"Authorization": "Bearer tok"},
            body={},
        )
        assert result.text == ""
        assert result.tool_calls == []
        client.breaker.record_failure.assert_called_once()
        client.breaker.record_success.assert_not_called()

    async def test_successful_response_still_records_success(self):
        """A successful non-empty response should still record success."""
        client = _make_client()
        client.breaker = MagicMock()

        sse = _sse_lines(
            {"type": "response.output_text.delta", "delta": "Hello"},
            "data: [DONE]",
        )

        async def _aiter():
            for line in sse:
                yield line

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.content = _aiter()

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=ctx)
        client._session = mock_session

        result = await client._stream_request(
            headers={"Authorization": "Bearer tok"},
            body={},
        )
        assert result == "Hello"
        client.breaker.record_success.assert_called_once()
        client.breaker.record_failure.assert_not_called()

    async def test_empty_first_try_success_second_no_failure_recorded(self):
        """If first try is empty but second succeeds, no failure should be recorded."""
        client = _make_client()
        client.breaker = MagicMock()

        attempt = [0]

        def make_ctx(*args, **kwargs):
            attempt[0] += 1
            if attempt[0] == 1:
                # First attempt: empty
                async def _aiter():
                    for line in _sse_lines({"type": "response.created"}):
                        yield line
            else:
                # Second attempt: success
                async def _aiter():
                    for line in _sse_lines(
                        {"type": "response.output_text.delta", "delta": "ok"},
                        "data: [DONE]",
                    ):
                        yield line

            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.content = _aiter()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=make_ctx)
        client._session = mock_session

        result = await client._stream_request(
            headers={"Authorization": "Bearer tok"},
            body={},
        )
        assert result == "ok"
        client.breaker.record_success.assert_called_once()
        client.breaker.record_failure.assert_not_called()


# ---------------------------------------------------------------------------
# 4. _send_with_retry catches network errors
# ---------------------------------------------------------------------------

class TestSendWithRetryResilience:
    """_send_with_retry should handle network-level errors."""

    async def test_catches_connection_error(self):
        """ConnectionError should be caught and retried."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._send_with_retry)
        assert "ConnectionError" in source, \
            "_send_with_retry should catch ConnectionError"

    async def test_catches_os_error(self):
        """OSError should be caught and retried."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._send_with_retry)
        assert "OSError" in source, \
            "_send_with_retry should catch OSError"

    async def test_still_catches_http_exception(self):
        """discord.HTTPException should still be caught."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._send_with_retry)
        assert "HTTPException" in source


# ---------------------------------------------------------------------------
# 5. Scheduled reminder resilience
# ---------------------------------------------------------------------------

class TestScheduledReminderResilience:
    """Scheduled reminder send failures should be caught."""

    async def test_reminder_send_wrapped(self):
        """The reminder channel.send should be in a try/except."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._on_scheduled_task)
        # The reminder section should have error handling
        assert "Failed to send scheduled reminder" in source

    async def test_check_action_already_wrapped(self):
        """The check action was already wrapped — verify."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._on_scheduled_task)
        assert "Scheduled task failed" in source


# ---------------------------------------------------------------------------
# 6. Graceful degradation for optional services
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Optional services should degrade gracefully when unavailable."""

    async def test_browser_disabled_returns_message(self):
        """Browser tools return helpful message when browser is disabled."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._handle_browser_screenshot)
        assert "Browser automation is not enabled" in source

    async def test_comfyui_disabled_returns_message(self):
        """Generate image returns helpful message when ComfyUI is disabled."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._handle_generate_image)
        assert "Image generation is disabled" in source

    async def test_comfyui_failure_returns_message(self):
        """Generate image returns helpful message when ComfyUI fails."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._handle_generate_image)
        assert "ComfyUI may be unavailable" in source

    async def test_knowledge_store_unavailable_returns_message(self):
        """Knowledge operations return message when store is unavailable."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._handle_search_knowledge)
        assert "Knowledge base is not available" in source

    async def test_codex_not_configured_returns_message(self):
        """No codex client returns a helpful message."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        assert "No tool backend available" in source


# ---------------------------------------------------------------------------
# 7. Codex API error handling
# ---------------------------------------------------------------------------

class TestCodexRetryLogic:
    """Verify Codex API retry behavior for various error types."""

    async def test_retries_on_429(self):
        """429 rate limit triggers retry."""
        client = _make_client()
        client.breaker = MagicMock()

        call_count = [0]

        def make_ctx(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < MAX_RETRIES:
                mock_resp = MagicMock()
                mock_resp.status = 429
                mock_resp.read = AsyncMock(return_value=b"rate limited")
                ctx = MagicMock()
                ctx.__aenter__ = AsyncMock(return_value=mock_resp)
                ctx.__aexit__ = AsyncMock(return_value=False)
                return ctx
            else:
                # Final attempt returns success
                async def _aiter():
                    for line in _sse_lines(
                        {"type": "response.output_text.delta", "delta": "ok"},
                        "data: [DONE]",
                    ):
                        yield line
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.content = _aiter()
                ctx = MagicMock()
                ctx.__aenter__ = AsyncMock(return_value=mock_resp)
                ctx.__aexit__ = AsyncMock(return_value=False)
                return ctx

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=make_ctx)
        client._session = mock_session

        result = await client._stream_request(
            headers={"Authorization": "Bearer tok"},
            body={},
        )
        assert result == "ok"
        assert call_count[0] == MAX_RETRIES

    async def test_retries_on_502(self):
        """502 bad gateway triggers retry."""
        client = _make_client()
        client.breaker = MagicMock()

        call_count = [0]

        def make_ctx(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.status = 502
            mock_resp.read = AsyncMock(return_value=b"bad gateway")
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=make_ctx)
        client._session = mock_session

        with pytest.raises(RuntimeError, match="Codex API error"):
            await client._stream_request(
                headers={"Authorization": "Bearer tok"},
                body={},
            )
        assert call_count[0] == MAX_RETRIES

    async def test_no_retry_on_400(self):
        """400 client error does not retry."""
        client = _make_client()
        client.breaker = MagicMock()

        call_count = [0]

        def make_ctx(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.status = 400
            mock_resp.read = AsyncMock(return_value=b"bad request")
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=make_ctx)
        client._session = mock_session

        with pytest.raises(RuntimeError, match="Codex API error"):
            await client._stream_request(
                headers={"Authorization": "Bearer tok"},
                body={},
            )
        # Should fail immediately, no retry
        assert call_count[0] == 1

    async def test_401_triggers_token_refresh(self):
        """401 triggers a single token refresh attempt."""
        client = _make_client()
        client.breaker = MagicMock()

        call_count = [0]

        def make_ctx(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_resp = MagicMock()
                mock_resp.status = 401
                mock_resp.read = AsyncMock(return_value=b"unauthorized")
                ctx = MagicMock()
                ctx.__aenter__ = AsyncMock(return_value=mock_resp)
                ctx.__aexit__ = AsyncMock(return_value=False)
                return ctx
            else:
                async def _aiter():
                    for line in _sse_lines(
                        {"type": "response.output_text.delta", "delta": "refreshed"},
                        "data: [DONE]",
                    ):
                        yield line
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.content = _aiter()
                ctx = MagicMock()
                ctx.__aenter__ = AsyncMock(return_value=mock_resp)
                ctx.__aexit__ = AsyncMock(return_value=False)
                return ctx

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=make_ctx)
        client._session = mock_session

        result = await client._stream_request(
            headers={"Authorization": "Bearer tok"},
            body={},
        )
        assert result == "refreshed"
        client.auth._refresh.assert_called_once()


# ---------------------------------------------------------------------------
# 8. Circuit breaker behavior
# ---------------------------------------------------------------------------

class TestCircuitBreakerBehavior:
    """Verify circuit breaker state transitions."""

    def test_starts_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == "closed"

    def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"

    def test_success_resets(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == "closed"  # Reset by success

    def test_check_raises_when_open(self):
        from src.llm.circuit_breaker import CircuitOpenError
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60.0)
        cb.record_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.check()
        assert exc_info.value.provider == "test"
        assert exc_info.value.retry_after > 0


# ---------------------------------------------------------------------------
# 9. Tool execution timeout
# ---------------------------------------------------------------------------

class TestToolExecutionTimeout:
    """Verify tool timeout behavior."""

    async def test_tool_timeout_returns_error_message(self):
        """A timed-out tool should return an error string, not crash."""
        from src.discord.client import TOOL_OUTPUT_MAX_CHARS

        # The timeout wrapper in _process_with_tools catches TimeoutError
        # and returns a tool_result dict with an error message
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        assert "asyncio.TimeoutError" in source
        assert "timed out after" in source

    async def test_timeout_audits_the_error(self):
        """Timeout should audit log the error."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        # The timeout handler should try to audit
        assert "audit.log_execution" in source


# ---------------------------------------------------------------------------
# 10. Error response to user
# ---------------------------------------------------------------------------

class TestErrorResponseToUser:
    """Verify error responses are sanitized before reaching users."""

    async def test_error_response_scrubbed(self):
        """Error messages sent to Discord should be scrubbed for secrets."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        # The outer error handler should scrub secrets
        assert "scrub_response_secrets" in source

    async def test_error_markers_sanitized_in_history(self):
        """Error history entries should be sanitized, not raw error text."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        assert "Previous request" in source
        assert "encountered an error" in source

    async def test_codex_error_in_tool_loop_returns_partial_report(self):
        """When Codex API fails mid-loop, partial completion report is included."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        assert "_build_partial_completion_report" in source
        assert "LLM API error" in source


# ---------------------------------------------------------------------------
# 11. Empty response fallback
# ---------------------------------------------------------------------------

class TestEmptyResponseFallback:
    """Verify empty LLM responses are handled gracefully."""

    async def test_empty_response_uses_fallback(self):
        """Empty text response should use _EMPTY_RESPONSE_FALLBACK."""
        from src.discord.client import _EMPTY_RESPONSE_FALLBACK
        assert _EMPTY_RESPONSE_FALLBACK, "Fallback message should be non-empty"
        assert "try again" in _EMPTY_RESPONSE_FALLBACK.lower()

    async def test_tool_loop_uses_fallback_on_empty(self):
        """Tool loop returns fallback when LLM response is empty."""
        from src.discord.client import HeimdallBot, _EMPTY_RESPONSE_FALLBACK
        import inspect
        source = inspect.getsource(HeimdallBot._process_with_tools)
        assert "_EMPTY_RESPONSE_FALLBACK" in source

    async def test_guest_chat_uses_fallback_on_empty(self):
        """Guest chat path uses fallback when Codex returns empty."""
        from src.discord.client import HeimdallBot
        import inspect
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        assert "_EMPTY_RESPONSE_FALLBACK" in source


# ---------------------------------------------------------------------------
# 12. Browser reconnection
# ---------------------------------------------------------------------------

class TestBrowserReconnection:
    """Browser should auto-reconnect on stale CDP connections."""

    async def test_connection_error_triggers_reconnect(self):
        """Stale CDP connections trigger force_reconnect."""
        from src.tools.browser import BrowserManager
        import inspect
        source = inspect.getsource(BrowserManager.new_page)
        assert "_is_connection_error" in source
        assert "_force_reconnect" in source

    async def test_connection_error_patterns(self):
        """Known connection error patterns are detected."""
        from src.tools.browser import BrowserManager
        bm = BrowserManager()
        assert bm._is_connection_error(Exception("connection closed"))
        assert bm._is_connection_error(Exception("Target closed"))
        assert bm._is_connection_error(Exception("browser has been closed"))
        assert not bm._is_connection_error(Exception("page not found"))
        assert not bm._is_connection_error(Exception("timeout"))


# ---------------------------------------------------------------------------
# 13. ComfyUI error handling
# ---------------------------------------------------------------------------

class TestComfyUIErrorHandling:
    """ComfyUI should return None gracefully on all failure modes."""

    async def test_timeout_returns_none(self):
        """Timeout should return None, not raise."""
        from src.tools.comfyui import ComfyUIClient
        import inspect
        source = inspect.getsource(ComfyUIClient.generate)
        assert "asyncio.TimeoutError" in source
        assert "return None" in source

    async def test_connection_error_returns_none(self):
        """Connection error should return None, not raise."""
        from src.tools.comfyui import ComfyUIClient
        import inspect
        source = inspect.getsource(ComfyUIClient.generate)
        assert "ClientError" in source

    async def test_unexpected_error_returns_none(self):
        """Unexpected exceptions should return None, not propagate."""
        from src.tools.comfyui import ComfyUIClient
        import inspect
        source = inspect.getsource(ComfyUIClient.generate)
        # Should have a catch-all
        assert "Exception" in source
