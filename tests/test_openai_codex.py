"""Tests for llm/openai_codex.py — Codex chat client with streaming SSE."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.llm.codex_auth import CodexAuth
from src.llm.openai_codex import CODEX_API_URL, MAX_RETRIES, CodexChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_auth(access_token: str = "test_token", account_id: str | None = "acct_1") -> MagicMock:
    """Build a mock CodexAuth that returns a fixed access token."""
    auth = MagicMock(spec=CodexAuth)
    auth.get_access_token = AsyncMock(return_value=access_token)
    auth.get_account_id = MagicMock(return_value=account_id)
    auth._refresh = AsyncMock()
    auth._load = MagicMock(return_value={})
    return auth


def _make_client(auth: MagicMock | None = None, model: str = "o4-mini") -> CodexChatClient:
    """Build a CodexChatClient with a mock auth."""
    return CodexChatClient(auth=auth or _make_auth(), model=model, max_tokens=4096)


def _sse_lines(*events: str | dict) -> list[bytes]:
    """Convert event dicts to raw SSE byte lines as aiohttp would yield them.

    Each item can be:
    - A dict: serialized as data: {json}\n\n
    - A string: used verbatim (e.g. "data: [DONE]")
    """
    lines = []
    for event in events:
        if isinstance(event, dict):
            lines.append(f"data: {json.dumps(event)}\n\n".encode())
        else:
            lines.append(f"{event}\n\n".encode())
    return lines


def _mock_aiohttp_response(status: int, sse_lines: list[bytes] | None = None,
                            body: bytes = b"") -> MagicMock:
    """Build a mock aiohttp response for session.post().

    For status 200 with SSE, provides an async iterator over content lines.
    For non-200, provides a read() that returns body.
    """
    mock_resp = MagicMock()
    mock_resp.status = status

    if sse_lines is not None:
        # Make resp.content an async iterator over the SSE lines
        mock_resp.content = _AsyncIterator(sse_lines)
    else:
        mock_resp.read = AsyncMock(return_value=body)

    return mock_resp


class _AsyncIterator:
    """Async iterator over a list of items, for mocking resp.content."""

    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _mock_session_post(mock_resp) -> MagicMock:
    """Build a mock aiohttp session whose post() returns mock_resp as async ctx mgr."""
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_ctx)
    mock_session.closed = False
    mock_session.close = AsyncMock()
    return mock_session


# ---------------------------------------------------------------------------
# _convert_messages
# ---------------------------------------------------------------------------
class TestConvertMessages:
    def test_converts_user_message(self):
        """User messages are converted with input_text content type."""
        client = _make_client()
        result = client._convert_messages([{"role": "user", "content": "hello"}])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["type"] == "message"
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "hello"

    def test_converts_assistant_message(self):
        """Assistant messages are converted with output_text content type."""
        client = _make_client()
        result = client._convert_messages([{"role": "assistant", "content": "hi back"}])
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "output_text"

    def test_preserves_developer_and_system_roles(self):
        """Developer and system roles are preserved by the Responses API."""
        client = _make_client()
        result = client._convert_messages([{"role": "system", "content": "sys msg"}])
        assert result[0]["role"] == "system"
        assert result[0]["content"][0]["type"] == "input_text"

        result = client._convert_messages([{"role": "developer", "content": "dev msg"}])
        assert result[0]["role"] == "developer"
        assert result[0]["content"][0]["type"] == "input_text"

    def test_maps_unknown_roles_to_user(self):
        """Roles other than user/assistant/developer/system are mapped to 'user'."""
        client = _make_client()
        result = client._convert_messages([{"role": "tool", "content": "tool msg"}])
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "input_text"

    def test_extracts_text_from_list_content(self):
        """Messages with list content have text extracted from blocks."""
        client = _make_client()
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": "disk usage: 80%"}]},
            {"role": "user", "content": "normal message"},
        ]
        result = client._convert_messages(messages)
        assert len(result) == 2
        assert "[Tool result: disk usage: 80%]" in result[0]["content"][0]["text"]
        assert result[1]["content"][0]["text"] == "normal message"

    def test_extracts_text_blocks_from_list(self):
        """Text blocks in list content are extracted."""
        client = _make_client()
        messages = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Here is the result"},
                {"type": "tool_use", "name": "check_disk"},
            ]},
        ]
        result = client._convert_messages(messages)
        assert len(result) == 1
        assert "Here is the result" in result[0]["content"][0]["text"]
        assert "[Used tool: check_disk]" in result[0]["content"][0]["text"]

    def test_skips_empty_list_content(self):
        """Messages with list content that produces no text are skipped."""
        client = _make_client()
        messages = [
            {"role": "user", "content": [{"type": "image", "source": "..."}]},
            {"role": "user", "content": "normal message"},
        ]
        result = client._convert_messages(messages)
        assert len(result) == 1
        assert result[0]["content"][0]["text"] == "normal message"

    def test_tool_result_with_list_content(self):
        """Tool result blocks with list-format content are summarized."""
        client = _make_client()
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": [
                    {"type": "text", "text": "Server is healthy"},
                ]},
            ]},
        ]
        result = client._convert_messages(messages)
        assert len(result) == 1
        assert "Server is healthy" in result[0]["content"][0]["text"]

    def test_skips_missing_content(self):
        """Messages without a content key are skipped."""
        client = _make_client()
        result = client._convert_messages([{"role": "user"}])
        # content defaults to "" which is a string, so it should be included
        assert len(result) == 1
        assert result[0]["content"][0]["text"] == ""

    def test_preserves_message_order(self):
        """Multiple messages maintain their original order."""
        client = _make_client()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        result = client._convert_messages(messages)
        assert len(result) == 3
        assert [m["content"][0]["text"] for m in result] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# _read_stream
# ---------------------------------------------------------------------------
class TestReadStream:
    async def test_collects_text_deltas(self):
        """Assembles response text from response.output_text.delta events."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": " world"},
            {"type": "response.completed", "response": {"output": []}},
        )
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "Hello world"

    async def test_uses_completed_output_when_no_deltas(self):
        """Falls back to response.completed output when no deltas were received."""
        client = _make_client()
        lines = _sse_lines({
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Final answer"}],
                }],
            },
        })
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "Final answer"

    async def test_prefers_deltas_over_completed(self):
        """When both deltas and completed output exist, uses the deltas."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_text.delta", "delta": "From deltas"},
            {
                "type": "response.completed",
                "response": {
                    "output": [{
                        "type": "message",
                        "content": [{"type": "output_text", "text": "From completed"}],
                    }],
                },
            },
        )
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "From deltas"

    async def test_handles_done_sentinel(self):
        """Stops reading at [DONE] sentinel."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_text.delta", "delta": "Before"},
            "data: [DONE]",
            {"type": "response.output_text.delta", "delta": "After"},
        )
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "Before"

    async def test_skips_non_data_lines(self):
        """Lines not starting with 'data: ' are ignored."""
        client = _make_client()
        lines = [
            b"event: ping\n\n",
            b": comment\n\n",
            f'data: {json.dumps({"type": "response.output_text.delta", "delta": "ok"})}\n\n'.encode(),
        ]
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "ok"

    async def test_skips_invalid_json(self):
        """Malformed JSON data lines are skipped without error."""
        client = _make_client()
        lines = [
            b"data: {invalid json\n\n",
            f'data: {json.dumps({"type": "response.output_text.delta", "delta": "valid"})}\n\n'.encode(),
        ]
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "valid"

    async def test_returns_empty_string_when_no_content(self):
        """Returns empty string when stream produces no text content."""
        client = _make_client()
        lines = _sse_lines({"type": "response.created"})
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == ""

    async def test_ignores_empty_deltas(self):
        """Delta events with empty string are skipped."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_text.delta", "delta": ""},
            {"type": "response.output_text.delta", "delta": "content"},
        )
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "content"

    async def test_ignores_unknown_event_types(self):
        """Unknown event types are silently ignored."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.unknown_event", "data": "stuff"},
            {"type": "response.output_text.delta", "delta": "real"},
        )
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "real"

    async def test_completed_skips_non_message_output(self):
        """Completed event ignores output items that aren't type=message."""
        client = _make_client()
        lines = _sse_lines({
            "type": "response.completed",
            "response": {
                "output": [
                    {"type": "function_call", "name": "foo"},
                    {"type": "message", "content": [{"type": "output_text", "text": "msg"}]},
                ],
            },
        })
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "msg"

    async def test_completed_skips_empty_text_blocks(self):
        """Completed event ignores content blocks with empty text."""
        client = _make_client()
        lines = _sse_lines({
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": ""},
                        {"type": "output_text", "text": "actual"},
                    ],
                }],
            },
        })
        mock_resp = _mock_aiohttp_response(200, lines)
        result = await client._read_stream(mock_resp)
        assert result == "actual"


# ---------------------------------------------------------------------------
# Session management: _get_session / close
# ---------------------------------------------------------------------------
class TestSessionManagement:
    async def test_get_session_creates_new(self):
        """_get_session creates a new aiohttp session when none exists."""
        client = _make_client()
        assert client._session is None
        with patch("src.llm.openai_codex.aiohttp.ClientSession") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_cls.return_value = mock_instance
            session = await client._get_session()
            assert session is mock_instance
            mock_cls.assert_called_once()

    async def test_get_session_reuses_open(self):
        """_get_session reuses an existing open session."""
        client = _make_client()
        mock_session = MagicMock()
        mock_session.closed = False
        client._session = mock_session
        session = await client._get_session()
        assert session is mock_session

    async def test_get_session_recreates_closed(self):
        """_get_session creates a new session if the existing one is closed."""
        client = _make_client()
        old_session = MagicMock()
        old_session.closed = True
        client._session = old_session

        with patch("src.llm.openai_codex.aiohttp.ClientSession") as mock_cls:
            new_session = MagicMock()
            new_session.closed = False
            mock_cls.return_value = new_session
            session = await client._get_session()
            assert session is new_session

    async def test_close_closes_session(self):
        """close() closes the aiohttp session."""
        client = _make_client()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._session = mock_session

        await client.close()
        mock_session.close.assert_awaited_once()

    async def test_close_noop_when_no_session(self):
        """close() is a no-op when no session exists."""
        client = _make_client()
        await client.close()  # should not raise

    async def test_close_noop_when_already_closed(self):
        """close() is a no-op when session is already closed."""
        client = _make_client()
        mock_session = MagicMock()
        mock_session.closed = True
        client._session = mock_session
        await client.close()  # should not raise or call close again


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------
class TestChat:
    async def test_chat_sends_correct_request(self):
        """chat() builds correct headers and body, calls _stream_request."""
        auth = _make_auth(access_token="bearer_tok", account_id="acct_42")
        client = _make_client(auth=auth, model="o4-mini")

        expected_result = "response text"
        client._stream_request = AsyncMock(return_value=expected_result)

        result = await client.chat(
            messages=[{"role": "user", "content": "test query"}],
            system="You are helpful.",
        )

        assert result == expected_result
        auth.get_access_token.assert_awaited_once()
        auth.get_account_id.assert_called_once()

        # Verify _stream_request was called with correct headers and body
        call_args = client._stream_request.call_args
        headers = call_args[0][0]
        body = call_args[0][1]

        assert headers["Authorization"] == "Bearer bearer_tok"
        assert headers["ChatGPT-Account-Id"] == "acct_42"
        assert body["model"] == "o4-mini"
        assert body["instructions"] == "You are helpful."
        assert body["stream"] is True
        assert body["store"] is False

    async def test_chat_omits_account_id_when_none(self):
        """chat() omits ChatGPT-Account-Id header when account_id is None."""
        auth = _make_auth(account_id=None)
        client = _make_client(auth=auth)
        client._stream_request = AsyncMock(return_value="ok")

        await client.chat(messages=[{"role": "user", "content": "hi"}], system="sys")

        headers = client._stream_request.call_args[0][0]
        assert "ChatGPT-Account-Id" not in headers


# ---------------------------------------------------------------------------
# _stream_request — success, retries, errors
# ---------------------------------------------------------------------------
class TestStreamRequest:
    async def test_success_on_first_attempt(self):
        """Returns response text on successful 200 response."""
        client = _make_client()
        sse = _sse_lines(
            {"type": "response.output_text.delta", "delta": "Hello world"},
        )
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        result = await client._stream_request(
            headers={"Authorization": "Bearer tok"},
            body={"model": "o4-mini"},
        )
        assert result == "Hello world"

    async def test_retries_on_500(self):
        """Retries on server error (500) and succeeds on subsequent attempt."""
        client = _make_client()

        # First call: 500 error
        error_resp = MagicMock()
        error_resp.status = 500
        error_resp.read = AsyncMock(return_value=b"server error")

        # Second call: 200 success
        sse = _sse_lines({"type": "response.output_text.delta", "delta": "ok"})
        success_resp = _mock_aiohttp_response(200, sse)

        call_count = 0
        mock_post_ctxs = []
        for resp in [error_resp, success_resp]:
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_post_ctxs.append(ctx)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=mock_post_ctxs)
        client._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._stream_request(
                headers={"Authorization": "Bearer tok"},
                body={},
            )
        assert result == "ok"
        assert mock_session.post.call_count == 2

    async def test_retries_on_429(self):
        """Retries on rate limit (429)."""
        client = _make_client()

        error_resp = MagicMock()
        error_resp.status = 429
        error_resp.read = AsyncMock(return_value=b"rate limited")

        sse = _sse_lines({"type": "response.output_text.delta", "delta": "done"})
        success_resp = _mock_aiohttp_response(200, sse)

        ctxs = []
        for resp in [error_resp, success_resp]:
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctxs.append(ctx)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=ctxs)
        client._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._stream_request(
                headers={"Authorization": "Bearer tok"},
                body={},
            )
        assert result == "done"

    async def test_refreshes_on_401(self):
        """On 401, refreshes auth token and retries."""
        auth = _make_auth(access_token="new_token", account_id="acct_1")
        client = _make_client(auth=auth)

        # First call: 401
        unauth_resp = MagicMock()
        unauth_resp.status = 401
        unauth_resp.read = AsyncMock(return_value=b"unauthorized")

        # Second call: 200
        sse = _sse_lines({"type": "response.output_text.delta", "delta": "refreshed"})
        success_resp = _mock_aiohttp_response(200, sse)

        ctxs = []
        for resp in [unauth_resp, success_resp]:
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctxs.append(ctx)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=ctxs)
        client._session = mock_session

        result = await client._stream_request(
            headers={"Authorization": "Bearer old_token"},
            body={},
        )
        assert result == "refreshed"
        auth._refresh.assert_awaited_once()

    async def test_raises_on_non_retryable_error(self):
        """Raises RuntimeError on non-retryable HTTP errors (e.g. 403)."""
        client = _make_client()

        error_resp = MagicMock()
        error_resp.status = 403
        error_resp.read = AsyncMock(return_value=b"forbidden")

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=error_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=ctx)
        client._session = mock_session

        with pytest.raises(RuntimeError, match="Codex API error.*403"):
            await client._stream_request(
                headers={"Authorization": "Bearer tok"},
                body={},
            )

    async def test_raises_after_max_retries(self):
        """Raises RuntimeError after exhausting all retry attempts on 500."""
        client = _make_client()

        ctxs = []
        for _ in range(MAX_RETRIES):
            resp = MagicMock()
            resp.status = 500
            resp.read = AsyncMock(return_value=b"server error")
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctxs.append(ctx)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=ctxs)
        client._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Codex API error.*500"):
                await client._stream_request(
                    headers={"Authorization": "Bearer tok"},
                    body={},
                )

    async def test_retries_on_connection_error(self):
        """Retries on aiohttp.ClientError and succeeds on next attempt."""
        client = _make_client()

        # First call: connection error
        error_ctx = MagicMock()
        error_ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("conn failed"))
        error_ctx.__aexit__ = AsyncMock(return_value=False)

        # Second call: success
        sse = _sse_lines({"type": "response.output_text.delta", "delta": "recovered"})
        success_resp = _mock_aiohttp_response(200, sse)
        success_ctx = MagicMock()
        success_ctx.__aenter__ = AsyncMock(return_value=success_resp)
        success_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=[error_ctx, success_ctx])
        client._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._stream_request(
                headers={"Authorization": "Bearer tok"},
                body={},
            )
        assert result == "recovered"

    async def test_raises_after_max_connection_errors(self):
        """Raises RuntimeError after exhausting retries on connection errors."""
        client = _make_client()

        ctxs = []
        for _ in range(MAX_RETRIES):
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("conn failed"))
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctxs.append(ctx)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=ctxs)
        client._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Codex API connection failed"):
                await client._stream_request(
                    headers={"Authorization": "Bearer tok"},
                    body={},
                )

    async def test_circuit_breaker_checked(self):
        """Circuit breaker is checked before making the request."""
        from src.llm.circuit_breaker import CircuitBreaker

        client = _make_client()
        client.breaker = MagicMock(spec=CircuitBreaker)
        client.breaker.check = MagicMock(side_effect=RuntimeError("circuit open"))

        mock_session = MagicMock()
        mock_session.closed = False
        client._session = mock_session

        with pytest.raises(RuntimeError, match="circuit open"):
            await client._stream_request(
                headers={"Authorization": "Bearer tok"},
                body={},
            )
        client.breaker.check.assert_called_once()

    async def test_circuit_breaker_success_recorded(self):
        """Records success on circuit breaker after successful response."""
        client = _make_client()
        client.breaker = MagicMock()

        sse = _sse_lines({"type": "response.output_text.delta", "delta": "ok"})
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        await client._stream_request(
            headers={"Authorization": "Bearer tok"},
            body={},
        )
        client.breaker.record_success.assert_called_once()

    async def test_empty_response_does_not_record_breaker_success(self):
        """Does not record breaker success when 200 response has no content."""
        client = _make_client()
        client.breaker = MagicMock()

        sse = _sse_lines({"type": "response.created"})
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        result = await client._stream_request(
            headers={"Authorization": "Bearer tok"},
            body={},
        )
        assert result == ""
        client.breaker.record_success.assert_not_called()

    async def test_circuit_breaker_failure_recorded_on_error(self):
        """Records failure on circuit breaker after server error."""
        client = _make_client()
        client.breaker = MagicMock()

        error_resp = MagicMock()
        error_resp.status = 403
        error_resp.read = AsyncMock(return_value=b"error")

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=error_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=ctx)
        client._session = mock_session

        with pytest.raises(RuntimeError):
            await client._stream_request(
                headers={"Authorization": "Bearer tok"},
                body={},
            )
        client.breaker.record_failure.assert_called()


# ---------------------------------------------------------------------------
# End-to-end: chat method integration
# ---------------------------------------------------------------------------
class TestChatIntegration:
    async def test_full_chat_flow(self):
        """Full chat flow: auth → message conversion → streaming → result."""
        auth = _make_auth(access_token="bearer_tok", account_id="acct_1")
        client = _make_client(auth=auth, model="o4-mini")

        sse = _sse_lines(
            {"type": "response.output_text.delta", "delta": "Hello "},
            {"type": "response.output_text.delta", "delta": "world!"},
            {"type": "response.completed", "response": {"output": []}},
        )
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        result = await client.chat(
            messages=[{"role": "user", "content": "Say hello"}],
            system="Be concise.",
        )
        assert result == "Hello world!"

        # Verify the POST was made to the correct URL
        post_call = mock_session.post.call_args
        assert post_call[0][0] == CODEX_API_URL
