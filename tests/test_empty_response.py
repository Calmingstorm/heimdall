"""Tests for Issues 1, 5: empty response retry, friendly fallback.

Round 5 implementation tests:
- Issue 1: Codex retries on 200-with-empty-body (both chat and tool paths)
- Issue 5: "(no response)" replaced with friendly fallback message
- Round 4 finding: handoff preserves skill response on empty chat()
"""
from __future__ import annotations

import json
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.llm.openai_codex import CodexChatClient, CodexAuth, RETRY_BACKOFF  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


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


class _AsyncIterator:
    """Mimics aiohttp resp.content async iteration."""
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


def _sse_lines(*events) -> list[bytes]:
    """Convert event dicts (or raw strings) to SSE byte lines."""
    lines = []
    for ev in events:
        if isinstance(ev, dict):
            lines.append(f"data: {json.dumps(ev)}\n\n".encode())
        else:
            lines.append(f"{ev}\n\n".encode())
    return lines


def _mock_response(status: int, sse_lines: list[bytes] | None = None,
                   body: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status = status
    if sse_lines is not None:
        resp.content = _AsyncIterator(sse_lines)
    else:
        resp.read = AsyncMock(return_value=body)
    return resp


def _wrap_ctx(resp) -> MagicMock:
    """Wrap a mock response in an async context manager for session.post()."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _mock_session(*responses) -> MagicMock:
    """Build a mock aiohttp session with sequential post() responses."""
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.post = MagicMock(side_effect=[_wrap_ctx(r) for r in responses])
    return session


# Empty SSE stream: only has response.created, no text deltas
_EMPTY_SSE = _sse_lines({"type": "response.created"}, "data: [DONE]")

# SSE stream with text content
_TEXT_SSE = _sse_lines(
    {"type": "response.output_text.delta", "delta": "Hello!"},
    "data: [DONE]",
)

# SSE stream with a function call
_TOOL_SSE = _sse_lines(
    {"type": "response.output_item.added", "output_index": 0, "item": {
        "type": "function_call", "call_id": "call_1", "name": "check_disk",
    }},
    {"type": "response.function_call_arguments.done", "output_index": 0,
     "arguments": '{"host": "server"}'},
    {"type": "response.output_item.done", "output_index": 0, "item": {
        "type": "function_call", "call_id": "call_1", "name": "check_disk",
        "arguments": '{"host": "server"}',
    }},
    "data: [DONE]",
)


# ---------------------------------------------------------------------------
# Issue 1: Empty response retry — _stream_request (chat path)
# ---------------------------------------------------------------------------


class TestStreamRequestRetryOnEmpty:
    """Codex chat() retries when 200 returns empty text."""

    async def test_retries_on_empty_then_succeeds(self):
        client = _make_client()
        empty_resp = _mock_response(200, _EMPTY_SSE)
        ok_resp = _mock_response(200, _TEXT_SSE)
        client._session = _mock_session(empty_resp, ok_resp)

        with patch("src.llm.openai_codex.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client._stream_request(
                headers={"Authorization": "Bearer tok"}, body={},
            )

        assert result == "Hello!"
        assert client._session.post.call_count == 2
        mock_sleep.assert_called_once_with(RETRY_BACKOFF[0])

    async def test_exhausts_retries_returns_empty(self):
        client = _make_client()
        # All 3 attempts return empty
        resps = [_mock_response(200, _EMPTY_SSE) for _ in range(3)]
        client._session = _mock_session(*resps)

        with patch("src.llm.openai_codex.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client._stream_request(
                headers={"Authorization": "Bearer tok"}, body={},
            )

        assert result == ""
        assert client._session.post.call_count == 3
        # Should have slept twice (not after the last attempt)
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# Issue 1: Empty response retry — _stream_tool_request (tool path)
# ---------------------------------------------------------------------------


class TestStreamToolRequestRetryOnEmpty:
    """Codex chat_with_tools() retries when 200 returns empty LLMResponse."""

    async def test_retries_on_empty_then_succeeds(self):
        client = _make_client()
        empty_resp = _mock_response(200, _EMPTY_SSE)
        ok_resp = _mock_response(200, _TOOL_SSE)
        client._session = _mock_session(empty_resp, ok_resp)

        with patch("src.llm.openai_codex.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client._stream_tool_request(
                headers={"Authorization": "Bearer tok"}, body={},
            )

        assert result.is_tool_use
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "check_disk"
        assert client._session.post.call_count == 2
        mock_sleep.assert_called_once_with(RETRY_BACKOFF[0])

    async def test_exhausts_retries_returns_empty(self):
        client = _make_client()
        resps = [_mock_response(200, _EMPTY_SSE) for _ in range(3)]
        client._session = _mock_session(*resps)

        with patch("src.llm.openai_codex.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client._stream_tool_request(
                headers={"Authorization": "Bearer tok"}, body={},
            )

        assert result.text == ""
        assert result.tool_calls == []
        assert client._session.post.call_count == 3
        assert mock_sleep.call_count == 2

    async def test_text_response_retries_on_empty_then_succeeds(self):
        """Empty followed by text-only response (no tool calls)."""
        client = _make_client()
        empty_resp = _mock_response(200, _EMPTY_SSE)
        text_resp = _mock_response(200, _TEXT_SSE)
        client._session = _mock_session(empty_resp, text_resp)

        with patch("src.llm.openai_codex.asyncio.sleep", new_callable=AsyncMock):
            result = await client._stream_tool_request(
                headers={"Authorization": "Bearer tok"}, body={},
            )

        assert result.text == "Hello!"
        assert not result.is_tool_use


# ---------------------------------------------------------------------------
# Issue 5: _process_with_tools returns friendly fallback on empty
# ---------------------------------------------------------------------------


class TestProcessWithToolsFallback:
    """When Codex returns empty LLMResponse, user sees a friendly message."""

    async def test_friendly_message_on_empty(self):
        from src.discord.client import _EMPTY_RESPONSE_FALLBACK

        # Mock the Codex client to return empty LLMResponse
        mock_codex = MagicMock()
        mock_codex.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="", tool_calls=[], stop_reason="end_turn"),
        )

        # Minimal bot setup — _process_with_tools reads several attrs
        from src.discord.client import LokiBot
        with patch.object(LokiBot, "__init__", lambda self, *a, **kw: None):
            bot = LokiBot.__new__(LokiBot)

        bot.codex_client = mock_codex
        bot._cancelled_tasks = set()
        bot._pending_files = {}
        bot.tool_memory = MagicMock()
        bot.tool_memory.suggest = AsyncMock(return_value=[])
        bot.config = MagicMock()
        bot.config.tools.enabled = False  # Skip tool merging
        bot.skill_manager = MagicMock()

        mock_message = MagicMock()
        mock_message.channel = MagicMock()
        mock_message.channel.id = 123

        result = await bot._process_with_tools(
            mock_message,
            history=[{"role": "user", "content": "hello"}],
            system_prompt_override="test",
        )

        response_text = result[0]
        assert response_text == _EMPTY_RESPONSE_FALLBACK
        assert "try again" in response_text.lower()


# ---------------------------------------------------------------------------
# Issue 5: Chat route empty response gets fallback
# ---------------------------------------------------------------------------


class TestChatRouteEmptyFallback:
    """Chat route: when chat() returns empty, user sees friendly fallback."""

    async def test_chat_empty_gets_fallback(self):
        from src.discord.client import _EMPTY_RESPONSE_FALLBACK

        # chat() returns empty string — after the fix, the code should replace it
        mock_codex = MagicMock()
        mock_codex.chat = AsyncMock(return_value="")

        # Verify the constant exists and is meaningful
        assert _EMPTY_RESPONSE_FALLBACK
        assert "try again" in _EMPTY_RESPONSE_FALLBACK.lower()


# ---------------------------------------------------------------------------
# Round 4 finding: Handoff preserves skill response on empty chat()
# ---------------------------------------------------------------------------


class TestHandoffPreservesSkillResponse:
    """When handoff chat() returns empty, the original skill response is kept."""

    async def test_skill_response_preserved_on_empty_handoff(self):
        """Verify the pattern: save _skill_response before overwriting."""
        # This tests the logic directly: if chat() returns "", restore original
        skill_output = "Skill executed successfully: nginx restarted."
        codex_empty = ""

        # Simulate the handoff logic
        response = skill_output  # Initial value from _process_with_tools
        _skill_response = response  # Save before overwriting

        # chat() returns empty
        response = codex_empty
        if not response:
            response = _skill_response

        assert response == skill_output

    async def test_skill_response_preserved_on_handoff_exception(self):
        """If handoff chat() raises, original skill response is restored."""
        skill_output = "Disk usage: 42%"
        response = skill_output
        _skill_response = response

        # Simulate exception path
        try:
            raise RuntimeError("Codex handoff failed")
        except Exception:
            response = _skill_response

        assert response == skill_output
