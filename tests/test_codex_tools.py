"""Tests for Codex tool calling — format conversion, message conversion, stream parsing.

Tests the tool-calling additions to CodexChatClient (Round 7):
- _convert_tools: internal → OpenAI format
- _convert_messages_with_tools: internal format → Responses API with tool blocks
- _read_tool_stream: SSE parsing for function_call events
- chat_with_tools: end-to-end with mocked HTTP
- LLMResponse / ToolCall dataclasses
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from src.llm.codex_auth import CodexAuth
from src.llm.openai_codex import CodexChatClient
from src.llm.types import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Helpers (reused from test_openai_codex.py patterns)
# ---------------------------------------------------------------------------
def _make_auth(access_token: str = "test_token", account_id: str | None = "acct_1") -> MagicMock:
    auth = MagicMock(spec=CodexAuth)
    auth.get_access_token = AsyncMock(return_value=access_token)
    auth.get_account_id = MagicMock(return_value=account_id)
    auth._refresh = AsyncMock()
    auth._load = MagicMock(return_value={})
    return auth


def _make_client(auth: MagicMock | None = None, model: str = "o4-mini") -> CodexChatClient:
    return CodexChatClient(auth=auth or _make_auth(), model=model, max_tokens=4096)


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


def _sse_lines(*events: str | dict) -> list[bytes]:
    lines = []
    for event in events:
        if isinstance(event, dict):
            lines.append(f"data: {json.dumps(event)}\n\n".encode())
        else:
            lines.append(f"{event}\n\n".encode())
    return lines


def _mock_aiohttp_response(status: int, sse_lines: list[bytes] | None = None,
                            body: bytes = b"") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status = status
    if sse_lines is not None:
        mock_resp.content = _AsyncIterator(sse_lines)
    else:
        mock_resp.read = AsyncMock(return_value=body)
    return mock_resp


def _mock_session_post(mock_resp) -> MagicMock:
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_ctx)
    mock_session.closed = False
    mock_session.close = AsyncMock()
    return mock_session


# ---------------------------------------------------------------------------
# LLMResponse / ToolCall dataclasses
# ---------------------------------------------------------------------------
class TestToolCall:
    def test_basic_creation(self):
        tc = ToolCall(id="call_1", name="check_disk", input={"host": "server"})
        assert tc.id == "call_1"
        assert tc.name == "check_disk"
        assert tc.input == {"host": "server"}

    def test_empty_input(self):
        tc = ToolCall(id="call_2", name="list_hosts", input={})
        assert tc.input == {}


class TestLLMResponse:
    def test_text_only_response(self):
        resp = LLMResponse(text="Hello world")
        assert resp.text == "Hello world"
        assert resp.tool_calls == []
        assert resp.stop_reason == "end_turn"
        assert not resp.is_tool_use

    def test_tool_use_response(self):
        tc = ToolCall(id="call_1", name="check_disk", input={"host": "server"})
        resp = LLMResponse(text="Let me check", tool_calls=[tc], stop_reason="tool_use")
        assert resp.is_tool_use
        assert len(resp.tool_calls) == 1
        assert resp.stop_reason == "tool_use"

    def test_is_tool_use_with_calls_but_end_turn(self):
        """is_tool_use is True if tool_calls exist even with end_turn stop_reason."""
        tc = ToolCall(id="call_1", name="check_disk", input={})
        resp = LLMResponse(tool_calls=[tc], stop_reason="end_turn")
        assert resp.is_tool_use

    def test_defaults(self):
        resp = LLMResponse()
        assert resp.text == ""
        assert resp.tool_calls == []
        assert resp.stop_reason == "end_turn"
        assert not resp.is_tool_use


# ---------------------------------------------------------------------------
# _convert_tools
# ---------------------------------------------------------------------------
class TestConvertTools:
    def test_basic_conversion(self):
        """Converts internal tool format to OpenAI function format."""
        tool_defs = [
            {
                "name": "check_disk",
                "description": "Check disk usage on a host",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "description": "Target host"},
                    },
                    "required": ["host"],
                },
            },
        ]
        result = CodexChatClient._convert_tools(tool_defs)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "check_disk"
        assert result[0]["description"] == "Check disk usage on a host"
        assert result[0]["parameters"]["type"] == "object"
        assert "host" in result[0]["parameters"]["properties"]
        assert result[0]["parameters"]["required"] == ["host"]

    def test_multiple_tools(self):
        """Converts multiple tools."""
        tools = [
            {"name": "check_disk", "description": "Disk", "input_schema": {"type": "object", "properties": {}}},
            {"name": "check_memory", "description": "Memory", "input_schema": {"type": "object", "properties": {}}},
            {"name": "restart_service", "description": "Restart", "input_schema": {"type": "object", "properties": {}}},
        ]
        result = CodexChatClient._convert_tools(tools)
        assert len(result) == 3
        assert [t["name"] for t in result] == ["check_disk", "check_memory", "restart_service"]
        assert all(t["type"] == "function" for t in result)

    def test_missing_description(self):
        """Tool with no description gets empty string."""
        tools = [{"name": "test", "input_schema": {"type": "object", "properties": {}}}]
        result = CodexChatClient._convert_tools(tools)
        assert result[0]["description"] == ""

    def test_missing_input_schema(self):
        """Tool with no input_schema gets default empty object schema."""
        tools = [{"name": "test", "description": "Test"}]
        result = CodexChatClient._convert_tools(tools)
        assert result[0]["parameters"] == {"type": "object", "properties": {}}

    def test_empty_tools_list(self):
        result = CodexChatClient._convert_tools([])
        assert result == []

    def test_complex_schema_preserved(self):
        """Complex nested schemas are passed through unchanged."""
        schema = {
            "type": "object",
            "properties": {
                "host": {"type": "string", "enum": ["desktop", "server"]},
                "options": {
                    "type": "object",
                    "properties": {"verbose": {"type": "boolean"}},
                },
            },
            "required": ["host"],
        }
        tools = [{"name": "run_command", "description": "Run cmd", "input_schema": schema}]
        result = CodexChatClient._convert_tools(tools)
        assert result[0]["parameters"] == schema


# ---------------------------------------------------------------------------
# _convert_messages_with_tools
# ---------------------------------------------------------------------------
class TestConvertMessagesWithTools:
    def test_plain_user_message(self):
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "user", "content": "Check the disk"},
        ])
        assert len(result) == 1
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "Check the disk"

    def test_plain_assistant_message(self):
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "assistant", "content": "Sure, checking now."},
        ])
        assert len(result) == 1
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "output_text"

    def test_tool_use_block(self):
        """tool_use blocks become function_call items."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_123", "name": "check_disk", "input": {"host": "server"}},
            ]},
        ])
        assert len(result) == 1
        assert result[0]["type"] == "function_call"
        assert result[0]["call_id"] == "tu_123"
        assert result[0]["name"] == "check_disk"
        assert json.loads(result[0]["arguments"]) == {"host": "server"}

    def test_tool_result_block(self):
        """tool_result blocks become function_call_output items."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_123", "content": "Disk 80% used"},
            ]},
        ])
        assert len(result) == 1
        assert result[0]["type"] == "function_call_output"
        assert result[0]["call_id"] == "tu_123"
        assert result[0]["output"] == "Disk 80% used"

    def test_text_and_tool_use_in_same_message(self):
        """Assistant message with text + tool_use: text flushed before function_call."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check the disk."},
                {"type": "tool_use", "id": "tu_1", "name": "check_disk", "input": {"host": "server"}},
            ]},
        ])
        assert len(result) == 2
        # First: flushed text
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "assistant"
        assert "Let me check the disk" in result[0]["content"][0]["text"]
        # Second: function call
        assert result[1]["type"] == "function_call"
        assert result[1]["name"] == "check_disk"

    def test_multiple_tool_results(self):
        """Multiple tool_result blocks produce multiple function_call_output items."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "Result 1"},
                {"type": "tool_result", "tool_use_id": "tu_2", "content": "Result 2"},
            ]},
        ])
        assert len(result) == 2
        assert all(r["type"] == "function_call_output" for r in result)
        assert result[0]["call_id"] == "tu_1"
        assert result[1]["call_id"] == "tu_2"

    def test_tool_result_with_list_content(self):
        """tool_result with list-format content extracts text."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": [
                    {"type": "text", "text": "Server healthy"},
                ]},
            ]},
        ])
        assert result[0]["output"] == "Server healthy"

    def test_image_block_conversion(self):
        """Internal image blocks converted to OpenAI input_image format."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBOR...",
                }},
            ]},
        ])
        assert len(result) == 1
        assert result[0]["type"] == "message"
        content = result[0]["content"]
        assert any(c["type"] == "input_image" for c in content)
        image_block = [c for c in content if c["type"] == "input_image"][0]
        assert image_block["image_url"] == "data:image/png;base64,iVBOR..."

    def test_text_with_image(self):
        """Message with both text and image."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "abc123",
                }},
            ]},
        ])
        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_image"

    def test_empty_string_content_skipped(self):
        """Messages with empty string content are skipped."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "user", "content": ""},
        ])
        assert result == []

    def test_unknown_role_mapped_to_user(self):
        """Non-standard roles are mapped to user; developer/system are preserved."""
        client = _make_client()
        # "system" and "developer" are now preserved (not mapped to "user")
        result = client._convert_messages_with_tools([
            {"role": "system", "content": "System message"},
        ])
        assert result[0]["role"] == "system"
        # Truly unknown roles still map to "user"
        result2 = client._convert_messages_with_tools([
            {"role": "observer", "content": "Observer message"},
        ])
        assert result2[0]["role"] == "user"

    def test_full_tool_loop_conversation(self):
        """A realistic multi-turn conversation with tool calls and results."""
        client = _make_client()
        messages = [
            {"role": "user", "content": "Check disk on server"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I'll check the disk usage."},
                {"type": "tool_use", "id": "tu_1", "name": "check_disk", "input": {"host": "server"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "/dev/sda1: 80% used"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "The server disk is 80% full."},
            ]},
        ]
        result = client._convert_messages_with_tools(messages)

        # Message 1: user text
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "user"
        # Message 2a: assistant text (flushed before tool_use)
        assert result[1]["type"] == "message"
        assert result[1]["role"] == "assistant"
        # Message 2b: function_call
        assert result[2]["type"] == "function_call"
        assert result[2]["name"] == "check_disk"
        # Message 3: function_call_output
        assert result[3]["type"] == "function_call_output"
        assert result[3]["call_id"] == "tu_1"
        # Message 4: assistant text
        assert result[4]["type"] == "message"
        assert result[4]["role"] == "assistant"

    def test_multiple_tool_uses_in_one_message(self):
        """Multiple tool_use blocks in one assistant message."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "check_disk", "input": {"host": "server"}},
                {"type": "tool_use", "id": "tu_2", "name": "check_memory", "input": {"host": "server"}},
            ]},
        ])
        assert len(result) == 2
        assert result[0]["type"] == "function_call"
        assert result[0]["name"] == "check_disk"
        assert result[1]["type"] == "function_call"
        assert result[1]["name"] == "check_memory"

    def test_non_dict_content_skipped(self):
        """Messages with non-string non-list content are skipped."""
        client = _make_client()
        result = client._convert_messages_with_tools([
            {"role": "user", "content": 42},
        ])
        assert result == []


# ---------------------------------------------------------------------------
# _read_tool_stream — SSE parsing
# ---------------------------------------------------------------------------
class TestReadToolStream:
    async def test_text_only_response(self):
        """Text-only response returns LLMResponse with end_turn."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_text.delta", "delta": "Hello "},
            {"type": "response.output_text.delta", "delta": "world"},
            {"type": "response.completed", "response": {"output": []}},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert result.text == "Hello world"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"
        assert not result.is_tool_use

    async def test_single_function_call(self):
        """Parses a single function call from streaming events."""
        client = _make_client()
        lines = _sse_lines(
            # Function call announced
            {"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "function_call", "call_id": "call_abc", "name": "check_disk",
            }},
            # Arguments streamed
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"ho'},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": 'st":'},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": ' "server"}'},
            # Arguments done
            {"type": "response.function_call_arguments.done", "output_index": 0},
            {"type": "response.completed", "response": {"output": []}},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert result.stop_reason == "tool_use"
        assert result.is_tool_use
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "check_disk"
        assert tc.input == {"host": "server"}

    async def test_text_plus_function_call(self):
        """Response with both text and a function call."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_text.delta", "delta": "Let me check. "},
            {"type": "response.output_item.added", "output_index": 1, "item": {
                "type": "function_call", "call_id": "call_1", "name": "check_disk",
            }},
            {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": '{}'},
            {"type": "response.function_call_arguments.done", "output_index": 1},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert result.text == "Let me check. "
        assert result.is_tool_use
        assert result.tool_calls[0].name == "check_disk"
        assert result.tool_calls[0].input == {}

    async def test_multiple_function_calls(self):
        """Multiple function calls in one response."""
        client = _make_client()
        lines = _sse_lines(
            # First function call
            {"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "check_disk",
            }},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"host": "server"}'},
            {"type": "response.function_call_arguments.done", "output_index": 0},
            # Second function call
            {"type": "response.output_item.added", "output_index": 1, "item": {
                "type": "function_call", "call_id": "call_2", "name": "check_memory",
            }},
            {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": '{"host": "server"}'},
            {"type": "response.function_call_arguments.done", "output_index": 1},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "check_disk"
        assert result.tool_calls[1].name == "check_memory"

    async def test_function_call_from_output_item_done(self):
        """Fallback: function_call parsed from output_item.done when arguments.done is missing."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "list_hosts",
            }},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{}'},
            # No arguments.done, but output_item.done has full item
            {"type": "response.output_item.done", "output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "list_hosts",
                "arguments": "{}",
            }},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "list_hosts"

    async def test_function_call_from_completed_fallback(self):
        """Fallback: function_call parsed from response.completed when events are missing."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.completed", "response": {
                "output": [{
                    "type": "function_call",
                    "call_id": "call_fallback",
                    "name": "check_disk",
                    "arguments": '{"host": "desktop"}',
                }],
            }},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_fallback"
        assert result.tool_calls[0].name == "check_disk"
        assert result.tool_calls[0].input == {"host": "desktop"}

    async def test_completed_does_not_duplicate_calls(self):
        """response.completed doesn't duplicate calls already parsed from events."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "check_disk",
            }},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{}'},
            {"type": "response.function_call_arguments.done", "output_index": 0},
            {"type": "response.completed", "response": {
                "output": [{
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "check_disk",
                    "arguments": "{}",
                }],
            }},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert len(result.tool_calls) == 1

    async def test_invalid_json_arguments_handled(self):
        """Invalid JSON in function call arguments doesn't crash, uses empty dict."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "test_tool",
            }},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{invalid json'},
            {"type": "response.function_call_arguments.done", "output_index": 0},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input == {}

    async def test_empty_response(self):
        """Empty response returns LLMResponse with no content."""
        client = _make_client()
        lines = _sse_lines({"type": "response.created"})
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert result.text == ""
        assert result.tool_calls == []
        assert not result.is_tool_use

    async def test_done_sentinel_stops_parsing(self):
        """[DONE] sentinel stops SSE parsing."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_text.delta", "delta": "Before"},
            "data: [DONE]",
            {"type": "response.output_text.delta", "delta": "After"},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert result.text == "Before"

    async def test_completed_text_fallback(self):
        """Text from response.completed is used when no deltas received."""
        client = _make_client()
        lines = _sse_lines({
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Final text"}],
                }],
            },
        })
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert result.text == "Final text"

    async def test_empty_arguments_parsed_as_empty_dict(self):
        """Empty string arguments are parsed as empty dict."""
        client = _make_client()
        lines = _sse_lines(
            {"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "list_hosts",
            }},
            # No argument deltas at all
            {"type": "response.function_call_arguments.done", "output_index": 0},
        )
        resp = _mock_aiohttp_response(200, lines)
        result = await client._read_tool_stream(resp)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input == {}


# ---------------------------------------------------------------------------
# chat_with_tools — end-to-end with mocked HTTP
# ---------------------------------------------------------------------------
class TestChatWithTools:
    async def test_text_response(self):
        """chat_with_tools returns text-only LLMResponse when no tools called."""
        auth = _make_auth()
        client = _make_client(auth=auth)

        sse = _sse_lines(
            {"type": "response.output_text.delta", "delta": "All disks look fine."},
            {"type": "response.completed", "response": {"output": []}},
        )
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        tools = [{"name": "check_disk", "description": "Check disk", "input_schema": {"type": "object", "properties": {}}}]
        result = await client.chat_with_tools(
            messages=[{"role": "user", "content": "How are the disks?"}],
            system="You are an infra assistant.",
            tools=tools,
        )
        assert isinstance(result, LLMResponse)
        assert result.text == "All disks look fine."
        assert not result.is_tool_use

    async def test_tool_call_response(self):
        """chat_with_tools returns LLMResponse with tool calls."""
        auth = _make_auth()
        client = _make_client(auth=auth)

        sse = _sse_lines(
            {"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "check_disk",
            }},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"host": "server"}'},
            {"type": "response.function_call_arguments.done", "output_index": 0},
            {"type": "response.completed", "response": {"output": []}},
        )
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        tools = [{"name": "check_disk", "description": "Check disk", "input_schema": {"type": "object", "properties": {"host": {"type": "string"}}}}]
        result = await client.chat_with_tools(
            messages=[{"role": "user", "content": "Check disk on server"}],
            system="You are an infra assistant.",
            tools=tools,
        )
        assert result.is_tool_use
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "check_disk"
        assert result.tool_calls[0].input == {"host": "server"}

    async def test_sends_correct_body(self):
        """chat_with_tools sends tools in OpenAI format and messages with tool support."""
        auth = _make_auth()
        client = _make_client(auth=auth, model="gpt-5.3")

        sse = _sse_lines(
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"output": []}},
        )
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        tool_defs = [
            {"name": "check_disk", "description": "Disk check", "input_schema": {"type": "object", "properties": {}}},
        ]
        await client.chat_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            system="sys prompt",
            tools=tool_defs,
        )

        # Verify the request body
        post_call = mock_session.post.call_args
        body = post_call[1].get("json") or post_call[0][1] if len(post_call[0]) > 1 else None
        if body is None:
            body = post_call[1]["json"]

        assert body["model"] == "gpt-5.3"
        assert body["instructions"] == "sys prompt"
        assert body["stream"] is True
        assert body["store"] is False
        # Verify tools are in OpenAI format
        assert len(body["tools"]) == 1
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["name"] == "check_disk"
        assert body["tools"][0]["parameters"] == {"type": "object", "properties": {}}

    async def test_includes_account_id_header(self):
        """chat_with_tools includes account ID header when available."""
        auth = _make_auth(account_id="acct_42")
        client = _make_client(auth=auth)

        sse = _sse_lines(
            {"type": "response.output_text.delta", "delta": "ok"},
        )
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        await client.chat_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            system="sys",
            tools=[{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
        )

        post_call = mock_session.post.call_args
        headers = post_call[1].get("headers") or post_call[0][0] if len(post_call[0]) > 0 else None
        if headers is None:
            headers = post_call[1]["headers"]
        assert headers["ChatGPT-Account-Id"] == "acct_42"

    async def test_omits_account_id_when_none(self):
        """chat_with_tools omits account ID header when not set."""
        auth = _make_auth(account_id=None)
        client = _make_client(auth=auth)

        sse = _sse_lines(
            {"type": "response.output_text.delta", "delta": "ok"},
        )
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        await client.chat_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            system="sys",
            tools=[{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
        )

        post_call = mock_session.post.call_args
        headers = post_call[1].get("headers") or post_call[0][0]
        assert "ChatGPT-Account-Id" not in headers

    async def test_with_tool_history(self):
        """chat_with_tools correctly converts a conversation with tool history."""
        client = _make_client()

        sse = _sse_lines(
            {"type": "response.output_text.delta", "delta": "Disk is 80% full."},
        )
        mock_resp = _mock_aiohttp_response(200, sse)
        mock_session = _mock_session_post(mock_resp)
        client._session = mock_session

        messages = [
            {"role": "user", "content": "Check disk on server"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "check_disk", "input": {"host": "server"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "/dev/sda1: 80%"},
            ]},
        ]
        result = await client.chat_with_tools(
            messages=messages,
            system="sys",
            tools=[{"name": "check_disk", "description": "Check", "input_schema": {"type": "object", "properties": {}}}],
        )
        assert result.text == "Disk is 80% full."

        # Verify converted messages include function_call and function_call_output
        post_call = mock_session.post.call_args
        body = post_call[1]["json"]
        input_items = body["input"]
        types = [item["type"] for item in input_items]
        assert "message" in types
        assert "function_call" in types
        assert "function_call_output" in types

    async def test_circuit_breaker_checked(self):
        """chat_with_tools checks circuit breaker before request."""
        client = _make_client()
        client.breaker = MagicMock()
        client.breaker.check = MagicMock(side_effect=RuntimeError("circuit open"))
        client._session = MagicMock()
        client._session.closed = False

        with pytest.raises(RuntimeError, match="circuit open"):
            await client.chat_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                system="sys",
                tools=[{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
            )
