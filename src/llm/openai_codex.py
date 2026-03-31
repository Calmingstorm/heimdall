from __future__ import annotations

import asyncio
import json

import aiohttp

from ..logging import get_logger
from .circuit_breaker import CircuitBreaker
from .codex_auth import CodexAuth, CodexAuthPool
from .types import LLMResponse, ToolCall

log = get_logger("codex")

CODEX_API_URL = "https://chatgpt.com/backend-api/codex/responses"
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]


class CodexChatClient:
    """Chat client using OpenAI Codex backend API (ChatGPT subscription)."""

    def __init__(self, auth: CodexAuth | CodexAuthPool, model: str, max_tokens: int) -> None:
        self.auth = auth
        self.model = model
        self.max_tokens = max_tokens
        self.breaker = CircuitBreaker("codex_api")
        self._session: aiohttp.ClientSession | None = None
        # Tool conversion cache — avoids re-converting same tools across tool loop iterations
        self._last_tools_list: list[dict] | None = None
        self._last_tools_converted: list[dict] = []

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,                # max connections (single-host API)
                limit_per_host=10,       # same as limit — one host only
                keepalive_timeout=30,    # reuse connections across requests
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                auto_decompress=False,
                headers={"Accept-Encoding": "identity"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat(
        self, messages: list[dict], system: str,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat request via the Codex backend API (streaming).

        Args:
            max_tokens: Per-call token limit override. Falls back to
                        ``self.max_tokens`` when *None*.
        """
        access_token = await self.auth.get_access_token()
        account_id = self.auth.get_account_id()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        body = {
            "model": self.model,
            "instructions": system,
            "input": self._convert_messages(messages),
            "store": False,
            "stream": True,
        }
        # Note: Codex Responses API does not support max_output_tokens.
        # Callers needing short responses should use prompt instructions instead.

        return await self._stream_request(headers, body)

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert internal message format to Codex Responses API format."""
        codex_messages = []
        for msg in messages:
            content = msg.get("content", "")
            # Extract text from list-format content blocks (tool_use, tool_result, etc.)
            if isinstance(content, list):
                text_parts = []
                image_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "image":
                            source = block.get("source", {})
                            if isinstance(source, dict) and source.get("type") == "base64":
                                media_type = source.get("media_type", "image/png")
                                data = source.get("data", "")
                                image_parts.append({
                                    "type": "input_image",
                                    "image_url": f"data:{media_type};base64,{data}",
                                })
                        elif block.get("type") == "tool_use":
                            text_parts.append(f"[Used tool: {block.get('name', 'unknown')}]")
                        elif block.get("type") == "tool_result":
                            result_content = block.get("content", "")
                            if isinstance(result_content, str):
                                summary = result_content[:200]
                            elif isinstance(result_content, list):
                                summary = " ".join(
                                    b.get("text", "")[:200]
                                    for b in result_content
                                    if isinstance(b, dict) and b.get("type") == "text"
                                )
                            else:
                                summary = str(result_content)[:200]
                            text_parts.append(f"[Tool result: {summary}]")
                # If we have images, build multimodal content
                if image_parts:
                    msg_content = []
                    if text_parts:
                        msg_content.append({"type": "input_text", "text": " ".join(text_parts)})
                    msg_content.extend(image_parts)
                    if msg_content:
                        codex_messages.append({
                            "type": "message",
                            "role": "user",
                            "content": msg_content,
                        })
                    continue
                content = " ".join(text_parts)
                if not content:
                    continue
            elif not isinstance(content, str):
                continue

            role = msg["role"]
            # Map roles: Responses API supports user, assistant, developer, system
            if role not in ("user", "assistant", "developer", "system"):
                role = "user"

            # User messages use input_text, assistant messages use output_text
            content_type = "output_text" if role == "assistant" else "input_text"

            codex_messages.append({
                "type": "message",
                "role": role,
                "content": [{"type": content_type, "text": content}],
            })
        return codex_messages

    # ------------------------------------------------------------------
    # Tool calling support
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert internal tool definitions to OpenAI function format.

        Internal:  {"name": ..., "description": ..., "input_schema": {...}}
        OpenAI:    {"type": "function", "name": ..., "description": ..., "parameters": {...}}
        """
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

    def _convert_messages_with_tools(self, messages: list[dict]) -> list[dict]:
        """Convert internal message format to Codex Responses API format with tool support.

        Unlike _convert_messages (which flattens tool blocks to text), this method
        preserves tool call / tool result structure for the Responses API:

        - Assistant text → {"type": "message", "role": "assistant", "content": [output_text]}
        - tool_use block → {"type": "function_call", "call_id": ..., "name": ..., "arguments": "..."}
        - tool_result block → {"type": "function_call_output", "call_id": ..., "output": "..."}
        - User text → {"type": "message", "role": "user", "content": [input_text]}
        - Image blocks → {"type": "message", "role": "user", "content": [input_image]}
        """
        codex_input: list[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                if not content:
                    continue
                ct = "output_text" if role == "assistant" else "input_text"
                codex_input.append({
                    "type": "message",
                    "role": role if role in ("user", "assistant", "developer", "system") else "user",
                    "content": [{"type": ct, "text": content}],
                })
                continue

            if not isinstance(content, list):
                continue

            # Process list-format content blocks
            text_parts: list[str] = []
            image_parts: list[dict] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")

                if btype == "text":
                    text_parts.append(block.get("text", ""))

                elif btype == "tool_use":
                    # Flush any accumulated text first
                    if text_parts:
                        codex_input.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": " ".join(text_parts)}],
                        })
                        text_parts = []
                    # Convert to OpenAI function_call item
                    tool_input = block.get("input", {})
                    codex_input.append({
                        "type": "function_call",
                        "call_id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
                    })

                elif btype == "tool_result":
                    # Convert to OpenAI function_call_output item
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        output = " ".join(
                            b.get("text", "")
                            for b in result_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    elif isinstance(result_content, str):
                        output = result_content
                    else:
                        output = str(result_content)
                    codex_input.append({
                        "type": "function_call_output",
                        "call_id": block.get("tool_use_id", ""),
                        "output": output,
                    })

                elif btype == "image":
                    # Convert internal base64 image to OpenAI input_image format
                    source = block.get("source", {})
                    if isinstance(source, dict) and source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        image_parts.append({
                            "type": "input_image",
                            "image_url": f"data:{media_type};base64,{data}",
                        })

            # Flush remaining text/image parts
            if text_parts or image_parts:
                msg_content: list[dict] = []
                if text_parts:
                    ct = "output_text" if role == "assistant" else "input_text"
                    msg_content.append({"type": ct, "text": " ".join(text_parts)})
                msg_content.extend(image_parts)
                if msg_content:
                    codex_input.append({
                        "type": "message",
                        "role": role if role in ("user", "assistant", "developer", "system") else "user",
                        "content": msg_content,
                    })

        return codex_input

    def _convert_tools_cached(self, tools: list[dict]) -> list[dict]:
        """Convert tools with identity-based caching.

        Within a tool loop, the same tools list object is passed on every
        iteration. This avoids re-converting 70+ tool definitions each time.
        """
        if tools is not self._last_tools_list:
            self._last_tools_converted = self._convert_tools(tools)
            self._last_tools_list = tools
        return self._last_tools_converted

    async def chat_with_tools(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        """Send a request with tool definitions, return structured LLMResponse.

        Args:
            messages: Conversation history in internal format.
            system: System prompt text.
            tools: Tool definitions in internal format (converted internally).

        Returns:
            LLMResponse with text, tool_calls, and stop_reason.
        """
        access_token = await self.auth.get_access_token()
        account_id = self.auth.get_account_id()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        body = {
            "model": self.model,
            "instructions": system,
            "input": self._convert_messages_with_tools(messages),
            "tools": self._convert_tools_cached(tools),
            "tool_choice": "auto",
            "store": False,
            "stream": True,
        }

        return await self._stream_tool_request(headers, body)

    async def _stream_tool_request(self, headers: dict, body: dict) -> LLMResponse:
        """Send a streaming request and parse both text and function_call events."""
        self.breaker.check()
        session = await self._get_session()
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    CODEX_API_URL,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        result = await self._read_tool_stream(resp)
                        if result.text or result.tool_calls:
                            self.breaker.record_success()
                            return result
                        # 200 but empty — treat as transient, retry
                        log.warning(
                            "Codex tool request returned 200 with empty response (attempt %d/%d)",
                            attempt + 1, MAX_RETRIES,
                        )
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_BACKOFF[attempt])
                            continue
                        # Exhausted retries on empty — record as degraded
                        self.breaker.record_failure()
                        return result

                    error_body = (await resp.read()).decode("utf-8", errors="replace")

                    if resp.status == 401 and attempt == 0:
                        log.warning("Codex auth expired, refreshing token...")
                        await self.auth._refresh(self.auth._load())
                        new_token = await self.auth.get_access_token()
                        headers["Authorization"] = f"Bearer {new_token}"
                        account_id = self.auth.get_account_id()
                        if account_id:
                            headers["ChatGPT-Account-Id"] = account_id
                        continue

                    if resp.status == 429:
                        # Rate limited — rotate to next account if pool
                        if hasattr(self.auth, "mark_current_limited"):
                            self.auth.mark_current_limited()
                        self.breaker.record_failure()
                        last_error = f"HTTP 429: {error_body[:200]}"
                        if attempt < MAX_RETRIES - 1:
                            wait = RETRY_BACKOFF[attempt]
                            log.warning(
                                "Codex rate limited (attempt %d/%d): %s. Rotating + retry in %ds...",
                                attempt + 1, MAX_RETRIES, last_error, wait,
                            )
                            await asyncio.sleep(wait)
                            # Refresh token from (possibly new) account
                            access_token = await self.auth.get_access_token()
                            headers["Authorization"] = f"Bearer {access_token}"
                            if hasattr(self.auth, "get_account_id"):
                                aid = self.auth.get_account_id()
                                if aid:
                                    headers["ChatGPT-Account-Id"] = aid
                            continue

                    if resp.status in (500, 502, 503, 504):
                        self.breaker.record_failure()
                        last_error = f"HTTP {resp.status}: {error_body[:200]}"
                        if attempt < MAX_RETRIES - 1:
                            wait = RETRY_BACKOFF[attempt]
                            log.warning(
                                "Codex tool API error (attempt %d/%d): %s. Retrying in %ds...",
                                attempt + 1, MAX_RETRIES, last_error, wait,
                            )
                            await asyncio.sleep(wait)
                            continue

                    self.breaker.record_failure()
                    raise RuntimeError(f"Codex API error ({resp.status}): {error_body[:500]}")

            except aiohttp.ClientError as e:
                self.breaker.record_failure()
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    log.warning(
                        "Codex tool connection error (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1, MAX_RETRIES, e, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Codex API connection failed: {e}") from e

        raise RuntimeError(f"Codex API failed after {MAX_RETRIES} retries: {last_error}")

    async def _read_tool_stream(self, resp: aiohttp.ClientResponse) -> LLMResponse:
        """Read SSE stream and extract text content and function calls.

        Handles these SSE event types:
        - response.output_text.delta: incremental text
        - response.output_item.added: new output item (detect function_call type)
        - response.function_call_arguments.delta: streaming JSON arguments
        - response.function_call_arguments.done: complete arguments
        - response.output_item.done: finalize the output item
        - response.completed: final response object (fallback)
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        # Track in-progress function calls by output_index
        pending_calls: dict[int, dict] = {}  # {index: {"call_id": ..., "name": ..., "args": ""}}
        event_types_seen: list[str] = []

        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()

            if not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            event_types_seen.append(event_type)

            # Incremental text
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    text_parts.append(delta)

            # Complete text (sometimes sent instead of deltas)
            elif event_type == "response.output_text.done":
                done_text = event.get("text", "")
                if done_text and not text_parts:
                    text_parts.append(done_text)

            # New output item — detect function_call type
            elif event_type == "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    idx = event.get("output_index", 0)
                    pending_calls[idx] = {
                        "call_id": item.get("call_id", ""),
                        "name": item.get("name", ""),
                        "args": "",
                    }

            # Streaming function call arguments
            elif event_type == "response.function_call_arguments.delta":
                idx = event.get("output_index", 0)
                if idx in pending_calls:
                    pending_calls[idx]["args"] += event.get("delta", "")

            # Function call arguments complete
            elif event_type == "response.function_call_arguments.done":
                idx = event.get("output_index", 0)
                if idx in pending_calls:
                    call_info = pending_calls[idx]
                    try:
                        parsed_args = json.loads(call_info["args"]) if call_info["args"] else {}
                    except json.JSONDecodeError:
                        parsed_args = {}
                        log.warning("Failed to parse function call arguments: %s", call_info["args"][:200])
                    tool_calls.append(ToolCall(
                        id=call_info["call_id"],
                        name=call_info["name"],
                        input=parsed_args,
                    ))

            # Output item done — finalize any remaining pending call at this index
            elif event_type == "response.output_item.done":
                item = event.get("item", {})
                idx = event.get("output_index", 0)
                if item.get("type") == "function_call" and idx in pending_calls:
                    # If arguments.done wasn't received, try to parse from the done item
                    call_info = pending_calls.pop(idx, None)
                    if call_info and not any(tc.id == call_info["call_id"] for tc in tool_calls):
                        args_str = item.get("arguments", call_info.get("args", ""))
                        try:
                            parsed_args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            parsed_args = {}
                        tool_calls.append(ToolCall(
                            id=call_info["call_id"],
                            name=call_info["name"],
                            input=parsed_args,
                        ))

            # Content filter / refusal / error events
            elif event_type in ("response.failed", "response.incomplete", "error"):
                log.warning("Codex stream event %s: %s", event_type, json.dumps(event)[:500])

            # Final response object — fallback
            elif event_type == "response.completed":
                response_obj = event.get("response", {})
                output = response_obj.get("output", [])
                for item in output:
                    item_type = item.get("type", "")
                    if item_type == "message" and not text_parts:
                        for block in item.get("content", []):
                            text = block.get("text", "")
                            if text:
                                text_parts.append(text)
                    elif item_type == "function_call":
                        # Fallback: pick up function calls from completed event
                        call_id = item.get("call_id", "")
                        if not any(tc.id == call_id for tc in tool_calls):
                            args_str = item.get("arguments", "")
                            try:
                                parsed_args = json.loads(args_str) if args_str else {}
                            except json.JSONDecodeError:
                                parsed_args = {}
                            tool_calls.append(ToolCall(
                                id=call_id,
                                name=item.get("name", ""),
                                input=parsed_args,
                            ))

        text = "".join(text_parts)
        if not text and not tool_calls:
            log.warning("Codex tool stream empty (events: %s, pending: %s)",
                        event_types_seen, list(pending_calls.keys()))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(text=text, tool_calls=tool_calls, stop_reason=stop_reason)

    async def _stream_request(self, headers: dict, body: dict) -> str:
        """Send a streaming request and collect the full response text."""
        self.breaker.check()  # Fast-fail if Codex is known to be down
        session = await self._get_session()
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    CODEX_API_URL,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        result = await self._read_stream(resp)
                        if result:
                            self.breaker.record_success()
                            return result
                        # 200 but empty — treat as transient, retry
                        log.warning(
                            "Codex returned 200 with empty response (attempt %d/%d)",
                            attempt + 1, MAX_RETRIES,
                        )
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_BACKOFF[attempt])
                            continue
                        # Exhausted retries on empty — record as degraded
                        self.breaker.record_failure()
                        return result

                    error_body = (await resp.read()).decode("utf-8", errors="replace")

                    # Auth expired — try refreshing once
                    if resp.status == 401 and attempt == 0:
                        log.warning("Codex auth expired, refreshing token...")
                        await self.auth._refresh(self.auth._load())
                        new_token = await self.auth.get_access_token()
                        headers["Authorization"] = f"Bearer {new_token}"
                        account_id = self.auth.get_account_id()
                        if account_id:
                            headers["ChatGPT-Account-Id"] = account_id
                        continue

                    if resp.status == 429:
                        if hasattr(self.auth, "mark_current_limited"):
                            self.auth.mark_current_limited()
                        self.breaker.record_failure()
                        last_error = f"HTTP 429: {error_body[:200]}"
                        if attempt < MAX_RETRIES - 1:
                            wait = RETRY_BACKOFF[attempt]
                            log.warning(
                                "Codex rate limited (attempt %d/%d): %s. Rotating + retry in %ds...",
                                attempt + 1, MAX_RETRIES, last_error, wait,
                            )
                            await asyncio.sleep(wait)
                            access_token = await self.auth.get_access_token()
                            headers["Authorization"] = f"Bearer {access_token}"
                            if hasattr(self.auth, "get_account_id"):
                                aid = self.auth.get_account_id()
                                if aid:
                                    headers["ChatGPT-Account-Id"] = aid
                            continue

                    if resp.status in (500, 502, 503, 504):
                        self.breaker.record_failure()
                        last_error = f"HTTP {resp.status}: {error_body[:200]}"
                        if attempt < MAX_RETRIES - 1:
                            wait = RETRY_BACKOFF[attempt]
                            log.warning(
                                "Codex API error (attempt %d/%d): %s. Retrying in %ds...",
                                attempt + 1, MAX_RETRIES, last_error, wait,
                            )
                            await asyncio.sleep(wait)
                            continue

                    self.breaker.record_failure()
                    raise RuntimeError(f"Codex API error ({resp.status}): {error_body[:500]}")

            except aiohttp.ClientError as e:
                self.breaker.record_failure()
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    log.warning(
                        "Codex connection error (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1, MAX_RETRIES, e, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Codex API connection failed: {e}") from e

        raise RuntimeError(f"Codex API failed after {MAX_RETRIES} retries: {last_error}")

    async def _read_stream(self, resp: aiohttp.ClientResponse) -> str:
        """Read SSE stream and extract text content."""
        text_parts = []

        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()

            if not line.startswith("data: "):
                continue

            data_str = line[6:]  # strip "data: " prefix
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Handle different SSE event types
            event_type = event.get("type", "")

            # response.output_text.delta — incremental text
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    text_parts.append(delta)

            # response.output_text.done — complete text (sometimes sent instead of deltas)
            elif event_type == "response.output_text.done":
                done_text = event.get("text", "")
                if done_text and not text_parts:
                    text_parts.append(done_text)

            # response.completed — final response object
            elif event_type == "response.completed":
                response = event.get("response", {})
                output = response.get("output", [])
                for item in output:
                    if item.get("type") == "message":
                        for block in item.get("content", []):
                            text = block.get("text", "")
                            if text:
                                # Only use completed output if we didn't get deltas
                                if not text_parts:
                                    text_parts.append(text)

        if not text_parts:
            log.warning("Codex stream returned 200 but produced no text content")
            return ""
        return "".join(text_parts)
