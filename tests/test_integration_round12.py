"""Integration tests for the free backend migration (Round 12).

Tests cover:
1. Multi-turn Codex tool loop (3+ iterations)
2. Haiku circuit breaker → heuristic fallback chain
3. Haiku malformed response handling (missing keys, error objects)
4. Codex error handling when Codex raises
5. End-to-end routing: classify → route → tool loop → result
6. Compaction and reflection with Codex callable
7. Codex tool loop edge cases
8. Chat route error handling
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402
from src.llm.haiku_classifier import HaikuClassifier  # noqa: E402
from src.llm.circuit_breaker import CircuitOpenError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal LokiBot stub with all required attributes."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._last_tool_use = {}
    stub._system_prompt = "test system prompt"
    stub._channel_locks = {}
    stub._processed_messages = MagicMock()
    stub._processed_messages_max = 100
    stub._background_tasks = {}
    stub._background_tasks_max = 20
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["12345"]
    stub.config.discord.channels = ["67890"]
    stub.config.tools.approval_timeout_seconds = 30
    stub.config.monitoring.alert_channel_id = "67890"
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.reset = MagicMock()
    stub.sessions.search_history = AsyncMock(return_value=[])
    stub.sessions.get_or_create = MagicMock()
    stub.classifier = MagicMock()
    stub.classifier.classify = AsyncMock(return_value="task")
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex chat response")
    stub.codex_client.chat_with_tools = AsyncMock(
        return_value=LLMResponse(text="Codex tool response", tool_calls=[], stop_reason="end_turn")
    )
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.requires_approval = MagicMock(return_value=None)
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub.tool_executor.set_user_context = MagicMock()
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._process_with_tools = AsyncMock(
        return_value=("response", False, False, [], False)
    )
    # Only non-approval tools for simpler testing (must exist in registry with requires_approval=False)
    stub._merged_tool_definitions = MagicMock(return_value=[
        {"name": "check_disk", "description": "Check disk usage", "input_schema": {"type": "object", "properties": {"host": {"type": "string"}}}},
        {"name": "check_service", "description": "Check service status", "input_schema": {"type": "object", "properties": {"host": {"type": "string"}, "service": {"type": "string"}}}},
        {"name": "check_docker", "description": "Check Docker containers", "input_schema": {"type": "object", "properties": {"host": {"type": "string"}}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.voice_manager = None
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._pending_files = {}
    stub._track_recent_action = MagicMock()
    return stub


def _make_message(channel_id="chan-1", author_id="12345"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.reply = AsyncMock()
    return msg


def _mock_haiku_response(text: str, status: int = 200, body: dict | None = None):
    """Create a mock aiohttp session for Anthropic Messages API.

    Returns a mock session that can be injected into classifier._session.
    """
    mock_resp = AsyncMock()
    mock_resp.status = status
    if body is not None:
        mock_resp.json = AsyncMock(return_value=body)
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
    mock_resp.text = AsyncMock(return_value=f"HTTP {status}")

    mock_post_ctx = AsyncMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_ctx)
    mock_session.closed = False

    return mock_session


def _inject_haiku_session(classifier, mock_session):
    """Inject a mock session into classifier so _get_session() returns it."""
    classifier._session = mock_session


# ---------------------------------------------------------------------------
# 1. Multi-Turn Codex Tool Loop (3+ iterations)
# ---------------------------------------------------------------------------

class TestMultiTurnToolLoop:
    """Test Codex tool loop with multiple iterations of tool use."""

    async def test_three_iteration_tool_loop(self):
        """Codex uses tools across 3 iterations before final text response."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="Checking disk...",
                tool_calls=[ToolCall(id="c1", name="check_disk", input={"host": "server"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="Now memory...",
                tool_calls=[ToolCall(id="c2", name="check_service", input={"host": "server"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="Now CPU...",
                tool_calls=[ToolCall(id="c3", name="check_docker", input={"host": "server"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="Server is healthy: disk 42%, memory 60%, CPU 15%.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ])
        stub.tool_executor.execute = AsyncMock(side_effect=[
            "Filesystem  Size  Used\n/  50G  21G",
            "Total: 16G  Used: 9.6G  Free: 6.4G",
            "CPU: 15% idle",
        ])
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, already_sent, is_error, tools_used, handoff = await stub._process_with_tools(
                msg, [], system_prompt_override="test prompt",
            )

        assert text == "Server is healthy: disk 42%, memory 60%, CPU 15%."
        assert tools_used == ["check_disk", "check_service", "check_docker"]
        assert stub.codex_client.chat_with_tools.call_count == 4
        assert stub.tool_executor.execute.call_count == 3
        assert is_error is False

    async def test_multi_turn_preserves_message_history(self):
        """Each iteration should include previous tool_use/tool_result in messages."""
        stub = _make_bot_stub()
        msg = _make_message()

        captured_messages = []

        async def capture_chat(messages, system, tools):
            captured_messages.append([dict(m) for m in messages])
            if len(captured_messages) == 1:
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="c1", name="check_disk", input={"host": "server"})],
                    stop_reason="tool_use",
                )
            elif len(captured_messages) == 2:
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="c2", name="check_service", input={"host": "server"})],
                    stop_reason="tool_use",
                )
            return LLMResponse(text="All good.", tool_calls=[], stop_reason="end_turn")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=capture_chat)
        stub.tool_executor.execute = AsyncMock(return_value="ok")
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [], system_prompt_override="test")

        # Third call should have 4 extra messages (2 tool exchanges)
        third_call_msgs = captured_messages[2]
        tool_msgs = [m for m in third_call_msgs if
                     m.get("role") == "assistant" and
                     any(b.get("type") == "tool_use" for b in m.get("content", []))]
        assert len(tool_msgs) == 2

    async def test_multi_tool_calls_per_iteration(self):
        """Multiple tool calls in a single iteration, then another iteration."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="Checking both hosts...",
                tool_calls=[
                    ToolCall(id="c1", name="check_disk", input={"host": "server"}),
                    ToolCall(id="c2", name="check_disk", input={"host": "desktop"}),
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="Now memory...",
                tool_calls=[ToolCall(id="c3", name="check_service", input={"host": "server"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="All systems healthy.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub.tool_executor.execute = AsyncMock(return_value="ok")
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [], system_prompt_override="test",
            )

        assert tools_used == ["check_disk", "check_disk", "check_service"]
        assert stub.tool_executor.execute.call_count == 3


# ---------------------------------------------------------------------------
# 2. Haiku Circuit Breaker → Heuristic Fallback
# ---------------------------------------------------------------------------

class TestHaikuFallbackChain:
    """Test Haiku classifier fallback behavior."""

    async def test_circuit_open_returns_chat(self):
        """When Haiku circuit breaker is open, classify returns 'chat'."""
        classifier = HaikuClassifier(api_key="test-key")
        classifier.breaker.check = MagicMock(
            side_effect=CircuitOpenError("haiku_classify", retry_after=60.0)
        )

        result = await classifier.classify("check disk on server")
        assert result == "chat"

    async def test_connection_error_returns_task(self):
        """When Haiku is unreachable, classify returns 'task' (fail-safe)."""
        classifier = HaikuClassifier(api_key="test-key")

        import aiohttp
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))
        _inject_haiku_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("check disk on server")
        assert result == "task"

    async def test_repeated_failures_open_circuit(self):
        """3 consecutive failures should open the circuit breaker."""
        classifier = HaikuClassifier(api_key="test-key")

        import aiohttp
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("refused"))
        _inject_haiku_session(classifier, mock_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            for _ in range(3):
                result = await classifier.classify("test")
                assert result == "task"

        # After 3 failures, circuit should be open
        result = await classifier.classify("test again")
        assert result == "chat"  # CircuitOpenError → "chat"

    async def test_success_after_failure_resets_circuit(self):
        """Successful classification after failure resets the breaker."""
        classifier = HaikuClassifier(api_key="test-key")

        # One failure
        import aiohttp
        fail_session = MagicMock()
        fail_session.closed = False
        fail_session.post = MagicMock(side_effect=aiohttp.ClientError("refused"))
        _inject_haiku_session(classifier, fail_session)

        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            await classifier.classify("test")

        # Then success
        _inject_haiku_session(classifier, _mock_haiku_response("task"))
        result = await classifier.classify("check disk")
        assert result == "task"

        # Should still work (circuit not open)
        _inject_haiku_session(classifier, _mock_haiku_response("chat"))
        result = await classifier.classify("hello")
        assert result == "chat"


# ---------------------------------------------------------------------------
# 3. Haiku Malformed Response Handling
# ---------------------------------------------------------------------------

class TestHaikuMalformedResponses:
    """Test Haiku classifier with malformed responses."""

    async def test_missing_content_key(self):
        """Haiku returns JSON without 'content' key → fallback to 'task'."""
        classifier = HaikuClassifier(api_key="test-key")
        _inject_haiku_session(classifier, _mock_haiku_response(
            "", body={"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}}
        ))
        result = await classifier.classify("check disk")
        assert result == "task"

    async def test_empty_response_string(self):
        """Haiku returns empty response string → fallback to 'task'."""
        classifier = HaikuClassifier(api_key="test-key")
        _inject_haiku_session(classifier, _mock_haiku_response(""))
        result = await classifier.classify("check disk")
        assert result == "task"

    async def test_multi_word_response(self):
        """Haiku returns multi-word response → fallback to 'task'."""
        classifier = HaikuClassifier(api_key="test-key")
        _inject_haiku_session(classifier, _mock_haiku_response(
            "I think this is a task"
        ))
        result = await classifier.classify("check disk")
        assert result == "task"

    async def test_http_error_falls_back_to_task(self):
        """Retryable HTTP errors (503) should fall back to 'task' after retry."""
        classifier = HaikuClassifier(api_key="test-key")
        _inject_haiku_session(classifier, _mock_haiku_response("", status=503))
        with patch("src.llm.haiku_classifier.asyncio.sleep", new_callable=AsyncMock):
            result = await classifier.classify("check disk")
        assert result == "task"

    async def test_empty_content_array(self):
        """Haiku returns JSON with empty content array → fallback to 'task'."""
        classifier = HaikuClassifier(api_key="test-key")
        _inject_haiku_session(classifier, _mock_haiku_response(
            "", body={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-haiku-4-5-20251001",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 0},
            }
        ))
        result = await classifier.classify("check disk")
        # Empty content array → data["content"][0] raises IndexError → "task"
        assert result == "task"


# ---------------------------------------------------------------------------
# 4. Codex Error Handling
# ---------------------------------------------------------------------------

class TestTaskRouteErrorHandling:
    """Tests for task route error handling (Codex only)."""

    async def test_codex_exception_sends_error_response(self):
        """When Codex raises, task route catches and sends error via _send_chunked."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="task")

        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("Codex API error"))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "check disk on server", "chan-1")

        # Codex exception is caught internally, error sent via _send_chunked
        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "tool execution failed" in sent_text.lower() or "codex api error" in sent_text.lower()

    async def test_codex_failure_returns_error(self):
        """When Codex tool loop fails, sends error message."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="task")

        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("Codex API error"))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Should send error since Codex failed
        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "error" in sent_text.lower() or "fail" in sent_text.lower() or "unavailable" in sent_text.lower()


# ---------------------------------------------------------------------------
# 5. End-to-End Routing: Classify → Route → Tool Loop → Result
# ---------------------------------------------------------------------------

class TestEndToEndRouting:
    """Test complete message flow from classification to response."""

    async def test_task_message_end_to_end(self):
        """Task message: Haiku classifies → Codex tool loop → response."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="task")

        # _process_with_tools always uses Codex (no use_codex parameter needed)
        stub._process_with_tools = AsyncMock(
            return_value=("Disk is 42% full on server.", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "check disk on server", "chan-1")

        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "42%" in sent_text

    async def test_chat_message_end_to_end(self):
        """Chat message: Haiku classifies → Codex chat → response."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub.codex_client.chat = AsyncMock(return_value="Hello! How can I help?")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello there", "chan-1")

        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "Hello" in sent_text

    async def test_keyword_bypass_skips_classifier(self):
        """Keyword-matched messages bypass classifier entirely."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="chat")  # Would be wrong if called

        async def mock_process(msg, hist, **kwargs):
            return ("Disk checked.", False, False, ["check_disk"], False)

        stub._process_with_tools = mock_process
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Classifier should NOT be called for keyword-matched messages
        stub.classifier.classify.assert_not_called()

    async def test_claude_code_message_end_to_end(self):
        """claude_code message: Haiku classifies → claude -p CLI → response."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="The function processes HTTP requests using async handlers."
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "explain the request handler", "chan-1")

        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "async handlers" in sent_text

    async def test_classifier_failure_falls_back_to_task(self):
        """When classifier raises, fallback to 'task' and use Codex tool loop."""
        stub = _make_bot_stub()
        # Simulate a real HaikuClassifier that fails and returns "task"
        stub.classifier.classify = AsyncMock(return_value="task")
        stub._process_with_tools = AsyncMock(
            return_value=("Here's the result.", False, False, ["some_tool"], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "ambiguous message", "chan-1")

        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "result" in sent_text

    async def test_circuit_open_routes_chat_end_to_end(self):
        """When classifier circuit opens (returns 'chat'), client routes to Codex chat."""
        stub = _make_bot_stub()
        # Simulate circuit breaker being open → classifier returns "chat"
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub.codex_client.chat = AsyncMock(return_value="I'm here to help!")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "check disk on server", "chan-1")

        # Even though this looks like a task, circuit open → "chat" → Codex chat
        stub.codex_client.chat.assert_called()
        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "help" in sent_text.lower()

    async def test_classifier_circuit_open_routes_to_chat(self):
        """When Haiku circuit breaker is open, classifier returns 'chat'."""
        classifier = HaikuClassifier(api_key="test-key")
        # Trip the circuit breaker
        for _ in range(3):
            classifier.breaker.record_failure()

        result = await classifier.classify("check disk on server")
        assert result == "chat"

    async def test_classifier_passes_skill_hints(self):
        """Classifier receives skill hints from the client context."""
        classifier = HaikuClassifier(api_key="test-key")
        mock_session = _mock_haiku_response("task")
        _inject_haiku_session(classifier, mock_session)

        result = await classifier.classify(
            "what's the weather",
            skill_hints="weather (get current weather), news (fetch headlines)",
        )
        assert result == "task"

        # Verify skill hints were included in the system prompt
        call_kwargs = mock_session.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "weather" in body["system"]
        assert "news" in body["system"]

    async def test_classifier_passes_recent_tool_use_context(self):
        """Classifier receives recent tool use context."""
        classifier = HaikuClassifier(api_key="test-key")
        mock_session = _mock_haiku_response("task")
        _inject_haiku_session(classifier, mock_session)

        result = await classifier.classify(
            "and the desktop?",
            has_recent_tool_use=True,
        )
        assert result == "task"

        # Verify tool use context was included in the system prompt
        call_kwargs = mock_session.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "recently ran tool commands" in body["system"]


# ---------------------------------------------------------------------------
# 6. Compaction and Reflection with Codex Callable
# ---------------------------------------------------------------------------

class TestCompactionReflectionIntegration:
    """Test compaction and reflection with Codex callable injection."""

    async def test_compaction_uses_codex_callable(self):
        """SessionManager._compact should use the injected Codex compaction fn."""
        import tempfile
        from src.sessions.manager import SessionManager, COMPACTION_THRESHOLD

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(max_history=50, max_age_hours=24, persist_dir=tmpdir)

            compaction_called = False
            async def mock_compaction(messages, system):
                nonlocal compaction_called
                compaction_called = True
                return "Summary: Users discussed disk usage."

            mgr.set_compaction_fn(mock_compaction)

            # Create a session with enough messages to exceed COMPACTION_THRESHOLD
            session = mgr.get_or_create("test-channel")
            for i in range(COMPACTION_THRESHOLD + 2):
                role = "user" if i % 2 == 0 else "assistant"
                mgr.add_message("test-channel", role, f"Message {i}")

            # Trigger compaction
            history = await mgr.get_history_with_compaction("test-channel")

            assert compaction_called
            assert session.summary is not None
            assert "disk usage" in session.summary.lower()

    async def test_compaction_fallback_on_failure(self):
        """When compaction callable fails, history should still be trimmed."""
        import tempfile
        from src.sessions.manager import SessionManager, COMPACTION_THRESHOLD

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(max_history=50, max_age_hours=24, persist_dir=tmpdir)

            async def failing_compaction(messages, system):
                raise RuntimeError("Codex is down")

            mgr.set_compaction_fn(failing_compaction)

            session = mgr.get_or_create("test-channel")
            for i in range(COMPACTION_THRESHOLD + 2):
                role = "user" if i % 2 == 0 else "assistant"
                mgr.add_message("test-channel", role, f"Message {i}")

            # Should not raise — fallback trims without summary
            history = await mgr.get_history_with_compaction("test-channel")
            assert isinstance(history, list)

    async def test_reflection_uses_codex_callable(self):
        """ConversationReflector should use injected text_fn."""
        import tempfile
        from src.learning.reflector import ConversationReflector

        with tempfile.TemporaryDirectory() as tmpdir:
            reflector = ConversationReflector(
                learned_path=f"{tmpdir}/learned.json",
            )

            text_fn_called = False
            async def mock_text_fn(messages, system):
                nonlocal text_fn_called
                text_fn_called = True
                return "[]"  # Empty insights list

            reflector.set_text_fn(mock_text_fn)

            # _reflect requires full=True/False keyword arg
            await reflector._reflect(
                "test conversation text", full=True, user_ids=["user1"]
            )
            assert text_fn_called


# ---------------------------------------------------------------------------
# 7. Tool Loop Edge Cases
# ---------------------------------------------------------------------------

class TestToolLoopEdgeCases:
    """Test edge cases in the Codex tool loop."""

    async def test_tool_executor_error_continues_loop(self):
        """When a tool execution fails, the error is sent back to Codex."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="check_disk", input={"host": "server"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="The disk check failed. Please check if the host is reachable.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ])
        stub.tool_executor.execute = AsyncMock(side_effect=Exception("SSH connection refused"))
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(
                msg, [], system_prompt_override="test",
            )

        # Tool error should be reported back to Codex, which then responds
        assert "disk check failed" in text.lower() or "reachable" in text.lower()
        assert stub.codex_client.chat_with_tools.call_count == 2

    async def test_empty_text_response_returns_fallback(self):
        """Codex returns empty text with no tools → returns fallback message."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="", tool_calls=[], stop_reason="end_turn")
        )
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        text, _, _, _, _ = await stub._process_with_tools(
            msg, [], system_prompt_override="test",
        )

        # Should return a non-empty fallback
        assert text  # Not empty
        assert len(text) > 0

    async def test_codex_returns_text_and_tools(self):
        """Codex returns both text and tool calls — text is preserved."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="I'll check the disk for you.",
                tool_calls=[ToolCall(id="c1", name="check_disk", input={"host": "server"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Disk is 42% full.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub.tool_executor.execute = AsyncMock(return_value="50G total, 21G used")
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [], system_prompt_override="test",
            )

        assert text == "Disk is 42% full."
        assert tools_used == ["check_disk"]

    async def test_no_codex_returns_error(self):
        """When Codex is not available and task is classified, error is returned."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub.classifier.classify = AsyncMock(return_value="task")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Should have sent an error message via _send_with_retry or _send_chunked
        assert stub._send_with_retry.called or stub._send_chunked.called


# ---------------------------------------------------------------------------
# 8. Chat Route Edge Cases
# ---------------------------------------------------------------------------

class TestChatRouteEdgeCases:
    """Test chat route error handling."""

    async def test_chat_codex_failure_returns_error(self):
        """When Codex chat fails, should return error message."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub.codex_client.chat = AsyncMock(side_effect=RuntimeError("Codex down"))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1")

        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        # Should be an error message
        assert "unavailable" in sent_text.lower() or "error" in sent_text.lower() or "try again" in sent_text.lower()

    async def test_no_codex_chat_returns_error_not_haiku(self):
        """When no Codex client, chat returns error."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1")

        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "not configured" in sent_text.lower() or "unavailable" in sent_text.lower() or "error" in sent_text.lower()


# ---------------------------------------------------------------------------
# 9. Claude Code Routing Edge Cases
# ---------------------------------------------------------------------------

class TestClaudeCodeRoutingEdgeCases:
    """Test claude_code routing edge cases — 'Unknown' prefix fallback."""

    async def test_unknown_prefix_triggers_codex_fallback(self):
        """Response starting with 'Unknown' triggers Codex fallback (line 1029)."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Unknown: could not determine target host"
        )
        stub.codex_client.chat = AsyncMock(return_value="I can help with that differently.")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review the auth module", "chan-1")

        # Verify Codex fallback was used after "Unknown" prefix
        stub.codex_client.chat.assert_called_once()
        stub._send_chunked.assert_called()

    async def test_claude_code_failed_prefix_triggers_codex_fallback(self):
        """Response starting with 'Claude Code failed' also triggers Codex fallback."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Claude Code failed: SSH connection timed out"
        )
        stub.codex_client.chat = AsyncMock(return_value="Let me try another approach.")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "analyze this code", "chan-1")

        stub.codex_client.chat.assert_called_once()
        stub._send_chunked.assert_called()

    async def test_unknown_prefix_no_codex_returns_error(self):
        """'Unknown' prefix with no Codex client returns error (not crash)."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Unknown: error details"
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review code", "chan-1")

        # Should still send a response (error), not crash
        stub._send_chunked.assert_called()


# ---------------------------------------------------------------------------
# 10. Image Blocks Force Task Route
# ---------------------------------------------------------------------------

class TestImageBlocksRouting:
    """Test that messages with image attachments force the 'task' route."""

    async def test_image_blocks_force_task_route(self):
        """When image_blocks is non-empty, msg_type is forced to 'task' (line 942)."""
        stub = _make_bot_stub()
        # Classifier would return "chat" — but images should override
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._process_with_tools = AsyncMock(
            return_value=("I can see the image.", False, False, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        image_blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        # Pass image_blocks directly to _handle_message_inner (same as on_message does)
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "describe this image", "chan-1", image_blocks=image_blocks
            )

        # Classifier should NOT have been called — images bypass it
        stub.classifier.classify.assert_not_called()
        # Should have routed to task (tools) path
        stub._process_with_tools.assert_called()

    async def test_no_image_blocks_uses_classifier(self):
        """When no image attachments, classifier is called normally."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="chat")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        # Pass empty image_blocks (same as on_message does when no attachments)
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "chan-1", image_blocks=[])

        # Classifier should have been called
        stub.classifier.classify.assert_called_once()


# ---------------------------------------------------------------------------
# 11. Classifier Graceful Shutdown
# ---------------------------------------------------------------------------

class TestClassifierGracefulShutdown:
    """Test that the classifier's close() method works correctly for shutdown."""

    async def test_close_after_classify(self):
        """Classifier can be closed after use without errors."""
        classifier = HaikuClassifier(api_key="test-key")
        mock_session = _mock_haiku_response("task")
        _inject_haiku_session(classifier, mock_session)

        result = await classifier.classify("check disk")
        assert result == "task"

        # Simulate shutdown — close should work
        mock_session.close = AsyncMock()
        await classifier.close()
        mock_session.close.assert_called_once()

    async def test_close_idempotent(self):
        """Calling close() multiple times is safe."""
        classifier = HaikuClassifier(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.closed = False
        classifier._session = mock_session

        await classifier.close()
        mock_session.close.assert_called_once()

        # Second close — session is now marked as closed
        mock_session.closed = True
        await classifier.close()
        # close() should not be called again
        assert mock_session.close.call_count == 1

    async def test_classify_after_close_recreates_session(self):
        """If classifier is closed and then used again, it recreates the session."""
        classifier = HaikuClassifier(api_key="test-key")

        # First: use with a mock session
        mock_session1 = _mock_haiku_response("task")
        _inject_haiku_session(classifier, mock_session1)
        result1 = await classifier.classify("test")
        assert result1 == "task"

        # Close it
        mock_session1.close = AsyncMock()
        await classifier.close()

        # Mark as closed so _get_session() recreates
        mock_session1.closed = True

        # Second: inject a new session (simulating _get_session recreation)
        mock_session2 = _mock_haiku_response("chat")
        _inject_haiku_session(classifier, mock_session2)
        result2 = await classifier.classify("hello")
        assert result2 == "chat"
