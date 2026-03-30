"""Integration tests for the free backend migration (Round 12).

Tests cover:
1. Multi-turn Codex tool loop (3+ iterations)
2. Codex error handling when Codex raises
3. End-to-end routing: all messages route to task → tool loop → result
4. Compaction and reflection with Codex callable
5. Codex tool loop edge cases
6. Image blocks force task route
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal HeimdallBot stub with all required attributes."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
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
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
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
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex chat response")
    stub.codex_client.chat_with_tools = AsyncMock(
        return_value=LLMResponse(text="Codex tool response", tool_calls=[], stop_reason="end_turn")
    )
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
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
    # Tool definitions for testing
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
    # Use the real static method so continuation detection works correctly
    stub._should_continue_task = HeimdallBot._should_continue_task
    return stub


def _make_message(channel_id="chan-1", author_id="12345"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.reply = AsyncMock()
    return msg


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
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
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
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
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
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [], system_prompt_override="test",
            )

        assert tools_used == ["check_disk", "check_disk", "check_service"]
        assert stub.tool_executor.execute.call_count == 3


# ---------------------------------------------------------------------------
# 2. Codex Error Handling
# ---------------------------------------------------------------------------

class TestTaskRouteErrorHandling:
    """Tests for task route error handling (Codex only)."""

    async def test_codex_exception_sends_error_response(self):
        """When Codex raises, task route catches and sends error via _send_chunked."""
        stub = _make_bot_stub()

        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("Codex API error"))
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk on server", "chan-1")

        # Codex exception is caught internally, error sent via _send_chunked
        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "tool execution failed" in sent_text.lower() or "codex api error" in sent_text.lower()

    async def test_codex_failure_returns_error(self):
        """When Codex tool loop fails, sends error message."""
        stub = _make_bot_stub()

        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("Codex API error"))
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Should send error since Codex failed
        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "error" in sent_text.lower() or "fail" in sent_text.lower() or "unavailable" in sent_text.lower()


# ---------------------------------------------------------------------------
# 3. End-to-End Routing: All Messages → Task Route → Tool Loop → Result
# ---------------------------------------------------------------------------

class TestEndToEndRouting:
    """Test complete message flow — all messages route to task (Codex with tools)."""

    async def test_task_message_end_to_end(self):
        """Task message routes to Codex tool loop and returns response."""
        stub = _make_bot_stub()

        stub._process_with_tools = AsyncMock(
            return_value=("Disk is 42% full on server.", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk on server", "chan-1")

        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "42%" in sent_text

    async def test_all_messages_route_to_task(self):
        """Any message (even casual 'hello') routes to the task path with tools."""
        stub = _make_bot_stub()

        stub._process_with_tools = AsyncMock(
            return_value=("Hello! How can I help you today?", False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "hello", "chan-1")

        # _process_with_tools should be called (task route), not codex_client.chat (chat route)
        stub._process_with_tools.assert_called()
        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "Hello" in sent_text

    async def test_keyword_bypass_routes_to_task(self):
        """Keyword-matched messages route to task (same as all other messages)."""
        stub = _make_bot_stub()

        async def mock_process(msg, hist, **kwargs):
            return ("Disk checked.", False, False, ["check_disk"], False)

        stub._process_with_tools = mock_process
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "Disk checked" in sent_text

    async def test_default_routes_to_task(self):
        """Non-keyword messages also route to task (the default route)."""
        stub = _make_bot_stub()
        stub._process_with_tools = AsyncMock(
            return_value=("Here's the result.", False, False, ["some_tool"], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "ambiguous message", "chan-1")

        stub._process_with_tools.assert_called()
        stub._send_chunked.assert_called()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "result" in sent_text


# ---------------------------------------------------------------------------
# 4. Compaction and Reflection with Codex Callable
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
# 5. Tool Loop Edge Cases
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
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
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
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

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
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [], system_prompt_override="test",
            )

        assert text == "Disk is 42% full."
        assert tools_used == ["check_disk"]

    async def test_no_codex_returns_error(self):
        """When Codex is not available, error is returned."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Should have sent an error message via _send_with_retry or _send_chunked
        assert stub._send_with_retry.called or stub._send_chunked.called


# ---------------------------------------------------------------------------
# 6. Image Blocks Force Task Route
# ---------------------------------------------------------------------------

class TestImageBlocksRouting:
    """Test that messages with image attachments force the 'task' route."""

    async def test_image_blocks_force_task_route(self):
        """When image_blocks is non-empty, msg_type is forced to 'task'."""
        stub = _make_bot_stub()
        stub._process_with_tools = AsyncMock(
            return_value=("I can see the image.", False, False, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        image_blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        await stub._handle_message_inner(
            msg, "describe this image", "chan-1", image_blocks=image_blocks
        )

        # Should have routed to task (tools) path
        stub._process_with_tools.assert_called()
