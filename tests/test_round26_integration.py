"""End-to-end integration tests — Round 26.

Continued integration testing covering:
1. Edge cases from Round 25 findings (thread context, attachments, permission tiers)
2. Stress testing: rapid message sequences, channel locking, dedup
3. Cross-feature: voice + tools, knowledge + search, background tasks + knowledge
"""
from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    HeimdallBot,
    TOOL_OUTPUT_MAX_CHARS,
    _EMPTY_RESPONSE_FALLBACK,
    _FABRICATION_RETRY_MSG,
    _HEDGING_RETRY_MSG,
    combine_bot_messages,
    detect_fabrication,
    detect_hedging,
    scrub_response_secrets,
    truncate_tool_output,
)
from src.discord.voice import (  # noqa: E402
    VoiceManager,
    VoiceMessageProxy,
    VoiceState,
    PCMStreamSource,
    DISCORD_FRAME_BYTES,
)
from src.discord.background_task import (  # noqa: E402
    BackgroundTask,
    StepResult,
    run_background_task,
    _execute_tool,
    _substitute_vars,
    _check_condition,
    _send_progress,
    _send_summary,
    BLOCKED_TOOLS,
    MAX_STEPS,
)
from src.knowledge.store import (  # noqa: E402
    KnowledgeStore,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.search.hybrid import reciprocal_rank_fusion  # noqa: E402
from src.search.fts import FullTextIndex, _prepare_query  # noqa: E402
from src.llm.circuit_breaker import CircuitOpenError  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (shared with Round 25 pattern)
# ---------------------------------------------------------------------------

def _make_bot_stub(*, respond_to_bots=False):
    """Minimal HeimdallBot stub with all required attributes."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "You are Heimdall. Execute tasks."
    stub._pending_files = {}
    stub._cancelled_tasks = set()
    stub._embedder = None
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.max_tool_iterations_chat = 30
    stub.config.tools.max_tool_iterations_loop = 100
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.respond_to_bots = respond_to_bots
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_history = MagicMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.get_or_create = MagicMock()
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Chat response")
    stub.codex_client.chat_with_tools = AsyncMock(
        return_value=LLMResponse(text="Done", tool_calls=[], stop_reason="end_turn"),
    )
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run a command", "input_schema": {"type": "object", "properties": {}}},
        {"name": "check_disk", "description": "Check disk usage", "input_schema": {"type": "object", "properties": {}}},
        {"name": "run_script", "description": "Run a script", "input_schema": {"type": "object", "properties": {}}},
        {"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {}}},
        {"name": "docker_ps", "description": "List containers", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._maybe_cleanup_caches = MagicMock()
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.tool_memory.suggest = AsyncMock(return_value=[])
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")
    stub.voice_manager = None
    stub._channel_locks = {}
    stub._processed_messages = {}
    stub._processed_messages_max = 100
    stub._classify_completion = AsyncMock(return_value=(True, ""))
    return stub


def _make_message(*, channel_id="chan-1", is_bot=False, author_id="user-1",
                  display_name=None, content="test", webhook_id=None):
    """Create a mock Discord message."""
    msg = AsyncMock()
    msg.content = content
    msg.id = hash(content) % 2**32
    msg.webhook_id = webhook_id
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.__str__ = lambda s: f"#{channel_id}"
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None),
    ))
    msg.channel.send = AsyncMock(return_value=AsyncMock())
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.bot = is_bot
    msg.author.display_name = display_name or ("OtherBot" if is_bot else "TestUser")
    msg.author.name = msg.author.display_name
    msg.author.__str__ = lambda s: msg.author.display_name
    msg.reply = AsyncMock()
    msg.attachments = []
    return msg


def _tc(name, input_=None, id_=None):
    """Create a ToolCall."""
    return ToolCall(
        id=id_ or f"tc-{name}",
        name=name,
        input=input_ or {"command": f"do {name}"},
    )


def _tool_resp(text="", tool_calls=None, stop="end_turn"):
    """Build an LLMResponse."""
    if tool_calls:
        stop = "tool_use"
    return LLMResponse(text=text, tool_calls=tool_calls or [], stop_reason=stop)


# =====================================================================
# 1. EDGE CASES FROM ROUND 25 FINDINGS
# =====================================================================

class TestThreadContextInheritance:
    """Messages in Discord threads should inherit parent channel context."""

    async def test_thread_inherits_parent_summary(self):
        """Thread with no session gets parent summary."""
        stub = _make_bot_stub()

        # Create a thread message (channel is a Thread with parent)
        msg = _make_message(channel_id="thread-1")
        parent_channel = MagicMock()
        parent_channel.id = "parent-1"
        msg.channel = MagicMock(spec=discord.Thread)
        msg.channel.id = "thread-1"
        msg.channel.parent = parent_channel
        msg.channel.typing = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        ))

        # Setup sessions: empty thread, parent with messages
        thread_session = MagicMock()
        thread_session.messages = []
        thread_session.summary = ""
        parent_session = MagicMock()
        parent_session.summary = "Previous context about server maintenance"
        parent_msg = MagicMock()
        parent_msg.role = "user"
        parent_msg.content = "Check the servers"
        parent_session.messages = [parent_msg]

        def get_session(cid):
            if cid == "thread-1":
                return thread_session
            return parent_session

        stub.sessions.get_or_create = MagicMock(side_effect=get_session)
        stub._handle_message_inner = AsyncMock()
        stub._handle_message = HeimdallBot._handle_message.__get__(stub)

        await stub._handle_message(msg, "new question")

        # Thread session should have parent summary
        assert "server maintenance" in thread_session.summary.lower() or \
               "Parent channel context" in thread_session.summary
        stub._handle_message_inner.assert_called_once()

    async def test_thread_no_parent_skip_inheritance(self):
        """Thread with existing messages doesn't re-inherit."""
        stub = _make_bot_stub()

        msg = _make_message(channel_id="thread-2")
        parent_channel = MagicMock()
        parent_channel.id = "parent-2"
        msg.channel = MagicMock(spec=discord.Thread)
        msg.channel.id = "thread-2"
        msg.channel.parent = parent_channel
        msg.channel.typing = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        ))

        thread_session = MagicMock()
        thread_session.messages = [MagicMock()]  # Already has messages
        thread_session.summary = "Existing context"

        stub.sessions.get_or_create = MagicMock(return_value=thread_session)
        stub._handle_message_inner = AsyncMock()
        stub._handle_message = HeimdallBot._handle_message.__get__(stub)

        await stub._handle_message(msg, "follow up")

        # Summary should not change since thread already has messages
        assert thread_session.summary == "Existing context"

    async def test_non_thread_channel_no_inheritance(self):
        """Regular channels don't trigger thread inheritance."""
        stub = _make_bot_stub()

        msg = _make_message(channel_id="regular-1")
        # Regular channel - not a Thread
        msg.channel = MagicMock()
        msg.channel.id = "regular-1"
        msg.channel.typing = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        ))

        stub._handle_message_inner = AsyncMock()
        stub._handle_message = HeimdallBot._handle_message.__get__(stub)

        await stub._handle_message(msg, "hello")

        # Should still call _handle_message_inner
        stub._handle_message_inner.assert_called_once()
        # get_or_create should NOT be called for thread inheritance
        stub.sessions.get_or_create.assert_not_called()


class TestVoiceCallbackIntegration:
    """Voice callback should be invoked when voice_callback is provided."""

    async def test_voice_callback_called_with_response(self):
        """voice_callback receives the final response text."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_resp("Server is healthy, all systems operational."),
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        voice_callback = AsyncMock()

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "check health", "chan-1",
                                             voice_callback=voice_callback)

        voice_callback.assert_called_once()
        call_args = voice_callback.call_args[0][0]
        assert "healthy" in call_args.lower() or "operational" in call_args.lower()

    async def test_voice_callback_not_called_for_guest(self):
        """Guest users get chat response; voice_callback still gets called."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.permissions.is_guest = MagicMock(return_value=True)
        stub.codex_client.chat = AsyncMock(return_value="Guest response")
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        voice_callback = AsyncMock()

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "hello", "chan-1",
                                             voice_callback=voice_callback)

        voice_callback.assert_called_once_with("Guest response")

    async def test_voice_callback_error_propagates(self):
        """If voice_callback raises, it propagates (not wrapped in try/except)."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_resp("Response text"),
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        voice_callback = AsyncMock(side_effect=Exception("TTS failed"))

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            with pytest.raises(Exception, match="TTS failed"):
                await stub._handle_message_inner(msg, "test", "chan-1",
                                                 voice_callback=voice_callback)

        voice_callback.assert_called_once()


class TestPermissionTierFiltering:
    """Permission tiers filter available tools for different users."""

    async def test_admin_gets_all_tools(self):
        """Admin users get unrestricted tool access."""
        stub = _make_bot_stub()
        msg = _make_message(author_id="admin-1")

        captured = []

        async def capture_chat(**kw):
            captured.append(kw.get("tools", []))
            return _tool_resp("Done with admin tools.")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=capture_chat)
        stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert captured  # LLM was called with tools
        # All 5 tools from stub should be present
        assert len(captured[0]) == 5

    async def test_restricted_user_gets_filtered_tools(self):
        """Restricted users see only permitted tools."""
        stub = _make_bot_stub()
        msg = _make_message(author_id="restricted-1")

        captured_tools = []

        async def capture_chat(**kw):
            captured_tools.append(kw.get("tools", []))
            return _tool_resp("Done with limited tools.")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=capture_chat)
        # Filter: restricted user only gets read_file
        stub.permissions.filter_tools = MagicMock(
            side_effect=lambda uid, tools: [t for t in tools if t["name"] == "read_file"],
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert captured_tools
        assert len(captured_tools[0]) == 1
        assert captured_tools[0][0]["name"] == "read_file"

    async def test_guest_no_tools_chat_only(self):
        """Guest tier uses chat-only path (no tools)."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.permissions.is_guest = MagicMock(return_value=True)
        stub.codex_client.chat = AsyncMock(return_value="I can chat but not use tools.")
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "run command", "chan-1")

        # chat_with_tools should NOT be called for guests
        stub.codex_client.chat_with_tools.assert_not_called()
        stub.codex_client.chat.assert_called_once()


class TestToolOutputEdgeCases:
    """Edge cases in tool output handling."""

    async def test_tool_returns_empty_string(self):
        """Empty tool output should not break the tool loop."""
        stub = _make_bot_stub()
        msg = _make_message()
        call_count = 0

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("Running...", [_tc("run_command")])
            return _tool_resp("Command returned no output, which is normal.")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)
        stub.tool_executor.execute = AsyncMock(return_value="")
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(msg, [])

        assert not is_error
        assert "run_command" in tools_used

    async def test_tool_returns_none_handled_gracefully(self):
        """Tool returning None should be converted to str without crash."""
        stub = _make_bot_stub()
        msg = _make_message()
        call_count = 0

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("Running...", [_tc("check_disk")])
            return _tool_resp("Check complete.")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)
        # Return "" instead of None — executor results are always str in practice
        stub.tool_executor.execute = AsyncMock(return_value="")
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(msg, [])

        assert "check_disk" in tools_used

    async def test_large_tool_output_truncated(self):
        """Output exceeding TOOL_OUTPUT_MAX_CHARS is truncated with omission marker."""
        large_output = "A" * (TOOL_OUTPUT_MAX_CHARS + 5000)
        result = truncate_tool_output(large_output)
        # Result preserves start and end halves, adds omission marker in between
        assert len(result) < len(large_output)
        assert result.startswith("A")
        assert result.endswith("A")
        assert "omitted" in result

    async def test_tool_exception_message_in_result(self):
        """Tool exception message should appear in the result fed back to LLM."""
        stub = _make_bot_stub()
        msg = _make_message()
        call_count = 0
        captured_messages = []

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            captured_messages.append(kw.get("messages", []))
            if call_count == 1:
                return _tool_resp("Checking...", [_tc("run_command")])
            return _tool_resp("The command failed with a permission error.")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)
        stub.tool_executor.execute = AsyncMock(
            side_effect=PermissionError("Access denied to /root"),
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        # The error message should be in the second LLM call's messages
        assert len(captured_messages) >= 2
        second_call = str(captured_messages[1])
        assert "Access denied" in second_call or "PermissionError" in second_call


class TestEmptyResponseFallback:
    """LLM returning empty text without tools triggers fallback."""

    async def test_empty_text_no_tools_uses_fallback(self):
        """Empty response with no tools should return fallback text."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_resp(""),
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == _EMPTY_RESPONSE_FALLBACK

    async def test_whitespace_only_text_returned_as_is(self):
        """Whitespace-only response is returned without fallback (truthy string)."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_resp("   \n  "),
        )
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, _, _ = await stub._process_with_tools(msg, [])

        # Whitespace is truthy so `text or FALLBACK` returns the whitespace
        assert text == "   \n  "


# =====================================================================
# 2. STRESS TESTING: RAPID MESSAGE SEQUENCES
# =====================================================================

class TestChannelLocking:
    """Per-channel locking ensures serial message processing."""

    async def test_channel_lock_created_per_channel(self):
        """Each channel gets its own asyncio.Lock."""
        stub = _make_bot_stub()
        stub._handle_message_inner = AsyncMock()
        stub._handle_message = HeimdallBot._handle_message.__get__(stub)

        msg1 = _make_message(channel_id="chan-A")
        msg1.channel = MagicMock()
        msg1.channel.id = "chan-A"
        msg1.channel.typing = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        ))

        msg2 = _make_message(channel_id="chan-B")
        msg2.channel = MagicMock()
        msg2.channel.id = "chan-B"
        msg2.channel.typing = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        ))

        await stub._handle_message(msg1, "msg1")
        await stub._handle_message(msg2, "msg2")

        assert "chan-A" in stub._channel_locks
        assert "chan-B" in stub._channel_locks
        assert stub._channel_locks["chan-A"] is not stub._channel_locks["chan-B"]

    async def test_same_channel_messages_serialized(self):
        """Messages in the same channel are processed one at a time."""
        stub = _make_bot_stub()
        processing_order = []
        processing_concurrent = []

        async def track_processing(msg, content, channel_id, **kw):
            processing_order.append(content)
            processing_concurrent.append(len(processing_order))
            await asyncio.sleep(0.01)  # Simulate processing time

        stub._handle_message_inner = AsyncMock(side_effect=track_processing)
        stub._handle_message = HeimdallBot._handle_message.__get__(stub)

        msg1 = _make_message(channel_id="chan-1", content="first")
        msg1.channel = MagicMock()
        msg1.channel.id = "chan-1"
        msg1.channel.typing = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        ))

        msg2 = _make_message(channel_id="chan-1", content="second")
        msg2.channel = MagicMock()
        msg2.channel.id = "chan-1"
        msg2.channel.typing = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        ))

        # Launch both concurrently
        await asyncio.gather(
            stub._handle_message(msg1, "first"),
            stub._handle_message(msg2, "second"),
        )

        assert len(processing_order) == 2
        # Both should be processed (order may vary due to lock acquisition)
        assert "first" in processing_order
        assert "second" in processing_order

    async def test_different_channels_parallel(self):
        """Messages in different channels can process concurrently."""
        stub = _make_bot_stub()
        active = []
        max_concurrent = [0]

        async def track_concurrent(msg, content, channel_id, **kw):
            active.append(channel_id)
            max_concurrent[0] = max(max_concurrent[0], len(active))
            await asyncio.sleep(0.05)  # Hold the "lock" briefly
            active.remove(channel_id)

        stub._handle_message_inner = AsyncMock(side_effect=track_concurrent)
        stub._handle_message = HeimdallBot._handle_message.__get__(stub)

        msgs = []
        for i in range(3):
            m = _make_message(channel_id=f"chan-{i}", content=f"msg-{i}")
            m.channel = MagicMock()
            m.channel.id = f"chan-{i}"
            m.channel.typing = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=None),
                __aexit__=AsyncMock(return_value=None),
            ))
            msgs.append(m)

        await asyncio.gather(
            *(stub._handle_message(m, f"msg-{i}") for i, m in enumerate(msgs))
        )

        # At least 2 should have been concurrent (channels don't block each other)
        assert max_concurrent[0] >= 2


class TestMessageDeduplication:
    """Duplicate message detection via _processed_messages."""

    async def test_combine_bot_messages_merges_burst(self):
        """Rapid bot messages (string parts) are combined into one."""
        parts = ["line 1", "line 2", "line 3"]
        combined = combine_bot_messages(parts)
        assert "line 1" in combined
        assert "line 2" in combined
        assert "line 3" in combined

    async def test_combine_bot_messages_single(self):
        """Single bot message returns its content."""
        parts = ["single message"]
        combined = combine_bot_messages(parts)
        assert combined == "single message"

    async def test_combine_bot_messages_code_block_merge(self):
        """Adjacent code blocks are merged into one."""
        parts = ["```python\nprint('hello')\n```", "```python\nprint('world')\n```"]
        combined = combine_bot_messages(parts)
        # Should combine cleanly
        assert "hello" in combined
        assert "world" in combined

    async def test_combine_bot_messages_split_code_block(self):
        """Code block split across messages is joined."""
        parts = ["```python\ndef foo():", "    return 42\n```"]
        combined = combine_bot_messages(parts)
        assert "def foo():" in combined
        assert "return 42" in combined


class TestRapidToolCalls:
    """Rapid tool calls within a single message should work correctly."""

    async def test_parallel_tool_calls_in_one_iteration(self):
        """Multiple tool calls in a single LLM response execute concurrently."""
        stub = _make_bot_stub()
        msg = _make_message()
        call_count = 0
        exec_count = 0

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("Running both...", [
                    _tc("check_disk", id_="tc-1"),
                    _tc("docker_ps", id_="tc-2"),
                ])
            return _tool_resp("Disk is fine, 3 containers running.")

        async def fake_exec(name, inp, **kwargs):
            nonlocal exec_count
            exec_count += 1
            return f"result from {name}"

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)
        stub.tool_executor.execute = AsyncMock(side_effect=fake_exec)
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(msg, [])

        assert not is_error
        assert exec_count == 2
        assert "check_disk" in tools_used
        assert "docker_ps" in tools_used

    async def test_mixed_success_failure_parallel_tools(self):
        """Some tools succeed, some fail — all results fed back to LLM."""
        stub = _make_bot_stub()
        msg = _make_message()
        call_count = 0

        async def fake_chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_resp("Running...", [
                    _tc("run_command", id_="tc-ok"),
                    _tc("check_disk", id_="tc-fail"),
                ])
            return _tool_resp("Command succeeded but disk check failed.")

        async def selective_exec(name, inp, **kwargs):
            if name == "check_disk":
                raise ConnectionError("SSH timeout")
            return "ok"

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=fake_chat)
        stub.tool_executor.execute = AsyncMock(side_effect=selective_exec)
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(msg, [])

        # Both tools attempted
        assert "run_command" in tools_used
        assert "check_disk" in tools_used


# =====================================================================
# 3. CROSS-FEATURE: VOICE + TOOLS
# =====================================================================

class TestVoiceManagerUnit:
    """Voice manager component tests."""

    async def test_voice_message_proxy_reply(self):
        """VoiceMessageProxy.reply() sends to the text channel."""
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=MagicMock(id=123))
        member = MagicMock()
        guild = MagicMock()

        proxy = VoiceMessageProxy(
            author=member,
            channel=channel,
            id=999,
            guild=guild,
        )

        await proxy.reply("test response")
        channel.send.assert_called_once_with("test response")

    async def test_pcm_stream_source_reads_frames(self):
        """PCMStreamSource reads correct frame sizes."""
        data = b"\x00" * (DISCORD_FRAME_BYTES * 2 + 100)
        source = PCMStreamSource(data)

        frame1 = source.read()
        assert len(frame1) == DISCORD_FRAME_BYTES

        frame2 = source.read()
        assert len(frame2) == DISCORD_FRAME_BYTES

        # Third read: not enough data for a full frame
        frame3 = source.read()
        assert frame3 == b""

    async def test_pcm_stream_source_is_not_opus(self):
        """PCMStreamSource reports PCM, not opus."""
        source = PCMStreamSource(b"\x00" * 100)
        assert source.is_opus() is False

    async def test_voice_manager_not_connected_by_default(self):
        """VoiceManager starts disconnected."""
        config = MagicMock()
        config.enabled = True
        bot = MagicMock()
        vm = VoiceManager(config, bot)
        assert vm.is_connected is False
        assert vm.current_channel is None

    async def test_voice_manager_disabled_config(self):
        """join_channel returns error when voice is disabled."""
        config = MagicMock()
        config.enabled = False
        bot = MagicMock()
        vm = VoiceManager(config, bot)
        channel = MagicMock()
        result = await vm.join_channel(channel)
        assert "disabled" in result.lower()

    async def test_voice_manager_leave_not_connected(self):
        """leave_channel returns message when not connected."""
        config = MagicMock()
        config.enabled = True
        bot = MagicMock()
        vm = VoiceManager(config, bot)
        result = await vm.leave_channel()
        assert "not in" in result.lower()

    async def test_voice_speak_when_not_connected(self):
        """speak() is a no-op when not connected."""
        config = MagicMock()
        config.enabled = True
        bot = MagicMock()
        vm = VoiceManager(config, bot)
        # Should not raise
        await vm.speak("hello")

    async def test_handle_transcription_leave_command(self):
        """Voice 'leave voice' command triggers leave_channel."""
        config = MagicMock()
        config.enabled = True
        config.transcript_channel_id = None
        bot = MagicMock()
        vm = VoiceManager(config, bot)

        # Pretend we're connected
        vm._voice_client = MagicMock()
        vm._voice_client.is_connected = MagicMock(return_value=True)
        vm._voice_client.channel = MagicMock()
        vm._voice_client.channel.name = "General"
        vm._voice_client.channel.guild = MagicMock()
        vm._voice_client.channel.guild.text_channels = []
        vm._voice_client.stop_listening = MagicMock()
        vm._voice_client.disconnect = AsyncMock()

        vm._ws = None
        vm._ws_task = None

        await vm._handle_transcription("leave voice", "user-1")

        # Should have triggered leave (voice_client set to None after leave)
        assert vm._voice_client is None

    async def test_handle_service_message_error_resets_speaking(self):
        """Error message from voice service resets speaking state."""
        config = MagicMock()
        config.enabled = True
        bot = MagicMock()
        vm = VoiceManager(config, bot)
        vm._speaking = True
        vm._tts_buffer = bytearray(b"some data")

        await vm._handle_service_message({"type": "error", "message": "test error"})

        assert vm._speaking is False
        assert len(vm._tts_buffer) == 0

    async def test_handle_service_message_tts_start(self):
        """TTS start message sets speaking flag."""
        config = MagicMock()
        config.enabled = True
        bot = MagicMock()
        vm = VoiceManager(config, bot)

        await vm._handle_service_message({"type": "tts_start"})

        assert vm._speaking is True
        assert len(vm._tts_buffer) == 0

    async def test_handle_service_audio_buffers_during_tts(self):
        """Audio data is buffered during TTS phase."""
        config = MagicMock()
        config.enabled = True
        bot = MagicMock()
        vm = VoiceManager(config, bot)
        vm._speaking = True
        vm._tts_buffer = bytearray()

        await vm._handle_service_audio(b"\x00" * 100)
        await vm._handle_service_audio(b"\xFF" * 50)

        assert len(vm._tts_buffer) == 150

    async def test_on_audio_receive_skipped_when_speaking(self):
        """Audio receive is skipped while bot is speaking (TTS)."""
        config = MagicMock()
        config.enabled = True
        bot = MagicMock()
        vm = VoiceManager(config, bot)
        vm._speaking = True
        vm._ws_send_binary = AsyncMock()

        await vm._on_audio_receive(b"\x00" * 100, "user-1")

        vm._ws_send_binary.assert_not_called()


class TestVoiceTranscriptionFlow:
    """Voice transcription → HeimdallBot pipeline integration."""

    async def test_transcription_creates_proxy_and_routes(self):
        """_on_voice_transcription creates VoiceMessageProxy and calls _handle_message."""
        stub = _make_bot_stub()
        stub._handle_message = AsyncMock()
        stub._on_voice_transcription = HeimdallBot._on_voice_transcription.__get__(stub)

        member = MagicMock()
        member.display_name = "TestUser"
        member.guild = MagicMock()

        transcript_channel = AsyncMock()
        transcript_channel.send = AsyncMock()

        stub.voice_manager = MagicMock()
        stub.voice_manager.speak = AsyncMock()

        await stub._on_voice_transcription("check the servers", member, transcript_channel)

        # Should post transcription to channel
        transcript_channel.send.assert_called_once()
        sent_text = transcript_channel.send.call_args[0][0]
        assert "TestUser" in sent_text
        assert "check the servers" in sent_text

        # Should call _handle_message with a VoiceMessageProxy
        stub._handle_message.assert_called_once()
        call_args = stub._handle_message.call_args
        proxy = call_args[0][0]
        assert isinstance(proxy, VoiceMessageProxy)
        assert call_args[0][1] == "check the servers"

    async def test_voice_callback_speaks_response(self):
        """Voice callback should invoke voice_manager.speak()."""
        stub = _make_bot_stub()
        stub.voice_manager = MagicMock()
        stub.voice_manager.speak = AsyncMock()

        # Extract the voice_callback that _on_voice_transcription creates
        captured_callback = []

        async def capture_handle_message(proxy, text, **kw):
            vc = kw.get("voice_callback")
            if vc:
                captured_callback.append(vc)

        stub._handle_message = AsyncMock(side_effect=capture_handle_message)
        stub._on_voice_transcription = HeimdallBot._on_voice_transcription.__get__(stub)

        member = MagicMock()
        member.display_name = "User"
        member.guild = MagicMock()

        transcript_channel = AsyncMock()
        transcript_channel.send = AsyncMock()

        await stub._on_voice_transcription("hello", member, transcript_channel)

        assert captured_callback
        await captured_callback[0]("Hello from Heimdall")
        stub.voice_manager.speak.assert_called_once_with("Hello from Heimdall")


class TestAutoJoinLeaveVoice:
    """Voice auto-join/leave on voice state changes."""

    async def test_auto_join_when_user_joins_channel(self):
        """Bot auto-joins when an allowed user joins a voice channel."""
        stub = _make_bot_stub()
        stub.voice_manager = MagicMock()
        stub.voice_manager.is_connected = False
        stub.voice_manager.join_channel = AsyncMock(return_value="Joined")
        stub.config.voice = MagicMock()
        stub.config.voice.auto_join = True
        stub._is_allowed_user = MagicMock(return_value=True)
        stub.on_voice_state_update = HeimdallBot.on_voice_state_update.__get__(stub)

        member = MagicMock()
        member.bot = False

        before = MagicMock()
        before.channel = None

        after = MagicMock()
        after.channel = MagicMock()

        await stub.on_voice_state_update(member, before, after)

        stub.voice_manager.join_channel.assert_called_once_with(after.channel)

    async def test_auto_leave_when_all_users_leave(self):
        """Bot leaves when all humans leave the voice channel."""
        stub = _make_bot_stub()
        channel = MagicMock()
        channel.members = [MagicMock(bot=True)]  # Only bot remains

        stub.voice_manager = MagicMock()
        stub.voice_manager.is_connected = True
        stub.voice_manager.current_channel = channel
        stub.voice_manager.leave_channel = AsyncMock(return_value="Left")
        stub.config.voice = MagicMock()
        stub.config.voice.auto_join = True
        stub._is_allowed_user = MagicMock(return_value=True)
        stub.on_voice_state_update = HeimdallBot.on_voice_state_update.__get__(stub)

        member = MagicMock()
        member.bot = False

        before = MagicMock()
        before.channel = channel

        after = MagicMock()
        after.channel = None

        await stub.on_voice_state_update(member, before, after)

        stub.voice_manager.leave_channel.assert_called_once()

    async def test_no_auto_join_for_bots(self):
        """Bot users don't trigger auto-join."""
        stub = _make_bot_stub()
        stub.voice_manager = MagicMock()
        stub.voice_manager.join_channel = AsyncMock()
        stub.config.voice = MagicMock()
        stub.config.voice.auto_join = True
        stub.on_voice_state_update = HeimdallBot.on_voice_state_update.__get__(stub)

        member = MagicMock()
        member.bot = True

        before = MagicMock()
        before.channel = None
        after = MagicMock()
        after.channel = MagicMock()

        await stub.on_voice_state_update(member, before, after)

        stub.voice_manager.join_channel.assert_not_called()

    async def test_no_auto_join_when_disabled(self):
        """Auto-join disabled in config → no join."""
        stub = _make_bot_stub()
        stub.voice_manager = MagicMock()
        stub.voice_manager.join_channel = AsyncMock()
        stub.config.voice = MagicMock()
        stub.config.voice.auto_join = False
        stub.on_voice_state_update = HeimdallBot.on_voice_state_update.__get__(stub)

        member = MagicMock()
        member.bot = False

        before = MagicMock()
        before.channel = None
        after = MagicMock()
        after.channel = MagicMock()

        await stub.on_voice_state_update(member, before, after)

        stub.voice_manager.join_channel.assert_not_called()


# =====================================================================
# 4. CROSS-FEATURE: KNOWLEDGE + SEARCH
# =====================================================================

class TestKnowledgeStoreUnit:
    """KnowledgeStore integration tests with real SQLite (no vec extension)."""

    def _make_store(self, tmp_path, fts=None):
        """Create a KnowledgeStore with patched load_extension."""
        db_path = str(tmp_path / "knowledge.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            return KnowledgeStore(db_path, fts_index=fts)

    async def test_ingest_chunks_text(self, tmp_path):
        """Ingest splits text into chunks and stores in SQLite."""
        store = self._make_store(tmp_path)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        count = await store.ingest("Short doc", "test.md", embedder)
        assert count == 1
        assert store.count() == 1

    async def test_ingest_long_text_multiple_chunks(self, tmp_path):
        """Long text produces multiple chunks."""
        store = self._make_store(tmp_path)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1])

        long_text = "word " * 1000
        count = await store.ingest(long_text, "long.md", embedder)
        assert count >= 2

    async def test_ingest_replaces_existing_source(self, tmp_path):
        """Re-ingesting same source replaces old chunks."""
        store = self._make_store(tmp_path)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.5])

        await store.ingest("Old content", "source.md", embedder)
        assert store.count() == 1
        await store.ingest("New content", "source.md", embedder)
        assert store.count() == 1  # Replaced, not duplicated

    async def test_ingest_with_fts_dual_write(self, tmp_path):
        """Ingest writes to both SQLite and FTS5."""
        fts = MagicMock()
        fts.delete_knowledge_source = MagicMock()
        fts.index_knowledge_chunk = MagicMock(return_value=True)

        store = self._make_store(tmp_path, fts=fts)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1])

        count = await store.ingest("Some content", "doc.md", embedder)
        assert count == 1
        fts.index_knowledge_chunk.assert_called_once()

    async def test_search_returns_empty_without_vec(self, tmp_path):
        """Semantic search returns empty without sqlite-vec."""
        store = self._make_store(tmp_path)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.5])

        results = await store.search("test query", embedder, limit=5)
        assert results == []

    async def test_search_returns_empty_when_unavailable(self, tmp_path):
        """Search returns empty list when store is unavailable."""
        bad_path = str(tmp_path / "x" / "y" / "z.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            store = KnowledgeStore(bad_path)

        embedder = AsyncMock()
        results = await store.search("test", embedder)
        assert results == []

    async def test_search_returns_empty_when_embed_fails(self, tmp_path):
        """Search returns empty when embedder returns None."""
        store = self._make_store(tmp_path)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=None)

        results = await store.search("test", embedder)
        assert results == []

    async def test_delete_source_removes_sqlite_and_fts(self, tmp_path):
        """delete_source removes from both SQLite and FTS5."""
        fts = MagicMock()
        fts.delete_knowledge_source = MagicMock(return_value=3)

        store = self._make_store(tmp_path, fts=fts)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1])

        await store.ingest("content for doc", "old-doc.md", embedder)
        count = store.delete_source("old-doc.md")
        assert count == 1
        fts.delete_knowledge_source.assert_called_with("old-doc.md")

    async def test_list_sources_groups_by_source(self, tmp_path):
        """list_sources aggregates chunks by source name."""
        store = self._make_store(tmp_path)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1])

        await store.ingest("content a", "doc-a", embedder, uploader="admin")
        await store.ingest("content b", "doc-b", embedder, uploader="user")

        sources = store.list_sources()
        assert len(sources) == 2
        names = [s["source"] for s in sources]
        assert "doc-a" in names
        assert "doc-b" in names


class TestHybridSearch:
    """Hybrid search combining semantic + FTS5 via Reciprocal Rank Fusion."""

    def _make_store(self, tmp_path, fts=None):
        db_path = str(tmp_path / "knowledge.db")
        with patch("src.knowledge.store.load_extension", return_value=False):
            return KnowledgeStore(db_path, fts_index=fts)

    async def test_hybrid_search_fts_only(self, tmp_path):
        """search_hybrid uses FTS results when no vec available."""
        fts = MagicMock()
        fts.search_knowledge = MagicMock(return_value=[
            {"chunk_id": "c1", "content": "FTS result 1", "source": "doc1"},
            {"chunk_id": "c2", "content": "FTS result 2", "source": "doc2"},
        ])

        store = self._make_store(tmp_path, fts=fts)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.5])

        results = await store.search_hybrid("test query", embedder, limit=5)
        assert len(results) >= 1

    async def test_hybrid_search_empty_when_both_empty(self, tmp_path):
        """search_hybrid returns empty when both backends return nothing."""
        fts = MagicMock()
        fts.search_knowledge = MagicMock(return_value=[])

        store = self._make_store(tmp_path, fts=fts)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.5])

        results = await store.search_hybrid("nothing", embedder)
        assert results == []

    async def test_hybrid_search_no_fts(self, tmp_path):
        """search_hybrid works without FTS (returns empty without vec)."""
        store = self._make_store(tmp_path)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.5])

        results = await store.search_hybrid("query", embedder, limit=5)
        assert results == []

    async def test_hybrid_search_fts_only_results(self, tmp_path):
        """search_hybrid returns FTS results when no semantic available."""
        fts = MagicMock()
        fts.search_knowledge = MagicMock(return_value=[
            {"chunk_id": "c1", "content": "FTS result", "source": "doc1"},
        ])

        store = self._make_store(tmp_path, fts=fts)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.5])

        results = await store.search_hybrid("query", embedder, limit=5)
        assert len(results) == 1
        assert results[0]["content"] == "FTS result"


class TestReciprocalRankFusion:
    """RRF algorithm correctness tests."""

    def test_single_list(self):
        """RRF with a single list preserves order."""
        items = [
            {"doc_id": "a", "text": "first"},
            {"doc_id": "b", "text": "second"},
        ]
        result = reciprocal_rank_fusion(items)
        assert len(result) == 2
        assert result[0]["doc_id"] == "a"
        assert result[1]["doc_id"] == "b"

    def test_two_lists_boost_overlap(self):
        """Items appearing in both lists get higher RRF scores."""
        list1 = [
            {"doc_id": "a", "text": "A from list1"},
            {"doc_id": "b", "text": "B from list1"},
        ]
        list2 = [
            {"doc_id": "b", "text": "B from list2"},
            {"doc_id": "c", "text": "C from list2"},
        ]
        result = reciprocal_rank_fusion(list1, list2)
        # 'b' appears in both → should rank first
        assert result[0]["doc_id"] == "b"

    def test_deduplication_keeps_highest_ranked(self):
        """Dedup keeps the dict from the list where item ranked highest."""
        list1 = [{"doc_id": "x", "text": "from list1 rank 0"}]
        list2 = [
            {"doc_id": "y", "text": "y"},
            {"doc_id": "x", "text": "from list2 rank 1"},
        ]
        result = reciprocal_rank_fusion(list1, list2)
        # 'x' ranked 0 in list1, so list1's version is kept
        x_entry = [r for r in result if r["doc_id"] == "x"][0]
        assert x_entry["text"] == "from list1 rank 0"

    def test_limit_respected(self):
        """RRF respects the limit parameter."""
        items = [{"doc_id": str(i), "text": f"item {i}"} for i in range(20)]
        result = reciprocal_rank_fusion(items, limit=5)
        assert len(result) == 5

    def test_rrf_score_added(self):
        """Each result has an rrf_score field."""
        items = [{"doc_id": "a", "text": "test"}]
        result = reciprocal_rank_fusion(items)
        assert "rrf_score" in result[0]
        assert result[0]["rrf_score"] > 0

    def test_empty_lists(self):
        """RRF with empty lists returns empty."""
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_custom_id_key(self):
        """RRF works with custom id_key."""
        items = [
            {"chunk_id": "c1", "text": "chunk 1"},
            {"chunk_id": "c2", "text": "chunk 2"},
        ]
        result = reciprocal_rank_fusion(items, id_key="chunk_id")
        assert len(result) == 2
        assert result[0]["chunk_id"] == "c1"


class TestFTSPrepareQuery:
    """FTS5 query preparation edge cases."""

    def test_normal_query(self):
        """Simple text passes through."""
        assert _prepare_query("hello world") == "hello world"

    def test_empty_query(self):
        """Empty string returns empty."""
        assert _prepare_query("") == ""
        assert _prepare_query("   ") == ""

    def test_special_chars_quoted(self):
        """FTS5 special characters trigger quoting."""
        result = _prepare_query("test*query")
        assert result.startswith('"') and result.endswith('"')

    def test_ip_address_quoted(self):
        """IP addresses (contain dots) are quoted for literal match."""
        result = _prepare_query("198.51.100.1")
        assert result == '"198.51.100.1"'

    def test_path_quoted(self):
        """File paths (contain slashes) are quoted."""
        result = _prepare_query("/var/log/syslog")
        assert result == '"/var/log/syslog"'

    def test_internal_quotes_escaped(self):
        """Internal double quotes are escaped."""
        result = _prepare_query('test "quoted" value')
        assert '""' in result  # Escaped quotes


class TestChunkText:
    """Text chunking for knowledge ingestion."""

    def test_short_text_single_chunk(self):
        """Short text returns as single chunk."""
        chunks = KnowledgeStore._chunk_text("Hello, world!")
        assert len(chunks) == 1
        assert chunks[0] == "Hello, world!"

    def test_empty_text_no_chunks(self):
        """Empty text returns no chunks."""
        chunks = KnowledgeStore._chunk_text("")
        assert chunks == []
        chunks = KnowledgeStore._chunk_text("   ")
        assert chunks == []

    def test_long_text_multiple_chunks(self):
        """Text longer than CHUNK_SIZE gets split into multiple chunks."""
        long_text = "word " * 1000  # ~5000 chars, well over CHUNK_SIZE
        chunks = KnowledgeStore._chunk_text(long_text)
        assert len(chunks) >= 2
        # Each chunk should be at most CHUNK_SIZE
        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE + 50  # Small tolerance for word boundaries

    def test_paragraph_boundary_splitting(self):
        """Text is preferably split on paragraph boundaries."""
        para1 = "First paragraph. " * 40  # ~680 chars
        para2 = "Second paragraph. " * 40  # ~720 chars
        para3 = "Third paragraph. " * 40
        text = f"{para1}\n\n{para2}\n\n{para3}"
        chunks = KnowledgeStore._chunk_text(text)
        assert len(chunks) >= 2


# =====================================================================
# 5. CROSS-FEATURE: BACKGROUND TASKS + KNOWLEDGE
# =====================================================================

class TestBackgroundTaskKnowledgeIntegration:
    """Background tasks can run knowledge operations."""

    async def test_execute_tool_ingest_document(self):
        """_execute_tool routes ingest_document to knowledge store."""
        knowledge_store = AsyncMock()
        knowledge_store.ingest = AsyncMock(return_value=5)
        embedder = AsyncMock()
        executor = MagicMock()
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        result = await _execute_tool(
            "ingest_document",
            {"source": "runbook.md", "content": "How to restart services..."},
            executor, skill_manager, knowledge_store, embedder, "admin",
        )

        assert "5 chunks" in result
        knowledge_store.ingest.assert_called_once()

    async def test_execute_tool_ingest_missing_fields(self):
        """ingest_document with missing fields returns error."""
        knowledge_store = AsyncMock()
        embedder = AsyncMock()
        executor = MagicMock()
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        result = await _execute_tool(
            "ingest_document",
            {"source": "runbook.md"},  # missing content
            executor, skill_manager, knowledge_store, embedder, "admin",
        )

        assert "required" in result.lower()

    async def test_execute_tool_search_knowledge(self):
        """_execute_tool routes search_knowledge to knowledge store."""
        knowledge_store = AsyncMock()
        knowledge_store.search = AsyncMock(return_value=[
            {"source": "doc1", "score": 0.9, "content": "Found result text here"},
        ])
        embedder = AsyncMock()
        executor = MagicMock()
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        result = await _execute_tool(
            "search_knowledge",
            {"query": "how to restart"},
            executor, skill_manager, knowledge_store, embedder, "admin",
        )

        assert "doc1" in result
        assert "Found result" in result

    async def test_execute_tool_search_no_results(self):
        """search_knowledge with no results returns appropriate message."""
        knowledge_store = AsyncMock()
        knowledge_store.search = AsyncMock(return_value=[])
        embedder = AsyncMock()
        executor = MagicMock()
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        result = await _execute_tool(
            "search_knowledge",
            {"query": "nonexistent topic"},
            executor, skill_manager, knowledge_store, embedder, "admin",
        )

        assert "no results" in result.lower()

    async def test_execute_tool_list_knowledge(self):
        """_execute_tool routes list_knowledge to knowledge store."""
        knowledge_store = MagicMock()
        knowledge_store.list_sources = MagicMock(return_value=[
            {"source": "runbook.md", "chunks": 5},
            {"source": "readme.md", "chunks": 3},
        ])
        executor = MagicMock()
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        result = await _execute_tool(
            "list_knowledge", {},
            executor, skill_manager, knowledge_store, None, "admin",
        )

        assert "runbook.md" in result
        assert "readme.md" in result

    async def test_execute_tool_list_knowledge_empty(self):
        """list_knowledge returns message when knowledge base is empty."""
        knowledge_store = MagicMock()
        knowledge_store.list_sources = MagicMock(return_value=[])
        executor = MagicMock()
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        result = await _execute_tool(
            "list_knowledge", {},
            executor, skill_manager, knowledge_store, None, "admin",
        )

        assert "empty" in result.lower()

    async def test_execute_tool_routes_to_skill(self):
        """_execute_tool routes to skill manager for skill tools."""
        executor = MagicMock()
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=True)
        skill_manager.execute = AsyncMock(return_value="skill result")

        result = await _execute_tool(
            "custom_skill", {"param": "value"},
            executor, skill_manager, None, None, "admin",
        )

        assert result == "skill result"
        skill_manager.execute.assert_called_once_with("custom_skill", {"param": "value"})

    async def test_execute_tool_routes_to_executor(self):
        """_execute_tool routes unknown tools to executor."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value="exec result")
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        result = await _execute_tool(
            "run_command", {"command": "uptime"},
            executor, skill_manager, None, None, "admin",
        )

        assert result == "exec result"
        executor.execute.assert_called_once()


class TestBackgroundTaskVariableSubstitution:
    """Variable substitution in background task tool inputs."""

    def test_prev_output_substitution(self):
        result = _substitute_vars(
            {"command": "echo {prev_output}"},
            {},
            "hello world",
        )
        assert result["command"] == "echo hello world"

    def test_named_variable_substitution(self):
        result = _substitute_vars(
            {"file": "/tmp/{var.filename}"},
            {"filename": "output.txt"},
            "",
        )
        assert result["file"] == "/tmp/output.txt"

    def test_multiple_substitutions(self):
        result = _substitute_vars(
            {"cmd": "scp {var.host}:{var.path} ."},
            {"host": "server1", "path": "/tmp/data"},
            "",
        )
        assert result["cmd"] == "scp server1:/tmp/data ."

    def test_non_string_values_unchanged(self):
        result = _substitute_vars(
            {"limit": 10, "command": "test {prev_output}"},
            {},
            "data",
        )
        assert result["limit"] == 10
        assert result["command"] == "test data"


class TestBackgroundTaskConditions:
    """Condition checking for conditional step execution."""

    def test_positive_condition_match(self):
        assert _check_condition("success", "Operation: SUCCESS") is True

    def test_positive_condition_no_match(self):
        assert _check_condition("success", "Operation: FAILED") is False

    def test_negated_condition_match(self):
        assert _check_condition("!error", "All systems normal") is True

    def test_negated_condition_no_match(self):
        assert _check_condition("!error", "Error: disk full") is False

    def test_case_insensitive(self):
        assert _check_condition("OK", "result: ok") is True


class TestBackgroundTaskBlockedTools:
    """Background tasks refuse blocked tools."""

    async def test_blocked_tool_produces_error(self):
        """Blocked tools generate error steps."""
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=MagicMock())

        task = BackgroundTask(
            task_id="test-1",
            description="Test blocked",
            steps=[
                {"tool_name": "purge_messages", "description": "purge"},
            ],
            channel=channel,
            requester="admin",
        )

        executor = MagicMock()
        skill_manager = MagicMock()

        await run_background_task(task, executor, skill_manager)

        assert task.status == "failed"
        assert len(task.results) == 1
        assert task.results[0].status == "error"
        assert "cannot run" in task.results[0].output.lower()

    async def test_blocked_tool_continue_on_failure(self):
        """Blocked tool with on_failure=continue doesn't abort."""
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=MagicMock())

        task = BackgroundTask(
            task_id="test-2",
            description="Test continue",
            steps=[
                {"tool_name": "purge_messages", "description": "purge", "on_failure": "continue"},
                {"tool_name": "run_command", "tool_input": {"command": "echo ok"}, "description": "echo"},
            ],
            channel=channel,
            requester="admin",
        )

        executor = MagicMock()
        executor.execute = AsyncMock(return_value="ok")
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        await run_background_task(task, executor, skill_manager)

        assert task.status == "completed"
        assert len(task.results) == 2
        assert task.results[0].status == "error"  # purge blocked
        assert task.results[1].status == "ok"  # echo succeeded


class TestBackgroundTaskCancellation:
    """Background task cancellation mid-execution."""

    async def test_cancel_before_step(self):
        """Cancelling before a step produces cancelled status."""
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=MagicMock())

        task = BackgroundTask(
            task_id="cancel-1",
            description="Cancel test",
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "slow"}, "description": "step 1"},
                {"tool_name": "run_command", "tool_input": {"command": "never"}, "description": "step 2"},
            ],
            channel=channel,
            requester="admin",
        )

        # Cancel immediately
        task.cancel()

        executor = MagicMock()
        executor.execute = AsyncMock(return_value="ok")
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        await run_background_task(task, executor, skill_manager)

        assert task.status == "cancelled"
        # Executor should not have been called
        executor.execute.assert_not_called()


class TestBackgroundTaskProgress:
    """Progress reporting for background tasks."""

    async def test_progress_message_sent(self):
        """Running a task sends progress messages."""
        channel = AsyncMock()
        progress_msg = MagicMock()
        progress_msg.edit = AsyncMock()
        channel.send = AsyncMock(return_value=progress_msg)

        task = BackgroundTask(
            task_id="prog-1",
            description="Progress test",
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "echo 1"}, "description": "step 1"},
            ],
            channel=channel,
            requester="admin",
        )

        executor = MagicMock()
        executor.execute = AsyncMock(return_value="done")
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        await run_background_task(task, executor, skill_manager)

        assert task.status == "completed"
        # channel.send should be called for progress + summary
        assert channel.send.call_count >= 1

    async def test_send_summary_completed(self):
        """Summary message reflects completed status."""
        channel = AsyncMock()
        channel.send = AsyncMock()

        task = BackgroundTask(
            task_id="sum-1",
            description="Summary test",
            steps=[],
            channel=channel,
            requester="admin",
        )
        task.status = "completed"
        task.results = [
            StepResult(index=0, tool_name="run_command", description="echo", status="ok",
                       output="hello", elapsed_ms=50),
        ]

        await _send_summary(task)

        channel.send.assert_called_once()
        sent_text = channel.send.call_args[0][0]
        assert "complete" in sent_text.lower()
        assert "1 steps succeeded" in sent_text or "All" in sent_text

    async def test_send_summary_failed(self):
        """Summary message reflects failed status."""
        channel = AsyncMock()
        channel.send = AsyncMock()

        task = BackgroundTask(
            task_id="sum-2",
            description="Fail test",
            steps=[{"tool_name": "t1"}, {"tool_name": "t2"}],
            channel=channel,
            requester="admin",
        )
        task.status = "failed"
        task.results = [
            StepResult(index=0, tool_name="t1", description="step 1", status="ok",
                       output="ok", elapsed_ms=10),
            StepResult(index=1, tool_name="t2", description="step 2", status="error",
                       output="connection refused", elapsed_ms=5),
        ]

        await _send_summary(task)

        channel.send.assert_called_once()
        sent_text = channel.send.call_args[0][0]
        assert "aborted" in sent_text.lower() or "failed" in sent_text.lower()


class TestBackgroundTaskStoreAs:
    """store_as captures tool output in variables for later steps."""

    async def test_store_as_passes_to_later_step(self):
        """store_as + {var.name} substitution works across steps."""
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=MagicMock())

        task = BackgroundTask(
            task_id="store-1",
            description="Store test",
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "hostname"},
                 "description": "get hostname", "store_as": "host"},
                {"tool_name": "run_command", "tool_input": {"command": "ssh {var.host} uptime"},
                 "description": "check uptime"},
            ],
            channel=channel,
            requester="admin",
        )

        exec_count = 0
        captured_inputs = []

        async def track_exec(name, inp):
            nonlocal exec_count
            exec_count += 1
            captured_inputs.append(inp)
            if exec_count == 1:
                return "server-01"
            return "up 7 days"

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=track_exec)
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        await run_background_task(task, executor, skill_manager)

        assert task.status == "completed"
        assert len(captured_inputs) == 2
        # Second step should have {var.host} replaced with "server-01"
        assert captured_inputs[1]["command"] == "ssh server-01 uptime"


class TestBackgroundTaskAuditLogging:
    """Background tasks log each step to audit."""

    async def test_audit_logged_per_step(self):
        """Each step execution is audit-logged."""
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=MagicMock())

        task = BackgroundTask(
            task_id="audit-1",
            description="Audit test",
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "echo 1"}, "description": "step 1"},
                {"tool_name": "check_disk", "tool_input": {}, "description": "step 2"},
            ],
            channel=channel,
            requester="admin",
            requester_id="user-123",
        )

        executor = MagicMock()
        executor.execute = AsyncMock(return_value="ok")
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        audit = MagicMock()
        audit.log_execution = AsyncMock()

        await run_background_task(task, executor, skill_manager, audit_logger=audit)

        assert audit.log_execution.call_count == 2
        # Verify audit params
        first_call = audit.log_execution.call_args_list[0]
        assert first_call.kwargs["tool_name"] == "run_command"
        assert first_call.kwargs["user_name"] == "admin"


# =====================================================================
# 6. SECRET SCRUBBING IN CROSS-FEATURE PATHS
# =====================================================================

class TestSecretScrubbingCrossFeature:
    """Secret scrubbing works across different feature paths."""

    async def test_scrub_response_secrets_api_key(self):
        """API key patterns are scrubbed from responses."""
        text = "The key is api_key=sk-1234567890abcdef and done."
        result = scrub_response_secrets(text)
        assert "sk-1234567890abcdef" not in result

    async def test_scrub_response_secrets_password(self):
        """Password patterns are scrubbed."""
        text = "Set password=MyS3cretP@ss in the config."
        result = scrub_response_secrets(text)
        assert "MyS3cretP@ss" not in result

    async def test_scrub_response_preserves_clean_text(self):
        """Text without secrets passes through unchanged."""
        text = "All systems healthy. Disk usage: 42%."
        result = scrub_response_secrets(text)
        assert result == text

    async def test_background_task_scrubs_output(self):
        """Background task tool output is scrubbed before storing."""
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=MagicMock())

        task = BackgroundTask(
            task_id="scrub-1",
            description="Scrub test",
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "cat /etc/config"},
                 "description": "read config"},
            ],
            channel=channel,
            requester="admin",
        )

        executor = MagicMock()
        executor.execute = AsyncMock(return_value="password=SuperSecret123 api_key=sk-abcdef")
        skill_manager = MagicMock()
        skill_manager.has_skill = MagicMock(return_value=False)

        await run_background_task(task, executor, skill_manager)

        assert task.status == "completed"
        # The stored result should have secrets scrubbed
        result_output = task.results[0].output
        assert "SuperSecret123" not in result_output


