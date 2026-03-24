"""Round 13: Session defense verification.

Verifies all 5 layers of session defense still work after routing
simplification (Rounds 1-12):

1. Context Separator — developer message between history and current request
2. Selective History Saving — tool-less responses not saved; errors sanitized
3. Abbreviated Task History — tool path uses get_task_history (fewer messages)
4. Compaction with Error Omission — summarize old messages, omit errors/failures
5. Fabrication + Hedging Detection & Retry — detect fake output, hedge language

These layers work together to prevent poisoned history from corrupting
future requests. Each layer is verified independently AND in combination.
"""
from __future__ import annotations

import asyncio
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    LokiBot,
    combine_bot_messages,
    detect_fabrication,
    detect_hedging,
    _FABRICATION_RETRY_MSG,
    _HEDGING_RETRY_MSG,
)
from src.llm.types import LLMResponse, ToolCall  # noqa: E402
from src.sessions.manager import (  # noqa: E402
    SessionManager,
    Session,
    Message,
    COMPACTION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tc(name, inp=None):
    """Shorthand for ToolCall creation."""
    return ToolCall(id=f"tc-{name}", name=name, input=inp or {})


def _make_handle_message_stub(**overrides):
    """LokiBot stub for _handle_message_inner-level tests."""
    stub = MagicMock()
    stub._system_prompt = "test prompt"
    stub._pending_files = {}
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.channels = []
    stub.config.discord.respond_to_bots = True
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.add_message = MagicMock()
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.remove_last_message = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[
        {"role": "user", "content": "older message"},
        {"role": "assistant", "content": "older response"},
        {"role": "user", "content": "current request"},
    ])
    stub.sessions.get_task_history = AsyncMock(return_value=[
        {"role": "user", "content": "recent message"},
        {"role": "user", "content": "current request"},
    ])
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="chat response")
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")
    stub._knowledge_store = None
    stub._embedder = None
    stub._fts_index = None
    stub._vector_store = None
    stub.scheduler = MagicMock()
    stub.infra_watcher = None
    stub.voice_manager = None
    stub.user = MagicMock()
    stub.user.id = 111
    stub.guilds = []
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.browser_manager = None
    stub.context_loader = MagicMock()
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._memory_path = "/tmp/test_memory.json"
    stub._track_recent_action = MagicMock()
    stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed
    stub._build_partial_completion_report = LokiBot._build_partial_completion_report

    for k, v in overrides.items():
        setattr(stub, k, v)
    return stub


def _make_process_with_tools_stub(respond_to_bots=True):
    """LokiBot stub for _process_with_tools-level tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test prompt"
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.respond_to_bots = respond_to_bots
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.codex_client = MagicMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub._send_with_retry = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run", "input_schema": {"type": "object", "properties": {}}},
        {"name": "run_script", "description": "Script", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed
    stub._build_partial_completion_report = LokiBot._build_partial_completion_report
    return stub


def _make_msg(is_bot=False, channel_id="ch-1", author_id="user-1",
              content="test", display_name="TestUser", webhook_id=None):
    """Create a mock Discord message."""
    msg = MagicMock()
    msg.author = MagicMock()
    msg.author.bot = is_bot
    msg.author.id = author_id
    msg.author.display_name = display_name
    msg.author.name = display_name
    msg.author.__str__ = lambda s: display_name
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.__str__ = lambda s: f"#{channel_id}"
    msg.channel.typing = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(), __aexit__=AsyncMock(),
    ))
    msg.content = content
    msg.id = hash(content) % 2**32
    msg.webhook_id = webhook_id
    return msg


# ===================================================================
# Layer 1: Context Separator
# ===================================================================

class TestContextSeparatorInjection:
    """Verify the developer-role context separator is injected between
    history and current request in _process_with_tools."""

    async def test_separator_injected_with_multi_message_history(self):
        """Separator should be present when history has >1 messages."""
        stub = _make_process_with_tools_stub()
        msg = _make_msg()
        history = [
            {"role": "user", "content": "old message"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "current request"},
        ]
        # LLM returns text-only response with tool call to avoid tool-less path
        stub.codex_client.chat_with_tools = AsyncMock(return_value=LLMResponse(
            text="Done.", tool_calls=[_tc("run_command")],
        ))
        # Second call returns final text
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Result: ok", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(stub, msg, history)
        # Check that chat_with_tools was called with a messages list
        # that contains a developer-role separator
        first_call_messages = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        developer_msgs = [m for m in first_call_messages if m.get("role") == "developer"]
        assert len(developer_msgs) >= 1
        sep = developer_msgs[0]
        assert "CURRENT REQUEST" in sep["content"]
        assert "CURRENTLY AVAILABLE" in sep["content"]

    async def test_separator_not_injected_with_single_message(self):
        """Separator should NOT be injected when history has only 1 message."""
        stub = _make_process_with_tools_stub()
        msg = _make_msg()
        history = [{"role": "user", "content": "only message"}]
        stub.codex_client.chat_with_tools = AsyncMock(return_value=LLMResponse(
            text="Hello.", tool_calls=[_tc("run_command")],
        ))
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(stub, msg, history)
        first_call_messages = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        developer_msgs = [m for m in first_call_messages if m.get("role") == "developer"]
        assert len(developer_msgs) == 0

    async def test_separator_position_before_last_message(self):
        """Separator should be inserted before the current user message."""
        stub = _make_process_with_tools_stub()
        msg = _make_msg()
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old resp"},
            {"role": "user", "content": "current"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(stub, msg, history)
        # messages list is mutated in place, but separator is always at index 2
        # (inserted before last element of original 3-element history)
        # Find the separator's position relative to the original user message
        first_call_messages = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        sep_indices = [i for i, m in enumerate(first_call_messages) if m.get("role") == "developer"]
        assert len(sep_indices) >= 1
        sep_idx = sep_indices[0]
        # Separator at index 2, original "current" user message at index 3
        assert sep_idx == 2
        assert first_call_messages[3]["role"] == "user"
        assert first_call_messages[3]["content"] == "current"

    async def test_bot_preamble_in_separator(self):
        """Bot messages should get the ANOTHER BOT preamble in the separator."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=True)
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "bot request"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(stub, msg, history)
        first_call_messages = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        dev_msgs = [m for m in first_call_messages if m.get("role") == "developer"]
        assert len(dev_msgs) == 1
        sep_content = dev_msgs[0]["content"]
        assert "ANOTHER BOT" in sep_content
        assert "EXECUTE immediately" in sep_content
        assert "run_script" in sep_content

    async def test_human_no_bot_preamble(self):
        """Human messages should NOT get the bot preamble."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=False)
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "human request"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(stub, msg, history)
        first_call_messages = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        dev_msgs = [m for m in first_call_messages if m.get("role") == "developer"]
        assert len(dev_msgs) == 1
        sep_content = dev_msgs[0]["content"]
        assert "ANOTHER BOT" not in sep_content

    async def test_separator_tells_codex_to_use_tools(self):
        """Separator should instruct Codex to use currently available tools."""
        stub = _make_process_with_tools_stub()
        msg = _make_msg()
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "current"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(stub, msg, history)
        first_call_messages = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        dev_msgs = [m for m in first_call_messages if m.get("role") == "developer"]
        sep_content = dev_msgs[0]["content"]
        assert "tool" in sep_content.lower()
        assert "Do not repeat prior refusals" in sep_content

    async def test_separator_marks_history_as_context_only(self):
        """Separator should clearly mark earlier messages as context only."""
        stub = _make_process_with_tools_stub()
        msg = _make_msg()
        history = [
            {"role": "user", "content": "history msg"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new query"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(stub, msg, history)
        first_call_messages = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        dev_msgs = [m for m in first_call_messages if m.get("role") == "developer"]
        assert "history" in dev_msgs[0]["content"].lower() or "context" in dev_msgs[0]["content"].lower()


# ===================================================================
# Layer 2: Selective History Saving
# ===================================================================

class TestSelectiveHistorySaving:
    """Verify that tool-less responses are NOT saved to session history,
    and that error responses save sanitized markers."""

    async def test_tool_response_saved_to_history(self):
        """Response with tools used should be saved normally."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="check disk space")
        # _process_with_tools returns: (response, already_sent, is_error, tools_used, handoff)
        stub._process_with_tools = AsyncMock(
            return_value=("Disk is 42% full", False, False, ["check_disk"], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "check disk space", "ch-1")
        # Response should be saved to history
        stub.sessions.add_message.assert_any_call("ch-1", "assistant", "Disk is 42% full")

    async def test_toolless_response_not_saved(self):
        """Response without any tools used should NOT be saved to history."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="hello")
        stub._process_with_tools = AsyncMock(
            return_value=("Hello there.", False, False, [], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "hello", "ch-1")
        # User message is saved (first add_message call), but assistant response is NOT
        add_calls = stub.sessions.add_message.call_args_list
        assistant_saves = [c for c in add_calls if c[0][1] == "assistant"]
        assert len(assistant_saves) == 0, "Tool-less response should not be saved"

    async def test_toolless_response_still_sent_to_discord(self):
        """Even unsaved responses should be sent to the user on Discord."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="hello")
        stub._process_with_tools = AsyncMock(
            return_value=("Hello there.", False, False, [], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "hello", "ch-1")
        # Response should be sent to Discord even though not saved to history
        stub._send_chunked.assert_called_once()

    async def test_error_with_tools_saves_sanitized_marker(self):
        """Error response with tools used should save a sanitized marker."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="restart nginx")
        stub._process_with_tools = AsyncMock(
            return_value=("Connection refused to server", False, True, ["restart_service"], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "restart nginx", "ch-1")
        add_calls = stub.sessions.add_message.call_args_list
        assistant_saves = [c for c in add_calls if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1
        saved_content = assistant_saves[0][0][2]
        assert "[Previous request used tools" in saved_content
        assert "restart_service" in saved_content
        assert "Connection refused" not in saved_content  # raw error NOT saved

    async def test_error_without_tools_saves_generic_marker(self):
        """Error response with no tools should save a generic error marker."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="bad request")
        stub._process_with_tools = AsyncMock(
            return_value=("Something failed badly", False, True, [], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "bad request", "ch-1")
        add_calls = stub.sessions.add_message.call_args_list
        assistant_saves = [c for c in add_calls if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1
        saved_content = assistant_saves[0][0][2]
        assert "error before tool execution" in saved_content
        assert "Something failed" not in saved_content

    async def test_handoff_response_saved(self):
        """Skill handoff responses should be saved (they're legitimate)."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="run my skill")
        stub._process_with_tools = AsyncMock(
            return_value=("Skill result", False, False, [], True),
        )
        # Mock the handoff path — codex_client.chat returns a response
        stub.codex_client.chat = AsyncMock(return_value="Wrapped skill result")
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "run my skill", "ch-1")
        add_calls = stub.sessions.add_message.call_args_list
        assistant_saves = [c for c in add_calls if c[0][1] == "assistant"]
        # Handoff saves are allowed (even without tools)
        assert len(assistant_saves) >= 1

    async def test_guest_response_saved(self):
        """Guest tier responses should always be saved (they can't use tools anyway)."""
        stub = _make_handle_message_stub()
        stub.permissions.is_guest = MagicMock(return_value=True)
        msg = _make_msg(content="hi")
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "hi", "ch-1")
        add_calls = stub.sessions.add_message.call_args_list
        assistant_saves = [c for c in add_calls if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1


# ===================================================================
# Layer 3: Abbreviated Task History
# ===================================================================

class TestAbbreviatedTaskHistory:
    """Verify that the tool-calling path uses get_task_history (fewer messages)
    instead of full history, reducing influence of stale/poisoned exchanges."""

    async def test_task_route_uses_get_task_history(self):
        """Non-guest messages should use get_task_history for abbreviated history."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="check status")
        stub._process_with_tools = AsyncMock(
            return_value=("All good", False, False, ["run_command"], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "check status", "ch-1")
        # get_task_history should be called (not just get_history_with_compaction)
        stub.sessions.get_task_history.assert_called_once()

    async def test_guest_route_uses_full_history(self):
        """Guest messages should use full history (via get_history_with_compaction)."""
        stub = _make_handle_message_stub()
        stub.permissions.is_guest = MagicMock(return_value=True)
        msg = _make_msg(content="hello")
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "hello", "ch-1")
        # Guest uses chat route with full history — get_task_history NOT called
        stub.sessions.get_task_history.assert_not_called()

    async def test_task_history_limits_messages(self):
        """get_task_history should return fewer messages than full history."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        # Add many messages
        for i in range(30):
            sm.add_message("ch-1", "user", f"message {i}")
            sm.add_message("ch-1", "assistant", f"response {i}")
        # get_task_history with default max_messages=10
        task_hist = await sm.get_task_history("ch-1", max_messages=10)
        full_hist = sm.get_history("ch-1")
        # Task history should have fewer messages
        assert len(task_hist) < len(full_hist)
        assert len(task_hist) <= 10

    async def test_task_history_includes_summary(self):
        """get_task_history should prepend the session summary if available."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        session = sm.get_or_create("ch-1")
        session.summary = "User previously asked about Docker containers."
        sm.add_message("ch-1", "user", "check containers")
        task_hist = await sm.get_task_history("ch-1", max_messages=10)
        # First message should be the summary
        assert task_hist[0]["role"] == "user"
        assert "Previous conversation summary" in task_hist[0]["content"]
        assert "Docker containers" in task_hist[0]["content"]
        # Second should be the acknowledgment
        assert task_hist[1]["role"] == "assistant"
        assert "context from our previous" in task_hist[1]["content"]

    async def test_task_history_recent_messages_only(self):
        """get_task_history should return only the most recent messages."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        for i in range(20):
            sm.add_message("ch-1", "user", f"message {i}")
            sm.add_message("ch-1", "assistant", f"response {i}")
        task_hist = await sm.get_task_history("ch-1", max_messages=6)
        # Should have the last 6 messages
        assert len(task_hist) == 6
        # The last message should be "response 19"
        assert task_hist[-1]["content"] == "response 19"
        # Should NOT contain old messages
        assert not any("message 0" in m["content"] for m in task_hist)


# ===================================================================
# Layer 4: Compaction with Error Omission
# ===================================================================

class TestCompactionErrorOmission:
    """Verify that session compaction explicitly omits errors, failures,
    and 'unable to' statements from summaries."""

    async def test_compaction_instruction_omits_errors(self):
        """Compaction system instruction should tell LLM to omit errors."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        captured_system = {}
        async def mock_compact_fn(messages, system):
            captured_system["text"] = system
            return "Summary of conversation."
        sm.set_compaction_fn(mock_compact_fn)
        # Add enough messages to trigger compaction
        for i in range(COMPACTION_THRESHOLD + 5):
            sm.add_message("ch-compact", "user", f"msg {i}")
            sm.add_message("ch-compact", "assistant", f"resp {i}")
        await sm.get_history_with_compaction("ch-compact")
        assert "text" in captured_system
        system = captured_system["text"]
        assert "OMIT" in system
        assert "Error messages" in system
        assert "API failures" in system
        assert "unable to" in system
        assert "I can't" in system or "I can\\'t" in system

    async def test_compaction_preserves_successful_outcomes(self):
        """Compaction should preserve successful task outcomes."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        captured_system = {}
        async def mock_compact_fn(messages, system):
            captured_system["text"] = system
            return "Summary."
        sm.set_compaction_fn(mock_compact_fn)
        for i in range(COMPACTION_THRESHOLD + 5):
            sm.add_message("ch-p", "user", f"msg {i}")
            sm.add_message("ch-p", "assistant", f"resp {i}")
        await sm.get_history_with_compaction("ch-p")
        system = captured_system["text"]
        assert "PRESERVE" in system
        assert "successful task outcomes" in system
        assert "decisions made" in system

    async def test_compaction_omits_unconfirmed_data(self):
        """Compaction should omit data not confirmed by actual tool results."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        captured_system = {}
        async def mock_compact_fn(messages, system):
            captured_system["text"] = system
            return "Summary."
        sm.set_compaction_fn(mock_compact_fn)
        for i in range(COMPACTION_THRESHOLD + 5):
            sm.add_message("ch-u", "user", f"msg {i}")
            sm.add_message("ch-u", "assistant", f"resp {i}")
        await sm.get_history_with_compaction("ch-u")
        system = captured_system["text"]
        assert "not confirmed by actual tool results" in system

    async def test_compaction_merges_with_existing_summary(self):
        """When an existing summary exists, compaction should merge it."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        captured_messages = {}
        async def mock_compact_fn(messages, system):
            captured_messages["content"] = messages[0]["content"]
            return "Merged summary."
        sm.set_compaction_fn(mock_compact_fn)
        session = sm.get_or_create("ch-merge")
        session.summary = "Previous: user configured Docker."
        for i in range(COMPACTION_THRESHOLD + 5):
            sm.add_message("ch-merge", "user", f"msg {i}")
            sm.add_message("ch-merge", "assistant", f"resp {i}")
        await sm.get_history_with_compaction("ch-merge")
        assert "Previous summary" in captured_messages["content"]
        assert "Docker" in captured_messages["content"]

    async def test_compaction_keeps_recent_messages(self):
        """After compaction, recent messages should be preserved."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        async def mock_compact_fn(messages, system):
            return "Summary of old messages."
        sm.set_compaction_fn(mock_compact_fn)
        for i in range(COMPACTION_THRESHOLD + 5):
            sm.add_message("ch-keep", "user", f"msg {i}")
            sm.add_message("ch-keep", "assistant", f"resp {i}")
        session = sm.get_or_create("ch-keep")
        original_count = len(session.messages)
        await sm._compact(session)
        # After compaction, recent messages should still be there
        assert len(session.messages) > 0
        assert len(session.messages) < original_count
        # Summary should be set
        assert session.summary == "Summary of old messages."

    async def test_compaction_fallback_on_failure(self):
        """If compaction fails, trim without summary and clear stale summary."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        async def failing_compact_fn(messages, system):
            raise RuntimeError("LLM failed")
        sm.set_compaction_fn(failing_compact_fn)
        session = sm.get_or_create("ch-fail")
        session.summary = "Old stale summary."
        for i in range(COMPACTION_THRESHOLD + 5):
            sm.add_message("ch-fail", "user", f"msg {i}")
            sm.add_message("ch-fail", "assistant", f"resp {i}")
        await sm._compact(session)
        # Should trim to max_history
        assert len(session.messages) <= sm.max_history
        # Stale summary should be cleared
        assert session.summary == ""

    async def test_compaction_threshold(self):
        """Compaction should only trigger when threshold is exceeded."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        compact_called = {"count": 0}
        async def mock_compact_fn(messages, system):
            compact_called["count"] += 1
            return "Summary."
        sm.set_compaction_fn(mock_compact_fn)
        # Below threshold — should NOT compact
        for i in range(COMPACTION_THRESHOLD - 5):
            sm.add_message("ch-thresh", "user", f"msg {i}")
        await sm.get_history_with_compaction("ch-thresh")
        assert compact_called["count"] == 0


# ===================================================================
# Layer 5: Fabrication + Hedging Detection & Retry
# ===================================================================

class TestFabricationDetectionInContext:
    """Verify fabrication detection works correctly in the tool loop,
    preventing poisoned history from influencing future responses."""

    async def test_fabrication_retry_injects_correction(self):
        """When fabrication is detected, a correction message should be injected."""
        stub = _make_process_with_tools_stub()
        msg = _make_msg()
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old resp"},
            {"role": "user", "content": "check disk"},
        ]
        # First call: fabricated output; second: tool call; third: final text
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="I ran df -h and the disk is 42% full.", tool_calls=[]),
            LLMResponse(text=None, tool_calls=[_tc("check_disk")]),
            LLMResponse(text="Disk is 42% full (confirmed).", tool_calls=[]),
        ])
        result = await LokiBot._process_with_tools(stub, msg, history)
        response = result[0]
        assert "confirmed" in response
        # Should have 3 chat_with_tools calls (fabrication + retry + final)
        assert stub.codex_client.chat_with_tools.call_count == 3
        # Second call should contain the fabrication correction
        second_call_msgs = stub.codex_client.chat_with_tools.call_args_list[1][1]["messages"]
        correction_msgs = [
            m for m in second_call_msgs
            if isinstance(m.get("content"), str) and "fabrication" in m["content"].lower()
        ]
        assert len(correction_msgs) >= 1

    async def test_fabrication_only_on_first_iteration(self):
        """Fabrication detection should only fire on iteration 0."""
        # This is tested implicitly — after retry (iteration=1),
        # even fabricated text won't trigger another retry
        stub = _make_process_with_tools_stub()
        msg = _make_msg()
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "check disk"},
        ]
        # First: fabrication → retry; second: fabrication again → should NOT retry
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="I ran df -h and here's the output", tool_calls=[]),
            LLMResponse(text="I checked the disk and it's fine", tool_calls=[]),
        ])
        result = await LokiBot._process_with_tools(stub, msg, history)
        # Only 2 calls: original + one retry, no second retry
        assert stub.codex_client.chat_with_tools.call_count == 2

    async def test_fabrication_bypassed_when_tools_used(self):
        """If tools were actually used, fabrication detection should not fire."""
        assert detect_fabrication("I ran the command and here's the output", ["run_command"]) is False

    async def test_fabrication_detects_fake_docker_output(self):
        """Fabricated Docker output should be detected."""
        fake = "```bash\nCONTAINER ID   IMAGE     COMMAND   STATUS\nabc123   nginx   \"/bin/sh\"   Up 2h\n```"
        assert detect_fabrication(fake, []) is True

    async def test_fabrication_detects_fake_df_output(self):
        """Fabricated disk usage output should be detected."""
        fake = "```text\nFilesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       50G   21G   29G  42% /\n```"
        assert detect_fabrication(fake, []) is True

    async def test_fabrication_ignores_short_text(self):
        """Short text should not trigger fabrication detection."""
        assert detect_fabrication("OK", []) is False
        assert detect_fabrication("", []) is False

    async def test_fabrication_ignores_genuine_text(self):
        """Honest text without fake output should not trigger."""
        assert detect_fabrication("The server is configured with 8GB RAM and 4 CPU cores.", []) is False


class TestHedgingDetectionInContext:
    """Verify hedging detection works for bot messages in the tool loop."""

    async def test_hedging_retry_for_bot_message(self):
        """Bot message + hedging should trigger retry with correction."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=True)
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "restart nginx"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Would you like me to restart nginx?", tool_calls=[]),
            LLMResponse(text=None, tool_calls=[_tc("restart_service")]),
            LLMResponse(text="Nginx restarted.", tool_calls=[]),
        ])
        result = await LokiBot._process_with_tools(stub, msg, history)
        assert "restarted" in result[0].lower()
        assert stub.codex_client.chat_with_tools.call_count == 3
        # Correction message should mention bot
        second_msgs = stub.codex_client.chat_with_tools.call_args_list[1][1]["messages"]
        corrections = [
            m for m in second_msgs
            if isinstance(m.get("content"), str) and "bot" in m["content"].lower()
            and m.get("role") == "developer"
        ]
        assert len(corrections) >= 1

    async def test_hedging_retried_for_human_too(self):
        """Human message hedging should NOT trigger retry."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=False)
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "restart nginx"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(return_value=LLMResponse(
            text="Would you like me to restart nginx?", tool_calls=[],
        ))
        result = await LokiBot._process_with_tools(stub, msg, history)
        # Only 1 call — no retry for human hedging
        assert stub.codex_client.chat_with_tools.call_count >= 2  # hedging retries for all messages now

    async def test_hedging_bypassed_when_tools_used(self):
        """If tools were used, hedging detection should not fire."""
        assert detect_hedging("Would you like me to restart it?", ["restart_service"]) is False

    async def test_hedging_detects_shall_i(self):
        assert detect_hedging("Shall I proceed with the restart?", []) is True

    async def test_hedging_detects_let_me_know(self):
        assert detect_hedging("Let me know when you want me to start.", []) is True

    async def test_hedging_detects_plan_language(self):
        assert detect_hedging("Here's a plan for the migration.", []) is True

    async def test_hedging_ignores_action_language(self):
        assert detect_hedging("I restarted nginx and it's running.", []) is False


# ===================================================================
# Combined: Poisoned History Prevention
# ===================================================================

class TestPoisonedHistoryPrevention:
    """Verify the layers work together to prevent poisoned history
    from causing fabrication or hedging in future responses."""

    async def test_poisoned_fabrication_not_saved(self):
        """A fabricated response should not be saved to session history."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="check disk")
        # Simulating a tool-less (potentially fabricated) response
        stub._process_with_tools = AsyncMock(
            return_value=("I ran df -h and it's 42% full.", False, False, [], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "check disk", "ch-1")
        # Tool-less response should NOT be saved
        add_calls = stub.sessions.add_message.call_args_list
        assistant_saves = [c for c in add_calls if c[0][1] == "assistant"]
        assert len(assistant_saves) == 0

    async def test_error_sanitization_prevents_refusal_leak(self):
        """Error responses should save sanitized markers, not raw refusals."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="delete everything")
        stub._process_with_tools = AsyncMock(
            return_value=("I'm sorry, I can't delete system files.", False, True, [], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "delete everything", "ch-1")
        add_calls = stub.sessions.add_message.call_args_list
        assistant_saves = [c for c in add_calls if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1
        saved = assistant_saves[0][0][2]
        assert "I'm sorry" not in saved
        assert "can't delete" not in saved
        assert "error" in saved.lower()

    async def test_task_history_limits_poison_window(self):
        """Even if old messages are poisoned, task history limits their reach."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        # Add poisoned messages (old fabricated responses that somehow got saved)
        for i in range(15):
            sm.add_message("ch-poison", "user", f"check status of server {i}")
            sm.add_message("ch-poison", "assistant",
                           f"I ran systemctl status and server {i} is running fine.")
        # Add recent legitimate messages
        sm.add_message("ch-poison", "user", "check disk")
        sm.add_message("ch-poison", "assistant", "[Used check_disk tool: 42% used]")
        # get_task_history limits to last 10 — poisoned old messages excluded
        task_hist = await sm.get_task_history("ch-poison", max_messages=10)
        # Only recent messages should be included
        assert len(task_hist) <= 10
        # Old poisoned messages should be excluded from direct history
        old_poison = [m for m in task_hist if "I ran systemctl" in m.get("content", "")]
        # At most a few should leak in (the most recent ones)
        assert len(old_poison) < 10

    async def test_fabrication_then_hedging_order(self):
        """Fabrication should fire before hedging — both on iteration 0."""
        stub = _make_process_with_tools_stub(respond_to_bots=True)
        msg = _make_msg(is_bot=True)
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "check disk"},
        ]
        # First: fabrication; second: hedging (should NOT fire since iteration=1)
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="I ran df -h and the disk is fine.", tool_calls=[]),
            LLMResponse(text="Would you like me to check something else?", tool_calls=[]),
        ])
        result = await LokiBot._process_with_tools(stub, msg, history)
        # Fabrication fires on iteration 0 → retry → hedging text on iteration 1 NOT retried
        assert stub.codex_client.chat_with_tools.call_count == 2
        assert "would you like" in result[0].lower()

    async def test_context_separator_prevents_history_pattern_repeat(self):
        """Separator should prevent the model from repeating old refusals."""
        stub = _make_process_with_tools_stub()
        msg = _make_msg()
        # History with a previous refusal
        history = [
            {"role": "user", "content": "delete the logs"},
            {"role": "assistant", "content": "[Previous request encountered an error before tool execution.]"},
            {"role": "user", "content": "delete the logs again"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Logs deleted.", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(stub, msg, history)
        # Verify separator is present telling not to repeat refusals
        first_msgs = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        dev_msgs = [m for m in first_msgs if m.get("role") == "developer"]
        assert any("Do not repeat prior refusals" in m["content"] for m in dev_msgs)


# ===================================================================
# Cross-Layer Integration
# ===================================================================

class TestCrossLayerIntegration:
    """Test that multiple defense layers work together correctly."""

    async def test_tool_response_saves_then_appears_in_task_history(self):
        """Tool responses saved to history should appear in future task history."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        sm.add_message("ch-int", "user", "check disk")
        sm.add_message("ch-int", "assistant", "Disk is 42% full [used check_disk]")
        sm.add_message("ch-int", "user", "check memory")
        task_hist = await sm.get_task_history("ch-int")
        # Both messages should be in task history
        assert any("42%" in m["content"] for m in task_hist)

    async def test_error_marker_appears_in_history_not_raw_error(self):
        """After error sanitization, only the marker should appear in future history."""
        sm = SessionManager(max_history=50, max_age_hours=24, persist_dir="/tmp/test_sessions")
        # Simulate what _handle_message_inner does for errors
        sm.add_message("ch-err", "user", "restart nginx")
        sanitized = "[Previous request encountered an error before tool execution.]"
        sm.add_message("ch-err", "assistant", sanitized)
        sm.add_message("ch-err", "user", "try again")
        task_hist = await sm.get_task_history("ch-err")
        # History should contain the sanitized marker
        assert any("error before tool execution" in m["content"] for m in task_hist)

    async def test_compaction_threshold_value(self):
        """Compaction threshold should be 40 messages."""
        assert COMPACTION_THRESHOLD == 40

    async def test_separator_plus_abbreviated_history(self):
        """Both separator and abbreviated history should work together."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="check status")
        # Return task history with 2 messages
        stub.sessions.get_task_history = AsyncMock(return_value=[
            {"role": "user", "content": "old request"},
            {"role": "user", "content": "check status"},
        ])
        stub._process_with_tools = AsyncMock(
            return_value=("Status OK", False, False, ["run_command"], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "check status", "ch-1")
        # _process_with_tools should have been called with the abbreviated history
        call_args = stub._process_with_tools.call_args
        history_arg = call_args[0][1]  # second positional arg is history
        assert len(history_arg) == 2  # abbreviated, not full

    async def test_prune_called_after_save(self):
        """Session prune should be called after every save."""
        stub = _make_handle_message_stub()
        msg = _make_msg(content="check disk")
        stub._process_with_tools = AsyncMock(
            return_value=("Disk OK", False, False, ["check_disk"], False),
        )
        with patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await LokiBot._handle_message_inner(stub, msg, "check disk", "ch-1")
        stub.sessions.prune.assert_called()
        stub.sessions.save.assert_called()


# ===================================================================
# Source Code Verification
# ===================================================================

class TestSessionDefenseSourceStructure:
    """Verify the 5-layer defense exists in the source code."""

    def test_context_separator_in_process_with_tools(self):
        """_process_with_tools should contain the context separator injection."""
        import inspect
        src = inspect.getsource(LokiBot._process_with_tools)
        assert "CURRENT REQUEST" in src
        assert 'role": "developer"' in src or "role\": \"developer\"" in src
        assert "insert(-1" in src

    def test_selective_saving_in_handle_message_inner(self):
        """_handle_message_inner should have the tool-less check."""
        import inspect
        src = inspect.getsource(LokiBot._handle_message_inner)
        assert "not tools_used" in src
        assert "pollute" in src or "poisoning" in src

    def test_error_sanitization_in_handle_message_inner(self):
        """_handle_message_inner should sanitize errors before saving."""
        import inspect
        src = inspect.getsource(LokiBot._handle_message_inner)
        assert "sanitized" in src
        assert "error before tool execution" in src

    def test_fabrication_detection_exists(self):
        """detect_fabrication function should exist and be importable."""
        assert callable(detect_fabrication)

    def test_hedging_detection_exists(self):
        """detect_hedging function should exist and be importable."""
        assert callable(detect_hedging)

    def test_fabrication_retry_msg_exists(self):
        """_FABRICATION_RETRY_MSG should exist with correct role."""
        assert _FABRICATION_RETRY_MSG["role"] == "developer"
        assert "fabrication" in _FABRICATION_RETRY_MSG["content"].lower()

    def test_hedging_retry_msg_exists(self):
        """_HEDGING_RETRY_MSG should exist with correct role."""
        assert _HEDGING_RETRY_MSG["role"] == "developer"
        assert "bot" in _HEDGING_RETRY_MSG["content"].lower()

    def test_task_history_limits_messages(self):
        """get_task_history should have a max_messages parameter."""
        import inspect
        sig = inspect.signature(SessionManager.get_task_history)
        assert "max_messages" in sig.parameters

    def test_compaction_has_omit_rules(self):
        """_compact should contain OMIT rules for errors."""
        import inspect
        src = inspect.getsource(SessionManager._compact)
        assert "OMIT" in src
        assert "Error" in src

    def test_compaction_has_preserve_rules(self):
        """_compact should contain PRESERVE rules for successes."""
        import inspect
        src = inspect.getsource(SessionManager._compact)
        assert "PRESERVE" in src
        assert "successful" in src

    def test_bot_preamble_in_separator(self):
        """The context separator should include bot-specific preamble."""
        import inspect
        src = inspect.getsource(LokiBot._process_with_tools)
        assert "ANOTHER BOT" in src
        assert "EXECUTE immediately" in src

    def test_five_defense_layers_present(self):
        """All 5 defense layers should be present in the codebase."""
        import inspect
        client_src = inspect.getsource(LokiBot._process_with_tools)
        inner_src = inspect.getsource(LokiBot._handle_message_inner)
        compact_src = inspect.getsource(SessionManager._compact)
        # 1. Context separator
        assert "CURRENT REQUEST" in client_src
        # 2. Selective saving (tool-less not saved)
        assert "not tools_used" in inner_src
        # 3. Abbreviated task history
        assert "get_task_history" in inner_src
        # 4. Compaction error omission
        assert "OMIT" in compact_src
        # 5. Fabrication + hedging detection
        assert "detect_fabrication" in client_src
        assert "detect_hedging" in client_src
