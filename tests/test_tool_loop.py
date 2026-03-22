"""Integration tests for the _process_with_tools tool execution loop.

Exercises the core loop in client.py that iterates tool calls: message
assembly, parallel tool execution, approval flow, progress messages,
tool result truncation, secret scrubbing, audit logging, recent-action
tracking, and max-iteration guard.

The loop uses Codex exclusively via self.codex_client.chat_with_tools()
which returns LLMResponse objects.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    LokiBot,
    MAX_TOOL_ITERATIONS,
    TOOL_OUTPUT_MAX_CHARS,
    truncate_tool_output,
)
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal LokiBot stub for _process_with_tools."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._last_tool_use = {}
    stub._system_prompt = "You are a bot."
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.config.tools.approval_timeout_seconds = 30
    stub.config.tools.auto_approve = False
    stub.codex_client = MagicMock()
    stub.codex_client.chat_with_tools = AsyncMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.requires_approval = MagicMock(return_value=None)
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="OK")
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[{"name": "test"}])
    stub._build_system_prompt = MagicMock(return_value="You are a bot.")
    stub._pending_files = {}
    # Bind real methods
    stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)
    stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)
    return stub


def _make_message(channel_id="chan-1"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.reply = AsyncMock()
    return msg


def _tool_response(tool_calls, text=""):
    """Create an LLMResponse with tool calls (stop_reason=tool_use)."""
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason="tool_use",
    )


def _text_response(text="Done."):
    """Create a final LLMResponse with just text (stop_reason=end_turn)."""
    return LLMResponse(text=text, stop_reason="end_turn")


def _codex_tool_then_text(tool_calls, final_text="Done."):
    """Return a side_effect list for codex_client.chat_with_tools:
    first call returns tool_use, second returns text."""
    return [
        _tool_response(tool_calls),
        _text_response(final_text),
    ]


# ---------------------------------------------------------------------------
# Tool loop iteration & message assembly
# ---------------------------------------------------------------------------

class TestToolLoopIteration:
    """Core iteration: tool call -> execute -> append result -> next iteration."""

    async def test_single_tool_call_and_response(self):
        """One tool call followed by a text response."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-1", name="check_disk", input={"host": "server"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc], "Disk is 42% full.")
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, already_sent, is_error, _tools, _handoff = await stub._process_with_tools(msg, [])

        assert text == "Disk is 42% full."
        assert already_sent is False
        assert is_error is False

    async def test_tool_result_appended_to_messages(self):
        """After tool execution, the result is appended as a user message."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-42", name="run_command", input={"command": "uptime"})
        stub.tool_executor.execute = AsyncMock(return_value="up 5 days")

        captured_messages = []
        original_side_effect = _codex_tool_then_text([tc])

        call_count = 0
        async def _capture_chat(messages, system, tools):
            nonlocal call_count
            captured_messages.append([dict(m) if isinstance(m, dict) else m for m in messages])
            result = original_side_effect[call_count]
            call_count += 1
            return result

        stub.codex_client.chat_with_tools = _capture_chat

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [{"role": "user", "content": "check uptime"}])

        # Second call to chat_with_tools should have tool result in messages
        assert len(captured_messages) == 2
        last_msg = captured_messages[1][-1]
        assert last_msg["role"] == "user"
        # Content is a list of tool_result dicts
        results = last_msg["content"]
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "tool-42"
        assert results[0]["content"] == "up 5 days"

    async def test_assistant_content_appended_before_tool_results(self):
        """The assistant's full response (with tool_use blocks) is appended
        before the user tool_result messages."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-1", name="check_disk", input={})

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            r = [
                _tool_response([tc]),
                _text_response("OK"),
            ][call_count]
            call_count += 1
            return r

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        # After first iteration, messages should have: assistant (tool_use) + user (tool_result)
        second_call_msgs = captured[1]
        assert second_call_msgs[-2]["role"] == "assistant"
        assert second_call_msgs[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# Parallel tool execution
# ---------------------------------------------------------------------------

class TestParallelToolExecution:
    """Multiple tool calls in one iteration are executed concurrently."""

    async def test_multiple_tools_executed_in_parallel(self):
        """Two tool calls in one response should both execute."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="t-1", name="check_disk", input={"host": "server"})
        tc2 = ToolCall(id="t-2", name="check_disk", input={"host": "desktop"})

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc1, tc2])
        )
        stub.tool_executor.execute = AsyncMock(return_value="50%")

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        assert stub.tool_executor.execute.call_count == 2

    async def test_parallel_results_have_correct_tool_use_ids(self):
        """Each tool result should reference the correct tool_use_id."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="id-alpha", name="check_disk", input={})
        tc2 = ToolCall(id="id-beta", name="run_command", input={})

        captured = []
        call_count = 0
        async def _cap(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            r = [
                _tool_response([tc1, tc2]),
                _text_response("Done"),
            ][call_count]
            call_count += 1
            return r

        stub.codex_client.chat_with_tools = _cap
        stub.tool_executor.execute = AsyncMock(return_value="result")

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        tool_results = captured[1][-1]["content"]
        ids = {r["tool_use_id"] for r in tool_results}
        assert ids == {"id-alpha", "id-beta"}


# ---------------------------------------------------------------------------
# Progress messages
# ---------------------------------------------------------------------------

class TestProgressMessages:
    """Progress updates sent to Discord during tool execution via embed."""

    async def test_progress_embed_sent_for_tool_call(self):
        """A progress embed with tool names should be sent."""
        stub = _make_bot_stub()
        msg = _make_message()
        embed_msg = AsyncMock()
        msg.channel.send = AsyncMock(return_value=embed_msg)
        stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed

        tc = ToolCall(id="tool-1", name="check_disk", input={})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        # Check progress embed was sent via channel.send
        msg.channel.send.assert_called()
        embed = msg.channel.send.call_args[1]["embed"]
        assert "`check_disk`" in embed.description

    async def test_progress_embed_no_gear_emoji(self):
        """Progress embeds must not contain gear emojis."""
        stub = _make_bot_stub()
        msg = _make_message()
        embed_msg = AsyncMock()
        msg.channel.send = AsyncMock(return_value=embed_msg)
        stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed

        tc = ToolCall(id="tool-1", name="run_command", input={})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        embed = msg.channel.send.call_args[1]["embed"]
        assert "\u2699" not in embed.description  # no gear emoji
        assert "\ufe0f" not in embed.description  # no variation selector

    async def test_progress_embed_lists_multiple_tools(self):
        """Progress embed should list all tool names when multiple called."""
        stub = _make_bot_stub()
        msg = _make_message()
        embed_msg = AsyncMock()
        msg.channel.send = AsyncMock(return_value=embed_msg)
        stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed

        tc1 = ToolCall(id="t1", name="check_disk", input={})
        tc2 = ToolCall(id="t2", name="check_memory", input={})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc1, tc2])
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        embed = msg.channel.send.call_args[1]["embed"]
        assert "`check_disk`" in embed.description
        assert "`check_memory`" in embed.description


# ---------------------------------------------------------------------------
# Tool approval flow
# ---------------------------------------------------------------------------

class TestToolApproval:
    """Approval required/denied flow for destructive tools."""

    async def test_approved_tool_executes(self):
        """When approval is granted, the tool runs normally."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-1", name="run_command", input={"command": "rm -rf /tmp/test"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )
        stub.tool_executor.execute = AsyncMock(return_value="removed")

        with patch("src.discord.client.requires_approval", return_value=True), \
             patch("src.discord.client.request_approval", new_callable=AsyncMock, return_value=True) as mock_approve, \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        mock_approve.assert_called_once()
        stub.tool_executor.execute.assert_called_once()

    async def test_denied_tool_returns_denial_message(self):
        """When approval is denied, the tool result says 'denied'."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-deny", name="run_command", input={"command": "reboot"})

        captured = []
        call_count = 0
        async def _cap(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            r = [
                _tool_response([tc]),
                _text_response("Action denied."),
            ][call_count]
            call_count += 1
            return r

        stub.codex_client.chat_with_tools = _cap

        with patch("src.discord.client.requires_approval", return_value=True), \
             patch("src.discord.client.request_approval", new_callable=AsyncMock, return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        # Tool executor should NOT have been called
        stub.tool_executor.execute.assert_not_called()
        # Audit log should record denial
        stub.audit.log_execution.assert_called_once()
        assert stub.audit.log_execution.call_args[1]["approved"] is False
        # Tool result should say denied
        tool_results = captured[1][-1]["content"]
        assert "denied" in tool_results[0]["content"].lower()

    async def test_skill_approval_takes_precedence(self):
        """Skill manager's approval setting takes precedence over registry."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-1", name="my_skill", input={})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )
        # Skill says requires approval
        stub.skill_manager.requires_approval = MagicMock(return_value=True)
        stub.skill_manager.has_skill = MagicMock(return_value=True)
        stub.skill_manager.execute = AsyncMock(return_value="skill result")

        with patch("src.discord.client.requires_approval", return_value=False) as reg_approval, \
             patch("src.discord.client.request_approval", new_callable=AsyncMock, return_value=True), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        # request_approval was called because skill said True, even though registry said False
        # The approval was requested (we patched it to return True)
        stub.skill_manager.execute.assert_called_once()


# ---------------------------------------------------------------------------
# Tool output truncation & scrubbing
# ---------------------------------------------------------------------------

class TestToolOutputProcessing:
    """Tool outputs are truncated and scrubbed before being added to messages."""

    async def test_large_output_truncated(self):
        """Tool output exceeding TOOL_OUTPUT_MAX_CHARS is truncated."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="t-big", name="query_prometheus", input={})
        big_output = "x" * 20000
        stub.tool_executor.execute = AsyncMock(return_value=big_output)

        captured = []
        call_count = 0
        async def _cap(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            r = [
                _tool_response([tc]),
                _text_response("Done"),
            ][call_count]
            call_count += 1
            return r

        stub.codex_client.chat_with_tools = _cap

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        tool_content = captured[1][-1]["content"][0]["content"]
        assert len(tool_content) < 20000
        assert "omitted" in tool_content

    async def test_secrets_scrubbed_from_tool_output(self):
        """Tool outputs containing secrets should be scrubbed."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-1", name="run_command", input={})
        stub.tool_executor.execute = AsyncMock(return_value="api_key=sk-abc123secret")

        captured = []
        call_count = 0
        async def _cap(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            r = [
                _tool_response([tc]),
                _text_response("Done"),
            ][call_count]
            call_count += 1
            return r

        stub.codex_client.chat_with_tools = _cap

        # Use real scrub_output_secrets
        with patch("src.discord.client.requires_approval", return_value=False):
            await stub._process_with_tools(msg, [])

        tool_content = captured[1][-1]["content"][0]["content"]
        assert "sk-abc123secret" not in tool_content


# ---------------------------------------------------------------------------
# Audit logging and recent-action tracking
# ---------------------------------------------------------------------------

class TestAuditAndTracking:
    """Audit log and recent-action tracking during tool execution."""

    async def test_audit_log_called_per_tool(self):
        """Each tool execution should produce an audit log entry."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="t1", name="check_disk", input={"host": "server"})
        tc2 = ToolCall(id="t2", name="run_command", input={"command": "ls"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc1, tc2])
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        assert stub.audit.log_execution.call_count == 2
        names = {c[1]["tool_name"] for c in stub.audit.log_execution.call_args_list}
        assert names == {"check_disk", "run_command"}

    async def test_recent_actions_tracked_per_tool(self):
        """Each tool execution should update recent actions for the channel."""
        stub = _make_bot_stub()
        msg = _make_message(channel_id="ch-42")

        tc = ToolCall(id="tool-1", name="docker_logs", input={"container": "bot"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )
        stub.tool_executor.execute = AsyncMock(return_value="log output")

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        assert "ch-42" in stub._recent_actions
        entries = stub._recent_actions["ch-42"]
        assert len(entries) == 1
        assert "docker_logs" in entries[0][1]

    async def test_audit_records_execution_time(self):
        """Audit log should include execution_time_ms."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-1", name="check_disk", input={})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        call_kwargs = stub.audit.log_execution.call_args[1]
        assert "execution_time_ms" in call_kwargs
        assert isinstance(call_kwargs["execution_time_ms"], int)
        assert call_kwargs["execution_time_ms"] >= 0

    async def test_audit_records_error_on_exception(self):
        """When a tool raises an exception, audit should record the error."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-1", name="check_disk", input={})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )
        stub.tool_executor.execute = AsyncMock(side_effect=RuntimeError("SSH timeout"))

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        call_kwargs = stub.audit.log_execution.call_args[1]
        assert call_kwargs["error"] == "SSH timeout"


# ---------------------------------------------------------------------------
# Tool exception handling
# ---------------------------------------------------------------------------

class TestToolExceptionHandling:
    """Tool exceptions should be caught and returned as error results."""

    async def test_exception_returns_error_result(self):
        """A tool that raises should produce an error tool_result, not crash."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="t-err", name="check_disk", input={})
        stub.tool_executor.execute = AsyncMock(side_effect=ValueError("bad input"))

        captured = []
        call_count = 0
        async def _cap(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            r = [
                _tool_response([tc]),
                _text_response("Error noted."),
            ][call_count]
            call_count += 1
            return r

        stub.codex_client.chat_with_tools = _cap

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, is_error, _tools, _handoff = await stub._process_with_tools(msg, [])

        assert is_error is False  # The loop handled the error; final response is valid
        tool_content = captured[1][-1]["content"][0]["content"]
        assert "Error executing check_disk" in tool_content
        assert "bad input" in tool_content

    async def test_one_failing_tool_doesnt_block_others(self):
        """When one tool in a parallel batch fails, others still execute."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="t-ok", name="check_disk", input={})
        tc2 = ToolCall(id="t-fail", name="run_command", input={})

        call_map = {
            "check_disk": "50%",
            "run_command": RuntimeError("connection refused"),
        }

        async def _route_execute(name, inp, **kwargs):
            result = call_map[name]
            if isinstance(result, Exception):
                raise result
            return result

        stub.tool_executor.execute = _route_execute

        captured = []
        call_count = 0
        async def _cap(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            r = [
                _tool_response([tc1, tc2]),
                _text_response("Mixed results."),
            ][call_count]
            call_count += 1
            return r

        stub.codex_client.chat_with_tools = _cap

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        results = captured[1][-1]["content"]
        contents = {r["tool_use_id"]: r["content"] for r in results}
        assert contents["t-ok"] == "50%"
        assert "Error" in contents["t-fail"]


# ---------------------------------------------------------------------------
# Tool selection & system prompt override
# ---------------------------------------------------------------------------

class TestToolSelectionAndPrompt:
    """Tools and system prompt are configured correctly."""

    async def test_tools_disabled_passes_empty_list(self):
        """When tools are disabled, tools=[] is passed to chat_with_tools."""
        stub = _make_bot_stub()
        stub.config.tools.enabled = False
        stub.config.tools.tool_timeout_seconds = 300
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_text_response("Response.")
        )

        await stub._process_with_tools(msg, [])

        call_kwargs = stub.codex_client.chat_with_tools.call_args
        # tools keyword argument should be empty list (None → [] via `tools or []`)
        assert call_kwargs[1]["tools"] == []

    async def test_system_prompt_override_used(self):
        """system_prompt_override should be used instead of _system_prompt."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_text_response("OK.")
        )

        await stub._process_with_tools(
            msg, [], system_prompt_override="Custom prompt here.",
        )

        call_kwargs = stub.codex_client.chat_with_tools.call_args[1]
        assert call_kwargs["system"] == "Custom prompt here."


# ---------------------------------------------------------------------------
# Max iterations guard
# ---------------------------------------------------------------------------

class TestMaxIterations:
    """The tool loop stops after MAX_TOOL_ITERATIONS."""

    async def test_max_iterations_returns_error(self):
        """After MAX_TOOL_ITERATIONS, returns error message."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-1", name="check_disk", input={})
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_response([tc])
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, already_sent, is_error, _tools, _handoff = await stub._process_with_tools(msg, [])

        assert is_error is True
        assert "Too many tool calls" in text
        assert stub.codex_client.chat_with_tools.call_count == MAX_TOOL_ITERATIONS

    async def test_max_iterations_constant_is_20(self):
        """MAX_TOOL_ITERATIONS should be 20."""
        assert MAX_TOOL_ITERATIONS == 20


# ---------------------------------------------------------------------------
# Discord-native tool dispatch
# ---------------------------------------------------------------------------

class TestDiscordNativeTools:
    """Tools handled directly by the bot (not through tool_executor)."""

    async def test_search_history_dispatched_to_handler(self):
        """search_history should call _handle_search_history, not tool_executor."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_search_history = AsyncMock(return_value="Found 3 messages.")

        tc = ToolCall(id="tool-1", name="search_history", input={"query": "deployment"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        stub._handle_search_history.assert_called_once()
        stub.tool_executor.execute.assert_not_called()

    async def test_user_skill_dispatched_to_skill_manager(self):
        """User-created skills go through skill_manager.execute."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.skill_manager.has_skill = MagicMock(return_value=True)
        stub.skill_manager.requires_approval = MagicMock(return_value=False)
        stub.skill_manager.execute = AsyncMock(return_value="skill output")

        tc = ToolCall(id="tool-1", name="my_custom_skill", input={"arg": "val"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        stub.skill_manager.execute.assert_called_once()
        stub.tool_executor.execute.assert_not_called()

    async def test_unknown_tool_falls_through_to_executor(self):
        """Tools not handled by Discord-native dispatch go to tool_executor."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tool-1", name="check_disk", input={"host": "server"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        stub.tool_executor.execute.assert_called_once_with("check_disk", {"host": "server"}, user_id="user-1")
