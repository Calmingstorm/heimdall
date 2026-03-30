"""Tests for per-tool timeout in the tool loop.

Round 12: Each tool execution is wrapped in asyncio.wait_for() with a
configurable timeout (config.tools.tool_timeout_seconds, default 300s).
If a tool hangs, the timeout fires, the tool is cancelled, and a clear
error message is returned to the LLM instead of blocking indefinitely.

The timeout is per-tool (not per-step), so if one tool in a parallel
group hangs, the other tools still complete normally.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(tool_timeout: int = 300):
    """Minimal HeimdallBot stub for timeout tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test prompt"
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = tool_timeout
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.respond_to_bots = False
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
        {"name": "run_command", "description": "Run command", "input_schema": {"type": "object", "properties": {}}},
        {"name": "check_disk", "description": "Check disk", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._should_continue_task = HeimdallBot._should_continue_task
    return stub


def _make_message():
    """Create a mock Discord message."""
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = "test-chan"
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None),
    ))
    msg.channel.send = AsyncMock(return_value=AsyncMock())
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    msg.attachments = []
    return msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPerToolTimeout:
    """Test that tool execution is wrapped in asyncio.wait_for with a timeout."""

    @pytest.mark.asyncio
    async def test_tool_timeout_returns_error_message(self):
        """A tool that exceeds the timeout returns a clear error message."""
        stub = _make_bot_stub(tool_timeout=1)  # 1 second timeout
        msg = _make_message()

        # Make the tool hang longer than the timeout
        async def _slow_execute(tool_name, tool_input, user_id=None):
            await asyncio.sleep(10)
            return "this should never be returned"

        stub.tool_executor.execute = _slow_execute

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="run_command", input={"command": "sleep 999"})],
            ),
            LLMResponse(text="The tool timed out.", tool_calls=[]),
        ])

        result = await HeimdallBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "run slow command"}]
        )

        # The LLM received the timeout error and responded with text
        text, _, _, tools_used, _ = result
        assert text == "The tool timed out."

        # Verify the timeout error was passed back to the LLM in messages
        second_call_messages = stub.codex_client.chat_with_tools.call_args_list[1][1]["messages"]
        tool_result_content = second_call_messages[-1]["content"]
        assert any(
            "timed out after 1s" in str(r.get("content", ""))
            for r in tool_result_content
        )

    @pytest.mark.asyncio
    async def test_timeout_audit_logged(self):
        """Timeout events are audit-logged with the correct error message."""
        stub = _make_bot_stub(tool_timeout=1)
        msg = _make_message()

        async def _slow_execute(tool_name, tool_input, user_id=None):
            await asyncio.sleep(10)
            return "never"

        stub.tool_executor.execute = _slow_execute

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="run_command", input={"command": "hang"})],
            ),
            LLMResponse(text="Done", tool_calls=[]),
        ])

        await HeimdallBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "run hanging command"}]
        )

        # Find the audit call for the timeout (may be mixed with normal audit calls)
        timeout_audit_calls = [
            c for c in stub.audit.log_execution.call_args_list
            if c[1].get("error") and "timed out" in c[1]["error"]
        ]
        assert len(timeout_audit_calls) == 1
        call_kwargs = timeout_audit_calls[0][1]
        assert call_kwargs["tool_name"] == "run_command"
        assert call_kwargs["error"] == "Tool 'run_command' timed out after 1s"
        assert call_kwargs["execution_time_ms"] == 1000
        assert call_kwargs["approved"] is True

    @pytest.mark.asyncio
    async def test_one_slow_tool_doesnt_block_fast_tool(self):
        """In a parallel group, one slow tool times out while the other completes."""
        stub = _make_bot_stub(tool_timeout=1)
        msg = _make_message()

        call_count = 0

        async def _mixed_execute(tool_name, tool_input, user_id=None):
            nonlocal call_count
            call_count += 1
            if tool_name == "run_command":
                await asyncio.sleep(10)  # will timeout
                return "never"
            return "fast result"

        stub.tool_executor.execute = _mixed_execute

        # Two tools called in parallel in the same iteration
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="tc1", name="run_command", input={"command": "hang"}),
                    ToolCall(id="tc2", name="check_disk", input={"host": "server"}),
                ],
            ),
            LLMResponse(text="Mixed results", tool_calls=[]),
        ])

        result = await HeimdallBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "check both"}]
        )

        # Verify the fast tool result and timeout error were both returned to LLM
        second_call_messages = stub.codex_client.chat_with_tools.call_args_list[1][1]["messages"]
        tool_results = second_call_messages[-1]["content"]
        results_content = [r["content"] for r in tool_results]

        # One result is the timeout error, the other is the fast result
        assert any("timed out" in c for c in results_content)
        assert any("fast result" in c for c in results_content)

    @pytest.mark.asyncio
    async def test_fast_tool_not_affected_by_timeout(self):
        """A tool that completes quickly is not affected by the timeout."""
        stub = _make_bot_stub(tool_timeout=300)
        msg = _make_message()

        stub.tool_executor.execute = AsyncMock(return_value="fast result")

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="check_disk", input={"host": "server"})],
            ),
            LLMResponse(text="Disk looks good", tool_calls=[]),
        ])

        result = await HeimdallBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "check disk"}]
        )

        text, _, _, tools_used, _ = result
        assert text == "Disk looks good"
        assert "check_disk" in tools_used

    @pytest.mark.asyncio
    async def test_timeout_value_from_config(self):
        """The timeout value comes from config.tools.tool_timeout_seconds."""
        stub = _make_bot_stub(tool_timeout=2)
        msg = _make_message()

        async def _slow_execute(tool_name, tool_input, user_id=None):
            await asyncio.sleep(10)
            return "never"

        stub.tool_executor.execute = _slow_execute

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="run_command", input={"command": "slow"})],
            ),
            LLMResponse(text="Timed out", tool_calls=[]),
        ])

        await HeimdallBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "run slow"}]
        )

        # Check error message references the configured timeout value (2s, not default 300s)
        second_call_messages = stub.codex_client.chat_with_tools.call_args_list[1][1]["messages"]
        tool_result_content = second_call_messages[-1]["content"]
        assert any("timed out after 2s" in str(r.get("content", "")) for r in tool_result_content)

    @pytest.mark.asyncio
    async def test_audit_failure_doesnt_crash_on_timeout(self):
        """If audit logging fails during timeout handling, the tool loop continues."""
        stub = _make_bot_stub(tool_timeout=1)
        msg = _make_message()

        async def _slow_execute(tool_name, tool_input, user_id=None):
            await asyncio.sleep(10)
            return "never"

        stub.tool_executor.execute = _slow_execute
        stub.audit.log_execution = AsyncMock(side_effect=Exception("audit broken"))

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="run_command", input={"command": "hang"})],
            ),
            LLMResponse(text="Recovered", tool_calls=[]),
        ])

        result = await HeimdallBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "run it"}]
        )

        # The tool loop continues despite audit failure
        text, _, _, _, _ = result
        assert text == "Recovered"


class TestToolTimeoutConfig:
    """Test that the tool_timeout_seconds config field works correctly."""

    def test_default_timeout_value(self):
        """Default tool_timeout_seconds is 300."""
        from src.config.schema import ToolsConfig
        config = ToolsConfig()
        assert config.tool_timeout_seconds == 300

    def test_custom_timeout_value(self):
        """tool_timeout_seconds can be set to a custom value."""
        from src.config.schema import ToolsConfig
        config = ToolsConfig(tool_timeout_seconds=120)
        assert config.tool_timeout_seconds == 120
