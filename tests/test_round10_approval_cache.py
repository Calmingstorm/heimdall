"""Tests for session approval cache in the tool loop.

Round 10: When a user approves a destructive tool (e.g., run_command), the
approval is cached for the duration of the current _process_with_tools call.
Subsequent calls to the SAME tool type skip the approval prompt. The cache
resets when the tool loop ends (new message = new approvals).

This reduces UX friction for multi-step tasks that use the same destructive
tool repeatedly (e.g., "check disk on all 5 servers" → 5x run_command).
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal LokiBot stub for approval cache tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._last_tool_use = {}
    stub._system_prompt = "test prompt"
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.tools.approval_timeout_seconds = 30
    stub.sessions = MagicMock()
    stub.codex_client = MagicMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.requires_approval = MagicMock(return_value=None)
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
    stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed
    return stub


def _make_message():
    """Create a mock Discord message with proper typing() setup."""
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

class TestApprovalCache:
    """Test that approval caching works for destructive tools in the tool loop."""

    @pytest.mark.asyncio
    @patch("src.discord.client.request_approval", new_callable=AsyncMock)
    @patch("src.discord.client.requires_approval")
    async def test_approval_cached_across_iterations(
        self, mock_requires_approval, mock_request_approval
    ):
        """First call to a destructive tool prompts approval; second call skips it."""
        stub = _make_bot_stub()
        msg = _make_message()

        # run_command requires approval
        mock_requires_approval.return_value = True
        mock_request_approval.return_value = True

        # Iteration 1: run_command (needs approval) → approved → cached
        # Iteration 2: run_command (cached) → skips approval
        # Iteration 3: text response (loop ends)
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc1", name="run_command", input={"command": "df -h"})]),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc2", name="run_command", input={"command": "free -m"})]),
            LLMResponse(text="Done!", tool_calls=[]),
        ])

        result = await LokiBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "check resources"}]
        )

        # request_approval called ONCE (first iteration), not twice
        assert mock_request_approval.call_count == 1
        assert mock_request_approval.call_args[1]["tool_name"] == "run_command"

    @pytest.mark.asyncio
    @patch("src.discord.client.request_approval", new_callable=AsyncMock)
    @patch("src.discord.client.requires_approval")
    async def test_different_tools_cached_independently(
        self, mock_requires_approval, mock_request_approval
    ):
        """Different destructive tools each need their own first approval."""
        stub = _make_bot_stub()
        msg = _make_message()

        mock_requires_approval.return_value = True
        mock_request_approval.return_value = True

        # Iteration 1: run_command → needs approval
        # Iteration 2: restart_service (different tool) → needs approval
        # Iteration 3: run_command (cached) → skips approval
        # Iteration 4: text response
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc1", name="run_command", input={"command": "df -h"})]),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc2", name="restart_service", input={"service": "nginx"})]),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc3", name="run_command", input={"command": "free -m"})]),
            LLMResponse(text="Done!", tool_calls=[]),
        ])

        result = await LokiBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "check and restart"}]
        )

        # Two approvals: run_command + restart_service (third run_command is cached)
        assert mock_request_approval.call_count == 2
        approved_tools = [c[1]["tool_name"] for c in mock_request_approval.call_args_list]
        assert "run_command" in approved_tools
        assert "restart_service" in approved_tools

    @pytest.mark.asyncio
    @patch("src.discord.client.request_approval", new_callable=AsyncMock)
    @patch("src.discord.client.requires_approval")
    async def test_denied_tool_not_cached(
        self, mock_requires_approval, mock_request_approval
    ):
        """A denied tool is NOT added to the cache — next call still prompts."""
        stub = _make_bot_stub()
        msg = _make_message()

        mock_requires_approval.return_value = True
        # First call: denied; second call: approved
        mock_request_approval.side_effect = [False, True]

        # Iteration 1: run_command → denied → returns denial result
        # Iteration 2: LLM tries run_command again → prompts approval again (not cached)
        # Iteration 3: text response
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc1", name="run_command", input={"command": "rm -rf /"})]),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc2", name="run_command", input={"command": "df -h"})]),
            LLMResponse(text="Done!", tool_calls=[]),
        ])

        result = await LokiBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "clean up"}]
        )

        # Both calls prompted (denied tool was NOT cached)
        assert mock_request_approval.call_count == 2

    @pytest.mark.asyncio
    @patch("src.discord.client.request_approval", new_callable=AsyncMock)
    @patch("src.discord.client.requires_approval")
    async def test_non_destructive_tools_skip_approval(
        self, mock_requires_approval, mock_request_approval
    ):
        """Non-destructive tools (requires_approval=False) never prompt."""
        stub = _make_bot_stub()
        msg = _make_message()

        mock_requires_approval.return_value = False
        mock_request_approval.return_value = True

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc1", name="check_disk", input={"host": "server"})]),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc2", name="check_disk", input={"host": "desktop"})]),
            LLMResponse(text="Done!", tool_calls=[]),
        ])

        result = await LokiBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "check disks"}]
        )

        # No approvals needed
        assert mock_request_approval.call_count == 0

    @pytest.mark.asyncio
    @patch("src.discord.client.request_approval", new_callable=AsyncMock)
    @patch("src.discord.client.requires_approval")
    async def test_cache_is_local_to_loop(
        self, mock_requires_approval, mock_request_approval
    ):
        """Each _process_with_tools call gets a fresh cache (no cross-call leaks)."""
        stub = _make_bot_stub()
        msg = _make_message()

        mock_requires_approval.return_value = True
        mock_request_approval.return_value = True

        # First call: run_command approved and cached
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc1", name="run_command", input={"command": "df -h"})]),
            LLMResponse(text="Done!", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "check disk"}]
        )
        assert mock_request_approval.call_count == 1

        # Second call: run_command should prompt AGAIN (fresh cache)
        mock_request_approval.reset_mock()
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc2", name="run_command", input={"command": "free -m"})]),
            LLMResponse(text="Done!", tool_calls=[]),
        ])
        await LokiBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "check memory"}]
        )
        assert mock_request_approval.call_count == 1

    @pytest.mark.asyncio
    @patch("src.discord.client.request_approval", new_callable=AsyncMock)
    @patch("src.discord.client.requires_approval")
    async def test_skill_approval_also_cached(
        self, mock_requires_approval, mock_request_approval
    ):
        """Skills with requires_approval=True also use the approval cache."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Skill manager says this tool requires approval (overrides registry)
        stub.skill_manager.requires_approval = MagicMock(return_value=True)
        mock_requires_approval.return_value = False  # registry says no, but skill says yes
        mock_request_approval.return_value = True

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc1", name="my_skill", input={"action": "deploy"})]),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc2", name="my_skill", input={"action": "verify"})]),
            LLMResponse(text="Done!", tool_calls=[]),
        ])

        # my_skill isn't in merged tools but the skill_manager handles it
        stub.skill_manager.has_skill = MagicMock(return_value=True)

        result = await LokiBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "deploy and verify"}]
        )

        # Only one approval prompt despite two my_skill calls
        assert mock_request_approval.call_count == 1

    @pytest.mark.asyncio
    @patch("src.discord.client.request_approval", new_callable=AsyncMock)
    @patch("src.discord.client.requires_approval")
    async def test_mixed_destructive_and_safe_tools(
        self, mock_requires_approval, mock_request_approval
    ):
        """Mix of destructive and safe tools: only destructive ones prompt."""
        stub = _make_bot_stub()
        msg = _make_message()

        # run_command requires approval, check_disk doesn't
        def approval_lookup(name):
            return name == "run_command"
        mock_requires_approval.side_effect = approval_lookup
        mock_request_approval.return_value = True

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc1", name="check_disk", input={"host": "server"})]),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc2", name="run_command", input={"command": "df -h"})]),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc3", name="check_disk", input={"host": "desktop"})]),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc4", name="run_command", input={"command": "free -m"})]),
            LLMResponse(text="Done!", tool_calls=[]),
        ])

        result = await LokiBot._process_with_tools(
            stub, msg, [{"role": "user", "content": "check everything"}]
        )

        # Only 1 approval: first run_command. Second run_command is cached.
        assert mock_request_approval.call_count == 1
        assert mock_request_approval.call_args[1]["tool_name"] == "run_command"
