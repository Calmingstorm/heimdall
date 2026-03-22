"""Tests for unified progress embed in the tool loop.

Round 9: The tool loop now sends a single editable Discord embed instead of
scattered "Running: tool_name..." text messages. The embed:
- Shows each step with status markers (checkmark for done, arrow for running)
- Includes elapsed time for completed steps
- Color-codes by status: blue=running, green=complete, red=error
- Updates in-place via message.edit() instead of new messages
- Includes LLM reasoning text from the running step (in italics)
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.client import AnsiblexBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal AnsiblexBot stub with embed support."""
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
    stub.config.discord.allowed_users = []
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
        {"name": "check_disk", "description": "Check disk", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._build_tool_progress_embed = AnsiblexBot._build_tool_progress_embed
    return stub


def _make_message():
    """Create a mock Discord message with embed support."""
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = "test-chan"
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None),
    ))
    embed_msg = AsyncMock()
    msg.channel.send = AsyncMock(return_value=embed_msg)
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    msg.attachments = []
    return msg


# ---------------------------------------------------------------------------
# _build_tool_progress_embed unit tests
# ---------------------------------------------------------------------------

class TestBuildToolProgressEmbed:
    """Unit tests for the static embed builder."""

    def test_running_step_shows_arrow(self):
        """Running steps show a right-arrow marker."""
        steps = [{"tools": ["check_disk"], "reasoning": None, "status": "running"}]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "running")
        assert "\u25b6 Step 1:" in embed.description  # ► arrow
        assert "`check_disk`" in embed.description

    def test_done_step_shows_checkmark(self):
        """Completed steps show a checkmark with elapsed time."""
        steps = [{"tools": ["check_disk"], "reasoning": None, "status": "done", "elapsed_ms": 1500}]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "running")
        assert "\u2713 Step 1:" in embed.description  # ✓ checkmark
        assert "(1.5s)" in embed.description

    def test_multiple_steps_shown(self):
        """Multiple steps are listed with sequential numbering."""
        steps = [
            {"tools": ["check_disk"], "reasoning": None, "status": "done", "elapsed_ms": 1000},
            {"tools": ["restart_service"], "reasoning": None, "status": "running"},
        ]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "running")
        assert "Step 1:" in embed.description
        assert "Step 2:" in embed.description
        assert "`check_disk`" in embed.description
        assert "`restart_service`" in embed.description

    def test_multiple_tools_in_one_step(self):
        """A step with multiple concurrent tools lists all names."""
        steps = [{"tools": ["check_disk", "check_memory"], "reasoning": None, "status": "running"}]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "running")
        assert "`check_disk`, `check_memory`" in embed.description

    def test_reasoning_shown_for_running_step(self):
        """Reasoning text appears in italics when the latest step is running."""
        steps = [{"tools": ["check_disk"], "reasoning": "I'll check the disk.", "status": "running"}]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "running")
        assert "*I'll check the disk.*" in embed.description

    def test_reasoning_not_shown_for_done_step(self):
        """Reasoning text is NOT shown when the latest step is done."""
        steps = [{"tools": ["check_disk"], "reasoning": "I'll check the disk.", "status": "done", "elapsed_ms": 500}]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "running")
        assert "*I'll check the disk.*" not in embed.description

    def test_running_color_blue(self):
        """Running status uses blue color."""
        steps = [{"tools": ["check_disk"], "reasoning": None, "status": "running"}]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "running")
        assert embed.color == discord.Color.blue()

    def test_complete_color_green(self):
        """Complete status uses green color."""
        steps = [{"tools": ["check_disk"], "reasoning": None, "status": "done", "elapsed_ms": 100}]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "complete")
        assert embed.color == discord.Color.green()

    def test_error_color_red(self):
        """Error status uses red color."""
        steps = [{"tools": ["check_disk"], "reasoning": None, "status": "running"}]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "error")
        assert embed.color == discord.Color.red()

    def test_empty_steps_shows_starting(self):
        """Empty steps list shows 'Starting...'."""
        embed = AnsiblexBot._build_tool_progress_embed([], "running")
        assert embed.description == "Starting..."

    def test_description_truncated_over_4000(self):
        """Very long descriptions are truncated at 4000 chars."""
        # Create a step list that would produce a very long description
        steps = [
            {"tools": [f"tool_{i}"], "reasoning": None, "status": "done", "elapsed_ms": 100}
            for i in range(500)
        ]
        embed = AnsiblexBot._build_tool_progress_embed(steps, "running")
        assert len(embed.description) <= 4020  # 4000 + truncation marker


# ---------------------------------------------------------------------------
# Integration: embed lifecycle in _process_with_tools
# ---------------------------------------------------------------------------

class TestProgressEmbedLifecycle:
    """Test embed creation, update, and final status in the tool loop."""

    async def test_embed_created_on_first_tool_call(self):
        """First tool call creates the progress embed via channel.send."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})], stop_reason="tool_use"),
            LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub._process_with_tools = AnsiblexBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        # Embed was created via channel.send
        msg.channel.send.assert_called()
        embed = msg.channel.send.call_args[1]["embed"]
        assert isinstance(embed, discord.Embed)

    async def test_embed_edited_on_step_completion(self):
        """After tool execution, the embed is edited to show step as done."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})], stop_reason="tool_use"),
            LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub._process_with_tools = AnsiblexBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        # embed_msg.edit should have been called (at least once for step completion)
        embed_msg = msg.channel.send.return_value
        assert embed_msg.edit.call_count >= 1

    async def test_embed_turns_green_on_completion(self):
        """When the loop completes, the embed is updated to green."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})], stop_reason="tool_use"),
            LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub._process_with_tools = AnsiblexBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        # The last edit call should have green color
        embed_msg = msg.channel.send.return_value
        last_edit = embed_msg.edit.call_args_list[-1]
        final_embed = last_edit[1]["embed"]
        assert final_embed.color == discord.Color.green()

    async def test_embed_turns_red_on_max_iterations(self):
        """When max iterations is hit, the embed turns red."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Return tool calls forever (will hit MAX_TOOL_ITERATIONS)
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})],
                stop_reason="tool_use",
            )
        )
        stub._process_with_tools = AnsiblexBot._process_with_tools.__get__(stub)

        text, _, is_error, _, _ = await stub._process_with_tools(msg, [])

        assert is_error is True
        # The last edit should have red color
        embed_msg = msg.channel.send.return_value
        last_edit = embed_msg.edit.call_args_list[-1]
        final_embed = last_edit[1]["embed"]
        assert final_embed.color == discord.Color.red()

    async def test_embed_edited_not_new_message_per_step(self):
        """Multiple tool iterations should EDIT the embed, not send new messages."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})], stop_reason="tool_use"),
            LLMResponse(text="", tool_calls=[ToolCall(id="tc-2", name="check_disk", input={})], stop_reason="tool_use"),
            LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub._process_with_tools = AnsiblexBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        # channel.send should be called only ONCE (for the initial embed)
        assert msg.channel.send.call_count == 1
        # embed_msg.edit should be called multiple times (step completions + new steps)
        embed_msg = msg.channel.send.return_value
        assert embed_msg.edit.call_count >= 2

    async def test_no_embed_for_text_only_response(self):
        """When LLM responds with text only (no tools), no embed is created."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Hello!", tool_calls=[], stop_reason="end_turn")
        )
        stub._process_with_tools = AnsiblexBot._process_with_tools.__get__(stub)

        text, _, _, _, _ = await stub._process_with_tools(msg, [])

        assert text == "Hello!"
        # No embed should be sent for text-only responses
        msg.channel.send.assert_not_called()

    async def test_embed_includes_elapsed_time(self):
        """Completed steps in the embed show elapsed time."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})], stop_reason="tool_use"),
            LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub._process_with_tools = AnsiblexBot._process_with_tools.__get__(stub)

        await stub._process_with_tools(msg, [])

        # After step completion, the embed should show elapsed time
        embed_msg = msg.channel.send.return_value
        # Find the edit call that marks step as done
        done_found = False
        for call in embed_msg.edit.call_args_list:
            embed = call[1]["embed"]
            if "\u2713" in embed.description:  # checkmark = done
                done_found = True
                # Should have elapsed time in parentheses
                assert "s)" in embed.description
                break
        assert done_found, "No step-done embed found in edit calls"

    async def test_embed_survives_channel_send_failure(self):
        """If channel.send fails, the tool loop still works."""
        stub = _make_bot_stub()
        msg = _make_message()
        msg.channel.send = AsyncMock(side_effect=discord.HTTPException(MagicMock(), "send failed"))

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="", tool_calls=[ToolCall(id="tc-1", name="check_disk", input={})], stop_reason="tool_use"),
            LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn"),
        ])
        stub._process_with_tools = AnsiblexBot._process_with_tools.__get__(stub)

        text, _, _, tools, _ = await stub._process_with_tools(msg, [])

        # Should complete successfully despite embed send failure
        assert text == "Done."
        assert tools == ["check_disk"]
