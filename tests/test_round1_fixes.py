"""Tests for Round 1-5 routing and prompt fixes.

Covers:
- P2: schedule_task proactive defense (requires scheduling intent in user message)
- P3: nonlocal system_prompt in _run_tool (skill CRUD updates prompt)
- P3: Compaction routed through compaction_fn (Codex)
- Round 3: nonlocal system_prompt behavioral test (prompt updates across iterations)
- Round 3: Codex fallback double-failure in claude_code path
"""
from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports trigger __init__
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402
from src.sessions.manager import (  # noqa: E402
    COMPACTION_THRESHOLD,
    SessionManager,
)


# ---------------------------------------------------------------------------
# P2: schedule_task proactive defense
# ---------------------------------------------------------------------------

class TestScheduleTaskDefense:
    """_handle_schedule_task rejects calls when user message lacks scheduling intent."""

    def _make_bot_stub(self, user_content: str):
        """Create a minimal stub with a Discord message containing user_content."""
        from src.discord.client import LokiBot

        stub = MagicMock()
        stub.scheduler = MagicMock()
        stub.scheduler.add = MagicMock(return_value={
            "id": "sched-1",
            "description": "Test task",
            "next_run": "soon",
        })
        # Bind the real _handle_schedule_task method
        stub._handle_schedule_task = LokiBot._handle_schedule_task.__get__(stub)
        # Also bind the class attribute for the regex
        stub._SCHEDULE_INTENT_RE = LokiBot._SCHEDULE_INTENT_RE

        msg = MagicMock()
        msg.content = user_content
        msg.channel = MagicMock()
        msg.channel.id = "67890"
        return stub, msg

    def test_allows_explicit_schedule_request(self):
        """schedule_task proceeds when user message says 'schedule'."""
        stub, msg = self._make_bot_stub("schedule a disk check every hour")
        result = stub._handle_schedule_task(msg, {
            "description": "Disk check",
            "action": "tool",
            "cron": "0 * * * *",
        })
        assert "Scheduled" in result
        stub.scheduler.add.assert_called_once()

    def test_allows_remind_request(self):
        """schedule_task proceeds when user message says 'remind me'."""
        stub, msg = self._make_bot_stub("remind me to check the logs tomorrow")
        result = stub._handle_schedule_task(msg, {
            "description": "Check logs",
            "action": "reminder",
        })
        assert "Scheduled" in result

    def test_allows_every_n_pattern(self):
        """schedule_task proceeds when user message says 'every 5 minutes'."""
        stub, msg = self._make_bot_stub("run this every 5 minutes")
        result = stub._handle_schedule_task(msg, {
            "description": "Recurring task",
            "action": "tool",
            "cron": "*/5 * * * *",
        })
        assert "Scheduled" in result

    def test_allows_at_time_pattern(self):
        """schedule_task proceeds when user message says 'at 3pm'."""
        stub, msg = self._make_bot_stub("run a backup at 3pm")
        result = stub._handle_schedule_task(msg, {
            "description": "Backup",
            "action": "tool",
        })
        assert "Scheduled" in result

    def test_allows_daily_pattern(self):
        """schedule_task proceeds when user message says 'daily'."""
        stub, msg = self._make_bot_stub("send me a daily digest")
        result = stub._handle_schedule_task(msg, {
            "description": "Daily digest",
            "action": "reminder",
        })
        assert "Scheduled" in result

    def test_blocks_proactive_scheduling(self):
        """schedule_task blocked when user message has no scheduling intent."""
        stub, msg = self._make_bot_stub("what's the CPU usage on the server?")
        result = stub._handle_schedule_task(msg, {
            "description": "CPU monitor",
            "action": "tool",
            "cron": "*/5 * * * *",
        })
        assert "doesn't appear to request scheduling" in result
        stub.scheduler.add.assert_not_called()

    def test_blocks_casual_conversation(self):
        """schedule_task blocked for casual conversation."""
        stub, msg = self._make_bot_stub("hey how's it going?")
        result = stub._handle_schedule_task(msg, {
            "description": "Random",
            "action": "reminder",
        })
        assert "doesn't appear to request scheduling" in result
        stub.scheduler.add.assert_not_called()

    def test_blocks_pizza_ordering_misroute(self):
        """schedule_task blocked for the 'pizza photo' misroute example."""
        stub, msg = self._make_bot_stub("look at this pizza photo")
        result = stub._handle_schedule_task(msg, {
            "description": "Pizza delivery",
            "action": "reminder",
        })
        assert "doesn't appear to request scheduling" in result

    def test_allows_in_n_minutes(self):
        """schedule_task proceeds when user says 'in 30 minutes'."""
        stub, msg = self._make_bot_stub("restart nginx in 30 minutes")
        result = stub._handle_schedule_task(msg, {
            "description": "Restart nginx",
            "action": "tool",
        })
        assert "Scheduled" in result

    def test_allows_tomorrow(self):
        """schedule_task proceeds when user says 'tomorrow'."""
        stub, msg = self._make_bot_stub("deploy the update tomorrow")
        result = stub._handle_schedule_task(msg, {
            "description": "Deploy update",
            "action": "tool",
        })
        assert "Scheduled" in result


# ---------------------------------------------------------------------------
# P3: Compaction routed through compaction_fn
# ---------------------------------------------------------------------------

class TestCompactionThroughChat:
    """Compaction now routes through a registered compaction_fn (Codex)."""

    @pytest.mark.asyncio
    async def test_compaction_uses_compaction_fn(self, tmp_dir):
        """Compaction calls the registered compaction_fn."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "test_ch"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        mock_fn = AsyncMock(return_value="Summarized conversation.")
        mgr.set_compaction_fn(mock_fn)

        await mgr.get_history_with_compaction(channel)

        mock_fn.assert_awaited_once()
        # First arg is messages list, second is system instruction
        args = mock_fn.call_args[0]
        assert len(args[0]) == 1  # single message
        assert args[0][0]["role"] == "user"
        assert isinstance(args[1], str)  # system instruction

    @pytest.mark.asyncio
    async def test_compaction_handles_fn_failure(self, tmp_dir):
        """Compaction degrades gracefully when compaction_fn fails."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "test_ch"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        failing_fn = AsyncMock(side_effect=RuntimeError("backend down"))
        mgr.set_compaction_fn(failing_fn)

        await mgr.get_history_with_compaction(channel)

        # Should have trimmed without summary
        session = mgr.get_or_create(channel)
        assert len(session.messages) <= 30
        assert session.summary == ""


# ---------------------------------------------------------------------------
# Round 3: nonlocal system_prompt behavioral test
# ---------------------------------------------------------------------------

class TestNonlocalSystemPromptBehavior:
    """Verify that skill CRUD in _run_tool actually updates system_prompt
    for subsequent tool loop iterations (the nonlocal fix)."""

    def _make_bot_stub(self):
        from src.discord.client import LokiBot
        from src.llm.types import LLMResponse, ToolCall

        stub = MagicMock()
        stub._recent_actions = {}
        stub._recent_actions_max = 10
        stub._recent_actions_expiry = 3600
        stub._last_tool_use = {}
        stub._system_prompt = "Original prompt"
        stub.config = MagicMock()
        stub.config.tools.enabled = True
        stub.config.tools.tool_timeout_seconds = 300
        stub.config.discord.allowed_users = []
        stub.config.tools.approval_timeout_seconds = 30
        stub.codex_client = MagicMock()
        stub.skill_manager = MagicMock()
        stub.skill_manager.requires_approval = MagicMock(return_value=None)
        stub.skill_manager.has_skill = MagicMock(return_value=False)
        stub.skill_manager.list_skills = MagicMock(return_value=[])
        stub.skill_manager.create_skill = MagicMock(return_value="Skill created.")
        stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
        stub.audit = MagicMock()
        stub.audit.log_execution = AsyncMock()
        stub.tool_executor = MagicMock()
        stub.tool_executor.execute = AsyncMock(return_value="OK")
        stub.tool_executor.set_user_context = MagicMock()
        stub._send_with_retry = AsyncMock()
        stub._send_chunked = AsyncMock()
        stub._merged_tool_definitions = MagicMock(return_value=[{"name": "test"}])
        stub._pending_files = {}
        stub.permissions = MagicMock()
        stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, t: t)

        # _build_system_prompt returns different values on successive calls:
        # First call (initial) returns "Original prompt" (from system_prompt_override)
        # After skill CRUD, returns "Updated prompt with skills"
        stub._build_system_prompt = MagicMock(return_value="Updated prompt with skills")

        # Bind real methods
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)
        stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)
        return stub

    def _make_message(self):
        msg = AsyncMock()
        msg.channel = MagicMock()
        msg.channel.id = "chan-1"
        msg.author = MagicMock()
        msg.author.id = "user-1"
        msg.content = "create a skill called test"
        return msg

    async def test_skill_create_updates_prompt_for_next_iteration(self):
        """After create_skill, the next tool loop iteration uses the updated system_prompt."""
        from src.llm.types import LLMResponse, ToolCall

        stub = self._make_bot_stub()
        msg = self._make_message()

        # Track system kwarg values passed to codex_client.chat_with_tools
        captured_prompts = []

        call_count = 0

        async def mock_chat_with_tools(*args, **kwargs):
            nonlocal call_count
            captured_prompts.append(kwargs.get("system"))
            call_count += 1
            if call_count == 1:
                # Iteration 1: returns create_skill tool call
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(
                        id="tool-create",
                        name="create_skill",
                        input={"name": "my_skill", "code": "def run(): pass"},
                    )],
                    stop_reason="tool_use",
                )
            # Iteration 2: returns final text
            return LLMResponse(text="Skill created successfully!", stop_reason="end_turn")

        stub.codex_client.chat_with_tools = mock_chat_with_tools

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, already_sent, is_error, tools_used, handoff = await stub._process_with_tools(
                msg, [{"role": "user", "content": "create a skill"}],
                system_prompt_override="Original prompt",
            )

        assert len(captured_prompts) == 2
        # First iteration: uses the original prompt
        assert captured_prompts[0] == "Original prompt"
        # Second iteration: uses the UPDATED prompt (from _build_system_prompt)
        assert captured_prompts[1] == "Updated prompt with skills"
        # Verify _build_system_prompt was called (by _run_tool after create_skill)
        stub._build_system_prompt.assert_called()

    async def test_non_skill_tool_does_not_change_prompt(self):
        """Regular tool calls should NOT change the system_prompt between iterations."""
        from src.llm.types import LLMResponse, ToolCall

        stub = self._make_bot_stub()
        msg = self._make_message()

        captured_prompts = []

        call_count = 0

        async def mock_chat_with_tools(*args, **kwargs):
            nonlocal call_count
            captured_prompts.append(kwargs.get("system"))
            call_count += 1
            if call_count == 1:
                # Iteration 1: returns check_disk tool call
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="tool-check", name="check_disk", input={"host": "server"})],
                    stop_reason="tool_use",
                )
            # Iteration 2: returns final text
            return LLMResponse(text="Disk is fine.", stop_reason="end_turn")

        stub.codex_client.chat_with_tools = mock_chat_with_tools

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "check the disk"}],
                system_prompt_override="Original prompt",
            )

        assert len(captured_prompts) == 2
        # Both iterations should use the same original prompt
        assert captured_prompts[0] == "Original prompt"
        assert captured_prompts[1] == "Original prompt"
        # _build_system_prompt should NOT have been called
        stub._build_system_prompt.assert_not_called()


# ---------------------------------------------------------------------------
# Round 3: Codex fallback double-failure in claude_code path
# ---------------------------------------------------------------------------

class TestClaudeCodeCodexDoubleFail:
    """When claude_code fails, budget is exceeded, AND Codex fallback also fails."""

    def _make_bot_stub(self):
        from src.discord.client import LokiBot
        stub = MagicMock()
        stub._recent_actions = {}
        stub._recent_actions_max = 10
        stub._last_tool_use = {}
        stub._system_prompt = "system prompt"
        stub.config = MagicMock()
        stub.config.tools.enabled = True
        stub.config.tools.tool_timeout_seconds = 300
        stub.config.discord.allowed_users = []
        stub.config.tools.approval_timeout_seconds = 30
        stub.sessions = MagicMock()
        stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
        stub.sessions.add_message = MagicMock()
        stub.sessions.remove_last_message = MagicMock(return_value=True)
        stub.sessions.prune = MagicMock()
        stub.sessions.save = MagicMock()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(side_effect=RuntimeError("Codex is down"))
        stub.skill_manager = MagicMock()
        stub.skill_manager.list_skills = MagicMock(return_value=[])
        stub.audit = MagicMock()
        stub.audit.log_execution = AsyncMock()
        stub.tool_executor = MagicMock()
        stub._build_system_prompt = MagicMock(return_value="full prompt")
        stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
        stub._send_with_retry = AsyncMock()
        stub._send_chunked = AsyncMock()
        stub._process_with_tools = AsyncMock(
            return_value=("nope", False, False, [], False)
        )
        stub._merged_tool_definitions = MagicMock(return_value=[])
        stub.permissions = MagicMock()
        stub.permissions.is_guest = MagicMock(return_value=False)
        stub.voice_manager = None
        stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
        stub._pending_files = {}
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)
        return stub

    def _make_message(self):
        msg = AsyncMock()
        msg.channel = MagicMock()
        msg.channel.id = "chan-1"
        msg.author = MagicMock()
        msg.author.id = "user-1"
        msg.reply = AsyncMock()
        return msg

    async def test_error_response_path_codex_also_fails(self):
        """claude_code returns error string + budget exceeded + Codex fails → error message."""
        stub = self._make_bot_stub()
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Claude Code failed: timeout"
        )
        msg = self._make_message()

        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # Codex should have been attempted
        stub.codex_client.chat.assert_awaited()
        # Should have sent the double-failure error message
        stub._send_chunked.assert_awaited()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "both failed" in sent_text

    async def test_exception_path_codex_also_fails(self):
        """claude_code raises exception + budget exceeded + Codex fails → error message."""
        stub = self._make_bot_stub()
        stub.tool_executor._handle_claude_code = AsyncMock(
            side_effect=RuntimeError("SSH timeout")
        )
        msg = self._make_message()

        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # Codex should have been attempted
        stub.codex_client.chat.assert_awaited()
        # Should have sent an error with the original exception info
        stub._send_chunked.assert_awaited()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "SSH timeout" in sent_text

    async def test_error_response_no_codex_budget_exceeded(self):
        """claude_code returns error string + budget exceeded + no Codex → original error sent."""
        stub = self._make_bot_stub()
        stub.codex_client = None  # No Codex available
        stub.tool_executor._handle_claude_code = AsyncMock(
            return_value="Claude Code failed: timeout"
        )
        msg = self._make_message()

        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # Should send the original Claude Code error (no Codex to try)
        stub._send_chunked.assert_awaited()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "Claude Code failed" in sent_text

    async def test_exception_path_no_codex_budget_exceeded(self):
        """claude_code raises exception + budget exceeded + no Codex → exception in message."""
        stub = self._make_bot_stub()
        stub.codex_client = None  # No Codex available
        stub.tool_executor._handle_claude_code = AsyncMock(
            side_effect=RuntimeError("SSH timeout")
        )
        msg = self._make_message()

        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review the code", "chan-1")

        # Should send error with the original exception info
        stub._send_chunked.assert_awaited()
        sent_text = stub._send_chunked.call_args[0][1]
        assert "SSH timeout" in sent_text
