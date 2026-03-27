"""Tests for Round 1-5 routing and prompt fixes.

Covers:
- P3: nonlocal system_prompt in _run_tool (skill CRUD updates prompt)
- P3: Compaction routed through compaction_fn (Codex)
- Round 3: nonlocal system_prompt behavioral test (prompt updates across iterations)
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports trigger __init__
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402
from src.sessions.manager import (  # noqa: E402
    COMPACTION_THRESHOLD,
    SessionManager,
)


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
        from src.discord.client import HeimdallBot
        from src.llm.types import LLMResponse, ToolCall

        stub = MagicMock()
        stub._recent_actions = {}
        stub._recent_actions_max = 10
        stub._recent_actions_expiry = 3600
        stub._system_prompt = "Original prompt"
        stub.config = MagicMock()
        stub.config.tools.enabled = True
        stub.config.tools.tool_timeout_seconds = 300
        stub.config.discord.allowed_users = []
        stub.config.discord.respond_to_bots = False
        stub.config.discord.require_mention = False
        stub.codex_client = MagicMock()
        stub.skill_manager = MagicMock()
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
        stub._process_with_tools = HeimdallBot._process_with_tools.__get__(stub)
        stub._track_recent_action = HeimdallBot._track_recent_action.__get__(stub)
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

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
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

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
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
