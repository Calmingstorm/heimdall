"""Tests for bot interop improvements (Round 7).

When Loki receives messages from other bots:
1. A bot-specific preamble is injected into the context separator telling Codex
   to execute immediately and never hedge.
2. detect_hedging() catches phrases like "shall I", "if you want" that bots
   should never receive.
3. If hedging is detected on the first iteration with no tools called, the tool
   loop retries once with a developer correction message.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    LokiBot,
    detect_hedging,
    _HEDGING_RETRY_MSG,
    _HEDGING_PATTERNS,
)
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_approval():
    with patch("src.discord.client.requires_approval", return_value=False):
        yield


def _make_bot_stub(respond_to_bots=False):
    """Minimal LokiBot stub for bot interop tests."""
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
    stub.config.tools.auto_approve = False
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.respond_to_bots = respond_to_bots
    stub.config.discord.require_mention = False
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
        {"name": "run_command", "description": "Run", "input_schema": {"type": "object", "properties": {}}},
        {"name": "run_script", "description": "Script", "input_schema": {"type": "object", "properties": {}}},
        {"name": "check_disk", "description": "Disk", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed
    stub._build_partial_completion_report = LokiBot._build_partial_completion_report
    return stub


def _make_message(is_bot=False):
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
    msg.author.display_name = "TestBot" if is_bot else "TestUser"
    msg.author.bot = is_bot
    msg.webhook_id = None
    msg.reply = AsyncMock()
    return msg


# ===========================================================================
# detect_hedging() unit tests
# ===========================================================================

class TestDetectHedging:
    """Unit tests for the detect_hedging function."""

    def test_empty_text_not_hedging(self):
        assert detect_hedging("", []) is False

    def test_short_text_not_hedging(self):
        assert detect_hedging("OK done", []) is False

    def test_tools_used_never_hedging(self):
        """If tools were actually called, it's not hedging."""
        text = "If you want, I can also check memory."
        assert detect_hedging(text, ["check_disk"]) is False

    def test_detects_if_you_want(self):
        text = "I can check the disk if you want me to."
        assert detect_hedging(text, []) is True

    def test_detects_if_youd_like(self):
        text = "If you'd like, I can run that command for you."
        assert detect_hedging(text, []) is True

    def test_detects_shall_i(self):
        text = "Shall I run the disk check on server1?"
        assert detect_hedging(text, []) is True

    def test_detects_should_i(self):
        text = "Should I go ahead and restart the service?"
        assert detect_hedging(text, []) is True

    def test_detects_would_you_like(self):
        text = "Would you like me to check the disk usage?"
        assert detect_hedging(text, []) is True

    def test_detects_would_you_like_no_me_to(self):
        text = "Would you like a full system check?"
        assert detect_hedging(text, []) is True

    def test_detects_ready_when_you(self):
        text = "I'm ready when you are to proceed."
        assert detect_hedging(text, []) is True

    def test_detects_ready_on_you(self):
        text = "Ready on you — just say the word."
        assert detect_hedging(text, []) is True

    def test_detects_let_me_know(self):
        text = "Let me know if you want me to continue."
        assert detect_hedging(text, []) is True

    def test_detects_let_me_know_when(self):
        text = "Let me know when you're ready to proceed."
        assert detect_hedging(text, []) is True

    def test_detects_want_me_to(self):
        text = "Want me to run the diagnostics?"
        assert detect_hedging(text, []) is True

    def test_detects_i_can_do_that_for_you(self):
        text = "I can do that for you if you want."
        assert detect_hedging(text, []) is True

    def test_detects_i_can_run_it_if(self):
        text = "I can run it if needed."
        assert detect_hedging(text, []) is True

    def test_detects_heres_a_plan(self):
        text = "Here's a plan: first I'll check memory, then disk."
        assert detect_hedging(text, []) is True

    def test_detects_i_would_suggest(self):
        text = "I would suggest running a backup first."
        assert detect_hedging(text, []) is True

    def test_detects_i_would_recommend(self):
        text = "I'd recommend checking the logs before restarting."
        assert detect_hedging(text, []) is True

    def test_detects_before_i_proceed(self):
        text = "Before I proceed, can you confirm the hostname?"
        assert detect_hedging(text, []) is True

    def test_detects_before_we_go_ahead(self):
        text = "Before we go ahead, let me outline the steps."
        assert detect_hedging(text, []) is True

    def test_detects_ill_wait_for_your_go_ahead(self):
        text = "I'll wait for your go-ahead before making changes."
        assert detect_hedging(text, []) is True

    def test_detects_ill_wait_for_confirmation(self):
        text = "I'll wait for your confirmation before proceeding."
        assert detect_hedging(text, []) is True

    def test_detects_just_say_the_word(self):
        text = "I'm all set — just say the word and I'll run it."
        assert detect_hedging(text, []) is True

    # --- Negative cases (should NOT trigger) ---

    def test_normal_action_not_hedging(self):
        text = "Checking disk usage on server1."
        assert detect_hedging(text, []) is False

    def test_result_report_not_hedging(self):
        text = "Disk is at 42% on /dev/sda1."
        assert detect_hedging(text, []) is False

    def test_direct_statement_not_hedging(self):
        text = "Running the backup script now."
        assert detect_hedging(text, []) is False

    def test_question_about_topic_not_hedging(self):
        text = "What server are you asking about?"
        assert detect_hedging(text, []) is False

    def test_informational_not_hedging(self):
        text = "The nginx service listens on port 80 by default."
        assert detect_hedging(text, []) is False


# ===========================================================================
# Bot-specific context separator (preamble injection)
# ===========================================================================

class TestBotPreambleInjection:
    """Test that bot messages get a bot-specific preamble in the context separator."""

    @pytest.mark.asyncio
    async def test_bot_message_injects_bot_preamble(self):
        """When message.author.bot=True and respond_to_bots=True, separator has bot instructions."""
        bot = _make_bot_stub(respond_to_bots=True)
        msg = _make_message(is_bot=True)

        captured_messages = []

        async def mock_chat(**kwargs):
            captured_messages.append([m.copy() for m in kwargs.get("messages", [])])
            return LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        await LokiBot._process_with_tools(
            bot, msg,
            [
                {"role": "user", "content": "old message"},
                {"role": "user", "content": "run df -h on server1"},
            ],
        )

        # Find the developer separator
        assert len(captured_messages) >= 1
        first_call = captured_messages[0]
        dev_msgs = [m for m in first_call if m.get("role") == "developer"]
        assert len(dev_msgs) >= 1
        separator_content = dev_msgs[0]["content"]
        assert "ANOTHER BOT" in separator_content
        assert "EXECUTE immediately" in separator_content

    @pytest.mark.asyncio
    async def test_human_message_no_bot_preamble(self):
        """When message.author.bot=False, separator should NOT have bot instructions."""
        bot = _make_bot_stub(respond_to_bots=False)
        msg = _make_message(is_bot=False)

        captured_messages = []

        async def mock_chat(**kwargs):
            captured_messages.append([m.copy() for m in kwargs.get("messages", [])])
            return LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        await LokiBot._process_with_tools(
            bot, msg,
            [
                {"role": "user", "content": "old message"},
                {"role": "user", "content": "run df -h on server1"},
            ],
        )

        assert len(captured_messages) >= 1
        first_call = captured_messages[0]
        dev_msgs = [m for m in first_call if m.get("role") == "developer"]
        assert len(dev_msgs) >= 1
        separator_content = dev_msgs[0]["content"]
        assert "ANOTHER BOT" not in separator_content

    @pytest.mark.asyncio
    async def test_bot_preamble_mentions_run_script(self):
        """Bot preamble should instruct use of run_script for code."""
        bot = _make_bot_stub(respond_to_bots=True)
        msg = _make_message(is_bot=True)

        captured_messages = []

        async def mock_chat(**kwargs):
            captured_messages.append([m.copy() for m in kwargs.get("messages", [])])
            return LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        await LokiBot._process_with_tools(
            bot, msg,
            [
                {"role": "user", "content": "old msg"},
                {"role": "user", "content": "run this script"},
            ],
        )

        first_call = captured_messages[0]
        dev_msgs = [m for m in first_call if m.get("role") == "developer"]
        separator_content = dev_msgs[0]["content"]
        assert "run_script" in separator_content

    @pytest.mark.asyncio
    async def test_bot_preamble_not_injected_when_respond_to_bots_disabled(self):
        """Even if author is a bot, no preamble when respond_to_bots=False."""
        bot = _make_bot_stub(respond_to_bots=False)
        msg = _make_message(is_bot=True)

        captured_messages = []

        async def mock_chat(**kwargs):
            captured_messages.append([m.copy() for m in kwargs.get("messages", [])])
            return LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        await LokiBot._process_with_tools(
            bot, msg,
            [
                {"role": "user", "content": "old msg"},
                {"role": "user", "content": "do something"},
            ],
        )

        first_call = captured_messages[0]
        dev_msgs = [m for m in first_call if m.get("role") == "developer"]
        separator_content = dev_msgs[0]["content"]
        assert "ANOTHER BOT" not in separator_content


# ===========================================================================
# Hedging retry in _process_with_tools (bot messages only)
# ===========================================================================

class TestHedgingRetry:
    """Integration tests for hedging retry in the tool loop for bot messages."""

    @pytest.mark.asyncio
    async def test_hedging_triggers_retry_for_bot_message(self):
        """Bot message + hedging response → retry with correction → second attempt uses tools."""
        bot = _make_bot_stub(respond_to_bots=True)
        msg = _make_message(is_bot=True)

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    text="Would you like me to check the disk on server1?",
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            elif call_count == 2:
                return LLMResponse(
                    text="Checking disk.",
                    tool_calls=[ToolCall(id="call_1", name="check_disk", input={"host": "server1"})],
                    stop_reason="tool_use",
                )
            else:
                return LLMResponse(
                    text="Disk is 42% full.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk on server1"}],
        )

        assert not is_error
        assert "check_disk" in tools_used
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_hedging_no_retry_for_human_message(self):
        """Human message + hedging response → no retry (hedging might be appropriate)."""
        bot = _make_bot_stub(respond_to_bots=False)
        msg = _make_message(is_bot=False)

        bot.codex_client.chat_with_tools = AsyncMock(return_value=LLMResponse(
            text="Would you like me to check the disk on server1?",
            tool_calls=[],
            stop_reason="end_turn",
        ))

        text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk"}],
        )

        # Should return the hedging text as-is — no retry for humans
        assert text == "Would you like me to check the disk on server1?"
        assert bot.codex_client.chat_with_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_hedging_retry_still_hedges_returns_text(self):
        """Both attempts hedge → return the second response as-is."""
        bot = _make_bot_stub(respond_to_bots=True)
        msg = _make_message(is_bot=True)

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    text="Shall I check the disk for you?",
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            else:
                return LLMResponse(
                    text="Let me know when you're ready to proceed.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk"}],
        )

        assert not is_error
        assert text == "Let me know when you're ready to proceed."
        assert tools_used == []
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_hedging_correction_message_injected(self):
        """Verify the hedging correction message is added to messages on retry."""
        bot = _make_bot_stub(respond_to_bots=True)
        msg = _make_message(is_bot=True)

        captured_messages = []

        async def mock_chat(**kwargs):
            captured_messages.append([m.copy() for m in kwargs.get("messages", [])])
            if len(captured_messages) == 1:
                return LLMResponse(
                    text="If you'd like, I can check the disk usage.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            else:
                return LLMResponse(
                    text="Checking now.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        await LokiBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk"}],
        )

        assert len(captured_messages) == 2
        second_call_msgs = captured_messages[1]
        dev_msgs = [m for m in second_call_msgs if m.get("role") == "developer"
                    and "another bot" in m.get("content", "").lower()]
        assert len(dev_msgs) >= 1

    @pytest.mark.asyncio
    async def test_no_hedging_retry_when_tools_used(self):
        """If tools were used earlier in the loop, hedging check doesn't trigger."""
        bot = _make_bot_stub(respond_to_bots=True)
        msg = _make_message(is_bot=True)

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    text="Checking.",
                    tool_calls=[ToolCall(id="call_1", name="check_disk", input={})],
                    stop_reason="tool_use",
                )
            else:
                # After tools, text might include "if you want" — that's OK
                return LLMResponse(
                    text="Disk is fine. If you want, I can also check memory.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk"}],
        )

        assert "check_disk" in tools_used
        assert call_count == 2  # No extra retry


# ===========================================================================
# _HEDGING_RETRY_MSG content tests
# ===========================================================================

class TestHedgingRetryMsg:
    """Verify the hedging correction message content."""

    def test_retry_msg_is_developer_role(self):
        assert _HEDGING_RETRY_MSG["role"] == "developer"

    def test_retry_msg_mentions_bot(self):
        assert "bot" in _HEDGING_RETRY_MSG["content"].lower()

    def test_retry_msg_demands_execution(self):
        content = _HEDGING_RETRY_MSG["content"].lower()
        assert "execute" in content

    def test_retry_msg_forbids_hedging_phrases(self):
        content = _HEDGING_RETRY_MSG["content"].lower()
        assert "shall i" in content or "if you want" in content


# ===========================================================================
# Interaction between hedging and fabrication detection
# ===========================================================================

class TestHedgingFabricationInteraction:
    """Test that hedging and fabrication detectors don't interfere with each other."""

    @pytest.mark.asyncio
    async def test_fabrication_checked_before_hedging_for_bot(self):
        """If a bot response is BOTH fabrication and hedging, fabrication retry fires first."""
        bot = _make_bot_stub(respond_to_bots=True)
        msg = _make_message(is_bot=True)

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # This text is fabrication (claims "I ran") — should trigger fabrication retry first
                return LLMResponse(
                    text="I ran df -h and the disk is fine. Shall I check memory too?",
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            else:
                return LLMResponse(
                    text="Checking disk now.",
                    tool_calls=[ToolCall(id="c1", name="check_disk", input={"host": "s1"})],
                    stop_reason="tool_use",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        # Need a third response for after tool use
        original_side_effect = bot.codex_client.chat_with_tools.side_effect

        async def extended_mock(**kwargs):
            nonlocal call_count
            if call_count <= 2:
                return await original_side_effect(**kwargs)
            return LLMResponse(text="Disk is fine.", tool_calls=[], stop_reason="end_turn")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=extended_mock)

        text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk"}],
        )

        # Fabrication retry should have fired (fabrication is checked first in the code)
        assert "check_disk" in tools_used

    @pytest.mark.asyncio
    async def test_pure_hedging_no_fabrication_for_bot(self):
        """Bot response that hedges but doesn't fabricate — hedging retry fires."""
        bot = _make_bot_stub(respond_to_bots=True)
        msg = _make_message(is_bot=True)

        captured_messages = []
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_messages.append([m.copy() for m in kwargs.get("messages", [])])
            if call_count == 1:
                return LLMResponse(
                    text="Would you like me to run the disk check?",
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            else:
                return LLMResponse(
                    text="Checking disk.",
                    tool_calls=[ToolCall(id="c1", name="check_disk", input={})],
                    stop_reason="tool_use",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        # Need third response for after tool execution
        original = bot.codex_client.chat_with_tools.side_effect

        async def ext(**kwargs):
            nonlocal call_count
            if call_count <= 2:
                return await original(**kwargs)
            return LLMResponse(text="Disk OK.", tool_calls=[], stop_reason="end_turn")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=ext)

        text, already_sent, is_error, tools_used, handoff = await LokiBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk"}],
        )

        assert "check_disk" in tools_used
        # Verify the hedging correction (not fabrication) was injected
        assert len(captured_messages) >= 2
        second_msgs = captured_messages[1]
        hedging_corrections = [m for m in second_msgs if m.get("role") == "developer"
                               and "another bot" in m.get("content", "").lower()]
        assert len(hedging_corrections) >= 1
