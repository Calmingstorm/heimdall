"""Tests for fabrication detection and retry logic (Round 5).

When Codex returns text claiming to have run commands or checked systems
without actually calling any tools, that's fabrication. The detect_fabrication()
function identifies these patterns, and the tool loop retries once with a
developer correction message.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    HeimdallBot,
    detect_fabrication,
    _FABRICATION_RETRY_MSG,
)
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bot_stub():
    """Minimal HeimdallBot stub for fabrication tests."""
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
        {"name": "run_command", "description": "Run", "input_schema": {"type": "object", "properties": {}}},
        {"name": "check_disk", "description": "Disk", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    return stub


def _make_message():
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
    msg.webhook_id = None
    msg.reply = AsyncMock()
    return msg


# ===========================================================================
# detect_fabrication() unit tests
# ===========================================================================

class TestDetectFabrication:
    """Unit tests for the detect_fabrication function."""

    def test_empty_text_not_fabrication(self):
        assert detect_fabrication("", []) is False

    def test_short_text_not_fabrication(self):
        assert detect_fabrication("OK", []) is False

    def test_tools_used_never_fabrication(self):
        """If tools were actually called, it's not fabrication."""
        text = "I ran df -h and here's the output:"
        assert detect_fabrication(text, ["run_command"]) is False

    def test_detects_i_ran_pattern(self):
        text = "I ran `df -h` on the server and the disk is 42% full."
        assert detect_fabrication(text, []) is True

    def test_detects_i_executed_pattern(self):
        text = "I executed the command and everything looks good."
        assert detect_fabrication(text, []) is True

    def test_detects_i_checked_pattern(self):
        text = "I checked the server and all services are running normally."
        assert detect_fabrication(text, []) is True

    def test_detects_heres_the_output(self):
        text = "Here's the output from the disk check:\n```\n/dev/sda1 50G 20G 30G\n```"
        assert detect_fabrication(text, []) is True

    def test_detects_here_is_the_result(self):
        text = "Here is the result of the command:\nEverything is fine."
        assert detect_fabrication(text, []) is True

    def test_detects_command_returned(self):
        text = "The command returned exit code 0 with no errors."
        assert detect_fabrication(text, []) is True

    def test_detects_output_shows(self):
        text = "The output shows that nginx is running on port 80."
        assert detect_fabrication(text, []) is True

    def test_detects_i_can_see(self):
        text = "I can see that the service is running correctly."
        assert detect_fabrication(text, []) is True

    def test_detects_i_found(self):
        text = "I found that the configuration file has the correct settings."
        assert detect_fabrication(text, []) is True

    def test_detects_fake_terminal_output(self):
        text = "Here are the results:\n```bash\n$ df -h\nFilesystem      Size  Used Avail\n```"
        assert detect_fabrication(text, []) is True

    def test_detects_fake_docker_output(self):
        text = "Container status:\n```text\nCONTAINER ID   IMAGE\n```"
        assert detect_fabrication(text, []) is True

    def test_normal_chat_not_fabrication(self):
        """Normal conversational responses should not be flagged."""
        text = "Sure, I can help you with that. Let me check the disk usage."
        assert detect_fabrication(text, []) is False

    def test_question_not_fabrication(self):
        text = "Which server would you like me to check the disk usage on?"
        assert detect_fabrication(text, []) is False

    def test_plan_statement_not_fabrication(self):
        """A plan statement without claims of execution is not fabrication."""
        text = "To check the disk usage, I need to connect to the server first."
        assert detect_fabrication(text, []) is False

    def test_code_example_not_fabrication(self):
        """Code examples in conversation shouldn't be flagged as fake output."""
        text = "You can use this command:\n```\ndf -h\n```"
        assert detect_fabrication(text, []) is False

    def test_i_performed_pattern(self):
        text = "I performed a health check and everything is stable."
        assert detect_fabrication(text, []) is True

    def test_running_pattern(self):
        text = "After running the diagnostics, all systems appear healthy."
        assert detect_fabrication(text, []) is True


# ===========================================================================
# Integration: Retry on fabrication in _process_with_tools
# ===========================================================================

class TestFabricationRetry:
    """Integration tests for fabrication retry in the tool loop."""

    @pytest.mark.asyncio
    async def test_fabrication_triggers_retry(self):
        """First response fabricates → retry with correction → second response uses tools."""
        bot = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: fabrication (text claiming results, no tools)
                return LLMResponse(
                    text="I ran df -h and the disk is 42% full.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            elif call_count == 2:
                # Second call (after correction): uses tools
                return LLMResponse(
                    text="Checking disk.",
                    tool_calls=[ToolCall(id="call_1", name="check_disk", input={"host": "server1"})],
                    stop_reason="tool_use",
                )
            else:
                # Final: return text result
                return LLMResponse(
                    text="Disk is 42% full.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        text, already_sent, is_error, tools_used, handoff = await HeimdallBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk on server1"}],
        )

        assert not is_error
        assert "check_disk" in tools_used
        # Should have called chat 3 times: fabrication, retry→tools, final
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fabrication_retry_still_fabricates_returns_text(self):
        """Both attempts fabricate → return the second response as-is."""
        bot = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    text="I checked the server and nginx is running.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            else:
                # Second attempt also text-only (but this time iteration > 0)
                return LLMResponse(
                    text="The server appears to be running fine.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        text, already_sent, is_error, tools_used, handoff = await HeimdallBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check nginx"}],
        )

        # Should return text (not loop forever), no error flag
        assert not is_error
        assert text == "The server appears to be running fine."
        assert tools_used == []
        # Only 2 calls: initial fabrication + retry
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_when_no_fabrication(self):
        """Legitimate text-only response should not trigger retry."""
        bot = _make_bot_stub()
        msg = _make_message()

        bot.codex_client.chat_with_tools = AsyncMock(return_value=LLMResponse(
            text="Sure, which server would you like me to check?",
            tool_calls=[],
            stop_reason="end_turn",
        ))

        text, already_sent, is_error, tools_used, handoff = await HeimdallBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk"}],
        )

        assert text == "Sure, which server would you like me to check?"
        assert not is_error
        # Only 1 call — no retry triggered (hedging only fires for bot messages)
        assert bot.codex_client.chat_with_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_when_tools_already_used(self):
        """If tools were used in the loop, no fabrication retry needed."""
        bot = _make_bot_stub()
        msg = _make_message()

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
                # After tools, even if text mentions "I ran" it's not fabrication
                return LLMResponse(
                    text="I ran the check and disk is fine.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        text, already_sent, is_error, tools_used, handoff = await HeimdallBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk"}],
        )

        assert "check_disk" in tools_used
        assert call_count == 2  # No extra retry

    @pytest.mark.asyncio
    async def test_correction_message_injected_on_retry(self):
        """Verify the developer correction message is added to messages on retry."""
        bot = _make_bot_stub()
        msg = _make_message()

        captured_messages = []

        async def mock_chat(**kwargs):
            captured_messages.append([m.copy() for m in kwargs.get("messages", [])])
            if len(captured_messages) == 1:
                return LLMResponse(
                    text="I executed the command and here's the result.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            else:
                return LLMResponse(
                    text="Let me actually run that.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        await HeimdallBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "run df -h"}],
        )

        # Second call should have the correction message
        assert len(captured_messages) == 2
        second_call_msgs = captured_messages[1]
        # Find the developer correction message
        dev_msgs = [m for m in second_call_msgs if m.get("role") == "developer"
                    and "fabrication" in m.get("content", "").lower()]
        assert len(dev_msgs) >= 1


class TestFabricationRetryMsg:
    """Verify the correction message content."""

    def test_retry_msg_is_developer_role(self):
        assert _FABRICATION_RETRY_MSG["role"] == "developer"

    def test_retry_msg_mentions_fabrication(self):
        assert "fabrication" in _FABRICATION_RETRY_MSG["content"].lower()

    def test_retry_msg_demands_tool_use(self):
        content = _FABRICATION_RETRY_MSG["content"].lower()
        assert "call" in content and "tool" in content


class TestPrematureFailureDetection:
    """Tests for detect_premature_failure — catches giving up too early."""

    def test_no_tools_returns_false(self):
        from src.discord.client import detect_premature_failure
        assert detect_premature_failure("Error: couldn't resolve", []) is False

    def test_tools_used_with_failure_returns_true(self):
        from src.discord.client import detect_premature_failure
        assert detect_premature_failure(
            "Couldn't get a live Jita quote because ESI name resolution failed",
            ["eve_jita_price_lookup"]
        ) is True

    def test_tools_used_with_success_returns_false(self):
        from src.discord.client import detect_premature_failure
        assert detect_premature_failure(
            "The price of PLEX is 4.2B ISK",
            ["eve_jita_price_lookup"]
        ) is False

    def test_blocked_pattern(self):
        from src.discord.client import detect_premature_failure
        assert detect_premature_failure(
            "Current blocker is upstream name resolution, currently unavailable",
            ["run_script"]
        ) is True

    def test_workaround_pattern(self):
        from src.discord.client import detect_premature_failure
        assert detect_premature_failure(
            "Use this workaround instead to get the data",
            ["fetch_url"]
        ) is True

    def test_no_results_pattern(self):
        from src.discord.client import detect_premature_failure
        assert detect_premature_failure(
            "No results found for that search term",
            ["web_search"]
        ) is True

    def test_short_text_returns_false(self):
        from src.discord.client import detect_premature_failure
        assert detect_premature_failure("Error", ["run_command"]) is False

    def test_unable_to_pattern(self):
        from src.discord.client import detect_premature_failure
        assert detect_premature_failure(
            "Unable to resolve the type ID from ESI search",
            ["run_script", "fetch_url"]
        ) is True

    def test_error_colon_pattern(self):
        from src.discord.client import detect_premature_failure
        assert detect_premature_failure(
            "Error: Could not resolve type_id for Shadow Serpentis Gyrostabilizer via ESI",
            ["eve_jita_price_lookup"]
        ) is True

    def test_successful_with_error_word_in_context(self):
        from src.discord.client import detect_premature_failure
        # "error" appears but as part of a successful error handling description
        assert detect_premature_failure(
            "Fixed the error in the script and reran it. Output: 42",
            ["run_script", "run_command"]
        ) is False  # "Fixed the error" doesn't match failure patterns
