"""Tests verifying bot config backward compatibility and Codex-only routing.

These tests verify:
  - Config loads without an anthropic section (backward compat)
  - Config loads with anthropic.api_key = ""
  - _process_with_tools uses only Codex
  - Task route errors gracefully when no Codex client
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.config.schema import AnthropicConfig, Config, DiscordConfig  # noqa: E402
from src.discord.client import LokiBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestAnthropicConfigOptional:
    """AnthropicConfig.api_key defaults to empty string."""

    def test_api_key_defaults_to_empty(self):
        cfg = AnthropicConfig()
        assert cfg.api_key == ""

    def test_config_without_anthropic_section(self):
        """Config can be created without specifying anthropic at all."""
        cfg = Config(discord=DiscordConfig(token="test-token"))
        assert cfg.anthropic.api_key == ""

    def test_config_with_empty_api_key(self):
        cfg = Config(
            discord=DiscordConfig(token="test-token"),
            anthropic=AnthropicConfig(api_key=""),
        )
        assert cfg.anthropic.api_key == ""

    def test_config_with_api_key_still_works(self):
        """Backward compat: providing an API key still works."""
        cfg = Config(
            discord=DiscordConfig(token="test-token"),
            anthropic=AnthropicConfig(api_key="sk-test-key"),
        )
        assert cfg.anthropic.api_key == "sk-test-key"


# ---------------------------------------------------------------------------
# Bot stub for routing tests
# ---------------------------------------------------------------------------

def _make_bot_stub():
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._last_tool_use = {}
    stub._system_prompt = "You are a bot."
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.config.tools.approval_timeout_seconds = 30
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Hello!")
    stub.codex_client.chat_with_tools = AsyncMock(
        return_value=LLMResponse(text="Done.")
    )
    stub.classifier = MagicMock()
    stub.classifier.classify = AsyncMock(return_value="task")
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.skill_manager = MagicMock()
    stub.skill_manager.requires_approval = MagicMock(return_value=None)
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock()
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="OK")
    stub.tool_executor.set_user_context = MagicMock()
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[{"name": "test"}])
    stub._build_system_prompt = MagicMock(return_value="You are a bot.")
    stub._build_chat_system_prompt = MagicMock(return_value="You are a chat bot.")
    stub._inject_tool_hints = AsyncMock(return_value="You are a bot.")
    stub._check_for_secrets = MagicMock(return_value=False)
    stub._is_allowed_channel = MagicMock(return_value=True)
    stub._is_allowed_user = MagicMock(return_value=True)
    stub.voice_manager = None
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.tool_memory.get_hint = AsyncMock(return_value=None)
    stub._embedder = None
    # Bind real methods
    stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)
    stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)
    stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)
    return stub


def _make_message(content="check disk on server", channel_id="chan-1"):
    msg = AsyncMock()
    msg.content = content
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.typing = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(), __aexit__=AsyncMock()
    ))
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.name = "testuser"
    msg.reply = AsyncMock()
    msg.attachments = []
    return msg


# ---------------------------------------------------------------------------
# Task route without Anthropic
# ---------------------------------------------------------------------------

class TestTaskRouteNoAnthropic:
    """Task route uses only Codex."""

    async def test_task_uses_codex_chat_with_tools(self):
        """Task route calls codex_client.chat_with_tools."""
        stub = _make_bot_stub()
        msg = _make_message()

        with patch("src.discord.client.is_task_by_keyword", return_value=True), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "check disk on server", "user-1")

        stub.codex_client.chat_with_tools.assert_called_once()

    async def test_task_no_codex_returns_error(self):
        """Without Codex, task route returns error immediately."""
        stub = _make_bot_stub()
        stub.codex_client = None
        msg = _make_message()

        with patch("src.discord.client.is_task_by_keyword", return_value=True), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "check disk on server", "user-1")

        stub._send_with_retry.assert_called()
        call_args = stub._send_with_retry.call_args
        assert "No tool backend available" in call_args[0][1]

    async def test_task_codex_failure_returns_error(self):
        """When Codex fails, error is returned."""
        stub = _make_bot_stub()
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=Exception("Codex down")
        )
        msg = _make_message()

        with patch("src.discord.client.is_task_by_keyword", return_value=True), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "check disk on server", "user-1")

        # Should send error message
        stub._send_chunked.assert_called()
        error_text = stub._send_chunked.call_args[0][1]
        assert "Tool execution failed" in error_text or "Something went wrong" in error_text

    async def test_task_route_uses_codex(self):
        """Task route uses Codex tool calling."""
        stub = _make_bot_stub()
        msg = _make_message()

        with patch("src.discord.client.is_task_by_keyword", return_value=True), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.scrub_response_secrets", side_effect=lambda x: x):
            await stub._handle_message_inner(msg, "check disk on server", "user-1")

        stub.codex_client.chat_with_tools.assert_called_once()


class TestProcessWithToolsCodexOnly:
    """_process_with_tools always uses Codex."""

    async def test_simple_text_response(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="All clear.")
        )

        text, already_sent, is_error, tools, handoff = await stub._process_with_tools(
            msg, [],
        )
        assert text == "All clear."
        assert is_error is False

    async def test_tool_loop_with_codex(self):
        """Tool call followed by text response."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                tool_calls=[ToolCall(id="t1", name="check_disk", input={"host": "server"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Disk is 42% full."),
        ])

        with patch("src.discord.client.requires_approval", return_value=False), \
             patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, already_sent, is_error, tools, handoff = await stub._process_with_tools(
                msg, [],
            )

        assert text == "Disk is 42% full."
        assert tools == ["check_disk"]
        assert stub.codex_client.chat_with_tools.call_count == 2
