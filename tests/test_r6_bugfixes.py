"""Tests for R6 cross-feature bug fixes.

Bug 1: _last_tools_used was an instance attribute shared across concurrent
channel processing, causing cross-channel contamination of tool memory recordings.
Fix: tools_used is now a local variable in _process_with_tools, returned as the
4th element of the tuple.

Bug 2: parse_time stripped timezone info from returned datetimes, producing naive
ISO strings. The scheduler running in a UTC Docker container would interpret them
as UTC, causing reminders to fire 4-5 hours early/late.
Fix: Use result.isoformat() which preserves the timezone offset.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from zoneinfo import ZoneInfo

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.tools.time_parser import parse_time, _default_tz  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def now() -> datetime:
    """Fixed reference time: Wednesday March 18 2026, 2:30 PM UTC."""
    return datetime(2026, 3, 18, 14, 30, 0, tzinfo=_default_tz)


# ── Bug 1: tools_used is now a local variable ─────────────────────────


class TestToolsUsedLocalVariable:
    """Verify _process_with_tools returns tools_used as 4th tuple element."""

    def _make_bot_stub(self):
        """Minimal LokiBot stub for _process_with_tools."""
        from src.discord.client import LokiBot  # noqa

        stub = SimpleNamespace()
        stub._system_prompt = "test prompt"
        stub.config = MagicMock()
        stub.config.tools = MagicMock(enabled=False)
        stub.permissions = MagicMock()
        stub.permissions.filter_tools = MagicMock(return_value=[])
        stub.permissions.is_guest = MagicMock(return_value=False)
        stub.tool_executor = MagicMock()
        stub.tool_executor.set_user_context = MagicMock()
        stub.tool_executor.execute = AsyncMock(return_value="OK")
        stub.codex_client = MagicMock()
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=LLMResponse(text="Hello!", stop_reason="end_turn")
        )
        stub._send_with_retry = AsyncMock()
        stub._merged_tool_definitions = MagicMock(return_value=[])
        stub.skill_manager = MagicMock()
        stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed

        msg = MagicMock()
        msg.author.id = 100000000000000001
        msg.channel.id = 123

        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)
        return stub, msg

    async def test_returns_5_tuple(self):
        """_process_with_tools returns (text, already_sent, is_error, tools_used, handoff)."""
        stub, msg = self._make_bot_stub()
        result = await stub._process_with_tools(msg, [])
        assert len(result) == 5

    async def test_tools_used_initially_empty(self):
        """When no tools are called, tools_used should be empty."""
        stub, msg = self._make_bot_stub()
        _text, _sent, _err, tools, _handoff = await stub._process_with_tools(msg, [])
        assert tools == []

    async def test_no_instance_attribute(self):
        """_process_with_tools should NOT set self._last_tools_used."""
        stub, msg = self._make_bot_stub()
        await stub._process_with_tools(msg, [])
        assert not hasattr(stub, "_last_tools_used")

    async def test_tools_used_accumulates_across_iterations(self):
        """Tools from multiple iterations should all appear in the returned list."""
        stub, msg = self._make_bot_stub()
        stub.config.tools.enabled = True
        stub.config.tools.tool_timeout_seconds = 300
        stub._merged_tool_definitions = MagicMock(return_value=[{"name": "t"}])

        # First iteration: tool call, second: final text
        call_count = 0

        async def fake_chat_with_tools(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="tool-1", name="check_disk", input={})],
                    stop_reason="tool_use",
                )
            return LLMResponse(text="Done", stop_reason="end_turn")

        stub.codex_client.chat_with_tools = fake_chat_with_tools

        # Mock the tool execution chain
        stub.skill_manager.has_skill = MagicMock(return_value=False)
        stub.tool_executor.execute = AsyncMock(return_value="OK")
        stub.audit = MagicMock()
        stub.audit.log_execution = AsyncMock()
        stub._track_recent_action = MagicMock()

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            _text, _sent, _err, tools, _handoff = await stub._process_with_tools(msg, [])

        assert "check_disk" in tools


# ── Bug 2: parse_time preserves timezone ───────────────────────────────


class TestParseTimeTimezonePreservation:
    """parse_time should return ISO strings WITH timezone offset."""

    def test_relative_time_has_offset(self, now):
        result = parse_time("in 30 minutes", now)
        assert "+00:00" in result

    def test_tomorrow_has_offset(self, now):
        result = parse_time("tomorrow at 9am", now)
        assert "+00:00" in result

    def test_today_has_offset(self, now):
        result = parse_time("today at 5pm", now)
        assert "+00:00" in result

    def test_next_day_has_offset(self, now):
        result = parse_time("next friday", now)
        assert "+00:00" in result

    def test_bare_day_has_offset(self, now):
        result = parse_time("friday at 3pm", now)
        assert "+00:00" in result

    def test_at_time_has_offset(self, now):
        result = parse_time("at 5pm", now)
        assert "+00:00" in result

    def test_bare_time_has_offset(self, now):
        result = parse_time("5pm", now)
        assert "+00:00" in result

    def test_roundtrip_preserves_tz(self, now):
        """Returned ISO string should roundtrip through fromisoformat with tz."""
        result = parse_time("tomorrow at 9am", now)
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None
        assert parsed.hour == 9

    def test_configured_tz_offset(self):
        """When default tz is configured, offset should match that timezone."""
        from zoneinfo import ZoneInfo
        # Use a tz-aware now in America/New_York (winter = EST = -05:00)
        winter_now = datetime(2026, 1, 15, 14, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        result = parse_time("tomorrow at 9am", winter_now)
        assert "-05:00" in result

    def test_no_naive_output(self, now):
        """Output should never be a naive datetime string."""
        result = parse_time("in 1 hour", now)
        # Naive ISO strings end with seconds: 2026-03-18T15:30:00
        # TZ-aware ones continue: 2026-03-18T15:30:00+00:00
        assert len(result) > len("2026-03-18T15:30:00")
