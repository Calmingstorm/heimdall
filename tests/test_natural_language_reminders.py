"""Tests for natural language reminders: time parser, tool definition, system prompt, handler."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.tools.time_parser import parse_time, _parse_time_of_day, _next_weekday, _default_tz, set_default_timezone
from src.tools.registry import TOOLS

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def now() -> datetime:
    """Fixed reference time: Wednesday March 18 2026, 2:30 PM UTC."""
    return datetime(2026, 3, 18, 14, 30, 0, tzinfo=_default_tz)


# ── _parse_time_of_day ───────────────────────────────────────────────


class TestParseTimeOfDay:
    def test_9am(self):
        assert _parse_time_of_day("9am") == (9, 0)

    def test_9pm(self):
        assert _parse_time_of_day("9pm") == (21, 0)

    def test_12am(self):
        assert _parse_time_of_day("12am") == (0, 0)

    def test_12pm(self):
        assert _parse_time_of_day("12pm") == (12, 0)

    def test_930am(self):
        assert _parse_time_of_day("9:30am") == (9, 30)

    def test_330pm(self):
        assert _parse_time_of_day("3:30pm") == (15, 30)

    def test_space_before_ampm(self):
        assert _parse_time_of_day("9 am") == (9, 0)

    def test_24hour(self):
        assert _parse_time_of_day("17:00") == (17, 0)

    def test_24hour_morning(self):
        assert _parse_time_of_day("09:30") == (9, 30)

    def test_bare_number_returns_none(self):
        assert _parse_time_of_day("9") is None

    def test_nonsense_returns_none(self):
        assert _parse_time_of_day("xyz") is None


# ── _next_weekday ────────────────────────────────────────────────────


class TestNextWeekday:
    def test_next_friday_from_wednesday(self, now):
        result = _next_weekday(now, 4)  # Friday
        assert result.weekday() == 4
        assert result.day == 20  # March 20

    def test_next_monday_from_wednesday(self, now):
        result = _next_weekday(now, 0)  # Monday
        assert result.weekday() == 0
        assert result.day == 23  # March 23

    def test_same_day_goes_to_next_week(self, now):
        result = _next_weekday(now, 2)  # Wednesday (same as now)
        assert result.weekday() == 2
        assert result.day == 25  # next Wednesday

    def test_next_day(self, now):
        result = _next_weekday(now, 3)  # Thursday
        assert result.weekday() == 3
        assert result.day == 19


# ── parse_time: relative expressions ────────────────────────────────


class TestParseTimeRelative:
    def test_in_30_minutes(self, now):
        result = parse_time("in 30 minutes", now)
        assert result == "2026-03-18T15:00:00+00:00"

    def test_in_2_hours(self, now):
        result = parse_time("in 2 hours", now)
        assert result == "2026-03-18T16:30:00+00:00"

    def test_in_1_day(self, now):
        result = parse_time("in 1 day", now)
        assert result == "2026-03-19T14:30:00+00:00"

    def test_in_1_week(self, now):
        result = parse_time("in 1 week", now)
        assert result == "2026-03-25T14:30:00+00:00"

    def test_in_5_mins(self, now):
        result = parse_time("in 5 mins", now)
        assert result == "2026-03-18T14:35:00+00:00"

    def test_in_3_hrs(self, now):
        result = parse_time("in 3 hrs", now)
        assert result == "2026-03-18T17:30:00+00:00"

    def test_unknown_unit_raises(self, now):
        with pytest.raises(ValueError, match="Unknown time unit"):
            parse_time("in 5 fortnights", now)


# ── parse_time: "tomorrow" ──────────────────────────────────────────


class TestParseTimeTomorrow:
    def test_tomorrow_at_9am(self, now):
        result = parse_time("tomorrow at 9am", now)
        assert result == "2026-03-19T09:00:00+00:00"

    def test_tomorrow_at_3_30pm(self, now):
        result = parse_time("tomorrow at 3:30pm", now)
        assert result == "2026-03-19T15:30:00+00:00"

    def test_tomorrow_no_time_defaults_9am(self, now):
        result = parse_time("tomorrow", now)
        assert result == "2026-03-19T09:00:00+00:00"

    def test_tomorrow_without_at(self, now):
        result = parse_time("tomorrow 9am", now)
        assert result == "2026-03-19T09:00:00+00:00"


# ── parse_time: "today" ─────────────────────────────────────────────


class TestParseTimeToday:
    def test_today_at_5pm(self, now):
        result = parse_time("today at 5pm", now)
        assert result == "2026-03-18T17:00:00+00:00"

    def test_today_no_time_raises(self, now):
        with pytest.raises(ValueError, match="requires a time"):
            parse_time("today", now)


# ── parse_time: "next DAY" ──────────────────────────────────────────


class TestParseTimeNextDay:
    def test_next_monday_at_3pm(self, now):
        result = parse_time("next monday at 3pm", now)
        assert result == "2026-03-23T15:00:00+00:00"

    def test_next_friday(self, now):
        result = parse_time("next friday", now)
        assert result == "2026-03-20T09:00:00+00:00"  # defaults 9am

    def test_next_sunday_at_10am(self, now):
        result = parse_time("next sunday at 10am", now)
        assert result == "2026-03-22T10:00:00+00:00"

    def test_abbreviated_day(self, now):
        result = parse_time("next fri at 2pm", now)
        assert result == "2026-03-20T14:00:00+00:00"


# ── parse_time: bare "DAY [at TIME]" ────────────────────────────────


class TestParseTimeBareDay:
    def test_friday_at_3pm(self, now):
        result = parse_time("friday at 3pm", now)
        assert result == "2026-03-20T15:00:00+00:00"

    def test_monday_no_time(self, now):
        result = parse_time("monday", now)
        assert result == "2026-03-23T09:00:00+00:00"


# ── parse_time: "at TIME" ───────────────────────────────────────────


class TestParseTimeAtTime:
    def test_at_5pm_future(self, now):
        result = parse_time("at 5pm", now)
        assert result == "2026-03-18T17:00:00+00:00"

    def test_at_9am_past_wraps_to_tomorrow(self, now):
        # 9am is before 2:30pm, so wraps to tomorrow
        result = parse_time("at 9am", now)
        assert result == "2026-03-19T09:00:00+00:00"

    def test_at_1700(self, now):
        result = parse_time("at 17:00", now)
        assert result == "2026-03-18T17:00:00+00:00"


# ── parse_time: bare time ───────────────────────────────────────────


class TestParseTimeBare:
    def test_5pm(self, now):
        result = parse_time("5pm", now)
        assert result == "2026-03-18T17:00:00+00:00"

    def test_9am_wraps(self, now):
        result = parse_time("9am", now)
        assert result == "2026-03-19T09:00:00+00:00"


# ── parse_time: edge cases ──────────────────────────────────────────


class TestParseTimeEdgeCases:
    def test_whitespace_stripped(self, now):
        result = parse_time("  in 30 minutes  ", now)
        assert result == "2026-03-18T15:00:00+00:00"

    def test_case_insensitive(self, now):
        result = parse_time("Tomorrow At 9AM", now)
        assert result == "2026-03-19T09:00:00+00:00"

    def test_unparseable_raises(self, now):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_time("when pigs fly", now)

    def test_empty_raises(self, now):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_time("", now)

    def test_no_now_uses_current_time(self):
        # Should not raise — just verify it returns something valid
        result = parse_time("in 30 minutes")
        assert "T" in result  # ISO format

    def test_naive_now_gets_tz(self):
        naive = datetime(2026, 3, 18, 14, 30, 0)
        result = parse_time("in 1 hour", naive)
        assert result == "2026-03-18T15:30:00+00:00"


# ── Tool registry ───────────────────────────────────────────────────


class TestParseTimeRegistry:
    def _tool(self):
        return next(t for t in TOOLS if t["name"] == "parse_time")

    def test_tool_exists(self):
        assert self._tool()

    def test_expression_required(self):
        schema = self._tool()["input_schema"]
        assert "expression" in schema["properties"]
        assert "expression" in schema["required"]

    def test_description_mentions_natural_language(self):
        assert "natural language" in self._tool()["description"].lower()


# ── Permission tier ─────────────────────────────────────────────────


class TestParseTimePermissions:
    def test_in_user_tier(self):
        from src.permissions.manager import USER_TIER_TOOLS

        assert "parse_time" in USER_TIER_TOOLS


# ── schedule_task tool description ───────────────────────────────────


class TestScheduleTaskDescription:
    def _tool(self):
        return next(t for t in TOOLS if t["name"] == "schedule_task")

    def test_mentions_natural_language(self):
        desc = self._tool()["description"].lower()
        assert "natural language" in desc

    def test_mentions_parse_time(self):
        desc = self._tool()["description"].lower()
        assert "parse_time" in desc

    def test_run_at_mentions_natural_language(self):
        run_at = self._tool()["input_schema"]["properties"]["run_at"]
        assert "natural language" in run_at["description"].lower()


# ── System prompt ────────────────────────────────────────────────────


class TestSystemPromptReminders:
    def test_context_has_reminders_section(self):
        from pathlib import Path

        arch = (Path(__file__).parent.parent / "data" / "context" / "architecture.md").read_text()
        assert "Reminders and Scheduling" in arch

    def test_context_has_time_examples(self):
        from pathlib import Path

        arch = (Path(__file__).parent.parent / "data" / "context" / "architecture.md").read_text()
        assert "parse_time" in arch

    def test_full_prompt_has_current_datetime(self):
        from src.llm.system_prompt import build_system_prompt

        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[]
        )
        assert "Current Date and Time" in prompt

    def test_chat_prompt_unchanged(self):
        from src.llm.system_prompt import build_chat_system_prompt

        prompt = build_chat_system_prompt()
        # Chat prompt should NOT have scheduling section (no tool access)
        assert "Reminders and Scheduling" not in prompt


# ── Client handler ───────────────────────────────────────────────────


class TestHandleParseTime:
    """Test the handler logic (replicated here to avoid LokiBot import chain)."""

    @staticmethod
    def _handle(inp: dict) -> str:
        expression = inp.get("expression", "")
        if not expression:
            return "Error: 'expression' is required (e.g. 'in 2 hours', 'tomorrow at 9am')"
        try:
            result = parse_time(expression)
            return f"Parsed '{expression}' \u2192 {result}"
        except ValueError as e:
            return f"Error: {e}"

    def test_valid_expression(self):
        result = self._handle({"expression": "in 30 minutes"})
        assert "in 30 minutes" in result
        assert "T" in result
        assert result.startswith("Parsed")

    def test_invalid_expression(self):
        result = self._handle({"expression": "when pigs fly"})
        assert result.startswith("Error:")

    def test_missing_expression(self):
        result = self._handle({})
        assert "required" in result.lower()

    def test_tomorrow_at_9am(self):
        result = self._handle({"expression": "tomorrow at 9am"})
        assert "09:00:00" in result

    def test_empty_expression(self):
        result = self._handle({"expression": ""})
        assert "required" in result.lower()

    def test_handler_in_client_dispatch(self):
        """Verify the dispatch case exists in client.py source."""
        from pathlib import Path

        client_src = Path("src/discord/client.py").read_text()
        assert 'tool_name == "parse_time"' in client_src
        assert "_handle_parse_time" in client_src


# ── Timezone configurability ────────────────────────────────────────


class TestTimeParserTimezone:
    """Round 6: time parser uses configurable default timezone."""

    def test_set_default_timezone(self):
        from src.tools.time_parser import set_default_timezone, _default_tz
        import src.tools.time_parser as tp
        original = tp._default_tz
        try:
            set_default_timezone("Asia/Tokyo")
            assert str(tp._default_tz) == "Asia/Tokyo"
        finally:
            tp._default_tz = original

    def test_default_timezone_is_utc(self):
        import src.tools.time_parser as tp
        assert str(tp._default_tz) == "UTC"

    def test_parse_time_respects_configured_tz(self):
        """parse_time with no explicit now uses the configured default timezone."""
        import src.tools.time_parser as tp
        original = tp._default_tz
        try:
            set_default_timezone("Asia/Tokyo")
            result = parse_time("in 1 hour")
            # Tokyo is UTC+9, so offset should be +09:00
            assert "+09:00" in result
        finally:
            tp._default_tz = original

    def test_parse_time_tool_description_no_eastern(self):
        """Tool description should not hardcode 'Eastern Time'."""
        tool = next(t for t in TOOLS if t["name"] == "parse_time")
        assert "Eastern Time" not in tool["description"]

    def test_set_default_timezone_called_in_client_init(self):
        """Verify client.py calls set_default_timezone during init."""
        from pathlib import Path
        client_src = Path("src/discord/client.py").read_text()
        assert "set_default_timezone" in client_src
