"""Natural language time expression parser.

Converts expressions like 'in 2 hours', 'tomorrow at 9am', 'next Monday at 3pm'
to ISO datetime strings. Used as a helper for the LLM when scheduling reminders.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_default_tz = ZoneInfo("UTC")


def set_default_timezone(tz_name: str) -> None:
    """Set the default timezone used by parse_time when no explicit time is given."""
    global _default_tz
    _default_tz = ZoneInfo(tz_name)

# Day name → weekday number (Monday=0)
DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3,
    "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}

UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
    "day": 86400, "days": 86400, "d": 86400,
    "week": 604800, "weeks": 604800, "w": 604800,
}


def _parse_time_of_day(text: str) -> tuple[int, int] | None:
    """Extract hour and minute from a time expression like '9am', '3:30pm', '17:00'."""
    text = text.strip().lower()

    # 12-hour: 9am, 9:30pm, 9:30 am
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        if m.group(3) == "pm" and hour != 12:
            hour += 12
        elif m.group(3) == "am" and hour == 12:
            hour = 0
        return (hour, minute)

    # 24-hour: 17:00, 09:30
    m = re.match(r"(\d{1,2}):(\d{2})$", text)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # Bare hour: "9" — too ambiguous, skip
    return None


def _next_weekday(now: datetime, target_weekday: int) -> datetime:
    """Return the next occurrence of target_weekday after now."""
    days_ahead = target_weekday - now.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return now + timedelta(days=days_ahead)


def parse_time(expression: str, now: datetime | None = None) -> str:
    """Parse a natural language time expression into an ISO datetime string.

    Args:
        expression: Natural language like 'in 2 hours', 'tomorrow at 9am',
                    'next Monday at 3pm', 'at 5pm'.
        now: Reference time (defaults to current time in configured timezone).

    Returns:
        ISO datetime string with timezone (e.g. '2026-03-18T17:00:00-04:00').

    Raises:
        ValueError: If the expression cannot be parsed.
    """
    if now is None:
        now = datetime.now(_default_tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_default_tz)

    text = expression.strip().lower()

    # --- "in X units" ---
    m = re.match(r"in\s+(\d+)\s+(\w+)", text)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        if unit not in UNIT_SECONDS:
            raise ValueError(f"Unknown time unit: {unit}")
        result = now + timedelta(seconds=amount * UNIT_SECONDS[unit])
        return result.isoformat()

    # --- "tomorrow [at TIME]" ---
    if text.startswith("tomorrow"):
        tomorrow = now + timedelta(days=1)
        time_part = re.sub(r"^tomorrow\s*(at\s*)?", "", text).strip()
        if time_part:
            parsed = _parse_time_of_day(time_part)
            if parsed is None:
                raise ValueError(f"Cannot parse time: {time_part}")
            hour, minute = parsed
        else:
            hour, minute = 9, 0  # default 9am
        result = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return result.isoformat()

    # --- "today [at TIME]" ---
    if text.startswith("today"):
        time_part = re.sub(r"^today\s*(at\s*)?", "", text).strip()
        if time_part:
            parsed = _parse_time_of_day(time_part)
            if parsed is None:
                raise ValueError(f"Cannot parse time: {time_part}")
            hour, minute = parsed
        else:
            raise ValueError("'today' requires a time (e.g. 'today at 5pm')")
        result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return result.isoformat()

    # --- "next DAYNAME [at TIME]" ---
    m = re.match(r"next\s+(\w+)(?:\s+at\s+(.+))?", text)
    if m:
        day_str = m.group(1)
        if day_str in DAY_NAMES:
            target_day = _next_weekday(now, DAY_NAMES[day_str])
            time_str = m.group(2)
            if time_str:
                parsed = _parse_time_of_day(time_str)
                if parsed is None:
                    raise ValueError(f"Cannot parse time: {time_str}")
                hour, minute = parsed
            else:
                hour, minute = 9, 0  # default 9am
            result = target_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return result.isoformat()

    # --- "DAYNAME [at TIME]" (without "next") ---
    first_word = text.split()[0] if text.split() else ""
    if first_word in DAY_NAMES:
        target_day = _next_weekday(now, DAY_NAMES[first_word])
        time_match = re.search(r"at\s+(.+)", text)
        if time_match:
            parsed = _parse_time_of_day(time_match.group(1))
            if parsed is None:
                raise ValueError(f"Cannot parse time: {time_match.group(1)}")
            hour, minute = parsed
        else:
            hour, minute = 9, 0
        result = target_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return result.isoformat()

    # --- "at TIME" (today if future, tomorrow if past) ---
    m = re.match(r"at\s+(.+)", text)
    if m:
        parsed = _parse_time_of_day(m.group(1))
        if parsed is None:
            raise ValueError(f"Cannot parse time: {m.group(1)}")
        hour, minute = parsed
        result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if result <= now:
            result += timedelta(days=1)
        return result.isoformat()

    # --- bare time like "5pm", "9:30am" ---
    parsed = _parse_time_of_day(text)
    if parsed:
        hour, minute = parsed
        result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if result <= now:
            result += timedelta(days=1)
        return result.isoformat()

    raise ValueError(
        f"Cannot parse time expression: '{expression}'. "
        "Try formats like: 'in 30 minutes', 'tomorrow at 9am', "
        "'next Monday at 3pm', 'at 5pm'"
    )
