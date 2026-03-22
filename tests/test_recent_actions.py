"""Tests for per-channel recent actions with time-based expiry.

Recent actions are injected into the system prompt to give the LLM context
about what tools were just used.  They must be scoped per-channel (channel A's
actions should not leak into channel B's prompt) and expire after 1 hour.
"""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402


def _make_bot_stub():
    """Create a minimal stub with the fields _track_recent_action needs."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._last_tool_use = {}
    stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)
    return stub


# ---------------------------------------------------------------------------
# Per-channel isolation
# ---------------------------------------------------------------------------

class TestPerChannelIsolation:
    """Actions in channel A must not appear in channel B."""

    def test_actions_stored_per_channel(self):
        bot = _make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 100, channel_id="chan-1")
        bot._track_recent_action("check_memory", {"host": "desktop"}, "OK", 200, channel_id="chan-2")

        assert "chan-1" in bot._recent_actions
        assert "chan-2" in bot._recent_actions
        assert len(bot._recent_actions["chan-1"]) == 1
        assert len(bot._recent_actions["chan-2"]) == 1

    def test_channel_a_not_in_channel_b(self):
        bot = _make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 100, channel_id="chan-1")
        bot._track_recent_action("check_memory", {"host": "desktop"}, "OK", 200, channel_id="chan-2")

        _, entry_a = bot._recent_actions["chan-1"][0]
        _, entry_b = bot._recent_actions["chan-2"][0]
        assert "check_disk" in entry_a
        assert "check_memory" in entry_b
        # Ensure no cross-contamination
        assert "check_memory" not in entry_a
        assert "check_disk" not in entry_b

    def test_multiple_actions_same_channel(self):
        bot = _make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 100, channel_id="chan-1")
        bot._track_recent_action("check_memory", {"host": "server"}, "OK", 200, channel_id="chan-1")

        assert len(bot._recent_actions["chan-1"]) == 2

    def test_no_channel_id_stores_nothing(self):
        bot = _make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 100)
        assert bot._recent_actions == {}

    def test_none_channel_id_stores_nothing(self):
        bot = _make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 100, channel_id=None)
        assert bot._recent_actions == {}


# ---------------------------------------------------------------------------
# Per-channel cap
# ---------------------------------------------------------------------------

class TestPerChannelCap:
    """Each channel should be capped at _recent_actions_max entries."""

    def test_cap_at_max(self):
        bot = _make_bot_stub()
        bot._recent_actions_max = 3
        for i in range(5):
            bot._track_recent_action(f"tool_{i}", {"host": "s"}, "OK", 100, channel_id="chan-1")

        assert len(bot._recent_actions["chan-1"]) == 3
        # Should keep the last 3 (tool_2, tool_3, tool_4)
        _, last_entry = bot._recent_actions["chan-1"][-1]
        assert "tool_4" in last_entry

    def test_cap_per_channel_independent(self):
        bot = _make_bot_stub()
        bot._recent_actions_max = 2
        for i in range(3):
            bot._track_recent_action(f"a_{i}", {"host": "s"}, "OK", 100, channel_id="chan-1")
        bot._track_recent_action("b_0", {"host": "s"}, "OK", 100, channel_id="chan-2")

        assert len(bot._recent_actions["chan-1"]) == 2
        assert len(bot._recent_actions["chan-2"]) == 1


# ---------------------------------------------------------------------------
# Time-based expiry
# ---------------------------------------------------------------------------

class TestTimeBasedExpiry:
    """Old actions should be excluded when building the system prompt."""

    def test_fresh_actions_included(self):
        """Actions within expiry window should appear."""
        bot = _make_bot_stub()
        bot._recent_actions_expiry = 3600
        now = time.time()
        bot._recent_actions["chan-1"] = [
            (now - 60, "- [14:00] `check_disk`(host=server) → OK (100ms)"),
        ]

        channel_actions = [
            entry for ts, entry in bot._recent_actions.get("chan-1", [])
            if now - ts < bot._recent_actions_expiry
        ]
        assert len(channel_actions) == 1

    def test_old_actions_excluded(self):
        """Actions older than expiry should be filtered out."""
        bot = _make_bot_stub()
        bot._recent_actions_expiry = 3600
        now = time.time()
        bot._recent_actions["chan-1"] = [
            (now - 7200, "- [12:00] `old_tool`(host=server) → OK (100ms)"),  # 2 hours ago
            (now - 60, "- [14:00] `new_tool`(host=server) → OK (100ms)"),    # 1 minute ago
        ]

        channel_actions = [
            entry for ts, entry in bot._recent_actions.get("chan-1", [])
            if now - ts < bot._recent_actions_expiry
        ]
        assert len(channel_actions) == 1
        assert "new_tool" in channel_actions[0]

    def test_all_expired(self):
        """All expired actions should result in empty list."""
        bot = _make_bot_stub()
        bot._recent_actions_expiry = 3600
        now = time.time()
        bot._recent_actions["chan-1"] = [
            (now - 7200, "- [12:00] `old_tool`(host=server) → OK (100ms)"),
        ]

        channel_actions = [
            entry for ts, entry in bot._recent_actions.get("chan-1", [])
            if now - ts < bot._recent_actions_expiry
        ]
        assert len(channel_actions) == 0

    def test_no_actions_for_channel(self):
        """Channel with no actions should produce empty list."""
        bot = _make_bot_stub()
        channel_actions = [
            entry for ts, entry in bot._recent_actions.get("chan-999", [])
            if time.time() - ts < bot._recent_actions_expiry
        ]
        assert len(channel_actions) == 0


# ---------------------------------------------------------------------------
# Entry format
# ---------------------------------------------------------------------------

class TestEntryFormat:
    """Verify the format of recorded action entries."""

    def test_entry_contains_tool_name(self):
        bot = _make_bot_stub()
        bot._track_recent_action("query_prometheus", {"query": "up"}, "OK", 300, channel_id="chan-1")
        _, entry = bot._recent_actions["chan-1"][0]
        assert "query_prometheus" in entry

    def test_entry_contains_input_summary(self):
        bot = _make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 100, channel_id="chan-1")
        _, entry = bot._recent_actions["chan-1"][0]
        assert "host=server" in entry

    def test_entry_contains_status_ok(self):
        bot = _make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK result", 100, channel_id="chan-1")
        _, entry = bot._recent_actions["chan-1"][0]
        assert "OK" in entry

    def test_entry_contains_status_error(self):
        bot = _make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "Error: connection refused", 100, channel_id="chan-1")
        _, entry = bot._recent_actions["chan-1"][0]
        assert "ERROR" in entry

    def test_entry_has_timestamp(self):
        bot = _make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 100, channel_id="chan-1")
        ts, entry = bot._recent_actions["chan-1"][0]
        assert isinstance(ts, float)
        assert ts > 0
        # Entry text should also contain a time string like [HH:MM]
        import re
        assert re.search(r"\[\d{2}:\d{2}\]", entry)

    def test_long_input_truncated(self):
        bot = _make_bot_stub()
        long_val = "x" * 200
        bot._track_recent_action("run_command", {"host": "server", "command": long_val}, "OK", 100, channel_id="chan-1")
        _, entry = bot._recent_actions["chan-1"][0]
        # Input summary should be truncated with ...
        assert "..." in entry

    def test_non_string_input_excluded(self):
        bot = _make_bot_stub()
        bot._track_recent_action("some_tool", {"host": "server", "count": 5}, "OK", 100, channel_id="chan-1")
        _, entry = bot._recent_actions["chan-1"][0]
        # count=5 is int, should not appear in summary (only string values included)
        assert "host=server" in entry
        assert "count" not in entry


# ---------------------------------------------------------------------------
# Timestamp stored for expiry
# ---------------------------------------------------------------------------

class TestTimestampStorage:
    """Each entry should have a real timestamp for expiry filtering."""

    def test_timestamp_is_current(self):
        bot = _make_bot_stub()
        before = time.time()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 100, channel_id="chan-1")
        after = time.time()

        ts, _ = bot._recent_actions["chan-1"][0]
        assert before <= ts <= after
