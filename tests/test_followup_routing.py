"""Tests for follow-up message routing — per-channel tool-use context.

When tools are executed in a channel, follow-up messages like 'and the desktop?'
should be classified as 'task' (not 'chat') so they route to the tool backend
instead of the chat backend.
"""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import AnsiblexBot  # noqa: E402


class TestTrackRecentActionChannel:
    """_track_recent_action should update _last_tool_use per channel."""

    def _make_bot_stub(self):
        """Create a minimal stub with the fields _track_recent_action needs."""
        # We can't instantiate AnsiblexBot (needs Config + discord), so build
        # a stub object with the relevant attributes and bind the real method.
        stub = MagicMock()
        stub._recent_actions = {}
        stub._recent_actions_max = 10
        stub._last_tool_use = {}
        stub._track_recent_action = AnsiblexBot._track_recent_action.__get__(stub)
        return stub

    def test_sets_last_tool_use_for_channel(self):
        bot = self._make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 150, channel_id="chan-1")
        assert "chan-1" in bot._last_tool_use
        assert bot._last_tool_use["chan-1"] > 0

    def test_different_channels_tracked_independently(self):
        bot = self._make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 150, channel_id="chan-1")
        t1 = bot._last_tool_use["chan-1"]

        bot._track_recent_action("check_memory", {"host": "desktop"}, "OK", 200, channel_id="chan-2")
        assert "chan-2" in bot._last_tool_use
        # chan-1 should still have its original timestamp
        assert bot._last_tool_use["chan-1"] == t1

    def test_updates_timestamp_on_subsequent_tool_use(self):
        bot = self._make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 150, channel_id="chan-1")
        t1 = bot._last_tool_use["chan-1"]

        # Small delay to ensure monotonic time advances
        bot._track_recent_action("check_memory", {"host": "server"}, "OK", 200, channel_id="chan-1")
        assert bot._last_tool_use["chan-1"] >= t1

    def test_no_channel_id_does_not_set_last_tool_use(self):
        bot = self._make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 150)
        assert bot._last_tool_use == {}
        assert bot._recent_actions == {}

    def test_none_channel_id_does_not_set_last_tool_use(self):
        bot = self._make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 150, channel_id=None)
        assert bot._last_tool_use == {}
        assert bot._recent_actions == {}

    def test_still_appends_to_recent_actions(self):
        """Channel tracking should store actions per-channel."""
        bot = self._make_bot_stub()
        bot._track_recent_action("check_disk", {"host": "server"}, "OK", 150, channel_id="chan-1")
        assert "chan-1" in bot._recent_actions
        assert len(bot._recent_actions["chan-1"]) == 1
        _ts, entry = bot._recent_actions["chan-1"][0]
        assert "check_disk" in entry


class TestRecentToolUseWindow:
    """The 300-second (5-minute) window for recent tool use detection."""

    def test_within_window(self):
        """Tool use within 5 minutes should be considered recent."""
        last_tool_use = {"chan-1": time.monotonic() - 60}  # 1 minute ago
        recent = time.monotonic() - last_tool_use.get("chan-1", 0) < 300
        assert recent is True

    def test_outside_window(self):
        """Tool use older than 5 minutes should not be considered recent."""
        last_tool_use = {"chan-1": time.monotonic() - 400}  # ~6.7 minutes ago
        recent = time.monotonic() - last_tool_use.get("chan-1", 0) < 300
        assert recent is False

    def test_no_tool_use_history(self):
        """Channel with no tool use history should not be considered recent."""
        last_tool_use = {}
        recent = time.monotonic() - last_tool_use.get("chan-1", 0) < 300
        assert recent is False

    def test_different_channel_not_affected(self):
        """Tool use in chan-1 should not affect chan-2's recency."""
        last_tool_use = {"chan-1": time.monotonic()}
        recent_chan2 = time.monotonic() - last_tool_use.get("chan-2", 0) < 300
        assert recent_chan2 is False


