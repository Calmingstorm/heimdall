"""Tests for Round 10: Topic change detection.

Validates that topic switches are detected based on keyword overlap with
recent history, that the separator is strengthened on topic changes, and
that the history window is reduced on topic changes.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.sessions.manager import (
    Message,
    SessionManager,
    score_relevance,
    TOPIC_CHANGE_SCORE_THRESHOLD,
    TOPIC_CHANGE_TIME_GAP,
    TOPIC_CHANGE_RECENT_WINDOW,
)


@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=100,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestTopicChangeConstants:
    def test_score_threshold_is_low(self):
        assert 0.0 < TOPIC_CHANGE_SCORE_THRESHOLD < 0.2

    def test_time_gap_is_five_minutes(self):
        assert TOPIC_CHANGE_TIME_GAP == 300

    def test_recent_window_positive(self):
        assert TOPIC_CHANGE_RECENT_WINDOW >= 2


# ---------------------------------------------------------------------------
# detect_topic_change — basic behavior
# ---------------------------------------------------------------------------

class TestDetectTopicChange:
    def test_no_history_no_change(self, session_mgr):
        """No messages → no topic change."""
        result = session_mgr.detect_topic_change("ch1", "hello world")
        assert result["is_topic_change"] is False
        assert result["time_gap"] == 0.0
        assert result["max_overlap"] == 0.0

    def test_same_topic_no_change(self, session_mgr):
        """Asking about same topic → no topic change."""
        session = session_mgr.get_or_create("ch1")
        session.messages = [
            Message(role="user", content="check disk usage on server-1"),
            Message(role="assistant", content="disk usage is 42% on server-1"),
            Message(role="user", content="also check server-2 disk"),
        ]
        result = session_mgr.detect_topic_change("ch1", "what about server-3 disk usage")
        assert result["is_topic_change"] is False
        assert result["max_overlap"] > TOPIC_CHANGE_SCORE_THRESHOLD

    def test_different_topic_detected(self, session_mgr):
        """Completely different topic → topic change detected."""
        session = session_mgr.get_or_create("ch1")
        session.messages = [
            Message(role="user", content="check nginx status on web-server"),
            Message(role="assistant", content="nginx is running on web-server"),
            Message(role="user", content="restart the nginx service"),
        ]
        result = session_mgr.detect_topic_change("ch1", "write me a haiku about cats")
        assert result["is_topic_change"] is True
        assert result["max_overlap"] < TOPIC_CHANGE_SCORE_THRESHOLD

    def test_single_message_no_change(self, session_mgr):
        """Only 1 message in history → not enough to call it a topic change."""
        session = session_mgr.get_or_create("ch1")
        session.messages = [
            Message(role="user", content="check nginx status"),
        ]
        result = session_mgr.detect_topic_change("ch1", "write me a haiku about cats")
        # Requires at least 2 recent messages to detect topic change
        assert result["is_topic_change"] is False

    def test_time_gap_detected(self, session_mgr):
        """Messages >5 min ago → has_time_gap is True."""
        session = session_mgr.get_or_create("ch1")
        old_time = time.time() - 600  # 10 min ago
        session.messages = [
            Message(role="user", content="check disk", timestamp=old_time),
            Message(role="assistant", content="disk ok", timestamp=old_time + 5),
        ]
        result = session_mgr.detect_topic_change("ch1", "write a poem")
        assert result["has_time_gap"] is True
        assert result["time_gap"] > TOPIC_CHANGE_TIME_GAP

    def test_no_time_gap_recent(self, session_mgr):
        """Messages just now → has_time_gap is False."""
        session = session_mgr.get_or_create("ch1")
        now = time.time()
        session.messages = [
            Message(role="user", content="check disk", timestamp=now - 10),
            Message(role="assistant", content="disk ok", timestamp=now - 5),
        ]
        result = session_mgr.detect_topic_change("ch1", "write a poem")
        assert result["has_time_gap"] is False
        assert result["time_gap"] < TOPIC_CHANGE_TIME_GAP

    def test_partial_overlap_no_change(self, session_mgr):
        """Some shared keywords → not a topic change."""
        session = session_mgr.get_or_create("ch1")
        session.messages = [
            Message(role="user", content="deploy nginx on server-1"),
            Message(role="assistant", content="deployed nginx on server-1"),
        ]
        result = session_mgr.detect_topic_change("ch1", "check nginx logs on server-1")
        assert result["is_topic_change"] is False

    def test_return_dict_keys(self, session_mgr):
        """Return dict has all expected keys."""
        session = session_mgr.get_or_create("ch1")
        session.messages = [
            Message(role="user", content="hello"),
            Message(role="user", content="world"),
        ]
        result = session_mgr.detect_topic_change("ch1", "test query")
        assert "is_topic_change" in result
        assert "time_gap" in result
        assert "has_time_gap" in result
        assert "max_overlap" in result


# ---------------------------------------------------------------------------
# detect_topic_change — edge cases
# ---------------------------------------------------------------------------

class TestTopicChangeEdgeCases:
    def test_empty_query(self, session_mgr):
        """Empty query → no meaningful tokens → depends on overlap."""
        session = session_mgr.get_or_create("ch1")
        session.messages = [
            Message(role="user", content="check disk"),
            Message(role="assistant", content="disk ok"),
        ]
        result = session_mgr.detect_topic_change("ch1", "")
        # Empty query produces 0 overlap, but score_relevance returns 0.0
        # which is below threshold → topic change
        assert result["max_overlap"] == 0.0

    def test_stop_words_only_query(self, session_mgr):
        """Query with only stop words → all tokens filtered → 0 overlap."""
        session = session_mgr.get_or_create("ch1")
        session.messages = [
            Message(role="user", content="check disk"),
            Message(role="assistant", content="disk ok"),
        ]
        result = session_mgr.detect_topic_change("ch1", "the is a to")
        assert result["max_overlap"] == 0.0

    def test_many_messages_uses_window(self, session_mgr):
        """Only checks the most recent TOPIC_CHANGE_RECENT_WINDOW messages."""
        session = session_mgr.get_or_create("ch1")
        # 10 messages about nginx, then 5 about cats
        for i in range(10):
            session.messages.append(
                Message(role="user", content=f"nginx config update {i}")
            )
        for i in range(TOPIC_CHANGE_RECENT_WINDOW):
            session.messages.append(
                Message(role="user", content=f"cats are wonderful creatures {i}")
            )
        # Query about cats should NOT be a topic change (recent window is about cats)
        result = session_mgr.detect_topic_change("ch1", "tell me about cats")
        assert result["is_topic_change"] is False

    def test_different_channel_isolated(self, session_mgr):
        """Topic detection is per-channel."""
        s1 = session_mgr.get_or_create("ch1")
        s1.messages = [
            Message(role="user", content="nginx deploy"),
            Message(role="assistant", content="deployed nginx"),
        ]
        s2 = session_mgr.get_or_create("ch2")
        s2.messages = [
            Message(role="user", content="database backup"),
            Message(role="assistant", content="backed up database"),
        ]
        # Query about nginx on ch2 (which discusses database) → topic change
        result = session_mgr.detect_topic_change("ch2", "deploy nginx")
        assert result["is_topic_change"] is True

    def test_logs_topic_change(self, session_mgr):
        """Topic change is logged."""
        session = session_mgr.get_or_create("ch1")
        session.messages = [
            Message(role="user", content="check nginx status on web-server"),
            Message(role="assistant", content="nginx is running on web-server"),
        ]
        with patch("src.sessions.manager.log") as mock_log:
            session_mgr.detect_topic_change("ch1", "write me a haiku about cats")
            mock_log.info.assert_called()
            call_args = str(mock_log.info.call_args)
            assert "Topic change" in call_args or "topic change" in call_args.lower()


# ---------------------------------------------------------------------------
# get_task_history with topic_change=True
# ---------------------------------------------------------------------------

class TestTaskHistoryTopicChange:
    async def test_topic_change_reduces_history(self, session_mgr):
        """topic_change=True → only last message in history."""
        session = session_mgr.get_or_create("ch1")
        for i in range(10):
            session.messages.append(
                Message(role="user" if i % 2 == 0 else "assistant",
                        content=f"message {i}")
            )
        history = await session_mgr.get_task_history(
            "ch1", max_messages=20, topic_change=True,
        )
        # Should have only the last message (no summary in this session)
        assert len(history) == 1
        assert history[0]["content"] == "message 9"

    async def test_topic_change_keeps_summary(self, session_mgr):
        """topic_change=True still includes summary for broad context."""
        session = session_mgr.get_or_create("ch1")
        session.summary = "[Topics: nginx, server-1] Deployed nginx."
        for i in range(5):
            session.messages.append(
                Message(role="user" if i % 2 == 0 else "assistant",
                        content=f"msg {i}")
            )
        history = await session_mgr.get_task_history(
            "ch1", max_messages=20, topic_change=True,
        )
        # Summary (2 messages) + 1 recent
        assert len(history) == 3
        assert "Previous conversation summary" in history[0]["content"]
        assert history[2]["content"] == "msg 4"

    async def test_no_topic_change_normal_history(self, session_mgr):
        """topic_change=False → normal history behavior."""
        session = session_mgr.get_or_create("ch1")
        for i in range(5):
            session.messages.append(
                Message(role="user" if i % 2 == 0 else "assistant",
                        content=f"msg {i}")
            )
        history = await session_mgr.get_task_history(
            "ch1", max_messages=20, topic_change=False,
        )
        assert len(history) == 5

    async def test_topic_change_empty_session(self, session_mgr):
        """topic_change=True on empty session → empty history."""
        history = await session_mgr.get_task_history(
            "ch1", max_messages=20, topic_change=True,
        )
        assert len(history) == 0

    async def test_topic_change_logs_reduction(self, session_mgr):
        """topic_change=True logs the reduction."""
        session = session_mgr.get_or_create("ch1")
        for i in range(5):
            session.messages.append(
                Message(role="user", content=f"msg {i}")
            )
        with patch("src.sessions.manager.log") as mock_log:
            await session_mgr.get_task_history(
                "ch1", max_messages=20, topic_change=True,
            )
            mock_log.info.assert_called()
            call_args = str(mock_log.info.call_args)
            assert "Topic change" in call_args or "topic change" in call_args.lower()


# ---------------------------------------------------------------------------
# Client integration — separator injection
# ---------------------------------------------------------------------------

class TestClientTopicChangeSeparator:
    """Test that the separator includes topic change notice."""

    async def test_topic_change_in_separator(self):
        """When topic change detected, separator includes TOPIC CHANGE notice."""
        from src.discord.client import HeimdallBot

        with patch("src.discord.client.discord.Client.__init__"):
            bot = HeimdallBot.__new__(HeimdallBot)
            bot.sessions = MagicMock()
            bot.sessions.detect_topic_change = MagicMock(return_value={
                "is_topic_change": True,
                "time_gap": 600.0,
                "has_time_gap": True,
                "max_overlap": 0.01,
            })
            bot.sessions.get_task_history = AsyncMock(return_value=[
                {"role": "user", "content": "old stuff"},
                {"role": "user", "content": "new topic"},
            ])

            # Build a mock message
            message = MagicMock()
            message.content = "write me a haiku"
            message.id = 12345
            message.author = MagicMock()
            message.author.id = 99
            message.author.display_name = "TestUser"
            message.author.bot = False
            message.channel = MagicMock()
            message.channel.typing = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(), __aexit__=AsyncMock()
            ))

            # Mock dependencies
            bot.codex_client = MagicMock()
            bot.config = MagicMock()
            bot.config.tools.enabled = True
            bot.config.discord.respond_to_bots = False
            bot.permissions = MagicMock()
            bot.permissions.filter_tools = MagicMock(return_value=[{"name": "test"}])
            bot._merged_tool_definitions = MagicMock(return_value=[{"name": "test"}])
            bot._system_prompt = "system"

            # Mock chat_with_tools to capture the messages sent
            captured_messages = []

            async def fake_chat_with_tools(messages, system, tools):
                captured_messages.extend(messages)
                resp = MagicMock()
                resp.text = "response"
                resp.tool_calls = []
                return resp

            bot.codex_client.chat_with_tools = fake_chat_with_tools

            await bot._process_with_tools(
                message,
                [{"role": "user", "content": "old stuff"},
                 {"role": "user", "content": "new topic"}],
                topic_change=True,
            )

            # Find the developer separator message
            sep_msgs = [m for m in captured_messages if m.get("role") == "developer"]
            assert len(sep_msgs) >= 1
            sep_content = sep_msgs[0]["content"]
            assert "TOPIC CHANGE DETECTED" in sep_content
            assert "fresh request" in sep_content.lower()

    async def test_no_topic_change_no_notice(self):
        """When no topic change, separator does NOT include TOPIC CHANGE."""
        from src.discord.client import HeimdallBot

        with patch("src.discord.client.discord.Client.__init__"):
            bot = HeimdallBot.__new__(HeimdallBot)

            message = MagicMock()
            message.content = "check disk"
            message.id = 12345
            message.author = MagicMock()
            message.author.id = 99
            message.author.display_name = "TestUser"
            message.author.bot = False

            bot.codex_client = MagicMock()
            bot.config = MagicMock()
            bot.config.tools.enabled = True
            bot.config.discord.respond_to_bots = False
            bot.permissions = MagicMock()
            bot.permissions.filter_tools = MagicMock(return_value=[{"name": "test"}])
            bot._merged_tool_definitions = MagicMock(return_value=[{"name": "test"}])
            bot._system_prompt = "system"

            captured_messages = []

            async def fake_chat_with_tools(messages, system, tools):
                captured_messages.extend(messages)
                resp = MagicMock()
                resp.text = "response"
                resp.tool_calls = []
                return resp

            bot.codex_client.chat_with_tools = fake_chat_with_tools

            await bot._process_with_tools(
                message,
                [{"role": "user", "content": "check disk usage"},
                 {"role": "user", "content": "check disk"}],
                topic_change=False,
            )

            sep_msgs = [m for m in captured_messages if m.get("role") == "developer"]
            assert len(sep_msgs) >= 1
            sep_content = sep_msgs[0]["content"]
            assert "TOPIC CHANGE" not in sep_content


# ---------------------------------------------------------------------------
# Source code verification
# ---------------------------------------------------------------------------

class TestSourceVerification:
    def test_manager_has_topic_constants(self):
        import src.sessions.manager as mod
        assert hasattr(mod, "TOPIC_CHANGE_SCORE_THRESHOLD")
        assert hasattr(mod, "TOPIC_CHANGE_TIME_GAP")
        assert hasattr(mod, "TOPIC_CHANGE_RECENT_WINDOW")

    def test_manager_has_detect_method(self):
        assert hasattr(SessionManager, "detect_topic_change")

    def test_task_history_accepts_topic_change(self):
        import inspect
        sig = inspect.signature(SessionManager.get_task_history)
        assert "topic_change" in sig.parameters

    def test_client_imports_and_calls(self):
        """Client source should call detect_topic_change."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert "detect_topic_change" in source
        assert "topic_change" in source

    def test_separator_has_topic_change_block(self):
        """Client source has the topic change separator injection."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client)
        assert "TOPIC CHANGE DETECTED" in source

    def test_process_with_tools_accepts_topic_change(self):
        import inspect
        from src.discord.client import HeimdallBot
        sig = inspect.signature(HeimdallBot._process_with_tools)
        assert "topic_change" in sig.parameters


# ---------------------------------------------------------------------------
# Integration: full flow
# ---------------------------------------------------------------------------

class TestTopicChangeIntegration:
    async def test_topic_change_flow(self, session_mgr):
        """Full flow: add messages, detect topic change, get reduced history."""
        ch = "integration_ch"
        # Add messages about nginx
        session_mgr.add_message(ch, "user", "deploy nginx to server-1")
        session_mgr.add_message(ch, "assistant", "deployed nginx to server-1")
        session_mgr.add_message(ch, "user", "check nginx status")
        session_mgr.add_message(ch, "assistant", "nginx running on server-1")

        # Detect topic change — new topic
        info = session_mgr.detect_topic_change(ch, "write me a poem about the ocean")
        assert info["is_topic_change"] is True

        # Get history with topic change — should be minimal
        history = await session_mgr.get_task_history(
            ch, max_messages=20, topic_change=True,
        )
        # Only last message
        assert len(history) == 1
        assert "nginx" in history[0]["content"]

    async def test_same_topic_flow(self, session_mgr):
        """Same topic: no topic change, full history returned."""
        ch = "integration_ch2"
        session_mgr.add_message(ch, "user", "deploy nginx to server-1")
        session_mgr.add_message(ch, "assistant", "deployed nginx to server-1")
        session_mgr.add_message(ch, "user", "check nginx status")

        info = session_mgr.detect_topic_change(ch, "restart nginx on server-1")
        assert info["is_topic_change"] is False

        history = await session_mgr.get_task_history(
            ch, max_messages=20, topic_change=False,
        )
        assert len(history) == 3

    async def test_topic_change_with_time_gap(self, session_mgr):
        """Topic change with time gap — both flags set."""
        ch = "integration_ch3"
        old_time = time.time() - 600  # 10 min ago
        session = session_mgr.get_or_create(ch)
        session.messages = [
            Message(role="user", content="nginx deploy", timestamp=old_time),
            Message(role="assistant", content="done", timestamp=old_time + 5),
        ]

        info = session_mgr.detect_topic_change(ch, "write a haiku about rain")
        assert info["is_topic_change"] is True
        assert info["has_time_gap"] is True
        assert info["time_gap"] > 300
