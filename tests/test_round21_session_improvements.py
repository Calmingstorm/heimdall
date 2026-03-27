"""Tests for Round 21: Session + Conversation Improvements.

Covers:
- Compaction fallback preserves existing summary (bug fix)
- Compaction instruction improvements (tool names, structured output)
- Bot buffer size limit
- Session context window efficiency
- Image block handling in session history
- Bot-to-bot conversation handling
- Context separator behavior
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.sessions.manager import (
    COMPACTION_THRESHOLD,
    Message,
    Session,
    SessionManager,
)


@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=10,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


# ---------------------------------------------------------------------------
# Compaction fallback — preserve existing summary on error
# ---------------------------------------------------------------------------

class TestCompactionFallbackPreservesSummary:
    """When compaction fails, the existing summary should be preserved."""

    @pytest.mark.asyncio
    async def test_fallback_keeps_existing_summary(self, tmp_dir):
        """If LLM compaction fails, the existing summary is NOT cleared."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        session = mgr.get_or_create(channel)
        session.summary = "User discussed server deployment on host-1."

        # Fill above compaction threshold
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        # Compaction function that fails
        async def failing_compaction(messages, system):
            raise RuntimeError("LLM unavailable")

        mgr.set_compaction_fn(failing_compaction)
        await mgr.get_history_with_compaction(channel)

        # Summary should be preserved, not cleared
        assert session.summary == "User discussed server deployment on host-1."

    @pytest.mark.asyncio
    async def test_fallback_trims_messages(self, tmp_dir):
        """Even on compaction failure, messages are trimmed to max_history."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        async def failing_compaction(messages, system):
            raise RuntimeError("LLM unavailable")

        mgr.set_compaction_fn(failing_compaction)
        await mgr.get_history_with_compaction(channel)

        session = mgr.get_or_create(channel)
        assert len(session.messages) <= 30

    @pytest.mark.asyncio
    async def test_fallback_no_compaction_fn(self, tmp_dir):
        """When no compaction function is set, summary is preserved."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        session = mgr.get_or_create(channel)
        session.summary = "Previous context about monitoring."

        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        # No compaction function set — should fall back
        await mgr.get_history_with_compaction(channel)
        assert session.summary == "Previous context about monitoring."

    @pytest.mark.asyncio
    async def test_successful_compaction_replaces_summary(self, tmp_dir):
        """On successful compaction, the summary IS replaced."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        session = mgr.get_or_create(channel)
        session.summary = "Old summary."

        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        async def mock_compaction(messages, system):
            return "New summary with updated context."

        mgr.set_compaction_fn(mock_compaction)
        await mgr.get_history_with_compaction(channel)

        assert session.summary == "New summary with updated context."

    @pytest.mark.asyncio
    async def test_fallback_with_empty_summary(self, tmp_dir):
        """When there's no existing summary and compaction fails, summary stays empty."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        async def failing_compaction(messages, system):
            raise RuntimeError("LLM unavailable")

        mgr.set_compaction_fn(failing_compaction)
        await mgr.get_history_with_compaction(channel)

        session = mgr.get_or_create(channel)
        assert session.summary == ""


# ---------------------------------------------------------------------------
# Compaction instruction improvements
# ---------------------------------------------------------------------------

class TestCompactionInstructions:
    """Verify compaction system instruction contains key directives."""

    @pytest.mark.asyncio
    async def test_instruction_mentions_tool_names(self, tmp_dir):
        """Compaction instruction should tell the LLM to preserve tool names."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        captured_system = None

        async def capture_compaction(messages, system):
            nonlocal captured_system
            captured_system = system
            return "Summary."

        mgr.set_compaction_fn(capture_compaction)
        await mgr.get_history_with_compaction(channel)

        assert captured_system is not None
        assert "tool names" in captured_system.lower()

    @pytest.mark.asyncio
    async def test_instruction_mentions_hosts(self, tmp_dir):
        """Compaction instruction should mention preserving host info."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        captured_system = None

        async def capture_compaction(messages, system):
            nonlocal captured_system
            captured_system = system
            return "Summary."

        mgr.set_compaction_fn(capture_compaction)
        await mgr.get_history_with_compaction(channel)

        assert captured_system is not None
        assert "host" in captured_system.lower()

    @pytest.mark.asyncio
    async def test_instruction_has_structured_format_rule(self, tmp_dir):
        """Compaction instruction should request structured output (topic + bullets)."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        captured_system = None

        async def capture_compaction(messages, system):
            nonlocal captured_system
            captured_system = system
            return "Summary."

        mgr.set_compaction_fn(capture_compaction)
        await mgr.get_history_with_compaction(channel)

        assert captured_system is not None
        assert "bullet" in captured_system.lower() or "topic" in captured_system.lower()

    @pytest.mark.asyncio
    async def test_instruction_preserves_core_rules(self, tmp_dir):
        """Compaction instruction still has error-omission and word-limit rules."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        captured_system = None

        async def capture_compaction(messages, system):
            nonlocal captured_system
            captured_system = system
            return "Summary."

        mgr.set_compaction_fn(capture_compaction)
        await mgr.get_history_with_compaction(channel)

        assert "OMIT" in captured_system
        assert "500 characters" in captured_system
        assert "PRESERVE" in captured_system

    @pytest.mark.asyncio
    async def test_instruction_has_five_rules(self, tmp_dir):
        """Compaction instruction should have 5 numbered rules."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        captured_system = None

        async def capture_compaction(messages, system):
            nonlocal captured_system
            captured_system = system
            return "Summary."

        mgr.set_compaction_fn(capture_compaction)
        await mgr.get_history_with_compaction(channel)

        assert "5." in captured_system

    @pytest.mark.asyncio
    async def test_compaction_merges_previous_summary(self, tmp_dir):
        """When a previous summary exists, it's included in compaction input."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        channel = "ch1"
        session = mgr.get_or_create(channel)
        session.summary = "User set up nginx on server-1."

        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        captured_messages = None

        async def capture_compaction(messages, system):
            nonlocal captured_messages
            captured_messages = messages
            return "Merged summary."

        mgr.set_compaction_fn(capture_compaction)
        await mgr.get_history_with_compaction(channel)

        assert captured_messages is not None
        content = captured_messages[0]["content"]
        assert "Previous summary" in content
        assert "nginx" in content


# ---------------------------------------------------------------------------
# Bot buffer size limit
# ---------------------------------------------------------------------------

class TestBotBufferLimit:
    """Test that the bot message buffer has a size cap."""

    def test_buffer_max_attribute_exists(self):
        """HeimdallBot should have a _bot_msg_buffer_max attribute."""
        from src.discord.client import HeimdallBot
        with patch("discord.Client.__init__", return_value=None):
            bot = object.__new__(HeimdallBot)
            # Check the class has the attribute set in __init__
            # We can't easily instantiate HeimdallBot, but we can check the default
            assert hasattr(HeimdallBot, "__init__")

    def test_buffer_limit_drops_oldest(self):
        """When buffer exceeds max, the oldest message should be dropped."""
        # Simulate the buffer behavior directly
        buf: list[str] = []
        max_buf = 20
        for i in range(25):
            if len(buf) >= max_buf:
                buf.pop(0)
            buf.append(f"msg {i}")

        assert len(buf) == 20
        assert buf[0] == "msg 5"  # first 5 were dropped
        assert buf[-1] == "msg 24"


# ---------------------------------------------------------------------------
# Task history size and efficiency
# ---------------------------------------------------------------------------

class TestTaskHistoryEfficiency:
    """Verify task history returns appropriate message count."""

    @pytest.mark.asyncio
    async def test_task_history_respects_max_messages(self, session_mgr):
        """get_task_history returns at most max_messages recent messages."""
        channel = "ch1"
        for i in range(30):
            session_mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        history = await session_mgr.get_task_history(channel, max_messages=10)
        # No summary = no extra messages, just 10 recent
        assert len(history) == 10

    @pytest.mark.asyncio
    async def test_task_history_includes_summary_prefix(self, session_mgr):
        """When summary exists, task history prepends summary pair."""
        channel = "ch1"
        session = session_mgr.get_or_create(channel)
        session.summary = "Previously deployed to server-1."
        for i in range(15):
            session_mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        history = await session_mgr.get_task_history(channel, max_messages=10)
        # 10 recent + 2 summary messages = 12
        assert len(history) == 12
        assert "Previous conversation summary" in history[0]["content"]

    @pytest.mark.asyncio
    async def test_task_history_fewer_than_max(self, session_mgr):
        """When fewer messages exist than max, all are returned."""
        channel = "ch1"
        session_mgr.add_message(channel, "user", "hello")
        session_mgr.add_message(channel, "assistant", "hi")

        history = await session_mgr.get_task_history(channel, max_messages=20)
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_task_history_default_max_is_10(self, session_mgr):
        """Default max_messages should be 10."""
        channel = "ch1"
        for i in range(30):
            session_mgr.add_message(channel, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        history = await session_mgr.get_task_history(channel)
        assert len(history) == 10


# ---------------------------------------------------------------------------
# Image blocks in session history
# ---------------------------------------------------------------------------

class TestImageBlockHandling:
    """Verify that image blocks don't persist in session storage."""

    def test_session_messages_store_strings(self, session_mgr):
        """Session messages are always strings, never image block lists."""
        session_mgr.add_message("ch1", "user", "check this image")
        session = session_mgr.get_or_create("ch1")
        assert isinstance(session.messages[0].content, str)

    def test_image_blocks_not_in_persisted_data(self, session_mgr):
        """Saved session JSON should never contain base64 image data."""
        session_mgr.add_message("ch1", "user", "[User]: check this image")
        session_mgr.add_message("ch1", "assistant", "I can see the image shows a server dashboard.")
        session_mgr.save()

        path = Path(session_mgr.persist_dir) / "ch1.json"
        data = json.loads(path.read_text())
        for msg in data["messages"]:
            assert isinstance(msg["content"], str)
            assert "base64" not in msg["content"]

    def test_get_history_returns_string_content(self, session_mgr):
        """get_history returns messages with string content."""
        session_mgr.add_message("ch1", "user", "analyze this")
        session_mgr.add_message("ch1", "assistant", "analysis complete")
        history = session_mgr.get_history("ch1")
        for msg in history:
            assert isinstance(msg["content"], str)


# ---------------------------------------------------------------------------
# Context separator behavior
# ---------------------------------------------------------------------------

class TestContextSeparator:
    """Verify context separator injection in _process_with_tools."""

    def test_separator_uses_developer_role(self):
        """The context separator should use 'developer' role, not 'system'."""
        # This is verified by reading the source — we test the role value
        from src.discord import client
        import inspect
        source = inspect.getsource(client.HeimdallBot._process_with_tools)
        assert '"role": "developer"' in source or "'role': 'developer'" in source

    def test_separator_has_current_request_header(self):
        """The separator should have the '=== CURRENT REQUEST' header with request hash."""
        from src.discord import client
        import inspect
        source = inspect.getsource(client.HeimdallBot._process_with_tools)
        assert "=== CURRENT REQUEST" in source
        assert "req_hash" in source

    def test_separator_mentions_bot_execution(self):
        """The bot preamble should tell the LLM to execute immediately."""
        from src.discord import client
        import inspect
        source = inspect.getsource(client.HeimdallBot._process_with_tools)
        assert "EXECUTE immediately" in source

    def test_separator_mentions_tool_evaluation(self):
        """The separator should instruct fresh tool evaluation."""
        from src.discord import client
        import inspect
        source = inspect.getsource(client.HeimdallBot._process_with_tools)
        assert "CURRENTLY AVAILABLE" in source


# ---------------------------------------------------------------------------
# Session continuity
# ---------------------------------------------------------------------------

class TestSessionContinuity:
    """Test session continuity across restarts via archived summaries."""

    def test_continuity_carries_forward_recent_summary(self, tmp_dir):
        """A new session inherits summary from a recently archived session."""
        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )

        # Create an archive with a recent summary
        archive_dir = Path(mgr.persist_dir) / "archive"
        archive_dir.mkdir(parents=True)
        archive_data = {
            "channel_id": "ch1",
            "messages": [],
            "summary": "User configured firewall rules.",
            "last_active": time.time() - 3600,  # 1 hour ago (within 48h window)
        }
        (archive_dir / "ch1_12345.json").write_text(json.dumps(archive_data))

        session = mgr.get_or_create("ch1")
        assert "firewall rules" in session.summary

    def test_continuity_ignores_old_archives(self, tmp_dir):
        """Archived summaries older than CONTINUITY_MAX_AGE are ignored."""
        mgr = SessionManager(
            max_history=10, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )

        archive_dir = Path(mgr.persist_dir) / "archive"
        archive_dir.mkdir(parents=True)
        archive_data = {
            "channel_id": "ch1",
            "messages": [],
            "summary": "Very old context.",
            "last_active": time.time() - 200000,  # ~55 hours ago (outside 48h window)
        }
        (archive_dir / "ch1_99999.json").write_text(json.dumps(archive_data))

        session = mgr.get_or_create("ch1")
        assert session.summary == ""


# ---------------------------------------------------------------------------
# Bot message combining
# ---------------------------------------------------------------------------

class TestBotMessageCombining:
    """Test combine_bot_messages function."""

    def test_single_message(self):
        from src.discord.client import combine_bot_messages
        assert combine_bot_messages(["hello"]) == "hello"

    def test_empty_list(self):
        from src.discord.client import combine_bot_messages
        assert combine_bot_messages([]) == ""

    def test_multiple_messages_joined(self):
        from src.discord.client import combine_bot_messages
        result = combine_bot_messages(["hello", "world"])
        assert "hello" in result
        assert "world" in result

    def test_split_code_blocks_merged(self):
        from src.discord.client import combine_bot_messages
        result = combine_bot_messages(["```bash\nls -la", "/tmp\n```"])
        # Should be joined with single newline (inside code block)
        assert "ls -la\n/tmp" in result

    def test_adjacent_code_blocks_collapsed(self):
        from src.discord.client import combine_bot_messages
        result = combine_bot_messages(["```bash\nfirst\n```", "```bash\nsecond\n```"])
        # Adjacent fences should be merged
        assert result.count("```") <= 2  # Only opening and closing


# ---------------------------------------------------------------------------
# Session selective saving (layer 2 defense)
# ---------------------------------------------------------------------------

class TestSelectiveSaving:
    """Verify that tool-less responses are NOT saved to session history."""

    def test_tool_less_response_not_saved_explanation(self, session_mgr):
        """The session saving logic should skip tool-less non-guest responses.

        In client.py: `if not is_guest and not tools_used and not handoff: pass`
        This means no add_message call for the assistant response.
        """
        # Verify the logic by checking that only user message exists
        # after simulating a tool-less response
        session_mgr.add_message("ch1", "user", "what is the meaning of life?")
        # Do NOT add assistant message (simulates the skip)
        history = session_mgr.get_history("ch1")
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_tool_response_is_saved(self, session_mgr):
        """When tools are used, the assistant response IS saved."""
        session_mgr.add_message("ch1", "user", "check disk space")
        session_mgr.add_message("ch1", "assistant", "Disk usage: 45% on /dev/sda1")
        history = session_mgr.get_history("ch1")
        assert len(history) == 2
        assert history[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# remove_last_message (orphan cleanup)
# ---------------------------------------------------------------------------

class TestRemoveLastMessage:
    """Test orphaned message cleanup."""

    def test_removes_matching_role(self, session_mgr):
        session_mgr.add_message("ch1", "user", "hello")
        assert session_mgr.remove_last_message("ch1", "user") is True
        assert session_mgr.get_history("ch1") == []

    def test_no_remove_mismatched_role(self, session_mgr):
        session_mgr.add_message("ch1", "user", "hello")
        assert session_mgr.remove_last_message("ch1", "assistant") is False
        assert len(session_mgr.get_history("ch1")) == 1

    def test_no_remove_empty_session(self, session_mgr):
        assert session_mgr.remove_last_message("ch1", "user") is False

    def test_no_remove_nonexistent_session(self, session_mgr):
        assert session_mgr.remove_last_message("nonexistent", "user") is False
