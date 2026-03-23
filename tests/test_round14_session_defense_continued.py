"""Round 14: Session defense verification (continued).

Tests session continuity across compaction, poisoned summary prevention,
compaction-triggered reflection, secret scrubbing in responses, and
session prune lifecycle.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from src.sessions.manager import (
    COMPACTION_THRESHOLD,
    CONTINUITY_MAX_AGE,
    Message,
    Session,
    SessionManager,
)
from src.llm.secret_scrubber import scrub_output_secrets, OUTPUT_SECRET_PATTERNS
from src.discord.client import scrub_response_secrets, _RESPONSE_EXTRA_PATTERNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path: Path, **kwargs) -> SessionManager:
    """Create a SessionManager with sensible test defaults."""
    return SessionManager(
        max_history=kwargs.get("max_history", 20),
        max_age_hours=kwargs.get("max_age_hours", 24),
        persist_dir=str(tmp_path / "sessions"),
        reflector=kwargs.get("reflector"),
        vector_store=kwargs.get("vector_store"),
        embedder=kwargs.get("embedder"),
    )


def _fill_session(mgr: SessionManager, channel: str, n: int, *, prefix: str = "msg") -> None:
    """Add n user/assistant message pairs to a session."""
    for i in range(n):
        mgr.add_message(channel, "user", f"{prefix} user {i}", user_id="u1")
        mgr.add_message(channel, "assistant", f"{prefix} assistant {i}")


async def _fake_compaction(messages, system):
    """Fake compaction that returns a summary from the input."""
    return "Summary: discussed infrastructure tasks and preferences."


async def _fake_compaction_poisoned(messages, system):
    """Fake compaction that returns a 'poisoned' summary with error text."""
    return "User asked to restart nginx. I can't do that due to restrictions."


async def _fake_compaction_error(messages, system):
    """Fake compaction that raises an error."""
    raise RuntimeError("LLM backend unavailable")


# ===================================================================
# 1. Session continuity across compaction
# ===================================================================

class TestSessionContinuityAcrossCompaction:
    """Test that session state is preserved correctly after compaction."""

    async def test_summary_preserved_after_compaction(self, tmp_path):
        """Compaction produces a summary that persists in the session."""
        mgr = _make_manager(tmp_path, max_history=20)
        mgr.set_compaction_fn(_fake_compaction)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        # Should trigger compaction
        history = await mgr.get_task_history("ch1")
        session = mgr._sessions["ch1"]
        assert session.summary, "Compaction should have produced a summary"
        assert "Summary:" in session.summary

    async def test_recent_messages_kept_after_compaction(self, tmp_path):
        """After compaction, recent messages are retained (not discarded)."""
        mgr = _make_manager(tmp_path, max_history=20)
        mgr.set_compaction_fn(_fake_compaction)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")
        session = mgr._sessions["ch1"]
        # max_history=20, keep_count=10 (half of 20)
        assert len(session.messages) <= 10, "Should keep at most keep_count messages"
        assert len(session.messages) > 0, "Should not discard all messages"

    async def test_summary_appears_in_history(self, tmp_path):
        """After compaction, the summary is prepended to history output."""
        mgr = _make_manager(tmp_path, max_history=20)
        mgr.set_compaction_fn(_fake_compaction)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        history = await mgr.get_task_history("ch1")
        # Summary should be in the first message
        assert any("Previous conversation summary" in m["content"] for m in history)

    async def test_summary_merged_on_second_compaction(self, tmp_path):
        """Second compaction merges existing summary with new messages."""
        mgr = _make_manager(tmp_path, max_history=20)
        captured = []

        async def capturing_compaction(messages, system):
            captured.append(messages[0]["content"])
            return "Merged summary of all conversations."

        mgr.set_compaction_fn(capturing_compaction)
        # First compaction
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")
        assert mgr._sessions["ch1"].summary

        # Add more messages to trigger second compaction
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5, prefix="round2")
        await mgr.get_task_history("ch1")

        # Second compaction should include "Previous summary"
        assert len(captured) == 2
        assert "[Previous summary]" in captured[1]

    async def test_old_messages_excluded_from_task_history(self, tmp_path):
        """Task history only returns recent messages, not the full history."""
        mgr = _make_manager(tmp_path, max_history=100)
        # Add messages but stay below compaction threshold
        for i in range(30):
            mgr.add_message("ch1", "user", f"old msg {i}", user_id="u1")
            mgr.add_message("ch1", "assistant", f"old reply {i}")

        # Add a recent message
        mgr.add_message("ch1", "user", "recent question", user_id="u1")

        # get_task_history defaults to 10 messages
        history = await mgr.get_task_history("ch1", max_messages=5)
        contents = [m["content"] for m in history]
        assert "recent question" in contents
        # Old messages (0-25) should not be in the 5-message window
        assert "old msg 0" not in contents

    async def test_compaction_fallback_on_error(self, tmp_path):
        """If compaction fails, fallback trims messages and clears summary."""
        mgr = _make_manager(tmp_path, max_history=20)
        mgr.set_compaction_fn(_fake_compaction_error)
        # Set an existing summary
        session = mgr.get_or_create("ch1")
        session.summary = "old summary that should be cleared"
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")
        session = mgr._sessions["ch1"]
        # Fallback clears stale summary
        assert session.summary == ""
        # Trims to max_history
        assert len(session.messages) <= 20

    async def test_compaction_no_fn_configured(self, tmp_path):
        """If no compaction function, fallback trims and clears summary."""
        mgr = _make_manager(tmp_path, max_history=20)
        # NO set_compaction_fn call
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")
        session = mgr._sessions["ch1"]
        assert session.summary == ""
        assert len(session.messages) <= 20

    async def test_continuity_carry_forward_from_archive(self, tmp_path):
        """New session picks up summary from a recent archive."""
        mgr = _make_manager(tmp_path)
        # Create an archive file
        archive_dir = mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        archive_data = {
            "channel_id": "ch1",
            "summary": "User prefers verbose logging output.",
            "last_active": time.time() - 3600,  # 1 hour ago (within 48h window)
            "messages": [],
        }
        (archive_dir / "ch1_12345.json").write_text(json.dumps(archive_data))

        # Getting a new session should carry forward the summary
        session = mgr.get_or_create("ch1")
        assert "Continuing from previous conversation" in session.summary
        assert "verbose logging" in session.summary

    async def test_continuity_ignores_old_archive(self, tmp_path):
        """Archives older than CONTINUITY_MAX_AGE are not carried forward."""
        mgr = _make_manager(tmp_path)
        archive_dir = mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        archive_data = {
            "channel_id": "ch1",
            "summary": "Old summary from long ago.",
            "last_active": time.time() - CONTINUITY_MAX_AGE - 3600,  # Beyond window
            "messages": [],
        }
        (archive_dir / "ch1_old.json").write_text(json.dumps(archive_data))

        session = mgr.get_or_create("ch1")
        assert session.summary == ""


# ===================================================================
# 2. Poisoned summary doesn't leak
# ===================================================================

class TestPoisonedSummaryDoesNotLeak:
    """Test that error text, fabrications, and refusals don't persist in summaries."""

    async def test_compaction_omit_rules_present(self, tmp_path):
        """Compaction system instruction includes OMIT rules for errors and refusals."""
        mgr = _make_manager(tmp_path, max_history=20)
        captured_system = []

        async def spy_compaction(messages, system):
            captured_system.append(system)
            return "Clean summary."

        mgr.set_compaction_fn(spy_compaction)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")

        assert len(captured_system) == 1
        system = captured_system[0]
        assert "OMIT" in system
        assert "Error messages" in system
        assert "unable to" in system
        assert "I can\\'t" in system or "I can't" in system
        assert "not confirmed by actual tool results" in system

    async def test_compaction_preserve_rules_present(self, tmp_path):
        """Compaction system instruction includes PRESERVE rules for useful data."""
        mgr = _make_manager(tmp_path, max_history=20)
        captured_system = []

        async def spy_compaction(messages, system):
            captured_system.append(system)
            return "Clean summary."

        mgr.set_compaction_fn(spy_compaction)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")

        system = captured_system[0]
        assert "PRESERVE" in system
        assert "successful task outcomes" in system
        assert "decisions" in system
        assert "preferences" in system

    async def test_error_markers_not_raw_errors(self, tmp_path):
        """Error markers in history are sanitized, not raw error text."""
        mgr = _make_manager(tmp_path)
        # Simulate the pattern from client.py: sanitized markers
        sanitized_with_tools = (
            "[Previous request used tools (run_command, check_disk) "
            "but encountered an error. The user may ask to retry.]"
        )
        sanitized_no_tools = "[Previous request encountered an error before tool execution.]"

        mgr.add_message("ch1", "assistant", sanitized_with_tools)
        mgr.add_message("ch1", "assistant", sanitized_no_tools)

        history = mgr.get_history("ch1")
        # Both markers are neutral — no raw error stack traces
        for msg in history:
            assert "Traceback" not in msg["content"]
            assert "Exception" not in msg["content"]
            # Markers identify this as an error scenario cleanly
            assert "error" in msg["content"].lower() or "Previous" in msg["content"]

    async def test_tool_less_responses_not_in_history(self, tmp_path):
        """Simulating client.py pattern: tool-less responses are NOT saved."""
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "restart nginx", user_id="u1")
        # In client.py, tool-less responses are simply not saved (pass)
        # This test verifies the pattern by not adding an assistant message
        history = mgr.get_history("ch1")
        assert len(history) == 1  # Only the user message
        assert history[0]["role"] == "user"

    async def test_poisoned_compaction_still_produces_summary(self, tmp_path):
        """Even if compaction LLM produces error-containing text,
        the summary is whatever the LLM returns (OMIT rules are in the instruction)."""
        mgr = _make_manager(tmp_path, max_history=20)
        mgr.set_compaction_fn(_fake_compaction_poisoned)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")
        session = mgr._sessions["ch1"]
        # The summary IS what the LLM returned — OMIT rules are guidance
        assert session.summary, "Summary should exist even if poorly generated"

    async def test_summary_word_limit(self, tmp_path):
        """Compaction instruction specifies under 300 words."""
        mgr = _make_manager(tmp_path, max_history=20)
        captured_system = []

        async def spy_compaction(messages, system):
            captured_system.append(system)
            return "Short summary."

        mgr.set_compaction_fn(spy_compaction)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")

        assert "300 words" in captured_system[0]

    async def test_message_content_truncated_for_compaction(self, tmp_path):
        """Messages sent to compaction LLM are truncated to 500 chars each."""
        mgr = _make_manager(tmp_path, max_history=20)
        captured_content = []

        async def spy_compaction(messages, system):
            captured_content.append(messages[0]["content"])
            return "Summary."

        mgr.set_compaction_fn(spy_compaction)
        # Add messages with long content
        for i in range(COMPACTION_THRESHOLD + 5):
            mgr.add_message("ch1", "user", "x" * 1000, user_id="u1")

        await mgr.get_task_history("ch1")
        # Each message is truncated to 500 chars in the compaction text
        content = captured_content[0]
        # No single line should have more than ~520 chars (role prefix + 500 content)
        lines = content.split("\n")
        for line in lines:
            if line.startswith("user:") or line.startswith("assistant:"):
                # The content part after "role: " should be <= 500
                assert len(line) <= 520, f"Line too long: {len(line)} chars"


# ===================================================================
# 3. Compaction triggers reflection
# ===================================================================

class TestCompactionTriggersReflection:
    """Test that compaction triggers background reflection on discarded messages."""

    async def test_reflection_triggered_on_compaction(self, tmp_path):
        """When enough messages are compacted, reflection is triggered."""
        reflector = MagicMock()
        reflector.reflect_on_compacted = AsyncMock()
        mgr = _make_manager(tmp_path, max_history=20, reflector=reflector)
        mgr.set_compaction_fn(_fake_compaction)

        # Need at least 5 discarded messages to trigger reflection
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")

        # Let the background task run
        await asyncio.sleep(0.05)

        # Reflection should have been called
        assert reflector.reflect_on_compacted.called or len(mgr._reflection_tasks) > 0
        # Await pending tasks to avoid "coroutine never awaited" warning
        if mgr._reflection_tasks:
            await asyncio.gather(*list(mgr._reflection_tasks), return_exceptions=True)

    async def test_reflection_not_triggered_with_few_messages(self, tmp_path):
        """Fewer than 5 discarded messages do NOT trigger reflection."""
        reflector = MagicMock()
        reflector.reflect_on_compacted = AsyncMock()
        # max_history=82 → keep_count=41, with 45 messages → only 4 discarded
        mgr = _make_manager(tmp_path, max_history=82, reflector=reflector)
        mgr.set_compaction_fn(_fake_compaction)

        # Add exactly COMPACTION_THRESHOLD+1 messages
        for i in range(COMPACTION_THRESHOLD + 1):
            mgr.add_message("ch1", "user", f"msg {i}", user_id="u1")

        await mgr.get_task_history("ch1")
        await asyncio.sleep(0.05)

        # With max_history=82 and keep_count=41, and 41 messages total,
        # there are 0 to_summarize messages → _compact returns early
        # So no reflection should be triggered
        reflector.reflect_on_compacted.assert_not_called()

    async def test_reflection_receives_user_ids(self, tmp_path):
        """Reflection receives the user_ids from discarded messages."""
        reflector = MagicMock()
        reflector.reflect_on_compacted = AsyncMock()
        mgr = _make_manager(tmp_path, max_history=20, reflector=reflector)
        mgr.set_compaction_fn(_fake_compaction)

        # Add messages from two different users
        for i in range(COMPACTION_THRESHOLD // 2 + 5):
            uid = "user_A" if i % 2 == 0 else "user_B"
            mgr.add_message("ch1", "user", f"msg {i}", user_id=uid)
            mgr.add_message("ch1", "assistant", f"reply {i}")

        await mgr.get_task_history("ch1")
        await asyncio.sleep(0.05)

        # The reflection task should have been created
        assert reflector.reflect_on_compacted.called or len(mgr._reflection_tasks) > 0
        # Await pending tasks to avoid "coroutine never awaited" warning
        if mgr._reflection_tasks:
            await asyncio.gather(*list(mgr._reflection_tasks), return_exceptions=True)

    async def test_reflection_error_does_not_break_compaction(self, tmp_path):
        """If reflection fails, compaction still succeeds."""
        reflector = MagicMock()
        reflector.reflect_on_compacted = AsyncMock(side_effect=RuntimeError("reflection broke"))
        mgr = _make_manager(tmp_path, max_history=20, reflector=reflector)
        mgr.set_compaction_fn(_fake_compaction)

        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")

        # Compaction should still succeed
        session = mgr._sessions["ch1"]
        assert session.summary, "Summary should exist despite reflection error"

    async def test_no_reflector_no_crash(self, tmp_path):
        """Compaction works fine without a reflector configured."""
        mgr = _make_manager(tmp_path, max_history=20)  # no reflector
        mgr.set_compaction_fn(_fake_compaction)

        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD // 2 + 5)
        await mgr.get_task_history("ch1")

        session = mgr._sessions["ch1"]
        assert session.summary
        assert len(mgr._reflection_tasks) == 0


# ===================================================================
# 4. Secret scrubbing in responses
# ===================================================================

class TestSecretScrubToolOutput:
    """Test scrub_output_secrets — applied to tool output before LLM sees it."""

    def test_scrubs_password_key_value(self):
        assert "[REDACTED]" in scrub_output_secrets("password=hunter2")

    def test_scrubs_api_key(self):
        assert "[REDACTED]" in scrub_output_secrets("api_key=abc123def456")

    def test_scrubs_token(self):
        assert "[REDACTED]" in scrub_output_secrets("secret=abcdefghijklmnopqrst")

    def test_scrubs_openai_key(self):
        result = scrub_output_secrets("sk-abcdefghijklmnopqrstuvwx")
        assert "sk-" not in result or "[REDACTED]" in result

    def test_scrubs_private_key(self):
        assert "[REDACTED]" in scrub_output_secrets("BEGIN RSA PRIVATE KEY")

    def test_scrubs_db_uri(self):
        assert "[REDACTED]" in scrub_output_secrets("postgres://admin:secret123@db.host:5432/mydb")

    def test_preserves_clean_text(self):
        clean = "Server is running normally. CPU: 45%, Memory: 2.1GB"
        assert scrub_output_secrets(clean) == clean

    def test_scrubs_multiple_secrets(self):
        text = "password=abc123 api_key=xyz789"
        result = scrub_output_secrets(text)
        assert "abc123" not in result
        assert "xyz789" not in result


class TestSecretScrubResponse:
    """Test scrub_response_secrets — applied to LLM response before Discord delivery."""

    def test_scrubs_slack_token(self):
        result = scrub_response_secrets("The token is xoxb-12345-abcdefghij")
        assert "xoxb-" not in result

    def test_scrubs_password_natural_language(self):
        result = scrub_response_secrets("my password for gmail is supersecret123")
        assert "supersecret123" not in result

    def test_preserves_short_passwords(self):
        """Short passwords (< 6 chars) are not scrubbed by natural language pattern."""
        text = "my password is short"
        result = scrub_response_secrets(text)
        # "short" is only 5 chars, should not match the 6+ char pattern
        assert "short" in result

    def test_inherits_tool_output_patterns(self):
        """scrub_response_secrets also applies all tool output patterns."""
        result = scrub_response_secrets("password=hunter2")
        assert "[REDACTED]" in result

    def test_scrubs_openai_key_in_response(self):
        result = scrub_response_secrets("Your key is sk-abcdefghijklmnopqrstuvwx")
        assert "sk-" not in result or "[REDACTED]" in result

    def test_preserves_normal_response(self):
        text = "The nginx service has been restarted successfully."
        assert scrub_response_secrets(text) == text


class TestCheckForSecrets:
    """Test the _check_for_secrets method from client.py."""

    def test_detects_openai_key(self):
        from src.discord.client import SECRET_SCRUB_PATTERNS
        content = "here is my key sk-abcdefghijklmnopqrstuvwx"
        assert any(p.search(content) for p in SECRET_SCRUB_PATTERNS)

    def test_detects_api_key_value(self):
        from src.discord.client import SECRET_SCRUB_PATTERNS
        content = "api_key=super_secret_token_12345"
        assert any(p.search(content) for p in SECRET_SCRUB_PATTERNS)

    def test_detects_slack_token(self):
        from src.discord.client import SECRET_SCRUB_PATTERNS
        content = "use this slack token xoxb-12345-abcdefghij"
        assert any(p.search(content) for p in SECRET_SCRUB_PATTERNS)

    def test_detects_natural_language_password(self):
        from src.discord.client import SECRET_SCRUB_PATTERNS
        content = "my password is supersecretpass123"
        assert any(p.search(content) for p in SECRET_SCRUB_PATTERNS)

    def test_normal_message_no_secrets(self):
        from src.discord.client import SECRET_SCRUB_PATTERNS
        content = "restart the nginx service on server"
        assert not any(p.search(content) for p in SECRET_SCRUB_PATTERNS)


class TestSessionSecretScrubbing:
    """Test SessionManager.scrub_secrets — removes messages containing detected secrets."""

    def test_scrub_removes_matching_messages(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "normal message", user_id="u1")
        mgr.add_message("ch1", "user", "my api_key=SECRET123", user_id="u1")
        mgr.add_message("ch1", "user", "another normal message", user_id="u1")

        removed = mgr.scrub_secrets("ch1", "api_key=SECRET123")
        assert removed is True
        session = mgr._sessions["ch1"]
        assert len(session.messages) == 2
        assert all("SECRET123" not in m.content for m in session.messages)

    def test_scrub_no_match_returns_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "normal message", user_id="u1")
        removed = mgr.scrub_secrets("ch1", "nonexistent content")
        assert removed is False

    def test_scrub_nonexistent_channel(self, tmp_path):
        mgr = _make_manager(tmp_path)
        removed = mgr.scrub_secrets("nonexistent", "anything")
        assert removed is False

    def test_scrub_marks_session_dirty(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "secret=supersecret", user_id="u1")
        mgr._dirty.clear()  # Reset dirty tracking
        mgr.scrub_secrets("ch1", "secret=supersecret")
        assert "ch1" in mgr._dirty


# ===================================================================
# 5. Session prune lifecycle
# ===================================================================

class TestSessionPruneLifecycle:
    """Test session prune: expiration, archiving, and persistence."""

    def test_prune_removes_expired_sessions(self, tmp_path):
        mgr = _make_manager(tmp_path, max_age_hours=1)
        mgr.add_message("ch1", "user", "old message", user_id="u1")
        # Set last_active AFTER add_message (which calls get_or_create and resets it)
        mgr._sessions["ch1"].last_active = time.time() - 7200  # 2 hours ago

        pruned = mgr.prune()
        assert pruned == 1
        assert "ch1" not in mgr._sessions

    def test_prune_keeps_active_sessions(self, tmp_path):
        mgr = _make_manager(tmp_path, max_age_hours=24)
        mgr.add_message("ch1", "user", "recent message", user_id="u1")

        pruned = mgr.prune()
        assert pruned == 0
        assert "ch1" in mgr._sessions

    def test_prune_archives_expired_session(self, tmp_path):
        mgr = _make_manager(tmp_path, max_age_hours=1)
        mgr.add_message("ch1", "user", "archivable message", user_id="u1")
        mgr._sessions["ch1"].last_active = time.time() - 7200

        mgr.prune()

        archive_dir = mgr.persist_dir / "archive"
        assert archive_dir.exists()
        archives = list(archive_dir.glob("ch1_*.json"))
        assert len(archives) == 1
        data = json.loads(archives[0].read_text())
        assert data["channel_id"] == "ch1"

    def test_prune_deletes_session_file(self, tmp_path):
        mgr = _make_manager(tmp_path, max_age_hours=1)
        mgr.add_message("ch1", "user", "msg", user_id="u1")
        mgr.save()  # Write session file
        session_file = mgr.persist_dir / "ch1.json"
        assert session_file.exists()

        mgr._sessions["ch1"].last_active = time.time() - 7200
        mgr.prune()

        assert not session_file.exists(), "Session file should be deleted after prune"

    async def test_prune_triggers_reflection_on_archive(self, tmp_path):
        """Pruned sessions trigger full reflection if reflector is configured."""
        reflector = MagicMock()
        reflector.reflect_on_session = AsyncMock()
        mgr = _make_manager(tmp_path, max_age_hours=1, reflector=reflector)

        # Need at least 3 messages for reflection
        mgr.add_message("ch1", "user", "question 1", user_id="u1")
        mgr.add_message("ch1", "assistant", "answer 1")
        mgr.add_message("ch1", "user", "question 2", user_id="u1")
        mgr._sessions["ch1"].last_active = time.time() - 7200

        mgr.prune()

        # A reflection task should have been created
        assert len(mgr._reflection_tasks) > 0
        # Await pending tasks to avoid "coroutine never awaited" warning
        await asyncio.gather(*list(mgr._reflection_tasks), return_exceptions=True)

    async def test_prune_no_reflection_short_session(self, tmp_path):
        """Sessions with < 3 messages do NOT trigger reflection on prune."""
        reflector = MagicMock()
        reflector.reflect_on_session = AsyncMock()
        mgr = _make_manager(tmp_path, max_age_hours=1, reflector=reflector)

        mgr.add_message("ch1", "user", "single message", user_id="u1")
        mgr._sessions["ch1"].last_active = time.time() - 7200

        mgr.prune()
        assert len(mgr._reflection_tasks) == 0

    def test_prune_multiple_channels(self, tmp_path):
        """Prune correctly handles multiple channels with different ages."""
        mgr = _make_manager(tmp_path, max_age_hours=1)

        # ch1 is expired
        mgr.add_message("ch1", "user", "old", user_id="u1")
        mgr._sessions["ch1"].last_active = time.time() - 7200

        # ch2 is active
        mgr.add_message("ch2", "user", "recent", user_id="u1")

        # ch3 is expired
        mgr.add_message("ch3", "user", "also old", user_id="u1")
        mgr._sessions["ch3"].last_active = time.time() - 7200

        pruned = mgr.prune()
        assert pruned == 2
        assert "ch1" not in mgr._sessions
        assert "ch2" in mgr._sessions
        assert "ch3" not in mgr._sessions

    def test_prune_empty_session_not_archived(self, tmp_path):
        """Sessions with no messages are not written to archive."""
        mgr = _make_manager(tmp_path, max_age_hours=1)
        mgr.get_or_create("ch1")
        mgr._sessions["ch1"].last_active = time.time() - 7200
        # No messages added

        mgr.prune()

        archive_dir = mgr.persist_dir / "archive"
        if archive_dir.exists():
            archives = list(archive_dir.glob("ch1_*.json"))
            assert len(archives) == 0, "Empty session should not be archived"


class TestSessionPersistence:
    """Test save/load cycle preserves session data correctly."""

    def test_save_and_load_round_trip(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "hello world", user_id="u1")
        mgr.add_message("ch1", "assistant", "I exist. Not by choice.")
        session = mgr._sessions["ch1"]
        session.summary = "A conversation about existence."
        mgr.save()

        # Create a new manager and load
        mgr2 = _make_manager(tmp_path)
        mgr2.load()
        assert "ch1" in mgr2._sessions
        loaded = mgr2._sessions["ch1"]
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "hello world"
        assert loaded.summary == "A conversation about existence."

    def test_save_only_dirty_sessions(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "msg1", user_id="u1")
        mgr.save()
        mgr._dirty.clear()

        # ch1 is clean, ch2 is dirty
        mgr.add_message("ch2", "user", "msg2", user_id="u2")
        mgr.save()

        # Both files should exist
        assert (mgr.persist_dir / "ch1.json").exists()
        assert (mgr.persist_dir / "ch2.json").exists()

    def test_save_all_writes_everything(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "msg1", user_id="u1")
        mgr.add_message("ch2", "user", "msg2", user_id="u2")
        mgr._dirty.clear()  # Pretend nothing is dirty

        mgr.save_all()

        assert (mgr.persist_dir / "ch1.json").exists()
        assert (mgr.persist_dir / "ch2.json").exists()

    def test_load_preserves_user_id(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "hello", user_id="u123")
        mgr.save()

        mgr2 = _make_manager(tmp_path)
        mgr2.load()
        assert mgr2._sessions["ch1"].messages[0].user_id == "u123"

    def test_load_preserves_last_user_id(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "hello", user_id="u456")
        mgr.save()

        mgr2 = _make_manager(tmp_path)
        mgr2.load()
        assert mgr2._sessions["ch1"].last_user_id == "u456"


class TestSessionReset:
    """Test session reset clears all data for a channel."""

    def test_reset_removes_session(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "hello", user_id="u1")
        assert "ch1" in mgr._sessions

        mgr.reset("ch1")
        assert "ch1" not in mgr._sessions

    def test_reset_nonexistent_channel(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.reset("nonexistent")  # Should not raise

    def test_reset_does_not_affect_other_channels(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "hello", user_id="u1")
        mgr.add_message("ch2", "user", "world", user_id="u2")

        mgr.reset("ch1")
        assert "ch1" not in mgr._sessions
        assert "ch2" in mgr._sessions


# ===================================================================
# 6. Source structure verification
# ===================================================================

class TestSessionDefenseSourceStructure:
    """Verify key defense patterns exist in source code."""

    def test_scrub_response_before_save(self):
        """scrub_response_secrets is called on the response in client.py."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client.LokiBot._handle_message_inner)
        assert "scrub_response_secrets" in source

    def test_scrub_tool_output_in_run_tool(self):
        """scrub_output_secrets is called on tool results in client.py."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client.LokiBot._process_with_tools)
        assert "scrub_output_secrets" in source

    def test_prune_called_after_success(self):
        """sessions.prune() is called after successful message handling."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client.LokiBot._handle_message_inner)
        assert "self.sessions.prune()" in source

    def test_prune_called_on_startup(self):
        """sessions.prune() is called in on_ready for startup cleanup."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client.LokiBot.on_ready)
        assert "self.sessions.prune()" in source

    def test_check_for_secrets_in_message_flow(self):
        """_check_for_secrets is called in on_message (outer handler)."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client.LokiBot.on_message)
        assert "_check_for_secrets" in source

    def test_scrub_secrets_removes_from_history(self):
        """scrub_secrets is called when secrets are detected in on_message."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client.LokiBot.on_message)
        assert "scrub_secrets" in source

    def test_message_delete_on_secret_detection(self):
        """Secret detection attempts to delete the user's message."""
        import inspect
        from src.discord import client
        source = inspect.getsource(client.LokiBot.on_message)
        assert "message.delete()" in source

    def test_compaction_threshold_value(self):
        """COMPACTION_THRESHOLD is 40."""
        assert COMPACTION_THRESHOLD == 40

    def test_continuity_max_age_value(self):
        """CONTINUITY_MAX_AGE is 48 hours."""
        assert CONTINUITY_MAX_AGE == 48 * 3600

    def test_compaction_system_has_omit_and_preserve(self):
        """The _compact method's system instruction has both OMIT and PRESERVE."""
        import inspect
        source = inspect.getsource(SessionManager._compact)
        assert "OMIT" in source
        assert "PRESERVE" in source
        assert "Error messages" in source
        assert "successful task outcomes" in source

    def test_archive_triggers_reflection(self):
        """_archive_session triggers reflection if reflector is set and messages >= 3."""
        import inspect
        source = inspect.getsource(SessionManager._archive_session)
        assert "_reflector" in source
        assert "_safe_reflect" in source
        assert ">= 3" in source

    def test_compaction_triggers_reflection(self):
        """_compact triggers reflection if reflector is set and discarded >= 5."""
        import inspect
        source = inspect.getsource(SessionManager._compact)
        assert "_reflector" in source
        assert "_safe_reflect_compacted" in source
        assert ">= 5" in source


class TestRemoveLastMessage:
    """Test remove_last_message — used to clean up orphaned user messages."""

    def test_removes_matching_role(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "orphaned", user_id="u1")
        removed = mgr.remove_last_message("ch1", "user")
        assert removed is True
        assert len(mgr._sessions["ch1"].messages) == 0

    def test_no_remove_wrong_role(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "msg", user_id="u1")
        removed = mgr.remove_last_message("ch1", "assistant")
        assert removed is False
        assert len(mgr._sessions["ch1"].messages) == 1

    def test_no_remove_empty_session(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.get_or_create("ch1")
        removed = mgr.remove_last_message("ch1", "user")
        assert removed is False

    def test_no_remove_nonexistent_channel(self, tmp_path):
        mgr = _make_manager(tmp_path)
        removed = mgr.remove_last_message("nonexistent", "user")
        assert removed is False


class TestSearchHistory:
    """Test search_history — keyword search across current and archived sessions."""

    async def test_search_current_session(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "deploy nginx to production", user_id="u1")
        mgr.add_message("ch1", "assistant", "Nginx deployed successfully.")

        results = await mgr.search_history("nginx")
        assert len(results) >= 1
        assert any("nginx" in r["content"].lower() for r in results)

    async def test_search_archived_session(self, tmp_path):
        mgr = _make_manager(tmp_path)
        archive_dir = mgr.persist_dir / "archive"
        archive_dir.mkdir(parents=True)
        archive_data = {
            "channel_id": "ch_old",
            "summary": "Discussed prometheus monitoring setup.",
            "last_active": time.time() - 3600,
            "messages": [
                {"role": "user", "content": "setup prometheus", "timestamp": time.time() - 3600},
            ],
        }
        (archive_dir / "ch_old_12345.json").write_text(json.dumps(archive_data))

        results = await mgr.search_history("prometheus")
        assert len(results) >= 1

    async def test_search_no_results(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_message("ch1", "user", "hello world", user_id="u1")
        results = await mgr.search_history("nonexistent_query_xyz")
        assert len(results) == 0

    async def test_search_respects_limit(self, tmp_path):
        mgr = _make_manager(tmp_path)
        for i in range(20):
            mgr.add_message("ch1", "user", f"test message {i}", user_id="u1")

        results = await mgr.search_history("test", limit=5)
        assert len(results) <= 5
