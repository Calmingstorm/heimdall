"""Tests for Round 9: Compaction quality improvements.

Verifies:
- Improved compaction prompt preserves identifiers and outcomes
- Topic tags are requested in the compaction format
- Summary length is enforced (< 500 chars)
- Truncation preserves complete lines
- Compaction prompt instructs to omit intermediate steps
- Source code contains the expected improvements
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.sessions.manager import (  # noqa: E402
    COMPACTION_MAX_CHARS,
    COMPACTION_THRESHOLD,
    SessionManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_session(mgr: SessionManager, channel: str, count: int) -> None:
    """Add *count* alternating user/assistant messages."""
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        mgr.add_message(channel, role, f"msg {i}")


def _fill_realistic_session(mgr: SessionManager, channel: str) -> None:
    """Add realistic infrastructure conversation messages."""
    messages = [
        ("user", "Check disk usage on server-a at 192.168.1.50"),
        ("assistant", "Ran df -h on server-a (192.168.1.50). /dev/sda1 is at 78% usage, /var/log at 92%."),
        ("user", "Clean up old logs on server-a"),
        ("assistant", "Deleted 3.2GB of rotated logs from /var/log on server-a. Usage now 45%."),
        ("user", "Deploy nginx to container web-prod-01"),
        ("assistant", "Installed nginx 1.24.0 on web-prod-01. Config at /etc/nginx/nginx.conf. Listening on port 443."),
        ("user", "Set up SSL cert for api.example.com"),
        ("assistant", "Generated Let's Encrypt cert for api.example.com. Stored at /etc/letsencrypt/live/api.example.com/. Nginx reloaded."),
        ("user", "Check if DNS resolves for api.example.com"),
        ("assistant", "DNS check: api.example.com resolves to 203.0.113.42. TTL 300s. CNAME to lb-east.example.com."),
        ("user", "What was the error on the payment service?"),
        ("assistant", "I can't access the payment service logs right now, the connection timed out."),
        ("user", "Restart the payment service on server-b"),
        ("assistant", "Restarted payment-api.service on server-b (10.0.0.5). Service is now active (running). PID 4521."),
    ]
    for role, content in messages:
        mgr.add_message(channel, role, content)
    # Pad to exceed threshold
    for i in range(COMPACTION_THRESHOLD - len(messages) + 5):
        role = "user" if i % 2 == 0 else "assistant"
        mgr.add_message(channel, role, f"padding msg {i}")


# ---------------------------------------------------------------------------
# COMPACTION_MAX_CHARS constant
# ---------------------------------------------------------------------------

class TestCompactionMaxChars:
    def test_constant_value(self):
        assert COMPACTION_MAX_CHARS == 500

    def test_constant_positive(self):
        assert COMPACTION_MAX_CHARS > 0


# ---------------------------------------------------------------------------
# Compaction prompt improvements
# ---------------------------------------------------------------------------

class TestCompactionPrompt:
    """Verify the compaction system instruction contains quality improvements."""

    async def test_prompt_requests_topic_tags(self, tmp_dir):
        """Prompt should instruct LLM to produce [Topics: ...] line."""
        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "[Topics: test]\n- did stuff"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        assert "[Topics:" in captured["system"]
        assert "comma-separated" in captured["system"]

    async def test_prompt_preserves_identifiers_verbatim(self, tmp_dir):
        """Prompt should instruct to preserve hostnames, IPs, paths verbatim."""
        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "[Topics: test]\n- stuff"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        system = captured["system"]
        assert "VERBATIM" in system
        assert "Hostnames" in system
        assert "IPs" in system
        assert "file paths" in system

    async def test_prompt_omits_intermediate_steps(self, tmp_dir):
        """Prompt should instruct to omit intermediate steps and retries."""
        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "[Topics: test]\n- stuff"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        system = captured["system"]
        assert "Intermediate steps" in system
        assert "retries" in system

    async def test_prompt_outcome_focused(self, tmp_dir):
        """Prompt should instruct WHAT → OUTCOME format."""
        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "[Topics: test]\n- stuff"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        system = captured["system"]
        assert "WHAT" in system
        assert "OUTCOME" in system

    async def test_prompt_requests_under_500_chars(self, tmp_dir):
        """Prompt should instruct LLM to keep summary under 500 chars."""
        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "[Topics: test]\n- stuff"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        assert "500 characters" in captured["system"]

    async def test_prompt_omits_conversational_filler(self, tmp_dir):
        """Prompt should instruct to omit greetings and filler."""
        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "[Topics: test]\n- stuff"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        system = captured["system"]
        assert "filler" in system.lower() or "greetings" in system.lower()

    async def test_prompt_still_contains_summarize(self, tmp_dir):
        """Backward compat: prompt still contains 'Summarize'."""
        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "summary"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        assert "Summarize" in captured["system"]

    async def test_prompt_preserves_error_omission(self, tmp_dir):
        """Prompt still omits errors and 'unable to' statements."""
        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "summary"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        system = captured["system"]
        assert "error" in system.lower()
        assert "unable to" in system.lower()


# ---------------------------------------------------------------------------
# Summary length enforcement
# ---------------------------------------------------------------------------

class TestSummaryLengthEnforcement:
    """Verify summaries are truncated to COMPACTION_MAX_CHARS."""

    async def test_short_summary_not_truncated(self, tmp_dir):
        """Summary under 500 chars is kept as-is."""
        short = "[Topics: nginx]\n- Deployed nginx on server-a"
        assert len(short) < COMPACTION_MAX_CHARS

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=short))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        assert session.summary == short

    async def test_long_summary_truncated(self, tmp_dir):
        """Summary over 500 chars is truncated."""
        lines = ["[Topics: test]"]
        for i in range(30):
            lines.append(f"- Action {i}: deployed service-{i} to host-{i}.example.com")
        long_summary = "\n".join(lines)
        assert len(long_summary) > COMPACTION_MAX_CHARS

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=long_summary))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        assert len(session.summary) <= COMPACTION_MAX_CHARS

    async def test_truncation_at_line_boundary(self, tmp_dir):
        """Truncation happens at a line boundary, not mid-word."""
        lines = ["[Topics: infra]"]
        # Build a summary that's well over 500 chars
        for i in range(25):
            lines.append(f"- Checked host-{i:02d} status OK")
        long_summary = "\n".join(lines)
        assert len(long_summary) > COMPACTION_MAX_CHARS

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=long_summary))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        # Should not end mid-line
        assert not session.summary.endswith("-")
        # Should end at a complete line
        last_line = session.summary.split("\n")[-1]
        assert last_line.startswith("- ")

    async def test_exactly_500_chars_not_truncated(self, tmp_dir):
        """Summary of exactly 500 chars is kept as-is."""
        summary = "x" * COMPACTION_MAX_CHARS
        assert len(summary) == COMPACTION_MAX_CHARS

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=summary))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        assert len(session.summary) == COMPACTION_MAX_CHARS

    async def test_501_chars_truncated(self, tmp_dir):
        """Summary of 501 chars is truncated."""
        # Build a 501-char string with newlines to test line-boundary truncation
        base = "[Topics: test]\n"
        filler = "- " + "a" * 50 + "\n"
        summary = base
        while len(summary) + len(filler) <= COMPACTION_MAX_CHARS:
            summary += filler
        summary += filler  # push over 500
        assert len(summary) > COMPACTION_MAX_CHARS

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=summary))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        assert len(session.summary) <= COMPACTION_MAX_CHARS

    async def test_no_newlines_truncates_at_limit(self, tmp_dir):
        """A single long line with no newlines truncates at the limit."""
        summary = "a" * (COMPACTION_MAX_CHARS + 100)

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=summary))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        # No newline found, falls through to the raw truncation
        assert len(session.summary) <= COMPACTION_MAX_CHARS

    async def test_whitespace_stripped(self, tmp_dir):
        """Summary with leading/trailing whitespace is stripped."""
        summary = "  \n [Topics: test]\n- item \n  "

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=summary))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        assert not session.summary.startswith(" ")
        assert not session.summary.endswith(" ")


# ---------------------------------------------------------------------------
# Realistic compaction scenarios
# ---------------------------------------------------------------------------

class TestRealisticCompaction:
    """Test compaction with realistic infrastructure conversations."""

    async def test_realistic_session_compaction(self, tmp_dir):
        """A realistic session is compacted through the improved prompt."""
        captured = {}

        async def capture_fn(messages, system):
            captured["messages"] = messages
            captured["system"] = system
            # Simulate what a good LLM summary should look like
            return (
                "[Topics: nginx, ssl, dns, payment-api, server-a, server-b]\n"
                "- Cleaned /var/log on server-a (192.168.1.50) → 45% usage\n"
                "- Deployed nginx 1.24.0 on web-prod-01, port 443\n"
                "- SSL cert for api.example.com via Let's Encrypt\n"
                "- api.example.com → 203.0.113.42 (CNAME lb-east)\n"
                "- Restarted payment-api.service on server-b (10.0.0.5)"
            )

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        _fill_realistic_session(mgr, "ch1")
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        assert "[Topics:" in session.summary
        assert len(session.summary) <= COMPACTION_MAX_CHARS
        # Key identifiers preserved
        assert "server-a" in session.summary
        assert "192.168.1.50" in session.summary or "server-a" in session.summary

    async def test_merged_summary_includes_previous(self, tmp_dir):
        """When session has existing summary, it's included for merging."""
        captured = {}

        async def capture_fn(messages, system):
            captured["content"] = messages[0]["content"]
            return "[Topics: merged]\n- old + new"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        session = mgr.get_or_create("ch1")
        session.summary = "[Topics: old]\n- Previously deployed redis on cache-01"
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        assert "Previous summary" in captured["content"]
        assert "redis" in captured["content"]

    async def test_error_messages_excluded_from_outcome(self, tmp_dir):
        """The prompt instructs to omit error/failure messages."""
        captured = {}

        async def capture_fn(messages, system):
            captured["system"] = system
            return "[Topics: test]\n- success only"

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(capture_fn)
        # Add messages including errors
        for i in range(COMPACTION_THRESHOLD + 5):
            role = "user" if i % 2 == 0 else "assistant"
            if i == 5:
                mgr.add_message("ch1", "assistant", "Error: connection refused to server-c")
            elif i == 7:
                mgr.add_message("ch1", "assistant", "I can't access the database right now")
            else:
                mgr.add_message("ch1", role, f"msg {i}")

        await mgr.get_history_with_compaction("ch1")

        system = captured["system"]
        assert "error" in system.lower()
        assert "unable to" in system.lower()


# ---------------------------------------------------------------------------
# Source code verification
# ---------------------------------------------------------------------------

class TestSourceVerification:
    """Verify source code contains expected improvements."""

    def test_manager_has_compaction_max_chars(self):
        source = Path("src/sessions/manager.py").read_text()
        assert "COMPACTION_MAX_CHARS" in source

    def test_manager_has_topic_tags_format(self):
        source = Path("src/sessions/manager.py").read_text()
        assert "[Topics:" in source

    def test_manager_has_verbatim_instruction(self):
        source = Path("src/sessions/manager.py").read_text()
        assert "VERBATIM" in source

    def test_manager_has_outcome_instruction(self):
        source = Path("src/sessions/manager.py").read_text()
        assert "OUTCOME" in source

    def test_manager_has_500_char_limit(self):
        source = Path("src/sessions/manager.py").read_text()
        assert "500 characters" in source

    def test_manager_has_truncation_logic(self):
        source = Path("src/sessions/manager.py").read_text()
        assert "COMPACTION_MAX_CHARS" in source
        assert "truncated" in source

    def test_manager_has_intermediate_steps_omission(self):
        source = Path("src/sessions/manager.py").read_text()
        assert "Intermediate steps" in source

    def test_manager_preserves_identifiers_list(self):
        source = Path("src/sessions/manager.py").read_text()
        for identifier in ["Hostnames", "IPs", "UUIDs", "file paths", "container names"]:
            assert identifier in source, f"Missing identifier type: {identifier}"

    def test_compaction_max_chars_is_500(self):
        assert COMPACTION_MAX_CHARS == 500

    def test_compaction_max_chars_exported(self):
        from src.sessions import manager
        assert hasattr(manager, "COMPACTION_MAX_CHARS")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCompactionEdgeCases:
    async def test_empty_summary_from_llm(self, tmp_dir):
        """Empty string from LLM results in empty summary."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=""))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        assert session.summary == ""

    async def test_whitespace_only_summary(self, tmp_dir):
        """Whitespace-only from LLM results in empty summary."""
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value="   \n\n  "))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        assert session.summary == ""

    async def test_summary_with_unicode(self, tmp_dir):
        """Unicode in summary is preserved."""
        summary = "[Topics: i18n]\n- Deployed app with 日本語 locale"
        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=summary))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        assert "日本語" in session.summary

    async def test_truncation_preserves_topic_line(self, tmp_dir):
        """When truncating, the [Topics:] line is preserved if it fits."""
        topic_line = "[Topics: nginx, ssl, dns]"
        lines = [topic_line]
        # Add enough bullet lines to exceed limit
        for i in range(30):
            lines.append(f"- Action {i}: deployed service-{i} on host-{i}")
        long_summary = "\n".join(lines)
        assert len(long_summary) > COMPACTION_MAX_CHARS

        mgr = SessionManager(
            max_history=30, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(AsyncMock(return_value=long_summary))
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        # Topic line should be preserved since it's at the start
        assert session.summary.startswith("[Topics:")

    async def test_compaction_fn_failure_preserves_existing_summary(self, tmp_dir):
        """When compaction fails, existing summary is preserved."""
        mgr = SessionManager(
            max_history=20, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        session = mgr.get_or_create("ch1")
        session.summary = "[Topics: old]\n- Previous work"
        _fill_session(mgr, "ch1", COMPACTION_THRESHOLD + 5)

        mgr.set_compaction_fn(AsyncMock(side_effect=RuntimeError("fail")))
        await mgr.get_history_with_compaction("ch1")

        session = mgr.get_or_create("ch1")
        # Existing summary preserved on failure
        assert "[Topics: old]" in session.summary
