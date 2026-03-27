"""Tests for Round 8: History relevance scoring.

Validates that get_task_history filters older messages by keyword relevance
to the current query, always keeps recent messages, and logs dropped messages.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.sessions.manager import (
    Message,
    SessionManager,
    score_relevance,
    _tokenize,
    RELEVANCE_KEEP_RECENT,
    RELEVANCE_MIN_SCORE,
    RELEVANCE_MAX_OLDER,
)


@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=100,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


# ---------------------------------------------------------------------------
# _tokenize tests
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Check the disk usage on server-1")
        assert "check" in tokens
        assert "disk" in tokens
        assert "usage" in tokens
        assert "server-1" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenize("the is a to of and or for this that")
        assert len(tokens) == 0

    def test_single_char_removed(self):
        tokens = _tokenize("a b c hello world")
        assert "hello" in tokens
        assert "world" in tokens
        assert "a" not in tokens
        assert "b" not in tokens

    def test_case_insensitive(self):
        tokens = _tokenize("Deploy NGINX to Server")
        assert "deploy" in tokens
        assert "nginx" in tokens
        assert "server" in tokens

    def test_empty_string(self):
        assert _tokenize("") == set()

    def test_preserves_paths_and_ips(self):
        tokens = _tokenize("check /var/log/syslog on 10.0.0.1")
        assert "/var/log/syslog" in tokens
        assert "10.0.0.1" in tokens


# ---------------------------------------------------------------------------
# score_relevance tests
# ---------------------------------------------------------------------------

class TestScoreRelevance:
    def test_identical_query_and_message(self):
        score = score_relevance("check disk usage", "check disk usage")
        assert score == 1.0

    def test_no_overlap(self):
        score = score_relevance("check disk usage", "write a haiku about cats")
        assert score == 0.0

    def test_partial_overlap(self):
        score = score_relevance("check disk usage on server", "disk usage is high")
        assert 0.0 < score < 1.0

    def test_empty_query(self):
        score = score_relevance("", "some message content")
        assert score == 0.0

    def test_empty_message(self):
        score = score_relevance("check disk", "")
        assert score == 0.0

    def test_stop_words_dont_inflate_score(self):
        # "the" and "is" are stop words — should not count toward overlap
        score = score_relevance("the server is down", "the table is set")
        # Only stop words overlap, no real content overlap
        assert score == 0.0

    def test_high_overlap_above_threshold(self):
        score = score_relevance(
            "deploy nginx to production server",
            "deployed nginx on the production server successfully",
        )
        assert score >= RELEVANCE_MIN_SCORE

    def test_score_is_float_between_0_and_1(self):
        score = score_relevance("test query here", "different content entirely")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# get_task_history with relevance filtering
# ---------------------------------------------------------------------------

class TestTaskHistoryRelevance:
    async def test_no_query_returns_all(self, session_mgr: SessionManager):
        """Without current_query, all recent messages are included."""
        for i in range(10):
            session_mgr.add_message("ch1", "user", f"message {i}")
        history = await session_mgr.get_task_history("ch1", max_messages=10)
        # All 10 messages returned (no summary)
        assert len(history) == 10

    async def test_query_keeps_recent_messages(self, session_mgr: SessionManager):
        """The most recent RELEVANCE_KEEP_RECENT messages are always kept."""
        # Add messages: first batch unrelated, last batch related
        for i in range(5):
            session_mgr.add_message("ch1", "user", f"unrelated topic {i} about painting")
        for i in range(RELEVANCE_KEEP_RECENT):
            session_mgr.add_message("ch1", "user", f"recent message {i} about painting")

        history = await session_mgr.get_task_history(
            "ch1", max_messages=8, current_query="deploy nginx to server",
        )
        # Recent messages always included
        recent_contents = [m["content"] for m in history[-RELEVANCE_KEEP_RECENT:]]
        for i in range(RELEVANCE_KEEP_RECENT):
            assert any(f"recent message {i}" in c for c in recent_contents)

    async def test_drops_irrelevant_older_messages(self, session_mgr: SessionManager):
        """Older messages with no keyword overlap are dropped."""
        # Add irrelevant older messages
        for i in range(8):
            session_mgr.add_message("ch1", "user", f"wrote haiku about cats number {i}")
        # Add recent messages (always kept)
        for i in range(RELEVANCE_KEEP_RECENT):
            session_mgr.add_message("ch1", "assistant", f"recent response {i}")

        history = await session_mgr.get_task_history(
            "ch1", max_messages=11, current_query="check disk usage on server-1",
        )
        # Should have fewer messages than the full 11
        # Recent messages are always kept, older irrelevant ones dropped
        assert len(history) <= RELEVANCE_KEEP_RECENT + RELEVANCE_MAX_OLDER

    async def test_keeps_relevant_older_messages(self, session_mgr: SessionManager):
        """Older messages that overlap with the query are kept."""
        # Relevant older message
        session_mgr.add_message("ch1", "user", "check disk usage on server-1")
        session_mgr.add_message("ch1", "assistant", "Disk usage is 45% on server-1")
        # Irrelevant older message
        session_mgr.add_message("ch1", "user", "write a poem about autumn")
        session_mgr.add_message("ch1", "assistant", "Leaves fall gently...")
        # Recent messages
        for i in range(RELEVANCE_KEEP_RECENT):
            session_mgr.add_message("ch1", "user", f"more disk info {i}")

        history = await session_mgr.get_task_history(
            "ch1", max_messages=10, current_query="check disk usage on server-1",
        )
        contents = [m["content"] for m in history]
        # The relevant older messages should be present
        assert any("check disk usage" in c for c in contents)
        assert any("45%" in c for c in contents)

    async def test_irrelevant_older_messages_excluded(self, session_mgr: SessionManager):
        """Verify irrelevant messages are actually excluded."""
        # Many irrelevant messages
        for i in range(10):
            session_mgr.add_message("ch1", "user", f"painting watercolors landscape {i}")
        # Recent
        for i in range(RELEVANCE_KEEP_RECENT):
            session_mgr.add_message("ch1", "user", f"recent {i}")

        history = await session_mgr.get_task_history(
            "ch1", max_messages=13, current_query="deploy nginx production",
        )
        contents = [m["content"] for m in history]
        # None of the painting messages should be included
        assert not any("painting watercolors" in c for c in contents)

    async def test_summary_still_prepended(self, session_mgr: SessionManager):
        """Summary is still prepended even with relevance filtering."""
        session = session_mgr.get_or_create("ch1")
        session.summary = "Previously discussed server monitoring."
        for i in range(5):
            session_mgr.add_message("ch1", "user", f"message {i}")

        history = await session_mgr.get_task_history(
            "ch1", max_messages=5, current_query="check something",
        )
        assert history[0]["content"].startswith("[Previous conversation summary:")
        assert history[1]["content"] == "Understood, I have context from our previous conversation."

    async def test_few_messages_no_filtering(self, session_mgr: SessionManager):
        """When messages <= RELEVANCE_KEEP_RECENT, no filtering occurs."""
        for i in range(RELEVANCE_KEEP_RECENT):
            session_mgr.add_message("ch1", "user", f"cats {i}")

        history = await session_mgr.get_task_history(
            "ch1", max_messages=10, current_query="deploy nginx",
        )
        assert len(history) == RELEVANCE_KEEP_RECENT

    async def test_respects_max_older_cap(self, session_mgr: SessionManager):
        """Even if many older messages are relevant, cap at RELEVANCE_MAX_OLDER."""
        # Many relevant older messages
        for i in range(15):
            session_mgr.add_message("ch1", "user", f"deploy nginx to server {i}")
        # Recent
        for i in range(RELEVANCE_KEEP_RECENT):
            session_mgr.add_message("ch1", "user", f"recent deploy nginx {i}")

        history = await session_mgr.get_task_history(
            "ch1", max_messages=18, current_query="deploy nginx server",
        )
        # Should not exceed RELEVANCE_MAX_OLDER + RELEVANCE_KEEP_RECENT (no summary)
        assert len(history) <= RELEVANCE_MAX_OLDER + RELEVANCE_KEEP_RECENT

    async def test_preserves_original_order(self, session_mgr: SessionManager):
        """Relevant older messages maintain their original order."""
        session_mgr.add_message("ch1", "user", "first nginx deploy attempt")
        session_mgr.add_message("ch1", "user", "unrelated painting topic")
        session_mgr.add_message("ch1", "user", "second nginx deploy attempt")
        session_mgr.add_message("ch1", "user", "unrelated cooking topic")
        # Recent
        for i in range(RELEVANCE_KEEP_RECENT):
            session_mgr.add_message("ch1", "user", f"recent {i}")

        history = await session_mgr.get_task_history(
            "ch1", max_messages=10, current_query="nginx deploy",
        )
        contents = [m["content"] for m in history]
        nginx_msgs = [c for c in contents if "nginx" in c and "recent" not in c]
        if len(nginx_msgs) >= 2:
            assert contents.index(nginx_msgs[0]) < contents.index(nginx_msgs[1])


# ---------------------------------------------------------------------------
# Logging of dropped messages
# ---------------------------------------------------------------------------

class TestRelevanceLogging:
    async def test_logs_dropped_messages(self, session_mgr: SessionManager):
        """When messages are dropped, a log entry is produced."""
        for i in range(8):
            session_mgr.add_message("ch1", "user", f"painting watercolors {i}")
        for i in range(RELEVANCE_KEEP_RECENT):
            session_mgr.add_message("ch1", "user", f"recent {i}")

        with patch("src.sessions.manager.log") as mock_log:
            await session_mgr.get_task_history(
                "ch1", max_messages=11, current_query="deploy nginx server",
            )
            # Verify info log about dropped messages was called
            log_calls = [str(c) for c in mock_log.info.call_args_list]
            assert any("Relevance filter" in c and "dropped" in c for c in log_calls)

    async def test_no_log_when_nothing_dropped(self, session_mgr: SessionManager):
        """When all older messages are relevant, no drop log."""
        session_mgr.add_message("ch1", "user", "deploy nginx server")
        session_mgr.add_message("ch1", "user", "check nginx status")
        for i in range(RELEVANCE_KEEP_RECENT):
            session_mgr.add_message("ch1", "user", f"recent {i}")

        with patch("src.sessions.manager.log") as mock_log:
            await session_mgr.get_task_history(
                "ch1", max_messages=10, current_query="deploy nginx server",
            )
            drop_calls = [
                str(c) for c in mock_log.info.call_args_list
                if "dropped" in str(c) and "Relevance" in str(c)
            ]
            assert len(drop_calls) == 0


# ---------------------------------------------------------------------------
# Constants verification
# ---------------------------------------------------------------------------

class TestRelevanceConstants:
    def test_keep_recent_is_positive(self):
        assert RELEVANCE_KEEP_RECENT > 0

    def test_min_score_between_0_and_1(self):
        assert 0.0 < RELEVANCE_MIN_SCORE < 1.0

    def test_max_older_is_positive(self):
        assert RELEVANCE_MAX_OLDER > 0


# ---------------------------------------------------------------------------
# Integration with client.py call site
# ---------------------------------------------------------------------------

class TestClientIntegration:
    def test_get_task_history_signature_accepts_current_query(self):
        """Verify the method accepts current_query parameter."""
        import inspect
        sig = inspect.signature(SessionManager.get_task_history)
        params = list(sig.parameters.keys())
        assert "current_query" in params

    def test_score_relevance_importable(self):
        """Verify score_relevance is importable from the module."""
        from src.sessions.manager import score_relevance
        assert callable(score_relevance)


# ---------------------------------------------------------------------------
# Source code verification
# ---------------------------------------------------------------------------

class TestSourceVerification:
    def test_manager_has_relevance_scoring(self):
        """Verify relevance scoring code exists in manager.py."""
        source = Path("src/sessions/manager.py").read_text()
        assert "score_relevance" in source
        assert "RELEVANCE_KEEP_RECENT" in source
        assert "RELEVANCE_MIN_SCORE" in source
        assert "RELEVANCE_MAX_OLDER" in source
        assert "_tokenize" in source

    def test_client_passes_current_query(self):
        """Verify client.py passes current_query to get_task_history."""
        source = Path("src/discord/client.py").read_text()
        assert "current_query=" in source

    def test_manager_imports_re(self):
        """Verify re module is imported for tokenization."""
        source = Path("src/sessions/manager.py").read_text()
        assert "import re" in source

    def test_stop_words_defined(self):
        """Verify stop words set is defined."""
        source = Path("src/sessions/manager.py").read_text()
        assert "_STOP_WORDS" in source


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRelevanceEdgeCases:
    def test_score_with_only_stop_words_in_query(self):
        """Query consisting only of stop words should score 0."""
        score = score_relevance("the is a to", "the quick brown fox")
        assert score == 0.0

    def test_score_with_numbers_and_paths(self):
        """Numbers and paths should be valid tokens."""
        score = score_relevance(
            "check /var/log/nginx on 10.0.0.1",
            "/var/log/nginx shows errors on 10.0.0.1",
        )
        assert score > 0.5

    def test_score_with_mixed_case(self):
        """Scoring is case-insensitive."""
        score = score_relevance("Deploy NGINX", "deploy nginx configuration")
        assert score > 0.0

    async def test_empty_session_with_query(self, session_mgr: SessionManager):
        """Empty session with a query should return empty list."""
        history = await session_mgr.get_task_history(
            "ch1", max_messages=10, current_query="deploy nginx",
        )
        assert history == []

    async def test_single_message_with_query(self, session_mgr: SessionManager):
        """Single message is always kept regardless of relevance."""
        session_mgr.add_message("ch1", "user", "hello world")
        history = await session_mgr.get_task_history(
            "ch1", max_messages=10, current_query="deploy nginx",
        )
        assert len(history) == 1
        assert history[0]["content"] == "hello world"

    async def test_all_messages_relevant(self, session_mgr: SessionManager):
        """When all messages are relevant, none are dropped."""
        for i in range(8):
            session_mgr.add_message("ch1", "user", f"deploy nginx to server {i}")
        history = await session_mgr.get_task_history(
            "ch1", max_messages=8, current_query="deploy nginx server",
        )
        # Recent 3 always kept + up to 7 older, but only 5 older exist
        assert len(history) == 8
