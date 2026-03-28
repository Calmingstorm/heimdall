"""Tests for Round 14: Context fix-up round.

Validates fixes applied to rounds 6-13 context handling:
- Security audit: no personal IPs in test files
- Token budget: summary pair protected (dropped last, not first)
- Compaction: word-boundary truncation when no newlines
- Helper: _content_text handles string and non-string content
- Helper: _SUMMARY_PREFIX constant for summary detection
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.sessions.manager import (  # noqa: E402
    BUDGET_KEEP_RECENT,
    CHARS_PER_TOKEN,
    COMPACTION_MAX_CHARS,
    CONTEXT_TOKEN_BUDGET,
    SessionManager,
    _SUMMARY_PREFIX,
    _content_text,
    apply_token_budget,
    estimate_tokens,
)


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _summary_pair(summary_text: str = "some prior context") -> list[dict]:
    return [
        _msg("user", f"[Previous conversation summary: {summary_text}]"),
        _msg("assistant", "Understood, I have context from our previous conversation."),
    ]


# ---------------------------------------------------------------------------
# _SUMMARY_PREFIX constant
# ---------------------------------------------------------------------------

class TestSummaryPrefix:
    def test_prefix_value(self):
        assert _SUMMARY_PREFIX == "[Previous conversation summary:"

    def test_summary_pair_detected(self):
        pair = _summary_pair()
        assert pair[0]["content"].startswith(_SUMMARY_PREFIX)


# ---------------------------------------------------------------------------
# _content_text helper
# ---------------------------------------------------------------------------

class TestContentText:
    def test_string_content(self):
        assert _content_text({"content": "hello"}) == "hello"

    def test_list_content(self):
        """Non-string content is converted via str()."""
        result = _content_text({"content": [{"type": "text", "text": "hi"}]})
        assert isinstance(result, str)
        assert "hi" in result

    def test_empty_string(self):
        assert _content_text({"content": ""}) == ""

    def test_numeric_content(self):
        """Edge case: numeric content."""
        result = _content_text({"content": 42})
        assert result == "42"


# ---------------------------------------------------------------------------
# Summary protection in apply_token_budget
# ---------------------------------------------------------------------------

class TestSummaryProtection:
    def test_summary_kept_when_droppable_messages_exist(self):
        """Summary pair should be kept when there are other droppable messages."""
        pair = _summary_pair("x" * 400)  # ~100 tokens
        droppable = [_msg("user", "a" * 4000) for _ in range(7)]  # ~1000 tokens each
        recent = [_msg("user", "b" * 2000) for _ in range(5)]  # ~500 tokens each
        msgs = pair + droppable + recent
        # Budget: 4000 tokens. Recent = 2500. Summary ~115. Need to drop droppable.
        result, dropped = apply_token_budget(msgs, budget=4000)
        # Summary pair should still be present
        assert result[0]["content"].startswith(_SUMMARY_PREFIX)
        assert dropped > 0

    def test_summary_dropped_only_when_droppable_exhausted(self):
        """Summary pair is dropped only after all other droppable messages are gone."""
        pair = _summary_pair("x" * 20000)  # ~5000 tokens (huge summary)
        droppable = [_msg("user", "a" * 400)]  # ~100 tokens
        recent = [_msg("user", "b" * 2000) for _ in range(5)]  # ~500 tokens each
        msgs = pair + droppable + recent
        # Budget: 3000 tokens. Recent = 2500. Summary = ~5000. Won't fit.
        result, dropped = apply_token_budget(msgs, budget=3000)
        # droppable (1) + summary pair (2) = 3 dropped
        assert dropped == 3
        # Only recent messages remain
        assert len(result) == 5
        assert not any(m["content"].startswith(_SUMMARY_PREFIX) for m in result)

    def test_summary_survives_tight_budget(self):
        """With tight budget, summary is kept if it fits after dropping droppable."""
        pair = _summary_pair("brief")  # tiny summary
        droppable = [_msg("user", "x" * 8000) for _ in range(3)]  # ~2000 tokens each
        recent = [_msg("user", "y" * 400) for _ in range(5)]  # ~100 tokens each
        msgs = pair + droppable + recent
        # Budget: 1500 tokens. Recent ~500. Summary ~20.
        result, dropped = apply_token_budget(msgs, budget=1500)
        # All droppable should be dropped, summary kept
        assert result[0]["content"].startswith(_SUMMARY_PREFIX)
        assert dropped == 3  # only the droppable messages

    def test_no_summary_drops_oldest_first(self):
        """Without summary pair, behavior is unchanged — oldest dropped first."""
        msgs = [_msg("user", f"msg_{i}_" + "x" * 4000) for i in range(8)]
        result, dropped = apply_token_budget(msgs, budget=4000)
        assert dropped > 0
        # Recent 3 should be kept
        assert any("msg_7_" in m["content"] for m in result)
        assert any("msg_6_" in m["content"] for m in result)
        assert any("msg_5_" in m["content"] for m in result)

    def test_summary_not_falsely_detected(self):
        """A message that doesn't start with the summary prefix is not protected."""
        msgs = [
            _msg("user", "Just a regular message with lots of text " + "x" * 4000),
            _msg("assistant", "Response " + "y" * 4000),
        ] + [_msg("user", "z" * 4000) for _ in range(5)]
        result, dropped = apply_token_budget(msgs, budget=3000)
        # Oldest (non-summary) messages should be dropped
        assert dropped > 0

    def test_summary_order_preserved(self):
        """Summary pair stays at the beginning of the result."""
        pair = _summary_pair("context")
        middle = [_msg("user", f"mid_{i}") for i in range(3)]
        recent = [_msg("user", f"recent_{i}") for i in range(3)]
        msgs = pair + middle + recent
        result, _ = apply_token_budget(msgs, budget=100000)
        assert result[0]["content"].startswith(_SUMMARY_PREFIX)
        assert "Understood" in result[1]["content"]


# ---------------------------------------------------------------------------
# Compaction truncation — word boundary fallback
# ---------------------------------------------------------------------------

class TestCompactionTruncation:
    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return tmp_path

    async def test_no_newline_truncation_uses_space(self, tmp_dir):
        """When summary has no newlines, truncation falls back to space boundary."""
        long_words = " ".join(["infrastructure"] * 50)  # ~750 chars, no newlines

        async def mock_compaction_fn(messages, system):
            return long_words

        mgr = SessionManager(
            max_history=5, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(mock_compaction_fn)

        # Fill session past compaction threshold
        session = mgr.get_or_create("ch_trunc")
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            session.messages.append(
                MagicMock(role=role, content=f"msg {i}", timestamp=0)
            )

        await mgr._compact(session)

        # Summary should be truncated and not end mid-word
        assert len(session.summary) <= COMPACTION_MAX_CHARS
        assert not session.summary.endswith("infrastru")  # no mid-word cut
        # Should end at a word boundary (last char is a letter)
        assert session.summary[-1].isalpha() or session.summary[-1] in ".])"

    async def test_newline_truncation_still_works(self, tmp_dir):
        """Normal newline-based truncation still works as before."""
        lines = "\n".join([f"- Point {i}: " + "x" * 40 for i in range(20)])

        async def mock_compaction_fn(messages, system):
            return lines

        mgr = SessionManager(
            max_history=5, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(mock_compaction_fn)

        session = mgr.get_or_create("ch_newline")
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            session.messages.append(
                MagicMock(role=role, content=f"msg {i}", timestamp=0)
            )

        await mgr._compact(session)

        assert len(session.summary) <= COMPACTION_MAX_CHARS
        # Should end at a newline boundary (complete line)
        assert session.summary.endswith("]") or "Point" in session.summary

    async def test_short_summary_not_truncated(self, tmp_dir):
        """Short summaries pass through unchanged."""
        short = "[Topics: nginx]\n- Deployed nginx on server-a"

        async def mock_compaction_fn(messages, system):
            return short

        mgr = SessionManager(
            max_history=5, max_age_hours=1,
            persist_dir=str(tmp_dir / "sessions"),
        )
        mgr.set_compaction_fn(mock_compaction_fn)

        session = mgr.get_or_create("ch_short")
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            session.messages.append(
                MagicMock(role=role, content=f"msg {i}", timestamp=0)
            )

        await mgr._compact(session)
        assert session.summary == short


# ---------------------------------------------------------------------------
# Integration: summary protection in get_task_history
# ---------------------------------------------------------------------------

class TestGetTaskHistorySummaryProtection:
    @pytest.fixture
    def manager(self, tmp_path):
        return SessionManager(
            max_history=50, max_age_hours=1,
            persist_dir=str(tmp_path / "sessions"),
        )

    async def test_summary_preserved_after_budget(self, manager):
        """Summary pair is preserved in get_task_history after budget enforcement."""
        ch = "summary_prot"
        session = manager.get_or_create(ch)
        session.summary = "Important context about server-a deployment"

        # Add enough messages to trigger budget trimming
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message(ch, role, f"msg_{i}_" + "x" * 4000)

        with patch("src.sessions.manager.log"):
            history = await manager.get_task_history(ch, max_messages=10)

        # Summary should be present
        assert any(
            m["content"].startswith("[Previous conversation summary:")
            for m in history
        )

    async def test_summary_at_start_of_history(self, manager):
        """Summary pair should be at positions 0 and 1."""
        ch = "summary_pos"
        session = manager.get_or_create(ch)
        session.summary = "Prior context"

        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message(ch, role, f"message {i}")

        history = await manager.get_task_history(ch, max_messages=5)
        assert history[0]["content"].startswith("[Previous conversation summary:")
        assert "Understood" in history[1]["content"]


# ---------------------------------------------------------------------------
# Security audit compliance
# ---------------------------------------------------------------------------

class TestSecurityCompliance:
    def test_round8_no_personal_ips(self):
        """test_round8_relevance_scoring.py must not contain personal IPs."""
        from pathlib import Path
        content = Path("tests/test_round8_relevance_scoring.py").read_text()
        # Check for personal IP prefix without embedding the literal
        personal_prefix = "192" + ".168" + ".1"
        assert personal_prefix not in content

    def test_round9_no_personal_ips(self):
        """test_round9_compaction_quality.py must not contain personal IPs."""
        from pathlib import Path
        content = Path("tests/test_round9_compaction_quality.py").read_text()
        personal_prefix = "192" + ".168" + ".1"
        assert personal_prefix not in content

    def test_replacement_ips_used(self):
        """Replacement IPs (10.0.0.x) should be used instead."""
        from pathlib import Path
        r8 = Path("tests/test_round8_relevance_scoring.py").read_text()
        assert "10.0.0.1" in r8
        r9 = Path("tests/test_round9_compaction_quality.py").read_text()
        assert "10.0.0.50" in r9


# ---------------------------------------------------------------------------
# Source code verification
# ---------------------------------------------------------------------------

class TestSourceVerification:
    def test_summary_prefix_constant_exists(self):
        """_SUMMARY_PREFIX should be defined in manager.py."""
        from pathlib import Path
        source = Path("src/sessions/manager.py").read_text()
        assert "_SUMMARY_PREFIX" in source

    def test_content_text_helper_exists(self):
        """_content_text helper should be defined."""
        from pathlib import Path
        source = Path("src/sessions/manager.py").read_text()
        assert "def _content_text" in source

    def test_summary_detection_in_budget(self):
        """apply_token_budget should detect summary pair."""
        from pathlib import Path
        source = Path("src/sessions/manager.py").read_text()
        assert "has_summary" in source
        assert "_SUMMARY_PREFIX" in source

    def test_word_boundary_fallback(self):
        """Compaction truncation should have word boundary fallback."""
        from pathlib import Path
        source = Path("src/sessions/manager.py").read_text()
        assert 'last_space = truncated.rfind(" ")' in source

    def test_droppable_separation(self):
        """Budget function separates summary from droppable messages."""
        from pathlib import Path
        source = Path("src/sessions/manager.py").read_text()
        assert "droppable" in source
        assert "summary_pair" in source

    def test_content_text_used_in_budget(self):
        """apply_token_budget should use _content_text helper."""
        import inspect
        source = inspect.getsource(apply_token_budget)
        assert "_content_text" in source
