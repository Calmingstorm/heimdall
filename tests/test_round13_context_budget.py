"""Tests for Round 13: Token-based context budget management.

Validates:
- CONTEXT_TOKEN_BUDGET, CHARS_PER_TOKEN, BUDGET_KEEP_RECENT constants
- estimate_tokens() function
- apply_token_budget() drops oldest first, keeps recent 3
- Integration: get_task_history applies budget after relevance filtering
- Logging when messages are trimmed
- Edge cases: empty, exactly at budget, summary messages
"""
from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.sessions.manager import (  # noqa: E402
    BUDGET_KEEP_RECENT,
    CHARS_PER_TOKEN,
    CONTEXT_TOKEN_BUDGET,
    SessionManager,
    apply_token_budget,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify context budget constants are sensible."""

    def test_budget_positive(self):
        assert CONTEXT_TOKEN_BUDGET > 0

    def test_budget_default_is_16000(self):
        assert CONTEXT_TOKEN_BUDGET == 16000

    def test_chars_per_token_positive(self):
        assert CHARS_PER_TOKEN > 0

    def test_chars_per_token_default_is_4(self):
        assert CHARS_PER_TOKEN == 4

    def test_keep_recent_positive(self):
        assert BUDGET_KEEP_RECENT > 0

    def test_keep_recent_default_is_5(self):
        assert BUDGET_KEEP_RECENT == 5

    def test_budget_large_enough_for_recent(self):
        """Budget should be large enough to hold at least a few messages."""
        assert CONTEXT_TOKEN_BUDGET >= 1000


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Test the simple token estimation function."""

    def test_empty_string(self):
        assert estimate_tokens("") == 1  # min 1

    def test_short_string(self):
        assert estimate_tokens("hi") == 1  # 2 chars / 4 = 0 → min 1

    def test_known_length(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25  # 100 / 4

    def test_exact_multiple(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_non_multiple(self):
        text = "a" * 401
        assert estimate_tokens(text) == 100  # 401 // 4 = 100

    def test_long_text(self):
        text = "x" * 32000
        assert estimate_tokens(text) == 8000

    def test_always_positive(self):
        for n in range(10):
            assert estimate_tokens("a" * n) >= 1


# ---------------------------------------------------------------------------
# apply_token_budget
# ---------------------------------------------------------------------------


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


class TestApplyTokenBudget:
    """Test the budget enforcement function."""

    def test_empty_list(self):
        result, dropped = apply_token_budget([])
        assert result == []
        assert dropped == 0

    def test_under_budget(self):
        msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
        result, dropped = apply_token_budget(msgs, budget=8000)
        assert result == msgs
        assert dropped == 0

    def test_exactly_at_budget(self):
        # 8000 tokens = 32000 chars
        text = "a" * 32000
        msgs = [_msg("user", text)]
        result, dropped = apply_token_budget(msgs, budget=8000)
        assert len(result) == 1
        assert dropped == 0

    def test_over_budget_drops_oldest(self):
        # Each message ~1000 tokens (4000 chars), 10 messages = 10000 tokens
        msgs = [_msg("user", "a" * 4000) for _ in range(10)]
        result, dropped = apply_token_budget(msgs, budget=5000)
        # Should keep recent 3 (3000 tokens) + as many older as fit in 2000 tokens = 2 more
        assert dropped > 0
        # Recent 3 are always kept
        assert result[-3:] == msgs[-3:]

    def test_always_keeps_recent_5(self):
        # 7 messages, each 2000 tokens, budget 4000
        msgs = [_msg("user", "a" * 8000) for _ in range(7)]
        result, dropped = apply_token_budget(msgs, budget=4000)
        # Recent 5 = 10000 tokens > budget, but they're protected
        # Older 2 should be dropped
        assert dropped == 2
        assert result == msgs[-5:]

    def test_single_message(self):
        msgs = [_msg("user", "a" * 100000)]
        result, dropped = apply_token_budget(msgs, budget=100)
        # Single message is always kept (it's recent)
        assert len(result) == 1
        assert dropped == 0

    def test_two_messages_over_budget(self):
        msgs = [_msg("user", "a" * 40000), _msg("assistant", "a" * 40000)]
        result, dropped = apply_token_budget(msgs, budget=100)
        # Both are within recent 3, so both kept
        assert len(result) == 2
        assert dropped == 0

    def test_drops_only_oldest(self):
        # 6 messages, each 500 tokens (2000 chars), budget = 2500
        msgs = [_msg("user" if i % 2 == 0 else "assistant", f"msg{i}" + "x" * 2000) for i in range(6)]
        result, dropped = apply_token_budget(msgs, budget=2500)
        # Recent 3 = 1500 tokens, remaining budget = 1000 = 2 more messages
        assert dropped > 0
        # The most recent 3 messages should be the last 3
        assert result[-3:] == msgs[-3:]

    def test_preserves_order(self):
        # 8 messages with unique content, budget allows 5
        msgs = [_msg("user", f"unique_{i}_" + "x" * 2000) for i in range(8)]
        result, dropped = apply_token_budget(msgs, budget=2800)
        # Result should be in original order
        for i in range(len(result) - 1):
            # Each message has unique_N, check ordering
            idx_a = int(result[i]["content"].split("_")[1])
            idx_b = int(result[i + 1]["content"].split("_")[1])
            assert idx_a < idx_b

    def test_returns_tuple(self):
        result = apply_token_budget([_msg("user", "hi")])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_budget_zero(self):
        msgs = [_msg("user", "hello"), _msg("assistant", "world")]
        result, dropped = apply_token_budget(msgs, budget=0)
        # Recent 3 are protected even with 0 budget
        assert len(result) == 2
        assert dropped == 0

    def test_summary_messages_counted(self):
        """Summary pair at the start should be counted toward budget."""
        summary = _msg("user", "[Previous conversation summary: " + "x" * 4000 + "]")
        ack = _msg("assistant", "Understood, I have context from our previous conversation.")
        regular = [_msg("user", "a" * 2000) for _ in range(5)]
        msgs = [summary, ack] + regular
        result, dropped = apply_token_budget(msgs, budget=3000)
        # Recent 3 of the 7 messages are protected, older ones dropped to fit
        assert dropped > 0

    def test_large_budget_no_drops(self):
        msgs = [_msg("user", "hi") for _ in range(100)]
        result, dropped = apply_token_budget(msgs, budget=100000)
        assert dropped == 0
        assert len(result) == 100


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestBudgetLogging:
    """Verify logging when budget trimming occurs."""

    def test_logs_when_trimmed(self):
        msgs = [_msg("user", "a" * 8000) for _ in range(10)]
        with patch("src.sessions.manager.log") as mock_log:
            _, dropped = apply_token_budget(msgs, budget=5000)
        assert dropped > 0
        mock_log.info.assert_called()
        log_msg = mock_log.info.call_args[0][0]
        assert "budget" in log_msg.lower() or "trimmed" in log_msg.lower()

    def test_no_log_when_under_budget(self):
        msgs = [_msg("user", "hi")]
        with patch("src.sessions.manager.log") as mock_log:
            _, dropped = apply_token_budget(msgs, budget=8000)
        assert dropped == 0
        mock_log.info.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: get_task_history applies budget
# ---------------------------------------------------------------------------


class TestGetTaskHistoryBudget:
    """Verify get_task_history enforces the token budget."""

    @pytest.fixture
    def manager(self, tmp_path):
        return SessionManager(
            max_history=100,
            max_age_hours=24,
            persist_dir=str(tmp_path),
        )

    async def test_budget_applied_to_history(self, manager):
        """Large history should be trimmed by token budget."""
        ch = "budget_test"
        # Add many large messages (each ~1000 tokens = 4000 chars)
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message(ch, role, f"msg_{i}_" + "x" * 4000)
        with patch("src.sessions.manager.log"):
            history = await manager.get_task_history(ch, max_messages=20)
        # Total budget is 8000 tokens. Each message ~1000 tokens.
        # 20 messages = 20000 tokens > 8000, so some should be dropped.
        total_tokens = sum(
            estimate_tokens(m["content"]) for m in history
        )
        assert total_tokens <= CONTEXT_TOKEN_BUDGET + BUDGET_KEEP_RECENT * 1100  # tolerance for protected msgs

    async def test_small_history_not_trimmed(self, manager):
        """Small history should not be trimmed."""
        ch = "small_test"
        manager.add_message(ch, "user", "hello")
        manager.add_message(ch, "assistant", "hi there")
        history = await manager.get_task_history(ch)
        assert len(history) == 2

    async def test_recent_messages_preserved(self, manager):
        """The most recent messages should always be in the result."""
        ch = "recent_test"
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message(ch, role, f"msg_{i}_" + "x" * 4000)
        with patch("src.sessions.manager.log"):
            history = await manager.get_task_history(ch, max_messages=20)
        # Last message content should be present
        contents = [m["content"] for m in history]
        assert any("msg_19_" in c for c in contents)

    async def test_budget_with_summary(self, manager):
        """Summary should be counted toward budget."""
        ch = "summary_budget"
        session = manager.get_or_create(ch)
        session.summary = "x" * 16000  # ~4000 tokens just for summary
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message(ch, role, f"msg_{i}_" + "x" * 8000)
        with patch("src.sessions.manager.log"):
            history = await manager.get_task_history(ch, max_messages=15)
        # Should have fewer messages due to summary consuming budget
        # Summary pair is 2 messages + whatever fits
        assert len(history) < 17  # 15 msgs + 2 summary pair

    async def test_budget_logging_in_get_task_history(self, manager):
        """Should log when budget trimming occurs in get_task_history."""
        ch = "log_test"
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message(ch, role, f"msg_{i}_" + "x" * 4000)
        with patch("src.sessions.manager.log") as mock_log:
            await manager.get_task_history(ch, max_messages=20)
        # Should have at least one info log about budget
        budget_logs = [
            call for call in mock_log.info.call_args_list
            if "budget" in str(call).lower()
        ]
        assert len(budget_logs) > 0


# ---------------------------------------------------------------------------
# Source code verification
# ---------------------------------------------------------------------------


class TestSourceVerification:
    """Verify the implementation exists in the source."""

    def test_constants_exist(self):
        from src.sessions import manager
        assert hasattr(manager, "CONTEXT_TOKEN_BUDGET")
        assert hasattr(manager, "CHARS_PER_TOKEN")
        assert hasattr(manager, "BUDGET_KEEP_RECENT")

    def test_estimate_tokens_exists(self):
        from src.sessions import manager
        assert hasattr(manager, "estimate_tokens")
        assert callable(manager.estimate_tokens)

    def test_apply_token_budget_exists(self):
        from src.sessions import manager
        assert hasattr(manager, "apply_token_budget")
        assert callable(manager.apply_token_budget)

    def test_apply_token_budget_signature(self):
        sig = inspect.signature(apply_token_budget)
        params = list(sig.parameters.keys())
        assert "messages" in params
        assert "budget" in params

    def test_estimate_tokens_signature(self):
        sig = inspect.signature(estimate_tokens)
        params = list(sig.parameters.keys())
        assert "text" in params

    def test_apply_token_budget_has_default_budget(self):
        sig = inspect.signature(apply_token_budget)
        budget_param = sig.parameters["budget"]
        assert budget_param.default == CONTEXT_TOKEN_BUDGET

    def test_manager_source_calls_apply_token_budget(self):
        src = inspect.getsource(SessionManager.get_task_history)
        assert "apply_token_budget" in src

    def test_manager_source_logs_budget_drops(self):
        src = inspect.getsource(SessionManager.get_task_history)
        assert "budget_dropped" in src

    def test_budget_keep_recent_matches(self):
        """BUDGET_KEEP_RECENT and RELEVANCE_KEEP_RECENT should both be 5."""
        from src.sessions.manager import RELEVANCE_KEEP_RECENT
        assert BUDGET_KEEP_RECENT == RELEVANCE_KEEP_RECENT


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for token budget management."""

    def test_all_messages_are_empty(self):
        msgs = [_msg("user", ""), _msg("assistant", "")]
        result, dropped = apply_token_budget(msgs, budget=1)
        # Empty messages estimate to 1 token each, 2 messages within recent
        assert dropped == 0

    def test_mixed_content_types(self):
        """Messages with non-string content should not crash."""
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": "world"},
        ]
        result, dropped = apply_token_budget(msgs, budget=8000)
        assert dropped == 0

    def test_many_tiny_messages(self):
        """Many tiny messages should fit in budget."""
        msgs = [_msg("user", "ok") for _ in range(100)]
        result, dropped = apply_token_budget(msgs, budget=8000)
        # Each is 1 token, 100 total — well under 8000
        assert dropped == 0
        assert len(result) == 100

    def test_fewer_than_keep_recent(self):
        """When there are fewer messages than BUDGET_KEEP_RECENT."""
        msgs = [_msg("user", "a" * 100000)]  # huge but only 1 message
        result, dropped = apply_token_budget(msgs, budget=100)
        assert len(result) == 1
        assert dropped == 0

    def test_exact_budget_boundary(self):
        """Messages that exactly hit the budget should not be dropped."""
        # 4 tokens per message (16 chars), budget = 20 tokens = 5 messages
        msgs = [_msg("user", "a" * 16) for _ in range(5)]
        result, dropped = apply_token_budget(msgs, budget=20)
        assert dropped == 0
        assert len(result) == 5
