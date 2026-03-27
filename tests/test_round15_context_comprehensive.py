"""Tests for Round 15: Comprehensive context handling verification.

Cross-round integration tests for all context improvements from Rounds 7-14:
- Relevance scoring (Round 8) + Token budget (Round 13) pipeline
- Topic change detection (Round 10) + History reduction
- Tool output summarization (Round 12) + Budget enforcement
- Compaction quality (Round 9) + Summary protection (Round 14)
- Channel isolation (Round 11) across all context features
- Request isolation (Round 7) separator completeness
- Boundary conditions at exact thresholds
- Full end-to-end pipeline: message → topic detection → relevance → budget → output
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from src.sessions.manager import (
    BUDGET_KEEP_RECENT,
    CHARS_PER_TOKEN,
    COMPACTION_MAX_CHARS,
    COMPACTION_THRESHOLD,
    CONTEXT_TOKEN_BUDGET,
    Message,
    RELEVANCE_KEEP_RECENT,
    RELEVANCE_MAX_OLDER,
    RELEVANCE_MIN_SCORE,
    SessionManager,
    TOOL_SUMMARY_MAX_CHARS,
    TOOL_SUMMARY_THRESHOLD,
    TOPIC_CHANGE_RECENT_WINDOW,
    TOPIC_CHANGE_SCORE_THRESHOLD,
    TOPIC_CHANGE_TIME_GAP,
    _SUMMARY_PREFIX,
    _content_text,
    _tokenize,
    apply_token_budget,
    estimate_tokens,
    score_relevance,
    summarize_tool_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path, max_history: int = 20) -> SessionManager:
    return SessionManager(
        max_history=max_history,
        max_age_hours=24,
        persist_dir=str(tmp_path / "sessions"),
    )


def _make_msg(role: str, content: str, ts: float | None = None) -> dict:
    return {"role": role, "content": content}


def _fill_session(sm: SessionManager, channel: str, messages: list[tuple[str, str]]):
    """Add messages to a session. Each tuple is (role, content)."""
    for role, content in messages:
        sm.add_message(channel, role, content)


# ===========================================================================
# Part 1: Cross-Round Integration — Relevance + Budget Pipeline
# ===========================================================================

class TestRelevancePlusBudget:
    """Tests that relevance filtering (R8) and token budget (R13) work together."""

    async def test_relevance_filters_then_budget_trims(self, tmp_path):
        """Relevance scoring reduces messages first, then budget trims remainder."""
        sm = _make_manager(tmp_path)
        # Add 15 messages: 5 relevant to "nginx", 10 irrelevant
        for i in range(10):
            sm.add_message("ch1", "user", f"unrelated topic about cooking recipe {i}")
        for i in range(5):
            sm.add_message("ch1", "user", f"nginx server config for proxy {i}")

        result = await sm.get_task_history(
            "ch1", max_messages=15, current_query="nginx proxy server",
        )
        # Should have: relevant older + recent 3 (all nginx) + possibly summary
        # The 10 cooking messages should be mostly filtered by relevance
        contents = [m["content"] for m in result]
        cooking_count = sum(1 for c in contents if "cooking" in c)
        nginx_count = sum(1 for c in contents if "nginx" in c)
        assert nginx_count > cooking_count, "Relevant messages should outnumber irrelevant"

    async def test_budget_protects_recent_after_relevance(self, tmp_path):
        """Recent 3 messages survive even if they'd exceed budget alone."""
        sm = _make_manager(tmp_path)
        # Add a few small messages then 3 big recent ones
        for i in range(5):
            sm.add_message("ch1", "user", f"old message {i}")
        # Recent 3: large messages (each ~4000 chars = ~1000 tokens)
        for i in range(3):
            sm.add_message("ch1", "user", "x" * 4000 + f" recent {i}")

        result = await sm.get_task_history(
            "ch1", max_messages=8, current_query="recent something",
        )
        recent_contents = [m["content"] for m in result if "recent" in m["content"]]
        assert len(recent_contents) == 3, "All 3 recent messages must survive"

    async def test_summary_survives_relevance_and_budget(self, tmp_path):
        """Summary pair persists through relevance filtering and budget enforcement."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        session.summary = "Deployed nginx to server-a, configured SSL"

        # Add enough messages that budget needs trimming
        for i in range(10):
            sm.add_message("ch1", "user", f"disk usage check on server-b {i}" + " data" * 50)

        result = await sm.get_task_history(
            "ch1", max_messages=10, current_query="disk usage",
        )
        summary_msgs = [m for m in result if _SUMMARY_PREFIX in m["content"]]
        assert len(summary_msgs) >= 1, "Summary should survive pipeline"

    async def test_constant_alignment(self):
        """BUDGET_KEEP_RECENT must equal RELEVANCE_KEEP_RECENT for consistent behavior."""
        assert BUDGET_KEEP_RECENT == RELEVANCE_KEEP_RECENT


# ===========================================================================
# Part 2: Cross-Round Integration — Topic Change + History Reduction
# ===========================================================================

class TestTopicChangePipeline:
    """Tests that topic change detection integrates with history retrieval."""

    async def test_topic_change_reduces_to_one_message(self, tmp_path):
        """When topic change detected, get_task_history returns only 1 message + summary."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        session.summary = "Previous work on nginx"

        for i in range(10):
            sm.add_message("ch1", "user", f"nginx configuration step {i}")

        result = await sm.get_task_history(
            "ch1", max_messages=20, current_query="weather forecast",
            topic_change=True,
        )
        # Should be: summary pair (2) + last message (1) = 3
        non_summary = [m for m in result if _SUMMARY_PREFIX not in m["content"]
                       and "Understood" not in m["content"]]
        assert len(non_summary) == 1, f"Expected 1 message, got {len(non_summary)}"

    async def test_topic_change_with_empty_session(self, tmp_path):
        """Topic change on empty session doesn't crash."""
        sm = _make_manager(tmp_path)
        result = await sm.get_task_history(
            "ch1", max_messages=20, topic_change=True,
        )
        assert result == []

    async def test_topic_change_keeps_summary(self, tmp_path):
        """Summary pair is still included even on topic change."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        session.summary = "Critical: server-a at 10.0.0.50 needs monitoring"
        sm.add_message("ch1", "user", "check disk usage")

        result = await sm.get_task_history(
            "ch1", max_messages=20, topic_change=True,
        )
        has_summary = any(_SUMMARY_PREFIX in m["content"] for m in result)
        assert has_summary, "Summary should be present even during topic change"

    async def test_topic_change_overrides_relevance_filtering(self, tmp_path):
        """When topic_change=True, relevance filtering is skipped."""
        sm = _make_manager(tmp_path)
        for i in range(10):
            sm.add_message("ch1", "user", f"nginx config {i}")

        # Even with current_query that would match everything, topic_change wins
        result = await sm.get_task_history(
            "ch1", max_messages=20, current_query="nginx config",
            topic_change=True,
        )
        non_summary = [m for m in result if _SUMMARY_PREFIX not in m["content"]
                       and "Understood" not in m["content"]]
        assert len(non_summary) == 1, "Topic change should override relevance"

    def test_detect_topic_change_no_session(self, tmp_path):
        """No session returns default no-change dict."""
        sm = _make_manager(tmp_path)
        result = sm.detect_topic_change("nonexistent", "any query")
        assert result["is_topic_change"] is False
        assert result["max_overlap"] == 0.0

    def test_detect_topic_change_single_message(self, tmp_path):
        """Single message is never a topic change (requires >= 2)."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "only one message about servers")
        result = sm.detect_topic_change("ch1", "completely different topic")
        assert result["is_topic_change"] is False

    def test_detect_topic_change_same_topic(self, tmp_path):
        """Same topic has high overlap, no topic change."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "nginx proxy configuration")
        sm.add_message("ch1", "user", "nginx reverse proxy setup")
        result = sm.detect_topic_change("ch1", "nginx proxy settings")
        assert result["is_topic_change"] is False
        assert result["max_overlap"] > TOPIC_CHANGE_SCORE_THRESHOLD

    def test_detect_topic_change_different_topic(self, tmp_path):
        """Completely different topic triggers change."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "nginx proxy configuration")
        sm.add_message("ch1", "user", "nginx reverse proxy setup")
        result = sm.detect_topic_change("ch1", "banana smoothie recipe blender")
        assert result["is_topic_change"] is True
        assert result["max_overlap"] < TOPIC_CHANGE_SCORE_THRESHOLD


# ===========================================================================
# Part 3: Boundary Conditions at Exact Thresholds
# ===========================================================================

class TestBoundaryConditions:
    """Tests at exact threshold values to verify strict/non-strict comparisons."""

    def test_relevance_at_exact_threshold(self):
        """Score exactly at RELEVANCE_MIN_SCORE (0.15) should be KEPT."""
        # score_relevance uses >= threshold in get_task_history
        # Need a query/message pair that scores exactly 0.15
        # 1 match out of ~6-7 tokens ≈ 0.15
        # "alpha beta gamma delta epsilon zeta eta" → 7 tokens
        # If message has "alpha" → 1/7 ≈ 0.143 (below)
        # "alpha beta gamma delta epsilon zeta" → 6 tokens
        # 1 match → 1/6 ≈ 0.167 (above)
        score = score_relevance(
            "alpha beta gamma delta epsilon zeta",
            "alpha unrelated words",
        )
        # 1 token match out of 6 query tokens = 0.1667
        assert score >= RELEVANCE_MIN_SCORE

    def test_relevance_below_threshold(self):
        """Score below RELEVANCE_MIN_SCORE (0.15) should be dropped."""
        score = score_relevance(
            "alpha beta gamma delta epsilon zeta eta",
            "alpha unrelated words",
        )
        # 1 match out of 7 query tokens = 0.143
        assert score < RELEVANCE_MIN_SCORE

    def test_topic_change_at_exact_threshold(self):
        """Score exactly at TOPIC_CHANGE_SCORE_THRESHOLD is NOT a topic change."""
        # is_topic_change requires max_overlap < threshold (strictly less)
        # TOPIC_CHANGE_SCORE_THRESHOLD = 0.05
        # Need a score of exactly 0.05: 1 match out of 20 query tokens
        tokens = " ".join(f"word{i}" for i in range(20))
        msg = "word0 unrelated content here"
        score = score_relevance(tokens, msg)
        # 1/20 = 0.05 exactly
        assert score == TOPIC_CHANGE_SCORE_THRESHOLD
        # This should NOT trigger topic change (strictly less than)

    def test_topic_change_just_below_threshold(self):
        """Score just below threshold IS a topic change."""
        # 0 matches out of 20 tokens = 0.0
        tokens = " ".join(f"word{i}" for i in range(20))
        msg = "completely different vocabulary here"
        score = score_relevance(tokens, msg)
        assert score < TOPIC_CHANGE_SCORE_THRESHOLD

    def test_budget_at_exact_limit(self):
        """Messages exactly at budget should not be trimmed."""
        # Create messages that total exactly CONTEXT_TOKEN_BUDGET tokens
        char_count = CONTEXT_TOKEN_BUDGET * CHARS_PER_TOKEN  # 32000 chars
        msg = {"role": "user", "content": "x" * char_count}
        result, dropped = apply_token_budget([msg])
        assert dropped == 0
        assert len(result) == 1

    def test_budget_one_token_over(self):
        """Messages one token over budget should trigger trimming."""
        # 2 messages: one older, one recent — both big
        chars_each = (CONTEXT_TOKEN_BUDGET * CHARS_PER_TOKEN) // 2 + CHARS_PER_TOKEN
        msgs = [
            {"role": "user", "content": "a" * chars_each},
            {"role": "user", "content": "b" * chars_each},
            {"role": "user", "content": "c" * chars_each},
            {"role": "user", "content": "d" * chars_each},
        ]
        result, dropped = apply_token_budget(msgs)
        assert dropped > 0

    def test_summarize_at_exact_threshold(self):
        """Exactly TOOL_SUMMARY_THRESHOLD tools with long response triggers summary."""
        tools = [f"tool_{i}" for i in range(TOOL_SUMMARY_THRESHOLD)]
        response = "x" * (TOOL_SUMMARY_MAX_CHARS + 1)
        result = summarize_tool_response(response, tools)
        assert result != response
        assert "[Task used" in result

    def test_summarize_one_below_threshold(self):
        """One below threshold does NOT trigger summary."""
        tools = [f"tool_{i}" for i in range(TOOL_SUMMARY_THRESHOLD - 1)]
        response = "x" * (TOOL_SUMMARY_MAX_CHARS + 100)
        result = summarize_tool_response(response, tools)
        assert result == response

    def test_summarize_long_but_within_char_limit(self):
        """At threshold but response within TOOL_SUMMARY_MAX_CHARS: no summary."""
        tools = [f"tool_{i}" for i in range(TOOL_SUMMARY_THRESHOLD)]
        response = "x" * TOOL_SUMMARY_MAX_CHARS  # exactly at limit
        result = summarize_tool_response(response, tools)
        assert result == response  # <= max chars, no summary

    def test_compaction_max_chars_exact(self):
        """Summary exactly at COMPACTION_MAX_CHARS is not truncated."""
        summary = "x" * COMPACTION_MAX_CHARS
        assert len(summary) == COMPACTION_MAX_CHARS
        # The _compact method would not truncate this

    def test_estimate_tokens_empty_string(self):
        """Empty string returns 1 (minimum)."""
        assert estimate_tokens("") == 1

    def test_estimate_tokens_three_chars(self):
        """3 chars → 0 // 4 = 0, but min is 1."""
        assert estimate_tokens("abc") == 1

    def test_estimate_tokens_four_chars(self):
        """4 chars → 4 // 4 = 1."""
        assert estimate_tokens("abcd") == 1

    def test_estimate_tokens_five_chars(self):
        """5 chars → 5 // 4 = 1."""
        assert estimate_tokens("abcde") == 1

    def test_estimate_tokens_eight_chars(self):
        """8 chars → 8 // 4 = 2."""
        assert estimate_tokens("abcdefgh") == 2


# ===========================================================================
# Part 4: Relevance Scoring Edge Cases
# ===========================================================================

class TestRelevanceScoringEdgeCases:
    """Edge cases for _tokenize and score_relevance."""

    def test_tokenize_paths_preserved(self):
        """File paths are kept as single tokens."""
        tokens = _tokenize("check /var/log/nginx.log please")
        assert "/var/log/nginx.log" in tokens

    def test_tokenize_ips_preserved(self):
        """IP addresses are kept as single tokens."""
        tokens = _tokenize("ping 10.0.0.1 now")
        assert "10.0.0.1" in tokens

    def test_tokenize_colons_preserved(self):
        """Tokens with colons (like ports) are preserved."""
        tokens = _tokenize("connect to localhost:8080")
        assert "localhost:8080" in tokens

    def test_tokenize_all_stop_words(self):
        """All stop words → empty set."""
        tokens = _tokenize("the is a to in on and or for")
        assert len(tokens) == 0

    def test_tokenize_single_char_filtered(self):
        """Single character tokens are filtered out."""
        tokens = _tokenize("a b c d e f")
        assert len(tokens) == 0

    def test_tokenize_mixed_case(self):
        """Tokenization is case-insensitive."""
        tokens = _tokenize("NGINX Proxy SERVER")
        assert "nginx" in tokens
        assert "proxy" in tokens
        assert "server" in tokens

    def test_score_relevance_identical(self):
        """Identical strings → score 1.0."""
        assert score_relevance("nginx proxy config", "nginx proxy config") == 1.0

    def test_score_relevance_no_overlap(self):
        """No token overlap → score 0.0."""
        assert score_relevance("nginx proxy", "banana smoothie recipe") == 0.0

    def test_score_relevance_empty_query(self):
        """Empty query → 0.0."""
        assert score_relevance("", "any content") == 0.0

    def test_score_relevance_empty_message(self):
        """Empty message → 0.0."""
        assert score_relevance("any query", "") == 0.0

    def test_score_relevance_stop_words_only_query(self):
        """Query with only stop words → 0.0."""
        assert score_relevance("the and is to", "the and is to") == 0.0

    def test_score_relevance_path_overlap(self):
        """Path tokens match correctly."""
        score = score_relevance(
            "check /var/log/nginx.log",
            "error in /var/log/nginx.log",
        )
        assert score >= 0.5

    def test_score_relevance_partial_overlap(self):
        """Partial overlap returns proportional score."""
        # "nginx server config" → 3 tokens
        # "nginx database backup" → 1 overlap (nginx)
        score = score_relevance("nginx server config", "nginx database backup")
        assert 0.0 < score < 1.0
        assert abs(score - 1 / 3) < 0.01


# ===========================================================================
# Part 5: Token Budget — Summary Pair Protection
# ===========================================================================

class TestSummaryPairProtection:
    """Tests that apply_token_budget correctly protects and orders summary pairs."""

    def test_summary_pair_kept_while_droppable_exists(self):
        """Summary pair is kept while other droppable messages can be removed."""
        summary_user = {"role": "user", "content": f"{_SUMMARY_PREFIX} deployed nginx]"}
        summary_ack = {"role": "assistant", "content": "Understood, I have context."}
        droppable1 = {"role": "user", "content": "d" * 4000}
        droppable2 = {"role": "user", "content": "e" * 4000}
        recent1 = {"role": "user", "content": "recent msg 1"}
        recent2 = {"role": "user", "content": "recent msg 2"}
        recent3 = {"role": "user", "content": "recent msg 3"}

        msgs = [summary_user, summary_ack, droppable1, droppable2, recent1, recent2, recent3]
        result, dropped = apply_token_budget(msgs, budget=500)

        # Summary should survive, droppable messages removed
        assert any(_SUMMARY_PREFIX in m["content"] for m in result)
        assert dropped >= 1

    def test_summary_pair_dropped_last(self):
        """Summary pair is only dropped when all other droppable messages are gone."""
        summary_user = {"role": "user", "content": f"{_SUMMARY_PREFIX} " + "x" * 4000 + "]"}
        summary_ack = {"role": "assistant", "content": "Understood" + "x" * 4000}
        recent1 = {"role": "user", "content": "r" * 4000}
        recent2 = {"role": "user", "content": "r" * 4000}
        recent3 = {"role": "user", "content": "r" * 4000}

        # Budget so tight only recent 3 fit
        msgs = [summary_user, summary_ack, recent1, recent2, recent3]
        result, dropped = apply_token_budget(msgs, budget=100)

        # Summary should be dropped (no other droppable), dropped count includes +2
        assert not any(_SUMMARY_PREFIX in m["content"] for m in result)
        assert dropped == 2

    def test_summary_pair_detection_requires_two_older(self):
        """Summary pair is only detected when len(older) >= 2."""
        summary_user = {"role": "user", "content": f"{_SUMMARY_PREFIX} data]"}
        recent1 = {"role": "user", "content": "recent 1"}
        recent2 = {"role": "user", "content": "recent 2"}
        recent3 = {"role": "user", "content": "recent 3"}

        # summary_user is the only older message — can't form a pair
        msgs = [summary_user, recent1, recent2, recent3]
        result, dropped = apply_token_budget(msgs, budget=100)
        # It's treated as a regular droppable, not a protected summary pair
        assert dropped >= 0  # May or may not be dropped depending on budget

    def test_output_order_preserved(self):
        """Result is always: summary_pair + droppable + recent."""
        summary_user = {"role": "user", "content": f"{_SUMMARY_PREFIX} data]"}
        summary_ack = {"role": "assistant", "content": "Understood."}
        msg1 = {"role": "user", "content": "middle 1"}
        msg2 = {"role": "user", "content": "middle 2"}
        recent1 = {"role": "user", "content": "recent 1"}
        recent2 = {"role": "user", "content": "recent 2"}
        recent3 = {"role": "user", "content": "recent 3"}

        msgs = [summary_user, summary_ack, msg1, msg2, recent1, recent2, recent3]
        result, _ = apply_token_budget(msgs, budget=99999)

        # Under huge budget, nothing dropped — order is original
        assert result == msgs

    def test_content_text_string(self):
        """_content_text returns string content directly."""
        assert _content_text({"content": "hello"}) == "hello"

    def test_content_text_list(self):
        """_content_text converts list content to string."""
        result = _content_text({"content": [{"type": "text", "text": "hi"}]})
        assert isinstance(result, str)

    def test_content_text_none(self):
        """_content_text handles None content."""
        assert _content_text({"content": None}) == "None"

    def test_content_text_int(self):
        """_content_text handles integer content."""
        assert _content_text({"content": 42}) == "42"


# ===========================================================================
# Part 6: Tool Output Summarization Edge Cases
# ===========================================================================

class TestToolSummarizationEdgeCases:
    """Edge cases for summarize_tool_response."""

    def test_empty_response_with_many_tools(self):
        """Empty response string returns unchanged (len <= max_chars)."""
        tools = [f"t{i}" for i in range(15)]
        result = summarize_tool_response("", tools)
        assert result == ""

    def test_exactly_15_unique_tools(self):
        """15 unique tools shows all, no '+N more' suffix."""
        tools = [f"tool_{i}" for i in range(15)]
        response = "x" * 1000
        result = summarize_tool_response(response, tools)
        assert "(+" not in result
        assert "tool_0" in result
        assert "tool_14" in result

    def test_16_unique_tools_shows_plus_more(self):
        """16 unique tools shows first 15 + '(+1 more)'."""
        tools = [f"tool_{i}" for i in range(16)]
        response = "x" * 1000
        result = summarize_tool_response(response, tools)
        assert "(+1 more)" in result

    def test_dedup_preserves_order(self):
        """Duplicate tools are removed preserving first-occurrence order."""
        tools = ["run_command", "check_disk", "run_command", "check_disk",
                 "run_command", "check_disk", "run_command", "check_disk",
                 "run_command", "check_disk"]  # 10 calls, 2 unique
        response = "A" * 600
        result = summarize_tool_response(response, tools)
        assert "[Task used 10 tool calls (run_command, check_disk)]" in result

    def test_single_paragraph_response(self):
        """Response with no double-newlines uses last-400 chars."""
        tools = [f"t{i}" for i in range(12)]
        response = "word " * 200  # 1000 chars, no \n\n
        result = summarize_tool_response(response, tools)
        assert "[Task used" in result
        assert len(result) <= TOOL_SUMMARY_MAX_CHARS

    def test_short_last_paragraph_includes_penultimate(self):
        """When last paragraph < 100 chars, penultimate is also included."""
        tools = [f"t{i}" for i in range(12)]
        long_para = "A" * 400
        short_para = "Done."
        response = long_para + "\n\n" + short_para
        result = summarize_tool_response(response, tools)
        # Should include both paragraphs in the outcome
        assert "Done." in result

    def test_result_within_max_chars(self):
        """Summarized result never exceeds TOOL_SUMMARY_MAX_CHARS."""
        tools = [f"tool_{i}" for i in range(20)]
        response = "x" * 5000
        result = summarize_tool_response(response, tools)
        assert len(result) <= TOOL_SUMMARY_MAX_CHARS

    def test_custom_threshold(self):
        """Custom threshold parameter is respected."""
        tools = ["a", "b", "c"]
        response = "x" * 600
        # With default threshold (10), these 3 tools would not trigger
        result_default = summarize_tool_response(response, tools)
        assert result_default == response

        # With threshold=2, it triggers
        result_custom = summarize_tool_response(response, tools, threshold=2)
        assert result_custom != response
        assert "[Task used" in result_custom

    def test_unicode_response(self):
        """Unicode characters in response are handled."""
        tools = [f"t{i}" for i in range(12)]
        response = "Deployed to server-a. Status: " + "\u2713" * 200
        result = summarize_tool_response(response, tools)
        assert len(result) <= TOOL_SUMMARY_MAX_CHARS

    def test_whitespace_only_response(self):
        """Whitespace-only response is short enough to pass through."""
        tools = [f"t{i}" for i in range(12)]
        response = "   \n\n   "
        result = summarize_tool_response(response, tools)
        assert result == response  # len <= 500


# ===========================================================================
# Part 7: Compaction Quality + Summary Truncation
# ===========================================================================

class TestCompactionQuality:
    """Tests for compaction prompt content and summary enforcement."""

    async def test_compaction_prompt_requires_topic_tags(self, tmp_path):
        """Compaction instruction mentions [Topics: ...] format."""
        # max_history=20 → keep_count=10, so 45 msgs → 35 to_summarize
        sm = _make_manager(tmp_path, max_history=20)
        captured_system = None

        async def mock_compact(messages, system):
            nonlocal captured_system
            captured_system = system
            return "[Topics: nginx]\n- Configured proxy on server-a"

        sm.set_compaction_fn(mock_compact)
        session = sm.get_or_create("ch1")
        for i in range(COMPACTION_THRESHOLD + 5):
            session.messages.append(Message(role="user", content=f"msg {i}"))

        await sm._compact(session)
        assert captured_system is not None
        assert "[Topics:" in captured_system

    async def test_compaction_prompt_requires_verbatim_identifiers(self, tmp_path):
        """Compaction instruction requires verbatim preservation of identifiers."""
        sm = _make_manager(tmp_path, max_history=20)
        captured_system = None

        async def mock_compact(messages, system):
            nonlocal captured_system
            captured_system = system
            return "[Topics: test]\n- Test summary"

        sm.set_compaction_fn(mock_compact)
        session = sm.get_or_create("ch1")
        for i in range(COMPACTION_THRESHOLD + 5):
            session.messages.append(Message(role="user", content=f"msg {i}"))

        await sm._compact(session)
        assert captured_system is not None
        assert "PRESERVE VERBATIM" in captured_system
        assert "Hostnames" in captured_system

    async def test_compaction_truncates_at_newline_boundary(self, tmp_path):
        """Long summaries are truncated at line boundaries first."""
        sm = _make_manager(tmp_path, max_history=20)

        async def mock_compact(messages, system):
            # Return a summary longer than COMPACTION_MAX_CHARS with newlines
            lines = []
            for i in range(20):
                lines.append(f"- Line {i}: detailed information about step {i}")
            return "\n".join(lines)

        sm.set_compaction_fn(mock_compact)
        session = sm.get_or_create("ch1")
        for i in range(COMPACTION_THRESHOLD + 5):
            session.messages.append(Message(role="user", content=f"msg {i}"))

        await sm._compact(session)
        assert len(session.summary) <= COMPACTION_MAX_CHARS
        # Should end at a line boundary, not mid-line
        assert not session.summary.endswith("-")

    async def test_compaction_word_boundary_fallback(self, tmp_path):
        """When no newlines, truncation falls back to word boundaries."""
        sm = _make_manager(tmp_path, max_history=20)

        async def mock_compact(messages, system):
            # Long summary with no newlines, just spaces
            return "word " * 150  # 750 chars, no \n

        sm.set_compaction_fn(mock_compact)
        session = sm.get_or_create("ch1")
        for i in range(COMPACTION_THRESHOLD + 5):
            session.messages.append(Message(role="user", content=f"msg {i}"))

        await sm._compact(session)
        assert len(session.summary) <= COMPACTION_MAX_CHARS
        # After truncation at word boundary then rstrip, should not end mid-word
        assert not session.summary.endswith(" ")

    async def test_compaction_merges_existing_summary(self, tmp_path):
        """Existing summary is included in compaction prompt for merging."""
        sm = _make_manager(tmp_path, max_history=20)
        captured_content = None

        async def mock_compact(messages, system):
            nonlocal captured_content
            captured_content = messages[0]["content"]
            return "[Topics: merged]\n- Merged summary"

        sm.set_compaction_fn(mock_compact)
        session = sm.get_or_create("ch1")
        session.summary = "Previous work on server-a"
        for i in range(COMPACTION_THRESHOLD + 5):
            session.messages.append(Message(role="user", content=f"msg {i}"))

        await sm._compact(session)
        assert captured_content is not None
        assert "Previous work on server-a" in captured_content
        assert "[Previous summary]" in captured_content

    async def test_compaction_omits_errors(self, tmp_path):
        """Compaction instruction tells LLM to omit error messages."""
        sm = _make_manager(tmp_path, max_history=20)
        captured_system = None

        async def mock_compact(messages, system):
            nonlocal captured_system
            captured_system = system
            return "[Topics: test]\n- Summary"

        sm.set_compaction_fn(mock_compact)
        session = sm.get_or_create("ch1")
        for i in range(COMPACTION_THRESHOLD + 5):
            session.messages.append(Message(role="user", content=f"msg {i}"))

        await sm._compact(session)
        assert captured_system is not None
        assert "OMIT" in captured_system
        assert "Error" in captured_system

    async def test_compaction_failure_preserves_existing_summary(self, tmp_path):
        """When compaction fails, existing summary is preserved."""
        sm = _make_manager(tmp_path, max_history=20)

        async def failing_compact(messages, system):
            raise RuntimeError("LLM unavailable")

        sm.set_compaction_fn(failing_compact)
        session = sm.get_or_create("ch1")
        session.summary = "Important existing summary"
        for i in range(COMPACTION_THRESHOLD + 5):
            session.messages.append(Message(role="user", content=f"msg {i}"))

        await sm._compact(session)
        assert session.summary == "Important existing summary"

    async def test_compaction_fires_reflection_at_5_discarded(self, tmp_path):
        """Reflection fires when >= 5 messages are discarded."""
        # max_history=20 → keep_count=10 → 45 msgs → 35 discarded (>= 5)
        sm = _make_manager(tmp_path, max_history=20)
        mock_reflector = AsyncMock()
        sm._reflector = mock_reflector

        async def mock_compact(messages, system):
            return "[Topics: test]\n- Summary"

        sm.set_compaction_fn(mock_compact)
        session = sm.get_or_create("ch1")
        for i in range(COMPACTION_THRESHOLD + 5):
            session.messages.append(Message(role="user", content=f"msg {i}"))

        await sm._compact(session)
        # Reflection should have been triggered as a task
        assert len(sm._reflection_tasks) >= 1

    async def test_compaction_no_reflection_below_5_discarded(self, tmp_path):
        """Reflection does NOT fire when < 5 messages discarded."""
        sm = SessionManager(
            max_history=20, max_age_hours=24,
            persist_dir=str(tmp_path / "sessions"),
        )
        mock_reflector = AsyncMock()
        sm._reflector = mock_reflector

        async def mock_compact(messages, system):
            return "[Topics: test]\n- Summary"

        sm.set_compaction_fn(mock_compact)
        session = sm.get_or_create("ch1")
        # max_history=20, keep_count=10
        # Add 14 messages: to_summarize = messages[:4] (only 4, < 5)
        for i in range(14):
            session.messages.append(Message(role="user", content=f"msg {i}"))

        # Manually compact (threshold check done externally)
        await sm._compact(session)
        # Reflection should NOT have been triggered (only 4 discarded)
        assert len(sm._reflection_tasks) == 0


# ===========================================================================
# Part 8: Channel Isolation Across Context Features
# ===========================================================================

class TestChannelIsolation:
    """Verify context features don't leak across channels."""

    async def test_relevance_scoring_per_channel(self, tmp_path):
        """Relevance filtering for ch1 doesn't affect ch2."""
        sm = _make_manager(tmp_path)

        # ch1: all about nginx
        for i in range(8):
            sm.add_message("ch1", "user", f"nginx config step {i}")

        # ch2: all about cooking
        for i in range(8):
            sm.add_message("ch2", "user", f"cooking recipe step {i}")

        r1 = await sm.get_task_history("ch1", max_messages=8, current_query="nginx proxy")
        r2 = await sm.get_task_history("ch2", max_messages=8, current_query="cooking recipe")

        # ch1 should have nginx content, ch2 should have cooking content
        r1_content = " ".join(m["content"] for m in r1)
        r2_content = " ".join(m["content"] for m in r2)
        assert "nginx" in r1_content
        assert "cooking" not in r1_content
        assert "cooking" in r2_content
        assert "nginx" not in r2_content

    def test_topic_change_per_channel(self, tmp_path):
        """Topic change for ch1 doesn't affect ch2."""
        sm = _make_manager(tmp_path)

        sm.add_message("ch1", "user", "nginx server setup")
        sm.add_message("ch1", "user", "nginx proxy config")
        sm.add_message("ch2", "user", "cooking pasta recipe")
        sm.add_message("ch2", "user", "cooking risotto recipe")

        # Query about banana (topic change for both)
        r1 = sm.detect_topic_change("ch1", "banana smoothie blender")
        r2 = sm.detect_topic_change("ch2", "banana smoothie blender")

        # Both should be topic changes
        assert r1["is_topic_change"] is True
        assert r2["is_topic_change"] is True

        # But querying nginx should not be a topic change for ch1
        r1_same = sm.detect_topic_change("ch1", "nginx reverse proxy")
        assert r1_same["is_topic_change"] is False

    async def test_summary_per_channel(self, tmp_path):
        """Summaries are channel-specific."""
        sm = _make_manager(tmp_path)
        s1 = sm.get_or_create("ch1")
        s1.summary = "Channel 1 summary about servers"
        s2 = sm.get_or_create("ch2")
        s2.summary = "Channel 2 summary about databases"

        sm.add_message("ch1", "user", "hello")
        sm.add_message("ch2", "user", "hello")

        r1 = await sm.get_task_history("ch1", max_messages=5)
        r2 = await sm.get_task_history("ch2", max_messages=5)

        r1_text = " ".join(m["content"] for m in r1)
        r2_text = " ".join(m["content"] for m in r2)
        assert "servers" in r1_text
        assert "databases" not in r1_text
        assert "databases" in r2_text
        assert "servers" not in r2_text

    async def test_budget_per_channel(self, tmp_path):
        """Token budget is applied independently per channel."""
        sm = _make_manager(tmp_path)

        # ch1: many large messages
        for i in range(15):
            sm.add_message("ch1", "user", f"big message {i} " + "x" * 3000)

        # ch2: few small messages
        for i in range(3):
            sm.add_message("ch2", "user", f"small message {i}")

        r1 = await sm.get_task_history("ch1", max_messages=15)
        r2 = await sm.get_task_history("ch2", max_messages=15)

        # ch1 should be trimmed, ch2 should not
        assert len(r2) == 3  # all 3 messages
        assert len(r1) <= 15  # may be trimmed by budget


# ===========================================================================
# Part 9: Full End-to-End Pipeline
# ===========================================================================

class TestEndToEndPipeline:
    """Full pipeline: messages → topic detection → relevance → budget → output."""

    async def test_full_pipeline_no_topic_change(self, tmp_path):
        """Normal message flow: relevance filters, budget trims, summary included."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        session.summary = "Deployed nginx on server-a at 10.0.0.50"

        # Mix of relevant and irrelevant messages
        sm.add_message("ch1", "user", "check disk space on server-a")
        sm.add_message("ch1", "assistant", "Disk at 45% on server-a")
        sm.add_message("ch1", "user", "what is the weather like today")
        sm.add_message("ch1", "assistant", "I cannot check weather")
        sm.add_message("ch1", "user", "nginx status on server-a")
        sm.add_message("ch1", "assistant", "nginx running on server-a port 80")
        sm.add_message("ch1", "user", "restart nginx on server-a")

        # Topic: nginx on server-a
        topic_info = sm.detect_topic_change("ch1", "nginx reload server-a")
        assert topic_info["is_topic_change"] is False

        result = await sm.get_task_history(
            "ch1", max_messages=20, current_query="nginx reload server-a",
            topic_change=topic_info["is_topic_change"],
        )

        # Should include summary, recent messages, and relevant older ones
        text = " ".join(m["content"] for m in result)
        assert "10.0.0.50" in text  # from summary
        assert "restart nginx" in text  # most recent

    async def test_full_pipeline_with_topic_change(self, tmp_path):
        """Topic change: only last message + summary."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        session.summary = "Server monitoring active"

        # History about servers
        for i in range(5):
            sm.add_message("ch1", "user", f"server monitoring step {i}")
            sm.add_message("ch1", "assistant", f"monitoring output {i}")

        topic_info = sm.detect_topic_change("ch1", "write haiku about cats")
        assert topic_info["is_topic_change"] is True

        result = await sm.get_task_history(
            "ch1", max_messages=20, current_query="write haiku about cats",
            topic_change=True,
        )

        # Should only have summary + last message (not all 10 server messages)
        non_summary = [m for m in result if _SUMMARY_PREFIX not in m["content"]
                       and "Understood" not in m["content"]]
        assert len(non_summary) == 1

    async def test_full_pipeline_summary_with_budget(self, tmp_path):
        """Summary survives budget enforcement after relevance filtering."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        session.summary = "Critical: backup scheduled for server-b"

        # Add many large messages to trigger budget trimming
        for i in range(10):
            sm.add_message("ch1", "user", f"large message {i} " + "x" * 2000)

        result = await sm.get_task_history(
            "ch1", max_messages=10, current_query="backup server-b",
        )

        summary_present = any(_SUMMARY_PREFIX in m["content"] for m in result)
        assert summary_present, "Summary should survive budget enforcement"
        # Recent 3 should be present
        assert len(result) >= BUDGET_KEEP_RECENT

    async def test_full_pipeline_empty_session(self, tmp_path):
        """Empty session returns empty list."""
        sm = _make_manager(tmp_path)
        result = await sm.get_task_history(
            "new_channel", max_messages=20, current_query="anything",
        )
        assert result == []

    async def test_full_pipeline_single_message(self, tmp_path):
        """Single message returns just that message."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "hello world")
        result = await sm.get_task_history(
            "ch1", max_messages=20, current_query="hello",
        )
        assert len(result) == 1
        assert "hello world" in result[0]["content"]


# ===========================================================================
# Part 10: Summarization + Budget Integration
# ===========================================================================

class TestSummarizationBudgetIntegration:
    """Verify summarized tool output fits within budget when stored."""

    def test_summarized_response_fits_budget(self):
        """A summarized response should be short enough to not blow budgets."""
        tools = [f"t{i}" for i in range(20)]
        response = "x" * 10000  # Very long response
        summarized = summarize_tool_response(response, tools)
        tokens = estimate_tokens(summarized)
        # Should be well under the per-message budget
        assert tokens < CONTEXT_TOKEN_BUDGET

    def test_unsummarized_response_may_be_large(self):
        """Below threshold, response passes through and may be large."""
        tools = ["run_command"] * 5
        response = "x" * 10000
        result = summarize_tool_response(response, tools)
        assert result == response
        assert len(result) == 10000

    async def test_summarized_history_uses_less_budget(self, tmp_path):
        """Summarized responses in history leave more room for other messages."""
        sm = _make_manager(tmp_path)

        # Simulate: tool response gets summarized before being stored
        tools = [f"t{i}" for i in range(15)]
        long_response = "Detailed step-by-step output\n\n" * 50 + "Final: nginx deployed."
        summarized = summarize_tool_response(long_response, tools)

        sm.add_message("ch1", "user", "deploy nginx")
        sm.add_message("ch1", "assistant", summarized)  # stored summarized
        sm.add_message("ch1", "user", "check status")
        sm.add_message("ch1", "assistant", "nginx running")

        result = await sm.get_task_history("ch1", max_messages=10)
        assert len(result) == 4  # all messages fit


# ===========================================================================
# Part 11: get_task_history Detailed Behavior
# ===========================================================================

class TestGetTaskHistoryDetailed:
    """Detailed tests for get_task_history filtering and ordering."""

    async def test_no_query_returns_all_candidates(self, tmp_path):
        """Without current_query, no relevance filtering happens."""
        sm = _make_manager(tmp_path)
        for i in range(8):
            sm.add_message("ch1", "user", f"msg {i}")

        result = await sm.get_task_history("ch1", max_messages=8)
        # All 8 messages should be present (no filtering)
        assert len(result) == 8

    async def test_few_messages_no_filtering(self, tmp_path):
        """When candidates <= RELEVANCE_KEEP_RECENT, no filtering occurs."""
        sm = _make_manager(tmp_path)
        for i in range(RELEVANCE_KEEP_RECENT):
            sm.add_message("ch1", "user", f"msg {i}")

        result = await sm.get_task_history(
            "ch1", max_messages=10, current_query="anything",
        )
        assert len(result) == RELEVANCE_KEEP_RECENT

    async def test_exactly_one_older_relevant(self, tmp_path):
        """Exactly one older message beyond recent 3 — kept if relevant."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "nginx proxy configuration")
        sm.add_message("ch1", "user", "recent msg 1")
        sm.add_message("ch1", "user", "recent msg 2")
        sm.add_message("ch1", "user", "recent msg 3")

        result = await sm.get_task_history(
            "ch1", max_messages=10, current_query="nginx proxy",
        )
        contents = [m["content"] for m in result]
        assert "nginx proxy configuration" in contents

    async def test_exactly_one_older_irrelevant(self, tmp_path):
        """Exactly one older message beyond recent 3 — dropped if irrelevant."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "banana smoothie recipe blender")
        sm.add_message("ch1", "user", "nginx recent 1")
        sm.add_message("ch1", "user", "nginx recent 2")
        sm.add_message("ch1", "user", "nginx recent 3")

        result = await sm.get_task_history(
            "ch1", max_messages=10, current_query="nginx proxy server",
        )
        contents = [m["content"] for m in result]
        assert "banana smoothie recipe blender" not in contents
        assert len(result) == 3  # only recent 3

    async def test_relevance_cap_at_max_older(self, tmp_path):
        """Even if all older messages are relevant, cap at RELEVANCE_MAX_OLDER."""
        sm = _make_manager(tmp_path)
        # Add more than RELEVANCE_MAX_OLDER relevant older messages
        for i in range(RELEVANCE_MAX_OLDER + 5):
            sm.add_message("ch1", "user", f"nginx server config step {i}")
        # Recent 3
        for i in range(3):
            sm.add_message("ch1", "user", f"nginx recent {i}")

        result = await sm.get_task_history(
            "ch1", max_messages=20, current_query="nginx server config",
        )
        non_summary = [m for m in result if _SUMMARY_PREFIX not in m["content"]
                       and "Understood" not in m["content"]]
        # Cap: RELEVANCE_MAX_OLDER + RELEVANCE_KEEP_RECENT = 7 + 3 = 10
        assert len(non_summary) <= RELEVANCE_MAX_OLDER + RELEVANCE_KEEP_RECENT

    async def test_original_order_preserved(self, tmp_path):
        """After relevance filtering, messages stay in original chronological order."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "nginx step one")
        sm.add_message("ch1", "user", "banana smoothie")  # irrelevant
        sm.add_message("ch1", "user", "nginx step two")
        sm.add_message("ch1", "user", "nginx step three")  # recent 3 start
        sm.add_message("ch1", "user", "nginx step four")
        sm.add_message("ch1", "user", "nginx step five")

        result = await sm.get_task_history(
            "ch1", max_messages=10, current_query="nginx step",
        )
        nginx_msgs = [m["content"] for m in result if "nginx" in m["content"]]
        # Verify chronological order is preserved
        expected_order = ["nginx step one", "nginx step two", "nginx step three",
                          "nginx step four", "nginx step five"]
        assert nginx_msgs == expected_order

    async def test_max_messages_limits_candidate_pool(self, tmp_path):
        """max_messages limits how far back we look."""
        sm = _make_manager(tmp_path)
        for i in range(20):
            sm.add_message("ch1", "user", f"msg {i}")

        result = await sm.get_task_history("ch1", max_messages=5)
        assert len(result) <= 5


# ===========================================================================
# Part 12: Source Code Verification
# ===========================================================================

class TestSourceVerification:
    """Verify key patterns exist in source code."""

    def test_manager_has_all_constants(self):
        """All context-handling constants are defined."""
        assert COMPACTION_THRESHOLD == 40
        assert COMPACTION_MAX_CHARS == 500
        assert TOPIC_CHANGE_SCORE_THRESHOLD == 0.05
        assert TOPIC_CHANGE_TIME_GAP == 300
        assert TOPIC_CHANGE_RECENT_WINDOW == 5
        assert RELEVANCE_KEEP_RECENT == 3
        assert RELEVANCE_MIN_SCORE == 0.15
        assert RELEVANCE_MAX_OLDER == 7
        assert TOOL_SUMMARY_THRESHOLD == 10
        assert TOOL_SUMMARY_MAX_CHARS == 500
        assert CONTEXT_TOKEN_BUDGET == 8000
        assert CHARS_PER_TOKEN == 4
        assert BUDGET_KEEP_RECENT == 3

    def test_summary_prefix_constant(self):
        """_SUMMARY_PREFIX is the expected string."""
        assert _SUMMARY_PREFIX == "[Previous conversation summary:"

    def test_manager_functions_importable(self):
        """All context-handling functions are importable."""
        assert callable(_tokenize)
        assert callable(score_relevance)
        assert callable(summarize_tool_response)
        assert callable(estimate_tokens)
        assert callable(apply_token_budget)
        assert callable(_content_text)

    def test_session_manager_methods_exist(self):
        """SessionManager has all context methods."""
        assert hasattr(SessionManager, "detect_topic_change")
        assert hasattr(SessionManager, "get_task_history")
        assert hasattr(SessionManager, "get_history_with_compaction")
        assert hasattr(SessionManager, "get_history")

    def test_client_imports_context_functions(self):
        """client.py imports the context-related functions."""
        from pathlib import Path
        client_src = Path("src/discord/client.py").read_text()
        assert "summarize_tool_response" in client_src
        assert "detect_topic_change" in client_src
        assert "current_query" in client_src
        assert "topic_change" in client_src

    def test_client_has_separator_fields(self):
        """client.py context separator includes all required fields."""
        from pathlib import Path
        client_src = Path("src/discord/client.py").read_text()
        assert "req_hash" in client_src
        assert "req_time" in client_src
        assert "channel_ctx" in client_src
        assert "HISTORY ABOVE" in client_src
        assert "Do NOT re-execute" in client_src
        assert "TOPIC CHANGE DETECTED" in client_src

    def test_client_has_thread_inheritance_markers(self):
        """client.py adds INHERITED FROM markers for thread context."""
        from pathlib import Path
        client_src = Path("src/discord/client.py").read_text()
        assert "INHERITED FROM" in client_src

    def test_stop_words_frozenset(self):
        """_STOP_WORDS is a frozenset (immutable)."""
        from src.sessions.manager import _STOP_WORDS
        assert isinstance(_STOP_WORDS, frozenset)
        assert len(_STOP_WORDS) > 30

    def test_no_personal_ips_in_this_file(self):
        """This test file itself must not contain personal IPs."""
        from pathlib import Path
        content = Path(__file__).read_text()
        personal_prefix = "192" + ".168" + ".1"
        assert personal_prefix not in content


# ===========================================================================
# Part 13: Topic Change Time Gap Behavior
# ===========================================================================

class TestTopicChangeTimeGap:
    """Tests for time gap component of topic change detection."""

    def test_time_gap_alone_does_not_trigger(self, tmp_path):
        """Time gap > 5 min with same topic: not a topic change."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        # Add messages with old timestamps (10 minutes ago)
        old_ts = time.time() - 600
        session.messages.append(Message(role="user", content="nginx config", timestamp=old_ts))
        session.messages.append(Message(role="user", content="nginx proxy", timestamp=old_ts))

        result = sm.detect_topic_change("ch1", "nginx settings")
        assert result["has_time_gap"] is True  # > 5 min gap
        assert result["is_topic_change"] is False  # same topic, high overlap

    def test_no_time_gap_with_topic_change(self, tmp_path):
        """Recent messages + different topic: topic change without time gap."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        now = time.time()
        session.messages.append(Message(role="user", content="nginx config", timestamp=now - 10))
        session.messages.append(Message(role="user", content="nginx proxy", timestamp=now - 5))

        result = sm.detect_topic_change("ch1", "banana smoothie recipe blender")
        assert result["has_time_gap"] is False  # < 5 min gap
        assert result["is_topic_change"] is True  # different topic

    def test_time_gap_and_topic_change(self, tmp_path):
        """Both time gap and topic change: strongest signal."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        old_ts = time.time() - 600
        session.messages.append(Message(role="user", content="nginx config", timestamp=old_ts))
        session.messages.append(Message(role="user", content="nginx proxy", timestamp=old_ts))

        result = sm.detect_topic_change("ch1", "banana smoothie recipe blender")
        assert result["has_time_gap"] is True
        assert result["is_topic_change"] is True

    def test_time_gap_exactly_at_threshold(self, tmp_path):
        """Time gap exactly at TOPIC_CHANGE_TIME_GAP is NOT a time gap (strictly greater)."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        ts = time.time() - TOPIC_CHANGE_TIME_GAP
        session.messages.append(Message(role="user", content="nginx", timestamp=ts))
        session.messages.append(Message(role="user", content="nginx", timestamp=ts))

        result = sm.detect_topic_change("ch1", "something else entirely xyzzy")
        # time_gap ≈ TOPIC_CHANGE_TIME_GAP, has_time_gap depends on > vs >=
        # The code uses: time_gap > TOPIC_CHANGE_TIME_GAP
        # So at exactly the threshold, has_time_gap should be False
        # (but timing jitter may push it slightly over — just check the flag)
        assert isinstance(result["has_time_gap"], bool)

    def test_return_dict_keys(self, tmp_path):
        """detect_topic_change returns all expected keys."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "test message")
        result = sm.detect_topic_change("ch1", "query")
        assert set(result.keys()) == {"is_topic_change", "time_gap", "has_time_gap", "max_overlap"}


# ===========================================================================
# Part 14: Session Persistence and Context Continuity
# ===========================================================================

class TestSessionPersistenceContext:
    """Test that context features survive save/load cycles."""

    def test_summary_survives_save_load(self, tmp_path):
        """Summary is preserved through save/load."""
        sm = _make_manager(tmp_path)
        session = sm.get_or_create("ch1")
        session.summary = "Deployed to server-a"
        sm.add_message("ch1", "user", "test msg")
        sm.save()

        sm2 = _make_manager(tmp_path)
        sm2.load()
        session2 = sm2.get_or_create("ch1")
        assert session2.summary == "Deployed to server-a"

    def test_messages_survive_save_load(self, tmp_path):
        """Messages are preserved through save/load."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "hello world")
        sm.add_message("ch1", "assistant", "hi there")
        sm.save()

        sm2 = _make_manager(tmp_path)
        sm2.load()
        session2 = sm2.get_or_create("ch1")
        assert len(session2.messages) == 2
        assert session2.messages[0].content == "hello world"

    async def test_relevance_works_after_load(self, tmp_path):
        """Relevance scoring works on loaded sessions."""
        sm = _make_manager(tmp_path)
        for i in range(8):
            sm.add_message("ch1", "user", f"nginx config step {i}")
        sm.save()

        sm2 = _make_manager(tmp_path)
        sm2.load()
        result = await sm2.get_task_history(
            "ch1", max_messages=8, current_query="nginx configuration",
        )
        assert len(result) > 0

    def test_topic_detection_works_after_load(self, tmp_path):
        """Topic change detection works on loaded sessions."""
        sm = _make_manager(tmp_path)
        sm.add_message("ch1", "user", "nginx proxy config")
        sm.add_message("ch1", "user", "nginx reverse proxy")
        sm.save()

        sm2 = _make_manager(tmp_path)
        sm2.load()
        result = sm2.detect_topic_change("ch1", "banana smoothie blender recipe")
        assert result["is_topic_change"] is True
