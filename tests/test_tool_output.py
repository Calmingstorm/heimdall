"""Tests for tool output truncation in the Discord client."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub out heavy discord extension before any src.discord imports trigger __init__
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import truncate_tool_output, TOOL_OUTPUT_MAX_CHARS  # noqa: E402


# ---------------------------------------------------------------------------
# truncate_tool_output — basic behavior
# ---------------------------------------------------------------------------

class TestTruncateToolOutput:
    """Tests for the truncate_tool_output() function."""

    def test_short_output_unchanged(self):
        """Output shorter than the limit passes through unchanged."""
        text = "OK — service started."
        assert truncate_tool_output(text) == text

    def test_empty_string_unchanged(self):
        assert truncate_tool_output("") == ""

    def test_exactly_at_limit_unchanged(self):
        """Output exactly at the limit is NOT truncated."""
        text = "x" * TOOL_OUTPUT_MAX_CHARS
        assert truncate_tool_output(text) == text

    def test_one_over_limit_is_truncated(self):
        """Output one character over the limit IS truncated (notice is added)."""
        text = "x" * (TOOL_OUTPUT_MAX_CHARS + 1)
        result = truncate_tool_output(text)
        # The middle is replaced with a truncation notice
        assert "characters omitted" in result
        assert result != text

    def test_large_output_truncated(self):
        """Large output is truncated with start + end preserved."""
        text = "A" * 5000 + "MIDDLE" + "B" * 5000 + "C" * 10000
        result = truncate_tool_output(text, max_chars=8000)
        assert len(result) < len(text)
        # Start is preserved
        assert result.startswith("A" * 100)
        # End is preserved
        assert result.endswith("C" * 100)
        # Truncation notice is present
        assert "characters omitted" in result

    def test_truncation_notice_correct_count(self):
        """The truncation notice reports the correct number of omitted characters."""
        size = 20000
        limit = 8000
        text = "x" * size
        result = truncate_tool_output(text, max_chars=limit)
        expected_omitted = size - limit
        assert f"{expected_omitted} characters omitted" in result

    def test_custom_max_chars(self):
        """Custom max_chars parameter is respected."""
        text = "x" * 500
        # Should not truncate at default limit
        assert truncate_tool_output(text) == text
        # Should truncate at custom limit
        result = truncate_tool_output(text, max_chars=100)
        assert len(result) < len(text)
        assert "characters omitted" in result

    def test_preserves_start_and_end_content(self):
        """Truncation preserves the first half and last half of the limit."""
        start = "START_MARKER_" * 100   # 1300 chars
        end = "_END_MARKER" * 100       # 1100 chars
        middle = "x" * 20000
        text = start + middle + end
        result = truncate_tool_output(text, max_chars=6000)
        # First 3000 chars should contain the start marker
        assert "START_MARKER_" in result[:3100]
        # Last 3000 chars should contain the end marker
        assert "_END_MARKER" in result[-3100:]

    def test_result_contains_three_parts(self):
        """Truncated output has: start + notice + end."""
        text = "A" * 10000 + "B" * 10000
        result = truncate_tool_output(text, max_chars=6000)
        parts = result.split("[...")
        assert len(parts) == 2, "Expected exactly one truncation marker"
        # First part is start content
        assert parts[0].startswith("A" * 100)
        # Second part contains the notice and end content
        assert "characters omitted" in parts[1]
        assert parts[1].rstrip().endswith("B" * 100)

    def test_multiline_output_truncated(self):
        """Realistic multi-line tool output (like df -h or logs) truncates correctly."""
        lines = [f"line-{i:04d}: " + "x" * 80 for i in range(200)]
        text = "\n".join(lines)  # ~18000 chars
        result = truncate_tool_output(text, max_chars=6000)
        # First line preserved
        assert "line-0000" in result
        # Last line preserved
        assert "line-0199" in result
        # Truncation happened
        assert "characters omitted" in result


# ---------------------------------------------------------------------------
# TOOL_OUTPUT_MAX_CHARS — constant sanity checks
# ---------------------------------------------------------------------------

class TestToolOutputConstant:
    """Ensure the constant is set to a reasonable value."""

    def test_max_chars_is_positive(self):
        assert TOOL_OUTPUT_MAX_CHARS > 0

    def test_max_chars_is_reasonable_range(self):
        """Limit should be between 4K and 50K chars (~1-12K tokens)."""
        assert 4000 <= TOOL_OUTPUT_MAX_CHARS <= 50000

    def test_default_matches_constant(self):
        """truncate_tool_output default matches TOOL_OUTPUT_MAX_CHARS."""
        text = "x" * (TOOL_OUTPUT_MAX_CHARS + 1)
        result = truncate_tool_output(text)
        assert "characters omitted" in result

        text_at_limit = "x" * TOOL_OUTPUT_MAX_CHARS
        assert truncate_tool_output(text_at_limit) == text_at_limit


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTruncateEdgeCases:
    """Edge cases for truncation."""

    def test_max_chars_of_one(self):
        """Degenerate max_chars=1 still works without crashing."""
        result = truncate_tool_output("hello world", max_chars=1)
        assert "characters omitted" in result

    def test_max_chars_of_zero(self):
        """max_chars=0: any non-empty input is truncated."""
        result = truncate_tool_output("hello", max_chars=0)
        assert "characters omitted" in result

    def test_unicode_content(self):
        """Unicode content is handled correctly."""
        text = "\U0001f600" * 10000  # emoji characters
        result = truncate_tool_output(text, max_chars=100)
        assert "characters omitted" in result

    def test_json_output(self):
        """JSON tool output (like Prometheus) truncates without crashing."""
        import json
        data = {"status": "success", "data": {"result": [{"metric": {"__name__": f"metric_{i}"}, "value": [1234567890, str(i)]} for i in range(500)]}}
        text = json.dumps(data)
        result = truncate_tool_output(text, max_chars=2000)
        # Start should have the JSON opening
        assert result.startswith('{"status"')
        assert "characters omitted" in result

    def test_binary_looking_output(self):
        """Output with null bytes or binary-ish content doesn't crash."""
        text = "normal text\x00\x01\x02" * 5000
        result = truncate_tool_output(text, max_chars=1000)
        assert "characters omitted" in result
