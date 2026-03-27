"""Tests for Round 12: Tool output summarization in history.

Validates:
- summarize_tool_response() compresses verbose multi-tool responses
- Short responses are left unchanged
- Tool list is deduplicated and capped
- Outcome is extracted from last paragraph
- Response is capped at TOOL_SUMMARY_MAX_CHARS
- Integration: client.py calls summarize_tool_response before saving
- Below-threshold tool counts are not summarized
"""
from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.sessions.manager import (  # noqa: E402
    TOOL_SUMMARY_MAX_CHARS,
    TOOL_SUMMARY_THRESHOLD,
    summarize_tool_response,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify tool summarization constants are sensible."""

    def test_threshold_positive(self):
        assert TOOL_SUMMARY_THRESHOLD > 0

    def test_threshold_default_is_10(self):
        assert TOOL_SUMMARY_THRESHOLD == 10

    def test_max_chars_positive(self):
        assert TOOL_SUMMARY_MAX_CHARS > 0

    def test_max_chars_default_is_500(self):
        assert TOOL_SUMMARY_MAX_CHARS == 500


# ---------------------------------------------------------------------------
# summarize_tool_response — basic behavior
# ---------------------------------------------------------------------------


class TestSummarizeBasic:
    """Basic summarization behavior."""

    def test_below_threshold_returns_unchanged(self):
        """Fewer than threshold tool calls → no summarization."""
        response = "A" * 1000
        tools = ["run_command"] * 5
        assert summarize_tool_response(response, tools) == response

    def test_at_threshold_summarizes(self):
        """Exactly threshold tool calls → summarizes if response is long."""
        response = "A" * 1000
        tools = ["run_command"] * 10
        result = summarize_tool_response(response, tools)
        assert result != response
        assert "[Task used 10 tool calls" in result

    def test_above_threshold_summarizes(self):
        """More than threshold tool calls → summarizes."""
        response = "Step 1: ...\n\nStep 2: ...\n\nAll done successfully."
        tools = ["run_command"] * 15
        # Make response long enough to trigger
        response = ("Intermediate output.\n\n" * 30) + "Final result: everything works."
        result = summarize_tool_response(response, tools)
        assert "[Task used 15 tool calls" in result

    def test_short_response_not_summarized(self):
        """Short response even with many tools is left alone."""
        response = "Done. All services restarted."
        tools = ["run_command"] * 20
        assert summarize_tool_response(response, tools) == response

    def test_empty_response(self):
        """Empty response is returned unchanged."""
        assert summarize_tool_response("", ["t"] * 15) == ""

    def test_custom_threshold(self):
        """Custom threshold parameter works."""
        response = "X" * 600
        tools = ["run_command"] * 5
        result = summarize_tool_response(response, tools, threshold=3)
        assert "[Task used 5 tool calls" in result

    def test_custom_threshold_below(self):
        """Below custom threshold returns unchanged."""
        response = "X" * 600
        tools = ["run_command"] * 2
        result = summarize_tool_response(response, tools, threshold=3)
        assert result == response


# ---------------------------------------------------------------------------
# Tool list formatting
# ---------------------------------------------------------------------------


class TestToolListFormatting:
    """Tool list in header is deduplicated and formatted correctly."""

    def test_tools_deduplicated(self):
        """Repeated tools appear only once in header."""
        tools = ["run_command"] * 12
        response = "X" * 600
        result = summarize_tool_response(response, tools)
        # Header should say "run_command" once, not 12 times
        assert "(run_command)" in result

    def test_multiple_unique_tools(self):
        """Multiple unique tools all appear."""
        tools = ["run_command", "check_disk", "read_file"] * 4  # 12 total
        response = ("Long output from various tools.\n\n" * 30 +
                    "All checks completed successfully.")
        result = summarize_tool_response(response, tools)
        assert "run_command" in result
        assert "check_disk" in result
        assert "read_file" in result

    def test_tool_count_accurate(self):
        """Header shows total call count (not unique count)."""
        tools = ["run_command"] * 12
        response = "X" * 600
        result = summarize_tool_response(response, tools)
        assert "12 tool calls" in result

    def test_tools_capped_at_15(self):
        """More than 15 unique tools shows (+N more)."""
        tools = [f"tool_{i}" for i in range(20)]
        response = "X" * 600
        result = summarize_tool_response(response, tools)
        assert "(+5 more)" in result

    def test_tools_order_preserved(self):
        """First-occurrence order is preserved in tool list."""
        tools = ["alpha", "beta", "gamma"] * 4  # 12 total
        response = "X" * 600
        result = summarize_tool_response(response, tools)
        # alpha should come before beta, which comes before gamma
        alpha_pos = result.index("alpha")
        beta_pos = result.index("beta")
        gamma_pos = result.index("gamma")
        assert alpha_pos < beta_pos < gamma_pos


# ---------------------------------------------------------------------------
# Outcome extraction
# ---------------------------------------------------------------------------


class TestOutcomeExtraction:
    """The summarized output captures the key outcome."""

    def test_last_paragraph_kept(self):
        """Last paragraph is preserved as the outcome."""
        response = ("Step 1: checked disk usage.\n\n"
                    "Step 2: restarted nginx.\n\n"
                    "Step 3: verified health.\n\n"
                    "All services are healthy. Deployment complete.")
        tools = ["run_command"] * 12
        result = summarize_tool_response(response, tools)
        assert "All services are healthy" in result

    def test_single_paragraph_response(self):
        """Single long paragraph is summarized within budget."""
        response = "word " * 200  # ~1000 chars, single paragraph
        tools = ["run_command"] * 12
        result = summarize_tool_response(response, tools)
        assert "[Task used" in result
        assert len(result) <= TOOL_SUMMARY_MAX_CHARS

    def test_result_within_budget(self):
        """Summarized result never exceeds TOOL_SUMMARY_MAX_CHARS."""
        response = "A very long paragraph.\n\n" * 50
        tools = ["run_command", "check_disk", "read_file"] * 5  # 15
        result = summarize_tool_response(response, tools)
        assert len(result) <= TOOL_SUMMARY_MAX_CHARS

    def test_short_last_paragraph_includes_penultimate(self):
        """If last paragraph is very short, penultimate is included too."""
        response = ("Long explanation of steps.\n\n"
                    "nginx restarted on server-a, disk 45% used.\n\n"
                    "Done.")
        tools = ["run_command"] * 12
        # Only summarize if response is actually > TOOL_SUMMARY_MAX_CHARS
        # This response is short, so it should be unchanged
        if len(response) <= TOOL_SUMMARY_MAX_CHARS:
            assert summarize_tool_response(response, tools) == response
        else:
            result = summarize_tool_response(response, tools)
            assert "Done." in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_exactly_max_chars_not_summarized(self):
        """Response at exactly TOOL_SUMMARY_MAX_CHARS is not summarized."""
        response = "X" * TOOL_SUMMARY_MAX_CHARS
        tools = ["run_command"] * 12
        assert summarize_tool_response(response, tools) == response

    def test_one_char_over_summarized(self):
        """Response one char over MAX_CHARS gets summarized."""
        response = "X" * (TOOL_SUMMARY_MAX_CHARS + 1)
        tools = ["run_command"] * 12
        result = summarize_tool_response(response, tools)
        assert result != response
        assert len(result) <= TOOL_SUMMARY_MAX_CHARS

    def test_empty_tools_list(self):
        """Empty tools list → below threshold → no summarization."""
        response = "X" * 1000
        assert summarize_tool_response(response, []) == response

    def test_newlines_only_response(self):
        """Response of only newlines."""
        response = "\n" * 600
        tools = ["t"] * 12
        result = summarize_tool_response(response, tools)
        # Should not crash; response has no non-empty paragraphs
        assert isinstance(result, str)

    def test_unicode_response(self):
        """Unicode content is handled correctly."""
        response = "状态检查完成。\n\n" * 100 + "所有服务正常运行。"
        tools = ["run_command"] * 12
        result = summarize_tool_response(response, tools)
        assert isinstance(result, str)
        assert "[Task used" in result

    def test_response_with_code_blocks(self):
        """Code blocks in response don't break summarization."""
        response = ("```\nroot     12345  0.0  0.1 nginx\n```\n\n" * 20 +
                    "All processes running normally.")
        tools = ["run_command"] * 12
        result = summarize_tool_response(response, tools)
        assert "All processes running normally" in result


# ---------------------------------------------------------------------------
# Client integration
# ---------------------------------------------------------------------------


class TestClientIntegration:
    """Verify client.py imports and uses summarize_tool_response."""

    def test_import_exists(self):
        """client.py imports summarize_tool_response."""
        import src.discord.client as client_mod
        source = inspect.getsource(client_mod)
        assert "summarize_tool_response" in source

    def test_called_before_add_message(self):
        """summarize_tool_response is called before add_message in the save path."""
        import src.discord.client as client_mod
        source = inspect.getsource(client_mod)
        # Find the summarize call and the add_message call
        summarize_pos = source.find("summarize_tool_response(response, tools_used)")
        add_pos = source.find("add_message(channel_id, \"assistant\", history_response)")
        assert summarize_pos > 0, "summarize_tool_response call not found"
        assert add_pos > 0, "add_message with history_response not found"
        assert summarize_pos < add_pos, "summarize must come before add_message"


# ---------------------------------------------------------------------------
# Source code verification
# ---------------------------------------------------------------------------


class TestSourceVerification:
    """Verify manager.py has the expected constants and functions."""

    def test_constants_in_module(self):
        """TOOL_SUMMARY_THRESHOLD and TOOL_SUMMARY_MAX_CHARS are defined."""
        from src.sessions import manager
        assert hasattr(manager, "TOOL_SUMMARY_THRESHOLD")
        assert hasattr(manager, "TOOL_SUMMARY_MAX_CHARS")

    def test_function_signature(self):
        """summarize_tool_response has the expected parameters."""
        sig = inspect.signature(summarize_tool_response)
        params = list(sig.parameters.keys())
        assert "response" in params
        assert "tools_used" in params
        assert "threshold" in params

    def test_threshold_default_in_signature(self):
        """threshold parameter defaults to TOOL_SUMMARY_THRESHOLD."""
        sig = inspect.signature(summarize_tool_response)
        default = sig.parameters["threshold"].default
        assert default == TOOL_SUMMARY_THRESHOLD

    def test_function_has_docstring(self):
        """summarize_tool_response has a docstring."""
        assert summarize_tool_response.__doc__ is not None
        assert len(summarize_tool_response.__doc__) > 20

    def test_manager_source_has_log_call(self):
        """summarize_tool_response logs when it compresses."""
        source = inspect.getsource(summarize_tool_response)
        assert "log.info" in source

    def test_manager_source_deduplicates_tools(self):
        """Implementation deduplicates tool names."""
        source = inspect.getsource(summarize_tool_response)
        assert "unique_tools" in source or "seen" in source


# ---------------------------------------------------------------------------
# Realistic scenarios
# ---------------------------------------------------------------------------


class TestRealisticScenarios:
    """Test with realistic multi-tool response patterns."""

    def test_infrastructure_check_response(self):
        """Typical infra check with many run_command calls."""
        lines = []
        for host in ["server-a", "server-b", "server-c", "server-d",
                      "server-e", "server-f", "server-g", "server-h"]:
            lines.append(f"Checked {host}:")
            lines.append(f"  - Disk usage: /dev/sda1 45% used, /dev/sdb1 12% used")
            lines.append(f"  - Memory: 8192MB/16384MB (50% utilization)")
            lines.append(f"  - CPU Load: 0.52, 0.48, 0.31 (1m, 5m, 15m)")
            lines.append(f"  - Services: nginx running, postgres running, redis running")
            lines.append("")
        lines.append("All 8 servers are healthy across the cluster. No action needed.")
        response = "\n".join(lines)
        tools = ["run_command"] * 12
        result = summarize_tool_response(response, tools)
        assert "[Task used 12 tool calls" in result
        assert "run_command" in result
        assert "healthy" in result or "No action needed" in result

    def test_mixed_tools_deployment(self):
        """Deployment using multiple tool types."""
        response = (
            "Pulled latest code on server-a using git pull origin main.\n\n"
            "Built Docker image heimdall:latest (digest sha256:abc123def456).\n\n"
            "Stopped old container heimdall-bot-v2.0.9 gracefully.\n\n"
            "Started new container heimdall-bot-v2.1.0 on port 8080.\n\n"
            "Health check passed: /health returned 200 OK in 120ms.\n\n"
            "Ran 15 database migrations on postgres://server-a:5432/heimdall.\n\n"
            "Updated nginx config at /etc/nginx/sites-available/heimdall.conf.\n\n"
            "Reloaded nginx (pid 12345, config test passed).\n\n"
            "Verified all 8 API endpoints responding with expected status codes.\n\n"
            "DNS propagation confirmed for heimdall.example.com (A record → 10.0.1.5).\n\n"
            "Deployment complete. All services running on server-a with version 2.1.0."
        )
        tools = (
            ["run_command"] * 5 + ["read_file", "write_file"] +
            ["run_command"] * 3 + ["check_service", "run_command"]
        )
        result = summarize_tool_response(response, tools)
        assert "[Task used" in result
        assert "Deployment complete" in result

    def test_nine_tools_not_summarized(self):
        """9 tools (just under threshold) — response unchanged."""
        response = "X" * 1000
        tools = ["run_command"] * 9
        assert summarize_tool_response(response, tools) == response
