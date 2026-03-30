"""Tests for run_command / run_command_multi output line truncation.

Session 20: Added per-handler line-aware truncation so the LLM receives
complete lines rather than mid-line character-based cuts from the
central truncate_tool_output.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.config.schema import ToolsConfig
from src.tools.executor import (
    ToolExecutor,
    _truncate_lines,
    _RUN_COMMAND_MAX_LINES,
)


# ---------------------------------------------------------------------------
# _truncate_lines unit tests
# ---------------------------------------------------------------------------


class TestTruncateLines:
    """Unit tests for the _truncate_lines helper."""

    def test_short_output_passthrough(self):
        """Output under the limit is returned unchanged."""
        text = "line1\nline2\nline3"
        assert _truncate_lines(text) == text

    def test_exact_limit_passthrough(self):
        """Output at exactly max_lines is returned unchanged."""
        lines = [f"line{i}" for i in range(_RUN_COMMAND_MAX_LINES)]
        text = "\n".join(lines)
        assert _truncate_lines(text) == text

    def test_over_limit_truncated(self):
        """Output exceeding max_lines is truncated with a notice."""
        count = _RUN_COMMAND_MAX_LINES + 100
        lines = [f"line{i}" for i in range(count)]
        result = _truncate_lines("\n".join(lines))
        result_lines = result.split("\n")
        # Should have max_lines + 1 (for the notice line)
        assert len(result_lines) == _RUN_COMMAND_MAX_LINES + 1

    def test_preserves_first_lines(self):
        """First half of kept lines comes from the start of output."""
        count = 300
        lines = [f"L{i}" for i in range(count)]
        result = _truncate_lines("\n".join(lines), max_lines=100)
        result_lines = result.split("\n")
        # First 50 lines should be L0..L49
        for i in range(50):
            assert result_lines[i] == f"L{i}"

    def test_preserves_last_lines(self):
        """Last half of kept lines comes from the end of output."""
        count = 300
        lines = [f"L{i}" for i in range(count)]
        result = _truncate_lines("\n".join(lines), max_lines=100)
        result_lines = result.split("\n")
        # Last 50 lines should be L250..L299
        for i in range(50):
            assert result_lines[-(50 - i)] == f"L{250 + i}"

    def test_notice_mentions_omitted_count(self):
        """The truncation notice states how many lines were omitted."""
        count = 500
        lines = [f"line{i}" for i in range(count)]
        result = _truncate_lines("\n".join(lines), max_lines=200)
        assert "300 lines omitted" in result

    def test_notice_suggests_pipe_commands(self):
        """The truncation notice suggests head/tail/grep."""
        lines = [f"x{i}" for i in range(300)]
        result = _truncate_lines("\n".join(lines), max_lines=100)
        assert "head" in result
        assert "tail" in result
        assert "grep" in result

    def test_empty_string(self):
        """Empty string is returned unchanged."""
        assert _truncate_lines("") == ""

    def test_single_line(self):
        """Single line is returned unchanged."""
        assert _truncate_lines("hello") == "hello"

    def test_custom_max_lines(self):
        """Custom max_lines parameter is respected."""
        lines = [f"L{i}" for i in range(50)]
        result = _truncate_lines("\n".join(lines), max_lines=10)
        result_lines = result.split("\n")
        # 5 first + 1 notice + 5 last = 11
        assert len(result_lines) == 11
        assert "40 lines omitted" in result

    def test_content_integrity(self):
        """Truncated output contains only original content plus the notice."""
        lines = [f"data_{i}" for i in range(100)]
        result = _truncate_lines("\n".join(lines), max_lines=20)
        result_lines = result.split("\n")
        # All non-notice lines should be from the original
        for line in result_lines:
            if "omitted" not in line:
                assert line.startswith("data_")

    def test_one_over_limit(self):
        """One line over the limit still truncates."""
        lines = [f"L{i}" for i in range(11)]
        result = _truncate_lines("\n".join(lines), max_lines=10)
        assert "1 lines omitted" in result
        result_lines = result.split("\n")
        # 5 first + 1 notice + 5 last = 11
        assert len(result_lines) == 11


# ---------------------------------------------------------------------------
# Constant validation
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify the constant is reasonable."""

    def test_max_lines_matches_convention(self):
        """_RUN_COMMAND_MAX_LINES should be 200, matching docker_logs/read_file convention."""
        assert _RUN_COMMAND_MAX_LINES == 200

    def test_max_lines_positive(self):
        assert _RUN_COMMAND_MAX_LINES > 0


# ---------------------------------------------------------------------------
# Integration: _handle_run_command applies truncation
# ---------------------------------------------------------------------------


@pytest.fixture
def executor(tools_config: ToolsConfig, tmp_dir: Path) -> ToolExecutor:
    return ToolExecutor(tools_config, memory_path=str(tmp_dir / "memory.json"))


class TestRunCommandTruncation:
    """Verify _handle_run_command applies line truncation."""

    @pytest.mark.asyncio
    async def test_short_output_unchanged(self, executor: ToolExecutor):
        """Short command output passes through without truncation."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "hello world")
            result = await executor.execute("run_command", {
                "host": "server", "command": "echo hello world",
            })
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_long_output_truncated(self, executor: ToolExecutor):
        """Command output exceeding _RUN_COMMAND_MAX_LINES is truncated."""
        long_output = "\n".join(f"proc{i}" for i in range(500))
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor.execute("run_command", {
                "host": "server", "command": "ps aux",
            })
        assert "lines omitted" in result
        assert "proc0" in result  # first lines preserved
        assert "proc499" in result  # last lines preserved

    @pytest.mark.asyncio
    async def test_error_output_not_truncated(self, executor: ToolExecutor):
        """Error messages from failed commands are not subject to line truncation."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "command not found")
            result = await executor.execute("run_command", {
                "host": "server", "command": "badcmd",
            })
        # Error wrapping from _run_on_host takes precedence
        assert "Command failed" in result


class TestRunCommandMultiTruncation:
    """Verify _handle_run_command_multi applies per-host line truncation."""

    @pytest.mark.asyncio
    async def test_per_host_truncation(self, executor: ToolExecutor):
        """Each host's output is individually truncated."""
        long_output = "\n".join(f"line{i}" for i in range(500))
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor.execute("run_command_multi", {
                "hosts": ["server", "desktop"],
                "command": "ps aux",
            })
        # Both hosts should have truncated output
        assert result.count("lines omitted") == 2
        # Both hosts should have headers
        assert "### server" in result
        assert "### desktop" in result

    @pytest.mark.asyncio
    async def test_short_multi_output_unchanged(self, executor: ToolExecutor):
        """Short output from multiple hosts is not truncated."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "all good")
            result = await executor.execute("run_command_multi", {
                "hosts": ["server", "desktop"],
                "command": "uptime",
            })
        assert "lines omitted" not in result
        assert "all good" in result

    @pytest.mark.asyncio
    async def test_all_hosts_expansion(self, executor: ToolExecutor):
        """'all' host target expands to all configured hosts."""
        long_output = "\n".join(f"L{i}" for i in range(300))
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor.execute("run_command_multi", {
                "hosts": ["all"],
                "command": "ps aux",
            })
        # Should have one section per configured host (server, desktop, macbook)
        assert "### server" in result
        assert "### desktop" in result
        assert "### macbook" in result
        # Each should be truncated
        assert result.count("lines omitted") == 3
