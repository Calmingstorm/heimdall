"""Tests for claude_code improvements.

Round 2, Session 7: Improvements to _handle_claude_code:

1. Configurable output limit: _handle_claude_code now accepts max_output_chars
   parameter (default 3000 for tool loop).

Note: claude_code routing path tests removed — all messages now route to "task"
(no classifier). The _handle_claude_code method is still used as a tool, so
the unit tests for max_output_chars remain valid.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# Configurable max_output_chars
# ---------------------------------------------------------------------------

class TestClaudeCodeMaxOutputChars:
    """_handle_claude_code should respect the max_output_chars parameter."""

    @pytest.fixture
    def executor(self):
        """Create a minimal ToolExecutor-like object for testing."""
        from src.tools.executor import ToolExecutor
        config = MagicMock()
        config.hosts = {"desktop": MagicMock(address="10.0.0.2", ssh_user="root", os="linux")}
        config.ssh_key_path = "/app/.ssh/id_ed25519"
        config.ssh_known_hosts_path = "/app/.ssh/known_hosts"
        config.command_timeout_seconds = 30
        config.memory_path = None
        return ToolExecutor(config)

    async def test_default_truncation_at_3000(self, executor):
        """Without max_output_chars, output should be truncated at 3000 chars."""
        long_output = "x" * 5000
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        assert len(result) < 5000
        assert "[... truncated ...]" in result
        # Default 3000: first 1500 + truncation notice + last 1500
        assert result.startswith("x" * 1500)
        assert result.endswith("x" * 1500)

    async def test_custom_max_output_chars_8000(self, executor):
        """With max_output_chars=8000, more output should be preserved."""
        long_output = "y" * 10000
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
                "max_output_chars": 8000,
            })
        assert "[... truncated ...]" in result
        # 8000: first 4000 + truncation notice + last 4000
        assert result.startswith("y" * 4000)
        assert result.endswith("y" * 4000)

    async def test_short_output_not_truncated_default(self, executor):
        """Short output should pass through unchanged regardless of limit."""
        short_output = "The function does X."
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, short_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        assert result == short_output

    async def test_short_output_not_truncated_custom(self, executor):
        """Short output should pass through unchanged even with custom limit."""
        short_output = "The function does X."
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, short_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
                "max_output_chars": 8000,
            })
        assert result == short_output

    async def test_output_at_limit_not_truncated(self, executor):
        """Output exactly at max_output_chars should NOT be truncated."""
        exact_output = "z" * 3000
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, exact_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        assert result == exact_output
        assert "[... truncated ...]" not in result

    async def test_output_one_over_limit_truncated(self, executor):
        """Output one char over max_output_chars should be truncated."""
        over_output = "a" * 3001
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, over_output)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        assert "[... truncated ...]" in result

    async def test_error_output_not_truncated(self, executor):
        """Failed command output (exit != 0) has its own truncation, not max_output_chars."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "error " * 1000)
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
            })
        # Error path uses output[-2000:], not max_output_chars
        assert result.startswith("Claude Code failed")


