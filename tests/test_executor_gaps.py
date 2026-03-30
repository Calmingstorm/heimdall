"""Tests for executor.py — covering untested handlers: browser, claude_code,
run_command_multi error branch."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolsConfig
from src.tools.executor import ToolExecutor


@pytest.fixture
def executor(tools_config: ToolsConfig, tmp_dir: Path) -> ToolExecutor:
    return ToolExecutor(tools_config, memory_path=str(tmp_dir / "memory.json"))


# ---------------------------------------------------------------------------
# Browser handlers — disabled path
# ---------------------------------------------------------------------------

class TestBrowserHandlersDisabled:
    """When browser_manager is None, all browser tools return a descriptive error."""

    @pytest.mark.asyncio
    async def test_browser_read_page_disabled(self, executor):
        """browser_read_page without browser manager returns disabled message."""
        result = await executor.execute("browser_read_page", {"url": "https://example.com"})
        assert "not enabled" in result.lower()

    @pytest.mark.asyncio
    async def test_browser_read_table_disabled(self, executor):
        """browser_read_table without browser manager returns disabled message."""
        result = await executor.execute("browser_read_table", {"url": "https://example.com"})
        assert "not enabled" in result.lower()

    @pytest.mark.asyncio
    async def test_browser_click_disabled(self, executor):
        """browser_click without browser manager returns disabled message."""
        result = await executor.execute("browser_click", {"selector": "#btn"})
        assert "not enabled" in result.lower()

    @pytest.mark.asyncio
    async def test_browser_fill_disabled(self, executor):
        """browser_fill without browser manager returns disabled message."""
        result = await executor.execute("browser_fill", {
            "selector": "#input", "value": "hello",
        })
        assert "not enabled" in result.lower()

    @pytest.mark.asyncio
    async def test_browser_evaluate_disabled(self, executor):
        """browser_evaluate without browser manager returns disabled message."""
        result = await executor.execute("browser_evaluate", {
            "expression": "document.title",
        })
        assert "not enabled" in result.lower()


# ---------------------------------------------------------------------------
# Browser handlers — enabled path (delegating to browser module)
# ---------------------------------------------------------------------------

class TestBrowserHandlersEnabled:
    @pytest.mark.asyncio
    async def test_browser_read_page_enabled(self, executor):
        """browser_read_page with browser manager delegates to handler."""
        executor._browser_manager = MagicMock()
        with patch(
            "src.tools.browser.handle_browser_read_page",
            new_callable=AsyncMock, return_value="Page text here",
        ):
            result = await executor.execute("browser_read_page", {"url": "https://example.com"})
        assert result == "Page text here"

    @pytest.mark.asyncio
    async def test_browser_click_enabled(self, executor):
        """browser_click with browser manager delegates to handler."""
        executor._browser_manager = MagicMock()
        with patch(
            "src.tools.browser.handle_browser_click",
            new_callable=AsyncMock, return_value="Clicked",
        ):
            result = await executor.execute("browser_click", {"selector": "#btn"})
        assert result == "Clicked"


# ---------------------------------------------------------------------------
# claude_code
# ---------------------------------------------------------------------------

class TestClaudeCode:
    @pytest.mark.asyncio
    async def test_claude_code_readonly(self, executor):
        """claude_code in read-only mode (default) runs without --dangerously-skip-permissions."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Analysis complete: all good")
            result = await executor.execute("claude_code", {
                "host": "server",
                "working_directory": "/opt/project",
                "prompt": "Check the code",
            })
        assert "Analysis complete" in result
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--print" in cmd
        assert "--dangerously-skip-permissions" not in cmd

    @pytest.mark.asyncio
    async def test_claude_code_with_edits(self, executor):
        """claude_code with allow_edits uses --dangerously-skip-permissions and su."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            # allow_edits=True triggers multiple SSH calls:
            # 1. mktemp (create temp dir)
            # 2. claude -p (actual work)
            # 3. find (list files in temp dir)
            # 4. cp (copy files to target) — only if find found files
            # 5. rm -rf (cleanup temp dir)
            mock_ssh.side_effect = [
                (0, "/tmp/claude_code_abc12345\n"),  # mktemp
                (0, "Fixed the bug"),                 # claude -p
                (0, ""),                              # find (no files)
                (0, ""),                              # rm cleanup
            ]
            result = await executor.execute("claude_code", {
                "host": "server",
                "working_directory": "/opt/project",
                "prompt": "Fix the bug",
                "allow_edits": True,
            })
        assert "Fixed the bug" in result
        # The claude -p call is the second SSH call (index 1)
        claude_call = mock_ssh.call_args_list[1]
        cmd = claude_call[1].get("command") or claude_call[0][1]
        assert "--dangerously-skip-permissions" in cmd
        assert "su - deploy" in cmd

    @pytest.mark.asyncio
    async def test_claude_code_failure(self, executor):
        """claude_code with non-zero exit reports the failure."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "Error: budget exceeded")
            result = await executor.execute("claude_code", {
                "host": "server",
                "working_directory": "/opt/project",
                "prompt": "Do stuff",
            })
        assert "failed" in result.lower()
        assert "budget" in result.lower()

    @pytest.mark.asyncio
    async def test_claude_code_output_truncation(self, executor):
        """claude_code truncates very long output at max_output_chars."""
        long_output = "x" * 10000
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor.execute("claude_code", {
                "host": "server",
                "working_directory": "/opt/project",
                "prompt": "Analyze",
                "max_output_chars": 200,
            })
        assert "[... truncated ...]" in result
        assert len(result) < len(long_output)

    @pytest.mark.asyncio
    async def test_claude_code_unknown_host(self, executor):
        """claude_code with unknown host returns error."""
        result = await executor.execute("claude_code", {
            "host": "nonexistent",
            "working_directory": "/tmp",
            "prompt": "hello",
        })
        assert "Unknown" in result or "disallowed" in result.lower()

    @pytest.mark.asyncio
    async def test_claude_code_allowed_tools(self, executor):
        """claude_code passes --allowedTools when specified."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "done")
            await executor.execute("claude_code", {
                "host": "server",
                "working_directory": "/opt/project",
                "prompt": "hello",
                "allowed_tools": "Read,Grep",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--allowedTools" in cmd


# ---------------------------------------------------------------------------
# run_command_multi — error branch
# ---------------------------------------------------------------------------

class TestRunCommandMultiErrors:
    @pytest.mark.asyncio
    async def test_run_command_multi_partial_failure(self, executor):
        """run_command_multi handles exceptions from individual hosts."""
        async def selective_ssh(**kwargs):
            if kwargs.get("host") == "10.0.0.1":
                raise ConnectionError("Connection refused")
            return (0, "uptime: up 5 days")

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.side_effect = selective_ssh
            result = await executor.execute("run_command_multi", {
                "hosts": ["server", "desktop"],
                "command": "uptime",
            })
        # Server should show error, desktop should show output
        assert "Error" in result or "error" in result
        assert "uptime" in result
