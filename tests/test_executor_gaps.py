"""Tests for executor.py — covering untested handlers: Incus, browser, claude_code,
prometheus_range, docker gaps, and restart_service."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolsConfig
from src.tools.executor import ToolExecutor


@pytest.fixture
def executor(tools_config: ToolsConfig, tmp_dir: Path) -> ToolExecutor:
    return ToolExecutor(tools_config, memory_path=str(tmp_dir / "memory.json"))


# ---------------------------------------------------------------------------
# check_docker — container-specific branch (line 209-215)
# ---------------------------------------------------------------------------

class TestCheckDocker:
    @pytest.mark.asyncio
    async def test_check_docker_all(self, executor):
        """check_docker without container shows all containers."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "NAME  STATUS  PORTS\nansiblex  Up 2 days")
            result = await executor.execute("check_docker", {"host": "server"})
        assert "ansiblex" in result
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        # No --filter when no container specified
        assert "--filter" not in cmd

    @pytest.mark.asyncio
    async def test_check_docker_specific_container(self, executor):
        """check_docker with container uses --filter name=..."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ansiblex  Up 2 days  8080->8080")
            result = await executor.execute("check_docker", {
                "host": "server",
                "container": "ansiblex",
            })
        assert "ansiblex" in result
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--filter name=" in cmd
        # Uses -a for all (including stopped)
        assert "-a" in cmd


# ---------------------------------------------------------------------------
# restart_service (lines 262-267)
# ---------------------------------------------------------------------------

class TestRestartService:
    @pytest.mark.asyncio
    async def test_restart_allowed_service(self, executor):
        """restart_service on allowed service runs systemctl restart + status."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "active (running)")
            result = await executor.execute("restart_service", {
                "host": "server",
                "service": "apache2",
            })
        assert "active" in result
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "systemctl restart" in cmd
        assert "systemctl status" in cmd

    @pytest.mark.asyncio
    async def test_restart_disallowed_service(self, executor):
        """restart_service on disallowed service is rejected."""
        result = await executor.execute("restart_service", {
            "host": "server",
            "service": "sshd",
        })
        assert "not in the allowlist" in result


# ---------------------------------------------------------------------------
# query_prometheus_range (lines 491-520)
# ---------------------------------------------------------------------------

class TestQueryPrometheusRange:
    @pytest.mark.asyncio
    async def test_range_query_success(self, executor):
        """prometheus_range query returns formatted output."""
        raw = json.dumps({
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {"__name__": "up", "job": "prometheus"},
                    "values": [[1645000000, "1"], [1645000300, "1"]],
                }],
            },
        })
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, raw)
            result = await executor.execute("query_prometheus_range", {
                "query": "up",
                "duration": "1h",
                "step": "5m",
            })
        assert "1 result" in result or "up" in result.lower()
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "query_range" in cmd

    @pytest.mark.asyncio
    async def test_range_query_failure(self, executor):
        """prometheus_range query with non-zero exit returns error."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "connection refused")
            result = await executor.execute("query_prometheus_range", {
                "query": "up",
            })
        assert "failed" in result.lower()


# ---------------------------------------------------------------------------
# Browser handlers — disabled path (lines 525-563)
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
# claude_code (lines 567-625)
# ---------------------------------------------------------------------------

class TestClaudeCode:
    @pytest.mark.asyncio
    async def test_claude_code_readonly(self, executor):
        """claude_code in read-only mode (default) runs without --dangerously-skip-permissions."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Analysis complete: all good")
            result = await executor.execute("claude_code", {
                "host": "server",
                "working_directory": "/opt/ansiblex",
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
                "working_directory": "/opt/ansiblex",
                "prompt": "Fix the bug",
                "allow_edits": True,
            })
        assert "Fixed the bug" in result
        # The claude -p call is the second SSH call (index 1)
        claude_call = mock_ssh.call_args_list[1]
        cmd = claude_call[1].get("command") or claude_call[0][1]
        assert "--dangerously-skip-permissions" in cmd
        assert "su - calmingstorm" in cmd

    @pytest.mark.asyncio
    async def test_claude_code_failure(self, executor):
        """claude_code with non-zero exit reports the failure."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "Error: budget exceeded")
            result = await executor.execute("claude_code", {
                "host": "server",
                "working_directory": "/opt/ansiblex",
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
                "working_directory": "/opt/ansiblex",
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
                "working_directory": "/opt/ansiblex",
                "prompt": "hello",
                "allowed_tools": "Read,Grep",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--allowedTools" in cmd


# ---------------------------------------------------------------------------
# Incus handlers (lines 629-771)
# ---------------------------------------------------------------------------

class TestIncusTools:
    @pytest.mark.asyncio
    async def test_incus_list_success(self, executor):
        """incus_list parses CSV output into formatted table."""
        csv_output = "myvm,RUNNING,CONTAINER,10.0.0.1\nweb,STOPPED,VIRTUAL-MACHINE,10.0.0.2"
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, csv_output)
            result = await executor.execute("incus_list", {})
        assert "myvm" in result
        assert "RUNNING" in result
        assert "web" in result

    @pytest.mark.asyncio
    async def test_incus_list_empty(self, executor):
        """incus_list with no instances returns message."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_list", {})
        assert "No Incus instances" in result

    @pytest.mark.asyncio
    async def test_incus_list_three_column_row(self, executor):
        """incus_list handles rows with only 3 columns (no IPv4)."""
        csv_output = "myvm,RUNNING,CONTAINER"
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, csv_output)
            result = await executor.execute("incus_list", {})
        assert "myvm" in result

    @pytest.mark.asyncio
    async def test_incus_list_command_failed(self, executor):
        """incus_list returns error when SSH command fails."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "Command failed (exit 1):\nPermission denied")
            result = await executor.execute("incus_list", {})
        assert "Command failed" in result

    @pytest.mark.asyncio
    async def test_incus_info(self, executor):
        """incus_info runs 'incus info <instance>'."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Name: myvm\nStatus: RUNNING")
            result = await executor.execute("incus_info", {"instance": "myvm"})
        assert "myvm" in result

    @pytest.mark.asyncio
    async def test_incus_exec(self, executor):
        """incus_exec runs a command inside an instance."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "root")
            result = await executor.execute("incus_exec", {
                "instance": "myvm",
                "command": "whoami",
            })
        assert "root" in result
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "incus exec" in cmd

    @pytest.mark.asyncio
    async def test_incus_exec_with_user(self, executor):
        """incus_exec with user flag passes --user."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ubuntu")
            await executor.execute("incus_exec", {
                "instance": "myvm",
                "command": "whoami",
                "user": "ubuntu",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--user" in cmd

    @pytest.mark.asyncio
    async def test_incus_start(self, executor):
        """incus_start starts an instance and returns success message."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_start", {"instance": "myvm"})
        assert "started" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_start_failure(self, executor):
        """incus_start returns error on failure."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "Command failed (exit 1):\nInstance not found")
            result = await executor.execute("incus_start", {"instance": "nonexistent"})
        assert "Command failed" in result

    @pytest.mark.asyncio
    async def test_incus_stop(self, executor):
        """incus_stop stops an instance."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_stop", {"instance": "myvm"})
        assert "stopped" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_stop_force(self, executor):
        """incus_stop with force=True passes --force flag."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("incus_stop", {
                "instance": "myvm",
                "force": True,
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--force" in cmd

    @pytest.mark.asyncio
    async def test_incus_restart(self, executor):
        """incus_restart restarts an instance."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_restart", {"instance": "myvm"})
        assert "restarted" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_restart_force(self, executor):
        """incus_restart with force passes --force."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("incus_restart", {
                "instance": "myvm",
                "force": True,
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--force" in cmd

    @pytest.mark.asyncio
    async def test_incus_restart_failure(self, executor):
        """incus_restart returns error message on failure."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "Command failed (exit 1):\ntimeout")
            result = await executor.execute("incus_restart", {"instance": "myvm"})
        assert "Command failed" in result

    @pytest.mark.asyncio
    async def test_incus_snapshot_list(self, executor):
        """incus_snapshot_list shows snapshots."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "snap0\nsnap1")
            result = await executor.execute("incus_snapshot_list", {"instance": "myvm"})
        assert "snap0" in result

    @pytest.mark.asyncio
    async def test_incus_snapshot_list_none(self, executor):
        """incus_snapshot_list returns message when no snapshots."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "No snapshots found")
            result = await executor.execute("incus_snapshot_list", {"instance": "myvm"})
        assert "No snapshots" in result

    @pytest.mark.asyncio
    async def test_incus_snapshot_list_empty(self, executor):
        """incus_snapshot_list returns message for empty output."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_snapshot_list", {"instance": "myvm"})
        assert "No snapshots" in result

    @pytest.mark.asyncio
    async def test_incus_snapshot_create(self, executor):
        """incus_snapshot create makes a named snapshot."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_snapshot", {
                "instance": "myvm",
                "action": "create",
                "snapshot": "before-update",
            })
        assert "created" in result.lower()
        assert "before-update" in result

    @pytest.mark.asyncio
    async def test_incus_snapshot_create_auto(self, executor):
        """incus_snapshot create without name uses auto."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_snapshot", {
                "instance": "myvm",
                "action": "create",
            })
        assert "auto" in result.lower() or "created" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_snapshot_create_failure(self, executor):
        """incus_snapshot create returns error on failure."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "Command failed (exit 1):\nno space")
            result = await executor.execute("incus_snapshot", {
                "instance": "myvm",
                "action": "create",
            })
        assert "Command failed" in result

    @pytest.mark.asyncio
    async def test_incus_snapshot_restore(self, executor):
        """incus_snapshot restore restores a snapshot."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_snapshot", {
                "instance": "myvm",
                "action": "restore",
                "snapshot": "before-update",
            })
        assert "restored" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_snapshot_restore_no_name(self, executor):
        """incus_snapshot restore without snapshot name returns error."""
        result = await executor.execute("incus_snapshot", {
            "instance": "myvm",
            "action": "restore",
        })
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_snapshot_delete(self, executor):
        """incus_snapshot delete removes a snapshot."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_snapshot", {
                "instance": "myvm",
                "action": "delete",
                "snapshot": "old-snap",
            })
        assert "deleted" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_snapshot_delete_no_name(self, executor):
        """incus_snapshot delete without snapshot name returns error."""
        result = await executor.execute("incus_snapshot", {
            "instance": "myvm",
            "action": "delete",
        })
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_snapshot_unknown_action(self, executor):
        """incus_snapshot with unknown action returns error."""
        result = await executor.execute("incus_snapshot", {
            "instance": "myvm",
            "action": "clone",
            "snapshot": "snap1",
        })
        assert "Unknown snapshot action" in result

    @pytest.mark.asyncio
    async def test_incus_launch_container(self, executor):
        """incus_launch creates a container by default."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_launch", {
                "image": "ubuntu:22.04",
                "name": "test-ct",
            })
        assert "launched" in result.lower()
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--vm" not in cmd

    @pytest.mark.asyncio
    async def test_incus_launch_vm(self, executor):
        """incus_launch with type=vm passes --vm."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("incus_launch", {
                "image": "ubuntu:22.04",
                "name": "test-vm",
                "type": "vm",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--vm" in cmd

    @pytest.mark.asyncio
    async def test_incus_launch_with_profile(self, executor):
        """incus_launch with profile passes --profile."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("incus_launch", {
                "image": "ubuntu:22.04",
                "name": "test-ct",
                "profile": "default",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--profile" in cmd

    @pytest.mark.asyncio
    async def test_incus_launch_failure(self, executor):
        """incus_launch returns error on SSH failure."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "Image not found")
            result = await executor.execute("incus_launch", {
                "image": "nonexistent:image",
                "name": "test-ct",
            })
        assert "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_delete(self, executor):
        """incus_delete removes an instance."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            result = await executor.execute("incus_delete", {"instance": "myvm"})
        assert "deleted" in result.lower()

    @pytest.mark.asyncio
    async def test_incus_delete_force(self, executor):
        """incus_delete with force passes --force."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("incus_delete", {
                "instance": "myvm",
                "force": True,
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--force" in cmd

    @pytest.mark.asyncio
    async def test_incus_logs(self, executor):
        """incus_logs retrieves console logs."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Boot log line 1\nBoot log line 2")
            result = await executor.execute("incus_logs", {
                "instance": "myvm",
                "lines": 10,
            })
        assert "Boot log" in result
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "tail -n 10" in cmd

    @pytest.mark.asyncio
    async def test_incus_logs_caps_lines(self, executor):
        """incus_logs caps lines at 200."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "logs")
            await executor.execute("incus_logs", {
                "instance": "myvm",
                "lines": 999,
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "tail -n 200" in cmd


# ---------------------------------------------------------------------------
# docker_compose_action — remaining branches (lines 352-356)
# ---------------------------------------------------------------------------

class TestDockerComposeActionBranches:
    @pytest.mark.asyncio
    async def test_docker_compose_down(self, executor):
        """docker_compose_action 'down' runs docker compose down."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Stopped")
            await executor.execute("docker_compose_action", {
                "host": "server",
                "project_dir": "/opt/ansiblex",
                "action": "down",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "down" in cmd

    @pytest.mark.asyncio
    async def test_docker_compose_pull(self, executor):
        """docker_compose_action 'pull' runs docker compose pull."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Pulling")
            await executor.execute("docker_compose_action", {
                "host": "server",
                "project_dir": "/opt/ansiblex",
                "action": "pull",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "pull" in cmd

    @pytest.mark.asyncio
    async def test_docker_compose_restart(self, executor):
        """docker_compose_action 'restart' runs docker compose restart."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Restarted")
            await executor.execute("docker_compose_action", {
                "host": "server",
                "project_dir": "/opt/ansiblex",
                "action": "restart",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "restart" in cmd

    @pytest.mark.asyncio
    async def test_docker_compose_build(self, executor):
        """docker_compose_action 'build' runs docker compose build."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Built")
            await executor.execute("docker_compose_action", {
                "host": "server",
                "project_dir": "/opt/ansiblex",
                "action": "build",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "build" in cmd


# ---------------------------------------------------------------------------
# docker_stats with container (lines 382-383)
# ---------------------------------------------------------------------------

class TestDockerStatsContainer:
    @pytest.mark.asyncio
    async def test_docker_stats_specific_container(self, executor):
        """docker_stats with container filters to that container."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ansiblex  5%  100MiB / 1GiB")
            result = await executor.execute("docker_stats", {
                "host": "server",
                "container": "ansiblex",
            })
        assert "ansiblex" in result
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "ansiblex" in cmd


# ---------------------------------------------------------------------------
# run_command_multi — error branch (line 483)
# ---------------------------------------------------------------------------

class TestRunCommandMultiErrors:
    @pytest.mark.asyncio
    async def test_run_command_multi_partial_failure(self, executor):
        """run_command_multi handles exceptions from individual hosts."""
        async def selective_ssh(**kwargs):
            if kwargs.get("host") == "192.168.1.13":
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


# ---------------------------------------------------------------------------
# Ansible playbook — with options (lines 284-306)
# ---------------------------------------------------------------------------

class TestAnsiblePlaybookOptions:
    @pytest.mark.asyncio
    async def test_ansible_with_limit_and_tags(self, executor):
        """ansible_playbook passes --limit and --tags when provided."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_ansible_playbook", {
                "playbook": "check-services.yml",
                "limit": "server",
                "tags": "packages",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--limit" in cmd
        assert "--tags" in cmd

    @pytest.mark.asyncio
    async def test_ansible_no_check_mode(self, executor):
        """ansible_playbook without check_mode omits --check."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_ansible_playbook", {
                "playbook": "check-services.yml",
                "check_mode": False,
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "--check" not in cmd

    @pytest.mark.asyncio
    async def test_ansible_failure(self, executor):
        """ansible_playbook reports failure on non-zero exit."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (2, "FAILED! => {}")
            result = await executor.execute("run_ansible_playbook", {
                "playbook": "check-services.yml",
            })
        assert "failed" in result.lower()


# ---------------------------------------------------------------------------
# Prometheus query failure (line 259)
# ---------------------------------------------------------------------------

class TestQueryPrometheusFailure:
    @pytest.mark.asyncio
    async def test_prometheus_query_ssh_failure(self, executor):
        """query_prometheus with non-zero exit code returns error."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "curl: connection refused")
            result = await executor.execute("query_prometheus", {
                "query": "up",
            })
        assert "failed" in result.lower()
