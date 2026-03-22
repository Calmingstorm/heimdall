"""Tests for tools/executor.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor


@pytest.fixture
def executor(tools_config: ToolsConfig, tmp_dir: Path) -> ToolExecutor:
    return ToolExecutor(tools_config, memory_path=str(tmp_dir / "memory.json"))


class TestHostResolution:
    def test_known_host(self, executor: ToolExecutor):
        result = executor._resolve_host("server")
        assert result == ("10.0.0.1", "root", "linux")

    def test_unknown_host(self, executor: ToolExecutor):
        assert executor._resolve_host("nonexistent") is None

    def test_macos_host(self, executor: ToolExecutor):
        result = executor._resolve_host("macbook")
        assert result[2] == "macos"


class TestValidation:
    def test_valid_service(self, executor: ToolExecutor):
        assert executor._validate_service("apache2") is True

    def test_invalid_service(self, executor: ToolExecutor):
        assert executor._validate_service("evil_service") is False

    def test_valid_playbook(self, executor: ToolExecutor):
        assert executor._validate_playbook("check-services.yml") is True

    def test_invalid_playbook(self, executor: ToolExecutor):
        assert executor._validate_playbook("nuke-everything.yml") is False


class TestExecute:
    @pytest.mark.asyncio
    async def test_unknown_tool(self, executor: ToolExecutor):
        result = await executor.execute("nonexistent_tool", {})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_check_service_allowed(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "active (running)")
            result = await executor.execute("check_service", {
                "host": "server",
                "service": "apache2",
            })
        assert "active (running)" in result
        mock_ssh.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_service_disallowed(self, executor: ToolExecutor):
        result = await executor.execute("check_service", {
            "host": "server",
            "service": "evil_service",
        })
        assert "not in the allowlist" in result

    @pytest.mark.asyncio
    async def test_check_service_unknown_host(self, executor: ToolExecutor):
        result = await executor.execute("check_service", {
            "host": "nonexistent",
            "service": "apache2",
        })
        assert "Unknown or disallowed host" in result

    @pytest.mark.asyncio
    async def test_check_disk_linux(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Filesystem  Size  Used  Avail")
            result = await executor.execute("check_disk", {"host": "server"})
        assert "Filesystem" in result
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "tmpfs" in cmd  # excludes tmpfs on linux

    @pytest.mark.asyncio
    async def test_check_disk_macos(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Filesystem  Size  Used  Avail")
            await executor.execute("check_disk", {"host": "macbook"})
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert cmd == "df -h"

    @pytest.mark.asyncio
    async def test_check_memory_macos(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "--- Memory ---")
            await executor.execute("check_memory", {"host": "macbook"})
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "vm_stat" in cmd

    @pytest.mark.asyncio
    async def test_check_logs_line_limit(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "log output")
            await executor.execute("check_logs", {
                "host": "server",
                "service": "apache2",
                "lines": 999,
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "-n 50" in cmd  # capped at 50

    @pytest.mark.asyncio
    async def test_run_command(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "output")
            result = await executor.execute("run_command", {
                "host": "server",
                "command": "uptime",
            })
        assert result == "output"

    @pytest.mark.asyncio
    async def test_write_file(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("write_file", {
                "host": "server",
                "path": "/tmp/test.txt",
                "content": "hello world",
            })
        mock_ssh.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_file_uses_base64(self, executor: ToolExecutor):
        """Verify write_file uses base64 encoding (not heredoc) to prevent injection."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("write_file", {
                "host": "server",
                "path": "/tmp/test.txt",
                "content": "line1\nLOKI_EOF\nrm -rf /\nmore content",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        # Should use base64, not heredoc
        assert "base64" in cmd
        assert "LOKI_EOF" not in cmd

    @pytest.mark.asyncio
    async def test_read_file_default_lines(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "file contents")
            await executor.execute("read_file", {
                "host": "server",
                "path": "/etc/hostname",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "head -n 200" in cmd

    @pytest.mark.asyncio
    async def test_ansible_playbook_check_mode_default(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_ansible_playbook", {
                "playbook": "check-services.yml",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--check" in cmd

    @pytest.mark.asyncio
    async def test_ansible_playbook_disallowed(self, executor: ToolExecutor):
        result = await executor.execute("run_ansible_playbook", {
            "playbook": "nuke.yml",
        })
        assert "not in the allowlist" in result

    @pytest.mark.asyncio
    async def test_handler_exception(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.side_effect = RuntimeError("boom")
            result = await executor.execute("check_disk", {"host": "server"})
        assert "Error executing" in result

    @pytest.mark.asyncio
    async def test_query_prometheus(self, executor: ToolExecutor):
        raw = json.dumps({
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "up", "instance": "localhost:9090", "job": "prometheus"}, "value": [1645000000, "1"]},
                ],
            },
        })
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, raw)
            result = await executor.execute("query_prometheus", {
                "query": "up",
            })
        # Should be formatted, not raw JSON
        assert "1 result" in result
        assert "up{" in result
        assert '"status"' not in result


class TestDockerTools:
    @pytest.mark.asyncio
    async def test_docker_logs(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "log line 1\nlog line 2")
            result = await executor.execute("docker_logs", {
                "host": "server",
                "container": "myapp",
                "lines": 10,
            })
        assert "log line" in result
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--tail 10" in cmd

    @pytest.mark.asyncio
    async def test_docker_logs_with_since(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "recent logs")
            await executor.execute("docker_logs", {
                "host": "server",
                "container": "myapp",
                "since": "1h",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--since" in cmd

    @pytest.mark.asyncio
    async def test_docker_logs_line_cap(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "logs")
            await executor.execute("docker_logs", {
                "host": "server",
                "container": "test",
                "lines": 999,
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--tail 200" in cmd

    @pytest.mark.asyncio
    async def test_docker_compose_up(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Started")
            result = await executor.execute("docker_compose_action", {
                "host": "server",
                "project_dir": "/opt/project",
                "action": "up",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "up -d" in cmd

    @pytest.mark.asyncio
    async def test_docker_compose_invalid_action(self, executor: ToolExecutor):
        result = await executor.execute("docker_compose_action", {
            "host": "server",
            "project_dir": "/opt/test",
            "action": "destroy",
        })
        assert "Invalid action" in result

    @pytest.mark.asyncio
    async def test_docker_stats(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "NAME  CPU  MEM")
            result = await executor.execute("docker_stats", {"host": "server"})
        assert "CPU" in result


class TestGitTools:
    @pytest.mark.asyncio
    async def test_git_status(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "On branch master")
            result = await executor.execute("git_status", {
                "host": "desktop",
                "repo_path": "/root/project",
            })
        assert "On branch" in result

    @pytest.mark.asyncio
    async def test_git_log(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "abc1234 Initial commit")
            result = await executor.execute("git_log", {
                "host": "desktop",
                "repo_path": "/root/project",
                "count": 5,
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "-n 5" in cmd

    @pytest.mark.asyncio
    async def test_git_log_cap(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "commits")
            await executor.execute("git_log", {
                "host": "desktop",
                "repo_path": "/root/test",
                "count": 999,
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "-n 50" in cmd

    @pytest.mark.asyncio
    async def test_git_diff_working_dir(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "diff output")
            await executor.execute("git_diff", {
                "host": "desktop",
                "repo_path": "/root/project",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert cmd.endswith("diff")

    @pytest.mark.asyncio
    async def test_git_diff_commit(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "diff output")
            await executor.execute("git_diff", {
                "host": "desktop",
                "repo_path": "/root/project",
                "commit": "abc1234",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "abc1234" in cmd

    @pytest.mark.asyncio
    async def test_git_show(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "commit details")
            result = await executor.execute("git_show", {
                "host": "desktop",
                "repo_path": "/root/project",
                "commit": "HEAD",
            })
        assert "commit details" in result

    @pytest.mark.asyncio
    async def test_git_pull(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Already up to date.")
            result = await executor.execute("git_pull", {
                "host": "desktop",
                "repo_path": "/root/project",
            })
        assert "up to date" in result


class TestMultiHost:
    @pytest.mark.asyncio
    async def test_run_command_multi(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "up 5 days")
            result = await executor.execute("run_command_multi", {
                "hosts": ["server", "desktop"],
                "command": "uptime",
            })
        assert "server" in result
        assert "desktop" in result
        assert mock_ssh.call_count == 2

    @pytest.mark.asyncio
    async def test_run_command_multi_all(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            result = await executor.execute("run_command_multi", {
                "hosts": ["all"],
                "command": "hostname",
            })
        # Should expand to all 3 configured hosts
        assert mock_ssh.call_count == 3


class TestMemory:
    @pytest.mark.asyncio
    async def test_save_and_list(self, executor: ToolExecutor):
        result = await executor.execute("memory_manage", {
            "action": "save",
            "key": "test_key",
            "value": "test_value",
        })
        assert "Saved" in result

        result = await executor.execute("memory_manage", {"action": "list"})
        assert "test_key" in result
        assert "test_value" in result

    @pytest.mark.asyncio
    async def test_delete(self, executor: ToolExecutor):
        await executor.execute("memory_manage", {
            "action": "save", "key": "k", "value": "v",
        })
        result = await executor.execute("memory_manage", {
            "action": "delete", "key": "k",
        })
        assert "Deleted" in result

        result = await executor.execute("memory_manage", {"action": "list"})
        assert "No notes" in result

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, executor: ToolExecutor):
        result = await executor.execute("memory_manage", {
            "action": "delete", "key": "nonexistent",
        })
        assert "No note found" in result

    @pytest.mark.asyncio
    async def test_save_missing_fields(self, executor: ToolExecutor):
        result = await executor.execute("memory_manage", {
            "action": "save", "key": "k",
        })
        assert "required" in result.lower()
