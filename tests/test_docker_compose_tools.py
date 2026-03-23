"""Tests for Docker Compose status, logs, and build tools."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.config.schema import ToolsConfig
from src.tools.executor import ToolExecutor
from src.tools.registry import TOOLS


@pytest.fixture
def executor(tools_config: ToolsConfig, tmp_dir: Path) -> ToolExecutor:
    return ToolExecutor(tools_config, memory_path=str(tmp_dir / "memory.json"))


# --- Registry tests ---


class TestDockerComposeStatusRegistry:
    def test_tool_exists(self):
        names = [t["name"] for t in TOOLS]
        assert "docker_compose_status" in names

    def test_required_fields(self):
        tool = next(t for t in TOOLS if t["name"] == "docker_compose_status")
        assert tool["input_schema"]["required"] == ["host", "project_dir"]

    def test_has_host_property(self):
        tool = next(t for t in TOOLS if t["name"] == "docker_compose_status")
        assert "host" in tool["input_schema"]["properties"]

    def test_has_project_dir_property(self):
        tool = next(t for t in TOOLS if t["name"] == "docker_compose_status")
        assert "project_dir" in tool["input_schema"]["properties"]


class TestDockerComposeLogsRegistry:
    def test_tool_exists(self):
        names = [t["name"] for t in TOOLS]
        assert "docker_compose_logs" in names

    def test_required_fields(self):
        tool = next(t for t in TOOLS if t["name"] == "docker_compose_logs")
        assert tool["input_schema"]["required"] == ["host", "project_dir"]

    def test_has_service_property(self):
        tool = next(t for t in TOOLS if t["name"] == "docker_compose_logs")
        assert "service" in tool["input_schema"]["properties"]

    def test_has_lines_property(self):
        tool = next(t for t in TOOLS if t["name"] == "docker_compose_logs")
        assert "lines" in tool["input_schema"]["properties"]

    def test_has_since_property(self):
        tool = next(t for t in TOOLS if t["name"] == "docker_compose_logs")
        assert "since" in tool["input_schema"]["properties"]


class TestDockerComposeActionBuild:
    def test_build_in_enum(self):
        tool = next(t for t in TOOLS if t["name"] == "docker_compose_action")
        assert "build" in tool["input_schema"]["properties"]["action"]["enum"]

    def test_description_mentions_build(self):
        tool = next(t for t in TOOLS if t["name"] == "docker_compose_action")
        assert "build" in tool["description"]


# --- Executor tests: docker_compose_status ---


class TestDockerComposeStatusExecutor:
    @pytest.mark.asyncio
    async def test_runs_compose_ps(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "NAME  STATUS  PORTS\nmyapp  running  8080->8080")
            result = await executor.execute("docker_compose_status", {
                "host": "server",
                "project_dir": "/opt/project",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "docker compose ps -a" in cmd
        assert "myapp" in result

    @pytest.mark.asyncio
    async def test_uses_project_dir(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "output")
            await executor.execute("docker_compose_status", {
                "host": "server",
                "project_dir": "/opt/myproject",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "/opt/myproject" in cmd

    @pytest.mark.asyncio
    async def test_shell_escapes_project_dir(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "output")
            await executor.execute("docker_compose_status", {
                "host": "server",
                "project_dir": "/opt/my project",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "'/opt/my project'" in cmd

    @pytest.mark.asyncio
    async def test_returns_ssh_output(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "NAME  IMAGE  STATUS\nweb  nginx  Up 2 hours")
            result = await executor.execute("docker_compose_status", {
                "host": "server",
                "project_dir": "/opt/test",
            })
        assert "web" in result
        assert "Up 2 hours" in result


# --- Executor tests: docker_compose_logs ---


class TestDockerComposeLogsExecutor:
    @pytest.mark.asyncio
    async def test_default_tail_50(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "log output")
            await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/project",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--tail 50" in cmd

    @pytest.mark.asyncio
    async def test_custom_lines(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "logs")
            await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/project",
                "lines": 100,
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--tail 100" in cmd

    @pytest.mark.asyncio
    async def test_line_cap_200(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "logs")
            await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/project",
                "lines": 999,
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--tail 200" in cmd

    @pytest.mark.asyncio
    async def test_with_service_filter(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "web logs")
            result = await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/project",
                "service": "web",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "web" in cmd
        assert "web logs" in result

    @pytest.mark.asyncio
    async def test_with_since(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "recent logs")
            await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/project",
                "since": "1h",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--since" in cmd

    @pytest.mark.asyncio
    async def test_shell_escapes_service(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "logs")
            await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/test",
                "service": "my service",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "'my service'" in cmd

    @pytest.mark.asyncio
    async def test_shell_escapes_since(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "logs")
            await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/test",
                "since": "2h",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--since '2h'" in cmd or "--since 2h" in cmd

    @pytest.mark.asyncio
    async def test_all_options_combined(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "filtered logs")
            result = await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/project",
                "service": "bot",
                "lines": 25,
                "since": "30m",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "--tail 25" in cmd
        assert "--since" in cmd
        assert "bot" in cmd
        assert "filtered logs" in result

    @pytest.mark.asyncio
    async def test_no_service_omits_filter(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "all logs")
            await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/test",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        # Command should end with "2>&1", no service name appended
        assert cmd.strip().endswith("2>&1")

    @pytest.mark.asyncio
    async def test_stderr_captured(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "error output on stderr")
            await executor.execute("docker_compose_logs", {
                "host": "server",
                "project_dir": "/opt/test",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "2>&1" in cmd


# --- Executor tests: docker_compose_action build ---


class TestDockerComposeActionBuildExecutor:
    @pytest.mark.asyncio
    async def test_build_action(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Building web...")
            result = await executor.execute("docker_compose_action", {
                "host": "server",
                "project_dir": "/opt/project",
                "action": "build",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "docker compose" in cmd
        assert "build" in cmd
        assert "Building" in result

    @pytest.mark.asyncio
    async def test_build_uses_project_dir(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "done")
            await executor.execute("docker_compose_action", {
                "host": "server",
                "project_dir": "/opt/myapp",
                "action": "build",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "/opt/myapp" in cmd

    @pytest.mark.asyncio
    async def test_invalid_action_still_rejected(self, executor: ToolExecutor):
        result = await executor.execute("docker_compose_action", {
            "host": "server",
            "project_dir": "/opt/test",
            "action": "destroy",
        })
        assert "Invalid action" in result
