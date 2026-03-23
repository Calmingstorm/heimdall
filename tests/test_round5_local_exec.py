"""Round 5: Tests for direct local execution (no SSH for localhost hosts)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolHost, ToolsConfig
from src.tools.executor import ToolExecutor
from src.tools.ssh import (
    MAX_OUTPUT_CHARS,
    is_local_address,
    run_local_command,
)


# ---------------------------------------------------------------------------
# Unit tests for is_local_address
# ---------------------------------------------------------------------------
class TestIsLocalAddress:
    def test_localhost(self):
        assert is_local_address("localhost") is True

    def test_ipv4_loopback(self):
        assert is_local_address("127.0.0.1") is True

    def test_ipv6_loopback(self):
        assert is_local_address("::1") is True

    def test_remote_ip(self):
        assert is_local_address("10.0.0.1") is False

    def test_remote_hostname(self):
        assert is_local_address("myserver.lan") is False

    def test_empty(self):
        assert is_local_address("") is False


# ---------------------------------------------------------------------------
# Unit tests for run_local_command
# ---------------------------------------------------------------------------
class TestRunLocalCommand:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello local", None))
        mock_proc.returncode = 0

        with patch("src.tools.ssh.asyncio.create_subprocess_shell", return_value=mock_proc):
            code, output = await run_local_command("echo hello")
        assert code == 0
        assert output == "hello local"

    @pytest.mark.asyncio
    async def test_nonzero_exit(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"err", None))
        mock_proc.returncode = 1

        with patch("src.tools.ssh.asyncio.create_subprocess_shell", return_value=mock_proc):
            code, output = await run_local_command("false")
        assert code == 1
        assert output == "err"

    @pytest.mark.asyncio
    async def test_timeout(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()

        with patch("src.tools.ssh.asyncio.create_subprocess_shell", return_value=mock_proc):
            code, output = await run_local_command("sleep 999", timeout=1)
        assert code == 1
        assert "timed out" in output.lower()
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        long_output = "x" * (MAX_OUTPUT_CHARS + 1000)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(long_output.encode(), None))
        mock_proc.returncode = 0

        with patch("src.tools.ssh.asyncio.create_subprocess_shell", return_value=mock_proc):
            code, output = await run_local_command("cat bigfile")
        assert code == 0
        assert len(output) < len(long_output)
        assert "truncated" in output.lower()

    @pytest.mark.asyncio
    async def test_exception(self):
        with patch(
            "src.tools.ssh.asyncio.create_subprocess_shell",
            side_effect=OSError("No such file"),
        ):
            code, output = await run_local_command("bad_cmd")
        assert code == 1
        assert "Local exec error" in output

    @pytest.mark.asyncio
    async def test_returncode_none_treated_as_zero(self):
        """Returncode None → 0 (same behavior as run_ssh_command)."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"ok", None))
        mock_proc.returncode = None

        with patch("src.tools.ssh.asyncio.create_subprocess_shell", return_value=mock_proc):
            code, output = await run_local_command("echo ok")
        assert code == 0


# ---------------------------------------------------------------------------
# Integration tests: ToolExecutor._exec_command dispatches correctly
# ---------------------------------------------------------------------------
@pytest.fixture
def local_tools_config(tmp_dir: Path) -> ToolsConfig:
    """Config with a localhost host and a remote host."""
    return ToolsConfig(
        ssh_key_path=str(tmp_dir / "id_ed25519"),
        ssh_known_hosts_path=str(tmp_dir / "known_hosts"),
        hosts={
            "local": ToolHost(address="127.0.0.1", ssh_user="root", os="linux"),
            "local2": ToolHost(address="localhost", ssh_user="admin", os="linux"),
            "remote": ToolHost(address="10.0.0.5", ssh_user="root", os="linux"),
        },
        allowed_services=["nginx"],
        command_timeout_seconds=5,
        prometheus_host="local",
        incus_host="local",
    )


@pytest.fixture
def local_executor(local_tools_config: ToolsConfig, tmp_dir: Path) -> ToolExecutor:
    return ToolExecutor(local_tools_config, memory_path=str(tmp_dir / "memory.json"))


class TestExecCommand:
    @pytest.mark.asyncio
    async def test_local_address_uses_run_local_command(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "local output")
            code, output = await local_executor._exec_command(
                "127.0.0.1", "echo hi", "root",
            )
        assert code == 0
        assert output == "local output"
        mock_local.assert_called_once_with("echo hi", timeout=5)

    @pytest.mark.asyncio
    async def test_remote_address_uses_run_ssh_command(self, local_executor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ssh output")
            code, output = await local_executor._exec_command(
                "10.0.0.5", "echo hi", "root",
            )
        assert code == 0
        assert output == "ssh output"
        mock_ssh.assert_called_once()

    @pytest.mark.asyncio
    async def test_localhost_string_uses_local(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "ok")
            await local_executor._exec_command("localhost", "ls", "admin")
        mock_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_timeout_forwarded(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "ok")
            await local_executor._exec_command(
                "127.0.0.1", "sleep 1", "root", timeout=120,
            )
        mock_local.assert_called_once_with("sleep 1", timeout=120)

    @pytest.mark.asyncio
    async def test_default_timeout_from_config(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "ok")
            await local_executor._exec_command("127.0.0.1", "ls", "root")
        # Config has command_timeout_seconds=5
        mock_local.assert_called_once_with("ls", timeout=5)


# ---------------------------------------------------------------------------
# Integration: _run_on_host uses local for localhost hosts
# ---------------------------------------------------------------------------
class TestRunOnHostLocal:
    @pytest.mark.asyncio
    async def test_localhost_host_uses_local(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "disk output")
            result = await local_executor._run_on_host("local", "df -h")
        assert result == "disk output"
        mock_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_remote_host_uses_ssh(self, local_executor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ssh result")
            result = await local_executor._run_on_host("remote", "df -h")
        assert result == "ssh result"
        mock_ssh.assert_called_once()

    @pytest.mark.asyncio
    async def test_localhost_no_ssh_key_needed(self, local_executor):
        """Local execution never calls run_ssh_command — no SSH key required."""
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_local.return_value = (0, "ok")
            await local_executor._run_on_host("local", "uptime")
        mock_local.assert_called_once()
        mock_ssh.assert_not_called()


# ---------------------------------------------------------------------------
# Tool handlers: verify local execution for localhost hosts
# ---------------------------------------------------------------------------
class TestToolHandlersLocal:
    @pytest.mark.asyncio
    async def test_run_command_local(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "hello")
            result = await local_executor.execute("run_command", {
                "host": "local", "command": "echo hello",
            })
        assert result == "hello"
        mock_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_disk_local(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "/dev/sda1  50G  30G  20G  60% /")
            result = await local_executor.execute("check_disk", {"host": "local"})
        assert "/dev/sda1" in result
        mock_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_memory_local(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "              total    used\nMem:    16G     8G")
            result = await local_executor.execute("check_memory", {"host": "local"})
        assert "16G" in result
        mock_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_script_local(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "script output")
            result = await local_executor.execute("run_script", {
                "host": "local",
                "script": "echo hello world",
                "interpreter": "bash",
            })
        assert result == "script output"
        mock_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_service_local(self, local_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "active (running)")
            result = await local_executor.execute("check_service", {
                "host": "local", "service": "nginx",
            })
        assert "active (running)" in result

    @pytest.mark.asyncio
    async def test_query_prometheus_local(self, local_executor):
        """Prometheus on a local host uses subprocess, not SSH."""
        prom_response = '{"status":"success","data":{"resultType":"scalar","result":[1234,"42"]}}'
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, prom_response)
            result = await local_executor.execute("query_prometheus", {
                "query": "up",
            })
        assert "42" in result
        mock_local.assert_called_once()


# ---------------------------------------------------------------------------
# Remote hosts still use SSH
# ---------------------------------------------------------------------------
class TestRemoteStillUsesSSH:
    @pytest.mark.asyncio
    async def test_run_command_remote(self, local_executor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "remote result")
            result = await local_executor.execute("run_command", {
                "host": "remote", "command": "uptime",
            })
        assert result == "remote result"
        mock_ssh.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_script_remote(self, local_executor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "remote script output")
            result = await local_executor.execute("run_script", {
                "host": "remote",
                "script": "echo test",
                "interpreter": "bash",
            })
        assert result == "remote script output"
        mock_ssh.assert_called_once()


# ---------------------------------------------------------------------------
# Verify _exec_command is used everywhere (no direct run_ssh_command calls)
# ---------------------------------------------------------------------------
class TestNoBareSSHCalls:
    def test_executor_only_uses_exec_command(self):
        """All run_ssh_command usage in executor.py is inside _exec_command."""
        import ast

        src = Path("src/tools/executor.py").read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                if node.name == "_exec_command":
                    continue  # _exec_command itself is allowed to call run_ssh_command
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute):
                        # No method should reference run_ssh_command except _exec_command
                        pass
                    if isinstance(child, ast.Name) and child.id == "run_ssh_command":
                        assert False, (
                            f"Function {node.name} calls run_ssh_command directly. "
                            f"Use self._exec_command instead."
                        )
