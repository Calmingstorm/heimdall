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
                "content": "line1\nHEIMDALL_EOF\nrm -rf /\nmore content",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        # Should use base64, not heredoc
        assert "base64" in cmd
        assert "HEIMDALL_EOF" not in cmd

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


class TestRunScript:
    """Tests for the run_script composite tool."""

    @pytest.mark.asyncio
    async def test_basic_bash_script(self, executor: ToolExecutor):
        """run_script writes to temp file, executes with bash, cleans up."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "hello world")
            result = await executor.execute("run_script", {
                "host": "server",
                "script": "#!/bin/bash\necho hello world",
            })
        assert result == "hello world"
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        # Must use base64 encoding to avoid heredoc issues
        assert "base64" in cmd
        # Must use mktemp for the temp file
        assert "mktemp" in cmd
        # Must execute with bash
        assert "bash" in cmd
        # Must clean up
        assert "rm -f" in cmd

    @pytest.mark.asyncio
    async def test_python_interpreter(self, executor: ToolExecutor):
        """run_script respects interpreter parameter."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "42")
            result = await executor.execute("run_script", {
                "host": "server",
                "script": "print(6 * 7)",
                "interpreter": "python3",
            })
        assert result == "42"
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "python3" in cmd

    @pytest.mark.asyncio
    async def test_script_failure_reports_exit_code(self, executor: ToolExecutor):
        """run_script reports exit code on failure."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "syntax error line 3")
            result = await executor.execute("run_script", {
                "host": "server",
                "script": "bad script content",
            })
        assert "Script failed (exit 1)" in result
        assert "syntax error" in result

    @pytest.mark.asyncio
    async def test_unknown_host(self, executor: ToolExecutor):
        """run_script rejects unknown hosts."""
        result = await executor.execute("run_script", {
            "host": "nonexistent",
            "script": "echo hi",
        })
        assert "Unknown or disallowed host" in result

    @pytest.mark.asyncio
    async def test_unsupported_interpreter(self, executor: ToolExecutor):
        """run_script rejects unsupported interpreters."""
        result = await executor.execute("run_script", {
            "host": "server",
            "script": "echo hi",
            "interpreter": "evil_binary",
        })
        assert "Unsupported interpreter" in result

    @pytest.mark.asyncio
    async def test_script_with_special_characters(self, executor: ToolExecutor):
        """run_script handles scripts with quotes, heredocs, and special chars via base64."""
        script = '''#!/bin/bash
echo "Hello $USER"
cat << 'EOF'
This has "quotes" and $variables and `backticks`
EOF
echo 'done'
'''
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Hello root\nThis has stuff\ndone")
            result = await executor.execute("run_script", {
                "host": "server",
                "script": script,
            })
        assert "done" in result
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        # The actual script content should NOT appear in the command (it's base64-encoded)
        assert "Hello $USER" not in cmd
        assert "base64" in cmd

    @pytest.mark.asyncio
    async def test_custom_filename(self, executor: ToolExecutor):
        """run_script uses custom filename in mktemp pattern."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_script", {
                "host": "server",
                "script": "echo ok",
                "filename": "deploy.sh",
            })
        cmd = mock_ssh.call_args[1]["command"] if mock_ssh.call_args[1] else mock_ssh.call_args[0][1]
        assert "deploy.sh" in cmd

    @pytest.mark.asyncio
    async def test_output_truncation(self, executor: ToolExecutor):
        """run_script truncates very long output."""
        long_output = "\n".join(f"line {i}" for i in range(300))
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor.execute("run_script", {
                "host": "server",
                "script": "generate lots of output",
            })
        assert "omitted" in result

    @pytest.mark.asyncio
    async def test_all_supported_interpreters(self, executor: ToolExecutor):
        """All documented interpreters are accepted."""
        for interp in ["bash", "sh", "python3", "python", "node", "ruby", "perl"]:
            with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
                mock_ssh.return_value = (0, "ok")
                result = await executor.execute("run_script", {
                    "host": "server",
                    "script": "echo ok",
                    "interpreter": interp,
                })
            assert result == "ok", f"Interpreter {interp} failed"




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
