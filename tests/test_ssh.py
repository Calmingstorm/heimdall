"""Tests for tools/ssh.py."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.ssh import run_ssh_command, MAX_OUTPUT_CHARS


class TestRunSSHCommand:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello world", None))
        mock_proc.returncode = 0

        with patch("src.tools.ssh.asyncio.create_subprocess_exec", return_value=mock_proc):
            code, output = await run_ssh_command(
                host="1.2.3.4",
                command="echo hello",
                ssh_key_path="/tmp/key",
                known_hosts_path="/tmp/hosts",
            )
        assert code == 0
        assert output == "hello world"

    @pytest.mark.asyncio
    async def test_nonzero_exit(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"error msg", None))
        mock_proc.returncode = 1

        with patch("src.tools.ssh.asyncio.create_subprocess_exec", return_value=mock_proc):
            code, output = await run_ssh_command(
                host="1.2.3.4",
                command="false",
                ssh_key_path="/tmp/key",
                known_hosts_path="/tmp/hosts",
            )
        assert code == 1
        assert output == "error msg"

    @pytest.mark.asyncio
    async def test_timeout(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()

        with patch("src.tools.ssh.asyncio.create_subprocess_exec", return_value=mock_proc):
            code, output = await run_ssh_command(
                host="1.2.3.4",
                command="sleep 999",
                ssh_key_path="/tmp/key",
                known_hosts_path="/tmp/hosts",
                timeout=1,
            )
        assert code == 1
        assert "timed out" in output.lower()
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        long_output = "x" * (MAX_OUTPUT_CHARS + 1000)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(long_output.encode(), None))
        mock_proc.returncode = 0

        with patch("src.tools.ssh.asyncio.create_subprocess_exec", return_value=mock_proc):
            code, output = await run_ssh_command(
                host="1.2.3.4",
                command="cat bigfile",
                ssh_key_path="/tmp/key",
                known_hosts_path="/tmp/hosts",
            )
        assert code == 0
        assert len(output) < len(long_output)
        assert "truncated" in output.lower()

    @pytest.mark.asyncio
    async def test_connection_error(self):
        with patch(
            "src.tools.ssh.asyncio.create_subprocess_exec",
            side_effect=OSError("Connection refused"),
        ):
            code, output = await run_ssh_command(
                host="1.2.3.4",
                command="echo test",
                ssh_key_path="/tmp/key",
                known_hosts_path="/tmp/hosts",
            )
        assert code == 1
        assert "SSH error" in output
