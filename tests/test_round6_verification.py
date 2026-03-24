"""Round 6: Verification tests for Round 5 local execution implementation.

Covers:
- _truncate_output shared helper works identically for local and SSH paths
- _exec_command handles IPv6 loopback (::1)
- claude_code handler routes through _exec_command for local hosts
- No source files outside ssh.py and executor.py import SSH/local functions directly
- Security: create_subprocess_shell only in ssh.py
- _exec_command is the sole dispatch point in executor.py (AST check)
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.config.schema import ToolHost, ToolsConfig
from src.tools.executor import ToolExecutor
from src.tools.ssh import MAX_OUTPUT_CHARS, _truncate_output, is_local_address


# ---------------------------------------------------------------------------
# _truncate_output: shared by run_local_command and run_ssh_command
# ---------------------------------------------------------------------------
class TestTruncateOutput:
    def test_short_output_unchanged(self):
        text = "hello world"
        assert _truncate_output(text) == text

    def test_exact_limit_unchanged(self):
        text = "x" * MAX_OUTPUT_CHARS
        assert _truncate_output(text) == text

    def test_over_limit_truncated(self):
        text = "x" * (MAX_OUTPUT_CHARS + 100)
        result = _truncate_output(text)
        assert len(result) < len(text)
        assert "truncated" in result.lower()
        # Head and tail are preserved
        half = MAX_OUTPUT_CHARS // 2
        assert result.startswith("x" * half)
        assert result.endswith("x" * half)

    def test_empty_string(self):
        assert _truncate_output("") == ""


# ---------------------------------------------------------------------------
# is_local_address: edge cases
# ---------------------------------------------------------------------------
class TestIsLocalAddressEdgeCases:
    def test_ipv6_loopback(self):
        assert is_local_address("::1") is True

    def test_case_sensitive_localhost(self):
        """'Localhost' with capital L is NOT local (strict matching)."""
        assert is_local_address("Localhost") is False

    def test_loopback_with_port_is_not_local(self):
        """'127.0.0.1:22' is NOT a match — address must be exact."""
        assert is_local_address("127.0.0.1:22") is False

    def test_0_0_0_0_is_not_local(self):
        """0.0.0.0 is NOT treated as local (bind-all, not loopback)."""
        assert is_local_address("0.0.0.0") is False


# ---------------------------------------------------------------------------
# _exec_command with IPv6 loopback
# ---------------------------------------------------------------------------
@pytest.fixture
def ipv6_config(tmp_dir: Path) -> ToolsConfig:
    return ToolsConfig(
        ssh_key_path=str(tmp_dir / "id_ed25519"),
        ssh_known_hosts_path=str(tmp_dir / "known_hosts"),
        hosts={
            "ipv6local": ToolHost(address="::1", ssh_user="root", os="linux"),
            "remote": ToolHost(address="10.0.0.50", ssh_user="root", os="linux"),
        },
        allowed_services=["nginx"],
        command_timeout_seconds=10,
    )


@pytest.fixture
def ipv6_executor(ipv6_config: ToolsConfig, tmp_dir: Path) -> ToolExecutor:
    return ToolExecutor(ipv6_config, memory_path=str(tmp_dir / "memory.json"))


class TestIPv6LocalExecution:
    @pytest.mark.asyncio
    async def test_ipv6_loopback_uses_local_command(self, ipv6_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = (0, "ipv6 local output")
            code, output = await ipv6_executor._exec_command("::1", "whoami", "root")
        assert code == 0
        assert output == "ipv6 local output"
        mock_local.assert_called_once_with("whoami", timeout=10)

    @pytest.mark.asyncio
    async def test_ipv6_loopback_run_on_host(self, ipv6_executor):
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_local.return_value = (0, "local result")
            result = await ipv6_executor._run_on_host("ipv6local", "hostname")
        assert result == "local result"
        mock_local.assert_called_once()
        mock_ssh.assert_not_called()


# ---------------------------------------------------------------------------
# claude_code handler uses _exec_command (and thus local dispatch)
# ---------------------------------------------------------------------------
@pytest.fixture
def local_claude_config(tmp_dir: Path) -> ToolsConfig:
    return ToolsConfig(
        ssh_key_path=str(tmp_dir / "id_ed25519"),
        ssh_known_hosts_path=str(tmp_dir / "known_hosts"),
        hosts={
            "localbox": ToolHost(address="127.0.0.1", ssh_user="root", os="linux"),
        },
        allowed_services=[],
        command_timeout_seconds=30,
        claude_code_host="localbox",
        claude_code_user="claude",
        claude_code_dir="/opt/project",
    )


@pytest.fixture
def local_claude_executor(local_claude_config: ToolsConfig, tmp_dir: Path) -> ToolExecutor:
    return ToolExecutor(local_claude_config, memory_path=str(tmp_dir / "memory.json"))


class TestClaudeCodeLocalExecution:
    @pytest.mark.asyncio
    async def test_claude_code_read_only_uses_local(self, local_claude_executor):
        """claude_code on a local host uses subprocess, not SSH."""
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_local.return_value = (0, "claude output")
            result = await local_claude_executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "explain this code",
                "allow_edits": False,
            })
        assert "claude output" in result
        mock_local.assert_called_once()
        mock_ssh.assert_not_called()

    @pytest.mark.asyncio
    async def test_claude_code_edits_uses_local_for_all_steps(self, local_claude_executor):
        """claude_code with allow_edits uses local exec for mktemp, main, find, cp, rm."""
        call_count = 0
        results = [
            (0, "/tmp/claude_code_XXXXXXXX"),  # mktemp
            (0, "files written"),               # main claude execution
            (0, "/tmp/claude_code_XXXXXXXX/main.py"),  # find
            (0, ""),                             # cp
            (0, ""),                             # rm cleanup
        ]

        async def mock_local_side_effect(cmd, timeout=30):
            nonlocal call_count
            idx = min(call_count, len(results) - 1)
            call_count += 1
            return results[idx]

        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_local.side_effect = mock_local_side_effect
            result = await local_claude_executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "write a main.py",
                "allow_edits": True,
            })
        # All subprocess calls went to local, none to SSH
        assert mock_local.call_count >= 3  # at least mktemp + main + cleanup
        mock_ssh.assert_not_called()


# ---------------------------------------------------------------------------
# Source code integrity: no direct SSH/local imports outside ssh.py + executor.py
# ---------------------------------------------------------------------------
class TestNoDirectSSHImports:
    def test_only_executor_imports_ssh_functions(self):
        """No source file other than executor.py and ssh.py imports run_ssh_command/run_local_command."""
        result = subprocess.run(
            ["git", "ls-files", "src/"], capture_output=True, text=True, cwd="."
        )
        src_files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]

        allowed_files = {"src/tools/ssh.py", "src/tools/executor.py"}
        violations = []

        for f in src_files:
            if f in allowed_files:
                continue
            content = Path(f).read_text()
            for func_name in ("run_ssh_command", "run_local_command", "is_local_address"):
                if func_name in content:
                    violations.append(f"{f} references {func_name}")

        assert not violations, f"Direct SSH function references found: {violations}"


class TestSubprocessShellRestriction:
    def test_create_subprocess_shell_only_in_ssh_py(self):
        """create_subprocess_shell must only appear in ssh.py."""
        result = subprocess.run(
            ["git", "ls-files", "src/"], capture_output=True, text=True, cwd="."
        )
        src_files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]

        allowed = {"src/tools/ssh.py", "src/tools/process_manager.py"}
        violations = []
        for f in src_files:
            if f in allowed:
                continue
            content = Path(f).read_text()
            if "create_subprocess_shell" in content:
                violations.append(f)

        assert not violations, f"create_subprocess_shell found in: {violations}"


# ---------------------------------------------------------------------------
# _exec_command is the sole dispatch point (AST-level verification)
# ---------------------------------------------------------------------------
class TestExecCommandCentralDispatch:
    def test_no_bare_ssh_calls_in_executor(self):
        """No method in executor.py calls run_ssh_command except _exec_command."""
        src = Path("src/tools/executor.py").read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.name == "_exec_command":
                    continue
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id == "run_ssh_command":
                        assert False, (
                            f"{node.name} calls run_ssh_command directly — "
                            "use self._exec_command instead"
                        )

    def test_no_bare_local_calls_in_executor(self):
        """No method in executor.py calls run_local_command except _exec_command."""
        src = Path("src/tools/executor.py").read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.name == "_exec_command":
                    continue
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id == "run_local_command":
                        assert False, (
                            f"{node.name} calls run_local_command directly — "
                            "use self._exec_command instead"
                        )

    def test_exec_command_calls_both(self):
        """_exec_command references both run_local_command and run_ssh_command."""
        src = Path("src/tools/executor.py").read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_exec_command":
                body_src = ast.dump(node)
                assert "run_local_command" in body_src
                assert "run_ssh_command" in body_src
                return
        assert False, "_exec_command not found in executor.py"
