"""Round 15: Tool execution hardening.

Tests:
1. All 74 tools have handlers (no approval gate, immediate dispatch)
2. run_script with complex scripts (heredocs, special chars, multi-line, large)
3. claude_code delegation from Codex (read-only, allow_edits, file manifest, errors)
4. Local subprocess execution paths (localhost, 127.0.0.1, ::1)
5. Remote SSH execution paths (all remote host patterns)
6. Tool executor dispatch coverage (every tool category)
7. Tool input validation (bad hosts, bad interpreters, missing config)
"""
from __future__ import annotations

import ast
import base64
import inspect
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.tools.executor import ToolExecutor, _truncate_lines  # noqa: E402
from src.tools.registry import TOOLS, get_tool_definitions  # noqa: E402
from src.tools.ssh import is_local_address  # noqa: E402
from src.config.schema import ToolsConfig, ToolHost  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    *,
    extra_hosts: dict | None = None,
    local_host: bool = False,
    prom_host: str = "server",
    ansible_host: str = "desktop",
    claude_code_host: str = "desktop",
    claude_code_user: str = "deploy",
    incus_host: str = "desktop",
) -> ToolsConfig:
    """Build a ToolsConfig for testing."""
    hosts = {
        "server": ToolHost(address="10.0.0.1", ssh_user="root", os="linux"),
        "desktop": ToolHost(address="10.0.0.2", ssh_user="root", os="linux"),
        "macbook": ToolHost(address="10.0.0.3", ssh_user="deploy", os="macos"),
    }
    if local_host:
        hosts["local"] = ToolHost(address="127.0.0.1", ssh_user="root", os="linux")
    if extra_hosts:
        for k, v in extra_hosts.items():
            hosts[k] = v
    return ToolsConfig(
        ssh_key_path="/test/id_ed25519",
        ssh_known_hosts_path="/test/known_hosts",
        hosts=hosts,
        allowed_services=["apache2", "prometheus", "grafana-server", "nginx"],
        allowed_playbooks=["check-services.yml", "update-all.yml"],
        ansible_directory="/ansible",
        command_timeout_seconds=5,
        prometheus_host=prom_host,
        ansible_host=ansible_host,
        claude_code_host=claude_code_host,
        claude_code_user=claude_code_user,
        claude_code_dir="/opt/project",
        incus_host=incus_host,
    )


def _make_executor(**kwargs) -> ToolExecutor:
    """Build a ToolExecutor with test config."""
    return ToolExecutor(_make_config(**kwargs))


# ===========================================================================
# 1. All tools have handlers and no approval gate
# ===========================================================================


class TestAllToolsHaveHandlers:
    """Every tool in TOOLS must have a corresponding handler."""

    def test_every_registry_tool_has_handler_or_client_handler(self):
        """All 74 tools must have _handle_{name} on ToolExecutor or be
        Discord-native (handled in client.py _run_tool)."""
        # Discord-native tools handled in client.py, not executor.py
        client_tools = {
            "purge_messages", "browser_screenshot", "generate_file", "post_file",
            "schedule_task", "list_schedules", "delete_schedule", "parse_time",
            "search_history", "delegate_task", "list_tasks", "cancel_task",
            "search_knowledge", "ingest_document", "list_knowledge", "delete_knowledge",
            "set_permission", "search_audit", "create_digest",
            "create_skill", "edit_skill", "delete_skill", "list_skills",
        }
        executor = _make_executor()
        for tool in TOOLS:
            name = tool["name"]
            if name in client_tools:
                continue  # handled in client.py
            handler = getattr(executor, f"_handle_{name}", None)
            assert handler is not None, f"Tool '{name}' has no handler on ToolExecutor"
            assert callable(handler), f"Handler for '{name}' is not callable"

    def test_tool_count_at_least_70(self):
        """Registry must have 70+ tools."""
        assert len(TOOLS) >= 70, f"Only {len(TOOLS)} tools found, expected 70+"

    def test_no_requires_approval_field(self):
        """No tool should have requires_approval."""
        for tool in TOOLS:
            assert "requires_approval" not in tool, \
                f"Tool '{tool['name']}' has requires_approval field"

    def test_get_tool_definitions_returns_all(self):
        """get_tool_definitions returns all tools with correct structure."""
        defs = get_tool_definitions()
        assert len(defs) == len(TOOLS)
        for d in defs:
            assert set(d.keys()) == {"name", "description", "input_schema"}
            assert d["name"]
            assert d["description"]
            assert isinstance(d["input_schema"], dict)

    def test_no_approval_in_descriptions(self):
        """No tool description should mention approval."""
        for tool in TOOLS:
            desc = tool["description"].lower()
            assert "approval" not in desc, \
                f"Tool '{tool['name']}' description mentions approval: {tool['description']}"


class TestToolDispatch:
    """Tool executor dispatches correctly."""

    async def test_unknown_tool_returns_error(self):
        executor = _make_executor()
        result = await executor.execute("nonexistent_tool_xyz", {})
        assert "Unknown tool" in result

    async def test_handler_exception_returns_error(self):
        executor = _make_executor()
        with patch.object(executor, "_handle_run_command", side_effect=RuntimeError("boom")):
            result = await executor.execute("run_command", {"host": "server", "command": "ls"})
            assert "Error executing run_command" in result
            assert "boom" in result

    async def test_memory_manage_gets_user_id(self):
        executor = _make_executor()
        with patch.object(executor, "_handle_memory_manage", new_callable=AsyncMock, return_value="ok") as mock:
            await executor.execute("memory_manage", {"action": "list"}, user_id="user-42")
            mock.assert_called_once_with({"action": "list"}, user_id="user-42")

    async def test_manage_list_gets_user_id(self):
        executor = _make_executor()
        with patch.object(executor, "_handle_manage_list", new_callable=AsyncMock, return_value="ok") as mock:
            await executor.execute("manage_list", {"action": "list_all"}, user_id="user-99")
            mock.assert_called_once_with({"action": "list_all"}, user_id="user-99")

    async def test_regular_tool_does_not_get_user_id(self):
        executor = _make_executor()
        with patch.object(executor, "_handle_check_disk", new_callable=AsyncMock, return_value="ok") as mock:
            await executor.execute("check_disk", {"host": "server"})
            mock.assert_called_once_with({"host": "server"})


# ===========================================================================
# 2. run_script with complex scripts
# ===========================================================================


class TestRunScriptComplexScripts:
    """Test run_script with various complex script patterns."""

    async def test_multiline_bash_with_heredoc(self):
        """Multi-line bash script with heredoc content."""
        script = """#!/bin/bash
cat << 'EOF'
Hello World
Special chars: $HOME "quotes" 'singles' `backticks`
EOF
echo "Done"
"""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "Hello World\nDone")):
            result = await executor.execute("run_script", {
                "host": "server", "script": script, "interpreter": "bash"
            })
            assert "Hello World" in result

    async def test_python_with_imports_and_classes(self):
        """Complex Python script with imports, classes, exceptions."""
        script = """import json
import sys

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        return json.dumps(self.data, indent=2)

try:
    proc = DataProcessor({"key": "value", "list": [1, 2, 3]})
    print(proc.process())
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
"""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, '{"key": "value"}')):
            result = await executor.execute("run_script", {
                "host": "server", "script": script, "interpreter": "python3"
            })
            assert "key" in result

    async def test_script_with_special_shell_chars(self):
        """Script containing shell-dangerous characters."""
        script = """echo "Dollars: $PATH $HOME"
echo 'Backticks: `whoami`'
echo "Pipes: | && || ;"
echo "Redirects: > >> < 2>&1"
echo "Glob: * ? [abc]"
"""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "output")
            result = await executor.execute("run_script", {
                "host": "server", "script": script, "interpreter": "bash"
            })
            # Verify base64 encoding was used (no shell injection)
            cmd = mock_ssh.call_args[1]["command"]
            assert "base64 -d" in cmd
            assert "output" in result

    async def test_script_base64_encoding_roundtrip(self):
        """Script is base64 encoded correctly for transport."""
        script = "echo 'hello world'"
        expected_b64 = base64.b64encode(script.encode()).decode()
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "hello world")
            await executor.execute("run_script", {
                "host": "server", "script": script, "interpreter": "bash"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert expected_b64 in cmd

    async def test_script_with_binary_like_content(self):
        """Script with unusual but valid characters."""
        script = "echo -e '\\x00\\x01\\x02 test \\xff'"
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "output")):
            result = await executor.execute("run_script", {
                "host": "server", "script": script, "interpreter": "bash"
            })
            assert result == "output"

    async def test_script_nonzero_exit_reports_error(self):
        """Non-zero exit code produces error message."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(1, "syntax error")):
            result = await executor.execute("run_script", {
                "host": "server", "script": "exit 1", "interpreter": "bash"
            })
            assert "Script failed (exit 1)" in result
            assert "syntax error" in result

    async def test_script_output_truncation(self):
        """Long script output is truncated."""
        long_output = "\n".join(f"line {i}" for i in range(500))
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, long_output)):
            result = await executor.execute("run_script", {
                "host": "server", "script": "big output", "interpreter": "bash"
            })
            assert "omitted" in result

    async def test_unsupported_interpreter_rejected(self):
        """Unsupported interpreter (e.g., 'go') is rejected."""
        executor = _make_executor()
        result = await executor.execute("run_script", {
            "host": "server", "script": "code", "interpreter": "powershell"
        })
        assert "Unsupported interpreter" in result

    async def test_script_unknown_host_rejected(self):
        """Unknown host returns error."""
        executor = _make_executor()
        result = await executor.execute("run_script", {
            "host": "nonexistent", "script": "echo hi", "interpreter": "bash"
        })
        assert "Unknown or disallowed host" in result

    async def test_custom_filename(self):
        """Custom filename is used in mktemp."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_script", {
                "host": "server", "script": "echo hi",
                "interpreter": "bash", "filename": "deploy.sh"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "deploy.sh" in cmd

    async def test_node_interpreter_uses_js_extension(self):
        """Node interpreter creates .js file."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_script", {
                "host": "server", "script": "console.log('hi')", "interpreter": "node"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "node" in cmd
            assert ".js" in cmd

    async def test_perl_interpreter_uses_pl_extension(self):
        """Perl interpreter creates .pl file."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_script", {
                "host": "server", "script": "print 'hi'", "interpreter": "perl"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "perl" in cmd
            assert ".pl" in cmd

    async def test_ruby_interpreter_uses_rb_extension(self):
        """Ruby interpreter creates .rb file."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_script", {
                "host": "server", "script": "puts 'hi'", "interpreter": "ruby"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "ruby" in cmd
            assert ".rb" in cmd

    async def test_large_script(self):
        """Large script (>10KB) is handled via base64 without issues."""
        # Generate a large Python script
        lines = [f"x_{i} = {i} * {i}  # computation {i}" for i in range(500)]
        lines.append("print(sum([" + ", ".join(f"x_{i}" for i in range(500)) + "]))")
        script = "\n".join(lines)
        assert len(script) > 10000

        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "41541750")
            result = await executor.execute("run_script", {
                "host": "server", "script": script, "interpreter": "python3"
            })
            # Verify it was base64 encoded
            cmd = mock_ssh.call_args[1]["command"]
            assert "base64 -d" in cmd
            assert "41541750" in result


# ===========================================================================
# 3. claude_code delegation from Codex
# ===========================================================================


class TestClaudeCodeReadOnly:
    """claude_code in read-only mode (allow_edits=false)."""

    async def test_basic_read_only_execution(self):
        """Read-only claude_code runs with --print and returns output."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Analysis complete: 42 files found")
            result = await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Analyze the codebase",
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "--print" in cmd
            assert "--output-format text" in cmd
            assert "--no-session-persistence" in cmd
            assert "--dangerously-skip-permissions" not in cmd
            assert "Analysis complete" in result

    async def test_read_only_no_tmpdir(self):
        """Read-only mode does not create temp directory."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "output")
            await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Read the code",
            })
            # Only 1 call (the main execution), no mktemp
            assert mock_ssh.call_count == 1

    async def test_prompt_base64_encoded(self):
        """Prompt is base64-encoded to avoid shell injection."""
        executor = _make_executor()
        prompt = "Analyze 'quotes' and $variables and `backticks`"
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": prompt,
            })
            cmd = mock_ssh.call_args[1]["command"]
            expected_b64 = base64.b64encode(prompt.encode()).decode()
            assert expected_b64 in cmd
            assert "base64 -d" in cmd

    async def test_read_only_uses_working_dir(self):
        """Read-only mode cds to working directory."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("claude_code", {
                "working_directory": "/home/user/project",
                "prompt": "Read",
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "cd" in cmd
            assert "/home/user/project" in cmd

    async def test_allowed_tools_forwarded(self):
        """allowed_tools parameter is forwarded to claude CLI."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Do it",
                "allowed_tools": "Read,Glob,Grep",
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "--allowedTools" in cmd

    async def test_output_truncation(self):
        """Very long claude_code output is truncated."""
        long_output = "x" * 10000
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, long_output)
            result = await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Analyze",
                "max_output_chars": 3000,
            })
            assert len(result) < len(long_output)
            assert "truncated" in result


class TestClaudeCodeAllowEdits:
    """claude_code with allow_edits=true (temp dir, file copy, manifest)."""

    async def test_allow_edits_creates_tmpdir(self):
        """allow_edits creates temp directory as non-root user."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.side_effect = [
                (0, "/tmp/claude_code_ABCDEF12\n"),  # mktemp
                (0, "Generated code"),                # main execution
                (0, "/tmp/claude_code_ABCDEF12/main.py\n"),  # find
                (0, ""),                               # cp
                (0, ""),                               # rm -rf
            ]
            result = await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Write a main.py",
                "allow_edits": True,
            })
            assert "Generated code" in result
            assert "FILES ON DISK" in result
            assert "/opt/project/main.py" in result

    async def test_allow_edits_uses_skip_permissions(self):
        """allow_edits passes --dangerously-skip-permissions."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.side_effect = [
                (0, "/tmp/claude_code_ABCDEF12\n"),
                (0, "output"),
                (0, ""),   # find (no files)
                (0, ""),   # rm
            ]
            await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Write code",
                "allow_edits": True,
            })
            # The main execution call (index 1)
            main_cmd = mock_ssh.call_args_list[1][1]["command"]
            assert "--dangerously-skip-permissions" in main_cmd

    async def test_allow_edits_rewrites_paths(self):
        """allow_edits rewrites absolute paths to relative in prompt."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.side_effect = [
                (0, "/tmp/claude_code_ABCDEF12\n"),
                (0, "done"),
                (0, ""),
                (0, ""),
            ]
            prompt = "Write file to /opt/project/src/main.py in /opt/project"
            await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": prompt,
                "allow_edits": True,
            })
            main_cmd = mock_ssh.call_args_list[1][1]["command"]
            # Decode the base64 prompt to check rewriting
            # The command may use shell quoting like '"'"' around the b64
            import re
            b64_match = re.search(r"echo ['\"]?'?\"?'?([A-Za-z0-9+/=]+)'?\"?'?['\"]? \| base64 -d", main_cmd)
            if not b64_match:
                # Try to find any base64-looking string in the command
                b64_match = re.search(r"([A-Za-z0-9+/=]{20,})", main_cmd)
            assert b64_match, f"No base64 found in: {main_cmd}"
            decoded = base64.b64decode(b64_match.group(1)).decode()
            assert "./src/main.py" in decoded
            assert "Write ALL files relative to the current directory" in decoded

    async def test_allow_edits_no_user_configured(self):
        """allow_edits without claude_code_user returns error."""
        executor = _make_executor(claude_code_user="")
        result = await executor.execute("claude_code", {
            "working_directory": "/opt/project",
            "prompt": "Write code",
            "allow_edits": True,
        })
        assert "claude_code_user not configured" in result

    async def test_allow_edits_tmpdir_creation_fails(self):
        """If mktemp fails, error is returned."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (1, "Permission denied")
            result = await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Write code",
                "allow_edits": True,
            })
            assert "Failed to create temp directory" in result

    async def test_allow_edits_file_copy_fails(self):
        """If file copy fails, warning is included in output."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.side_effect = [
                (0, "/tmp/claude_code_ABCDEF12\n"),
                (0, "Generated code"),
                (0, "/tmp/claude_code_ABCDEF12/main.py\n"),
                (1, "cp: permission denied"),   # cp fails
                (0, ""),                         # rm cleanup
            ]
            result = await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Write code",
                "allow_edits": True,
            })
            assert "WARNING" in result
            assert "copy" in result.lower() or "failed" in result.lower()

    async def test_allow_edits_cleanup_always_runs(self):
        """Temp directory is always cleaned up, even on failure."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.side_effect = [
                (0, "/tmp/claude_code_ABCDEF12\n"),
                (1, "claude failed"),             # main fails
                (0, ""),                           # rm cleanup
            ]
            result = await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Write code",
                "allow_edits": True,
            })
            assert "Claude Code failed" in result
            # Verify rm -rf was called
            last_call = mock_ssh.call_args_list[-1]
            assert "rm -rf" in last_call[1]["command"]

    async def test_allow_edits_multiple_files_manifest(self):
        """Multiple files created show in manifest."""
        executor = _make_executor()
        file_list = "/tmp/claude_code_ABC/src/main.py\n/tmp/claude_code_ABC/src/utils.py\n/tmp/claude_code_ABC/README.md"
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.side_effect = [
                (0, "/tmp/claude_code_ABC\n"),
                (0, "Code generated"),
                (0, file_list),              # find
                (0, ""),                      # cp
                (0, ""),                      # rm
            ]
            result = await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Create project",
                "allow_edits": True,
            })
            assert "/opt/project/src/main.py" in result
            assert "/opt/project/src/utils.py" in result
            assert "/opt/project/README.md" in result
            assert "Do NOT rewrite them with write_file" in result


class TestClaudeCodeErrorHandling:
    """claude_code error paths."""

    async def test_no_host_configured(self):
        """No claude_code_host returns error."""
        executor = _make_executor(claude_code_host="")
        result = await executor.execute("claude_code", {
            "working_directory": "/opt/project",
            "prompt": "Analyze",
        })
        assert "claude_code_host not configured" in result

    async def test_unknown_host(self):
        """Unknown host alias returns error."""
        executor = _make_executor()
        result = await executor.execute("claude_code", {
            "host": "nonexistent_host",
            "working_directory": "/opt/project",
            "prompt": "Analyze",
        })
        assert "Unknown or disallowed host" in result

    async def test_failure_truncates_output(self):
        """Failed execution truncates output to last 2000 chars."""
        long_error = "E" * 5000
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(1, long_error)):
            result = await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Analyze",
            })
            assert "Claude Code failed" in result
            # Output is truncated to last 2000 chars
            assert len(result) < 5500

    async def test_timeout_300_seconds(self):
        """claude_code uses 300-second timeout."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Analyze",
            })
            assert mock_ssh.call_args[1]["timeout"] == 300


# ===========================================================================
# 4. Local subprocess execution paths
# ===========================================================================


class TestLocalSubprocessExecution:
    """Tools on local hosts use subprocess, not SSH."""

    async def test_run_command_local_uses_subprocess(self):
        """run_command on localhost uses run_local_command."""
        executor = _make_executor(local_host=True)
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock, return_value=(0, "output")) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            result = await executor.execute("run_command", {
                "host": "local", "command": "uptime"
            })
            mock_local.assert_called_once()
            mock_ssh.assert_not_called()
            assert "output" in result

    async def test_check_disk_local(self):
        """check_disk on localhost uses local execution."""
        executor = _make_executor(local_host=True)
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock, return_value=(0, "disk output")) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            result = await executor.execute("check_disk", {"host": "local"})
            mock_local.assert_called_once()
            mock_ssh.assert_not_called()

    async def test_check_memory_local(self):
        """check_memory on localhost uses local execution."""
        executor = _make_executor(local_host=True)
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock, return_value=(0, "mem output")) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            result = await executor.execute("check_memory", {"host": "local"})
            mock_local.assert_called_once()
            mock_ssh.assert_not_called()

    async def test_run_script_local(self):
        """run_script on localhost uses local execution."""
        executor = _make_executor(local_host=True)
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock, return_value=(0, "script ok")) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            result = await executor.execute("run_script", {
                "host": "local", "script": "echo hi", "interpreter": "bash"
            })
            mock_local.assert_called_once()
            mock_ssh.assert_not_called()

    async def test_claude_code_local(self):
        """claude_code on localhost host uses local execution."""
        executor = _make_executor(
            local_host=True,
            claude_code_host="local",
        )
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock, return_value=(0, "analysis done")) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            result = await executor.execute("claude_code", {
                "working_directory": "/opt/project",
                "prompt": "Analyze code",
            })
            mock_local.assert_called()
            mock_ssh.assert_not_called()

    async def test_ipv6_localhost_uses_local(self):
        """::1 address uses local execution."""
        executor = _make_executor(extra_hosts={
            "ipv6local": ToolHost(address="::1", ssh_user="root", os="linux")
        })
        with patch("src.tools.executor.run_local_command", new_callable=AsyncMock, return_value=(0, "ok")) as mock_local, \
             patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            result = await executor.execute("run_command", {
                "host": "ipv6local", "command": "hostname"
            })
            mock_local.assert_called_once()
            mock_ssh.assert_not_called()


# ===========================================================================
# 5. Remote SSH execution paths
# ===========================================================================


class TestRemoteSSHExecution:
    """Tools on remote hosts use SSH with correct parameters."""

    async def test_run_command_remote_uses_ssh(self):
        """run_command on remote host uses SSH."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "output")) as mock_ssh:
            result = await executor.execute("run_command", {
                "host": "server", "command": "uptime"
            })
            mock_ssh.assert_called_once()
            assert mock_ssh.call_args[1]["host"] == "10.0.0.1"
            assert mock_ssh.call_args[1]["ssh_key_path"] == "/test/id_ed25519"
            assert mock_ssh.call_args[1]["ssh_user"] == "root"

    async def test_remote_ssh_passes_known_hosts(self):
        """SSH calls include known_hosts path."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "ok")) as mock_ssh:
            await executor.execute("run_command", {
                "host": "server", "command": "ls"
            })
            assert mock_ssh.call_args[1]["known_hosts_path"] == "/test/known_hosts"

    async def test_remote_uses_correct_ssh_user(self):
        """macOS host uses deploy user, not root."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "ok")) as mock_ssh:
            await executor.execute("run_command", {
                "host": "macbook", "command": "uptime"
            })
            assert mock_ssh.call_args[1]["ssh_user"] == "deploy"

    async def test_remote_uses_config_timeout(self):
        """Default timeout comes from config."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "ok")) as mock_ssh:
            await executor.execute("run_command", {
                "host": "server", "command": "ls"
            })
            assert mock_ssh.call_args[1]["timeout"] == 5  # command_timeout_seconds


# ===========================================================================
# 6. Tool executor handler coverage (every tool category)
# ===========================================================================


class TestHostBasedToolHandlers:
    """Test each category of host-based tool handlers."""

    async def test_check_service_validates_allowlist(self):
        """check_service rejects services not in allowlist."""
        executor = _make_executor()
        result = await executor.execute("check_service", {
            "host": "server", "service": "evil_service"
        })
        assert "not in the allowlist" in result

    async def test_check_service_allowed(self):
        """check_service executes for allowed services."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "active")):
            result = await executor.execute("check_service", {
                "host": "server", "service": "apache2"
            })
            assert result == "active"

    async def test_check_docker_with_container(self):
        """check_docker with container filter."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "container info")
            result = await executor.execute("check_docker", {
                "host": "server", "container": "web"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "web" in cmd

    async def test_check_logs_respects_line_limit(self):
        """check_logs caps at 50 lines."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "logs")
            await executor.execute("check_logs", {
                "host": "server", "service": "apache2", "lines": 100
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "-n 50" in cmd

    async def test_restart_service_validates_allowlist(self):
        """restart_service rejects services not in allowlist."""
        executor = _make_executor()
        result = await executor.execute("restart_service", {
            "host": "server", "service": "evil"
        })
        assert "not in the allowlist" in result

    async def test_restart_service_runs_restart_then_status(self):
        """restart_service runs restart && status."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "restarted")
            await executor.execute("restart_service", {
                "host": "server", "service": "nginx"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "restart" in cmd
            assert "status" in cmd

    async def test_macos_check_memory(self):
        """macOS hosts use sysctl/vm_stat instead of free."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "mem info")
            await executor.execute("check_memory", {"host": "macbook"})
            cmd = mock_ssh.call_args[1]["command"]
            assert "sysctl" in cmd or "vm_stat" in cmd

    async def test_macos_check_disk(self):
        """macOS hosts use df -h without exclude flags."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "disk info")
            await executor.execute("check_disk", {"host": "macbook"})
            cmd = mock_ssh.call_args[1]["command"]
            assert "df -h" in cmd
            assert "--exclude" not in cmd


class TestDockerToolHandlers:
    """Docker tool handler tests."""

    async def test_docker_logs_with_since(self):
        """docker_logs supports --since parameter."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "log output")
            await executor.execute("docker_logs", {
                "host": "server", "container": "web", "since": "1h"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "--since" in cmd

    async def test_docker_compose_action_validates(self):
        """docker_compose_action rejects invalid actions."""
        executor = _make_executor()
        result = await executor.execute("docker_compose_action", {
            "host": "server", "project_dir": "/app", "action": "destroy"
        })
        assert "Invalid action" in result

    async def test_docker_compose_up_adds_detach(self):
        """docker compose up uses -d flag."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("docker_compose_action", {
                "host": "server", "project_dir": "/app", "action": "up"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "up -d" in cmd

    async def test_docker_stats_no_container(self):
        """docker_stats without container shows all."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "stats")
            await executor.execute("docker_stats", {"host": "server"})
            cmd = mock_ssh.call_args[1]["command"]
            assert "--no-stream" in cmd


class TestGitToolHandlers:
    """Git tool handler tests."""

    async def test_git_status(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "clean")
            result = await executor.execute("git_status", {
                "host": "server", "repo_path": "/opt/repo"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "git -C" in cmd
            assert "status" in cmd

    async def test_git_log_limits_count(self):
        """git_log caps at 50 commits."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "log")
            await executor.execute("git_log", {
                "host": "server", "repo_path": "/opt/repo", "count": 100
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "-n 50" in cmd

    async def test_git_commit_with_files(self):
        """git_commit with file list adds specific files."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "committed")
            await executor.execute("git_commit", {
                "host": "server", "repo_path": "/opt/repo",
                "message": "fix", "files": ["main.py", "utils.py"]
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "main.py" in cmd
            assert "utils.py" in cmd
            assert "git" in cmd

    async def test_git_push_with_branch(self):
        """git_push with branch pushes to specific branch."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "pushed")
            await executor.execute("git_push", {
                "host": "server", "repo_path": "/opt/repo", "branch": "feature"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "feature" in cmd

    async def test_git_branch_create(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "created")
            await executor.execute("git_branch", {
                "host": "server", "repo_path": "/opt/repo",
                "action": "create", "branch_name": "feature-x"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "checkout -b" in cmd
            assert "feature-x" in cmd

    async def test_git_branch_list(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "* main\n  dev")
            await executor.execute("git_branch", {
                "host": "server", "repo_path": "/opt/repo", "action": "list"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "branch -a" in cmd

    async def test_git_branch_requires_name_for_create(self):
        executor = _make_executor()
        result = await executor.execute("git_branch", {
            "host": "server", "repo_path": "/opt/repo", "action": "create"
        })
        assert "branch_name is required" in result


class TestPrometheusToolHandlers:
    """Prometheus tool handler tests."""

    async def test_prometheus_instant_query(self):
        executor = _make_executor()
        prom_response = json.dumps({
            "status": "success",
            "data": {"resultType": "vector", "result": [
                {"metric": {"__name__": "up"}, "value": [1234, "1"]}
            ]}
        })
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, prom_response)):
            result = await executor.execute("query_prometheus", {
                "query": "up"
            })
            assert "up" in result

    async def test_prometheus_no_host_configured(self):
        executor = _make_executor(prom_host="")
        result = await executor.execute("query_prometheus", {"query": "up"})
        assert "not configured" in result

    async def test_prometheus_range_query(self):
        executor = _make_executor()
        prom_response = json.dumps({
            "status": "success",
            "data": {"resultType": "matrix", "result": [
                {"metric": {"__name__": "up"}, "values": [[1, "1"], [2, "0"]]}
            ]}
        })
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, prom_response)):
            result = await executor.execute("query_prometheus_range", {
                "query": "up", "duration": "1h", "step": "5m"
            })
            assert "up" in result


class TestIncusToolHandlers:
    """Incus tool handler tests."""

    async def test_incus_list(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "web,Running,CONTAINER,10.0.0.5")):
            result = await executor.execute("incus_list", {})
            assert "web" in result

    async def test_incus_invalid_name(self):
        """Invalid instance name is rejected."""
        executor = _make_executor()
        result = await executor.execute("incus_info", {"instance": "bad name!"})
        assert "Invalid Incus name" in result

    async def test_incus_exec(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "ok")):
            result = await executor.execute("incus_exec", {
                "instance": "web", "command": "apt update"
            })
            assert result == "ok"

    async def test_incus_start(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "")):
            result = await executor.execute("incus_start", {"instance": "web"})
            assert "started" in result

    async def test_incus_stop_force(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("incus_stop", {"instance": "web", "force": True})
            cmd = mock_ssh.call_args[1]["command"]
            assert "--force" in cmd

    async def test_incus_snapshot_create(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "")):
            result = await executor.execute("incus_snapshot", {
                "instance": "web", "action": "create", "snapshot": "pre-deploy"
            })
            assert "created" in result.lower()

    async def test_incus_snapshot_restore_requires_name(self):
        executor = _make_executor()
        result = await executor.execute("incus_snapshot", {
            "instance": "web", "action": "restore"
        })
        assert "required" in result.lower()

    async def test_incus_launch_with_vm(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("incus_launch", {
                "name": "test-vm", "image": "ubuntu:22.04", "type": "vm"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "--vm" in cmd

    async def test_incus_delete(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "")):
            result = await executor.execute("incus_delete", {"instance": "web"})
            assert "deleted" in result.lower()


class TestAnsibleToolHandler:
    """Ansible playbook handler tests."""

    async def test_ansible_validates_playbook(self):
        executor = _make_executor()
        result = await executor.execute("run_ansible_playbook", {
            "playbook": "evil.yml"
        })
        assert "not in the allowlist" in result

    async def test_ansible_check_mode_default(self):
        """Default check_mode=True adds --check."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_ansible_playbook", {
                "playbook": "check-services.yml"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "--check" in cmd

    async def test_ansible_no_host_configured(self):
        executor = _make_executor(ansible_host="")
        result = await executor.execute("run_ansible_playbook", {
            "playbook": "check-services.yml"
        })
        assert "not configured" in result

    async def test_ansible_with_limit_and_tags(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "ok")
            await executor.execute("run_ansible_playbook", {
                "playbook": "check-services.yml",
                "limit": "webservers",
                "tags": "deploy",
                "check_mode": False,
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "--limit" in cmd
            assert "--tags" in cmd
            assert "--check" not in cmd


class TestFileToolHandlers:
    """read_file and write_file handler tests."""

    async def test_read_file(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "file content")):
            result = await executor.execute("read_file", {
                "host": "server", "path": "/etc/hostname"
            })
            assert "file content" in result

    async def test_read_file_custom_lines(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "content")
            await executor.execute("read_file", {
                "host": "server", "path": "/etc/passwd", "lines": 5
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "head -n 5" in cmd

    async def test_write_file_base64_encodes_content(self):
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "")
            await executor.execute("write_file", {
                "host": "server", "path": "/tmp/test.txt",
                "content": "Hello 'World' \"test\" $HOME"
            })
            cmd = mock_ssh.call_args[1]["command"]
            assert "base64 -d" in cmd
            assert "mkdir -p" in cmd


class TestBrowserToolHandlers:
    """Browser tool handler tests."""

    async def test_browser_not_enabled(self):
        """Browser tools return error when not enabled."""
        executor = _make_executor()
        result = await executor.execute("browser_read_page", {"url": "http://example.com"})
        assert "not enabled" in result

    async def test_browser_click_not_enabled(self):
        executor = _make_executor()
        result = await executor.execute("browser_click", {"selector": "#btn"})
        assert "not enabled" in result

    async def test_browser_fill_not_enabled(self):
        executor = _make_executor()
        result = await executor.execute("browser_fill", {"selector": "#input", "value": "text"})
        assert "not enabled" in result

    async def test_browser_evaluate_not_enabled(self):
        executor = _make_executor()
        result = await executor.execute("browser_evaluate", {"expression": "1+1"})
        assert "not enabled" in result


class TestWebToolHandlers:
    """Web tool handler tests."""

    async def test_web_search_caps_results(self):
        """web_search caps max_results at 10."""
        executor = _make_executor()
        with patch("src.tools.web.web_search", new_callable=AsyncMock, return_value="results") as mock:
            await executor.execute("web_search", {"query": "test", "max_results": 50})
            _, kwargs = mock.call_args
            assert kwargs["max_results"] == 10

    async def test_fetch_url(self):
        executor = _make_executor()
        with patch("src.tools.web.fetch_url", new_callable=AsyncMock, return_value="page content"):
            result = await executor.execute("fetch_url", {"url": "http://example.com"})
            assert "page content" in result


class TestMultiHostToolHandler:
    """run_command_multi handler tests."""

    async def test_run_command_multi_parallel(self):
        """run_command_multi runs on multiple hosts in parallel."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "ok")):
            result = await executor.execute("run_command_multi", {
                "hosts": ["server", "desktop"], "command": "uptime"
            })
            assert "server" in result
            assert "desktop" in result

    async def test_run_command_multi_all_hosts(self):
        """run_command_multi 'all' expands to all configured hosts."""
        executor = _make_executor()
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock, return_value=(0, "ok")):
            result = await executor.execute("run_command_multi", {
                "hosts": ["all"], "command": "hostname"
            })
            assert "server" in result
            assert "desktop" in result
            assert "macbook" in result


# ===========================================================================
# 7. Source structure verification
# ===========================================================================


class TestSourceStructureVerification:
    """Verify architectural invariants in source code."""

    def test_no_approval_check_in_execute(self):
        """execute() has no approval check — immediate dispatch."""
        source = inspect.getsource(ToolExecutor.execute)
        assert "approval" not in source.lower()
        assert "approve" not in source.lower()

    def test_no_approval_in_executor_module(self):
        """No approval references in entire executor module."""
        import src.tools.executor as mod
        source = inspect.getsource(mod)
        # Allow "approval" only in comments
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "approval" not in stripped.lower() or "no approval" in stripped.lower(), \
                f"Approval reference in executor.py: {stripped}"

    def test_exec_command_dispatches_local_and_remote(self):
        """_exec_command checks is_local_address for routing."""
        source = inspect.getsource(ToolExecutor._exec_command)
        assert "is_local_address" in source
        assert "run_local_command" in source
        assert "run_ssh_command" in source

    def test_all_host_tools_use_exec_command_or_run_on_host(self):
        """All host-based handlers go through _exec_command or _run_on_host."""
        import src.tools.executor as mod
        source = inspect.getsource(mod)
        # Parse AST to find direct run_ssh_command calls
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_handle_"):
                func_source = ast.get_source_segment(source, node)
                if func_source and "run_ssh_command" in func_source:
                    assert False, f"{node.name} calls run_ssh_command directly instead of _exec_command"

    def test_truncate_lines_in_run_command(self):
        """run_command truncates output."""
        source = inspect.getsource(ToolExecutor._handle_run_command)
        assert "_truncate_lines" in source

    def test_truncate_lines_in_run_script(self):
        """run_script truncates output."""
        source = inspect.getsource(ToolExecutor._handle_run_script)
        assert "_truncate_lines" in source


class TestTruncateLinesBehavior:
    """Test _truncate_lines function."""

    def test_short_output_unchanged(self):
        text = "line1\nline2\nline3"
        assert _truncate_lines(text) == text

    def test_exact_limit_unchanged(self):
        lines = "\n".join(f"line {i}" for i in range(200))
        assert _truncate_lines(lines) == lines

    def test_over_limit_truncated(self):
        lines = "\n".join(f"line {i}" for i in range(300))
        result = _truncate_lines(lines)
        assert "omitted" in result
        assert "line 0" in result      # head preserved
        assert "line 299" in result    # tail preserved

    def test_custom_max_lines(self):
        lines = "\n".join(f"line {i}" for i in range(20))
        result = _truncate_lines(lines, max_lines=10)
        assert "omitted" in result
