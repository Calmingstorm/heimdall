"""Tests for Round 5 Quick Wins — 5 fixes across background_task.py and executor.py.

Covers:
1. Secret scrubbing in background task tool output
2. requester_id field on BackgroundTask + audit logging integration
3. _send_summary activation in run_background_task
4. cp -a return code check in _handle_claude_code
5. timeout 1200 wrapper for remote claude command
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.background_task import (  # noqa: E402
    BackgroundTask,
    StepResult,
    run_background_task,
    _send_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_channel():
    ch = AsyncMock()
    msg = AsyncMock()
    msg.edit = AsyncMock()
    ch.send = AsyncMock(return_value=msg)
    ch.id = 12345
    return ch, msg


def _make_task(steps=None, description="Test task", channel=None, **kwargs):
    ch = channel or _make_channel()[0]
    return BackgroundTask(
        task_id="abc12345",
        description=description,
        steps=steps or [],
        channel=ch,
        requester="user123",
        **kwargs,
    )


def _make_executor(return_value="executed ok"):
    ex = MagicMock()
    ex.execute = AsyncMock(return_value=return_value)
    return ex


def _make_skill_manager():
    sm = MagicMock()
    sm.has_skill = MagicMock(return_value=False)
    sm.execute = AsyncMock(return_value="skill result")
    return sm


# ---------------------------------------------------------------------------
# Fix 1: Secret scrubbing in background task tool output
# ---------------------------------------------------------------------------

class TestBackgroundTaskSecretScrubbing:
    async def test_api_key_scrubbed_from_step_result(self):
        """Tool output containing an API key should be scrubbed in StepResult."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "run_command", "tool_input": {"command": "env"},
                    "description": "show env"}],
            channel=ch,
        )
        ex = _make_executor(return_value="api_key=sk-abcdefghijklmnopqrstuvwxyz12345")

        await run_background_task(task, ex, _make_skill_manager())

        assert task.status == "completed"
        assert "sk-abc" not in task.results[0].output
        assert "[REDACTED]" in task.results[0].output

    async def test_password_scrubbed_from_step_result(self):
        """Tool output containing a password should be scrubbed."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "run_command", "tool_input": {},
                    "description": "show config"}],
            channel=ch,
        )
        ex = _make_executor(return_value="password=hunter2 and more text")

        await run_background_task(task, ex, _make_skill_manager())

        assert "hunter2" not in task.results[0].output
        assert "[REDACTED]" in task.results[0].output

    async def test_scrubbed_value_propagated_to_prev_output(self):
        """The scrubbed output should be used for {prev_output} in the next step."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {},
                 "description": "get secret"},
                {"tool_name": "run_command",
                 "tool_input": {"data": "got: {prev_output}"},
                 "description": "use output"},
            ],
            channel=ch,
        )
        call_count = [0]

        async def side_effect(name, inp):
            call_count[0] += 1
            if call_count[0] == 1:
                return "password=supersecret123"
            return f"received: {inp.get('data', '')}"

        ex = _make_executor()
        ex.execute = AsyncMock(side_effect=side_effect)

        await run_background_task(task, ex, _make_skill_manager())

        # The second step's input should contain the scrubbed version
        second_call = ex.execute.call_args_list[1]
        assert "supersecret123" not in second_call[0][1]["data"]
        assert "[REDACTED]" in second_call[0][1]["data"]


# ---------------------------------------------------------------------------
# Fix 2: requester_id field + audit logging
# ---------------------------------------------------------------------------

class TestBackgroundTaskRequesterIdField:
    def test_requester_id_default_empty(self):
        """BackgroundTask.requester_id defaults to empty string."""
        task = _make_task()
        assert task.requester_id == ""

    def test_requester_id_set_explicitly(self):
        """BackgroundTask.requester_id can be set."""
        task = _make_task(requester_id="123456789")
        assert task.requester_id == "123456789"


class TestBackgroundTaskAuditLogging:
    async def test_audit_logger_called_on_success(self):
        """Audit logger is called for each successful step."""
        ch, _ = _make_channel()
        ch.id = 99999
        task = _make_task(
            steps=[
                {"tool_name": "check_disk", "tool_input": {}, "description": "check"},
                {"tool_name": "check_memory", "tool_input": {}, "description": "mem"},
            ],
            channel=ch,
            requester_id="111222",
        )
        audit = AsyncMock()
        audit.log_execution = AsyncMock()

        await run_background_task(
            task, _make_executor(), _make_skill_manager(),
            audit_logger=audit,
        )

        assert audit.log_execution.call_count == 2
        first_call = audit.log_execution.call_args_list[0]
        assert first_call[1]["user_id"] == "111222"
        assert first_call[1]["user_name"] == "user123"
        assert first_call[1]["tool_name"] == "check_disk"
        assert first_call[1]["approved"] is True
        assert first_call[1].get("error") is None

    async def test_audit_logger_called_on_error(self):
        """Audit logger is called with error details on tool failure."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "run_command", "tool_input": {},
                    "description": "fail", "on_failure": "continue"}],
            channel=ch,
            requester_id="333",
        )
        ex = _make_executor()
        ex.execute = AsyncMock(side_effect=RuntimeError("SSH timeout"))
        audit = AsyncMock()
        audit.log_execution = AsyncMock()

        await run_background_task(
            task, ex, _make_skill_manager(),
            audit_logger=audit,
        )

        assert audit.log_execution.call_count == 1
        call_kwargs = audit.log_execution.call_args_list[0][1]
        assert call_kwargs["error"] == "SSH timeout"
        assert call_kwargs["tool_name"] == "run_command"

    async def test_audit_logger_failure_does_not_crash_task(self):
        """If audit logging fails, the task continues normally."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "check_disk", "tool_input": {}, "description": "check"}],
            channel=ch,
        )
        audit = AsyncMock()
        audit.log_execution = AsyncMock(side_effect=Exception("Disk full"))

        await run_background_task(
            task, _make_executor(), _make_skill_manager(),
            audit_logger=audit,
        )

        # Task should still complete despite audit failure
        assert task.status == "completed"
        assert len(task.results) == 1
        assert task.results[0].status == "ok"

    async def test_no_audit_logger_is_fine(self):
        """Task works normally without an audit logger."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "check_disk", "tool_input": {}, "description": "check"}],
            channel=ch,
        )

        await run_background_task(task, _make_executor(), _make_skill_manager())

        assert task.status == "completed"


# ---------------------------------------------------------------------------
# Fix 3: _send_summary activation
# ---------------------------------------------------------------------------

class TestSendSummaryActivation:
    async def test_send_summary_called_after_completion(self):
        """run_background_task should call _send_summary after completing."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "check_disk", "tool_input": {}, "description": "check"}],
            channel=ch,
        )

        with patch("src.discord.background_task._send_summary", new_callable=AsyncMock) as mock_summary:
            await run_background_task(task, _make_executor(), _make_skill_manager())
            mock_summary.assert_called_once_with(task)

    async def test_send_summary_called_on_failure(self):
        """_send_summary is called even when the task fails."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "run_command", "tool_input": {},
                    "description": "fail", "on_failure": "abort"}],
            channel=ch,
        )
        ex = _make_executor()
        ex.execute = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("src.discord.background_task._send_summary", new_callable=AsyncMock) as mock_summary:
            await run_background_task(task, ex, _make_skill_manager())
            mock_summary.assert_called_once_with(task)
            assert task.status == "failed"

    async def test_send_summary_called_on_cancellation(self):
        """_send_summary is called when a task is cancelled."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "run_command", "tool_input": {}, "description": "step"}],
            channel=ch,
        )
        task.cancel()

        with patch("src.discord.background_task._send_summary", new_callable=AsyncMock) as mock_summary:
            await run_background_task(task, _make_executor(), _make_skill_manager())
            mock_summary.assert_called_once_with(task)
            assert task.status == "cancelled"


# ---------------------------------------------------------------------------
# Fix 4: cp -a return code check in _handle_claude_code
# ---------------------------------------------------------------------------

class TestClaudeCodeCpReturnCode:
    @pytest.fixture
    def executor(self):
        from src.tools.executor import ToolExecutor
        config = MagicMock()
        config.hosts = {"desktop": MagicMock(address="10.0.0.2", ssh_user="root", os="linux")}
        config.ssh_key_path = "/app/.ssh/id_ed25519"
        config.ssh_known_hosts_path = "/app/.ssh/known_hosts"
        config.command_timeout_seconds = 30
        config.claude_code_host = "desktop"
        config.claude_code_user = "deploy"
        config.claude_code_dir = "/opt/project"
        config.memory_path = None
        return ToolExecutor(config)

    async def test_cp_failure_shows_warning(self, executor):
        """When cp -a fails, the output should contain a WARNING about copy failure."""
        call_count = [0]
        tmpdir = "/tmp/claude_code_abcd1234"

        async def mock_ssh(host, command, timeout, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # mktemp -d
                return (0, tmpdir)
            elif call_count[0] == 2:
                # claude -p execution succeeds
                return (0, "Code written successfully")
            elif call_count[0] == 3:
                # find command returns files
                return (0, f"{tmpdir}/main.py\n{tmpdir}/util.py")
            elif call_count[0] == 4:
                # cp -a FAILS (disk full)
                return (1, "cp: error writing '/root/project/main.py': No space left on device")
            elif call_count[0] == 5:
                # rm -rf cleanup
                return (0, "")
            return (0, "")

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock,
                    side_effect=mock_ssh):
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "write a script",
                "allow_edits": True,
            })

        assert "WARNING" in result
        assert "failed" in result
        assert "FILES ON DISK" not in result

    async def test_cp_success_shows_manifest(self, executor):
        """When cp -a succeeds, the output shows FILES ON DISK as before."""
        call_count = [0]
        tmpdir = "/tmp/claude_code_abcd1234"

        async def mock_ssh(host, command, timeout, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # mktemp -d
                return (0, tmpdir)
            elif call_count[0] == 2:
                # claude -p
                return (0, "Code written")
            elif call_count[0] == 3:
                # find
                return (0, f"{tmpdir}/main.py")
            elif call_count[0] == 4:
                # cp -a succeeds
                return (0, "")
            elif call_count[0] == 5:
                # rm -rf cleanup
                return (0, "")
            return (0, "")

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock,
                    side_effect=mock_ssh):
            result = await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "write a script",
                "allow_edits": True,
            })

        assert "FILES ON DISK" in result
        assert "WARNING" not in result


# ---------------------------------------------------------------------------
# Fix 5: timeout 1200 wrapper for remote claude command
# ---------------------------------------------------------------------------

class TestClaudeCodeTimeoutWrapper:
    @pytest.fixture
    def executor(self):
        from src.tools.executor import ToolExecutor
        config = MagicMock()
        config.hosts = {"desktop": MagicMock(address="10.0.0.2", ssh_user="root", os="linux")}
        config.ssh_key_path = "/app/.ssh/id_ed25519"
        config.ssh_known_hosts_path = "/app/.ssh/known_hosts"
        config.command_timeout_seconds = 30
        config.claude_code_host = "desktop"
        config.claude_code_user = "deploy"
        config.claude_code_dir = "/opt/project"
        config.memory_path = None
        return ToolExecutor(config)

    async def test_allow_edits_command_has_timeout(self, executor):
        """allow_edits=True path should include 'timeout 1200' in the SSH command."""
        call_count = [0]
        tmpdir = "/tmp/claude_code_abcd1234"

        async def mock_ssh(host, command, timeout, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # mktemp -d
                return (0, tmpdir)
            return (0, "done")

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock,
                    side_effect=mock_ssh) as mock_ssh_ref:
            await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "test",
                "allow_edits": True,
            })

        # Second call is the main claude execution (first is mktemp)
        second_call = mock_ssh_ref.call_args_list[1]
        cmd_arg = second_call[1].get("command", second_call[0][1] if len(second_call[0]) > 1 else "")
        assert "timeout 1200" in cmd_arg

    async def test_readonly_command_has_timeout(self, executor):
        """allow_edits=False (read-only) path should include 'timeout 1200' in the SSH command."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "analysis result")
            await executor._handle_claude_code({
                "host": "desktop",
                "working_directory": "/root/project",
                "prompt": "review this code",
            })

        first_call_cmd = mock_ssh.call_args_list[0][1].get("command") or mock_ssh.call_args_list[0][0][1]
        if not isinstance(first_call_cmd, str):
            first_call_cmd = str(mock_ssh.call_args_list[0])
        assert "timeout 1200" in first_call_cmd
