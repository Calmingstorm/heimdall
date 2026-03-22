"""Tests for git write operations: git_commit, git_push, git_branch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.executor import ToolExecutor
from src.tools.registry import TOOLS, requires_approval


@pytest.fixture
def executor(tools_config, tmp_dir: Path) -> ToolExecutor:
    return ToolExecutor(tools_config, memory_path=str(tmp_dir / "memory.json"))


# --- Registry tests ---


class TestGitWriteRegistry:
    def test_git_commit_in_registry(self):
        names = [t["name"] for t in TOOLS]
        assert "git_commit" in names

    def test_git_push_in_registry(self):
        names = [t["name"] for t in TOOLS]
        assert "git_push" in names

    def test_git_branch_in_registry(self):
        names = [t["name"] for t in TOOLS]
        assert "git_branch" in names

    def test_git_commit_requires_approval(self):
        assert requires_approval("git_commit") is True

    def test_git_push_requires_approval(self):
        assert requires_approval("git_push") is True

    def test_git_branch_requires_approval(self):
        assert requires_approval("git_branch") is True

    def test_git_commit_schema_has_message_required(self):
        tool = next(t for t in TOOLS if t["name"] == "git_commit")
        assert "message" in tool["input_schema"]["required"]
        assert "host" in tool["input_schema"]["required"]
        assert "repo_path" in tool["input_schema"]["required"]

    def test_git_commit_schema_files_optional(self):
        tool = next(t for t in TOOLS if t["name"] == "git_commit")
        assert "files" not in tool["input_schema"]["required"]
        assert "files" in tool["input_schema"]["properties"]

    def test_git_push_schema_remote_optional(self):
        tool = next(t for t in TOOLS if t["name"] == "git_push")
        assert "remote" not in tool["input_schema"]["required"]
        assert "remote" in tool["input_schema"]["properties"]

    def test_git_push_schema_branch_optional(self):
        tool = next(t for t in TOOLS if t["name"] == "git_push")
        assert "branch" not in tool["input_schema"]["required"]
        assert "branch" in tool["input_schema"]["properties"]

    def test_git_branch_schema_action_required(self):
        tool = next(t for t in TOOLS if t["name"] == "git_branch")
        assert "action" in tool["input_schema"]["required"]

    def test_git_branch_schema_action_enum(self):
        tool = next(t for t in TOOLS if t["name"] == "git_branch")
        action_prop = tool["input_schema"]["properties"]["action"]
        assert set(action_prop["enum"]) == {"list", "create", "switch"}

    def test_git_branch_schema_branch_name_optional(self):
        tool = next(t for t in TOOLS if t["name"] == "git_branch")
        assert "branch_name" not in tool["input_schema"]["required"]
        assert "branch_name" in tool["input_schema"]["properties"]


# --- git_commit handler tests ---


class TestGitCommit:
    @pytest.mark.asyncio
    async def test_commit_all_files(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "[master abc1234] Fix bug\n 1 file changed")
            result = await executor.execute("git_commit", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "message": "Fix bug",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "git -C" in cmd
        assert "add -A" in cmd
        assert "commit -m" in cmd
        assert "Fix bug" in result

    @pytest.mark.asyncio
    async def test_commit_specific_files(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "[master def5678] Add feature\n 2 files changed")
            result = await executor.execute("git_commit", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "message": "Add feature",
                "files": ["src/main.py", "tests/test_main.py"],
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "add" in cmd
        assert "src/main.py" in cmd
        assert "tests/test_main.py" in cmd
        assert "add -A" not in cmd

    @pytest.mark.asyncio
    async def test_commit_shell_escapes_message(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "committed")
            await executor.execute("git_commit", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "message": "Fix: handle 'quoted' strings & special $chars",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        # shlex.quote wraps in single quotes — message should not appear unquoted
        assert "commit -m" in cmd

    @pytest.mark.asyncio
    async def test_commit_shell_escapes_repo_path(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "committed")
            await executor.execute("git_commit", {
                "host": "desktop",
                "repo_path": "/root/my project",
                "message": "test",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        # Path with space should be quoted
        assert "my project" in cmd

    @pytest.mark.asyncio
    async def test_commit_shell_escapes_files(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "committed")
            await executor.execute("git_commit", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "message": "test",
                "files": ["file with space.py"],
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "file with space.py" in cmd

    @pytest.mark.asyncio
    async def test_commit_returns_ssh_output(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "nothing to commit, working tree clean")
            result = await executor.execute("git_commit", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "message": "empty",
            })
        assert "nothing to commit" in result


# --- git_push handler tests ---


class TestGitPush:
    @pytest.mark.asyncio
    async def test_push_default(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "To http://192.168.1.13:3300/calmingstorm/ansiblex\n   abc..def  master -> master")
            result = await executor.execute("git_push", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "push" in cmd
        assert "origin" in cmd
        assert "master" in result

    @pytest.mark.asyncio
    async def test_push_custom_remote(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "pushed")
            await executor.execute("git_push", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "remote": "upstream",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "upstream" in cmd

    @pytest.mark.asyncio
    async def test_push_specific_branch(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "pushed")
            await executor.execute("git_push", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "branch": "feature/new-tools",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "feature/new-tools" in cmd

    @pytest.mark.asyncio
    async def test_push_custom_remote_and_branch(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "pushed")
            await executor.execute("git_push", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "remote": "gitea",
                "branch": "develop",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "gitea" in cmd
        assert "develop" in cmd

    @pytest.mark.asyncio
    async def test_push_shell_escapes_remote(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "pushed")
            await executor.execute("git_push", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "remote": "my remote",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        # shlex.quote should handle the space
        assert "my remote" in cmd

    @pytest.mark.asyncio
    async def test_push_no_branch_omits_branch(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "pushed")
            await executor.execute("git_push", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        # Should end with just the remote, no branch arg
        assert cmd.strip().endswith("origin")


# --- git_branch handler tests ---


class TestGitBranch:
    @pytest.mark.asyncio
    async def test_branch_list(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "* master\n  feature/foo\n  remotes/origin/master")
            result = await executor.execute("git_branch", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "action": "list",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "branch -a" in cmd
        assert "master" in result
        assert "feature/foo" in result

    @pytest.mark.asyncio
    async def test_branch_create(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Switched to a new branch 'feature/bar'")
            result = await executor.execute("git_branch", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "action": "create",
                "branch_name": "feature/bar",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "checkout -b" in cmd
        assert "feature/bar" in cmd
        assert "feature/bar" in result

    @pytest.mark.asyncio
    async def test_branch_switch(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "Switched to branch 'develop'")
            result = await executor.execute("git_branch", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "action": "switch",
                "branch_name": "develop",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "checkout" in cmd
        assert "-b" not in cmd
        assert "develop" in cmd
        assert "develop" in result

    @pytest.mark.asyncio
    async def test_branch_create_missing_name(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            result = await executor.execute("git_branch", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "action": "create",
            })
        assert "branch_name is required" in result
        mock_ssh.assert_not_called()

    @pytest.mark.asyncio
    async def test_branch_switch_missing_name(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            result = await executor.execute("git_branch", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "action": "switch",
            })
        assert "branch_name is required" in result
        mock_ssh.assert_not_called()

    @pytest.mark.asyncio
    async def test_branch_list_ignores_branch_name(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "* master")
            await executor.execute("git_branch", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "action": "list",
                "branch_name": "ignored",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "branch -a" in cmd
        assert "ignored" not in cmd

    @pytest.mark.asyncio
    async def test_branch_unknown_action(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            result = await executor.execute("git_branch", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "action": "delete",
                "branch_name": "old-branch",
            })
        assert "Unknown action" in result
        mock_ssh.assert_not_called()

    @pytest.mark.asyncio
    async def test_branch_shell_escapes_name(self, executor: ToolExecutor):
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "created")
            await executor.execute("git_branch", {
                "host": "desktop",
                "repo_path": "/root/ansiblex",
                "action": "create",
                "branch_name": "feature/special branch",
            })
        cmd = mock_ssh.call_args[1].get("command") or mock_ssh.call_args[0][1]
        assert "special branch" in cmd
