"""Round 32 — Skill fix-up tests.

Tests for bug fixes: file limit constant separation, cache invalidation
for install_skill in loop iteration path, and private IP test corrections.
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import pytest

from src.config.schema import ToolsConfig
from src.tools.skill_context import (
    MAX_SKILL_FILES,
    MAX_SKILL_MESSAGES,
    ResourceTracker,
    SkillContext,
)
from src.tools.executor import ToolExecutor


# ---------------------------------------------------------------------------
# MAX_SKILL_FILES constant
# ---------------------------------------------------------------------------


class TestFileLimit:
    """File limit should use its own constant, not MAX_SKILL_MESSAGES."""

    def test_max_skill_files_exists(self):
        assert MAX_SKILL_FILES == 10

    def test_max_skill_files_separate_from_messages(self):
        """File and message limits are independent constants."""
        from src.tools import skill_context
        assert hasattr(skill_context, "MAX_SKILL_FILES")
        assert hasattr(skill_context, "MAX_SKILL_MESSAGES")

    async def test_file_limit_uses_max_skill_files(self, tools_config):
        """post_file() should check against MAX_SKILL_FILES, not MAX_SKILL_MESSAGES."""
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(files_sent=MAX_SKILL_FILES)
        callback = AsyncMock()
        ctx = SkillContext(
            executor, "test_skill",
            file_callback=callback,
            resource_tracker=tracker,
        )
        await ctx.post_file(b"data", "file.txt")
        callback.assert_not_called()

    async def test_file_limit_independent_of_messages(self, tools_config):
        """Sending max messages should NOT block file posting."""
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(messages_sent=MAX_SKILL_MESSAGES, files_sent=0)
        callback = AsyncMock()
        ctx = SkillContext(
            executor, "test_skill",
            file_callback=callback,
            resource_tracker=tracker,
        )
        await ctx.post_file(b"data", "file.txt")
        callback.assert_called_once()

    async def test_message_limit_independent_of_files(self, tools_config):
        """Sending max files should NOT block message posting."""
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(files_sent=MAX_SKILL_FILES, messages_sent=0)
        callback = AsyncMock()
        ctx = SkillContext(
            executor, "test_skill",
            message_callback=callback,
            resource_tracker=tracker,
        )
        await ctx.post_message("hello")
        callback.assert_called_once_with("hello")

    async def test_file_under_limit_allowed(self, tools_config):
        """Files under the limit should go through."""
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(files_sent=MAX_SKILL_FILES - 1)
        callback = AsyncMock()
        ctx = SkillContext(
            executor, "test_skill",
            file_callback=callback,
            resource_tracker=tracker,
        )
        await ctx.post_file(b"data", "file.txt", "caption")
        callback.assert_called_once_with(b"data", "file.txt", "caption")


# ---------------------------------------------------------------------------
# Cache invalidation for install_skill in loop iteration
# ---------------------------------------------------------------------------


class TestInstallSkillCacheInvalidation:
    """install_skill should invalidate caches in the loop iteration tool handler."""

    def test_install_skill_in_loop_cache_invalidation(self):
        """The loop iteration tool handler should invalidate caches for install_skill."""
        import src.discord.client as client_mod
        src = inspect.getsource(client_mod.HeimdallBot._run_loop_iteration)
        assert "install_skill" in src

    def test_all_skill_crud_tools_in_cache_check(self):
        """All skill CRUD tools should be in the loop cache invalidation set."""
        import src.discord.client as client_mod
        src = inspect.getsource(client_mod.HeimdallBot._run_loop_iteration)
        for tool in ("create_skill", "edit_skill", "delete_skill",
                      "enable_skill", "disable_skill", "install_skill"):
            assert tool in src, f"{tool} missing from loop iteration cache invalidation"


# ---------------------------------------------------------------------------
# Source code checks for correct constant usage
# ---------------------------------------------------------------------------


class TestSourceCorrectness:
    """Verify source code uses the correct constants."""

    def test_post_file_uses_max_skill_files(self):
        """post_file should reference MAX_SKILL_FILES, not MAX_SKILL_MESSAGES."""
        from src.tools import skill_context
        src = inspect.getsource(skill_context.SkillContext.post_file)
        assert "MAX_SKILL_FILES" in src
        assert "MAX_SKILL_MESSAGES" not in src

    def test_post_message_uses_max_skill_messages(self):
        """post_message should reference MAX_SKILL_MESSAGES."""
        from src.tools import skill_context
        src = inspect.getsource(skill_context.SkillContext.post_message)
        assert "MAX_SKILL_MESSAGES" in src

    def test_no_personal_ips_in_sandbox_tests(self):
        """Round 30 sandbox tests should not use 192.168.1.x addresses."""
        from pathlib import Path
        content = Path("tests/test_round30_skill_sandbox.py").read_text()
        assert "192.168.1" not in content
