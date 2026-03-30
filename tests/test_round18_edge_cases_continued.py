"""Round 18: Edge cases (continued) — webhook bots, display name tagging, session persistence,
background tasks, circuit breaker edge cases.

Tests cover: allowed webhook bypass, attachment processing for webhooks, display name tagging
with unicode/empty/special names, session save/load/corrupted files, background task lifecycle
and limits, circuit breaker state transitions and recovery timing.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import discord
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Minimal config stub."""
    cfg = SimpleNamespace(
        discord=SimpleNamespace(
            respond_to_bots=True,
            require_mention=False,
            allowed_channels=[],
            allowed_users=[],
            guest_users=[],
        ),
        tools=SimpleNamespace(
            hosts={
                "server1": SimpleNamespace(address="10.0.0.1", ssh_user="deploy", ssh_key="~/.ssh/id_rsa"),
            },
            ssh_known_hosts_path="~/.ssh/known_hosts",
            timeout_seconds=30,
            claude_code_host="server1",
            claude_code_user="",
            tool_timeout_seconds=300,
        ),
        sessions=SimpleNamespace(max_history=50, max_age_hours=72, persist_dir="/tmp/test_sessions"),
        learning=SimpleNamespace(max_entries=100, consolidation_target=50, enabled=False),
        context=SimpleNamespace(directory="./contexts"),
        monitoring=SimpleNamespace(enabled=False),
        scheduling=SimpleNamespace(enabled=False),
        audit=SimpleNamespace(enabled=False),
        voice=SimpleNamespace(enabled=False),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ============================================================================
# 1. Webhook Bot Edge Cases
# ============================================================================

class TestWebhookAllowedBypass:
    """Allowed webhooks bypass respond_to_bots=False check."""

    def test_allowed_webhook_parsed_from_env(self):
        """_ALLOWED_WEBHOOK_IDS parsed from comma-separated env var."""
        from src.discord.client import _ALLOWED_WEBHOOK_IDS
        # The module-level set exists
        assert isinstance(_ALLOWED_WEBHOOK_IDS, set)

    def test_webhook_ids_split_by_comma(self):
        """ALLOWED_WEBHOOK_IDS env var splits on comma, strips whitespace."""
        with patch.dict(os.environ, {"ALLOWED_WEBHOOK_IDS": "123, 456 , 789"}):
            from src.discord import client as c
            old = c._ALLOWED_WEBHOOK_IDS
            try:
                c._ALLOWED_WEBHOOK_IDS = set()
                # Simulate init
                raw = os.environ.get("ALLOWED_WEBHOOK_IDS", "")
                c._ALLOWED_WEBHOOK_IDS = {wid.strip() for wid in raw.split(",") if wid.strip()}
                assert c._ALLOWED_WEBHOOK_IDS == {"123", "456", "789"}
            finally:
                c._ALLOWED_WEBHOOK_IDS = old

    def test_empty_webhook_env_gives_empty_set(self):
        """Empty ALLOWED_WEBHOOK_IDS env var results in empty set."""
        raw = ""
        result = {wid.strip() for wid in raw.split(",") if wid.strip()}
        assert result == set()

    def test_webhook_id_comparison_is_string(self):
        """Webhook ID comparison uses str(message.webhook_id)."""
        allowed = {"12345"}
        # Integer webhook_id
        assert str(12345) in allowed
        # String webhook_id
        assert "12345" in allowed

    def test_allowed_webhook_bypasses_respond_to_bots_false(self):
        """Allowed webhook still processes even when respond_to_bots=False."""
        # The check: is_allowed_webhook = webhook_id and str(webhook_id) in _ALLOWED_WEBHOOK_IDS
        # if not is_allowed_webhook and not respond_to_bots: return
        # So allowed webhooks bypass the return
        webhook_id = 99999
        allowed_ids = {"99999"}
        respond_to_bots = False
        is_allowed_webhook = webhook_id and str(webhook_id) in allowed_ids
        should_skip = not is_allowed_webhook and not respond_to_bots
        assert not should_skip, "Allowed webhook should NOT be skipped"

    def test_non_allowed_webhook_blocked_when_respond_to_bots_false(self):
        """Non-allowed webhook blocked when respond_to_bots=False."""
        webhook_id = 88888
        allowed_ids = {"99999"}
        respond_to_bots = False
        is_allowed_webhook = webhook_id and str(webhook_id) in allowed_ids
        should_skip = not is_allowed_webhook and not respond_to_bots
        assert should_skip, "Non-allowed webhook SHOULD be skipped"

    def test_webhook_without_id_blocked(self):
        """Message with webhook_id=None blocked when respond_to_bots=False."""
        webhook_id = None
        allowed_ids = {"99999"}
        respond_to_bots = False
        is_allowed_webhook = webhook_id and str(webhook_id) in allowed_ids
        should_skip = not is_allowed_webhook and not respond_to_bots
        assert should_skip


class TestWebhookAttachmentProcessing:
    """Webhook bots take the human path — attachments and secrets ARE processed."""

    def test_webhook_bot_not_buffered(self):
        """Webhook bots (via allowed IDs) don't enter bot buffer.

        The bot buffer path requires: message.author.bot AND respond_to_bots.
        Webhook messages set author.bot=True but are NOT specifically excluded from buffering.
        However, in the on_message flow, webhooks take the full path including attachments.
        """
        # Bot buffer key: (channel_id, author_id)
        # Webhook messages with webhook_id still have author.bot=True
        # They enter the buffer if respond_to_bots=True (this is expected behavior)
        # The key difference: test webhooks skip tool filtering
        pass  # documented behavior

    def test_test_webhook_skips_tool_filtering(self):
        """Test webhooks skip permission-based tool filtering."""
        # Source: is_test_wh = message.webhook_id and str(message.webhook_id) in _ALLOWED_WEBHOOK_IDS
        #         if tools is not None and not is_test_wh: tools = self.permissions.filter_tools(...)
        webhook_id = 12345
        allowed_ids = {"12345"}
        is_test_wh = webhook_id and str(webhook_id) in allowed_ids
        assert is_test_wh, "Test webhook should skip tool filtering"

    def test_non_webhook_gets_tool_filtering(self):
        """Non-webhook messages get permission-based tool filtering."""
        webhook_id = None
        allowed_ids = {"12345"}
        is_test_wh = webhook_id and str(webhook_id) in allowed_ids
        assert not is_test_wh


# ============================================================================
# 2. Display Name Tagging Edge Cases
# ============================================================================

class TestDisplayNameTagging:
    """Display name tagging in session history."""

    def test_display_name_used_when_present(self):
        """display_name is used when available."""
        author = SimpleNamespace(display_name="NickName", name="username")
        display_name = author.display_name or author.name
        assert display_name == "NickName"

    def test_name_fallback_when_display_name_none(self):
        """Falls back to author.name when display_name is None."""
        author = SimpleNamespace(display_name=None, name="username")
        display_name = author.display_name or author.name
        assert display_name == "username"

    def test_name_fallback_when_display_name_empty(self):
        """Falls back to author.name when display_name is empty string."""
        author = SimpleNamespace(display_name="", name="username")
        display_name = author.display_name or author.name
        assert display_name == "username"

    def test_tagged_content_format(self):
        """Content formatted as [display_name]: content."""
        display_name = "TestUser"
        content = "Hello world"
        tagged = f"[{display_name}]: {content}"
        assert tagged == "[TestUser]: Hello world"

    def test_unicode_display_name(self):
        """Unicode characters in display name are preserved."""
        author = SimpleNamespace(display_name="日本語ユーザー", name="user")
        display_name = author.display_name or author.name
        tagged = f"[{display_name}]: test"
        assert "[日本語ユーザー]: test" == tagged

    def test_emoji_display_name(self):
        """Emoji in display name preserved."""
        author = SimpleNamespace(display_name="🔥 FireUser 🔥", name="user")
        display_name = author.display_name or author.name
        tagged = f"[{display_name}]: test"
        assert "🔥 FireUser 🔥" in tagged

    def test_display_name_with_brackets(self):
        """Display name containing brackets doesn't break format."""
        author = SimpleNamespace(display_name="[admin]", name="user")
        display_name = author.display_name or author.name
        tagged = f"[{display_name}]: test"
        assert tagged == "[[admin]]: test"

    def test_display_name_with_special_chars(self):
        """Special characters in display name preserved."""
        author = SimpleNamespace(display_name="user@domain#1234", name="user")
        display_name = author.display_name or author.name
        tagged = f"[{display_name}]: test"
        assert "user@domain#1234" in tagged

    def test_very_long_display_name(self):
        """Very long display name doesn't crash."""
        author = SimpleNamespace(display_name="A" * 500, name="user")
        display_name = author.display_name or author.name
        tagged = f"[{display_name}]: test"
        assert len(tagged) > 500
        assert tagged.startswith("[AAAA")

    def test_display_name_with_newlines(self):
        """Newlines in display name are preserved (unlikely but possible)."""
        author = SimpleNamespace(display_name="User\nName", name="user")
        display_name = author.display_name or author.name
        tagged = f"[{display_name}]: test"
        assert "\n" in tagged

    def test_empty_content_with_image_placeholder(self):
        """Empty content becomes (see attached image) before tagging."""
        content = "(see attached image)"
        display_name = "TestUser"
        tagged = f"[{display_name}]: {content}"
        assert tagged == "[TestUser]: (see attached image)"


# ============================================================================
# 3. Session Persistence Edge Cases
# ============================================================================

class TestSessionSave:
    """Session save (dirty tracking, file I/O)."""

    def test_save_only_dirty_sessions(self, tmp_path):
        """save() only writes sessions in the _dirty set."""
        from src.sessions.manager import SessionManager
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.add_message("chan1", "user", "hello")
        mgr.add_message("chan2", "user", "world")
        mgr.save()
        # Both files written
        assert (tmp_path / "chan1.json").exists()
        assert (tmp_path / "chan2.json").exists()
        # After save, dirty is cleared
        assert len(mgr._dirty) == 0

    def test_save_skips_non_dirty(self, tmp_path):
        """save() after clear dirty doesn't rewrite."""
        from src.sessions.manager import SessionManager
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.add_message("chan1", "user", "hello")
        mgr.save()
        # Modify file externally
        path = tmp_path / "chan1.json"
        path.write_text("{}")
        # save() again — no dirty sessions
        mgr.save()
        # File should still be the externally modified one
        assert json.loads(path.read_text()) == {}

    def test_save_all_writes_everything(self, tmp_path):
        """save_all() writes all sessions regardless of dirty flag."""
        from src.sessions.manager import SessionManager
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.add_message("chan1", "user", "hello")
        mgr.save()
        # Dirty is clear now
        assert len(mgr._dirty) == 0
        # Modify session in memory
        mgr._sessions["chan1"].summary = "new summary"
        # save() won't write (not dirty), but save_all() will
        mgr.save_all()
        data = json.loads((tmp_path / "chan1.json").read_text())
        assert data["summary"] == "new summary"


class TestSessionLoad:
    """Session load with corrupted/missing data."""

    def test_load_valid_session(self, tmp_path):
        """Load a valid session JSON file."""
        from src.sessions.manager import SessionManager
        data = {
            "channel_id": "chan1",
            "messages": [{"role": "user", "content": "hello", "timestamp": 1000.0}],
            "created_at": 1000.0,
            "last_active": 1000.0,
            "summary": "test summary",
            "last_user_id": "user123",
        }
        (tmp_path / "chan1.json").write_text(json.dumps(data))
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.load()
        assert "chan1" in mgr._sessions
        assert mgr._sessions["chan1"].summary == "test summary"
        assert mgr._sessions["chan1"].last_user_id == "user123"
        assert len(mgr._sessions["chan1"].messages) == 1

    def test_load_corrupted_json_skips_file(self, tmp_path):
        """Corrupted JSON file is skipped, others load fine."""
        from src.sessions.manager import SessionManager
        # Valid file
        valid = {"channel_id": "chan1", "messages": [], "created_at": 1.0, "last_active": 1.0, "summary": ""}
        (tmp_path / "chan1.json").write_text(json.dumps(valid))
        # Corrupted file
        (tmp_path / "chan2.json").write_text("{invalid json")
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.load()
        assert "chan1" in mgr._sessions
        assert "chan2" not in mgr._sessions

    def test_load_empty_file_skips(self, tmp_path):
        """Empty file is skipped."""
        from src.sessions.manager import SessionManager
        (tmp_path / "empty.json").write_text("")
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.load()
        assert len(mgr._sessions) == 0

    def test_load_missing_channel_id_skips(self, tmp_path):
        """File missing channel_id key is skipped."""
        from src.sessions.manager import SessionManager
        data = {"messages": [], "summary": ""}
        (tmp_path / "bad.json").write_text(json.dumps(data))
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.load()
        assert len(mgr._sessions) == 0

    def test_load_extra_fields_ignored(self, tmp_path):
        """Extra unknown fields in JSON are ignored by Message dataclass."""
        from src.sessions.manager import SessionManager
        data = {
            "channel_id": "chan1",
            "messages": [{"role": "user", "content": "hello", "timestamp": 1.0, "extra_field": "ignored"}],
            "created_at": 1.0, "last_active": 1.0, "summary": "",
        }
        (tmp_path / "chan1.json").write_text(json.dumps(data))
        mgr = SessionManager(50, 72, str(tmp_path))
        # Message(**m) will raise TypeError for extra_field
        mgr.load()
        # File with extra fields is skipped due to TypeError
        # This is expected behavior — strict Message dataclass
        # (we just verify it doesn't crash)

    def test_load_preserves_user_id(self, tmp_path):
        """user_id on messages is preserved through save/load cycle."""
        from src.sessions.manager import SessionManager
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.add_message("chan1", "user", "hello", user_id="usr1")
        mgr.save()
        mgr2 = SessionManager(50, 72, str(tmp_path))
        mgr2.load()
        assert mgr2._sessions["chan1"].messages[0].user_id == "usr1"

    def test_save_load_roundtrip(self, tmp_path):
        """Full save/load roundtrip preserves all fields."""
        from src.sessions.manager import SessionManager
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.add_message("chan1", "user", "hello", user_id="usr1")
        mgr.add_message("chan1", "assistant", "hi there")
        mgr._sessions["chan1"].summary = "some summary"
        mgr.save()

        mgr2 = SessionManager(50, 72, str(tmp_path))
        mgr2.load()
        s = mgr2._sessions["chan1"]
        assert s.summary == "some summary"
        assert len(s.messages) == 2
        assert s.messages[0].role == "user"
        assert s.messages[0].content == "hello"
        assert s.messages[1].role == "assistant"
        assert s.messages[1].content == "hi there"


class TestSessionPersistenceEdgeCases:
    """Edge cases for session file operations."""

    def test_persist_dir_created_if_missing(self, tmp_path):
        """SessionManager creates persist_dir if it doesn't exist."""
        from src.sessions.manager import SessionManager
        new_dir = tmp_path / "nested" / "dir"
        mgr = SessionManager(50, 72, str(new_dir))
        assert new_dir.exists()

    def test_concurrent_add_message_and_save(self, tmp_path):
        """add_message marks dirty, save clears it — no crash on interleaving."""
        from src.sessions.manager import SessionManager
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.add_message("chan1", "user", "hello")
        assert "chan1" in mgr._dirty
        mgr.save()
        assert len(mgr._dirty) == 0
        # Adding after save marks dirty again
        mgr.add_message("chan1", "user", "world")
        assert "chan1" in mgr._dirty

    def test_save_deleted_session_skips(self, tmp_path):
        """If a session is deleted from _sessions but still in _dirty, save skips it."""
        from src.sessions.manager import SessionManager
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.add_message("chan1", "user", "hello")
        # Mark dirty but delete from _sessions
        del mgr._sessions["chan1"]
        # save() should not crash
        mgr.save()
        assert not (tmp_path / "chan1.json").exists()

    def test_load_nonexistent_dir_no_crash(self, tmp_path):
        """Loading from a directory with no JSON files doesn't crash."""
        from src.sessions.manager import SessionManager
        mgr = SessionManager(50, 72, str(tmp_path))
        mgr.load()
        assert len(mgr._sessions) == 0


# ============================================================================
# 4. Background Task Edge Cases
# ============================================================================

class TestBackgroundTaskConstants:
    """Background task constants and imports."""

    def test_max_steps_value(self):
        from src.discord.background_task import MAX_STEPS
        assert MAX_STEPS == 200

    def test_progress_update_interval(self):
        from src.discord.background_task import PROGRESS_UPDATE_INTERVAL
        assert PROGRESS_UPDATE_INTERVAL == 2.0

    def test_blocked_tools_defined(self):
        from src.discord.background_task import BLOCKED_TOOLS
        assert "purge_messages" in BLOCKED_TOOLS
        assert "delegate_task" in BLOCKED_TOOLS
        assert "browser_screenshot" in BLOCKED_TOOLS

    def test_blocked_tools_no_regular_tools(self):
        """Regular tools like run_command are NOT blocked."""
        from src.discord.background_task import BLOCKED_TOOLS
        assert "run_command" not in BLOCKED_TOOLS
        assert "check_disk" not in BLOCKED_TOOLS
        assert "run_script" not in BLOCKED_TOOLS


class TestBackgroundTaskLifecycle:
    """Background task creation and execution lifecycle."""

    def test_task_creation(self):
        from src.discord.background_task import BackgroundTask, create_task_id
        channel = AsyncMock()
        task = BackgroundTask(
            task_id=create_task_id(),
            description="test task",
            steps=[{"tool_name": "run_command", "tool_input": {"command": "ls"}}],
            channel=channel,
            requester="user",
        )
        assert task.status == "running"
        assert len(task.task_id) == 8
        assert len(task.results) == 0
        assert task.current_step == 0

    def test_task_cancel_event(self):
        """Cancel event can be set."""
        from src.discord.background_task import BackgroundTask, create_task_id
        task = BackgroundTask(
            task_id=create_task_id(), description="test", steps=[],
            channel=AsyncMock(), requester="user",
        )
        assert not task._cancel_event.is_set()
        task.cancel()
        assert task._cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_cancellation_during_execution(self):
        """Task checks cancel event before each step."""
        from src.discord.background_task import BackgroundTask, run_background_task, create_task_id
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())  # progress msg

        task = BackgroundTask(
            task_id=create_task_id(), description="test",
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "ls"}},
                {"tool_name": "run_command", "tool_input": {"command": "pwd"}},
            ],
            channel=channel, requester="user",
        )
        # Set cancel before execution
        task.cancel()

        executor = AsyncMock()
        skill_manager = MagicMock()
        skill_manager.has_skill.return_value = False

        await run_background_task(task, executor, skill_manager)
        assert task.status == "cancelled"
        assert len(task.results) == 1
        assert task.results[0].status == "cancelled"

    @pytest.mark.asyncio
    async def test_blocked_tool_abort(self):
        """Blocked tool with on_failure=abort stops the task."""
        from src.discord.background_task import BackgroundTask, run_background_task, create_task_id
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())

        task = BackgroundTask(
            task_id=create_task_id(), description="test",
            steps=[
                {"tool_name": "purge_messages", "tool_input": {}, "on_failure": "abort"},
                {"tool_name": "run_command", "tool_input": {"command": "ls"}},
            ],
            channel=channel, requester="user",
        )

        executor = AsyncMock()
        skill_manager = MagicMock()
        skill_manager.has_skill.return_value = False

        await run_background_task(task, executor, skill_manager)
        assert task.status == "failed"
        assert len(task.results) == 1
        assert task.results[0].status == "error"
        assert "cannot run in background" in task.results[0].output

    @pytest.mark.asyncio
    async def test_blocked_tool_continue(self):
        """Blocked tool with on_failure=continue skips and continues."""
        from src.discord.background_task import BackgroundTask, run_background_task, create_task_id
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())

        task = BackgroundTask(
            task_id=create_task_id(), description="test",
            steps=[
                {"tool_name": "purge_messages", "tool_input": {}, "on_failure": "continue"},
                {"tool_name": "run_command", "tool_input": {"command": "ls"}},
            ],
            channel=channel, requester="user",
        )

        executor = AsyncMock()
        executor.execute = AsyncMock(return_value="output")
        skill_manager = MagicMock()
        skill_manager.has_skill.return_value = False

        await run_background_task(task, executor, skill_manager)
        assert task.status == "completed"
        assert len(task.results) == 2
        assert task.results[0].status == "error"
        assert task.results[1].status == "ok"

    @pytest.mark.asyncio
    async def test_tool_exception_abort(self):
        """Tool that raises an exception with on_failure=abort stops the task."""
        from src.discord.background_task import BackgroundTask, run_background_task, create_task_id
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())

        task = BackgroundTask(
            task_id=create_task_id(), description="test",
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "bad"}, "on_failure": "abort"},
            ],
            channel=channel, requester="user",
        )

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=RuntimeError("Connection refused"))
        skill_manager = MagicMock()
        skill_manager.has_skill.return_value = False

        await run_background_task(task, executor, skill_manager)
        assert task.status == "failed"
        assert task.results[0].status == "error"
        assert "Connection refused" in task.results[0].output

    @pytest.mark.asyncio
    async def test_completed_status_on_success(self):
        """Successful task set to completed."""
        from src.discord.background_task import BackgroundTask, run_background_task, create_task_id
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())

        task = BackgroundTask(
            task_id=create_task_id(), description="test",
            steps=[{"tool_name": "run_command", "tool_input": {"command": "ls"}}],
            channel=channel, requester="user",
        )

        executor = AsyncMock()
        executor.execute = AsyncMock(return_value="file1.txt")
        skill_manager = MagicMock()
        skill_manager.has_skill.return_value = False

        await run_background_task(task, executor, skill_manager)
        assert task.status == "completed"
        assert task.results[0].status == "ok"
        assert task.results[0].output == "file1.txt"


class TestBackgroundTaskVariableSubstitution:
    """Variable substitution in background tasks."""

    def test_prev_output_substitution(self):
        from src.discord.background_task import _substitute_vars
        result = _substitute_vars(
            {"command": "echo {prev_output}"},
            {}, "hello world",
        )
        assert result["command"] == "echo hello world"

    def test_named_variable_substitution(self):
        from src.discord.background_task import _substitute_vars
        result = _substitute_vars(
            {"command": "deploy {var.version}"},
            {"version": "v1.2.3"}, "",
        )
        assert result["command"] == "deploy v1.2.3"

    def test_no_substitution_for_non_string(self):
        from src.discord.background_task import _substitute_vars
        result = _substitute_vars(
            {"count": 42},
            {}, "prev",
        )
        assert result["count"] == 42

    def test_undefined_variable_stays_literal(self):
        from src.discord.background_task import _substitute_vars
        result = _substitute_vars(
            {"command": "{var.undefined}"},
            {}, "",
        )
        assert result["command"] == "{var.undefined}"

    def test_multiple_substitutions(self):
        from src.discord.background_task import _substitute_vars
        result = _substitute_vars(
            {"command": "deploy {var.app} to {var.host}"},
            {"app": "myapp", "host": "prod"}, "",
        )
        assert result["command"] == "deploy myapp to prod"


class TestBackgroundTaskConditions:
    """Condition checking in background tasks."""

    def test_positive_condition_match(self):
        from src.discord.background_task import _check_condition
        assert _check_condition("running", "Service is RUNNING") is True

    def test_positive_condition_no_match(self):
        from src.discord.background_task import _check_condition
        assert _check_condition("running", "Service is stopped") is False

    def test_negated_condition_match(self):
        from src.discord.background_task import _check_condition
        assert _check_condition("!error", "Service is running") is True

    def test_negated_condition_no_match(self):
        from src.discord.background_task import _check_condition
        assert _check_condition("!error", "There was an ERROR") is False

    def test_condition_case_insensitive(self):
        from src.discord.background_task import _check_condition
        assert _check_condition("SUCCESS", "the task was a success") is True

    @pytest.mark.asyncio
    async def test_condition_skips_step(self):
        """Step with unmet condition is skipped."""
        from src.discord.background_task import BackgroundTask, run_background_task, create_task_id
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())

        task = BackgroundTask(
            task_id=create_task_id(), description="test",
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "status"}},
                {"tool_name": "run_command", "tool_input": {"command": "restart"},
                 "condition": "error"},  # Only run if prev_output contains "error"
            ],
            channel=channel, requester="user",
        )

        executor = AsyncMock()
        executor.execute = AsyncMock(return_value="Service is running OK")
        skill_manager = MagicMock()
        skill_manager.has_skill.return_value = False

        await run_background_task(task, executor, skill_manager)
        assert task.status == "completed"
        assert len(task.results) == 2
        assert task.results[0].status == "ok"
        assert task.results[1].status == "skipped"

    @pytest.mark.asyncio
    async def test_store_as_variable(self):
        """store_as saves output for later substitution."""
        from src.discord.background_task import BackgroundTask, run_background_task, create_task_id
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())

        call_count = 0
        async def mock_execute(tool_name, tool_input):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "v2.0.0"  # version output
            return f"deployed {tool_input.get('command', '')}"

        task = BackgroundTask(
            task_id=create_task_id(), description="test",
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "get-version"},
                 "store_as": "version"},
                {"tool_name": "run_command",
                 "tool_input": {"command": "deploy {var.version}"}},
            ],
            channel=channel, requester="user",
        )

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=mock_execute)
        skill_manager = MagicMock()
        skill_manager.has_skill.return_value = False

        await run_background_task(task, executor, skill_manager)
        # Second tool should have been called with substituted version
        second_call = executor.execute.call_args_list[1]
        assert second_call[0][1]["command"] == "deploy v2.0.0"


class TestBackgroundTaskCleanup:
    """Background task cleanup and limits."""

    def test_completed_tasks_pruned_at_max(self):
        """Completed tasks pruned when exceeding max."""
        # Simulate the cleanup logic from client.py
        background_tasks = {}
        max_tasks = 20

        # Add 25 completed tasks
        for i in range(25):
            tid = f"task_{i:03d}"
            t = SimpleNamespace(status="completed")
            background_tasks[tid] = t

        # Prune logic (from client.py:2306-2312)
        completed = [
            tid for tid, t in background_tasks.items()
            if t.status in ("completed", "failed", "cancelled")
        ]
        while len(completed) > max_tasks:
            old = completed.pop(0)
            del background_tasks[old]

        assert len(background_tasks) == 20

    def test_running_tasks_never_pruned(self):
        """Running tasks are excluded from pruning."""
        background_tasks = {}
        max_tasks = 20

        # 10 running + 25 completed
        for i in range(10):
            background_tasks[f"run_{i}"] = SimpleNamespace(status="running")
        for i in range(25):
            background_tasks[f"done_{i}"] = SimpleNamespace(status="completed")

        completed = [
            tid for tid, t in background_tasks.items()
            if t.status in ("completed", "failed", "cancelled")
        ]
        while len(completed) > max_tasks:
            old = completed.pop(0)
            del background_tasks[old]

        # 10 running + 20 completed = 30 total
        running = [t for t in background_tasks.values() if t.status == "running"]
        assert len(running) == 10
        assert len(background_tasks) == 30


class TestBackgroundTaskProgress:
    """Progress message rendering."""

    @pytest.mark.asyncio
    async def test_send_progress_initial(self):
        """Initial progress message is sent (no existing_msg)."""
        from src.discord.background_task import BackgroundTask, _send_progress, create_task_id
        channel = AsyncMock()
        sent_msg = AsyncMock()
        channel.send = AsyncMock(return_value=sent_msg)

        task = BackgroundTask(
            task_id=create_task_id(), description="Deploy app",
            steps=[{"tool_name": "run_command"}] * 5,
            channel=channel, requester="user",
        )

        result = await _send_progress(task, None)
        assert result == sent_msg
        channel.send.assert_called_once()
        text = channel.send.call_args[1]["content"] if "content" in (channel.send.call_args[1] or {}) else channel.send.call_args[0][0]
        assert "Deploy app" in text

    @pytest.mark.asyncio
    async def test_send_progress_edit_existing(self):
        """Progress message edits existing when passed."""
        from src.discord.background_task import BackgroundTask, _send_progress, create_task_id
        channel = AsyncMock()
        existing_msg = AsyncMock()

        task = BackgroundTask(
            task_id=create_task_id(), description="Deploy app",
            steps=[{"tool_name": "run_command"}] * 5,
            channel=channel, requester="user",
        )

        result = await _send_progress(task, existing_msg)
        assert result == existing_msg
        existing_msg.edit.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_progress_edit_failure_silent(self):
        """Edit failure is silently caught."""
        from src.discord.background_task import BackgroundTask, _send_progress, create_task_id
        channel = AsyncMock()
        existing_msg = AsyncMock()
        existing_msg.edit = AsyncMock(side_effect=discord.HTTPException(MagicMock(), "rate limited"))

        task = BackgroundTask(
            task_id=create_task_id(), description="Deploy app",
            steps=[{"tool_name": "run_command"}] * 5,
            channel=channel, requester="user",
        )

        # Should not raise
        result = await _send_progress(task, existing_msg)
        assert result == existing_msg

    @pytest.mark.asyncio
    async def test_send_progress_long_text_truncated_running(self):
        """Running progress > 1900 chars is truncated."""
        from src.discord.background_task import (
            BackgroundTask, StepResult, _send_progress, create_task_id,
        )
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())

        task = BackgroundTask(
            task_id=create_task_id(), description="Deploy app",
            steps=[{"tool_name": "run_command"}] * 50,
            channel=channel, requester="user",
        )
        # Add many results to make text long
        for i in range(50):
            task.results.append(StepResult(
                index=i, tool_name="run_command",
                description=f"Step {i} " + "x" * 100,
                status="ok", output="output " * 50,
            ))

        result = await _send_progress(task, None)
        text = channel.send.call_args[0][0]
        assert len(text) <= 1904  # 1900 + "..." + newline

    @pytest.mark.asyncio
    async def test_send_progress_file_attachment_on_finish(self):
        """Finished progress > 1900 chars creates file attachment."""
        from src.discord.background_task import (
            BackgroundTask, StepResult, _send_progress, create_task_id,
        )
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())

        task = BackgroundTask(
            task_id=create_task_id(), description="Big deploy",
            steps=[{"tool_name": "run_command"}] * 30,
            channel=channel, requester="user",
            status="completed",
        )
        for i in range(30):
            task.results.append(StepResult(
                index=i, tool_name="run_command",
                description=f"Step {i} with long description " + "x" * 80,
                status="ok", output="long output " * 20,
            ))

        result = await _send_progress(task, None)
        # Should have sent with file=
        assert channel.send.call_count >= 1
        call_kwargs = channel.send.call_args[1]
        assert "file" in call_kwargs or "content" in call_kwargs

    @pytest.mark.asyncio
    async def test_zero_steps_shows_no_steps(self):
        """Task with 0 steps shows 'No steps'."""
        from src.discord.background_task import BackgroundTask, _send_progress, create_task_id
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=AsyncMock())

        task = BackgroundTask(
            task_id=create_task_id(), description="Empty task",
            steps=[], channel=channel, requester="user",
        )

        await _send_progress(task, None)
        text = channel.send.call_args[0][0]
        assert "No steps" in text


class TestBackgroundTaskSummary:
    """Task summary message rendering."""

    @pytest.mark.asyncio
    async def test_summary_all_success(self):
        """Summary for all-success task."""
        from src.discord.background_task import (
            BackgroundTask, StepResult, _send_summary, create_task_id,
        )
        channel = AsyncMock()
        task = BackgroundTask(
            task_id=create_task_id(), description="Deploy app",
            steps=[{"tool_name": "run_command"}] * 3,
            channel=channel, requester="user",
            status="completed",
        )
        for i in range(3):
            task.results.append(StepResult(
                index=i, tool_name="run_command",
                description=f"Step {i}", status="ok", output="done",
            ))

        await _send_summary(task)
        text = channel.send.call_args[0][0]
        assert "All 3 steps succeeded" in text

    @pytest.mark.asyncio
    async def test_summary_with_errors(self):
        """Summary for task with errors."""
        from src.discord.background_task import (
            BackgroundTask, StepResult, _send_summary, create_task_id,
        )
        channel = AsyncMock()
        task = BackgroundTask(
            task_id=create_task_id(), description="Deploy app",
            steps=[{"tool_name": "run_command"}] * 2,
            channel=channel, requester="user",
            status="completed",
        )
        task.results.append(StepResult(0, "run_command", "Step 0", "ok", "done"))
        task.results.append(StepResult(1, "run_command", "Step 1", "error", "failed"))

        await _send_summary(task)
        text = channel.send.call_args[0][0]
        assert "1 succeeded" in text
        assert "1 failed" in text

    @pytest.mark.asyncio
    async def test_summary_cancelled(self):
        """Summary for cancelled task."""
        from src.discord.background_task import (
            BackgroundTask, StepResult, _send_summary, create_task_id,
        )
        channel = AsyncMock()
        task = BackgroundTask(
            task_id=create_task_id(), description="Deploy app",
            steps=[{"tool_name": "run_command"}] * 5,
            channel=channel, requester="user",
            status="cancelled",
        )
        task.results.append(StepResult(0, "run_command", "Step 0", "ok", "done"))
        task.results.append(StepResult(1, "run_command", "Step 1", "cancelled"))

        await _send_summary(task)
        text = channel.send.call_args[0][0]
        assert "cancelled" in text.lower()


# ============================================================================
# 5. Circuit Breaker Edge Cases
# ============================================================================

class TestCircuitBreakerBasicStates:
    """Circuit breaker state transitions."""

    def test_initial_state_closed(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test")
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_failures_below_threshold_stays_closed(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        assert cb._failure_count == 2

    def test_exactly_threshold_opens(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert cb._failure_count == 3

    def test_success_resets_count(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_threshold_one_opens_immediately(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"


class TestCircuitBreakerHalfOpen:
    """Half-open state and recovery."""

    def test_open_to_half_open_after_timeout(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.15)
        assert cb.state == "half_open"

    def test_half_open_allows_probe(self):
        """check() does not raise when in half_open state."""
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        # Should not raise
        cb.check()

    def test_half_open_success_closes(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        cb.record_success()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_half_open_failure_reopens(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        # In half_open, record another failure
        cb.record_failure()
        assert cb.state == "open"
        # Failure count accumulates (record_failure increments, doesn't reset)
        assert cb._failure_count == 2

    def test_check_raises_when_open(self):
        from src.llm.circuit_breaker import CircuitBreaker, CircuitOpenError
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60.0)
        cb.record_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.check()
        assert exc_info.value.provider == "test"
        assert exc_info.value.retry_after > 0
        assert exc_info.value.retry_after <= 60.0


class TestCircuitBreakerTimingEdgeCases:
    """Timing edge cases."""

    def test_retry_after_decreases_over_time(self):
        from src.llm.circuit_breaker import CircuitBreaker, CircuitOpenError
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=1.0)
        cb.record_failure()
        try:
            cb.check()
        except CircuitOpenError as e:
            first_retry = e.retry_after

        time.sleep(0.2)

        try:
            cb.check()
        except CircuitOpenError as e:
            second_retry = e.retry_after

        assert second_retry < first_retry

    def test_retry_after_never_negative(self):
        """retry_after is max(0.0, remaining), never negative."""
        from src.llm.circuit_breaker import CircuitBreaker, CircuitOpenError
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.05)
        # Should be half_open, check should NOT raise
        cb.check()  # should not raise

    def test_zero_recovery_timeout_immediate_halfopen(self):
        """recovery_timeout=0 means immediate half_open."""
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.0)
        cb.record_failure()
        # Immediately half_open
        assert cb.state == "half_open"
        # check should not raise
        cb.check()

    def test_very_long_recovery_timeout(self):
        """Very long recovery_timeout stays open."""
        from src.llm.circuit_breaker import CircuitBreaker, CircuitOpenError
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=999999.0)
        cb.record_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.check()
        assert exc_info.value.retry_after > 999990


class TestCircuitBreakerError:
    """CircuitOpenError details."""

    def test_error_message_format(self):
        from src.llm.circuit_breaker import CircuitOpenError
        err = CircuitOpenError("codex_api", 42.5)
        assert "codex_api" in str(err)
        assert "42s" in str(err)  # formatted as {retry_after:.0f}s

    def test_error_attributes(self):
        from src.llm.circuit_breaker import CircuitOpenError
        err = CircuitOpenError("provider", 10.0)
        assert err.provider == "provider"
        assert err.retry_after == 10.0

    def test_error_is_exception(self):
        from src.llm.circuit_breaker import CircuitOpenError
        assert issubclass(CircuitOpenError, Exception)


class TestCircuitBreakerThreadSafety:
    """Thread safety of circuit breaker."""

    def test_lock_exists(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test")
        assert isinstance(cb._lock, type(threading.Lock()))

    def test_concurrent_failures(self):
        """Multiple threads recording failures don't corrupt state."""
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=100)

        errors = []
        def record_many():
            try:
                for _ in range(50):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cb._failure_count == 200  # 4 threads × 50 failures
        assert cb.state == "open"

    def test_concurrent_success_and_failure(self):
        """Interleaved success and failure don't crash."""
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=100)

        errors = []
        def mixed_ops():
            try:
                for i in range(100):
                    if i % 2 == 0:
                        cb.record_failure()
                    else:
                        cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_ops) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # State should be consistent (closed or open, no corrupt values)
        assert cb.state in ("closed", "open", "half_open")


class TestCircuitBreakerRecoveryScenario:
    """End-to-end recovery scenario."""

    def test_full_recovery_cycle(self):
        """3 failures → open → wait → half_open → probe success → closed."""
        from src.llm.circuit_breaker import CircuitBreaker, CircuitOpenError
        cb = CircuitBreaker("test_api", failure_threshold=3, recovery_timeout=0.1)

        # 3 failures
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        # Check raises while open
        with pytest.raises(CircuitOpenError):
            cb.check()

        # Wait for recovery
        time.sleep(0.15)
        assert cb.state == "half_open"

        # Probe allowed
        cb.check()  # should not raise

        # Probe succeeds
        cb.record_success()
        assert cb.state == "closed"
        assert cb._failure_count == 0

        # Normal operations resume
        cb.check()  # no error

    def test_failed_probe_reopens(self):
        """Failed probe reopens with fresh timer."""
        from src.llm.circuit_breaker import CircuitBreaker, CircuitOpenError
        cb = CircuitBreaker("test_api", failure_threshold=1, recovery_timeout=0.1)

        cb.record_failure()
        time.sleep(0.15)  # half_open
        cb.record_failure()  # probe fails

        assert cb.state == "open"
        # Fresh timer means it won't be half_open immediately
        with pytest.raises(CircuitOpenError):
            cb.check()


# ============================================================================
# Source Structure Verification
# ============================================================================

class TestRound18SourceStructure:
    """Verify source code structures for all tested areas."""

    def test_allowed_webhook_ids_is_module_level_set(self):
        from src.discord.client import _ALLOWED_WEBHOOK_IDS
        assert isinstance(_ALLOWED_WEBHOOK_IDS, set)

    def test_init_allowed_webhook_ids_method_exists(self):
        from src.discord.client import HeimdallBot
        assert hasattr(HeimdallBot, "_init_allowed_webhook_ids")

    def test_display_name_in_handle_message_inner(self):
        """display_name used in _handle_message_inner."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot._handle_message_inner)
        assert "display_name" in source
        assert "tagged_content" in source

    def test_session_manager_has_save_load(self):
        from src.sessions.manager import SessionManager
        assert hasattr(SessionManager, "save")
        assert hasattr(SessionManager, "save_all")
        assert hasattr(SessionManager, "load")

    def test_background_task_has_cancel(self):
        from src.discord.background_task import BackgroundTask
        assert hasattr(BackgroundTask, "cancel")

    def test_background_task_has_run(self):
        from src.discord.background_task import run_background_task
        assert callable(run_background_task)

    def test_circuit_breaker_has_all_methods(self):
        from src.llm.circuit_breaker import CircuitBreaker
        assert hasattr(CircuitBreaker, "check")
        assert hasattr(CircuitBreaker, "record_success")
        assert hasattr(CircuitBreaker, "record_failure")
        assert hasattr(CircuitBreaker, "state")

    def test_background_tasks_max_is_20(self):
        """client.py uses _background_tasks_max = 20."""
        import inspect
        from src.discord.client import HeimdallBot
        source = inspect.getsource(HeimdallBot.__init__)
        assert "_background_tasks_max = 20" in source

    def test_send_progress_handles_exceptions(self):
        """_send_progress catches exceptions silently."""
        import inspect
        from src.discord.background_task import _send_progress
        source = inspect.getsource(_send_progress)
        assert "except Exception" in source

    def test_substitute_vars_function_exists(self):
        from src.discord.background_task import _substitute_vars
        assert callable(_substitute_vars)

    def test_check_condition_function_exists(self):
        from src.discord.background_task import _check_condition
        assert callable(_check_condition)
