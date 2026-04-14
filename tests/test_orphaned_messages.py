"""Tests for orphaned user message cleanup on error paths.

When processing fails (budget exceeded, API error, generic exception, or
_process_with_tools returning is_error=True), the user message added at the
start of _handle_message_inner must be removed from session history so it
doesn't persist as an orphaned entry — wasting tokens on subsequent requests
and potentially confusing the model with consecutive user messages.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.sessions.manager import SessionManager, Message, Session  # noqa: E402
from src.discord.client import HeimdallBot  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session_mgr(tmp_dir: Path) -> SessionManager:
    return SessionManager(
        max_history=20,
        max_age_hours=1,
        persist_dir=str(tmp_dir / "sessions"),
    )


# ---------------------------------------------------------------------------
# SessionManager.remove_last_message unit tests
# ---------------------------------------------------------------------------

class TestRemoveLastMessage:
    """Unit tests for SessionManager.remove_last_message()."""

    def test_removes_matching_role(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "hello")
        result = session_mgr.remove_last_message("ch1", "user")
        assert result is True
        session = session_mgr.get_or_create("ch1")
        assert len(session.messages) == 0

    def test_does_not_remove_mismatched_role(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "hello")
        session_mgr.add_message("ch1", "assistant", "hi there")
        result = session_mgr.remove_last_message("ch1", "user")
        assert result is False
        session = session_mgr.get_or_create("ch1")
        assert len(session.messages) == 2

    def test_removes_only_last_message(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "first")
        session_mgr.add_message("ch1", "assistant", "response")
        session_mgr.add_message("ch1", "user", "second")
        result = session_mgr.remove_last_message("ch1", "user")
        assert result is True
        session = session_mgr.get_or_create("ch1")
        assert len(session.messages) == 2
        assert session.messages[-1].content == "response"

    def test_nonexistent_channel(self, session_mgr: SessionManager):
        result = session_mgr.remove_last_message("nonexistent", "user")
        assert result is False

    def test_empty_session(self, session_mgr: SessionManager):
        session_mgr.get_or_create("ch1")
        result = session_mgr.remove_last_message("ch1", "user")
        assert result is False

    def test_channel_isolation(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "ch1 msg")
        session_mgr.add_message("ch2", "user", "ch2 msg")
        session_mgr.remove_last_message("ch1", "user")
        # ch1 should be empty, ch2 should still have its message
        assert len(session_mgr.get_or_create("ch1").messages) == 0
        assert len(session_mgr.get_or_create("ch2").messages) == 1

    def test_preserves_earlier_messages(self, session_mgr: SessionManager):
        session_mgr.add_message("ch1", "user", "msg1")
        session_mgr.add_message("ch1", "assistant", "reply1")
        session_mgr.add_message("ch1", "user", "msg2")
        session_mgr.remove_last_message("ch1", "user")
        session = session_mgr.get_or_create("ch1")
        assert len(session.messages) == 2
        assert session.messages[0].content == "msg1"
        assert session.messages[1].content == "reply1"


# ---------------------------------------------------------------------------
# Helpers for _handle_message_inner tests
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Create a minimal HeimdallBot stub."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._system_prompt = "You are a bot."
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.max_tool_iterations_chat = 30
    stub.config.tools.max_tool_iterations_loop = 100
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex chat response")
    stub.codex_client.chat_with_tools = AsyncMock(return_value=MagicMock())
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub._build_system_prompt = MagicMock(return_value="system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._pending_files = {}
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    return stub


def _make_message(channel_id="chan-1", author_id="user-1"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.reply = AsyncMock()
    return msg


# ---------------------------------------------------------------------------
# Error path orphan cleanup in _handle_message_inner
# ---------------------------------------------------------------------------

class TestOrphanCleanupOnGenericException:
    """Generic exception saves error to history for checkpoint-save (Round 14)."""

    async def test_generic_exception_saves_error_to_history(self):
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("boom"))
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Error saved as sanitized marker (not removed) for checkpoint-save
        stub.sessions.remove_last_message.assert_not_called()
        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1
        # Sanitized marker instead of raw error text
        assert "error" in assistant_saves[0][0][2].lower()

    async def test_generic_exception_still_sends_error_to_user(self):
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("boom"))
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Inner except catches Codex exception and sends error via _send_chunked
        stub._send_chunked.assert_called_once()
        assert "boom" in str(stub._send_chunked.call_args)


class TestOrphanCleanupOnIsError:
    """is_error=True saves error to history for checkpoint-save (Round 14)."""

    async def test_is_error_true_saves_error_to_history(self):
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("API overloaded", False, True, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Error saved as sanitized marker (not removed) for checkpoint-save
        stub.sessions.remove_last_message.assert_not_called()
        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1
        # Sanitized marker: no tools used -> "[Previous request encountered an error before tool execution.]"
        assert "error" in assistant_saves[0][0][2].lower()

    async def test_is_error_true_triggers_save(self):
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("API overloaded", False, True, [], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub.sessions.save.assert_called()


class TestNoOrphanCleanupOnSuccess:
    """Successful responses should NOT trigger remove_last_message."""

    async def test_success_does_not_remove_user_message(self):
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("Disk is 42% full.", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub.sessions.remove_last_message.assert_not_called()

    async def test_success_saves_both_messages(self):
        stub = _make_bot_stub()
        msg = _make_message()

        stub._process_with_tools = AsyncMock(
            return_value=("Disk is 42% full.", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", "chan-1")

        add_calls = stub.sessions.add_message.call_args_list
        assert len(add_calls) == 2
        # Content is prefixed with display name: "[TestUser]: check disk"
        assert add_calls[0][0][0] == "chan-1"
        assert add_calls[0][0][1] == "user"
        assert "check disk" in add_calls[0][0][2]
        assert add_calls[1][0] == ("chan-1", "assistant", "Disk is 42% full.")

    async def test_codex_chat_success_does_not_remove_user_message(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Hey there!")
        # Guest tier forces the chat route (classifier removed)
        stub.permissions.is_guest = MagicMock(return_value=True)

        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "hey whats up", "chan-1")

        stub.sessions.remove_last_message.assert_not_called()

    async def test_codex_success_no_orphan(self):
        """When Codex succeeds on the task route, user message should be kept
        (not orphaned) and the assistant response saved."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Here you go!")

        stub._process_with_tools = AsyncMock(
            return_value=("Here you go!", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = HeimdallBot._handle_message_inner.__get__(stub)

        await stub._handle_message_inner(msg, "check disk", "chan-1")

        # Should NOT remove the user message since response succeeded
        stub.sessions.remove_last_message.assert_not_called()
        # Should save the assistant response
        add_calls = stub.sessions.add_message.call_args_list
        assert any(c[0][1] == "assistant" for c in add_calls)
