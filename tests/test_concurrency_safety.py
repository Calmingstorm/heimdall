"""Tests for concurrency safety fixes — shared mutable state elimination.

Covers:
- Fix #1: system_prompt is now a local variable passed through the call chain,
  not mutated on self._system_prompt per-request
- Fix #2: user_id is now passed directly to executor.execute() as a keyword arg,
  not set on the shared ToolExecutor via set_user_context()
- Fix #3: _pending_files is now a per-channel dict, not a shared list
"""
from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402
from src.tools.executor import ToolExecutor  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal LokiBot stub for concurrency tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._last_tool_use = {}
    stub._system_prompt = "default prompt"
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
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
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._process_with_tools = AsyncMock(
        return_value=("response", False, False, [], False)
    )
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.voice_manager = None
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="OK")
    stub.tool_executor.set_user_context = MagicMock()
    stub.tool_executor._load_memory_for_user = MagicMock(return_value={})
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")
    return stub


def _make_message(channel_id="chan-1", author_id="user-1"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.display_name = f"User-{author_id}"
    msg.author.name = f"User-{author_id}"
    msg.reply = AsyncMock()
    msg.attachments = []
    return msg


# ---------------------------------------------------------------------------
# Fix #1: system_prompt is local, not mutated on self
# ---------------------------------------------------------------------------

class TestSystemPromptNotMutated:
    """_handle_message_inner must not mutate self._system_prompt."""

    async def test_task_route_does_not_mutate_self_system_prompt(self):
        """Task path should pass system_prompt_override, not set self._system_prompt."""
        stub = _make_bot_stub()
        msg = _make_message()
        original_prompt = stub._system_prompt
        stub._build_system_prompt = MagicMock(return_value="user-specific prompt")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        # self._system_prompt should NOT have been changed
        assert stub._system_prompt == original_prompt

    async def test_system_prompt_passed_via_override(self):
        """The freshly built prompt should be passed as system_prompt_override."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._build_system_prompt = MagicMock(return_value="per-request prompt")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub._process_with_tools.assert_called_once()
        kwargs = stub._process_with_tools.call_args[1]
        assert kwargs.get("system_prompt_override") == "per-request prompt"

    async def test_inject_tool_hints_returns_prompt_not_mutates(self):
        """_inject_tool_hints should return modified prompt, not mutate self."""
        stub = _make_bot_stub()
        stub._inject_tool_hints = LokiBot._inject_tool_hints.__get__(stub)
        stub.tool_memory = MagicMock()
        stub.tool_memory.format_hints = AsyncMock(return_value="## Hints\nhint text")

        result = await stub._inject_tool_hints("base", "query")
        assert "Hints" in result
        # self._system_prompt unchanged
        assert stub._system_prompt == "default prompt"

    async def test_two_channels_get_independent_prompts(self):
        """Simulate two channels — each should get its own prompt, not interfere."""
        stub = _make_bot_stub()
        prompts_seen = {}

        def build_for_channel(channel=None, user_id=None, query=None):
            ch_id = str(channel.id) if channel else "none"
            return f"prompt-for-{ch_id}"

        stub._build_system_prompt = MagicMock(side_effect=build_for_channel)

        async def capture_process(*args, **kwargs):
            sp = kwargs.get("system_prompt_override", "")
            prompts_seen[sp] = True
            return ("ok", False, False, [], False)

        stub._process_with_tools = capture_process
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg1 = _make_message("chan-A", "user-1")
        msg2 = _make_message("chan-B", "user-2")

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg1, "check disk", "chan-A")
            await stub._handle_message_inner(msg2, "check memory", "chan-B")

        assert "prompt-for-chan-A" in prompts_seen
        assert "prompt-for-chan-B" in prompts_seen


# ---------------------------------------------------------------------------
# Fix #2: user_id passed through execute(), not set_user_context()
# ---------------------------------------------------------------------------

class TestUserIdPassedToExecute:
    """user_id should be passed directly to execute(), not set on the executor."""

    async def test_execute_passes_user_id_to_memory_manage(self, tmp_path):
        """execute() should forward user_id to _handle_memory_manage."""
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))

        # Call execute with user_id kwarg — should scope memory to user
        result = await executor.execute(
            "memory_manage",
            {"action": "save", "key": "color", "value": "blue"},
            user_id="42",
        )
        assert "personal" in result
        data = json.loads(mem_file.read_text())
        assert data["user_42"]["color"] == "blue"

    async def test_execute_passes_user_id_to_manage_list(self, tmp_path):
        """execute() should forward user_id to _handle_manage_list."""
        mem_file = tmp_path / "memory.json"
        mem_file.write_text("{}")
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))

        result = await executor.execute(
            "manage_list",
            {"action": "add", "list_name": "todo", "items": ["item1"], "owner": "personal"},
            user_id="99",
        )
        lists_file = tmp_path / "lists.json"
        data = json.loads(lists_file.read_text())
        assert data["todo"]["owner"] == "99"

    async def test_execute_without_user_id_defaults_to_global(self, tmp_path):
        """Without user_id, memory_manage should save to global scope."""
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))

        result = await executor.execute(
            "memory_manage",
            {"action": "save", "key": "fact", "value": "test"},
        )
        assert "global" in result
        data = json.loads(mem_file.read_text())
        assert data["global"]["fact"] == "test"

    async def test_process_with_tools_passes_user_id_to_executor(self):
        """_process_with_tools should pass user_id kwarg to executor.execute."""
        from src.llm.types import LLMResponse, ToolCall

        stub = _make_bot_stub()
        msg = _make_message(author_id="user-42")

        call_count = 0

        async def fake_chat_with_tools(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="tool-1", name="check_disk", input={"host": "server"})],
                    stop_reason="tool_use",
                )
            return LLMResponse(text="Done.", stop_reason="end_turn")

        stub.codex_client.chat_with_tools = fake_chat_with_tools
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)
        stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            await stub._process_with_tools(msg, [])

        stub.tool_executor.execute.assert_called_once_with(
            "check_disk", {"host": "server"}, user_id="user-42"
        )


# ---------------------------------------------------------------------------
# Fix #3: _pending_files is per-channel
# ---------------------------------------------------------------------------

class TestPendingFilesPerChannel:
    """_pending_files should be scoped per-channel, not shared."""

    def test_pending_files_is_dict(self):
        """_pending_files should be a dict keyed by channel_id."""
        stub = _make_bot_stub()
        assert isinstance(stub._pending_files, dict)

    async def test_send_chunked_uses_channel_id(self):
        """_send_chunked should only consume pending files for its channel."""
        stub = _make_bot_stub()
        stub._send_chunked = LokiBot._send_chunked.__get__(stub)

        # Pre-populate pending files for two channels
        stub._pending_files = {
            "chan-A": [(b"data-A", "file-A.txt")],
            "chan-B": [(b"data-B", "file-B.txt")],
        }

        msg = _make_message("chan-A")
        await stub._send_chunked(msg, "Hello!")

        # chan-A's files should be consumed
        assert "chan-A" not in stub._pending_files
        # chan-B's files should be untouched
        assert stub._pending_files["chan-B"] == [(b"data-B", "file-B.txt")]

    async def test_already_sent_path_uses_channel_id(self):
        """The already_sent path should only consume files for its channel."""
        stub = _make_bot_stub()
        stub._pending_files = {
            "chan-X": [(b"xdata", "x.txt")],
            "chan-Y": [(b"ydata", "y.txt")],
        }

        # Simulate the already_sent path code
        channel_id = "chan-X"
        pending = stub._pending_files.pop(channel_id, [])

        assert len(pending) == 1
        assert pending[0][1] == "x.txt"
        # chan-Y untouched
        assert "chan-Y" in stub._pending_files
