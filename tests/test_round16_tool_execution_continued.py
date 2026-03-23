"""Round 16: Tool execution hardening (continued).

Tests:
1. Discord-native tools (purge, generate_file, post_file, schedule, knowledge, skills)
2. Tool loop iteration (multi-step: Codex → tool → result → Codex → tool → final)
3. Concurrent tool execution within a single tool loop iteration
4. Tool output scrubbing (secrets removed before LLM sees output)
5. Tool timeout handling (per-tool timeouts with audit logging)
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    LokiBot,
    MAX_TOOL_ITERATIONS,
    TOOL_OUTPUT_MAX_CHARS,
    truncate_tool_output,
)
from src.llm.types import LLMResponse, ToolCall  # noqa: E402
from src.llm.secret_scrubber import scrub_output_secrets  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal LokiBot stub for _process_with_tools and Discord-native tools."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "You are a bot."
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.codex_client = MagicMock()
    stub.codex_client.chat_with_tools = AsyncMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="OK")
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[{"name": "test"}])
    stub._build_system_prompt = MagicMock(return_value="You are a bot.")
    stub._build_tool_progress_embed = MagicMock(return_value=MagicMock())
    stub._build_partial_completion_report = MagicMock(return_value="")
    stub._pending_files = {}
    stub.sessions = MagicMock()
    stub.sessions.reset = MagicMock()
    stub.sessions.search_history = AsyncMock(return_value=[])
    stub.scheduler = MagicMock()
    stub.browser_manager = None
    stub._knowledge_store = None
    stub._embedder = None
    stub._background_tasks = {}
    stub._background_tasks_max = 10
    # Bind real methods
    stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)
    stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)
    stub._handle_purge = LokiBot._handle_purge.__get__(stub)
    stub._handle_generate_file = LokiBot._handle_generate_file.__get__(stub)
    stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)
    stub._handle_schedule_task = LokiBot._handle_schedule_task.__get__(stub)
    stub._handle_list_schedules = LokiBot._handle_list_schedules.__get__(stub)
    stub._handle_delete_schedule = LokiBot._handle_delete_schedule.__get__(stub)
    stub._handle_parse_time = LokiBot._handle_parse_time.__get__(stub)
    stub._handle_search_history = LokiBot._handle_search_history.__get__(stub)
    stub._handle_search_knowledge = LokiBot._handle_search_knowledge.__get__(stub)
    stub._handle_list_knowledge = LokiBot._handle_list_knowledge.__get__(stub)
    stub._handle_delete_knowledge = LokiBot._handle_delete_knowledge.__get__(stub)
    stub._handle_ingest_document = LokiBot._handle_ingest_document.__get__(stub)
    stub._handle_set_permission = LokiBot._handle_set_permission.__get__(stub)
    stub._handle_search_audit = LokiBot._handle_search_audit.__get__(stub)
    stub._handle_create_digest = LokiBot._handle_create_digest.__get__(stub)
    stub._handle_delegate_task = LokiBot._handle_delegate_task.__get__(stub)
    stub._handle_list_tasks = LokiBot._handle_list_tasks.__get__(stub)
    stub._handle_cancel_task = LokiBot._handle_cancel_task.__get__(stub)
    return stub


def _make_message(channel_id="chan-1", author_id="user-1", author_bot=False, webhook_id=None):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock(return_value=MagicMock())
    msg.channel.purge = AsyncMock(return_value=[MagicMock() for _ in range(5)])
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.bot = author_bot
    msg.author.__str__ = lambda self: "TestUser#1234"
    msg.webhook_id = webhook_id
    msg.reply = AsyncMock()
    return msg


def _tool_response(tool_calls, text=""):
    return LLMResponse(text=text, tool_calls=tool_calls, stop_reason="tool_use")


def _text_response(text="Done."):
    return LLMResponse(text=text, stop_reason="end_turn")


def _codex_tool_then_text(tool_calls, final_text="Done."):
    return [_tool_response(tool_calls), _text_response(final_text)]


# ===========================================================================
# 1. Discord-native tool handlers
# ===========================================================================


class TestPurgeHandler:
    """Tests for _handle_purge (purge_messages tool)."""

    async def test_purge_deletes_messages_and_resets_session(self):
        stub = _make_bot_stub()
        msg = _make_message()
        deleted = [MagicMock() for _ in range(10)]
        msg.channel.purge = AsyncMock(return_value=deleted)

        result = await stub._handle_purge(msg, {"count": 10})

        assert "Deleted 10 messages" in result
        assert "reset conversation history" in result
        stub.sessions.reset.assert_called_once_with(str(msg.channel.id))

    async def test_purge_caps_at_500(self):
        stub = _make_bot_stub()
        msg = _make_message()

        await stub._handle_purge(msg, {"count": 9999})
        msg.channel.purge.assert_called_once_with(limit=500)

    async def test_purge_defaults_to_100(self):
        stub = _make_bot_stub()
        msg = _make_message()

        await stub._handle_purge(msg, {})
        msg.channel.purge.assert_called_once_with(limit=100)

    async def test_purge_forbidden_error(self):
        import discord
        stub = _make_bot_stub()
        msg = _make_message()
        msg.channel.purge = AsyncMock(
            side_effect=discord.Forbidden(MagicMock(status=403), "no perms")
        )

        result = await stub._handle_purge(msg, {"count": 5})
        assert "permission" in result.lower()

    async def test_purge_through_tool_loop(self):
        """purge_messages dispatched correctly in _process_with_tools."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="purge_messages", input={"count": 3})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc], "Purged.")
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "purge messages"}]
            )

        assert is_error is False
        assert "purge_messages" in tools_used


class TestGenerateFileHandler:
    """Tests for _handle_generate_file (generate_file tool)."""

    async def test_generate_file_posts_attachment(self):
        stub = _make_bot_stub()
        msg = _make_message()

        result = await stub._handle_generate_file(
            msg, {"filename": "output.py", "content": "print('hello')", "caption": "Here's the file"}
        )

        assert "output.py" in result
        assert "bytes" in result
        msg.channel.send.assert_called_once()
        call_kwargs = msg.channel.send.call_args
        assert call_kwargs[1].get("content") == "Here's the file" or \
               call_kwargs.kwargs.get("content") == "Here's the file"

    async def test_generate_file_default_filename(self):
        stub = _make_bot_stub()
        msg = _make_message()

        result = await stub._handle_generate_file(msg, {"content": "data"})
        assert "output.txt" in result

    async def test_generate_file_empty_content(self):
        stub = _make_bot_stub()
        msg = _make_message()

        result = await stub._handle_generate_file(msg, {})
        assert "output.txt" in result
        assert "0 bytes" in result

    async def test_generate_file_through_tool_loop(self):
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="generate_file",
                      input={"filename": "test.txt", "content": "hello"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc], "File posted.")
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "generate a file"}]
            )

        assert "generate_file" in tools_used


class TestPostFileHandler:
    """Tests for _handle_post_file (post_file tool)."""

    async def test_post_file_missing_host_or_path(self):
        stub = _make_bot_stub()
        msg = _make_message()

        result = await stub._handle_post_file(msg, {"host": "server"})
        assert "required" in result.lower()

        result = await stub._handle_post_file(msg, {"path": "/tmp/f"})
        assert "required" in result.lower()

    async def test_post_file_unknown_host(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.tool_executor._resolve_host = MagicMock(return_value=None)

        result = await stub._handle_post_file(
            msg, {"host": "badhost", "path": "/tmp/file.txt"}
        )
        assert "Unknown" in result or "disallowed" in result

    async def test_post_file_success(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.tool_executor._resolve_host = MagicMock(
            return_value=("10.0.0.1", "root", "linux")
        )
        file_content = b"file data here"
        b64 = base64.b64encode(file_content)

        with patch("asyncio.create_subprocess_exec") as mock_proc:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b64, b""))
            proc.returncode = 0
            mock_proc.return_value = proc

            result = await stub._handle_post_file(
                msg, {"host": "server", "path": "/tmp/test.txt", "caption": "Here"}
            )

        assert "test.txt" in result
        assert "KB" in result or "bytes" in result.lower()

    async def test_post_file_too_large(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.tool_executor._resolve_host = MagicMock(
            return_value=("10.0.0.1", "root", "linux")
        )
        # 26 MB file (over 25 MB limit)
        large_content = b"x" * (26 * 1024 * 1024)
        b64 = base64.b64encode(large_content)

        with patch("asyncio.create_subprocess_exec") as mock_proc:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b64, b""))
            proc.returncode = 0
            mock_proc.return_value = proc

            result = await stub._handle_post_file(
                msg, {"host": "server", "path": "/tmp/big.bin"}
            )

        assert "too large" in result.lower() or "25 MB" in result


class TestScheduleHandlers:
    """Tests for schedule_task, list_schedules, delete_schedule, parse_time."""

    async def test_schedule_task_cron(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.scheduler.add = MagicMock(return_value={
            "id": "sched-1", "description": "Check disk", "cron": "0 * * * *",
            "next_run": "2026-03-22T10:00:00",
        })

        result = stub._handle_schedule_task(msg, {
            "description": "Check disk", "action": "reminder",
            "cron": "0 * * * *",
        })

        assert "sched-1" in result
        assert "recurring" in result
        assert "Check disk" in result

    async def test_schedule_task_one_time(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.scheduler.add = MagicMock(return_value={
            "id": "sched-2", "description": "Reminder",
            "next_run": "2026-03-22T15:00:00",
        })

        result = stub._handle_schedule_task(msg, {
            "description": "Reminder", "action": "reminder",
            "run_at": "2026-03-22T15:00:00",
        })

        assert "one-time" in result

    async def test_schedule_task_with_trigger(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.scheduler.add = MagicMock(return_value={
            "id": "sched-3", "description": "On push",
            "trigger": {"event": "push", "repo": "loki"},
        })

        result = stub._handle_schedule_task(msg, {
            "description": "On push", "trigger": {"event": "push", "repo": "loki"},
        })

        assert "webhook-triggered" in result.lower() or "trigger" in result.lower()

    async def test_schedule_task_error(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.scheduler.add = MagicMock(side_effect=ValueError("bad cron"))

        result = stub._handle_schedule_task(msg, {"cron": "bad"})
        assert "Failed" in result

    async def test_list_schedules_empty(self):
        stub = _make_bot_stub()
        stub.scheduler.list_all = MagicMock(return_value=[])

        result = stub._handle_list_schedules()
        assert "No scheduled" in result

    async def test_list_schedules_with_entries(self):
        stub = _make_bot_stub()
        stub.scheduler.list_all = MagicMock(return_value=[
            {"id": "s1", "description": "Disk check", "cron": "*/5 * * * *",
             "next_run": "2026-03-22T10:05:00", "last_run": "2026-03-22T10:00:00"},
            {"id": "s2", "description": "Backup", "next_run": "2026-03-23",
             "last_run": "never"},
        ])

        result = stub._handle_list_schedules()
        assert "s1" in result
        assert "s2" in result
        assert "Disk check" in result

    async def test_delete_schedule_found(self):
        stub = _make_bot_stub()
        stub.scheduler.delete = MagicMock(return_value=True)

        result = stub._handle_delete_schedule({"schedule_id": "s1"})
        assert "Deleted" in result

    async def test_delete_schedule_not_found(self):
        stub = _make_bot_stub()
        stub.scheduler.delete = MagicMock(return_value=False)

        result = stub._handle_delete_schedule({"schedule_id": "nope"})
        assert "not found" in result

    async def test_parse_time_success(self):
        stub = _make_bot_stub()

        with patch("src.discord.client.parse_time", create=True):
            from src.tools.time_parser import parse_time
            with patch("src.tools.time_parser.parse_time", return_value="2026-03-22T15:00:00Z"):
                result = stub._handle_parse_time({"expression": "in 2 hours"})

        assert "2026-03-22" in result or "Parsed" in result

    async def test_parse_time_empty(self):
        stub = _make_bot_stub()
        result = stub._handle_parse_time({})
        assert "required" in result.lower()


class TestKnowledgeHandlers:
    """Tests for search_knowledge, ingest_document, list_knowledge, delete_knowledge."""

    async def test_search_knowledge_not_available(self):
        stub = _make_bot_stub()
        stub._knowledge_store = None

        result = await stub._handle_search_knowledge({"query": "test"})
        assert "not available" in result.lower()

    async def test_search_knowledge_empty_query(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._embedder = MagicMock()

        result = await stub._handle_search_knowledge({})
        assert "required" in result.lower()

    async def test_search_knowledge_with_results(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.search_hybrid = AsyncMock(return_value=[
            {"source": "doc1.md", "score": 0.95, "content": "This is the answer"},
            {"source": "doc2.md", "score": 0.82, "content": "Another result"},
        ])
        stub._embedder = MagicMock()

        result = await stub._handle_search_knowledge({"query": "test query"})
        assert "doc1.md" in result
        assert "doc2.md" in result
        assert "0.95" in result

    async def test_search_knowledge_no_results(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.search_hybrid = AsyncMock(return_value=[])
        stub._embedder = MagicMock()

        result = await stub._handle_search_knowledge({"query": "nothing"})
        assert "No knowledge" in result

    async def test_search_knowledge_limit_capped_at_10(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.search_hybrid = AsyncMock(return_value=[])
        stub._embedder = MagicMock()

        await stub._handle_search_knowledge({"query": "x", "limit": 50})
        call_args = stub._knowledge_store.search_hybrid.call_args
        assert call_args.kwargs.get("limit", call_args[0][2] if len(call_args[0]) > 2 else None) <= 10

    async def test_ingest_document_success(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.ingest = AsyncMock(return_value=5)
        stub._embedder = MagicMock()

        result = await stub._handle_ingest_document(
            {"source": "readme.md", "content": "# Hello"}, "user1"
        )
        assert "5 chunks" in result
        assert "readme.md" in result

    async def test_ingest_document_missing_fields(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._embedder = MagicMock()

        result = await stub._handle_ingest_document({"source": ""}, "user1")
        assert "required" in result.lower()

    async def test_list_knowledge_empty(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.list_sources = MagicMock(return_value=[])

        result = stub._handle_list_knowledge()
        assert "empty" in result.lower()

    async def test_list_knowledge_with_sources(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.list_sources = MagicMock(return_value=[
            {"source": "api.md", "chunks": 12, "uploader": "admin", "ingested_at": "2026-03-22T10:00:00"},
        ])

        result = stub._handle_list_knowledge()
        assert "api.md" in result
        assert "12 chunks" in result

    async def test_delete_knowledge_success(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.delete_source = MagicMock(return_value=5)

        result = stub._handle_delete_knowledge({"source": "old.md"})
        assert "Deleted" in result
        assert "5 chunks" in result

    async def test_delete_knowledge_not_found(self):
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.delete_source = MagicMock(return_value=0)

        result = stub._handle_delete_knowledge({"source": "nope"})
        assert "No document" in result


class TestSkillHandlers:
    """Tests for create_skill, edit_skill, delete_skill, list_skills in tool loop."""

    async def test_list_skills_empty(self):
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="list_skills", input={})
        stub.skill_manager.list_skills = MagicMock(return_value=[])
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc], "No skills.")
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "list skills"}]
            )

        assert "list_skills" in tools_used

    async def test_list_skills_with_skills(self):
        stub = _make_bot_stub()
        msg = _make_message()

        stub.skill_manager.list_skills = MagicMock(return_value=[
            {"name": "weather", "description": "Get weather"},
            {"name": "calc", "description": "Calculator"},
        ])

        tc = ToolCall(id="tc-1", name="list_skills", input={})

        captured = []
        call_count = 0
        responses = _codex_tool_then_text([tc], "Listed.")

        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            r = responses[call_count]
            call_count += 1
            return r

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "list skills"}]
            )

        # Check that the tool result contains skill info
        tool_result_msg = captured[1][-1]
        result_content = tool_result_msg["content"][0]["content"]
        assert "weather" in result_content
        assert "calc" in result_content

    async def test_create_skill_rebuilds_system_prompt(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.skill_manager.create_skill = MagicMock(return_value="Skill created.")

        tc = ToolCall(id="tc-1", name="create_skill",
                      input={"name": "test_skill", "code": "def run(): pass"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc], "Created.")
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "create a skill"}]
            )

        # _build_system_prompt should have been called to rebuild after skill creation
        stub._build_system_prompt.assert_called()

    async def test_custom_skill_dispatched_via_skill_manager(self):
        stub = _make_bot_stub()
        msg = _make_message()
        # First call: has_skill returns False for known tools, True for custom_skill
        stub.skill_manager.has_skill = MagicMock(side_effect=lambda n: n == "my_custom_skill")
        stub.skill_manager.execute = AsyncMock(return_value="Skill executed!")

        tc = ToolCall(id="tc-1", name="my_custom_skill", input={"arg": "val"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc], "Done.")
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            _, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "run my skill"}]
            )

        assert "my_custom_skill" in tools_used
        stub.skill_manager.execute.assert_called_once()


class TestPermissionAndAuditHandlers:
    """Tests for set_permission and search_audit."""

    async def test_set_permission_admin_only(self):
        stub = _make_bot_stub()
        stub.permissions.is_admin = MagicMock(return_value=False)

        result = stub._handle_set_permission("user-1", {"user_id": "user-2", "tier": "admin"})
        assert "denied" in result.lower()

    async def test_set_permission_success(self):
        stub = _make_bot_stub()
        stub.permissions.is_admin = MagicMock(return_value=True)
        stub.permissions.set_tier = MagicMock()

        result = stub._handle_set_permission("admin-1", {"user_id": "user-2", "tier": "operator"})
        assert "operator" in result

    async def test_search_audit_no_results(self):
        stub = _make_bot_stub()
        stub.audit.search = AsyncMock(return_value=[])

        result = await stub._handle_search_audit({"tool_name": "check_disk"})
        assert "No audit" in result

    async def test_search_audit_with_results(self):
        stub = _make_bot_stub()
        stub.audit.search = AsyncMock(return_value=[
            {"timestamp": "2026-03-22T10:00:00Z", "tool_name": "run_command",
             "user_name": "admin", "approved": True, "execution_time_ms": 150,
             "result_summary": "OK", "error": None},
        ])

        result = await stub._handle_search_audit({"tool_name": "run_command"})
        assert "run_command" in result
        assert "admin" in result


class TestDelegateAndTaskHandlers:
    """Tests for delegate_task, list_tasks, cancel_task."""

    async def test_delegate_task_no_steps(self):
        stub = _make_bot_stub()
        msg = _make_message()

        result = await stub._handle_delegate_task(msg, {"description": "test"})
        assert "No steps" in result

    async def test_delegate_task_too_many_steps(self):
        stub = _make_bot_stub()
        msg = _make_message()

        steps = [{"tool_name": f"tool_{i}"} for i in range(200)]
        with patch("src.discord.client.MAX_STEPS", 50):
            result = await stub._handle_delegate_task(
                msg, {"description": "big task", "steps": steps}
            )
        assert "Too many" in result

    async def test_delegate_task_invalid_step(self):
        stub = _make_bot_stub()
        msg = _make_message()

        result = await stub._handle_delegate_task(
            msg, {"description": "test", "steps": [{"no_tool": True}]}
        )
        assert "must have" in result

    async def test_list_tasks_empty(self):
        stub = _make_bot_stub()
        result = stub._handle_list_tasks()
        assert "No background" in result

    async def test_list_tasks_with_tasks(self):
        stub = _make_bot_stub()
        task = MagicMock()
        task.task_id = "t-1"
        task.description = "Backup"
        task.status = "running"
        task.steps = [MagicMock(), MagicMock()]
        task.results = [MagicMock(status="ok")]
        stub._background_tasks = {"t-1": task}

        result = stub._handle_list_tasks()
        assert "t-1" in result
        assert "Backup" in result

    async def test_list_tasks_detail_view(self):
        stub = _make_bot_stub()
        task = MagicMock()
        task.task_id = "t-1"
        task.description = "Deploy"
        task.status = "completed"
        task.steps = [MagicMock()]
        r = MagicMock()
        r.status = "ok"
        r.index = 0
        r.description = "Step 1"
        r.elapsed_ms = 200
        r.output = "Deployed."
        task.results = [r]
        stub._background_tasks = {"t-1": task}

        result = stub._handle_list_tasks({"task_id": "t-1"})
        assert "Deploy" in result
        assert "Deployed" in result

    async def test_cancel_task_not_found(self):
        stub = _make_bot_stub()
        result = stub._handle_cancel_task({"task_id": "nope"})
        assert "No task" in result

    async def test_cancel_task_not_running(self):
        stub = _make_bot_stub()
        task = MagicMock()
        task.status = "completed"
        stub._background_tasks = {"t-1": task}

        result = stub._handle_cancel_task({"task_id": "t-1"})
        assert "not running" in result

    async def test_cancel_task_success(self):
        stub = _make_bot_stub()
        task = MagicMock()
        task.status = "running"
        stub._background_tasks = {"t-1": task}

        result = stub._handle_cancel_task({"task_id": "t-1"})
        assert "Cancellation" in result
        task.cancel.assert_called_once()


class TestCreateDigestHandler:
    """Tests for create_digest."""

    async def test_create_digest_success(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.scheduler.add = MagicMock(return_value={
            "id": "d-1", "description": "Daily Digest",
            "next_run": "2026-03-23T08:00:00",
        })

        result = stub._handle_create_digest(msg, {"cron": "0 8 * * *"})
        assert "d-1" in result
        assert "Digest" in result or "digest" in result

    async def test_create_digest_error(self):
        stub = _make_bot_stub()
        msg = _make_message()
        stub.scheduler.add = MagicMock(side_effect=ValueError("bad cron"))

        result = stub._handle_create_digest(msg, {"cron": "invalid"})
        assert "Failed" in result


# ===========================================================================
# 2. Tool loop iteration (multi-step)
# ===========================================================================


class TestToolLoopMultiStep:
    """Tool loop: Codex → tool1 → result → Codex → tool2 → result → final text."""

    async def test_two_step_tool_chain(self):
        """Codex calls tool1, gets result, calls tool2, gets result, then responds."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="tc-1", name="check_disk", input={"host": "server"})
        tc2 = ToolCall(id="tc-2", name="check_memory", input={"host": "server"})

        call_count = 0
        async def _multi_step(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response([tc1], "Checking disk first.")
            elif call_count == 2:
                return _tool_response([tc2], "Now checking memory.")
            else:
                return _text_response("Disk 42%, memory 3.2GB free.")

        stub.codex_client.chat_with_tools = _multi_step

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "check server health"}]
            )

        assert text == "Disk 42%, memory 3.2GB free."
        assert is_error is False
        assert "check_disk" in tools_used
        assert "check_memory" in tools_used
        assert call_count == 3

    async def test_three_step_chain_message_growth(self):
        """Messages grow correctly over 3 iterations: each adds assistant + user."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="tc-1", name="run_command", input={"command": "ls"})
        tc2 = ToolCall(id="tc-2", name="run_command", input={"command": "pwd"})
        tc3 = ToolCall(id="tc-3", name="run_command", input={"command": "whoami"})

        captured_msgs = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured_msgs.append(len(messages))
            call_count += 1
            if call_count == 1:
                return _tool_response([tc1])
            elif call_count == 2:
                return _tool_response([tc2])
            elif call_count == 3:
                return _tool_response([tc3])
            return _text_response("Done.")

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "run commands"}]
            )

        # Message count should grow: each iteration adds assistant + user (tool result)
        assert captured_msgs[0] < captured_msgs[1] < captured_msgs[2] < captured_msgs[3]

    async def test_max_iterations_guard(self):
        """Tool loop stops after MAX_TOOL_ITERATIONS with error."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-loop", name="run_command", input={"command": "loop"})
        # Always return a tool call — never a final text
        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_response([tc])
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "infinite loop"}]
            )

        assert is_error is True
        assert "too many" in text.lower() or "simpler" in text.lower()
        assert len(tools_used) == MAX_TOOL_ITERATIONS

    async def test_tools_used_tracks_all_iterations(self):
        """tools_used accumulates across all iterations."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="tc-1", name="check_disk", input={})
        tc2 = ToolCall(id="tc-2", name="check_memory", input={})

        call_count = 0
        async def _steps(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response([tc1])
            elif call_count == 2:
                return _tool_response([tc2])
            return _text_response("Done")

        stub.codex_client.chat_with_tools = _steps

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            _, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "check all"}]
            )

        assert tools_used == ["check_disk", "check_memory"]

    async def test_tool_error_continues_loop(self):
        """If a tool raises an exception, error is captured and loop continues."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="run_command", input={"command": "fail"})
        stub.tool_executor.execute = AsyncMock(side_effect=RuntimeError("connection refused"))

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc], "Command failed.")
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, is_error, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "run command"}]
            )

        assert is_error is False  # The loop completed normally (LLM responded after error)
        assert "run_command" in tools_used

    async def test_tool_result_format_in_messages(self):
        """Tool results are sent back as user messages with tool_result format."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-42", name="check_disk", input={"host": "server"})

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append([m for m in messages])
            call_count += 1
            if call_count == 1:
                return _tool_response([tc])
            return _text_response("Done")

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "check disk"}]
            )

        # Second call should have tool_result in last user message
        second_msgs = captured[1]
        tool_result_msg = second_msgs[-1]
        assert tool_result_msg["role"] == "user"
        results = tool_result_msg["content"]
        assert isinstance(results, list)
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "tc-42"


# ===========================================================================
# 3. Concurrent tool execution
# ===========================================================================


class TestConcurrentToolExecution:
    """Multiple tool calls in a single iteration run concurrently via asyncio.gather."""

    async def test_parallel_tools_all_results_returned(self):
        """Multiple tools called in one iteration — all results come back."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="tc-1", name="check_disk", input={"host": "server"})
        tc2 = ToolCall(id="tc-2", name="check_memory", input={"host": "server"})
        tc3 = ToolCall(id="tc-3", name="run_command", input={"command": "uptime"})

        exec_count = 0
        async def _mock_exec(name, inp, **kwargs):
            nonlocal exec_count
            exec_count += 1
            return f"result-{name}"

        stub.tool_executor.execute = _mock_exec

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            call_count += 1
            if call_count == 1:
                return _tool_response([tc1, tc2, tc3])
            return _text_response("All good")

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "check everything"}]
            )

        assert text == "All good"
        assert set(tools_used) == {"check_disk", "check_memory", "run_command"}
        # All 3 tool results should be in the messages
        tool_results_msg = captured[1][-1]
        assert len(tool_results_msg["content"]) == 3

    async def test_parallel_tools_one_fails_others_succeed(self):
        """If one tool fails, others still return results."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="tc-1", name="check_disk", input={})
        tc2 = ToolCall(id="tc-2", name="run_command", input={"command": "fail"})

        async def _mock_exec(name, inp, **kwargs):
            if name == "run_command":
                raise RuntimeError("SSH failed")
            return "disk OK"

        stub.tool_executor.execute = _mock_exec

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            call_count += 1
            if call_count == 1:
                return _tool_response([tc1, tc2])
            return _text_response("Partial results")

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "check"}]
            )

        # Both tools should be in tools_used
        assert "check_disk" in tools_used
        assert "run_command" in tools_used

        # Results should have one success and one error
        results = captured[1][-1]["content"]
        assert len(results) == 2
        result_contents = [r["content"] for r in results]
        assert any("disk OK" in c for c in result_contents)
        assert any("Error" in c for c in result_contents)

    async def test_parallel_tools_all_have_correct_tool_use_ids(self):
        """Each tool result is tagged with the correct tool_use_id."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="id-aaa", name="check_disk", input={})
        tc2 = ToolCall(id="id-bbb", name="check_memory", input={})

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            call_count += 1
            if call_count == 1:
                return _tool_response([tc1, tc2])
            return _text_response("OK")

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "check"}]
            )

        results = captured[1][-1]["content"]
        ids = {r["tool_use_id"] for r in results}
        assert ids == {"id-aaa", "id-bbb"}

    async def test_concurrent_execution_timing(self):
        """Concurrent tools run in parallel, not sequentially."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="tc-1", name="run_command", input={"command": "sleep1"})
        tc2 = ToolCall(id="tc-2", name="run_command", input={"command": "sleep2"})

        async def _slow_exec(name, inp, **kwargs):
            await asyncio.sleep(0.05)  # 50ms per tool
            return "done"

        stub.tool_executor.execute = _slow_exec

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[_tool_response([tc1, tc2]), _text_response("Done")]
        )

        t0 = time.monotonic()
        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "run parallel"}]
            )
        elapsed = time.monotonic() - t0

        # If sequential: ~100ms. If parallel: ~50ms.
        # Allow generous margin but verify it's not sequential
        assert elapsed < 0.15, f"Tools ran sequentially ({elapsed:.3f}s), expected parallel"


# ===========================================================================
# 4. Tool output secret scrubbing
# ===========================================================================


class TestToolOutputScrubbing:
    """Secrets are scrubbed from tool output before LLM sees it."""

    async def test_scrub_password_in_tool_output(self):
        """Password patterns are redacted."""
        result = scrub_output_secrets("Connection string: password=s3cr3t123!")
        assert "s3cr3t123" not in result
        assert "[REDACTED]" in result

    async def test_scrub_api_key_in_tool_output(self):
        result = scrub_output_secrets("Config: api_key=abc123xyz789secret")
        assert "abc123xyz789" not in result
        assert "[REDACTED]" in result

    async def test_scrub_openai_key(self):
        result = scrub_output_secrets("Key is sk-abc123def456ghi789jklmnop012345")
        assert "sk-abc123" not in result
        assert "[REDACTED]" in result

    async def test_scrub_private_key_header(self):
        result = scrub_output_secrets("-----BEGIN RSA PRIVATE KEY-----")
        assert "BEGIN RSA PRIVATE KEY" not in result

    async def test_scrub_database_uri(self):
        result = scrub_output_secrets("postgres://admin:password123@db.example.com:5432/mydb")
        assert "password123" not in result
        assert "[REDACTED]" in result

    async def test_scrub_multiple_secrets(self):
        text = "password=abc123 and api_key=xyz789 and sk-longenoughkeyhere12345678"
        result = scrub_output_secrets(text)
        assert "abc123" not in result
        assert "xyz789" not in result
        assert "sk-longenoughkeyhere" not in result

    async def test_clean_text_unchanged(self):
        text = "Disk usage: 42% used, 58% free."
        result = scrub_output_secrets(text)
        assert result == text

    async def test_scrub_in_tool_loop(self):
        """Tool output is scrubbed before being added to messages for LLM."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="run_command", input={"command": "cat /etc/config"})
        stub.tool_executor.execute = AsyncMock(
            return_value="DB_PASSWORD=super_secret_password123"
        )

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            call_count += 1
            if call_count == 1:
                return _tool_response([tc])
            return _text_response("Config checked.")

        stub.codex_client.chat_with_tools = _capture

        # Use the real scrub_output_secrets (don't mock it)
        text, _, _, _, _ = await stub._process_with_tools(
            msg, [{"role": "user", "content": "show config"}]
        )

        # The tool result sent to LLM should have the secret scrubbed
        tool_result_msg = captured[1][-1]
        tool_output = tool_result_msg["content"][0]["content"]
        assert "super_secret_password" not in tool_output
        assert "[REDACTED]" in tool_output

    async def test_scrub_token_in_tool_output(self):
        result = scrub_output_secrets("token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdef")
        assert "eyJhbGciOiJ" not in result
        assert "[REDACTED]" in result

    async def test_scrub_mongodb_srv_uri(self):
        result = scrub_output_secrets("mongodb+srv://user:pass123@cluster.example.net/db")
        assert "pass123" not in result


# ===========================================================================
# 5. Tool timeout handling
# ===========================================================================


class TestToolTimeoutHandling:
    """Per-tool timeouts via asyncio.wait_for in _run_tool_with_timeout."""

    async def test_tool_timeout_returns_error_message(self):
        """When a tool exceeds the timeout, an error message is returned."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.config.tools.tool_timeout_seconds = 0.05  # 50ms timeout

        tc = ToolCall(id="tc-1", name="run_command", input={"command": "sleep 999"})

        async def _slow_exec(name, inp, **kwargs):
            await asyncio.sleep(10)  # Way longer than timeout
            return "never reaches here"

        stub.tool_executor.execute = _slow_exec

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            call_count += 1
            if call_count == 1:
                return _tool_response([tc])
            return _text_response("Timed out.")

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "run slow command"}]
            )

        # The timeout error should be in the tool result sent to LLM
        tool_result_msg = captured[1][-1]
        tool_output = tool_result_msg["content"][0]["content"]
        assert "timed out" in tool_output.lower()
        assert "run_command" in tools_used

    async def test_timeout_logs_to_audit(self):
        """Tool timeout should be recorded in audit log."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.config.tools.tool_timeout_seconds = 0.05

        tc = ToolCall(id="tc-1", name="check_disk", input={"host": "server"})

        async def _slow_exec(name, inp, **kwargs):
            await asyncio.sleep(10)
            return "never"

        stub.tool_executor.execute = _slow_exec

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[_tool_response([tc]), _text_response("Timed out")]
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "check disk"}]
            )

        # Audit log should record the timeout
        audit_calls = stub.audit.log_execution.call_args_list
        timeout_calls = [c for c in audit_calls if c.kwargs.get("error") and "timed out" in c.kwargs["error"].lower()]
        assert len(timeout_calls) >= 1

    async def test_timeout_one_tool_others_succeed(self):
        """In parallel execution, one timing out doesn't affect others."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.config.tools.tool_timeout_seconds = 0.1  # 100ms

        tc1 = ToolCall(id="tc-1", name="check_disk", input={})
        tc2 = ToolCall(id="tc-2", name="run_command", input={"command": "slow"})

        async def _mixed_exec(name, inp, **kwargs):
            if name == "run_command":
                await asyncio.sleep(10)  # Times out
                return "never"
            return "disk OK"

        stub.tool_executor.execute = _mixed_exec

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            call_count += 1
            if call_count == 1:
                return _tool_response([tc1, tc2])
            return _text_response("Partial results")

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, _ = await stub._process_with_tools(
                msg, [{"role": "user", "content": "check and run"}]
            )

        # Both tools should be tracked
        assert "check_disk" in tools_used
        assert "run_command" in tools_used

        # Results should have one success and one timeout
        results = captured[1][-1]["content"]
        assert len(results) == 2
        contents = [r["content"] for r in results]
        assert any("disk OK" in c for c in contents)
        assert any("timed out" in c.lower() for c in contents)

    async def test_timeout_uses_config_value(self):
        """Timeout value comes from config.tools.tool_timeout_seconds."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub.config.tools.tool_timeout_seconds = 42

        tc = ToolCall(id="tc-1", name="run_command", input={})

        # Tool that never finishes
        async def _hang(name, inp, **kwargs):
            await asyncio.sleep(999)

        stub.tool_executor.execute = _hang

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            call_count += 1
            if call_count == 1:
                return _tool_response([tc])
            return _text_response("OK")

        stub.codex_client.chat_with_tools = _capture

        # Override wait_for to verify the timeout value
        original_wait_for = asyncio.wait_for
        recorded_timeouts = []

        async def _capture_wait_for(coro, *, timeout):
            recorded_timeouts.append(timeout)
            raise asyncio.TimeoutError()

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("asyncio.wait_for", side_effect=_capture_wait_for):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "run"}]
            )

        assert 42 in recorded_timeouts


# ===========================================================================
# 6. Tool output truncation
# ===========================================================================


class TestToolOutputTruncation:
    """truncate_tool_output caps large outputs before LLM sees them."""

    def test_short_output_unchanged(self):
        text = "Disk usage: 42%"
        assert truncate_tool_output(text) == text

    def test_exact_limit_unchanged(self):
        text = "x" * TOOL_OUTPUT_MAX_CHARS
        assert truncate_tool_output(text) == text

    def test_over_limit_truncated(self):
        text = "x" * (TOOL_OUTPUT_MAX_CHARS + 5000)
        result = truncate_tool_output(text)
        assert len(result) < len(text)
        assert "omitted" in result
        # Preserves start and end
        assert result.startswith("x")
        assert result.endswith("x")

    def test_custom_max_chars(self):
        text = "abcdefghijklmnop"  # 16 chars
        result = truncate_tool_output(text, max_chars=10)
        assert "omitted" in result
        assert len(result) < 16 + 50  # Original minus omitted plus marker

    def test_truncation_preserves_head_and_tail(self):
        text = "HEAD" + "x" * 20000 + "TAIL"
        result = truncate_tool_output(text, max_chars=100)
        assert "HEAD" in result
        assert "TAIL" in result

    async def test_truncation_in_tool_loop(self):
        """Large tool output is truncated before being sent to LLM."""
        stub = _make_bot_stub()
        msg = _make_message()

        large_output = "x" * (TOOL_OUTPUT_MAX_CHARS + 5000)
        tc = ToolCall(id="tc-1", name="run_command", input={"command": "big output"})
        stub.tool_executor.execute = AsyncMock(return_value=large_output)

        captured = []
        call_count = 0
        async def _capture(messages, system, tools):
            nonlocal call_count
            captured.append(list(messages))
            call_count += 1
            if call_count == 1:
                return _tool_response([tc])
            return _text_response("Done")

        stub.codex_client.chat_with_tools = _capture

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "big command"}]
            )

        # Verify the tool output sent to LLM is truncated
        tool_result = captured[1][-1]["content"][0]["content"]
        assert len(tool_result) < len(large_output)
        assert "omitted" in tool_result


# ===========================================================================
# 7. Audit logging in tool loop
# ===========================================================================


class TestToolAuditLogging:
    """Every tool execution is logged to the audit system."""

    async def test_audit_log_called_for_each_tool(self):
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="tc-1", name="check_disk", input={"host": "server"})
        tc2 = ToolCall(id="tc-2", name="check_memory", input={"host": "server"})

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[_tool_response([tc1, tc2]), _text_response("Done")]
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "check all"}]
            )

        # Should have 2 audit log calls (one per tool)
        assert stub.audit.log_execution.call_count == 2
        logged_tools = [c.kwargs["tool_name"] for c in stub.audit.log_execution.call_args_list]
        assert "check_disk" in logged_tools
        assert "check_memory" in logged_tools

    async def test_audit_log_records_approved_true(self):
        """All tools are approved=True (no approval system)."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="run_command", input={"command": "ls"})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "list files"}]
            )

        call_kwargs = stub.audit.log_execution.call_args.kwargs
        assert call_kwargs["approved"] is True

    async def test_audit_log_records_error(self):
        """Tool errors are recorded in audit log."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="run_command", input={"command": "fail"})
        stub.tool_executor.execute = AsyncMock(side_effect=RuntimeError("broke"))

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "fail"}]
            )

        call_kwargs = stub.audit.log_execution.call_args.kwargs
        assert call_kwargs["error"] is not None
        assert "broke" in call_kwargs["error"]

    async def test_audit_log_records_execution_time(self):
        """Execution time is captured in milliseconds."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="run_command", input={})

        async def _slow_exec(name, inp, **kwargs):
            await asyncio.sleep(0.05)
            return "done"

        stub.tool_executor.execute = _slow_exec

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "run"}]
            )

        call_kwargs = stub.audit.log_execution.call_args.kwargs
        assert call_kwargs["execution_time_ms"] >= 40  # At least ~50ms with some tolerance


# ===========================================================================
# 8. Progress embed and cancel in tool loop
# ===========================================================================


class TestProgressEmbedInToolLoop:
    """Progress embed tracks tool steps and cancel button works."""

    async def test_progress_embed_created_on_first_tool_call(self):
        """Progress embed is sent when the first tool call happens."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="check_disk", input={})
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=_codex_tool_then_text([tc])
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "check disk"}]
            )

        # _build_tool_progress_embed should have been called
        stub._build_tool_progress_embed.assert_called()

    async def test_tools_tracked_in_progress_steps(self):
        """Each iteration adds a step to progress_steps."""
        stub = _make_bot_stub()
        msg = _make_message()

        tc1 = ToolCall(id="tc-1", name="check_disk", input={})
        tc2 = ToolCall(id="tc-2", name="check_memory", input={})

        call_count = 0
        async def _steps(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response([tc1])
            elif call_count == 2:
                return _tool_response([tc2])
            return _text_response("Done")

        stub.codex_client.chat_with_tools = _steps

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            await stub._process_with_tools(
                msg, [{"role": "user", "content": "check all"}]
            )

        # _build_tool_progress_embed called multiple times with accumulating steps
        all_calls = stub._build_tool_progress_embed.call_args_list
        assert len(all_calls) >= 2  # At least 2 iterations


# ===========================================================================
# 9. Skill handoff
# ===========================================================================


class TestSkillHandoff:
    """When all tools in an iteration are handoff skills, the loop returns with handoff=True."""

    async def test_skill_handoff_returns_true(self):
        stub = _make_bot_stub()
        msg = _make_message()

        tc = ToolCall(id="tc-1", name="my_skill", input={})
        stub.skill_manager.has_skill = MagicMock(side_effect=lambda n: n == "my_skill")
        stub.skill_manager.execute = AsyncMock(return_value="Skill result!")
        stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=True)

        stub.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_response([tc])
        )

        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x):
            text, _, _, tools_used, handoff = await stub._process_with_tools(
                msg, [{"role": "user", "content": "run skill"}]
            )

        assert handoff is True
        assert "my_skill" in tools_used


# ===========================================================================
# 10. Source structure verification
# ===========================================================================


class TestSourceStructureVerification:
    """Verify source code structure for tool execution paths."""

    def test_all_discord_native_tools_in_run_tool(self):
        """All Discord-native tool names appear in _run_tool dispatch."""
        import inspect
        source = inspect.getsource(LokiBot._process_with_tools)

        discord_native = [
            "purge_messages", "browser_screenshot", "generate_file", "post_file",
            "schedule_task", "list_schedules", "delete_schedule", "parse_time",
            "search_history", "delegate_task", "list_tasks", "cancel_task",
            "search_knowledge", "ingest_document", "list_knowledge", "delete_knowledge",
            "set_permission", "search_audit", "create_digest",
            "create_skill", "edit_skill", "delete_skill", "list_skills",
        ]
        for tool_name in discord_native:
            assert tool_name in source, f"Discord-native tool {tool_name} not in _process_with_tools"

    def test_scrub_output_secrets_called_in_run_tool(self):
        source = inspect.getsource(LokiBot._process_with_tools)
        assert "scrub_output_secrets" in source

    def test_audit_log_called_in_run_tool(self):
        source = inspect.getsource(LokiBot._process_with_tools)
        assert "audit.log_execution" in source

    def test_truncate_tool_output_called_in_run_tool(self):
        source = inspect.getsource(LokiBot._process_with_tools)
        assert "truncate_tool_output" in source

    def test_asyncio_gather_for_parallel_execution(self):
        source = inspect.getsource(LokiBot._process_with_tools)
        assert "asyncio.gather" in source

    def test_wait_for_timeout_wrapping(self):
        source = inspect.getsource(LokiBot._process_with_tools)
        assert "asyncio.wait_for" in source
        assert "tool_timeout" in source

    def test_max_tool_iterations_guard(self):
        source = inspect.getsource(LokiBot._process_with_tools)
        assert "MAX_TOOL_ITERATIONS" in source

    def test_fallback_to_tool_executor(self):
        """Unknown tools fall through to tool_executor.execute."""
        source = inspect.getsource(LokiBot._process_with_tools)
        assert "tool_executor.execute" in source

    def test_tool_choice_is_auto(self):
        """chat_with_tools uses tool_choice='auto'."""
        import inspect
        source = inspect.getsource(LokiBot)
        # Check in codex client
        from src.llm.openai_codex import CodexChatClient
        codex_source = inspect.getsource(CodexChatClient.chat_with_tools)
        assert '"auto"' in codex_source

    def test_no_approval_in_run_tool(self):
        """No approval checks in the tool execution path."""
        source = inspect.getsource(LokiBot._process_with_tools)
        assert "request_approval" not in source
        assert "requires_approval" not in source
        assert "ApprovalView" not in source


import inspect  # noqa: E402 — needed for source inspection tests
