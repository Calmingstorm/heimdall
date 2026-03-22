"""Tests for discord/background_task.py — background task delegation system.

Covers:
- _substitute_vars: variable and prev_output replacement in tool inputs
- _check_condition: positive and negated substring matching
- _send_progress: progress message creation, editing, truncation, file attachment
- _send_summary: completed/failed/cancelled summary messages
- run_background_task: full orchestrator with step execution, cancellation,
  blocked tools, condition skipping, error handling (abort vs continue),
  variable storage, and progress updates
- _execute_tool: routing to knowledge store, skill manager, and executor
- create_task_id: unique ID generation
- BackgroundTask.cancel: cancellation event
- StepResult dataclass
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.background_task import (  # noqa: E402
    BackgroundTask,
    StepResult,
    BLOCKED_TOOLS,
    MAX_STEPS,
    PROGRESS_UPDATE_INTERVAL,
    run_background_task,
    _execute_tool,
    _substitute_vars,
    _check_condition,
    _send_progress,
    _send_summary,
    create_task_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_channel():
    """Mock Discord channel with send/edit."""
    ch = AsyncMock()
    msg = AsyncMock()
    msg.edit = AsyncMock()
    ch.send = AsyncMock(return_value=msg)
    return ch, msg


def _make_task(steps=None, description="Test task", channel=None):
    """Create a BackgroundTask with sensible defaults."""
    ch = channel or _make_channel()[0]
    return BackgroundTask(
        task_id="abc12345",
        description=description,
        steps=steps or [],
        channel=ch,
        requester="user123",
    )


def _make_executor():
    """Mock ToolExecutor."""
    ex = MagicMock()
    ex.execute = AsyncMock(return_value="executed ok")
    return ex


def _make_skill_manager(has=False):
    """Mock SkillManager."""
    sm = MagicMock()
    sm.has_skill = MagicMock(return_value=has)
    sm.execute = AsyncMock(return_value="skill result")
    return sm


# ---------------------------------------------------------------------------
# create_task_id
# ---------------------------------------------------------------------------

class TestCreateTaskId:
    def test_returns_8_char_hex(self):
        """create_task_id returns an 8-character hex string."""
        tid = create_task_id()
        assert len(tid) == 8
        int(tid, 16)  # should not raise

    def test_unique(self):
        """Successive calls produce different IDs."""
        ids = {create_task_id() for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# StepResult dataclass
# ---------------------------------------------------------------------------

class TestStepResult:
    def test_defaults(self):
        """StepResult has sensible defaults for output and elapsed_ms."""
        r = StepResult(index=0, tool_name="test", description="desc", status="ok")
        assert r.output == ""
        assert r.elapsed_ms == 0

    def test_with_values(self):
        """StepResult stores all provided values."""
        r = StepResult(
            index=2, tool_name="run_command", description="run ls",
            status="error", output="permission denied", elapsed_ms=150,
        )
        assert r.index == 2
        assert r.tool_name == "run_command"
        assert r.status == "error"
        assert r.output == "permission denied"
        assert r.elapsed_ms == 150


# ---------------------------------------------------------------------------
# BackgroundTask
# ---------------------------------------------------------------------------

class TestBackgroundTask:
    def test_cancel_sets_event(self):
        """cancel() sets the internal cancellation event."""
        task = _make_task()
        assert not task._cancel_event.is_set()
        task.cancel()
        assert task._cancel_event.is_set()

    def test_default_status_is_running(self):
        """New tasks start with status 'running'."""
        task = _make_task()
        assert task.status == "running"

    def test_results_empty_by_default(self):
        """New tasks have no results."""
        task = _make_task()
        assert task.results == []
        assert task.current_step == 0


# ---------------------------------------------------------------------------
# _substitute_vars
# ---------------------------------------------------------------------------

class TestSubstituteVars:
    def test_prev_output_replacement(self):
        """Replaces {prev_output} in string values."""
        result = _substitute_vars(
            {"cmd": "echo {prev_output}"}, {}, "hello world",
        )
        assert result["cmd"] == "echo hello world"

    def test_named_variable_replacement(self):
        """Replaces {var.name} with stored variables."""
        result = _substitute_vars(
            {"host": "{var.target}", "port": "{var.port}"},
            {"target": "server", "port": "8080"},
            "",
        )
        assert result["host"] == "server"
        assert result["port"] == "8080"

    def test_non_string_values_unchanged(self):
        """Non-string values (int, bool, list) pass through unchanged."""
        result = _substitute_vars(
            {"count": 5, "enabled": True, "tags": ["a", "b"]},
            {"x": "y"}, "prev",
        )
        assert result["count"] == 5
        assert result["enabled"] is True
        assert result["tags"] == ["a", "b"]

    def test_combined_vars_and_prev_output(self):
        """Both {prev_output} and {var.x} work in the same string."""
        result = _substitute_vars(
            {"msg": "{var.prefix}: {prev_output}"},
            {"prefix": "Result"}, "success",
        )
        assert result["msg"] == "Result: success"

    def test_no_vars_no_change(self):
        """Input without placeholders is returned as-is."""
        result = _substitute_vars({"cmd": "ls -la"}, {}, "")
        assert result["cmd"] == "ls -la"

    def test_empty_input(self):
        """Empty dict returns empty dict."""
        assert _substitute_vars({}, {}, "prev") == {}


# ---------------------------------------------------------------------------
# _check_condition
# ---------------------------------------------------------------------------

class TestCheckCondition:
    def test_positive_match(self):
        """Positive condition matches when substring is present."""
        assert _check_condition("success", "Operation success!") is True

    def test_positive_no_match(self):
        """Positive condition fails when substring is absent."""
        assert _check_condition("success", "Operation failed!") is False

    def test_negated_match(self):
        """Negated condition (!) matches when substring is NOT present."""
        assert _check_condition("!error", "Operation success!") is True

    def test_negated_no_match(self):
        """Negated condition fails when substring IS present."""
        assert _check_condition("!error", "Got an error here") is False

    def test_case_insensitive(self):
        """Condition matching is case-insensitive."""
        assert _check_condition("SUCCESS", "operation success") is True
        assert _check_condition("!ERROR", "all good") is True


# ---------------------------------------------------------------------------
# _send_progress
# ---------------------------------------------------------------------------

class TestSendProgress:
    async def test_creates_new_message(self):
        """Posts a new message when existing_msg is None."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "t"}], channel=ch)
        msg = await _send_progress(task, None)
        ch.send.assert_called_once()
        assert msg is not None

    async def test_edits_existing_message(self):
        """Edits the existing message when one is provided."""
        ch, msg = _make_channel()
        task = _make_task(steps=[{"tool_name": "t"}], channel=ch)
        result = await _send_progress(task, msg)
        msg.edit.assert_called_once()
        assert result is msg

    async def test_shows_step_progress(self):
        """Progress message includes step count and progress bar."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "a"}, {"tool_name": "b"}],
            channel=ch,
        )
        task.current_step = 0
        task.results.append(StepResult(0, "a", "step a", "ok", "done"))
        await _send_progress(task, None)
        text = ch.send.call_args[1].get("content") or ch.send.call_args[0][0]
        assert "1/2" in text

    async def test_completed_status_icon(self):
        """Completed tasks show DONE in the progress message."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "a"}], channel=ch)
        task.status = "completed"
        task.results.append(StepResult(0, "a", "step a", "ok", "done"))
        await _send_progress(task, None)
        text = ch.send.call_args[1].get("content") or ch.send.call_args[0][0]
        assert "DONE" in text

    async def test_failed_status_icon(self):
        """Failed tasks show FAILED in the progress message."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "a"}], channel=ch)
        task.status = "failed"
        task.results.append(StepResult(0, "a", "step a", "error", "boom"))
        await _send_progress(task, None)
        text = ch.send.call_args[1].get("content") or ch.send.call_args[0][0]
        assert "FAILED" in text

    async def test_cancelled_status_icon(self):
        """Cancelled tasks show CANCELLED in the progress message."""
        ch, _ = _make_channel()
        task = _make_task(steps=[], channel=ch)
        task.status = "cancelled"
        await _send_progress(task, None)
        text = ch.send.call_args[1].get("content") or ch.send.call_args[0][0]
        assert "CANCELLED" in text

    async def test_no_steps(self):
        """Tasks with no steps show 'No steps'."""
        ch, _ = _make_channel()
        task = _make_task(steps=[], channel=ch)
        await _send_progress(task, None)
        text = ch.send.call_args[1].get("content") or ch.send.call_args[0][0]
        assert "No steps" in text

    async def test_running_shows_last_3_results(self):
        """While running, only the last 3 results are shown."""
        ch, _ = _make_channel()
        steps = [{"tool_name": f"t{i}"} for i in range(5)]
        task = _make_task(steps=steps, channel=ch)
        for i in range(5):
            task.results.append(StepResult(i, f"t{i}", f"step {i}", "ok", f"out{i}"))
        # Still running — should only show last 3
        await _send_progress(task, None)
        text = ch.send.call_args[1].get("content") or ch.send.call_args[0][0]
        assert "Step 3" in text
        assert "Step 5" in text

    async def test_finished_shows_all_results(self):
        """When completed, all results are shown."""
        ch, _ = _make_channel()
        steps = [{"tool_name": f"t{i}"} for i in range(5)]
        task = _make_task(steps=steps, channel=ch)
        task.status = "completed"
        for i in range(5):
            task.results.append(StepResult(i, f"t{i}", f"step {i}", "ok", f"out{i}"))
        await _send_progress(task, None)
        text = ch.send.call_args[1].get("content") or ch.send.call_args[0][0]
        assert "Step 1" in text
        assert "Step 5" in text

    async def test_long_text_truncated_while_running(self):
        """Long progress text is truncated to 1900 chars while running."""
        ch, _ = _make_channel()
        steps = [{"tool_name": f"t{i}"} for i in range(3)]
        task = _make_task(steps=steps, channel=ch)
        # Add results with very long output
        for i in range(3):
            task.results.append(
                StepResult(i, f"t{i}", f"step {i}", "ok", "x" * 2000)
            )
        await _send_progress(task, None)
        text = ch.send.call_args[1].get("content") or ch.send.call_args[0][0]
        assert len(text) <= 1905  # 1900 + "..." + newline

    async def test_long_finished_text_sends_file(self):
        """Long finished progress sends a file attachment."""
        ch, _ = _make_channel()
        steps = [{"tool_name": f"t{i}"} for i in range(50)]
        task = _make_task(steps=steps, channel=ch)
        task.status = "completed"
        for i in range(50):
            task.results.append(
                StepResult(i, f"t{i}", f"step {i} long desc", "ok", "output " * 20)
            )
        await _send_progress(task, None)
        # Should have called send with a file
        found_file = False
        for c in ch.send.call_args_list:
            if c[1].get("file") is not None:
                found_file = True
                break
        assert found_file

    async def test_handles_send_exception(self):
        """Gracefully handles Discord send failures."""
        ch, _ = _make_channel()
        ch.send = AsyncMock(side_effect=Exception("Discord down"))
        task = _make_task(steps=[], channel=ch)
        # Should not raise
        result = await _send_progress(task, None)
        assert result is None  # returns existing_msg (None)

    async def test_ok_error_skipped_counts(self):
        """Progress message shows OK, Errors, Skipped counts."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "a"}] * 4, channel=ch)
        task.status = "completed"
        task.results = [
            StepResult(0, "a", "s0", "ok", ""),
            StepResult(1, "a", "s1", "ok", ""),
            StepResult(2, "a", "s2", "error", ""),
            StepResult(3, "a", "s3", "skipped", ""),
        ]
        await _send_progress(task, None)
        text = ch.send.call_args[1].get("content") or ch.send.call_args[0][0]
        assert "OK: 2" in text
        assert "Errors: 1" in text
        assert "Skipped: 1" in text


# ---------------------------------------------------------------------------
# _send_summary
# ---------------------------------------------------------------------------

class TestSendSummary:
    async def test_completed_no_errors(self):
        """Summary for a fully successful task."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "a"}],
            channel=ch,
        )
        task.status = "completed"
        task.results = [StepResult(0, "a", "run a", "ok", "all good")]
        await _send_summary(task)
        text = ch.send.call_args[0][0]
        assert "All 1 steps succeeded" in text
        assert "all good" in text

    async def test_completed_with_errors(self):
        """Summary for a task that completed but had some errors."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "a"}] * 2, channel=ch)
        task.status = "completed"
        task.results = [
            StepResult(0, "a", "step 1", "ok", "fine"),
            StepResult(1, "a", "step 2", "error", "oops"),
        ]
        await _send_summary(task)
        text = ch.send.call_args[0][0]
        assert "1 succeeded" in text
        assert "1 failed" in text

    async def test_failed_summary(self):
        """Summary for an aborted task."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "a"}] * 5, channel=ch)
        task.status = "failed"
        task.results = [
            StepResult(0, "a", "step 1", "ok", "fine"),
            StepResult(1, "a", "step 2", "error", "boom"),
        ]
        await _send_summary(task)
        text = ch.send.call_args[0][0]
        assert "aborted" in text
        assert "2 of 5" in text

    async def test_cancelled_summary(self):
        """Summary for a cancelled task."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "a"}] * 3, channel=ch)
        task.status = "cancelled"
        task.results = [StepResult(0, "a", "step 1", "ok", "fine")]
        await _send_summary(task)
        text = ch.send.call_args[0][0]
        assert "cancelled" in text
        assert "1 of 3" in text

    async def test_skipped_steps_omitted(self):
        """Skipped steps are not shown in the summary detail."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "a"}] * 2, channel=ch)
        task.status = "completed"
        task.results = [
            StepResult(0, "a", "step 1", "ok", "fine"),
            StepResult(1, "a", "step 2", "skipped", "condition not met"),
        ]
        await _send_summary(task)
        text = ch.send.call_args[0][0]
        assert "condition not met" not in text

    async def test_long_output_truncated_in_summary(self):
        """Long step outputs are truncated to 200 chars in summary."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "a"}], channel=ch)
        task.status = "completed"
        task.results = [StepResult(0, "a", "step 1", "ok", "x" * 300)]
        await _send_summary(task)
        text = ch.send.call_args[0][0]
        assert "..." in text

    async def test_very_long_summary_sends_file(self):
        """Very long summaries are sent as file attachments."""
        ch, _ = _make_channel()
        task = _make_task(steps=[{"tool_name": "a"}] * 50, channel=ch)
        task.status = "completed"
        task.results = [
            StepResult(i, "a", f"long description step {i}", "ok", "output " * 40)
            for i in range(50)
        ]
        await _send_summary(task)
        found_file = False
        for c in ch.send.call_args_list:
            if c[1].get("file") is not None:
                found_file = True
                break
        assert found_file

    async def test_handles_send_failure(self):
        """Gracefully handles Discord send failures in summary."""
        ch, _ = _make_channel()
        ch.send = AsyncMock(side_effect=Exception("Discord error"))
        task = _make_task(steps=[], channel=ch)
        task.status = "completed"
        # Should not raise
        await _send_summary(task)


# ---------------------------------------------------------------------------
# _execute_tool
# ---------------------------------------------------------------------------

class TestExecuteTool:
    async def test_routes_ingest_document_to_knowledge_store(self):
        """ingest_document routes to knowledge_store.ingest."""
        ks = AsyncMock()
        ks.ingest = AsyncMock(return_value=5)
        emb = AsyncMock()
        result = await _execute_tool(
            "ingest_document",
            {"source": "test.md", "content": "Hello world"},
            _make_executor(), _make_skill_manager(),
            knowledge_store=ks, embedder=emb, requester="user1",
        )
        ks.ingest.assert_called_once_with(
            content="Hello world", source="test.md",
            embedder=emb, uploader="user1",
        )
        assert "5 chunks" in result

    async def test_ingest_missing_fields(self):
        """ingest_document returns error when source or content missing."""
        ks = AsyncMock()
        emb = AsyncMock()
        result = await _execute_tool(
            "ingest_document", {"source": ""},
            _make_executor(), _make_skill_manager(),
            knowledge_store=ks, embedder=emb, requester="user1",
        )
        assert "required" in result.lower()

    async def test_routes_search_knowledge(self):
        """search_knowledge routes to knowledge_store.search."""
        ks = AsyncMock()
        ks.search = AsyncMock(return_value=[
            {"source": "doc1", "score": 0.9, "content": "hello world"},
        ])
        emb = AsyncMock()
        result = await _execute_tool(
            "search_knowledge", {"query": "hello", "limit": 3},
            _make_executor(), _make_skill_manager(),
            knowledge_store=ks, embedder=emb, requester="user1",
        )
        assert "doc1" in result
        assert "0.9" in result

    async def test_search_knowledge_no_results(self):
        """search_knowledge returns 'no results' when empty."""
        ks = AsyncMock()
        ks.search = AsyncMock(return_value=[])
        emb = AsyncMock()
        result = await _execute_tool(
            "search_knowledge", {"query": "nothing"},
            _make_executor(), _make_skill_manager(),
            knowledge_store=ks, embedder=emb, requester="user1",
        )
        assert "No results" in result

    async def test_search_knowledge_limits_to_10(self):
        """search_knowledge caps limit at 10."""
        ks = AsyncMock()
        ks.search = AsyncMock(return_value=[])
        emb = AsyncMock()
        await _execute_tool(
            "search_knowledge", {"query": "q", "limit": 50},
            _make_executor(), _make_skill_manager(),
            knowledge_store=ks, embedder=emb, requester="user1",
        )
        _, kwargs = ks.search.call_args
        assert kwargs["limit"] == 10

    async def test_routes_list_knowledge(self):
        """list_knowledge routes to knowledge_store.list_sources."""
        ks = MagicMock()
        ks.list_sources = MagicMock(return_value=[
            {"source": "doc1", "chunks": 3},
            {"source": "doc2", "chunks": 7},
        ])
        result = await _execute_tool(
            "list_knowledge", {},
            _make_executor(), _make_skill_manager(),
            knowledge_store=ks, embedder=None, requester="user1",
        )
        assert "doc1" in result
        assert "3 chunks" in result
        assert "doc2" in result

    async def test_list_knowledge_empty(self):
        """list_knowledge returns 'empty' when no sources."""
        ks = MagicMock()
        ks.list_sources = MagicMock(return_value=[])
        result = await _execute_tool(
            "list_knowledge", {},
            _make_executor(), _make_skill_manager(),
            knowledge_store=ks, embedder=None, requester="user1",
        )
        assert "empty" in result.lower()

    async def test_routes_to_skill_manager(self):
        """Tool names matching a skill are routed to skill_manager."""
        sm = _make_skill_manager(has=True)
        result = await _execute_tool(
            "my_custom_skill", {"arg": "val"},
            _make_executor(), sm,
            knowledge_store=None, embedder=None, requester="user1",
        )
        sm.execute.assert_called_once_with("my_custom_skill", {"arg": "val"})
        assert result == "skill result"

    async def test_routes_to_executor(self):
        """Unknown tool names fall through to executor."""
        ex = _make_executor()
        sm = _make_skill_manager(has=False)
        result = await _execute_tool(
            "run_command", {"command": "ls"},
            ex, sm,
            knowledge_store=None, embedder=None, requester="user1",
        )
        ex.execute.assert_called_once_with("run_command", {"command": "ls"})
        assert result == "executed ok"


# ---------------------------------------------------------------------------
# run_background_task — orchestrator
# ---------------------------------------------------------------------------

class TestRunBackgroundTask:
    async def test_simple_successful_task(self):
        """Runs all steps and ends with status 'completed'."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "ls"}, "description": "list files"},
                {"tool_name": "check_disk", "tool_input": {}, "description": "check disk"},
            ],
            channel=ch,
        )
        ex = _make_executor()
        sm = _make_skill_manager()

        await run_background_task(task, ex, sm)

        assert task.status == "completed"
        assert len(task.results) == 2
        assert all(r.status == "ok" for r in task.results)

    async def test_blocked_tool_abort(self):
        """A blocked tool with on_failure=abort stops the task."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "purge_messages", "tool_input": {}, "description": "purge",
                 "on_failure": "abort"},
            ],
            channel=ch,
        )
        await run_background_task(task, _make_executor(), _make_skill_manager())

        assert task.status == "failed"
        assert task.results[0].status == "error"
        assert "cannot run" in task.results[0].output.lower()

    async def test_blocked_tool_continue(self):
        """A blocked tool with on_failure=continue skips and continues."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "purge_messages", "tool_input": {}, "description": "purge",
                 "on_failure": "continue"},
                {"tool_name": "check_disk", "tool_input": {}, "description": "check"},
            ],
            channel=ch,
        )
        await run_background_task(task, _make_executor(), _make_skill_manager())

        assert task.status == "completed"
        assert task.results[0].status == "error"
        assert task.results[1].status == "ok"

    async def test_condition_skip(self):
        """A step with a false condition is skipped."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "echo success"},
                 "description": "step 1"},
                {"tool_name": "run_command", "tool_input": {"command": "echo fix"},
                 "description": "step 2 (only on error)", "condition": "error"},
            ],
            channel=ch,
        )
        ex = _make_executor()
        ex.execute = AsyncMock(return_value="all good, success")

        await run_background_task(task, ex, _make_skill_manager())

        assert task.status == "completed"
        assert task.results[0].status == "ok"
        assert task.results[1].status == "skipped"

    async def test_condition_negated_match(self):
        """A negated condition that matches allows the step to run."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {}, "description": "step 1"},
                {"tool_name": "run_command", "tool_input": {},
                 "description": "step 2 (if no error)", "condition": "!error"},
            ],
            channel=ch,
        )
        ex = _make_executor()
        ex.execute = AsyncMock(return_value="all good")

        await run_background_task(task, ex, _make_skill_manager())

        assert len(task.results) == 2
        assert task.results[1].status == "ok"

    async def test_cancellation(self):
        """Cancelling a task during execution stops processing."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {}, "description": "step 1"},
                {"tool_name": "run_command", "tool_input": {}, "description": "step 2"},
            ],
            channel=ch,
        )
        # Cancel before the first step
        task.cancel()
        await run_background_task(task, _make_executor(), _make_skill_manager())

        assert task.status == "cancelled"
        assert len(task.results) == 1
        assert task.results[0].status == "cancelled"

    async def test_error_abort(self):
        """A tool execution error with on_failure=abort stops the task."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {}, "description": "step 1",
                 "on_failure": "abort"},
                {"tool_name": "run_command", "tool_input": {}, "description": "step 2"},
            ],
            channel=ch,
        )
        ex = _make_executor()
        ex.execute = AsyncMock(side_effect=RuntimeError("SSH failed"))

        await run_background_task(task, ex, _make_skill_manager())

        assert task.status == "failed"
        assert len(task.results) == 1
        assert task.results[0].status == "error"
        assert "SSH failed" in task.results[0].output

    async def test_error_continue(self):
        """A tool error with on_failure=continue proceeds to next step."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {}, "description": "step 1",
                 "on_failure": "continue"},
                {"tool_name": "run_command", "tool_input": {}, "description": "step 2"},
            ],
            channel=ch,
        )
        ex = _make_executor()
        call_count = [0]

        async def side_effect(name, inp):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first fails")
            return "second ok"

        ex.execute = AsyncMock(side_effect=side_effect)

        await run_background_task(task, ex, _make_skill_manager())

        assert task.status == "completed"
        assert task.results[0].status == "error"
        assert task.results[1].status == "ok"

    async def test_store_as_variable(self):
        """store_as saves tool output to variables for later substitution."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {"command": "hostname"},
                 "description": "get hostname", "store_as": "host"},
                {"tool_name": "run_command",
                 "tool_input": {"command": "ping {var.host}"},
                 "description": "ping host"},
            ],
            channel=ch,
        )
        ex = _make_executor()
        call_count = [0]

        async def side_effect(name, inp):
            call_count[0] += 1
            if call_count[0] == 1:
                return "myserver"
            return f"pinged {inp.get('command', '')}"

        ex.execute = AsyncMock(side_effect=side_effect)

        await run_background_task(task, ex, _make_skill_manager())

        assert task.status == "completed"
        # The second step should have had the variable substituted
        second_call = ex.execute.call_args_list[1]
        assert second_call[0][1]["command"] == "ping myserver"

    async def test_prev_output_substitution(self):
        """prev_output is passed between steps."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {}, "description": "step 1"},
                {"tool_name": "run_command",
                 "tool_input": {"data": "got: {prev_output}"},
                 "description": "step 2"},
            ],
            channel=ch,
        )
        ex = _make_executor()
        ex.execute = AsyncMock(return_value="step1_result")

        await run_background_task(task, ex, _make_skill_manager())

        second_call = ex.execute.call_args_list[1]
        assert second_call[0][1]["data"] == "got: step1_result"

    async def test_output_truncated_to_500(self):
        """Step output is truncated to 500 chars in results."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "run_command", "tool_input": {}, "description": "step 1"}],
            channel=ch,
        )
        ex = _make_executor()
        ex.execute = AsyncMock(return_value="x" * 1000)

        await run_background_task(task, ex, _make_skill_manager())

        assert len(task.results[0].output) == 500

    async def test_empty_steps_completes(self):
        """A task with no steps completes immediately."""
        ch, _ = _make_channel()
        task = _make_task(steps=[], channel=ch)

        await run_background_task(task, _make_executor(), _make_skill_manager())

        assert task.status == "completed"
        assert len(task.results) == 0

    async def test_condition_with_no_prev_output_runs(self):
        """Condition on first step (no prev_output) does not skip."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[
                {"tool_name": "run_command", "tool_input": {},
                 "description": "step 1", "condition": "something"},
            ],
            channel=ch,
        )
        # condition="something" with empty prev_output => condition check evaluates:
        # prev_output is "" which is falsy, so `if condition and prev_output` is False
        # meaning the step runs
        await run_background_task(task, _make_executor(), _make_skill_manager())

        assert task.results[0].status == "ok"

    async def test_elapsed_ms_recorded(self):
        """elapsed_ms is recorded for each step."""
        ch, _ = _make_channel()
        task = _make_task(
            steps=[{"tool_name": "run_command", "tool_input": {}, "description": "step 1"}],
            channel=ch,
        )
        await run_background_task(task, _make_executor(), _make_skill_manager())

        assert task.results[0].elapsed_ms >= 0


# ---------------------------------------------------------------------------
# BLOCKED_TOOLS constant
# ---------------------------------------------------------------------------

class TestBlockedTools:
    def test_blocked_tools_contains_expected(self):
        """BLOCKED_TOOLS includes key dangerous/interactive tools."""
        assert "purge_messages" in BLOCKED_TOOLS
        assert "delegate_task" in BLOCKED_TOOLS
        assert "browser_screenshot" in BLOCKED_TOOLS

    def test_run_command_not_blocked(self):
        """Standard tools like run_command are not blocked."""
        assert "run_command" not in BLOCKED_TOOLS
        assert "check_disk" not in BLOCKED_TOOLS
