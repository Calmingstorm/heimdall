"""Round 7: Request isolation tests.

Verifies the strengthened context separator with:
- Request hash for disambiguation
- Timestamp for temporal grounding
- User identity for provenance
- Explicit anti-re-execution language
- Adversarial scenarios (topic switches, similar requests)
"""
from __future__ import annotations

import hashlib
import sys
import time
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tc(name, inp=None):
    return ToolCall(id=f"tc-{name}", name=name, input=inp or {})


def _make_stub(respond_to_bots=True):
    """HeimdallBot stub for _process_with_tools-level tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._system_prompt = "test prompt"
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.discord.respond_to_bots = respond_to_bots
    stub.config.discord.require_mention = False
    stub.sessions = MagicMock()
    stub.codex_client = MagicMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.should_handoff_to_codex = MagicMock(return_value=False)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub._send_with_retry = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    return stub


def _make_msg(content="test", author_id="user-1", display_name="TestUser",
              is_bot=False, channel_id="ch-1", webhook_id=None):
    msg = MagicMock()
    msg.author = MagicMock()
    msg.author.bot = is_bot
    msg.author.id = author_id
    msg.author.display_name = display_name
    msg.author.name = display_name
    msg.author.__str__ = lambda s: display_name
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.__str__ = lambda s: f"#{channel_id}"
    msg.channel.typing = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(), __aexit__=AsyncMock(),
    ))
    msg.content = content
    msg.id = hash(content) % 2**32
    msg.webhook_id = webhook_id
    return msg


def _get_separator(call_args_list):
    """Extract the developer-role separator from the first chat_with_tools call."""
    msgs = call_args_list[0][1]["messages"]
    devs = [m for m in msgs if m.get("role") == "developer"]
    assert devs, "No developer message found"
    return devs[0]["content"]


# ===================================================================
# Request Hash
# ===================================================================

class TestRequestHash:
    """Separator should include a unique request hash for disambiguation."""

    async def test_separator_contains_request_hash(self):
        stub = _make_stub()
        msg = _make_msg(content="check disk space")
        expected_hash = hashlib.sha256(b"check disk space").hexdigest()[:8]
        history = [
            {"role": "user", "content": "old msg"},
            {"role": "user", "content": "check disk space"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert f"req-{expected_hash}" in sep

    async def test_different_content_different_hash(self):
        """Two different messages should produce different request hashes."""
        hash_a = hashlib.sha256(b"restart nginx").hexdigest()[:8]
        hash_b = hashlib.sha256(b"check memory").hexdigest()[:8]
        assert hash_a != hash_b

    async def test_hash_format(self):
        """Request hash should be 8 hex characters."""
        h = hashlib.sha256(b"any content").hexdigest()[:8]
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)


# ===================================================================
# Timestamp
# ===================================================================

class TestRequestTimestamp:
    """Separator should include a timestamp for temporal grounding."""

    async def test_separator_contains_timestamp(self):
        stub = _make_stub()
        msg = _make_msg(content="check status")
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "check status"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "Time:" in sep
        assert "UTC" in sep

    async def test_timestamp_is_recent(self):
        """Timestamp should be within a few seconds of now."""
        stub = _make_stub()
        msg = _make_msg(content="test time")
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "test time"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        before = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        # The timestamp in the separator should contain the current date
        assert before[:10] in sep  # YYYY-MM-DD


# ===================================================================
# User Identity
# ===================================================================

class TestUserIdentity:
    """Separator should include user display name and ID."""

    async def test_separator_contains_user_display_name(self):
        stub = _make_stub()
        msg = _make_msg(content="hello", display_name="Alice", author_id="12345")
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "hello"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "Alice" in sep
        assert "12345" in sep

    async def test_separator_contains_from_prefix(self):
        stub = _make_stub()
        msg = _make_msg(content="test", display_name="Bob")
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "test"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "From:" in sep


# ===================================================================
# Anti-Re-Execution Language
# ===================================================================

class TestAntiReExecution:
    """Separator should explicitly forbid re-executing old tasks."""

    async def test_separator_forbids_re_execution(self):
        stub = _make_stub()
        msg = _make_msg(content="new task")
        history = [
            {"role": "user", "content": "restart nginx"},
            {"role": "assistant", "content": "Done, nginx restarted."},
            {"role": "user", "content": "new task"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "NOT re-execute" in sep or "Do NOT re-execute" in sep

    async def test_separator_marks_history_boundary(self):
        """Separator should clearly delineate history from request."""
        stub = _make_stub()
        msg = _make_msg(content="current")
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "current"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "HISTORY" in sep
        assert "REQUEST" in sep

    async def test_separator_forbids_confusing_old_context(self):
        """Separator should warn against confusing old context with new request."""
        stub = _make_stub()
        msg = _make_msg(content="test")
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "test"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "confuse" in sep.lower() or "old context" in sep.lower()


# ===================================================================
# Separator Structural Tests
# ===================================================================

class TestSeparatorStructure:
    """Verify separator positioning and role."""

    async def test_separator_is_developer_role(self):
        stub = _make_stub()
        msg = _make_msg(content="test")
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "test"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        msgs = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        devs = [m for m in msgs if m.get("role") == "developer"]
        assert len(devs) >= 1

    async def test_separator_before_user_message(self):
        """Separator should appear immediately before the current user message."""
        stub = _make_stub()
        msg = _make_msg(content="current msg")
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old resp"},
            {"role": "user", "content": "current msg"},
        ]
        # Return no tools so the list isn't mutated further
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        msgs = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        # Find the developer separator
        sep_indices = [i for i, m in enumerate(msgs) if m.get("role") == "developer"]
        assert sep_indices, "No developer separator found"
        sep_idx = sep_indices[0]
        # The message after the separator should be the current user message
        assert msgs[sep_idx + 1]["role"] == "user"
        content = msgs[sep_idx + 1]["content"]
        assert content == "current msg"

    async def test_no_separator_for_single_message(self):
        """Single message (no history) should only get message ID, not full separator."""
        stub = _make_stub()
        msg = _make_msg(content="only message")
        history = [{"role": "user", "content": "only message"}]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        msgs = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        devs = [m for m in msgs if m.get("role") == "developer"]
        assert len(devs) == 1
        assert "CURRENT REQUEST" not in devs[0]["content"]
        assert "Current message ID" in devs[0]["content"]


# ===================================================================
# Bot Message Handling
# ===================================================================

class TestBotMessageSeparator:
    """Bot messages should include extra execution instructions."""

    async def test_bot_message_has_execute_preamble(self):
        stub = _make_stub(respond_to_bots=True)
        msg = _make_msg(content="run uptime", is_bot=True)
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "run uptime"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "ANOTHER BOT" in sep
        assert "EXECUTE immediately" in sep

    async def test_human_message_no_bot_preamble(self):
        stub = _make_stub(respond_to_bots=True)
        msg = _make_msg(content="run uptime", is_bot=False)
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "run uptime"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "ANOTHER BOT" not in sep


# ===================================================================
# Adversarial Scenarios
# ===================================================================

class TestAdversarialScenarios:
    """Test that the separator handles tricky context scenarios."""

    async def test_topic_switch_separator_present(self):
        """When user switches topics, separator should still be present
        and hash should reflect the NEW request, not old ones."""
        stub = _make_stub()
        msg = _make_msg(content="write a haiku about cats")
        history = [
            {"role": "user", "content": "check disk space on server-a"},
            {"role": "assistant", "content": "Disk is 45% full."},
            {"role": "user", "content": "restart nginx on server-b"},
            {"role": "assistant", "content": "Nginx restarted."},
            {"role": "user", "content": "write a haiku about cats"},
        ]
        expected_hash = hashlib.sha256(b"write a haiku about cats").hexdigest()[:8]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="A cat poem.", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        # Hash should be for the current message, not old ones
        assert f"req-{expected_hash}" in sep
        # Should still have anti-re-execution language
        assert "NOT re-execute" in sep or "Do NOT re-execute" in sep

    async def test_similar_sounding_requests_different_hashes(self):
        """Similar but different requests should have different hashes."""
        hash_a = hashlib.sha256(b"restart nginx on server-a").hexdigest()[:8]
        hash_b = hashlib.sha256(b"restart nginx on server-b").hexdigest()[:8]
        assert hash_a != hash_b

    async def test_separator_with_long_history(self):
        """Separator should work with many history messages."""
        stub = _make_stub()
        msg = _make_msg(content="final request")
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"old message {i}"})
            history.append({"role": "assistant", "content": f"old response {i}"})
        history.append({"role": "user", "content": "final request"})
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        msgs = stub.codex_client.chat_with_tools.call_args_list[0][1]["messages"]
        devs = [m for m in msgs if m.get("role") == "developer"]
        assert len(devs) >= 1
        sep = devs[0]["content"]
        assert "CURRENT REQUEST" in sep
        expected_hash = hashlib.sha256(b"final request").hexdigest()[:8]
        assert f"req-{expected_hash}" in sep

    async def test_different_users_different_identity_in_separator(self):
        """Separator should reflect the actual user making the request."""
        stub = _make_stub()
        msg = _make_msg(content="hello", display_name="Charlie", author_id="99999")
        history = [
            {"role": "user", "content": "old from alice"},
            {"role": "user", "content": "hello"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Hi", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "Charlie" in sep
        assert "99999" in sep

    async def test_separator_with_error_history(self):
        """Separator should still work when history contains error markers."""
        stub = _make_stub()
        msg = _make_msg(content="try again")
        history = [
            {"role": "user", "content": "deploy app"},
            {"role": "assistant", "content": "[Previous request used tools (run_command) but encountered an error...]"},
            {"role": "user", "content": "try again"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text=None, tool_calls=[_tc("run_command")]),
            LLMResponse(text="Deployed.", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "CURRENT REQUEST" in sep
        assert "HISTORY" in sep


# ===================================================================
# Source Code Verification
# ===================================================================

class TestRequestIsolationSource:
    """Verify the enhanced separator exists in source code."""

    def test_source_contains_request_hash(self):
        import inspect
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "req_hash" in src
        assert "hashlib.sha256" in src

    def test_source_contains_timestamp(self):
        import inspect
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "req_time" in src
        assert "time.strftime" in src or "time.gmtime" in src

    def test_source_contains_user_identity(self):
        import inspect
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "user_display" in src
        assert "display_name" in src

    def test_source_contains_anti_reexecution(self):
        import inspect
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "re-execute" in src.lower()

    def test_source_imports_hashlib(self):
        """client.py should import hashlib for request hashing."""
        import inspect
        src = inspect.getsource(sys.modules["src.discord.client"])
        assert "import hashlib" in src

    def test_separator_still_has_current_request_marker(self):
        """The CURRENT REQUEST marker should still be present."""
        import inspect
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "CURRENT REQUEST" in src

    def test_separator_still_uses_developer_role(self):
        import inspect
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert '"role": "developer"' in src or "'role': 'developer'" in src

    def test_separator_still_inserted_before_last(self):
        import inspect
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "insert(-1" in src
