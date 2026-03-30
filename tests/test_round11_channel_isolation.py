"""Tests for Round 11: Per-channel isolation + channel context.

Validates:
- Sessions are keyed per-channel — different channels never share state
- Channel name appears in the context separator for spatial awareness
- Thread context shows parent→thread in separator
- Thread-inherited summaries carry [INHERITED FROM #parent] markers
- No cross-channel context leaking
"""
from __future__ import annotations

import asyncio
import inspect
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

import discord  # noqa: E402
from src.discord.client import HeimdallBot  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402
from src.sessions.manager import Message, SessionManager  # noqa: E402


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
              is_bot=False, channel_id="ch-1", channel_name="general",
              webhook_id=None, is_thread=False, parent_name=None):
    """Create a mock Discord message with channel context.

    If *is_thread* is True, the channel will be a mock discord.Thread with a
    parent channel whose name is *parent_name* (default 'general').
    """
    msg = MagicMock()
    msg.author = MagicMock()
    msg.author.bot = is_bot
    msg.author.id = author_id
    msg.author.display_name = display_name
    msg.author.name = display_name
    msg.author.__str__ = lambda s: display_name

    if is_thread:
        # Make the channel look like a discord.Thread
        ch = MagicMock(spec=discord.Thread)
        ch.id = channel_id
        ch.name = channel_name
        ch.parent = MagicMock()
        ch.parent.id = "parent-ch-1"
        ch.parent.name = parent_name or "general"
    else:
        ch = MagicMock()
        ch.id = channel_id
        ch.name = channel_name
        # Ensure isinstance(ch, discord.Thread) is False
        ch.__class__ = type("TextChannel", (), {})

    ch.typing = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(), __aexit__=AsyncMock(),
    ))
    ch.__str__ = lambda s: f"#{channel_name}"
    msg.channel = ch
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


def _get_dev_content(call_args_list):
    """Extract the developer message content (works for both history and no-history cases)."""
    msgs = call_args_list[0][1]["messages"]
    devs = [m for m in msgs if m.get("role") == "developer"]
    assert devs, "No developer message found"
    return devs[0]["content"]


# ===================================================================
# Per-channel session isolation
# ===================================================================

class TestPerChannelIsolation:
    """Sessions must be strictly keyed by channel_id."""

    def test_different_channels_separate_sessions(self, tmp_dir):
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        mgr.add_message("ch-1", "user", "Hello from channel 1")
        mgr.add_message("ch-2", "user", "Hello from channel 2")

        h1 = mgr.get_history("ch-1")
        h2 = mgr.get_history("ch-2")
        assert len(h1) == 1
        assert len(h2) == 1
        assert "channel 1" in h1[0]["content"]
        assert "channel 2" in h2[0]["content"]

    def test_channel_session_no_cross_contamination(self, tmp_dir):
        """Adding messages to ch-1 must not affect ch-2."""
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        mgr.add_message("ch-1", "user", "msg1")
        mgr.add_message("ch-1", "user", "msg2")
        mgr.add_message("ch-1", "user", "msg3")

        h2 = mgr.get_history("ch-2")
        assert len(h2) == 0  # no messages in ch-2

    def test_channel_reset_only_affects_target(self, tmp_dir):
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        mgr.add_message("ch-1", "user", "msg1")
        mgr.add_message("ch-2", "user", "msg2")
        mgr.reset("ch-1")

        h1 = mgr.get_history("ch-1")
        h2 = mgr.get_history("ch-2")
        assert len(h1) == 0
        assert len(h2) == 1

    def test_channel_summary_isolated(self, tmp_dir):
        """Summary for ch-1 should not appear in ch-2."""
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        s1 = mgr.get_or_create("ch-1")
        s1.summary = "nginx deployed on server-a"
        s2 = mgr.get_or_create("ch-2")

        h2 = mgr.get_history("ch-2")
        for msg in h2:
            assert "nginx" not in msg["content"]

    def test_session_keyed_by_str_channel_id(self, tmp_dir):
        """Sessions use str(channel.id) as key — numeric IDs work."""
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        mgr.add_message("123456789", "user", "numeric id")
        h = mgr.get_history("123456789")
        assert len(h) == 1

    def test_thread_and_parent_separate_sessions(self, tmp_dir):
        """Thread sessions are distinct from parent channel sessions."""
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        mgr.add_message("parent-ch", "user", "parent message")
        mgr.add_message("thread-ch", "user", "thread message")

        hp = mgr.get_history("parent-ch")
        ht = mgr.get_history("thread-ch")
        assert len(hp) == 1
        assert len(ht) == 1
        assert "parent message" in hp[0]["content"]
        assert "thread message" in ht[0]["content"]


# ===================================================================
# Channel context in separator
# ===================================================================

class TestChannelContextInSeparator:
    """Separator should include channel name for spatial awareness."""

    async def test_separator_has_channel_name(self):
        """Regular channel messages include 'Channel: #name'."""
        stub = _make_stub()
        msg = _make_msg(content="check disk", channel_name="ops-alerts")
        history = [
            {"role": "user", "content": "old msg"},
            {"role": "user", "content": "check disk"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "Channel: #ops-alerts" in sep

    async def test_thread_separator_shows_parent_and_thread(self):
        """Thread messages include parent→thread in channel context."""
        stub = _make_stub()
        msg = _make_msg(
            content="check disk", channel_name="fix-nginx",
            is_thread=True, parent_name="devops",
        )
        history = [
            {"role": "user", "content": "old msg"},
            {"role": "user", "content": "check disk"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        assert "#devops" in sep
        assert "thread" in sep.lower()
        assert "fix-nginx" in sep

    async def test_no_history_still_has_channel_context(self):
        """Single-message case (no history) still includes channel context."""
        stub = _make_stub()
        msg = _make_msg(content="hello", channel_name="general")
        history = [{"role": "user", "content": "hello"}]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Hi", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        dev = _get_dev_content(stub.codex_client.chat_with_tools.call_args_list)
        assert "Channel: #general" in dev

    async def test_channel_context_precedes_request_boundary(self):
        """Channel context line comes before the HISTORY ABOVE separator."""
        stub = _make_stub()
        msg = _make_msg(content="deploy app", channel_name="deployments")
        history = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "deploy app"},
        ]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Done", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
        ch_pos = sep.find("Channel:")
        hist_pos = sep.find("HISTORY ABOVE")
        assert ch_pos < hist_pos, "Channel context should appear before history boundary"


# ===================================================================
# Thread context inheritance markers
# ===================================================================

class TestThreadInheritedMarkers:
    """Thread-inherited context must be marked so the LLM knows its origin."""

    def _make_bot_stub_for_handle_message(self):
        """Create a HeimdallBot stub for _handle_message-level tests."""
        stub = MagicMock()
        stub.sessions = SessionManager(
            max_history=100, max_age_hours=1,
            persist_dir="/tmp/test_sessions_marker",
        )
        stub._channel_locks = {}
        stub._handle_message_inner = AsyncMock()
        return stub

    async def test_thread_inherits_summary_with_marker(self):
        """Thread inheriting parent summary gets [INHERITED FROM #parent]."""
        stub = self._make_bot_stub_for_handle_message()
        # Set up parent session with summary
        parent = stub.sessions.get_or_create("parent-ch")
        parent.summary = "nginx deployed on server-a"

        # Create a mock thread message
        msg = MagicMock()
        msg.channel = MagicMock(spec=discord.Thread)
        msg.channel.id = "thread-ch"
        msg.channel.parent = MagicMock()
        msg.channel.parent.id = "parent-ch"
        msg.channel.parent.name = "ops"

        await HeimdallBot._handle_message(
            stub, msg, "thread msg", image_blocks=[],
        )

        thread_session = stub.sessions.get_or_create("thread-ch")
        assert "[INHERITED FROM #ops]" in thread_session.summary

    async def test_thread_inherits_parent_context_with_marker(self):
        """Thread inheriting parent messages gets [INHERITED FROM #parent] marker."""
        stub = self._make_bot_stub_for_handle_message()
        parent = stub.sessions.get_or_create("parent-ch")
        parent.messages.append(Message(role="user", content="check nginx status"))
        parent.messages.append(Message(role="assistant", content="nginx is running"))

        msg = MagicMock()
        msg.channel = MagicMock(spec=discord.Thread)
        msg.channel.id = "thread-ch-2"
        msg.channel.parent = MagicMock()
        msg.channel.parent.id = "parent-ch"
        msg.channel.parent.name = "infra"

        await HeimdallBot._handle_message(
            stub, msg, "fix it", image_blocks=[],
        )

        thread_session = stub.sessions.get_or_create("thread-ch-2")
        assert "[INHERITED FROM #infra]" in thread_session.summary
        assert "Parent channel context" in thread_session.summary

    async def test_thread_no_inheritance_when_parent_empty(self):
        """Empty parent session → no inheritance, no markers."""
        stub = self._make_bot_stub_for_handle_message()
        # Parent has empty session (no messages, no summary)
        stub.sessions.get_or_create("parent-empty")

        msg = MagicMock()
        msg.channel = MagicMock(spec=discord.Thread)
        msg.channel.id = "thread-empty"
        msg.channel.parent = MagicMock()
        msg.channel.parent.id = "parent-empty"
        msg.channel.parent.name = "empty-channel"

        await HeimdallBot._handle_message(
            stub, msg, "hello", image_blocks=[],
        )

        thread_session = stub.sessions.get_or_create("thread-empty")
        assert not thread_session.summary

    async def test_thread_inherits_both_summary_and_messages(self):
        """Thread with parent that has both summary and messages."""
        stub = self._make_bot_stub_for_handle_message()
        parent = stub.sessions.get_or_create("parent-both")
        parent.summary = "[Topics: dns] Configured dns on server-b"
        parent.messages.append(Message(role="user", content="set up dns"))

        msg = MagicMock()
        msg.channel = MagicMock(spec=discord.Thread)
        msg.channel.id = "thread-both"
        msg.channel.parent = MagicMock()
        msg.channel.parent.id = "parent-both"
        msg.channel.parent.name = "networking"

        await HeimdallBot._handle_message(
            stub, msg, "thread task", image_blocks=[],
        )

        thread_session = stub.sessions.get_or_create("thread-both")
        # Should have inherited tag
        assert "[INHERITED FROM #networking]" in thread_session.summary
        # Should contain original summary content
        assert "dns" in thread_session.summary.lower()

    async def test_second_message_in_thread_no_reinheritance(self):
        """Second message in a thread should not re-inherit from parent."""
        stub = self._make_bot_stub_for_handle_message()
        parent = stub.sessions.get_or_create("parent-re")
        parent.summary = "some parent context"
        parent.messages.append(Message(role="user", content="parent msg"))

        # First message seeds the thread
        msg1 = MagicMock()
        msg1.channel = MagicMock(spec=discord.Thread)
        msg1.channel.id = "thread-re"
        msg1.channel.parent = MagicMock()
        msg1.channel.parent.id = "parent-re"
        msg1.channel.parent.name = "main"

        await HeimdallBot._handle_message(stub, msg1, "first", image_blocks=[])

        # Thread session now has messages (from _handle_message_inner being called)
        thread = stub.sessions.get_or_create("thread-re")
        original_summary = thread.summary

        # Simulate that _handle_message_inner added a message
        thread.messages.append(Message(role="user", content="first"))

        # Second message should not re-inherit (thread already has messages)
        msg2 = MagicMock()
        msg2.channel = msg1.channel  # same thread

        await HeimdallBot._handle_message(stub, msg2, "second", image_blocks=[])

        # Summary should be unchanged
        assert thread.summary == original_summary


# ===================================================================
# Cross-channel isolation verification
# ===================================================================

class TestCrossChannelIsolation:
    """Verify no cross-channel data leaking."""

    async def test_task_history_per_channel(self, tmp_dir):
        """get_task_history returns only the target channel's messages."""
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        mgr.add_message("ch-a", "user", "deploy nginx on server-a")
        mgr.add_message("ch-b", "user", "check dns on server-b")

        ha = await mgr.get_task_history("ch-a")
        hb = await mgr.get_task_history("ch-b")

        # Each should only contain its own message
        all_a = " ".join(m["content"] for m in ha)
        all_b = " ".join(m["content"] for m in hb)
        assert "nginx" in all_a
        assert "server-b" not in all_a
        assert "dns" in all_b
        assert "server-a" not in all_b

    async def test_compaction_per_channel(self, tmp_dir):
        """Compaction only affects the target channel."""
        mgr = SessionManager(max_history=10, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        mgr.set_compaction_fn(AsyncMock(return_value="Compacted ch-a"))

        # Fill ch-a past compaction threshold
        for i in range(50):
            mgr.add_message("ch-a", "user", f"ch-a msg {i}")

        # ch-b has a few messages
        mgr.add_message("ch-b", "user", "ch-b msg")

        await mgr.get_task_history("ch-a")

        # ch-b should be unaffected
        hb = mgr.get_history("ch-b")
        assert len(hb) == 1
        assert "ch-b msg" in hb[0]["content"]

    def test_detect_topic_change_per_channel(self, tmp_dir):
        """Topic change detection only looks at the target channel."""
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        mgr.add_message("ch-a", "user", "deploy nginx")
        mgr.add_message("ch-a", "user", "restart nginx")

        # Query about dns to ch-b (no history) → no topic change
        result = mgr.detect_topic_change("ch-b", "configure dns")
        assert result["is_topic_change"] is False

    def test_scrub_secrets_only_target_channel(self, tmp_dir):
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        mgr.add_message("ch-1", "user", "password=secret123")
        mgr.add_message("ch-2", "user", "password=secret123")

        mgr.scrub_secrets("ch-1", "password=secret123")

        h1 = mgr.get_history("ch-1")
        h2 = mgr.get_history("ch-2")
        assert len(h1) == 0
        assert len(h2) == 1


# ===================================================================
# Source code verification
# ===================================================================

class TestSourceVerification:
    """Verify implementation details are present in source."""

    def test_separator_has_channel_ctx_variable(self):
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "channel_ctx" in src

    def test_separator_builds_channel_line(self):
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "Channel:" in src

    def test_thread_check_in_separator(self):
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "_is_thread" in src

    def test_thread_inheritance_has_inherited_tag(self):
        src = inspect.getsource(HeimdallBot._handle_message)
        assert "INHERITED FROM" in src

    def test_thread_inheritance_includes_parent_name(self):
        src = inspect.getsource(HeimdallBot._handle_message)
        assert "parent_name" in src

    def test_no_history_case_has_channel_ctx(self):
        """Even single-message case includes channel context."""
        src = inspect.getsource(HeimdallBot._process_with_tools)
        assert "channel_ctx" in src

    def test_session_keyed_by_channel_id(self):
        """SessionManager uses channel_id as dict key."""
        src = inspect.getsource(SessionManager.get_or_create)
        assert "channel_id" in src

    def test_channel_locks_keyed_by_channel_id(self):
        src = inspect.getsource(HeimdallBot._handle_message)
        assert "_channel_locks" in src


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases for channel isolation and context."""

    async def test_dm_channel_has_no_name_fallback(self):
        """DM channels may not have a name — should fallback to ID."""
        stub = _make_stub()
        msg = _make_msg(content="hello", channel_id="dm-123")
        # Simulate a DM channel without a name attribute
        msg.channel.name = None
        msg.channel.__class__ = type("DMChannel", (), {})
        history = [{"role": "user", "content": "hello"}]
        stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
            LLMResponse(text="Hi", tool_calls=[]),
        ])
        await HeimdallBot._process_with_tools(stub, msg, history)
        dev = _get_dev_content(stub.codex_client.chat_with_tools.call_args_list)
        # Should include channel ID as fallback
        assert "Channel:" in dev
        assert "dm-123" in dev

    async def test_many_channels_no_leaking(self, tmp_dir):
        """10 channels, each with unique content — no cross-contamination."""
        mgr = SessionManager(max_history=100, max_age_hours=1,
                             persist_dir=str(tmp_dir / "sessions"))
        for i in range(10):
            mgr.add_message(f"ch-{i}", "user", f"unique-content-{i}")

        for i in range(10):
            h = mgr.get_history(f"ch-{i}")
            assert len(h) == 1
            assert f"unique-content-{i}" in h[0]["content"]
            # Must NOT contain content from other channels
            for j in range(10):
                if j != i:
                    assert f"unique-content-{j}" not in h[0]["content"]

    async def test_persist_and_reload_isolated(self, tmp_dir):
        """Sessions persist and reload with isolation intact."""
        mgr1 = SessionManager(max_history=100, max_age_hours=1,
                              persist_dir=str(tmp_dir / "sessions"))
        mgr1.add_message("ch-x", "user", "data-x")
        mgr1.add_message("ch-y", "user", "data-y")
        mgr1.save()

        mgr2 = SessionManager(max_history=100, max_age_hours=1,
                              persist_dir=str(tmp_dir / "sessions"))
        mgr2.load()

        hx = mgr2.get_history("ch-x")
        hy = mgr2.get_history("ch-y")
        assert len(hx) == 1
        assert len(hy) == 1
        assert "data-x" in hx[0]["content"]
        assert "data-y" in hy[0]["content"]
        assert "data-y" not in hx[0]["content"]

    async def test_separator_channel_different_per_channel(self):
        """Two messages to different channels should have different Channel: lines."""
        results = {}
        for ch_name in ("ops", "dev"):
            stub = _make_stub()
            msg = _make_msg(content="test", channel_name=ch_name)
            history = [
                {"role": "user", "content": "old"},
                {"role": "user", "content": "test"},
            ]
            stub.codex_client.chat_with_tools = AsyncMock(side_effect=[
                LLMResponse(text="ok", tool_calls=[]),
            ])
            await HeimdallBot._process_with_tools(stub, msg, history)
            sep = _get_separator(stub.codex_client.chat_with_tools.call_args_list)
            results[ch_name] = sep

        assert f"Channel: #ops" in results["ops"]
        assert f"Channel: #dev" in results["dev"]
        assert "dev" not in results["ops"]
        assert "ops" not in results["dev"]
