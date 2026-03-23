"""Tests for Round 19 bug fixes.

Fix 1: Circuit breaker recovery embed now passes view=cancel_view to keep
       the cancel button visible during the recovery wait period.

Fix 2: ToolLoopCancelView.disable() now calls self.stop() to unregister
       the view from discord.py's event listener when the tool loop ends.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    LokiBot,
    MAX_TOOL_ITERATIONS,
    ToolLoopCancelView,
)
from src.llm.circuit_breaker import CircuitOpenError  # noqa: E402
from src.llm.types import LLMResponse, ToolCall  # noqa: E402


@pytest.fixture(autouse=True)
def _no_approval():
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal LokiBot stub."""
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
    stub.config.discord.respond_to_bots = False
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
        {"name": "check_disk", "description": "Disk", "input_schema": {"type": "object", "properties": {}}},
    ])
    stub.permissions = MagicMock()
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub._track_recent_action = MagicMock()
    stub._build_tool_progress_embed = LokiBot._build_tool_progress_embed
    stub._build_partial_completion_report = LokiBot._build_partial_completion_report
    return stub


def _make_message():
    """Create a mock Discord message."""
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = "test-chan"
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None),
    ))
    embed_msg = AsyncMock()
    msg.channel.send = AsyncMock(return_value=embed_msg)
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    return msg


def _tool_response(text, tools=None):
    tc = tools or []
    return LLMResponse(
        text=text or "",
        tool_calls=tc,
        stop_reason="tool_use" if tc else "end_turn",
    )


# ---------------------------------------------------------------------------
# Fix 1: Cancel button stays visible during circuit breaker recovery
# ---------------------------------------------------------------------------

class TestCancelViewDuringCircuitBreakerRecovery:
    """The cancel button should remain on the embed during CB recovery wait."""

    @pytest.mark.asyncio
    async def test_recovery_embed_edit_passes_view(self):
        """When CB triggers and embed is updated with recovery message,
        view=cancel_view should be passed to keep the cancel button visible."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Step 1: tool call (embed created with cancel view)
        # Step 2: circuit breaker opens (embed should keep cancel button)
        # Retry: succeeds
        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                _tool_response("Checking", [ToolCall(id="tc1", name="check_disk", input={})]),
                CircuitOpenError("codex_api", retry_after=0.01),
                _tool_response("Done!"),
            ]
        )

        embed_msg = AsyncMock()
        msg.channel.send = AsyncMock(return_value=embed_msg)

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock):
            text, already_sent, is_error, tools_used, handoff = (
                await LokiBot._process_with_tools(
                    stub, msg, [{"role": "user", "content": "check disk"}]
                )
            )

        assert not is_error
        assert text == "Done!"

        # Find the edit call that shows the recovery message
        edit_calls = embed_msg.edit.call_args_list
        recovery_edits = [
            c for c in edit_calls
            if "recovering" in str(getattr(c.kwargs.get("embed"), "description", "")).lower()
            or "retrying" in str(getattr(c.kwargs.get("embed"), "description", "")).lower()
        ]
        assert len(recovery_edits) >= 1, "Expected at least one recovery embed edit"

        # The recovery edit should include view= parameter (cancel button kept)
        for rc in recovery_edits:
            assert "view" in rc.kwargs, (
                "Recovery embed edit should pass view=cancel_view to keep cancel button visible"
            )
            assert rc.kwargs["view"] is not None

    @pytest.mark.asyncio
    async def test_cancel_during_circuit_breaker_wait_detected(self):
        """If cancel is pressed during CB wait, it should be detected at next iteration."""
        stub = _make_bot_stub()
        msg = _make_message()

        # Step 1: tool call
        # Step 2: CB opens — during sleep, cancel is pressed
        # Retry succeeds, but cancel is detected at iteration 3 start
        call_count = 0

        async def _chat_with_tools_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("Checking", [ToolCall(id="tc1", name="check_disk", input={})])
            elif call_count == 2:
                raise CircuitOpenError("codex_api", retry_after=0.01)
            else:
                # Retry succeeds — but cancel should catch at next iteration
                return _tool_response("Continuing", [ToolCall(id="tc2", name="run_command", input={})])

        stub.codex_client.chat_with_tools = AsyncMock(side_effect=_chat_with_tools_side_effect)

        embed_msg = AsyncMock()
        cancel_view_ref = []

        original_send = msg.channel.send

        async def _capture_send(**kwargs):
            result = await original_send(**kwargs)
            if "view" in kwargs and isinstance(kwargs["view"], ToolLoopCancelView):
                cancel_view_ref.append(kwargs["view"])
            return result

        msg.channel.send = AsyncMock(side_effect=_capture_send)
        msg.channel.send.return_value = embed_msg

        # During CB sleep, set the cancel event
        async def _sleep_and_cancel(secs):
            if cancel_view_ref:
                cancel_view_ref[0]._cancel_event.set()

        with patch("src.discord.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = _sleep_and_cancel
            text, already_sent, is_error, tools_used, handoff = (
                await LokiBot._process_with_tools(
                    stub, msg, [{"role": "user", "content": "check disk"}]
                )
            )

        # The cancel should be detected — task should report cancellation or partial completion
        # After CB recovery, the tool call response is received, then at next iteration
        # the cancel check fires
        assert is_error or "cancel" in text.lower() or "cancelled" in text.lower()


# ---------------------------------------------------------------------------
# Fix 2: disable() calls stop() to clean up discord.py event listener
# ---------------------------------------------------------------------------

class TestDisableCallsStop:
    """ToolLoopCancelView.disable() should call self.stop() to unregister
    the view from discord.py's internal listener registry."""

    def test_disable_calls_stop(self):
        """disable() should call stop() to clean up the event listener."""
        view = ToolLoopCancelView(allowed_user_ids=["user-1"])
        with patch.object(view, "stop") as mock_stop:
            view.disable()
            mock_stop.assert_called_once()

    def test_disable_disables_buttons(self):
        """disable() still disables all buttons (original behavior preserved)."""
        view = ToolLoopCancelView(allowed_user_ids=["user-1"])
        view.disable()
        for item in view.children:
            if isinstance(item, discord.ui.Button):
                assert item.disabled

    def test_double_stop_is_safe(self):
        """Calling disable() after the view is already stopped doesn't crash."""
        view = ToolLoopCancelView(allowed_user_ids=["user-1"])
        view.stop()  # First stop (e.g., from cancel button callback)
        view.disable()  # Second stop via disable() — should not raise

    @pytest.mark.asyncio
    async def test_view_disabled_on_normal_completion(self):
        """After a normal tool loop completion, the cancel view buttons should be disabled."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                _tool_response("Working", [ToolCall(id="tc1", name="check_disk", input={})]),
                _tool_response("Done!"),
            ]
        )

        embed_msg = AsyncMock()
        view_ref = []

        async def _capture_send(**kwargs):
            if "view" in kwargs and isinstance(kwargs["view"], ToolLoopCancelView):
                view_ref.append(kwargs["view"])
            return embed_msg

        msg.channel.send = AsyncMock(side_effect=_capture_send)

        text, already_sent, is_error, tools_used, handoff = (
            await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check disk"}]
            )
        )

        assert not is_error
        assert text == "Done!"
        assert len(view_ref) == 1, "Expected cancel view to be attached to embed"
        # All buttons should be disabled after completion
        for item in view_ref[0].children:
            if isinstance(item, discord.ui.Button):
                assert item.disabled, "Cancel button should be disabled after completion"

    @pytest.mark.asyncio
    async def test_view_disabled_on_error(self):
        """After a tool loop error, the cancel view buttons should be disabled."""
        stub = _make_bot_stub()
        msg = _make_message()

        stub.codex_client.chat_with_tools = AsyncMock(
            side_effect=[
                _tool_response("Working", [ToolCall(id="tc1", name="check_disk", input={})]),
                RuntimeError("API down"),
            ]
        )

        embed_msg = AsyncMock()
        view_ref = []

        async def _capture_send(**kwargs):
            if "view" in kwargs and isinstance(kwargs["view"], ToolLoopCancelView):
                view_ref.append(kwargs["view"])
            return embed_msg

        msg.channel.send = AsyncMock(side_effect=_capture_send)

        text, already_sent, is_error, tools_used, handoff = (
            await LokiBot._process_with_tools(
                stub, msg, [{"role": "user", "content": "check disk"}]
            )
        )

        assert is_error
        assert len(view_ref) == 1, "Expected cancel view to be attached to embed"
        for item in view_ref[0].children:
            if isinstance(item, discord.ui.Button):
                assert item.disabled, "Cancel button should be disabled after error"
