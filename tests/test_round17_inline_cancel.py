"""Tests for inline task cancellation via cancel button on progress embed.

Round 17: A cancel button is attached to the progress embed. When pressed,
an asyncio.Event is set and the tool loop checks it at each iteration boundary,
cleanly stopping with a partial completion report.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    AnsiblexBot,
    MAX_TOOL_ITERATIONS,
    ToolLoopCancelView,
)
from src.llm.types import LLMResponse, ToolCall  # noqa: E402

# Patch requires_approval to return False for test tools
@pytest.fixture(autouse=True)
def _no_approval():
    """Patch requires_approval to return False for all tools in all tests."""
    with patch("src.discord.client.requires_approval", return_value=False):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub():
    """Minimal AnsiblexBot stub for cancellation tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._last_tool_use = {}
    stub._system_prompt = "test prompt"
    stub._pending_files = {}
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["user-1"]
    stub.config.tools.approval_timeout_seconds = 30
    stub.sessions = MagicMock()
    stub.codex_client = MagicMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.requires_approval = MagicMock(return_value=None)
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
    stub._build_tool_progress_embed = AnsiblexBot._build_tool_progress_embed
    stub._build_partial_completion_report = AnsiblexBot._build_partial_completion_report
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
    # Return a mock that also supports .edit(embed=..., view=...)
    embed_msg = AsyncMock()
    msg.channel.send = AsyncMock(return_value=embed_msg)
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    return msg


def _tool_response(text: str | None, tools: list[ToolCall] | None = None) -> LLMResponse:
    """Build an LLMResponse with optional tool calls."""
    tc = tools or []
    return LLMResponse(
        text=text or "",
        tool_calls=tc,
        stop_reason="tool_use" if tc else "end_turn",
    )


def _tool_call(name: str = "run_command", input_: dict | None = None) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, input=input_ or {"command": "echo hi"})


# ===========================================================================
# ToolLoopCancelView unit tests
# ===========================================================================

class TestToolLoopCancelView:
    """Unit tests for the ToolLoopCancelView class."""

    def test_initial_state_not_cancelled(self):
        view = ToolLoopCancelView(allowed_user_ids=["123"])
        assert view.is_cancelled is False

    def test_cancel_event_sets_is_cancelled(self):
        view = ToolLoopCancelView(allowed_user_ids=["123"])
        view._cancel_event.set()
        assert view.is_cancelled is True

    def test_disable_disables_buttons(self):
        view = ToolLoopCancelView(allowed_user_ids=["123"])
        view.disable()
        for item in view.children:
            if isinstance(item, discord.ui.Button):
                assert item.disabled is True

    def test_allowed_users_stored(self):
        view = ToolLoopCancelView(allowed_user_ids=["111", "222"])
        assert view._allowed == {"111", "222"}

    def test_default_timeout(self):
        view = ToolLoopCancelView(allowed_user_ids=["123"])
        assert view.timeout == 600


# ===========================================================================
# Cancel button interaction tests
# ===========================================================================

class TestCancelButtonInteraction:
    """Test the cancel button's interaction handler."""

    @pytest.mark.asyncio
    async def test_authorized_user_cancels(self):
        view = ToolLoopCancelView(allowed_user_ids=["123"])
        interaction = AsyncMock()
        interaction.user = MagicMock()
        interaction.user.id = 123

        # The decorated button callback is on the Button item
        button = view.children[0]
        await button.callback(interaction)

        assert view.is_cancelled is True
        interaction.response.edit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected(self):
        view = ToolLoopCancelView(allowed_user_ids=["123"])
        interaction = AsyncMock()
        interaction.user = MagicMock()
        interaction.user.id = 999

        button = view.children[0]
        await button.callback(interaction)

        assert view.is_cancelled is False
        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args
        assert "not authorized" in call_args[0][0]
        assert call_args[1].get("ephemeral") is True


# ===========================================================================
# Integration: Cancellation during tool loop
# ===========================================================================

class TestCancelDuringToolLoop:
    """Integration tests: cancel button pressed during tool loop execution."""

    @pytest.mark.asyncio
    async def test_cancel_after_one_step_reports_partial_completion(self):
        """Cancel pressed after 1 completed step — report shows that step."""
        bot = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def mock_chat_with_tools(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("Checking disk.", [_tool_call("check_disk")])
            return _tool_response("Done.")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)

        # Intercept channel.send to capture the cancel view and set cancel
        async def send_with_cancel_capture(*args, **kwargs):
            result = AsyncMock()
            if "view" in kwargs and kwargs["view"] is not None:
                kwargs["view"]._cancel_event.set()
            return result

        msg.channel.send = AsyncMock(side_effect=send_with_cancel_capture)

        text, already_sent, is_error, tools_used, handoff = await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "check disk"}],
        )

        assert is_error is True
        assert "cancelled by user" in text.lower()
        assert "check_disk" in tools_used

    @pytest.mark.asyncio
    async def test_cancel_before_any_tool_not_possible(self):
        """Cancel button doesn't exist before first tool call, so normal completion."""
        bot = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("Plan.", [_tool_call()])
            return _tool_response("Done.")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        # Don't set cancel — verify normal completion
        text, _, is_error, _, _ = await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "test"}],
        )

        assert is_error is False
        assert "cancelled" not in text.lower()

    @pytest.mark.asyncio
    async def test_cancel_after_two_steps(self):
        """Cancel pressed after 2 steps — partial report shows both."""
        bot = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _tool_response(
                    f"Step {call_count}.",
                    [_tool_call("check_disk" if call_count == 1 else "run_command")],
                )
            return _tool_response("Final.")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        embed_msg = AsyncMock()
        cancel_views = []

        async def send_with_view(*args, **kwargs):
            if "view" in kwargs and kwargs["view"] is not None:
                cancel_views.append(kwargs["view"])
            return embed_msg

        msg.channel.send = AsyncMock(side_effect=send_with_view)

        # Set cancel after step 2 completes (3rd edit: step1 done, step2 running, step2 done)
        edit_count = 0

        async def edit_and_cancel(*args, **kwargs):
            nonlocal edit_count
            edit_count += 1
            # Edit 3 = step 2 done. After that, loop will iterate and check cancel.
            if edit_count == 3 and cancel_views:
                cancel_views[0]._cancel_event.set()

        embed_msg.edit = AsyncMock(side_effect=edit_and_cancel)

        text, _, is_error, tools_used, _ = await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "test"}],
        )

        assert is_error is True
        assert "cancelled by user" in text.lower()
        assert "check_disk" in tools_used
        assert "run_command" in tools_used
        assert "Partial completion" in text

    @pytest.mark.asyncio
    async def test_cancel_is_error_true(self):
        """Cancelled tasks return is_error=True for checkpoint-save."""
        bot = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("Go.", [_tool_call()])
            return _tool_response("Final.")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        async def send_and_cancel(*args, **kwargs):
            result = AsyncMock()
            if "view" in kwargs and kwargs["view"] is not None:
                kwargs["view"]._cancel_event.set()
            return result

        msg.channel.send = AsyncMock(side_effect=send_and_cancel)

        text, _, is_error, _, handoff = await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "test"}],
        )

        assert is_error is True
        assert handoff is False


# ===========================================================================
# Embed updates on cancellation
# ===========================================================================

class TestCancelEmbedUpdates:
    """Test that the progress embed is properly updated on cancel."""

    @pytest.mark.asyncio
    async def test_embed_shows_cancelled_footer(self):
        """On cancel, embed footer shows 'Cancelled by user.'."""
        bot = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("Check.", [_tool_call("check_disk")])
            return _tool_response("Final.")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        embed_msg = AsyncMock()
        cancel_views = []

        async def send_with_view(*args, **kwargs):
            if "view" in kwargs and kwargs["view"] is not None:
                cancel_views.append(kwargs["view"])
            return embed_msg

        msg.channel.send = AsyncMock(side_effect=send_with_view)

        # Set cancel after first step completes
        edit_count = 0

        async def edit_and_cancel(*args, **kwargs):
            nonlocal edit_count
            edit_count += 1
            if edit_count == 1 and cancel_views:
                cancel_views[0]._cancel_event.set()

        embed_msg.edit = AsyncMock(side_effect=edit_and_cancel)

        text, _, is_error, _, _ = await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "test"}],
        )

        assert is_error is True
        # Check that the last embed edit includes the cancel footer
        last_edit = embed_msg.edit.call_args_list[-1]
        if "embed" in last_edit.kwargs:
            embed = last_edit.kwargs["embed"]
            assert "Cancelled by user" in embed.description

    @pytest.mark.asyncio
    async def test_embed_color_is_red_on_cancel(self):
        """On cancel, embed color is red (error color)."""
        bot = _make_bot_stub()
        msg = _make_message()

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("Go.", [_tool_call()])
            return _tool_response("Done.")

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=mock_chat)

        embed_msg = AsyncMock()
        cancel_views = []

        async def send_with_view(*args, **kwargs):
            if "view" in kwargs and kwargs["view"] is not None:
                cancel_views.append(kwargs["view"])
            return embed_msg

        msg.channel.send = AsyncMock(side_effect=send_with_view)

        edit_count = 0

        async def edit_and_cancel(*args, **kwargs):
            nonlocal edit_count
            edit_count += 1
            if edit_count == 1 and cancel_views:
                cancel_views[0]._cancel_event.set()

        embed_msg.edit = AsyncMock(side_effect=edit_and_cancel)

        await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "test"}],
        )

        # Last edit should have red color embed
        last_edit = embed_msg.edit.call_args_list[-1]
        if "embed" in last_edit.kwargs:
            embed = last_edit.kwargs["embed"]
            assert embed.color == discord.Color.red()


# ===========================================================================
# Cancel view attached to embed
# ===========================================================================

class TestCancelViewAttachedToEmbed:
    """Test that the cancel view is attached to the progress embed."""

    @pytest.mark.asyncio
    async def test_cancel_view_sent_with_first_embed(self):
        """First embed send includes a ToolLoopCancelView."""
        bot = _make_bot_stub()
        msg = _make_message()

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_response("Plan.", [_tool_call()]),
            _tool_response("Done."),
        ])

        embed_msg = AsyncMock()
        msg.channel.send = AsyncMock(return_value=embed_msg)

        await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "test"}],
        )

        # channel.send should have been called with view kwarg
        send_call = msg.channel.send.call_args
        assert "view" in send_call.kwargs
        assert isinstance(send_call.kwargs["view"], ToolLoopCancelView)

    @pytest.mark.asyncio
    async def test_cancel_view_disabled_on_completion(self):
        """Cancel view buttons are disabled when the loop completes."""
        bot = _make_bot_stub()
        msg = _make_message()

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_response("Plan.", [_tool_call()]),
            _tool_response("Done."),
        ])

        embed_msg = AsyncMock()
        views_sent = []

        async def capture_send(*args, **kwargs):
            if "view" in kwargs:
                views_sent.append(kwargs["view"])
            return embed_msg

        msg.channel.send = AsyncMock(side_effect=capture_send)

        await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "test"}],
        )

        assert len(views_sent) == 1
        view = views_sent[0]
        # Buttons should be disabled after completion
        for item in view.children:
            if isinstance(item, discord.ui.Button):
                assert item.disabled is True

    @pytest.mark.asyncio
    async def test_no_cancel_view_for_text_only_response(self):
        """No cancel view when there are no tool calls."""
        bot = _make_bot_stub()
        msg = _make_message()

        bot.codex_client.chat_with_tools = AsyncMock(
            return_value=_tool_response("Just text."),
        )

        await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "hello"}],
        )

        # channel.send should NOT have been called with a ToolLoopCancelView
        if msg.channel.send.called:
            for call in msg.channel.send.call_args_list:
                view = call.kwargs.get("view")
                assert not isinstance(view, ToolLoopCancelView)


# ===========================================================================
# No regression
# ===========================================================================

class TestCancelNoRegression:
    """Ensure cancel mechanism doesn't break normal operations."""

    @pytest.mark.asyncio
    async def test_normal_completion_unaffected(self):
        """Tool loop completes normally when cancel is not pressed."""
        bot = _make_bot_stub()
        msg = _make_message()

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_response("Step 1.", [_tool_call("check_disk")]),
            _tool_response("Step 2.", [_tool_call("run_command")]),
            _tool_response("All done."),
        ])

        embed_msg = AsyncMock()
        msg.channel.send = AsyncMock(return_value=embed_msg)

        text, _, is_error, tools_used, _ = await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "test"}],
        )

        assert is_error is False
        assert text == "All done."
        assert "check_disk" in tools_used
        assert "run_command" in tools_used

    @pytest.mark.asyncio
    async def test_cancel_view_not_cancelled_by_default(self):
        """The cancel view is not in cancelled state by default."""
        bot = _make_bot_stub()
        msg = _make_message()

        bot.codex_client.chat_with_tools = AsyncMock(side_effect=[
            _tool_response("Go.", [_tool_call()]),
            _tool_response("Done."),
        ])

        views_captured = []
        embed_msg = AsyncMock()

        async def capture_send(*args, **kwargs):
            if "view" in kwargs and kwargs["view"] is not None:
                views_captured.append(kwargs["view"])
            return embed_msg

        msg.channel.send = AsyncMock(side_effect=capture_send)

        await AnsiblexBot._process_with_tools(
            bot, msg, [{"role": "user", "content": "test"}],
        )

        assert len(views_captured) == 1
        # The cancel event should not be set (user didn't press the button)
        assert views_captured[0]._cancel_event.is_set() is False
