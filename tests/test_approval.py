"""Tests for discord/approval.py — button-based approval for tool execution.

Covers:
- ApprovalView:
  - Initialization with allowed users and timeout
  - approve_button: authorized vs unauthorized user
  - deny_button: authorized vs unauthorized user
  - wait_for_result: approval, denial, and timeout
  - on_timeout: sets event
  - disable_all: disables all buttons
- request_approval:
  - Sends embed with tool info, waits for approval
  - Approved flow: green embed, returns True
  - Denied flow: red embed, returns False
  - Timeout flow: red embed with timeout text, returns False
  - Handles long parameter values (truncation)
  - Handles very long descriptions (>4000 chars)
  - Handles edit failure gracefully
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.approval import ApprovalView, request_approval  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_interaction(user_id="12345"):
    """Create a mock Discord interaction."""
    interaction = MagicMock()
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    return interaction


def _get_button(view: ApprovalView, label: str) -> discord.ui.Button:
    """Find a button in the view by label."""
    for item in view.children:
        if isinstance(item, discord.ui.Button) and item.label == label:
            return item
    raise ValueError(f"No button with label '{label}' found")


def _make_channel():
    """Create a mock messageable channel."""
    ch = AsyncMock()
    msg = AsyncMock()
    msg.edit = AsyncMock()
    ch.send = AsyncMock(return_value=msg)
    return ch, msg


# ---------------------------------------------------------------------------
# ApprovalView — initialization
# ---------------------------------------------------------------------------

class TestApprovalViewInit:
    def test_default_state(self):
        """ApprovalView starts unapproved with unset event."""
        view = ApprovalView(allowed_users=["123"], timeout=60)
        assert view._allowed_users == ["123"]
        assert view._approved is False
        assert not view._event.is_set()

    def test_custom_timeout(self):
        """Custom timeout is passed to parent View."""
        view = ApprovalView(allowed_users=["123"], timeout=30)
        assert view.timeout == 30

    def test_multiple_allowed_users(self):
        """Supports multiple allowed users."""
        view = ApprovalView(allowed_users=["123", "456", "789"])
        assert len(view._allowed_users) == 3


# ---------------------------------------------------------------------------
# ApprovalView — approve_button
# ---------------------------------------------------------------------------

class TestApproveButton:
    async def test_authorized_user_approves(self):
        """Authorized user clicking Approve sets approved=True."""
        view = ApprovalView(allowed_users=["12345"], timeout=60)
        interaction = _make_interaction(user_id="12345")
        btn = _get_button(view, "Approve")

        await btn.callback(interaction)

        assert view._approved is True
        assert view._event.is_set()
        interaction.response.defer.assert_called_once()

    async def test_unauthorized_user_rejected(self):
        """Unauthorized user gets ephemeral rejection message."""
        view = ApprovalView(allowed_users=["12345"], timeout=60)
        interaction = _make_interaction(user_id="99999")
        btn = _get_button(view, "Approve")

        await btn.callback(interaction)

        assert view._approved is False
        assert not view._event.is_set()
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert "not authorized" in call_kwargs[0][0].lower()
        assert call_kwargs[1]["ephemeral"] is True


# ---------------------------------------------------------------------------
# ApprovalView — deny_button
# ---------------------------------------------------------------------------

class TestDenyButton:
    async def test_authorized_user_denies(self):
        """Authorized user clicking Deny sets approved=False."""
        view = ApprovalView(allowed_users=["12345"], timeout=60)
        interaction = _make_interaction(user_id="12345")
        btn = _get_button(view, "Deny")

        await btn.callback(interaction)

        assert view._approved is False
        assert view._event.is_set()
        interaction.response.defer.assert_called_once()

    async def test_unauthorized_user_rejected(self):
        """Unauthorized user gets ephemeral rejection on deny."""
        view = ApprovalView(allowed_users=["12345"], timeout=60)
        interaction = _make_interaction(user_id="99999")
        btn = _get_button(view, "Deny")

        await btn.callback(interaction)

        assert not view._event.is_set()
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert "not authorized" in call_kwargs[0][0].lower()
        assert call_kwargs[1]["ephemeral"] is True


# ---------------------------------------------------------------------------
# ApprovalView — wait_for_result
# ---------------------------------------------------------------------------

class TestWaitForResult:
    async def test_returns_true_when_approved(self):
        """wait_for_result returns True when approved."""
        view = ApprovalView(allowed_users=["123"], timeout=60)
        view._approved = True
        view._event.set()

        result = await view.wait_for_result()
        assert result is True

    async def test_returns_false_when_denied(self):
        """wait_for_result returns False when denied."""
        view = ApprovalView(allowed_users=["123"], timeout=60)
        view._approved = False
        view._event.set()

        result = await view.wait_for_result()
        assert result is False

    async def test_returns_false_on_timeout(self):
        """wait_for_result returns False on timeout."""
        view = ApprovalView(allowed_users=["123"], timeout=0.01)

        result = await view.wait_for_result()
        assert result is False


# ---------------------------------------------------------------------------
# ApprovalView — on_timeout / disable_all
# ---------------------------------------------------------------------------

class TestOnTimeoutAndDisable:
    async def test_on_timeout_sets_event(self):
        """on_timeout sets the event to unblock wait_for_result."""
        view = ApprovalView(allowed_users=["123"], timeout=60)
        assert not view._event.is_set()

        await view.on_timeout()
        assert view._event.is_set()

    def test_disable_all_disables_buttons(self):
        """disable_all sets disabled=True on all button children."""
        view = ApprovalView(allowed_users=["123"], timeout=60)

        # Verify there are button children
        buttons = [c for c in view.children if isinstance(c, discord.ui.Button)]
        assert len(buttons) >= 2

        view.disable_all()
        for btn in buttons:
            assert btn.disabled is True


# ---------------------------------------------------------------------------
# request_approval
# ---------------------------------------------------------------------------

class TestRequestApproval:
    async def test_approved_flow(self):
        """Full approved flow: sends embed, user approves, returns True."""
        ch, msg = _make_channel()
        bot = MagicMock()

        # Patch ApprovalView to simulate immediate approval
        with patch("src.discord.approval.ApprovalView") as MockView:
            mock_view = MagicMock()
            mock_view.wait_for_result = AsyncMock(return_value=True)
            mock_view._event = MagicMock()
            mock_view._event.is_set = MagicMock(return_value=True)
            mock_view.disable_all = MagicMock()
            mock_view.children = []
            MockView.return_value = mock_view

            result = await request_approval(
                bot, ch, "run_command",
                {"command": "rm -rf /tmp/test", "host": "server"},
                allowed_users=["12345"],
                timeout=60,
            )

        assert result is True
        ch.send.assert_called_once()
        # Embed should be sent
        send_kwargs = ch.send.call_args[1]
        assert "embed" in send_kwargs
        assert "view" in send_kwargs
        # Message should be edited with green color
        msg.edit.assert_called_once()

    async def test_denied_flow(self):
        """Full denied flow: user denies, returns False."""
        ch, msg = _make_channel()
        bot = MagicMock()

        with patch("src.discord.approval.ApprovalView") as MockView:
            mock_view = MagicMock()
            mock_view.wait_for_result = AsyncMock(return_value=False)
            mock_view._event = MagicMock()
            mock_view._event.is_set = MagicMock(return_value=True)
            mock_view.disable_all = MagicMock()
            mock_view.children = []
            MockView.return_value = mock_view

            result = await request_approval(
                bot, ch, "run_command",
                {"command": "dangerous"},
                allowed_users=["12345"],
            )

        assert result is False
        msg.edit.assert_called_once()

    async def test_timeout_flow(self):
        """Timeout flow: no response, returns False."""
        ch, msg = _make_channel()
        bot = MagicMock()

        with patch("src.discord.approval.ApprovalView") as MockView:
            mock_view = MagicMock()
            mock_view.wait_for_result = AsyncMock(return_value=False)
            mock_view._event = MagicMock()
            mock_view._event.is_set = MagicMock(return_value=False)  # timeout, not explicit deny
            mock_view.disable_all = MagicMock()
            mock_view.children = []
            MockView.return_value = mock_view

            result = await request_approval(
                bot, ch, "write_file",
                {"path": "/etc/hosts"},
                allowed_users=["12345"],
                timeout=5,
            )

        assert result is False

    async def test_long_parameter_truncated(self):
        """Long parameter values are truncated in the embed."""
        ch, msg = _make_channel()
        bot = MagicMock()

        with patch("src.discord.approval.ApprovalView") as MockView:
            mock_view = MagicMock()
            mock_view.wait_for_result = AsyncMock(return_value=True)
            mock_view._event = MagicMock()
            mock_view._event.is_set = MagicMock(return_value=True)
            mock_view.disable_all = MagicMock()
            mock_view.children = []
            MockView.return_value = mock_view

            result = await request_approval(
                bot, ch, "run_command",
                {"command": "x" * 500},  # very long value
                allowed_users=["12345"],
            )

        assert result is True
        # Verify the embed was created (send was called with embed)
        send_kwargs = ch.send.call_args[1]
        embed = send_kwargs["embed"]
        # The description should contain truncated value
        assert "chars)" in embed.description

    async def test_very_long_description_truncated(self):
        """Very long embed descriptions are truncated to 4000 chars."""
        ch, msg = _make_channel()
        bot = MagicMock()

        with patch("src.discord.approval.ApprovalView") as MockView:
            mock_view = MagicMock()
            mock_view.wait_for_result = AsyncMock(return_value=True)
            mock_view._event = MagicMock()
            mock_view._event.is_set = MagicMock(return_value=True)
            mock_view.disable_all = MagicMock()
            mock_view.children = []
            MockView.return_value = mock_view

            # Many parameters to make description very long
            params = {f"param_{i}": "x" * 200 for i in range(30)}
            result = await request_approval(
                bot, ch, "complex_tool", params,
                allowed_users=["12345"],
            )

        assert result is True
        send_kwargs = ch.send.call_args[1]
        embed = send_kwargs["embed"]
        assert len(embed.description) <= 4020  # 4000 + "(truncated)" suffix

    async def test_edit_failure_handled(self):
        """Gracefully handles HTTPException during message edit."""
        ch, msg = _make_channel()
        msg.edit = AsyncMock(side_effect=discord.HTTPException(
            MagicMock(status=500), "Internal Server Error",
        ))
        bot = MagicMock()

        with patch("src.discord.approval.ApprovalView") as MockView:
            mock_view = MagicMock()
            mock_view.wait_for_result = AsyncMock(return_value=True)
            mock_view._event = MagicMock()
            mock_view._event.is_set = MagicMock(return_value=True)
            mock_view.disable_all = MagicMock()
            mock_view.children = []
            MockView.return_value = mock_view

            # Should not raise
            result = await request_approval(
                bot, ch, "run_command", {"cmd": "ls"},
                allowed_users=["12345"],
            )

        assert result is True

    async def test_embed_title_contains_tool_name(self):
        """The embed title includes the tool name."""
        ch, msg = _make_channel()
        bot = MagicMock()

        with patch("src.discord.approval.ApprovalView") as MockView:
            mock_view = MagicMock()
            mock_view.wait_for_result = AsyncMock(return_value=False)
            mock_view._event = MagicMock()
            mock_view._event.is_set = MagicMock(return_value=True)
            mock_view.disable_all = MagicMock()
            mock_view.children = []
            MockView.return_value = mock_view

            await request_approval(
                bot, ch, "restart_service",
                {"service": "nginx"},
                allowed_users=["12345"],
            )

        send_kwargs = ch.send.call_args[1]
        embed = send_kwargs["embed"]
        assert "restart_service" in embed.title

    async def test_empty_params(self):
        """Works with empty tool_input dict."""
        ch, msg = _make_channel()
        bot = MagicMock()

        with patch("src.discord.approval.ApprovalView") as MockView:
            mock_view = MagicMock()
            mock_view.wait_for_result = AsyncMock(return_value=True)
            mock_view._event = MagicMock()
            mock_view._event.is_set = MagicMock(return_value=True)
            mock_view.disable_all = MagicMock()
            mock_view.children = []
            MockView.return_value = mock_view

            result = await request_approval(
                bot, ch, "check_disk", {},
                allowed_users=["12345"],
            )

        assert result is True
