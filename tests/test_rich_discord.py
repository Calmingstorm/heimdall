"""Tests for rich Discord tools: add_reaction, create_poll, broadcast.

Covers handler logic, validation, error handling, embed construction,
and poll constraints.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import discord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot():
    """Create a minimal mock bot with the relevant handler methods."""
    from src.discord.client import LokiBot
    bot = MagicMock(spec=LokiBot)
    # Bind real async methods from the class
    bot._handle_add_reaction = LokiBot._handle_add_reaction.__get__(bot, LokiBot)
    bot._handle_create_poll = LokiBot._handle_create_poll.__get__(bot, LokiBot)
    bot._handle_broadcast = LokiBot._handle_broadcast.__get__(bot, LokiBot)
    return bot


def _make_message(channel=None):
    """Create a mock Discord message with a mock channel."""
    msg = AsyncMock(spec=discord.Message)
    msg.channel = channel or AsyncMock()
    return msg


# ---------------------------------------------------------------------------
# add_reaction handler
# ---------------------------------------------------------------------------

class TestAddReaction:
    """Tests for _handle_add_reaction."""

    async def test_add_reaction_success(self):
        bot = _make_bot()
        msg = _make_message()
        target_msg = AsyncMock()
        msg.channel.fetch_message = AsyncMock(return_value=target_msg)

        result = await bot._handle_add_reaction(msg, {
            "message_id": "123456",
            "emoji": "\U0001f44d",
        })

        msg.channel.fetch_message.assert_awaited_once_with(123456)
        target_msg.add_reaction.assert_awaited_once_with("\U0001f44d")
        assert result == "Reaction added."

    async def test_add_reaction_missing_message_id_uses_current(self):
        """Missing message_id falls back to the triggering message."""
        bot = _make_bot()
        msg = _make_message()
        msg.id = 999888
        target_msg = AsyncMock()
        msg.channel.fetch_message = AsyncMock(return_value=target_msg)
        result = await bot._handle_add_reaction(msg, {"emoji": "\U0001f44d"})
        msg.channel.fetch_message.assert_awaited_once_with(999888)
        assert result == "Reaction added."

    async def test_add_reaction_missing_emoji(self):
        bot = _make_bot()
        msg = _make_message()
        result = await bot._handle_add_reaction(msg, {"message_id": "123"})
        assert "required" in result.lower()

    async def test_add_reaction_empty_inputs(self):
        bot = _make_bot()
        msg = _make_message()
        result = await bot._handle_add_reaction(msg, {})
        assert "required" in result.lower()

    async def test_add_reaction_message_not_found(self):
        bot = _make_bot()
        msg = _make_message()
        msg.channel.fetch_message = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "not found")
        )
        result = await bot._handle_add_reaction(msg, {
            "message_id": "999",
            "emoji": "\U0001f44d",
        })
        assert "not found" in result.lower()

    async def test_add_reaction_forbidden(self):
        bot = _make_bot()
        msg = _make_message()
        msg.channel.fetch_message = AsyncMock(
            side_effect=discord.Forbidden(MagicMock(status=403), "forbidden")
        )
        result = await bot._handle_add_reaction(msg, {
            "message_id": "123",
            "emoji": "\U0001f44d",
        })
        assert "permission" in result.lower() or "denied" in result.lower()

    async def test_add_reaction_generic_error(self):
        bot = _make_bot()
        msg = _make_message()
        msg.channel.fetch_message = AsyncMock(side_effect=RuntimeError("boom"))
        result = await bot._handle_add_reaction(msg, {
            "message_id": "123",
            "emoji": "\U0001f44d",
        })
        assert "failed" in result.lower()

    async def test_add_reaction_custom_emoji(self):
        """Custom Discord emoji format <:name:id> should work."""
        bot = _make_bot()
        msg = _make_message()
        target_msg = AsyncMock()
        msg.channel.fetch_message = AsyncMock(return_value=target_msg)

        result = await bot._handle_add_reaction(msg, {
            "message_id": "123456",
            "emoji": "<:custom:111222333>",
        })
        target_msg.add_reaction.assert_awaited_once_with("<:custom:111222333>")
        assert result == "Reaction added."


# ---------------------------------------------------------------------------
# create_poll handler
# ---------------------------------------------------------------------------

class TestCreatePoll:
    """Tests for _handle_create_poll."""

    async def test_create_poll_success(self):
        bot = _make_bot()
        msg = _make_message()

        with patch("discord.Poll") as MockPoll:
            poll_instance = MagicMock()
            MockPoll.return_value = poll_instance

            result = await bot._handle_create_poll(msg, {
                "question": "Best language?",
                "options": ["Python", "Rust", "Go"],
            })

            MockPoll.assert_called_once()
            call_kwargs = MockPoll.call_args[1]
            assert call_kwargs["question"] == "Best language?"
            assert call_kwargs["multiple"] is False
            # duration defaults to 24h
            assert call_kwargs["duration"] == timedelta(hours=24)
            # Each option should be added
            assert poll_instance.add_answer.call_count == 3
            msg.channel.send.assert_awaited_once()
            assert result == "Poll created."

    async def test_create_poll_missing_question(self):
        bot = _make_bot()
        msg = _make_message()
        result = await bot._handle_create_poll(msg, {
            "options": ["A", "B"],
        })
        assert "required" in result.lower()

    async def test_create_poll_missing_options(self):
        bot = _make_bot()
        msg = _make_message()
        result = await bot._handle_create_poll(msg, {
            "question": "Test?",
        })
        assert "required" in result.lower()

    async def test_create_poll_empty_options(self):
        bot = _make_bot()
        msg = _make_message()
        result = await bot._handle_create_poll(msg, {
            "question": "Test?",
            "options": [],
        })
        assert "required" in result.lower()

    async def test_poll_max_options_exceeded(self):
        bot = _make_bot()
        msg = _make_message()
        result = await bot._handle_create_poll(msg, {
            "question": "Too many?",
            "options": [f"opt{i}" for i in range(11)],
        })
        assert "10" in result or "maximum" in result.lower()

    async def test_poll_exactly_10_options(self):
        """10 options should succeed (max allowed)."""
        bot = _make_bot()
        msg = _make_message()

        with patch("discord.Poll") as MockPoll:
            poll_instance = MagicMock()
            MockPoll.return_value = poll_instance

            result = await bot._handle_create_poll(msg, {
                "question": "Pick one?",
                "options": [f"opt{i}" for i in range(10)],
            })
            assert poll_instance.add_answer.call_count == 10
            assert result == "Poll created."

    async def test_poll_custom_duration(self):
        bot = _make_bot()
        msg = _make_message()

        with patch("discord.Poll") as MockPoll:
            poll_instance = MagicMock()
            MockPoll.return_value = poll_instance

            await bot._handle_create_poll(msg, {
                "question": "Test?",
                "options": ["A", "B"],
                "duration_hours": 48,
            })
            call_kwargs = MockPoll.call_args[1]
            assert call_kwargs["duration"] == timedelta(hours=48)

    async def test_poll_duration_clamped_to_168(self):
        """Duration should be capped at 168 hours (7 days)."""
        bot = _make_bot()
        msg = _make_message()

        with patch("discord.Poll") as MockPoll:
            poll_instance = MagicMock()
            MockPoll.return_value = poll_instance

            await bot._handle_create_poll(msg, {
                "question": "Test?",
                "options": ["A", "B"],
                "duration_hours": 500,
            })
            call_kwargs = MockPoll.call_args[1]
            assert call_kwargs["duration"] == timedelta(hours=168)

    async def test_poll_multiple_selection(self):
        bot = _make_bot()
        msg = _make_message()

        with patch("discord.Poll") as MockPoll:
            poll_instance = MagicMock()
            MockPoll.return_value = poll_instance

            await bot._handle_create_poll(msg, {
                "question": "Pick any?",
                "options": ["A", "B", "C"],
                "multiple": True,
            })
            call_kwargs = MockPoll.call_args[1]
            assert call_kwargs["multiple"] is True

    async def test_poll_send_failure(self):
        bot = _make_bot()
        msg = _make_message()

        with patch("discord.Poll") as MockPoll:
            poll_instance = MagicMock()
            MockPoll.return_value = poll_instance
            msg.channel.send = AsyncMock(side_effect=RuntimeError("Discord down"))

            result = await bot._handle_create_poll(msg, {
                "question": "Test?",
                "options": ["A", "B"],
            })
            assert "failed" in result.lower()

    async def test_poll_options_converted_to_strings(self):
        """Non-string options should be converted via str()."""
        bot = _make_bot()
        msg = _make_message()

        with patch("discord.Poll") as MockPoll:
            poll_instance = MagicMock()
            MockPoll.return_value = poll_instance

            await bot._handle_create_poll(msg, {
                "question": "Numbers?",
                "options": [1, 2, 3],
            })
            calls = poll_instance.add_answer.call_args_list
            assert calls[0][1]["text"] == "1"
            assert calls[1][1]["text"] == "2"
            assert calls[2][1]["text"] == "3"


# ---------------------------------------------------------------------------
# broadcast handler
# ---------------------------------------------------------------------------

class TestBroadcast:
    """Tests for _handle_broadcast."""

    async def test_broadcast_plain_text(self):
        bot = _make_bot()
        msg = _make_message()

        result = await bot._handle_broadcast(msg, {"text": "Hello world"})
        msg.channel.send.assert_awaited_once_with(
            content="Hello world", embed=None
        )
        assert result == "Message sent."

    async def test_broadcast_with_embed(self):
        bot = _make_bot()
        msg = _make_message()

        result = await bot._handle_broadcast(msg, {
            "text": "Check this out",
            "embed": {
                "title": "Status Report",
                "description": "All systems operational.",
                "color": "#00ff00",
                "fields": [
                    {"name": "Uptime", "value": "99.9%", "inline": True},
                    {"name": "CPU", "value": "12%"},
                ],
            },
        })

        msg.channel.send.assert_awaited_once()
        call_kwargs = msg.channel.send.call_args[1]
        assert call_kwargs["content"] == "Check this out"
        embed_obj = call_kwargs["embed"]
        assert isinstance(embed_obj, discord.Embed)
        assert embed_obj.title == "Status Report"
        assert embed_obj.description == "All systems operational."
        assert embed_obj.color.value == 0x00FF00
        assert len(embed_obj.fields) == 2
        assert embed_obj.fields[0].name == "Uptime"
        assert embed_obj.fields[0].value == "99.9%"
        assert embed_obj.fields[0].inline is True
        assert embed_obj.fields[1].name == "CPU"
        assert embed_obj.fields[1].inline is False
        assert result == "Message sent."

    async def test_broadcast_embed_only(self):
        """Embed without text should still send."""
        bot = _make_bot()
        msg = _make_message()

        result = await bot._handle_broadcast(msg, {
            "embed": {
                "title": "Alert",
                "description": "Something happened.",
            },
        })

        msg.channel.send.assert_awaited_once()
        call_kwargs = msg.channel.send.call_args[1]
        assert call_kwargs["content"] is None
        assert isinstance(call_kwargs["embed"], discord.Embed)
        assert result == "Message sent."

    async def test_broadcast_empty_input(self):
        bot = _make_bot()
        msg = _make_message()
        result = await bot._handle_broadcast(msg, {})
        assert "provide" in result.lower()

    async def test_broadcast_no_text_no_embed(self):
        bot = _make_bot()
        msg = _make_message()
        result = await bot._handle_broadcast(msg, {"text": "", "embed": None})
        assert "provide" in result.lower()

    async def test_broadcast_embed_invalid_color(self):
        """Invalid color string should default to 0 (black)."""
        bot = _make_bot()
        msg = _make_message()

        await bot._handle_broadcast(msg, {
            "embed": {
                "title": "Test",
                "description": "Desc",
                "color": "not-a-color",
            },
        })

        call_kwargs = msg.channel.send.call_args[1]
        embed_obj = call_kwargs["embed"]
        assert embed_obj.color.value == 0

    async def test_broadcast_embed_no_color(self):
        """Missing color should default to #000000 (0)."""
        bot = _make_bot()
        msg = _make_message()

        await bot._handle_broadcast(msg, {
            "embed": {
                "title": "Test",
                "description": "Desc",
            },
        })

        call_kwargs = msg.channel.send.call_args[1]
        embed_obj = call_kwargs["embed"]
        assert embed_obj.color.value == 0

    async def test_broadcast_embed_with_empty_fields(self):
        """Fields list can be empty."""
        bot = _make_bot()
        msg = _make_message()

        result = await bot._handle_broadcast(msg, {
            "embed": {
                "title": "Test",
                "description": "Desc",
                "fields": [],
            },
        })

        call_kwargs = msg.channel.send.call_args[1]
        embed_obj = call_kwargs["embed"]
        assert len(embed_obj.fields) == 0
        assert result == "Message sent."

    async def test_broadcast_embed_field_defaults(self):
        """Fields with missing name/value get zero-width space defaults."""
        bot = _make_bot()
        msg = _make_message()

        await bot._handle_broadcast(msg, {
            "embed": {
                "title": "Test",
                "description": "Desc",
                "fields": [{}],  # Missing name and value
            },
        })

        call_kwargs = msg.channel.send.call_args[1]
        embed_obj = call_kwargs["embed"]
        assert embed_obj.fields[0].name == "\u200b"
        assert embed_obj.fields[0].value == "\u200b"

    async def test_broadcast_text_and_embed_together(self):
        """Both text and embed should be sent in the same message."""
        bot = _make_bot()
        msg = _make_message()

        result = await bot._handle_broadcast(msg, {
            "text": "Important:",
            "embed": {"title": "Info", "description": "Details here"},
        })

        msg.channel.send.assert_awaited_once()
        call_kwargs = msg.channel.send.call_args[1]
        assert call_kwargs["content"] == "Important:"
        assert call_kwargs["embed"] is not None
        assert result == "Message sent."

    async def test_broadcast_embed_not_dict(self):
        """Non-dict embed should be treated as no embed."""
        bot = _make_bot()
        msg = _make_message()

        result = await bot._handle_broadcast(msg, {
            "text": "Hello",
            "embed": "not a dict",
        })

        msg.channel.send.assert_awaited_once()
        call_kwargs = msg.channel.send.call_args[1]
        assert call_kwargs["embed"] is None
        assert result == "Message sent."

    async def test_broadcast_color_without_hash(self):
        """Color like 'ff0000' without '#' should fail gracefully."""
        bot = _make_bot()
        msg = _make_message()

        await bot._handle_broadcast(msg, {
            "embed": {
                "title": "Test",
                "description": "Desc",
                "color": "ff0000",
            },
        })

        call_kwargs = msg.channel.send.call_args[1]
        embed_obj = call_kwargs["embed"]
        # "ff0000" after lstrip("#") is "ff0000" — should parse as 0xff0000
        assert embed_obj.color.value == 0xFF0000
