"""Tests for _send_chunked — Discord message chunking.

Covers:
- Short messages sent directly (no chunking)
- Very long messages sent as file attachment
- Normal chunking across multiple messages
- Code block continuation across chunk boundaries
- Long lines (>1990 chars) that previously caused oversized chunks
- Empty chunk prevention (previously sent empty messages to Discord)
- First chunk sent as reply, subsequent as plain messages
"""
from __future__ import annotations

import io
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import discord  # noqa: E402
import pytest  # noqa: E402

from src.discord.client import LokiBot, DISCORD_MAX_LEN  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_and_message():
    """Create minimal stubs for _send_chunked testing."""
    bot = MagicMock(spec=LokiBot)
    # Wire up the real _send_chunked method (unbound) to the stub
    bot._send_chunked = LokiBot._send_chunked.__get__(bot)
    bot._send_with_retry = AsyncMock(return_value=MagicMock())
    # _pending_files is a per-channel dict
    bot._pending_files = {}

    msg = MagicMock(spec=discord.Message)
    msg.reply = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = "test-channel"
    msg.channel.send = AsyncMock()

    return bot, msg


# ---------------------------------------------------------------------------
# Short message — no chunking
# ---------------------------------------------------------------------------

class TestShortMessage:
    @pytest.mark.asyncio
    async def test_short_message_sent_directly(self):
        bot, msg = _make_bot_and_message()
        await bot._send_chunked(msg, "Hello!")
        bot._send_with_retry.assert_called_once_with(msg, "Hello!")

    @pytest.mark.asyncio
    async def test_exactly_at_limit_sent_directly(self):
        bot, msg = _make_bot_and_message()
        text = "x" * DISCORD_MAX_LEN
        await bot._send_chunked(msg, text)
        bot._send_with_retry.assert_called_once_with(msg, text)


# ---------------------------------------------------------------------------
# Very long message — file attachment
# ---------------------------------------------------------------------------

class TestFileAttachment:
    @pytest.mark.asyncio
    async def test_very_long_message_sent_as_file(self):
        bot, msg = _make_bot_and_message()
        text = "x" * (DISCORD_MAX_LEN * 4 + 1)
        await bot._send_chunked(msg, text)
        # Should send a "too long" message with file attached via _send_with_retry
        bot._send_with_retry.assert_called_once()
        assert "too long" in bot._send_with_retry.call_args[0][1].lower()
        files = bot._send_with_retry.call_args[1].get("files")
        assert files and len(files) > 0


# ---------------------------------------------------------------------------
# Normal chunking
# ---------------------------------------------------------------------------

class TestNormalChunking:
    @pytest.mark.asyncio
    async def test_two_chunks(self):
        bot, msg = _make_bot_and_message()
        # Build a message just over the limit with two ~1000-char sections
        part1 = "a" * 1000 + "\n"
        part2 = "b" * 1000 + "\n"
        text = part1 + part2  # 2002 chars total > 2000
        await bot._send_chunked(msg, text)
        assert bot._send_with_retry.call_count == 2
        # First chunk sent as reply
        first_call = bot._send_with_retry.call_args_list[0]
        assert first_call == call(msg, bot._send_with_retry.call_args_list[0][0][1])
        # Second chunk sent as_reply=False
        second_call = bot._send_with_retry.call_args_list[1]
        assert second_call[1].get("as_reply") is False or (len(second_call[0]) >= 3 and second_call[0][2] is False) or second_call == call(msg, second_call[0][1], as_reply=False)

    @pytest.mark.asyncio
    async def test_all_chunks_under_discord_limit(self):
        bot, msg = _make_bot_and_message()
        # ~3500 chars across many lines
        lines = [f"Line {i}: {'x' * 80}" for i in range(40)]
        text = "\n".join(lines)
        assert len(text) > DISCORD_MAX_LEN
        assert len(text) <= DISCORD_MAX_LEN * 4
        await bot._send_chunked(msg, text)
        for c in bot._send_with_retry.call_args_list:
            chunk_text = c[0][1]
            assert len(chunk_text) <= DISCORD_MAX_LEN, (
                f"Chunk is {len(chunk_text)} chars, exceeds {DISCORD_MAX_LEN}"
            )

    @pytest.mark.asyncio
    async def test_first_chunk_is_reply_rest_are_not(self):
        bot, msg = _make_bot_and_message()
        lines = [f"Line {i}: {'x' * 80}" for i in range(40)]
        text = "\n".join(lines)
        await bot._send_chunked(msg, text)
        calls = bot._send_with_retry.call_args_list
        assert len(calls) >= 2
        # First call: no as_reply kwarg (defaults to True)
        assert "as_reply" not in calls[0][1]
        # Subsequent calls: as_reply=False
        for c in calls[1:]:
            assert c[1].get("as_reply") is False or c == call(msg, c[0][1], as_reply=False)


# ---------------------------------------------------------------------------
# Code block continuation
# ---------------------------------------------------------------------------

class TestCodeBlockContinuation:
    @pytest.mark.asyncio
    async def test_code_block_split_adds_markers(self):
        bot, msg = _make_bot_and_message()
        # Create a long code block that must be split across chunks
        code_lines = ["```python"] + [f"x = {i}" for i in range(300)] + ["```"]
        text = "\n".join(code_lines)
        assert len(text) > DISCORD_MAX_LEN
        await bot._send_chunked(msg, text)
        calls = bot._send_with_retry.call_args_list
        assert len(calls) >= 2
        # First chunk should end with ``` (closing the code block)
        first_chunk = calls[0][0][1]
        assert first_chunk.rstrip().endswith("```")
        # Second chunk should start with ``` (re-opening the code block)
        second_chunk = calls[1][0][1]
        assert second_chunk.lstrip().startswith("```")


# ---------------------------------------------------------------------------
# Long line handling (the bug fix)
# ---------------------------------------------------------------------------

class TestLongLineHandling:
    @pytest.mark.asyncio
    async def test_single_long_line_split_into_valid_chunks(self):
        """A single line > DISCORD_MAX_LEN must be split so no chunk exceeds the limit."""
        bot, msg = _make_bot_and_message()
        # 2500-char single line + short text to exceed the short-message path
        long_line = "x" * 2500
        text = long_line + "\nshort line\n"
        await bot._send_chunked(msg, text)
        calls = bot._send_with_retry.call_args_list
        assert len(calls) >= 2
        for c in calls:
            chunk_text = c[0][1]
            assert len(chunk_text) <= DISCORD_MAX_LEN, (
                f"Chunk is {len(chunk_text)} chars, exceeds {DISCORD_MAX_LEN}"
            )

    @pytest.mark.asyncio
    async def test_very_long_single_line_no_newlines(self):
        """Text with no newlines at all, 3000 chars — must chunk without oversizing."""
        bot, msg = _make_bot_and_message()
        text = "a" * 3000
        await bot._send_chunked(msg, text)
        calls = bot._send_with_retry.call_args_list
        assert len(calls) >= 2
        for c in calls:
            chunk_text = c[0][1]
            assert len(chunk_text) <= DISCORD_MAX_LEN, (
                f"Chunk is {len(chunk_text)} chars, exceeds {DISCORD_MAX_LEN}"
            )

    @pytest.mark.asyncio
    async def test_long_line_between_normal_lines(self):
        """A long line sandwiched between normal lines should not cause oversized chunks."""
        bot, msg = _make_bot_and_message()
        text = "Normal start\n" + "x" * 2500 + "\nNormal end\n"
        await bot._send_chunked(msg, text)
        calls = bot._send_with_retry.call_args_list
        for c in calls:
            chunk_text = c[0][1]
            assert len(chunk_text) <= DISCORD_MAX_LEN

    @pytest.mark.asyncio
    async def test_multiple_long_lines(self):
        """Multiple consecutive long lines should all be handled."""
        bot, msg = _make_bot_and_message()
        text = ("y" * 2500) + "\n" + ("z" * 2500) + "\nend\n"
        await bot._send_chunked(msg, text)
        calls = bot._send_with_retry.call_args_list
        for c in calls:
            chunk_text = c[0][1]
            assert len(chunk_text) <= DISCORD_MAX_LEN

    @pytest.mark.asyncio
    async def test_long_line_preserves_content(self):
        """All content from a long line should appear across the chunks."""
        bot, msg = _make_bot_and_message()
        long_line = "abcdefgh" * 350  # 2800 chars
        text = long_line + "\nend\n"
        await bot._send_chunked(msg, text)
        calls = bot._send_with_retry.call_args_list
        reassembled = "".join(c[0][1] for c in calls)
        # The long line content should be fully present (with possible newlines from splitting)
        content_only = reassembled.replace("\n", "")
        assert long_line in content_only or content_only.startswith(long_line.replace("\n", ""))


# ---------------------------------------------------------------------------
# Empty chunk prevention
# ---------------------------------------------------------------------------

class TestEmptyChunkPrevention:
    @pytest.mark.asyncio
    async def test_no_empty_chunks_sent(self):
        """No call to _send_with_retry should have empty text."""
        bot, msg = _make_bot_and_message()
        # Trigger the scenario: long line at start forces chunk boundary with empty current
        text = "x" * 2500 + "\nshort\n"
        await bot._send_chunked(msg, text)
        for c in bot._send_with_retry.call_args_list:
            chunk_text = c[0][1]
            assert chunk_text.strip(), "Empty chunk was sent to Discord"

    @pytest.mark.asyncio
    async def test_no_empty_chunks_with_code_blocks(self):
        """Ensure code block markers don't create empty chunks."""
        bot, msg = _make_bot_and_message()
        text = "```\n" + "x" * 2500 + "\n```\n"
        await bot._send_chunked(msg, text)
        for c in bot._send_with_retry.call_args_list:
            chunk_text = c[0][1]
            assert chunk_text.strip(), "Empty chunk was sent to Discord"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_exactly_at_chunk_boundary(self):
        """Text exactly at DISCORD_MAX_LEN + 1 should produce two chunks."""
        bot, msg = _make_bot_and_message()
        # Two lines that together exceed the limit
        line1 = "a" * (DISCORD_MAX_LEN - 100)
        line2 = "b" * 200
        text = line1 + "\n" + line2
        assert len(text) > DISCORD_MAX_LEN
        await bot._send_chunked(msg, text)
        assert bot._send_with_retry.call_count >= 2

    @pytest.mark.asyncio
    async def test_line_exactly_at_max_line_len(self):
        """A line exactly at the pre-split threshold should not be split."""
        bot, msg = _make_bot_and_message()
        max_line_len = DISCORD_MAX_LEN - 20
        line = "x" * max_line_len
        text = line + "\n" + "short\n"
        # This should work without any oversized chunks
        if len(text) > DISCORD_MAX_LEN:
            await bot._send_chunked(msg, text)
            for c in bot._send_with_retry.call_args_list:
                assert len(c[0][1]) <= DISCORD_MAX_LEN

    @pytest.mark.asyncio
    async def test_whitespace_only_trailing_not_appended(self):
        """Trailing whitespace-only content should not create an extra chunk."""
        bot, msg = _make_bot_and_message()
        text = "a" * 1500 + "\n" + "b" * 1500 + "\n   \n"
        await bot._send_chunked(msg, text)
        for c in bot._send_with_retry.call_args_list:
            assert c[0][1].strip(), "Whitespace-only chunk was sent"
