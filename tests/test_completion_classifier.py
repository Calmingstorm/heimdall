"""Tests for the completion classifier (Round 1: stub).

Round 1 installs a stub _classify_completion() that always returns
(True, "")—i.e., COMPLETE.  These tests verify:
1. The stub exists and returns the expected sentinel values.
2. The removed functions (_should_continue_task, _is_mid_task_checkpoint,
   _CHECKPOINT_PATTERNS, _CONTINUATION_MAX_CHARS) are actually gone.
3. _CONTINUATION_MSG and continuation_count/max_continuations survive.
4. max_tokens is now forwarded by CodexChatClient.chat().
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    HeimdallBot,
    _CONTINUATION_MSG,
)


# ---------------------------------------------------------------------------
# 1. Stub returns COMPLETE
# ---------------------------------------------------------------------------


class TestClassifyCompletionStub:
    """Round 1 stub always returns (True, '')."""

    @pytest.mark.asyncio
    async def test_stub_returns_complete(self):
        stub = MagicMock(spec=HeimdallBot)
        is_complete, reason = await HeimdallBot._classify_completion(
            stub, "check disk", "Disk is fine.", ["run_command"],
        )
        assert is_complete is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_stub_with_empty_inputs(self):
        stub = MagicMock(spec=HeimdallBot)
        is_complete, reason = await HeimdallBot._classify_completion(
            stub, "", "", [],
        )
        assert is_complete is True
        assert reason == ""


# ---------------------------------------------------------------------------
# 2. Removed items are gone
# ---------------------------------------------------------------------------


class TestRemovedItems:
    """Verify old regex-based continuation items were deleted."""

    def test_no_checkpoint_patterns(self):
        import src.discord.client as mod
        assert not hasattr(mod, "_CHECKPOINT_PATTERNS")

    def test_no_continuation_max_chars(self):
        import src.discord.client as mod
        assert not hasattr(mod, "_CONTINUATION_MAX_CHARS")

    def test_no_is_mid_task_checkpoint(self):
        import src.discord.client as mod
        assert not hasattr(mod, "_is_mid_task_checkpoint")

    def test_no_should_continue_task(self):
        import src.discord.client as mod
        assert not hasattr(mod, "_should_continue_task")


# ---------------------------------------------------------------------------
# 3. Kept items still present
# ---------------------------------------------------------------------------


class TestKeptItems:
    """Items that must survive the removal."""

    def test_continuation_msg_exists(self):
        assert _CONTINUATION_MSG["role"] == "developer"
        assert "continue" in _CONTINUATION_MSG["content"].lower()

    def test_classify_completion_is_method(self):
        assert hasattr(HeimdallBot, "_classify_completion")


# ---------------------------------------------------------------------------
# 4. max_tokens forwarding in CodexChatClient.chat()
# ---------------------------------------------------------------------------


class TestMaxTokensForwarding:
    """Verify that chat() passes max_tokens to the API body."""

    @pytest.mark.asyncio
    async def test_max_tokens_added_to_body(self):
        """When max_tokens is provided, body should include max_output_tokens."""
        from src.llm.openai_codex import CodexChatClient

        client = MagicMock(spec=CodexChatClient)
        client.model = "gpt-4"
        client.max_tokens = None
        client.auth = MagicMock()
        client.auth.get_access_token = AsyncMock(return_value="tok")
        client.auth.get_account_id = MagicMock(return_value=None)
        client._convert_messages = MagicMock(return_value=[])

        captured_body = {}

        async def fake_stream(headers, body):
            captured_body.update(body)
            return "ok"

        client._stream_request = fake_stream

        result = await CodexChatClient.chat(
            client, messages=[], system="sys", max_tokens=10,
        )
        assert result == "ok"
        assert captured_body.get("max_output_tokens") == 10

    @pytest.mark.asyncio
    async def test_no_max_tokens_when_none(self):
        """When max_tokens is None and self.max_tokens is None, no key in body."""
        from src.llm.openai_codex import CodexChatClient

        client = MagicMock(spec=CodexChatClient)
        client.model = "gpt-4"
        client.max_tokens = None
        client.auth = MagicMock()
        client.auth.get_access_token = AsyncMock(return_value="tok")
        client.auth.get_account_id = MagicMock(return_value=None)
        client._convert_messages = MagicMock(return_value=[])

        captured_body = {}

        async def fake_stream(headers, body):
            captured_body.update(body)
            return "ok"

        client._stream_request = fake_stream

        result = await CodexChatClient.chat(
            client, messages=[], system="sys",
        )
        assert "max_output_tokens" not in captured_body

    @pytest.mark.asyncio
    async def test_instance_max_tokens_used_as_fallback(self):
        """When per-call max_tokens is None, self.max_tokens is used."""
        from src.llm.openai_codex import CodexChatClient

        client = MagicMock(spec=CodexChatClient)
        client.model = "gpt-4"
        client.max_tokens = 500
        client.auth = MagicMock()
        client.auth.get_access_token = AsyncMock(return_value="tok")
        client.auth.get_account_id = MagicMock(return_value=None)
        client._convert_messages = MagicMock(return_value=[])

        captured_body = {}

        async def fake_stream(headers, body):
            captured_body.update(body)
            return "ok"

        client._stream_request = fake_stream

        result = await CodexChatClient.chat(
            client, messages=[], system="sys",
        )
        assert captured_body.get("max_output_tokens") == 500
