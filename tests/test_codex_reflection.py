"""Tests for Codex-based reflection and digest summarization.

Verifies that:
- ConversationReflector.set_text_fn() stores a callable used by _reflect/_consolidate
- _reflect() uses text_fn when set
- _consolidate() uses text_fn when set
- Without text_fn, reflection is skipped gracefully
- Failure in text_fn is handled gracefully
- Digest summarization uses Codex when available
- Dead code verification in _process_with_tools
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.learning.reflector import ConversationReflector  # noqa: E402
from src.sessions.manager import Message, Session  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(num_messages: int = 6) -> Session:
    """Build a minimal Session with alternating user/assistant messages."""
    session = Session(channel_id="ch1")
    for i in range(num_messages):
        role = "user" if i % 2 == 0 else "assistant"
        session.messages.append(
            Message(role=role, content=f"msg {i}", user_id="u1" if role == "user" else None)
        )
    session.last_user_id = "u1"
    return session


def _make_reflector(tmp_dir: Path, text_fn=None) -> ConversationReflector:
    """Create a reflector for testing."""
    reflector = ConversationReflector(
        learned_path=str(tmp_dir / "learned.json"),
        max_entries=5,
        consolidation_target=3,
    )
    if text_fn:
        reflector.set_text_fn(text_fn)
    return reflector


# ---------------------------------------------------------------------------
# set_text_fn
# ---------------------------------------------------------------------------

class TestSetTextFn:
    def test_stores_callable(self, tmp_dir):
        reflector = _make_reflector(tmp_dir)
        assert reflector._text_fn is None
        fn = AsyncMock(return_value="[]")
        reflector.set_text_fn(fn)
        assert reflector._text_fn is fn


# ---------------------------------------------------------------------------
# _reflect() with text_fn
# ---------------------------------------------------------------------------

class TestReflectWithTextFn:
    async def test_uses_text_fn_when_set(self, tmp_dir):
        """When text_fn is set, _reflect() calls it for reflection."""
        codex_fn = AsyncMock(return_value='[{"key": "test_key", "category": "fact", "content": "learned fact"}]')
        reflector = _make_reflector(tmp_dir, text_fn=codex_fn)

        session = _make_session()
        await reflector.reflect_on_session(session, user_ids=["u1"])

        codex_fn.assert_awaited_once()
        # Verify the call args: messages list and system string
        args = codex_fn.call_args[0]
        assert len(args[0]) == 1  # single message
        assert args[0][0]["role"] == "user"
        assert isinstance(args[1], str)  # system instruction

    async def test_text_fn_result_is_parsed(self, tmp_dir):
        """Entries from text_fn should be saved to learned.json."""
        codex_fn = AsyncMock(return_value='[{"key": "codex_fact", "category": "operational", "content": "Codex learned this"}]')
        reflector = _make_reflector(tmp_dir, text_fn=codex_fn)

        session = _make_session()
        await reflector.reflect_on_session(session, user_ids=["u1"])

        data = json.loads((tmp_dir / "learned.json").read_text())
        assert any(e["key"] == "codex_fact" for e in data["entries"])

    async def test_text_fn_is_active_path(self, tmp_dir):
        """When text_fn is set, it should be the active reflection path."""
        codex_fn = AsyncMock(return_value="[]")
        reflector = _make_reflector(tmp_dir, text_fn=codex_fn)

        session = _make_session()
        await reflector.reflect_on_session(session, user_ids=["u1"])

        # text_fn was called — that's sufficient to prove it's the active path
        codex_fn.assert_awaited_once()

    async def test_text_fn_failure_logged(self, tmp_dir):
        """When text_fn raises, _reflect handles it gracefully."""
        failing_fn = AsyncMock(side_effect=RuntimeError("Codex down"))
        reflector = _make_reflector(tmp_dir, text_fn=failing_fn)

        session = _make_session()
        # Should not raise
        await reflector.reflect_on_session(session, user_ids=["u1"])

        # No learned.json created since reflection failed
        assert not (tmp_dir / "learned.json").exists()

    async def test_compacted_reflection_uses_text_fn(self, tmp_dir):
        """reflect_on_compacted should also use text_fn."""
        codex_fn = AsyncMock(return_value='[{"key": "compacted_fact", "category": "correction", "content": "fix"}]')
        reflector = _make_reflector(tmp_dir, text_fn=codex_fn)

        messages = [
            Message(role="user", content=f"msg {i}", user_id="u1")
            for i in range(6)
        ]
        await reflector.reflect_on_compacted(messages, "old summary", user_ids=["u1"])

        codex_fn.assert_awaited_once()
        data = json.loads((tmp_dir / "learned.json").read_text())
        assert any(e["key"] == "compacted_fact" for e in data["entries"])


# ---------------------------------------------------------------------------
# _consolidate() with text_fn
# ---------------------------------------------------------------------------

class TestConsolidateWithTextFn:
    async def test_uses_text_fn_for_consolidation(self, tmp_dir):
        """When text_fn is set, _consolidate uses it for consolidation."""
        codex_fn = AsyncMock(return_value='[{"key": "merged", "category": "fact", "content": "merged fact"}]')
        reflector = _make_reflector(tmp_dir, text_fn=codex_fn)

        entries = [
            {"key": f"k{i}", "category": "fact", "content": f"fact {i}",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"}
            for i in range(6)
        ]
        result = await reflector._consolidate(entries)

        # text_fn is called twice in this case: once for reflect, but
        # here we only call _consolidate directly
        assert codex_fn.await_count >= 1
        assert any(e["key"] == "merged" for e in result)

    async def test_consolidation_fallback_on_text_fn_failure(self, tmp_dir):
        """When text_fn fails during consolidation, falls back to trim."""
        failing_fn = AsyncMock(side_effect=RuntimeError("Codex down"))
        reflector = _make_reflector(tmp_dir, text_fn=failing_fn)

        entries = [
            {"key": f"k{i}", "category": "fact", "content": f"fact {i}",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"}
            for i in range(6)
        ]
        result = await reflector._consolidate(entries)

        # Should fall back to keeping most recent entries
        assert len(result) <= reflector._consolidation_target


# ---------------------------------------------------------------------------
# No text_fn configured
# ---------------------------------------------------------------------------

class TestNoTextFnPath:
    async def test_skips_reflection_when_no_text_fn(self, tmp_dir):
        """Without text_fn, _reflect returns early without crashing."""
        reflector = _make_reflector(tmp_dir)
        assert reflector._text_fn is None

        session = _make_session()
        # Should not raise — just logs a warning and returns
        await reflector.reflect_on_session(session, user_ids=["u1"])

        # No learned.json created since reflection was skipped
        assert not (tmp_dir / "learned.json").exists()


# ---------------------------------------------------------------------------
# Dead code verification in _process_with_tools
# ---------------------------------------------------------------------------

class TestHaikuCodeRemoved:
    def test_no_haiku_streaming_check(self):
        """_process_with_tools should no longer check model_override == HAIKU_MODEL
        for streaming decisions."""
        import inspect
        from src.discord.client import AnsiblexBot

        source = inspect.getsource(AnsiblexBot._process_with_tools)
        # The old Haiku-specific logic should be gone
        assert "model_override == HAIKU_MODEL" not in source
        assert "escalate" not in source.lower()

    def test_codex_path_no_use_codex_flag(self):
        """_process_with_tools no longer uses a use_codex flag — it always
        uses self.codex_client.chat_with_tools() directly."""
        import inspect
        from src.discord.client import AnsiblexBot

        source = inspect.getsource(AnsiblexBot._process_with_tools)
        # Old flags should be gone
        assert "use_codex" not in source
        assert "use_streaming" not in source
        # Codex path is always used
        assert "chat_with_tools" in source


# ---------------------------------------------------------------------------
# Digest summarization uses Codex
# ---------------------------------------------------------------------------

class TestDigestSummarization:
    async def test_digest_uses_codex_when_available(self):
        """When codex_client is available, digest should use it."""
        from src.discord.client import AnsiblexBot

        with patch.object(AnsiblexBot, "__init__", lambda self, *a, **kw: None):
            bot = AnsiblexBot.__new__(AnsiblexBot)

        bot.codex_client = MagicMock()
        bot.codex_client.chat = AsyncMock(return_value="All systems healthy. No issues detected.")
        bot.claude = MagicMock()
        bot.audit = MagicMock()
        bot.audit.log_execution = AsyncMock()

        # Mock channel
        channel = MagicMock()
        channel.send = AsyncMock()

        # Mock _format_digest_raw
        bot._format_digest_raw = AsyncMock(return_value="CPU: 20%, RAM: 40%, Disk: 60%")

        # Mock get_channel
        bot.get_channel = MagicMock(return_value=channel)

        # Call _on_scheduled_digest
        bot._on_scheduled_digest = AnsiblexBot._on_scheduled_digest.__get__(bot)
        await bot._on_scheduled_digest({"id": "digest-1", "channel_id": "123"})

        # Codex should be called, not Claude
        bot.codex_client.chat.assert_awaited_once()
        bot.claude.chat.assert_not_called()
        channel.send.assert_called_once()
        sent = channel.send.call_args[0][0]
        assert "Daily Infrastructure Digest" in sent
