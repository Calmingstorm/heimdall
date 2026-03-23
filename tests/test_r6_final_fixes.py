"""Tests for R6 final quality fixes (session 48).

Bug 1: Session.load() didn't restore last_user_id from serialized JSON.
After a bot restart, sessions lost track of which user was last active,
causing reflections during compaction to lose per-user context.
Fix: Add last_user_id=data.get("last_user_id") to Session constructor in load().

Bug 2: Consolidation in reflector dropped user_id from entries.
The consolidation prompt schema didn't mention user_id, and post-processing
only preserved created_at/updated_at from originals, not user_id.
Fix: Add user_id to prompt schema and preserve from originals in post-processing.

Bug 3: ToolMemory.record() never called _expire(), so in a long-running
instance entries older than EXPIRY_DAYS accumulated in memory.
Fix: Call _expire() in record() before capping.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.sessions.manager import SessionManager, Session, Message  # noqa: E402
from src.learning.reflector import ConversationReflector  # noqa: E402
from src.tools.tool_memory import ToolMemory, EXPIRY_DAYS  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_claude():
    client = MagicMock()
    client.usage = MagicMock()
    client.usage.record = MagicMock()
    return client


@pytest.fixture
def mock_text_fn():
    return AsyncMock(return_value="[]")


@pytest.fixture
def reflector(tmp_dir, mock_text_fn):
    r = ConversationReflector(
        learned_path=str(tmp_dir / "learned.json"),
        max_entries=5,
        consolidation_target=3,
    )
    r.set_text_fn(mock_text_fn)
    return r


# ── Bug 1: Session.load() restores last_user_id ──────────────────────

class TestSessionLoadLastUserId:
    """Session.load() must restore last_user_id from serialized data."""

    def test_last_user_id_preserved_through_save_load(self, tmp_dir):
        """Save a session with last_user_id, reload it, verify it persists."""
        persist_dir = tmp_dir / "sessions"
        persist_dir.mkdir()
        mgr1 = SessionManager(
            max_history=10, max_age_hours=1, persist_dir=str(persist_dir),
        )
        mgr1.add_message("ch1", "user", "hello")
        session = mgr1._sessions["ch1"]
        session.last_user_id = "100000000000000001"
        mgr1.save()

        mgr2 = SessionManager(
            max_history=10, max_age_hours=1, persist_dir=str(persist_dir),
        )
        mgr2.load()
        loaded = mgr2._sessions["ch1"]
        assert loaded.last_user_id == "100000000000000001"

    def test_last_user_id_none_when_absent(self, tmp_dir):
        """Old session files without last_user_id field default to None."""
        persist_dir = tmp_dir / "sessions"
        persist_dir.mkdir()
        data = {
            "channel_id": "ch1",
            "messages": [{"role": "user", "content": "hello"}],
            "created_at": time.time(),
            "last_active": time.time(),
            "summary": "",
        }
        (persist_dir / "ch1.json").write_text(json.dumps(data))

        mgr = SessionManager(
            max_history=10, max_age_hours=1, persist_dir=str(persist_dir),
        )
        mgr.load()
        assert mgr._sessions["ch1"].last_user_id is None

    def test_last_user_id_in_serialized_json(self, tmp_dir):
        """Verify last_user_id appears in the serialized JSON output."""
        persist_dir = tmp_dir / "sessions"
        persist_dir.mkdir()
        mgr = SessionManager(
            max_history=10, max_age_hours=1, persist_dir=str(persist_dir),
        )
        mgr.add_message("ch1", "user", "hello")
        mgr._sessions["ch1"].last_user_id = "12345"
        mgr.save()

        raw = json.loads((persist_dir / "ch1.json").read_text())
        assert raw["last_user_id"] == "12345"

    def test_last_user_id_roundtrip_multiple_sessions(self, tmp_dir):
        """Multiple sessions with different user IDs survive save/load."""
        persist_dir = tmp_dir / "sessions"
        persist_dir.mkdir()
        mgr1 = SessionManager(
            max_history=10, max_age_hours=1, persist_dir=str(persist_dir),
        )
        mgr1.add_message("ch1", "user", "hello")
        mgr1._sessions["ch1"].last_user_id = "user_a"
        mgr1.add_message("ch2", "user", "hi")
        mgr1._sessions["ch2"].last_user_id = "user_b"
        mgr1.save()

        mgr2 = SessionManager(
            max_history=10, max_age_hours=1, persist_dir=str(persist_dir),
        )
        mgr2.load()
        assert mgr2._sessions["ch1"].last_user_id == "user_a"
        assert mgr2._sessions["ch2"].last_user_id == "user_b"


# ── Bug 2: Consolidation preserves user_id ────────────────────────────

class TestConsolidationUserIdPreservation:
    """Consolidation must preserve user_id from original entries."""

    async def test_consolidation_prompt_mentions_user_id(self, reflector, mock_text_fn):
        """The consolidation prompt schema must include user_id."""
        entries = [
            {"key": f"k{i}", "category": "preference", "content": f"pref {i}",
             "user_id": "100000000000000001",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"}
            for i in range(6)
        ]

        await reflector._consolidate(entries)

        # The prompt sent to text_fn should mention user_id in the schema
        prompt_text = mock_text_fn.call_args[0][0][0]["content"]
        assert "user_id" in prompt_text

    async def test_user_id_preserved_from_original_when_llm_drops_it(
        self, reflector, mock_text_fn,
    ):
        """If LLM output lacks user_id, it should be restored from originals."""
        entries = [
            {"key": "aaron_tz", "category": "preference",
             "content": "prefers Eastern Time", "user_id": "100000000000000001",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"},
            {"key": "aaron_emoji", "category": "preference",
             "content": "dislikes emoji", "user_id": "100000000000000001",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"},
        ]

        # LLM returns merged entry WITHOUT user_id
        mock_text_fn.return_value = json.dumps([
            {"key": "aaron_tz", "category": "preference",
             "content": "prefers Eastern Time"},
        ])

        result = await reflector._consolidate(entries)

        assert len(result) == 1
        assert result[0]["user_id"] == "100000000000000001"

    async def test_user_id_kept_when_llm_includes_it(
        self, reflector, mock_text_fn,
    ):
        """If LLM output includes user_id, that value is kept."""
        entries = [
            {"key": "k1", "category": "preference", "content": "pref",
             "user_id": "111", "created_at": "2026-01-01",
             "updated_at": "2026-01-01"},
        ]

        mock_text_fn.return_value = json.dumps([
            {"key": "k1", "category": "preference", "content": "pref",
             "user_id": "111"},
        ])

        result = await reflector._consolidate(entries)

        assert result[0]["user_id"] == "111"

    async def test_global_entries_no_user_id_after_consolidation(
        self, reflector, mock_text_fn,
    ):
        """Global entries (no user_id) should NOT get a user_id after consolidation."""
        entries = [
            {"key": "server_ip", "category": "fact", "content": "10.0.0.1",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"},
        ]

        mock_text_fn.return_value = json.dumps([
            {"key": "server_ip", "category": "fact", "content": "10.0.0.1"},
        ])

        result = await reflector._consolidate(entries)

        assert "user_id" not in result[0]

    async def test_mixed_user_and_global_entries(self, reflector, mock_text_fn):
        """Consolidation preserves user_id for user entries, omits for global."""
        entries = [
            {"key": "pref1", "category": "preference", "content": "dark mode",
             "user_id": "111", "created_at": "2026-01-01",
             "updated_at": "2026-01-01"},
            {"key": "fact1", "category": "fact", "content": "5 machines",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"},
        ]

        # LLM drops user_id from both
        mock_text_fn.return_value = json.dumps([
            {"key": "pref1", "category": "preference", "content": "dark mode"},
            {"key": "fact1", "category": "fact", "content": "5 machines"},
        ])

        result = await reflector._consolidate(entries)

        pref = next(e for e in result if e["key"] == "pref1")
        fact = next(e for e in result if e["key"] == "fact1")
        assert pref["user_id"] == "111"
        assert "user_id" not in fact


# ── Bug 3: ToolMemory.record() expires stale entries ─────────────────

class TestToolMemoryRecordExpiry:
    """record() must expire stale entries so they don't accumulate."""

    async def test_record_expires_old_entries(self, tmp_dir):
        """Old entries should be removed when record() is called."""
        path = tmp_dir / "tool_memory.json"
        old_ts = (
            datetime.now(timezone.utc) - timedelta(days=EXPIRY_DAYS + 1)
        ).isoformat()
        now_ts = datetime.now(timezone.utc).isoformat()

        # Pre-populate with one old and one recent entry
        entries = [
            {"query": "old query", "keywords": ["old"], "tools_used": ["t1", "t2"],
             "success": True, "timestamp": old_ts},
            {"query": "recent query", "keywords": ["recent"],
             "tools_used": ["t3", "t4"], "success": True, "timestamp": now_ts},
        ]
        path.write_text(json.dumps(entries))

        tm = ToolMemory(data_path=str(path))
        # At init, _load() calls _expire(), so old entry should already be gone
        assert len(tm._entries) == 1

        # Now re-add an old entry manually to simulate accumulation
        tm._entries.append({
            "query": "stale", "keywords": ["stale"],
            "tools_used": ["t5"], "success": True, "timestamp": old_ts,
        })
        assert len(tm._entries) == 2

        # record() should expire the stale entry
        await tm.record("new check disk query", ["check_disk", "check_memory"])
        # Should have: recent + new (old was expired)
        assert len(tm._entries) == 2
        queries = [e["query"] for e in tm._entries]
        assert "stale" not in queries
        assert "recent query" in queries

    async def test_record_without_expiry_accumulates(self, tmp_dir):
        """Verify the fix is actually needed — without expire, old entries stay."""
        path = tmp_dir / "tool_memory.json"
        tm = ToolMemory(data_path=str(path))

        # Record many entries — all should persist since they're not expired
        for i in range(5):
            await tm.record(f"check disk on server {i}", ["check_disk", "run_command"])

        assert len(tm._entries) == 5

    async def test_record_expiry_preserves_recent(self, tmp_dir):
        """Recent entries should survive the expiry call in record()."""
        path = tmp_dir / "tool_memory.json"
        tm = ToolMemory(data_path=str(path))

        await tm.record("check disk", ["check_disk", "run_command"])
        await tm.record("check memory", ["check_memory", "query_prometheus"])
        await tm.record("deploy", ["run_command", "check_docker"])

        assert len(tm._entries) == 3
        # All entries should be recent and survive
        for entry in tm._entries:
            assert entry["timestamp"] >= (
                datetime.now(timezone.utc) - timedelta(seconds=10)
            ).isoformat()
