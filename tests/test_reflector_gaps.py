"""Tests for reflector.py coverage gaps.

Targets uncovered lines: 59-60, 98, 101, 112, 114, 160-162, 177,
197, 207, 267-268, 299-303, 309.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.learning.reflector import ConversationReflector


@pytest.fixture
def mock_text_fn():
    return AsyncMock(return_value="[]")


@pytest.fixture
def reflector(tmp_dir: Path, mock_text_fn):
    r = ConversationReflector(
        learned_path=str(tmp_dir / "learned.json"),
        max_entries=5,
        consolidation_target=3,
    )
    r.set_text_fn(mock_text_fn)
    return r


def _make_session(n_messages=5, summary="test summary"):
    """Create a mock Session with n messages."""
    session = MagicMock()
    session.summary = summary
    msgs = []
    for i in range(n_messages):
        m = MagicMock()
        m.role = "user" if i % 2 == 0 else "assistant"
        m.content = f"Message {i}"
        msgs.append(m)
    session.messages = msgs
    return session


class TestLoadError:
    def test_load_corrupt_json(self, tmp_dir: Path):
        """Corrupt learned.json triggers error path (lines 59-60)."""
        path = tmp_dir / "learned.json"
        path.write_text("not valid json {{{")
        r = ConversationReflector(
            learned_path=str(path),
        )
        # Should return default structure, not crash
        result = r.get_prompt_section()
        assert result == ""


class TestReflectOnSessionGuards:
    @pytest.mark.asyncio
    async def test_disabled_reflector_returns_early(self, tmp_dir: Path):
        """Disabled reflector returns immediately (line 98)."""
        r = ConversationReflector(
            learned_path=str(tmp_dir / "learned.json"),
            enabled=False,
        )
        session = _make_session(10)
        await r.reflect_on_session(session)
        # No learned.json should be created
        assert not (tmp_dir / "learned.json").exists()

    @pytest.mark.asyncio
    async def test_short_session_skipped(self, reflector):
        """Session with < 3 messages is skipped (line 101)."""
        session = _make_session(2)
        await reflector.reflect_on_session(session)
        # Short session should not create learned.json
        assert not reflector._path.exists()


class TestReflectOnCompactedGuards:
    @pytest.mark.asyncio
    async def test_disabled_reflector_compacted(self, tmp_dir: Path):
        """Disabled reflector skips compacted reflection (line 112)."""
        r = ConversationReflector(
            learned_path=str(tmp_dir / "learned.json"),
            enabled=False,
        )
        msgs = [MagicMock(role="user", content=f"msg {i}") for i in range(10)]
        await r.reflect_on_compacted(msgs, "summary")
        # No learned.json should be created
        assert not (tmp_dir / "learned.json").exists()

    @pytest.mark.asyncio
    async def test_short_compacted_skipped(self, reflector):
        """Compacted with < 5 messages is skipped (line 114)."""
        msgs = [MagicMock(role="user", content=f"msg {i}") for i in range(3)]
        await reflector.reflect_on_compacted(msgs, "summary")
        # Short compacted should not create learned.json
        assert not reflector._path.exists()


class TestReflectAPIError:
    @pytest.mark.asyncio
    async def test_api_error_returns_early(self, reflector, mock_text_fn):
        """API call failure logs error and returns (lines 160-162)."""
        session = _make_session(5)

        mock_text_fn.side_effect = Exception("API down")
        await reflector.reflect_on_session(session)

        # No data should be saved
        assert not Path(reflector._path).exists()


class TestReflectCompactedFiltering:
    @pytest.mark.asyncio
    async def test_compacted_filters_non_correction_operational(self, reflector, mock_text_fn):
        """Non-full reflection filters out preference/fact entries (line 177)."""
        msgs = [MagicMock(role="user", content=f"msg {i}") for i in range(10)]

        # Return only preference entries, which get filtered in non-full mode
        mock_text_fn.return_value = json.dumps([
            {"key": "k1", "category": "preference", "content": "likes dark mode"},
        ])

        await reflector.reflect_on_compacted(msgs, "summary")

        # Since only preferences were returned and non-full filters them, nothing saved
        if reflector._path.exists():
            data = json.loads(reflector._path.read_text())
            assert len(data.get("entries", [])) == 0


class TestReflectUserIdTagging:
    @pytest.mark.asyncio
    async def test_user_id_tagged_on_existing_entry(self, reflector, mock_text_fn, tmp_dir):
        """user_id is set on existing entries during merge (line 197)."""
        # Pre-populate with an existing entry
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "k1", "category": "correction", "content": "old content",
                 "created_at": "2026-01-01", "updated_at": "2026-01-01"},
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(data))

        session = _make_session(5)
        mock_text_fn.return_value = json.dumps([
            {"key": "k1", "category": "correction", "content": "updated content"},
        ])

        await reflector.reflect_on_session(session, user_id="user123")

        saved = json.loads((tmp_dir / "learned.json").read_text())
        entries = saved["entries"]
        assert len(entries) == 1
        assert entries[0]["user_id"] == "user123"
        assert entries[0]["content"] == "updated content"


class TestConsolidationTrigger:
    @pytest.mark.asyncio
    async def test_consolidation_triggered_when_over_max(self, reflector, mock_text_fn, tmp_dir):
        """Consolidation is triggered when entries exceed max_entries (line 207)."""
        # Pre-populate with entries at the limit
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": f"k{i}", "category": "fact", "content": f"fact {i}",
                 "created_at": "2026-01-01", "updated_at": "2026-01-01"}
                for i in range(5)  # max_entries is 5
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(data))

        session = _make_session(5)

        # First call: reflection returns a new entry (pushing over limit)
        reflect_text = json.dumps([
            {"key": "k_new", "category": "fact", "content": "brand new fact"},
        ])

        # Second call: consolidation returns trimmed entries
        consolidate_text = json.dumps([
            {"key": "k0", "category": "fact", "content": "merged fact 0"},
            {"key": "k1", "category": "fact", "content": "merged fact 1"},
            {"key": "k_new", "category": "fact", "content": "brand new fact"},
        ])

        call_count = 0
        async def side_effect(messages, system):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return reflect_text
            return consolidate_text

        mock_text_fn.side_effect = side_effect
        await reflector.reflect_on_session(session)

        assert call_count == 2  # Reflection + consolidation
        saved = json.loads((tmp_dir / "learned.json").read_text())
        assert len(saved["entries"]) == 3  # Consolidated


class TestConsolidateNewEntryTimestamps:
    async def test_new_entries_get_timestamps(self, reflector, mock_text_fn):
        """New entries from consolidation get created_at/updated_at (lines 267-268)."""
        entries = [
            {"key": f"k{i}", "category": "fact", "content": f"fact {i}",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"}
            for i in range(6)
        ]

        # Return a new entry not in originals
        mock_text_fn.return_value = json.dumps([
            {"key": "k0", "category": "fact", "content": "existing"},
            {"key": "brand_new", "category": "fact", "content": "totally new"},
        ])

        result = await reflector._consolidate(entries)

        # brand_new should have timestamps
        new_entry = [e for e in result if e["key"] == "brand_new"][0]
        assert "created_at" in new_entry
        assert "updated_at" in new_entry


class TestParseEntriesEdgeCases:
    def test_embedded_json_array(self):
        """JSON array embedded in text is extracted (lines 299-303)."""
        raw = 'Here are the entries: [{"key": "k1", "category": "fact", "content": "test"}] done.'
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 1
        assert result[0]["key"] == "k1"

    def test_no_json_array_in_text(self):
        """Text with no JSON array returns empty (line 305-306)."""
        raw = "Just some text with no brackets"
        result = ConversationReflector._parse_entries(raw)
        assert result == []

    def test_invalid_json_within_brackets(self):
        """Malformed JSON inside brackets returns empty (lines 301-303)."""
        raw = '[{"key": bad json here}]'
        result = ConversationReflector._parse_entries(raw)
        assert result == []

    def test_non_list_json(self):
        """JSON that parses but isn't a list returns empty (line 309)."""
        raw = '{"key": "k1", "category": "fact", "content": "test"}'
        result = ConversationReflector._parse_entries(raw)
        assert result == []

    async def test_consolidation_empty_returns_trimmed(self, reflector, mock_text_fn):
        """Consolidation returning empty entries falls back to trimmed originals (lines 285-288)."""
        # LLM returns empty/invalid JSON → _parse_entries returns [] → fallback to trimmed
        mock_text_fn.return_value = "not valid json"
        entries = [
            {"key": f"k{i}", "category": "fact", "content": f"entry {i}",
             "updated_at": f"2024-01-0{i+1}T00:00:00+00:00"}
            for i in range(5)
        ]
        result = await reflector._consolidate(entries)
        # Should return trimmed list (consolidation_target=3), sorted by updated_at desc
        assert len(result) == 3
        assert result[0]["key"] == "k4"  # most recent first

    async def test_user_id_preserved_in_consolidation(self, reflector, mock_text_fn):
        """user_id from original is preserved when consolidation drops it (lines 299-301)."""
        # Original entries have user_id, consolidated entries don't → should be restored
        originals = [
            {"key": "k1", "category": "preference", "content": "likes dark mode",
             "user_id": "user_42", "updated_at": "2024-01-01T00:00:00+00:00"},
            {"key": "k2", "category": "fact", "content": "runs Ubuntu",
             "user_id": "user_99", "updated_at": "2024-01-02T00:00:00+00:00"},
        ]
        # LLM returns consolidated entries WITHOUT user_id
        mock_text_fn.return_value = json.dumps([
            {"key": "k1", "category": "preference", "content": "likes dark mode"},
            {"key": "k2", "category": "fact", "content": "runs Ubuntu"},
        ])
        result = await reflector._consolidate(originals)
        assert len(result) == 2
        assert result[0]["user_id"] == "user_42"
        assert result[1]["user_id"] == "user_99"

    def test_parse_entries_preserves_user_id(self):
        """_parse_entries should preserve user_id when present in LLM response."""
        raw = json.dumps([
            {"key": "k1", "category": "preference", "content": "likes dark mode",
             "user_id": "user_42"},
            {"key": "k2", "category": "operational", "content": "server runs Ubuntu"},
        ])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 2
        assert result[0].get("user_id") == "user_42"
        assert "user_id" not in result[1]

    def test_parse_entries_ignores_empty_user_id(self):
        """_parse_entries should not include user_id when it's empty/None."""
        raw = json.dumps([
            {"key": "k1", "category": "fact", "content": "test", "user_id": ""},
            {"key": "k2", "category": "fact", "content": "test2", "user_id": None},
        ])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 2
        assert "user_id" not in result[0]
        assert "user_id" not in result[1]

    def test_consolidation_preserves_user_id_through_parse(self):
        """When consolidation LLM returns entries with user_id, they survive _parse_entries."""
        raw = json.dumps([
            {"key": "k1", "category": "preference", "content": "likes verbose",
             "user_id": "user_A"},
            {"key": "k2", "category": "correction", "content": "no emojis",
             "user_id": "user_B"},
        ])
        result = ConversationReflector._parse_entries(raw)
        assert result[0]["user_id"] == "user_A"
        assert result[1]["user_id"] == "user_B"

    def test_parse_entries_fence_stripping_extracts_inner_json(self):
        """Fence stripping should only keep content inside ``` fences."""
        inner = json.dumps([
            {"key": "k1", "category": "fact", "content": "inside fence"},
        ])
        raw = f"Here are the entries:\n```json\n{inner}\n```\nHope that helps!"
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 1
        assert result[0]["content"] == "inside fence"

    def test_parse_entries_fence_stripping_excludes_outer_text(self):
        """Text outside fences with ] character should not break parsing."""
        inner = json.dumps([
            {"key": "k1", "category": "operational", "content": "data"},
        ])
        raw = f"Results below:\n```\n{inner}\n```\nDone] extra bracket"
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 1
        assert result[0]["key"] == "k1"


class TestMultiUserReflection:
    """Tests for multi-user reflection attribution (audit issue 10)."""

    @pytest.mark.asyncio
    async def test_multi_user_reflect_includes_user_hint(self, reflector, mock_text_fn):
        """When multiple user_ids are passed, the prompt should instruct LLM to attribute."""
        session = _make_session(5)
        mock_text_fn.return_value = json.dumps([
            {"key": "k1", "category": "preference", "content": "likes dark mode",
             "user_id": "user_A"},
        ])

        await reflector._reflect(
            "test conversation", full=True,
            user_ids=["user_A", "user_B"],
        )

        # Verify the prompt sent to text_fn mentions the participant IDs
        prompt_text = mock_text_fn.call_args[0][0][0]["content"]
        assert "user_A" in prompt_text
        assert "user_B" in prompt_text
        assert "Multiple users participated" in prompt_text

    @pytest.mark.asyncio
    async def test_single_user_reflect_no_hint(self, reflector, mock_text_fn):
        """When only one user_id is passed, no multi-user hint in prompt."""
        session = _make_session(5)
        mock_text_fn.return_value = json.dumps([
            {"key": "k1", "category": "preference", "content": "likes dark mode"},
        ])

        await reflector._reflect(
            "test conversation", full=True,
            user_ids=["user_A"],
        )

        prompt_text = mock_text_fn.call_args[0][0][0]["content"]
        assert "Multiple users participated" not in prompt_text

    @pytest.mark.asyncio
    async def test_multi_user_llm_attributed_user_id_preserved(self, reflector, mock_text_fn, tmp_dir):
        """When LLM returns user_id in multi-user mode, it should be saved."""
        mock_text_fn.return_value = json.dumps([
            {"key": "pref_a", "category": "preference", "content": "likes verbose",
             "user_id": "user_A"},
            {"key": "pref_b", "category": "correction", "content": "no emojis",
             "user_id": "user_B"},
        ])

        await reflector._reflect(
            "test conversation", full=True,
            user_ids=["user_A", "user_B"],
        )

        data = json.loads((tmp_dir / "learned.json").read_text())
        entries = {e["key"]: e for e in data["entries"]}
        assert entries["pref_a"]["user_id"] == "user_A"
        assert entries["pref_b"]["user_id"] == "user_B"


class TestFormatConversationUserAttribution:
    """Tests for _format_conversation including user_id in output."""

    def test_format_includes_user_id(self, reflector):
        """Messages with user_id should include it in formatted output."""
        m1 = MagicMock()
        m1.role = "user"
        m1.content = "hello"
        m1.user_id = "42"
        m2 = MagicMock()
        m2.role = "assistant"
        m2.content = "hi"
        m2.user_id = None

        result = reflector._format_conversation([m1, m2])
        assert "user [user_id=42]:" in result
        assert "assistant:" in result  # no user_id for assistant

    def test_format_without_user_id(self, reflector):
        """Messages without user_id should format normally."""
        m1 = MagicMock()
        m1.role = "user"
        m1.content = "hello"
        m1.user_id = None

        result = reflector._format_conversation([m1])
        assert "user:" in result
        assert "user_id" not in result


class TestGetPromptSectionFiltering:
    def test_filters_by_user_id(self, reflector, tmp_dir):
        """Entries for other users are excluded (lines 80-87)."""
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "k1", "category": "fact", "content": "global fact"},
                {"key": "k2", "category": "preference", "content": "user A pref", "user_id": "userA"},
                {"key": "k3", "category": "preference", "content": "user B pref", "user_id": "userB"},
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(data))

        result = reflector.get_prompt_section(user_id="userA")
        assert "global fact" in result
        assert "user A pref" in result
        assert "user B pref" not in result

    def test_no_user_id_shows_global_only(self, reflector, tmp_dir):
        """Without user_id, only global entries shown."""
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "k1", "category": "fact", "content": "global fact"},
                {"key": "k2", "category": "preference", "content": "user pref", "user_id": "userA"},
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(data))

        result = reflector.get_prompt_section()
        assert "global fact" in result
        assert "user pref" not in result

    def test_all_filtered_returns_empty(self, reflector, tmp_dir):
        """When all entries belong to other users, returns empty (line 88-89)."""
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "k1", "category": "preference", "content": "other user", "user_id": "otherUser"},
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(data))

        result = reflector.get_prompt_section(user_id="myUser")
        assert result == ""
