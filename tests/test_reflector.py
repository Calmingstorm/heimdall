"""Tests for learning/reflector.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.learning.reflector import ConversationReflector


@pytest.fixture
def reflector(tmp_dir: Path):
    r = ConversationReflector(
        learned_path=str(tmp_dir / "learned.json"),
        max_entries=5,
        consolidation_target=3,
    )
    r.set_text_fn(AsyncMock(return_value="[]"))
    return r


class TestConsolidate:
    async def test_consolidate_calls_text_fn(self, reflector):
        """_consolidate calls the registered text_fn."""
        entries = [
            {"key": f"k{i}", "category": "fact", "content": f"fact {i}",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"}
            for i in range(6)
        ]

        consolidated_json = json.dumps([
            {"key": "k0", "category": "fact", "content": "merged fact"},
            {"key": "k1", "category": "fact", "content": "another fact"},
        ])
        reflector._text_fn = AsyncMock(return_value=consolidated_json)

        result = await reflector._consolidate(entries)

        reflector._text_fn.assert_awaited_once()
        assert len(result) == 2

    async def test_consolidate_fallback_when_no_text_fn(self):
        """Without text_fn, consolidation falls back to trimming."""
        r = ConversationReflector(
            learned_path="/tmp/fake_learned.json",
            max_entries=5,
            consolidation_target=3,
        )
        # No text_fn set
        entries = [
            {"key": f"k{i}", "category": "fact", "content": f"fact {i}",
             "created_at": "2026-01-01", "updated_at": f"2026-01-0{i + 1}"}
            for i in range(6)
        ]

        result = await r._consolidate(entries)
        # Should fall back to trimming to consolidation_target (3)
        assert len(result) == 3

    async def test_consolidate_fallback_on_error(self, reflector):
        """On API failure, should fall back to keeping most recent entries."""
        entries = [
            {"key": f"k{i}", "category": "fact", "content": f"fact {i}",
             "created_at": "2026-01-01", "updated_at": f"2026-01-0{i + 1}"}
            for i in range(6)
        ]

        reflector._text_fn = AsyncMock(side_effect=Exception("API down"))
        result = await reflector._consolidate(entries)

        # Should fall back to trimming to consolidation_target (3)
        assert len(result) == 3


class TestParseEntries:
    def test_parses_valid_json(self):
        raw = json.dumps([
            {"key": "k1", "category": "fact", "content": "test fact"},
            {"key": "k2", "category": "correction", "content": "test correction"},
        ])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 2
        assert result[0]["key"] == "k1"

    def test_handles_markdown_fences(self):
        raw = '```json\n[{"key": "k1", "category": "fact", "content": "test"}]\n```'
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 1

    def test_rejects_invalid_category(self):
        raw = json.dumps([
            {"key": "k1", "category": "invalid_cat", "content": "test"},
        ])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 0

    def test_returns_empty_on_garbage(self):
        result = ConversationReflector._parse_entries("not json at all")
        assert result == []


class TestGetPromptSection:
    def test_empty_when_no_file(self, reflector):
        assert reflector.get_prompt_section() == ""

    def test_formats_entries(self, reflector, tmp_dir):
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "k1", "category": "fact", "content": "servers run Ubuntu"},
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(data))
        result = reflector.get_prompt_section()
        assert "## Learned Context" in result
        assert "[fact] servers run Ubuntu" in result
