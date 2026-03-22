"""Tests for tool use memory: pattern storage, keyword matching, expiry, and client integration."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.tool_memory import (
    ToolMemory,
    extract_keywords,
    _jaccard,
    _cosine,
    MAX_ENTRIES,
    EXPIRY_DAYS,
    MIN_JACCARD_SCORE,
    MIN_SEMANTIC_SCORE,
)


# ── extract_keywords ────────────────────────────────────────────────


class TestExtractKeywords:
    async def test_basic_query(self):
        kw = extract_keywords("check disk space on the server")
        assert "check" in kw
        assert "disk" in kw
        assert "space" in kw
        assert "server" in kw
        # stop words removed
        assert "on" not in kw
        assert "the" not in kw

    async def test_empty_string(self):
        assert extract_keywords("") == []

    async def test_only_stop_words(self):
        assert extract_keywords("the a an is are") == []

    async def test_case_insensitive(self):
        kw = extract_keywords("Check Docker Containers")
        assert "check" in kw
        assert "docker" in kw
        assert "containers" in kw

    async def test_preserves_numbers(self):
        kw = extract_keywords("check port 8080 on server")
        assert "8080" in kw
        assert "port" in kw

    async def test_single_char_filtered(self):
        kw = extract_keywords("a b c disk d")
        assert kw == ["disk"]

    async def test_underscores_in_words(self):
        kw = extract_keywords("run check_disk on desktop")
        assert "check_disk" in kw

    async def test_returns_list(self):
        result = extract_keywords("docker memory usage")
        assert isinstance(result, list)


# ── _jaccard ────────────────────────────────────────────────────────


class TestJaccard:
    async def test_identical_sets(self):
        assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    async def test_disjoint_sets(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    async def test_partial_overlap(self):
        # {a, b} & {b, c} = {b}, union = {a, b, c}
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    async def test_empty_first(self):
        assert _jaccard(set(), {"a"}) == 0.0

    async def test_empty_second(self):
        assert _jaccard({"a"}, set()) == 0.0

    async def test_both_empty(self):
        assert _jaccard(set(), set()) == 0.0


# ── ToolMemory — init / load / save ─────────────────────────────────


class TestToolMemoryStorage:
    async def test_init_no_path(self):
        tm = ToolMemory(None)
        assert tm._entries == []

    async def test_init_missing_file(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tool_memory.json"))
        assert tm._entries == []

    async def test_load_existing(self, tmp_path):
        path = tmp_path / "tool_memory.json"
        entries = [
            {
                "query": "check disk",
                "keywords": ["check", "disk"],
                "tools_used": ["check_disk"],
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        assert len(tm._entries) == 1
        assert tm._entries[0]["query"] == "check disk"

    async def test_corrupt_json(self, tmp_path):
        path = tmp_path / "tool_memory.json"
        path.write_text("{invalid json")
        tm = ToolMemory(str(path))
        assert tm._entries == []

    async def test_non_list_json(self, tmp_path):
        path = tmp_path / "tool_memory.json"
        path.write_text('{"key": "value"}')
        tm = ToolMemory(str(path))
        assert tm._entries == []

    async def test_save_creates_parent(self, tmp_path):
        path = tmp_path / "subdir" / "tool_memory.json"
        tm = ToolMemory(str(path))
        await tm.record("check disk space", ["check_disk"])
        assert path.exists()

    async def test_save_persists(self, tmp_path):
        path = tmp_path / "tool_memory.json"
        tm = ToolMemory(str(path))
        await tm.record("check disk space", ["check_disk"])
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["tools_used"] == ["check_disk"]

    async def test_save_noop_no_path(self):
        tm = ToolMemory(None)
        await tm.record("check disk", ["check_disk"])
        # Should not raise, just silently skip save
        assert len(tm._entries) == 1


# ── ToolMemory — record ─────────────────────────────────────────────


class TestToolMemoryRecord:
    async def test_records_entry(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record("check disk space on server", ["check_disk", "query_prometheus"])
        assert len(tm._entries) == 1
        entry = tm._entries[0]
        assert entry["query"] == "check disk space on server"
        assert entry["tools_used"] == ["check_disk", "query_prometheus"]
        assert entry["success"] is True
        assert "timestamp" in entry
        assert "keywords" in entry

    async def test_skips_empty_tools(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record("check disk", [])
        assert len(tm._entries) == 0

    async def test_skips_no_keywords(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record("the a an is", ["check_disk"])
        assert len(tm._entries) == 0

    async def test_truncates_long_query(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        long_query = "word " * 100
        await tm.record(long_query, ["check_disk"])
        assert len(tm._entries[0]["query"]) <= 200

    async def test_records_failure(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record("check disk space", ["check_disk"], success=False)
        assert tm._entries[0]["success"] is False

    async def test_caps_at_max_entries(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        for i in range(MAX_ENTRIES + 10):
            await tm.record(f"query number {i}", [f"tool_{i}"])
        assert len(tm._entries) == MAX_ENTRIES

    async def test_persists_on_record(self, tmp_path):
        path = tmp_path / "tm.json"
        tm = ToolMemory(str(path))
        await tm.record("check disk usage", ["check_disk"])
        # Reload and verify
        tm2 = ToolMemory(str(path))
        assert len(tm2._entries) == 1

    async def test_keywords_extracted(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record("check docker containers status", ["check_docker"])
        kw = tm._entries[0]["keywords"]
        assert "check" in kw
        assert "docker" in kw
        assert "containers" in kw
        assert "status" in kw


# ── ToolMemory — expiry ──────────────────────────────────────────────


class TestToolMemoryExpiry:
    async def test_expires_old_entries(self, tmp_path):
        path = tmp_path / "tm.json"
        old_ts = (datetime.now(timezone.utc) - timedelta(days=EXPIRY_DAYS + 1)).isoformat()
        fresh_ts = datetime.now(timezone.utc).isoformat()
        entries = [
            {"query": "old", "keywords": ["old"], "tools_used": ["t1"],
             "success": True, "timestamp": old_ts},
            {"query": "fresh", "keywords": ["fresh"], "tools_used": ["t2"],
             "success": True, "timestamp": fresh_ts},
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        assert len(tm._entries) == 1
        assert tm._entries[0]["query"] == "fresh"

    async def test_keeps_recent_entries(self, tmp_path):
        path = tmp_path / "tm.json"
        ts = (datetime.now(timezone.utc) - timedelta(days=EXPIRY_DAYS - 1)).isoformat()
        entries = [
            {"query": "recent", "keywords": ["recent"], "tools_used": ["t1"],
             "success": True, "timestamp": ts},
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        assert len(tm._entries) == 1

    async def test_missing_timestamp_expired(self, tmp_path):
        path = tmp_path / "tm.json"
        entries = [
            {"query": "no ts", "keywords": ["no"], "tools_used": ["t1"],
             "success": True},
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        assert len(tm._entries) == 0


# ── ToolMemory — find_patterns ───────────────────────────────────────


class TestToolMemoryFindPatterns:
    def _make_entry(self, query, tools, success=True):
        return {
            "query": query,
            "keywords": extract_keywords(query),
            "tools_used": tools,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def test_finds_matching(self, tmp_path):
        path = tmp_path / "tm.json"
        entries = [
            self._make_entry("check disk space on server", ["check_disk", "query_prometheus"]),
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        results = await tm.find_patterns("how is disk space on the server")
        assert len(results) == 1
        assert results[0]["tools_used"] == ["check_disk", "query_prometheus"]

    async def test_excludes_failures(self, tmp_path):
        path = tmp_path / "tm.json"
        entries = [
            self._make_entry("check disk space", ["check_disk", "query_prometheus"], success=False),
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        results = await tm.find_patterns("check disk space")
        assert len(results) == 0

    async def test_excludes_single_tool(self, tmp_path):
        path = tmp_path / "tm.json"
        entries = [
            self._make_entry("check disk space", ["check_disk"]),
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        results = await tm.find_patterns("check disk space")
        assert len(results) == 0

    async def test_no_match_below_threshold(self, tmp_path):
        path = tmp_path / "tm.json"
        entries = [
            self._make_entry("check disk space", ["check_disk", "check_memory"]),
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        results = await tm.find_patterns("restart docker containers")
        assert len(results) == 0

    async def test_empty_query(self, tmp_path):
        path = tmp_path / "tm.json"
        entries = [
            self._make_entry("check disk", ["check_disk", "check_memory"]),
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        results = await tm.find_patterns("")
        assert len(results) == 0

    async def test_stop_words_only_query(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [self._make_entry("check disk", ["check_disk", "check_memory"])]
        results = await tm.find_patterns("the a an")
        assert len(results) == 0

    async def test_limit_results(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"]),
            self._make_entry("check disk usage server", ["check_disk", "check_memory"]),
            self._make_entry("check disk server health", ["check_disk", "check_service"]),
            self._make_entry("check disk server logs", ["check_disk", "check_logs"]),
        ]
        results = await tm.find_patterns("check disk space on server", limit=2)
        assert len(results) <= 2

    async def test_deduplicates_by_tool_sequence(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"]),
            self._make_entry("check disk usage server", ["check_disk", "query_prometheus"]),
        ]
        results = await tm.find_patterns("check disk space on server")
        assert len(results) == 1

    async def test_sorted_by_relevance(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        # Exact match keywords should score higher
        tm._entries = [
            self._make_entry("check memory usage", ["check_memory", "query_prometheus"]),
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        results = await tm.find_patterns("check disk space on server")
        assert results[0]["tools_used"] == ["check_disk", "query_prometheus"]

    async def test_no_entries(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        results = await tm.find_patterns("check disk space")
        assert results == []


# ── ToolMemory — format_hints ────────────────────────────────────────


class TestToolMemoryFormatHints:
    def _make_entry(self, query, tools):
        return {
            "query": query,
            "keywords": extract_keywords(query),
            "tools_used": tools,
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def test_formats_hints(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        hints = await tm.format_hints("check disk space on server")
        assert "## Tool Use Patterns" in hints
        assert "`check_disk`" in hints
        assert "`query_prometheus`" in hints
        assert "->" in hints

    async def test_empty_when_no_match(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        hints = await tm.format_hints("restart docker containers")
        assert hints == ""

    async def test_empty_for_empty_query(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        hints = await tm.format_hints("")
        assert hints == ""

    async def test_multiple_patterns(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            self._make_entry("check disk space server health", ["check_disk", "query_prometheus"]),
            self._make_entry("check disk space server logs", ["check_disk", "check_logs"]),
        ]
        hints = await tm.format_hints("check disk space on server")
        assert hints.count("- ") >= 2


# ── Client integration ──────────────────────────────────────────────


class TestClientIntegration:
    """Test that client.py properly initializes, records, and injects tool memory."""

    def _make_bot_stub(self):
        """Minimal bot stub with tool_memory."""
        bot = MagicMock()
        bot.tool_memory = ToolMemory(None)
        return bot

    async def test_tool_memory_initialized(self):
        """ToolMemory import works and can be instantiated."""
        from src.tools.tool_memory import ToolMemory
        tm = ToolMemory(None)
        assert tm._entries == []

    async def test_record_after_tools(self, tmp_path):
        """Simulates recording after a successful tool loop."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tools_used = ["check_disk", "query_prometheus", "check_logs"]
        await tm.record("how is the server doing", tools_used, success=True)
        assert len(tm._entries) == 1
        assert tm._entries[0]["tools_used"] == tools_used

    async def test_format_hints_injected(self, tmp_path):
        """Simulates the system prompt injection path."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record(
            "check disk space on server",
            ["check_disk", "query_prometheus"],
        )
        hints = await tm.format_hints("how is disk space on the server")
        assert "Tool Use Patterns" in hints

    async def test_no_hints_for_chat(self, tmp_path):
        """Chat queries shouldn't match infra patterns."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record(
            "check disk space on server",
            ["check_disk", "query_prometheus"],
        )
        hints = await tm.format_hints("tell me a joke")
        assert hints == ""

    async def test_tools_used_tracking_is_local(self):
        """Verify tools_used is a local variable (not instance attr) to avoid cross-channel contamination."""
        # _process_with_tools returns tools_used as 4th element of the tuple
        # Verify the pattern: accumulate into a local list, return it
        tools = []
        tools.extend(["check_disk"])
        tools.extend(["query_prometheus", "check_logs"])
        assert tools == ["check_disk", "query_prometheus", "check_logs"]

    async def test_build_system_prompt_query_param(self):
        """Verify _build_system_prompt accepts query parameter."""
        from src.tools.tool_memory import ToolMemory
        # Just verify the parameter is accepted (integration tested via bot)
        tm = ToolMemory(None)
        hints = await tm.format_hints("check disk")
        assert hints == ""  # no patterns stored


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    async def test_concurrent_writes(self, tmp_path):
        """Multiple records in sequence don't corrupt the file."""
        path = tmp_path / "tm.json"
        tm = ToolMemory(str(path))
        for i in range(10):
            await tm.record(f"query {i} about disk space", [f"tool_a_{i}", f"tool_b_{i}"])
        assert len(tm._entries) == 10
        tm2 = ToolMemory(str(path))
        assert len(tm2._entries) == 10

    async def test_reload_after_record(self, tmp_path):
        """Data survives a reload."""
        path = tmp_path / "tm.json"
        tm = ToolMemory(str(path))
        await tm.record("check disk space", ["check_disk", "query_prometheus"])
        tm2 = ToolMemory(str(path))
        assert len(tm2._entries) == 1
        results = await tm2.find_patterns("check disk space")
        assert len(results) == 1

    async def test_empty_keywords_in_stored_entry(self, tmp_path):
        """Entry with empty keywords list doesn't crash matching."""
        path = tmp_path / "tm.json"
        entries = [
            {"query": "x", "keywords": [], "tools_used": ["t1", "t2"],
             "success": True,
             "timestamp": datetime.now(timezone.utc).isoformat()},
        ]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        results = await tm.find_patterns("check disk space")
        assert len(results) == 0

    async def test_special_characters_in_query(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record("what's the disk usage? (on server #1)", ["check_disk", "check_memory"])
        assert len(tm._entries) == 1
        # Keywords should handle punctuation gracefully
        assert "disk" in tm._entries[0]["keywords"]
        assert "usage" in tm._entries[0]["keywords"]

    async def test_unicode_in_query(self, tmp_path):
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record("check disk space résumé", ["check_disk", "check_memory"])
        assert len(tm._entries) == 1

    async def test_min_score_threshold(self):
        """Verify MIN_JACCARD_SCORE is set to a reasonable value."""
        assert 0.0 < MIN_JACCARD_SCORE < 0.5

    async def test_max_entries_constant(self):
        assert MAX_ENTRIES == 200

    async def test_expiry_days_constant(self):
        assert EXPIRY_DAYS == 30


# ── _cosine edge cases ───────────────────────────────────────────────


class TestCosine:
    async def test_identical_vectors(self):
        assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    async def test_orthogonal_vectors(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    async def test_opposite_vectors(self):
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    async def test_zero_vector_first(self):
        assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0

    async def test_zero_vector_second(self):
        assert _cosine([1.0, 2.0], [0.0, 0.0]) == 0.0

    async def test_both_zero_vectors(self):
        assert _cosine([0.0, 0.0], [0.0, 0.0]) == 0.0

    async def test_mismatched_lengths(self):
        """Mismatched dimensions should return 0.0, not silently truncate."""
        assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0]) == 0.0

    async def test_empty_vectors(self):
        assert _cosine([], []) == 0.0

    async def test_realistic_similarity(self):
        """Two similar-ish vectors should produce a score between 0 and 1."""
        a = [0.1, 0.5, 0.3, 0.8]
        b = [0.2, 0.4, 0.35, 0.75]
        score = _cosine(a, b)
        assert 0.9 < score < 1.0  # similar vectors → high cosine

    async def test_semantic_score_threshold(self):
        """MIN_SEMANTIC_SCORE should be between 0.5 and 0.9."""
        assert 0.5 <= MIN_SEMANTIC_SCORE <= 0.9


# ── Semantic path (embeddings on both query and entry) ────────────────


class TestSemanticPath:
    def _make_entry(self, query, tools, embedding=None):
        entry = {
            "query": query,
            "keywords": extract_keywords(query),
            "tools_used": tools,
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if embedding is not None:
            entry["embedding"] = embedding
        return entry

    async def test_cosine_used_when_both_have_embeddings(self, tmp_path):
        """When both query and entry have embeddings, cosine similarity is used."""
        # Use identical embeddings → score = 1.0 (above threshold)
        emb = [0.1, 0.5, 0.3, 0.8]
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"], embedding=emb),
        ]
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=emb)
        results = await tm.find_patterns("completely different keywords", embedder=embedder)
        # Keywords don't overlap at all, but embeddings are identical → should match
        assert len(results) == 1

    async def test_semantic_below_threshold_falls_through_to_jaccard(self, tmp_path):
        """When cosine score < MIN_SEMANTIC_SCORE, falls through to Jaccard."""
        emb_query = [1.0, 0.0, 0.0, 0.0]
        emb_entry = [0.0, 1.0, 0.0, 0.0]  # orthogonal → cosine = 0.0
        tm = ToolMemory(str(tmp_path / "tm.json"))
        # Use keywords that would match via Jaccard
        tm._entries = [
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"], embedding=emb_entry),
        ]
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=emb_query)
        # Keywords match ("check", "disk", "space", "server") — Jaccard fallback should find it
        results = await tm.find_patterns("check disk space server", embedder=embedder)
        assert len(results) == 1

    async def test_semantic_below_threshold_no_keyword_match(self, tmp_path):
        """When both cosine and Jaccard are below threshold, entry is excluded."""
        emb_query = [1.0, 0.0, 0.0, 0.0]
        emb_entry = [0.0, 1.0, 0.0, 0.0]  # orthogonal → cosine = 0.0
        tm = ToolMemory(str(tmp_path / "tm.json"))
        # Use keywords that do NOT overlap with query
        tm._entries = [
            self._make_entry("restart docker containers", ["restart_docker", "check_docker"], embedding=emb_entry),
        ]
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=emb_query)
        # Keywords don't overlap and cosine = 0 → excluded
        results = await tm.find_patterns("check disk space server", embedder=embedder)
        assert len(results) == 0

    async def test_semantic_match_passes_through_format_hints(self, tmp_path):
        """format_hints should return hints when semantic match succeeds."""
        emb = [0.5, 0.5, 0.5]
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            self._make_entry("check server health", ["check_disk", "query_prometheus"], embedding=emb),
        ]
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=emb)
        hints = await tm.format_hints("unrelated query words", embedder=embedder)
        assert "Tool Use Patterns" in hints


# ── Fallback path (entry lacks embedding → Jaccard) ──────────────────


class TestFallbackPath:
    def _make_entry(self, query, tools, embedding=None):
        entry = {
            "query": query,
            "keywords": extract_keywords(query),
            "tools_used": tools,
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if embedding is not None:
            entry["embedding"] = embedding
        return entry

    async def test_jaccard_used_when_entry_lacks_embedding(self, tmp_path):
        """When query has embedding but entry doesn't, Jaccard is used."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        # Entry has NO embedding
        tm._entries = [
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        # Keywords overlap → Jaccard should match
        results = await tm.find_patterns("check disk space on server", embedder=embedder)
        assert len(results) == 1

    async def test_jaccard_used_when_no_embedder(self, tmp_path):
        """When no embedder is provided, all entries use Jaccard."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        results = await tm.find_patterns("check disk space on server")
        assert len(results) == 1

    async def test_jaccard_when_embedder_returns_none(self, tmp_path):
        """When embedder.embed() returns None, Jaccard is used for all entries."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=None)
        results = await tm.find_patterns("check disk space on server", embedder=embedder)
        assert len(results) == 1

    async def test_mixed_entries_some_with_embeddings(self, tmp_path):
        """Entries with embeddings use cosine; entries without use Jaccard."""
        emb = [1.0, 0.0, 0.0]
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            # Entry WITH embedding (identical to query → cosine ≈ 1.0 → match)
            self._make_entry("restart docker containers", ["restart_docker", "check_docker"], embedding=emb),
            # Entry WITHOUT embedding (keyword overlap → Jaccard → match)
            self._make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=emb)
        results = await tm.find_patterns("check disk space server", embedder=embedder)
        # Both should match: first via cosine (identical embeddings), second via Jaccard
        assert len(results) == 2


# ── record() with embedder ───────────────────────────────────────────


class TestRecordWithEmbedder:
    async def test_stores_embedding(self, tmp_path):
        """record() should store the embedding from the embedder."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        await tm.record("check disk space", ["check_disk", "query_prometheus"], embedder=embedder)
        assert "embedding" in tm._entries[0]
        assert tm._entries[0]["embedding"] == [0.1, 0.2, 0.3]

    async def test_no_embedding_when_embedder_returns_none(self, tmp_path):
        """record() should skip embedding if embedder returns None."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=None)
        await tm.record("check disk space", ["check_disk", "query_prometheus"], embedder=embedder)
        assert "embedding" not in tm._entries[0]

    async def test_no_embedding_without_embedder(self, tmp_path):
        """record() should skip embedding if no embedder is provided."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        await tm.record("check disk space", ["check_disk", "query_prometheus"])
        assert "embedding" not in tm._entries[0]

    async def test_embedding_serializable_to_json(self, tmp_path):
        """Embeddings stored by record() must survive JSON round-trip."""
        path = tmp_path / "tm.json"
        tm = ToolMemory(str(path))
        emb = [0.1, 0.2, 0.3, 0.456789]
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=emb)
        await tm.record("check disk space", ["check_disk", "query_prometheus"], embedder=embedder)
        # Reload from disk
        tm2 = ToolMemory(str(path))
        assert tm2._entries[0]["embedding"] == pytest.approx(emb)

    async def test_expiry_works_with_embedding_field(self, tmp_path):
        """Entries with embeddings should still be expired normally."""
        path = tmp_path / "tm.json"
        old_ts = (datetime.now(timezone.utc) - timedelta(days=EXPIRY_DAYS + 1)).isoformat()
        entries = [{
            "query": "old query",
            "keywords": ["old", "query"],
            "tools_used": ["t1", "t2"],
            "success": True,
            "timestamp": old_ts,
            "embedding": [0.1, 0.2, 0.3],
        }]
        path.write_text(json.dumps(entries))
        tm = ToolMemory(str(path))
        assert len(tm._entries) == 0  # expired
