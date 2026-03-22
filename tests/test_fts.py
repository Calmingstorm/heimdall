"""Tests for SQLite FTS5 full-text search (fts.py).

Uses real SQLite via tmp_path — no mocking needed. Tests verify:
- Index creation and availability
- Session indexing, searching, upsert (delete+insert), and has_session
- Knowledge chunk indexing, searching, deletion, and has_knowledge_chunk
- _prepare_query: quoting special chars, IPs, paths, empty strings
- Error handling: unavailable index returns safe defaults
"""
from __future__ import annotations

import sqlite3

import pytest

from src.search.fts import FullTextIndex, _prepare_query


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fts(tmp_path):
    """Create a FullTextIndex backed by a temp SQLite database."""
    return FullTextIndex(str(tmp_path / "test_fts.db"))


@pytest.fixture
def broken_fts():
    """Create a FullTextIndex that failed init (unavailable)."""
    # A non-writable path will fail
    idx = FullTextIndex("/nonexistent/dir/impossible.db")
    return idx


# ── Availability ────────────────────────────────────────────────────


class TestAvailability:
    async def test_available_after_init(self, fts):
        """Successful init sets available=True."""
        assert fts.available is True

    async def test_unavailable_on_bad_path(self, broken_fts):
        """Failed init sets available=False."""
        assert broken_fts.available is False


# ── Session indexing ────────────────────────────────────────────────


class TestSessionIndexing:
    async def test_index_session_returns_true(self, fts):
        """Indexing a session returns True on success."""
        result = fts.index_session("s1", "hello world", "chan1", 1000.0)
        assert result is True

    async def test_has_session_after_index(self, fts):
        """has_session returns True for indexed session."""
        fts.index_session("s1", "test content", "chan1", 1000.0)
        assert fts.has_session("s1") is True

    async def test_has_session_missing(self, fts):
        """has_session returns False for non-existent session."""
        assert fts.has_session("nonexistent") is False

    async def test_index_session_upsert(self, fts):
        """Re-indexing a session replaces old content."""
        fts.index_session("s1", "original content", "chan1", 1000.0)
        fts.index_session("s1", "updated content", "chan1", 2000.0)
        # Search for old content should miss
        old_results = fts.search_sessions("original")
        assert len(old_results) == 0
        # Search for new content should hit
        new_results = fts.search_sessions("updated")
        assert len(new_results) == 1
        assert new_results[0]["doc_id"] == "s1"

    async def test_index_session_when_unavailable(self, broken_fts):
        """Indexing returns False when FTS is unavailable."""
        assert broken_fts.index_session("s1", "text", "c", 0.0) is False

    async def test_has_session_when_unavailable(self, broken_fts):
        """has_session returns False when FTS is unavailable."""
        assert broken_fts.has_session("s1") is False


# ── Session searching ───────────────────────────────────────────────


class TestSessionSearch:
    async def test_search_returns_matching_session(self, fts):
        """Basic keyword search finds indexed sessions."""
        fts.index_session("s1", "prometheus alerting rules configuration", "c1", 100.0)
        fts.index_session("s2", "docker container management", "c1", 200.0)
        results = fts.search_sessions("prometheus")
        assert len(results) == 1
        assert results[0]["doc_id"] == "s1"

    async def test_search_returns_correct_fields(self, fts):
        """Results contain doc_id, content, channel_id, timestamp, type, rank."""
        fts.index_session("s1", "network configuration guide", "c42", 123.5)
        results = fts.search_sessions("network")
        assert len(results) == 1
        r = results[0]
        assert r["doc_id"] == "s1"
        assert "network" in r["content"].lower()
        assert r["channel_id"] == "c42"
        assert r["timestamp"] == 123.5
        assert r["type"] == "fts"
        assert "rank" in r

    async def test_search_empty_query(self, fts):
        """Empty query returns empty list."""
        fts.index_session("s1", "some text", "c1", 100.0)
        assert fts.search_sessions("") == []
        assert fts.search_sessions("   ") == []

    async def test_search_no_match(self, fts):
        """Query that doesn't match anything returns empty list."""
        fts.index_session("s1", "grafana dashboard setup", "c1", 100.0)
        results = fts.search_sessions("kubernetes")
        assert results == []

    async def test_search_respects_limit(self, fts):
        """Limit parameter caps the number of results."""
        for i in range(10):
            fts.index_session(f"s{i}", f"server monitoring topic {i}", f"c1", float(i))
        results = fts.search_sessions("server", limit=3)
        assert len(results) == 3

    async def test_search_multiple_matches(self, fts):
        """Multiple matching sessions are returned."""
        fts.index_session("s1", "disk space on server", "c1", 100.0)
        fts.index_session("s2", "disk usage alert", "c1", 200.0)
        fts.index_session("s3", "memory usage report", "c1", 300.0)
        results = fts.search_sessions("disk")
        assert len(results) == 2
        ids = {r["doc_id"] for r in results}
        assert ids == {"s1", "s2"}

    async def test_search_when_unavailable(self, broken_fts):
        """Searching when unavailable returns empty list."""
        assert broken_fts.search_sessions("test") == []


# ── Knowledge chunk indexing ────────────────────────────────────────


class TestKnowledgeIndexing:
    async def test_index_knowledge_returns_true(self, fts):
        """Indexing a knowledge chunk returns True."""
        result = fts.index_knowledge_chunk("k1", "ansible playbook docs", "wiki", 0)
        assert result is True

    async def test_has_knowledge_chunk_after_index(self, fts):
        """has_knowledge_chunk returns True for indexed chunk."""
        fts.index_knowledge_chunk("k1", "test content", "wiki", 0)
        assert fts.has_knowledge_chunk("k1") is True

    async def test_has_knowledge_chunk_missing(self, fts):
        """has_knowledge_chunk returns False for non-existent chunk."""
        assert fts.has_knowledge_chunk("nonexistent") is False

    async def test_index_knowledge_upsert(self, fts):
        """Re-indexing a chunk replaces old content."""
        fts.index_knowledge_chunk("k1", "original docs", "wiki", 0)
        fts.index_knowledge_chunk("k1", "updated docs", "wiki", 0)
        old = fts.search_knowledge("original")
        assert len(old) == 0
        new = fts.search_knowledge("updated")
        assert len(new) == 1

    async def test_index_knowledge_when_unavailable(self, broken_fts):
        """Indexing returns False when FTS is unavailable."""
        assert broken_fts.index_knowledge_chunk("k1", "t", "s", 0) is False

    async def test_has_knowledge_when_unavailable(self, broken_fts):
        """has_knowledge_chunk returns False when unavailable."""
        assert broken_fts.has_knowledge_chunk("k1") is False


# ── Knowledge searching ────────────────────────────────────────────


class TestKnowledgeSearch:
    async def test_search_returns_matching_chunk(self, fts):
        """Basic keyword search finds indexed knowledge chunks."""
        fts.index_knowledge_chunk("k1", "ansible vault encryption setup", "docs", 0)
        fts.index_knowledge_chunk("k2", "docker compose networking", "docs", 1)
        results = fts.search_knowledge("ansible")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "k1"

    async def test_search_returns_correct_fields(self, fts):
        """Knowledge results contain chunk_id, content, source, chunk_index, type, rank."""
        fts.index_knowledge_chunk("k1", "pihole dns configuration", "wiki", 3)
        results = fts.search_knowledge("pihole")
        assert len(results) == 1
        r = results[0]
        assert r["chunk_id"] == "k1"
        assert "pihole" in r["content"].lower()
        assert r["source"] == "wiki"
        assert r["chunk_index"] == 3
        assert r["type"] == "fts"
        assert "rank" in r

    async def test_search_empty_query(self, fts):
        """Empty query returns empty list."""
        fts.index_knowledge_chunk("k1", "some text", "wiki", 0)
        assert fts.search_knowledge("") == []

    async def test_search_no_match(self, fts):
        """Non-matching query returns empty list."""
        fts.index_knowledge_chunk("k1", "grafana alerts", "wiki", 0)
        results = fts.search_knowledge("terraform")
        assert results == []

    async def test_search_respects_limit(self, fts):
        """Limit parameter caps knowledge results."""
        for i in range(10):
            fts.index_knowledge_chunk(f"k{i}", f"ansible playbook topic {i}", "docs", i)
        results = fts.search_knowledge("ansible", limit=3)
        assert len(results) == 3

    async def test_search_when_unavailable(self, broken_fts):
        """Searching when unavailable returns empty list."""
        assert broken_fts.search_knowledge("test") == []


# ── Knowledge deletion ──────────────────────────────────────────────


class TestKnowledgeDeletion:
    async def test_delete_by_source(self, fts):
        """delete_knowledge_source removes all chunks from that source."""
        fts.index_knowledge_chunk("k1", "doc one", "wiki", 0)
        fts.index_knowledge_chunk("k2", "doc two", "wiki", 1)
        fts.index_knowledge_chunk("k3", "doc three", "blog", 0)
        deleted = fts.delete_knowledge_source("wiki")
        assert deleted == 2
        assert fts.has_knowledge_chunk("k1") is False
        assert fts.has_knowledge_chunk("k2") is False
        assert fts.has_knowledge_chunk("k3") is True

    async def test_delete_nonexistent_source(self, fts):
        """Deleting a non-existent source returns 0."""
        deleted = fts.delete_knowledge_source("nonexistent")
        assert deleted == 0

    async def test_delete_when_unavailable(self, broken_fts):
        """Deletion returns 0 when unavailable."""
        assert broken_fts.delete_knowledge_source("wiki") == 0


# ── _prepare_query ──────────────────────────────────────────────────


class TestPrepareQuery:
    async def test_plain_text_passthrough(self):
        """Plain text query passes through unchanged."""
        assert _prepare_query("hello world") == "hello world"

    async def test_empty_string(self):
        """Empty string returns empty string."""
        assert _prepare_query("") == ""

    async def test_whitespace_only(self):
        """Whitespace-only returns empty string."""
        assert _prepare_query("   ") == ""

    async def test_ip_address_quoted(self):
        """IP address (contains dots) gets quoted."""
        result = _prepare_query("192.168.1.13")
        assert result == '"192.168.1.13"'

    async def test_path_quoted(self):
        """Path (contains slashes) gets quoted."""
        result = _prepare_query("/opt/project/data")
        assert result == '"/opt/project/data"'

    async def test_special_chars_quoted(self):
        """FTS5 special characters trigger quoting."""
        assert _prepare_query("test*") == '"test*"'
        assert _prepare_query("field:value") == '"field:value"'
        assert _prepare_query("(grouped)") == '"(grouped)"'

    async def test_internal_quotes_escaped(self):
        """Internal double quotes are escaped when wrapping."""
        result = _prepare_query('say "hello"')
        assert result == '"say ""hello"""'

    async def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        assert _prepare_query("  hello  ") == "hello"

    async def test_promql_expression_quoted(self):
        """PromQL-like expression with special chars is quoted."""
        result = _prepare_query("rate(http_requests_total[5m])")
        assert result.startswith('"')
        assert result.endswith('"')
