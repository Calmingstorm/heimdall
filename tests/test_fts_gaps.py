"""Tests for fts.py coverage gaps.

Targets uncovered lines: 74-76, 103-105, 132-134, 161-163, 174-176.
These are all exception paths in FTS5 operations.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.search.fts import FullTextIndex


@pytest.fixture
def fts(tmp_path):
    return FullTextIndex(str(tmp_path / "test_fts.db"))


class TestSessionIndexException:
    def test_index_session_catches_exception(self, fts):
        """Exception during session indexing returns False (lines 74-76)."""
        # Replace _conn with a mock that raises on execute
        fts._conn = MagicMock()
        fts._conn.execute = MagicMock(side_effect=Exception("DB error"))
        result = fts.index_session("s1", "text", "c1", 1000.0)
        assert result is False


class TestSessionSearchException:
    def test_search_sessions_catches_exception(self, fts):
        """Exception during session search returns [] (lines 103-105)."""
        fts._conn = MagicMock()
        fts._conn.execute = MagicMock(side_effect=Exception("Query error"))
        result = fts.search_sessions("hello")
        assert result == []


class TestKnowledgeIndexException:
    def test_index_knowledge_catches_exception(self, fts):
        """Exception during knowledge indexing returns False (lines 132-134)."""
        fts._conn = MagicMock()
        fts._conn.execute = MagicMock(side_effect=Exception("DB error"))
        result = fts.index_knowledge_chunk("k1", "text", "src", 0)
        assert result is False


class TestKnowledgeSearchException:
    def test_search_knowledge_catches_exception(self, fts):
        """Exception during knowledge search returns [] (lines 161-163)."""
        fts._conn = MagicMock()
        fts._conn.execute = MagicMock(side_effect=Exception("Query error"))
        result = fts.search_knowledge("ansible")
        assert result == []


class TestKnowledgeDeleteException:
    def test_delete_knowledge_catches_exception(self, fts):
        """Exception during knowledge deletion returns 0 (lines 174-176)."""
        fts._conn = MagicMock()
        fts._conn.execute = MagicMock(side_effect=Exception("Delete error"))
        result = fts.delete_knowledge_source("wiki")
        assert result == 0
