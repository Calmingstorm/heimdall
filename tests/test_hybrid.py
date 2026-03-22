"""Tests for Reciprocal Rank Fusion (hybrid.py).

Pure logic — no external services to mock. Tests verify:
- Basic RRF scoring formula
- Multiple list merging with deduplication
- Limit enforcement
- Edge cases: empty lists, missing id_key, custom k
"""
from __future__ import annotations

import pytest

from src.search.hybrid import reciprocal_rank_fusion


class TestRRFBasicScoring:
    """Verify the core RRF formula: score = sum(1 / (k + rank_1based))."""

    async def test_single_list_single_item(self):
        """One item at rank 1 → score = 1/(60+1) = 1/61."""
        results = reciprocal_rank_fusion(
            [{"doc_id": "a", "text": "hello"}],
        )
        assert len(results) == 1
        assert results[0]["doc_id"] == "a"
        expected = round(1.0 / 61, 6)
        assert results[0]["rrf_score"] == expected

    async def test_single_list_preserves_rank_order(self):
        """Items from a single list keep their original order."""
        items = [{"doc_id": f"d{i}", "val": i} for i in range(5)]
        results = reciprocal_rank_fusion(items)
        assert [r["doc_id"] for r in results] == [f"d{i}" for i in range(5)]

    async def test_single_list_scores_decrease(self):
        """Higher-ranked items get higher RRF scores."""
        items = [{"doc_id": f"d{i}"} for i in range(5)]
        results = reciprocal_rank_fusion(items)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        # All scores should be distinct
        assert len(set(scores)) == 5

    async def test_two_lists_same_item_scores_accumulate(self):
        """An item appearing in both lists gets scores from both."""
        list_a = [{"doc_id": "x", "source": "a"}]
        list_b = [{"doc_id": "x", "source": "b"}]
        results = reciprocal_rank_fusion(list_a, list_b)
        assert len(results) == 1
        # 1/(60+1) + 1/(60+1) = 2/61
        expected = round(2.0 / 61, 6)
        assert results[0]["rrf_score"] == expected

    async def test_two_lists_shared_item_keeps_first_occurrence_data(self):
        """Dedup keeps the dict from the highest-ranked (first) occurrence."""
        list_a = [{"doc_id": "x", "source": "a"}]
        list_b = [{"doc_id": "x", "source": "b"}]
        results = reciprocal_rank_fusion(list_a, list_b)
        assert results[0]["source"] == "a"

    async def test_score_formula_with_default_k(self):
        """Verify exact formula: rank_0=0 → 1/(60+0+1), rank_0=1 → 1/(60+1+1)."""
        items = [{"doc_id": "a"}, {"doc_id": "b"}]
        results = reciprocal_rank_fusion(items)
        assert results[0]["rrf_score"] == round(1.0 / 61, 6)
        assert results[1]["rrf_score"] == round(1.0 / 62, 6)


class TestRRFMerging:
    """Test merging across multiple result lists."""

    async def test_disjoint_lists_merge(self):
        """Items from different lists are all included."""
        list_a = [{"doc_id": "a1"}, {"doc_id": "a2"}]
        list_b = [{"doc_id": "b1"}, {"doc_id": "b2"}]
        results = reciprocal_rank_fusion(list_a, list_b)
        ids = {r["doc_id"] for r in results}
        assert ids == {"a1", "a2", "b1", "b2"}

    async def test_disjoint_lists_top_items_tie(self):
        """First items from two disjoint lists have equal RRF scores."""
        list_a = [{"doc_id": "a1"}]
        list_b = [{"doc_id": "b1"}]
        results = reciprocal_rank_fusion(list_a, list_b)
        assert results[0]["rrf_score"] == results[1]["rrf_score"]

    async def test_item_in_both_lists_ranks_higher(self):
        """An item in both lists outranks items in only one list."""
        list_a = [{"doc_id": "shared"}, {"doc_id": "a_only"}]
        list_b = [{"doc_id": "shared"}, {"doc_id": "b_only"}]
        results = reciprocal_rank_fusion(list_a, list_b)
        assert results[0]["doc_id"] == "shared"
        assert results[0]["rrf_score"] > results[1]["rrf_score"]

    async def test_three_lists(self):
        """Three lists merge correctly with accumulation."""
        list_a = [{"doc_id": "x"}]
        list_b = [{"doc_id": "x"}]
        list_c = [{"doc_id": "x"}]
        results = reciprocal_rank_fusion(list_a, list_b, list_c)
        assert len(results) == 1
        expected = round(3.0 / 61, 6)
        assert results[0]["rrf_score"] == expected

    async def test_different_ranks_across_lists(self):
        """Item at rank 0 in list A, rank 2 in list B → correct sum."""
        list_a = [{"doc_id": "x"}]
        list_b = [{"doc_id": "y"}, {"doc_id": "z"}, {"doc_id": "x"}]
        results = reciprocal_rank_fusion(list_a, list_b)
        # x: 1/61 (list_a rank 0) + 1/63 (list_b rank 2)
        x_score = round(1.0 / 61 + 1.0 / 63, 6)
        x_result = [r for r in results if r["doc_id"] == "x"][0]
        assert x_result["rrf_score"] == x_score


class TestRRFLimit:
    """Test the limit parameter."""

    async def test_limit_truncates(self):
        """Results are capped at limit."""
        items = [{"doc_id": f"d{i}"} for i in range(20)]
        results = reciprocal_rank_fusion(items, limit=5)
        assert len(results) == 5

    async def test_default_limit_is_10(self):
        """Default limit=10 caps results."""
        items = [{"doc_id": f"d{i}"} for i in range(15)]
        results = reciprocal_rank_fusion(items)
        assert len(results) == 10

    async def test_limit_larger_than_results(self):
        """Limit larger than available items returns all items."""
        items = [{"doc_id": "a"}, {"doc_id": "b"}]
        results = reciprocal_rank_fusion(items, limit=100)
        assert len(results) == 2


class TestRRFCustomK:
    """Test the k parameter changes scoring."""

    async def test_custom_k(self):
        """k=10 changes the formula: score = 1/(10+rank+1)."""
        items = [{"doc_id": "a"}]
        results = reciprocal_rank_fusion(items, k=10)
        expected = round(1.0 / 11, 6)
        assert results[0]["rrf_score"] == expected

    async def test_k_zero(self):
        """k=0 gives higher absolute scores: 1/(0+1) = 1.0 for rank 0."""
        items = [{"doc_id": "a"}]
        results = reciprocal_rank_fusion(items, k=0)
        assert results[0]["rrf_score"] == 1.0


class TestRRFEdgeCases:
    """Edge cases and unusual inputs."""

    async def test_empty_list(self):
        """No input lists → empty results."""
        results = reciprocal_rank_fusion()
        assert results == []

    async def test_single_empty_list(self):
        """One empty list → empty results."""
        results = reciprocal_rank_fusion([])
        assert results == []

    async def test_multiple_empty_lists(self):
        """Multiple empty lists → empty results."""
        results = reciprocal_rank_fusion([], [], [])
        assert results == []

    async def test_missing_id_key_uses_rank(self):
        """Items without the id_key fall back to rank-based dedup."""
        items = [{"text": "hello"}, {"text": "world"}]
        results = reciprocal_rank_fusion(items)
        assert len(results) == 2

    async def test_custom_id_key(self):
        """Custom id_key is respected for deduplication."""
        list_a = [{"chunk_id": "c1", "text": "a"}]
        list_b = [{"chunk_id": "c1", "text": "b"}]
        results = reciprocal_rank_fusion(list_a, list_b, id_key="chunk_id")
        assert len(results) == 1
        assert results[0]["rrf_score"] == round(2.0 / 61, 6)

    async def test_rrf_score_added_to_result_dicts(self):
        """Results have rrf_score field that wasn't in the original dict."""
        items = [{"doc_id": "a", "color": "red"}]
        results = reciprocal_rank_fusion(items)
        assert "rrf_score" in results[0]
        assert results[0]["color"] == "red"

    async def test_original_dicts_not_mutated(self):
        """Original input dicts are not modified (copies are used)."""
        original = {"doc_id": "a", "text": "hello"}
        reciprocal_rank_fusion([original])
        assert "rrf_score" not in original
