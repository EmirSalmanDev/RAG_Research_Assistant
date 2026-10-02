"""Retrieval metrics against values worked out by hand."""

import math

import pytest

from eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_counts_reference_chunks_in_the_top_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, {"b", "e"}, k=3) == 0.5  # b found, e never retrieved
    assert recall_at_k(retrieved, {"b", "e"}, k=1) == 0.0
    assert recall_at_k(retrieved, {"a", "d"}, k=4) == 1.0
    assert recall_at_k([], {"a"}, k=8) == 0.0


def test_recall_ignores_order_inside_the_top_k() -> None:
    assert recall_at_k(["x", "a"], {"a"}, k=2) == recall_at_k(["a", "x"], {"a"}, k=2) == 1.0


def test_a_repeated_chunk_counts_once() -> None:
    assert recall_at_k(["a", "a", "b"], {"b"}, k=2) == 1.0  # [a, b] after de-duplication
    assert reciprocal_rank(["a", "a", "b"], {"b"}) == 0.5


def test_reciprocal_rank_is_one_over_the_first_relevant_rank() -> None:
    assert reciprocal_rank(["b", "x"], {"b"}) == 1.0
    assert reciprocal_rank(["x", "y", "b", "e"], {"b", "e"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], {"b"}) == 0.0


def test_ndcg_by_hand() -> None:
    # Relevant b at rank 2 and e at rank 4: DCG = 1/log2(3) + 1/log2(5).
    # Ideal: both at ranks 1 and 2: IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3).
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "b", "y", "e"], {"b", "e"}, k=10) == pytest.approx(expected)
    assert expected == pytest.approx(0.6509, abs=1e-4)


def test_ndcg_is_one_for_a_perfect_ranking_and_zero_for_no_hit() -> None:
    assert ndcg_at_k(["b", "e", "x"], {"b", "e"}, k=10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"b"}, k=10) == 0.0


def test_ndcg_only_looks_at_the_top_k() -> None:
    assert ndcg_at_k(["x", "b"], {"b"}, k=1) == 0.0
    # More reference chunks than k: the ideal list is capped at k, so k hits score 1.0.
    assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, k=2) == pytest.approx(1.0)


def test_metrics_need_reference_chunks() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=8)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), k=10)


def test_mean() -> None:
    assert mean([1.0, 0.5, 0.0]) == 0.5
    assert mean([]) is None
