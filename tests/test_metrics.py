import math

import pytest

from recclens.eval.metrics import match_f1, mrr, ndcg_at_k, recall_at_k


def test_match_f1_perfect_and_partial():
    truth = [{"a", "b"}, {"a", "b"}, {"c"}]
    assert match_f1(truth, truth) == 1.0
    # Row 1 predicts only itself: F1 = 2*1/(1+2) = 2/3.
    predicted = [{"a"}, {"a", "b"}, {"c"}]
    assert match_f1(predicted, truth) == pytest.approx((2 / 3 + 1 + 1) / 3)


def test_recall_at_k_caps_denominator_at_k():
    ranked = [[1, 2, 3]]
    relevant = [{1, 2, 3, 4, 5}]
    assert recall_at_k(ranked, relevant, k=2) == 1.0
    assert recall_at_k(ranked, relevant, k=3) == 1.0
    assert recall_at_k([[9, 1, 8]], [{1, 2}], k=3) == 0.5


def test_ndcg_at_k():
    assert ndcg_at_k([[1, 2]], [{1}], k=2) == 1.0
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k([[2, 1]], [{1}], k=2) == pytest.approx(expected)


def test_mrr_skips_queries_without_relevant_items():
    assert mrr([[3, 1], [5, 6]], [{1}, set()]) == 0.5
    assert mrr([[3, 4]], [{1}]) == 0.0
