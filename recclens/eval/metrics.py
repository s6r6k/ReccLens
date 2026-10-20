"""Matching and retrieval metrics.

Matching follows the Kaggle Price Match metric: for each listing, F1 between the
predicted set of matching posting_ids and the true set (its whole label group,
itself included), averaged over listings.

Retrieval metrics take, per query, a ranked list of item ids and the set of
relevant item ids.
"""

from collections.abc import Sequence

import numpy as np


def match_f1(predicted: Sequence[set], truth: Sequence[set]) -> float:
    scores = []
    for pred, true in zip(predicted, truth, strict=True):
        overlap = len(pred & true)
        scores.append(2 * overlap / (len(pred) + len(true)) if pred or true else 1.0)
    return float(np.mean(scores))


def recall_at_k(ranked: Sequence[Sequence], relevant: Sequence[set], k: int) -> float:
    """Fraction of each query's relevant items found in its top k, averaged."""
    scores = [
        len(set(r[:k]) & rel) / min(len(rel), k)
        for r, rel in zip(ranked, relevant, strict=True)
        if rel
    ]
    return float(np.mean(scores))


def ndcg_at_k(ranked: Sequence[Sequence], relevant: Sequence[set], k: int) -> float:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for r, rel in zip(ranked, relevant, strict=True):
        if not rel:
            continue
        gains = np.array([1.0 if item in rel else 0.0 for item in r[:k]])
        dcg = float((gains * discounts[: len(gains)]).sum())
        idcg = float(discounts[: min(len(rel), k)].sum())
        scores.append(dcg / idcg)
    return float(np.mean(scores))


def mrr(ranked: Sequence[Sequence], relevant: Sequence[set]) -> float:
    scores = []
    for r, rel in zip(ranked, relevant, strict=True):
        if not rel:
            continue
        rank = next((i for i, item in enumerate(r, start=1) if item in rel), None)
        scores.append(1.0 / rank if rank else 0.0)
    return float(np.mean(scores))
