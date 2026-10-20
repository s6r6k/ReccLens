from pathlib import Path

import numpy as np
import pandas as pd

from recclens.baselines.tfidf import embed_titles
from recclens.evaluate import evaluate_items, evaluate_queries

FIXTURES = Path(__file__).parent / "fixtures"


def one_hot_by_group(items: pd.DataFrame) -> np.ndarray:
    codes = pd.factorize(items["label_group"])[0]
    return np.eye(codes.max() + 1, dtype=np.float32)[codes]


def test_oracle_embeddings_score_perfectly():
    items = pd.read_csv(FIXTURES / "sample_items.csv")
    results = evaluate_items(items, one_hot_by_group(items), k=10, threshold=0.9, max_matches=50)
    assert results["match_f1"] == 1.0
    assert results["item_recall@10"] == 1.0
    assert results["item_mrr"] == 1.0


def test_tfidf_baseline_end_to_end():
    items = pd.read_csv(FIXTURES / "sample_items.csv")
    queries = pd.read_csv(FIXTURES / "sample_queries.csv")
    item_emb, query_emb = embed_titles(items["title"].tolist(), queries["query"].tolist(), dim=16)

    item_results = evaluate_items(items, item_emb, k=10, threshold=0.5, max_matches=50)
    query_results = evaluate_queries(items, item_emb, queries, query_emb, k=10)

    assert item_results["match_f1"] > 0.5
    assert query_results["query_recall@10"] > 0.3
    for lang in queries["language"].unique():
        assert f"query_recall@10/{lang}" in query_results
