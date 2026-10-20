"""Evaluate any model from its embeddings.

Item embeddings are row-aligned with the split CSV. Query embeddings, if given,
are row-aligned with a queries CSV (columns: posting_id, language, query).

    python -m recclens.evaluate eval.item_embeddings=outputs/emb/test_items.npy
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from recclens.eval.metrics import match_f1, mrr, ndcg_at_k, recall_at_k


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def evaluate_items(
    items: pd.DataFrame, emb: np.ndarray, k: int, threshold: float, max_matches: int
) -> dict:
    emb = l2_normalize(emb.astype(np.float32))
    sims = emb @ emb.T
    groups = items["label_group"].to_numpy()
    ids = items["posting_id"].to_numpy()

    by_group: dict = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, set()).add(i)

    neighbours = top_k(sims, max_matches + 1)
    predicted = [
        {ids[j] for j in row if sims[i, j] >= threshold} | {ids[i]}
        for i, row in enumerate(neighbours)
    ]
    truth = [{ids[j] for j in by_group[g]} for g in groups]

    # Item-to-item retrieval: rank everything except the query listing itself.
    np.fill_diagonal(sims, -np.inf)
    ranked = top_k(sims, k).tolist()
    relevant = [by_group[g] - {i} for i, g in enumerate(groups)]

    return {
        "match_f1": match_f1(predicted, truth),
        f"item_recall@{k}": recall_at_k(ranked, relevant, k),
        f"item_ndcg@{k}": ndcg_at_k(ranked, relevant, k),
        "item_mrr": mrr(ranked, relevant),
    }


def evaluate_queries(
    items: pd.DataFrame, item_emb: np.ndarray, queries: pd.DataFrame, query_emb: np.ndarray, k: int
) -> dict:
    item_emb = l2_normalize(item_emb.astype(np.float32))
    query_emb = l2_normalize(query_emb.astype(np.float32))
    ranked_all = top_k(query_emb @ item_emb.T, k)

    group_of = dict(zip(items["posting_id"], items["label_group"]))
    rows_of_group: dict = {}
    for i, g in enumerate(items["label_group"]):
        rows_of_group.setdefault(g, set()).add(i)
    relevant_all = [rows_of_group[group_of[pid]] for pid in queries["posting_id"]]

    results = {}
    for lang, idx in queries.groupby("language").indices.items():
        ranked = ranked_all[idx].tolist()
        relevant = [relevant_all[i] for i in idx]
        results[f"query_recall@{k}/{lang}"] = recall_at_k(ranked, relevant, k)
        results[f"query_ndcg@{k}/{lang}"] = ndcg_at_k(ranked, relevant, k)
        results[f"query_mrr/{lang}"] = mrr(ranked, relevant)
    ranked = ranked_all.tolist()
    results[f"query_recall@{k}"] = recall_at_k(ranked, relevant_all, k)
    results[f"query_ndcg@{k}"] = ndcg_at_k(ranked, relevant_all, k)
    results["query_mrr"] = mrr(ranked, relevant_all)
    return results


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    ev = cfg.eval
    if ev.item_embeddings is None:
        raise ValueError("set eval.item_embeddings=<path to .npy>")

    items = pd.read_csv(root / ev.items_csv)
    item_emb = np.load(root / ev.item_embeddings)
    if len(items) != len(item_emb):
        raise ValueError(f"{len(items)} items but {len(item_emb)} embeddings")

    results = evaluate_items(items, item_emb, ev.k, ev.match_threshold, ev.max_matches)
    if ev.queries_csv and ev.query_embeddings:
        queries = pd.read_csv(root / ev.queries_csv)
        queries = queries[queries["posting_id"].isin(set(items["posting_id"]))]
        query_emb = np.load(root / ev.query_embeddings)[queries.index.to_numpy()]
        results |= evaluate_queries(items, item_emb, queries.reset_index(drop=True), query_emb, ev.k)

    print(json.dumps(results, indent=2))
    Path("metrics.json").write_text(json.dumps(results, indent=2))

    if cfg.wandb.enabled:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            job_type="eval",
        )
        run.summary.update(results)
        run.finish()


if __name__ == "__main__":
    main()
