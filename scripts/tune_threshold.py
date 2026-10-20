"""Pick the match threshold on val, then report Match F1 on test at that threshold.

Choosing it on test would leak; this sweeps val only.

    python scripts/tune_threshold.py \
        --val-items data/processed/val.csv --val-emb outputs/run/val_items.npy \
        --test-items data/processed/test.csv --test-emb outputs/run/test_items.npy
"""

import argparse
import json

import numpy as np
import pandas as pd

from recclens.evaluate import evaluate_items


def sweep(items: pd.DataFrame, emb: np.ndarray, thresholds: np.ndarray, max_matches: int) -> dict:
    return {
        float(t): evaluate_items(items, emb, k=10, threshold=float(t), max_matches=max_matches)[
            "match_f1"
        ]
        for t in thresholds
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-items", required=True)
    parser.add_argument("--val-emb", required=True)
    parser.add_argument("--test-items", required=True)
    parser.add_argument("--test-emb", required=True)
    parser.add_argument("--low", type=float, default=0.3)
    parser.add_argument("--high", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--max-matches", type=int, default=50)
    args = parser.parse_args()

    thresholds = np.arange(args.low, args.high + 1e-9, args.step)
    val_scores = sweep(
        pd.read_csv(args.val_items), np.load(args.val_emb), thresholds, args.max_matches
    )
    best = max(val_scores, key=val_scores.get)
    test_f1 = evaluate_items(
        pd.read_csv(args.test_items), np.load(args.test_emb), 10, best, args.max_matches
    )["match_f1"]

    for t, f1 in val_scores.items():
        print(f"{t:.3f}  val F1 {f1:.4f}{'  <- best' if t == best else ''}")
    summary = {"threshold": best, "val_match_f1": val_scores[best], "test_match_f1": test_f1}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
