"""TF-IDF baseline on listing titles.

Character n-grams cope with mixed Indonesian/English titles and model numbers
without a tokenizer. Writes dense embeddings for evaluate.py.

    python -m recclens.baselines.tfidf --items data/processed/test.csv \
        --out outputs/tfidf/test_items.npy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def embed_titles(
    titles: list[str], queries: list[str] | None = None, dim: int = 256
) -> tuple[np.ndarray, np.ndarray | None]:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
    item_tfidf = vectorizer.fit_transform(titles)
    dim = min(dim, item_tfidf.shape[1] - 1, item_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=dim, random_state=0)
    items = svd.fit_transform(item_tfidf).astype(np.float32)
    query_emb = None
    if queries is not None:
        query_emb = svd.transform(vectorizer.transform(queries)).astype(np.float32)
    return items, query_emb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--queries-out", type=Path, default=None)
    parser.add_argument("--dim", type=int, default=256)
    args = parser.parse_args()

    items = pd.read_csv(args.items)
    queries = pd.read_csv(args.queries)["query"].astype(str).tolist() if args.queries else None
    item_emb, query_emb = embed_titles(items["title"].astype(str).tolist(), queries, args.dim)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, item_emb)
    print(f"Saved {item_emb.shape} to {args.out}")
    if query_emb is not None:
        out = args.queries_out or args.out.with_name("queries.npy")
        np.save(out, query_emb)
        print(f"Saved {query_emb.shape} to {out}")


if __name__ == "__main__":
    main()
