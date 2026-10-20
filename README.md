# ReccLens

Multimodal, multilingual product matching and search for Southeast Asian e-commerce.
Give it a product photo, a text query in English, Indonesian, Malay, Thai or Vietnamese,
or both, and it returns the top 10 matching listings and groups listings that are the
same product. Trained on Shopee's public Price Match data; served with Triton and TensorRT.

> Status: week 2 of 6 (baselines and first trained model). See [PLAN.md](PLAN.md) for the full build plan.

## Results

Measured on the 5,202 test listings (1,652 held-out label groups; no test product is seen
in training). Match F1 uses the threshold that maximised F1 on val. Item Recall@10 and
NDCG@10 rank the other test listings for each listing. Text-to-product numbers need the
generated queries.

| Model | Match F1 | Item Recall@10 | Item NDCG@10 | Recall@10 (text→product) |
|---|---|---|---|---|
| TF-IDF titles (char 2–4-grams + SVD 256) | 0.734 | 0.853 | 0.805 | – |
| Zero-shot SigLIP 2 (image) | 0.751 | 0.917 | 0.886 | – |
| Zero-shot SigLIP 2 + e5 (concat) | 0.799 | 0.946 | 0.918 | – |
| ReccLens week 2 (ArcFace + LoRA, concat head) | – | – | – | – |
| ReccLens (ArcFace + InfoNCE + MoE) | – | – | – | – |
| + hybrid retrieval + reranker | – | – | – | – |

Tuned thresholds: TF-IDF 0.70, SigLIP 0.80, SigLIP + e5 0.85.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data,queries]"

# 1. Data (needs a Kaggle account with the competition rules accepted)
python -m recclens.data.download
python -m recclens.data.split

# 2. Multilingual queries (needs ANTHROPIC_API_KEY)
python -m recclens.data.gen_queries submit --limit 200   # sample to hand-check first
python -m recclens.data.gen_queries collect

# 3. Baselines + evaluation
python -m recclens.baselines.tfidf --items data/processed/test.csv \
    --queries data/processed/queries.csv --out outputs/tfidf/items.npy
python -m recclens.evaluate eval.item_embeddings=outputs/tfidf/items.npy \
    eval.queries_csv=data/processed/queries.csv eval.query_embeddings=outputs/tfidf/queries.npy
python -m recclens.baselines.zero_shot --queries data/processed/queries.csv

# 4. Train (ArcFace on label groups, LoRA on SigLIP 2 and e5), then embed and score test
pip install -e ".[train]"
python -m recclens.train
python -m recclens.embed model.checkpoint=checkpoints/siglip_e5/best.pt \
    eval.queries_csv=data/processed/queries.csv hydra.run.dir=outputs/arcface
python -m recclens.evaluate eval.item_embeddings=outputs/arcface/items.npy \
    eval.queries_csv=data/processed/queries.csv eval.query_embeddings=outputs/arcface/queries.npy

# 5. Pick the match threshold on val (embed with eval.split=val first), report on test
python scripts/tune_threshold.py --val-items data/processed/val.csv --val-emb <val items.npy> \
    --test-items data/processed/test.csv --test-emb outputs/arcface/items.npy
```

`model.mode` picks the towers: `image`, `text` or `concat`. Training runs on CUDA, Apple MPS or CPU.

Add `wandb.enabled=true` to any Hydra command to log to Weights & Biases.

## Evaluation

`recclens/evaluate.py` scores any model from its embeddings:

- **Match F1**: the Kaggle Price Match metric. Per listing, F1 between predicted matches
  (cosine similarity above `eval.match_threshold`) and its true label group.
- **Recall@10, NDCG@10, MRR**: item-to-item retrieval, and text-to-product retrieval
  reported per language so the weakest language is visible.

## Data

The Shopee Price Match competition data is not redistributed here. The download script
fetches it from Kaggle into `data/raw/`, which is git-ignored.

## Tests

```bash
pytest -q
ruff check .
```
