# ReccLens build plan

ReccLens is a multimodal, multilingual product-matching and search engine for
Southeast Asian e-commerce. It takes a product photo, a text query in any SEA
language, or both, returns the top 10 matching listings, and groups listings
that are the same product. It is trained on Shopee's public Price Match
(Kaggle 2021) data and served at production latency with Triton and TensorRT.

Build window: 19 Oct to 29 Nov 2026. Each week ends with something that runs
and is measured. If a week slips, cut its stretch goal, not the next week's start.

## Architecture

```
offline                                      online (one GPU pass per query)
-------                                      -------------------------------
Price Match data ──► group split (70/15/15)   query (image and/or text)
LLM queries (5 SEA languages)                   │
        │                                       ▼
        ▼                                  Triton ensemble
SigLIP 2 (LoRA) + multilingual-e5 (LoRA)     DALI decode ► encoders ► MoE fusion
ArcFace + InfoNCE + hard negatives              │
MoE fusion head                                 ▼
        │                                  FAISS IVF-PQ  +  BM25
        ▼                                       │ reciprocal rank fusion
ONNX ► TensorRT FP16 / INT8                     ▼
FAISS + BM25 indexes                       XLM-R cross-encoder rerank (top 50)
VLM-labelled pairs ► distilled reranker         │
                                                ▼
                                           top 10 + same-product groups
```

## Repository layout

| Path | Purpose | Week |
|---|---|---|
| `configs/` | Hydra configs (data, model, train, eval) | 1 |
| `recclens/data/download.py` | Fetch Kaggle data (data never committed) | 1 |
| `recclens/data/split.py` | Split by `label_group`, 70/15/15 | 1 |
| `recclens/data/gen_queries.py` | LLM shopper queries in en/id/ms/th/vi | 1 |
| `recclens/eval/metrics.py` | Match F1, Recall@K, NDCG@K, MRR | 1 |
| `recclens/evaluate.py` | One eval entry point for any model's embeddings | 1 |
| `recclens/baselines/` | TF-IDF, zero-shot SigLIP, SigLIP + e5 concat | 2 |
| `recclens/models/` | Encoders with LoRA, ArcFace, InfoNCE, MoE head | 2–3 |
| `recclens/train.py` | Training loop, W&B logging | 2–3 |
| `recclens/retrieval/` | FAISS IVF-PQ / HNSW, BM25, RRF, grouping | 3–4 |
| `recclens/rerank/` | VLM labelling, cross-encoder distillation | 4 |
| `export/` | ONNX export, TensorRT build, INT8 calibration | 5 |
| `serving/triton/` | Model repository, ensemble, DALI, BLS | 5 |
| `serving/gateway/` | FastAPI, Redis cache, Prometheus metrics | 5–6 |
| `bench/` | Latency benchmarks, Locust, nprobe sweep | 5–6 |
| `demo/` | Gradio app for Hugging Face Spaces | 6 |
| `docker-compose.yml` | Triton + gateway + Redis + Prometheus + Grafana | 6 |
| `.github/workflows/ci.yml` | Lint, tests, eval on a tiny sample | 1 |

## Weekly milestones

### Week 1 (19–25 Oct): data and evaluation harness
- [ ] Download script; read competition rules on redistribution.
- [ ] Group split 70/15/15 so test products never appear in training.
- [ ] Generate 3–5 queries per listing in en, id, ms, th, vi; hand-check 200.
- [ ] `evaluate.py` reports F1, Recall@10, NDCG@10, MRR for any model.
- [ ] Repo skeleton, Hydra configs, W&B logging, CI running eval on a sample.

### Week 2 (26 Oct–1 Nov): baselines, then first model
- [ ] Baselines: TF-IDF titles, zero-shot SigLIP images, SigLIP + e5 concat.
- [ ] Image and text towers with ArcFace on label groups, LoRA on both.
- [ ] Baseline table in the README.

### Week 3 (2–8 Nov): fusion and hard negatives
- [ ] MoE fusion head (2–4 experts, learned gate) and InfoNCE image-text loss.
- [ ] Hard negatives mined from the model's own neighbours each epoch.
- [ ] FAISS IVF-PQ and BM25 indexes, reciprocal rank fusion.
- [ ] Ablation table.

### Week 4 (9–15 Nov): model freeze and reranker
- [ ] Freeze the embedding model on 11 Nov.
- [ ] VLM scores ~20k query/candidate pairs; distil into XLM-R cross-encoder.
- [ ] Same-product grouping: threshold + connected components.
- [ ] Offline results with and without the reranker.

### Week 5 (16–22 Nov): TensorRT and Triton
- [ ] ONNX export; TensorRT FP16 and INT8 (calibrated on 1,000 train images).
- [ ] INT8 keeps Recall@10 within 1 point of FP32, else FP16 sensitive layers.
- [ ] Triton ensemble: DALI ► encoders ► fusion, dynamic batching; FAISS and reranker via BLS.
- [ ] Latency table: PyTorch, ONNX Runtime, TRT FP16, TRT INT8.

### Week 6 (23–29 Nov): load test, demo and write-up
- [ ] Locust at rising QPS: p50/p99 vs throughput; nprobe recall/latency curve.
- [ ] Prometheus + Grafana dashboard, Gradio demo on HF Spaces, 2-minute video.
- [ ] Public repo, live demo and blog post by 29 Nov.

## Metrics (measured on held-out label groups only)

| Metric | Target |
|---|---|
| Match F1 (Kaggle metric) | Clearly above the best zero-shot baseline |
| Recall@10, text to product | Reported per language |
| NDCG@10 after reranking | Above retrieval alone, in every language |
| Per-component gain | One ablation row each |
| INT8 accuracy loss | Recall@10 within 1 point of FP32 |
| Encoder latency per image | Batch 1 and batch 32 |
| End-to-end p99 latency | Under 100 ms at max sustained QPS on one GPU |
| Recall vs latency | Curve over FAISS nprobe with operating point marked |

Rules: split by label group, never by row; mean and spread over 3 seeds for the
headline model; latency after warm-up at fixed batch size on a named GPU; keep a
failure-analysis page with 20 wrong matches.
