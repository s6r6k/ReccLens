"""Write item (and query) embeddings for a split, for recclens.evaluate.

Zero-shot when model.checkpoint is null, otherwise loads the trained adapters.

    python -m recclens.embed model.mode=image
    python -m recclens.embed model.checkpoint=checkpoints/siglip_e5/best.pt \
        eval.queries_csv=data/processed/queries.csv
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from recclens.data.dataset import Collator, ShopeeDataset
from recclens.models.model import ReccLensModel, build_model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


@torch.no_grad()
def embed_items(
    model: ReccLensModel, items: pd.DataFrame, image_dir: Path | None, collator: Collator,
    device: torch.device, batch_size: int = 128, num_workers: int = 4,
) -> np.ndarray:
    dataset = ShopeeDataset(items, image_dir, load_images=model.mode != "text")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator
    )
    model.eval()
    chunks = [
        model.encode_items(to_device(batch, device)).float().cpu()
        for batch in tqdm(loader, desc="items")
    ]
    return torch.cat(chunks).numpy()


@torch.no_grad()
def embed_queries(
    model: ReccLensModel, queries: list[str], tokenizer, device: torch.device,
    max_length: int = 64, batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    chunks = []
    for start in tqdm(range(0, len(queries), batch_size), desc="queries"):
        text = tokenizer(
            ["query: " + q for q in queries[start : start + batch_size]],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        ).to(device)
        chunks.append(model.encode_queries(text).float().cpu())
    return torch.cat(chunks).numpy()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    device = pick_device()

    model, processor, tokenizer = build_model(cfg.model, train=False)
    model.to(device)
    collator = Collator(processor, tokenizer, cfg.model.text.max_length)

    items = pd.read_csv(root / cfg.eval.items_csv)
    item_emb = embed_items(
        model, items, root / cfg.data.image_dir, collator, device,
        num_workers=cfg.train.num_workers,
    )
    np.save(out_dir / "items.npy", item_emb)
    print(f"Saved items {item_emb.shape} to {out_dir / 'items.npy'}")

    if cfg.eval.queries_csv and model.has_queries:
        queries = pd.read_csv(root / cfg.eval.queries_csv)
        query_emb = embed_queries(
            model, queries["query"].astype(str).tolist(), tokenizer, device,
            cfg.model.text.max_length,
        )
        np.save(out_dir / "queries.npy", query_emb)
        print(f"Saved queries {query_emb.shape} to {out_dir / 'queries.npy'}")


if __name__ == "__main__":
    main()
