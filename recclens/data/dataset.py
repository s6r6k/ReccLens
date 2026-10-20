"""Torch dataset over a split CSV, plus the batch collator.

Images and titles are returned raw; the collator runs the image processor and
tokenizer once per batch, so the same dataset serves image-only, text-only and
concat models.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import Dataset


class ShopeeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path | None, load_images: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir) if image_dir else None
        self.load_images = load_images and self.image_dir is not None
        self.labels = torch.as_tensor(pd.factorize(self.df["label_group"])[0])
        self.num_classes = int(self.labels.max()) + 1 if len(self.labels) else 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        item = {"title": str(row["title"]), "label": self.labels[idx]}
        if self.load_images:
            item["image"] = Image.open(self.image_dir / row["image"]).convert("RGB")
        return item


class Collator:
    def __init__(self, image_processor=None, tokenizer=None, max_length: int = 64):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        out = {"label": torch.stack([b["label"] for b in batch])}
        if self.image_processor is not None and "image" in batch[0]:
            images = [b["image"] for b in batch]
            out["pixel_values"] = self.image_processor(
                images=images, return_tensors="pt"
            )["pixel_values"]
        if self.tokenizer is not None:
            out["text"] = self.tokenizer(
                ["passage: " + b["title"] for b in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        return out


def group_sampler(dataset: ShopeeDataset, m_per_class: int, batch_size: int) -> MPerClassSampler:
    """Batches hold m listings from each of batch_size / m label groups, so every
    batch has positives for the metric loss."""
    return MPerClassSampler(
        dataset.labels, m=m_per_class, batch_size=batch_size, length_before_new_iter=len(dataset)
    )
