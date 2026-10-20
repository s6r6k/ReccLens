"""Fusion of image and text embeddings into one item embedding.

ConcatHead is the Week 2 fusion: concatenate both towers and project. Week 3
replaces it with a mixture-of-experts gate (MoEHead) that weighs image against
text per item, so a listing with a weak photo leans on its title.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ConcatHead(nn.Module):
    def __init__(self, image_dim: int, text_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(image_dim + text_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(torch.cat([image_emb, text_emb], dim=-1)), dim=-1)


class MoEHead(nn.Module):
    """Week 3: write this yourself.

    Sketch: 2-4 expert MLPs over [image_emb; text_emb], and a gate (Linear +
    softmax) that outputs per-item expert weights. Output is the gate-weighted
    sum of expert outputs, L2-normalised, the same shape as ConcatHead.
    """

    def __init__(self, image_dim: int, text_dim: int, embed_dim: int, num_experts: int = 4):
        super().__init__()
        raise NotImplementedError("MoE fusion head: Week 3")
