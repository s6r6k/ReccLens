"""ReccLensModel: the towers plus fusion, behind one encode_items / encode_queries API.

mode="image"  items from the image tower; no text queries.
mode="text"   items and queries from the text tower.
mode="concat" zero-shot: items are [image; text] and queries are [0; text], so
              text queries are scored against the title half. With a trained
              head: both go through ConcatHead (queries with a zero image).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from recclens.models.encoders import ImageEncoder, TextEncoder
from recclens.models.heads import ConcatHead


class ReccLensModel(nn.Module):
    def __init__(
        self,
        mode: str,
        image_encoder: ImageEncoder | None,
        text_encoder: TextEncoder | None,
        embed_dim: int | None = None,
    ):
        super().__init__()
        if mode not in ("image", "text", "concat"):
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.head = None
        if mode == "concat" and embed_dim:
            self.head = ConcatHead(image_encoder.dim, text_encoder.dim, embed_dim)

    @property
    def has_queries(self) -> bool:
        return self.text_encoder is not None

    def _fuse(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if self.head is not None:
            return self.head(image_emb, text_emb)
        return F.normalize(torch.cat([image_emb, text_emb], dim=-1), dim=-1)

    def encode_items(self, batch: dict) -> torch.Tensor:
        if self.mode == "image":
            return self.image_encoder(batch["pixel_values"])
        if self.mode == "text":
            return self.text_encoder(**batch["text"])
        image_emb = self.image_encoder(batch["pixel_values"])
        return self._fuse(image_emb, self.text_encoder(**batch["text"]))

    def encode_queries(self, text: dict) -> torch.Tensor:
        text_emb = self.text_encoder(**text)
        if self.mode == "text":
            return text_emb
        image_emb = text_emb.new_zeros(text_emb.shape[0], self.image_encoder.dim)
        return self._fuse(image_emb, text_emb)

    def trainable_state_dict(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names or k.startswith("head.")}

    def save(self, path: Path, cfg: DictConfig) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.trainable_state_dict(), "model_cfg": OmegaConf.to_container(cfg)},
            path,
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        missing_trained = set(ckpt["state_dict"]) - set(self.state_dict())
        if missing_trained:
            raise KeyError(f"checkpoint keys not in model: {sorted(missing_trained)[:5]}")
        self.load_state_dict(ckpt["state_dict"], strict=False)


def build_model(cfg: DictConfig, train: bool):
    """Returns (model, image_processor, tokenizer). Downloads backbones on first use."""
    from recclens.models.encoders import build_image_encoder, build_text_encoder

    trained = train or cfg.checkpoint is not None
    image_encoder = processor = text_encoder = tokenizer = None
    if cfg.mode in ("image", "concat"):
        image_encoder, processor = build_image_encoder(cfg.image, trained)
    if cfg.mode in ("text", "concat"):
        text_encoder, tokenizer = build_text_encoder(cfg.text, trained)
    model = ReccLensModel(
        cfg.mode, image_encoder, text_encoder, cfg.embed_dim if trained else None
    )
    if cfg.checkpoint is not None:
        model.load(Path(cfg.checkpoint))
    return model, processor, tokenizer
