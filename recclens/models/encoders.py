"""Image and text towers. Each returns L2-normalised embeddings.

ImageEncoder wraps the SigLIP 2 vision tower (pooled output). TextEncoder wraps
multilingual-e5 with mean pooling; e5 expects "query: " on search queries and
"passage: " on listing titles. Either can be wrapped with LoRA so only the
adapter weights train.
"""

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel


def with_lora(model: nn.Module, lora_cfg) -> nn.Module:
    if lora_cfg is None or not lora_cfg.enabled:
        return model
    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
    )
    return get_peft_model(model, config)


class ImageEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, lora_cfg=None):
        super().__init__()
        self.backbone = with_lora(backbone, lora_cfg)
        self.dim = backbone.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone(pixel_values=pixel_values).pooler_output
        return F.normalize(pooled, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, lora_cfg=None):
        super().__init__()
        self.backbone = with_lora(backbone, lora_cfg)
        self.dim = backbone.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return F.normalize(pooled, dim=-1)


def build_image_encoder(cfg, train: bool) -> tuple[ImageEncoder, SiglipImageProcessor]:
    backbone = SiglipVisionModel.from_pretrained(cfg.backbone)
    processor = SiglipImageProcessor.from_pretrained(cfg.backbone)
    return ImageEncoder(backbone, cfg.lora if train else None), processor


def build_text_encoder(cfg, train: bool) -> tuple[TextEncoder, AutoTokenizer]:
    backbone = AutoModel.from_pretrained(cfg.backbone, add_pooling_layer=False)
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    return TextEncoder(backbone, cfg.lora if train else None), tokenizer
