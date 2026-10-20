"""Train the towers with ArcFace on train label groups (LoRA on both encoders).

Each epoch embeds the val split and scores Match F1 with recclens.evaluate; the
best epoch's adapters and head are saved to train.checkpoint_dir/best.pt.

    python -m recclens.train
    python -m recclens.train model.mode=image wandb.enabled=true
"""

import math
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_metric_learning.losses import ArcFaceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from recclens.data.dataset import Collator, ShopeeDataset, group_sampler
from recclens.embed import embed_items, pick_device, to_device
from recclens.evaluate import evaluate_items
from recclens.models.model import ReccLensModel, build_model


def output_dim(model: ReccLensModel) -> int:
    if model.head is not None:
        return model.head.proj[0].out_features
    if model.mode == "image":
        return model.image_encoder.dim
    if model.mode == "text":
        return model.text_encoder.dim
    return model.image_encoder.dim + model.text_encoder.dim


def build_optimizer(model: ReccLensModel, loss_fn: ArcFaceLoss, cfg) -> torch.optim.Optimizer:
    lora = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    groups = [
        {"params": lora, "lr": cfg.lr_lora},
        {"params": head, "lr": cfg.lr_head},
        {"params": loss_fn.parameters(), "lr": cfg.lr_loss, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=cfg.weight_decay)


def train_one_epoch(
    model: ReccLensModel,
    loss_fn: ArcFaceLoss,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    amp: bool = False,
    grad_clip: float | None = None,
) -> float:
    model.train()
    use_amp = amp and device.type == "cuda"
    total, steps = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            emb = model.encode_items(batch)
        loss = loss_fn(emb.float(), batch["label"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], grad_clip
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    torch.manual_seed(cfg.seed)
    device = pick_device()
    tc = cfg.train

    model, processor, tokenizer = build_model(cfg.model, train=True)
    model.to(device)
    collator = Collator(processor, tokenizer, cfg.model.text.max_length)
    image_dir = root / cfg.data.image_dir

    train_df = pd.read_csv(root / cfg.data.processed_dir / "train.csv")
    val_df = pd.read_csv(root / cfg.data.processed_dir / "val.csv")
    train_set = ShopeeDataset(train_df, image_dir, load_images=model.mode != "text")
    loader = DataLoader(
        train_set,
        batch_size=tc.batch_size,
        sampler=group_sampler(train_set, tc.m_per_class, tc.batch_size),
        num_workers=tc.num_workers,
        collate_fn=collator,
        drop_last=True,
    )

    loss_fn = ArcFaceLoss(
        num_classes=train_set.num_classes,
        embedding_size=output_dim(model),
        margin=tc.arcface.margin,
        scale=tc.arcface.scale,
    ).to(device)
    optimizer = build_optimizer(model, loss_fn, tc)
    total_steps = tc.epochs * len(loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, math.ceil(tc.warmup_ratio * total_steps), total_steps
    )

    run = None
    if cfg.wandb.enabled:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            job_type="train",
        )

    ckpt_dir = root / tc.checkpoint_dir
    best_f1 = -1.0
    for epoch in range(tc.epochs):
        train_loss = train_one_epoch(
            model, loss_fn, loader, optimizer, scheduler, device, tc.amp, tc.grad_clip
        )
        val_emb = embed_items(
            model, val_df, image_dir, collator, device, num_workers=tc.num_workers
        )
        metrics = evaluate_items(
            val_df, val_emb, cfg.eval.k, cfg.eval.match_threshold, cfg.eval.max_matches
        )
        print(f"epoch {epoch}: loss {train_loss:.4f}, val {metrics}")
        if run is not None:
            run.log({"epoch": epoch, "train/loss": train_loss}
                    | {f"val/{k}": v for k, v in metrics.items()})
        if metrics["match_f1"] > best_f1:
            best_f1 = metrics["match_f1"]
            model.save(ckpt_dir / "best.pt", cfg.model)
            print(f"saved best (val match_f1 {best_f1:.4f}) to {ckpt_dir / 'best.pt'}")

    if run is not None:
        run.summary["best_val_match_f1"] = best_f1
        run.finish()


if __name__ == "__main__":
    main()
