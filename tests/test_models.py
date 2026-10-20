"""CPU smoke tests with tiny random-init backbones (no model downloads)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from omegaconf import OmegaConf  # noqa: E402
from transformers import (  # noqa: E402
    SiglipVisionConfig,
    SiglipVisionModel,
    XLMRobertaConfig,
    XLMRobertaModel,
)

from recclens.models.encoders import ImageEncoder, TextEncoder  # noqa: E402
from recclens.models.model import ReccLensModel  # noqa: E402

DIM = 32


def lora(targets):
    return OmegaConf.create(
        {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": targets}
    )


def tiny_model(mode="concat", embed_dim=16):
    torch.manual_seed(0)
    image = ImageEncoder(
        SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=DIM,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=32,
                patch_size=8,
            )
        ),
        lora(["q_proj", "v_proj"]),
    )
    text = TextEncoder(
        XLMRobertaModel(
            XLMRobertaConfig(
                vocab_size=100,
                hidden_size=DIM,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
            ),
            add_pooling_layer=False,
        ),
        lora(["query", "value"]),
    )
    return ReccLensModel(mode, image, text, embed_dim)


def tiny_batch(n=4):
    ids = torch.randint(5, 100, (n, 6))
    return {
        "pixel_values": torch.randn(n, 3, 32, 32),
        "text": {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
        "label": torch.tensor([0, 0, 1, 1])[:n],
    }


@pytest.mark.parametrize(("mode", "embed_dim", "out_dim"), [
    ("image", None, DIM),
    ("text", None, DIM),
    ("concat", None, 2 * DIM),
    ("concat", 16, 16),
])
def test_item_and_query_embeddings_are_normalised(mode, embed_dim, out_dim):
    model = tiny_model(mode, embed_dim).eval()
    batch = tiny_batch()
    with torch.no_grad():
        items = model.encode_items(batch)
        assert items.shape == (4, out_dim)
        assert torch.allclose(items.norm(dim=-1), torch.ones(4), atol=1e-5)
        if mode != "image":
            queries = model.encode_queries(batch["text"])
            assert queries.shape == (4, out_dim)


def test_only_lora_and_head_are_trainable():
    model = tiny_model()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in n or n.startswith("head.") for n in trainable)


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model().eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.1)
    model.save(tmp_path / "ckpt.pt", OmegaConf.create({"mode": "concat"}))

    fresh = tiny_model().eval()
    fresh.load(tmp_path / "ckpt.pt")
    batch = tiny_batch()
    with torch.no_grad():
        assert torch.allclose(model.encode_items(batch), fresh.encode_items(batch), atol=1e-6)


class FakeTokenizer:
    """Hash characters into the tiny vocab; enough to exercise the text path."""

    def __call__(self, texts, padding=True, truncation=True, max_length=16, return_tensors="pt"):
        from transformers import BatchEncoding

        rows = [[5 + (ord(c) % 90) for c in t[:max_length]] or [5] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [1] * (width - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        return BatchEncoding({"input_ids": ids, "attention_mask": mask})


def test_train_step_updates_lora_and_head():
    from pytorch_metric_learning.losses import ArcFaceLoss

    from recclens.train import build_optimizer, output_dim, train_one_epoch

    model = tiny_model()
    loss_fn = ArcFaceLoss(num_classes=2, embedding_size=output_dim(model))
    cfg = OmegaConf.create({"lr_lora": 1e-2, "lr_head": 1e-2, "lr_loss": 1e-2,
                            "weight_decay": 0.0})
    optimizer = build_optimizer(model, loss_fn, cfg)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    loss = train_one_epoch(model, loss_fn, [tiny_batch()] * 3, optimizer, None,
                           torch.device("cpu"), grad_clip=1.0)

    assert loss > 0
    changed = [n for n, p in model.named_parameters() if p.requires_grad
               and not torch.equal(p, before[n])]
    assert any(n.startswith("head.") for n in changed)
    assert any("lora_B" in n for n in changed)


def test_embed_text_mode_on_sample_fixture():
    from pathlib import Path

    import pandas as pd

    from recclens.data.dataset import Collator
    from recclens.embed import embed_items, embed_queries

    fixtures = Path(__file__).parent / "fixtures"
    items = pd.read_csv(fixtures / "sample_items.csv")
    queries = pd.read_csv(fixtures / "sample_queries.csv")
    model = tiny_model("text", None)
    tokenizer = FakeTokenizer()

    item_emb = embed_items(model, items, None, Collator(None, tokenizer), torch.device("cpu"),
                           batch_size=8, num_workers=0)
    query_emb = embed_queries(model, queries["query"].tolist(), tokenizer, torch.device("cpu"))

    assert item_emb.shape == (len(items), DIM)
    assert query_emb.shape == (len(queries), DIM)
