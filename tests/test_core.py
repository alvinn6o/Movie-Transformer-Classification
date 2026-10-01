"""Fast checks that need neither the dataset nor pretrained weights."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.data import parse_genres
from src.evaluate import (binary_metrics, expected_calibration_error, multilabel_metrics,
                          pick_threshold)
from src.lora import LoRALinear, apply_lora
from src.model import (ConcatHead, FusionTransformerHead, GatedFusionHead, build_head, mean_pool,
                       pool_patches)


def test_parse_genres_splits_and_strips():
    assert parse_genres("Drama - Comedy - Documentary") == ["Drama", "Comedy", "Documentary"]


@pytest.mark.parametrize("value", [None, float("nan"), "  "])
def test_parse_genres_rejects_missing(value):
    with pytest.raises(ValueError):
        parse_genres(value)


def test_comedy_label_is_membership_not_exclusive():
    genres = pd.Series(["Comedy", "Comedy - Romance", "Horror - Thriller"]).apply(parse_genres)
    assert genres.apply(lambda g: int("Comedy" in g)).tolist() == [1, 1, 0]


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))


def feats(B=4, D=8, Lt=6, P=16):
    mask = torch.ones(B, Lt, dtype=torch.long)
    mask[:, 4:] = 0
    return {"text": torch.randn(B, D), "image": torch.randn(B, D),
            "text_tokens": torch.randn(B, Lt, D), "text_mask": mask,
            "image_tokens": torch.randn(B, P, D).half()}


@pytest.mark.parametrize("kind,modalities", [
    ("concat", ("text",)), ("concat", ("image",)), ("concat", ("text", "image")),
    ("gmu", ("text", "image")), ("transformer", ("text", "image")),
])
@pytest.mark.parametrize("num_labels,shape", [(1, (4,)), (5, (4, 5))])
def test_head_output_shapes(kind, modalities, num_labels, shape):
    head = build_head(kind, modalities, num_labels=num_labels, dim=8, hidden=4, d_model=8, heads=2)
    assert head(feats()).shape == shape


def test_heads_only_declare_needed_inputs():
    assert ConcatHead(("text",), dim=8).inputs == ["text"]
    assert GatedFusionHead(dim=8).inputs == ["text", "image"]
    assert FusionTransformerHead(dim=8, d_model=8, heads=2).inputs == ["text_tokens", "text_mask", "image_tokens"]


def test_fusion_transformer_ignores_padded_text_tokens():
    torch.manual_seed(0)
    head = FusionTransformerHead(dim=8, d_model=8, heads=2, image_pool=1).eval()
    f = feats(B=1)
    out1 = head(f)
    f["text_tokens"][:, 4:] = 1e3  # change only padding positions
    assert torch.allclose(out1, head(f), atol=1e-5)


def test_fusion_transformer_attention_share_sums_to_one_with_fuse():
    head = FusionTransformerHead(dim=8, d_model=8, heads=2).eval()
    _, share = head(feats(), return_attn=True)
    total = share["text"] + share["image"]
    assert ((total > 0) & (total <= 1 + 1e-5)).all()  # remainder goes to [FUSE] itself


@pytest.mark.parametrize("n,expected", [(196, 49), (197, 50)])  # SigLIP grid; ViT [CLS] + grid
def test_pool_patches(n, expected):
    assert pool_patches(torch.randn(2, n, 8), 2).shape == (2, expected, 8)


def test_lora_starts_as_identity_and_trains_only_adapters():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    model[0].q_proj = None  # no-op attribute; targets are matched by module name below
    x = torch.randn(3, 8)
    before = model(x)
    wrapped = nn.ModuleDict({"q_proj": model[0]})
    names = apply_lora(wrapped, ("q_proj",), r=2, alpha=4)
    assert names == ["q_proj"] and isinstance(wrapped["q_proj"], LoRALinear)
    assert torch.allclose(wrapped["q_proj"](x), model[0](x))  # B = 0 at init
    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert trainable == ["q_proj.lora_A.weight", "q_proj.lora_B.weight"]
    assert torch.allclose(before, model(x))


def test_threshold_and_metrics():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    t = pick_threshold(y, s)
    m = binary_metrics(y, s, t)
    assert m["recall"] == 1.0 and m["n_positive"] == 2


def test_ece_zero_for_perfectly_calibrated_bins():
    y = np.array([0, 1] * 50)
    s = np.full(100, 0.5)
    assert expected_calibration_error(y, s) == pytest.approx(0.0)


def test_multilabel_metrics():
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    S = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.2]])
    m = multilabel_metrics(Y, S, np.array([0.5, 0.5]), ["A", "B"])
    assert m["micro_f1"] == 1.0 and m["macro_ap"] == 1.0
    assert m["mean_predicted_tags"] == m["mean_true_tags"] == 1.0
