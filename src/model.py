"""Classification heads and the end-to-end multimodal model.

Heads over frozen-encoder outputs:
  ConcatHead             pooled vectors -> LayerNorm -> concat -> MLP        (late fusion)
  GatedFusionHead        pooled vectors -> GMU gate z * h_text + (1-z) * h_image
  FusionTransformerHead  token sequences -> [FUSE] + text tokens + image patches
                         -> pre-norm self-attention blocks -> [FUSE] -> logits
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from src.config import EMBED_DIM
from src.encoders import mean_pool  # noqa: F401  (re-exported for tests/notebook)


class ConcatHead(nn.Module):
    """MLP over one or more pooled modality embeddings (late fusion by concatenation)."""

    token_level = False

    def __init__(self, modalities=("text", "image"), dim=EMBED_DIM, hidden=256,
                 dropout=0.2, num_labels=1, **_):
        super().__init__()
        self.modalities = tuple(modalities)
        self.num_labels = num_labels
        # Per-modality LayerNorm puts the two encoders' vectors on a similar scale.
        self.norms = nn.ModuleDict({m: nn.LayerNorm(dim) for m in self.modalities})
        self.mlp = nn.Sequential(
            nn.Linear(dim * len(self.modalities), hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    @property
    def inputs(self):
        return list(self.modalities)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([self.norms[m](feats[m]) for m in self.modalities], dim=-1)
        logits = self.mlp(x)
        return logits.squeeze(-1) if self.num_labels == 1 else logits


GenreHead = ConcatHead  # name used by the first version


class GatedFusionHead(nn.Module):
    """Gated Multimodal Unit (Arevalo et al., 2017, the MM-IMDb paper).

    h_t = tanh(W_t x_t), h_v = tanh(W_v x_v), z = sigmoid(W_z [x_t; x_v])
    h   = z * h_t + (1 - z) * h_v
    The gate z decides, per movie and per hidden unit, how much to trust each modality.
    """

    token_level = False
    modalities = ("text", "image")

    def __init__(self, dim=EMBED_DIM, hidden=256, dropout=0.2, num_labels=1, **_):
        super().__init__()
        self.num_labels = num_labels
        self.norm_t, self.norm_v = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.h_t, self.h_v = nn.Linear(dim, hidden), nn.Linear(dim, hidden)
        self.gate = nn.Linear(2 * dim, hidden)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_labels))

    @property
    def inputs(self):
        return ["text", "image"]

    def forward(self, feats, return_gate=False):
        xt, xv = self.norm_t(feats["text"]), self.norm_v(feats["image"])
        z = torch.sigmoid(self.gate(torch.cat([xt, xv], dim=-1)))
        h = z * torch.tanh(self.h_t(xt)) + (1 - z) * torch.tanh(self.h_v(xv))
        logits = self.out(h)
        logits = logits.squeeze(-1) if self.num_labels == 1 else logits
        return (logits, z) if return_gate else logits


class AttentionBlock(nn.Module):
    """Pre-norm transformer encoder block with explicit multi-head attention."""

    def __init__(self, d_model=256, heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.heads, self.head_dim = heads, d_model // heads
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(mlp_ratio * d_model, d_model))
        self.attn_drop, self.drop = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, keep_mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)          # [B, H, L, L]
        scores = scores.masked_fill(~keep_mask[:, None, None, :], float("-inf"))  # ignore padding keys
        attn = scores.softmax(dim=-1)
        out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, attn


def pool_patches(tokens: torch.Tensor, k: int) -> torch.Tensor:
    """Average-pool a square grid of patch tokens k x k (a leading [CLS] is kept)."""
    n = tokens.shape[1]
    side = math.isqrt(n)
    cls = None
    if side * side != n:  # ViT: [CLS] + 14x14 patches
        cls, tokens, side = tokens[:, :1], tokens[:, 1:], math.isqrt(n - 1)
    B, _, D = tokens.shape
    grid = tokens.float().reshape(B, side, side, D).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, k).flatten(2).transpose(1, 2)
    return pooled if cls is None else torch.cat([cls.float(), pooled], dim=1)


class FusionTransformerHead(nn.Module):
    """Token-level fusion transformer over frozen encoder outputs.

    Sequence = [FUSE] + text tokens + image patch tokens, each projected 768 -> d_model
    and tagged with a learned modality embedding. Self-attention lets every plot token
    attend to every poster patch (and vice versa); the [FUSE] output is classified.
    Positional information is already inside the encoder outputs. Poster patches are
    average-pooled 2x2 (196 -> 49 tokens), which cuts the sequence ~2.3x.
    """

    token_level = True

    def __init__(self, modalities=("text", "image"), dim=EMBED_DIM, d_model=256, heads=4,
                 layers=2, dropout=0.1, num_labels=1, image_pool=2, **_):
        super().__init__()
        self.modalities = tuple(modalities)
        self.image_pool = image_pool
        self.num_labels = num_labels
        self.proj = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d_model))
                                   for m in self.modalities})
        self.type_emb = nn.ParameterDict({m: nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
                                          for m in self.modalities})
        self.fuse_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([AttentionBlock(d_model, heads, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_labels))

    @property
    def inputs(self):
        keys = {"text": ["text_tokens", "text_mask"], "image": ["image_tokens"]}
        return [k for m in self.modalities for k in keys[m]]

    def forward(self, feats, return_attn=False):
        B = next(iter(feats.values())).shape[0]
        seqs = [self.fuse_token.expand(B, -1, -1)]
        masks = [torch.ones(B, 1, dtype=torch.bool, device=seqs[0].device)]
        spans = {}
        for m in self.modalities:
            raw = feats[f"{m}_tokens"]
            if m == "image":
                raw = pool_patches(raw, self.image_pool) if self.image_pool > 1 else raw
                mask = torch.ones(raw.shape[:2], dtype=torch.bool, device=raw.device)
            else:
                mask = feats["text_mask"].bool()
            tokens = self.proj[m](raw.float()) + self.type_emb[m]
            start = sum(s.shape[1] for s in seqs)
            spans[m] = (start, start + tokens.shape[1])
            seqs.append(tokens)
            masks.append(mask)
        x, keep = torch.cat(seqs, dim=1), torch.cat(masks, dim=1)
        for block in self.blocks:
            x, attn = block(x, keep)
        logits = self.out(self.norm(x[:, 0]))
        logits = logits.squeeze(-1) if self.num_labels == 1 else logits
        if not return_attn:
            return logits
        # Share of [FUSE]'s last-layer attention landing on each modality (mean over heads).
        fuse_attn = attn[:, :, 0].mean(1)  # [B, L]
        share = {m: fuse_attn[:, a:b].sum(-1) for m, (a, b) in spans.items()}
        return logits, share


HEADS = {"concat": ConcatHead, "gmu": GatedFusionHead, "transformer": FusionTransformerHead}


def build_head(kind: str, modalities, num_labels=1, **kwargs) -> nn.Module:
    if kind == "gmu":
        return GatedFusionHead(num_labels=num_labels, **kwargs)
    return HEADS[kind](modalities=modalities, num_labels=num_labels, **kwargs)


class MultimodalGenreClassifier(nn.Module):
    """Backbone (two transformer encoders) + head, run end to end from raw inputs.
    Used for LoRA fine-tuning and for inference."""

    def __init__(self, backbone: nn.Module, head: nn.Module, freeze_encoders: bool = True):
        super().__init__()
        self.backbone = backbone
        self.head = head
        if freeze_encoders:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def features(self, input_ids=None, attention_mask=None, pixel_values=None) -> dict:
        needs = set(self.head.inputs)
        feats = {}
        if needs & {"text", "text_tokens"}:
            pooled, tokens = self.backbone.encode_text(input_ids, attention_mask)
            feats.update(text=pooled, text_tokens=tokens, text_mask=attention_mask)
        if needs & {"image", "image_tokens"}:
            pooled, tokens = self.backbone.encode_image(pixel_values)
            feats.update(image=pooled, image_tokens=tokens)
        return feats

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        return self.head(self.features(input_ids, attention_mask, pixel_values))
