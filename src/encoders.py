"""Pretrained transformer encoders behind one interface.

Text encoders:   tokenize(texts) -> {"input_ids", "attention_mask"};  forward -> (pooled, tokens)
Image encoders:  preprocess(images) -> pixel_values;                  forward -> (pooled, tokens)

A backbone is a (text encoder, image encoder) pair from config.BACKBONES, so the
same code path serves caching, heads, LoRA fine-tuning, and inference.
"""
import torch
from torch import nn
from transformers import (AutoImageProcessor, AutoModel, AutoTokenizer, SiglipTextModel,
                          SiglipVisionModel, logging)

from src.config import BACKBONES, CHECKPOINTS

logging.set_verbosity_error()


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors, ignoring padding positions."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class DistilBERTText(nn.Module):
    """DistilBERT; masked mean pool over the last hidden layer."""

    max_tokens = 128  # longest plot is 92 tokens: no truncation

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(CHECKPOINTS["distilbert"])
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS["distilbert"])

    def tokenize(self, texts):
        t = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_tokens,
                           return_tensors="pt")
        return {"input_ids": t["input_ids"], "attention_mask": t["attention_mask"]}

    def n_tokens(self, text) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return mean_pool(hidden, attention_mask), hidden


class SigLIPText(nn.Module):
    """SigLIP 2 text tower; pooled = its own last-token head (the contrastive embedding)."""

    max_tokens = 64  # pretraining length; truncates 0.2% of plots

    def __init__(self):
        super().__init__()
        self.model = SiglipTextModel.from_pretrained(CHECKPOINTS["siglip2"])
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS["siglip2"])

    def tokenize(self, texts):
        # Pretrained on max_length-padded text without an attention mask; the mask
        # returned here is only used by token-level fusion heads.
        ids = self.tokenizer(texts, padding="max_length", truncation=True,
                             max_length=self.max_tokens, return_tensors="pt")["input_ids"]
        return {"input_ids": ids, "attention_mask": (ids != self.tokenizer.pad_token_id).long()}

    def n_tokens(self, text) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def forward(self, input_ids, attention_mask=None):
        out = self.model(input_ids=input_ids)
        return out.pooler_output, out.last_hidden_state


class ViTImage(nn.Module):
    """ViT-B/16 pretrained on ImageNet-21k; pooled = [CLS]; tokens = [CLS] + 196 patches."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(CHECKPOINTS["vit"], add_pooling_layer=False)
        self.processor = AutoImageProcessor.from_pretrained(CHECKPOINTS["vit"])

    def preprocess(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def forward(self, pixel_values):
        hidden = self.model(pixel_values=pixel_values).last_hidden_state
        return hidden[:, 0], hidden


class SigLIPImage(nn.Module):
    """SigLIP 2 vision tower; pooled = attention-pool head; tokens = 196 patches."""

    def __init__(self):
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(CHECKPOINTS["siglip2"])
        self.processor = AutoImageProcessor.from_pretrained(CHECKPOINTS["siglip2"])

    def preprocess(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.pooler_output, out.last_hidden_state


TEXT_ENCODERS = {"distilbert": DistilBERTText, "siglip2": SigLIPText}
IMAGE_ENCODERS = {"vit": ViTImage, "siglip2": SigLIPImage}


class Backbone(nn.Module):
    def __init__(self, name: str, modalities=("text", "image")):
        super().__init__()
        self.name = name
        cfg = BACKBONES[name]
        self.text = TEXT_ENCODERS[cfg["text"]]() if "text" in modalities else None
        self.image = IMAGE_ENCODERS[cfg["image"]]() if "image" in modalities else None

    def tokenize(self, texts):
        return self.text.tokenize(texts)

    def preprocess(self, images):
        return self.image.preprocess(images)

    def encode_text(self, input_ids, attention_mask):
        return self.text(input_ids, attention_mask)

    def encode_image(self, pixel_values):
        return self.image(pixel_values)

    def revisions(self):
        return {m: enc.model.config._commit_hash
                for m, enc in (("text", self.text), ("image", self.image)) if enc is not None}


def load_backbone(name: str, modalities=("text", "image")) -> Backbone:
    return Backbone(name, modalities)
