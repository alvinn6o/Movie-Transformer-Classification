"""Run a frozen backbone once over every movie and cache its outputs.

    python -m src.embed --backbone vit_distilbert
    python -m src.embed --backbone siglip2

Writes artifacts/features_<backbone>.pt (pooled, fp32) and
artifacts/tokens_<backbone>.pt (token sequences, fp16, for the fusion transformer).
"""
import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import BACKBONES, features_path, tokens_path
from src.data import load_dataset, load_poster
from src.encoders import load_backbone


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_texts(texts, backbone, device, batch_size=64):
    pooled, tokens, masks = [], [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="text", mininterval=10):
        batch = {k: v.to(device) for k, v in backbone.tokenize(texts[i:i + batch_size]).items()}
        p, t = backbone.encode_text(batch["input_ids"], batch["attention_mask"])
        pooled.append(p.float().cpu())
        tokens.append(t.half().cpu())
        masks.append(batch["attention_mask"].bool().cpu())
    # Batches are padded to their own longest plot; right-pad all to one length.
    L = max(t.shape[1] for t in tokens)
    tokens = torch.cat([F.pad(t, (0, 0, 0, L - t.shape[1])) for t in tokens])
    masks = torch.cat([F.pad(m, (0, L - m.shape[1])) for m in masks])
    return torch.cat(pooled), tokens, masks


@torch.no_grad()
def embed_images(paths, backbone, device, batch_size=32):
    pooled, tokens = [], []
    for i in tqdm(range(0, len(paths), batch_size), desc="image", mininterval=10):
        pixels = backbone.preprocess([load_poster(p) for p in paths[i:i + batch_size]]).to(device)
        p, t = backbone.encode_image(pixels)
        pooled.append(p.float().cpu())
        tokens.append(t.half().cpu())
    return torch.cat(pooled), torch.cat(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["vit_distilbert", "siglip2"], default="vit_distilbert")
    args = parser.parse_args()

    df = load_dataset()
    device = get_device()
    backbone = load_backbone(args.backbone).eval().to(device)

    start = time.perf_counter()
    # Only plot text and poster files go in; genres stay in df for labels.
    text, text_tokens, text_mask = embed_texts(df["plot"].tolist(), backbone, device)
    image, image_tokens = embed_images(df["image_file"].tolist(), backbone, device)
    elapsed = time.perf_counter() - start

    meta = {
        "backbone": args.backbone,
        "encoders": BACKBONES[args.backbone],
        "revisions": backbone.revisions(),
        "text_tokens_shape": list(text_tokens.shape),
        "image_tokens_shape": list(image_tokens.shape),
        "plots_truncated": int(sum(backbone.text.n_tokens(t) > backbone.text.max_tokens
                                   for t in df["plot"])),
        "device": str(device),
        "seconds": round(elapsed, 1),
    }

    ids = df["movie_id"].tolist()
    torch.save({"movie_id": ids, "text": text, "image": image, "meta": meta},
               features_path(args.backbone))
    torch.save({"movie_id": ids, "text_tokens": text_tokens, "text_mask": text_mask,
                "image_tokens": image_tokens, "meta": meta}, tokens_path(args.backbone))
    print(f"{args.backbone}: {len(ids)} movies, text tokens {tuple(text_tokens.shape)}, "
          f"image tokens {tuple(image_tokens.shape)}, {elapsed:.0f}s on {device}, "
          f"{meta['plots_truncated']} plots truncated")


if __name__ == "__main__":
    main()
