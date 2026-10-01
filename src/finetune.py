"""LoRA fine-tuning of both encoders, end to end from raw posters and plots.

    python -m src.finetune --backbone siglip2_distilbert --head concat --seeds 0 1 2

Recipe (LP-FT, Kumar et al., 2022): start from the head trained on frozen features
(same backbone/head/seed), then train LoRA adapters in every attention query/value
projection of both encoders plus the head. LoRA's B matrices start at zero, so epoch 0
reproduces the frozen model exactly; that is checked and logged.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, BACKBONES, RUNS_DIR, SPLITS
from src.data import load_dataset, load_poster
from src.encoders import load_backbone
from src.evaluate import binary_metrics, pick_threshold, save_run
from src.experiments import HEAD_CONFIGS
from src.lora import apply_lora, lora_state_dict
from src.model import MultimodalGenreClassifier, build_head
from src.train import set_seed

SHORT = {"vit_distilbert": "vd", "siglip2": "sg", "siglip2_distilbert": "sd"}
CONFIG = {
    "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "v_proj", "q_lin", "v_lin"],
    "lora_lr": 2e-4, "head_lr": 1e-4, "weight_decay": 0.01,
    "batch_size": 32, "epochs": 6, "patience": 2, "warmup_frac": 0.05,
}


def device():
    return torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")


def pixel_cache(backbone, df) -> torch.Tensor:
    """Preprocess every poster once (fp16), so epochs skip JPEG decoding."""
    path = ARTIFACTS_DIR / f"pixels_{BACKBONES[backbone.name]['image']}.pt"
    if path.exists():
        return torch.load(path)
    chunks = [backbone.preprocess([load_poster(p) for p in df["image_file"].iloc[i:i + 64]]).half()
              for i in tqdm(range(0, len(df), 64), desc="pixels", mininterval=10)]
    pixels = torch.cat(chunks)
    torch.save(pixels, path)
    return pixels


@torch.no_grad()
def predict(model, data, idx, dev, bs=64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        logits = model(data["input_ids"][b].to(dev), data["attention_mask"][b].to(dev),
                       data["pixels"][b].to(dev).float())
        out.append(torch.sigmoid(logits).float().cpu())
    return torch.cat(out).numpy()


def run(backbone_name: str, head_kind: str, seed: int, data, df, split_idx, y):
    set_seed(seed)
    dev = device()
    short = SHORT[backbone_name]
    name = f"{short}_lora_{head_kind}"
    frozen_run = RUNS_DIR / "binary" / f"{short}_{head_kind}" / f"seed{seed}"

    backbone = load_backbone(backbone_name)
    head = build_head(head_kind, ("text", "image"), **HEAD_CONFIGS[head_kind])
    if (frozen_run / "head.pt").exists():  # LP-FT: start from the frozen-feature head
        head.load_state_dict(torch.load(frozen_run / "head.pt")["state_dict"])
        init = str(frozen_run.relative_to(RUNS_DIR))
    else:
        init = "random"
    model = MultimodalGenreClassifier(backbone, head, freeze_encoders=True)
    adapted = apply_lora(model.backbone, CONFIG["lora_targets"], CONFIG["lora_r"],
                         CONFIG["lora_alpha"], CONFIG["lora_dropout"])
    model.to(dev)

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    head_params = list(model.head.parameters())
    n_lora, n_head = sum(p.numel() for p in lora_params), sum(p.numel() for p in head_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{name} seed {seed}: {len(adapted)} LoRA layers, trainable {n_lora + n_head:,} "
          f"({100 * (n_lora + n_head) / n_total:.2f}% of {n_total:,}); head init: {init}", flush=True)

    opt = torch.optim.AdamW([{"params": lora_params, "lr": CONFIG["lora_lr"]},
                             {"params": head_params, "lr": CONFIG["head_lr"]}],
                            weight_decay=CONFIG["weight_decay"])
    tr = split_idx["train"]
    steps = CONFIG["epochs"] * math.ceil(len(tr) / CONFIG["batch_size"])
    warmup = max(1, int(CONFIG["warmup_frac"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    y_tr = y[tr]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=((y_tr == 0).sum() / (y_tr == 1).sum()).to(dev))
    y_dev = y[split_idx["dev"]].numpy()

    def dev_ap():
        return binary_metrics(y_dev, predict(model, data, split_idx["dev"], dev), 0.5)["average_precision"]

    history = [{"epoch": 0, "dev_ap": dev_ap(), "train_loss": None, "seconds": 0.0}]
    print(f"  epoch 0 (frozen start) dev AP {history[0]['dev_ap']:.4f}", flush=True)
    best = {"ap": history[0]["dev_ap"], "epoch": 0,
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                      if "lora_" in k or k.startswith("head.")}}
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        start = time.perf_counter()
        perm = tr[torch.randperm(len(tr))]
        total = 0.0
        for i in range(0, len(perm), CONFIG["batch_size"]):
            b = perm[i:i + CONFIG["batch_size"]]
            logits = model(data["input_ids"][b].to(dev), data["attention_mask"][b].to(dev),
                           data["pixels"][b].to(dev).float())
            loss = loss_fn(logits, y[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        train_seconds = time.perf_counter() - start
        ap = dev_ap()
        history.append({"epoch": epoch, "dev_ap": ap, "train_loss": total / len(tr),
                        "seconds": round(train_seconds, 1)})
        print(f"  epoch {epoch} loss {total / len(tr):.4f} dev AP {ap:.4f} ({train_seconds:.0f}s train)",
              flush=True)
        if ap > best["ap"]:
            best = {"ap": ap, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                              if "lora_" in k or k.startswith("head.")}}
        elif epoch - best["epoch"] >= CONFIG["patience"]:
            break

    model.load_state_dict(best["state"], strict=False)
    dev_s = predict(model, data, split_idx["dev"], dev)
    test_s = predict(model, data, split_idx["test"], dev)
    y_test = y[split_idx["test"]].numpy()
    threshold = pick_threshold(y_dev, dev_s)
    dev_m, test_m = binary_metrics(y_dev, dev_s, threshold), binary_metrics(y_test, test_s, threshold)
    info = {
        "backbone": backbone_name, "head": head_kind, "modalities": ["text", "image"],
        "inputs": "plot + poster",
        "trainable": f"LoRA r={CONFIG['lora_r']} on {len(adapted)} q/v projections + head",
        "trainable_params": n_lora + n_head, "total_params": n_total, "head_init": init,
        "config": CONFIG, "best_epoch": best["epoch"], "history": history,
    }
    run_dir = save_run("binary", name, seed, info, dev_m, test_m,
                       {"dev_scores": dev_s, "dev_labels": y_dev, "test_scores": test_s, "test_labels": y_test})
    torch.save({"lora": lora_state_dict(model), "head": model.head.state_dict(),
                "backbone": backbone_name, "head_kind": head_kind, "modalities": ("text", "image"),
                "head_config": HEAD_CONFIGS[head_kind], "lora_config": CONFIG,
                "thresholds": [threshold]}, run_dir / "model.pt")
    print(f"  -> best epoch {best['epoch']}  dev AP {dev_m['average_precision']:.4f}  "
          f"test AP {test_m['average_precision']:.4f}  AUC {test_m['roc_auc']:.4f}", flush=True)
    del model, opt
    if dev.type == "mps":
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=list(BACKBONES), default="siglip2_distilbert")
    parser.add_argument("--head", choices=list(HEAD_CONFIGS), default="concat")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    args = parser.parse_args()
    CONFIG["epochs"] = args.epochs

    df = load_dataset()
    backbone = load_backbone(args.backbone)
    tokens = backbone.tokenize(df["plot"].tolist())
    data = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
            "pixels": pixel_cache(backbone, df)}
    del backbone
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
    split_idx = {s: torch.from_numpy(np.flatnonzero(df["split"].to_numpy() == s)) for s in SPLITS}
    for seed in args.seeds:
        run(args.backbone, args.head, seed, data, df, split_idx, y)


if __name__ == "__main__":
    main()
