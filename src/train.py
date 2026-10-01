"""Train heads on cached frozen-encoder outputs.

    python -m src.train --task binary --experiments vd_concat sg_xattn
    python -m src.train --task binary --experiments all
    python -m src.train --task multilabel --experiments sg_text sg_image sg_concat

For each experiment: learning rate chosen from TRAIN_CONFIG["lr_grid"] on dev with
seed 0, then trained with every seed at that rate. Test scores are recorded but never
used for any choice.
"""
import argparse
import random
import time

import numpy as np
import torch
from torch import nn

from src.config import MIN_GENRE_TRAIN_SUPPORT, SPLITS, cache_source, features_path, tokens_path
from src.data import load_dataset
from src.evaluate import (binary_metrics, macro_ap, multilabel_metrics, pick_genre_thresholds,
                          pick_threshold, save_run)
from src.experiments import EXPERIMENTS, HEAD_CONFIGS, TRAIN_CONFIG
from src.model import build_head

_FEATURE_CACHE = {}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(token_level: bool) -> torch.device:
    # Pooled heads are tiny and train fastest (and deterministically) on CPU.
    if token_level and torch.backends.mps.is_available():
        return torch.device("mps")
    if token_level and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def genre_vocab(df) -> list[str]:
    counts = df[df["split"] == "train"].explode("genres")["genres"].value_counts()
    return sorted(counts[counts >= MIN_GENRE_TRAIN_SUPPORT].index)


def load_task(backbone: str, task: str, need_tokens: bool):
    """Feature tensors (aligned to cache order), targets, split indices, genre list."""
    key = (backbone, need_tokens)
    if key not in _FEATURE_CACHE:
        _FEATURE_CACHE.clear()  # keep at most one set of token tensors in memory
        tensors, meta, ids = {}, {}, None
        for m in ("text", "image"):
            src = cache_source(backbone, m)
            pooled = torch.load(features_path(src))
            assert ids is None or pooled["movie_id"] == ids
            ids = pooled["movie_id"]
            tensors[m] = pooled[m]
            meta[m] = {"cache": src, **pooled["meta"]}
            if need_tokens:
                tok = torch.load(tokens_path(src), mmap=True)
                keys = ["text_tokens", "text_mask"] if m == "text" else ["image_tokens"]
                tensors.update({k: tok[k].clone() for k in keys})
                del tok
        _FEATURE_CACHE[key] = (ids, tensors, meta)
    ids, tensors, meta = _FEATURE_CACHE[key]

    df = load_dataset().set_index("movie_id").loc[ids]
    if task == "binary":
        genres = ["Comedy"]
        y = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
    else:
        genres = genre_vocab(df.reset_index())
        y = torch.tensor([[int(g in tags) for g in genres] for tags in df["genres"]],
                         dtype=torch.float32)
    split_idx = {s: torch.from_numpy(np.flatnonzero(df["split"].to_numpy() == s)) for s in SPLITS}
    return tensors, y, split_idx, genres, meta


def batch(tensors, idx, keys, device):
    return {k: tensors[k][idx].to(device) for k in keys}


@torch.no_grad()
def predict(model, tensors, idx, device, bs=256) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(batch(tensors, idx[i:i + bs], model.inputs, device))).float().cpu()
           for i in range(0, len(idx), bs)]
    return torch.cat(out).numpy()


def selection_metric(task, y, scores) -> float:
    if task == "binary":
        return binary_metrics(y, scores, 0.5)["average_precision"]
    return macro_ap(y, scores)


def train_head(spec, task, tensors, y, split_idx, seed, lr, cfg=TRAIN_CONFIG):
    set_seed(seed)
    head_cfg = HEAD_CONFIGS[spec["head"]]
    model = build_head(spec["head"], spec["modalities"], num_labels=y.shape[1] if y.ndim == 2 else 1,
                       **head_cfg)
    device = get_device(model.token_level)
    model.to(device)

    tr = split_idx["train"]
    y_tr = y[tr]
    if task == "binary":
        # Up-weight positives: Comedy is ~34% of training movies.
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum())
    else:
        loss_fn = nn.BCEWithLogitsLoss()  # per-genre thresholds handle imbalance
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    y_dev = y[split_idx["dev"]].numpy()

    best = {"metric": -1.0, "epoch": 0, "state": None}
    history = []
    start = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        total = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            loss = loss_fn(model(batch(tensors, b, model.inputs, device)), y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        dev_metric = selection_metric(task, y_dev, predict(model, tensors, split_idx["dev"], device))
        history.append({"epoch": epoch, "train_loss": total / len(tr), "dev_metric": dev_metric})
        if dev_metric > best["metric"]:
            best = {"metric": dev_metric, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= cfg["patience"]:
            break
    model.load_state_dict(best["state"])
    return model, device, {"best_epoch": best["epoch"], "dev_selection_metric": best["metric"],
                           "train_seconds": round(time.perf_counter() - start, 1),
                           "history": history}


def evaluate_and_save(name, spec, task, model, device, train_info, tensors, y, split_idx,
                      genres, meta, seed, lr, lr_search):
    dev_s = predict(model, tensors, split_idx["dev"], device)
    test_s = predict(model, tensors, split_idx["test"], device)
    y_dev, y_test = y[split_idx["dev"]].numpy(), y[split_idx["test"]].numpy()
    if task == "binary":
        threshold = pick_threshold(y_dev, dev_s)
        dev, test = binary_metrics(y_dev, dev_s, threshold), binary_metrics(y_test, test_s, threshold)
        thresholds = [threshold]
    else:
        thresholds = pick_genre_thresholds(y_dev, dev_s)
        dev = multilabel_metrics(y_dev, dev_s, thresholds, genres)
        test = multilabel_metrics(y_test, test_s, thresholds, genres)
        thresholds = thresholds.tolist()

    info = {
        "backbone": spec["backbone"], "head": spec["head"], "modalities": list(spec["modalities"]),
        "inputs": " + ".join({"text": "plot", "image": "poster"}[m] for m in spec["modalities"]),
        "trainable": "head only (encoders frozen)",
        "trainable_params": sum(p.numel() for p in model.parameters()),
        "lr": lr, "lr_search": lr_search, "head_config": HEAD_CONFIGS[spec["head"]],
        "train_config": TRAIN_CONFIG, "genres": genres, "encoder_meta": meta, **train_info,
    }
    run_dir = save_run(task, name, seed, info, dev, test,
                       {"dev_scores": dev_s, "dev_labels": y_dev,
                        "test_scores": test_s, "test_labels": y_test})
    torch.save({"state_dict": model.cpu().state_dict(), "backbone": spec["backbone"],
                "head": spec["head"], "modalities": spec["modalities"],
                "head_config": HEAD_CONFIGS[spec["head"]], "num_labels": len(genres),
                "genres": genres, "thresholds": thresholds}, run_dir / "head.pt")
    key = "average_precision" if task == "binary" else "macro_ap"
    print(f"  {name:10s} seed {seed} lr {lr:.0e} epoch {train_info['best_epoch']:3d} "
          f"({train_info['train_seconds']:.0f}s)  dev {dev[key]:.3f}  test {test[key]:.3f}", flush=True)


def run_experiment(name: str, task: str, seeds, lr_grid):
    spec = EXPERIMENTS[name]
    need_tokens = spec["head"] == "transformer"
    tensors, y, split_idx, genres, meta = load_task(spec["backbone"], task, need_tokens)
    print(f"{name} [{task}]  backbone={spec['backbone']} head={spec['head']} "
          f"inputs={spec['modalities']} genres={len(genres)}", flush=True)

    # 1) learning-rate search on dev with seed 0
    lr_search, trained = {}, {}
    for lr in lr_grid:
        model, device, info = train_head(spec, task, tensors, y, split_idx, seeds[0], lr)
        lr_search[str(lr)] = info["dev_selection_metric"]
        trained[lr] = (model, device, info)
    best_lr = max(lr_grid, key=lambda lr: lr_search[str(lr)])

    # 2) all seeds at the chosen rate (seed 0 reuses the search run)
    for seed in seeds:
        if seed == seeds[0]:
            model, device, info = trained[best_lr]
        else:
            model, device, info = train_head(spec, task, tensors, y, split_idx, seed, best_lr)
        evaluate_and_save(name, spec, task, model, device, info, tensors, y, split_idx,
                          genres, meta, seed, best_lr, lr_search)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "multilabel"], default="binary")
    parser.add_argument("--experiments", nargs="+", default=["all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr", type=float, nargs="+", default=TRAIN_CONFIG["lr_grid"])
    args = parser.parse_args()

    names = list(EXPERIMENTS) if args.experiments == ["all"] else args.experiments
    # Group by backbone so each token cache is loaded once.
    names.sort(key=lambda n: (EXPERIMENTS[n]["backbone"], EXPERIMENTS[n]["head"] == "transformer"))
    for name in names:
        run_experiment(name, args.task, args.seeds, args.lr)


if __name__ == "__main__":
    main()
