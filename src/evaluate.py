"""Metrics: thresholds, binary and multilabel scores, calibration, run saving."""
import json

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from src.config import RUNS_DIR


def pick_threshold(y_true, scores) -> float:
    """Threshold that maximizes F1. Only ever called on dev data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def expected_calibration_error(y_true, scores, bins: int = 15) -> float:
    """Weighted mean |accuracy - confidence| over equal-width score bins (positive class)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = ids == b
        if sel.any():
            ece += sel.mean() * abs(y_true[sel].mean() - scores[sel].mean())
    return float(ece)


def binary_metrics(y_true, scores, threshold: float) -> dict:
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "ece": expected_calibration_error(y_true, scores),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def macro_ap(Y, S) -> float:
    cols = [k for k in range(Y.shape[1]) if 0 < Y[:, k].sum() < len(Y)]
    return float(np.mean([average_precision_score(Y[:, k], S[:, k]) for k in cols]))


def pick_genre_thresholds(Y_dev, S_dev) -> np.ndarray:
    return np.array([pick_threshold(Y_dev[:, k], S_dev[:, k]) for k in range(Y_dev.shape[1])])


def multilabel_metrics(Y, S, thresholds, genres) -> dict:
    """Thresholds: one per genre (chosen on dev)."""
    Y, S = np.asarray(Y), np.asarray(S)
    P = (S >= np.asarray(thresholds)[None, :]).astype(int)
    per_genre = {}
    for k, g in enumerate(genres):
        per_genre[g] = {
            "support": int(Y[:, k].sum()),
            "average_precision": float(average_precision_score(Y[:, k], S[:, k])) if Y[:, k].any() else None,
            "f1": float(f1_score(Y[:, k], P[:, k], zero_division=0)),
            "precision": float(precision_score(Y[:, k], P[:, k], zero_division=0)),
            "recall": float(recall_score(Y[:, k], P[:, k], zero_division=0)),
            "threshold": float(thresholds[k]),
        }
    return {
        "n": int(len(Y)),
        "macro_ap": macro_ap(Y, S),
        "micro_ap": float(average_precision_score(Y.ravel(), S.ravel())),
        "micro_f1": float(f1_score(Y, P, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y, P, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(Y, P, average="samples", zero_division=0)),
        "mean_true_tags": float(Y.sum(1).mean()),
        "mean_predicted_tags": float(P.sum(1).mean()),
        "per_genre": per_genre,
    }


def save_run(task: str, experiment: str, seed, info: dict, dev: dict, test: dict,
             scores: dict | None = None):
    """artifacts/runs/<task>/<experiment>/seed<k>/{metrics.json, scores.npz}"""
    run_dir = RUNS_DIR / task / experiment / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "experiment": experiment, "seed": seed, **info, "dev": dev, "test": test}
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    if scores is not None:
        np.savez(run_dir / "scores.npz", **scores)
    return run_dir
