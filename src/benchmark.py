"""Latency of end-to-end inference (preprocessing + both encoders + head), batch 1.

    python -m src.benchmark --runs binary/vd_concat/seed0 binary/sd_gmu/seed0 --devices cpu mps

Also checks that live predictions match the scores computed from cached features.
"""
import argparse
import json
import platform

import numpy as np
import torch

from src.config import REPORTS_DIR, RUNS_DIR
from src.data import load_dataset
from src.predict import ComedyPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    df = load_dataset()
    test = df[df["split"] == "test"].head(args.n)
    results = {"hardware": platform.machine(), "processor": platform.processor(),
               "torch_threads": torch.get_num_threads(), "n_movies": len(test), "batch_size": 1, "runs": {}}
    for run in args.runs:
        cached = np.load(RUNS_DIR / run / "scores.npz")["test_scores"][:args.n]
        for device in args.devices:
            predictor = ComedyPredictor(run, device)
            for row in test.head(3).itertuples():  # warm-up
                predictor(row.image_file, row.plot)
            out = [predictor(row.image_file, row.plot) for row in test.itertuples()]
            lat = np.array([o["latency_ms"] for o in out])
            live = np.array([o["comedy_score"] for o in out])
            n_params = sum(p.numel() for p in predictor.model.parameters())
            results["runs"].setdefault(run, {})[device] = {
                "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                "params": int(n_params), "max_abs_diff_vs_cached": float(np.abs(live - cached).max()),
            }
            print(f"{run:32s} {device:4s} p50 {np.percentile(lat, 50):6.1f} ms  p95 {np.percentile(lat, 95):6.1f} ms  "
                  f"params {n_params / 1e6:.0f}M  max|live-cached| {np.abs(live - cached).max():.1e}", flush=True)
            del predictor
    (REPORTS_DIR / "latency.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
