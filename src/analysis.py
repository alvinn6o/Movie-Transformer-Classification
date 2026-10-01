"""Error analysis and modality-reliance diagnostics (dev split only).

    python -m src.analysis --run binary/sd_gmu/seed0

Writes reports/error_analysis.md and reports/figures/error_rates_by_genre.png.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import FIGURES_DIR, REPORTS_DIR, RUNS_DIR  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.model import build_head  # noqa: E402
from src.report import GRID, INK, INK_2, SERIES  # noqa: E402  (also applies plot style)
from src.train import load_task, predict  # noqa: E402


def dev_frame(run: str) -> pd.DataFrame:
    metrics = json.loads((RUNS_DIR / run / "metrics.json").read_text())
    scores = np.load(RUNS_DIR / run / "scores.npz")
    df = load_dataset()
    dev = df[df["split"] == "dev"].copy()  # same order as training (dataset order)
    dev["score"] = scores["dev_scores"]
    dev["pred"] = (dev["score"] >= metrics["dev"]["threshold"]).astype(int)
    assert (dev["label"].to_numpy() == scores["dev_labels"]).all()
    return dev, metrics


def error_rates_by_genre(dev: pd.DataFrame, min_n=25) -> pd.DataFrame:
    """Miss rate among Comedies and false-alarm rate among non-Comedies, by co-occurring genre."""
    rows = []
    ex = dev.explode("genres")
    for g, part in ex[ex["genres"] != "Comedy"].groupby("genres"):
        pos, neg = part[part["label"] == 1], part[part["label"] == 0]
        rows.append({"genre": g,
                     "comedies_with_genre": len(pos),
                     "miss_rate": (pos["pred"] == 0).mean() if len(pos) >= min_n else np.nan,
                     "non_comedies_with_genre": len(neg),
                     "false_alarm_rate": (neg["pred"] == 1).mean() if len(neg) >= min_n else np.nan})
    return pd.DataFrame(rows).set_index("genre")


@torch.no_grad()
def modality_reliance(run: str) -> dict:
    """GMU: mean gate on the text branch. Fusion transformer: [FUSE] attention share."""
    ckpt = torch.load(RUNS_DIR / run / "head.pt")
    head = build_head(ckpt["head"], ckpt["modalities"], **ckpt["head_config"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    tensors, y, split_idx, _, _ = load_task(ckpt["backbone"], "binary", ckpt["head"] == "transformer")
    idx = split_idx["dev"]
    f = {k: tensors[k][idx] for k in head.inputs}
    yd = y[idx].bool()
    if ckpt["head"] == "gmu":
        _, z = head(f, return_gate=True)
        z = z.mean(1)
        return {"kind": "GMU text-gate mean (1 = all plot, 0 = all poster)",
                "all": float(z.mean()), "comedy": float(z[yd].mean()), "not_comedy": float(z[~yd].mean())}
    if ckpt["head"] == "transformer":
        _, share = head(f, return_attn=True)
        return {"kind": "[FUSE] last-layer attention share",
                "text": float(share["text"].mean()), "image": float(share["image"].mean())}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="binary/sd_gmu/seed0")
    parser.add_argument("--reliance-runs", nargs="+", default=["binary/sd_gmu/seed0", "binary/sd_xattn/seed0"])
    args = parser.parse_args()

    dev, metrics = dev_frame(args.run)
    rates = error_rates_by_genre(dev)
    fp = dev[(dev["pred"] == 1) & (dev["label"] == 0)].nlargest(10, "score")
    fn = dev[(dev["pred"] == 0) & (dev["label"] == 1)].nsmallest(10, "score")

    def table(part):
        lines = ["| Title | Reference genres | Score |", "| --- | --- | ---: |"]
        lines += [f"| {r.title} | {', '.join(r.genres)} | {r.score:.3f} |" for r in part.itertuples()]
        return lines

    lines = [
        "# Error analysis (dev split)",
        "",
        f"Model: `{args.run}` · threshold {metrics['dev']['threshold']:.3f} (max-F1 on dev) · "
        f"dev P {metrics['dev']['precision']:.3f} / R {metrics['dev']['recall']:.3f} / F1 {metrics['dev']['f1']:.3f}.",
        "Dev is used so the test split stays untouched. Titles and genre tags only; no posters or plots.",
        "",
        "## Error rates by co-occurring genre (n ≥ 25 per cell)",
        "",
        "| Genre | Comedies with genre | Miss rate | Non-comedies with genre | False-alarm rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for g, r in rates.sort_values("false_alarm_rate", ascending=False).iterrows():
        mr = "—" if np.isnan(r.miss_rate) else f"{r.miss_rate:.2f}"
        fa = "—" if np.isnan(r.false_alarm_rate) else f"{r.false_alarm_rate:.2f}"
        lines.append(f"| {g} | {int(r.comedies_with_genre)} | {mr} | {int(r.non_comedies_with_genre)} | {fa} |")
    lines += ["", "## Most confident false positives", ""] + table(fp)
    lines += ["", "## Most confident misses", ""] + table(fn)

    lines += ["", "## Modality reliance", ""]
    for run in args.reliance_runs:
        if (RUNS_DIR / run / "head.pt").exists():
            rel = modality_reliance(run)
            lines.append(f"- `{run}` {rel.pop('kind')}: " + ", ".join(f"{k} {v:.3f}" for k, v in rel.items()))
    lines.append("")
    lines.append("![Error rates](figures/error_rates_by_genre.png)")
    (REPORTS_DIR / "error_analysis.md").write_text("\n".join(lines) + "\n")

    plot = rates.dropna(subset=["false_alarm_rate"]).sort_values("false_alarm_rate")
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(plot) + 1.2))
    y = np.arange(len(plot))
    ax.barh(y, plot["false_alarm_rate"], color=SERIES[1], height=0.6, zorder=2)
    for i, v in enumerate(plot["false_alarm_rate"]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=7, color=INK)
    overall = ((dev["pred"] == 1) & (dev["label"] == 0)).sum() / (dev["label"] == 0).sum()
    ax.axvline(overall, color=INK_2, ls="--", lw=1)
    ax.text(overall + 0.005, len(plot) - 0.6, f"all non-comedies ({overall:.2f})", fontsize=7, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(plot.index, plot["non_comedies_with_genre"])], fontsize=8)
    ax.set(xlabel="False-alarm rate: non-Comedy movies predicted Comedy (dev)",
           title="Which genres get mistaken for Comedy?")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_rates_by_genre.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
