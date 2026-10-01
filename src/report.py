"""Aggregate every run into reports/results.md, reports/results.json, and figures.

    python -m src.report

The "selected" model is the one with the best mean dev score; test numbers are
reported for everything but never used to choose.
"""
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from src.config import FIGURES_DIR, REPORTS_DIR, RUNS_DIR  # noqa: E402
from src.evaluate import expected_calibration_error  # noqa: E402
from src.experiments import describe  # noqa: E402

# Reference categorical palette (light mode), fixed slot order; grey for baselines.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
GREY, INK, INK_2, SURFACE, GRID = "#9a9993", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
BACKBONE_COLOR = {"vd": SERIES[0], "sg": SERIES[1], "sd": SERIES[2]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def load_runs(task):
    runs = defaultdict(list)
    for path in sorted((RUNS_DIR / task).glob("*/seed*/metrics.json")):
        m = json.loads(path.read_text())
        m["_dir"] = path.parent
        runs[m["experiment"]].append(m)
    return runs


def mean_std(values):
    v = np.array(values, dtype=float)
    return float(v.mean()), float(v.std())


def fmt(values, digits=3):
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}" if len(values) > 1 else f"{m:.{digits}f}"


BASELINES = ["majority", "tfidf_logreg", "sg_zeroshot"]
HEAD_ORDER = ["text", "image", "concat", "gmu", "xattn", "lora_concat", "lora_gmu", "lora_xattn"]


def order_key(exp):
    """Baselines first, then by backbone (vd, sg, sd), then by head."""
    if exp in BASELINES:
        return (0, BASELINES.index(exp))
    prefix, _, rest = exp.partition("_")
    return (1 + ["vd", "sg", "sd"].index(prefix), HEAD_ORDER.index(rest))


def scores(run, split):
    d = np.load(run["_dir"] / "scores.npz")
    return d[f"{split}_labels"], d[f"{split}_scores"]


# ----------------------------------------------------------------------------- binary
def binary_section(runs):
    exps = sorted(runs, key=order_key)
    selected = max((e for e in exps if e not in ("majority",)),
                   key=lambda e: mean_std([r["dev"]["average_precision"] for r in runs[e]])[0])
    lines = [
        "## Version 1: Comedy vs. not Comedy",
        "",
        "Supplied splits: train 2,142 · dev 1,071 · test 1,072 (344 Comedy, prevalence 0.321). "
        "Learning rate (and C for TF-IDF) chosen on dev; decision threshold = max-F1 on dev. "
        "Mean ± std over seeds. **Bold = selected on dev AP.** ECE uses 15 bins on raw scores.",
        "",
        "| Backbone | Model | Inputs | Trainable params | Seeds | Dev AP | Test AP | Test ROC AUC | Test F1 | Brier | ECE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    table = {}
    for e in exps:
        rs = runs[e]
        bb, model = describe(e)
        get = lambda split, k: [r[split][k] for r in rs]  # noqa: E731
        params = rs[0].get("trainable_params", 0)
        row = [bb, model, rs[0].get("inputs", ""), f"{params:,}", str(len(rs)),
               fmt(get("dev", "average_precision")), fmt(get("test", "average_precision")),
               fmt(get("test", "roc_auc")), fmt(get("test", "f1")),
               fmt(get("test", "brier")), fmt(get("test", "ece"))]
        if e == selected:
            row = [f"**{c}**" for c in row]
        lines.append("| " + " | ".join(row) + " |")
        table[e] = {f"{s}_{k}": dict(zip(("mean", "std"), mean_std(get(s, k))))
                    for s in ("dev", "test") for k in ("average_precision", "roc_auc", "f1", "brier", "ece")}
        table[e].update(trainable_params=params, seeds=len(rs), description=f"{bb} | {model}")
    return lines, table, selected


def fig_binary_bars(runs, path):
    exps = [e for e in sorted(runs, key=order_key)]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(exps) + 1.2))
    for i, e in enumerate(exps):
        vals = [r["test"]["average_precision"] for r in runs[e]]
        m = np.mean(vals)
        color = BACKBONE_COLOR.get(e.split("_")[0], GREY) if "zeroshot" not in e else SERIES[1]
        ax.barh(i, m, color=color, height=0.62, zorder=2)
        ax.scatter(vals, [i] * len(vals), s=10, color=INK, zorder=3, linewidths=0)
        bb, model = describe(e)
        ax.text(max(vals) + 0.008, i, f"{m:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels([f"{describe(e)[1]}  [{e}]" for e in exps], fontsize=8)
    ax.invert_yaxis()
    t = next(iter(runs.values()))[0]["test"]
    prev = t["n_positive"] / t["n"]
    ax.axvline(prev, color=INK_2, ls="--", lw=1, zorder=1)
    ax.text(prev + 0.004, len(exps) - 0.4, "prevalence", fontsize=7, color=INK_2)
    ax.set_xlim(0.25, 0.9)
    ax.set_xlabel("Test AP (bar = mean over seeds, dots = seeds)")
    ax.set_title("Comedy classification: all models, test split", pad=40)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [GREY, *BACKBONE_COLOR.values()]]
    ax.legend(handles, ["baseline", "ViT + DistilBERT (vd)", "SigLIP 2 (sg)", "SigLIP 2 image + DistilBERT (sd)"],
              loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=7, handlelength=1)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_pr(runs, selected, path):
    picks = [("tfidf_logreg", GREY), ("vd_concat", SERIES[0]), ("sg_image", SERIES[1]), ("sd_concat", SERIES[2])]
    if selected not in dict(picks):
        picks.append((selected, SERIES[3]))
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for e, color in picks:
        if e not in runs:
            continue
        y, s = scores(sorted(runs[e], key=lambda r: r["seed"])[0], "test")
        p, r, _ = precision_recall_curve(y, s)
        bb, model = describe(e)
        label = model if bb == "—" else f"{bb} · {model}"
        ax.plot(r, p, color=color, lw=2, label=f"{label} (AP {average_precision_score(y, s):.3f})")
    ax.axhline(y.mean(), color=INK_2, ls="--", lw=1, label=f"prevalence ({y.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision-recall, test split (seed 0)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_learning_curves(runs, path):
    panels = [e for e in ("sd_concat", "sd_gmu", "sd_xattn") if e in runs]
    lora = [e for e in runs if "lora" in e]
    fig, axes = plt.subplots(1, len(panels) + bool(lora), figsize=(3.2 * (len(panels) + bool(lora)), 2.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, panels):
        r = next(r for r in runs[e] if r["seed"] == 0)
        h = r["history"]
        ax.plot([x["epoch"] for x in h], [x["dev_metric"] for x in h], color=SERIES[2], lw=2)
        ax.axvline(r["best_epoch"], color=INK_2, ls=":", lw=1)
        ax.set(title=f"{describe(e)[1]}\nlr {r['lr']:.0e}", xlabel="epoch")
    if lora:
        ax = axes[-1]
        for i, r in enumerate(sorted(runs[lora[0]], key=lambda r: r["seed"])):
            h = r["history"]
            ax.plot([x["epoch"] for x in h], [x["dev_ap"] for x in h], color=SERIES[6], lw=2,
                    alpha=1 - 0.25 * i, marker="o", ms=4, label=f"seed {r['seed']}")
        ax.set(title="LoRA fine-tuning\n(epoch 0 = frozen start)", xlabel="epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("dev average precision")
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle("Training curves (dotted line = early-stopping epoch)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration(runs, selected, path):
    """Raw scores are inflated by pos_weight; Platt scaling fit on dev fixes that."""
    r = sorted(runs[selected], key=lambda r: r["seed"])[0]
    yd, sd = scores(r, "dev")
    yt, st = scores(r, "test")
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))  # noqa: E731
    platt = LogisticRegression(C=1e6).fit(logit(sd)[:, None], yd)
    st_cal = platt.predict_proba(logit(st)[:, None])[:, 1]
    out = {"raw_ece": expected_calibration_error(yt, st), "platt_ece": expected_calibration_error(yt, st_cal),
           "raw_brier": float(np.mean((st - yt) ** 2)), "platt_brier": float(np.mean((st_cal - yt) ** 2))}

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    edges = np.linspace(0, 1, 11)
    for s, color, label in [(st, SERIES[1], f"raw (ECE {out['raw_ece']:.3f})"),
                            (st_cal, SERIES[0], f"Platt-scaled on dev (ECE {out['platt_ece']:.3f})")]:
        ids = np.clip(np.digitize(s, edges[1:-1]), 0, 9)
        xs = [s[ids == b].mean() for b in range(10) if (ids == b).sum() >= 5]
        ys = [yt[ids == b].mean() for b in range(10) if (ids == b).sum() >= 5]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, label=label)
    ax.plot([0, 1], [0, 1], color=INK_2, ls="--", lw=1, label="perfect calibration")
    ax.set(xlabel="mean predicted score", ylabel="observed Comedy rate", xlim=(0, 1), ylim=(0, 1),
           title=f"Reliability, test split\n{selected} (seed 0)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------- multilabel
def multilabel_section(runs):
    if not runs:
        return [], {}
    exps = sorted(runs, key=order_key)
    lines = [
        "## Version 2: multilabel genres",
        "",
        f"{len(runs[exps[0]][0]['genres'])} genres with ≥ 50 training movies (News dropped). "
        "One sigmoid output per genre, BCE loss; per-genre thresholds = max-F1 on dev. "
        "Learning rate selected on dev macro AP. Mean ± std over seeds.",
        "",
        "| Backbone | Model | Inputs | Seeds | Dev macro AP | Test macro AP | Test micro AP | Test micro F1 | Test macro F1 | Tags predicted / true |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    table = {}
    for e in exps:
        rs = runs[e]
        bb, model = describe(e)
        get = lambda split, k: [r[split][k] for r in rs]  # noqa: E731
        lines.append("| " + " | ".join([
            bb, model, rs[0]["inputs"], str(len(rs)), fmt(get("dev", "macro_ap")), fmt(get("test", "macro_ap")),
            fmt(get("test", "micro_ap")), fmt(get("test", "micro_f1")), fmt(get("test", "macro_f1")),
            f"{np.mean(get('test', 'mean_predicted_tags')):.2f} / {rs[0]['test']['mean_true_tags']:.2f}"]) + " |")
        table[e] = {f"test_{k}": dict(zip(("mean", "std"), mean_std(get("test", k))))
                    for k in ("macro_ap", "micro_ap", "micro_f1", "macro_f1")}
        genres = rs[0]["genres"]
        table[e]["per_genre_test_ap"] = {g: float(np.mean([r["test"]["per_genre"][g]["average_precision"] for r in rs]))
                                         for g in genres}
    return lines, table


def fig_per_genre(table, runs, path):
    fusion = [e for e in runs if e.startswith("sd_")]
    best_fusion = max(fusion, key=lambda e: np.mean([r["dev"]["macro_ap"] for r in runs[e]]))
    trio = [e for e in ("vd_text", "sg_image", best_fusion) if e in table]
    if len(trio) < 2:
        return []
    genres = list(table[trio[-1]]["per_genre_test_ap"])
    support = {g: runs[trio[-1]][0]["test"]["per_genre"][g]["support"] for g in genres}
    genres.sort(key=lambda g: table[trio[-1]]["per_genre_test_ap"][g])
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(genres) + 1.3))
    y = np.arange(len(genres))
    for e, color, off in zip(trio, [SERIES[0], SERIES[1], SERIES[2]], (-0.22, 0, 0.22)):
        vals = [table[e]["per_genre_test_ap"][g] for g in genres]
        ax.scatter(vals, y + off, s=26, color=color, zorder=3, label=f"{describe(e)[0]} · {describe(e)[1]}",
                   edgecolors=SURFACE, linewidths=1)
    for i, g in enumerate(genres):
        vals = [table[e]["per_genre_test_ap"][g] for e in trio]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=3, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} (n={support[g]})" for g in genres], fontsize=8)
    ax.set(xlabel="Test average precision (mean over seeds)", xlim=(0, 1),
           title="Per-genre AP: best plot-only vs best poster-only vs fusion")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1)
    ax.set_title(ax.get_title(), pad=48)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    gains = {g: table[trio[-1]]["per_genre_test_ap"][g] - max(table[e]["per_genre_test_ap"][g] for e in trio[:-1])
             for g in genres}
    img_vs_txt = {g: table["sg_image"]["per_genre_test_ap"][g] - table["vd_text"]["per_genre_test_ap"][g]
                  for g in genres} if {"sg_image", "vd_text"} <= set(table) else {}
    return gains, img_vs_txt


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    binary = load_runs("binary")
    lines = ["# Results", "", "Generated by `python -m src.report` from `artifacts/runs/`.", ""]
    b_lines, b_table, selected = binary_section(binary)
    lines += b_lines
    fig_binary_bars(binary, FIGURES_DIR / "binary_test_ap.png")
    fig_pr(binary, selected, FIGURES_DIR / "pr_curves.png")
    fig_learning_curves(binary, FIGURES_DIR / "learning_curves.png")
    cal = calibration(binary, selected, FIGURES_DIR / "reliability.png")
    lines += ["", f"Selected on dev: **{selected}** ({' · '.join(describe(selected))}).", "",
              f"Calibration of the selected model (seed 0, test): ECE {cal['raw_ece']:.3f} raw → "
              f"{cal['platt_ece']:.3f} after Platt scaling fit on dev; Brier {cal['raw_brier']:.3f} → "
              f"{cal['platt_brier']:.3f}. Training up-weights positives (pos_weight), which can push raw "
              "scores high; check the reliability plot before reading scores as probabilities.", "",
              "![All models](figures/binary_test_ap.png)", "", "![PR curves](figures/pr_curves.png)", "",
              "![Training curves](figures/learning_curves.png)", "", "![Reliability](figures/reliability.png)", ""]

    lat_path = REPORTS_DIR / "latency.json"
    if lat_path.exists():
        lat = json.loads(lat_path.read_text())
        lines += ["## Inference latency", "",
                  f"End to end (poster decode + preprocessing + both encoders + head), batch 1, warm, "
                  f"{lat['n_movies']} test movies, Apple M4 ({lat['hardware']}). `python -m src.benchmark`. "
                  "Not a serving benchmark (no API, no concurrency).", "",
                  "| Run | Params | CPU p50 / p95 (ms) | MPS p50 / p95 (ms) | max abs(live − cached score) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for run, d in lat["runs"].items():
            cells = [f"{d[k]['p50_ms']:.0f} / {d[k]['p95_ms']:.0f}" if k in d else "—" for k in ("cpu", "mps")]
            any_d = next(iter(d.values()))
            lines.append(f"| `{run}` | {any_d['params'] / 1e6:.0f}M | {cells[0]} | {cells[1]} | "
                         f"{any_d['max_abs_diff_vs_cached']:.0e} |")
        lines += ["", "Live scores are rounded to 4 decimals (5e-5 floor). The LoRA run's stored scores came "
                  "from an fp16 pixel cache used for training speed, hence its ~1e-3 gap.", ""]

    multi = load_runs("multilabel")
    m_lines, m_table = multilabel_section(multi)
    if m_lines:
        lines += m_lines
        res = fig_per_genre(m_table, multi, FIGURES_DIR / "multilabel_per_genre.png")
        if res:
            gains, img_vs_txt = res
            top = sorted(img_vs_txt.items(), key=lambda kv: kv[1])
            lines += ["", "![Per-genre AP](figures/multilabel_per_genre.png)", "",
                      "Poster beats plot most on: " + ", ".join(f"{g} (+{d:.2f})" for g, d in top[::-1][:4]) + ". "
                      "Plot beats poster most on: " + ", ".join(f"{g} ({d:+.2f})" for g, d in top[:4]) + ".", "",
                      f"Fusion beats the better single modality on {sum(v > 0 for v in gains.values())} of "
                      f"{len(gains)} genres.", ""]

    (REPORTS_DIR / "results.md").write_text("\n".join(lines) + "\n")
    (REPORTS_DIR / "results.json").write_text(json.dumps(
        {"selected_binary": selected, "binary": b_table, "calibration": cal, "multilabel": m_table}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
