# Progress log

Scope: the ML pipeline only. Comedy (binary) and 23-genre multilabel classification, comparing encoders, fusion architectures, and frozen vs LoRA-adapted encoders. The README also describes an API, a Streamlit demo, a SQLite review queue, and Docker. Those are out of scope for now.

## ML components

| # | Component | What it does | Code | Status |
| --- | --- | --- | --- | --- |
| 1 | Data loading & validation | Load CSV, resolve posters, check missing/corrupt images, duplicates, splits | `src/data.py` | Done |
| 2 | Labels | Comedy membership (binary); multi-hot over 23 genres with ≥ 50 train movies | `src/data.py`, `src/train.py` | Done |
| 3 | Encoders | DistilBERT, SigLIP 2 text, ViT-in21k, SigLIP 2 vision behind one interface | `src/encoders.py` | Done |
| 4 | Embedding cache | Pooled + token-level outputs for both encoder families, fp16 tokens | `src/embed.py` | Done |
| 5 | Concat head | Per-modality LayerNorm → concat → MLP | `src/model.py` | Done |
| 6 | Gated fusion (GMU) | Learned per-unit gate between plot and poster | `src/model.py` | Done |
| 7 | Fusion transformer | [FUSE] + plot tokens + pooled poster patches, written-out masked multi-head attention | `src/model.py` | Done |
| 8 | Training loop | BCE (+pos_weight for binary), AdamW, early stopping, dev LR search, 3 seeds | `src/train.py` | Done |
| 9 | LoRA fine-tuning | Hand-written LoRA on q/v of both encoders, LP-FT init, cosine schedule | `src/lora.py`, `src/finetune.py` | Done |
| 10 | Baselines | Majority, TF-IDF + LR (C on dev), SigLIP 2 zero-shot | `src/baselines.py`, `src/zeroshot.py` | Done |
| 11 | Metrics | AP, ROC AUC, F1/P/R at dev threshold, Brier, ECE; macro/micro AP & F1, per-genre | `src/evaluate.py` | Done |
| 12 | Reporting | Results tables + figures (bars, PR, learning curves, reliability, per-genre) | `src/report.py` | Done |
| 13 | Error analysis | Error rates by co-genre, confident FP/FN, modality reliance (GMU gate, attention share) | `src/analysis.py` | Done |
| 14 | Inference + latency | Raw poster + plot → score for any run; CPU/MPS p50/p95; live-vs-cached check | `src/predict.py`, `src/benchmark.py` | Done |
| 15 | Tests | Labels, pooling, head shapes, padding mask, patch pooling, LoRA identity, metrics | `tests/test_core.py` | Done (25) |
| 16 | Withheld-tag recovery / synthetic corruption | README "catalog-assistance" evaluation | — | Next |
| 17 | Multilabel LoRA, modality dropout | Fine-tune for 23 genres; robustness to a missing poster | — | Optional |

## Log

### 2026-09-30: setup, baselines, first full results
- Environment: Python 3.12 venv via `uv`; direct deps pinned in `requirements.txt`, full lock in `requirements-lock.txt`. Installed wheels only, with versions released at least 7 days earlier (`--exclude-newer 2026-09-23`). Added `torchvision` (needed by the transformers 5 image processors) and `pytest` the same way.
- Data checks: 4,285 rows; train 2,142 / dev 1,071 / test 1,072; Comedy 734 / 378 / 344. 0 missing or undecodable posters. 1 duplicated plot crosses train/test ("Treasure Island", different genres).
- First version (ViT-in21k + DistilBERT, concat head, lr 1e-3): fusion test AP 0.710 vs poster 0.655 vs plot 0.619. Superseded by the LR-searched runs below; numbers are in the same range.

### 2026-10-01: modern encoders, fusion architectures, LoRA, multilabel
**Encoders.** Added SigLIP 2 base/16 (`google/siglip2-base-patch16-224`). Its 64-token text limit truncates 10 of 4,285 plots (0.2%); DistilBERT at 128 truncates none. Refactored encoders into per-modality parts so backbones can mix. Cache time on the M4 GPU: 117 s (vd), 170 s (sg). Token caches: 1.9 GB + 1.7 GB, fp16, gitignored.

**Fusion transformer speed.** At first a step took 200 ms because there were 196 poster patch tokens per movie. Average-pooling the patches 2×2 (196 → 49 tokens) inside the head cut that to 65 ms. Text tokens keep full resolution.

**Binary results** (test AP, mean of 3 seeds; LR and model chosen on dev; full table in `reports/results.md`):

| | concat | GMU | fusion transformer |
| --- | ---: | ---: | ---: |
| ViT + DistilBERT | 0.704 | 0.708 | 0.681 |
| SigLIP 2 | 0.763 | 0.770 | 0.761 |
| SigLIP 2 image + DistilBERT | 0.766 | 0.772 | 0.784 |
| + LoRA (concat) | **0.775** (selected, dev 0.823) | | |

Single inputs: SigLIP poster 0.764, ViT poster 0.658, DistilBERT plot 0.615, SigLIP plot 0.569. Zero-shot SigLIP poster 0.557. TF-IDF 0.489.

**LoRA** (r=8, α=16, q/v of 36 attention layers; 0.84M of 160M params trainable):
- The epoch-0 dev AP matched the frozen head exactly (0.8184 vs 0.818; 0.8221 vs 0.822), which confirms the LoRA B=0 identity and that the training pixel cache is consistent.
- Gains: +0.005 dev AP and +0.009 test AP, the same direction on all 3 seeds.
- Cost: 180–245 s per training epoch on the M4 GPU, about 20 min per seed.
- Training loss keeps falling (0.46 → 0.25) while dev AP plateaus after 1–2 epochs.

**Multilabel** (23 genres, test macro AP):
- Mixed concat 0.709 vs DistilBERT plot 0.612 vs SigLIP poster 0.574. The fusion transformer reached 0.698.
- Fusion beats the better single input on 20 of 23 genres.
- ViT-in21k poster-only reached only 0.376. That holds across all 3 LRs, and mean-pooling the patches instead of `[CLS]` gave only 0.412 on dev (ad-hoc probe, seed 0), so the encoder itself is the weak point.

**Analysis** (dev split, selected model):
- False alarms: Musical 72%, Music 44%, Family 39%, Animation 31%, vs 17% for non-comedies overall.
- Misses: Comedy + Horror 46%, + Thriller 44%, + Action 34%.
- GMU plot gate 0.33; fusion transformer [FUSE] attention 72% on poster tokens.

**Calibration:** the selected model's ECE is 0.047 on raw test scores, and Platt scaling on dev doesn't improve it. The frozen concat head had ECE 0.091 → 0.042 after Platt.

**Latency** (batch 1, warm, 50 test movies, M4): CPU p50 65–72 ms, GPU p50 33–35 ms for every architecture. Live-vs-cached score difference is ≤ 6e-5 for frozen runs. For the LoRA run it is 1.1e-3, because training used an fp16 pixel cache.

### Observations / open questions
- Dev and test disagree on the ranking of the three fusion heads (dev spread 0.007; test spread 0.018). Treat the head choice as unresolved. With ~1k dev movies, differences under ~0.01 AP are not reliable\*.
- Dev AP runs above test AP for every model. Dev has more Comedy (0.353 vs 0.321), and AP's floor equals prevalence, so part of the gap is expected\*.
- The fusion transformer overfits by epoch ~4 at every LR tried. More data, stronger regularization, or modality dropout might change the ranking\*.
- LoRA was only run with the concat head and one hyperparameter setting (r=8, lr 2e-4).
- \*Inference, not tested.

## Reproduce

See the README Quickstart. Total compute on an Apple M4 (16 GB): about 5 min of caching, about 15 min for all frozen-head runs, and about 60 min for LoRA (3 seeds).
