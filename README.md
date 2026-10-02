# Movie Catalog Tagging Assistant

A multimodal machine learning project that uses movie posters and plot outlines to suggest genre tags and identify catalog entries that deserve human review. Experimental notebook with help of AI.

**Status:** Versions 1 (Comedy vs. not Comedy) and 2 (23-genre multilabel) are trained and evaluated. The comparison covers two encoder families (ViT + DistilBERT, SigLIP 2) and three fusion heads (concat MLP, gated multimodal unit, token-level fusion transformer), plus LoRA fine-tuning. See [Results](#results), [reports/results.md](reports/results.md), and [PROGRESS.md](PROGRESS.md). The API and review interface are not built. No business-impact results are claimed.

## Business problem

Consider a hypothetical movie-discovery website that receives movie metadata from several content providers. Some listings arrive with incomplete genre tags, while others use inconsistent or unsuitable tags.

A missing Comedy tag can keep a relevant movie out of a Comedy filter. An unsuitable tag can place a movie in a browsing category where viewers would not expect it. Catalog teams must investigate these issues, while users may need more effort to find something they want to watch.

The business hypothesis is that more useful genre metadata could improve discovery, user satisfaction, and engagement while reducing manual catalog work. Revenue and retention are possible downstream outcomes; this dataset cannot establish those effects.

This project explores whether a model can assist the catalog team by learning genre signals from two information sources:

- **Poster:** visual style, composition, characters, and other imagery.
- **Plot outline:** story content, themes, and events.

The first version answers one question: **Should this movie carry a Comedy tag?** A later version predicts several overlapping genres.

## Product use cases

| Use case | Example | Proposed behavior |
| --- | --- | --- |
| New movie onboarding | A provider supplies a poster and plot but no genre tags | Suggest likely tags for a curator |
| Missing-tag review | A romantic comedy is listed only as Romance | Suggest reviewing the missing Comedy tag |
| Existing-tag review | A movie has a Comedy tag but the model gives it a low Comedy score | Flag the disagreement for review |
| Ongoing catalog maintenance | A curator approves or rejects a suggestion | Save the review for auditing and a future training snapshot |

These examples are hypothetical. The dataset is not a verified collection of incorrectly tagged movies. A disagreement between model and catalog is a reason to investigate, not proof that either is wrong.

The MVP supports suggestions and human review. Changes to a catalog are explicit curator decisions.

## Learning objectives

- Build a reproducible PyTorch classification pipeline.
- Combine image and text representations from pretrained transformers.
- Compare text-only, image-only, and multimodal models.
- Extend binary classification to multilabel classification.
- Separate model scores from application decisions.
- Package inference in a small API and review interface.
- Measure predictive quality and inference latency.

## Inputs, labels, and application metadata

| Field | Role |
| --- | --- |
| Movie poster | Image input |
| Plot outline | Primary text input |
| Movie title | Display metadata initially; optional additional text in a later experiment |
| Dataset genre list | Reference labels used for training and evaluation |
| Dataset genre ID | Encoded label information; never an input feature |
| Existing website tags | Optional application metadata used after prediction to identify disagreements |
| Movie ID or image path | Joining, deduplication, and display; never a predictive feature |

**Genres are targets, not model inputs.** The predictor receives the poster and plot without the answer. Existing website tags can be compared with predictions in a separate application step.

Poster and plot are two modalities. Adding a movie title adds more text; it does not introduce a third modality.

The title is excluded from the first model to keep the experiment focused on poster and plot content. A later title-input experiment can measure whether it changes performance. Title removal does not guarantee that a pretrained encoder has never encountered the movie.

## Dataset

Source: [Tiny MM-IMDb on Kaggle](https://www.kaggle.com/datasets/gabrieltardochi/tiny-mm-imdb).

The inspected CSV has 4,285 records, with poster paths, titles, plot outlines, genre annotations, and a supplied split column. Kaggle's file listing includes poster JPEGs. A complete image-integrity audit is an implementation step.

| Split | Records |
| --- | ---: |
| Train | 2,142 |
| Development | 1,071 |
| Test | 1,072 |
| Total | 4,285 |

The CSV fields are:

| Field | Meaning |
| --- | --- |
| image_path | Poster location |
| title | Movie title |
| plot outline | Plot text |
| genre | One or more genre tags |
| genre_id | Encoding associated with the genre combination |
| split | Supplied train, development, or test assignment |

The inspected Tiny subset contains 24 individual genre tags across 689 combinations. These are properties of this Kaggle subset, not claims about every MM-IMDb release. [Kaggle source](https://www.kaggle.com/datasets/gabrieltardochi/tiny-mm-imdb)

The original MM-IMDb research task also uses posters and plots for multilabel movie-genre prediction. [Original paper](https://arxiv.org/abs/1702.01992)

### Access

Install the small set of packages needed to inspect the data:

~~~bash
pip install kagglehub pandas pillow
~~~

Download the dataset:

~~~python
from pathlib import Path
import kagglehub

data_root = Path(
    kagglehub.dataset_download("gabrieltardochi/tiny-mm-imdb")
)
print(data_root)
~~~

KaggleHub downloads to a local cache outside Kaggle and attaches resources within Kaggle notebooks. [Official KaggleHub documentation](https://github.com/Kaggle/kagglehub)

The expected CSV is under `tinymmimdb/data.csv`, with poster files under `tinymmimdb/images/`. Resolve the image_path values against the extracted data location and verify every file before training.

### Quickstart

~~~bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python -m src.data                               # data-quality report
.venv/bin/python -m src.embed --backbone vit_distilbert    # cache pooled + token outputs (~2 min, M4)
.venv/bin/python -m src.embed --backbone siglip2           # (~3 min, M4)
.venv/bin/python -m src.baselines                          # majority + TF-IDF/logistic regression
.venv/bin/python -m src.zeroshot                           # SigLIP 2 zero-shot poster baseline
.venv/bin/python -m src.train --task binary                # 14 head experiments x 3 seeds, LR chosen on dev
.venv/bin/python -m src.finetune --backbone siglip2_distilbert --head concat   # LoRA, ~20 min/seed (M4)
.venv/bin/python -m src.train --task multilabel --experiments vd_text vd_image sg_text sg_image sd_concat sd_gmu sd_xattn
.venv/bin/python -m src.benchmark --runs binary/sd_concat/seed0 binary/sd_lora_concat/seed0 --devices cpu mps
.venv/bin/python -m src.report                             # reports/results.md + reports/figures/
.venv/bin/python -m src.analysis --run binary/sd_lora_concat/seed0   # reports/error_analysis.md
.venv/bin/python -m src.predict --movie-id 0077598         # end to end through both transformers
.venv/bin/python -m pytest -q
~~~

For a walkthrough, start with `notebooks/01_multimodal_demo.ipynb`. It is a self-contained prototype on a small subset that was later refactored into `src/`.

## Version 1: Comedy versus Not comedy

Use every usable movie record and derive a binary label from its genre list:

- **1:** Comedy is among the movie's genre tags.
- **0:** Comedy is absent from the movie's genre tags.

A movie can be both Comedy and Romance and still have a positive Comedy label. Other genres do not make it a negative example.

| Reference genres | Binary label |
| --- | ---: |
| Comedy | 1 |
| Comedy - Romance | 1 |
| Drama - Comedy | 1 |
| Horror - Thriller | 0 |
| Action - Adventure | 0 |

The inspected CSV contains 1,456 Comedy examples and 2,829 non-Comedy examples. These counts describe reference annotations, not an objective or exhaustive definition of humor. [Dataset](https://www.kaggle.com/datasets/gabrieltardochi/tiny-mm-imdb)

Example label preparation:

~~~python
import pandas as pd

df = pd.read_csv(data_root / "tinymmimdb" / "data.csv")

def parse_genres(value):
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Missing genre annotation; inspect this record.")
    return {tag.strip() for tag in str(value).split(" - ") if tag.strip()}

df["genre_labels"] = df["genre"].apply(parse_genres)
df["is_comedy"] = df["genre_labels"].apply(
    lambda tags: int("Comedy" in tags)
)

print(df[["title", "genre", "is_comedy", "split"]].head())
~~~

The genre columns are used to build y. They must never be concatenated with the plot to build X.

### Binary output

Use one output logit per movie. Train with `BCEWithLogitsLoss`, then apply sigmoid when computing prediction scores. Select a decision threshold using development data.

A score is a model estimate. It is not automatically a calibrated probability or a guarantee that a tag is appropriate.

## Version 2: Multilabel genre prediction

Movies can belong to several genres at once. Therefore, the extension is **multilabel classification**, rather than multiclass classification that forces one exclusive category. [Problem-type definitions](https://scikit-learn.org/stable/modules/multiclass.html)

For an illustrative ordered genre vocabulary:

~~~text
[Action, Comedy, Drama, Horror, Romance]
~~~

a movie tagged Comedy and Romance has this target:

~~~text
[0, 1, 0, 0, 1]
~~~

The model outputs one score per genre. Several tags can exceed their thresholds.

Implementation plan:

1. Build and save a genre vocabulary from training annotations or a predefined dataset schema.
2. Convert each genre list to a multi-hot vector.
3. Replace the one-output head with a K-output head.
4. Train with binary cross-entropy across the K outputs.
5. Apply sigmoid independently to each logit.
6. Begin with a shared threshold; consider per-genre thresholds only when development support is sufficient.

Do not use a softmax across genres: the tags are not mutually exclusive. PyTorch's binary cross-entropy loss supports the multilabel setup. [BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html)

Start with a few sufficiently represented genres if needed. Report rare or unsupported genres explicitly before expanding coverage. Avoid treating every genre combination as a separate class.

## Proposed model architecture

The initial model uses two pretrained transformer encoders and a small fusion classifier.

| Branch | Starting checkpoint | Representation |
| --- | --- | --- |
| Text | distilbert/distilbert-base-uncased | Mask-aware pooled plot representation |
| Image | google/vit-base-patch16-224-in21k | Poster classification-token representation |
| Fusion | Trainable MLP | Concatenated image and text vectors |
| Output | One logit initially | Comedy score after sigmoid |

Both chosen encoders have a hidden width of 768. Concatenation therefore gives a 1,536-dimensional vector, with a suggested classifier of 1536 → 256 → 1. [DistilBERT configuration](https://huggingface.co/distilbert/distilbert-base-uncased/blob/main/config.json), [ViT configuration](https://huggingface.co/google/vit-base-patch16-224-in21k/blob/main/config.json)

~~~mermaid
flowchart TD
    P["Movie poster"] --> V["Vision transformer"]
    T["Plot outline"] --> B["Text transformer"]
    V --> I["Image representation"]
    B --> X["Text representation"]
    I --> F["Concatenate and classify"]
    X --> F
    F --> S["Comedy score"]
~~~

This is a multimodal model using transformer encoders. In the initial architecture, fusion occurs after each modality is encoded. It does not perform joint attention between individual plot tokens and poster patches.

### Implemented architectures

Encoders (all 768-d, `src/encoders.py`):

| Backbone key | Text encoder | Image encoder | Why |
| --- | --- | --- | --- |
| `vit_distilbert` (vd) | DistilBERT, masked mean pool | ViT-B/16 ImageNet-21k, `[CLS]` | Separately pretrained unimodal encoders (original design) |
| `siglip2` (sg) | SigLIP 2 text tower | SigLIP 2 vision tower (attention-pool head) | Image–text contrastive pretraining (2025) |
| `siglip2_distilbert` (sd) | DistilBERT | SigLIP 2 vision tower | Strongest encoder per modality on dev |

Heads (`src/model.py`):

| Head | Input | Mechanism | Params |
| --- | --- | --- | ---: |
| `concat` | pooled vectors | per-modality LayerNorm → concat → MLP | 0.40M |
| `gmu` | pooled vectors | Gated Multimodal Unit: z·tanh(W_t x_t) + (1−z)·tanh(W_v x_v), z = σ(W_z[x_t; x_v]) (from the MM-IMDb paper) | 0.79M |
| `transformer` | token sequences | [FUSE] + plot tokens + 2×2-pooled poster patches, modality embeddings, 2 pre-norm self-attention blocks (written out, with padding masks), classify [FUSE] | 1.45M |

Fine-tuning (`src/lora.py`, `src/finetune.py`): LoRA (r = 8, α = 16), implemented in the repo rather than imported, on every attention query/value projection of both encoders (36 layers). It follows LP-FT: start from the head trained on frozen features. Because LoRA's B matrices start at zero, epoch 0 reproduces the frozen model exactly; the run logs this as a check. Trainable: 0.84M of 160M parameters (0.52%).

### Training progression

1. Freeze both encoders and train the fusion classifier.
2. Cache deterministic encoder representations to make head experiments inexpensive.
3. Compare single-modality and combined-input baselines.
4. If justified by development results, unfreeze selected final encoder blocks and fine-tune with a lower learning rate.
5. Consider a joint multimodal transformer or an attention-based fusion module as a later architecture experiment.

Frozen feature extraction and fine-tuning are different experiments. Report which pretrained weights were actually updated.

DistilBERT supports up to 512 text positions. Begin with a documented token limit and measure truncation. If important plot content is routinely removed, revisit the limit or use a documented chunking method. [Text-model configuration](https://huggingface.co/distilbert/distilbert-base-uncased/blob/main/config.json)

Use the model's image processor consistently and inspect whether preprocessing preserves useful poster content.

## Experiments

Use the same split and reference targets for every experiment.

| Experiment | Purpose |
| --- | --- |
| Majority-class baseline | Establish a trivial reference |
| TF-IDF plus logistic regression | Establish a cheap text baseline |
| Text transformer plus classifier | Measure plot-only performance |
| Vision transformer plus classifier | Measure poster-only performance |
| Both transformer representations plus classifier | Measure the contribution of combining modalities |
| Selective fine-tuning | Determine whether adapting the encoders justifies its cost |

Title input is an optional ablation after the main poster-and-plot comparison. Additional architecture variants should answer a specific question rather than expand the project indefinitely.

Multimodal improvement is an experimental question. The final system should use the model justified by validation results, cost, and the intended workflow.

## Evaluation

### Predictive quality

For the Comedy task, report:

- Comedy precision, recall, and F1.
- Average precision.
- ROC AUC as a supplementary ranking metric.
- Confusion matrix and evaluated sample counts.
- The operating threshold chosen on development data.

For multilabel prediction, report micro F1, macro F1, and per-genre precision/recall/F1 with support counts. Report how many tags are typically suggested. Do not use a single multiclass accuracy value as the complete evaluation.

Keep test data out of threshold selection and hyperparameter tuning. Repeat promising training configurations with a few seeds before treating small gains as decisive.

### Catalog-assistance quality

Two evaluations connect the classifier to the business scenario:

**Tag completion:** Hide the Comedy annotation from a held-out catalog view, predict from poster and plot, and compare the suggestion with the held-out reference. The predictor never receives the reference tags.

**Synthetic metadata review, optional:** Make a separate copy of held-out catalog metadata. Remove selected positive tags or insert selected incorrect tags using a documented seed and corruption rate. Keep the original annotations unchanged as the evaluation reference.

Measure how often the system recovers withheld tags, identifies injected errors, and incorrectly flags unmodified metadata. Report missing-tag and spurious-tag performance separately. Include the number of reviews needed per recovered tag or detected injected error.

Describe the second experiment as simulated metadata corruption. Its results do not establish the error rate in a real website's catalog.

### Application policy

Let s be the Comedy score and let t_low and t_high be thresholds chosen on development data.

| Catalog state | Model signal | Application suggestion |
| --- | --- | --- |
| Comedy absent | s at or above t_high | Review adding Comedy |
| Comedy present | s at or below t_low | Review the existing Comedy tag |
| Either | s between thresholds | No strong automated recommendation; review if needed |
| Tag and confident prediction agree | Agreement | No issue flagged |

The thresholds control workload and error tradeoffs. The classifier's ordinary binary benchmark can use its own stated decision threshold. Keep model evaluation and triage policy results distinguishable.

### Business outcomes requiring additional evidence

A production study could measure:

| Outcome | Possible measurement |
| --- | --- |
| Curator usefulness | Suggestion acceptance rate and time per reviewed item |
| Better genre discovery | Relevance judgments or success on genre-filtered searches |
| Engagement | Watch starts or meaningful interactions after genre browsing |
| Retention or revenue | A controlled online experiment with an appropriate follow-up window |

Offline classification scores alone cannot establish improved engagement, reduced labor, retention, or revenue.

## System design

The proposed system has an offline training workflow and an online catalog-review workflow.

~~~mermaid
flowchart TD
    D["Validated dataset"] --> T["Train and evaluate"]
    T --> A["Versioned model artifact"]
    U["Poster and plot"] --> API["Prediction API"]
    A --> API
    API --> S["Genre scores"]
    C["Existing catalog tags"] --> R["Review policy"]
    S --> R
    R --> Q["Curator review queue"]
    Q --> L["Decision log"]
    L -. "Curated future labels" .-> D
~~~

Existing catalog tags enter the review policy after inference. They are not part of the predictor's input.

### Planned components

| Component | Initial implementation |
| --- | --- |
| Data processing | pandas, Pillow, versioned manifest |
| Model training | PyTorch and Hugging Face Transformers |
| Evaluation | scikit-learn |
| Experiment records | Local run artifacts; MLflow if useful |
| Inference | FastAPI |
| Demo | Streamlit |
| Prediction and review storage | SQLite |
| Packaging | Environment lockfile and Docker |

A synchronous API and a database-backed review list are sufficient for the MVP.

For each prediction, record a prediction ID, movie identifier, model version, score, policy decision, and latency. Store curator decisions separately with timestamps and link them by prediction ID.

Reviewed decisions can enter a future curated training dataset. Retraining and model promotion are explicit steps.

Measure end-to-end latency including input decoding, preprocessing, both encoders, and the classifier. Report p50/p95 latency together with hardware, batch size, concurrency, and whether the service was warm. Cached-feature head timing is not the latency of a new movie request.

## Data quality and reproducibility

Before training:

- Confirm each poster path exists and the file decodes.
- Check missing plots, malformed genre strings, and split values.
- Audit duplicate movies, near-identical plots, and duplicate poster images.
- Preserve supplied splits for the primary evaluation; document any separate cleaned protocol.
- If the same movie crosses splits, report the supplied-split score as a benchmark result and use a separate split that keeps each movie's duplicates together for claims about generalization to unseen movies.
- Fit learned preprocessing and select supported labels using training data.
- Keep genre annotations and genre IDs out of all input-building functions.
- Log truncation, image preprocessing, and missing-data decisions.

For each run, save the dataset version or hashes, split assignment, genre vocabulary, preprocessing settings, encoder revisions, trainable layers, seed, hyperparameters, weights, and thresholds.

A movie held out from this project's supervised training may still have appeared in an encoder's pretraining data. Keep generalization claims within the evidence provided by the evaluation.

## Implementation roadmap

- [x] Load the dataset and validate image/text/label pairing.
- [x] Build the Comedy membership label using the supplied split.
- [x] Train the simple and single-modality baselines.
- [x] Train the multimodal classifier and compare results.
- [x] Select a threshold and perform error analysis.
- [ ] Package the selected model for inference. *(CLI predictor in `src/predict.py`; no API yet)*
- [ ] Build an upload form and catalog-review interface.
- [ ] Measure latency and demonstrate withheld-tag recovery. *(latency measured offline; tag recovery pending)*
- [ ] Add the optional synthetic metadata-error experiment.
- [x] Extend the output head to multilabel genre prediction.

Finish the binary end-to-end workflow before expanding the label space.

### Repository organization

| Location | Purpose |
| --- | --- |
| README.md | Project purpose, setup, design, and results |
| PROGRESS.md | Component checklist and dated progress log |
| notebooks/01_multimodal_demo.ipynb | Small end-to-end prototype on a data subset |
| src/config.py | Paths, checkpoints, backbones, constants |
| src/data.py | Dataset parsing, labels, validation, poster loading |
| src/encoders.py | DistilBERT / SigLIP text and ViT / SigLIP image encoders behind one interface |
| src/model.py | Concat, GMU, and fusion-transformer heads; end-to-end model |
| src/lora.py | LoRA layer and injection |
| src/embed.py | Frozen-encoder cache (pooled + token outputs) |
| src/experiments.py | Experiment registry and hyperparameters |
| src/train.py | Head training with dev LR search, binary and multilabel |
| src/finetune.py | LoRA fine-tuning end to end |
| src/baselines.py, src/zeroshot.py | Majority, TF-IDF, SigLIP 2 zero-shot |
| src/evaluate.py | Metrics: thresholds, AP/AUC/F1, Brier, ECE, multilabel |
| src/report.py, src/analysis.py | Results tables, figures, error analysis |
| src/benchmark.py, src/predict.py | Latency benchmark; end-to-end inference |
| reports/ | Generated results, figures, error analysis, latency |
| tests/ | Labels, pooling, heads, masking, LoRA, metrics |
| *Not yet built* | `app/api.py`, `app/demo.py` |

## Results

Full tables, figures, and per-run details: [reports/results.md](reports/results.md). Error analysis: [reports/error_analysis.md](reports/error_analysis.md).

### Version 1: Comedy vs. not Comedy

Supplied test split (n = 1,072, 344 Comedy). Everything was chosen on dev: the learning rate, the decision threshold, and the final model. Test is used only for reporting. Mean ± std over 3 seeds.

| Model | Trainable | Dev AP | Test AP | Test ROC AUC | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority class | 0 | 0.353 | 0.321 | 0.500 | 0.000 |
| TF-IDF + logistic regression (plot) | 8.6K | 0.542 | 0.489 | 0.681 | 0.534 |
| SigLIP 2 zero-shot (poster, no training) | 0 | 0.597 | 0.557 | 0.693 | 0.531 |
| ViT + DistilBERT, concat fusion | 0.40M | 0.757 | 0.704 ± 0.002 | 0.828 | 0.656 |
| SigLIP 2, poster only | 0.20M | 0.787 | 0.764 ± 0.004 | 0.853 | 0.674 |
| SigLIP 2 image + DistilBERT, concat fusion | 0.40M | 0.818 | 0.766 ± 0.002 | 0.868 | 0.706 |
| SigLIP 2 image + DistilBERT, GMU | 0.79M | 0.816 | 0.772 ± 0.001 | 0.871 | 0.701 |
| SigLIP 2 image + DistilBERT, fusion transformer | 1.45M | 0.811 | 0.784 ± 0.003 | 0.878 | 0.724 |
| **SigLIP 2 image + DistilBERT, LoRA + concat (selected)** | 0.84M | **0.823** | **0.775 ± 0.001** | 0.868 | 0.708 |

![All models](reports/figures/binary_test_ap.png)

What the comparison shows:

1. **The encoder matters most.** Swapping the ImageNet-21k ViT for the SigLIP 2 vision tower raised test AP by about 0.06 with the same head. SigLIP's own text tower was the weakest plot encoder, so the best backbone mixes SigLIP 2 for posters with DistilBERT for plots.
2. **Fusion helps, but the fusion head choice is within noise for Comedy.** On dev, concat, GMU, and the fusion transformer are within 0.007 AP. On test the transformer scores highest (0.784), but the selection rule uses dev. The token-level transformer overfits after about 4 epochs on 2,142 training movies.
3. **LoRA fine-tuning adds a little.** Dev AP rose 0.818 → 0.823 and test AP 0.766 → 0.775, consistently across 3 seeds. It costs 3–4 minutes per training epoch on an M4 GPU, vs about 1 second for a frozen head.
4. **The models rely mostly on the poster for Comedy.** The GMU gate weights the plot at 0.33, and the fusion transformer's [FUSE] token puts 72% of its attention on poster tokens.
5. **Errors follow visual style.** On dev, 72% of non-comedy Musicals and 39% of non-comedy Family films are flagged as Comedy. Comedies that also carry Thriller, Horror, or Action tags are missed most often.

### Version 2: 23-genre multilabel

| Model | Test macro AP | Test micro F1 |
| --- | ---: | ---: |
| DistilBERT plot only | 0.612 | 0.604 |
| SigLIP 2 poster only | 0.574 | 0.584 |
| **SigLIP 2 image + DistilBERT, concat fusion (selected on dev)** | **0.709** | **0.659** |
| SigLIP 2 image + DistilBERT, fusion transformer | 0.698 | 0.653 |

Across genres, fusion beats the better single input on 20 of 23 genres (+0.10 macro AP). Posters carry Animation (+0.43 AP over plot), Film-Noir, and Family. Plots carry Documentary (+0.35 over poster), Biography, Music, and History.

![Per-genre AP](reports/figures/multilabel_per_genre.png)

### Cost

End-to-end latency (decode + preprocess + both encoders + head), batch 1, warm, Apple M4: about 65–72 ms on CPU and 33–35 ms p50 on the M4 GPU for every architecture. The encoders dominate, so the head choice adds no meaningful cost. Details are in [reports/results.md](reports/results.md#inference-latency). This is not a serving benchmark.

Business-impact results should be reported only if a separate curator study or online experiment has been conducted.

## Scope and attribution

This is a portfolio prototype for genre prediction and catalog assistance. Genre annotations can be subjective or incomplete, and model scores do not establish a single correct interpretation of a movie.

Credit the Kaggle subset and original MM-IMDb research. Keep downloaded dataset assets out of the code repository; link to the source and review applicable terms before redistributing posters or plot text. Choose a license for original project code separately.

### References

- [Tiny MM-IMDb dataset](https://www.kaggle.com/datasets/gabrieltardochi/tiny-mm-imdb)
- [Original MM-IMDb paper: Gated Multimodal Units for Information Fusion](https://arxiv.org/abs/1702.01992)
- [KaggleHub](https://github.com/Kaggle/kagglehub)
- [DistilBERT checkpoint](https://huggingface.co/distilbert/distilbert-base-uncased)
- [Vision Transformer checkpoint](https://huggingface.co/google/vit-base-patch16-224-in21k)
- [PyTorch binary cross-entropy loss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html)
- [Multiclass and multilabel problem definitions](https://scikit-learn.org/stable/modules/multiclass.html)
