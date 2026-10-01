"""SigLIP 2 zero-shot Comedy scoring: no training at all.

A poster's score is SigLIP's sigmoid image-text match against comedy prompts
(averaged over templates). Uses cached poster embeddings.

    python -m src.zeroshot
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from src.config import CHECKPOINTS, SPLITS, features_path
from src.data import load_dataset
from src.encoders import SigLIPText
from src.evaluate import binary_metrics, pick_threshold, save_run

TEMPLATES = [
    "a poster of a comedy movie.",
    "a movie poster for a funny comedy film.",
    "a comedy film poster.",
    "the poster of a hilarious comedy.",
]


@torch.no_grad()
def main():
    text_encoder = SigLIPText().eval()
    text_emb, _ = text_encoder(**text_encoder.tokenize(TEMPLATES))
    text_emb = F.normalize(text_emb, dim=-1)
    # Learned temperature and bias of the sigmoid contrastive loss
    model = AutoModel.from_pretrained(CHECKPOINTS["siglip2"])

    cache = torch.load(features_path("siglip2"))
    image_emb = F.normalize(cache["image"], dim=-1)
    logits = image_emb @ text_emb.T * model.logit_scale.exp() + model.logit_bias
    scores = torch.sigmoid(logits).mean(1).numpy()

    df = load_dataset().set_index("movie_id").loc[cache["movie_id"]]
    sel = {s: df["split"].to_numpy() == s for s in SPLITS}
    y = {s: df["label"].to_numpy()[sel[s]] for s in SPLITS}
    s = {k: scores[sel[k]] for k in SPLITS}
    threshold = pick_threshold(y["dev"], s["dev"])
    test = binary_metrics(y["test"], s["test"], threshold)
    save_run("binary", "sg_zeroshot", 0,
             {"backbone": "siglip2", "inputs": "poster", "trainable_params": 0, "templates": TEMPLATES},
             binary_metrics(y["dev"], s["dev"], threshold), test,
             {"dev_scores": s["dev"], "dev_labels": y["dev"], "test_scores": s["test"], "test_labels": y["test"]})
    print(f"sg_zeroshot  test AP {test['average_precision']:.3f}  AUC {test['roc_auc']:.3f}  F1 {test['f1']:.3f}")


if __name__ == "__main__":
    main()
