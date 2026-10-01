"""Experiment registry: which backbone, head, and inputs each named run uses.

Prefixes: vd = ViT + DistilBERT, sg = SigLIP 2 (both towers),
sd = SigLIP 2 image tower + DistilBERT (strongest encoder per modality).
"""
HEAD_CONFIGS = {
    "concat": {"hidden": 256, "dropout": 0.2},
    "gmu": {"hidden": 256, "dropout": 0.2},
    "transformer": {"d_model": 256, "heads": 4, "layers": 2, "dropout": 0.1, "image_pool": 2},
}

TRAIN_CONFIG = {
    "batch_size": 64,
    "weight_decay": 0.01,
    "max_epochs": 100,
    "patience": 10,  # early stopping on the dev selection metric
    "lr_grid": [1e-4, 3e-4, 1e-3],  # searched with seed 0, chosen on dev
}

EXPERIMENTS = {}
for short, backbone in [("vd", "vit_distilbert"), ("sg", "siglip2")]:
    EXPERIMENTS.update({
        f"{short}_text": dict(backbone=backbone, head="concat", modalities=("text",)),
        f"{short}_image": dict(backbone=backbone, head="concat", modalities=("image",)),
        f"{short}_concat": dict(backbone=backbone, head="concat", modalities=("text", "image")),
        f"{short}_gmu": dict(backbone=backbone, head="gmu", modalities=("text", "image")),
        f"{short}_xattn": dict(backbone=backbone, head="transformer", modalities=("text", "image")),
    })
# Mixed backbone: unimodal runs would duplicate vd_text / sg_image, so fusion heads only.
EXPERIMENTS.update({
    "sd_concat": dict(backbone="siglip2_distilbert", head="concat", modalities=("text", "image")),
    "sd_gmu": dict(backbone="siglip2_distilbert", head="gmu", modalities=("text", "image")),
    "sd_xattn": dict(backbone="siglip2_distilbert", head="transformer", modalities=("text", "image")),
})

DESCRIPTIONS = {
    "majority": "Majority class",
    "tfidf_logreg": "TF-IDF + logistic regression",
    "sg_zeroshot": "SigLIP 2 zero-shot (poster vs. prompt)",
    "text": "plot only, MLP",
    "image": "poster only, MLP",
    "concat": "late fusion: concat + MLP",
    "gmu": "gated fusion (GMU)",
    "xattn": "fusion transformer over tokens",
    "lora_concat": "LoRA fine-tuned encoders + concat MLP",
}
BACKBONE_NAMES = {"vd": "ViT-in21k + DistilBERT", "sg": "SigLIP 2",
                  "sd": "SigLIP 2 image + DistilBERT"}


def describe(experiment: str) -> tuple[str, str]:
    """(backbone label, model label) for tables."""
    prefix, _, rest = experiment.partition("_")
    if prefix in BACKBONE_NAMES and rest in DESCRIPTIONS:
        return BACKBONE_NAMES[prefix], DESCRIPTIONS[rest]
    return "—", DESCRIPTIONS.get(experiment, experiment)
