"""Paths, backbones, and task constants shared by every script."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_DATASET = "gabrieltardochi/tiny-mm-imdb"

# Pretrained checkpoints (all 768-d outputs)
CHECKPOINTS = {
    "distilbert": "distilbert/distilbert-base-uncased",
    "vit": "google/vit-base-patch16-224-in21k",  # ImageNet-21k supervised
    "siglip2": "google/siglip2-base-patch16-224",  # image-text sigmoid contrastive
}

# A backbone = (text encoder, image encoder).
BACKBONES = {
    "vit_distilbert": {"text": "distilbert", "image": "vit"},
    "siglip2": {"text": "siglip2", "image": "siglip2"},
    "siglip2_distilbert": {"text": "distilbert", "image": "siglip2"},  # best encoder per modality
}
EMBED_DIM = 768

# Kept for the original single-backbone scripts / notebook
TEXT_MODEL = CHECKPOINTS["distilbert"]
IMAGE_MODEL = CHECKPOINTS["vit"]
MAX_TOKENS = 128

TARGET_GENRE = "Comedy"
SPLITS = ("train", "dev", "test")
MIN_GENRE_TRAIN_SUPPORT = 50  # multilabel vocabulary: drops News (8 train movies)


def features_path(backbone: str) -> Path:
    """Pooled (one vector per movie per modality) embeddings."""
    return ARTIFACTS_DIR / f"features_{backbone}.pt"


def tokens_path(backbone: str) -> Path:
    """Token-level encoder outputs (fp16) for the fusion transformer."""
    return ARTIFACTS_DIR / f"tokens_{backbone}.pt"


def cache_source(backbone: str, modality: str) -> str:
    """Which embedded backbone's cache holds this modality's encoder outputs.
    Mixed backbones reuse caches instead of re-running the encoders."""
    encoder = BACKBONES[backbone][modality]
    for name in ("vit_distilbert", "siglip2"):
        if BACKBONES[name][modality] == encoder:
            return name
    raise KeyError(encoder)
