"""Dataset loading, label construction, and validation.

Inputs (X) are only the poster file and the plot text. Genre columns are used
to build targets (y) and never reach the encoders.
"""
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import KAGGLE_DATASET, TARGET_GENRE

# A few posters are ~100 MP; they come from the dataset, not user uploads.
Image.MAX_IMAGE_PIXELS = 200_000_000


def get_data_root() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET)) / "tinymmimdb"


def parse_genres(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        raise ValueError(f"Missing genre annotation: {value!r}")
    return [tag.strip() for tag in str(value).split(" - ") if tag.strip()]


def load_poster(path, size: int = 448) -> Image.Image:
    """Open a poster as RGB. JPEG draft mode decodes at reduced resolution,
    which is much faster for huge files; the ViT processor resizes to 224 anyway."""
    im = Image.open(path)
    im.draft("RGB", (size, size))
    return im.convert("RGB")


def load_dataset(root: Path | None = None) -> pd.DataFrame:
    """Return one row per movie with model inputs, targets, and the supplied split."""
    root = root or get_data_root()
    raw = pd.read_csv(root / "data.csv")
    df = pd.DataFrame(
        {
            "movie_id": raw["image_path"].str.replace(".jpeg", "", regex=False),
            "title": raw["title"],  # display only, not a model input
            "image_file": [str(root / "images" / p) for p in raw["image_path"]],
            "plot": raw["plot outline"].astype(str),
            "genres": raw["genre"].apply(parse_genres),
            "split": raw["split"],
        }
    )
    df["label"] = df["genres"].apply(lambda tags: int(TARGET_GENRE in tags))
    return df


def validate(df: pd.DataFrame, check_decode: bool = True) -> dict:
    """Data-quality checks run before training. Returns a JSON-able report."""
    report = {
        "rows": len(df),
        "split_counts": df["split"].value_counts().to_dict(),
        "positives_by_split": df.groupby("split")["label"].sum().to_dict(),
        "duplicate_movie_ids": int(df["movie_id"].duplicated().sum()),
        "empty_plots": int((df["plot"].str.strip() == "").sum()),
        "missing_images": int(sum(not Path(p).exists() for p in df["image_file"])),
    }
    dup_plots = df[df["plot"].duplicated(keep=False)]
    report["duplicate_plots_crossing_splits"] = int(
        (dup_plots.groupby("plot")["split"].nunique() > 1).sum()
    )
    if check_decode:
        bad = []
        for path in df["image_file"]:
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                bad.append(path)
        report["undecodable_images"] = len(bad)
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(validate(load_dataset()), indent=2))
