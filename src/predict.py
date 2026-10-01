"""End-to-end inference: raw poster + plot -> Comedy score through both transformers.

    python -m src.predict --movie-id 0077598
    python -m src.predict --poster path/to.jpg --plot "Two friends ..." --run binary/sd_lora_concat/seed0

Loads any saved run: frozen-encoder heads (head.pt) or LoRA fine-tuned models (model.pt).
"""
import argparse
import time

import torch

from src.config import RUNS_DIR
from src.data import load_poster
from src.encoders import load_backbone
from src.lora import apply_lora
from src.model import MultimodalGenreClassifier, build_head


def load_model(run: str, device="cpu"):
    run_dir = RUNS_DIR / run
    if (run_dir / "model.pt").exists():  # LoRA fine-tuned
        ckpt = torch.load(run_dir / "model.pt")
        head_kind, head_state = ckpt["head_kind"], ckpt["head"]
    else:
        ckpt = torch.load(run_dir / "head.pt")
        head_kind, head_state = ckpt["head"], ckpt["state_dict"]
    head = build_head(head_kind, ckpt["modalities"], num_labels=ckpt.get("num_labels", 1),
                      **ckpt["head_config"])
    head.load_state_dict(head_state)
    backbone = load_backbone(ckpt["backbone"], ckpt["modalities"])
    model = MultimodalGenreClassifier(backbone, head)
    if "lora" in ckpt:
        cfg = ckpt["lora_config"]
        apply_lora(model.backbone, cfg["lora_targets"], cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"])
        model.load_state_dict(ckpt["lora"], strict=False)
    return model.eval().to(device), ckpt


class ComedyPredictor:
    def __init__(self, run: str = "binary/sd_lora_concat/seed0", device: str = "cpu"):
        self.model, ckpt = load_model(run, device)
        self.threshold = ckpt["thresholds"][0]
        self.modalities = ckpt["modalities"]
        self.device = device

    @torch.no_grad()
    def __call__(self, poster_path: str, plot: str) -> dict:
        start = time.perf_counter()
        bb = self.model.backbone
        inputs = {}
        if "text" in self.modalities:
            inputs.update({k: v.to(self.device) for k, v in bb.tokenize([plot]).items()})
        if "image" in self.modalities:
            inputs["pixel_values"] = bb.preprocess([load_poster(poster_path)]).to(self.device)
        score = torch.sigmoid(self.model(**inputs)).item()
        if self.device == "mps":
            torch.mps.synchronize()
        return {"comedy_score": round(score, 4), "is_comedy": score >= self.threshold,
                "threshold": round(self.threshold, 4),
                "latency_ms": round((time.perf_counter() - start) * 1000, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poster")
    parser.add_argument("--plot")
    parser.add_argument("--movie-id")
    parser.add_argument("--run", default="binary/sd_lora_concat/seed0")
    args = parser.parse_args()

    if args.movie_id:
        from src.data import load_dataset
        row = load_dataset().set_index("movie_id").loc[args.movie_id]
        args.poster, args.plot = row["image_file"], row["plot"]
        print(f"{row['title']} | reference genres: {', '.join(row['genres'])}")
    print(ComedyPredictor(args.run)(args.poster, args.plot))


if __name__ == "__main__":
    main()
