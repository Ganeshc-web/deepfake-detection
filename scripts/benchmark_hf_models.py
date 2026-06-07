"""Benchmark free HF deepfake models on local validation samples."""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

CANDIDATES = [
    "prithivMLmods/Deepfake-Detect-Siglip2",
    "prithivMLmods/deepfake-detector-model-v1",
    "capcheck/ai-image-detection",
    "dima806/deepfake_vs_real_image_detection",
    "prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0",
]


def benchmark_model(model_id: str, samples: dict[str, list[Path]]) -> tuple[int, int]:
    """Return (correct, total) for a model."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    fake_indices = [
        i for i, name in id2label.items() if "fake" in name or "ai" in name
    ]
    real_indices = [i for i, name in id2label.items() if "real" in name]

    correct = 0
    total = 0
    for cls, paths in samples.items():
        for path in paths:
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=1).squeeze()

            fake_prob = sum(probs[i].item() for i in fake_indices)
            real_prob = sum(probs[i].item() for i in real_indices)
            is_fake = fake_prob >= real_prob
            expected_fake = cls == "fake"
            correct += int(is_fake == expected_fake)
            total += 1

    return correct, total


def main() -> None:
    samples: dict[str, list[Path]] = {}
    for cls in ("fake", "real"):
        folder = PROJECT_ROOT / "data" / "valid" / cls
        imgs = list(folder.glob("*.jpg"))
        samples[cls] = random.sample(imgs, min(5, len(imgs)))

    for model_id in CANDIDATES:
        print(f"\n=== {model_id} ===")
        try:
            correct, total = benchmark_model(model_id, samples)
            print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
        except Exception as exc:
            print(f"  FAILED: {exc}")


if __name__ == "__main__":
    main()
