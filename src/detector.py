"""
Multi-model ensemble detector for real vs AI-generated / deepfake images.

Combines three free Hugging Face models with test-time augmentation (TTA).
No training or API key required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

ENSEMBLE_MODELS: tuple[tuple[str, str], ...] = (
    ("v1", "prithivMLmods/deepfake-detector-model-v1"),
    ("vit", "capcheck/ai-image-detection"),
    ("3class", "prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0"),
)

# Display weights (v1 is most reliable on real face photos)
DISPLAY_WEIGHTS: dict[str, float] = {"v1": 0.65, "vit": 0.20, "3class": 0.15}
FAKE_THRESHOLD_LINE = 0.50  # shown on explanation chart


@dataclass
class ModelScore:
    """Fake probability from one backbone after TTA."""

    name: str
    fake_prob: float


@dataclass
class PredictionResult:
    """Structured ensemble prediction output."""

    label: str
    confidence: float
    fake_prob: float
    real_prob: float
    model_scores: list[ModelScore] = field(default_factory=list)


def _tta_variants(pil_image: Image.Image) -> list[Image.Image]:
    """Build original, flipped, and center-cropped views for TTA."""
    image = pil_image.convert("RGB")
    variants = [image, ImageOps.mirror(image)]

    width, height = image.size
    crop_ratio = 0.85
    crop_w = max(1, int(width * crop_ratio))
    crop_h = max(1, int(height * crop_ratio))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    variants.append(image.crop((left, top, left + crop_w, top + crop_h)))

    return variants


class _SingleHFModel:
    """One Hugging Face image-classification backbone."""

    def __init__(self, short_name: str, model_id: str, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.short_name = short_name
        self.model_id = model_id
        self.device = device

        print(f"  Loading {short_name}: {model_id} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.fake_indices = [
            i for i, name in id2label.items() if "fake" in name or name == "ai"
        ]
        self.real_indices = [i for i, name in id2label.items() if "real" in name]

        if not self.fake_indices:
            raise ValueError(f"No fake/AI labels found for {model_id}: {id2label}")

    def fake_prob_for_image(self, pil_image: Image.Image) -> float:
        """Return P(fake/AI) for a single image view."""
        inputs = self.processor(images=pil_image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze().cpu()

        fake_prob = sum(probs[i].item() for i in self.fake_indices)
        if self.real_indices:
            real_prob = sum(probs[i].item() for i in self.real_indices)
            total = fake_prob + real_prob
            if total > 0:
                fake_prob /= total
        return fake_prob

    def fake_prob_with_tta(self, pil_image: Image.Image) -> float:
        """Average fake probability across TTA views."""
        scores = [self.fake_prob_for_image(view) for view in _tta_variants(pil_image)]
        return sum(scores) / len(scores)


class EnsembleDetector:
    """
    Three-model ensemble with TTA for robust AI / deepfake detection.

    Models:
      - deepfake-detector-model-v1 (SigLIP deepfake)
      - capcheck/ai-image-detection (ViT AI vs real)
      - AI-vs-Deepfake-vs-Real-v2.0 (3-class: AI + deepfake vs real)
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading ensemble on {self.device} ...")

        self.backends: list[_SingleHFModel] = []
        for short_name, model_id in ENSEMBLE_MODELS:
            self.backends.append(_SingleHFModel(short_name, model_id, self.device))

        print("Ensemble ready.")

    @property
    def display_name(self) -> str:
        return "3-model ensemble (v1 + ViT + 3-class) + TTA"

    def _collect_scores(self, pil_image: Image.Image) -> list[ModelScore]:
        return [
            ModelScore(name=backend.short_name, fake_prob=backend.fake_prob_with_tta(pil_image))
            for backend in self.backends
        ]

    def _decide_fake(self, model_scores: list[ModelScore]) -> tuple[bool, float, float]:
        """
        Combine per-model scores with v1 as the primary signal.

        The ViT model (capcheck) often scores real photos as fake; v1 is used
        to veto those false positives while vit/3class help catch modern AI
        images when v1 is uncertain.
        """
        by_name = {score.name: score.fake_prob for score in model_scores}
        v1 = by_name["v1"]
        vit = by_name["vit"]
        c3 = by_name["3class"]

        weighted_fake = sum(DISPLAY_WEIGHTS[name] * by_name[name] for name in DISPLAY_WEIGHTS)

        # --- Decision tree (v1-anchored) ---
        if v1 < 0.22:
            # v1 strongly says real — only flag AI-art style fakes with 3class + vit
            is_fake = c3 >= 0.40 and vit >= 0.78
        elif v1 > 0.52:
            is_fake = True
        elif v1 < 0.42:
            # v1 moderately real — vit can confirm fake but v1 must not be near-zero
            is_fake = vit >= 0.82 and v1 >= 0.25
        else:
            # v1 borderline (0.42–0.52) — need v1 and vit to agree strongly
            is_fake = v1 >= 0.48 and vit >= 0.85

        # Modern AI path: 3class confident even when v1 misses face fakes
        if not is_fake and c3 >= 0.45 and vit >= 0.72:
            is_fake = True

        real_prob = 1.0 - weighted_fake
        return is_fake, weighted_fake, real_prob

    def predict_with_details(self, pil_image: Image.Image) -> PredictionResult:
        """Run ensemble + TTA and return full prediction details."""
        model_scores = self._collect_scores(pil_image)
        is_fake, ensemble_fake, real_prob = self._decide_fake(model_scores)
        max_fake = max(score.fake_prob for score in model_scores)

        if is_fake:
            label = "FAKE 🔴"
            confidence = max(ensemble_fake, max_fake) * 100.0
        else:
            label = "REAL 🟢"
            confidence = real_prob * 100.0

        return PredictionResult(
            label=label,
            confidence=confidence,
            fake_prob=ensemble_fake,
            real_prob=real_prob,
            model_scores=model_scores,
        )

    def predict_image(self, pil_image: Image.Image) -> tuple[str, float]:
        """Predict REAL or FAKE with confidence percentage."""
        result = self.predict_with_details(pil_image)
        return result.label, result.confidence

    def generate_explanation(
        self,
        pil_image: Image.Image,
        result: PredictionResult | None = None,
    ) -> Image.Image:
        """Build a chart showing ensemble and per-model fake scores."""
        if result is None:
            result = self.predict_with_details(pil_image)

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), facecolor="#1a1a1a")
        for ax in axes:
            ax.set_facecolor("#1a1a1a")

        # Left: ensemble fake vs real
        ensemble_labels = ["Fake", "Real"]
        ensemble_probs = [result.fake_prob * 100, result.real_prob * 100]
        bars = axes[0].barh(ensemble_labels, ensemble_probs, color=["#e74c3c", "#2ecc71"], height=0.5)
        axes[0].set_xlim(0, 100)
        axes[0].set_xlabel("Probability (%)", color="#e0e0e0", fontsize=9)
        axes[0].set_title("Ensemble", color="#e0e0e0", fontsize=11)
        axes[0].tick_params(colors="#e0e0e0")
        for spine in ("top", "right"):
            axes[0].spines[spine].set_visible(False)
        for bar, prob in zip(bars, ensemble_probs):
            axes[0].text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%",
                va="center",
                color="#e0e0e0",
                fontsize=9,
            )

        # Right: per-model fake scores (after TTA)
        model_names = [score.name for score in result.model_scores]
        model_fake = [score.fake_prob * 100 for score in result.model_scores]
        bars = axes[1].barh(model_names, model_fake, color="#e67e22", height=0.5)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Fake score (%)", color="#e0e0e0", fontsize=9)
        axes[1].set_title("Per model (TTA)", color="#e0e0e0", fontsize=11)
        axes[1].tick_params(colors="#e0e0e0")
        axes[1].axvline(FAKE_THRESHOLD_LINE * 100, color="#888", linestyle="--", linewidth=1)
        for spine in ("top", "right"):
            axes[1].spines[spine].set_visible(False)
        for bar, prob in zip(bars, model_fake):
            axes[1].text(
                min(bar.get_width() + 1, 92),
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%",
                va="center",
                color="#e0e0e0",
                fontsize=9,
            )

        plt.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=120, facecolor="#1a1a1a")
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


# Backward-compatible alias for type hints in predict.py
HuggingFaceDetector = EnsembleDetector


def load_detector() -> EnsembleDetector:
    """Load the three-model ensemble detector."""
    return EnsembleDetector()
