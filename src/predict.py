"""
Inference utilities for deepfake detection on images and videos.

Primary backend: 3-model Hugging Face ensemble with TTA (free, no API key).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

from src.detector import HuggingFaceDetector, PredictionResult, load_detector

# Re-export for app startup
__all__ = [
    "load_detector",
    "predict_image",
    "predict_video",
    "generate_explanation",
    "HuggingFaceDetector",
]


def predict_image(
    detector: HuggingFaceDetector,
    pil_image: Image.Image,
) -> tuple[str, float]:
    """Predict whether a single image is real or fake."""
    return detector.predict_image(pil_image)


def generate_explanation(
    detector: HuggingFaceDetector,
    pil_image: Image.Image,
    result: PredictionResult | None = None,
) -> Image.Image:
    """Return a probability chart explaining the prediction."""
    return detector.generate_explanation(pil_image, result=result)


def predict_video(
    detector: HuggingFaceDetector,
    video_path: str | Path,
) -> tuple[str, float]:
    """
    Predict whether a video is real or fake using frame sampling.

    Samples every 15th frame and returns majority-vote label with average confidence.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video file: {video_path}")

    labels: list[str] = []
    confidences: list[float] = []
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % 15 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                label, confidence = predict_image(detector, pil_image)
                labels.append(label)
                confidences.append(confidence)

            frame_index += 1
    finally:
        cap.release()

    if not labels:
        raise ValueError(
            f"No frames could be read from video: {video_path}. "
            "The file may be corrupt or empty."
        )

    majority_label = Counter(labels).most_common(1)[0][0]
    avg_confidence = sum(confidences) / len(confidences)
    return majority_label, avg_confidence
