"""OpenCV Haar cascade face detection for local EfficientNet / Grad-CAM."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def detect_and_crop_face(pil_image: Image.Image, padding: float = 0.2) -> Image.Image:
    """
    Detect the largest face and return a cropped PIL image.

    Falls back to the full image when no face is found.
    """
    image = pil_image.convert("RGB")
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    width, height = image.size
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(width, x + w + pad_w)
    y2 = min(height, y + h + pad_h)
    return image.crop((x1, y1, x2, y2))
