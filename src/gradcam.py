"""
Grad-CAM visualization for EfficientNet-B0 deepfake detection.

Generates a heatmap overlay highlighting facial regions that most
influenced the model's real/fake prediction.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from src.predict import _detect_and_crop_face


class GradCAM:
    """
    Grad-CAM implementation that hooks into EfficientNet-B0's last feature block.

    Captures forward activations and backward gradients to produce a spatial
    importance map over the input image.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Last convolutional block in EfficientNet-B0 features
        self.target_layer = model.features[-1]
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def forward_hook(_module, _inputs, output):
            self.activations = output

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_index: int = 0) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            class_index: Output index to explain (0 for single-logit model).

        Returns:
            Normalized heatmap as a 2D numpy array in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor).squeeze()

        # Single-logit model returns a scalar tensor after squeeze
        if output.dim() == 0:
            score = output
        else:
            score = output[class_index]

        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Global average pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def _overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """
    Resize heatmap to image size and blend with the original using a colormap.
    """
    image_rgb = np.array(pil_image.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend original image with heatmap (60% image, 40% heatmap)
    overlay = np.clip(0.6 * image_rgb + 0.4 * heatmap_color, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def generate_gradcam(model: torch.nn.Module, pil_image: Image.Image) -> Image.Image:
    """
    Generate a Grad-CAM heatmap overlay for a PIL image.

    Applies the same face detection/cropping pipeline as inference so the
    heatmap aligns with the region the model actually evaluated.

    Args:
        model: Trained EfficientNet-B0 in eval mode (will temporarily enable grad).
        pil_image: Input PIL Image.

    Returns:
        PIL Image with heatmap overlay.
    """
    face_image = _detect_and_crop_face(pil_image)
    transform = get_transforms(train=False)
    input_tensor = transform(face_image.convert("RGB")).unsqueeze(0)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    was_training = model.training
    model.eval()

    gradcam = GradCAM(model)
    input_tensor.requires_grad_(True)

    with torch.enable_grad():
        heatmap = gradcam.generate(input_tensor)

    if was_training:
        model.train()

    return _overlay_heatmap(face_image, heatmap)
