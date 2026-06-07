"""
EfficientNet-B0 model definition for binary deepfake classification.

Loads ImageNet-pretrained weights, freezes most layers, and replaces the
classifier head with a dropout + single-logit output layer.
"""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNet-B0 configured for binary deepfake detection.

    Steps:
      1. Load pretrained ImageNet weights (optional).
      2. Freeze all parameters except the last 20.
      3. Replace classifier with Dropout(0.4) -> Linear(1280, 1).

    Returns:
        Model ready for fine-tuning / inference.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Freeze the entire backbone first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 20 backbone (feature) parameters for light fine-tuning
    backbone_params = list(model.features.parameters())
    for param in backbone_params[-20:]:
        param.requires_grad = True

    # Replace classifier head — train the full new head (not just 20 scalars)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features=1280, out_features=1),
    )
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
