"""
Training script for the deepfake detection model.

Trains EfficientNet-B0 for 10 epochs, saves the best checkpoint by validation
accuracy, plots training curves, and prints a validation confusion matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as: python src/train.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from src.dataset import get_dataloaders
from src.model import build_model, get_device


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model and return loss, accuracy, and raw predictions/labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[float] = []
    all_labels: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def plot_training_curve(
    history: dict[str, list[float]],
    save_path: Path,
) -> None:
    """Plot and save training loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss subplot
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy subplot
    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved to {save_path}")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a labeled confusion matrix for validation results."""
    # Labels: 0=real, 1=fake (matching our binary target convention)
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    print("\nValidation Confusion Matrix:")
    print("                 Predicted")
    print("                 REAL   FAKE")
    print(f"Actual REAL      {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"Actual FAKE      {cm[1, 0]:5d}  {cm[1, 1]:5d}")


def main() -> None:
    """Main training loop."""
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    try:
        train_loader, val_loader = get_dataloaders(batch_size=32)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    # Build model, loss, optimizer, and scheduler
    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    num_epochs = 10
    best_val_acc = 0.0
    best_model_path = PROJECT_ROOT / "deepfake_model.pth"
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"\nTraining for {num_epochs} epochs...\n")

    final_val_preds = np.array([])
    final_val_labels = np.array([])

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model checkpoint by validation accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            final_val_preds = val_preds
            final_val_labels = val_labels

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to {best_model_path}")

    # Plot and save training curves
    curve_path = PROJECT_ROOT / "training_curve.png"
    plot_training_curve(history, curve_path)

    # Print confusion matrix from best epoch validation predictions
    if final_val_preds.size > 0:
        print_confusion_matrix(final_val_labels, final_val_preds)


if __name__ == "__main__":
    main()
