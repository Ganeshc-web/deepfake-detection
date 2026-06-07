"""
Fast training script — trains on a small data subset in minutes.

Use this while full training (python src/train.py) runs in the background.
Saves to deepfake_model_fast.pth so it won't conflict with the full model.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import build_model, get_device
from src.train import evaluate, plot_training_curve, print_confusion_matrix

# Fast-run defaults: stronger subset training, ~15-30 min on CPU
MAX_SAMPLES_PER_CLASS = 2000
NUM_EPOCHS = 5
BATCH_SIZE = 32
MODEL_PATH = PROJECT_ROOT / "deepfake_model_fast.pth"
CURVE_PATH = PROJECT_ROOT / "training_curve_fast.png"


def train_one_epoch_fast(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Train one epoch with a progress bar."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", unit="batch")
    for images, labels in progress:
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

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main() -> None:
    """Run quick training on a subset and save deepfake_model_fast.pth."""
    device = get_device()
    print(f"Using device: {device}")
    print(
        f"\nFAST MODE: {MAX_SAMPLES_PER_CLASS} samples/class, "
        f"{NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}"
    )
    print(f"Output model: {MODEL_PATH.name}\n")

    try:
        train_loader, val_loader = get_dataloaders(
            batch_size=BATCH_SIZE,
            max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    print(f"Training samples: {train_size} | Validation samples: {val_size}\n")

    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    final_val_preds = None
    final_val_labels = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch_fast(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )
        print(f"  Validating epoch {epoch}...")
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            final_val_preds = val_preds
            final_val_labels = val_labels

    print(f"\nFast training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {MODEL_PATH}")
    print(f"\nRun the app now:  python app.py")

    plot_training_curve(history, CURVE_PATH)

    if final_val_preds is not None and final_val_labels is not None:
        print_confusion_matrix(final_val_labels, final_val_preds)


if __name__ == "__main__":
    main()
