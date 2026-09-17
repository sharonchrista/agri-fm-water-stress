"""
Classification heads and supervised training loop for HSI tasks.

Three tasks share the same SpectralTransformerEncoder backbone:
  - crop_variety        : 10-class head
  - groundnut_stress    : 2-class head
  - pearl_millet_stress : 2-class head

Training modes:
  1. full_finetune  : train encoder + head jointly from scratch
  2. linear_probe   : freeze encoder, train head only (for transfer eval)
  3. few_shot       : prototypical network evaluation (no gradient update)

Usage:
    python train_hsi.py --task crop_variety --mode full_finetune --epochs 50
    python train_hsi.py --task groundnut_stress --mode full_finetune --epochs 30
    python train_hsi.py --task pearl_millet_stress --mode linear_probe --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from hsi_dataset import build_hsi_loaders
from hsi_encoder import SpectralTransformerEncoder, SpectralCNNEncoder, count_parameters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()

TASK_NUM_CLASSES = {
    "crop_variety": 10,
    "groundnut_stress": 2,
    "pearl_millet_stress": 2,
}

TrainingMode = Literal["full_finetune", "linear_probe"]


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Linear classification head attached to the encoder output.

    A single linear layer is used deliberately — the encoder embedding
    should carry all the representational work. A non-linear head would
    mask encoder quality during ablation comparisons.
    """

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, embed_dim) L2-normalised embeddings
        return self.fc(x)


# ---------------------------------------------------------------------------
# Full model: encoder + head
# ---------------------------------------------------------------------------

class HSIClassifier(nn.Module):
    """
    Complete HSI classification model: encoder + linear head.

    Parameters
    ----------
    encoder : SpectralTransformerEncoder or SpectralCNNEncoder
    num_classes : int
    freeze_encoder : bool
        If True, encoder parameters are frozen (linear probe mode).
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(
            embed_dim=encoder.embed_dim if hasattr(encoder, "embed_dim") else 128,
            num_classes=num_classes,
        )
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder(x)
        else:
            embeddings = self.encoder(x)
        return self.head(embeddings)

    def get_embeddings(self, x: Tensor) -> Tensor:
        """Extract embeddings without classification head — for few-shot eval."""
        with torch.no_grad():
            return self.encoder(x)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(logits: Tensor, labels: Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean().item()


def compute_per_class_accuracy(
    logits: Tensor,
    labels: Tensor,
    num_classes: int,
) -> dict[int, float]:
    predictions = logits.argmax(dim=-1)
    per_class: dict[int, float] = {}
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        per_class[cls] = (predictions[mask] == labels[mask]).float().mean().item()
    return per_class


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: HSIClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool = True,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for patches, labels in loader:
        patches = patches.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(patches)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += compute_accuracy(logits.detach(), labels)
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_acc / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: HSIClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    use_amp: bool = True,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for patches, labels in loader:
        patches = patches.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(patches)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    per_class = compute_per_class_accuracy(all_logits, all_labels, num_classes)
    mean_per_class_acc = sum(per_class.values()) / len(per_class)

    return {
        "loss": total_loss / len(loader),
        "accuracy": compute_accuracy(all_logits, all_labels),
        "mean_per_class_accuracy": mean_per_class_acc,
        "per_class_accuracy": per_class,
    }


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: HSIClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    task_name: str,
    encoder_type: str,
) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{task_name}_{encoder_type}_epoch{epoch:03d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "task_name": task_name,
        "encoder_type": encoder_type,
    }, path)
    return path


def load_checkpoint(
    path: Path,
    model: HSIClassifier,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    task_name: str,
    encoder_type: str = "transformer",
    mode: TrainingMode = "full_finetune",
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    embed_dim: int = 128,
    num_workers: int = 4,
    device_id: int = 0,
    save_every: int = 10,
    use_amp: bool = True,
) -> dict:
    """
    Train a single HSI classification task.

    Parameters
    ----------
    task_name    : one of "crop_variety", "groundnut_stress", "pearl_millet_stress"
    encoder_type : "transformer" or "cnn"
    mode         : "full_finetune" or "linear_probe"
    epochs       : number of training epochs
    batch_size   : samples per batch
    learning_rate: initial LR for AdamW
    weight_decay : L2 regularisation
    embed_dim    : encoder embedding dimension
    num_workers  : DataLoader workers
    device_id    : GPU index (always 0 on tyrone-hpc)
    save_every   : save checkpoint every N epochs

    Returns
    -------
    dict of best test metrics
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {task_name} | Encoder: {encoder_type} | Mode: {mode}")

    # Data
    loaders = build_hsi_loaders(
        data_root=DATA_ROOT,
        batch_size=batch_size,
        num_workers=num_workers,
        augment_train=True,
        use_weighted_sampler=True,
    )
    train_loader = loaders[task_name]["train"]
    test_loader = loaders[task_name]["test"]
    num_classes = TASK_NUM_CLASSES[task_name]

    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # Encoder
    if encoder_type == "transformer":
        encoder = SpectralTransformerEncoder(
            num_bands=282,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=4,
            ffn_dim=embed_dim * 4,
            dropout=0.1,
        )
    elif encoder_type == "cnn":
        encoder = SpectralCNNEncoder(
            num_bands=282,
            embed_dim=embed_dim,
            hidden_dim=embed_dim * 2,
            num_layers=4,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    model = HSIClassifier(
        encoder=encoder,
        num_classes=num_classes,
        freeze_encoder=(mode == "linear_probe"),
    ).to(device)

    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss — use label smoothing for crop_variety (10 classes, some imbalance)
    label_smoothing = 0.1 if task_name == "crop_variety" else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimiser — lower LR for linear probe since only head is trained
    lr = learning_rate if mode == "full_finetune" else learning_rate * 10
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{task_name}_{encoder_type}_{mode}.json"
    history = []
    best_acc = 0.0
    best_metrics = {}

    print(f"\nStarting training for {epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Test Loss':>10} {'Test Acc':>9} {'mCA':>8} {'Time':>7}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            use_amp=use_amp,
        )
        test_metrics = evaluate(
            model, test_loader, criterion, device, num_classes,
            use_amp=use_amp,
        )
        scheduler.step()

        elapsed = time.time() - t0
        log_entry = {
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(log_entry)

        print(
            f"{epoch:>6} "
            f"{train_metrics['loss']:>12.4f} "
            f"{train_metrics['accuracy']:>10.4f} "
            f"{test_metrics['loss']:>10.4f} "
            f"{test_metrics['accuracy']:>9.4f} "
            f"{test_metrics['mean_per_class_accuracy']:>8.4f} "
            f"{elapsed:>6.1f}s"
        )

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            best_metrics = test_metrics
            save_checkpoint(model, optimizer, epoch, test_metrics, task_name, encoder_type)
            print(f"  --> New best: {best_acc:.4f} (checkpoint saved)")

        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, test_metrics, task_name, encoder_type)

    # Save full log
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Log saved to {log_path}")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Best mean per-class accuracy: {best_metrics.get('mean_per_class_accuracy', 0):.4f}")

    return best_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HSI classification model")
    parser.add_argument(
        "--task",
        choices=list(TASK_NUM_CLASSES.keys()),
        default="crop_variety",
        help="Which HSI task to train",
    )
    parser.add_argument(
        "--encoder",
        choices=["transformer", "cnn"],
        default="transformer",
        help="Encoder architecture",
    )
    parser.add_argument(
        "--mode",
        choices=["full_finetune", "linear_probe"],
        default="full_finetune",
        help="Training mode",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP mixed precision")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        task_name=args.task,
        encoder_type=args.encoder,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embed_dim=args.embed_dim,
        num_workers=args.num_workers,
        save_every=args.save_every,
        use_amp=not args.no_amp,
    )