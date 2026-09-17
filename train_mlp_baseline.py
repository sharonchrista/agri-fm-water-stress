"""
Minimal MLP baseline for groundnut HSI stress classification.

Groundnut patches are (N, 1, 1, 282) — single pixel spectra with no spatial
context. A spatial encoder (CNN or Transformer) adds unnecessary complexity.
This script uses a simple 3-layer MLP directly on the 282-band spectrum.

Run: python train_mlp_baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset — flat spectrum, no spatial dims
# ---------------------------------------------------------------------------

class SpectralDataset(Dataset):
    """Flat spectrum dataset — returns (282,) vectors and integer labels."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        if indices is not None:
            X = X[indices]
            y = y[indices]

        # Flatten spatial dims: (N,1,1,282) or (N,282) -> (N,282)
        self.X = X.reshape(len(X), -1).astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.X[idx].copy()
        if self.augment:
            # Spectral jitter — mild Gaussian noise
            x += np.random.normal(0, 0.005, size=x.shape).astype(np.float32)
            x = np.clip(x, 0.0, 1.0)
            # Random band dropout — zero out 5% of bands
            mask = np.random.rand(len(x)) > 0.05
            x = x * mask.astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

    def class_weights(self) -> Tensor:
        unique, counts = np.unique(self.y, return_counts=True)
        weights = 1.0 / counts.astype(np.float32)
        weights = weights / weights.sum()
        sample_weights = np.array([weights[lbl] for lbl in self.y], dtype=np.float32)
        return torch.from_numpy(sample_weights)


# ---------------------------------------------------------------------------
# Model — lightweight MLP
# ---------------------------------------------------------------------------

class SpectralMLP(nn.Module):
    """
    3-layer MLP for hyperspectral stress classification.

    Deliberately small — 282-dim input, two hidden layers of 256 and 64,
    binary output. Strong dropout for regularisation on small dataset.
    Total params: ~90K — appropriate for 13,333 training samples.
    """

    def __init__(
        self,
        num_bands: int = 282,
        hidden_dims: tuple[int, ...] = (256, 64),
        num_classes: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        dims = [num_bands] + list(hidden_dims) + [num_classes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        # Initialise with small weights to prevent early saturation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Return penultimate layer activations for few-shot evaluation."""
        with torch.no_grad():
            for layer in list(self.net.children())[:-1]:
                x = layer(x)
        return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_accuracy(logits: Tensor, labels: Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def compute_per_class_accuracy(
    logits: Tensor, labels: Tensor, num_classes: int
) -> dict[int, float]:
    preds = logits.argmax(dim=-1)
    return {
        cls: (preds[labels == cls] == cls).float().mean().item()
        for cls in range(num_classes)
        if (labels == cls).any()
    }


def train_epoch(
    model: SpectralMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> dict[str, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_acc += compute_accuracy(logits.detach(), y)
        n += 1
    return {"loss": total_loss / n, "accuracy": total_acc / n}


@torch.no_grad()
def evaluate(
    model: SpectralMLP,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        total_loss += criterion(logits, y).item()
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
        n += 1
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    per_class = compute_per_class_accuracy(all_logits, all_labels, num_classes)
    return {
        "loss": total_loss / n,
        "accuracy": compute_accuracy(all_logits, all_labels),
        "mean_per_class_accuracy": sum(per_class.values()) / len(per_class),
        "per_class_accuracy": per_class,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")

    # Load data
    proc = DATA_ROOT / "processed" / "hsi"
    X = np.load(proc / "gn_X_patch.npy")      # (16667, 1, 1, 282)
    y = np.load(proc / "gn_y.npy")             # (16667,) int64
    train_idx = np.load(proc / "gn_train_idx.npy")
    test_idx = np.load(proc / "gn_test_idx.npy")

    train_ds = SpectralDataset(X, y, train_idx, augment=True)
    test_ds = SpectralDataset(X, y, test_idx, augment=False)

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
    print(f"Train label dist: {dict(zip(*np.unique(y[train_idx], return_counts=True)))}")
    print(f"Test  label dist: {dict(zip(*np.unique(y[test_idx], return_counts=True)))}")

    sampler = WeightedRandomSampler(
        train_ds.class_weights(), len(train_ds), replacement=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=128, sampler=sampler,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Model
    model = SpectralMLP(
        num_bands=282,
        hidden_dims=(256, 64),
        num_classes=2,
        dropout=0.4,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    print(model)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5
    )

    # Train
    EPOCHS = 100
    best_acc = 0.0
    history = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} "
          f"{'TeLoss':>8} {'TeAcc':>7} {'mCA':>7} {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion)
        te = evaluate(model, test_loader, criterion, num_classes=2)
        scheduler.step()

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "test": te})

        print(f"{epoch:>6} {tr['loss']:>8.4f} {tr['accuracy']:>7.4f} "
              f"{te['loss']:>8.4f} {te['accuracy']:>7.4f} "
              f"{te['mean_per_class_accuracy']:>7.4f} {elapsed:>5.1f}s")

        if te["accuracy"] > best_acc:
            best_acc = te["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": te,
            }, CHECKPOINT_DIR / "groundnut_mlp_best.pt")
            print(f"  --> best: {best_acc:.4f}")

    with open(LOG_DIR / "groundnut_mlp.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()