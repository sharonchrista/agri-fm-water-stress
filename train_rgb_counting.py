"""
RGB paddy panicle counting training via density map regression.

Architecture: CSRNet-style encoder-decoder
  Encoder: VGG16 frontend (pretrained ImageNet, first 10 conv layers)
  Decoder: dilated conv backend -> density map same size as input / 8

Loss: MSE between predicted and ground truth Gaussian density maps
Metric: MAE (Mean Absolute Error) on panicle count
        count = sum of density map values

The density map approach is standard for point-supervised counting.
Summing the predicted density map gives the estimated panicle count.

Run: python train_rgb_counting.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from rgb_fast_dataset import build_rgb_fast_loaders


DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model — lightweight encoder-decoder for density map regression
# ---------------------------------------------------------------------------

class DensityEncoder(nn.Module):
    """
    Lightweight CNN encoder for density map regression.
    Does not use pretrained VGG to keep memory footprint small
    and avoid ImageNet-domain mismatch with agricultural UAV imagery.

    Input : (B, 3, H, W)
    Output: (B, 1, H//8, W//8) — density map at 1/8 resolution
    """

    def __init__(self) -> None:
        super().__init__()

        # Frontend: progressive downsampling x8
        self.frontend = nn.Sequential(
            # Block 1: (B, 3, H, W) -> (B, 32, H/2, W/2)
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: -> (B, 64, H/4, W/4)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: -> (B, 128, H/8, W/8)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Backend: dilated convs to expand receptive field without downsampling
        self.backend = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),   # final 1x1 conv -> density map
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 3, H, W)
        x = self.frontend(x)    # (B, 128, H/8, W/8)
        x = self.backend(x)     # (B, 1, H/8, W/8)
        return F.relu(x)        # density must be non-negative


# ---------------------------------------------------------------------------
# Count metrics
# ---------------------------------------------------------------------------

def compute_count_mae(
    pred_density: Tensor,
    gt_density: Tensor,
) -> float:
    """
    Mean Absolute Error on panicle counts.
    Count = sum of density map values.
    """
    pred_counts = pred_density.sum(dim=(1, 2, 3))
    gt_counts = gt_density.sum(dim=(1, 2, 3))
    return (pred_counts - gt_counts).abs().mean().item()


def compute_count_mse(
    pred_density: Tensor,
    gt_density: Tensor,
) -> float:
    pred_counts = pred_density.sum(dim=(1, 2, 3))
    gt_counts = gt_density.sum(dim=(1, 2, 3))
    return ((pred_counts - gt_counts) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: DensityEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    for images, gt_density, gt_counts in loader:
        images = images.to(device, non_blocking=True)
        gt_density = gt_density.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred_density = model(images)

        # gt_density is already at model output resolution — no interpolation needed
        loss = F.mse_loss(pred_density, gt_density.to(device, non_blocking=True))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        gt_density_gpu = gt_density.to(device, non_blocking=True)
        total_mae += compute_count_mae(pred_density.detach(), gt_density_gpu.detach())
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mae": total_mae / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: DensityEncoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_pred_counts = []
    all_gt_counts = []
    total_loss = 0.0
    n_batches = 0

    for images, gt_density, gt_counts in loader:
        images = images.to(device, non_blocking=True)
        gt_density = gt_density.to(device, non_blocking=True)

        pred_density = model(images)

        # gt_density already at model output resolution
        gt_density_gpu = gt_density.to(device, non_blocking=True)
        loss = F.mse_loss(pred_density, gt_density_gpu)
        total_loss += loss.item()

        pred_counts = pred_density.sum(dim=(1, 2, 3)).cpu()
        gt_c = gt_density_gpu.sum(dim=(1, 2, 3)).cpu()
        all_pred_counts.append(pred_counts)
        all_gt_counts.append(gt_c)
        n_batches += 1

    all_pred = torch.cat(all_pred_counts)
    all_gt = torch.cat(all_gt_counts)
    mae = (all_pred - all_gt).abs().mean().item()
    mse = ((all_pred - all_gt) ** 2).mean().item()
    rmse = mse ** 0.5

    # Relative MAE — normalised by mean ground truth count
    mean_gt = all_gt.mean().item()
    rel_mae = mae / (mean_gt + 1e-6)

    return {
        "loss": total_loss / n_batches,
        "mae": mae,
        "rmse": rmse,
        "rel_mae": rel_mae,
        "mean_gt_count": mean_gt,
        "mean_pred_count": all_pred.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")

    # Hyperparameters
    BATCH_SIZE = 8       # larger batch fine with in-memory data
    EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    # Data
    print("\nBuilding RGB loaders...")
    loaders = build_rgb_fast_loaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    print(f"Train: {len(loaders['train'])} batches | "
          f"Test: {len(loaders['test'])} batches")

    # Model
    model = DensityEncoder().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(DEVICE) / 1024 ** 2
        print(f"VRAM after model init: {allocated:.1f} MB")

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    # Training
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_mae = float("inf")

    print(f"\n{'Epoch':>6} {'TrLoss':>10} {'TrMAE':>8} "
          f"{'TeLoss':>10} {'TeMAE':>8} {'RMSE':>8} {'RelMAE':>8} {'Time':>7}")
    print("-" * 72)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr = train_epoch(model, loaders["train"], optimizer, DEVICE)
        te = evaluate(model, loaders["test"], DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "test": te})

        print(f"{epoch:>6} {tr['loss']:>10.4f} {tr['mae']:>8.2f} "
              f"{te['loss']:>10.4f} {te['mae']:>8.2f} "
              f"{te['rmse']:>8.2f} {te['rel_mae']:>8.4f} {elapsed:>6.1f}s")

        if te["mae"] < best_mae:
            best_mae = te["mae"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": te,
            }, CHECKPOINT_DIR / "rgb_density_best.pt")
            print(f"  --> best MAE: {best_mae:.2f} "
                  f"(mean gt count: {te['mean_gt_count']:.1f}, "
                  f"mean pred: {te['mean_pred_count']:.1f})")

        if epoch % 25 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": te,
            }, CHECKPOINT_DIR / f"rgb_density_epoch{epoch:03d}.pt")

    with open(LOG_DIR / "rgb_counting.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best test MAE: {best_mae:.2f} panicles")


if __name__ == "__main__":
    main()