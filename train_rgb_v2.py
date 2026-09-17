"""
Fixed RGB panicle counting training.

Fixes three bugs from the original:
  1. Density maps generated at model output resolution (53x71) — no scale factor
  2. Combined loss: MSE on density map + L1 on predicted vs GT count
  3. Count loss weight ensures model cannot ignore density signal

The count loss term directly penalises zero-count predictions,
forcing the model to learn the density signal rather than predicting zeros.

Run: python train_rgb_v2.py
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
from torch.utils.data import DataLoader, Dataset


DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ORIG_H, ORIG_W = 850, 1150
TARGET_H, TARGET_W = 425, 575
MODEL_H = TARGET_H // 8   # 53
MODEL_W = TARGET_W // 8   # 71

# Gaussian kernel
SIGMA = 4     # smaller sigma at model output resolution (53x71)
KSIZE = 21    # 5*sigma + 1


def _gaussian_kernel(sigma: float, ksize: int) -> Tensor:
    coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    k = torch.exp(-(gx**2 + gy**2) / (2 * sigma**2))
    return k / k.sum()


_KERNEL = _gaussian_kernel(SIGMA, KSIZE)


def points_to_density(
    points: np.ndarray,
    h: int,
    w: int,
    orig_h: int = ORIG_H,
    orig_w: int = ORIG_W,
) -> np.ndarray:
    """Convert (n,2) [x,y] points in original coords to (h,w) density map."""
    dm = torch.zeros(1, 1, h, w)
    pad = KSIZE // 2
    for x_orig, y_orig in points:
        x = int(round(float(x_orig) * w / orig_w))
        y = int(round(float(y_orig) * h / orig_h))
        if 0 <= x < w and 0 <= y < h:
            dm[0, 0, y, x] += 1.0
    k4d = _KERNEL.unsqueeze(0).unsqueeze(0)
    dm = F.conv2d(dm, k4d, padding=pad)
    return dm.squeeze().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PaddyDataset(Dataset):

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        images: np.ndarray,
        point_labels: np.ndarray,
        counts: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.augment = augment
        n = len(images)

        # Resize images to TARGET resolution
        print(f"  Resizing {n} images...")
        imgs_t = torch.from_numpy(images).permute(0, 3, 1, 2)
        imgs_r = F.interpolate(
            imgs_t, size=(TARGET_H, TARGET_W),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).numpy()
        self.images = ((imgs_r - self._MEAN) / (self._STD + 1e-6)).astype(np.float32)

        # Generate density maps at MODEL output resolution — no scale factor
        print(f"  Generating density maps at ({MODEL_H}x{MODEL_W})...")
        self.density_maps = np.zeros((n, MODEL_H, MODEL_W), dtype=np.float32)
        for i, pts in enumerate(point_labels):
            if len(pts) > 0:
                self.density_maps[i] = points_to_density(
                    pts, MODEL_H, MODEL_W, ORIG_H, ORIG_W
                )
            if (i + 1) % 300 == 0:
                print(f"    {i+1}/{n}")

        self.counts = counts.astype(np.float32)

        # Verify density map counts match GT
        dm_counts = self.density_maps.sum(axis=(1, 2))
        mae = np.abs(dm_counts - self.counts).mean()
        print(f"  Density map count MAE vs GT: {mae:.2f} "
              f"(expect small — boundary clipping only)")

    def __len__(self) -> int:
        return len(self.counts)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        img = self.images[idx].copy()
        dm = self.density_maps[idx].copy()

        if self.augment:
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
                dm = dm[:, ::-1].copy()
            if np.random.rand() > 0.5:
                img = img[::-1, :, :].copy()
                dm = dm[::-1, :].copy()
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, -3.0, 3.0)

        img_t = torch.from_numpy(img.transpose(2, 0, 1))
        dm_t = torch.from_numpy(dm).unsqueeze(0)
        count_t = torch.tensor(self.counts[idx], dtype=torch.float32)
        return img_t, dm_t, count_t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DensityNet(nn.Module):
    """
    Lightweight encoder-decoder for density map regression.
    Same architecture as before but with proper output scaling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        return F.relu(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CountAwareLoss(nn.Module):
    """
    Combined loss:
      - MSE on density map spatial distribution
      - L1 on total predicted count vs GT count

    The count term prevents the model from predicting zero everywhere.
    alpha controls the weight of the count loss term.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        pred_dm: Tensor,
        gt_dm: Tensor,
        gt_counts: Tensor,
    ) -> tuple[Tensor, dict]:
        # Density map MSE
        mse_loss = F.mse_loss(pred_dm, gt_dm)

        # Count L1
        pred_counts = pred_dm.sum(dim=(1, 2, 3))
        count_loss = F.l1_loss(pred_counts, gt_counts)

        total = mse_loss + self.alpha * count_loss

        return total, {
            "mse": mse_loss.item(),
            "count_l1": count_loss.item(),
            "total": total.item(),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: DensityNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CountAwareLoss,
    scaler: torch.amp.GradScaler,
) -> dict:
    model.train()
    totals = {"loss": 0, "mse": 0, "count_l1": 0, "mae": 0}
    n = 0

    for imgs, gt_dm, gt_counts in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        gt_dm = gt_dm.to(DEVICE, non_blocking=True)
        gt_counts = gt_counts.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            pred = model(imgs)
            loss, components = criterion(pred, gt_dm, gt_counts)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        pred_counts = pred.detach().sum(dim=(1, 2, 3))
        mae = (pred_counts - gt_counts).abs().mean().item()

        totals["loss"] += components["total"]
        totals["mse"] += components["mse"]
        totals["count_l1"] += components["count_l1"]
        totals["mae"] += mae
        n += 1

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model: DensityNet, loader: DataLoader) -> dict:
    model.eval()
    all_pred, all_gt = [], []

    for imgs, gt_dm, gt_counts in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        gt_dm = gt_dm.to(DEVICE, non_blocking=True)
        pred = model(imgs)
        all_pred.append(pred.sum(dim=(1, 2, 3)).cpu())
        all_gt.append(gt_counts)

    pred_c = torch.cat(all_pred)
    gt_c = torch.cat(all_gt)
    mae = (pred_c - gt_c).abs().mean().item()
    rmse = ((pred_c - gt_c) ** 2).mean().sqrt().item()
    rel_mae = mae / (gt_c.mean().item() + 1e-6)
    return {
        "mae": mae,
        "rmse": rmse,
        "rel_mae": rel_mae,
        "mean_pred": pred_c.mean().item(),
        "mean_gt": gt_c.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")

    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 3e-4
    COUNT_LOSS_WEIGHT = 0.1

    rgb_dir = DATA_ROOT / "processed" / "rgb"
    print("Loading arrays...")
    t0 = time.time()
    tr_imgs = np.load(rgb_dir / "train_images.npy")
    tr_pts = np.load(rgb_dir / "train_point_labels.npy", allow_pickle=True)
    tr_cnt = np.load(rgb_dir / "train_point_counts.npy")
    te_imgs = np.load(rgb_dir / "test_images.npy")
    te_pts = np.load(rgb_dir / "test_point_labels.npy", allow_pickle=True)
    te_cnt = np.load(rgb_dir / "test_point_counts.npy")
    print(f"Loaded in {time.time()-t0:.1f}s")

    print("\nBuilding train dataset:")
    train_ds = PaddyDataset(tr_imgs, tr_pts, tr_cnt, augment=True)
    print("\nBuilding test dataset:")
    test_ds = PaddyDataset(te_imgs, te_pts, te_cnt, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f"\nTrain: {len(train_loader)} batches | Test: {len(test_loader)} batches")

    model = DensityNet().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CountAwareLoss(alpha=COUNT_LOSS_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_mae = float("inf")

    print(f"\n{'Ep':>4} {'TrLoss':>8} {'TrMAE':>7} {'TeMAE':>7} "
          f"{'RMSE':>7} {'RelMAE':>8} {'PredMn':>7} {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, scaler)
        te = evaluate(model, test_loader)
        scheduler.step()

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "test": te})

        print(f"{epoch:>4} {tr['loss']:>8.4f} {tr['mae']:>7.2f} "
              f"{te['mae']:>7.2f} {te['rmse']:>7.2f} "
              f"{te['rel_mae']:>8.4f} {te['mean_pred']:>7.1f} {elapsed:>5.1f}s")

        if te["mae"] < best_mae:
            best_mae = te["mae"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": te,
            }, CHECKPOINT_DIR / "rgb_density_v2_best.pt")
            print(f"  --> best MAE: {best_mae:.2f} "
                  f"(pred mean: {te['mean_pred']:.1f} vs gt: {te['mean_gt']:.1f})")

        if epoch % 25 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": te,
            }, CHECKPOINT_DIR / f"rgb_density_v2_epoch{epoch:03d}.pt")

    with open(LOG_DIR / "rgb_counting_v2.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest test MAE: {best_mae:.2f} panicles")
    print(f"Mean GT count: {te['mean_gt']:.1f} panicles")


if __name__ == "__main__":
    main()