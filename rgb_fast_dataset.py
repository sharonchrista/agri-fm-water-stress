"""
Fast RGB paddy dataset using pre-loaded numpy arrays.

Pre-generates all density maps at startup (stored in RAM) rather than
computing them on-the-fly per batch. With 503 GB RAM available this is
the correct strategy — startup takes ~30 seconds, then each epoch is fast.

Images are resized to (425, 575) at load time (half of original 850x1150).
Density maps are generated at the same resolution.

Usage:
    from rgb_fast_dataset import build_rgb_fast_loaders
    loaders = build_rgb_fast_loaders(data_root="~/agri_foundation/data")
    for images, density_maps, counts in loaders["train"]:
        ...
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORIG_H, ORIG_W = 850, 1150
TARGET_H, TARGET_W = 425, 575     # half resolution
GAUSSIAN_SIGMA = 15
GAUSSIAN_KERNEL_SIZE = 61          # must be odd


# ---------------------------------------------------------------------------
# Gaussian kernel
# ---------------------------------------------------------------------------

def _build_gaussian_kernel(sigma: float, kernel_size: int) -> Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


_KERNEL = _build_gaussian_kernel(GAUSSIAN_SIGMA, GAUSSIAN_KERNEL_SIZE)


def points_to_density_map(
    points: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Convert (n, 2) point array [x, y] in original coordinates
    to a (height, width) density map at target resolution.
    """
    density = torch.zeros(1, 1, height, width)
    pad = GAUSSIAN_KERNEL_SIZE // 2

    for x_orig, y_orig in points:
        x = int(round(float(x_orig) * width / ORIG_W))
        y = int(round(float(y_orig) * height / ORIG_H))
        if 0 <= x < width and 0 <= y < height:
            density[0, 0, y, x] += 1.0

    kernel_4d = _KERNEL.unsqueeze(0).unsqueeze(0)
    density = F.conv2d(density, kernel_4d, padding=pad)
    return density.squeeze().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset — all data in RAM
# ---------------------------------------------------------------------------

class RGBPaddyDataset(Dataset):
    """
    In-memory RGB paddy dataset.

    All images are resized and all density maps are pre-generated at init.
    Augmentation (flip) is applied per-item with consistent spatial transform
    on both image and density map.

    Parameters
    ----------
    images : np.ndarray (N, H, W, 3) float32 [0,1]
    point_labels : np.ndarray (N,) object — each element is (n_i, 2) array
    counts : np.ndarray (N,) int64
    augment : bool
    """

    # ImageNet normalisation for RGB
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

        print(f"  Pre-processing {n} images to ({TARGET_H}, {TARGET_W})...")
        t0 = time.time()

        # Resize images: (N, 850, 1150, 3) -> (N, 425, 575, 3)
        # Use torch interpolate for speed
        imgs_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)  # (N,3,H,W)
        imgs_resized = F.interpolate(
            imgs_tensor, size=(TARGET_H, TARGET_W),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).numpy()  # (N, 425, 575, 3)

        # Normalise
        self.images = (imgs_resized - self._MEAN) / (self._STD + 1e-6)
        self.images = self.images.astype(np.float32)

        # Pre-generate all density maps
        # Pre-generate density maps at model output resolution (H//8, W//8)
        # This eliminates the expensive F.interpolate call in the training loop
        MODEL_H = TARGET_H // 8   # 53
        MODEL_W = TARGET_W // 8   # 71
        print(f"  Generating {n} density maps at model resolution ({MODEL_H}x{MODEL_W})...")
        self.density_maps = np.zeros(
            (n, MODEL_H, MODEL_W), dtype=np.float32
        )
        # Scale factor to preserve count after resolution reduction
        scale = (TARGET_H * TARGET_W) / (MODEL_H * MODEL_W)
        for i, pts in enumerate(point_labels):
            if len(pts) > 0:
                dm_full = points_to_density_map(pts, TARGET_H, TARGET_W)
                # Downsample to model output resolution
                dm_tensor = torch.from_numpy(dm_full).unsqueeze(0).unsqueeze(0)
                dm_down = F.interpolate(
                    dm_tensor, size=(MODEL_H, MODEL_W),
                    mode="bilinear", align_corners=False
                ) * scale
                self.density_maps[i] = dm_down.squeeze().numpy()
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{n} done...")

        self.counts = counts.astype(np.int64)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s. "
              f"Images: {self.images.nbytes/1e9:.2f} GB, "
              f"Density maps: {self.density_maps.nbytes/1e9:.2f} GB")

    def __len__(self) -> int:
        return len(self.counts)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        img = self.images[idx].copy()           # (H, W, 3)
        dm = self.density_maps[idx].copy()      # (H, W)

        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
                dm = dm[:, ::-1].copy()
            # Vertical flip
            if np.random.rand() > 0.5:
                img = img[::-1, :, :].copy()
                dm = dm[::-1, :].copy()
            # Colour jitter on image only (brightness, contrast)
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, -3.0, 3.0)

        img_tensor = torch.from_numpy(
            img.transpose(2, 0, 1)     # (3, H, W)
        )
        dm_tensor = torch.from_numpy(dm).unsqueeze(0)   # (1, H, W)

        return img_tensor, dm_tensor, int(self.counts[idx])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rgb_fast_loaders(
    data_root: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,    # 0 = main process only — data already in RAM
) -> dict[str, DataLoader]:
    """
    Build fast in-memory RGB DataLoaders.

    num_workers=0 is correct here — all data is in RAM, multiprocessing
    would just copy the arrays to worker processes wasting memory.

    Returns {"train": DataLoader, "test": DataLoader}
    """
    root = Path(data_root).expanduser()
    rgb_dir = root / "processed" / "rgb"

    print("Loading RGB arrays from disk...")
    t0 = time.time()
    train_images = np.load(rgb_dir / "train_images.npy")
    train_labels = np.load(rgb_dir / "train_point_labels.npy", allow_pickle=True)
    train_counts = np.load(rgb_dir / "train_point_counts.npy")
    test_images = np.load(rgb_dir / "test_images.npy")
    test_labels = np.load(rgb_dir / "test_point_labels.npy", allow_pickle=True)
    test_counts = np.load(rgb_dir / "test_point_counts.npy")
    print(f"Arrays loaded in {time.time()-t0:.1f}s")

    print("\nBuilding train dataset:")
    train_ds = RGBPaddyDataset(
        train_images, train_labels, train_counts, augment=True
    )
    print("\nBuilding test dataset:")
    test_ds = RGBPaddyDataset(
        test_images, test_labels, test_counts, augment=False
    )

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(data_root: str = "~/agri_foundation/data") -> None:
    print("Building fast RGB loaders...")
    loaders = build_rgb_fast_loaders(data_root, batch_size=4, num_workers=0)

    for split, loader in loaders.items():
        imgs, dms, counts = next(iter(loader))
        pred_counts = dms.sum(dim=(1, 2, 3))
        print(f"\n--- {split} ---")
        print(f"  images      : {tuple(imgs.shape)} {imgs.dtype}")
        print(f"  density_maps: {tuple(dms.shape)} {dms.dtype}")
        print(f"  gt counts   : {counts}")
        print(f"  dm counts   : {[f'{c:.1f}' for c in pred_counts.tolist()]}")
        print(f"  total batches: {len(loader)}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    smoke_test()