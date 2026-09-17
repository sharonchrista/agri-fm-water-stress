"""
Multispectral dataset loader for agri_foundation.

Covers two unlabelled MS datasets used for self-supervised pretraining:
  - maize : (302, 960, 1280, 5) float32, Blue/Green/Red/RedEdge/NIR
  - paddy : (315, 960, 1280, 5) float32, same band order

These datasets have no crop stress labels. They are used for contrastive
pretraining of the MS encoder branch only — not for supervised classification.

Memory strategy: full arrays are ~7.4 GB combined. We load both into RAM
once at startup (tyrone-hpc has sufficient RAM) and sample random 64x64
spatial tiles on the fly. Each tile is one training sample.

Usage:
    from ms_dataset import build_ms_loaders
    loaders = build_ms_loaders(data_root="~/agri_foundation/data")
    for view_a, view_b in loaders["pretrain"]:
        # view_a, view_b: two augmented views of same tile for contrastive loss
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_SIZE = 64           # spatial crop size (pixels)
NUM_MS_BANDS = 5         # Blue, Green, Red, RedEdge, NIR
MS_BAND_NAMES = ("Blue", "Green", "Red", "RedEdge", "NIR")

# Per-band normalisation statistics computed from dataset range [0.05, 1.0]
# Using dataset min/max as a proxy — replace with actual per-band mean/std
# once computed from the full dataset for production training.
MS_BAND_MEAN = (0.45, 0.45, 0.45, 0.45, 0.45)
MS_BAND_STD = (0.22, 0.22, 0.22, 0.22, 0.22)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _random_tile(
    image: np.ndarray,
    tile_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Extract a random tile_size x tile_size spatial crop from a (H, W, C) image.
    Returns (C, tile_size, tile_size) in CHW order.
    """
    h, w, c = image.shape
    top = rng.integers(0, h - tile_size)
    left = rng.integers(0, w - tile_size)
    tile = image[top:top + tile_size, left:left + tile_size, :]  # (T, T, C)
    return tile.transpose(2, 0, 1)                                # (C, T, T)


def _spectral_dropout(tile: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    """
    Randomly zero out individual bands with probability drop_prob.
    Simulates missing band acquisitions — forces the encoder to learn
    representations robust to partial band availability.
    """
    mask = np.random.rand(tile.shape[0]) > drop_prob
    return tile * mask[:, np.newaxis, np.newaxis]


def _spectral_jitter(tile: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Additive Gaussian noise on all bands."""
    noise = np.random.normal(0.0, sigma, size=tile.shape).astype(np.float32)
    return np.clip(tile + noise, 0.0, 1.0)


def _random_flip(tile: np.ndarray) -> np.ndarray:
    """Random horizontal and vertical flip."""
    if np.random.rand() > 0.5:
        tile = tile[:, :, ::-1].copy()   # horizontal flip on W axis
    if np.random.rand() > 0.5:
        tile = tile[:, ::-1, :].copy()   # vertical flip on H axis
    return tile


def _normalise(tile: np.ndarray) -> np.ndarray:
    """Per-band standardisation using dataset statistics."""
    mean = np.array(MS_BAND_MEAN, dtype=np.float32)[:, np.newaxis, np.newaxis]
    std = np.array(MS_BAND_STD, dtype=np.float32)[:, np.newaxis, np.newaxis]
    return (tile - mean) / (std + 1e-6)


def _augment_view(tile: np.ndarray) -> np.ndarray:
    """Full augmentation pipeline for one contrastive view."""
    tile = _random_flip(tile)
    tile = _spectral_jitter(tile, sigma=0.02)
    tile = _spectral_dropout(tile, drop_prob=0.05)
    tile = _normalise(tile)
    return tile


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultispectralTileDataset(Dataset):
    """
    Tile-based multispectral dataset for self-supervised contrastive pretraining.

    Each __getitem__ call samples a random 64x64 tile from a random image
    and returns two independently augmented views of the same tile.
    The contrastive objective maximises agreement between the two views.

    Parameters
    ----------
    arrays : list of np.ndarray
        Each array is (N, H, W, C) float32. Multiple arrays (maize + paddy)
        are concatenated along the image axis at init time.
    tiles_per_epoch : int
        Virtual epoch size — number of tiles to sample per epoch.
        Since we sample randomly, this controls training iterations per epoch.
    seed : int
        Base random seed for reproducible tile sampling.
    """

    def __init__(
        self,
        arrays: list[np.ndarray],
        tiles_per_epoch: int = 10000,
        seed: int = 42,
    ) -> None:
        # Concatenate along image axis: (N_total, H, W, C)
        self.data = np.concatenate(arrays, axis=0)
        self.n_images, self.H, self.W, self.C = self.data.shape
        self.tiles_per_epoch = tiles_per_epoch
        self.rng = np.random.default_rng(seed)

        print(
            f"MultispectralTileDataset: {self.n_images} images "
            f"({self.H}x{self.W}x{self.C}), "
            f"{tiles_per_epoch} tiles/epoch"
        )

    def __len__(self) -> int:
        return self.tiles_per_epoch

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Sample a random image index — idx is ignored (virtual epoch)
        img_idx = self.rng.integers(0, self.n_images)
        image = self.data[img_idx]   # (H, W, C)

        # Extract a random tile — same spatial location, two different augmentations
        tile = _random_tile(image, TILE_SIZE, self.rng)   # (C, T, T)
        tile = tile.astype(np.float32)

        view_a = _augment_view(tile.copy())
        view_b = _augment_view(tile.copy())

        return (
            torch.from_numpy(view_a),
            torch.from_numpy(view_b),
        )


class MultispectralSupervisedDataset(Dataset):
    """
    Deterministic tile dataset for evaluation / downstream supervised tasks.
    Tiles the full image set with a fixed stride — no randomness.
    Used to extract MS features for linear probing or fine-tuning.

    Parameters
    ----------
    arrays : list of np.ndarray
        Each array is (N, H, W, C) float32.
    stride : int
        Tile extraction stride. stride=TILE_SIZE gives non-overlapping tiles.
    """

    def __init__(
        self,
        arrays: list[np.ndarray],
        stride: int = TILE_SIZE,
    ) -> None:
        self.data = np.concatenate(arrays, axis=0)
        self.n_images, self.H, self.W, self.C = self.data.shape
        self.stride = stride
        self.tiles = self._index_tiles()
        print(
            f"MultispectralSupervisedDataset: {self.n_images} images -> "
            f"{len(self.tiles)} non-overlapping tiles"
        )

    def _index_tiles(self) -> list[tuple[int, int, int]]:
        """Pre-compute all (image_idx, top, left) tile positions."""
        tiles = []
        for img_idx in range(self.n_images):
            for top in range(0, self.H - TILE_SIZE + 1, self.stride):
                for left in range(0, self.W - TILE_SIZE + 1, self.stride):
                    tiles.append((img_idx, top, left))
        return tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Tensor:
        img_idx, top, left = self.tiles[idx]
        tile = self.data[img_idx, top:top + TILE_SIZE, left:left + TILE_SIZE, :]
        tile = tile.astype(np.float32).transpose(2, 0, 1)   # (C, T, T)
        tile = _normalise(tile)
        return torch.from_numpy(tile)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _load_ms_arrays(data_root: Path) -> dict[str, np.ndarray]:
    """Load stacked MS arrays for maize and paddy."""
    ms_dir = data_root / "processed" / "ms"
    crops = {}

    for crop in ("maize", "paddy"):
        crop_dirs = list(ms_dir.glob(f"{crop}*"))
        if not crop_dirs:
            # Try nested folder with sensor name in path
            crop_dirs = [ms_dir / crop]
        if not crop_dirs:
            raise FileNotFoundError(f"MS directory for {crop} not found in {ms_dir}")

        # Find ms_stacked.npy — may be inside a sensor-named subfolder
        npy_candidates = list(crop_dirs[0].rglob("ms_stacked.npy"))
        if not npy_candidates:
            raise FileNotFoundError(
                f"ms_stacked.npy not found under {crop_dirs[0]}"
            )

        print(f"Loading {crop} MS array from {npy_candidates[0]} ...")
        crops[crop] = np.load(npy_candidates[0])   # (N, H, W, 5)
        print(f"  {crop}: {crops[crop].shape}")

    return crops


def build_ms_loaders(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    tiles_per_epoch: int = 10000,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Build MS DataLoaders for contrastive pretraining and evaluation.

    Returns
    -------
    {
        "pretrain"  : DataLoader yielding (view_a, view_b) tile pairs,
        "maize_eval": DataLoader yielding single tiles from maize only,
        "paddy_eval": DataLoader yielding single tiles from paddy only,
    }
    """
    root = Path(data_root).expanduser()
    arrays = _load_ms_arrays(root)

    pretrain_dataset = MultispectralTileDataset(
        arrays=list(arrays.values()),
        tiles_per_epoch=tiles_per_epoch,
        seed=seed,
    )

    maize_eval = MultispectralSupervisedDataset([arrays["maize"]], stride=TILE_SIZE)
    paddy_eval = MultispectralSupervisedDataset([arrays["paddy"]], stride=TILE_SIZE)

    return {
        "pretrain": DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "maize_eval": DataLoader(
            maize_eval,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "paddy_eval": DataLoader(
            paddy_eval,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


# ---------------------------------------------------------------------------
# Band statistics computation
# ---------------------------------------------------------------------------

def compute_band_statistics(
    data_root: str | Path,
    sample_fraction: float = 0.1,
) -> None:
    """
    Compute per-band mean and std from a random sample of tiles.
    Run once to replace the placeholder MS_BAND_MEAN / MS_BAND_STD constants.
    """
    root = Path(data_root).expanduser()
    arrays = _load_ms_arrays(root)
    combined = np.concatenate(list(arrays.values()), axis=0)

    n_sample = max(1, int(len(combined) * sample_fraction))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(combined), size=n_sample, replace=False)
    sample = combined[indices]   # (n_sample, H, W, 5)

    # Reshape to (n_samples * H * W, 5) for per-band stats
    flat = sample.reshape(-1, sample.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)

    print("Per-band statistics (use these to replace placeholders):")
    for i, name in enumerate(MS_BAND_NAMES):
        print(f"  {name:8s}: mean={mean[i]:.4f}  std={std[i]:.4f}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(data_root: str = "~/agri_foundation/data") -> None:
    """
    Quick sanity check. Run directly: python ms_dataset.py
    Also computes actual band statistics.
    """
    print("Computing band statistics first...")
    compute_band_statistics(data_root, sample_fraction=0.05)

    print("\nBuilding MS loaders...")
    loaders = build_ms_loaders(
        data_root=data_root,
        batch_size=8,
        num_workers=0,
        tiles_per_epoch=100,
    )

    print("\n--- pretrain (contrastive pairs) ---")
    view_a, view_b = next(iter(loaders["pretrain"]))
    print(f"  view_a: {tuple(view_a.shape)} dtype={view_a.dtype}")
    print(f"  view_b: {tuple(view_b.shape)} dtype={view_b.dtype}")
    print(f"  range a: [{view_a.min():.3f}, {view_a.max():.3f}]")

    print("\n--- maize_eval (deterministic tiles) ---")
    tiles = next(iter(loaders["maize_eval"]))
    print(f"  tiles: {tuple(tiles.shape)} dtype={tiles.dtype}")
    print(f"  total maize eval tiles: {len(loaders['maize_eval'].dataset)}")

    print("\n--- paddy_eval (deterministic tiles) ---")
    tiles = next(iter(loaders["paddy_eval"]))
    print(f"  tiles: {tuple(tiles.shape)} dtype={tiles.dtype}")
    print(f"  total paddy eval tiles: {len(loaders['paddy_eval'].dataset)}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    smoke_test()