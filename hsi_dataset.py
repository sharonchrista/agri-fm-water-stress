"""
HSI dataset loader for agri_foundation.

Covers three hyperspectral tasks:
  - crop_variety        : 10-class classification, patches (N, 11, 11, 282)
  - groundnut_stress    : binary stress, patches (N, 1, 1, 282) -> reshaped to (N, 11, 11, 282)
  - pearl_millet_stress : binary stress, patches (N, 11, 11, 282)

All three are unified into a single HSIDataset class that returns
(patch_tensor, label_tensor) with consistent shape (282, 11, 11) in CHW order
for compatibility with Conv2d-based and ViT-based encoders.

Usage:
    from hsi_dataset import HSIDataset, build_hsi_loaders
    loaders = build_hsi_loaders(data_root="~/agri_foundation/data")
    for patches, labels in loaders["crop_variety"]["train"]:
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# Canonical patch spatial size used across all HSI datasets.
# Groundnut (1x1) is zero-padded to this size so all datasets share one encoder.
PATCH_SIZE = 11
NUM_BANDS = 282

TaskName = Literal["crop_variety", "groundnut_stress", "pearl_millet_stress"]


class HSIDataset(Dataset):
    """
    Unified hyperspectral patch dataset.

    Parameters
    ----------
    patches : np.ndarray
        Shape (N, H, W, C) or (N, C) — any spatial size, any number of bands.
    labels : np.ndarray
        Shape (N,) — integer class labels.
    indices : np.ndarray | None
        Optional index subset for train/test splitting when a shared array
        is split by index file rather than separate X files (groundnut, millet).
    augment : bool
        Whether to apply spectral jitter augmentation during training.
    """

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        indices: Optional[np.ndarray] = None,
        augment: bool = False,
    ) -> None:
        if indices is not None:
            patches = patches[indices]
            labels = labels[indices]

        self.patches = patches
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        patch = self.patches[idx].astype(np.float32)  # (H, W, C) or (C,)
        label = self.labels[idx]

        patch = self._ensure_spatial(patch)   # -> (H, W, C)
        patch = self._pad_to_target(patch)    # -> (PATCH_SIZE, PATCH_SIZE, C)
        patch = self._to_chw(patch)           # -> (C, H, W) = (282, 11, 11)

        if self.augment:
            patch = self._spectral_jitter(patch)

        return torch.from_numpy(patch), torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    # Internal transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_spatial(patch: np.ndarray) -> np.ndarray:
        """Convert flat (C,) or squeezed (1,1,C) patches to (H,W,C)."""
        if patch.ndim == 1:
            # Flat vector -> (1, 1, C)
            return patch[np.newaxis, np.newaxis, :]
        if patch.ndim == 3:
            return patch
        raise ValueError(f"Unexpected patch ndim={patch.ndim}, shape={patch.shape}")

    @staticmethod
    def _pad_to_target(patch: np.ndarray) -> np.ndarray:
        """
        Zero-pad spatial dimensions to PATCH_SIZE x PATCH_SIZE.
        Padding is symmetric; no cropping — caller must ensure patch <= target.
        """
        h, w, c = patch.shape
        if h == PATCH_SIZE and w == PATCH_SIZE:
            return patch

        pad_h = PATCH_SIZE - h
        pad_w = PATCH_SIZE - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return np.pad(
            patch,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

    @staticmethod
    def _to_chw(patch: np.ndarray) -> np.ndarray:
        """Transpose (H, W, C) to (C, H, W) for PyTorch Conv2d."""
        return np.transpose(patch, (2, 0, 1))

    @staticmethod
    def _spectral_jitter(patch: np.ndarray) -> np.ndarray:
        """
        Mild spectral augmentation: additive Gaussian noise on band axis.
        Kept subtle (std=0.01) to avoid distorting spectral signatures.
        """
        noise = np.random.normal(0.0, 0.01, size=patch.shape).astype(np.float32)
        return np.clip(patch + noise, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Class weight utility for imbalanced tasks
    # ------------------------------------------------------------------

    def class_weights(self) -> Tensor:
        """Inverse-frequency weights over classes for weighted loss or sampler."""
        unique, counts = np.unique(self.labels, return_counts=True)
        freq = counts / counts.sum()
        weights = 1.0 / (freq + 1e-6)
        weights = weights / weights.sum()
        # Map class index -> weight for every sample
        weight_per_sample = np.array([weights[lbl] for lbl in self.labels], dtype=np.float32)
        return torch.from_numpy(weight_per_sample)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _load_crop_variety(proc_dir: Path, augment_train: bool) -> dict[str, HSIDataset]:
    """Load pre-split crop variety train/test arrays."""
    X_train = np.load(proc_dir / "cv_X_train.npy")
    y_train = np.load(proc_dir / "cv_y_train.npy")
    X_test = np.load(proc_dir / "cv_X_test.npy")
    y_test = np.load(proc_dir / "cv_y_test.npy")

    return {
        "train": HSIDataset(X_train, y_train, augment=augment_train),
        "test": HSIDataset(X_test, y_test, augment=False),
    }


def _load_stress_dataset(
    proc_dir: Path,
    x_file: str,
    y_file: str,
    train_idx_file: str,
    test_idx_file: str,
    augment_train: bool,
) -> dict[str, HSIDataset]:
    """Load index-split stress dataset (groundnut or pearl millet)."""
    X = np.load(proc_dir / x_file)
    y = np.load(proc_dir / y_file)
    train_idx = np.load(proc_dir / train_idx_file)
    test_idx = np.load(proc_dir / test_idx_file)

    return {
        "train": HSIDataset(X, y, indices=train_idx, augment=augment_train),
        "test": HSIDataset(X, y, indices=test_idx, augment=False),
    }


def build_hsi_datasets(
    data_root: str | Path,
    augment_train: bool = True,
) -> dict[str, dict[str, HSIDataset]]:
    """
    Build all available HSI datasets.

    Skips tasks whose processed arrays are missing or corrupted (all-NaN).
    Returns only tasks that are clean and ready for training.

    Returns
    -------
    dict with available task keys mapping to {"train": HSIDataset, "test": HSIDataset}.
    """
    proc_dir = Path(data_root).expanduser() / "processed" / "hsi"

    if not proc_dir.exists():
        raise FileNotFoundError(f"Processed HSI directory not found: {proc_dir}")

    datasets: dict[str, dict[str, HSIDataset]] = {}

    # Crop variety — requires pre-split arrays
    cv_train = proc_dir / "cv_X_train.npy"
    if cv_train.exists():
        arr = np.load(cv_train)
        if not np.isnan(arr).all() and not np.isinf(arr).any():
            datasets["crop_variety"] = _load_crop_variety(proc_dir, augment_train)
            print("  crop_variety: loaded")
        else:
            print("  crop_variety: SKIPPED (corrupted arrays)")
    else:
        print("  crop_variety: SKIPPED (files not found)")

    # Groundnut stress
    gn_patch = proc_dir / "gn_X_patch.npy"
    if gn_patch.exists():
        arr = np.load(gn_patch)
        if not np.isnan(arr).all() and not np.isinf(arr).any():
            datasets["groundnut_stress"] = _load_stress_dataset(
                proc_dir,
                x_file="gn_X_patch.npy",
                y_file="gn_y.npy",
                train_idx_file="gn_train_idx.npy",
                test_idx_file="gn_test_idx.npy",
                augment_train=augment_train,
            )
            print("  groundnut_stress: loaded")
        else:
            print("  groundnut_stress: SKIPPED (corrupted arrays)")
    else:
        print("  groundnut_stress: SKIPPED (files not found)")

    # Pearl millet stress
    pm = proc_dir / "pm_X.npy"
    if pm.exists():
        arr = np.load(pm)
        if not np.isnan(arr).all() and not np.isinf(arr).any():
            datasets["pearl_millet_stress"] = _load_stress_dataset(
                proc_dir,
                x_file="pm_X.npy",
                y_file="pm_y.npy",
                train_idx_file="pm_train_idx.npy",
                test_idx_file="pm_test_idx.npy",
                augment_train=augment_train,
            )
            print("  pearl_millet_stress: loaded")
        else:
            print("  pearl_millet_stress: SKIPPED (corrupted arrays)")
    else:
        print("  pearl_millet_stress: SKIPPED (files not found)")

    if not datasets:
        raise RuntimeError("No clean HSI datasets found. Check processed/hsi/ directory.")

    return datasets


def build_hsi_loaders(
    data_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    augment_train: bool = True,
    use_weighted_sampler: bool = True,
) -> dict[str, dict[str, DataLoader]]:
    """
    Build DataLoaders for all three HSI tasks.

    Weighted sampling is applied to training splits to handle class imbalance.
    Test splits always use sequential sampling.

    Returns
    -------
    Nested dict: loaders[task_name]["train" | "test"] -> DataLoader
    """
    datasets = build_hsi_datasets(data_root, augment_train)
    loaders: dict[str, dict[str, DataLoader]] = {}

    for task_name, splits in datasets.items():
        loaders[task_name] = {}

        for split_name, dataset in splits.items():
            is_train = split_name == "train"

            if is_train and use_weighted_sampler:
                sample_weights = dataset.class_weights()
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    replacement=True,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                )
            else:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=False,
                )

            loaders[task_name][split_name] = loader

    return loaders


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(data_root: str = "~/agri_foundation/data") -> None:
    """
    Quick sanity check — loads one batch from each task and prints shapes.
    Run directly: python hsi_dataset.py
    """
    print("Building HSI loaders...")
    loaders = build_hsi_loaders(
        data_root=data_root,
        batch_size=32,
        num_workers=0,    # 0 workers for quick test; increase for training
        augment_train=True,
        use_weighted_sampler=True,
    )

    for task_name, splits in loaders.items():
        print(f"\n--- {task_name} ---")
        for split_name, loader in splits.items():
            patches, labels = next(iter(loader))
            print(f"  {split_name:5s} | patches: {tuple(patches.shape)} "
                  f"dtype: {patches.dtype} | labels: {tuple(labels.shape)} "
                  f"classes: {labels.unique().tolist()}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    smoke_test()