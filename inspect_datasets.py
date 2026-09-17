"""
Dataset inspection script for agri_foundation.
Run from the project root: python inspect_datasets.py
Prints shapes, dtypes, label distributions, and value ranges
for all preprocessed numpy arrays across HSI, MS, and RGB modalities.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Config — adjust BASE_DIR to match your server mount point
# ---------------------------------------------------------------------------
BASE_DIR = Path("~/agri_foundation/data").expanduser()

HSI_RAW_DIR = BASE_DIR / "hyperspectral"
HSI_PROC_DIR = BASE_DIR / "processed" / "hsi"
MS_PROC_DIR = BASE_DIR / "processed" / "ms"
META_DIR = BASE_DIR / "processed" / "metadata"
RGB_DIR = BASE_DIR / "rgb_paddy"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npy(path: Path) -> np.ndarray | None:
    """Load a .npy file, returning None and printing a warning on failure."""
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"  WARNING: could not load {path.name} — {exc}")
        return None


def describe_array(arr: np.ndarray, name: str) -> None:
    """Print shape, dtype, min/max and — for label arrays — class distribution."""
    print(f"  {name}")
    print(f"    shape : {arr.shape}")
    print(f"    dtype : {arr.dtype}")

    # String arrays have no numeric range — show unique samples instead
    if np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.bytes_):
        unique_vals = np.unique(arr.ravel())
        preview = list(unique_vals[:8])
        print(f"    unique ({len(unique_vals)} total, first 8) : {preview}")
        return

    print(f"    range : [{arr.min():.4f}, {arr.max():.4f}]")
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        flat = arr.ravel().astype(int) if np.issubdtype(arr.dtype, np.integer) else arr.ravel()
        if np.issubdtype(arr.dtype, np.integer) or len(np.unique(flat)) <= 20:
            unique, counts = np.unique(flat, return_counts=True)
            dist = {int(u): int(c) for u, c in zip(unique, counts)}
            print(f"    classes : {dist}")


def section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"  WARNING: could not read {path.name} — {exc}")
        return None


# ---------------------------------------------------------------------------
# 1. HSI — Raw splits (crop variety, groundnut, pearl millet)
# ---------------------------------------------------------------------------

def inspect_hsi_raw() -> None:
    section("HSI RAW SPLITS")

    datasets = {
        "crop_variety": HSI_RAW_DIR / "crop_variety" / "Crop_dataset",
        "groundnut_stress": HSI_RAW_DIR / "groundnut_stress" / "GROUNDNUT WATER STRESS",
        "pearl_millet_stress": HSI_RAW_DIR / "pearl_millet_stress",
    }

    for name, folder in datasets.items():
        print(f"\n--- {name} ({folder}) ---")
        if not folder.exists():
            print(f"  Directory not found: {folder}")
            continue

        npy_files = sorted(folder.glob("*.npy"))
        if not npy_files:
            print("  No .npy files found.")
            continue

        for fpath in npy_files:
            arr = load_npy(fpath)
            if arr is not None:
                describe_array(arr, fpath.name)

        # Print any readme/docx names for manual follow-up
        docs = list(folder.glob("*.docx")) + list(folder.glob("*.txt")) + list(folder.glob("*.pdf"))
        if docs:
            print(f"  Documentation found: {[d.name for d in docs]}")


# ---------------------------------------------------------------------------
# 2. HSI — Processed splits
# ---------------------------------------------------------------------------

def inspect_hsi_processed() -> None:
    section("HSI PROCESSED (processed/hsi/)")

    if not HSI_PROC_DIR.exists():
        print(f"  Directory not found: {HSI_PROC_DIR}")
        return

    file_groups = {
        "crop_variety": ["cv_X_train.npy", "cv_X_test.npy", "cv_y_train.npy", "cv_y_test.npy"],
        "groundnut": [
            "gn_X_flat.npy", "gn_X_patch.npy", "gn_y.npy",
            "gn_train_idx.npy", "gn_test_idx.npy",
        ],
        "pearl_millet": ["pm_X.npy", "pm_y.npy", "pm_train_idx.npy", "pm_test_idx.npy"],
    }

    for group, files in file_groups.items():
        print(f"\n--- {group} ---")
        for fname in files:
            fpath = HSI_PROC_DIR / fname
            if not fpath.exists():
                print(f"  {fname} : NOT FOUND")
                continue
            arr = load_npy(fpath)
            if arr is not None:
                describe_array(arr, fname)


# ---------------------------------------------------------------------------
# 3. Multispectral — Processed stacked arrays
# ---------------------------------------------------------------------------

def inspect_ms_processed() -> None:
    section("MULTISPECTRAL PROCESSED (processed/ms/)")

    if not MS_PROC_DIR.exists():
        print(f"  Directory not found: {MS_PROC_DIR}")
        return

    crops = {
        "maize": MS_PROC_DIR / "maize",
        "paddy": MS_PROC_DIR / "paddy",
    }

    for crop, folder in crops.items():
        print(f"\n--- {crop} ---")
        if not folder.exists():
            print(f"  Directory not found: {folder}")
            continue

        for fname in ["ms_stacked.npy", "stacked_ids.npy"]:
            fpath = folder / fname
            # Folder name may contain the full sensor string — search recursively
            matches = list(folder.rglob(fname))
            if not matches:
                print(f"  {fname} : NOT FOUND")
                continue
            arr = load_npy(matches[0])
            if arr is not None:
                describe_array(arr, fname)


# ---------------------------------------------------------------------------
# 4. Metadata JSONs
# ---------------------------------------------------------------------------

def inspect_metadata() -> None:
    section("METADATA JSONs (processed/metadata/)")

    if not META_DIR.exists():
        print(f"  Directory not found: {META_DIR}")
        return

    for jpath in sorted(META_DIR.glob("*.json")):
        data = load_json(jpath)
        if data is not None:
            print(f"\n--- {jpath.name} ---")
            # Print top-level keys and values (truncated for readability)
            for key, val in data.items():
                if isinstance(val, (dict, list)) and len(str(val)) > 120:
                    print(f"  {key}: [{type(val).__name__}, {len(val)} items]")
                else:
                    print(f"  {key}: {val}")


# ---------------------------------------------------------------------------
# 5. RGB Paddy — folder structure and image counts
# ---------------------------------------------------------------------------

def inspect_rgb_paddy() -> None:
    section("RGB PADDY (rgb_paddy/)")

    if not RGB_DIR.exists():
        print(f"  Directory not found: {RGB_DIR}")
        return

    for split in ["train", "test"]:
        split_dir = RGB_DIR / split
        if not split_dir.exists():
            print(f"  {split}/ not found")
            continue

        class_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
        print(f"\n--- {split} ---")
        if not class_dirs:
            # Flat folder — just count images
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")) + \
                     list(split_dir.glob("*.tif")) + list(split_dir.glob("*.jpeg"))
            print(f"  Flat folder: {len(images)} images")
        else:
            # Class-structured
            total = 0
            for cls_dir in class_dirs:
                images = list(cls_dir.glob("*"))
                count = len([f for f in images if f.is_file()])
                print(f"  class '{cls_dir.name}': {count} images")
                total += count
            print(f"  Total: {total} images across {len(class_dirs)} classes")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Base directory: {BASE_DIR}")
    print(f"Exists: {BASE_DIR.exists()}")

    inspect_hsi_raw()
    inspect_hsi_processed()
    inspect_ms_processed()
    inspect_metadata()
    inspect_rgb_paddy()

    print("\n" + "=" * 60)
    print("  Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()