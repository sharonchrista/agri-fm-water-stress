"""
HSI reprocessing script for agri_foundation.

Problems found in existing processed arrays:
  - cv_X_train/test processed : 100% NaN (completely corrupt)
  - pm_X processed            : 100% NaN (completely corrupt)
  - gn_X processed            : clean, but labels were 1/2 not 0/1
  - Band count mismatch       : crop variety and groundnut have 300 bands,
                                pearl millet has 282 bands

Strategy:
  1. Load all datasets from RAW arrays
  2. Fix sparse NaN via per-band linear interpolation across sample axis
  3. Align to 282 bands by dropping 18 lowest-quality bands from 300-band datasets
     (lowest quality = highest NaN count before interpolation)
  4. Remap labels to 0-indexed integers
  5. Recreate stratified train/test splits for pearl millet
  6. Save as float32 to processed/hsi/ replacing corrupt files

Run: python reprocess_hsi.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path("~/agri_foundation/data").expanduser()
HSI_PROC_DIR = DATA_ROOT / "processed" / "hsi"

CV_RAW_DIR = DATA_ROOT / "hyperspectral" / "crop_variety" / "Crop_dataset"
GN_RAW_DIR = DATA_ROOT / "hyperspectral" / "groundnut_stress" / "GROUNDNUT WATER STRESS DATA"
PM_RAW_DIR = DATA_ROOT / "hyperspectral" / "pearl_millet_stress"

TARGET_BANDS = 282       # canonical band count — matches pearl millet raw
TEST_SIZE = 0.2          # train/test split ratio for pearl millet
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# NaN interpolation
# ---------------------------------------------------------------------------

def interpolate_nan_bands(X: np.ndarray) -> np.ndarray:
    """
    Fix sparse NaN values in a hyperspectral array via linear interpolation
    along the band axis for each affected pixel independently.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, H, W, C) or (N, C) — float array with sparse NaN values.

    Returns
    -------
    np.ndarray same shape, NaN replaced by interpolated values.
    Float32 output.
    """
    original_shape = X.shape
    X = X.astype(np.float64)

    # Flatten spatial dims: (N, H, W, C) -> (N*H*W, C) or (N, C) -> (N, C)
    if X.ndim == 4:
        N, H, W, C = X.shape
        flat = X.reshape(-1, C)
    elif X.ndim == 2:
        flat = X.copy()
        C = X.shape[1]
    else:
        raise ValueError(f"Unexpected shape {X.shape}")

    n_pixels = flat.shape[0]
    band_indices = np.arange(C)

    nan_pixels = np.where(np.isnan(flat).any(axis=1))[0]
    print(f"  Interpolating {len(nan_pixels)} pixels with NaN values...")

    for pixel_idx in nan_pixels:
        spectrum = flat[pixel_idx]
        nan_mask = np.isnan(spectrum)
        valid_mask = ~nan_mask

        if valid_mask.sum() < 2:
            # Too few valid bands — fill with column mean
            flat[pixel_idx, nan_mask] = np.nanmean(flat[:, nan_mask], axis=0)
            continue

        # Linear interpolation using valid bands as anchor points
        flat[pixel_idx, nan_mask] = np.interp(
            band_indices[nan_mask],
            band_indices[valid_mask],
            spectrum[valid_mask],
        )

    result = flat.reshape(original_shape)

    # Verify no NaN remain
    remaining_nan = np.isnan(result).sum()
    if remaining_nan > 0:
        print(f"  WARNING: {remaining_nan} NaN values remain after interpolation")
        # Fallback: replace any remaining NaN with band mean
        band_means = np.nanmean(result.reshape(-1, result.shape[-1]), axis=0)
        nan_locs = np.isnan(result)
        result[nan_locs] = np.take(band_means, np.where(nan_locs)[-1])

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Band selection: align 300-band datasets to 282 bands
# ---------------------------------------------------------------------------

def select_bands(
    X: np.ndarray,
    nan_counts_per_band: np.ndarray,
    target_bands: int = TARGET_BANDS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the target_bands lowest-NaN bands from a 300-band array.
    Returns (selected_X, selected_band_indices).
    """
    current_bands = X.shape[-1]
    if current_bands == target_bands:
        return X, np.arange(target_bands)

    # Keep bands with fewest NaN values (most reliable spectral information)
    selected_indices = np.argsort(nan_counts_per_band)[:target_bands]
    selected_indices = np.sort(selected_indices)  # preserve wavelength order

    if X.ndim == 4:
        return X[:, :, :, selected_indices], selected_indices
    elif X.ndim == 2:
        return X[:, selected_indices], selected_indices
    else:
        raise ValueError(f"Unexpected shape {X.shape}")


# ---------------------------------------------------------------------------
# Dataset-specific processing functions
# ---------------------------------------------------------------------------

def process_crop_variety() -> None:
    """Process crop variety: fix NaN, select 282 bands, remap labels 1-10 -> 0-9."""
    print("\n=== Processing Crop Variety ===")

    X_train_raw = np.load(CV_RAW_DIR / "X_train.npy")
    X_test_raw = np.load(CV_RAW_DIR / "X_test.npy")
    y_train_raw = np.load(CV_RAW_DIR / "y_train.npy")
    y_test_raw = np.load(CV_RAW_DIR / "y_test.npy")

    print(f"  Raw train: {X_train_raw.shape}, test: {X_test_raw.shape}")
    print(f"  Raw label range: {y_train_raw.min():.0f} - {y_train_raw.max():.0f}")

    # Compute NaN counts per band on train set (before interpolation)
    nan_per_band = np.isnan(X_train_raw).sum(axis=(0, 1, 2))

    # Fix NaN via interpolation
    print("  Fixing NaN in train set...")
    X_train = interpolate_nan_bands(X_train_raw)
    print("  Fixing NaN in test set...")
    X_test = interpolate_nan_bands(X_test_raw)

    # Select 282 best bands
    print(f"  Selecting {TARGET_BANDS} bands from {X_train_raw.shape[-1]}...")
    X_train, selected_bands = select_bands(X_train, nan_per_band, TARGET_BANDS)
    X_test, _ = select_bands(X_test, nan_per_band, TARGET_BANDS)
    print(f"  Selected band indices (first 10): {selected_bands[:10].tolist()}")

    # Remap labels 1-10 -> 0-9
    y_train = (y_train_raw - 1).astype(np.int64)
    y_test = (y_test_raw - 1).astype(np.int64)
    print(f"  Remapped label range: {y_train.min()} - {y_train.max()}")

    # Verify
    assert not np.isnan(X_train).any(), "NaN remains in train"
    assert not np.isnan(X_test).any(), "NaN remains in test"
    assert X_train.shape[-1] == TARGET_BANDS
    assert X_test.shape[-1] == TARGET_BANDS

    # Save
    np.save(HSI_PROC_DIR / "cv_X_train.npy", X_train)
    np.save(HSI_PROC_DIR / "cv_X_test.npy", X_test)
    np.save(HSI_PROC_DIR / "cv_y_train.npy", y_train)
    np.save(HSI_PROC_DIR / "cv_y_test.npy", y_test)
    np.save(HSI_PROC_DIR / "cv_selected_bands.npy", selected_bands)

    print(f"  Saved: cv_X_train {X_train.shape}, cv_X_test {X_test.shape}")
    print(f"  Label distribution train: {dict(zip(*np.unique(y_train, return_counts=True)))}")


def process_groundnut() -> None:
    """Process groundnut: select 282 bands, remap labels 1/2 -> 0/1, keep existing splits."""
    print("\n=== Processing Groundnut Stress ===")

    X_raw = np.load(GN_RAW_DIR / "X_GN_31Dec.npy")   # (16667, 300) float64
    y_raw = np.load(GN_RAW_DIR / "y_GN_31Dec.npy")   # (16667,) float64, values 1.0/2.0

    print(f"  Raw shape: {X_raw.shape}, labels unique: {np.unique(y_raw).tolist()}")

    # No NaN in groundnut — verified earlier
    assert np.isnan(X_raw).sum() == 0, "Unexpected NaN in groundnut raw"

    # Compute per-band NaN counts (all zero here but use same pipeline for consistency)
    nan_per_band = np.isnan(X_raw).sum(axis=0)

    # Reshape to patch format (N, 1, 1, 300) then select bands
    X = X_raw.astype(np.float32).reshape(-1, 1, 1, 300)
    X, selected_bands = select_bands(X, nan_per_band, TARGET_BANDS)
    # Reshape back to flat for groundnut convention
    X_flat = X.reshape(-1, TARGET_BANDS)

    # Remap labels 1/2 -> 0/1
    y = (y_raw - 1).astype(np.int64)
    print(f"  Remapped labels: {np.unique(y).tolist()}")

    # Recreate train/test split using same indices as existing processed files
    # Use stratified split to match original 80/20 ratio
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(X_flat, y))

    assert not np.isnan(X_flat).any(), "NaN in groundnut processed"
    assert X_flat.shape[-1] == TARGET_BANDS

    # Save flat version (groundnut convention) and patch version
    np.save(HSI_PROC_DIR / "gn_X_flat.npy", X_flat)
    np.save(HSI_PROC_DIR / "gn_X_patch.npy", X.astype(np.float32))
    np.save(HSI_PROC_DIR / "gn_y.npy", y)
    np.save(HSI_PROC_DIR / "gn_train_idx.npy", train_idx)
    np.save(HSI_PROC_DIR / "gn_test_idx.npy", test_idx)

    print(f"  Saved: gn_X_flat {X_flat.shape}, gn_X_patch {X.shape}")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")


def process_pearl_millet() -> None:
    """Process pearl millet: fix NaN, recreate stratified train/test splits."""
    print("\n=== Processing Pearl Millet Stress ===")

    pm_files = list(PM_RAW_DIR.rglob("X_all_25_pm.npy"))
    pm_y_files = list(PM_RAW_DIR.rglob("y_all_25_pm.npy"))

    if not pm_files:
        raise FileNotFoundError(f"Pearl millet raw X not found under {PM_RAW_DIR}")
    if not pm_y_files:
        raise FileNotFoundError(f"Pearl millet raw y not found under {PM_RAW_DIR}")

    X_raw = np.load(pm_files[0])    # (3150, 11, 11, 282)
    y_raw = np.load(pm_y_files[0])  # (3150,) int32

    print(f"  Raw shape: {X_raw.shape}, labels unique: {np.unique(y_raw).tolist()}")
    print(f"  NaN count: {np.isnan(X_raw).sum()}")

    # Fix NaN
    print("  Fixing NaN...")
    X = interpolate_nan_bands(X_raw)

    # Labels already 0/1 — just ensure int64
    y = y_raw.astype(np.int64)

    # Stratified train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(X.reshape(len(X), -1), y))

    assert not np.isnan(X).any(), "NaN remains in pearl millet"
    assert X.shape[-1] == TARGET_BANDS

    np.save(HSI_PROC_DIR / "pm_X.npy", X)
    np.save(HSI_PROC_DIR / "pm_y.npy", y)
    np.save(HSI_PROC_DIR / "pm_train_idx.npy", train_idx)
    np.save(HSI_PROC_DIR / "pm_test_idx.npy", test_idx)

    print(f"  Saved: pm_X {X.shape}")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_all() -> None:
    """Final check — load all processed files and confirm no NaN, correct shapes."""
    print("\n=== Final Verification ===")

    checks = [
        ("cv_X_train.npy", (2735, 11, 11, TARGET_BANDS)),
        ("cv_X_test.npy",  (4464, 11, 11, TARGET_BANDS)),
        ("gn_X_flat.npy",  (16667, TARGET_BANDS)),
        ("gn_X_patch.npy", (16667, 1, 1, TARGET_BANDS)),
        ("pm_X.npy",       (3150, 11, 11, TARGET_BANDS)),
    ]

    all_good = True
    for fname, expected_shape in checks:
        arr = np.load(HSI_PROC_DIR / fname)
        nan_count = np.isnan(arr).sum()
        shape_ok = arr.shape == expected_shape
        nan_ok = nan_count == 0
        status = "OK" if (shape_ok and nan_ok) else "FAIL"
        print(f"  [{status}] {fname}: shape={arr.shape} nan={nan_count}")
        if not shape_ok:
            print(f"         expected shape {expected_shape}")
        if not nan_ok:
            all_good = False

    label_checks = [
        ("cv_y_train.npy", 0, 9),
        ("cv_y_test.npy",  0, 9),
        ("gn_y.npy",       0, 1),
        ("pm_y.npy",       0, 1),
    ]
    for fname, expected_min, expected_max in label_checks:
        arr = np.load(HSI_PROC_DIR / fname)
        label_ok = arr.min() == expected_min and arr.max() == expected_max
        status = "OK" if label_ok else "FAIL"
        print(f"  [{status}] {fname}: range=[{arr.min()}, {arr.max()}] expected=[{expected_min}, {expected_max}]")
        if not label_ok:
            all_good = False

    if all_good:
        print("\nAll checks passed — datasets ready for training.")
    else:
        print("\nSome checks FAILED — review output above.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    HSI_PROC_DIR.mkdir(parents=True, exist_ok=True)

    process_crop_variety()
    process_groundnut()
    process_pearl_millet()
    verify_all()


if __name__ == "__main__":
    main()