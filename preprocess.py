"""
=============================================================================
AGRI FOUNDATION — Preprocessing Pipeline v2
=============================================================================
Changes from v1:
  - NaN handling: spectral interpolation imputation (replaces drop)
  - RGB labels: fixed path matching (handles separate label folders)
  - HSI: added NaN pattern diagnostics
  - Crop variety: class distribution analysis across train/test
=============================================================================
"""

import os
import sys
import glob
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import interpolate
from sklearn.model_selection import train_test_split
from collections import Counter

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE        = Path.home() / 'agri_foundation'
DATA        = BASE / 'data'
PROCESSED   = DATA / 'processed'
HSI_DIR     = DATA / 'hyperspectral'
MS_DIR      = DATA / 'multispectral' / 'raw' / 'Agri'
RGB_DIR     = DATA / 'rgb_paddy'

for d in [PROCESSED / 'hsi', PROCESSED / 'ms',
          PROCESSED / 'rgb', PROCESSED / 'metadata']:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = PROCESSED / 'preprocessing_v2_log.txt'
# Clear previous log
if LOG_FILE.exists():
    LOG_FILE.unlink()

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TARGET_BANDS     = 282
PATCH_SIZE       = 11
WAVELENGTHS_300  = np.linspace(385, 1020, 300)
WAVELENGTHS_282  = np.linspace(400, 1000, 282)
MS_BANDS         = {1: 'Blue', 2: 'Green', 3: 'Red', 4: 'RedEdge', 5: 'NIR'}

# ─────────────────────────────────────────────
# UTILITY: NaN DIAGNOSTICS
# ─────────────────────────────────────────────

def nan_diagnostics(X, name):
    """Report NaN pattern — per band and per sample."""
    if X.ndim == 4:
        X_flat = X.reshape(-1, X.shape[-1])
    else:
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X

    total_samples     = len(X_flat)
    nan_per_sample    = np.isnan(X_flat).any(axis=1)
    n_affected        = nan_per_sample.sum()
    nan_per_band      = np.isnan(X_flat).sum(axis=0)
    bands_with_nan    = (nan_per_band > 0).sum()
    nan_counts_sample = np.isnan(X_flat).sum(axis=1)

    log(f"  [{name}] NaN diagnostics:")
    log(f"    Samples with ANY NaN : {n_affected}/{total_samples} ({n_affected/total_samples*100:.1f}%)")
    log(f"    Bands with NaN       : {bands_with_nan}/{X_flat.shape[1]}")
    log(f"    First 10 band NaN ct : {nan_per_band[:10].tolist()}")
    log(f"    Last  10 band NaN ct : {nan_per_band[-10:].tolist()}")
    if n_affected > 0:
        affected = nan_counts_sample[nan_counts_sample > 0]
        log(f"    NaN per affected smp : min={affected.min()}, max={affected.max()}, mean={affected.mean():.1f}")

    return nan_per_band, n_affected

# ─────────────────────────────────────────────
# UTILITY: SPECTRAL INTERPOLATION IMPUTATION
# ─────────────────────────────────────────────

def impute_nan_spectral(X, dataset_name):
    """
    Impute NaN values using spectral interpolation per pixel spectrum.
    Exploits strong spectral continuity in hyperspectral data.

    For each pixel spectrum with NaN bands:
      - Use scipy interp1d on valid bands to fill NaN bands (linear)
      - Extrapolate edge bands where interpolation not possible
      - Fallback to band mean for pixels with too few valid bands (<10)

    X shape: (N, H, W, B) or (N, B)
    Returns: X with NaN imputed, same shape
    """
    shape  = X.shape
    B      = shape[-1]
    X_work = X.reshape(-1, B).copy().astype(np.float64)
    band_idx = np.arange(B, dtype=np.float64)

    # Precompute band means from valid (non-NaN) values for fallback
    band_means = np.nanmean(X_work, axis=0)
    # If a band is ALL NaN across all samples, use 0
    band_means = np.where(np.isnan(band_means), 0.0, band_means)

    n_imputed      = 0
    n_fallback     = 0
    n_clean        = 0
    min_valid_bands = 10  # minimum valid bands required for interpolation

    for i in range(len(X_work)):
        nan_mask = np.isnan(X_work[i])
        if not nan_mask.any():
            n_clean += 1
            continue

        valid_mask = ~nan_mask
        n_valid    = valid_mask.sum()

        if n_valid >= min_valid_bands:
            # Spectral interpolation + extrapolation at edges
            f = interpolate.interp1d(
                band_idx[valid_mask],
                X_work[i][valid_mask],
                kind='linear',
                bounds_error=False,
                fill_value=(
                    X_work[i][valid_mask][0],   # left edge: first valid value
                    X_work[i][valid_mask][-1]   # right edge: last valid value
                )
            )
            X_work[i][nan_mask] = f(band_idx[nan_mask])
            n_imputed += 1
        else:
            # Too few valid bands — use band mean
            X_work[i][nan_mask] = band_means[nan_mask]
            n_fallback += 1

    total = len(X_work)
    log(f"  [{dataset_name}] Imputation complete:")
    log(f"    Clean (no NaN)     : {n_clean}/{total} ({n_clean/total*100:.1f}%)")
    log(f"    Spectral interp    : {n_imputed}/{total} ({n_imputed/total*100:.1f}%)")
    log(f"    Band mean fallback : {n_fallback}/{total} ({n_fallback/total*100:.1f}%)")

    # Verify no NaN remains
    remaining_nan = np.isnan(X_work).sum()
    if remaining_nan > 0:
        log(f"    WARNING: {remaining_nan} NaN values remain — filling with 0")
        X_work = np.nan_to_num(X_work, nan=0.0)
    else:
        log(f"    ✓ No NaN values remain")

    return X_work.reshape(shape).astype(np.float32)


# ─────────────────────────────────────────────
# UTILITY: BAND INTERPOLATION (300 → 282)
# ─────────────────────────────────────────────

def interpolate_bands(X, src_wav, tgt_wav, name):
    if len(src_wav) == len(tgt_wav):
        log(f"  [{name}] Band count matches ({len(src_wav)}) — skipping interpolation")
        return X.astype(np.float32)

    log(f"  [{name}] Resampling bands: {len(src_wav)} → {len(tgt_wav)}")
    shape  = X.shape
    B_in   = shape[-1]
    B_out  = len(tgt_wav)
    X_flat = X.reshape(-1, B_in).astype(np.float64)
    X_out  = np.zeros((len(X_flat), B_out), dtype=np.float32)

    for i in range(len(X_flat)):
        X_out[i] = np.interp(tgt_wav, src_wav, X_flat[i])

    return X_out.reshape(shape[:-1] + (B_out,))


# ─────────────────────────────────────────────
# UTILITY: NORMALISATION
# ─────────────────────────────────────────────

def normalise_hsi(X, name, method='minmax'):
    log(f"  [{name}] Normalising ({method})...")
    shape  = X.shape
    B      = shape[-1]
    X_flat = X.reshape(-1, B).astype(np.float32)

    if method == 'minmax':
        b_min = X_flat.min(axis=0, keepdims=True)
        b_max = X_flat.max(axis=0, keepdims=True)
        denom = b_max - b_min
        denom[denom == 0] = 1e-8
        X_norm = (X_flat - b_min) / denom
        stats  = {'method': 'minmax',
                  'min': b_min.squeeze().tolist(),
                  'max': b_max.squeeze().tolist()}

    elif method == 'zscore':
        b_mean = X_flat.mean(axis=0, keepdims=True)
        b_std  = X_flat.std(axis=0, keepdims=True)
        b_std[b_std == 0] = 1e-8
        X_norm = (X_flat - b_mean) / b_std
        stats  = {'method': 'zscore',
                  'mean': b_mean.squeeze().tolist(),
                  'std':  b_std.squeeze().tolist()}

    return X_norm.reshape(shape).astype(np.float32), stats


def zero_index_labels(y, name):
    y_z    = (np.array(y) - 1).astype(np.int64)
    unique = np.unique(y_z)
    counts = {int(c): int((y_z == c).sum()) for c in unique}
    log(f"  [{name}] Labels 0-indexed: classes={unique.tolist()}, counts={counts}")
    return y_z


# ─────────────────────────────────────────────
# 1. HSI PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_hsi():
    log("\n" + "=" * 60)
    log("STEP 1: HSI PREPROCESSING")
    log("=" * 60)
    hsi_meta = {}

    # ── 1A. Crop Variety ──────────────────────
    log("\n[1A] Crop Variety (10 classes, 300 bands)")
    cv_dir = HSI_DIR / 'crop_variety' / 'Crop_dataset'

    X_tr = np.load(cv_dir / 'X_train.npy', allow_pickle=True)
    y_tr = np.load(cv_dir / 'y_train.npy', allow_pickle=True)
    X_te = np.load(cv_dir / 'X_test.npy',  allow_pickle=True)
    y_te = np.load(cv_dir / 'y_test.npy',  allow_pickle=True)
    log(f"  Loaded: X_train{X_tr.shape}, X_test{X_te.shape}")

    # Diagnostics
    nan_diagnostics(X_tr, 'CropVar_train')
    nan_diagnostics(X_te, 'CropVar_test')

    # Impute NaN
    X_tr = impute_nan_spectral(X_tr, 'CropVar_train')
    X_te = impute_nan_spectral(X_te, 'CropVar_test')
    log(f"  After imputation: X_train{X_tr.shape}, X_test{X_te.shape}")

    # Resample 300 → 282 bands
    X_tr = interpolate_bands(X_tr, WAVELENGTHS_300, WAVELENGTHS_282, 'CropVar_train')
    X_te = interpolate_bands(X_te, WAVELENGTHS_300, WAVELENGTHS_282, 'CropVar_test')

    # Normalise
    X_tr, stats = normalise_hsi(X_tr, 'CropVar_train', 'minmax')
    X_te, _     = normalise_hsi(X_te, 'CropVar_test',  'minmax')

    # Labels
    y_tr = zero_index_labels(y_tr, 'CropVar_train')
    y_te = zero_index_labels(y_te, 'CropVar_test')

    # Class distribution check
    tr_classes = set(np.unique(y_tr).tolist())
    te_classes = set(np.unique(y_te).tolist())
    missing_in_train = te_classes - tr_classes
    if missing_in_train:
        log(f"  WARNING: classes {missing_in_train} appear in test but NOT in train")
        log(f"  This is a dataset split issue — noted in metadata")

    # Save
    np.save(PROCESSED / 'hsi' / 'cv_X_train.npy', X_tr)
    np.save(PROCESSED / 'hsi' / 'cv_y_train.npy', y_tr)
    np.save(PROCESSED / 'hsi' / 'cv_X_test.npy',  X_te)
    np.save(PROCESSED / 'hsi' / 'cv_y_test.npy',  y_te)
    log(f"  ✓ Saved: cv_X_train{X_tr.shape}, cv_X_test{X_te.shape}")

    hsi_meta['crop_variety'] = {
        'task': 'crop_type_classification',
        'num_classes': 10,
        'bands': TARGET_BANDS,
        'patch_size': PATCH_SIZE,
        'train_samples': int(len(X_tr)),
        'test_samples': int(len(X_te)),
        'train_classes': sorted([int(c) for c in tr_classes]),
        'test_classes':  sorted([int(c) for c in te_classes]),
        'missing_in_train': sorted([int(c) for c in missing_in_train]),
        'wavelength_range': '400-1000nm (resampled from 385-1020nm)',
        'nan_handling': 'spectral_interpolation',
        'norm_stats': stats
    }

    # ── 1B. Groundnut Water Stress ────────────
    log("\n[1B] Groundnut Water Stress (binary, 300 bands, flat)")
    gn_dir = HSI_DIR / 'groundnut_stress' / 'GROUNDNUT WATER STRESS DATA'

    X_gn = np.load(gn_dir / 'X_GN_31Dec.npy', allow_pickle=True)
    y_gn = np.load(gn_dir / 'y_GN_31Dec.npy', allow_pickle=True)
    log(f"  Loaded: X_gn{X_gn.shape}")
    log(f"  Classes: WW={int((y_gn==1).sum())}, WS={int((y_gn==2).sum())}")

    nan_diagnostics(X_gn, 'Groundnut')
    X_gn = impute_nan_spectral(X_gn, 'Groundnut')
    X_gn = interpolate_bands(X_gn, WAVELENGTHS_300, WAVELENGTHS_282, 'Groundnut')
    X_gn, stats_gn = normalise_hsi(X_gn, 'Groundnut', 'minmax')
    y_gn = zero_index_labels(y_gn, 'Groundnut')

    # Keep flat + reshape to patch format
    X_gn_patch = X_gn.reshape(len(X_gn), 1, 1, TARGET_BANDS)

    # Stratified 80/20 split
    idx = np.arange(len(X_gn))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                       random_state=42, stratify=y_gn)
    np.save(PROCESSED / 'hsi' / 'gn_X_flat.npy',      X_gn)
    np.save(PROCESSED / 'hsi' / 'gn_X_patch.npy',     X_gn_patch)
    np.save(PROCESSED / 'hsi' / 'gn_y.npy',           y_gn)
    np.save(PROCESSED / 'hsi' / 'gn_train_idx.npy',   idx_tr)
    np.save(PROCESSED / 'hsi' / 'gn_test_idx.npy',    idx_te)
    log(f"  ✓ Saved: gn_X_flat{X_gn.shape}, gn_X_patch{X_gn_patch.shape}")

    hsi_meta['groundnut'] = {
        'task': 'water_stress_detection',
        'num_classes': 2,
        'bands': TARGET_BANDS,
        'total_samples': int(len(X_gn)),
        'train_samples': int(len(idx_tr)),
        'test_samples':  int(len(idx_te)),
        'label_map': {0: 'well_watered', 1: 'water_stressed'},
        'acquisition_date': '31_Dec_2021',
        'days_after_stress': 10,
        'nan_handling': 'spectral_interpolation',
        'norm_stats': stats_gn,
        'reference': 'Sankararao et al. 2023, IEEE GRSL'
    }

    # ── 1C. Pearl Millet Water Stress ─────────
    log("\n[1C] Pearl Millet Water Stress (binary, 282 bands, patches)")
    pm_dir = HSI_DIR / 'pearl_millet_stress' / 'pearl millet water stress dataset'

    X_pm = np.load(pm_dir / 'X_all_25_pm.npy', allow_pickle=True)
    y_pm = np.load(pm_dir / 'y_all_25_pm.npy', allow_pickle=True)
    log(f"  Loaded: X_pm{X_pm.shape}")
    log(f"  Classes: WW={int((y_pm==1).sum())}, WS={int((y_pm==2).sum())}")

    nan_diagnostics(X_pm, 'PearlMillet')
    X_pm = impute_nan_spectral(X_pm, 'PearlMillet')
    # 282 bands already match target
    X_pm, stats_pm = normalise_hsi(X_pm, 'PearlMillet', 'minmax')
    y_pm = zero_index_labels(y_pm, 'PearlMillet')

    idx = np.arange(len(X_pm))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                       random_state=42, stratify=y_pm)
    np.save(PROCESSED / 'hsi' / 'pm_X.npy',          X_pm)
    np.save(PROCESSED / 'hsi' / 'pm_y.npy',          y_pm)
    np.save(PROCESSED / 'hsi' / 'pm_train_idx.npy',  idx_tr)
    np.save(PROCESSED / 'hsi' / 'pm_test_idx.npy',   idx_te)
    log(f"  ✓ Saved: pm_X{X_pm.shape}")

    hsi_meta['pearl_millet'] = {
        'task': 'water_stress_detection',
        'num_classes': 2,
        'bands': TARGET_BANDS,
        'total_samples': int(len(X_pm)),
        'train_samples': int(len(idx_tr)),
        'test_samples':  int(len(idx_te)),
        'label_map': {0: 'well_watered', 1: 'water_stressed'},
        'acquisition_day': 25,
        'nan_handling': 'spectral_interpolation',
        'norm_stats': stats_pm
    }

    with open(PROCESSED / 'metadata' / 'hsi_meta.json', 'w') as f:
        json.dump(hsi_meta, f, indent=2, default=str)
    log("\n  ✓ HSI metadata saved")
    return hsi_meta


# ─────────────────────────────────────────────
# 2. MULTISPECTRAL PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_ms():
    log("\n" + "=" * 60)
    log("STEP 2: MULTISPECTRAL PREPROCESSING (skipping re-stack — already done)")
    log("=" * 60)
    log("  MS stacks already saved in v1 — loading and verifying only")

    ms_meta = {}
    for crop in ['maize', 'paddy']:
        crop_dir = PROCESSED / 'ms' / crop
        if not crop_dir.exists():
            log(f"  ✗ {crop} processed dir not found")
            continue
        for season_dir in crop_dir.iterdir():
            stack_file = season_dir / 'ms_stacked.npy'
            if stack_file.exists():
                arr = np.load(stack_file, mmap_mode='r')
                log(f"  ✓ {crop}/{season_dir.name}: {arr.shape} "
                    f"(min={arr.min():.4f}, max={arr.max():.4f})")
                nan_count = np.isnan(arr).sum()
                log(f"    NaN values: {nan_count}")
                ms_meta[f'{crop}_{season_dir.name}'] = {
                    'shape': list(arr.shape),
                    'nan_count': int(nan_count),
                    'band_order': 'Blue, Green, Red, RedEdge, NIR'
                }

    with open(PROCESSED / 'metadata' / 'ms_meta.json', 'w') as f:
        json.dump(ms_meta, f, indent=2, default=str)
    log("  ✓ MS metadata saved")
    return ms_meta


# ─────────────────────────────────────────────
# 3. RGB PREPROCESSING — FIXED LABEL LOADING
# ─────────────────────────────────────────────

def find_label_file(img_path, split_dir):
    """
    Try multiple strategies to find the label file for an image.
    Returns Path or None.
    """
    stem = img_path.stem

    # Strategy 1: same directory as image
    p = img_path.parent / f"{stem}.txt"
    if p.exists():
        return p

    # Strategy 2: split root directory
    p = split_dir / f"{stem}.txt"
    if p.exists():
        return p

    # Strategy 3: labels/ subfolder next to images
    p = split_dir / 'labels' / f"{stem}.txt"
    if p.exists():
        return p

    # Strategy 4: ../labels/ relative to image parent
    p = img_path.parent.parent / 'labels' / f"{stem}.txt"
    if p.exists():
        return p

    # Strategy 5: search entire split_dir tree
    matches = list(split_dir.rglob(f"{stem}.txt"))
    if matches:
        return matches[0]

    return None


def preprocess_rgb():
    log("\n" + "=" * 60)
    log("STEP 3: RGB PREPROCESSING")
    log("=" * 60)

    try:
        from PIL import Image as PILImage
    except ImportError:
        log("  ✗ PIL not available")
        return {}

    # First: discover where label files actually are
    log("\n  Discovering label file locations...")
    all_txts = list(RGB_DIR.rglob('*.txt'))
    log(f"  Total .txt files found: {len(all_txts)}")
    if all_txts:
        log(f"  Sample label paths:")
        for p in all_txts[:5]:
            log(f"    {p}")
        # Show a sample label content
        log(f"\n  Sample label content ({all_txts[0].name}):")
        with open(all_txts[0]) as f:
            lines = f.readlines()[:5]
        for l in lines:
            log(f"    {l.rstrip()}")

    rgb_meta = {}

    for split in ['train', 'test']:
        log(f"\n[3{'A' if split=='train' else 'B'}] RGB Paddy {split}")
        split_dir = RGB_DIR / split

        img_files = sorted(split_dir.rglob('*.jpg')) + \
                    sorted(split_dir.rglob('*.png'))
        log(f"  Found {len(img_files)} images")

        images       = []
        point_labels = []
        img_names    = []
        img_sizes    = []
        point_counts = []
        labels_found = 0

        for img_path in img_files:
            label_path = find_label_file(img_path, split_dir)

            try:
                img     = PILImage.open(img_path).convert('RGB')
                img_arr = np.array(img, dtype=np.float32) / 255.0
                W, H    = img.size

                points = []
                if label_path is not None:
                    labels_found += 1
                    with open(label_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            # Handle both "x y" and "class x y w h" (YOLO) formats
                            if len(parts) == 2:
                                x, y = float(parts[0]), float(parts[1])
                                points.append([x, y])
                            elif len(parts) == 5:
                                # YOLO format: class cx cy w h (normalised 0-1)
                                cx = float(parts[1]) * W
                                cy = float(parts[2]) * H
                                points.append([cx, cy])
                            elif len(parts) >= 2:
                                try:
                                    x, y = float(parts[0]), float(parts[1])
                                    points.append([x, y])
                                except ValueError:
                                    pass

                images.append(img_arr)
                pts = np.array(points, dtype=np.float32) if points \
                      else np.zeros((0, 2), dtype=np.float32)
                point_labels.append(pts)
                img_names.append(img_path.name)
                img_sizes.append((H, W))
                point_counts.append(len(points))

            except Exception as e:
                log(f"    Warning: {img_path.name}: {e}")
                continue

        log(f"  Loaded: {len(images)} images")
        log(f"  Labels matched: {labels_found}/{len(images)}")
        if point_counts:
            log(f"  Points/image: min={min(point_counts)}, "
                f"max={max(point_counts)}, mean={np.mean(point_counts):.1f}")
            log(f"  Images with ≥1 point: {sum(1 for p in point_counts if p > 0)}")

        size_counts      = Counter(img_sizes)
        most_common_size = size_counts.most_common(1)[0][0]
        log(f"  Image sizes: {dict(size_counts.most_common(3))}")

        # Save
        out_rgb = PROCESSED / 'rgb'
        np.save(out_rgb / f'{split}_point_labels.npy',
                np.array(point_labels, dtype=object))
        np.save(out_rgb / f'{split}_image_names.npy',
                np.array(img_names))
        np.save(out_rgb / f'{split}_point_counts.npy',
                np.array(point_counts))

        uniform = [(img, pts) for img, pts, sz
                   in zip(images, point_labels, img_sizes)
                   if sz == most_common_size]
        if uniform:
            imgs_arr = np.array([x[0] for x in uniform], dtype=np.float32)
            np.save(out_rgb / f'{split}_images.npy', imgs_arr)
            log(f"  ✓ Saved uniform images: {imgs_arr.shape}")

        meta_json = {
            'names':  img_names,
            'sizes':  img_sizes,
            'counts': point_counts
        }
        with open(PROCESSED / 'metadata' / f'rgb_{split}_meta.json', 'w') as f:
            json.dump(meta_json, f, indent=2)

        rgb_meta[split] = {
            'total_images':        len(images),
            'uniform_images':      len(uniform),
            'labels_matched':      labels_found,
            'most_common_size':    most_common_size,
            'annotation_type':     'point_xy_pixel',
            'mean_points_per_img': float(np.mean(point_counts)) if point_counts else 0,
            'task':                'panicle_detection_counting'
        }

    with open(PROCESSED / 'metadata' / 'rgb_meta.json', 'w') as f:
        json.dump(rgb_meta, f, indent=2)
    log("\n  ✓ RGB metadata saved")
    return rgb_meta


# ─────────────────────────────────────────────
# 4. MANIFEST
# ─────────────────────────────────────────────

def create_manifest(hsi_meta, ms_meta, rgb_meta):
    log("\n" + "=" * 60)
    log("STEP 4: DATASET MANIFEST")
    log("=" * 60)

    manifest = {
        'project':         'Multi-Modal Foundation Models for Comprehensive Remote Sensing Based Precision Crop Monitoring',
        'version':         'v2',
        'created':         datetime.now().isoformat(),
        'target_bands':    TARGET_BANDS,
        'wavelength_range': f'400-1000nm ({TARGET_BANDS} bands)',
        'datasets':        {'hsi': hsi_meta, 'ms': ms_meta, 'rgb': rgb_meta},
        'nan_strategy':    'spectral_interpolation (linear interp on valid bands per pixel; band-mean fallback if <10 valid bands)',
        'tasks': [
            'crop_type_classification       — 10 classes — HSI crop_variety',
            'water_stress_binary            — 2 classes  — HSI groundnut',
            'water_stress_binary            — 2 classes  — HSI pearl_millet',
            'panicle_detection_counting     — point annot — RGB paddy',
            'crop_monitoring_multispectral  — 5 bands    — MS maize + paddy',
        ]
    }

    with open(PROCESSED / 'metadata' / 'dataset_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    log("\n" + "─" * 62)
    log(f"  {'Dataset':<32} {'Samples':>8} {'Shape':>18}")
    log("─" * 62)

    if 'crop_variety' in hsi_meta:
        cv = hsi_meta['crop_variety']
        log(f"  {'HSI CropVariety train':<32} {cv['train_samples']:>8} {'(N,11,11,282)':>18}")
        log(f"  {'HSI CropVariety test':<32} {cv['test_samples']:>8} {'(N,11,11,282)':>18}")
    if 'groundnut' in hsi_meta:
        gn = hsi_meta['groundnut']
        log(f"  {'HSI Groundnut (flat)':<32} {gn['total_samples']:>8} {'(N,282)':>18}")
        log(f"  {'HSI Groundnut (patch)':<32} {gn['total_samples']:>8} {'(N,1,1,282)':>18}")
    if 'pearl_millet' in hsi_meta:
        pm = hsi_meta['pearl_millet']
        log(f"  {'HSI Pearl Millet':<32} {pm['total_samples']:>8} {'(N,11,11,282)':>18}")
    if 'train' in rgb_meta:
        log(f"  {'RGB Paddy train':<32} {rgb_meta['train']['total_images']:>8} {'(N,850,1150,3)':>18}")
        log(f"  {'RGB Paddy test':<32} {rgb_meta['test']['total_images']:>8} {'(N,850,1150,3)':>18}")
    log("─" * 62)
    log("  ✓ Manifest saved")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    log("=" * 60)
    log("AGRI FOUNDATION — PREPROCESSING PIPELINE v2")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    hsi_meta = preprocess_hsi()
    ms_meta  = preprocess_ms()
    rgb_meta = preprocess_rgb()
    create_manifest(hsi_meta, ms_meta, rgb_meta)

    log("\n" + "=" * 60)
    log(f"COMPLETE: {datetime.now().strftime('%H:%M:%S')}")
    log(f"Output  : {PROCESSED}")
    log("=" * 60)