"""
Cross-modal transfer evaluation for agri_foundation.

Evaluates whether the self-supervised MS encoder (trained with no labels)
learns representations that are linearly separable by crop stress proxies.

Protocol:
  1. Load pretrained MS encoder (frozen, epoch 78)
  2. Compute NDVI from MS tiles: (NIR - Red) / (NIR + Red)
     Band order: Blue=0, Green=1, Red=2, RedEdge=3, NIR=4
  3. Threshold NDVI to create binary pseudo-labels:
     NDVI > 0.4 = healthy (0), NDVI <= 0.4 = stressed (1)
  4. Extract MS embeddings from all tiles
  5. Train a linear probe (logistic regression) on embeddings
  6. Evaluate accuracy, AUC — if high, encoder learned stress-relevant features
  7. Also evaluate cross-crop: train on maize tiles, test on paddy tiles

This directly supports the paper's claim that contrastive pretraining
produces modality-agnostic representations relevant to crop stress.

Run: python cross_modal_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TILE_SIZE = 64
NDVI_THRESHOLD = 0.4
BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# MS Encoder (must match train_ms_pretrain.py exactly)
# ---------------------------------------------------------------------------

class MSEncoder(nn.Module):
    def __init__(self, in_channels: int = 5, base_channels: int = 32, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = nn.Sequential(
            self._conv_block(in_channels, base_channels, stride=2),
            self._conv_block(base_channels, base_channels * 2, stride=2),
            self._conv_block(base_channels * 2, base_channels * 4, stride=2),
            self._conv_block(base_channels * 4, embed_dim, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# NDVI computation
# ---------------------------------------------------------------------------

def compute_ndvi(tiles: np.ndarray) -> np.ndarray:
    """
    Compute per-tile mean NDVI from 5-band MS tiles.
    tiles: (N, 5, H, W) — bands: B=0, G=1, R=2, RE=3, NIR=4
    Returns: (N,) mean NDVI per tile
    """
    nir = tiles[:, 4, :, :]   # NIR band
    red = tiles[:, 2, :, :]   # Red band
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi.mean(axis=(1, 2))   # mean NDVI per tile


def ndvi_to_labels(ndvi: np.ndarray, threshold: float = NDVI_THRESHOLD) -> np.ndarray:
    """
    Binary pseudo-labels from NDVI.
    NDVI > threshold = healthy (0)
    NDVI <= threshold = stressed (1)
    """
    return (ndvi <= threshold).astype(np.int64)


# ---------------------------------------------------------------------------
# Tile extraction
# ---------------------------------------------------------------------------

def extract_random_tiles(
    images: np.ndarray,
    tile_size: int = TILE_SIZE,
    tiles_per_image: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Extract random tiles from (N, H, W, C) image array.
    Returns (N * tiles_per_image, C, tile_size, tile_size) in CHW order.
    """
    rng = np.random.default_rng(seed)
    N, H, W, C = images.shape
    tiles = []

    for img in images:
        for _ in range(tiles_per_image):
            top = rng.integers(0, H - tile_size)
            left = rng.integers(0, W - tile_size)
            tile = img[top:top+tile_size, left:left+tile_size, :]
            tiles.append(tile.transpose(2, 0, 1))  # (C, T, T)

    return np.stack(tiles, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    encoder: MSEncoder,
    tiles: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Extract L2-normalised embeddings for all tiles."""
    encoder.eval()
    all_emb = []

    for start in range(0, len(tiles), batch_size):
        batch = torch.from_numpy(tiles[start:start+batch_size]).to(DEVICE)
        emb = encoder(batch)
        all_emb.append(emb.cpu().numpy())

    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# Linear probe evaluation
# ---------------------------------------------------------------------------

def linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "",
) -> dict:
    """
    Train logistic regression on embeddings, evaluate on test set.
    Embeddings are already L2-normalised — StandardScaler still helps
    by adjusting scale differences across dimensions.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- {name} ---")
    print(f"  Train: {len(y_train)} tiles | Test: {len(y_test)} tiles")
    print(f"  Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test  label dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Healthy", "Stressed"],
                                 digits=4))

    return {"accuracy": acc, "auc": auc, "name": name}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")

    # Load encoder
    encoder = MSEncoder(in_channels=5, base_channels=32, embed_dim=256).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "ms_encoder_best.pt", map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder_state"])
    print(f"Loaded MS encoder: epoch={ckpt['epoch']} loss={ckpt['loss']:.6f}")

    # Load MS arrays
    ms_dir = DATA_ROOT / "processed" / "ms"
    print("\nLoading MS arrays...")

    maize_path = list(ms_dir.glob("maize/**/ms_stacked.npy"))[0]
    paddy_path = list(ms_dir.glob("paddy/**/ms_stacked.npy"))[0]

    maize = np.load(maize_path)   # (302, 960, 1280, 5)
    paddy = np.load(paddy_path)   # (315, 960, 1280, 5)
    print(f"Maize: {maize.shape} | Paddy: {paddy.shape}")

    # Normalise using computed band statistics
    MS_MEAN = np.array([0.2541, 0.2613, 0.2608, 0.3284, 0.2856], dtype=np.float32)
    MS_STD = np.array([0.1356, 0.1386, 0.1438, 0.1477, 0.1460], dtype=np.float32)

    # Extract tiles
    print("\nExtracting tiles...")
    TILES_PER_IMG = 15
    maize_tiles = extract_random_tiles(maize, TILE_SIZE, TILES_PER_IMG, seed=42)
    paddy_tiles = extract_random_tiles(paddy, TILE_SIZE, TILES_PER_IMG, seed=42)
    print(f"Maize tiles: {maize_tiles.shape} | Paddy tiles: {paddy_tiles.shape}")

    # Normalise tiles
    mean = MS_MEAN[:, None, None]
    std = MS_STD[:, None, None]
    maize_tiles_n = (maize_tiles - mean) / (std + 1e-6)
    paddy_tiles_n = (paddy_tiles - mean) / (std + 1e-6)

    # Compute NDVI pseudo-labels on raw (unnormalised) tiles
    maize_ndvi = compute_ndvi(maize_tiles)
    paddy_ndvi = compute_ndvi(paddy_tiles)
    maize_labels = ndvi_to_labels(maize_ndvi, NDVI_THRESHOLD)
    paddy_labels = ndvi_to_labels(paddy_ndvi, NDVI_THRESHOLD)

    print(f"\nNDVI statistics:")
    print(f"  Maize NDVI: mean={maize_ndvi.mean():.4f} std={maize_ndvi.std():.4f}")
    print(f"  Paddy NDVI: mean={paddy_ndvi.mean():.4f} std={paddy_ndvi.std():.4f}")
    print(f"  Maize labels: {dict(zip(*np.unique(maize_labels, return_counts=True)))}")
    print(f"  Paddy labels: {dict(zip(*np.unique(paddy_labels, return_counts=True)))}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    maize_emb = extract_embeddings(encoder, maize_tiles_n)
    paddy_emb = extract_embeddings(encoder, paddy_tiles_n)
    print(f"Maize embeddings: {maize_emb.shape}")
    print(f"Paddy embeddings: {paddy_emb.shape}")

    # Embedding space health check
    print(f"\nEmbedding space:")
    print(f"  Maize std: {maize_emb.std(axis=0).mean():.4f}")
    print(f"  Paddy std: {paddy_emb.std(axis=0).mean():.4f}")

    results = []

    # Experiment 1: Within-crop linear probe (maize)
    # 80/20 split on maize
    n_maize = len(maize_emb)
    split = int(0.8 * n_maize)
    idx = np.random.default_rng(42).permutation(n_maize)
    res = linear_probe(
        maize_emb[idx[:split]], maize_labels[idx[:split]],
        maize_emb[idx[split:]], maize_labels[idx[split:]],
        name="Within-crop: Maize train -> Maize test"
    )
    results.append(res)

    # Experiment 2: Within-crop linear probe (paddy)
    n_paddy = len(paddy_emb)
    split_p = int(0.8 * n_paddy)
    idx_p = np.random.default_rng(42).permutation(n_paddy)
    res = linear_probe(
        paddy_emb[idx_p[:split_p]], paddy_labels[idx_p[:split_p]],
        paddy_emb[idx_p[split_p:]], paddy_labels[idx_p[split_p:]],
        name="Within-crop: Paddy train -> Paddy test"
    )
    results.append(res)

    # Experiment 3: Cross-crop transfer (train maize, test paddy)
    res = linear_probe(
        maize_emb, maize_labels,
        paddy_emb, paddy_labels,
        name="Cross-crop transfer: Maize train -> Paddy test"
    )
    results.append(res)

    # Experiment 4: Cross-crop transfer (train paddy, test maize)
    res = linear_probe(
        paddy_emb, paddy_labels,
        maize_emb, maize_labels,
        name="Cross-crop transfer: Paddy train -> Maize test"
    )
    results.append(res)

    # Summary
    print("\n" + "=" * 60)
    print("CROSS-MODAL EVALUATION SUMMARY (for paper Table 3)")
    print("=" * 60)
    print(f"{'Experiment':<45} {'Acc':>6} {'AUC':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<45} {r['accuracy']:>6.4f} {r['auc']:>6.4f}")

    print(f"\nNDVI threshold: {NDVI_THRESHOLD}")
    print(f"Encoder: MS contrastive (epoch {ckpt['epoch']}, frozen)")
    print(f"Probe: Logistic regression (linear)")

    # Save
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "cross_modal_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {LOG_DIR / 'cross_modal_eval.json'}")


if __name__ == "__main__":
    main()