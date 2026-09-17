"""
Cross-Geographic MS Evaluation — Frozen Prithvi-EO-2.0.

Evaluates cross-geographic transfer using frozen Prithvi backbone
with linear probe. No JEPA — this is the foundation model baseline.

Train: India maize + paddy MS
Test:  Turkey maize MS (Zenodo), Sri Lanka paddy MS

This gives the cross-geographic baseline before Prithvi-JEPA completes.

Run: CUDA_VISIBLE_DEVICES=1 python cross_geo_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PRITHVI_DIR = Path("~/agri_foundation/models/prithvi").expanduser()
sys.path.insert(0, str(PRITHVI_DIR))

DATA_ROOT = Path("~/agri_foundation/data").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TILE_SIZE = 64
PRITHVI_IMG_SIZE = 224
PRITHVI_NUM_FRAMES = 4
PRITHVI_MEAN = torch.tensor([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0])
PRITHVI_STD  = torch.tensor([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0])
MS_MEAN = np.array([0.2541, 0.2613, 0.2608, 0.3284, 0.2856], dtype=np.float32)
MS_STD  = np.array([0.1356, 0.1386, 0.1438, 0.1477, 0.1460], dtype=np.float32)


# ---------------------------------------------------------------------------
# Prithvi loader
# ---------------------------------------------------------------------------

def load_prithvi(device: torch.device) -> nn.Module:
    from prithvi_mae import PrithviMAE
    cfg = json.load(open(PRITHVI_DIR / "config.json"))["pretrained_cfg"]
    mae = PrithviMAE(
        img_size=cfg["img_size"], num_frames=cfg["num_frames"],
        patch_size=cfg["patch_size"], in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"], depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        decoder_embed_dim=cfg["decoder_embed_dim"],
        decoder_depth=cfg["decoder_depth"],
        decoder_num_heads=cfg["decoder_num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
    )
    ckpt = torch.load(
        PRITHVI_DIR / "Prithvi_EO_V2_300M.pt",
        map_location="cpu", weights_only=False
    )
    mae.load_state_dict(ckpt.get("model", ckpt), strict=False)
    enc = mae.encoder
    for p in enc.parameters():
        p.requires_grad = False
    print(f"Prithvi loaded on {device}")
    return enc.to(device)


# ---------------------------------------------------------------------------
# Band adapter
# ---------------------------------------------------------------------------

def to_prithvi(tiles: Tensor, n_bands: int = 5) -> Tensor:
    """(B, C, H, W) normalised -> (B, 6, 4, 224, 224)"""
    B, C, H, W = tiles.shape
    if C == 5:
        scale = torch.tensor(
            [1087/0.2541, 1342/0.2613, 1433/0.2608,
             2734/0.3284, 1958/0.2856],
            device=tiles.device, dtype=tiles.dtype)
    elif C == 4:
        scale = torch.tensor(
            [1342/0.2613, 1433/0.2608, 2734/0.3284, 1958/0.2856],
            device=tiles.device, dtype=tiles.dtype)
    else:
        scale = torch.ones(C, device=tiles.device) * 1500.0

    tiles_dn = tiles * scale[None, :, None, None]
    pad = torch.zeros(B, 6 - C, H, W, device=tiles.device, dtype=tiles.dtype)
    tiles_6 = torch.cat([tiles_dn, pad], dim=1) if C < 6 else tiles_dn[:, :6]
    tiles_r = F.interpolate(tiles_6, size=(224, 224),
                            mode="bilinear", align_corners=False)
    mean = PRITHVI_MEAN.to(tiles.device)[None, :, None, None]
    std  = PRITHVI_STD.to(tiles.device)[None, :, None, None]
    tiles_n = (tiles_r - mean) / (std + 1e-8)
    return tiles_n.unsqueeze(2).repeat(1, 1, 4, 1, 1)  # (B,6,4,224,224)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(enc: nn.Module, tiles: np.ndarray,
                        batch_size: int = 8) -> np.ndarray:
    enc.eval()
    all_emb = []
    for start in range(0, len(tiles), batch_size):
        batch = torch.from_numpy(
            tiles[start:start+batch_size].transpose(0,3,1,2)
        ).to(DEVICE)
        x = to_prithvi(batch, n_bands=batch.shape[1])
        feat = enc.forward_features(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.ndim == 3:
            feat = feat.mean(dim=1)
        elif feat.shape[1] == 1:
            feat = feat.squeeze(1)
        all_emb.append(F.normalize(feat, dim=-1).cpu().numpy())
    return np.concatenate(all_emb)


# ---------------------------------------------------------------------------
# NDVI labels
# ---------------------------------------------------------------------------

def ndvi_labels(tiles: np.ndarray,
                nir_idx: int = 4, red_idx: int = 2) -> np.ndarray:
    nir = tiles[:, :, :, nir_idx].mean(axis=(1,2))
    red = tiles[:, :, :, red_idx].mean(axis=(1,2))
    ndvi = (nir - red) / (nir + red + 1e-8)
    return (ndvi > np.median(ndvi)).astype(np.int64)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def linear_probe(X_tr, y_tr, X_te, y_te, name="") -> dict:
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    clf = LogisticRegression(max_iter=1000, C=1.0,
                             random_state=42, class_weight="balanced")
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    try:
        auc = roc_auc_score(y_te, y_prob)
    except ValueError:
        auc = float("nan")
    print(f"  {name}: Acc={acc:.4f} AUC={auc:.4f} "
          f"(train={len(y_tr)}, test={len(y_te)})")
    return {"name": name, "accuracy": acc, "auc": auc}


# ---------------------------------------------------------------------------
# Tile extractor
# ---------------------------------------------------------------------------

def extract_tiles(arr: np.ndarray, n_bands: int,
                  tiles_per_image: int = 10,
                  seed: int = 42) -> np.ndarray:
    """(N, H, W, C) -> (N*tiles_per_image, 64, 64, n_bands) normalised"""
    rng = np.random.default_rng(seed)
    N, H, W, C = arr.shape
    tiles = []
    for img in arr:
        for _ in range(tiles_per_image):
            if H < TILE_SIZE or W < TILE_SIZE:
                continue
            t = rng.integers(0, H - TILE_SIZE)
            l = rng.integers(0, W - TILE_SIZE)
            tile = img[t:t+TILE_SIZE, l:l+TILE_SIZE, :n_bands].astype(np.float32)
            tile = (tile - MS_MEAN[:n_bands]) / (MS_STD[:n_bands] + 1e-6)
            tiles.append(tile)
    return np.stack(tiles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load Prithvi
    enc = load_prithvi(DEVICE)

    # ----------------------------------------------------------------
    # Load all datasets
    # ----------------------------------------------------------------
    ms_dir = DATA_ROOT / "processed" / "ms"
    results = []

    # India — Maize + Paddy (5 bands)
    print("\nLoading India MS...")
    maize_files = list(ms_dir.glob("maize/**/ms_stacked.npy"))
    paddy_files = list(ms_dir.glob("paddy/**/ms_stacked.npy"))
    india_maize = extract_tiles(np.load(maize_files[0]), 5, tiles_per_image=10)
    india_paddy = extract_tiles(np.load(paddy_files[0]), 5, tiles_per_image=10)
    india_tiles = np.concatenate([india_maize, india_paddy], axis=0)
    india_labels = ndvi_labels(india_tiles)
    print(f"  India: {len(india_tiles)} tiles")

    print("Extracting India embeddings...")
    india_emb = extract_embeddings(enc, india_tiles)

    # Turkey — Zenodo Maize (6 bands, drop Alpha)
    print("\nLoading Turkey MS (Zenodo maize)...")
    zenodo_dir = (DATA_ROOT / "zenodo_maize_ms" /
                  "processed_patches" / "water_2025" / "images")
    turkey_tiles_list = []
    for f in sorted(zenodo_dir.glob("*.npy"))[:200]:
        tile = np.load(f).astype(np.float32)         # (224, 224, 6)
        tile_5 = tile[:, :, :5]                       # drop Alpha
        # Downsample 224->64
        t = torch.from_numpy(tile_5.transpose(2,0,1)).unsqueeze(0)
        t64 = F.interpolate(t, size=(64,64),
                            mode='bilinear', align_corners=False)
        turkey_tiles_list.append(t64.squeeze(0).permute(1,2,0).numpy())
    turkey_tiles = np.stack(turkey_tiles_list)
    turkey_labels = ndvi_labels(turkey_tiles)
    print(f"  Turkey: {len(turkey_tiles)} tiles")

    print("Extracting Turkey embeddings...")
    turkey_emb = extract_embeddings(enc, turkey_tiles)

    # Sri Lanka — Paddy (4 bands: G, R, RE, NIR)
    print("\nLoading Sri Lanka MS...")
    sl_path = DATA_ROOT / "paddy_srilanka" / "ms_stacked_srilanka.npy"
    sl_arr = np.load(sl_path)                         # (67, 1944, 2592, 4)
    # Pad to 5 bands (prepend Blue=0)
    sl_5 = np.concatenate(
        [np.zeros((*sl_arr.shape[:3], 1), dtype=np.float32), sl_arr],
        axis=-1
    )
    sl_tiles = extract_tiles(sl_5, 5, tiles_per_image=10)
    sl_labels = ndvi_labels(sl_tiles)
    print(f"  Sri Lanka: {len(sl_tiles)} tiles")

    print("Extracting Sri Lanka embeddings...")
    sl_emb = extract_embeddings(enc, sl_tiles)

    # ----------------------------------------------------------------
    # Cross-geographic evaluation
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("CROSS-GEOGRAPHIC EVALUATION — Frozen Prithvi-EO-2.0")
    print("="*60)

    # Within-country baselines
    print("\n[Within-country baselines]")
    n = len(india_emb)
    idx = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    r = linear_probe(india_emb[idx[:split]], india_labels[idx[:split]],
                     india_emb[idx[split:]], india_labels[idx[split:]],
                     name="Within: India→India")
    results.append(r)

    n = len(turkey_emb)
    idx = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    r = linear_probe(turkey_emb[idx[:split]], turkey_labels[idx[:split]],
                     turkey_emb[idx[split:]], turkey_labels[idx[split:]],
                     name="Within: Turkey→Turkey")
    results.append(r)

    n = len(sl_emb)
    idx = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    r = linear_probe(sl_emb[idx[:split]], sl_labels[idx[:split]],
                     sl_emb[idx[split:]], sl_labels[idx[split:]],
                     name="Within: Sri Lanka→Sri Lanka")
    results.append(r)

    # Cross-geographic transfers
    print("\n[Cross-geographic transfers]")

    # India -> Turkey (different country, same crop: maize)
    r = linear_probe(india_emb, india_labels,
                     turkey_emb, turkey_labels,
                     name="Cross-geo: India→Turkey (maize)")
    results.append(r)

    # India -> Sri Lanka (different country, different crop)
    r = linear_probe(india_emb, india_labels,
                     sl_emb, sl_labels,
                     name="Cross-geo: India→Sri Lanka (paddy)")
    results.append(r)

    # Turkey -> Sri Lanka
    r = linear_probe(turkey_emb, turkey_labels,
                     sl_emb, sl_labels,
                     name="Cross-geo: Turkey→Sri Lanka")
    results.append(r)

    # All countries combined -> each held-out country
    print("\n[Multi-country train → held-out country test]")

    # Train: India + Turkey → Test: Sri Lanka
    train_emb = np.concatenate([india_emb, turkey_emb])
    train_labels = np.concatenate([india_labels, turkey_labels])
    r = linear_probe(train_emb, train_labels,
                     sl_emb, sl_labels,
                     name="Multi→Sri Lanka (India+Turkey→SL)")
    results.append(r)

    # Train: India + Sri Lanka → Test: Turkey
    train_emb = np.concatenate([india_emb, sl_emb])
    train_labels = np.concatenate([india_labels, sl_labels])
    r = linear_probe(train_emb, train_labels,
                     turkey_emb, turkey_labels,
                     name="Multi→Turkey (India+SL→Turkey)")
    results.append(r)

    # Train: Turkey + Sri Lanka → Test: India
    train_emb = np.concatenate([turkey_emb, sl_emb])
    train_labels = np.concatenate([turkey_labels, sl_labels])
    r = linear_probe(train_emb, train_labels,
                     india_emb, india_labels,
                     name="Multi→India (Turkey+SL→India)")
    results.append(r)

    # Summary table
    print("\n" + "="*65)
    print(f"{'Experiment':<45} {'Acc':>8} {'AUC':>8}")
    print("-"*65)
    for r in results:
        print(f"  {r['name']:<43} {r['accuracy']:.4f}  {r['auc']:.4f}")

    # Save
    with open(LOG_DIR / "cross_geo_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {LOG_DIR / 'cross_geo_eval.json'}")


if __name__ == "__main__":
    main()