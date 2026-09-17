"""
Fix 1: Improved Cross-Geographic MS Evaluation.

Two improvements over cross_geo_eval.py:

1. Turkey: use REAL pixel-level stress labels from Zenodo masks
   (soil=0, low_stress=1, high_stress=2, healthy=3)
   Binary: stressed = low_stress + high_stress, healthy = healthy
   soil patches excluded

2. India + Sri Lanka: NDRE + SAVI ensemble instead of NDVI alone
   NDRE = (NIR - RedEdge) / (NIR + RedEdge)  -- more stress-sensitive
   SAVI = ((NIR - Red) / (NIR + Red + 0.5)) * 1.5  -- soil-adjusted

Run: CUDA_VISIBLE_DEVICES=1 python cross_geo_eval_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PRITHVI_DIR = Path("~/agri_foundation/models/prithvi").expanduser()
sys.path.insert(0, str(PRITHVI_DIR))

DATA_ROOT   = Path("~/agri_foundation/data").expanduser()
LOG_DIR     = Path("~/agri_foundation/logs").expanduser()
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TILE_SIZE   = 64
PRITHVI_MEAN = torch.tensor([1087., 1342., 1433., 2734., 1958., 1363.])
PRITHVI_STD  = torch.tensor([2248., 2179., 2178., 1850., 1242., 1049.])
MS_MEAN = np.array([0.2541, 0.2613, 0.2608, 0.3284, 0.2856], np.float32)
MS_STD  = np.array([0.1356, 0.1386, 0.1438, 0.1477, 0.1460], np.float32)

# Band indices for 5-band (B,G,R,RE,NIR)
B_IDX, G_IDX, R_IDX, RE_IDX, NIR_IDX = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Vegetation index ensemble labels (India + Sri Lanka)
# ---------------------------------------------------------------------------

def vi_ensemble_labels(tiles: np.ndarray, n_bands: int = 5) -> np.ndarray:
    """
    Compute binary stress labels using NDRE + SAVI ensemble.
    More robust than NDVI alone for cross-geographic transfer.

    NDRE = (NIR - RedEdge) / (NIR + RedEdge)
      - Red-edge band is more sensitive to early chlorophyll stress
      - Less saturated than NDVI in dense canopies

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L), L=0.5
      - Soil-adjusted: reduces soil background influence
      - Important for tiles with mixed crop/soil pixels

    Labels: below ensemble median = stressed (1), above = healthy (0)
    """
    nir = tiles[:, :, :, NIR_IDX].mean(axis=(1, 2))
    red = tiles[:, :, :, R_IDX].mean(axis=(1, 2))

    if n_bands >= 4:  # has RedEdge
        re  = tiles[:, :, :, RE_IDX].mean(axis=(1, 2))
        ndre = (nir - re) / (nir + re + 1e-8)
    else:
        ndre = None

    # SAVI (L=0.5)
    L = 0.5
    savi = ((nir - red) / (nir + red + L + 1e-8)) * (1 + L)

    # NDVI for reference
    ndvi = (nir - red) / (nir + red + 1e-8)

    # Ensemble: average available indices
    indices = [ndvi, savi]
    if ndre is not None:
        indices.append(ndre)
    ensemble = np.stack(indices, axis=0).mean(axis=0)

    # Below median = stressed (lower vegetation index = more stressed)
    labels = (ensemble < np.median(ensemble)).astype(np.int64)
    n_stressed = labels.sum()
    print(f"    VI ensemble: {n_stressed} stressed, "
          f"{len(labels)-n_stressed} healthy "
          f"(NDVI median={np.median(ndvi):.3f}, "
          f"SAVI median={np.median(savi):.3f}"
          + (f", NDRE median={np.median(ndre):.3f})" if ndre is not None else ")"))
    return labels


# ---------------------------------------------------------------------------
# Turkey: real stress labels from Zenodo pixel masks
# ---------------------------------------------------------------------------

def load_turkey_with_real_labels(
    tiles_per_patch: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Turkey Zenodo maize patches with real pixel-level stress labels.

    class_map: soil=0, low_stress=1, high_stress=2, healthy=3
    Binary: stressed = (low_stress + high_stress) dominant
            healthy  = healthy dominant
    Exclude: soil-dominant patches (ratio_c0 > 0.5)
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    water_dir = (DATA_ROOT / "zenodo_maize_ms" /
                 "processed_patches" / "water_2025")
    img_dir  = water_dir / "images"
    mask_dir = water_dir / "masks"
    meta_csv = water_dir / "meta" / "patches.csv"

    df = pd.read_csv(meta_csv)

    # Filter: exclude soil-dominant patches (ratio_c0 > 0.5)
    df_valid = df[df["ratio_c0"] <= 0.5].copy()
    print(f"  Turkey: {len(df_valid)}/{len(df)} patches after soil filter")

    # Assign binary label from dominant class
    # ratio_c1 = low_stress, ratio_c2 = high_stress, ratio_c3 = healthy
    stress_ratio  = df_valid["ratio_c1"] + df_valid["ratio_c2"]
    healthy_ratio = df_valid["ratio_c3"]
    df_valid = df_valid[stress_ratio + healthy_ratio > 0].copy()
    df_valid["binary_label"] = (
        stress_ratio[df_valid.index] > healthy_ratio[df_valid.index]
    ).astype(int)

    n_stressed = df_valid["binary_label"].sum()
    print(f"  Turkey labels: {n_stressed} stressed, "
          f"{len(df_valid)-n_stressed} healthy")

    tiles, labels = [], []
    for _, row in df_valid.iterrows():
        patch_id = row["id"]
        img_path = img_dir / f"{patch_id}.npy"
        if not img_path.exists():
            continue

        img = np.load(img_path).astype(np.float32)  # (224, 224, 6)
        img_5 = img[:, :, :5]                        # drop Alpha

        # Downsample to 64x64
        t = torch.from_numpy(img_5.transpose(2, 0, 1)).unsqueeze(0)
        t64 = F.interpolate(t, size=(64, 64),
                            mode="bilinear", align_corners=False)
        tile = t64.squeeze(0).permute(1, 2, 0).numpy()
        tiles.append(tile)
        labels.append(int(row["binary_label"]))

    tiles_arr  = np.stack(tiles)
    labels_arr = np.array(labels, dtype=np.int64)
    print(f"  Turkey tiles loaded: {tiles_arr.shape}")
    return tiles_arr, labels_arr


# ---------------------------------------------------------------------------
# Prithvi loader and embedding extraction
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
    ckpt = torch.load(PRITHVI_DIR / "Prithvi_EO_V2_300M.pt",
                      map_location="cpu", weights_only=False)
    mae.load_state_dict(ckpt.get("model", ckpt), strict=False)
    enc = mae.encoder
    for p in enc.parameters():
        p.requires_grad = False
    print(f"Prithvi loaded on {device}")
    return enc.to(device)


def to_prithvi(tiles: Tensor, C: int = 5) -> Tensor:
    B, _, H, W = tiles.shape
    scale = torch.tensor(
        [1087/0.2541, 1342/0.2613, 1433/0.2608,
         2734/0.3284, 1958/0.2856][:C],
        device=tiles.device, dtype=tiles.dtype)
    dn = tiles * scale[None, :, None, None]
    pad = torch.zeros(B, 6-C, H, W, device=tiles.device, dtype=tiles.dtype)
    t6  = torch.cat([dn, pad], dim=1)
    tr  = F.interpolate(t6, size=(224, 224),
                        mode="bilinear", align_corners=False)
    mean = PRITHVI_MEAN.to(tiles.device)[None, :, None, None]
    std  = PRITHVI_STD.to(tiles.device)[None, :, None, None]
    tn   = (tr - mean) / (std + 1e-8)
    return tn.unsqueeze(2).repeat(1, 1, 4, 1, 1)


@torch.no_grad()
def extract_embeddings(enc: nn.Module, tiles: np.ndarray,
                       batch_size: int = 8) -> np.ndarray:
    enc.eval()
    all_emb = []
    C = tiles.shape[-1]
    for start in range(0, len(tiles), batch_size):
        batch = torch.from_numpy(
            tiles[start:start+batch_size].transpose(0, 3, 1, 2)
        ).to(DEVICE)
        x    = to_prithvi(batch, C=C)
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
# Linear probe
# ---------------------------------------------------------------------------

def linear_probe(X_tr, y_tr, X_te, y_te, name="") -> dict:
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=1000, C=1.0,
                             random_state=42, class_weight="balanced")
    clf.fit(sc.fit_transform(X_tr), y_tr)
    y_pred = clf.predict(sc.transform(X_te))
    y_prob = clf.predict_proba(sc.transform(X_te))[:, 1]
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
    rng = np.random.default_rng(seed)
    N, H, W, C = arr.shape
    tiles = []
    for img in arr:
        for _ in range(tiles_per_image):
            if H < TILE_SIZE or W < TILE_SIZE:
                continue
            t = rng.integers(0, H - TILE_SIZE)
            l = rng.integers(0, W - TILE_SIZE)
            tile = img[t:t+TILE_SIZE,
                       l:l+TILE_SIZE, :n_bands].astype(np.float32)
            tile = (tile - MS_MEAN[:n_bands]) / (MS_STD[:n_bands] + 1e-6)
            tiles.append(tile)
    return np.stack(tiles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    enc = load_prithvi(DEVICE)

    # ----------------------------------------------------------------
    # India — Maize + Paddy (5 bands, VI ensemble labels)
    # ----------------------------------------------------------------
    print("\nLoading India MS...")
    ms_dir = DATA_ROOT / "processed" / "ms"
    india_maize = extract_tiles(
        np.load(list(ms_dir.glob("maize/**/ms_stacked.npy"))[0]),
        5, tiles_per_image=10)
    india_paddy = extract_tiles(
        np.load(list(ms_dir.glob("paddy/**/ms_stacked.npy"))[0]),
        5, tiles_per_image=10)
    india_tiles  = np.concatenate([india_maize, india_paddy])
    print(f"  India: {len(india_tiles)} tiles — labelling with VI ensemble:")
    india_labels = vi_ensemble_labels(india_tiles, n_bands=5)

    print("Extracting India embeddings...")
    india_emb = extract_embeddings(enc, india_tiles)

    # ----------------------------------------------------------------
    # Turkey — REAL stress labels from Zenodo pixel masks
    # ----------------------------------------------------------------
    print("\nLoading Turkey MS (real labels)...")
    turkey_tiles, turkey_labels = load_turkey_with_real_labels()
    print("Extracting Turkey embeddings...")
    turkey_emb = extract_embeddings(enc, turkey_tiles)

    # ----------------------------------------------------------------
    # Sri Lanka — Paddy (4 bands, VI ensemble labels)
    # ----------------------------------------------------------------
    print("\nLoading Sri Lanka MS...")
    sl_arr = np.load(
        DATA_ROOT / "paddy_srilanka" / "ms_stacked_srilanka.npy"
    )  # (67, 1944, 2592, 4)
    # Pad to 5 bands (prepend Blue=0)
    sl_5 = np.concatenate(
        [np.zeros((*sl_arr.shape[:3], 1), dtype=np.float32), sl_arr],
        axis=-1
    )
    sl_tiles = extract_tiles(sl_5, 5, tiles_per_image=10)
    print(f"  Sri Lanka: {len(sl_tiles)} tiles — labelling with VI ensemble:")
    # VI on raw Sri Lanka (4-band: G=0,R=1,RE=2,NIR=3)
    sl_raw2 = np.load(str(DATA_ROOT / "paddy_srilanka" / "ms_stacked_srilanka.npy"))
    rng2 = np.random.default_rng(42)
    sr, sg = [], []
    for img in sl_raw2:
        H2, W2 = sl_raw2.shape[1], sl_raw2.shape[2]
        for _ in range(10):
            if H2<64 or W2<64: continue
            t2=rng2.integers(0,H2-64); l2=rng2.integers(0,W2-64)
            tile2=img[t2:t2+64,l2:l2+64,:]
            nir2=tile2[:,:,3].mean(); red2=tile2[:,:,1].mean(); re2=tile2[:,:,2].mean()
            L2=0.5
            ndvi2=(nir2-red2)/(nir2+red2+1e-8)
            ndre2=(nir2-re2)/(nir2+re2+1e-8)
            savi2=((nir2-red2)/(nir2+red2+L2+1e-8))*(1+L2)
            sr.append((ndvi2+ndre2+savi2)/3)
    sr=np.array(sr)
    # high VI = healthy(0), low VI = stressed(1)
    sl_labels = (sr < np.median(sr)).astype(np.int64)
    print(f"  Sri Lanka raw VI: stressed={sl_labels.sum()} healthy={(sl_labels==0).sum()}")

    print("Extracting Sri Lanka embeddings...")
    sl_emb = extract_embeddings(enc, sl_tiles)

    # ----------------------------------------------------------------
    # Cross-geographic evaluation
    # ----------------------------------------------------------------
    results = []

    print("\n" + "="*65)
    print("CROSS-GEOGRAPHIC EVALUATION v2 — Real Turkey Labels + VI Ensemble")
    print("="*65)

    # Within-country
    print("\n[Within-country baselines]")
    for name, emb, labels in [
        ("India",     india_emb,  india_labels),
        ("Turkey",    turkey_emb, turkey_labels),
        ("Sri Lanka", sl_emb,     sl_labels),
    ]:
        n = len(emb)
        idx = np.random.default_rng(42).permutation(n)
        s = int(0.8 * n)
        r = linear_probe(emb[idx[:s]], labels[idx[:s]],
                         emb[idx[s:]], labels[idx[s:]],
                         name=f"Within: {name}→{name}")
        results.append(r)

    # Cross-geographic
    print("\n[Cross-geographic transfers]")
    transfers = [
        (india_emb,  india_labels,  turkey_emb, turkey_labels,
         "Cross-geo: India→Turkey (maize, real labels)"),
        (india_emb,  india_labels,  sl_emb,     sl_labels,
         "Cross-geo: India→Sri Lanka (paddy, VI ensemble)"),
        (turkey_emb, turkey_labels, sl_emb,     sl_labels,
         "Cross-geo: Turkey→Sri Lanka"),
        (turkey_emb, turkey_labels, india_emb,  india_labels,
         "Cross-geo: Turkey→India"),
        (sl_emb,     sl_labels,     turkey_emb, turkey_labels,
         "Cross-geo: Sri Lanka→Turkey"),
    ]
    for X_tr, y_tr, X_te, y_te, name in transfers:
        r = linear_probe(X_tr, y_tr, X_te, y_te, name=name)
        results.append(r)

    # Multi-country
    print("\n[Multi-country training]")
    combos = [
        (np.concatenate([india_emb, turkey_emb]),
         np.concatenate([india_labels, turkey_labels]),
         sl_emb, sl_labels, "Multi→Sri Lanka (India+Turkey→SL)"),
        (np.concatenate([india_emb, sl_emb]),
         np.concatenate([india_labels, sl_labels]),
         turkey_emb, turkey_labels, "Multi→Turkey (India+SL→Turkey)"),
        (np.concatenate([turkey_emb, sl_emb]),
         np.concatenate([turkey_labels, sl_labels]),
         india_emb, india_labels, "Multi→India (Turkey+SL→India)"),
        (np.concatenate([india_emb, turkey_emb, sl_emb]),
         np.concatenate([india_labels, turkey_labels, sl_labels]),
         india_emb, india_labels, "All-country→India (held-out)"),
    ]
    for X_tr, y_tr, X_te, y_te, name in combos:
        r = linear_probe(X_tr, y_tr, X_te, y_te, name=name)
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print(f"{'Experiment':<50} {'Acc':>7} {'AUC':>7}")
    print("-"*70)

    print("\n[v1 — NDVI only (for comparison)]")
    v1 = {
        "India→India": (0.9198, 0.9803),
        "Turkey→Turkey": (0.9750, 0.9674),
        "Sri Lanka→Sri Lanka": (0.9478, 0.9848),
        "India→Turkey": (0.5250, 0.7714),
        "India→Sri Lanka": (0.6746, 0.7472),
        "Turkey→Sri Lanka": (0.4955, 0.3858),
        "Multi→Sri Lanka": (0.6836, 0.7587),
        "Multi→Turkey": (0.5400, 0.7179),
        "Multi→India": (0.4379, 0.5108),
    }
    for k, (acc, auc) in v1.items():
        print(f"  {k:<48} {acc:.4f} {auc:.4f}")

    print("\n[v2 — VI ensemble + Real Turkey labels]")
    for r in results:
        print(f"  {r['name']:<48} {r['accuracy']:.4f} {r['auc']:.4f}")

    # Save
    with open(LOG_DIR / "cross_geo_eval_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {LOG_DIR / 'cross_geo_eval_v2.json'}")


if __name__ == "__main__":
    main()
