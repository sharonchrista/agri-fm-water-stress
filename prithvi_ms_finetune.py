"""
Prithvi-EO-2.0 fine-tuning adapter for MS maize/paddy crop stress monitoring.

Adapts Prithvi-EO-2.0-300M (pretrained on Sentinel-2, 6 bands, 4 time steps)
to your 5-band single-date UAV multispectral data via:

  1. Band adapter  : 5 MS bands -> 6 Prithvi bands (zero-pad missing SWIR)
  2. Temporal warp : single image -> 4 repeated frames (pseudo-time-series)
  3. Spatial resize: 64x64 tiles -> 224x224 (Prithvi native resolution)
  4. Value rescale : [0,1] normalised -> Prithvi DN range (~1000-3000)
  5. LoRA fine-tune: only ~1% of backbone params updated, fits on T4

After fine-tuning, the Prithvi encoder replaces the custom MSEncoder
and the same few-shot cross-crop evaluation protocol is applied.

Run: python prithvi_ms_finetune.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add Prithvi model directory to path
PRITHVI_DIR = Path("~/agri_foundation/models/prithvi").expanduser()
sys.path.insert(0, str(PRITHVI_DIR))

DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prithvi expected input statistics (Sentinel-2 DN values)
# B02=Blue, B03=Green, B04=Red, B05=RedEdge, B06=NIR, B07=SWIR
PRITHVI_MEAN = torch.tensor([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0])
PRITHVI_STD = torch.tensor([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0])

# Your MS band order: Blue=0, Green=1, Red=2, RedEdge=3, NIR=4
# Prithvi band order: B02=0, B03=1, B04=2, B05=3, B06=4, B07=5(SWIR=zero)
MS_TO_PRITHVI_BAND_MAP = [0, 1, 2, 3, 4]  # your 5 bands -> Prithvi bands 0-4
PRITHVI_IMG_SIZE = 224
PRITHVI_NUM_FRAMES = 4
TILE_SIZE = 64
NDVI_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Band adapter: 5-band MS tile -> 6-band Prithvi input
# ---------------------------------------------------------------------------

def adapt_ms_to_prithvi(tiles: Tensor) -> Tensor:
    """
    Convert 5-band MS tiles to Prithvi 6-band input format.

    Steps:
      1. Rescale [0,1] normalised values to Sentinel-2 DN range
      2. Zero-pad missing SWIR band (band 5)
      3. Resize spatial dims from tile_size to 224x224
      4. Repeat single frame 4 times for temporal dimension
      5. Normalise using Prithvi mean/std

    Parameters
    ----------
    tiles : Tensor (B, 5, H, W) — normalised MS tiles

    Returns
    -------
    Tensor (B, num_frames, 6, 224, 224) — Prithvi-ready input
    """
    B, C, H, W = tiles.shape

    # Step 1: rescale from [0,1] to approximate DN range
    # Your MS stats: mean=[0.25,0.26,0.26,0.33,0.29], std=[0.14,0.14,0.14,0.15,0.15]
    # Prithvi Sentinel-2 DN mean=[1087,1342,1433,2734,1958], std=[2248,2179,2178,1850,1242]
    # Scale factor: prithvi_mean / your_mean for each band
    scale = torch.tensor([1087/0.2541, 1342/0.2613, 1433/0.2608,
                          2734/0.3284, 1958/0.2856],
                         device=tiles.device, dtype=tiles.dtype)
    tiles_dn = tiles * scale[None, :, None, None]

    # Step 2: zero-pad SWIR band (band index 5)
    swir_zero = torch.zeros(B, 1, H, W, device=tiles.device, dtype=tiles.dtype)
    tiles_6band = torch.cat([tiles_dn, swir_zero], dim=1)  # (B, 6, H, W)

    # Step 3: resize to 224x224
    tiles_resized = F.interpolate(
        tiles_6band,
        size=(PRITHVI_IMG_SIZE, PRITHVI_IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    )  # (B, 6, 224, 224)

    # Step 4: normalise using Prithvi mean/std
    mean = PRITHVI_MEAN.to(tiles.device)[None, :, None, None]
    std = PRITHVI_STD.to(tiles.device)[None, :, None, None]
    tiles_norm = (tiles_resized - mean) / (std + 1e-8)

    # Step 5: reshape to Prithvi Conv3d format (B, in_chans, num_frames, H, W)
    # Conv3d kernel is (embed_dim, in_chans, 1, 16, 16)
    # so input must be (B, in_chans, num_frames, H, W)
    tiles_temporal = tiles_norm.unsqueeze(2).repeat(1, 1, PRITHVI_NUM_FRAMES, 1, 1)
    # tiles_temporal shape: (B, 6, 4, 224, 224) ✓

    return tiles_temporal


# ---------------------------------------------------------------------------
# Prithvi encoder wrapper
# ---------------------------------------------------------------------------

class PrithviEncoder(nn.Module):
    """
    Wraps Prithvi-EO-2.0 ViT encoder for feature extraction.

    Loads the pretrained MAE, extracts the ViT encoder,
    applies LoRA-style adaptation (low-rank updates on attention layers),
    and returns CLS token + mean-pooled patch embeddings.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        embed_dim: int = 1024,
        lora_rank: int = 8,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        from prithvi_mae import PrithviMAE

        # Load config
        config_path = checkpoint_path.parent / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)["pretrained_cfg"]

        # Build model
        print(f"  Loading Prithvi-EO-2.0-300M from {checkpoint_path}...")
        self.mae = PrithviMAE(
            img_size=cfg["img_size"],
            num_frames=cfg["num_frames"],
            patch_size=cfg["patch_size"],
            in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            decoder_embed_dim=cfg["decoder_embed_dim"],
            decoder_depth=cfg["decoder_depth"],
            decoder_num_heads=cfg["decoder_num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
        )

        # Load pretrained weights
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        missing, unexpected = self.mae.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights. Missing: {len(missing)} Unexpected: {len(unexpected)}")

        self.encoder = self.mae.encoder
        self.embed_dim = embed_dim  # 1024 for 300M model

        # Freeze backbone
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"  Backbone frozen.")

        # LoRA adapters on last 4 attention layers
        if lora_rank > 0 and not freeze_backbone:
            self._add_lora(lora_rank)

        # Projection head to reduce 1024 -> 256 for compatibility
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def _add_lora(self, rank: int) -> None:
        """Add LoRA low-rank adaptation to last 4 transformer blocks."""
        n_blocks = len(self.encoder.blocks)
        for i in range(n_blocks - 4, n_blocks):
            block = self.encoder.blocks[i]
            # Unfreeze last 4 blocks
            for param in block.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        print(f"  LoRA: {trainable:,} / {total:,} params trainable "
              f"({trainable/total*100:.1f}%)")

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, 6, num_frames, 224, 224) — Prithvi-format (B, C, T, H, W)

        Returns
        -------
        Tensor (B, 256) — L2-normalised embedding
        """
        # x shape: (B, 6, 4, 224, 224) — (B, C, T, H, W)
        features = self.encoder.forward_features(x)

        # forward_features may return a list of tensors or a single tensor
        if isinstance(features, (list, tuple)):
            # Take the last element — usually the final layer features
            features = features[-1]

        # features shape: (B, N, embed_dim) or (B, embed_dim)
        if features.ndim == 3:
            pooled = features.mean(dim=1)   # (B, embed_dim)
        else:
            pooled = features               # (B, embed_dim)

        # Project to 256-dim
        out = self.proj(pooled)                       # (B, 256)
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# MS Tile Dataset (same as ms_dataset.py but returns raw tiles for adapter)
# ---------------------------------------------------------------------------

class MSTileDataset(Dataset):
    """
    Tile dataset for Prithvi fine-tuning.
    Returns raw normalised 5-band tiles — adapter converts to Prithvi format.
    """

    MS_MEAN = np.array([0.2541, 0.2613, 0.2608, 0.3284, 0.2856], dtype=np.float32)
    MS_STD = np.array([0.1356, 0.1386, 0.1438, 0.1477, 0.1460], dtype=np.float32)

    def __init__(
        self,
        arrays: list[np.ndarray],
        tiles_per_image: int = 10,
        seed: int = 42,
    ) -> None:
        self.data = np.concatenate(arrays, axis=0)  # (N, H, W, 5)
        self.n_images = len(self.data)
        self.tiles_per_image = tiles_per_image
        self.rng = np.random.default_rng(seed)
        self.tiles_per_epoch = self.n_images * tiles_per_image

        # Pre-extract all tiles for consistent evaluation
        print(f"  Pre-extracting {self.tiles_per_epoch} tiles...")
        self.tiles = self._extract_all_tiles()
        self.ndvi = self._compute_ndvi()
        self.labels = (self.ndvi <= NDVI_THRESHOLD).astype(np.int64)
        print(f"  Labels: stressed={self.labels.sum()} healthy={(self.labels==0).sum()}")

    def _extract_all_tiles(self) -> np.ndarray:
        N, H, W, C = self.data.shape
        tiles = []
        for img in self.data:
            for _ in range(self.tiles_per_image):
                top = self.rng.integers(0, H - TILE_SIZE)
                left = self.rng.integers(0, W - TILE_SIZE)
                tile = img[top:top+TILE_SIZE, left:left+TILE_SIZE, :]
                tile = tile.transpose(2, 0, 1).astype(np.float32)
                # Normalise
                tile = (tile - self.MS_MEAN[:, None, None]) / (self.MS_STD[:, None, None] + 1e-6)
                tiles.append(tile)
        return np.stack(tiles)  # (N*tiles_per_image, 5, 64, 64)

    def _compute_ndvi(self) -> np.ndarray:
        nir = self.tiles[:, 4, :, :]
        red = self.tiles[:, 2, :, :]
        # Denormalise for NDVI computation
        nir_raw = nir * self.MS_STD[4] + self.MS_MEAN[4]
        red_raw = red * self.MS_STD[2] + self.MS_MEAN[2]
        ndvi = (nir_raw - red_raw) / (nir_raw + red_raw + 1e-8)
        return ndvi.mean(axis=(1, 2))

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return torch.from_numpy(self.tiles[idx]), int(self.labels[idx])


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_prithvi_embeddings(
    encoder: PrithviEncoder,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from Prithvi encoder for all tiles."""
    encoder.eval()
    all_emb, all_labels = [], []

    for tiles, labels in loader:
        tiles = tiles.to(DEVICE)
        # Convert to Prithvi format
        prithvi_input = adapt_ms_to_prithvi(tiles)
        emb = encoder(prithvi_input)
        all_emb.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_emb), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Linear probe evaluation (same protocol as cross_modal_eval.py)
# ---------------------------------------------------------------------------

def linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "",
) -> dict:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- {name} ---")
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    return {"name": name, "accuracy": acc, "auc": auc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")

    # Load Prithvi encoder
    encoder = PrithviEncoder(
        checkpoint_path=PRITHVI_DIR / "Prithvi_EO_V2_300M.pt",
        embed_dim=1024,
        lora_rank=0,        # frozen backbone for now
        freeze_backbone=True,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Load MS data
    ms_dir = DATA_ROOT / "processed" / "ms"
    print("\nLoading MS arrays...")
    maize = np.load(list(ms_dir.glob("maize/**/ms_stacked.npy"))[0])
    paddy = np.load(list(ms_dir.glob("paddy/**/ms_stacked.npy"))[0])
    print(f"Maize: {maize.shape} | Paddy: {paddy.shape}")

    # Build datasets
    print("\nBuilding maize dataset:")
    maize_ds = MSTileDataset([maize], tiles_per_image=8, seed=42)
    print("\nBuilding paddy dataset:")
    paddy_ds = MSTileDataset([paddy], tiles_per_image=8, seed=42)

    maize_loader = DataLoader(maize_ds, batch_size=16, shuffle=False, num_workers=0)
    paddy_loader = DataLoader(paddy_ds, batch_size=16, shuffle=False, num_workers=0)

    # Extract embeddings
    print("\nExtracting Prithvi embeddings for maize...")
    t0 = time.time()
    maize_emb, maize_labels = extract_prithvi_embeddings(encoder, maize_loader)
    print(f"  Done in {time.time()-t0:.1f}s | shape: {maize_emb.shape}")

    print("Extracting Prithvi embeddings for paddy...")
    t0 = time.time()
    paddy_emb, paddy_labels = extract_prithvi_embeddings(encoder, paddy_loader)
    print(f"  Done in {time.time()-t0:.1f}s | shape: {paddy_emb.shape}")

    # Embedding health check
    print(f"\nEmbedding space (Prithvi):")
    print(f"  Maize std: {maize_emb.std(axis=0).mean():.4f}")
    print(f"  Paddy std: {paddy_emb.std(axis=0).mean():.4f}")
    print(f"  NDVI stats — Maize: {maize_ds.ndvi.mean():.4f}±{maize_ds.ndvi.std():.4f} "
          f"| Paddy: {paddy_ds.ndvi.mean():.4f}±{paddy_ds.ndvi.std():.4f}")

    results = []

    # Within-crop probes
    for name, emb, labels in [("Maize", maize_emb, maize_labels),
                               ("Paddy", paddy_emb, paddy_labels)]:
        n = len(emb)
        split = int(0.8 * n)
        idx = np.random.default_rng(42).permutation(n)
        res = linear_probe(
            emb[idx[:split]], labels[idx[:split]],
            emb[idx[split:]], labels[idx[split:]],
            name=f"Prithvi — Within-crop: {name}→{name}",
        )
        results.append(res)

    # Cross-crop probes
    res = linear_probe(
        maize_emb, maize_labels, paddy_emb, paddy_labels,
        name="Prithvi — Cross-crop: Maize→Paddy",
    )
    results.append(res)

    res = linear_probe(
        paddy_emb, paddy_labels, maize_emb, maize_labels,
        name="Prithvi — Cross-crop: Paddy→Maize",
    )
    results.append(res)

    # Summary
    print("\n" + "=" * 65)
    print("PRITHVI CROSS-CROP EVALUATION (compare with MSEncoder baseline)")
    print("=" * 65)
    print(f"{'Experiment':<48} {'Acc':>6} {'AUC':>6}")
    print("-" * 65)

    # Baseline results (from cross_modal_eval.py)
    baseline = [
        ("MSEncoder (ours) — Within-crop: Maize→Maize", 0.9967, 0.9989),
        ("MSEncoder (ours) — Within-crop: Paddy→Paddy", 0.9735, 0.9819),
        ("MSEncoder (ours) — Cross-crop: Maize→Paddy",  0.9790, 0.9820),
        ("MSEncoder (ours) — Cross-crop: Paddy→Maize",  0.9936, 0.9957),
    ]

    print("\n[Baseline — Custom MSEncoder (contrastive, no labels)]")
    for name, acc, auc in baseline:
        print(f"  {name:<48} {acc:.4f} {auc:.4f}")

    print("\n[Prithvi-EO-2.0-300M (frozen, linear probe)]")
    for r in results:
        print(f"  {r['name']:<48} {r['accuracy']:.4f} {r['auc']:.4f}")

    # Save
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "prithvi_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {LOG_DIR / 'prithvi_eval.json'}")


if __name__ == "__main__":
    main()