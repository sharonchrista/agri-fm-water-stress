"""
Prithvi-EO-2.0 Spatial-Physics JEPA Adapter
for Cross-Geographic MS Crop Stress Monitoring.

Architecture:
  Prithvi-EO-2.0 (frozen, 300M params)
      ↓
  Spatial-Physics JEPA Adapter (trainable, ~500K params)
  - Context encoder: mask 40% spatial patches
  - Physics constraint: preferentially mask high-NDVI patches
    (vegetation-dense regions carry most stress signal)
  - Target encoder: EMA copy (τ=0.996)
  - Loss: cosine similarity + variance regularisation
      ↓
  256-dim adapted MS embedding
      ↓
  Prototypical few-shot head (cross-crop)
  GP uncertainty head

Datasets (multi-country MS pretraining):
  India   -- Maize + Paddy MS 5-band (TIAND)
  Turkey  -- Maize MS 6-band (Zenodo 22062459)
  Sri Lanka -- Paddy MS 4-band (Mendeley)

Cross-geographic evaluation:
  Train: India maize/paddy
  Test:  Turkey maize, Sri Lanka paddy

Run: python prithvi_jepa_ms.py
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

PRITHVI_DIR = Path("~/agri_foundation/models/prithvi").expanduser()
sys.path.insert(0, str(PRITHVI_DIR))

DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prithvi config
PRITHVI_EMBED_DIM = 1024   # 300M model
PRITHVI_NUM_FRAMES = 4
PRITHVI_IMG_SIZE = 224
PRITHVI_BANDS = 6

# JEPA config
ADAPTER_DIM = 256
TILE_SIZE = 64
PATCH_SIZE = 8              # spatial patch for masking (8x8 pixels)
MASK_RATIO = 0.40           # 40% of patches masked
NDVI_PHYSICS_PROB = 0.70    # 70% of masked patches from high-NDVI regions
EMA_DECAY = 0.996
BATCH_SIZE = 8              # small — Prithvi is 300M params
EPOCHS = 100
LR = 5e-5

# MS band indices (5-band: B, G, R, RE, NIR)
NIR_IDX = 4
RED_IDX = 2

# Prithvi normalisation (Sentinel-2 DN statistics)
PRITHVI_MEAN = torch.tensor([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0])
PRITHVI_STD = torch.tensor([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0])

# MS dataset statistics (normalised [0,1])
MS_MEAN = np.array([0.2541, 0.2613, 0.2608, 0.3284, 0.2856], dtype=np.float32)
MS_STD = np.array([0.1356, 0.1386, 0.1438, 0.1477, 0.1460], dtype=np.float32)


# ---------------------------------------------------------------------------
# Band adapter: N-band MS tile -> 6-band Prithvi input
# ---------------------------------------------------------------------------

def adapt_ms_to_prithvi(tiles: Tensor, n_bands: int = 5) -> Tensor:
    """
    Convert N-band MS tiles to Prithvi 6-band spatiotemporal input.

    Steps:
      1. Rescale [0,1] normalised values to Sentinel-2 DN range
      2. Zero-pad missing bands to reach 6 bands
      3. Resize spatial dims to 224x224
      4. Normalise using Prithvi mean/std
      5. Repeat single frame 4 times -> temporal dimension

    Parameters
    ----------
    tiles : Tensor (B, N, H, W) — normalised MS tiles
    n_bands : number of input bands (5 for India/TIAND, 4 for Sri Lanka,
              5 for Turkey after dropping Alpha)

    Returns
    -------
    Tensor (B, 6, 4, 224, 224) — Prithvi-ready (C, T, H, W) format
    """
    B, C, H, W = tiles.shape

    # Rescale to approximate DN range using band-specific scale factors
    if C == 5:
        scale = torch.tensor(
            [1087/0.2541, 1342/0.2613, 1433/0.2608, 2734/0.3284, 1958/0.2856],
            device=tiles.device, dtype=tiles.dtype
        )
    elif C == 4:  # Sri Lanka: G, R, RE, NIR — no Blue
        scale = torch.tensor(
            [1342/0.2613, 1433/0.2608, 2734/0.3284, 1958/0.2856],
            device=tiles.device, dtype=tiles.dtype
        )
    else:
        scale = torch.ones(C, device=tiles.device, dtype=tiles.dtype) * 1500.0

    tiles_dn = tiles * scale[None, :, None, None]

    # Zero-pad to 6 bands
    if C < 6:
        pad = torch.zeros(B, 6 - C, H, W, device=tiles.device, dtype=tiles.dtype)
        tiles_6 = torch.cat([tiles_dn, pad], dim=1)
    else:
        tiles_6 = tiles_dn[:, :6]

    # Resize to 224x224
    tiles_resized = F.interpolate(
        tiles_6, size=(PRITHVI_IMG_SIZE, PRITHVI_IMG_SIZE),
        mode="bilinear", align_corners=False
    )

    # Normalise
    mean = PRITHVI_MEAN.to(tiles.device)[None, :, None, None]
    std = PRITHVI_STD.to(tiles.device)[None, :, None, None]
    tiles_norm = (tiles_resized - mean) / (std + 1e-8)

    # Temporal: repeat 4 times -> (B, 6, 4, 224, 224)
    tiles_temporal = tiles_norm.unsqueeze(2).repeat(1, 1, PRITHVI_NUM_FRAMES, 1, 1)

    return tiles_temporal


# ---------------------------------------------------------------------------
# Compute NDVI for physics-aware patch masking
# ---------------------------------------------------------------------------

def compute_tile_ndvi(tile: np.ndarray, nir_idx: int, red_idx: int) -> np.ndarray:
    """
    Compute per-pixel NDVI for a tile.
    tile: (H, W, C) normalised
    Returns: (H, W) NDVI map in [-1, 1]
    """
    nir = tile[:, :, nir_idx] * MS_STD[nir_idx] + MS_MEAN[nir_idx]
    red = tile[:, :, red_idx] * MS_STD[red_idx] + MS_MEAN[red_idx]
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi.clip(-1, 1)


def physics_spatial_mask(
    H: int,
    W: int,
    patch_size: int,
    ndvi_map: np.ndarray,
    mask_ratio: float = MASK_RATIO,
    physics_prob: float = NDVI_PHYSICS_PROB,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Physics-constrained spatial patch mask.
    High-NDVI patches (vegetation-dense) carry most stress signal —
    preferentially mask these.

    Returns: (n_patches_h, n_patches_w) bool mask
    """
    if rng is None:
        rng = np.random.default_rng()

    n_h = H // patch_size
    n_w = W // patch_size
    n_patches = n_h * n_w
    n_masked = max(1, int(n_patches * mask_ratio))

    # Compute mean NDVI per patch
    ndvi_patches = np.zeros(n_h * n_w)
    for i in range(n_h):
        for j in range(n_w):
            patch_ndvi = ndvi_map[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ]
            ndvi_patches[i * n_w + j] = patch_ndvi.mean()

    # High-NDVI patches (vegetation) — physics-motivated
    median_ndvi = np.median(ndvi_patches)
    high_ndvi_idx = np.where(ndvi_patches >= median_ndvi)[0]
    low_ndvi_idx = np.where(ndvi_patches < median_ndvi)[0]

    mask_flat = np.zeros(n_patches, dtype=bool)
    n_physics = int(n_masked * physics_prob)
    n_random = n_masked - n_physics

    if n_physics > 0 and len(high_ndvi_idx) > 0:
        sampled = rng.choice(
            high_ndvi_idx,
            size=min(n_physics, len(high_ndvi_idx)),
            replace=False
        )
        mask_flat[sampled] = True

    remaining = np.where(~mask_flat)[0]
    if n_random > 0 and len(remaining) > 0:
        sampled = rng.choice(
            remaining,
            size=min(n_random, len(remaining)),
            replace=False
        )
        mask_flat[sampled] = True

    return mask_flat.reshape(n_h, n_w)


# ---------------------------------------------------------------------------
# Multi-country MS tile dataset
# ---------------------------------------------------------------------------

class MultiCountryMSDataset(Dataset):
    """
    Combined MS tiles from India, Turkey, Sri Lanka for JEPA pretraining.
    Returns tile pairs for JEPA training (context + mask).
    """

    def __init__(
        self,
        tiles: np.ndarray,          # (N, H, W, C) normalised
        ndvi_maps: np.ndarray,      # (N, H, W) NDVI
        n_bands: int = 5,
        use_physics_masking: bool = True,
    ) -> None:
        self.tiles = tiles.astype(np.float32)
        self.ndvi_maps = ndvi_maps.astype(np.float32)
        self.n_bands = n_bands
        self.use_physics_masking = use_physics_masking
        self.rng = np.random.default_rng(42)
        _, self.H, self.W, self.C = tiles.shape
        self.n_h = self.H // PATCH_SIZE
        self.n_w = self.W // PATCH_SIZE
        print(f"  MS dataset: {len(tiles)} tiles "
              f"({self.H}x{self.W}x{self.C}), "
              f"patch grid: {self.n_h}x{self.n_w}")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        tile = self.tiles[idx]       # (H, W, C)
        ndvi = self.ndvi_maps[idx]   # (H, W)

        # Generate spatial patch mask
        if self.use_physics_masking:
            mask = physics_spatial_mask(
                self.H, self.W, PATCH_SIZE, ndvi,
                mask_ratio=MASK_RATIO,
                physics_prob=NDVI_PHYSICS_PROB,
                rng=self.rng,
            )
        else:
            n_patches = self.n_h * self.n_w
            n_masked = max(1, int(n_patches * MASK_RATIO))
            flat = np.zeros(n_patches, dtype=bool)
            flat[self.rng.choice(n_patches, n_masked, replace=False)] = True
            mask = flat.reshape(self.n_h, self.n_w)

        # Convert tile to (C, H, W) tensor
        tile_t = torch.from_numpy(tile.transpose(2, 0, 1))  # (C, H, W)
        mask_t = torch.from_numpy(mask)                      # (n_h, n_w)

        return tile_t, mask_t


# ---------------------------------------------------------------------------
# Prithvi backbone loader
# ---------------------------------------------------------------------------

def load_prithvi_encoder(device: torch.device) -> nn.Module:
    """Load Prithvi-EO-2.0-300M encoder, frozen."""
    from prithvi_mae import PrithviMAE
    import json

    config_path = PRITHVI_DIR / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)["pretrained_cfg"]

    mae = PrithviMAE(
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

    ckpt = torch.load(
        PRITHVI_DIR / "Prithvi_EO_V2_300M.pt",
        map_location="cpu", weights_only=False
    )
    state = ckpt.get("model", ckpt)
    mae.load_state_dict(state, strict=False)

    encoder = mae.encoder
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"Prithvi encoder loaded and frozen "
          f"({sum(p.numel() for p in encoder.parameters()):,} params)")
    return encoder.to(device)


# ---------------------------------------------------------------------------
# Spatial JEPA components for Prithvi
# ---------------------------------------------------------------------------

class PrithviContextEncoder(nn.Module):
    """
    Lightweight spatial JEPA context encoder on top of frozen Prithvi.
    Masks spatial patches, encodes visible context.
    """

    def __init__(
        self,
        prithvi_encoder: nn.Module,
        prithvi_embed_dim: int = PRITHVI_EMBED_DIM,
        adapter_dim: int = ADAPTER_DIM,
    ) -> None:
        super().__init__()
        self.prithvi = prithvi_encoder

        # Spatial adapter: project Prithvi 1024-dim to 256-dim
        self.adapter = nn.Sequential(
            nn.Linear(prithvi_embed_dim, adapter_dim * 2),
            nn.LayerNorm(adapter_dim * 2),
            nn.GELU(),
            nn.Linear(adapter_dim * 2, adapter_dim),
            nn.LayerNorm(adapter_dim),
        )
        self.embed_dim = adapter_dim

    def _apply_spatial_mask(
        self,
        prithvi_input: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Zero out masked spatial regions in Prithvi input.
        prithvi_input: (B, 6, 4, 224, 224)
        mask: (B, n_h, n_w) bool — True = masked
        """
        B, C, T, H, W = prithvi_input.shape
        n_h, n_w = mask.shape[1], mask.shape[2]
        ph = H // n_h
        pw = W // n_w

        masked = prithvi_input.clone()
        for i in range(n_h):
            for j in range(n_w):
                m = mask[:, i, j]  # (B,) bool
                if m.any():
                    masked[m, :, :,
                           i*ph:(i+1)*ph,
                           j*pw:(j+1)*pw] = 0.0
        return masked

    def forward(self, prithvi_input: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        prithvi_input : (B, 6, 4, 224, 224)
        mask : (B, n_h, n_w) bool

        Returns
        -------
        (B, adapter_dim) adapted embedding
        """
        # Apply spatial mask
        masked_input = self._apply_spatial_mask(prithvi_input, mask)

        # Prithvi forward (frozen)
        with torch.no_grad():
            features = self.prithvi.forward_features(masked_input)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            if features.ndim == 3:
                features = features.mean(dim=1)
            elif features.ndim == 2 and features.shape[1] == 1:
                features = features.squeeze(1)

        # Adapter
        return F.normalize(self.adapter(features), dim=-1)


class PrithviTargetEncoder(nn.Module):
    """EMA target encoder for Prithvi spatial JEPA."""

    def __init__(self, context_encoder: PrithviContextEncoder) -> None:
        super().__init__()
        import copy
        self.encoder = copy.deepcopy(context_encoder)
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(
        self,
        context_encoder: PrithviContextEncoder,
        decay: float = EMA_DECAY,
    ) -> None:
        for pt, pc in zip(
            self.encoder.parameters(),
            context_encoder.parameters(),
        ):
            pt.data = decay * pt.data + (1 - decay) * pc.data

    def forward(self, prithvi_input: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            return self.encoder(prithvi_input, mask)


class SpatialJEPAPredictor(nn.Module):
    def __init__(self, embed_dim: int = ADAPTER_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Load multi-country MS data
# ---------------------------------------------------------------------------

def load_multicountry_ms(
    data_root: Path,
    tiles_per_image: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load MS tiles from India, Turkey, Sri Lanka.
    Returns combined tiles (N, 64, 64, 5), NDVI maps (N, 64, 64), metadata.
    """
    rng = np.random.default_rng(seed)
    all_tiles = []
    all_ndvi = []
    metadata = {}

    def extract_tiles(arr: np.ndarray, n_bands: int, country: str, crop: str):
        """Extract random 64x64 tiles from (N, H, W, C) array."""
        N, H, W, C = arr.shape
        tiles, ndvi_maps = [], []
        for img in arr:
            for _ in range(tiles_per_image):
                if H < TILE_SIZE or W < TILE_SIZE:
                    continue
                top = rng.integers(0, H - TILE_SIZE)
                left = rng.integers(0, W - TILE_SIZE)
                tile = img[top:top+TILE_SIZE, left:left+TILE_SIZE, :n_bands]
                tile = tile.astype(np.float32)
                # Normalise
                tile = (tile - MS_MEAN[:n_bands]) / (MS_STD[:n_bands] + 1e-6)
                tile = tile.clip(-3, 3)
                tiles.append(tile)
                # NDVI (use available NIR and Red)
                nir_i = min(n_bands-1, NIR_IDX)
                red_i = min(2, RED_IDX)
                ndvi = compute_tile_ndvi(tile, nir_i, red_i)
                ndvi_maps.append(ndvi)

        print(f"  {country} ({crop}): {len(tiles)} tiles from {N} images")
        metadata[f"{country}_{crop}"] = {
            "n_tiles": len(tiles), "n_images": N,
            "bands": n_bands, "country": country, "crop": crop
        }
        return tiles, ndvi_maps

    # India — Maize + Paddy MS (5 bands)
    ms_dir = data_root / "processed" / "ms"
    maize_files = list(ms_dir.glob("maize/**/ms_stacked.npy"))
    paddy_files = list(ms_dir.glob("paddy/**/ms_stacked.npy"))

    if maize_files:
        maize = np.load(maize_files[0])
        t, n = extract_tiles(maize, 5, "India", "maize")
        all_tiles.extend(t); all_ndvi.extend(n)

    if paddy_files:
        paddy = np.load(paddy_files[0])
        t, n = extract_tiles(paddy, 5, "India", "paddy")
        all_tiles.extend(t); all_ndvi.extend(n)

    # Turkey — Zenodo Maize MS (6 bands, drop Alpha band 5)
    zenodo_dir = data_root / "zenodo_maize_ms" / "processed_patches"
    water_dir = zenodo_dir / "water_2025" / "images"
    if water_dir.exists():
        zenodo_tiles, zenodo_ndvi = [], []
        npy_files = list(water_dir.glob("*.npy"))
        for f in npy_files[:100]:  # limit to 100 patches
            tile = np.load(f).astype(np.float32)  # (224, 224, 6)
            tile_5 = tile[:, :, :5]  # drop Alpha
            # Downsample to 64x64
            tile_64 = torch.from_numpy(
                tile_5.transpose(2,0,1)
            ).unsqueeze(0)
            tile_64 = F.interpolate(
                tile_64, size=(TILE_SIZE, TILE_SIZE),
                mode='bilinear', align_corners=False
            ).squeeze(0).permute(1,2,0).numpy()
            ndvi = compute_tile_ndvi(tile_64, NIR_IDX, RED_IDX)
            zenodo_tiles.append(tile_64)
            zenodo_ndvi.append(ndvi)
        print(f"  Turkey (maize): {len(zenodo_tiles)} tiles from Zenodo")
        metadata["Turkey_maize"] = {
            "n_tiles": len(zenodo_tiles), "bands": 5, "country": "Turkey"
        }
        all_tiles.extend(zenodo_tiles)
        all_ndvi.extend(zenodo_ndvi)

    # Sri Lanka — Paddy MS (4 bands: G, R, RE, NIR)
    srilanka_path = data_root / "paddy_srilanka" / "ms_stacked_srilanka.npy"
    if srilanka_path.exists():
        sl = np.load(srilanka_path)  # (67, 1944, 2592, 4)
        # Pad to 5 bands (add Blue=0)
        sl_5 = np.concatenate(
            [np.zeros((*sl.shape[:3], 1), dtype=np.float32), sl],
            axis=-1
        )
        t, n = extract_tiles(sl_5, 5, "Sri Lanka", "paddy")
        all_tiles.extend(t); all_ndvi.extend(n)

    # Stack and pad all tiles to same band count (5)
    combined_tiles = []
    for tile in all_tiles:
        if tile.shape[2] < 5:
            pad = np.zeros((*tile.shape[:2], 5 - tile.shape[2]),
                          dtype=np.float32)
            tile = np.concatenate([tile, pad], axis=-1)
        combined_tiles.append(tile)

    tiles_arr = np.stack(combined_tiles, axis=0)   # (N, H, W, 5)
    ndvi_arr = np.stack(all_ndvi, axis=0)           # (N, H, W)

    print(f"\nTotal MS tiles: {len(tiles_arr)} from "
          f"{len(metadata)} country-crop combinations")
    return tiles_arr, ndvi_arr, metadata


# ---------------------------------------------------------------------------
# JEPA training
# ---------------------------------------------------------------------------

def train_prithvi_jepa_epoch(
    ctx: PrithviContextEncoder,
    tgt: PrithviTargetEncoder,
    pred: SpatialJEPAPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> float:
    ctx.train()
    pred.train()
    total_loss = 0.0
    n = 0

    for tiles, masks in loader:
        # tiles: (B, C, H, W), masks: (B, n_h, n_w)
        tiles = tiles.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        # Convert to Prithvi format
        prithvi_input = adapt_ms_to_prithvi(tiles, n_bands=tiles.shape[1])

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            ctx_emb = ctx(prithvi_input, masks)
            pred_emb = pred(ctx_emb)

            with torch.no_grad():
                tgt_emb = tgt(prithvi_input, masks)

            # JEPA loss
            pred_n = F.normalize(pred_emb, dim=-1)
            tgt_n = F.normalize(tgt_emb.detach(), dim=-1)
            cos_loss = 1.0 - (pred_n * tgt_n).sum(dim=-1).mean()
            var_loss = F.relu(1.0 - pred_emb.std(dim=0).mean())
            loss = cos_loss + 0.1 * var_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(ctx.adapter.parameters()) + list(pred.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        tgt.update_ema(ctx)

        total_loss += loss.item()
        n += 1

    return total_loss / n


# ---------------------------------------------------------------------------
# Cross-crop evaluation with linear probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_prithvi_jepa_embeddings(
    ctx: PrithviContextEncoder,
    tiles: np.ndarray,
    batch_size: int = 4,
) -> np.ndarray:
    """Extract adapted Prithvi embeddings (no masking at inference)."""
    ctx.eval()
    all_emb = []

    for start in range(0, len(tiles), batch_size):
        batch = torch.from_numpy(
            tiles[start:start+batch_size].transpose(0, 3, 1, 2)
        ).to(DEVICE)
        # No masking at inference
        n_h = tiles.shape[1] // PATCH_SIZE
        n_w = tiles.shape[2] // PATCH_SIZE
        B = batch.shape[0]
        mask = torch.zeros(B, n_h, n_w, dtype=torch.bool, device=DEVICE)
        prithvi_input = adapt_ms_to_prithvi(batch, n_bands=batch.shape[1])
        emb = ctx(prithvi_input, mask)
        all_emb.append(emb.cpu().numpy())

    return np.concatenate(all_emb)


def linear_probe_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "",
) -> dict:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(
        max_iter=1000, C=1.0, random_state=42, class_weight="balanced"
    )
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")
    print(f"  {name}: Acc={acc:.4f} AUC={auc:.4f}")
    return {"name": name, "accuracy": acc, "auc": auc}


def ndvi_labels(tiles: np.ndarray, nir_idx: int = 4, red_idx: int = 2) -> np.ndarray:
    """Median-NDVI binary labels for tiles."""
    nir = tiles[:, :, :, nir_idx].mean(axis=(1, 2))
    red = tiles[:, :, :, red_idx].mean(axis=(1, 2))
    ndvi = (nir - red) / (nir + red + 1e-8)
    return (ndvi > np.median(ndvi)).astype(np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load multi-country MS data
    print("\nLoading multi-country MS tiles...")
    tiles, ndvi_maps, metadata = load_multicountry_ms(
        DATA_ROOT, tiles_per_image=8
    )

    # Load Prithvi encoder
    print("\nLoading Prithvi-EO-2.0...")
    prithvi = load_prithvi_encoder(DEVICE)

    results_all = {}

    for use_physics in [True, False]:
        mask_type = "physics_constrained" if use_physics else "random"
        print(f"\n{'='*55}")
        print(f"Prithvi-JEPA Masking: {mask_type.upper()}")
        print(f"{'='*55}")

        # Build JEPA components
        ctx = PrithviContextEncoder(prithvi, PRITHVI_EMBED_DIM, ADAPTER_DIM).to(DEVICE)
        tgt = PrithviTargetEncoder(ctx).to(DEVICE)
        pred = SpatialJEPAPredictor(ADAPTER_DIM).to(DEVICE)

        trainable = sum(p.numel() for p in ctx.adapter.parameters()) + \
                    sum(p.numel() for p in pred.parameters())
        print(f"Trainable params: {trainable:,}")

        # Dataset and loader
        dataset = MultiCountryMSDataset(
            tiles, ndvi_maps,
            use_physics_masking=use_physics,
        )
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=2,
            pin_memory=True, drop_last=True,
        )
        print(f"Loader: {len(loader)} batches/epoch")

        optimizer = torch.optim.AdamW(
            list(ctx.adapter.parameters()) + list(pred.parameters()),
            lr=LR, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR * 0.01
        )
        scaler = torch.amp.GradScaler("cuda",
                                       enabled=torch.cuda.is_available())

        # Baseline before JEPA
        print("\n--- Extracting baseline embeddings (no JEPA)...")
        india_tiles = tiles[:len(tiles)//2]  # approximate India portion
        turkey_start = len(tiles) - 800      # approximate Turkey start
        india_emb = extract_prithvi_jepa_embeddings(ctx, india_tiles)
        turkey_emb = extract_prithvi_jepa_embeddings(
            ctx, tiles[turkey_start:]
        )
        india_labels = ndvi_labels(india_tiles)
        turkey_labels = ndvi_labels(tiles[turkey_start:])

        print("Baseline cross-geographic (India→Turkey):")
        baseline = linear_probe_eval(
            india_emb, india_labels,
            turkey_emb, turkey_labels,
            name=f"Baseline India→Turkey"
        )

        # JEPA training
        print(f"\nTraining Prithvi-JEPA ({mask_type}) for {EPOCHS} epochs...")
        print(f"{'Epoch':>6} {'Loss':>10} {'Time':>7}")
        print("-" * 28)

        best_loss = float("inf")
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            loss = train_prithvi_jepa_epoch(
                ctx, tgt, pred, loader, optimizer, scaler
            )
            scheduler.step()
            elapsed = time.time() - t0

            if epoch % 10 == 0 or epoch == 1:
                print(f"{epoch:>6} {loss:>10.4f} {elapsed:>6.1f}s")

            if loss < best_loss:
                best_loss = loss
                torch.save({
                    "epoch": epoch,
                    "context_encoder": ctx.state_dict(),
                    "predictor": pred.state_dict(),
                    "loss": best_loss,
                    "mask_type": mask_type,
                }, CHECKPOINT_DIR / f"prithvi_jepa_{mask_type}_best.pt")

        print(f"Best loss: {best_loss:.4f}")

        # Post-JEPA cross-geographic evaluation
        print("\n--- Post-JEPA cross-geographic evaluation...")
        india_emb_post = extract_prithvi_jepa_embeddings(ctx, india_tiles)
        turkey_emb_post = extract_prithvi_jepa_embeddings(
            ctx, tiles[turkey_start:]
        )

        post = linear_probe_eval(
            india_emb_post, india_labels,
            turkey_emb_post, turkey_labels,
            name=f"Post-JEPA ({mask_type}) India→Turkey"
        )

        results_all[mask_type] = {
            "baseline": baseline,
            "post_jepa": post,
            "best_loss": best_loss,
        }

    # Summary
    print("\n" + "="*60)
    print("PRITHVI SPATIAL-JEPA — CROSS-GEOGRAPHIC SUMMARY")
    print("="*60)
    print(f"{'Method':<35} {'Acc':>8} {'AUC':>8}")
    print("-"*55)
    for mask_type, res in results_all.items():
        print(f"Baseline ({mask_type}):"
              f"  {res['baseline']['accuracy']:.4f}  "
              f"{res['baseline']['auc']:.4f}")
        print(f"Post-JEPA ({mask_type}):"
              f"  {res['post_jepa']['accuracy']:.4f}  "
              f"{res['post_jepa']['auc']:.4f}")

    print(f"\nPrithvi linear probe reference (no JEPA):")
    print(f"  Maize→Paddy: 97.90% | Paddy→Maize: 99.92%")

    # Save
    with open(LOG_DIR / "prithvi_jepa_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {LOG_DIR / 'prithvi_jepa_results.json'}")


if __name__ == "__main__":
    main()