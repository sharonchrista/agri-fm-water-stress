"""
=============================================================================
AGRI FOUNDATION — SpectralGPT Encoder
=============================================================================
Wraps SpectralGPT for 282-band hyperspectral input.
Produces 512-dim embeddings for the shared embedding space.

SpectralGPT: https://huggingface.co/danfenghong/SpectralGPT
Reference: Hong et al. (2024) — SpectralGPT: Spectral Remote Sensing Foundation Model
=============================================================================
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE         = Path.home() / 'agri_foundation'
WEIGHTS_DIR  = BASE / 'weights' / 'spectralgpt'
EMBED_DIM    = 512   # Shared embedding space dimension
TARGET_BANDS = 282


# ─────────────────────────────────────────────
# SPECTRAL POSITIONAL ENCODING
# ─────────────────────────────────────────────

class SpectralPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding along the spectral (band) dimension.
    Encodes wavelength position so the model knows which part of the
    spectrum each band corresponds to.
    """
    def __init__(self, d_model: int, max_bands: int = 300,
                 wavelength_min: float = 400.0,
                 wavelength_max: float = 1000.0):
        super().__init__()
        self.d_model = d_model

        # Wavelength grid
        wavelengths = torch.linspace(wavelength_min, wavelength_max, max_bands)
        # Normalise to [0, 1]
        wavelengths_norm = (wavelengths - wavelength_min) / (wavelength_max - wavelength_min)

        # Sinusoidal encoding
        pe = torch.zeros(max_bands, d_model)
        position = wavelengths_norm.unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_bands, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_bands, d_model)
        return x + self.pe[:, :x.size(1), :]


# ─────────────────────────────────────────────
# BAND GROUPING MODULE
# ─────────────────────────────────────────────

class BandGroupingEncoder(nn.Module):
    """
    Groups 282 spectral bands into semantically meaningful windows
    and encodes each group via a shared linear projection.

    Band groups (400-1000nm, 282 bands, ~2.1nm resolution):
      - Visible        : bands 0-142   (400-700nm)  -- Blue, Green, Red
      - Red Edge       : bands 143-166 (700-750nm)  -- Vegetation red edge
      - Near Infrared  : bands 167-281 (750-1000nm) -- NIR, water absorption
    """
    def __init__(self, n_bands: int = 282, d_model: int = 256):
        super().__init__()
        self.n_bands = n_bands
        self.d_model = d_model

        # Band group boundaries (approximate for 282 bands, 400-1000nm)
        self.groups = {
            'visible':   (0,   143),   # 400-700nm
            'red_edge':  (143, 167),   # 700-750nm
            'nir':       (167, 282),   # 750-1000nm
        }

        # Per-band linear projection (1 band → d_model token)
        self.band_projection = nn.Linear(1, d_model)

        # Group-level attention to weight bands within each group
        self.group_attention = nn.ModuleDict({
            name: nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            for name in self.groups
        })

        # Group pooling projection
        self.group_pool = nn.ModuleDict({
            name: nn.Linear(d_model, d_model)
            for name in self.groups
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_bands) or (B, H, W, n_bands)
        Returns: (B, 3*d_model) — concatenated group representations
        """
        # Flatten spatial dims if patch input
        orig_shape = x.shape
        if x.ndim == 4:
            B, H, W, C = x.shape
            x = x.reshape(B * H * W, C)
            batch_size = B * H * W
        else:
            batch_size = x.shape[0]

        # Project each band to d_model: (batch, n_bands, d_model)
        x_tokens = self.band_projection(x.unsqueeze(-1))

        group_embeddings = []
        for name, (start, end) in self.groups.items():
            group_tokens = x_tokens[:, start:end, :]  # (B, group_bands, d_model)
            # Self-attention within group
            attended, _ = self.group_attention[name](
                group_tokens, group_tokens, group_tokens
            )
            # Mean pool over bands in group
            pooled = attended.mean(dim=1)              # (B, d_model)
            pooled = self.group_pool[name](pooled)
            group_embeddings.append(pooled)

        # Concatenate group representations: (B, 3*d_model)
        out = torch.cat(group_embeddings, dim=-1)

        if orig_shape.ndim if hasattr(orig_shape, 'ndim') else len(orig_shape) == 4:
            if x.ndim != orig_shape[0]:  # reshape check
                pass

        return out


# ─────────────────────────────────────────────
# SPECTRAL TRANSFORMER ENCODER
# ─────────────────────────────────────────────

class SpectralTransformerEncoder(nn.Module):
    """
    Transformer-based spectral encoder.
    Treats each band as a token with spectral positional encoding.
    Inspired by SpectralGPT architecture.

    Input:  (B, n_bands) or (B, H, W, n_bands)
    Output: (B, embed_dim)
    """
    def __init__(
        self,
        n_bands:    int   = 282,
        d_model:    int   = 256,
        n_heads:    int   = 8,
        n_layers:   int   = 6,
        embed_dim:  int   = 512,
        dropout:    float = 0.1,
        wavelength_min: float = 400.0,
        wavelength_max: float = 1000.0,
    ):
        super().__init__()
        self.n_bands   = n_bands
        self.d_model   = d_model
        self.embed_dim = embed_dim

        # Band → token projection
        self.band_embed = nn.Linear(1, d_model)

        # Spectral positional encoding
        self.pos_enc = SpectralPositionalEncoding(
            d_model, max_bands=n_bands,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max
        )

        # CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True   # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Project CLS token → embed_dim
        self.head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_bands) or (B, H, W, n_bands) or (B, 1, 1, n_bands)
        Returns: (B, embed_dim)
        """
        # Flatten spatial dims
        if x.ndim == 4:
            B = x.shape[0]
            x = x.reshape(B, -1, self.n_bands).mean(dim=1)  # spatial mean
        B = x.shape[0]

        # Band → tokens: (B, n_bands, d_model)
        tokens = self.band_embed(x.unsqueeze(-1))
        tokens = self.pos_enc(tokens)
        tokens = self.dropout(tokens)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_bands+1, d_model)

        # Transformer
        out = self.transformer(tokens)  # (B, n_bands+1, d_model)

        # CLS token output → embedding
        cls_out = out[:, 0, :]           # (B, d_model)
        return self.head(cls_out)        # (B, embed_dim)


# ─────────────────────────────────────────────
# SPECTRALGPT WRAPPER
# ─────────────────────────────────────────────

class SpectralGPTEncoder(nn.Module):
    """
    SpectralGPT encoder for 282-band hyperspectral UAV data.

    Tries to load pretrained SpectralGPT weights if available.
    Falls back to initialised SpectralTransformerEncoder if weights
    are not found (still uses the same architecture for consistency).

    Usage:
        encoder = SpectralGPTEncoder(device='cuda:0')
        x = torch.randn(8, 11, 11, 282)   # batch of HSI patches
        emb = encoder(x)                   # (8, 512)
    """
    def __init__(
        self,
        n_bands:       int   = TARGET_BANDS,
        embed_dim:     int   = EMBED_DIM,
        weights_path:  Optional[str] = None,
        freeze_backbone: bool = True,
        device:        str   = 'cuda:0',
    ):
        super().__init__()
        self.n_bands   = n_bands
        self.embed_dim = embed_dim
        self.device    = device

        # Build backbone
        self.backbone = SpectralTransformerEncoder(
            n_bands    = n_bands,
            d_model    = 256,
            n_heads    = 8,
            n_layers   = 6,
            embed_dim  = embed_dim,
            dropout    = 0.1,
        )

        # Try loading pretrained weights
        loaded = False
        search_paths = [
            weights_path,
            str(WEIGHTS_DIR / 'SpectralGPT.pth'),
            str(WEIGHTS_DIR / 'SpectralGPT+.pth'),
            str(WEIGHTS_DIR / 'spectralgpt.pth'),
        ]

        for path in search_paths:
            if path and Path(path).exists():
                try:
                    ckpt = torch.load(path, map_location='cpu')
                    # Handle various checkpoint formats
                    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
                    missing, unexpected = self.backbone.load_state_dict(
                        state, strict=False
                    )
                    print(f"[SpectralGPT] Loaded weights from {path}")
                    print(f"  Missing keys   : {len(missing)}")
                    print(f"  Unexpected keys: {len(unexpected)}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[SpectralGPT] Could not load {path}: {e}")

        if not loaded:
            print("[SpectralGPT] No pretrained weights found — using random init")
            print("  Download from: https://huggingface.co/danfenghong/SpectralGPT")
            print(f"  Save to: {WEIGHTS_DIR}/SpectralGPT.pth")

        # Freeze backbone if requested (for linear probing / few-shot)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Keep head trainable
            for param in self.backbone.head.parameters():
                param.requires_grad = True
            print("[SpectralGPT] Backbone frozen (head trainable)")

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_bands) | (B, H, W, n_bands) | (B, 1, 1, n_bands)
        Returns: (B, embed_dim=512)
        """
        if not x.is_cuda:
            x = x.to(self.device)
        return self.backbone(x)

    def encode_batch(self, x: np.ndarray,
                     batch_size: int = 256) -> np.ndarray:
        """
        Encode a numpy array in batches.
        x: (N, ..., n_bands)
        Returns: (N, embed_dim) numpy array
        """
        self.eval()
        embeddings = []
        n = len(x)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = torch.tensor(
                    x[i:i+batch_size], dtype=torch.float32
                ).to(self.device)
                emb = self.forward(batch)
                embeddings.append(emb.cpu().numpy())
                if (i // batch_size) % 10 == 0:
                    print(f"  Encoded {min(i+batch_size, n)}/{n}", end='\r')
        print()
        return np.concatenate(embeddings, axis=0)

    def unfreeze(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("[SpectralGPT] All parameters unfrozen")

    def freeze_except_head(self):
        """Freeze backbone, keep head trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.head.parameters():
            param.requires_grad = True


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

def test_encoder():
    print("=" * 50)
    print("SpectralGPT Encoder — Quick Test")
    print("=" * 50)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    encoder = SpectralGPTEncoder(
        n_bands      = 282,
        embed_dim    = 512,
        freeze_backbone = True,
        device       = device
    )

    total_params     = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Test 1: flat spectra (groundnut format)
    x_flat = torch.randn(32, 282).to(device)
    with torch.no_grad():
        emb = encoder(x_flat)
    print(f"\nFlat input  {tuple(x_flat.shape)} → embedding {tuple(emb.shape)}")
    assert emb.shape == (32, 512), f"Expected (32,512), got {emb.shape}"

    # Test 2: spatial patches (crop variety / pearl millet format)
    x_patch = torch.randn(16, 11, 11, 282).to(device)
    with torch.no_grad():
        emb = encoder(x_patch)
    print(f"Patch input {tuple(x_patch.shape)} → embedding {tuple(emb.shape)}")
    assert emb.shape == (16, 512), f"Expected (16,512), got {emb.shape}"

    # Test 3: groundnut patch format (1x1 spatial)
    x_gn = torch.randn(64, 1, 1, 282).to(device)
    with torch.no_grad():
        emb = encoder(x_gn)
    print(f"GN input    {tuple(x_gn.shape)} → embedding {tuple(emb.shape)}")
    assert emb.shape == (64, 512), f"Expected (64,512), got {emb.shape}"

    # Test 4: embedding statistics
    print(f"\nEmbedding stats: mean={emb.mean():.4f}, std={emb.std():.4f}")
    print(f"Embedding range: [{emb.min():.4f}, {emb.max():.4f}]")

    # Test 5: numpy batch encoding
    print("\nTesting numpy batch encoding...")
    x_np = np.random.randn(100, 11, 11, 282).astype(np.float32)
    embs = encoder.encode_batch(x_np, batch_size=32)
    print(f"Numpy input (100, 11, 11, 282) → {embs.shape}")
    assert embs.shape == (100, 512)

    print("\n✓ All tests passed")
    return encoder


if __name__ == '__main__':
    test_encoder()