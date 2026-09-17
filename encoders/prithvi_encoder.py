"""
=============================================================================
AGRI FOUNDATION — Prithvi-EO-2.0 Encoder + MicaSense Band Adapter
=============================================================================
Wraps Prithvi-EO-2.0 for 5-band MicaSense RedEdge-MX multispectral input.

Prithvi-EO-2.0 expects 6 HLS bands:
  [Blue(450nm), Green(560nm), Red(665nm), NIR(865nm), SWIR1(1610nm), SWIR2(2190nm)]

MicaSense RedEdge-MX provides 5 bands:
  [Blue(475nm), Green(560nm), Red(668nm), RedEdge(717nm), NIR(840nm)]

Band adapter strategy:
  - Blue, Green, Red, NIR: direct mapping (wavelength-close)
  - RedEdge(717nm) → substitutes SWIR1 (carries vegetation stress info)
  - SWIR2: zero-padded or learned substitution (not available)

Reference: Jakubik et al. (2024) — Prithvi-EO-2.0
Model: ibm-nasa-geospatial/Prithvi-EO-2.0-300M
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

BASE        = Path.home() / 'agri_foundation'
WEIGHTS_DIR = BASE / 'weights' / 'prithvi'
EMBED_DIM   = 512

# MicaSense RedEdge-MX band wavelengths (nm)
MICASENSE_WAVELENGTHS = [475, 560, 668, 717, 840]
# Prithvi-EO-2.0 HLS band wavelengths (nm)
PRITHVI_WAVELENGTHS   = [450, 560, 665, 865, 1610, 2190]

# Band mapping: MicaSense index → Prithvi index
# Blue(0)→Blue(0), Green(1)→Green(1), Red(2)→Red(2),
# NIR(4)→NIR(3), RedEdge(3)→SWIR1(4), [SWIR2(5) learned]
BAND_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3, 3: 4}


# ─────────────────────────────────────────────
# BAND ADAPTER
# ─────────────────────────────────────────────

class MicaSenseToPrithviAdapter(nn.Module):
    """
    Learnable adapter mapping 5 MicaSense RedEdge-MX bands
    to 6 Prithvi-EO-2.0 HLS bands.

    Architecture:
    - Fixed channel permutation for wavelength-close bands
    - Learnable 1×1 conv for SWIR2 synthesis from available bands
    - Optional attention-weighted band mixing

    Input:  (B, 5, H, W)  — 5-band MicaSense
    Output: (B, 6, H, W)  — 6-band Prithvi-compatible
    """
    def __init__(self, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        # Learnable SWIR2 synthesis from 5 input bands
        # SWIR2 (2190nm) is typically correlated with moisture/dry matter
        # Best approximated from NIR + RedEdge combination
        self.swir2_synthesis = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Reflectance is bounded [0,1]
        )

        # Optional band mixing attention
        if use_attention:
            self.band_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),       # (B, 5, 1, 1)
                nn.Flatten(1),                  # (B, 5)
                nn.Linear(5, 5),
                nn.Softmax(dim=1)
            )

        # Learnable scale/bias per output band
        self.band_norm = nn.GroupNorm(1, 6)

        self._init_weights()

    def _init_weights(self):
        # Initialise SWIR2 synthesis to use NIR (band 4) primarily
        # NIR is the closest available to SWIR spectral region
        with torch.no_grad():
            self.swir2_synthesis[0].weight.zero_()
            self.swir2_synthesis[0].bias.zero_()
            # Primarily use NIR (input channel 4) for SWIR2
            self.swir2_synthesis[0].weight[0, 4, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 5, H, W) — [Blue, Green, Red, RedEdge, NIR]
        Returns: (B, 6, H, W) — [Blue, Green, Red, NIR, RedEdge, SWIR2_synth]
        """
        B, C, H, W = x.shape
        assert C == 5, f"Expected 5 bands, got {C}"

        # Optional attention weighting
        if self.use_attention:
            attn = self.band_attention(x).view(B, 5, 1, 1)
            x_weighted = x * attn
        else:
            x_weighted = x

        # Channel permutation to Prithvi order:
        # [Blue(0), Green(1), Red(2), NIR(4), RedEdge(3)]
        x_permuted = torch.stack([
            x_weighted[:, 0, :, :],   # Blue   → Prithvi Blue
            x_weighted[:, 1, :, :],   # Green  → Prithvi Green
            x_weighted[:, 2, :, :],   # Red    → Prithvi Red
            x_weighted[:, 4, :, :],   # NIR    → Prithvi NIR
            x_weighted[:, 3, :, :],   # RedEdge→ Prithvi SWIR1 (substitute)
        ], dim=1)                      # (B, 5, H, W)

        # Synthesise SWIR2
        swir2 = self.swir2_synthesis(x)  # (B, 1, H, W)

        # Concatenate: (B, 6, H, W)
        out = torch.cat([x_permuted, swir2], dim=1)

        # Normalise across bands
        out = self.band_norm(out)

        return out


# ─────────────────────────────────────────────
# LIGHTWEIGHT PRITHVI-STYLE ViT BACKBONE
# ─────────────────────────────────────────────

class PrithviStyleViT(nn.Module):
    """
    Lightweight ViT backbone inspired by Prithvi-EO-2.0 architecture.
    Used when pretrained weights are not available.

    Prithvi-EO-2.0-300M architecture:
    - Patch size: 16×16
    - Hidden dim: 1024
    - Depth: 24 layers
    - Heads: 16
    - Input: (B, T, C, H, W) — temporal stacks

    This implementation:
    - Single-temporal (T=1)
    - Reduced depth (configurable)
    - 6-band input after adapter
    """
    def __init__(
        self,
        img_size:    int = 64,
        patch_size:  int = 8,
        in_channels: int = 6,
        d_model:     int = 512,
        n_heads:     int = 8,
        n_layers:    int = 6,
        embed_dim:   int = 512,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.n_patches   = (img_size // patch_size) ** 2
        self.d_model     = d_model

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, d_model)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6, H, W)
        Returns: (B, embed_dim)
        """
        B = x.shape[0]

        # Patch embedding: (B, d_model, n_h, n_w) → (B, n_patches, d_model)
        patches = self.patch_embed(x)
        patches = patches.flatten(2).transpose(1, 2)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.dropout(tokens)

        # Transformer
        out = self.transformer(tokens)
        cls_out = out[:, 0, :]

        return self.head(cls_out)


# ─────────────────────────────────────────────
# PRITHVI ENCODER (MAIN CLASS)
# ─────────────────────────────────────────────

class PrithviEncoder(nn.Module):
    """
    Prithvi-EO-2.0 encoder for 5-band MicaSense RedEdge-MX data.

    Pipeline:
      (B, 5, H, W) → BandAdapter → (B, 6, H, W) → Prithvi ViT → (B, 512)

    Loads HuggingFace Prithvi-EO-2.0-300M weights if available.
    Falls back to PrithviStyleViT if not.

    Usage:
        encoder = PrithviEncoder(device='cuda:1')
        # MS image: (B, H, W, 5) — numpy style
        x = torch.randn(4, 5, 64, 64)  # (B, C, H, W)
        emb = encoder(x)               # (B, 512)
    """
    def __init__(
        self,
        embed_dim:       int   = EMBED_DIM,
        img_size:        int   = 64,
        weights_path:    Optional[str] = None,
        freeze_backbone: bool  = True,
        device:          str   = 'cuda:1',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size  = img_size
        self.device    = device

        # Band adapter (always trainable)
        self.band_adapter = MicaSenseToPrithviAdapter(use_attention=True)

        # Try loading HuggingFace Prithvi
        self.backbone = None
        loaded = False

        # Strategy 1: HuggingFace transformers
        try:
            from transformers import AutoModel, AutoConfig
            hf_model_id = 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M'

            local_path = str(WEIGHTS_DIR)
            if Path(local_path).exists() and any(Path(local_path).iterdir()):
                print(f"[Prithvi] Loading from local: {local_path}")
                config   = AutoConfig.from_pretrained(local_path,
                               trust_remote_code=True)
                backbone = AutoModel.from_pretrained(local_path,
                               trust_remote_code=True)
            else:
                print(f"[Prithvi] Attempting HuggingFace download: {hf_model_id}")
                backbone = AutoModel.from_pretrained(hf_model_id,
                               trust_remote_code=True)

            # Extract the encoder part and wrap
            self.backbone      = backbone
            self.hf_projection = nn.Linear(1024, embed_dim)  # Prithvi hidden=1024
            self.using_hf      = True
            loaded             = True
            print(f"[Prithvi] HuggingFace model loaded")

        except Exception as e:
            print(f"[Prithvi] HuggingFace load failed: {e}")

        # Strategy 2: local .pth weights
        if not loaded:
            search_paths = [
                weights_path,
                str(WEIGHTS_DIR / 'Prithvi_EO_V2_300M.pt'),
                str(WEIGHTS_DIR / 'prithvi_eo_v2_300m.pt'),
                str(WEIGHTS_DIR / 'pytorch_model.bin'),
            ]
            for path in search_paths:
                if path and Path(path).exists():
                    try:
                        self.backbone      = PrithviStyleViT(
                            img_size=img_size, embed_dim=embed_dim
                        )
                        self.using_hf = False
                        ckpt  = torch.load(path, map_location='cpu')
                        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
                        missing, _ = self.backbone.load_state_dict(
                            state, strict=False
                        )
                        print(f"[Prithvi] Loaded weights from {path} "
                              f"(missing: {len(missing)})")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"[Prithvi] Load failed {path}: {e}")

        # Strategy 3: random init fallback
        if not loaded:
            print("[Prithvi] No pretrained weights — using PrithviStyleViT (random init)")
            print(f"  Download from: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M")
            self.backbone = PrithviStyleViT(
                img_size=img_size, in_channels=6,
                d_model=512, n_heads=8, n_layers=6,
                embed_dim=embed_dim
            )
            self.using_hf = False

        # Freeze backbone if requested
        if freeze_backbone and self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[Prithvi] Backbone frozen")

        # Band adapter always trainable
        for param in self.band_adapter.parameters():
            param.requires_grad = True

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 5, H, W)  — [Blue, Green, Red, RedEdge, NIR]
        Returns: (B, embed_dim)
        """
        if not x.is_cuda:
            x = x.to(self.device)

        # Resize to expected input size
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        # Band adaptation: (B, 5, H, W) → (B, 6, H, W)
        x_6band = self.band_adapter(x)

        # Backbone forward
        if self.using_hf:
            # Prithvi-EO-2.0 HF expects (B, T, C, H, W)
            x_hf = x_6band.unsqueeze(1)  # add temporal dim
            out  = self.backbone(x_hf)
            # Extract CLS token or pooled output
            if hasattr(out, 'last_hidden_state'):
                feat = out.last_hidden_state[:, 0, :]  # CLS
            else:
                feat = out[0][:, 0, :]
            return self.hf_projection(feat)
        else:
            return self.backbone(x_6band)

    def encode_patches(self, x: np.ndarray,
                       patch_size: int = 64,
                       batch_size: int = 32) -> np.ndarray:
        """
        Encode MS image array by extracting patches.
        x: (N, H, W, 5) numpy array
        Returns: (N, embed_dim)
        """
        self.eval()
        embeddings = []
        n = len(x)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = x[i:i+batch_size]
                # (B, H, W, 5) → (B, 5, H, W)
                batch_t = torch.tensor(
                    batch, dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)
                emb = self.forward(batch_t)
                embeddings.append(emb.cpu().numpy())
                print(f"  Encoded {min(i+batch_size, n)}/{n}", end='\r')
        print()
        return np.concatenate(embeddings, axis=0)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

def test_encoder():
    print("=" * 50)
    print("Prithvi Encoder — Quick Test")
    print("=" * 50)

    device = 'cuda:1' if torch.cuda.device_count() > 1 else \
             'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    encoder = PrithviEncoder(
        embed_dim       = 512,
        img_size        = 64,
        freeze_backbone = True,
        device          = device
    )

    total_p     = sum(p.numel() for p in encoder.parameters())
    trainable_p = sum(p.numel() for p in encoder.parameters()
                      if p.requires_grad)
    print(f"\nTotal params    : {total_p:,}")
    print(f"Trainable params: {trainable_p:,}")

    # Test 1: single MS image patch
    x = torch.randn(4, 5, 64, 64).to(device)
    with torch.no_grad():
        emb = encoder(x)
    print(f"\nInput  {tuple(x.shape)} → embedding {tuple(emb.shape)}")
    assert emb.shape == (4, 512)

    # Test 2: different spatial size (auto-resized)
    x2 = torch.randn(2, 5, 128, 128).to(device)
    with torch.no_grad():
        emb2 = encoder(x2)
    print(f"Input  {tuple(x2.shape)} → embedding {tuple(emb2.shape)}")
    assert emb2.shape == (2, 512)

    # Test 3: band adapter output check
    x3   = torch.randn(1, 5, 64, 64).to(device)
    adapted = encoder.band_adapter(x3)
    print(f"\nBand adapter: {tuple(x3.shape)} → {tuple(adapted.shape)}")
    assert adapted.shape == (1, 6, 64, 64)

    # Test 4: numpy batch
    print("\nTesting numpy batch encoding...")
    x_np = np.random.randn(20, 64, 64, 5).astype(np.float32)
    # Clip to [0,1] reflectance range
    x_np = np.clip(x_np, 0, 1)
    embs = encoder.encode_patches(x_np, batch_size=8)
    print(f"Numpy input (20, 64, 64, 5) → {embs.shape}")
    assert embs.shape == (20, 512)

    print("\n✓ All tests passed")
    return encoder


if __name__ == '__main__':
    test_encoder()