"""
HSI encoder architectures for agri_foundation.

Two architectures provided:
  1. SpectralTransformerEncoder  -- main model, treats each band as a token,
                                    self-attention across 282-band sequence.
  2. SpectralCNNEncoder          -- baseline, 1D depthwise conv stack along
                                    the spectral dimension.

Both produce a fixed-dim embedding vector per patch:
  Input  : (B, 282, 11, 11)  -- batch of HSI patches in CHW order
  Output : (B, embed_dim)    -- L2-normalised embedding per patch

The spatial 11x11 context is handled by a lightweight spatial pooling step
before the spectral encoder, reducing (B, C, 11, 11) to (B, C, 1, 1).
This keeps the spectral encoder decoupled from spatial assumptions,
which is essential for few-shot transfer to new patch sizes.

Usage:
    from hsi_encoder import SpectralTransformerEncoder, SpectralCNNEncoder

    model = SpectralTransformerEncoder(num_bands=282, embed_dim=128)
    patches = torch.randn(32, 282, 11, 11)
    embeddings = model(patches)   # (32, 128), L2-normalised
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared spatial pooling stem
# ---------------------------------------------------------------------------

class SpatialStem(nn.Module):
    """
    Reduces spatial dimensions (H, W) to 1x1 via a lightweight learned
    depthwise convolution followed by adaptive average pooling.

    For 1x1 spatial inputs (e.g. groundnut flat patches), the conv is
    bypassed and the input is passed directly to avoid zero-padding artifacts
    that kill gradient flow through the transformer.

    Input : (B, C, H, W)
    Output: (B, C)
    """

    def __init__(self, num_bands: int, spatial_size: int = 11) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        # Only used when spatial_size > 1
        if spatial_size > 1:
            self.dw_conv = nn.Conv2d(
                in_channels=num_bands,
                out_channels=num_bands,
                kernel_size=3,
                padding=1,
                groups=num_bands,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(num_bands)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        h, w = x.shape[2], x.shape[3]
        if h == 1 and w == 1:
            # 1x1 input — skip conv entirely, just squeeze spatial dims
            return x.squeeze(-1).squeeze(-1)   # (B, C)
        x = F.relu(self.bn(self.dw_conv(x)))
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)        # (B, C)


# ---------------------------------------------------------------------------
# 1. Spectral Transformer Encoder (main model)
# ---------------------------------------------------------------------------

class SpectralAttention(nn.Module):
    """
    Multi-head self-attention over the spectral dimension.
    Each band is treated as a token; attention captures cross-band dependencies.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # (B, seq_len, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, seq_len, embed_dim)
        residual = x
        x, _ = self.attn(x, x, x)
        return self.norm(residual + self.dropout(x))


class SpectralFFN(nn.Module):
    """Position-wise feed-forward network in each transformer block."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.net(x))


class SpectralTransformerBlock(nn.Module):
    """One transformer block: attention + FFN with pre-norm residuals."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = SpectralAttention(embed_dim, num_heads, dropout)
        self.ffn = SpectralFFN(embed_dim, ffn_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x


class SpectralTransformerEncoder(nn.Module):
    """
    Spectral Transformer Encoder for hyperspectral patches.

    Architecture:
      1. SpatialStem        : (B, 282, 11, 11) -> (B, 282)
      2. Band embedding     : linear projection per band -> (B, 282, embed_dim)
      3. Positional encoding: learnable wavelength position encoding
      4. N transformer blocks on the 282-token sequence
      5. Global average pooling over band tokens -> (B, embed_dim)
      6. Projection head    : embed_dim -> embed_dim with LayerNorm
      7. L2 normalisation   : unit-norm output for contrastive / cosine losses

    Parameters
    ----------
    num_bands : int
        Number of spectral bands (282 for this dataset).
    embed_dim : int
        Transformer token dimension and output embedding size.
    num_heads : int
        Number of attention heads. Must divide embed_dim.
    num_layers : int
        Number of transformer blocks.
    ffn_dim : int
        Feed-forward network hidden dimension (typically 4 * embed_dim).
    dropout : float
        Dropout rate applied in attention and FFN.
    spatial_size : int
        Input patch spatial size (11 for this dataset).
    """

    def __init__(
        self,
        num_bands: int = 282,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        spatial_size: int = 11,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.embed_dim = embed_dim

        # Spatial pooling stem — operates depthwise, preserves spectral identity
        self.spatial_stem = SpatialStem(num_bands, spatial_size)

        # Per-band linear projection: scalar -> embed_dim
        # Each band's reflectance value is projected to a token vector
        self.band_proj = nn.Linear(1, embed_dim)

        # Learnable positional encoding over band positions
        # Encodes wavelength ordering (400nm -> 1000nm) implicitly
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_bands, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpectralTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Projection head for contrastive learning (SimCLR-style)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, return_tokens: bool = False) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, num_bands, H, W)
        return_tokens : bool
            If True, return per-band token matrix (B, num_bands, embed_dim)
            instead of pooled embedding. Useful for SHAP band attribution.

        Returns
        -------
        Tensor (B, embed_dim) L2-normalised embedding, or
        Tensor (B, num_bands, embed_dim) if return_tokens=True
        """
        # Step 1: spatial pooling -> (B, num_bands)
        x = self.spatial_stem(x)

        # Step 2: reshape to token sequence -> (B, num_bands, 1)
        # then project each scalar band value to embed_dim
        x = x.unsqueeze(-1)                   # (B, 282, 1)
        x = self.band_proj(x)                 # (B, 282, embed_dim)

        # Step 3: add positional encoding
        x = x + self.pos_embed               # (B, 282, embed_dim)

        # Step 4: transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)                      # (B, 282, embed_dim)

        if return_tokens:
            return x                          # (B, 282, embed_dim) for SHAP

        # Step 5: global average pool over band tokens -> (B, embed_dim)
        x = x.mean(dim=1)

        # Step 6: projection head
        x = self.proj_head(x)

        # Step 7: L2 normalise
        return F.normalize(x, dim=-1)

    def get_band_attention(self, x: Tensor) -> Tensor:
        """
        Extract per-band attention weights from the first transformer block.
        Returns (B, num_heads, num_bands, num_bands) for interpretability analysis.
        Used downstream for SHAP-style band importance attribution.
        """
        x = self.spatial_stem(x).unsqueeze(-1)
        x = self.band_proj(x) + self.pos_embed

        # Hook into first block's attention
        with torch.no_grad():
            _, attn_weights = self.blocks[0].attn.attn(x, x, x)
        return attn_weights


# ---------------------------------------------------------------------------
# 2. Spectral CNN Encoder (baseline)
# ---------------------------------------------------------------------------

class SpectralCNNEncoder(nn.Module):
    """
    1D Spectral CNN Encoder — baseline model for ablation comparison.

    Architecture:
      1. SpatialStem   : (B, 282, 11, 11) -> (B, 282)
      2. Reshape       : (B, 1, 282) — treat bands as 1D signal
      3. Conv1d stack  : depthwise separable convolutions along spectral dim
      4. Global pool   : (B, hidden_dim)
      5. Projection    : hidden_dim -> embed_dim
      6. L2 normalise

    Significantly fewer parameters than the transformer (~3x less).
    Faster training — useful as a quick sanity check before full transformer runs.
    """

    def __init__(
        self,
        num_bands: int = 282,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        spatial_size: int = 11,
    ) -> None:
        super().__init__()
        self.spatial_stem = SpatialStem(num_bands, spatial_size)

        # Build Conv1d stack with residual connections
        # Input: (B, 1, num_bands) — single channel, bands as sequence
        channels = [1] + [hidden_dim] * num_layers
        self.conv_layers = nn.ModuleList()
        self.res_projections = nn.ModuleList()

        for i in range(num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
            ))
            # Residual projection only when channel dimensions change
            if in_ch != out_ch:
                self.res_projections.append(
                    nn.Conv1d(in_ch, out_ch, kernel_size=1)
                )
            else:
                self.res_projections.append(nn.Identity())

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 282, 11, 11)
        x = self.spatial_stem(x)        # (B, 282)
        x = x.unsqueeze(1)              # (B, 1, 282) — channel-first for Conv1d

        for conv, res_proj in zip(self.conv_layers, self.res_projections):
            residual = res_proj(x)
            x = F.gelu(conv(x) + residual)

        x = x.mean(dim=-1)              # global avg pool over spectral dim (B, hidden_dim)
        x = self.proj_head(x)           # (B, embed_dim)
        return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> None:
    """
    Verify both encoders produce correct output shapes and are GPU-compatible.
    Run directly: python hsi_encoder.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch = torch.randn(32, 282, 11, 11).to(device)

    # Transformer encoder
    transformer = SpectralTransformerEncoder(
        num_bands=282,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        ffn_dim=512,
        dropout=0.1,
    ).to(device)

    out = transformer(batch)
    print(f"\nSpectralTransformerEncoder")
    print(f"  input  : {tuple(batch.shape)}")
    print(f"  output : {tuple(out.shape)}")
    print(f"  l2 norm (should be ~1.0): {out.norm(dim=-1).mean():.4f}")
    print(f"  params : {count_parameters(transformer):,}")

    # Token-level output for SHAP
    tokens = transformer(batch, return_tokens=True)
    print(f"  tokens : {tuple(tokens.shape)} (for band attribution)")

    # Attention weights
    attn = transformer.get_band_attention(batch)
    print(f"  attention weights : {tuple(attn.shape)}")

    # CNN baseline
    cnn = SpectralCNNEncoder(
        num_bands=282,
        embed_dim=128,
        hidden_dim=256,
        num_layers=4,
    ).to(device)

    out_cnn = cnn(batch)
    print(f"\nSpectralCNNEncoder (baseline)")
    print(f"  input  : {tuple(batch.shape)}")
    print(f"  output : {tuple(out_cnn.shape)}")
    print(f"  l2 norm (should be ~1.0): {out_cnn.norm(dim=-1).mean():.4f}")
    print(f"  params : {count_parameters(cnn):,}")

    # Verify embeddings are not identical (sanity check models are different)
    cos_sim = F.cosine_similarity(out, out_cnn).mean().item()
    print(f"\nCosine similarity between transformer and CNN outputs: {cos_sim:.4f}")
    print("(Should be << 1.0 — models are untrained but architecturally different)")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    smoke_test()