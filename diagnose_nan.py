"""
NaN diagnostic script for HSI training.
Runs a single forward+backward pass with detailed checks at each stage.
Run: python diagnose_nan.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from hsi_dataset import build_hsi_loaders
from hsi_encoder import SpectralTransformerEncoder
from pathlib import Path

DATA_ROOT = Path("~/agri_foundation/data").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_tensor(t: torch.Tensor, name: str) -> bool:
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    print(f"  {name:40s} shape={tuple(t.shape)} "
          f"min={t.float().min():.4f} max={t.float().max():.4f} "
          f"nan={has_nan} inf={has_inf}")
    return has_nan or has_inf


def diagnose() -> None:
    print("=" * 60)
    print("Step 1: Check raw data")
    print("=" * 60)

    loaders = build_hsi_loaders(
        data_root=DATA_ROOT,
        batch_size=32,
        num_workers=0,
        augment_train=False,       # no augmentation during diagnosis
        use_weighted_sampler=False,
    )
    loader = loaders["crop_variety"]["train"]
    patches, labels = next(iter(loader))

    bad = check_tensor(patches, "patches (raw from loader)")
    check_tensor(labels.float(), "labels")
    print(f"  Label unique values: {labels.unique().tolist()}")

    if bad:
        print("  DATA IS CORRUPT — NaN in input patches. Check preprocessing.")
        return

    print("\n" + "=" * 60)
    print("Step 2: Check encoder forward pass (float32, no AMP)")
    print("=" * 60)

    encoder = SpectralTransformerEncoder(
        num_bands=282,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        ffn_dim=512,
        dropout=0.0,    # disable dropout for diagnosis
    ).to(DEVICE)

    patches = patches.to(DEVICE)

    # Spatial stem
    stem_out = encoder.spatial_stem(patches)
    bad = check_tensor(stem_out, "spatial_stem output")
    if bad:
        print("  NaN in spatial stem. Check depthwise conv init.")
        return

    # Band projection
    tokens = encoder.band_proj(stem_out.unsqueeze(-1))
    bad = check_tensor(tokens, "band_proj output (tokens)")
    if bad:
        print("  NaN in band projection. Likely init issue.")
        return

    # Positional encoding
    tokens = tokens + encoder.pos_embed
    bad = check_tensor(tokens, "after positional encoding")
    if bad:
        print("  NaN from positional encoding.")
        return

    # Each transformer block
    for i, block in enumerate(encoder.blocks):
        tokens = block(tokens)
        bad = check_tensor(tokens, f"transformer block {i} output")
        if bad:
            print(f"  NaN in transformer block {i}.")
            return

    # Full forward
    embeddings = encoder(patches)
    bad = check_tensor(embeddings, "final embeddings (L2 norm)")
    if bad:
        print("  NaN after L2 normalisation — likely zero-norm vector.")
        return

    print("\n" + "=" * 60)
    print("Step 3: Check loss computation (no AMP)")
    print("=" * 60)

    labels = labels.to(DEVICE)
    head = nn.Linear(128, 10).to(DEVICE)
    logits = head(embeddings)
    check_tensor(logits, "logits")

    # Without label smoothing
    loss_no_smooth = nn.CrossEntropyLoss()(logits, labels)
    print(f"  Loss (no smoothing)    : {loss_no_smooth.item()}")

    # With label smoothing
    loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
    print(f"  Loss (smoothing=0.1)   : {loss_smooth.item()}")

    print("\n" + "=" * 60)
    print("Step 4: Check AMP forward pass")
    print("=" * 60)

    with torch.cuda.amp.autocast():
        embeddings_amp = encoder(patches)
        logits_amp = head(embeddings_amp)
        loss_amp = nn.CrossEntropyLoss()(logits_amp, labels)

    check_tensor(embeddings_amp.float(), "embeddings (AMP)")
    check_tensor(logits_amp.float(), "logits (AMP)")
    print(f"  Loss (AMP)             : {loss_amp.item()}")

    print("\n" + "=" * 60)
    print("Step 5: Check gradient flow")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=3e-4
    )
    scaler = torch.cuda.amp.GradScaler()

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = nn.CrossEntropyLoss()(head(encoder(patches)), labels)

    scaler.scale(loss).backward()

    # Check gradients before unscaling
    for name, param in list(encoder.named_parameters())[:5]:
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            print(f"  grad {name:35s} nan={has_nan} inf={has_inf} "
                  f"norm={param.grad.float().norm():.4f}")

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(head.parameters()),
        max_norm=1.0
    )
    print(f"  Gradient norm (before clip): {grad_norm:.4f}")

    print("\n" + "=" * 60)
    print("Diagnosis complete.")
    print("=" * 60)


if __name__ == "__main__":
    diagnose()