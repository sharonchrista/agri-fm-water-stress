"""
Multispectral contrastive pretraining for agri_foundation.

Uses SimCLR-style NT-Xent loss to train a 5-band MS encoder on
unlabelled maize and paddy multispectral tiles.

Architecture:
  Input  : (B, 5, 64, 64) — two augmented views per tile
  Encoder: MSEncoder — lightweight CNN extracting per-tile embeddings
  Head   : 2-layer projection MLP -> 128-dim L2-normalised vector
  Loss   : NT-Xent (normalised temperature-scaled cross entropy)

After pretraining, the encoder backbone is frozen and used for:
  - Linear probing on downstream labelled tasks
  - Few-shot prototypical evaluation
  - Cross-modal transfer (HSI -> MS domain shift experiment)

Run: python train_ms_pretrain.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ms_dataset import build_ms_loaders


DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# MS Encoder
# ---------------------------------------------------------------------------

class MSEncoder(nn.Module):
    """
    Lightweight CNN encoder for 5-band multispectral tiles (64x64).

    Architecture:
      4 conv blocks with stride-2 downsampling -> (B, 256, 4, 4)
      Global average pool -> (B, 256)
      L2 normalisation -> unit-norm embedding

    Designed to be small enough to coexist with other GPU jobs:
    ~1.2M parameters, ~400 MB VRAM at batch size 64.
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 32,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Progressive downsampling: 64 -> 32 -> 16 -> 8 -> 4
        self.backbone = nn.Sequential(
            self._conv_block(in_channels, base_channels, stride=2),      # 32x32
            self._conv_block(base_channels, base_channels * 2, stride=2), # 16x16
            self._conv_block(base_channels * 2, base_channels * 4, stride=2), # 8x8
            self._conv_block(base_channels * 4, embed_dim, stride=2),    # 4x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B, embed_dim, 1, 1)

        self._init_weights()

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 5, 64, 64)
        x = self.backbone(x)          # (B, embed_dim, 4, 4)
        x = self.pool(x)              # (B, embed_dim, 1, 1)
        x = x.flatten(1)             # (B, embed_dim)
        return F.normalize(x, dim=-1) # unit-norm


# ---------------------------------------------------------------------------
# Projection head (SimCLR-style)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head for contrastive loss.
    Applied during pretraining only — discarded for downstream tasks.
    Output is L2-normalised for NT-Xent loss computation.
    """

    def __init__(self, embed_dim: int = 256, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# NT-Xent loss (SimCLR)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy loss for contrastive learning.

    For a batch of N tile pairs (view_a, view_b):
      - 2N embeddings total
      - Each view_a[i] should be closest to view_b[i] (its augmented pair)
      - All other 2N-2 samples are negatives

    Temperature controls sharpness — lower = harder negatives emphasised.
    """

    def __init__(self, temperature: float = 0.07, batch_size: int = 64) -> None:
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z_a, z_b : Tensor (B, proj_dim) — L2-normalised projections of view pairs

        Returns
        -------
        Scalar loss
        """
        batch_size = z_a.shape[0]
        device = z_a.device

        # Concatenate both views: (2B, proj_dim)
        z = torch.cat([z_a, z_b], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair indices:
        # For view_a[i] (index i), its positive is view_b[i] (index i + B)
        # For view_b[i] (index i + B), its positive is view_a[i] (index i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    encoder: MSEncoder,
    proj_head: ProjectionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NTXentLoss,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    encoder.train()
    proj_head.train()
    total_loss = 0.0
    n_batches = 0

    for view_a, view_b in loader:
        view_a = view_a.to(DEVICE, non_blocking=True)
        view_b = view_b.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            z_a = proj_head(encoder(view_a))
            z_b = proj_head(encoder(view_b))
            loss = criterion(z_a, z_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(proj_head.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


# ---------------------------------------------------------------------------
# Collapse diagnostic — checks if encoder has collapsed to constant output
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_collapse(
    encoder: MSEncoder,
    loader: DataLoader,
    n_batches: int = 5,
) -> dict[str, float]:
    """
    Representation collapse check.
    A collapsed encoder maps all inputs to the same point.
    Healthy encoder: std across batch >> 0, mean cosine sim << 1.
    Returns empty dict if insufficient data to compute metrics reliably.
    """
    encoder.eval()
    embeddings = []
    for i, (view_a, _) in enumerate(loader):
        if i >= n_batches:
            break
        emb = encoder(view_a.to(DEVICE))
        embeddings.append(emb.cpu())

    emb = torch.cat(embeddings, dim=0)   # (N, D)

    # Guard against degenerate batches
    per_dim_std = emb.std(dim=0)
    if per_dim_std.mean().item() < 1e-8:
        return {"embedding_std": 0.0, "mean_pairwise_cosine_sim": 1.0}

    std = per_dim_std.mean().item()

    # Average cosine similarity between random pairs
    n_sample = min(256, len(emb))
    idx = torch.randperm(len(emb))[:n_sample]
    sample = F.normalize(emb[idx], dim=-1)
    sim_matrix = sample @ sample.T
    mask = ~torch.eye(n_sample, dtype=torch.bool)
    off_diag_vals = sim_matrix[mask]

    if len(off_diag_vals) == 0:
        return {"embedding_std": std, "mean_pairwise_cosine_sim": float("nan")}

    off_diag = off_diag_vals.mean().item()
    return {"embedding_std": std, "mean_pairwise_cosine_sim": off_diag}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")

    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 100
    TILES_PER_EPOCH = 8000
    LR = 3e-4
    TEMPERATURE = 0.07
    EMBED_DIM = 256
    PROJ_DIM = 128

    # Data
    print("\nLoading MS data...")
    loaders = build_ms_loaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=4,
        tiles_per_epoch=TILES_PER_EPOCH,
    )
    pretrain_loader = loaders["pretrain"]
    print(f"Pretrain loader: {len(pretrain_loader)} batches/epoch "
          f"({TILES_PER_EPOCH} tiles, batch {BATCH_SIZE})")

    # Model
    encoder = MSEncoder(in_channels=5, base_channels=32, embed_dim=EMBED_DIM).to(DEVICE)
    proj_head = ProjectionHead(embed_dim=EMBED_DIM, proj_dim=PROJ_DIM).to(DEVICE)

    total_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in proj_head.parameters())
    )
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Proj head params: {sum(p.numel() for p in proj_head.parameters()):,}")
    print(f"Total params: {total_params:,}")

    # Check initial VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(DEVICE) / 1024**2
        print(f"VRAM allocated after model init: {allocated:.1f} MB")

    # Optimiser and scheduler
    criterion = NTXentLoss(temperature=TEMPERATURE, batch_size=BATCH_SIZE)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )
    # AMP enabled — MS encoder uses Conv2d which benefits from float16
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # Training
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")

    print(f"\n{'Epoch':>6} {'Loss':>10} {'EmbStd':>8} {'CosSim':>8} "
          f"{'LR':>10} {'Time':>7}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            encoder, proj_head, pretrain_loader,
            optimizer, criterion, scaler,
        )
        scheduler.step()

        # Collapse check every 5 epochs
        collapse_metrics = {}
        if epoch % 5 == 0 or epoch == 1:
            collapse_metrics = check_collapse(encoder, pretrain_loader)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        log_entry = {
            "epoch": epoch,
            "train": train_metrics,
            "collapse": collapse_metrics,
            "lr": lr,
        }
        history.append(log_entry)

        emb_std = collapse_metrics.get("embedding_std", float("nan"))
        cos_sim = collapse_metrics.get("mean_pairwise_cosine_sim", float("nan"))

        print(f"{epoch:>6} {train_metrics['loss']:>10.4f} "
              f"{emb_std:>8.4f} {cos_sim:>8.4f} "
              f"{lr:>10.6f} {elapsed:>6.1f}s")

        # Save best checkpoint
        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "proj_head_state": proj_head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": best_loss,
                "config": {
                    "embed_dim": EMBED_DIM,
                    "proj_dim": PROJ_DIM,
                    "batch_size": BATCH_SIZE,
                    "temperature": TEMPERATURE,
                },
            }, CHECKPOINT_DIR / "ms_encoder_best.pt")

        # Save periodic checkpoint every 25 epochs
        if epoch % 25 == 0:
            torch.save({
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "loss": train_metrics["loss"],
            }, CHECKPOINT_DIR / f"ms_encoder_epoch{epoch:03d}.pt")
            print(f"  --> checkpoint saved (epoch {epoch})")

    # Save training log
    with open(LOG_DIR / "ms_pretrain.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nPretraining complete. Best loss: {best_loss:.4f}")
    print(f"Encoder checkpoint: {CHECKPOINT_DIR / 'ms_encoder_best.pt'}")

    # Final collapse check
    print("\nFinal collapse check:")
    final_metrics = check_collapse(encoder, pretrain_loader, n_batches=20)
    print(f"  Embedding std:          {final_metrics['embedding_std']:.4f} "
          f"(healthy > 0.1)")
    print(f"  Mean pairwise cos sim:  {final_metrics['mean_pairwise_cosine_sim']:.4f} "
          f"(healthy < 0.3)")


if __name__ == "__main__":
    main()