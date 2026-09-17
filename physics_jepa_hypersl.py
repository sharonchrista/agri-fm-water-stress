"""
Physics-Constrained Spectral JEPA for Cross-Geographic Crop Stress Monitoring.

Architecture:
  HyperSL backbone (frozen) -> JEPA adapter (trainable) -> GP head + few-shot head

Physics-constrained masking strategy:
  Instead of random band masking, preferentially mask physiologically meaningful
  spectral regions confirmed by SHAP analysis:
    Group 1: Red-edge (700-750nm)      -- chlorophyll stress indicator
    Group 2: Water absorption (880-970nm) -- leaf water content (SHAP top bands)
    Group 3: NIR plateau (750-880nm)   -- structural integrity
  
  Masking: 70% physics bands (Groups 1+2), 30% random
  JEPA objective: context encoder predicts target encoder embeddings of masked bands
  This forces the model to learn physiologically meaningful spectral relationships.

Datasets (multi-country):
  India   -- Groundnut HSI 282-band (ICRISAT Hyderabad)
  USA     -- Wheat HSI 190-band (UMN Minnesota, zero-padded to 282)
  Turkey  -- Maize MS 6-band (Zenodo 22062459)
  Sri Lanka -- Paddy MS 4-band (Mendeley)
  India   -- Maize/Paddy MS 5-band (TIAND)

Run: python physics_jepa_hypersl.py
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

sys.path.insert(0, str(Path("~/agri_foundation/models/hypersl_code").expanduser()))

DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_BANDS = 282        # canonical band count
EMBED_DIM = 256        # HyperSL embedding dimension
PROJ_DIM = 128
TEMPERATURE = 0.07
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-4
EMA_DECAY = 0.996      # exponential moving average for target encoder

NUM_EPISODES = 1000
N_SHOT_VALUES = [1, 5, 10, 20]
NUM_QUERY = 50


# ---------------------------------------------------------------------------
# Wavelength definitions (400-1000nm, 282 bands, ~2.1nm spacing)
# ---------------------------------------------------------------------------

WAVELENGTHS = np.linspace(400, 1000, NUM_BANDS)  # nm

# Physics-informed band groups for water stress detection
# Confirmed by SHAP analysis: top bands are 880-965nm
PHYSICS_GROUPS = {
    "red_edge": {
        "range": (700, 750),
        "description": "Chlorophyll red-edge — stress shifts absorption edge",
        "indices": np.where((WAVELENGTHS >= 700) & (WAVELENGTHS <= 750))[0],
    },
    "water_absorption": {
        "range": (880, 970),
        "description": "Water absorption — SHAP top bands, leaf water content",
        "indices": np.where((WAVELENGTHS >= 880) & (WAVELENGTHS <= 970))[0],
    },
    "nir_plateau": {
        "range": (750, 880),
        "description": "NIR structural reflectance — cell turgor stress indicator",
        "indices": np.where((WAVELENGTHS >= 750) & (WAVELENGTHS <= 880))[0],
    },
}

# All physics band indices combined
ALL_PHYSICS_INDICES = np.concatenate([
    g["indices"] for g in PHYSICS_GROUPS.values()
])
ALL_OTHER_INDICES = np.setdiff1d(np.arange(NUM_BANDS), ALL_PHYSICS_INDICES)

print(f"Physics bands: {len(ALL_PHYSICS_INDICES)} "
      f"({len(ALL_PHYSICS_INDICES)/NUM_BANDS*100:.1f}% of spectrum)")
print(f"  Red-edge: {len(PHYSICS_GROUPS['red_edge']['indices'])} bands (700-750nm)")
print(f"  Water abs: {len(PHYSICS_GROUPS['water_absorption']['indices'])} bands (880-970nm)")
print(f"  NIR plateau: {len(PHYSICS_GROUPS['nir_plateau']['indices'])} bands (750-880nm)")


# ---------------------------------------------------------------------------
# Physics-constrained masking
# ---------------------------------------------------------------------------

def physics_constrained_mask(
    n_bands: int = NUM_BANDS,
    mask_ratio: float = 0.30,
    physics_prob: float = 0.70,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a physics-constrained band mask.

    With probability physics_prob, masked bands are drawn from physiologically
    meaningful spectral regions (red-edge, water absorption, NIR).
    With probability (1 - physics_prob), bands are masked randomly.

    Parameters
    ----------
    n_bands : total number of spectral bands
    mask_ratio : fraction of bands to mask
    physics_prob : probability of masking physics bands vs random
    rng : random number generator

    Returns
    -------
    mask : np.ndarray bool (n_bands,) — True = masked (to predict)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_masked = max(1, int(n_bands * mask_ratio))
    mask = np.zeros(n_bands, dtype=bool)

    n_physics = int(n_masked * physics_prob)
    n_random = n_masked - n_physics

    # Sample physics bands
    if n_physics > 0 and len(ALL_PHYSICS_INDICES) > 0:
        physics_sampled = rng.choice(
            ALL_PHYSICS_INDICES,
            size=min(n_physics, len(ALL_PHYSICS_INDICES)),
            replace=False,
        )
        mask[physics_sampled] = True

    # Sample remaining from non-physics bands
    if n_random > 0:
        remaining_indices = np.where(~mask)[0]
        random_sampled = rng.choice(
            remaining_indices,
            size=min(n_random, len(remaining_indices)),
            replace=False,
        )
        mask[random_sampled] = True

    return mask


def random_mask(
    n_bands: int = NUM_BANDS,
    mask_ratio: float = 0.30,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pure random masking — ablation baseline."""
    if rng is None:
        rng = np.random.default_rng()
    n_masked = max(1, int(n_bands * mask_ratio))
    indices = rng.choice(n_bands, size=n_masked, replace=False)
    mask = np.zeros(n_bands, dtype=bool)
    mask[indices] = True
    return mask


# ---------------------------------------------------------------------------
# HyperSL backbone loader
# ---------------------------------------------------------------------------

def load_hypersl_encoder(
    checkpoint_path: Path,
    device: torch.device,
    freeze: bool = True,
) -> nn.Module:
    """
    Load pretrained HyperSL encoder (SpectralSharedEncoder).
    Strips DataParallel 'module.' prefix from state dict.
    Returns only the encoder portion with frozen weights.
    """
    from engine.model import SpectralSharedEncoder

    model = SpectralSharedEncoder(
        embedding_dim=EMBED_DIM,
        num_heads=8,
        decoder_depth=4,
        encoder_depth=8,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    # Strip DataParallel 'module.' prefix
    clean_state = {
        k.replace("module.", ""): v
        for k, v in state.items()
    }

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"HyperSL loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        print("HyperSL backbone frozen")

    return model.to(device)


# ---------------------------------------------------------------------------
# JEPA components
# ---------------------------------------------------------------------------

class JEPAContextEncoder(nn.Module):
    """
    Lightweight JEPA context encoder adapter on top of frozen HyperSL.
    Takes visible (unmasked) spectral bands -> context embedding.
    Trainable — adapts HyperSL representations to agricultural domain.
    """

    def __init__(
        self,
        hypersl: nn.Module,
        embed_dim: int = EMBED_DIM,
        adapter_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hypersl = hypersl  # frozen backbone

        # Lightweight adapter: 2-layer MLP on top of HyperSL embedding
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        x: Tensor,
        wavelengths: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : (B, num_bands) — full spectrum
        wavelengths : (num_bands,) — wavelength values in nm
        mask : (B, num_bands) bool — True = masked bands

        Returns
        -------
        context_emb : (B, embed_dim) — adapted context embedding
        """
        # Zero out masked bands — encoder only sees visible bands
        x_visible = x.clone()
        # mask shape: (B, num_bands) bool
        x_visible[mask] = 0.0

        # HyperSL encoder forward (frozen)
        # HyperSL expects (B, num_bands) and wavelengths
        with torch.no_grad():
            hypersl_emb = self._hypersl_encode(x_visible, wavelengths)

        # Adapter (trainable)
        return self.adapter(hypersl_emb)

    def _hypersl_encode(self, x: Tensor, wavelengths: Tensor) -> Tensor:
        """
        Extract HyperSL embeddings.
        HyperSL encoder_forward expects:
          x: (B, 1, C) — batch, 1 spatial token, C bands
          wave: (B, C) — wavelength values per band
          mask_ratio: 0.0 at inference
        Returns z: (B, embed_dim) after aggregation
        """
        B = x.shape[0]
        # Reshape: (B, C) -> (B, 1, C)
        x_in = x.unsqueeze(1)
        # Wavelengths: (C,) -> (B, C)
        wave = wavelengths.unsqueeze(0).expand(B, -1)

        z, k, v, ids_restore, pos_tokens, shape = \
            self.hypersl.encoder_forward(x_in, wave, mask_ratio=0.0)

        # z shape: (B, embed_dim) after aggregate()
        if z.ndim == 3:
            z = z.squeeze(1) if z.shape[1] == 1 else z.mean(dim=1)
        return z


class JEPATargetEncoder(nn.Module):
    """
    EMA (exponential moving average) copy of context encoder.
    No gradients — provides stable target representations.
    """

    def __init__(self, context_encoder: JEPAContextEncoder) -> None:
        super().__init__()
        import copy
        self.encoder = copy.deepcopy(context_encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_ema(
        self,
        context_encoder: JEPAContextEncoder,
        decay: float = EMA_DECAY,
    ) -> None:
        """Update target encoder as EMA of context encoder."""
        for param_t, param_c in zip(
            self.encoder.parameters(),
            context_encoder.parameters(),
        ):
            param_t.data = decay * param_t.data + (1 - decay) * param_c.data

    def forward(
        self,
        x: Tensor,
        wavelengths: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Encode full spectrum (including masked bands) — target."""
        with torch.no_grad():
            return self.encoder(x, wavelengths, mask)


class JEPAPredictor(nn.Module):
    """
    Predicts target encoder embedding from context encoder embedding.
    3-layer MLP with residual connection.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, context_emb: Tensor) -> Tensor:
        return self.net(context_emb)


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, proj_dim: int = PROJ_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Multi-country HSI dataset
# ---------------------------------------------------------------------------

class MultiCountryHSIDataset(Dataset):
    """
    Combined HSI spectra from multiple countries for JEPA pretraining.
    No labels needed — purely self-supervised.
    
    Sources:
      India (groundnut): 282 bands, 400-1000nm
      USA (wheat):       190 bands, 400-900nm, zero-padded to 282
    """

    def __init__(
        self,
        spectra: np.ndarray,           # (N, 282) float32
        wavelengths: np.ndarray,       # (282,) nm values
        mask_ratio: float = 0.30,
        physics_prob: float = 0.70,
        use_physics_masking: bool = True,
    ) -> None:
        self.spectra = spectra.astype(np.float32)
        self.wavelengths = wavelengths.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.physics_prob = physics_prob
        self.use_physics_masking = use_physics_masking
        self.rng = np.random.default_rng(42)
        print(f"  Dataset: {len(spectra)} spectra, {spectra.shape[1]} bands")

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        x = self.spectra[idx].copy()

        # Generate mask
        if self.use_physics_masking:
            mask = physics_constrained_mask(
                len(x), self.mask_ratio, self.physics_prob, self.rng
            )
        else:
            mask = random_mask(len(x), self.mask_ratio, self.rng)

        return (
            torch.from_numpy(x),
            torch.from_numpy(self.wavelengths),
            torch.from_numpy(mask),
        )


# ---------------------------------------------------------------------------
# Load multi-country data
# ---------------------------------------------------------------------------

def load_multicountry_hsi(data_root: Path) -> tuple[np.ndarray, dict]:
    """Load and combine HSI spectra from India and USA."""
    spectra_list = []
    metadata = {}

    # India — Groundnut (282 bands)
    proc = data_root / "processed" / "hsi"
    gn_flat = np.load(proc / "gn_X_flat.npy").astype(np.float32)
    spectra_list.append(gn_flat)
    metadata["India_groundnut"] = {
        "n_samples": len(gn_flat),
        "bands": 282,
        "wavelength_range": "400-1000nm",
        "crop": "groundnut",
        "task": "water_stress",
    }
    print(f"  India (groundnut): {gn_flat.shape}")

    # USA — Wheat (190 bands, zero-padded to 282)
    wheat_path = data_root / "umn_wheat" / "wheat_X.npy"
    if wheat_path.exists():
        wheat = np.load(wheat_path).astype(np.float32)  # (15315, 190)
        pad = NUM_BANDS - wheat.shape[1]
        wheat_padded = np.pad(wheat, ((0,0),(0,pad)), constant_values=0)
        spectra_list.append(wheat_padded)
        metadata["USA_wheat"] = {
            "n_samples": len(wheat),
            "bands": 190,
            "padded_to": 282,
            "wavelength_range": "400-900nm",
            "crop": "wheat",
            "task": "yield_proxy_stress",
        }
        print(f"  USA (wheat): {wheat.shape} -> padded to {wheat_padded.shape}")

    combined = np.concatenate(spectra_list, axis=0)
    print(f"  Combined: {combined.shape} from {len(spectra_list)} countries")
    return combined, metadata


# ---------------------------------------------------------------------------
# JEPA training
# ---------------------------------------------------------------------------

def train_jepa_epoch(
    context_enc: JEPAContextEncoder,
    target_enc: JEPATargetEncoder,
    predictor: JEPAPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> float:
    context_enc.train()
    predictor.train()
    total_loss = 0.0
    n = 0

    for x, wavelengths, mask in loader:
        x = x.to(DEVICE, non_blocking=True)
        wavelengths = wavelengths[0].to(DEVICE)  # same for all in batch
        mask = mask.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            # Context: visible bands -> context embedding
            context_emb = context_enc(x, wavelengths, mask)
            # Predict target embedding from context
            predicted_emb = predictor(context_emb)

            # Target: full spectrum -> target embedding (EMA encoder, no grad)
            with torch.no_grad():
                target_emb = target_enc(x, wavelengths, mask)

            # JEPA loss: cosine similarity + variance regularisation
            # Cosine similarity loss (main JEPA objective)
            pred_norm = F.normalize(predicted_emb, dim=-1)
            tgt_norm = F.normalize(target_emb.detach(), dim=-1)
            cos_loss = 1.0 - (pred_norm * tgt_norm).sum(dim=-1).mean()

            # Variance regularisation: prevent collapse to constant embedding
            pred_std = predicted_emb.std(dim=0).mean()
            var_loss = F.relu(1.0 - pred_std)

            loss = cos_loss + 0.1 * var_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(context_enc.adapter.parameters()) +
            list(predictor.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        # Update target encoder EMA
        target_enc.update_ema(context_enc, decay=EMA_DECAY)

        total_loss += loss.item()
        n += 1

    return total_loss / n


# ---------------------------------------------------------------------------
# Few-shot evaluation (prototypical network)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_jepa_embeddings(
    context_enc: JEPAContextEncoder,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract JEPA context embeddings (no masking at inference)."""
    context_enc.eval()
    wavelengths = torch.from_numpy(WAVELENGTHS.astype(np.float32)).to(DEVICE)
    all_emb = []

    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start+batch_size]).to(DEVICE)
        # No masking at inference — encode full spectrum
        mask = torch.zeros(len(batch), NUM_BANDS, dtype=torch.bool, device=DEVICE)
        emb = context_enc(batch, wavelengths, mask)
        all_emb.append(F.normalize(emb, dim=-1).cpu().numpy())

    return np.concatenate(all_emb)


def proto_episode(
    emb: np.ndarray,
    labels: np.ndarray,
    n_shot: int,
    n_query: int,
    rng: np.random.Generator,
) -> float:
    sup_e, sup_l, qry_e, qry_l = [], [], [], []
    for cls in range(2):
        idx = np.where(labels == cls)[0]
        if len(idx) < n_shot + n_query:
            return float("nan")
        s = rng.choice(idx, n_shot + n_query, replace=False)
        sup_e.append(emb[s[:n_shot]])
        sup_l += [cls] * n_shot
        qry_e.append(emb[s[n_shot:]])
        qry_l += [cls] * n_query
    sup_e = np.concatenate(sup_e)
    qry_e = np.concatenate(qry_e)
    qry_l = np.array(qry_l)
    protos = np.stack([sup_e[np.array(sup_l)==c].mean(0) for c in range(2)])
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    preds = (qry_e @ protos.T).argmax(axis=1)
    return float((preds == qry_l).mean())


def run_fewshot_eval(
    context_enc: JEPAContextEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str = "",
) -> dict:
    emb = extract_jepa_embeddings(context_enc, X_test)
    rng = np.random.default_rng(42)
    results = {}

    print(f"\n{'N-shot':>8} {'Mean Acc':>10} {'95% CI':>8}  [{label}]")
    print("-" * 45)

    for n_shot in N_SHOT_VALUES:
        accs = [proto_episode(emb, y_test, n_shot, NUM_QUERY, rng)
                for _ in range(NUM_EPISODES)]
        accs = [a for a in accs if not np.isnan(a)]
        mean = np.mean(accs)
        ci = 1.96 * np.std(accs) / len(accs)**0.5
        results[n_shot] = {"mean": float(mean), "ci_95": float(ci)}
        print(f"{n_shot:>6}-shot {mean*100:>10.2f}% {ci*100:>7.2f}%")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load groundnut test set for evaluation
    proc = DATA_ROOT / "processed" / "hsi"
    gn_flat = np.load(proc / "gn_X_flat.npy").astype(np.float32)
    gn_y = np.load(proc / "gn_y.npy")
    test_idx = np.load(proc / "gn_test_idx.npy")
    X_test = gn_flat[test_idx]
    y_test = gn_y[test_idx]
    print(f"Groundnut test set: {X_test.shape}")

    # Load multi-country pretraining data
    print("\nLoading multi-country HSI data:")
    combined, metadata = load_multicountry_hsi(DATA_ROOT)

    # Load HyperSL backbone
    print("\nLoading HyperSL backbone...")
    hypersl = load_hypersl_encoder(
        checkpoint_path=Path("models/hypersl_weights/10_base_mask95_checkpoint.pt"),
        device=DEVICE,
        freeze=True,
    )

    # Build JEPA components
    context_enc = JEPAContextEncoder(hypersl, embed_dim=EMBED_DIM).to(DEVICE)
    target_enc = JEPATargetEncoder(context_enc).to(DEVICE)
    predictor = JEPAPredictor(embed_dim=EMBED_DIM).to(DEVICE)
    proj_head = ProjectionHead(embed_dim=EMBED_DIM, proj_dim=PROJ_DIM).to(DEVICE)

    trainable = sum(p.numel() for p in context_enc.adapter.parameters()) + \
                sum(p.numel() for p in predictor.parameters())
    total = sum(p.numel() for p in context_enc.parameters()) + \
            sum(p.numel() for p in predictor.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({trainable/total*100:.1f}%)")

    # Run ablation: physics masking vs random masking
    all_results = {}

    for use_physics in [True, False]:
        mask_type = "physics_constrained" if use_physics else "random"
        print(f"\n{'='*55}")
        print(f"JEPA Masking: {mask_type.upper()}")
        print(f"{'='*55}")

        # Reset JEPA components for fair comparison
        import copy
        ctx = JEPAContextEncoder(hypersl, embed_dim=EMBED_DIM).to(DEVICE)
        tgt = JEPATargetEncoder(ctx).to(DEVICE)
        pred = JEPAPredictor(embed_dim=EMBED_DIM).to(DEVICE)

        # Dataset
        dataset = MultiCountryHSIDataset(
            combined,
            WAVELENGTHS,
            mask_ratio=0.30,
            physics_prob=0.70 if use_physics else 0.0,
            use_physics_masking=use_physics,
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Loader: {len(loader)} batches/epoch")

        # Optimizer — only adapter and predictor are trainable
        optimizer = torch.optim.AdamW(
            list(ctx.adapter.parameters()) + list(pred.parameters()),
            lr=LR,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, epochs=EPOCHS,
            steps_per_epoch=len(loader), pct_start=0.1
        )
        scaler = torch.amp.GradScaler("cuda",
                                       enabled=torch.cuda.is_available())

        # Baseline before pretraining
        print("\n--- Before JEPA pretraining ---")
        baseline = run_fewshot_eval(ctx, X_test, y_test, "random init")

        # Train
        print(f"\nTraining JEPA ({mask_type}) for {EPOCHS} epochs...")
        print(f"{'Epoch':>6} {'Loss':>10} {'Time':>7}")
        print("-" * 28)

        history = []
        best_loss = float("inf")

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            loss = train_jepa_epoch(ctx, tgt, pred, loader, optimizer, scaler)
            scheduler.step()
            elapsed = time.time() - t0
            history.append({"epoch": epoch, "loss": loss})

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
                }, CHECKPOINT_DIR / f"jepa_{mask_type}_best.pt")

        print(f"Best loss: {best_loss:.4f}")

        # Post-training evaluation
        print(f"\n--- After JEPA pretraining ({mask_type}) ---")
        post = run_fewshot_eval(ctx, X_test, y_test, mask_type)

        all_results[mask_type] = {
            "baseline": baseline,
            "post_jepa": post,
            "best_loss": best_loss,
        }

    # Final comparison
    print("\n" + "="*70)
    print("PHYSICS-CONSTRAINED JEPA — FULL COMPARISON")
    print("="*70)
    print(f"{'N-shot':>8} {'Supervised':>12} {'Random-JEPA':>13} {'Physics-JEPA':>14}")
    print("-"*55)

    supervised = {1: 0.9433, 5: 0.9775, 10: 0.9813, 20: 0.9824}
    for n_shot in N_SHOT_VALUES:
        sup = supervised[n_shot] * 100
        rnd = all_results["random"]["post_jepa"][n_shot]["mean"] * 100
        phy = all_results["physics_constrained"]["post_jepa"][n_shot]["mean"] * 100
        delta = phy - rnd
        print(f"{n_shot:>6}-shot {sup:>12.2f}% {rnd:>12.2f}% "
              f"{phy:>13.2f}% (Δ{delta:+.2f}%)")

    print(f"\nDatasets used: {list(metadata.keys())}")
    print(f"Countries: India (groundnut), USA (wheat)")
    print(f"Next: add Turkey (maize MS) and Sri Lanka (paddy MS) — see ms_jepa.py")

    # Save
    output = {
        "results": all_results,
        "supervised_reference": supervised,
        "metadata": metadata,
        "physics_groups": {
            k: {"range": v["range"], "n_bands": len(v["indices"])}
            for k, v in PHYSICS_GROUPS.items()
        },
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "mask_ratio": 0.30,
            "physics_prob": 0.70,
            "ema_decay": EMA_DECAY,
            "backbone": "HyperSL",
        }
    }

    with open(LOG_DIR / "physics_jepa_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {LOG_DIR / 'physics_jepa_results.json'}")


if __name__ == "__main__":
    main()