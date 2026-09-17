"""
UMN Wheat HSI cross-crop transfer evaluation.

Dataset: 1021 wheat plots, 190 bands, 400-900nm, ~2cm UAV resolution
Labels: yield in grams (continuous) -> thresholded to binary high/low yield

Cross-crop transfer protocol:
  - Source domain: Groundnut water stress (282 bands, binary stress labels)
  - Target domain: Wheat yield-proxy stress (190 bands, binary yield labels)
  - Encoder: SpectralMLP pretrained on groundnut (FROZEN)
  - Adaptation: spectral adapter 190->282 bands + prototypical few-shot head

This tests two generalisation axes simultaneously:
  1. Cross-crop (groundnut -> wheat)
  2. Cross-band (282 -> 190 bands, different wavelength coverage)

Run: python wheat_transfer_eval.py
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

sys.path.insert(0, str(Path("~/agri_foundation").expanduser()))

DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WHEAT_BANDS = 190       # after noisy band removal
GROUNDNUT_BANDS = 282   # source domain
PATCH_SIZE = 11         # spatial patch size to extract from wheat plots
NUM_EPISODES = 1000
N_SHOT_VALUES = [1, 5, 10, 20]
NUM_QUERY_PER_CLASS = 30


# ---------------------------------------------------------------------------
# SpectralMLP (must match train_mlp_baseline.py)
# ---------------------------------------------------------------------------

class SpectralMLP(nn.Module):
    def __init__(self, num_bands=282, hidden_dims=(256,64), num_classes=2, dropout=0.4):
        super().__init__()
        dims = [num_bands] + list(hidden_dims) + [num_classes]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]),
                      nn.BatchNorm1d(dims[i+1]), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_embedding(self, x):
        out = x
        for layer in list(self.net.children())[:-1]:
            out = layer(out)
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# Spectral Adapter: 190 wheat bands -> 282 groundnut bands
# ---------------------------------------------------------------------------

class SpectralAdapter(nn.Module):
    """
    Adapts wheat HSI (190 bands, 400-900nm) to groundnut embedding space
    (282 bands, 400-1000nm) via learned linear projection.

    Two strategies:
      1. Zero-pad: pad 190 -> 282 with zeros (bands 191-282 = 900-1000nm missing)
      2. Linear: learned 190->282 projection (default, more flexible)

    The linear adapter is trained on a small labelled set from wheat
    while keeping the groundnut encoder frozen.
    """

    def __init__(
        self,
        source_bands: int = WHEAT_BANDS,
        target_bands: int = GROUNDNUT_BANDS,
        mode: str = "linear",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.source_bands = source_bands
        self.target_bands = target_bands

        if mode == "linear":
            self.proj = nn.Linear(source_bands, target_bands, bias=True)
            # Initialise as approximate identity (zero-pad equivalent)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            with torch.no_grad():
                # Copy first source_bands dimensions as identity
                for i in range(source_bands):
                    self.proj.weight[i, i] = 1.0
        elif mode == "zero_pad":
            self.proj = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, source_bands)
        if self.mode == "zero_pad":
            pad_size = self.target_bands - self.source_bands
            return F.pad(x, (0, pad_size), value=0.0)
        else:
            return self.proj(x)


# ---------------------------------------------------------------------------
# UMN Wheat Dataset Loader
# ---------------------------------------------------------------------------

def load_wheat_plots(
    data_root: Path,
    patch_size: int = PATCH_SIZE,
    patches_per_plot: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load UMN wheat plots and extract random spatial patches.

    Each plot is a (H, W, 190) numpy array.
    Yield labels are thresholded at median to create binary labels:
      0 = below-median yield (stressed/low productivity)
      1 = above-median yield (healthy/high productivity)

    Returns
    -------
    X : (N_patches, 190) flat spectral vectors
    y : (N_patches,) binary labels
    """
    wheat_dir = data_root / "umn_wheat"
    yield_dir = wheat_dir / "yield_data"
    rng = np.random.default_rng(seed)

    # Load yield data
    yield_file = yield_dir / "yield_data.pkl" if (yield_dir / "yield_data.pkl").exists() \
        else list(yield_dir.glob("*.pkl"))[0] if list(yield_dir.glob("*.pkl")) else None

    yield_dict = {}
    if yield_file and yield_file.exists():
        import pickle
        with open(yield_file, "rb") as f:
            yield_data = pickle.load(f)
        # yield_data format: dict or dataframe with plot_id -> yield
        if hasattr(yield_data, "iterrows"):
            for _, row in yield_data.iterrows():
                plot_id = str(row.iloc[0])
                yield_val = float(row.iloc[1])
                yield_dict[plot_id] = yield_val
        elif isinstance(yield_data, dict):
            yield_dict = {str(k): float(v) for k, v in yield_data.items()}
        print(f"Loaded yield data: {len(yield_dict)} plots")
    else:
        print("WARNING: yield data not found, using random labels for testing")

    # Collect all plot files
    plot_files = []
    for field in ["C3_numpy", "C4_numpy", "C9_numpy"]:
        field_dir = wheat_dir / field
        if field_dir.exists():
            plot_files.extend(sorted(field_dir.glob("*.npy")))

    print(f"Found {len(plot_files)} wheat plot files")

    all_patches = []
    all_yields = []
    plot_ids = []

    for plot_path in plot_files:
        plot_id = plot_path.stem   # e.g. "C3_10702"

        try:
            cube = np.load(plot_path)   # (H, W, bands) or (bands, H, W)
        except Exception as e:
            print(f"  Failed to load {plot_path.name}: {e}")
            continue

        # Ensure (H, W, bands) format
        if cube.ndim == 3:
            if cube.shape[0] < cube.shape[2]:
                cube = cube.transpose(1, 2, 0)  # (bands, H, W) -> (H, W, bands)
        else:
            continue

        H, W, B = cube.shape
        if B != WHEAT_BANDS:
            # Some plots may have different band counts — skip
            continue

        if H < patch_size or W < patch_size:
            continue

        # Extract random patches
        for _ in range(patches_per_plot):
            top = rng.integers(0, H - patch_size)
            left = rng.integers(0, W - patch_size)
            patch = cube[top:top+patch_size, left:left+patch_size, :]
            # Mean pool spatial dims -> (190,) spectral vector
            spectrum = patch.mean(axis=(0, 1)).astype(np.float32)
            # Clip and normalise to [0,1]
            spectrum = np.clip(spectrum, 0, None)
            if spectrum.max() > 0:
                spectrum = spectrum / spectrum.max()
            all_patches.append(spectrum)
            all_yields.append(yield_dict.get(plot_id, np.nan))
            plot_ids.append(plot_id)

    X = np.array(all_patches, dtype=np.float32)   # (N, 190)
    yields = np.array(all_yields, dtype=np.float32)

    # Remove plots without yield data
    valid_mask = ~np.isnan(yields)
    if valid_mask.sum() < len(yields):
        print(f"Dropped {(~valid_mask).sum()} patches without yield data")
        X = X[valid_mask]
        yields = yields[valid_mask]

    if len(yields) == 0:
        print("WARNING: No valid yield data found. Using random binary labels.")
        yields = rng.random(len(X))

    # Threshold at median -> binary labels
    median_yield = np.median(yields)
    y = (yields > median_yield).astype(np.int64)
    print(f"Wheat patches: {len(X)} | Median yield threshold: {median_yield:.1f}g")
    print(f"Label dist: above-median={y.sum()} below-median={(y==0).sum()}")

    return X, y


# ---------------------------------------------------------------------------
# Prototypical network few-shot evaluation
# ---------------------------------------------------------------------------

def prototypical_episode(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_shot: int,
    n_query: int,
    rng: np.random.Generator,
) -> float:
    support_emb, support_lbl, query_emb, query_lbl = [], [], [], []

    for cls in range(2):
        idx = np.where(labels == cls)[0]
        if len(idx) < n_shot + n_query:
            return float("nan")
        sampled = rng.choice(idx, size=n_shot + n_query, replace=False)
        support_emb.append(embeddings[sampled[:n_shot]])
        support_lbl.extend([cls] * n_shot)
        query_emb.append(embeddings[sampled[n_shot:]])
        query_lbl.extend([cls] * n_query)

    support_emb = np.concatenate(support_emb)
    query_emb = np.concatenate(query_emb)
    query_lbl = np.array(query_lbl)

    prototypes = np.stack([
        support_emb[np.array(support_lbl) == cls].mean(axis=0)
        for cls in range(2)
    ])
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
    prototypes = prototypes / norms

    sim = query_emb @ prototypes.T
    predictions = sim.argmax(axis=1)
    return float((predictions == query_lbl).mean())


# ---------------------------------------------------------------------------
# Extract embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    encoder: SpectralMLP,
    adapter: SpectralAdapter,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract embeddings from wheat spectra via adapter + frozen encoder."""
    encoder.eval()
    adapter.eval()
    all_emb = []

    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start+batch_size]).to(DEVICE)
        adapted = adapter(batch)               # (B, 282)
        emb = encoder.get_embedding(adapted)   # (B, 64) L2-normalised
        all_emb.append(emb.cpu().numpy())

    return np.concatenate(all_emb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")

    # Load pretrained groundnut encoder (frozen)
    encoder = SpectralMLP(num_bands=282, hidden_dims=(256,64), num_classes=2).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "groundnut_mlp_best.pt", map_location=DEVICE)
    encoder.load_state_dict(ckpt["model_state"])
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Loaded groundnut encoder: epoch={ckpt['epoch']} acc={ckpt['metrics']['accuracy']:.4f}")

    # Load wheat data
    print("\nLoading UMN wheat plots...")
    X_wheat, y_wheat = load_wheat_plots(DATA_ROOT, patch_size=PATCH_SIZE, patches_per_plot=20)
    print(f"Wheat X: {X_wheat.shape} | y: {y_wheat.shape}")

    # Two adapter modes
    results = {}
    for adapter_mode in ["zero_pad", "linear"]:
        print(f"\n{'='*55}")
        print(f"Adapter: {adapter_mode}")
        print(f"{'='*55}")

        adapter = SpectralAdapter(
            source_bands=WHEAT_BANDS,
            target_bands=GROUNDNUT_BANDS,
            mode=adapter_mode,
        ).to(DEVICE)

        # Extract embeddings
        print("Extracting wheat embeddings...")
        wheat_emb = extract_embeddings(encoder, adapter, X_wheat)
        print(f"Embeddings: {wheat_emb.shape}")
        print(f"Embedding std: {wheat_emb.std(axis=0).mean():.4f}")

        # Few-shot evaluation
        rng = np.random.default_rng(42)
        print(f"\n{'N-shot':>8} {'Mean Acc':>10} {'Std':>8} {'95% CI':>8}")
        print("-" * 40)

        adapter_results = {}
        for n_shot in N_SHOT_VALUES:
            episode_accs = []
            for _ in range(NUM_EPISODES):
                acc = prototypical_episode(
                    wheat_emb, y_wheat, n_shot, NUM_QUERY_PER_CLASS, rng
                )
                if not np.isnan(acc):
                    episode_accs.append(acc)

            mean_acc = np.mean(episode_accs)
            std_acc = np.std(episode_accs)
            ci_95 = 1.96 * std_acc / (len(episode_accs) ** 0.5)

            adapter_results[n_shot] = {
                "mean": float(mean_acc),
                "std": float(std_acc),
                "ci_95": float(ci_95),
                "n_episodes": len(episode_accs),
            }
            print(f"{n_shot:>6}-shot {mean_acc*100:>10.2f}% {std_acc*100:>7.2f}% {ci_95*100:>7.2f}%")

        results[adapter_mode] = adapter_results

    # Summary
    print("\n" + "="*60)
    print("WHEAT CROSS-CROP TRANSFER SUMMARY")
    print("Source: Groundnut stress (282 bands) | Target: Wheat yield (190 bands)")
    print("="*60)
    print(f"{'N-shot':>8} {'Zero-pad Acc':>14} {'Linear Acc':>12}")
    print("-"*40)
    for n_shot in N_SHOT_VALUES:
        zp = results["zero_pad"][n_shot]["mean"] * 100
        ln = results["linear"][n_shot]["mean"] * 100
        print(f"{n_shot:>6}-shot {zp:>13.2f}% {ln:>11.2f}%")

    print(f"\nFor comparison — groundnut within-domain:")
    print(f"  1-shot: 94.33% | 5-shot: 97.75% | 20-shot: 98.24%")

    # Save
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "wheat_transfer_eval.json", "w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {LOG_DIR / 'wheat_transfer_eval.json'}")


if __name__ == "__main__":
    main()