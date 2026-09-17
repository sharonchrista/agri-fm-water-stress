"""
Few-shot prototypical network evaluation for groundnut HSI stress classification.

Evaluates how well the trained SpectralMLP encoder generalises to new
labelled examples using prototypical networks — the core few-shot mechanism
in the foundation model framework.

Protocol:
  - Load pretrained SpectralMLP encoder (frozen)
  - From the test set, sample N-shot episodes (N = 1, 5, 10, 20)
  - Each episode: N labelled support examples per class -> class prototypes
  - Classify remaining query examples by nearest prototype in embedding space
  - Repeat 1000 episodes per N-shot setting, report mean accuracy ± std

This directly validates the few-shot generalisation claim in the paper.

Run: python few_shot_eval.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 1000
N_SHOT_VALUES = [1, 5, 10, 20]
NUM_CLASSES = 2
NUM_QUERY_PER_CLASS = 50   # query examples per class per episode


# ---------------------------------------------------------------------------
# Model definition (must match train_mlp_baseline.py exactly)
# ---------------------------------------------------------------------------

class SpectralMLP(nn.Module):
    def __init__(
        self,
        num_bands: int = 282,
        hidden_dims: tuple[int, ...] = (256, 64),
        num_classes: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        dims = [num_bands] + list(hidden_dims) + [num_classes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Return penultimate layer activations — the representation space."""
        # Pass through all layers except the final classification linear
        out = x
        for layer in list(self.net.children())[:-1]:
            out = layer(out)
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# Prototypical network evaluation
# ---------------------------------------------------------------------------

def prototypical_episode(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_shot: int,
    n_query: int,
    rng: np.random.Generator,
) -> float:
    """
    Run one prototypical network episode.

    For each class:
      - Sample n_shot support examples -> compute prototype (mean embedding)
      - Sample n_query query examples (disjoint from support)

    Classify each query by nearest prototype using cosine distance.
    Returns episode accuracy.
    """
    support_embeddings = []
    support_labels = []
    query_embeddings = []
    query_labels = []

    for cls in range(NUM_CLASSES):
        cls_indices = np.where(labels == cls)[0]

        if len(cls_indices) < n_shot + n_query:
            # Not enough samples — skip this class in episode
            return float("nan")

        # Sample support and query indices (disjoint)
        sampled = rng.choice(cls_indices, size=n_shot + n_query, replace=False)
        support_idx = sampled[:n_shot]
        query_idx = sampled[n_shot:]

        support_embeddings.append(embeddings[support_idx])
        support_labels.extend([cls] * n_shot)
        query_embeddings.append(embeddings[query_idx])
        query_labels.extend([cls] * n_query)

    # Stack into arrays
    support_emb = np.concatenate(support_embeddings, axis=0)   # (n_shot*C, D)
    query_emb = np.concatenate(query_embeddings, axis=0)        # (n_query*C, D)
    query_lbl = np.array(query_labels)

    # Compute class prototypes — mean of support embeddings per class
    prototypes = np.stack([
        support_emb[np.array(support_labels) == cls].mean(axis=0)
        for cls in range(NUM_CLASSES)
    ])   # (C, D)

    # Normalise prototypes
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
    prototypes = prototypes / norms

    # Classify by cosine similarity (embeddings already L2-normalised)
    # sim[i, j] = cosine similarity between query i and prototype j
    sim = query_emb @ prototypes.T   # (n_query*C, C)
    predictions = sim.argmax(axis=1)

    accuracy = (predictions == query_lbl).mean()
    return float(accuracy)


# ---------------------------------------------------------------------------
# Extract embeddings from pretrained encoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: SpectralMLP,
    X: np.ndarray,
) -> np.ndarray:
    """Extract penultimate-layer embeddings for all samples."""
    model.eval()
    X_tensor = torch.from_numpy(X.reshape(len(X), -1).astype(np.float32))

    batch_size = 512
    all_embeddings = []

    for start in range(0, len(X_tensor), batch_size):
        batch = X_tensor[start:start + batch_size].to(DEVICE)
        emb = model.get_embedding(batch)
        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")

    # Load data
    proc = DATA_ROOT / "processed" / "hsi"
    X = np.load(proc / "gn_X_patch.npy")
    y = np.load(proc / "gn_y.npy")
    test_idx = np.load(proc / "gn_test_idx.npy")

    X_test = X[test_idx]
    y_test = y[test_idx]
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Load pretrained model
    model = SpectralMLP(num_bands=282, hidden_dims=(256, 64), num_classes=2, dropout=0.4)
    ckpt = torch.load(
        CHECKPOINT_DIR / "groundnut_mlp_best.pt",
        map_location="cpu",
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(full supervision acc: {ckpt['metrics']['accuracy']:.4f})")

    # Extract embeddings for all test samples
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model, X_test)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding norm range: [{np.linalg.norm(embeddings, axis=1).min():.4f}, "
          f"{np.linalg.norm(embeddings, axis=1).max():.4f}]")

    # Run few-shot episodes
    rng = np.random.default_rng(42)

    print(f"\n{'N-shot':>8} {'Mean Acc':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)

    results = {}
    for n_shot in N_SHOT_VALUES:
        episode_accs = []
        for _ in range(NUM_EPISODES):
            acc = prototypical_episode(
                embeddings, y_test, n_shot, NUM_QUERY_PER_CLASS, rng
            )
            if not np.isnan(acc):
                episode_accs.append(acc)

        mean_acc = np.mean(episode_accs)
        std_acc = np.std(episode_accs)
        min_acc = np.min(episode_accs)
        max_acc = np.max(episode_accs)

        results[n_shot] = {
            "mean": float(mean_acc),
            "std": float(std_acc),
            "min": float(min_acc),
            "max": float(max_acc),
            "n_episodes": len(episode_accs),
        }

        print(f"{n_shot:>6}-shot {mean_acc:>10.4f} {std_acc:>8.4f} "
              f"{min_acc:>8.4f} {max_acc:>8.4f}")

    # Upper bound — full supervision accuracy
    print(f"\n{'Full':>6}-supv {ckpt['metrics']['accuracy']:>10.4f} "
          f"{'—':>8} {'—':>8} {'—':>8}  (all {len(test_idx)} test samples)")

    # Summary for paper
    print("\n" + "=" * 50)
    print("FEW-SHOT RESULTS SUMMARY (for paper Table 2)")
    print("=" * 50)
    print(f"Task: Groundnut Water Stress (HSI, 282 bands, binary)")
    print(f"Encoder: SpectralMLP (pretrained, frozen)")
    print(f"Episodes: {NUM_EPISODES} per N-shot setting")
    print(f"Query: {NUM_QUERY_PER_CLASS} samples per class per episode")
    print()
    for n_shot, res in results.items():
        ci_95 = 1.96 * res['std'] / (res['n_episodes'] ** 0.5)
        print(f"  {n_shot:2d}-shot: {res['mean']*100:.2f}% ± {ci_95*100:.2f}% (95% CI)")
    print(f"  Full supervision: {ckpt['metrics']['accuracy']*100:.2f}%")

    # Save results
    import json
    out_path = Path("~/agri_foundation/logs/few_shot_results.json").expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()