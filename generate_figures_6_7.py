"""
Figure 6: SHAP band attribution for groundnut water stress (SpectralMLP)
Figure 7: Density map sample predictions (DensityNet RGB)

Run: python generate_figures_6_7.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FIG_DIR = Path("~/agri_foundation/figures").expanduser()
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path("~/agri_foundation/data").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

C_STRESSED = "#E63946"
C_HEALTHY = "#52B788"
C_NEUTRAL = "#ADB5BD"


# ---------------------------------------------------------------------------
# SpectralMLP definition (must match train_mlp_baseline.py)
# ---------------------------------------------------------------------------

class SpectralMLP(nn.Module):
    def __init__(self, num_bands=282, hidden_dims=(256, 64), num_classes=2, dropout=0.4):
        super().__init__()
        dims = [num_bands] + list(hidden_dims) + [num_classes]
        layers = []
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Figure 6 — SHAP Band Attribution
# ---------------------------------------------------------------------------

def fig6_shap() -> None:
    print("Generating Figure 6: SHAP band attribution...")

    try:
        import shap
    except ImportError:
        print("  Installing shap...")
        import subprocess
        subprocess.run(["pip", "install", "shap", "-q"], check=True)
        import shap

    # Load model and data
    model = SpectralMLP().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "groundnut_mlp_best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded SpectralMLP epoch {ckpt['epoch']}")

    proc = DATA_ROOT / "processed" / "hsi"
    X = np.load(proc / "gn_X_patch.npy").reshape(-1, 282).astype(np.float32)
    y = np.load(proc / "gn_y.npy")
    test_idx = np.load(proc / "gn_test_idx.npy")
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Use 200 background samples and 500 test samples for SHAP
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_test), size=200, replace=False)
    explain_idx = rng.choice(len(X_test), size=500, replace=False)

    X_bg = torch.from_numpy(X_test[bg_idx]).to(DEVICE)
    X_explain = torch.from_numpy(X_test[explain_idx]).to(DEVICE)
    y_explain = y_test[explain_idx]

    print(f"  Running SHAP GradientExplainer on {len(X_explain)} samples...")

    # GradientExplainer works better with BatchNorm layers than DeepExplainer
    explainer = shap.GradientExplainer(model, X_bg)
    shap_values = explainer.shap_values(X_explain)

    # shap_values: list of [n_samples, n_features] per class
    if isinstance(shap_values, list):
        shap_stressed = np.array(shap_values[1])
        shap_healthy = np.array(shap_values[0])
    else:
        shap_stressed = np.array(shap_values)
        shap_healthy = -shap_stressed

    # Squeeze any extra dims and ensure 2D (n_samples, n_bands)
    shap_stressed = shap_stressed.reshape(len(X_explain), -1)
    shap_healthy = shap_healthy.reshape(len(X_explain), -1)

    # If SHAP returns doubled features (BatchNorm artifact), take first 282
    if shap_stressed.shape[1] != 282:
        print(f"  SHAP returned {shap_stressed.shape[1]} features, truncating to 282")
        shap_stressed = shap_stressed[:, :282]
        shap_healthy = shap_healthy[:, :282]

    # Mean absolute SHAP per band — strictly 1D (282,)
    mean_shap_stressed = np.abs(shap_stressed).mean(axis=0)
    mean_shap_healthy = np.abs(shap_healthy).mean(axis=0)
    directional = shap_stressed.mean(axis=0)

    print(f"  SHAP arrays: stressed={shap_stressed.shape} directional={directional.shape}")

    # Approximate wavelengths: 400-1000nm over 282 bands
    wavelengths = np.linspace(400, 1000, 282)

    # Known spectral regions for annotation
    regions = [
        (400, 500, "Blue",     "#4895EF", 0.08),
        (500, 600, "Green",    "#52B788", 0.08),
        (600, 700, "Red",      "#E63946", 0.08),
        (700, 750, "Red-Edge", "#FF9F1C", 0.12),
        (750, 900, "NIR",      "#9B5DE5", 0.08),
        (900, 1000,"SWIR",     "#8B4513", 0.06),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- Top panel: Mean absolute SHAP by band ---
    ax = axes[0]

    # Colour bars by direction
    colours = [C_STRESSED if float(d) > 0 else C_HEALTHY for d in directional]
    ax.bar(wavelengths, mean_shap_stressed, color=colours, alpha=0.7,
           width=2.2, label="Mean |SHAP| per band")

    # Shade spectral regions
    y_max = mean_shap_stressed.max()
    for wl_start, wl_end, name, color, alpha in regions:
        ax.axvspan(wl_start, wl_end, alpha=alpha, color=color, zorder=0)
        mid = (wl_start + wl_end) / 2
        ax.text(mid, y_max * 0.92, name, ha="center", fontsize=8,
                color="black", alpha=0.7)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(
        "(a) Per-band SHAP attribution — Groundnut Water Stress (SpectralMLP, 282 bands)\n"
        "Red = pushes toward stressed prediction | Green = pushes toward healthy prediction"
    )
    ax.set_xlim(400, 1000)

    stressed_patch = plt.Rectangle((0,0),1,1, fc=C_STRESSED, alpha=0.7)
    healthy_patch = plt.Rectangle((0,0),1,1, fc=C_HEALTHY, alpha=0.7)
    ax.legend([stressed_patch, healthy_patch],
              ["Positive attribution (stressed)", "Negative attribution (healthy)"],
              loc="upper left", fontsize=9)

    # --- Bottom panel: Top 20 most important bands ---
    ax2 = axes[1]
    top20_idx = np.argsort(mean_shap_stressed)[::-1][:20]
    top20_wl = wavelengths[top20_idx]
    top20_shap = mean_shap_stressed[top20_idx]
    top20_dir = directional[top20_idx]
    top20_colours = [C_STRESSED if float(d) > 0 else C_HEALTHY for d in top20_dir]

    bars = ax2.barh(range(20), top20_shap, color=top20_colours, alpha=0.8)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([f"{wl:.0f} nm" for wl in top20_wl], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_title("(b) Top 20 most important spectral bands")

    for i, (bar, shap_val) in enumerate(zip(bars, top20_shap)):
        ax2.text(shap_val + max(top20_shap)*0.01, i,
                 f"{shap_val:.4f}", va="center", fontsize=8)

    fig.tight_layout(pad=2.5)
    out = FIG_DIR / "fig6_shap_attribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")

    # Print top bands for paper text
    print("\n  Top 10 most important bands:")
    for i in range(10):
        idx = top20_idx[i]
        direction = "stressed" if float(directional[idx]) > 0 else "healthy"
        print(f"    {wavelengths[idx]:.0f} nm — SHAP {mean_shap_stressed[idx]:.4f} → {direction}")


# ---------------------------------------------------------------------------
# Figure 7 — Density Map Samples
# ---------------------------------------------------------------------------

def fig7_density_samples() -> None:
    print("\nGenerating Figure 7: Density map samples...")

    import sys
    sys.path.insert(0, str(Path("~/agri_foundation").expanduser()))

    try:
        from rgb_fast_dataset import build_rgb_fast_loaders
        from train_rgb_v2 import DensityNet
    except ImportError as e:
        print(f"  WARNING: import failed — {e}")
        return

    model = DensityNet().to(DEVICE)
    ckpt_path = CHECKPOINT_DIR / "rgb_density_v2_best.pt"
    if not ckpt_path.exists():
        print("  WARNING: rgb_density_v2_best.pt not found")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded DensityNet epoch {ckpt['epoch']} MAE={ckpt['metrics']['mae']:.2f}")

    # Load a small test batch
    print("  Loading RGB test data (this takes a few minutes)...")
    loaders = build_rgb_fast_loaders(DATA_ROOT, batch_size=16, num_workers=0)
    imgs, gt_dms, gt_counts = next(iter(loaders["test"]))

    with torch.no_grad():
        pred_dms = model(imgs.to(DEVICE)).cpu()

    pred_counts = pred_dms.sum(dim=(1,2,3))
    gt_counts_dm = gt_dms.sum(dim=(1,2,3))

    # Pick 4 samples: 2 low count, 2 high count
    counts_arr = gt_counts_dm.numpy()
    sorted_idx = np.argsort(counts_arr)
    sample_indices = [
        sorted_idx[2], sorted_idx[4],           # low count
        sorted_idx[-3], sorted_idx[-1],          # high count
    ]

    # Denormalise images for display
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    col_titles = ["Input Image (RGB)", "Ground Truth Density Map", "Predicted Density Map"]

    for row, sample_idx in enumerate(sample_indices):
        img = imgs[sample_idx].permute(1,2,0).numpy()
        img = np.clip(img * STD + MEAN, 0, 1)

        gt_dm = gt_dms[sample_idx, 0].numpy()
        pred_dm = pred_dms[sample_idx, 0].numpy()

        gt_count = gt_counts_dm[sample_idx].item()
        pred_count = pred_counts[sample_idx].item()
        error = abs(pred_count - gt_count)

        # Image
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(img)
        ax0.set_title(f"GT count: {gt_count:.0f}", fontsize=9)
        ax0.axis("off")
        if row == 0:
            ax0.text(0.5, 1.12, col_titles[0], transform=ax0.transAxes,
                     ha="center", fontsize=11, fontweight="bold")

        # GT density
        ax1 = fig.add_subplot(gs[row, 1])
        im1 = ax1.imshow(gt_dm, cmap="hot", interpolation="nearest")
        ax1.set_title(f"Sum: {gt_count:.1f}", fontsize=9)
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        if row == 0:
            ax1.text(0.5, 1.12, col_titles[1], transform=ax1.transAxes,
                     ha="center", fontsize=11, fontweight="bold")

        # Predicted density
        ax2 = fig.add_subplot(gs[row, 2])
        im2 = ax2.imshow(pred_dm, cmap="hot", interpolation="nearest")
        colour = "#52B788" if error < 5 else "#E63946"
        ax2.set_title(f"Pred: {pred_count:.1f} (err: {error:.1f})", fontsize=9,
                      color=colour)
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        if row == 0:
            ax2.text(0.5, 1.12, col_titles[2], transform=ax2.transAxes,
                     ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Panicle Density Map Predictions — DensityNet (RGB Paddy)\n"
        f"Best test MAE: {ckpt['metrics']['mae']:.2f} panicles "
        f"(mean GT count: {ckpt['metrics']['mean_gt']:.1f})",
        fontsize=12, y=1.01
    )

    out = FIG_DIR / "fig7_density_maps.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    sys.path.insert(0, str(Path("~/agri_foundation").expanduser()))

    print(f"Output: {FIG_DIR}\n")
    fig6_shap()
    fig7_density_samples()

    print(f"\nFigures saved:")
    all_figs = sorted(FIG_DIR.glob("fig6*.png")) + sorted(FIG_DIR.glob("fig7*.png"))
    for f in all_figs:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()