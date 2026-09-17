"""
Figure generation for agri_foundation paper.
Generates Figures 2, 3, and 4 from existing log files and results.

Figures produced:
  fig2_fewshot_curve.pdf      -- Few-shot accuracy vs N-shot
  fig3_training_curves.pdf    -- Loss/accuracy curves for HSI and RGB
  fig4_crossmodal_bars.pdf    -- Cross-crop transfer accuracy + AUC

Run: python generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG_DIR = Path("~/agri_foundation/logs").expanduser()
FIG_DIR = Path("~/agri_foundation/figures").expanduser()
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style settings
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# Colour palette — consistent across all figures
C_HSI = "#2E86AB"       # blue — HSI
C_MS = "#A23B72"        # purple — MS
C_RGB = "#F18F01"       # orange — RGB
C_FEW = "#C73E1D"       # red — few-shot
C_FULL = "#3B1F2B"      # dark — full supervision
C_HEALTHY = "#52B788"   # green — healthy/high NDVI
C_STRESSED = "#E63946"  # red — stressed/low NDVI


# ---------------------------------------------------------------------------
# Figure 2 — Few-Shot Accuracy Curve
# ---------------------------------------------------------------------------

def fig2_fewshot() -> None:
    print("Generating Figure 2: Few-shot accuracy curve...")

    # Results from few_shot_eval.py
    n_shots = [1, 5, 10, 20]
    accuracies = [0.9433, 0.9775, 0.9813, 0.9824]
    ci_95 = [0.0044, 0.0011, 0.0009, 0.0008]
    full_sup = 0.9841

    fig, ax = plt.subplots(figsize=(6, 4))

    # Few-shot curve with error bands
    ax.plot(n_shots, [a * 100 for a in accuracies],
            color=C_FEW, marker="o", markersize=7,
            linewidth=2, label="Few-shot (prototypical)", zorder=3)

    ax.fill_between(
        n_shots,
        [(a - ci) * 100 for a, ci in zip(accuracies, ci_95)],
        [(a + ci) * 100 for a, ci in zip(accuracies, ci_95)],
        color=C_FEW, alpha=0.15, label="95% CI"
    )

    # Full supervision baseline
    ax.axhline(full_sup * 100, color=C_FULL, linewidth=1.5,
               linestyle="--", label=f"Full supervision ({full_sup*100:.2f}%)", zorder=2)

    # Annotate key point
    ax.annotate(
        f"5-shot: {accuracies[1]*100:.2f}%\n(−{(full_sup-accuracies[1])*100:.2f}% vs full)",
        xy=(5, accuracies[1] * 100),
        xytext=(8, 96.5),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
        color="gray",
    )

    ax.set_xlabel("Number of labelled support examples per class (N-shot)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Few-Shot Generalisation — Groundnut Water Stress (HSI, 282 bands)")
    ax.set_xticks(n_shots)
    ax.set_xticklabels([f"{n}-shot" for n in n_shots])
    ax.set_ylim(92, 100)
    ax.legend(loc="lower right")

    out = FIG_DIR / "fig2_fewshot_curve.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig2_fewshot_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Training Curves (HSI MLP + RGB Density)
# ---------------------------------------------------------------------------

def fig3_training_curves() -> None:
    print("Generating Figure 3: Training curves...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # --- Left: Groundnut MLP ---
    hsi_log = LOG_DIR / "groundnut_mlp.json"
    if hsi_log.exists():
        with open(hsi_log) as f:
            hsi_data = json.load(f)

        epochs = [d["epoch"] for d in hsi_data]
        tr_loss = [d["train"]["loss"] for d in hsi_data]
        te_loss = [d["test"]["loss"] for d in hsi_data]
        tr_acc = [d["train"]["accuracy"] * 100 for d in hsi_data]
        te_acc = [d["test"]["accuracy"] * 100 for d in hsi_data]

        # Smooth noisy curves with rolling average (window=7)
        def smooth(vals, window=7):
            result = []
            for i in range(len(vals)):
                start = max(0, i - window // 2)
                end = min(len(vals), i + window // 2 + 1)
                result.append(sum(vals[start:end]) / (end - start))
            return result

        tr_loss_s = smooth(tr_loss)
        te_loss_s = smooth(te_loss)
        tr_acc_s = smooth(tr_acc)
        te_acc_s = smooth(te_acc)

        ax = axes[0]
        ax2 = ax.twinx()

        # Raw curves as faint background
        ax.plot(epochs, tr_loss, color=C_HSI, linewidth=0.5, alpha=0.2)
        ax.plot(epochs, te_loss, color=C_HSI, linewidth=0.5, alpha=0.2, linestyle="--")
        ax2.plot(epochs, tr_acc, color=C_FEW, linewidth=0.5, alpha=0.2)
        ax2.plot(epochs, te_acc, color=C_FEW, linewidth=0.5, alpha=0.2, linestyle="--")

        # Smoothed curves as main lines
        l1, = ax.plot(epochs, tr_loss_s, color=C_HSI, linewidth=2.0,
                      alpha=0.9, label="Train loss (smoothed)")
        l2, = ax.plot(epochs, te_loss_s, color=C_HSI, linewidth=2.0,
                      linestyle="--", label="Test loss (smoothed)")
        l3, = ax2.plot(epochs, tr_acc_s, color=C_FEW, linewidth=2.0,
                       alpha=0.9, label="Train acc (smoothed)")
        l4, = ax2.plot(epochs, te_acc_s, color=C_FEW, linewidth=2.0,
                       linestyle="--", label="Test acc (smoothed)")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color=C_HSI)
        ax2.set_ylabel("Accuracy (%)", color=C_FEW)
        ax.tick_params(axis="y", labelcolor=C_HSI)
        ax2.tick_params(axis="y", labelcolor=C_FEW)
        ax.set_title("(a) Groundnut Stress — SpectralMLP (HSI)")
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        lines = [l1, l2, l3, l4]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="center right", fontsize=9)

        # Mark best epoch
        best_epoch = max(range(len(te_acc_s)), key=lambda i: te_acc_s[i]) + 1
        ax2.axvline(best_epoch, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax2.text(best_epoch + 1, min(te_acc_s) + 1,
                 f"Best\nepoch {best_epoch}", fontsize=8, color="gray")
    else:
        axes[0].text(0.5, 0.5, "HSI log not found\n(groundnut_mlp.json)",
                     ha="center", va="center", transform=axes[0].transAxes)
        print("  WARNING: groundnut_mlp.json not found")

    # --- Right: RGB Panicle Counting ---
    rgb_log = LOG_DIR / "rgb_counting_v2.json"
    if rgb_log.exists():
        with open(rgb_log) as f:
            rgb_data = json.load(f)

        epochs = [d["epoch"] for d in rgb_data]
        tr_mae = [d["train"]["mae"] for d in rgb_data]
        te_mae = [d["test"]["mae"] for d in rgb_data]
        te_pred = [d["test"]["mean_pred"] for d in rgb_data]

        ax = axes[1]
        ax.plot(epochs, tr_mae, color=C_RGB, linewidth=1.5,
                alpha=0.7, label="Train MAE")
        ax.plot(epochs, te_mae, color=C_RGB, linewidth=1.5,
                linestyle="--", label="Test MAE")

        # Mark best MAE
        best_idx = min(range(len(te_mae)), key=lambda i: te_mae[i])
        ax.scatter([epochs[best_idx]], [te_mae[best_idx]],
                   color=C_RGB, s=60, zorder=5)
        ax.annotate(f"Best MAE: {te_mae[best_idx]:.2f}",
                    xy=(epochs[best_idx], te_mae[best_idx]),
                    xytext=(epochs[best_idx] + 5, te_mae[best_idx] + 2),
                    fontsize=9, color=C_RGB,
                    arrowprops=dict(arrowstyle="->", color=C_RGB, lw=1.0))

        # GT count reference line
        mean_gt = rgb_data[0]["test"]["mean_gt"]
        ax.axhline(mean_gt, color="gray", linestyle=":",
                   linewidth=1, alpha=0.7, label=f"Mean GT count ({mean_gt:.1f})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Absolute Error (panicles)")
        ax.set_title("(b) Panicle Counting — DensityNet (RGB)")
        ax.legend(loc="upper right", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "RGB log not found\n(rgb_counting_v2.json)",
                     ha="center", va="center", transform=axes[1].transAxes)
        print("  WARNING: rgb_counting_v2.json not found")

    fig.tight_layout(pad=2.0)
    out = FIG_DIR / "fig3_training_curves.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig3_training_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4 — Cross-Modal Transfer Bar Chart
# ---------------------------------------------------------------------------

def fig4_crossmodal() -> None:
    print("Generating Figure 4: Cross-modal transfer bar chart...")

    cross_modal_log = LOG_DIR / "cross_modal_eval.json"
    if cross_modal_log.exists():
        with open(cross_modal_log) as f:
            results = json.load(f)
    else:
        # Use hardcoded results if log not available
        results = [
            {"name": "Within-crop: Maize→Maize", "accuracy": 0.9967, "auc": 0.9989},
            {"name": "Within-crop: Paddy→Paddy", "accuracy": 0.9735, "auc": 0.9819},
            {"name": "Cross-crop: Maize→Paddy",  "accuracy": 0.9790, "auc": 0.9820},
            {"name": "Cross-crop: Paddy→Maize",  "accuracy": 0.9936, "auc": 0.9957},
        ]

    labels = [
        "Maize→Maize\n(within-crop)",
        "Paddy→Paddy\n(within-crop)",
        "Maize→Paddy\n(cross-crop)",
        "Paddy→Maize\n(cross-crop)",
    ]
    accuracies = [r["accuracy"] * 100 for r in results]
    aucs = [r["auc"] * 100 for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width/2, accuracies, width,
                   color=[C_HSI, C_HSI, C_MS, C_MS],
                   alpha=0.85, label="Accuracy (%)", zorder=3)
    bars2 = ax.bar(x + width/2, aucs, width,
                   color=[C_HSI, C_HSI, C_MS, C_MS],
                   alpha=0.45, label="AUC-ROC (%)", zorder=3,
                   edgecolor=[C_HSI, C_HSI, C_MS, C_MS], linewidth=1.2)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5)

    # Shade cross-crop region
    ax.axvspan(1.5, 3.5, alpha=0.04, color=C_MS, label="Cross-crop transfer region")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(94, 101)
    ax.set_ylabel("Performance (%)")
    ax.set_title(
        "Cross-Crop Generalisation via MS Contrastive Encoder (Linear Probe, NDVI Pseudo-Labels)\n"
        "Frozen encoder trained with no labels — evaluated across crop types"
    )

    within_patch = mpatches.Patch(color=C_HSI, alpha=0.85, label="Within-crop")
    cross_patch = mpatches.Patch(color=C_MS, alpha=0.85, label="Cross-crop transfer")
    acc_patch = mpatches.Patch(color="gray", alpha=0.85, label="Accuracy")
    auc_patch = mpatches.Patch(color="gray", alpha=0.45, label="AUC-ROC")
    ax.legend(handles=[within_patch, cross_patch, acc_patch, auc_patch],
              loc="lower right", fontsize=9, ncol=2)

    fig.tight_layout()
    out = FIG_DIR / "fig4_crossmodal_bars.pdf"
    fig.savefig(out)

    # Also save PNG for quick preview
    fig.savefig(FIG_DIR / "fig4_crossmodal_bars.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5 — t-SNE of MS Embeddings
# ---------------------------------------------------------------------------

def fig5_tsne() -> None:
    print("Generating Figure 5: t-SNE of MS embeddings...")

    try:
        from sklearn.manifold import TSNE
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ms_dataset import build_ms_loaders

        CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
        DATA_ROOT = Path("~/agri_foundation/data").expanduser()
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Rebuild encoder
        class MSEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                def cb(ic, oc, s):
                    return nn.Sequential(
                        nn.Conv2d(ic, oc, 3, stride=s, padding=1, bias=False),
                        nn.BatchNorm2d(oc), nn.GELU(),
                        nn.Conv2d(oc, oc, 3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(oc), nn.GELU(),
                    )
                self.backbone = nn.Sequential(cb(5,32,2), cb(32,64,2), cb(64,128,2), cb(128,256,2))
                self.pool = nn.AdaptiveAvgPool2d(1)
            def forward(self, x):
                return F.normalize(self.pool(self.backbone(x)).flatten(1), dim=-1)

        encoder = MSEncoder().to(DEVICE)
        ckpt = torch.load(CHECKPOINT_DIR / "ms_encoder_best.pt", map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder_state"])
        encoder.eval()

        # Load MS arrays
        ms_dir = DATA_ROOT / "processed" / "ms"
        maize = np.load(list(ms_dir.glob("maize/**/ms_stacked.npy"))[0])
        paddy = np.load(list(ms_dir.glob("paddy/**/ms_stacked.npy"))[0])

        MS_MEAN = np.array([0.2541, 0.2613, 0.2608, 0.3284, 0.2856], dtype=np.float32)
        MS_STD = np.array([0.1356, 0.1386, 0.1438, 0.1477, 0.1460], dtype=np.float32)

        def get_tiles(imgs, n_per_img=5, seed=0):
            rng = np.random.default_rng(seed)
            N, H, W, C = imgs.shape
            tiles = []
            ndvi_vals = []
            for img in imgs:
                for _ in range(n_per_img):
                    t = rng.integers(0, H - 64)
                    l = rng.integers(0, W - 64)
                    tile = img[t:t+64, l:l+64, :].transpose(2, 0, 1).astype(np.float32)
                    ndvi = (tile[4] - tile[2]) / (tile[4] + tile[2] + 1e-8)
                    ndvi_vals.append(ndvi.mean())
                    tile = (tile - MS_MEAN[:, None, None]) / (MS_STD[:, None, None] + 1e-6)
                    tiles.append(tile)
            return np.stack(tiles), np.array(ndvi_vals)

        print("  Extracting tiles for t-SNE (this takes a few minutes)...")
        maize_tiles, maize_ndvi = get_tiles(maize, n_per_img=5)
        paddy_tiles, paddy_ndvi = get_tiles(paddy, n_per_img=5)

        all_tiles = np.concatenate([maize_tiles, paddy_tiles], axis=0)
        all_ndvi = np.concatenate([maize_ndvi, paddy_ndvi], axis=0)
        crop_labels = np.array(
            ["Maize"] * len(maize_tiles) + ["Paddy"] * len(paddy_tiles)
        )

        # Extract embeddings in batches
        all_emb = []
        with torch.no_grad():
            for i in range(0, len(all_tiles), 64):
                batch = torch.from_numpy(all_tiles[i:i+64]).to(DEVICE)
                all_emb.append(encoder(batch).cpu().numpy())
        all_emb = np.concatenate(all_emb, axis=0)

        # Run t-SNE
        print("  Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
        emb_2d = tsne.fit_transform(all_emb)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: coloured by crop type
        for crop, color in [("Maize", C_RGB), ("Paddy", C_HSI)]:
            mask = crop_labels == crop
            axes[0].scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                c=color, alpha=0.4, s=8, label=crop, rasterized=True
            )
        axes[0].set_title("(a) t-SNE coloured by crop type")
        axes[0].legend(markerscale=3)
        axes[0].set_xlabel("t-SNE dim 1")
        axes[0].set_ylabel("t-SNE dim 2")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Right: coloured by NDVI value
        sc = axes[1].scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=all_ndvi, cmap="RdYlGn", alpha=0.5, s=8,
            vmin=0.1, vmax=0.7, rasterized=True
        )
        plt.colorbar(sc, ax=axes[1], label="Mean NDVI (tile)")
        axes[1].set_title("(b) t-SNE coloured by NDVI (stress proxy)")
        axes[1].set_xlabel("t-SNE dim 1")
        axes[1].set_ylabel("t-SNE dim 2")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        fig.suptitle(
            "MS Encoder Embedding Space (t-SNE)\n"
            "Frozen encoder trained with contrastive learning (no labels)",
            fontsize=11
        )
        fig.tight_layout()
        out = FIG_DIR / "fig5_tsne.pdf"
        fig.savefig(out)
        fig.savefig(FIG_DIR / "fig5_tsne.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    except Exception as e:
        print(f"  WARNING: t-SNE figure failed — {e}")
        print("  Run manually after checking imports.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Output directory: {FIG_DIR}\n")
    fig2_fewshot()
    fig3_training_curves()
    fig4_crossmodal()
    fig5_tsne()
    print(f"\nAll figures saved to {FIG_DIR}")
    print("Files:")
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()