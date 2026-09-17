"""
Gaussian Process Uncertainty Head for Crop Stress Monitoring.

Trains a sparse GP classifier on top of frozen foundation model embeddings
(HyperSL for HSI, Prithvi for MS) to provide calibrated uncertainty
alongside stress predictions.

GP provides:
  - P(stressed | embedding) — stress probability
  - σ(stressed | embedding) — predictive uncertainty
  - Calibration metrics (ECE, reliability diagram)

Two GP models:
  1. HSI-GP: on HyperSL embeddings from groundnut data
  2. MS-GP: on Prithvi embeddings from maize/paddy data

Run: CUDA_VISIBLE_DEVICES=1 python gp_uncertainty_head.py
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
from sklearn.preprocessing import StandardScaler

DATA_ROOT = Path("~/agri_foundation/data").expanduser()
PRITHVI_DIR = Path("~/agri_foundation/models/prithvi").expanduser()
CHECKPOINT_DIR = Path("~/agri_foundation/checkpoints").expanduser()
LOG_DIR = Path("~/agri_foundation/logs").expanduser()
sys.path.insert(0, str(PRITHVI_DIR))
sys.path.insert(0, str(Path("~/agri_foundation/models/hypersl_code").expanduser()))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GP config
N_INDUCING = 128       # sparse GP inducing points
GP_EPOCHS = 100
GP_LR = 0.01
BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Install gpytorch if needed
# ---------------------------------------------------------------------------

try:
    import gpytorch
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "gpytorch", "-q"])
    import gpytorch


# ---------------------------------------------------------------------------
# Sparse GP model (SVGP — Stochastic Variational GP)
# ---------------------------------------------------------------------------

class SVGPClassifier(gpytorch.models.ApproximateGP):
    """
    Sparse Variational Gaussian Process for binary stress classification.
    Uses inducing points for scalability to thousands of embeddings.
    """

    def __init__(
        self,
        inducing_points: Tensor,
    ) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ---------------------------------------------------------------------------
# GP training and evaluation
# ---------------------------------------------------------------------------

def train_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    name: str = "",
) -> tuple[SVGPClassifier, gpytorch.likelihoods.BernoulliLikelihood]:
    """Train sparse GP classifier on embeddings."""
    # Standardise
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    X_t = torch.from_numpy(X_s).float().to(DEVICE)
    y_t = torch.from_numpy(y_train).float().to(DEVICE)

    # Inducing points — sample from training data
    n_inducing = min(N_INDUCING, len(X_train))
    idx = np.random.default_rng(42).choice(len(X_train), n_inducing, replace=False)
    inducing_points = X_t[idx].clone()

    # GP model
    model = SVGPClassifier(inducing_points).to(DEVICE)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(DEVICE)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=GP_LR)

    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=len(y_train)
    )

    print(f"\nTraining GP ({name}): {len(X_train)} train, "
          f"{n_inducing} inducing points...")

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(1, GP_EPOCHS + 1):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}: ELBO = {-total_loss/len(loader):.4f}")

    return model, likelihood, scaler


@torch.no_grad()
def evaluate_gp(
    model: SVGPClassifier,
    likelihood: gpytorch.likelihoods.BernoulliLikelihood,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "",
) -> dict:
    """Evaluate GP with uncertainty quantification."""
    model.eval()
    likelihood.eval()

    X_s = scaler.transform(X_test)
    X_t = torch.from_numpy(X_s).float().to(DEVICE)

    # Predict in batches
    all_mean, all_var = [], []
    for start in range(0, len(X_t), BATCH_SIZE):
        batch = X_t[start:start+BATCH_SIZE]
        with gpytorch.settings.fast_pred_var():
            pred = likelihood(model(batch))
        all_mean.append(pred.mean.cpu().numpy())
        all_var.append(pred.variance.cpu().numpy())

    probs = np.concatenate(all_mean)      # P(stressed)
    variances = np.concatenate(all_var)   # GP predictive variance
    preds = (probs > 0.5).astype(int)

    # Accuracy
    acc = (preds == y_test).mean()

    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    calibration_data = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
        if mask.sum() > 0:
            bin_conf = probs[mask].mean()
            bin_acc = y_test[mask].mean()
            bin_size = mask.sum() / len(y_test)
            ece += bin_size * abs(bin_conf - bin_acc)
            calibration_data.append({
                "bin_centre": float((bin_edges[i] + bin_edges[i+1]) / 2),
                "confidence": float(bin_conf),
                "accuracy": float(bin_acc),
                "fraction": float(bin_size),
            })

    # Uncertainty statistics
    mean_uncertainty = float(variances.mean())
    high_conf_mask = (probs > 0.8) | (probs < 0.2)
    high_conf_acc = float((preds[high_conf_mask] == y_test[high_conf_mask]).mean()) \
        if high_conf_mask.sum() > 0 else float("nan")

    print(f"\n--- GP Results: {name} ---")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  ECE:               {ece:.4f} (lower = better calibrated)")
    print(f"  Mean uncertainty:  {mean_uncertainty:.4f}")
    print(f"  High-conf samples: {high_conf_mask.sum()}/{len(y_test)} "
          f"(acc={high_conf_acc:.4f})")

    return {
        "name": name,
        "accuracy": float(acc),
        "ece": float(ece),
        "mean_uncertainty": mean_uncertainty,
        "high_conf_accuracy": high_conf_acc,
        "high_conf_fraction": float(high_conf_mask.mean()),
        "calibration_curve": calibration_data,
    }


# ---------------------------------------------------------------------------
# Extract HyperSL embeddings
# ---------------------------------------------------------------------------

def get_hypersl_embeddings(
    X: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """Extract frozen HyperSL embeddings from flat HSI spectra."""
    from engine.model import SpectralSharedEncoder

    model = SpectralSharedEncoder(
        embedding_dim=256, num_heads=8,
        decoder_depth=4, encoder_depth=8
    )
    ckpt = torch.load(
        "models/hypersl_weights/10_base_mask95_checkpoint.pt",
        map_location="cpu", weights_only=False
    )
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    print(f"HyperSL loaded for GP embedding extraction")
    all_emb = []
    X_t = torch.from_numpy(X.astype(np.float32))
    wave = torch.from_numpy(wavelengths.astype(np.float32))

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = X_t[start:start+BATCH_SIZE].to(DEVICE)
            wave_b = wave.unsqueeze(0).expand(len(batch), -1).to(DEVICE)
            x_in = batch.unsqueeze(1)  # (B, 1, 282)
            z, _, _, _, _, _ = model.encoder_forward(x_in, wave_b, mask_ratio=0.0)
            if z.ndim == 3:
                z = z.squeeze(1)
            all_emb.append(F.normalize(z, dim=-1).cpu().numpy())

    return np.concatenate(all_emb)


# ---------------------------------------------------------------------------
# Extract Prithvi embeddings
# ---------------------------------------------------------------------------

def get_prithvi_embeddings(tiles: np.ndarray) -> np.ndarray:
    """Extract frozen Prithvi-EO-2.0 embeddings from MS tiles."""
    from prithvi_mae import PrithviMAE

    PRITHVI_MEAN_T = torch.tensor([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0])
    PRITHVI_STD_T  = torch.tensor([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0])

    cfg = json.load(open(PRITHVI_DIR / "config.json"))["pretrained_cfg"]
    mae = PrithviMAE(
        img_size=cfg["img_size"], num_frames=cfg["num_frames"],
        patch_size=cfg["patch_size"], in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"], depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        decoder_embed_dim=cfg["decoder_embed_dim"],
        decoder_depth=cfg["decoder_depth"],
        decoder_num_heads=cfg["decoder_num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
    )
    ckpt = torch.load(
        PRITHVI_DIR / "Prithvi_EO_V2_300M.pt",
        map_location="cpu", weights_only=False
    )
    mae.load_state_dict(ckpt.get("model", ckpt), strict=False)
    enc = mae.encoder
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval().to(DEVICE)
    print("Prithvi loaded for GP embedding extraction")

    all_emb = []
    with torch.no_grad():
        for start in range(0, len(tiles), 8):
            batch = torch.from_numpy(
                tiles[start:start+8].transpose(0,3,1,2)
            ).float().to(DEVICE)
            B, C, H, W = batch.shape
            scale = torch.tensor(
                [1087/0.2541, 1342/0.2613, 1433/0.2608,
                 2734/0.3284, 1958/0.2856],
                device=DEVICE
            )
            tiles_dn = batch * scale[None, :, None, None]
            pad = torch.zeros(B, 1, H, W, device=DEVICE)
            tiles_6 = torch.cat([tiles_dn, pad], dim=1)
            tiles_r = F.interpolate(tiles_6, size=(224, 224),
                                    mode="bilinear", align_corners=False)
            mean = PRITHVI_MEAN_T.to(DEVICE)[None, :, None, None]
            std  = PRITHVI_STD_T.to(DEVICE)[None, :, None, None]
            tiles_n = (tiles_r - mean) / (std + 1e-8)
            x = tiles_n.unsqueeze(2).repeat(1, 1, 4, 1, 1)
            feat = enc.forward_features(x)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            if feat.ndim == 3:
                feat = feat.mean(dim=1)
            elif feat.shape[1] == 1:
                feat = feat.squeeze(1)
            all_emb.append(F.normalize(feat, dim=-1).cpu().numpy())

    return np.concatenate(all_emb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # ----------------------------------------------------------------
    # GP 1 — HSI branch: HyperSL embeddings on groundnut stress
    # ----------------------------------------------------------------
    print("\n" + "="*55)
    print("GP 1: HSI Branch (HyperSL + groundnut)")
    print("="*55)

    proc = DATA_ROOT / "processed" / "hsi"
    gn_flat = np.load(proc / "gn_X_flat.npy").astype(np.float32)
    gn_y = np.load(proc / "gn_y.npy").astype(np.int64)
    test_idx = np.load(proc / "gn_test_idx.npy")
    train_mask = np.ones(len(gn_flat), dtype=bool)
    train_mask[test_idx] = False

    X_train_hsi = gn_flat[train_mask]
    y_train_hsi = gn_y[train_mask]
    X_test_hsi = gn_flat[test_idx]
    y_test_hsi = gn_y[test_idx]

    wavelengths = np.linspace(400, 1000, 282)

    print("Extracting HyperSL embeddings...")
    train_emb_hsi = get_hypersl_embeddings(X_train_hsi, wavelengths)
    test_emb_hsi  = get_hypersl_embeddings(X_test_hsi, wavelengths)
    print(f"Train emb: {train_emb_hsi.shape} | Test emb: {test_emb_hsi.shape}")

    # Train GP
    model_hsi, lik_hsi, scaler_hsi = train_gp(
        train_emb_hsi, y_train_hsi, name="HSI-GP (HyperSL)"
    )

    # Evaluate
    res = evaluate_gp(
        model_hsi, lik_hsi, scaler_hsi,
        test_emb_hsi, y_test_hsi,
        name="HSI-GP: HyperSL → Groundnut Stress"
    )
    all_results.append(res)

    # Save HSI GP checkpoint
    torch.save({
        "model": model_hsi.state_dict(),
        "likelihood": lik_hsi.state_dict(),
    }, CHECKPOINT_DIR / "gp_hsi_best.pt")

    # ----------------------------------------------------------------
    # GP 2 — MS branch: Prithvi embeddings on maize/paddy
    # ----------------------------------------------------------------
    print("\n" + "="*55)
    print("GP 2: MS Branch (Prithvi + maize/paddy)")
    print("="*55)

    ms_dir = DATA_ROOT / "processed" / "ms"
    MS_MEAN_A = np.array([0.2541, 0.2613, 0.2608, 0.3284, 0.2856], np.float32)
    MS_STD_A  = np.array([0.1356, 0.1386, 0.1438, 0.1477, 0.1460], np.float32)

    def extract_ms_tiles(arr, tiles_per_image=8, seed=42):
        rng = np.random.default_rng(seed)
        N, H, W, C = arr.shape
        tiles = []
        for img in arr:
            for _ in range(tiles_per_image):
                if H < 64 or W < 64:
                    continue
                t = rng.integers(0, H-64)
                l = rng.integers(0, W-64)
                tile = img[t:t+64, l:l+64, :].astype(np.float32)
                tile = (tile - MS_MEAN_A[:C]) / (MS_STD_A[:C] + 1e-6)
                tiles.append(tile)
        return np.stack(tiles)

    maize_arr = np.load(list(ms_dir.glob("maize/**/ms_stacked.npy"))[0])
    paddy_arr = np.load(list(ms_dir.glob("paddy/**/ms_stacked.npy"))[0])
    ms_tiles = np.concatenate([
        extract_ms_tiles(maize_arr),
        extract_ms_tiles(paddy_arr),
    ], axis=0)

    # NDVI labels
    nir = ms_tiles[:, :, :, 4].mean(axis=(1,2))
    red = ms_tiles[:, :, :, 2].mean(axis=(1,2))
    ndvi = (nir - red) / (nir + red + 1e-8)
    ms_labels = (ndvi > np.median(ndvi)).astype(np.int64)

    # Train/test split
    rng = np.random.default_rng(42)
    n = len(ms_tiles)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    X_train_ms = ms_tiles[idx[:split]]
    y_train_ms = ms_labels[idx[:split]]
    X_test_ms  = ms_tiles[idx[split:]]
    y_test_ms  = ms_labels[idx[split:]]

    print("Extracting Prithvi embeddings...")
    train_emb_ms = get_prithvi_embeddings(X_train_ms)
    test_emb_ms  = get_prithvi_embeddings(X_test_ms)
    print(f"Train emb: {train_emb_ms.shape} | Test emb: {test_emb_ms.shape}")

    # Train GP
    model_ms, lik_ms, scaler_ms = train_gp(
        train_emb_ms, y_train_ms, name="MS-GP (Prithvi)"
    )

    # Evaluate
    res = evaluate_gp(
        model_ms, lik_ms, scaler_ms,
        test_emb_ms, y_test_ms,
        name="MS-GP: Prithvi → Maize/Paddy NDVI"
    )
    all_results.append(res)

    torch.save({
        "model": model_ms.state_dict(),
        "likelihood": lik_ms.state_dict(),
    }, CHECKPOINT_DIR / "gp_ms_best.pt")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("GP UNCERTAINTY HEAD — SUMMARY")
    print("="*60)
    print(f"{'Model':<40} {'Acc':>6} {'ECE':>6} {'Unc':>6}")
    print("-"*56)
    for r in all_results:
        print(f"  {r['name']:<38} "
              f"{r['accuracy']:.4f} "
              f"{r['ece']:.4f} "
              f"{r['mean_uncertainty']:.4f}")

    print(f"\nECE interpretation: <0.05 = well calibrated, "
          f">0.10 = poorly calibrated")

    with open(LOG_DIR / "gp_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {LOG_DIR / 'gp_results.json'}")


if __name__ == "__main__":
    main()