import pickle, numpy as np, json
from pathlib import Path

base = Path('data/umn_wheat')
DATA_ROOT = Path('data')
LOG_DIR = Path('logs')
CHECKPOINT_DIR = Path('checkpoints')

# Load yield data
yield_path = base / 'yield_data/Yield_data/yield_data.pickle'
with open(yield_path, 'rb') as f:
    yield_raw = pickle.load(f)

# Build lookup: "C3_10702" -> yield_value
yield_dict = {}
for field_name, df in yield_raw.items():
    median_yield = df['Yield'].median()
    for _, row in df.iterrows():
        key = f"{field_name}_{int(row['plot_ID'])}"
        yield_dict[key] = float(row['Yield'])

print(f"Yield lookup built: {len(yield_dict)} plots")
print(f"Sample: {list(yield_dict.items())[:3]}")

# Load all wheat plots and extract patches
import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 11
PATCHES_PER_PLOT = 15
WHEAT_BANDS = 190
GROUNDNUT_BANDS = 282

rng = np.random.default_rng(42)
all_patches = []
all_yields = []

for field in ['C3_numpy', 'C4_numpy', 'C9_numpy']:
    field_name = field.split('_')[0]  # C3, C4, C9
    field_dir = base / field / field  # nested folder
    plot_files = sorted(field_dir.glob('*.npy'))
    print(f"{field}: {len(plot_files)} plots")

    for plot_path in plot_files:
        plot_id = plot_path.stem  # e.g. C3_10702
        if plot_id not in yield_dict:
            continue

        cube = np.load(plot_path).astype(np.float32)  # (H, W, 190)
        H, W, B = cube.shape
        if H < PATCH_SIZE or W < PATCH_SIZE or B != WHEAT_BANDS:
            continue

        for _ in range(PATCHES_PER_PLOT):
            top = rng.integers(0, H - PATCH_SIZE)
            left = rng.integers(0, W - PATCH_SIZE)
            patch = cube[top:top+PATCH_SIZE, left:left+PATCH_SIZE, :]
            spectrum = patch.mean(axis=(0,1))
            all_patches.append(spectrum)
            all_yields.append(yield_dict[plot_id])

X = np.array(all_patches, dtype=np.float32)
yields = np.array(all_yields, dtype=np.float32)

# Per-field median threshold for balanced labels
all_labels = np.zeros(len(yields), dtype=np.int64)
median_yield = np.median(yields)
all_labels = (yields > median_yield).astype(np.int64)

print(f"\nTotal patches: {len(X)}")
print(f"Median yield threshold: {median_yield:.1f}g")
print(f"Above median (healthy): {all_labels.sum()}")
print(f"Below median (stressed): {(all_labels==0).sum()}")
print(f"X range: [{X.min():.4f}, {X.max():.4f}]")

# Save processed wheat dataset
np.save('data/umn_wheat/wheat_X.npy', X)
np.save('data/umn_wheat/wheat_y.npy', all_labels)
np.save('data/umn_wheat/wheat_yields.npy', yields)
print("\nSaved: wheat_X.npy, wheat_y.npy, wheat_yields.npy")

# Now run few-shot evaluation
class SpectralMLP(nn.Module):
    def __init__(self, num_bands=282, hidden_dims=(256,64), num_classes=2, dropout=0.4):
        super().__init__()
        dims = [num_bands] + list(hidden_dims) + [num_classes]
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i],dims[i+1]),
                      nn.BatchNorm1d(dims[i+1]), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)
    def get_embedding(self, x):
        out = x
        for layer in list(self.net.children())[:-1]:
            out = layer(out)
        return F.normalize(out, dim=-1)

encoder = SpectralMLP().to(DEVICE)
ckpt = torch.load('checkpoints/groundnut_mlp_best.pt', map_location=DEVICE)
encoder.load_state_dict(ckpt['model_state'])
for p in encoder.parameters():
    p.requires_grad = False
encoder.eval()
print(f"\nLoaded groundnut encoder epoch={ckpt['epoch']}")

# Spectral adapter: 190 -> 282 via zero padding
def adapt_zero_pad(x_np):
    x = torch.from_numpy(x_np).to(DEVICE)
    pad = torch.zeros(len(x), GROUNDNUT_BANDS - WHEAT_BANDS, device=DEVICE)
    return torch.cat([x, pad], dim=1)

# Extract embeddings
BATCH = 512
all_emb = []
with torch.no_grad():
    for i in range(0, len(X), BATCH):
        batch = adapt_zero_pad(X[i:i+BATCH])
        emb = encoder.get_embedding(batch)
        all_emb.append(emb.cpu().numpy())
wheat_emb = np.concatenate(all_emb)
print(f"Embeddings: {wheat_emb.shape} std={wheat_emb.std(axis=0).mean():.4f}")

# Few-shot episodes
def proto_episode(emb, labels, n_shot, n_query, rng):
    sup_e, sup_l, qry_e, qry_l = [], [], [], []
    for cls in range(2):
        idx = np.where(labels==cls)[0]
        if len(idx) < n_shot + n_query: return float('nan')
        s = rng.choice(idx, n_shot+n_query, replace=False)
        sup_e.append(emb[s[:n_shot]]); sup_l += [cls]*n_shot
        qry_e.append(emb[s[n_shot:]]); qry_l += [cls]*n_query
    sup_e = np.concatenate(sup_e)
    qry_e = np.concatenate(qry_e)
    qry_l = np.array(qry_l)
    protos = np.stack([sup_e[np.array(sup_l)==c].mean(0) for c in range(2)])
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True)+1e-8)
    preds = (qry_e @ protos.T).argmax(axis=1)
    return float((preds==qry_l).mean())

rng2 = np.random.default_rng(42)
print(f"\n{'N-shot':>8} {'Mean Acc':>10} {'95% CI':>8}")
print("-"*30)
results = {}
for n_shot in [1, 5, 10, 20]:
    accs = [proto_episode(wheat_emb, all_labels, n_shot, 30, rng2)
            for _ in range(1000)]
    accs = [a for a in accs if not np.isnan(a)]
    mean = np.mean(accs)
    ci = 1.96 * np.std(accs) / len(accs)**0.5
    results[n_shot] = {'mean': float(mean), 'ci_95': float(ci)}
    print(f"{n_shot:>6}-shot {mean*100:>10.2f}% {ci*100:>7.2f}%")

print(f"\nGroundnut within-domain (reference):")
print(f"  1-shot: 94.33% | 5-shot: 97.75% | 20-shot: 98.24%")

with open('logs/wheat_transfer_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to logs/wheat_transfer_eval.json")
