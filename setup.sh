#!/bin/bash
# =============================================================================
# AGRI FOUNDATION — Environment Setup Script
# Run on: 172.21.1.158 (tyrone-hpc) | User: sharon
# Usage: bash setup_env.sh 2>&1 | tee setup_env_log.txt
# =============================================================================

echo "=============================================="
echo " AGRI FOUNDATION — Environment Setup"
echo " $(date)"
echo "=============================================="

# ------------------------------------------------------------------------------
# STEP 1 — Check CUDA and GPU availability
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 1: SYSTEM CHECK"
echo "----------------------------------------------"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Python: $(python3 --version 2>/dev/null || echo 'not found')"
echo "Conda: $(conda --version 2>/dev/null || echo 'not found')"
echo ""
echo "--- GPU Info ---"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
echo ""
echo "--- CUDA Version ---"
nvcc --version 2>/dev/null || echo "nvcc not found — will infer from nvidia-smi"
echo ""
echo "--- Disk space ---"
df -h ~/agri_foundation/

# ------------------------------------------------------------------------------
# STEP 2 — Create conda environment
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 2: CREATING CONDA ENVIRONMENT (agri-foundation)"
echo "----------------------------------------------"

# Remove existing env if present
conda env remove -n agri-foundation -y 2>/dev/null && echo "Removed existing agri-foundation env" || echo "No existing env found"

# Create fresh environment
conda create -n agri-foundation python=3.10 -y
echo "✓ Conda environment created: agri-foundation (Python 3.10)"

# ------------------------------------------------------------------------------
# STEP 3 — Activate and install core dependencies
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 3: INSTALLING DEPENDENCIES"
echo "----------------------------------------------"

# Use conda run to execute in the environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate agri-foundation

echo ""
echo "[3.1] Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
echo "✓ PyTorch installed"

echo ""
echo "[3.2] Installing geospatial and remote sensing libraries..."
pip install -q \
    rasterio \
    spectral \
    pyproj \
    shapely \
    fiona \
    geopandas \
    earthpy \
    h5py \
    netCDF4
echo "✓ Geospatial libraries installed"

echo ""
echo "[3.3] Installing vision and ML libraries..."
pip install -q \
    einops \
    timm \
    scikit-learn \
    scikit-image \
    opencv-python-headless \
    Pillow \
    albumentations \
    kornia
echo "✓ Vision / ML libraries installed"

echo ""
echo "[3.4] Installing HuggingFace ecosystem..."
pip install -q \
    transformers \
    huggingface_hub \
    datasets \
    accelerate \
    peft
echo "✓ HuggingFace installed"

echo ""
echo "[3.5] Installing SAM2 (Segment Anything Model 2)..."
pip install -q git+https://github.com/facebookresearch/sam2.git
echo "✓ SAM2 installed"

echo ""
echo "[3.6] Installing GroundingDINO..."
pip install -q \
    groundingdino-py 2>/dev/null || \
pip install -q \
    git+https://github.com/IDEA-Research/GroundingDINO.git 2>/dev/null || \
echo "  GroundingDINO: manual install may be needed — see note below"

echo ""
echo "[3.7] Installing data science and analysis libraries..."
pip install -q \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    plotly \
    scipy \
    statsmodels
echo "✓ Data science libraries installed"

echo ""
echo "[3.8] Installing explainability libraries..."
pip install -q \
    shap \
    captum \
    lime
echo "✓ Explainability libraries installed"

echo ""
echo "[3.9] Installing ERA5 / meteorological libraries..."
pip install -q \
    cdsapi \
    earthkit-data \
    xarray \
    cfgrib \
    eccodes \
    metpy
echo "✓ Meteorological libraries installed"

echo ""
echo "[3.10] Installing TurboVec (vector retrieval)..."
pip install -q turbovec 2>/dev/null || echo "  TurboVec: not available on PyPI — using FAISS as fallback"
pip install -q faiss-gpu 2>/dev/null || pip install -q faiss-cpu
echo "✓ Vector retrieval installed"

echo ""
echo "[3.11] Installing utility libraries..."
pip install -q \
    tqdm \
    wandb \
    tensorboard \
    omegaconf \
    hydra-core \
    rich \
    ipython \
    jupyter \
    ipykernel
echo "✓ Utility libraries installed"

# ------------------------------------------------------------------------------
# STEP 4 — Download model weights
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 4: DOWNLOADING MODEL WEIGHTS"
echo "----------------------------------------------"

WEIGHTS_DIR=~/agri_foundation/weights
mkdir -p $WEIGHTS_DIR

echo "[4.1] Downloading SAM2 weights (ViT-H)..."
mkdir -p $WEIGHTS_DIR/sam2
wget -q --show-progress \
    https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_large.pt \
    -O $WEIGHTS_DIR/sam2/sam2.1_hiera_large.pt 2>/dev/null || \
wget -q --show-progress \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O $WEIGHTS_DIR/sam2/sam_vit_h_4b8939.pth
echo "✓ SAM2 weights downloaded"

echo ""
echo "[4.2] Downloading Prithvi-EO-2.0 weights from HuggingFace..."
mkdir -p $WEIGHTS_DIR/prithvi
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ibm-nasa-geospatial/Prithvi-EO-2.0-300M',
    local_dir='$WEIGHTS_DIR/prithvi',
    ignore_patterns=['*.msgpack', '*.h5']
)
print('✓ Prithvi-EO-2.0 weights downloaded')
" 2>/dev/null || echo "  Prithvi: HuggingFace download — may need token for gated models"

echo ""
echo "[4.3] SpectralGPT weights..."
mkdir -p $WEIGHTS_DIR/spectralgpt
python3 -c "
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(
        repo_id='danfenghong/SpectralGPT',
        filename='SpectralGPT.pth',
        local_dir='$WEIGHTS_DIR/spectralgpt'
    )
    print('✓ SpectralGPT weights downloaded')
except Exception as e:
    print(f'  SpectralGPT: {e}')
    print('  Manual download may be needed from: https://huggingface.co/danfenghong/SpectralGPT')
" 2>/dev/null || echo "  SpectralGPT: check HuggingFace availability"

echo ""
echo "[4.4] GroundingDINO weights..."
mkdir -p $WEIGHTS_DIR/groundingdino
wget -q --show-progress \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -O $WEIGHTS_DIR/groundingdino/groundingdino_swint_ogc.pth 2>/dev/null
echo "✓ GroundingDINO weights downloaded"

# ------------------------------------------------------------------------------
# STEP 5 — Register Jupyter kernel
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 5: REGISTERING JUPYTER KERNEL"
echo "----------------------------------------------"
python3 -m ipykernel install --user --name agri-foundation --display-name "Agri Foundation (Python 3.10)"
echo "✓ Jupyter kernel registered"

# ------------------------------------------------------------------------------
# STEP 6 — Verify installation
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 6: VERIFICATION"
echo "----------------------------------------------"

python3 << 'PYEOF'
import sys
print(f"Python: {sys.version}")

results = []

# PyTorch + CUDA
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    results.append(f"  ✓ PyTorch {torch.__version__} | CUDA: {cuda_ok} | GPUs: {gpu_count}")
    if cuda_ok:
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            results.append(f"    GPU {i}: {name} ({mem:.1f} GB)")
except ImportError as e:
    results.append(f"  ✗ PyTorch: {e}")

# Core libraries
libs = [
    ('rasterio', 'rasterio'),
    ('spectral', 'spectral'),
    ('transformers', 'transformers'),
    ('einops', 'einops'),
    ('timm', 'timm'),
    ('sklearn', 'scikit-learn'),
    ('shap', 'shap'),
    ('cdsapi', 'cdsapi'),
    ('xarray', 'xarray'),
    ('faiss', 'faiss-gpu/cpu'),
    ('huggingface_hub', 'huggingface_hub'),
    ('geopandas', 'geopandas'),
    ('albumentations', 'albumentations'),
]

for lib, name in libs:
    try:
        mod = __import__(lib)
        ver = getattr(mod, '__version__', 'ok')
        results.append(f"  ✓ {name}: {ver}")
    except ImportError:
        results.append(f"  ✗ {name}: NOT INSTALLED")

# SAM2
try:
    import sam2
    results.append(f"  ✓ SAM2: installed")
except ImportError:
    try:
        from segment_anything import sam_model_registry
        results.append(f"  ✓ SAM (v1): installed")
    except ImportError:
        results.append(f"  ✗ SAM2: NOT INSTALLED")

# TurboVec
try:
    import turbovec
    results.append(f"  ✓ TurboVec: installed")
except ImportError:
    results.append(f"  ~ TurboVec: not installed (using FAISS fallback)")

print("\n--- Package Status ---")
for r in results:
    print(r)

# Weight files
import os
weights = {
    'SAM2': os.path.expanduser('~/agri_foundation/weights/sam2/'),
    'Prithvi': os.path.expanduser('~/agri_foundation/weights/prithvi/'),
    'SpectralGPT': os.path.expanduser('~/agri_foundation/weights/spectralgpt/'),
    'GroundingDINO': os.path.expanduser('~/agri_foundation/weights/groundingdino/'),
}
print("\n--- Weight Files ---")
for name, path in weights.items():
    if os.path.exists(path):
        files = os.listdir(path)
        total_size = sum(os.path.getsize(os.path.join(path, f))
                        for f in files if os.path.isfile(os.path.join(path, f)))
        print(f"  ✓ {name}: {len(files)} file(s), {total_size/1e9:.2f} GB")
    else:
        print(f"  ✗ {name}: directory not found")

print("\n=== SETUP VERIFICATION COMPLETE ===")
PYEOF

# ------------------------------------------------------------------------------
# STEP 7 — Create .env config file for the project
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 7: CREATING PROJECT CONFIG"
echo "----------------------------------------------"

cat > ~/agri_foundation/.env << 'EOF'
# AGRI FOUNDATION — Project Config
BASE_DIR=~/agri_foundation
DATA_DIR=~/agri_foundation/data
WEIGHTS_DIR=~/agri_foundation/weights
RESULTS_DIR=~/agri_foundation/results

# Data paths
RGB_TRAIN=~/agri_foundation/data/rgb_paddy/train
RGB_TEST=~/agri_foundation/data/rgb_paddy/test
MS_RAW=~/agri_foundation/data/multispectral/raw
HSI_CROP=~/agri_foundation/data/hyperspectral/crop_variety
HSI_GROUNDNUT=~/agri_foundation/data/hyperspectral/groundnut_stress
HSI_MILLET=~/agri_foundation/data/hyperspectral/pearl_millet_stress

# Model weights
SAM2_WEIGHTS=~/agri_foundation/weights/sam2/sam2.1_hiera_large.pt
PRITHVI_WEIGHTS=~/agri_foundation/weights/prithvi/
SPECTRALGPT_WEIGHTS=~/agri_foundation/weights/spectralgpt/SpectralGPT.pth
GDINO_WEIGHTS=~/agri_foundation/weights/groundingdino/groundingdino_swint_ogc.pth

# GPU config
CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Training defaults
BATCH_SIZE=16
NUM_WORKERS=8
EMBEDDING_DIM=512
FEW_SHOT_K=5
SEED=42

# ERA5 / CDS API (fill in your key)
CDS_API_KEY=your-cds-api-key-here
CDS_API_URL=https://cds.climate.copernicus.eu/api/v2

# WandB (optional)
WANDB_PROJECT=agri-foundation
WANDB_ENTITY=your-wandb-username
EOF

echo "✓ .env config created at ~/agri_foundation/.env"
echo "  → Edit CDS_API_KEY and WANDB_ENTITY before running experiments"

echo ""
echo "=============================================="
echo " SETUP COMPLETE — $(date)"
echo " Activate env with: conda activate agri-foundation"
echo " Log saved to: setup_env_log.txt"
echo "=============================================="