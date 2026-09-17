#!/bin/bash
# =============================================================================
# AGRI FOUNDATION — Phase 2 Setup and Encoder Tests
# =============================================================================

BASE=~/agri_foundation
cd $BASE

echo "=============================================="
echo " AGRI FOUNDATION — Phase 2: Encoders"
echo " $(date)"
echo "=============================================="

# Copy encoder files to correct locations
mkdir -p $BASE/encoders
cp spectralgpt_encoder.py $BASE/encoders/
cp prithvi_encoder.py $BASE/encoders/
cp sam2_encoder.py $BASE/encoders/
touch $BASE/encoders/__init__.py
echo "✓ Encoder files copied to ~/agri_foundation/encoders/"

# Check available weights
echo ""
echo ">>> WEIGHTS CHECK"
echo "----------------------------------------------"
find $BASE/weights -type f 2>/dev/null | sort
echo ""
echo "Weights directory sizes:"
du -sh $BASE/weights/*/ 2>/dev/null

# Try downloading missing weights
echo ""
echo ">>> DOWNLOADING MISSING WEIGHTS"
echo "----------------------------------------------"

# SAM2
SAM2_PT="$BASE/weights/sam2/sam2.1_hiera_large.pt"
if [ ! -f "$SAM2_PT" ]; then
    echo "Downloading SAM2 large weights..."
    mkdir -p $BASE/weights/sam2
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        -O "$SAM2_PT" && echo "✓ SAM2 downloaded" || \
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
        -O "$BASE/weights/sam2/sam_vit_h_4b8939.pth" && echo "✓ SAM v1 downloaded"
else
    echo "✓ SAM2 weights already present: $(du -sh $SAM2_PT | cut -f1)"
fi

# Prithvi from HuggingFace
PRITHVI_DIR="$BASE/weights/prithvi"
if [ ! "$(ls -A $PRITHVI_DIR 2>/dev/null)" ]; then
    echo "Downloading Prithvi-EO-2.0-300M from HuggingFace..."
    mkdir -p $PRITHVI_DIR
    python3 -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id='ibm-nasa-geospatial/Prithvi-EO-2.0-300M',
        local_dir='$PRITHVI_DIR',
        ignore_patterns=['*.msgpack','*.h5','flax_model*']
    )
    print('✓ Prithvi-EO-2.0-300M downloaded')
except Exception as e:
    print(f'Prithvi download failed: {e}')
    print('Manual: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M')
"
else
    echo "✓ Prithvi weights directory not empty"
    ls -lh $PRITHVI_DIR | head -5
fi

# SpectralGPT from HuggingFace
SGPT_DIR="$BASE/weights/spectralgpt"
if [ ! "$(ls -A $SGPT_DIR 2>/dev/null)" ]; then
    echo "Downloading SpectralGPT from HuggingFace..."
    mkdir -p $SGPT_DIR
    python3 -c "
from huggingface_hub import hf_hub_download, snapshot_download
try:
    # Try direct file download first
    hf_hub_download(
        repo_id='danfenghong/SpectralGPT',
        filename='SpectralGPT.pth',
        local_dir='$SGPT_DIR'
    )
    print('✓ SpectralGPT.pth downloaded')
except Exception as e1:
    print(f'Direct download failed: {e1}')
    try:
        snapshot_download(
            repo_id='danfenghong/SpectralGPT',
            local_dir='$SGPT_DIR'
        )
        print('✓ SpectralGPT snapshot downloaded')
    except Exception as e2:
        print(f'Snapshot download failed: {e2}')
        print('Manual: https://huggingface.co/danfenghong/SpectralGPT')
"
else
    echo "✓ SpectralGPT weights directory not empty"
    ls -lh $SGPT_DIR | head -5
fi

# GroundingDINO
GDINO_PT="$BASE/weights/groundingdino/groundingdino_swint_ogc.pth"
if [ ! -f "$GDINO_PT" ]; then
    echo "Downloading GroundingDINO weights..."
    mkdir -p $BASE/weights/groundingdino
    wget -q --show-progress \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
        -O "$GDINO_PT" && echo "✓ GroundingDINO downloaded" || echo "✗ GroundingDINO download failed"
else
    echo "✓ GroundingDINO: $(du -sh $GDINO_PT | cut -f1)"
fi

echo ""
echo ">>> RUNNING ENCODER TESTS"
echo "----------------------------------------------"

# Test SpectralGPT
echo ""
echo "[1/3] SpectralGPT Encoder"
cd $BASE
python3 -c "
import sys
sys.path.insert(0, 'encoders')
from spectralgpt_encoder import test_encoder
test_encoder()
" 2>&1 | tee $BASE/results/test_spectralgpt.log
echo ""

# Test Prithvi
echo "[2/3] Prithvi Encoder"
python3 -c "
import sys
sys.path.insert(0, 'encoders')
from prithvi_encoder import test_encoder
test_encoder()
" 2>&1 | tee $BASE/results/test_prithvi.log
echo ""

# Test SAM2
echo "[3/3] SAM2 Encoder"
python3 -c "
import sys
sys.path.insert(0, 'encoders')
from sam2_encoder import test_encoder
test_encoder()
" 2>&1 | tee $BASE/results/test_sam2.log
echo ""

echo "=============================================="
echo " ENCODER TESTS COMPLETE — $(date)"
echo " Logs: ~/agri_foundation/results/test_*.log"
echo "=============================================="