#!/bin/bash
# =============================================================================
# AGRI FOUNDATION — Deep Dataset Inspection
# =============================================================================

BASE=~/agri_foundation
DATA=$BASE/data

echo "=============================================="
echo " AGRI FOUNDATION — Deep Inspection"
echo " $(date)"
echo "=============================================="

# ------------------------------------------------------------------------------
# FIX: Re-extract groundnut zip (spaces in filename need quoting)
# ------------------------------------------------------------------------------
echo ""
echo ">>> FIX: RE-EXTRACTING GROUNDNUT ZIP"
echo "----------------------------------------------"
cd $DATA
ls -lah "$DATA/GROUNDNUT WATER STRESS DATA.zip" 2>/dev/null && \
    unzip -o "$DATA/GROUNDNUT WATER STRESS DATA.zip" -d "$DATA/hyperspectral/groundnut_stress/" && \
    echo "✓ Groundnut extracted" || echo "✗ Still failed"

echo "Groundnut folder after extraction:"
find $DATA/hyperspectral/groundnut_stress -type f | head -20
echo "Size: $(du -sh $DATA/hyperspectral/groundnut_stress/)"

# ------------------------------------------------------------------------------
# INSPECT: Multispectral structure
# ------------------------------------------------------------------------------
echo ""
echo ">>> MULTISPECTRAL DEEP INSPECTION"
echo "----------------------------------------------"

echo "--- Top-level crop/season folders ---"
find $DATA/multispectral/raw/Agri -maxdepth 1 -type d

echo ""
echo "--- All unique crop types ---"
find $DATA/multispectral/raw/Agri -maxdepth 1 -type d | xargs -I{} basename {}

echo ""
echo "--- Season/flight folders per crop ---"
find $DATA/multispectral/raw/Agri -maxdepth 2 -type d

echo ""
echo "--- Date/season summary (from folder names) ---"
find $DATA/multispectral/raw -type d | grep -oP '\d{8}' | sort | uniq

echo ""
echo "--- Count of images per band ---"
echo "Band 1 (Blue):    $(find $DATA/multispectral/raw -name '*_1.tif' | wc -l)"
echo "Band 2 (Green):   $(find $DATA/multispectral/raw -name '*_2.tif' | wc -l)"
echo "Band 3 (Red):     $(find $DATA/multispectral/raw -name '*_3.tif' | wc -l)"
echo "Band 4 (RedEdge): $(find $DATA/multispectral/raw -name '*_4.tif' | wc -l)"
echo "Band 5 (NIR):     $(find $DATA/multispectral/raw -name '*_5.tif' | wc -l)"

echo ""
echo "--- Sample single image metadata (rasterio) ---"
SAMPLE=$(find $DATA/multispectral/raw -name '*_1.tif' | head -1)
echo "Sample file: $SAMPLE"

python3 << PYEOF
import rasterio
import os
sample = "$SAMPLE"
if os.path.exists(sample):
    with rasterio.open(sample) as src:
        print(f"  Shape: {src.height} x {src.width}")
        print(f"  Bands: {src.count}")
        print(f"  CRS: {src.crs}")
        print(f"  Resolution: {src.res}")
        print(f"  Dtype: {src.dtypes}")
        print(f"  Bounds: {src.bounds}")
        tags = src.tags()
        if tags:
            print(f"  Tags (first 10): {dict(list(tags.items())[:10])}")
        # Check EXIF/metadata for DLS calibration
        for i in range(1, src.count+1):
            band_tags = src.tags(i)
            if band_tags:
                print(f"  Band {i} tags: {band_tags}")
else:
    print(f"  File not found: {sample}")
PYEOF

# ------------------------------------------------------------------------------
# INSPECT: RGB Paddy (YOLO label format)
# ------------------------------------------------------------------------------
echo ""
echo ">>> RGB PADDY DEEP INSPECTION"
echo "----------------------------------------------"

echo "--- Train folder contents ---"
find $DATA/rgb_paddy/train -type d | head -10
find $DATA/rgb_paddy/train -type f | head -20

echo ""
echo "--- Test folder contents ---"
find $DATA/rgb_paddy/test -type d | head -10
find $DATA/rgb_paddy/test -type f | head -20

echo ""
echo "--- Train image count ---"
find $DATA/rgb_paddy/train -name '*.jpg' | wc -l
find $DATA/rgb_paddy/train -name '*.png' | wc -l

echo ""
echo "--- Test image count ---"
find $DATA/rgb_paddy/test -name '*.jpg' | wc -l
find $DATA/rgb_paddy/test -name '*.png' | wc -l

echo ""
echo "--- Label format: sample .txt file ---"
SAMPLE_TXT=$(find $DATA/rgb_paddy -name '*.txt' | head -1)
echo "File: $SAMPLE_TXT"
echo "Content:"
cat "$SAMPLE_TXT" 2>/dev/null | head -10

echo ""
echo "--- Classes: unique class IDs in labels ---"
find $DATA/rgb_paddy -name '*.txt' -exec cat {} \; | awk '{print $1}' | sort | uniq -c | sort -rn | head -10

echo ""
echo "--- Sample image dimensions ---"
python3 << PYEOF
from PIL import Image
import os, glob

for split in ['train', 'test']:
    folder = f"$DATA/rgb_paddy/{split}"
    imgs = glob.glob(f"{folder}/**/*.jpg", recursive=True) + \
           glob.glob(f"{folder}/**/*.png", recursive=True)
    if imgs:
        img = Image.open(imgs[0])
        print(f"  [{split}] Sample: {os.path.basename(imgs[0])} | Size: {img.size} | Mode: {img.mode} | Total: {len(imgs)}")
    else:
        print(f"  [{split}] No images found")
PYEOF

# ------------------------------------------------------------------------------
# INSPECT: Hyperspectral (.npy and .dat files)
# ------------------------------------------------------------------------------
echo ""
echo ">>> HYPERSPECTRAL DEEP INSPECTION"
echo "----------------------------------------------"

for SUBSET in crop_variety groundnut_stress pearl_millet_stress; do
    echo ""
    echo "--- Subset: $SUBSET ---"
    find $DATA/hyperspectral/$SUBSET -type f | sort
    echo ""

    # Inspect .npy files
    NPY=$(find $DATA/hyperspectral/$SUBSET -name '*.npy' | head -1)
    if [ -n "$NPY" ]; then
        echo "  .npy file: $NPY"
        python3 << PYEOF
import numpy as np
import os
f = "$NPY"
if os.path.exists(f):
    arr = np.load(f, allow_pickle=True)
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}")
    if arr.ndim == 3:
        print(f"  Interpretation: (rows={arr.shape[0]}, cols={arr.shape[1]}, bands={arr.shape[2]})")
    elif arr.ndim == 2:
        print(f"  Interpretation: 2D array — could be label/mask or single band")
    elif arr.ndim == 1:
        print(f"  Interpretation: 1D — could be wavelength centers or labels vector")
PYEOF
    fi

    # Inspect .dat files
    DAT=$(find $DATA/hyperspectral/$SUBSET -name '*.dat' | head -1)
    if [ -n "$DAT" ]; then
        echo ""
        echo "  .dat file: $DAT"
        echo "  File size: $(du -sh "$DAT" | cut -f1)"
        echo "  First bytes (hex):"
        xxd "$DAT" | head -3
        # Check for accompanying .hdr
        HDR="${DAT%.dat}.hdr"
        if [ -f "$HDR" ]; then
            echo "  Companion .hdr found: $HDR"
            cat "$HDR"
        else
            echo "  No companion .hdr found"
            # Look for any .hdr in same folder
            find "$(dirname $DAT)" -name '*.hdr' | head -3
        fi
    fi

    echo ""
    echo "  All files in subset:"
    find $DATA/hyperspectral/$SUBSET -type f -exec ls -lh {} \;
done

# ------------------------------------------------------------------------------
# SUMMARY TABLE
# ------------------------------------------------------------------------------
echo ""
echo ">>> DATASET SUMMARY"
echo "----------------------------------------------"
python3 << PYEOF
import os, glob

data = os.path.expanduser("~/agri_foundation/data")

datasets = {
    "RGB Paddy Train":        (f"{data}/rgb_paddy/train", ['.jpg','.png']),
    "RGB Paddy Test":         (f"{data}/rgb_paddy/test",  ['.jpg','.png']),
    "RGB Labels (txt)":       (f"{data}/rgb_paddy",       ['.txt']),
    "MS Band 1 (Blue)":       (f"{data}/multispectral/raw", ['_1.tif']),
    "MS Band 5 (NIR)":        (f"{data}/multispectral/raw", ['_5.tif']),
    "HSI Crop Variety":       (f"{data}/hyperspectral/crop_variety", ['.npy','.dat','.tif','.hdr']),
    "HSI Groundnut":          (f"{data}/hyperspectral/groundnut_stress", ['.npy','.dat','.tif','.hdr']),
    "HSI Pearl Millet":       (f"{data}/hyperspectral/pearl_millet_stress", ['.npy','.dat','.tif','.hdr']),
}

print(f"\n{'Dataset':<25} {'Files':>8} {'Size':>10}")
print("-" * 48)
for name, (path, exts) in datasets.items():
    files = []
    for ext in exts:
        if ext.startswith('_'):
            files += glob.glob(f"{path}/**/*{ext}", recursive=True)
        else:
            files += glob.glob(f"{path}/**/*{ext}", recursive=True)
    total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    size_mb = total_size / 1e6
    print(f"{name:<25} {len(files):>8} {size_mb:>9.1f}M")
PYEOF

echo ""
echo "=============================================="
echo " DEEP INSPECTION COMPLETE — $(date)"
echo "=============================================="