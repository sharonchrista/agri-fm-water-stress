#!/bin/bash
# =============================================================================
# AGRI FOUNDATION — Fixed Extraction Script
# Zip files are in: ~/agri_foundation/data/
# =============================================================================

BASE=~/agri_foundation
DATA=$BASE/data

echo "=============================================="
echo " AGRI FOUNDATION — Data Extraction (Fixed)"
echo " $(date)"
echo "=============================================="

# ------------------------------------------------------------------------------
# STEP 1 — Confirm zip files are present
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 1: LOCATING ZIP FILES IN $DATA"
echo "----------------------------------------------"
ls -lh $DATA/*.zip 2>/dev/null || echo "No .zip files found at $DATA/"
echo ""
echo "All files in $DATA/:"
ls -lah $DATA/

# ------------------------------------------------------------------------------
# STEP 2 — Extract each zip to its correct target folder
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 2: EXTRACTING"
echo "----------------------------------------------"

# RGB Paddy train
if [ -f "$DATA/train.zip" ]; then
    echo "[1/5] Extracting train.zip → data/rgb_paddy/train/"
    mkdir -p $DATA/rgb_paddy/train
    unzip -o "$DATA/train.zip" -d "$DATA/rgb_paddy/train/" && echo "  ✓ Done" || echo "  ✗ Failed"
else
    echo "[1/5] train.zip not found at $DATA/ — searching..."
    find $BASE -name "train.zip" 2>/dev/null
fi

# RGB Paddy test
if [ -f "$DATA/test.zip" ]; then
    echo "[2/5] Extracting test.zip → data/rgb_paddy/test/"
    mkdir -p $DATA/rgb_paddy/test
    unzip -o "$DATA/test.zip" -d "$DATA/rgb_paddy/test/" && echo "  ✓ Done" || echo "  ✗ Failed"
else
    echo "[2/5] test.zip not found at $DATA/ — searching..."
    find $BASE -name "test.zip" 2>/dev/null
fi

# Multispectral
if [ -f "$DATA/Agri.zip" ]; then
    echo "[3/5] Extracting Agri.zip → data/multispectral/raw/ (5.68 GB — may take a few minutes)"
    mkdir -p $DATA/multispectral/raw
    unzip -o "$DATA/Agri.zip" -d "$DATA/multispectral/raw/" && echo "  ✓ Done" || echo "  ✗ Failed"
else
    echo "[3/5] Agri.zip not found at $DATA/ — searching..."
    find $BASE -name "Agri.zip" 2>/dev/null
fi

# Hyperspectral crop variety
if [ -f "$DATA/Crop_dataset.zip" ]; then
    echo "[4/5] Extracting Crop_dataset.zip → data/hyperspectral/crop_variety/"
    mkdir -p $DATA/hyperspectral/crop_variety
    unzip -o "$DATA/Crop_dataset.zip" -d "$DATA/hyperspectral/crop_variety/" && echo "  ✓ Done" || echo "  ✗ Failed"
else
    echo "[4/5] Crop_dataset.zip not found — searching..."
    find $BASE -name "*.zip" 2>/dev/null | grep -i crop
fi

# Hyperspectral groundnut (filename has spaces)
GROUNDNUT_ZIP=$(find $DATA -maxdepth 1 -name "*GROUNDNUT*" -o -name "*groundnut*" 2>/dev/null | head -1)
if [ -n "$GROUNDNUT_ZIP" ]; then
    echo "[5/5] Extracting $(basename "$GROUNDNUT_ZIP") → data/hyperspectral/groundnut_stress/"
    mkdir -p $DATA/hyperspectral/groundnut_stress
    unzip -o "$GROUNDNUT_ZIP" -d "$DATA/hyperspectral/groundnut_stress/" && echo "  ✓ Done" || echo "  ✗ Failed"
else
    echo "[5/5] Groundnut zip not found — searching..."
    find $BASE -name "*roundnut*" 2>/dev/null
fi

# Pearl millet — check if it was inside Crop_dataset.zip or separate
MILLET_ZIP=$(find $DATA -maxdepth 1 -name "*millet*" -o -name "*pearl*" -o -name "*Millet*" 2>/dev/null | head -1)
if [ -n "$MILLET_ZIP" ]; then
    echo "[+] Found millet zip: $MILLET_ZIP — extracting..."
    mkdir -p $DATA/hyperspectral/pearl_millet_stress
    unzip -o "$MILLET_ZIP" -d "$DATA/hyperspectral/pearl_millet_stress/" && echo "  ✓ Done"
else
    echo "[+] No separate millet zip found — may be inside Crop_dataset.zip"
fi

# ------------------------------------------------------------------------------
# STEP 3 — Check for any other zips not yet handled
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 3: REMAINING UNHANDLED ZIPS"
echo "----------------------------------------------"
find $DATA -name "*.zip" 2>/dev/null
echo ""
echo "All zip files found anywhere under $BASE:"
find $BASE -name "*.zip" 2>/dev/null

# ------------------------------------------------------------------------------
# STEP 4 — Size summary after extraction
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 4: FOLDER SIZE SUMMARY AFTER EXTRACTION"
echo "----------------------------------------------"
du -sh $DATA/rgb_paddy/train/ 2>/dev/null
du -sh $DATA/rgb_paddy/test/ 2>/dev/null
du -sh $DATA/multispectral/raw/ 2>/dev/null
du -sh $DATA/hyperspectral/crop_variety/ 2>/dev/null
du -sh $DATA/hyperspectral/groundnut_stress/ 2>/dev/null
du -sh $DATA/hyperspectral/pearl_millet_stress/ 2>/dev/null
echo ""
echo "Total data directory size:"
du -sh $DATA/

# ------------------------------------------------------------------------------
# STEP 5 — Quick structure check
# ------------------------------------------------------------------------------
echo ""
echo ">>> STEP 5: EXTRACTED CONTENT PREVIEW"
echo "----------------------------------------------"

echo "--- RGB Paddy Train (first 20 files) ---"
find $DATA/rgb_paddy/train -type f | head -20

echo ""
echo "--- RGB Paddy Test (first 20 files) ---"
find $DATA/rgb_paddy/test -type f | head -20

echo ""
echo "--- Multispectral raw (first 20 files) ---"
find $DATA/multispectral/raw -type f | head -20

echo ""
echo "--- HSI Crop Variety (first 20 files) ---"
find $DATA/hyperspectral/crop_variety -type f | head -20

echo ""
echo "--- HSI Groundnut (first 20 files) ---"
find $DATA/hyperspectral/groundnut_stress -type f | head -20

echo ""
echo "--- HSI Pearl Millet (first 20 files) ---"
find $DATA/hyperspectral/pearl_millet_stress -type f | head -20

echo ""
echo "--- File extension summary across ALL data ---"
find $DATA -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn

echo ""
echo "=============================================="
echo " EXTRACTION COMPLETE — $(date)"
echo "=============================================="