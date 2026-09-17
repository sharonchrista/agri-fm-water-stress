# Dataset Download Instructions

All five datasets used in this study are publicly available. Download
each dataset and place it under the corresponding subdirectory below.

## Expected Directory Structure After Download

```
data/
├── processed/
│   ├── hsi/
│   │   ├── gn_X_flat.npy          # Groundnut HSI spectra (16667, 282)
│   │   ├── gn_y.npy               # Groundnut labels (16667,)
│   │   └── gn_test_idx.npy        # Test set indices (3334,)
│   └── ms/
│       ├── maize/
│       │   └── ms_stacked.npy     # India maize MS (N, H, W, 5)
│       └── paddy/
│           └── ms_stacked.npy     # India paddy MS (N, H, W, 5)
├── umn_wheat/
│   ├── wheat_X.npy                # Wheat HSI spectra (15315, 190)
│   └── wheat_y.npy                # Wheat yield binary labels (15315,)
├── zenodo_maize_ms/
│   └── processed_patches/
│       └── water_2025/
│           ├── images/            # Turkey maize patches (*.npy)
│           ├── masks/             # Pixel-level stress masks
│           └── meta/
│               ├── patches.csv
│               └── class_map.json
├── paddy_srilanka/
│   └── ms_stacked_srilanka.npy    # Sri Lanka paddy MS (67, 1944, 2592, 4)
└── processed/
    └── rgb/
        ├── train_images/          # Paddy RGB training images
        ├── test_images/           # Paddy RGB test images
        └── annotations/           # Point annotation JSON files
```

## Dataset 1 — Groundnut Water Stress HSI (India)

- **Source:** TIAND dataset, ICRISAT Hyderabad
- **Place at:** `data/processed/hsi/`

## Dataset 2 — UMN Wheat HSI (USA)

- **Source:** University of Minnesota Data Repository
- **DOI:** 10.13020/0ch0-vb18
- **Download:** https://conservancy.umn.edu/handle/11299/204290
- **Place at:** `data/umn_wheat/`

## Dataset 3 — Zenodo Maize MS with Real Stress Labels (Turkey)

- **Source:** Zenodo
- **DOI:** 10.5281/zenodo.22062459
- **Download:** https://zenodo.org/records/22062459
- **Place at:** `data/zenodo_maize_ms/`

## Dataset 4 — Paddy MS (Sri Lanka)

- **Source:** Mendeley Data
- **DOI:** 10.17632/h8s5mn52j6.1
- **Download:** https://data.mendeley.com/datasets/h8s5mn52j6/1
- **Place at:** `data/paddy_srilanka/`

## Dataset 5 — Paddy Panicle RGB (India)

- **Source:** TIAND dataset
- **Place at:** `data/processed/rgb/`