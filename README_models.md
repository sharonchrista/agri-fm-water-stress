# Pretrained Model Checkpoint Download Instructions

The two foundation model backbones used in this study must be downloaded
separately due to file size. Neither is included in this repository.

## HyperSL (HSI Foundation Model)

- **Paper:** Kong et al., IEEE TGRS 2025
- **Parameters:** 9.49M
- **Checkpoint file:** `10_base_mask95_checkpoint.pt` (105 MB)
- **Download:** [HyperSL official repository — SEARCH AND ADD URL]
- **Place at:** `models/hypersl_weights/10_base_mask95_checkpoint.pt`
- **Also required:** HyperSL source code
  - **Place at:** `models/hypersl_code/`

## Prithvi-EO-2.0 (MS Foundation Model)

- **Paper:** Jakubik et al., arXiv 2023
- **Parameters:** 300M
- **Checkpoint file:** `Prithvi_EO_V2_300M.pt` (1.3 GB)
- **Download:** https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0
- **Place at:** `models/prithvi/Prithvi_EO_V2_300M.pt`
- **Also required:** `config.json` from the same HuggingFace repository
  - **Place at:** `models/prithvi/config.json`
- **Also required:** Prithvi source code (`prithvi_mae.py`)
  - **Place at:** `models/prithvi/prithvi_mae.py`

## Expected Directory Structure After Download

```
models/
├── hypersl_weights/
│   └── 10_base_mask95_checkpoint.pt
├── hypersl_code/
│   └── engine/
│       └── model.py
├── prithvi/
│   ├── Prithvi_EO_V2_300M.pt
│   ├── config.json
│   └── prithvi_mae.py
└── README_models.md
```