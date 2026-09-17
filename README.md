# Physics-Constrained Few-Shot Multimodal Foundation Models for Cross-Geographic Crop Stress Monitoring

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)

Official code and pretrained adapters for the paper:

> **Physics-Constrained Few-Shot Multimodal Foundation Models for Cross-Geographic Crop Stress Monitoring: A Multi-Country UAV Benchmark**
> Sharon Christa, Ram Avtar
> *Remote Sensing of Environment* (under review)

---

## Overview

This repository provides the full experimental pipeline for a multi-country UAV crop stress monitoring benchmark spanning hyperspectral (HSI), multispectral (MS), and RGB sensing across India, USA, Turkey, and Sri Lanka.

**Key results:**

| Branch | Task | Result |
|---|---|---|
| HSI (SpectralMLP) | Groundnut water stress | 98.41% accuracy |
| HSI (SpectralMLP) | 5-shot few-shot | 97.75% (±0.11%) |
| HSI (GP on HyperSL) | Calibrated uncertainty | 99.85%, ECE = 0.0084 |
| MS (Prithvi-EO-2.0) | Cross-crop transfer | 99.92% Paddy to Maize |
| MS (Prithvi-EO-2.0) | Cross-geographic transfer | 99.20% India to Turkey |
| RGB (DensityNet) | Panicle counting | MAE 7.39, RelMAE 0.244 |
| HSI cross-crop | Groundnut to Wheat | 50.15% (random chance, negative result) |

---

## Repository Structure

```
agri-fm-water-stress/
├── data/
│   └── README_data.md          # Dataset download instructions
├── models/
│   └── README_models.md        # Pretrained checkpoint download instructions
├── checkpoints/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── figures/
│   ├── fig1_architecture.png
│   ├── fig2_fewshot_curve.png
│   ├── fig3_training_curves.png
│   ├── fig4_crossmodal_bars.png
│   ├── fig5_tsne.png
│   ├── fig6_shap_attribution.png
│   ├── fig7_density_maps.png
│   ├── fig8_gp_calibration.png
│   └── fig9_physics_ablation.png
├── paper/
│   ├── main.tex
│   └── references.bib
├── train_mlp_baseline.py       # SpectralMLP supervised training
├── few_shot_eval.py            # Prototypical few-shot evaluation
├── prithvi_ms_finetune.py      # Prithvi-EO-2.0 MS linear probe
├── cross_geo_eval_v2.py        # Cross-geographic transfer (real Turkey labels)
├── gp_uncertainty_head.py      # Sparse GP uncertainty classifier
├── train_rgb_v2.py             # DensityNet panicle counting
├── generate_figures.py         # Figures 2-5
├── generate_figures_6_7.py     # SHAP + density map figures
├── fix_wheat.py                # Wheat data loading + cross-crop transfer
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

---

## Datasets

All datasets used in this study are publicly available. Download instructions are in `data/README_data.md`.

| Dataset | Country | Modality | Crop | Source |
|---|---|---|---|---|
| Groundnut water stress | India | HSI, 282-band | Groundnut | ICRISAT / TIAND |
| UMN wheat | USA | HSI, 190-band | Wheat | UMN Data Repository |
| Zenodo maize | Turkey | MS, 6-band | Maize | [Zenodo 22062459](https://zenodo.org/records/22062459) |
| Paddy MS | Sri Lanka | MS, 4-band | Paddy | [Mendeley Data](https://doi.org/10.17632/h8s5mn52j6.1) |
| Paddy panicle RGB | India | RGB | Paddy | TIAND |

After downloading, place datasets under `data/` following the structure described in `data/README_data.md`.

---

## Pretrained Foundation Model Checkpoints

The following pretrained model weights are required and must be downloaded separately due to size.

| Model | Parameters | Source |
|---|---|---|
| HyperSL | 9.49M | [HyperSL GitHub](https://github.com/hypersl/hypersl) |
| Prithvi-EO-2.0 | 300M | [HuggingFace](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0) |

Place checkpoints under `models/hypersl_weights/` and `models/prithvi/` respectively. See `models/README_models.md` for exact filenames.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sharonchrista/agri-fm-water-stress.git
cd agri-fm-water-stress

# Create conda environment
conda env create -f environment.yml
conda activate agri-fm-env

# Or with pip
pip install -r requirements.txt
```

**Requirements:** Python 3.11, PyTorch 2.0, CUDA 12.4, NVIDIA GPU (16GB+ VRAM recommended)

---

## Usage

### 1. HSI Branch: SpectralMLP Supervised Training

```bash
python train_mlp_baseline.py
```

Trains the SpectralMLP encoder on groundnut water stress data. Saves checkpoint to `checkpoints/groundnut_mlp_best.pt`.

### 2. HSI Branch: Few-Shot Prototypical Evaluation

```bash
python few_shot_eval.py
```

Evaluates the frozen SpectralMLP encoder under prototypical few-shot protocol (1/5/10/20-shot, 1000 episodes). Results saved to `logs/fewshot_results.json`.

### 3. HSI Branch: GP Uncertainty Head

```bash
CUDA_VISIBLE_DEVICES=0 python gp_uncertainty_head.py
```

Trains sparse GP classifier on HyperSL embeddings and evaluates calibration (ECE). Results saved to `logs/gp_results.json`.

### 4. HSI Branch: Cross-Crop Transfer (Groundnut to Wheat)

```bash
python fix_wheat.py
```

Evaluates groundnut encoder on wheat data across all band subsets and N-shot values. Results saved to `logs/cross_crop_hsi_results.json`.

### 5. MS Branch: Prithvi-EO-2.0 Cross-Crop Linear Probe

```bash
CUDA_VISIBLE_DEVICES=0 python prithvi_ms_finetune.py
```

Evaluates frozen Prithvi-EO-2.0 on India maize and paddy MS tiles. Results saved to `logs/prithvi_eval.json`.

### 6. MS Branch: Cross-Geographic Transfer

```bash
CUDA_VISIBLE_DEVICES=0 python cross_geo_eval_v2.py
```

Evaluates cross-geographic transfer across India, Turkey, and Sri Lanka using real Turkey stress labels and vegetation index ensemble labels for India and Sri Lanka. Results saved to `logs/cross_geo_eval_v2.json`.

### 7. RGB Branch: DensityNet Panicle Counting

```bash
CUDA_VISIBLE_DEVICES=0 python train_rgb_v2.py
```

Trains DensityNet on paddy panicle RGB imagery. Best checkpoint saved to `checkpoints/densitynet_best.pt`.

### 8. Generate Figures

```bash
python generate_figures.py        # Figures 2-5
python generate_figures_6_7.py    # Figures 6 (SHAP) and 7 (density maps)
```

---

## Pretrained Adapters

Pretrained SpectralMLP and DensityNet checkpoints from this study are available for download:

| Checkpoint | Description | Link |
|---|---|---|
| `groundnut_mlp_best.pt` | SpectralMLP, epoch 87, 98.41% | [SEARCH AND ADD] |
| `densitynet_best.pt` | DensityNet, epoch 89, MAE 7.39 | [SEARCH AND ADD] |
| `gp_hsi_best.pt` | SVGP on HyperSL, ECE 0.0084 | [SEARCH AND ADD] |

---

## Results Reproduction

All results in the paper can be reproduced by running the scripts in the order above. Expected runtimes on a single NVIDIA Tesla T4 (16GB):

| Script | Runtime |
|---|---|
| `train_mlp_baseline.py` | ~25 minutes |
| `few_shot_eval.py` | ~15 minutes |
| `gp_uncertainty_head.py` | ~45 minutes |
| `prithvi_ms_finetune.py` | ~1 hour |
| `cross_geo_eval_v2.py` | ~1 hour |
| `train_rgb_v2.py` | ~3 hours |

---

## Citation

If you use this code or benchmark in your research, please cite:

```bibtex
@article{christa2025agrifm,
  author  = {Christa, Sharon and Avtar, Ram},
  title   = {Physics-Constrained Few-Shot Multimodal Foundation Models
             for Cross-Geographic Crop Stress Monitoring:
             A Multi-Country UAV Benchmark},
  journal = {Remote Sensing of Environment},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The pretrained HyperSL and Prithvi-EO-2.0 checkpoints are subject to their respective licenses. Please refer to the original repositories for terms of use.

---

## Acknowledgements

[SEARCH AND ADD: funding acknowledgements]

The authors thank ICRISAT Hyderabad for providing the groundnut hyperspectral dataset, the University of Minnesota for the wheat HSI dataset, and the Zenodo maize dataset contributors for releasing real stress annotations.

---

## Contact

Sharon Christa — MIT Art, Design and Technology University, Pune, India
sharon@mituniversity.edu.in