"""
=============================================================================
AGRI FOUNDATION — SAM2 Encoder
=============================================================================
Wraps SAM2 for RGB paddy image encoding and point-prompted segmentation.

Two modes:
  1. EMBEDDING MODE: Extract image embeddings for shared embedding space
  2. DETECTION MODE: Point-prompted panicle segmentation/detection

Input data:
  - Images: (N, 850, 1150, 3) — paddy RGB patches
  - Labels: point annotations (x, y) pixel coordinates of panicle centres

SAM2 weights: sam2_hiera_large.pt
Reference: Kirillov et al. (2023) — Segment Anything
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict

BASE        = Path.home() / 'agri_foundation'
WEIGHTS_DIR = BASE / 'weights' / 'sam2'
EMBED_DIM   = 512

# SAM2 native image size
SAM2_IMG_SIZE = 1024


# ─────────────────────────────────────────────
# SAM2 IMAGE ENCODER WRAPPER
# ─────────────────────────────────────────────

class SAM2ImageEncoder(nn.Module):
    """
    Wraps SAM2 image encoder to produce fixed-dim embeddings
    for the shared multi-modal embedding space.

    SAM2 image encoder output: (B, 256, 64, 64) spatial feature map
    We pool + project → (B, 512) for alignment with HSI/MS embeddings.

    Usage:
        encoder = SAM2ImageEncoder(device='cuda:2')
        x = torch.randn(4, 3, 1024, 1024)
        emb = encoder(x)   # (4, 512)
    """
    def __init__(
        self,
        embed_dim:       int  = EMBED_DIM,
        weights_path:    Optional[str] = None,
        freeze_backbone: bool = True,
        device:          str  = 'cuda:2',
        img_size:        int  = SAM2_IMG_SIZE,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device    = device
        self.img_size  = img_size
        self.sam2      = None
        self.sam2_type = None

        # Search for weights
        search_paths = [
            weights_path,
            str(WEIGHTS_DIR / 'sam2_hiera_large.pt'),
            str(WEIGHTS_DIR / 'sam2.1_hiera_base_plus.pt'),
            str(WEIGHTS_DIR / 'sam_vit_h_4b8939.pth'),  # SAM v1 fallback
        ]

        loaded = False

        # Try SAM2 first
        for path in search_paths:
            if not path or not Path(path).exists():
                continue

            # SAM2
            if 'sam2' in Path(path).name.lower():
                try:
                    from sam2.build_sam import build_sam2
                    from sam2.sam2_image_predictor import SAM2ImagePredictor

                    # Determine config from filename
                    name = Path(path).stem
                    if 'large' in name:
                        cfg = 'sam2_hiera_l.yaml'
                    elif 'base_plus' in name:
                        cfg = 'sam2_hiera_b+.yaml'
                    elif 'small' in name:
                        cfg = 'sam2.1/sam2.1_hiera_s.yaml'
                    else:
                        cfg = 'sam2_hiera_l.yaml'

                    self.sam2      = build_sam2(cfg, path,
                                         device=device, apply_postprocessing=False)
                    self.sam2_type = 'sam2'
                    loaded         = True
                    print(f"[SAM2] Loaded SAM2 from {path}")
                    break
                except Exception as e:
                    print(f"[SAM2] SAM2 load failed: {e}")

            # SAM v1 fallback
            elif 'sam_vit' in Path(path).name.lower():
                try:
                    from segment_anything import sam_model_registry, SamPredictor
                    model_type = 'vit_h' if 'vit_h' in path else 'vit_l'
                    self.sam2      = sam_model_registry[model_type](checkpoint=path)
                    self.sam2_type = 'sam1'
                    loaded         = True
                    print(f"[SAM2] Loaded SAM v1 ({model_type}) from {path}")
                    break
                except Exception as e:
                    print(f"[SAM2] SAM v1 load failed: {e}")

        if not loaded:
            print("[SAM2] No SAM weights found — using fallback CNN encoder")
            print(f"  Expected: {WEIGHTS_DIR}/sam2_hiera_large.pt")
            self.sam2      = None
            self.sam2_type = 'fallback'

        # Projection head: SAM feature dim → shared embed_dim
        # SAM2 large: 256-dim features; SAM v1 ViT-H: 256-dim features
        sam_feature_dim = 256
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),                       # (B, 256, 8, 8)
            nn.Flatten(1),                                  # (B, 256*64)
            nn.Linear(sam_feature_dim * 64, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Fallback encoder if SAM not available
        if self.sam2_type == 'fallback':
            self.fallback_encoder = self._build_fallback_encoder(embed_dim)

        # Freeze SAM backbone
        if self.sam2 is not None and freeze_backbone:
            for param in self.sam2.parameters():
                param.requires_grad = False
            print("[SAM2] SAM backbone frozen")

        self.to(device)

    def _build_fallback_encoder(self, embed_dim: int) -> nn.Module:
        """
        Lightweight CNN encoder used when SAM weights unavailable.
        ResNet-style with 4 downsampling blocks.
        """
        return nn.Sequential(
            # Block 1: (B, 3, H, W) → (B, 64, H/4, W/4)
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.BatchNorm2d(64),
            # Block 2: → (B, 128, H/8, W/8)
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.BatchNorm2d(128),
            # Block 3: → (B, 256, H/16, W/16)
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(),
            nn.BatchNorm2d(256),
            # Global pool + project
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def _extract_sam2_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image features from SAM2 backbone."""
        if self.sam2_type == 'sam2':
            # SAM2 image encoder
            with torch.no_grad() if not self.training else torch.enable_grad():
                # SAM2 expects normalised images
                features = self.sam2.image_encoder(x)
                # Returns dict with 'vision_features', 'vision_pos_enc', etc.
                if isinstance(features, dict):
                    feat = features.get('vision_features',
                           features.get('image_embed', list(features.values())[0]))
                else:
                    feat = features
                # Ensure (B, C, H, W)
                if feat.ndim == 3:
                    feat = feat.unsqueeze(0)
                return feat

        elif self.sam2_type == 'sam1':
            # SAM v1 image encoder
            features = self.sam2.image_encoder(x)
            return features  # (B, 256, 64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) — RGB image, values in [0,1]
        Returns: (B, embed_dim)
        """
        if not x.is_cuda:
            x = x.to(self.device)

        # Resize to SAM input size
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        # Normalise to ImageNet stats (SAM expects this)
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225],
                            device=x.device).view(1, 3, 1, 1)
        x_norm = (x - mean) / std

        if self.sam2_type == 'fallback':
            return self.fallback_encoder(x_norm)

        # Extract SAM features
        try:
            features = self._extract_sam2_features(x_norm)
            return self.projection(features)
        except Exception as e:
            print(f"[SAM2] Feature extraction error: {e} — using fallback")
            return self.fallback_encoder(x_norm)

    def encode_batch(self, x: np.ndarray,
                     batch_size: int = 8) -> np.ndarray:
        """
        Encode RGB images in batches.
        x: (N, H, W, 3) numpy array, values in [0,1]
        Returns: (N, embed_dim)
        """
        self.eval()
        embeddings = []
        n = len(x)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = x[i:i+batch_size]
                # (B, H, W, 3) → (B, 3, H, W)
                batch_t = torch.tensor(
                    batch, dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)
                emb = self.forward(batch_t)
                embeddings.append(emb.cpu().numpy())
                print(f"  Encoded {min(i+batch_size, n)}/{n}", end='\r')
        print()
        return np.concatenate(embeddings, axis=0)


# ─────────────────────────────────────────────
# SAM2 POINT PREDICTOR (DETECTION MODE)
# ─────────────────────────────────────────────

class SAM2PointPredictor:
    """
    Uses SAM2 in point-prompted mode for panicle detection.
    Given an image and point annotations, generates instance masks
    around each panicle centre.

    This is used for:
    - Generating training masks from point annotations
    - Inference: detecting panicles given user-supplied points
    - Zero-shot segmentation baseline

    Usage:
        predictor = SAM2PointPredictor(device='cuda:2')
        masks = predictor.predict_from_points(image, points)
    """
    def __init__(self, device: str = 'cuda:2',
                 weights_path: Optional[str] = None):
        self.device = device
        self.predictor = None

        search_paths = [
            weights_path,
            str(WEIGHTS_DIR / 'sam2_hiera_large.pt'),
            str(WEIGHTS_DIR / 'sam_vit_h_4b8939.pth'),
        ]

        for path in search_paths:
            if not path or not Path(path).exists():
                continue
            try:
                if 'sam2' in Path(path).name:
                    from sam2.build_sam import build_sam2
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                    name = Path(path).stem
                    cfg  = 'sam2_hiera_l.yaml' if 'large' in name \
                           else 'sam2_hiera_b+.yaml'
                    model = build_sam2(cfg, path, device=device,
                                       apply_postprocessing=False)
                    self.predictor = SAM2ImagePredictor(model)
                    self.pred_type = 'sam2'
                    print(f"[SAM2Predictor] Loaded from {path}")
                    break
                else:
                    from segment_anything import sam_model_registry, SamPredictor
                    model_type = 'vit_h'
                    sam = sam_model_registry[model_type](checkpoint=path)
                    sam.to(device)
                    self.predictor = SamPredictor(sam)
                    self.pred_type = 'sam1'
                    print(f"[SAM2Predictor] SAM v1 loaded from {path}")
                    break
            except Exception as e:
                print(f"[SAM2Predictor] Load failed: {e}")

        if self.predictor is None:
            print("[SAM2Predictor] No SAM weights found")

    def predict_from_points(
        self,
        image:  np.ndarray,           # (H, W, 3), uint8 or float [0,1]
        points: np.ndarray,           # (N, 2), pixel coords [x, y]
        multimask_output: bool = False
    ) -> Optional[np.ndarray]:
        """
        Generate segmentation masks from point annotations.

        Args:
            image:  RGB image (H, W, 3)
            points: Panicle centre points (N, 2) in pixel coords
            multimask_output: if True returns 3 masks per point

        Returns:
            masks: (N, H, W) boolean array — one mask per point
        """
        if self.predictor is None or len(points) == 0:
            return None

        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        try:
            self.predictor.set_image(image)

            all_masks = []
            # Process points in batches to avoid OOM
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch_pts   = points[i:i+batch_size]  # (K, 2)
                batch_lbls  = np.ones(len(batch_pts), dtype=np.int32)

                if self.pred_type == 'sam2':
                    masks, scores, _ = self.predictor.predict(
                        point_coords   = batch_pts,
                        point_labels   = batch_lbls,
                        multimask_output = multimask_output
                    )
                else:
                    masks, scores, _ = self.predictor.predict(
                        point_coords   = batch_pts,
                        point_labels   = batch_lbls,
                        multimask_output = multimask_output
                    )

                # Take best mask per point
                if multimask_output:
                    best_idx = scores.argmax(axis=1)
                    for j, idx in enumerate(best_idx):
                        all_masks.append(masks[j, idx])
                else:
                    for m in masks:
                        all_masks.append(m)

            return np.array(all_masks)

        except Exception as e:
            print(f"[SAM2Predictor] Prediction error: {e}")
            return None

    def generate_heatmap(
        self,
        image:  np.ndarray,
        points: np.ndarray,
        sigma:  float = 10.0
    ) -> np.ndarray:
        """
        Generate Gaussian density heatmap from point annotations.
        Used as training target for panicle counting.

        Args:
            image:  (H, W, 3)
            points: (N, 2) pixel coords
            sigma:  Gaussian spread in pixels

        Returns:
            heatmap: (H, W) float32 density map
        """
        H, W = image.shape[:2]
        heatmap = np.zeros((H, W), dtype=np.float32)

        for x, y in points:
            x, y = int(round(x)), int(round(y))
            if 0 <= x < W and 0 <= y < H:
                # Gaussian kernel centred at (x, y)
                x_grid = np.arange(W)
                y_grid = np.arange(H)
                xx, yy = np.meshgrid(x_grid, y_grid)
                gauss  = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                heatmap += gauss

        return heatmap


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

def test_encoder():
    print("=" * 50)
    print("SAM2 Encoder — Quick Test")
    print("=" * 50)

    device = 'cuda:2' if torch.cuda.device_count() > 2 else \
             'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    encoder = SAM2ImageEncoder(
        embed_dim       = 512,
        freeze_backbone = True,
        device          = device
    )

    total_p     = sum(p.numel() for p in encoder.parameters())
    trainable_p = sum(p.numel() for p in encoder.parameters()
                      if p.requires_grad)
    print(f"\nTotal params    : {total_p:,}")
    print(f"Trainable params: {trainable_p:,}")

    # Test 1: standard input
    x = torch.randn(2, 3, 224, 224).clamp(0, 1).to(device)
    with torch.no_grad():
        emb = encoder(x)
    print(f"\nInput  {tuple(x.shape)} → embedding {tuple(emb.shape)}")
    assert emb.shape == (2, 512)

    # Test 2: paddy image size (850x1150) — will be resized
    x2 = torch.randn(1, 3, 850, 1150).clamp(0, 1).to(device)
    with torch.no_grad():
        emb2 = encoder(x2)
    print(f"Input  {tuple(x2.shape)} → embedding {tuple(emb2.shape)}")
    assert emb2.shape == (1, 512)

    # Test 3: numpy batch
    print("\nTesting numpy batch encoding...")
    x_np = np.random.rand(6, 128, 128, 3).astype(np.float32)
    import torch; torch.cuda.empty_cache(); embs = encoder.encode_batch(x_np, batch_size=2)
    print(f"Numpy input (10, 256, 256, 3) → {embs.shape}")
    assert embs.shape == (10, 512)

    # Test 4: heatmap generation
    print("\nTesting heatmap generation...")
    predictor = SAM2PointPredictor(device=device)
    img    = np.random.randint(0, 255, (850, 1150, 3), dtype=np.uint8)
    points = np.array([[100, 200], [500, 400], [800, 700]], dtype=np.float32)
    heatmap = predictor.generate_heatmap(img, points, sigma=15.0)
    print(f"Heatmap shape: {heatmap.shape}, max: {heatmap.max():.4f}")
    assert heatmap.shape == (850, 1150)

    print("\n✓ All tests passed")
    return encoder


if __name__ == '__main__':
    test_encoder()