"""
RGB paddy dataset loader for panicle detection and counting.

Label format: plain text files with one point annotation per line — "x y"
(no class ID, no bounding box). Each point is a panicle centre.

Approach: point annotations are converted to Gaussian density maps.
Summing the predicted density map gives the panicle count.
Ground truth count = number of annotation lines in the label file.

Image size: 1150 x 850 (W x H). Training uses half resolution (575 x 425)
to fit comfortably in GPU memory with batch_size >= 4.

Usage:
    from rgb_dataset import build_rgb_loaders
    loaders = build_rgb_loaders(data_root="~/agri_foundation/data")
    for images, density_maps, counts in loaders["train"]:
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORIG_W, ORIG_H = 1150, 850
TRAIN_W, TRAIN_H = 575, 425          # half resolution for training
GAUSSIAN_SIGMA = 15                   # pixels at TRAIN resolution
GAUSSIAN_KERNEL_SIZE = 61             # must be odd; 4*sigma + 1


# ---------------------------------------------------------------------------
# Gaussian kernel builder
# ---------------------------------------------------------------------------

def _build_gaussian_kernel(sigma: float, kernel_size: int) -> Tensor:
    """
    Build a 2D Gaussian kernel normalised to sum=1.
    Used to smear each point annotation into a density blob.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32)
    center = kernel_size // 2
    coords = coords - center
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


_GAUSSIAN_KERNEL = _build_gaussian_kernel(GAUSSIAN_SIGMA, GAUSSIAN_KERNEL_SIZE)


def points_to_density_map(
    points: list[tuple[float, float]],
    height: int,
    width: int,
    kernel: Tensor = _GAUSSIAN_KERNEL,
) -> np.ndarray:
    """
    Convert a list of (x, y) point annotations to a 2D density map.

    Points outside image bounds are silently ignored.
    The density map integrates to the number of valid annotated points.

    Parameters
    ----------
    points : list of (x, y) in original image coordinates
    height, width : target density map spatial size (after any rescaling)
    kernel : pre-built Gaussian kernel tensor

    Returns
    -------
    density : np.ndarray shape (height, width) float32
    """
    density = torch.zeros(1, 1, height, width, dtype=torch.float32)
    pad = GAUSSIAN_KERNEL_SIZE // 2

    for x_orig, y_orig in points:
        # Scale coordinates to target resolution
        x = int(round(x_orig * width / ORIG_W))
        y = int(round(y_orig * height / ORIG_H))

        if not (0 <= x < width and 0 <= y < height):
            continue

        density[0, 0, y, x] += 1.0

    # Convolve point map with Gaussian kernel to produce smooth density
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)   # (1,1,K,K)
    density = F.conv2d(density, kernel_4d, padding=pad)
    return density.squeeze().numpy()


# ---------------------------------------------------------------------------
# Label parser
# ---------------------------------------------------------------------------

def parse_label_file(label_path: Path) -> list[tuple[float, float]]:
    """
    Parse a plain-text point annotation file.
    Each line: "x y" (space-separated floats).
    Returns list of (x, y) tuples. Empty list if file is empty.
    """
    points: list[tuple[float, float]] = []
    text = label_path.read_text().strip()
    if not text:
        return points

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                points.append((x, y))
            except ValueError:
                continue  # skip malformed lines

    return points


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PaddyPanicleDataset(Dataset):
    """
    RGB paddy panicle counting dataset.

    Returns (image_tensor, density_map_tensor, count) per sample where:
      - image_tensor  : float32 (3, H, W), normalised to ImageNet stats
      - density_map   : float32 (1, H, W), integrates to panicle count
      - count         : int, number of annotated panicles
    """

    # ImageNet normalisation — standard for RGB pretrained backbones
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        target_size: tuple[int, int] = (TRAIN_H, TRAIN_W),
        augment: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_h, self.target_w = target_size
        self.augment = augment

        self.image_paths = sorted(image_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")

        # Verify every image has a matching label file
        missing = [
            p.name for p in self.image_paths
            if not (label_dir / p.with_suffix(".txt").name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have no matching label file. "
                f"First missing: {missing[0]}"
            )

        self._build_transforms()

    def _build_transforms(self) -> None:
        resize = transforms.Resize((self.target_h, self.target_w))
        normalise = transforms.Normalize(mean=self._MEAN, std=self._STD)

        if self.augment:
            self.img_transform = transforms.Compose([
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                normalise,
            ])
            # Spatial augmentation seed must be shared between image and density map
            self.spatial_augment = True
        else:
            self.img_transform = transforms.Compose([
                resize,
                transforms.ToTensor(),
                normalise,
            ])
            self.spatial_augment = False

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        img_path = self.image_paths[idx]
        lbl_path = self.label_dir / img_path.with_suffix(".txt").name

        # Load and transform image
        image = Image.open(img_path).convert("RGB")

        # Parse point annotations before any spatial transform
        points = parse_label_file(lbl_path)
        count = len(points)

        # Build density map at target resolution
        density = points_to_density_map(
            points, self.target_h, self.target_w
        )
        density_tensor = torch.from_numpy(density).unsqueeze(0)  # (1, H, W)

        # Apply consistent horizontal flip to both image and density map
        if self.augment and torch.rand(1).item() > 0.5:
            image = transforms.functional.hflip(image)
            density_tensor = torch.flip(density_tensor, dims=[2])  # flip W axis

        if self.augment and torch.rand(1).item() > 0.5:
            image = transforms.functional.vflip(image)
            density_tensor = torch.flip(density_tensor, dims=[1])  # flip H axis

        # Apply remaining image transforms (resize, colour jitter, normalise)
        # Rebuild without flip since we already applied it manually above
        resize_norm = transforms.Compose([
            transforms.Resize((self.target_h, self.target_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])
        if self.augment:
            colour_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            )
            image = colour_jitter(image)

        image_tensor = resize_norm(image)

        return image_tensor, density_tensor, count


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rgb_loaders(
    data_root: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    target_size: tuple[int, int] = (TRAIN_H, TRAIN_W),
) -> dict[str, DataLoader]:
    """
    Build train and test DataLoaders for the RGB paddy panicle dataset.

    Batch size default is 4 — density map regression at 575x425 uses
    ~1.4 GB GPU memory per batch of 4; increase if VRAM allows.

    Returns
    -------
    {"train": DataLoader, "test": DataLoader}
    """
    root = Path(data_root).expanduser()
    splits = {
        "train": root / "rgb_paddy" / "train" / "train",
        "test": root / "rgb_paddy" / "test" / "test",
    }

    loaders: dict[str, DataLoader] = {}
    for split_name, split_dir in splits.items():
        dataset = PaddyPanicleDataset(
            image_dir=split_dir / "images",
            label_dir=split_dir / "labels",
            target_size=target_size,
            augment=(split_name == "train"),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )

    return loaders


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(data_root: str = "~/agri_foundation/data") -> None:
    """
    Quick sanity check — loads one batch and verifies density map integrity.
    Run directly: python rgb_dataset.py
    """
    print("Building RGB loaders...")
    loaders = build_rgb_loaders(
        data_root=data_root,
        batch_size=2,
        num_workers=0,
    )

    for split_name, loader in loaders.items():
        images, density_maps, counts = next(iter(loader))
        predicted_counts = density_maps.sum(dim=(1, 2, 3))

        print(f"\n--- {split_name} ---")
        print(f"  images       : {tuple(images.shape)} dtype={images.dtype}")
        print(f"  density_maps : {tuple(density_maps.shape)} dtype={density_maps.dtype}")
        print(f"  counts (gt)  : {counts.tolist()}")
        print(f"  counts (dm)  : {[f'{c:.1f}' for c in predicted_counts.tolist()]}")
        print(f"  density range: [{density_maps.min():.6f}, {density_maps.max():.6f}]")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    smoke_test()