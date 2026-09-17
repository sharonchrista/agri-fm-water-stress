"""
Deep metadata inspection for agri_foundation.
Run from project root: python inspect_meta.py
Reads all JSON metadata in full and inspects HSI processed arrays completely.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


BASE_DIR = Path("~/agri_foundation/data").expanduser()
HSI_PROC_DIR = BASE_DIR / "processed" / "hsi"
META_DIR = BASE_DIR / "processed" / "metadata"
RGB_DIR = BASE_DIR / "rgb_paddy"
HSI_RAW_DIR = BASE_DIR / "hyperspectral"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_json(path: Path) -> dict | list | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"  WARNING: {path.name} — {exc}")
        return None


def load_npy(path: Path) -> np.ndarray | None:
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"  WARNING: {path.name} — {exc}")
        return None


def describe_array(arr: np.ndarray, name: str) -> None:
    print(f"  {name}")
    print(f"    shape  : {arr.shape}")
    print(f"    dtype  : {arr.dtype}")
    if np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.bytes_):
        unique_vals = np.unique(arr.ravel())
        print(f"    unique ({len(unique_vals)} total) : {list(unique_vals[:10])}")
        return
    print(f"    range  : [{arr.min():.4f}, {arr.max():.4f}]")
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        flat = arr.ravel()
        unique, counts = np.unique(flat, return_counts=True)
        if len(unique) <= 30:
            dist = {str(u): int(c) for u, c in zip(unique, counts)}
            print(f"    classes: {dist}")
        else:
            print(f"    unique values: {len(unique)} (continuous)")


# ---------------------------------------------------------------------------
# 1. Full JSON dumps
# ---------------------------------------------------------------------------

def inspect_all_json() -> None:
    section("FULL JSON CONTENTS")

    json_files = sorted(META_DIR.glob("*.json"))
    for jpath in json_files:
        data = load_json(jpath)
        if data is None:
            continue
        print(f"\n{'─' * 55}")
        print(f"  {jpath.name}")
        print(f"{'─' * 55}")
        print(json.dumps(data, indent=2)[:4000])  # cap at 4000 chars per file
        if len(json.dumps(data)) > 4000:
            print("  ... [truncated — file larger than 4000 chars]")


# ---------------------------------------------------------------------------
# 2. HSI processed arrays — full inspection
# ---------------------------------------------------------------------------

def inspect_hsi_processed() -> None:
    section("HSI PROCESSED ARRAYS (processed/hsi/)")

    files = sorted(HSI_PROC_DIR.glob("*.npy")) if HSI_PROC_DIR.exists() else []
    if not files:
        print(f"  No files found in {HSI_PROC_DIR}")
        return

    for fpath in files:
        arr = load_npy(fpath)
        if arr is not None:
            describe_array(arr, fpath.name)


# ---------------------------------------------------------------------------
# 3. HSI raw README content
# ---------------------------------------------------------------------------

def inspect_hsi_readme() -> None:
    section("HSI RAW README (groundnut)")

    readme_candidates = list(HSI_RAW_DIR.rglob("*.docx")) + \
                        list(HSI_RAW_DIR.rglob("*.txt")) + \
                        list(HSI_RAW_DIR.rglob("*.md"))

    if not readme_candidates:
        print("  No documentation files found.")
        return

    for doc in readme_candidates:
        print(f"\n  Found: {doc}")
        if doc.suffix == ".txt" or doc.suffix == ".md":
            try:
                print(doc.read_text()[:2000])
            except Exception as exc:
                print(f"  Could not read: {exc}")
        else:
            print("  .docx — open manually to read label descriptions")


# ---------------------------------------------------------------------------
# 4. RGB paddy — look inside train/test more carefully
# ---------------------------------------------------------------------------

def inspect_rgb_deep() -> None:
    section("RGB PADDY — DEEP INSPECTION")

    for split in ["train", "test"]:
        split_dir = RGB_DIR / split
        print(f"\n--- {split}/ ---")
        if not split_dir.exists():
            print(f"  Not found: {split_dir}")
            continue

        all_items = list(split_dir.iterdir())
        print(f"  Items in {split}/: {[x.name for x in all_items[:20]]}")

        # Walk one level deeper if subdirs exist
        for item in all_items:
            if item.is_dir():
                sub_items = list(item.iterdir())
                images = [f for f in sub_items if f.suffix.lower()
                          in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
                print(f"    subdir '{item.name}': {len(sub_items)} items, "
                      f"{len(images)} images")
                if images:
                    print(f"      sample: {images[0].name}")


# ---------------------------------------------------------------------------
# 5. Tasks list from manifest
# ---------------------------------------------------------------------------

def inspect_tasks() -> None:
    section("TASKS FROM DATASET MANIFEST")

    manifest_path = META_DIR / "dataset_manifest.json"
    data = load_json(manifest_path)
    if data is None:
        return

    tasks = data.get("tasks", [])
    print(f"  {len(tasks)} tasks defined:")
    for i, task in enumerate(tasks, 1):
        print(f"\n  Task {i}:")
        if isinstance(task, dict):
            for k, v in task.items():
                print(f"    {k}: {v}")
        else:
            print(f"    {task}")

    datasets = data.get("datasets", {})
    print(f"\n  Datasets block:")
    if isinstance(datasets, dict):
        for k, v in datasets.items():
            print(f"    {k}: {v}")
    elif isinstance(datasets, list):
        for item in datasets:
            print(f"    {item}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    inspect_tasks()
    inspect_hsi_processed()
    inspect_hsi_readme()
    inspect_rgb_deep()
    inspect_all_json()


if __name__ == "__main__":
    main()