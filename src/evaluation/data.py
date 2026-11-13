"""Shared data plumbing for the evaluations.

A distilled set is an ImageFolder directory (``{data}/{class:04d}/*.png``) whose
sorted subfolder order is the dataset's class index.  Real data lives under
``--data_root`` in the official layouts::

    {data_root}/imagenet/{train,val}/<wnid>/...      ImageNet-1k
    {data_root}/imagenette2/{train,val}/<wnid>/...   ImageNette (official split)
    {data_root}/imagewoof2/{train,val}/<wnid>/...    ImageWoof (official split)

ImageNette and ImageWoof are downloaded there on first use (``src/download.py``).

Sorted wnid folders give torchvision's class indices, matching the distilled sets.
"""

from __future__ import annotations

import os

from ..download import ARCHIVES, ensure_archive

DATASET_DIRS = {"imagenet-1k": "imagenet", "imagenette": "imagenette2", "imagewoof": "imagewoof2"}
NUM_CLASSES = {"imagenet-1k": 1000, "imagenette": 10, "imagewoof": 10}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def real_dir(data_root: str, dataset: str, split: str = "val") -> str:
    """ImageFolder directory of a real split (ImageNette/ImageWoof downloaded if missing)."""
    if dataset in ARCHIVES:
        ensure_archive(dataset, data_root)
    return os.path.join(data_root, DATASET_DIRS[dataset], split)


def check_distilled(path: str, dataset: str) -> int:
    """Fail fast unless ``path`` has the dataset's class count and the same number
    of images in every class; return that number (the IPC)."""
    if not os.path.isdir(path):
        raise SystemExit(f"[distilled] dir missing: {path}")
    class_dirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    counts = [
        sum(1 for f in os.listdir(os.path.join(path, c)) if os.path.splitext(f)[1].lower() in _IMG_EXTS)
        for c in class_dirs
    ]
    if len(class_dirs) != NUM_CLASSES[dataset]:
        raise SystemExit(
            f"[distilled] expected {NUM_CLASSES[dataset]} classes for {dataset}, " f"found {len(class_dirs)} in {path}"
        )
    if len(set(counts)) != 1 or counts[0] == 0:
        raise SystemExit(
            f"[distilled] unequal/empty classes in {path}: " f"min {min(counts)}, max {max(counts)} images/class"
        )
    print(f"[distilled] OK: {len(class_dirs)} classes x {counts[0]} = {sum(counts)} images", flush=True)
    return counts[0]
