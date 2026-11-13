"""Real images of one class, read from an ImageFolder-style directory.

``root`` holds one subdirectory per class; class ``k`` is the ``k``-th
subdirectory in sorted order.  This matches torchvision's class indices for
ImageNet (``train/`` with sorted wnid folders) and the official ImageNette /
ImageWoof releases (``imagenette2/train``, ``imagewoof2/train``), and works for
any other folder of images (e.g. one folder per painter).
"""

from __future__ import annotations

import os

import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS


def class_dirs(root: str) -> list[str]:
    return sorted(e.name for e in os.scandir(root) if e.is_dir())


def class_image_paths(root: str, class_idx: int) -> list[str]:
    """Sorted image paths of class ``class_idx`` under ``root``."""
    d = os.path.join(root, class_dirs(root)[class_idx])
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXTENSIONS))


class ClassFolder(Dataset):
    """All images of one class, resized (shorter side) and center-cropped to ``res``.

    Returns ``(image [3, res, res] in [0, 1], class_idx)``.
    """

    def __init__(self, root: str, class_idx: int, res: int = 256):
        self.paths = class_image_paths(root, class_idx)
        self.class_idx = class_idx
        self.transform = T.Compose([T.Resize(res), T.CenterCrop(res), T.ToTensor()])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[Tensor, int]:
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.class_idx
