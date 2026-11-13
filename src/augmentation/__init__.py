"""Augmentation schemes used during distillation.

* ``none``     — center crop to ``crop_res`` (used on the real images).
* ``resize``   — one deterministic resized view (the "None" row of the
                 augmentations-per-step ablation).
* ``standard`` — horizontal flip, random resized crop, Gaussian noise (the
                 paper's ``P_A``).
* ``audio``    — random 1-D crop to ``crop_res`` samples + Gaussian noise
                 (waveforms ``(B, C, T)``).
"""

from __future__ import annotations

import torch.nn as nn

from .ops import CenterCrop, RandomGaussianNoise, RandomHorizontalFlip, RandomResizedCrop, Resize
from .ops_1d import RandomCrop1D


def build_aug(
    name: str,
    crop_res: int,
    noise_std: float = 0.1,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
) -> nn.Module:
    if name == "none":
        return CenterCrop(crop_res)
    if name == "resize":
        return Resize(crop_res)
    if name == "standard":
        return nn.Sequential(
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop(crop_res, scale=scale, ratio=ratio),
            RandomGaussianNoise(std=noise_std),
        )
    if name == "audio":
        return nn.Sequential(RandomCrop1D(crop_res), RandomGaussianNoise(std=noise_std))
    raise ValueError(f"Unknown augmentation {name!r}; choose from none, resize, standard, audio.")


__all__ = ["build_aug"]
