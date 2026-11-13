"""Differentiable, batched augmentation ops.

All ops accept ``(B, C, H, W)`` images or ``(B, T, C, H, W)`` clips; for clips
the same random transform is applied to every frame of a clip.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class CenterCrop(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = (size, size)

    def forward(self, images: Tensor) -> Tensor:
        H, W = images.shape[-2], images.shape[-1]
        out_h, out_w = self.size
        h0 = (H - out_h) // 2
        w0 = (W - out_w) // 2
        return images[..., h0 : h0 + out_h, w0 : w0 + out_w]


class Resize(nn.Module):
    """Deterministic (antialiased bilinear) resize of the whole image."""

    def __init__(self, size: int):
        super().__init__()
        self.size = (size, size)

    def forward(self, images: Tensor) -> Tensor:
        video = images.ndim == 5
        if video:
            B = images.shape[0]
            images = rearrange(images, "b t c h w -> (b t) c h w")
        out = F.interpolate(images, size=self.size, mode="bilinear", align_corners=False, antialias=True)
        return rearrange(out, "(b t) c h w -> b t c h w", b=B) if video else out


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, images: Tensor) -> Tensor:
        B = images.shape[0]
        with torch.no_grad():
            flip = torch.rand(B, device=images.device) < self.p
        if not flip.any():
            return images
        images = images.clone()
        images[flip] = torch.flip(images[flip], dims=[-1])
        return images


class RandomResizedCrop(nn.Module):
    """Differentiable random resized crop via ``affine_grid``/``grid_sample``.

    Args:
        size: Output side length.
        scale: Range of the crop's area fraction.
        ratio: Range of the crop's aspect ratio (sampled log-uniformly).
    """

    def __init__(
        self, size: int, scale: tuple[float, float] = (0.08, 1.0), ratio: tuple[float, float] = (3 / 4, 4 / 3)
    ):
        super().__init__()
        self.size = (size, size)
        self.scale = scale
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    def forward(self, images: Tensor) -> Tensor:
        video = images.ndim == 5
        if video:
            B, T, C, H, W = images.shape
            images = rearrange(images, "b t c h w -> (b t) c h w")
        else:
            B, C, H, W = images.shape
            T = 1
        device = images.device

        with torch.no_grad():
            r = torch.rand(B, 4, device=device)
            scales = r[:, 0] * (self.scale[1] - self.scale[0]) + self.scale[0]
            ratios = torch.exp(r[:, 1] * (self.log_ratio[1] - self.log_ratio[0]) + self.log_ratio[0])
            scale_w = torch.sqrt(scales * ratios)
            scale_h = torch.sqrt(scales / ratios)
            x1 = r[:, 2] * (1 - scale_w)
            y1 = r[:, 3] * (1 - scale_h)
            offset_x = (x1 + scale_w / 2 - 0.5) * 2
            offset_y = (y1 + scale_h / 2 - 0.5) * 2

        theta = torch.zeros(B, 2, 3, dtype=images.dtype, device=device)
        theta[:, 0, 0] = scale_w
        theta[:, 1, 1] = scale_h
        theta[:, 0, 2] = offset_x
        theta[:, 1, 2] = offset_y
        if video:
            theta = theta.repeat_interleave(T, dim=0)

        grid = F.affine_grid(theta, (B * T, C, *self.size), align_corners=False)
        out = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return rearrange(out, "(b t) c h w -> b t c h w", b=B, t=T) if video else out


class RandomGaussianNoise(nn.Module):
    """Adds ``N(0, std^2)`` noise to each sample with probability ``p``."""

    def __init__(self, std: float = 0.1, p: float = 0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, images: Tensor) -> Tensor:
        B = images.shape[0]
        with torch.no_grad():
            apply = torch.rand(B, device=images.device) < self.p
        if apply.any():
            noise = torch.randn_like(images) * self.std
            mask = apply.view(B, *([1] * (images.ndim - 1))).float()
            images = images + mask * noise
        return images
