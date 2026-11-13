"""Differentiable augmentation ops for ``(B, C, T)`` waveforms."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class RandomCrop1D(nn.Module):
    """Independent random crop of each clip to ``size`` samples (via ``gather``)."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, wav: Tensor) -> Tensor:
        N, C, T = wav.shape
        if T == self.size:
            return wav
        starts = torch.randint(0, T - self.size + 1, (N,), device=wav.device)
        offsets = torch.arange(self.size, device=wav.device)
        idx = rearrange(starts, "n -> n 1 1") + rearrange(offsets, "t -> 1 1 t")
        return wav.gather(2, idx.expand(N, C, self.size))
