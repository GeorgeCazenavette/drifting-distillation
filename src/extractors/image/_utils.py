"""Helpers shared by the image extractors."""

from __future__ import annotations

import torch
from einops import rearrange

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# OpenAI-CLIP normalization (used by CLIP and EVA-02).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet input normalization to a raw ``[0,1]`` image batch."""
    mean = rearrange(torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype), "c -> 1 c 1 1")
    std = rearrange(torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype), "c -> 1 c 1 1")
    return (x - mean) / std
