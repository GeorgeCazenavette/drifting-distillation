"""DINOv3 ViT feature extractor."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModel

from ._utils import imagenet_norm


class DINOv3FeatureExtractor(nn.Module):
    """Wraps a DINOv3 ViT and extracts final-layer (post-LN) features.

    Token layout in each hidden state::

        [CLS, reg_0, ..., reg_n, patch_0, ..., patch_N]

    Patch tokens are ``hs[:, 1 + num_register_tokens:, :]``; register tokens are
    excluded from all patch-based features.

    Supported modes
    ---------------
    ``"cls"``      — CLS token: ``{"cls": [N, hidden]}``.
    ``"cls_mean"`` — ``cat([CLS, mean(patches)])``: ``{"cls_mean": [N, 2*hidden]}``.
    ``"cls_grid"`` — CLS prepended to patch tokens as one group:
                     ``{"cls_grid": [N*(1+num_patches), hidden]}``.
    """

    def __init__(self, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", mode: str = "cls"):
        super().__init__()
        if mode not in ("cls", "cls_mean", "cls_grid"):
            raise ValueError(f"DINOv3FeatureExtractor does not support mode {mode!r}")

        self.model = AutoModel.from_pretrained(model_name)
        self.num_register_tokens: int = getattr(self.model.config, "num_register_tokens", 0)
        self.mode = mode

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from raw ``[0,1]`` images ``[N, C, H, W]``."""
        x = imagenet_norm(x)  # raw [0,1] -> ImageNet norm (extractor owns normalization)
        skip = 1 + self.num_register_tokens  # index of first patch token
        hs = self.model(pixel_values=x).last_hidden_state  # [N, 1+nreg+num_patches, hidden], post-LN

        if self.mode == "cls":
            return {"cls": hs[:, 0, :]}

        if self.mode == "cls_mean":
            return {"cls_mean": torch.cat([hs[:, 0, :], hs[:, skip:, :].mean(dim=1)], dim=1)}

        # cls_grid: CLS + patch tokens (register tokens excluded) as one location group
        tokens = torch.cat([hs[:, :1, :], hs[:, skip:, :]], dim=1)
        return {"cls_grid": rearrange(tokens, "n l c -> (n l) c")}
