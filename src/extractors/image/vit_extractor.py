"""Torchvision ViT-B/16 feature extractor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from einops import rearrange

from ._utils import imagenet_norm


class ViTFeatureExtractor(nn.Module):
    """Wraps torchvision's vit_b_16 and extracts final-layer features.

    Token layout::

        [CLS, patch_0, patch_1, ..., patch_N]

    Supported modes
    ---------------
    ``"cls"``      — CLS token: ``{"cls": [N, hidden]}``.
    ``"cls_mean"`` — ``cat([CLS, mean(patches)])``: ``{"cls_mean": [N, 2*hidden]}``.
    ``"cls_grid"`` — CLS prepended to patch tokens as one group:
                     ``{"cls_grid": [N*(1+num_patches), hidden]}``.
    """

    def __init__(self, weights: str = "IMAGENET1K_V1", mode: str = "cls"):
        super().__init__()
        if mode not in ("cls", "cls_mean", "cls_grid"):
            raise ValueError(f"ViTFeatureExtractor does not support mode {mode!r}")

        w = tvm.ViT_B_16_Weights[weights]
        self.model = tvm.vit_b_16(weights=w)
        self.model.eval()
        self.mode = mode

        for p in self.model.parameters():
            p.requires_grad = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch embedding + CLS prepend + encoder. Returns encoder output."""
        N = x.shape[0]
        tokens = self.model._process_input(x)  # [N, num_patches, hidden]
        cls = self.model.class_token.expand(N, -1, -1)  # [N, 1, hidden]
        tokens = torch.cat([cls, tokens], dim=1)  # [N, 1+num_patches, hidden]
        return self.model.encoder(tokens)  # [N, 1+num_patches, hidden]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from raw ``[0,1]`` images ``[N, C, H, W]`` (H, W divisible by 16)."""
        x = imagenet_norm(x)  # raw [0,1] -> ImageNet norm (extractor owns normalization)
        enc_out = self._encode(x)  # [N, 1+num_patches, hidden]

        if self.mode == "cls":
            return {"cls": enc_out[:, 0, :]}

        if self.mode == "cls_mean":
            return {"cls_mean": torch.cat([enc_out[:, 0, :], enc_out[:, 1:, :].mean(dim=1)], dim=1)}

        # cls_grid: CLS + all patch tokens as one location group
        return {"cls_grid": rearrange(enc_out, "n l c -> (n l) c")}
