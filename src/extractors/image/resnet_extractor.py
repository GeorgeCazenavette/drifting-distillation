"""Torchvision ResNet feature extractor (global-average-pooled vector and grids)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from ._utils import imagenet_norm


class ResNetFeatureExtractor(nn.Module):
    """Wraps a torchvision ResNet and extracts features.

    Supported modes
    ---------------
    ``"cls"``       — global average pool (GAP) of the final stage: ``{"cls": [N, C]}``.
    ``"cls_grid2"`` — GAP prepended to the final stage's 7×7 grid as one group
                      (``{"cls_grid": [N*(1+49), C]}``) plus the penultimate
                      stage's 14×14 grid (``{"grid2": [N*196, C']}``; different
                      channel count, so it is a separate key).

    ``standardize`` standardizes every returned feature vector across channels
    (LayerNorm without affine), giving every latent row a constant norm (√C),
    as post-LN ViT latents have architecturally.
    """

    def __init__(self, backbone: nn.Module, mode: str = "cls", standardize: bool = False):
        super().__init__()
        if mode not in ("cls", "cls_grid2"):
            raise ValueError(f"ResNetFeatureExtractor does not support mode {mode!r}")
        self.backbone = backbone
        self.mode = mode
        self.standardize = standardize

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from raw ``[0,1]`` images ``[N, C, H, W]``."""
        x = imagenet_norm(x)  # raw [0,1] -> ImageNet norm (extractor owns normalization)
        features = self._extract(x)
        if self.standardize:
            features = {k: F.layer_norm(v, v.shape[-1:]) for k, v in features.items()}
        return features

    def _extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        penult_feat = self.backbone.layer3(x)  # [N, C', 14, 14]
        final_feat = self.backbone.layer4(penult_feat)  # [N, C, 7, 7]
        gap = reduce(final_feat, "n c h w -> n c", "mean")  # raw global-average-pool

        if self.mode == "cls":
            return {"cls": gap}

        # cls_grid2: GAP prepended as location 0 (50 = 1+49), mirroring the ViT
        # CLS+patches convention, plus the penultimate grid as its own key.
        grid = rearrange(final_feat, "n c h w -> n (h w) c")  # [N, HW, C]
        fused = torch.cat([gap.unsqueeze(1), grid], dim=1)  # [N, 1+HW, C]
        return {
            "cls_grid": rearrange(fused, "n l c -> (n l) c"),
            "grid2": rearrange(penult_feat, "n c h w -> (n h w) c"),
        }
