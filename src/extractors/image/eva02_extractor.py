"""EVA-02 ViT (timm) feature extractor."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from einops import rearrange

from ._utils import CLIP_MEAN, CLIP_STD


class EVA02FeatureExtractor(nn.Module):
    """Wraps an EVA-02 ViT (timm) and extracts final post-norm features.

    Token layout::

        [CLS, patch_0, patch_1, ..., patch_N]

    Supported modes
    ---------------
    ``"cls"``      — CLS token: ``{"cls": [N, hidden]}``.
    ``"cls_mean"`` — ``cat([CLS, mean(patches)])``: ``{"cls_mean": [N, 2*hidden]}``.
    ``"cls_grid"`` — CLS prepended to patch tokens as one group:
                     ``{"cls_grid": [N*(1+num_patches), hidden]}``.
    """

    def __init__(self, model_name: str = "eva02_base_patch14_224.mim_in22k", mode: str = "cls"):
        super().__init__()
        if mode not in ("cls", "cls_mean", "cls_grid"):
            raise ValueError(f"EVA02FeatureExtractor does not support mode {mode!r}")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        # EVA-02 (mim_in22k) expects OpenAI-CLIP normalization (NOT ImageNet).
        self.register_buffer("_norm_mean", rearrange(torch.tensor(CLIP_MEAN), "c -> 1 c 1 1"), persistent=False)
        self.register_buffer("_norm_std", rearrange(torch.tensor(CLIP_STD), "c -> 1 c 1 1"), persistent=False)
        self.num_prefix_tokens: int = getattr(self.model, "num_prefix_tokens", 1)
        self.mode = mode

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from raw ``[0,1]`` images ``[N, C, H, W]`` (H, W divisible by patch size)."""
        x = (x - self._norm_mean.to(x.dtype)) / self._norm_std.to(x.dtype)  # raw [0,1] -> CLIP norm
        skip = self.num_prefix_tokens
        hs = self.model.forward_features(x)  # [N, skip+num_patches, hidden]

        if self.mode == "cls":
            return {"cls": hs[:, 0, :]}
        if self.mode == "cls_mean":
            return {"cls_mean": torch.cat([hs[:, 0, :], hs[:, skip:, :].mean(dim=1)], dim=1)}
        tokens = torch.cat([hs[:, :1, :], hs[:, skip:, :]], dim=1)
        return {"cls_grid": rearrange(tokens, "n l c -> (n l) c")}
