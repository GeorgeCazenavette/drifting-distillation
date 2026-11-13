"""MoCo-v3 ViT feature extractor."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from ._moco_vit import VisionTransformerMoCoV3
from ._utils import imagenet_norm

# MoCo-v3 ViT architectures. Mirrors vit_base in the source repo (LTH14/rcg).
# Selected by substring match on ``model_name``.
_MOCOV3_ARCH: dict[str, dict] = {
    "vit-b": dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
}


def _arch_for(model_name: str) -> dict:
    for key, cfg in _MOCOV3_ARCH.items():
        if key in model_name:
            return dict(
                **cfg,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
    raise ValueError(
        f"Unsupported MoCo-v3 checkpoint: {model_name!r}. " f"Expected one of {list(_MOCOV3_ARCH)} as a substring."
    )


class MoCoV3FeatureExtractor(nn.Module):
    """Wraps a MoCo-v3 ViT (timm-based) and extracts final post-norm features.

    Token layout::

        [CLS, patch_0, patch_1, ..., patch_N]

    Position embedding is fixed 2-D sin-cos baked at construction for
    ``IMG_SIZE``×``IMG_SIZE`` inputs; ``forward`` raises ``ValueError`` for
    other input sizes.

    Supported modes
    ---------------
    ``"cls"``      — CLS token: ``{"cls": [N, hidden]}``.
    ``"cls_mean"`` — ``cat([CLS, mean(patches)])``: ``{"cls_mean": [N, 2*hidden]}``.
    ``"cls_grid"`` — CLS prepended to patch tokens as one group:
                     ``{"cls_grid": [N*(1+num_patches), hidden]}``.
    """

    IMG_SIZE = 224  # pos_embed is sincos baked at this resolution

    def __init__(self, model_name: str = "nyu-visionx/moco-v3-vit-b", mode: str = "cls"):
        super().__init__()
        if mode not in ("cls", "cls_mean", "cls_grid"):
            raise ValueError(f"MoCoV3FeatureExtractor does not support mode {mode!r}")

        cfg = _arch_for(model_name)
        self.model = VisionTransformerMoCoV3.from_pretrained(model_name, num_classes=0, **cfg)
        self.mode = mode

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from raw ``[0,1]`` images ``[N, C, IMG_SIZE, IMG_SIZE]``."""
        if x.shape[2] != self.IMG_SIZE or x.shape[3] != self.IMG_SIZE:
            raise ValueError(
                f"MoCo-v3 has fixed sin-cos pos_embed; input must be "
                f"{self.IMG_SIZE}×{self.IMG_SIZE}, got {tuple(x.shape[2:])}."
            )
        x = imagenet_norm(x)  # raw [0,1] -> ImageNet norm (extractor owns normalization)
        hs = self.model.forward_features(x)  # [N, 1+num_patches, hidden]

        if self.mode == "cls":
            return {"cls": hs[:, 0, :]}
        if self.mode == "cls_mean":
            return {"cls_mean": torch.cat([hs[:, 0, :], hs[:, 1:, :].mean(dim=1)], dim=1)}
        return {"cls_grid": rearrange(hs, "n l c -> (n l) c")}
