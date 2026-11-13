"""EUPE-ViT feature extractor.

Loads via torch.hub from facebookresearch/EUPE with weights from HuggingFace.
The model uses a custom DinoVisionTransformer with fused QKV projections, RoPE
embeddings, and 4 storage tokens prepended after the CLS token.

Token layout: [CLS, storage_0..storage_k, patch_0..patch_N]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import hf_hub_download

from ._utils import imagenet_norm

_HUB_ENTRY: dict[str, str] = {
    "facebook/EUPE-ViT-B": "eupe_vitb16",
}

_HUB_FILENAME: dict[str, str] = {
    "facebook/EUPE-ViT-B": "EUPE-ViT-B.pt",
}


class EUPEFeatureExtractor(nn.Module):
    """Wraps an EUPE ViT and extracts final-layer (normalized) features.

    Supported modes
    ---------------
    ``"cls"``      — CLS token: ``{"cls": [N, hidden]}``.
    ``"cls_mean"`` — ``cat([CLS, mean(patches)])``: ``{"cls_mean": [N, 2*hidden]}``.
    ``"cls_grid"`` — CLS prepended to patch tokens as one group:
                     ``{"cls_grid": [N*(1+num_patches), hidden]}``.
    """

    def __init__(self, model_name: str = "facebook/EUPE-ViT-B", mode: str = "cls"):
        super().__init__()
        if mode not in ("cls", "cls_mean", "cls_grid"):
            raise ValueError(f"EUPEFeatureExtractor does not support mode {mode!r}")
        if model_name not in _HUB_ENTRY:
            raise ValueError(f"Unknown EUPE model {model_name!r}. " f"Choose from: {sorted(_HUB_ENTRY)}")

        ckpt_path = hf_hub_download(model_name, _HUB_FILENAME[model_name])
        self.model = torch.hub.load(
            "facebookresearch/EUPE:main",
            _HUB_ENTRY[model_name],
            weights=ckpt_path,
            source="github",
            verbose=False,
        )
        self.mode = mode

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from raw ``[0,1]`` images ``[N, C, H, W]``."""
        x = imagenet_norm(x)  # raw [0,1] -> ImageNet norm (extractor owns normalization)
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]  # [N, hidden]
        patches = out["x_norm_patchtokens"]  # [N, num_patches, hidden]

        if self.mode == "cls":
            return {"cls": cls}

        if self.mode == "cls_mean":
            return {"cls_mean": torch.cat([cls, patches.mean(dim=1)], dim=1)}

        # cls_grid: CLS + patches as one flat group
        combined = torch.cat([cls.unsqueeze(1), patches], dim=1)
        return {"cls_grid": rearrange(combined, "n l c -> (n l) c")}
