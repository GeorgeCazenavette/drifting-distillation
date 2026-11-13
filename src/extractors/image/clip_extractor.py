"""CLIP vision ViT feature extractor."""

from __future__ import annotations

import torch
import torch.nn as nn
import transformers
from einops import rearrange
from transformers import CLIPModel

from ._utils import CLIP_MEAN, CLIP_STD


class CLIPFeatureExtractor(nn.Module):
    """Wraps the vision encoder of a CLIP model and extracts final-layer features.

    Token layout::

        [CLS, patch_0, patch_1, ..., patch_N]

    Supported modes
    ---------------
    ``"cls"``      — Projected CLS embedding (via ``visual_projection``):
                     ``{"cls": [N, proj_dim]}``.
    ``"cls_mean"`` — ``cat([CLS, mean(patches)])`` from last hidden state:
                     ``{"cls_mean": [N, 2*hidden]}``.
    ``"cls_grid"`` — CLS prepended to patch tokens as one group:
                     ``{"cls_grid": [N*(1+num_patches), hidden]}``.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", mode: str = "cls"):
        super().__init__()
        if mode not in ("cls", "cls_mean", "cls_grid"):
            raise ValueError(f"CLIPFeatureExtractor does not support mode {mode!r}")

        _prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        _full = CLIPModel.from_pretrained(model_name)
        transformers.logging.set_verbosity(_prev_verbosity)
        self.model = _full.vision_model  # CLIPVisionTransformer
        self.visual_projection = _full.visual_projection  # Linear(hidden, proj_dim)
        # CLIP expects OpenAI-CLIP normalization (NOT ImageNet).
        self.register_buffer("_norm_mean", rearrange(torch.tensor(CLIP_MEAN), "c -> 1 c 1 1"), persistent=False)
        self.register_buffer("_norm_std", rearrange(torch.tensor(CLIP_STD), "c -> 1 c 1 1"), persistent=False)
        self.mode = mode

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.visual_projection.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from raw ``[0,1]`` images ``[N, C, H, W]``."""
        x = (x - self._norm_mean.to(x.dtype)) / self._norm_std.to(x.dtype)  # raw [0,1] -> CLIP norm
        outputs = self.model(pixel_values=x)
        hs = outputs.last_hidden_state  # [N, 1+num_patches, hidden]

        if self.mode == "cls":
            # Match clip.load().visual convention: projected CLS
            return {"cls": self.visual_projection(outputs.pooler_output)}

        if self.mode == "cls_mean":
            return {"cls_mean": torch.cat([hs[:, 0, :], hs[:, 1:, :].mean(dim=1)], dim=1)}

        # cls_grid: CLS + all patch tokens as one location group
        return {"cls_grid": rearrange(hs, "n l c -> (n l) c")}
