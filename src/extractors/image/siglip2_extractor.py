"""SigLIP2 vision ViT feature extractor.

The fixed-resolution SigLIP2 checkpoints (e.g. ``google/siglip2-base-patch16-224``)
ship under the original SigLIP architecture (``model_type: siglip``), so they are
loaded via ``SiglipModel``.

Unlike CLIP/DINO, SigLIP has **no CLS token**: ``last_hidden_state`` holds only
patch tokens, and the global image embedding is produced by a multi-head
attention-pooling head (``pooler_output``).  This is SigLIP's canonical image
feature (there is no separate ``visual_projection``).  The pooled head output is
therefore used wherever a CLS token would be.

SigLIP also normalizes inputs to ``[-1, 1]`` (mean/std = 0.5), **not** ImageNet
stats; the extractor consumes raw ``[0, 1]`` images and applies this internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import transformers
from einops import rearrange
from transformers import SiglipModel


class SigLIP2FeatureExtractor(nn.Module):
    """Wraps the vision encoder of a SigLIP2 model and extracts final-layer features.

    Token layout (no CLS, no register tokens)::

        [patch_0, patch_1, ..., patch_N]

    Supported modes
    ---------------
    ``"cls"``      — Pooled image embedding (attention-pooling head):
                     ``{"cls": [N, hidden]}``.
    ``"cls_mean"`` — ``cat([pooled, mean(patches)])``:
                     ``{"cls_mean": [N, 2*hidden]}``.
    ``"cls_grid"`` — Pooled token prepended to the patch grid as one group:
                     ``{"cls_grid": [N*(1+num_patches), hidden]}``.
    """

    # SigLIP normalization: x in [0, 1] -> (x - 0.5) / 0.5 = 2x - 1.
    _NORM_MEAN = 0.5
    _NORM_STD = 0.5

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", mode: str = "cls"):
        super().__init__()
        if mode not in ("cls", "cls_mean", "cls_grid"):
            raise ValueError(f"SigLIP2FeatureExtractor does not support mode {mode!r}")

        # The fixed-res SigLIP2 checkpoints store a nested SiglipConfig
        # (vision + text); SiglipVisionModel mis-resolves it, so load the full
        # model and keep only the vision tower (the text tower is freed when the
        # temporary goes out of scope).
        _prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        _full = SiglipModel.from_pretrained(model_name)
        transformers.logging.set_verbosity(_prev_verbosity)
        self.model = _full.vision_model  # SiglipVisionTransformer
        self.mode = mode

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from raw ``[0,1]`` images ``[N, C, H, W]``."""
        x = (x - self._NORM_MEAN) / self._NORM_STD  # raw [0,1] -> [-1,1]
        outputs = self.model(pixel_values=x)
        hs = outputs.last_hidden_state  # [N, num_patches, hidden]
        pooled = outputs.pooler_output  # [N, hidden] (attention-pooling head)

        if self.mode == "cls":
            return {"cls": pooled}

        if self.mode == "cls_mean":
            return {"cls_mean": torch.cat([pooled, hs.mean(dim=1)], dim=1)}

        # cls_grid: pooled token prepended to the patch grid as one group.
        grid = torch.cat([pooled.unsqueeze(1), hs], dim=1)  # [N, 1+num_patches, hidden]
        return {"cls_grid": rearrange(grid, "n l c -> (n l) c")}
