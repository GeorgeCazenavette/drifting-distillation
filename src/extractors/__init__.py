"""Frozen feature extractors.

Every extractor consumes raw ``[0, 1]`` inputs (it owns its own input
normalization) and returns ``{feature_name: Tensor[N * locs, C]}``; the drifting
loss is applied to each feature separately.

Image modes (distillation and evaluation):
    ``cls``       CLS token (ViT) / global-average-pooled vector (ResNet-18).
    ``cls_mean``  CLS token and mean patch token (two features).
    ``cls_grid``  CLS token and every patch token (ViT).
    ``cls_grid2`` pooled vector and the per-location vectors of the last two
                  stages (ResNet-18).
Video/audio modes (``videomaev2``, ``musicfm-fma``):
    ``mean_all``  mean token of every transformer layer (distillation, herding).
    ``mean``      mean token of the last layer (evaluation; VideoMAE-v2 applies
                  its ``fc_norm``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

BACKBONE_DEFAULTS: dict[str, str] = {
    # image
    "resnet18": "IMAGENET1K_V1",
    "vit": "IMAGENET1K_V1",
    "clip": "openai/clip-vit-base-patch16",
    "siglip2": "google/siglip2-base-patch16-224",
    "eva02": "eva02_base_patch14_224.mim_in22k",
    "mocov3": "nyu-visionx/moco-v3-vit-b",
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov2": "facebook/dinov2-base",  # not in the paper: the demo's fallback when DINOv3 is gated
    "eupe": "facebook/EUPE-ViT-B",
    # video / audio
    "videomaev2": "OpenGVLab/VideoMAEv2-Base",
    "musicfm-fma": "minzwon/MusicFM",
}


def get_extractor(backbone: str, mode: str = "cls", standardize: bool = False, compile: bool = False) -> nn.Module:
    """Instantiate a frozen extractor (not yet moved to a device)."""
    if backbone not in BACKBONE_DEFAULTS:
        raise ValueError(f"Unknown backbone {backbone!r}; choose from {sorted(BACKBONE_DEFAULTS)}")
    if standardize and backbone != "resnet18":
        raise ValueError("standardize is only supported for resnet18")
    name = BACKBONE_DEFAULTS[backbone]

    if backbone == "resnet18":
        import torchvision.models as tvm

        from .image.resnet_extractor import ResNetFeatureExtractor

        model = tvm.resnet18(weights=tvm.ResNet18_Weights[name]).eval()
        extractor = ResNetFeatureExtractor(model, mode=mode, standardize=standardize)
    elif backbone == "vit":
        from .image.vit_extractor import ViTFeatureExtractor

        extractor = ViTFeatureExtractor(weights=name, mode=mode)
    elif backbone == "clip":
        from .image.clip_extractor import CLIPFeatureExtractor

        extractor = CLIPFeatureExtractor(model_name=name, mode=mode)
    elif backbone == "siglip2":
        from .image.siglip2_extractor import SigLIP2FeatureExtractor

        extractor = SigLIP2FeatureExtractor(model_name=name, mode=mode)
    elif backbone == "eva02":
        from .image.eva02_extractor import EVA02FeatureExtractor

        extractor = EVA02FeatureExtractor(model_name=name, mode=mode)
    elif backbone == "mocov3":
        from .image.mocov3_extractor import MoCoV3FeatureExtractor

        extractor = MoCoV3FeatureExtractor(model_name=name, mode=mode)
    elif backbone.startswith("dinov"):
        # DINOv2 has no register tokens; the DINOv3 wrapper handles both.
        from .image.dinov3_extractor import DINOv3FeatureExtractor

        extractor = DINOv3FeatureExtractor(model_name=name, mode=mode)
    elif backbone == "eupe":
        from .image.eupe_extractor import EUPEFeatureExtractor

        extractor = EUPEFeatureExtractor(model_name=name, mode=mode)
    elif backbone == "videomaev2":
        from .video.videomaev2_extractor import VideoMAEv2Extractor

        extractor = VideoMAEv2Extractor(model_name=name, mode=mode)
    else:  # musicfm-fma
        from .audio.musicfm_extractor import MusicFMExtractor

        extractor = MusicFMExtractor(model_name=name, variant="fma", mode=mode)

    if compile:
        extractor = torch.compile(extractor)
    return extractor


__all__ = ["BACKBONE_DEFAULTS", "get_extractor"]
