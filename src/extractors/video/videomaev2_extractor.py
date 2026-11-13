"""VideoMAE-v2 feature extractor (``OpenGVLab/VideoMAEv2-Base``).

Modes:
    ``mean_all``  mean token of every transformer block,
                  ``{"mean_layer{i}": [N, C]}`` (distillation and herding).
    ``mean``      mean token of the last block through ``fc_norm`` (the
                  model's classification pooling), ``{"mean": [N, C]}``
                  (evaluation).

Input: ``[N, T, 3, H, W]`` raw ``[0, 1]`` clips, 16 frames at 224 px.  The
model's ``preprocessor_config.json`` uses ImageNet mean/std, applied here.

Loading: the hub's ``PreTrainedModel`` wrapper calls ``Tensor.item()`` in
``__init__``, which breaks under newer transformers' meta-device init, so the
``VisionTransformer`` is imported straight from the snapshot directory and the
safetensors weights are loaded into it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_vision_transformer(snapshot_dir: Path) -> type[nn.Module]:
    """Import ``VisionTransformer`` from the snapshot's ``modeling_videomaev2.py``.

    That file uses a package-relative import (``from .modeling_config import
    ...``), so a synthetic parent package mapped to the snapshot is registered
    first.
    """
    pkg_name = f"_videomaev2_pkg_{snapshot_dir.name}"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(snapshot_dir)]
    pkg.__package__ = pkg_name
    pkg.__spec__ = None  # type: ignore[assignment]
    sys.modules[pkg_name] = pkg

    def _load_submodule(stem: str) -> types.ModuleType:
        name = f"{pkg_name}.{stem}"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            name,
            snapshot_dir / f"{stem}.py",
            submodule_search_locations=[str(snapshot_dir)],
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        mod.__package__ = pkg_name
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    _load_submodule("modeling_config")
    return _load_submodule("modeling_videomaev2").VisionTransformer


class VideoMAEv2Extractor(nn.Module):
    def __init__(self, model_name: str = "OpenGVLab/VideoMAEv2-Base", mode: str = "mean_all"):
        super().__init__()
        if mode not in ("mean", "mean_all"):
            raise ValueError(f"VideoMAEv2Extractor: unknown mode {mode!r}; choose mean or mean_all.")
        self.mode = mode

        from huggingface_hub import snapshot_download

        snapshot_dir = Path(snapshot_download(model_name))
        with open(snapshot_dir / "config.json") as f:
            mc: dict = json.load(f)["model_config"]
        self.vit: nn.Module = _load_vision_transformer(snapshot_dir)(**mc)

        from safetensors.torch import load_file

        state = load_file(str(snapshot_dir / "model.safetensors"))
        # Keys carry the PreTrainedModel wrapper's leading "model." prefix.
        state = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in state.items()}
        missing, _ = self.vit.load_state_dict(state, strict=False)
        if missing:
            warnings.warn(f"VideoMAEv2Extractor: missing keys: {missing[:5]}", stacklevel=2)

        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)

    def _blocks(self, x: Tensor) -> list[Tensor]:
        """Token sequence ``[N, L, C]`` after every block, for ``[N, 3, T, H, W]`` input."""
        vit = self.vit
        x = vit.patch_embed(x)
        if vit.pos_embed is not None:
            x = x + vit.pos_embed.expand(x.size(0), -1, -1).type_as(x).to(x.device).clone().detach()
        x = vit.pos_drop(x)
        hidden = []
        for blk in vit.blocks:
            x = blk(x)
            hidden.append(x)
        return hidden

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        mean = rearrange(torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype), "c -> 1 1 c 1 1")
        std = rearrange(torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype), "c -> 1 1 c 1 1")
        x = (x - mean) / std
        hidden = self._blocks(rearrange(x, "n t c h w -> n c t h w"))
        if self.mode == "mean":
            return {"mean": self.vit.fc_norm(hidden[-1].mean(dim=1))}
        return {f"mean_layer{i}": h.mean(dim=1) for i, h in enumerate(hidden)}
