"""Spatiotemporal pyramid parameterization of distilled video clips.

Video analogue of :class:`~.image_pyramid.ImagePyramid`.  Each level is a
learnable ``(num_clips, 3, t, h, h)`` tensor; a clip is decoded by trilinearly
upsampling every level to ``(num_frames, syn_res, syn_res)``, summing, applying
the fixed color decorrelation, and squashing with ``sigmoid(2x)``.

Each :meth:`extend_pyramid` call grows the pyramid in one of two ways:

* ``growth="alternate"``: add a level with doubled spatial
  resolution, then one with doubled temporal resolution.  With ``start_res=8``,
  ``start_frames=1``, 16 frames at 224 px and a call every 200 steps, the
  pyramid grows from 4 to 13 levels by step 1000.
* ``growth="joint"`` (the paper's run): add one level with both axes doubled.

Each axis stops at its target resolution.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .image_pyramid import linear_decorrelate_color
from .matmul_upsample import MatmulUpsample3D


class VideoPyramid:
    """Coarse-to-fine pyramid of learnable clips.

    Args:
        num_clips: Number of distilled clips.
        num_frames: Output frames per clip.
        syn_res: Output spatial resolution.
        start_res: Finest initial spatial level (power of 2).
        start_frames: Finest initial temporal level (power of 2).
        lr: Adam learning rate.
        decorrelate_color: Apply the color decorrelation in the decoder.
        growth: ``"alternate"`` (spatial level, then temporal level per
            growth step) or ``"joint"`` (one level doubling both axes).
        device: Torch device.
    """

    def __init__(
        self,
        num_clips: int,
        num_frames: int,
        syn_res: int = 224,
        start_res: int = 8,
        start_frames: int = 1,
        lr: float = 0.001,
        decorrelate_color: bool = True,
        growth: str = "joint",
        device: str | torch.device = "cuda",
    ) -> None:
        if growth not in ("alternate", "joint"):
            raise ValueError(f"growth must be 'alternate' or 'joint', got {growth!r}")
        self.growth = growth
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.syn_res = syn_res
        self.start_res = start_res
        self.start_frames = start_frames
        self.lr = lr
        self.decorrelate_color = decorrelate_color
        self.device = device

        # Per-axis upsample operators, keyed by source length.
        self._A_cache_t: dict[int, Tensor] = {}
        self._A_cache_s: dict[int, Tensor] = {}

        self.pyramid: list[Tensor] = []
        self.extend_log: list[str] = []
        self._init_pyramid()
        self.optimizer = self._init_optimizer()

    # ── Initialization ───────────────────────────────────────────────

    def _init_pyramid(self) -> None:
        pyramid: list[Tensor] = []
        res = 1
        while res <= max(self.start_res, self.start_frames):
            t = min(res, self.start_frames, self.num_frames)
            h = min(res, self.start_res, self.syn_res)
            pyramid.insert(0, torch.randn((self.num_clips, 3, t, h, h), device=self.device))
            res *= 2
        pyramid = [p / len(pyramid) for p in pyramid]
        for p in pyramid:
            p.requires_grad_(True)
        self.pyramid = pyramid
        self.extend_log = []

    def _init_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam([{"params": p, "lr": self.lr} for p in self.pyramid])

    # ── Progressive growing ──────────────────────────────────────────

    def _add_level(self, new_t: int, new_h: int, kind: str) -> None:
        old_len = len(self.pyramid)
        new_len = old_len + 1
        self.pyramid = [p.detach().clone() * (old_len / new_len) for p in self.pyramid]
        new_layer = torch.randn((self.num_clips, 3, new_t, new_h, new_h), device=self.device) / new_len
        self.pyramid.insert(0, new_layer)
        for p in self.pyramid:
            p.requires_grad_(True)
        self.optimizer = self._init_optimizer()
        self.extend_log.append(kind)

    def extend_spatial(self) -> bool:
        """Add a level with doubled H, W; ``False`` if already at ``syn_res``."""
        old_t, old_h = self.pyramid[0].shape[-3], self.pyramid[0].shape[-1]
        if old_h == self.syn_res:
            return False
        self._add_level(old_t, min(old_h * 2, self.syn_res), "spatial")
        return True

    def extend_temporal(self) -> bool:
        """Add a level with doubled T; ``False`` if already at ``num_frames``."""
        old_t, old_h = self.pyramid[0].shape[-3], self.pyramid[0].shape[-1]
        if old_t == self.num_frames:
            return False
        self._add_level(min(old_t * 2, self.num_frames), old_h, "temporal")
        return True

    def extend_pyramid(self) -> bool:
        """One growth step (see ``growth``); ``False`` if both axes are at full resolution."""
        if self.growth == "joint":
            old_t, old_h = self.pyramid[0].shape[-3], self.pyramid[0].shape[-1]
            if old_t == self.num_frames and old_h == self.syn_res:
                return False
            self._add_level(min(old_t * 2, self.num_frames), min(old_h * 2, self.syn_res), "spatiotemporal")
            return True
        grew_s = self.extend_spatial()
        grew_t = self.extend_temporal()
        return grew_s or grew_t

    def num_levels(self) -> int:
        return len(self.pyramid)

    # ── Decoding ─────────────────────────────────────────────────────

    def _build_A(self, src: int, target: int) -> Tensor:
        """Dense ``(target, src)`` operator of a 1-D linear upsample.

        Trilinear interpolation factors into one such operator per axis; it is
        built by pushing the identity basis through ``F.interpolate``.
        """
        col = torch.eye(src, device=self.device).unsqueeze(1)  # (src, 1, src)
        up = F.interpolate(col, size=target, mode="linear", align_corners=False)
        return rearrange(up, "j 1 R -> R j")

    def _upsample(self, p: Tensor) -> Tensor:
        """Trilinear upsample of one level to ``(num_frames, syn_res, syn_res)``."""
        t, h, w = p.shape[-3:]
        assert h == w, f"pyramid levels must be spatially square, got {h}x{w}"
        if (t, h) == (self.num_frames, self.syn_res):
            return p
        if t not in self._A_cache_t:
            self._A_cache_t[t] = self._build_A(t, self.num_frames)
        if h not in self._A_cache_s:
            self._A_cache_s[h] = self._build_A(h, self.syn_res)
        return MatmulUpsample3D.apply(p, self._A_cache_t[t], self._A_cache_s[h])

    def get_data(self) -> Tensor:
        """Decoded clips ``(num_clips, num_frames, 3, syn_res, syn_res)`` in ``[0, 1]``."""
        result = torch.sum(torch.stack([self._upsample(p) for p in self.pyramid]), dim=0)  # (N, 3, T, H, W)
        if self.decorrelate_color:
            N, _, T, _, _ = result.shape
            result = linear_decorrelate_color(rearrange(result, "n c t h w -> (n t) c h w"))
            return torch.sigmoid(2 * rearrange(result, "(n t) c h w -> n t c h w", n=N, t=T))
        return torch.sigmoid(2 * rearrange(result, "n c t h w -> n t c h w")).contiguous()

    # ── Checkpointing ────────────────────────────────────────────────

    def get_save_dict(self) -> dict[str, Any]:
        return {"pyramid": self.pyramid, "extend_log": self.extend_log, "opt_state": self.optimizer.state_dict()}

    def load_from_dict(self, load_dict: dict[str, Any]) -> None:
        self.pyramid = []
        for loaded_p in load_dict["pyramid"]:
            p = loaded_p.detach().to(self.device).clone()
            p.requires_grad_(True)
            self.pyramid.append(p)
        self.extend_log = list(load_dict.get("extend_log", []))
        self.optimizer = self._init_optimizer()
        self.optimizer.load_state_dict(load_dict["opt_state"])
