"""1-D pyramid parameterization of distilled audio clips.

Each level is a learnable ``(num_clips, 1, n)`` tensor, ``n = 1, 2, 4, ...``.
A waveform is decoded by linearly upsampling every level to
``target_samples``, summing, and applying ``tanh`` (output in ``(-1, 1)``).
Levels start at zero (silence); each new level is initialized to the current
decoded signal at its resolution, so adding it leaves the output unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .matmul_upsample import MatmulUpsample1D

# Cap on a cached dense upsample operator ``(target_samples, n)``.  At the
# default 524288 samples it admits n <= 32 -- the coarse levels whose
# atomicAdd-based interpolate backward is slowest; finer levels fall back to
# ``F.interpolate``, whose backward is already cheap there.
_MAX_A_BYTES = 64 * 2**20


class AudioPyramid:
    """Coarse-to-fine pyramid of learnable waveforms.

    Args:
        num_clips: Number of distilled clips.
        target_samples: Output length in samples.
        start_samples: Finest initial level (power of 2).
        lr: Adam learning rate.
        device: Torch device.
    """

    def __init__(
        self,
        num_clips: int,
        target_samples: int,
        start_samples: int = 1,
        lr: float = 0.001,
        device: str | torch.device = "cuda",
    ) -> None:
        self.num_clips = num_clips
        self.target_samples = target_samples
        self.start_samples = start_samples
        self.lr = lr
        self.device = device
        self._A_cache: dict[int, Tensor | None] = {}  # None: too large, use F.interpolate

        self.pyramid: list[Tensor] = []
        self._init_pyramid()
        self.optimizer = self._init_optimizer()

    def _init_pyramid(self) -> None:
        pyramid: list[Tensor] = []
        n = 1
        while n <= self.start_samples:
            if n > self.target_samples:
                n = self.target_samples
            # Drawn, then zeroed, as in the paper runs (keeps the RNG stream).
            pyramid.insert(0, torch.randn((self.num_clips, 1, n), device=self.device) * 0)
            n *= 2
        pyramid = [p / len(pyramid) for p in pyramid]
        for p in pyramid:
            p.requires_grad_(True)
        self.pyramid = pyramid

    def _init_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam([{"params": p, "lr": self.lr} for p in self.pyramid])

    def extend_pyramid(self) -> bool:
        """Add a level of doubled length; ``False`` if already at ``target_samples``."""
        old_len = len(self.pyramid)
        old_n = self.pyramid[0].shape[-1]
        if old_n == self.target_samples:
            return False
        new_len = old_len + 1
        new_n = min(old_n * 2, self.target_samples)

        self.pyramid = [p.detach().clone() * old_len / new_len for p in self.pyramid]
        new_layer = (
            torch.sum(
                torch.stack([F.interpolate(p, size=new_n, mode="linear", align_corners=False) for p in self.pyramid]),
                dim=0,
            )
            / old_len
        )
        self.pyramid.insert(0, new_layer)
        for p in self.pyramid:
            p.requires_grad_(True)
        self.optimizer = self._init_optimizer()
        return True

    def num_levels(self) -> int:
        return len(self.pyramid)

    def _build_A(self, n: int) -> Tensor | None:
        """Dense ``(target_samples, n)`` linear-upsample operator, built by pushing
        the identity basis through ``F.interpolate``; ``None`` if over the cap."""
        if 4 * self.target_samples * n > _MAX_A_BYTES:
            return None
        col = torch.eye(n, device=self.device).unsqueeze(1)  # (n, 1, n)
        up = F.interpolate(col, size=self.target_samples, mode="linear", align_corners=False)
        return rearrange(up, "j 1 R -> R j")

    def _upsample(self, p: Tensor) -> Tensor:
        n = p.shape[-1]
        if n == self.target_samples:
            return p
        if n not in self._A_cache:
            self._A_cache[n] = self._build_A(n)
        A = self._A_cache[n]
        if A is None:
            return F.interpolate(p, size=self.target_samples, mode="linear", align_corners=False)
        return MatmulUpsample1D.apply(p, A)

    def get_data(self) -> Tensor:
        """Decoded waveforms ``(num_clips, 1, target_samples)`` in ``(-1, 1)``."""
        return torch.tanh(torch.sum(torch.stack([self._upsample(p) for p in self.pyramid]), dim=0))

    def get_save_dict(self) -> dict[str, Any]:
        return {"pyramid": self.pyramid, "opt_state": self.optimizer.state_dict()}

    def load_from_dict(self, load_dict: dict[str, Any]) -> None:
        while len(self.pyramid) < len(load_dict["pyramid"]):
            self.extend_pyramid()
        with torch.no_grad():
            for p, loaded_p in zip(self.pyramid, load_dict["pyramid"], strict=True):
                p.copy_(loaded_p)
        self.optimizer.load_state_dict(load_dict["opt_state"])
