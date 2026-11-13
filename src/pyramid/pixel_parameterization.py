"""Raw-pixel parameterization: the "-pyramid" ablation (appendix, Parameterization).

A single learnable ``(num_images, 3, syn_res, syn_res)`` tensor pushed through
the same decode tail as :class:`ImagePyramid` (color decorrelation, then
``sigmoid(2x)``), so the only difference is the multi-scale structure.
``init_std`` should match the pyramid's per-pixel std at initialization,
``1/sqrt(L)`` for ``L`` initial levels.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .image_pyramid import linear_decorrelate_color


class PixelParameterization:
    def __init__(
        self,
        num_images: int,
        syn_res: int = 256,
        lr: float = 0.01,
        init_mode: str = "noise",
        decorrelate_color: bool = True,
        device: str | torch.device = "cuda",
        init_std: float = 0.5,
    ):
        self.num_images = num_images
        self.syn_res = syn_res
        self.lr = lr
        self.decorrelate_color = decorrelate_color
        self.device = device

        pixels = torch.randn((num_images, 3, syn_res, syn_res), device=device) * init_std
        if init_mode == "zero":
            pixels = pixels * 0
        self.pixels = pixels.requires_grad_(True)
        self.optimizer = self._init_optimizer()

    def _init_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam([{"params": self.pixels, "lr": self.lr}])

    def decode_pyramid(self) -> Tensor:
        result = self.pixels
        if self.decorrelate_color:
            result = linear_decorrelate_color(result)
        return torch.sigmoid(2 * result)

    def get_data(self) -> Tensor:
        return self.decode_pyramid()

    def init_from_images(self, target_images: Tensor, fit_steps: int = 300, fit_lr: float = 0.05) -> float:
        target = target_images.to(self.device).detach().clamp(1e-4, 1 - 1e-4)
        assert target.shape[0] == self.num_images, f"Expected {self.num_images} images, got {target.shape[0]}"
        opt = torch.optim.Adam([self.pixels], lr=fit_lr)
        final_loss = 0.0
        for _ in range(fit_steps):
            opt.zero_grad()
            loss = F.mse_loss(self.decode_pyramid(), target)
            loss.backward()
            opt.step()
            final_loss = loss.item()
        self.pixels = self.pixels.detach().requires_grad_(True)
        self.optimizer = self._init_optimizer()
        return final_loss

    def extend_pyramid(self) -> bool:
        return False

    def num_levels(self) -> int:
        return 1

    def get_save_dict(self) -> dict[str, Any]:
        return {"pixels": self.pixels, "opt_state": self.optimizer.state_dict()}

    def load_from_dict(self, load_dict: dict[str, Any]) -> None:
        with torch.no_grad():
            self.pixels.copy_(load_dict["pixels"])
        self.optimizer.load_state_dict(load_dict["opt_state"])
