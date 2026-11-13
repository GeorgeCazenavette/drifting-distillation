"""Laplacian-pyramid parameterization of the distilled images (paper Sec. 3.3).

Each level is a learnable ``(num_images, 3, r, r)`` tensor, ``r = 1, 2, 4, ...``.
An image is decoded by bicubically upsampling every level to ``syn_res``,
summing, applying a fixed 3x3 color decorrelation, and squashing with
``sigmoid(2x)``.  Training starts with the coarse levels only and adds a finer
level every ``extend_every`` steps.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .matmul_upsample import MatmulUpsample2D

# Reach of the bicubic kernel, in source pixels, on either side of a sample
# point.  Sets how much circular context a tileable level needs.
_BICUBIC_PAD = 2


# ── Color decorrelation ──────────────────────────────────────────────

# sqrt of the ImageNet RGB covariance, in its eigenbasis (Olah et al., 2017).
COLOR_CORRELATION_SVD_SQRT = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])


def linear_decorrelate_color(im: Tensor) -> Tensor:
    """Map the decorrelated latent channels back to RGB."""
    svd = COLOR_CORRELATION_SVD_SQRT.to(im.device, im.dtype)
    color_corr = svd / torch.max(torch.linalg.norm(svd, dim=0))

    h, w = im.shape[-2:]
    im = rearrange(im, "b c h w -> (b h w) c")
    im = im @ color_corr.T
    return rearrange(im, "(b h w) c -> b c h w", h=h, w=w)


# ── Pyramid ──────────────────────────────────────────────────────────


class ImagePyramid:
    """Coarse-to-fine pyramid of learnable images.

    Args:
        num_images: Number of distilled images.
        syn_res: Output resolution.
        start_res: Finest initial level (power of 2); levels ``1..start_res``
            exist at step 0.
        lr: Adam learning rate.
        init_mode: ``"noise"`` (standard Gaussian levels) or ``"zero"``
            (used before fitting to real images).
        tileable: Circular boundary conditions in the upsampler (seamless
            textures).
        decorrelate_color: Apply the color decorrelation in the decoder.
        device: Torch device.
    """

    def __init__(
        self,
        num_images: int,
        syn_res: int = 256,
        start_res: int = 8,
        lr: float = 0.001,
        init_mode: str = "noise",
        tileable: bool = False,
        decorrelate_color: bool = True,
        device: str | torch.device = "cuda",
    ) -> None:
        self.num_images = num_images
        self.syn_res = syn_res
        self.start_res = start_res
        self.lr = lr
        self.init_mode = init_mode
        self.tileable = tileable
        self.decorrelate_color = decorrelate_color
        self.device = device

        # Upsample operators, keyed by level resolution.
        self._A_cache: dict[int, Tensor] = {}

        self.pyramid: list[Tensor] = []
        self._init_pyramid()
        self.optimizer = self._init_optimizer()

    # ── Initialisation ────────────────────────────────────────────

    def _init_pyramid(self) -> None:
        pyramid: list[Tensor] = []
        res = 1
        while res <= self.start_res:
            if res > self.syn_res:
                res = self.syn_res
            level = torch.randn((self.num_images, 3, res, res), device=self.device)
            if self.init_mode == "zero":
                level = level * 0
            pyramid.insert(0, level)
            res *= 2

        pyramid = [p / len(pyramid) for p in pyramid]
        for p in pyramid:
            p.requires_grad_(True)
        self.pyramid = pyramid

    def _init_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam([{"params": p, "lr": self.lr} for p in self.pyramid])

    def init_from_images(self, target_images: Tensor, fit_steps: int = 300, fit_lr: float = 0.05) -> float:
        """Fit the pyramid to ``target_images`` ``(N, 3, H, W)`` in ``[0, 1]``.

        Grows the pyramid to full resolution first, then minimizes the MSE
        between the decoded images and the targets.  Returns the final MSE.
        """
        while self.pyramid[0].shape[-1] < self.syn_res:
            self.extend_pyramid()

        target = target_images.to(self.device).detach().clamp(1e-4, 1 - 1e-4)
        assert target.shape[0] == self.num_images, f"Expected {self.num_images} images, got {target.shape[0]}"
        opt = torch.optim.Adam(self.pyramid, lr=fit_lr)
        final_loss = 0.0
        for _ in range(fit_steps):
            opt.zero_grad()
            loss = F.mse_loss(self.decode_pyramid(), target)
            loss.backward()
            opt.step()
            final_loss = loss.item()
        self.pyramid = [p.detach().requires_grad_(True) for p in self.pyramid]
        self.optimizer = self._init_optimizer()
        return final_loss

    # ── Progressive growing ───────────────────────────────────────

    def extend_pyramid(self) -> bool:
        """Add the next finer level.  Returns ``False`` if already at ``syn_res``."""
        old_len = len(self.pyramid)
        old_res = self.pyramid[0].shape[-1]
        if old_res == self.syn_res:
            return False

        new_len = old_len + 1
        new_res = min(old_res * 2, self.syn_res)
        self.pyramid = [p.detach().clone() * old_len / new_len for p in self.pyramid]

        if self.init_mode == "zero":
            new_layer = (
                torch.sum(
                    torch.stack(
                        [F.interpolate(p, (new_res, new_res), antialias=False, mode="bicubic") for p in self.pyramid]
                    ),
                    dim=0,
                )
                / old_len
            )
        else:
            new_layer = torch.randn((self.num_images, 3, new_res, new_res), device=self.device) / new_len

        self.pyramid.insert(0, new_layer)
        for p in self.pyramid:
            p.requires_grad_(True)
        self.optimizer = self._init_optimizer()
        return True

    # ── Upsampling ────────────────────────────────────────────────

    def _build_A(self, h: int) -> Tensor:
        """Dense ``(syn_res, h)`` operator for the 1-D bicubic upsample of a level.

        Built by pushing the identity basis through ``F.interpolate`` itself, so
        it inherits PyTorch's exact bicubic coefficients and edge handling.
        For ``tileable`` levels the basis is circularly padded (or, for
        ``h == 2``, tiled 3x) before upsampling, then the central window kept.
        """
        eye = torch.eye(h, device=self.device)
        col = rearrange(eye, "j h -> j 1 h 1")

        if not self.tileable or h == 1:
            up = F.interpolate(col, (self.syn_res, 1), antialias=False, mode="bicubic")
            return rearrange(up, "j 1 R 1 -> R j")

        k = self.syn_res // h
        if h == 2:
            col = torch.cat([col, col, col], dim=-2)
        else:
            col = F.pad(col, (0, 0, _BICUBIC_PAD, _BICUBIC_PAD), mode="circular")

        up = F.interpolate(col, scale_factor=(k, 1), antialias=False, mode="bicubic")
        s = _BICUBIC_PAD * k
        return rearrange(up[:, :, s : s + self.syn_res, :], "j 1 R 1 -> R j")

    def _upsample(self, p: Tensor) -> Tensor:
        """Bicubic upsample of one level to ``syn_res`` as ``A @ p @ A.T``.

        Equivalent to ``F.interpolate`` (to fp32 rounding) but with a much
        cheaper, deterministic backward; see :mod:`.matmul_upsample`.
        """
        h = p.shape[-1]
        if h == self.syn_res:
            return p
        if h not in self._A_cache:
            self._A_cache[h] = self._build_A(h)
        return MatmulUpsample2D.apply(p, self._A_cache[h])

    # ── Decoding ──────────────────────────────────────────────────

    def decode_pyramid(self) -> Tensor:
        """Decode to ``(num_images, 3, syn_res, syn_res)`` images in ``[0, 1]``."""
        result = torch.sum(torch.stack([self._upsample(p) for p in self.pyramid]), dim=0)
        if self.decorrelate_color:
            result = linear_decorrelate_color(result)
        return torch.sigmoid(2 * result)

    def get_data(self) -> Tensor:
        return self.decode_pyramid()

    def num_levels(self) -> int:
        return len(self.pyramid)

    # ── Checkpointing ─────────────────────────────────────────────

    def get_save_dict(self) -> dict[str, Any]:
        return {"pyramid": self.pyramid, "opt_state": self.optimizer.state_dict()}

    def load_from_dict(self, load_dict: dict[str, Any]) -> None:
        loaded = load_dict["pyramid"]
        while len(self.pyramid) < len(loaded):
            self.extend_pyramid()
        with torch.no_grad():
            for p, q in zip(self.pyramid, loaded, strict=True):
                p.copy_(q)
        self.optimizer.load_state_dict(load_dict["opt_state"])
