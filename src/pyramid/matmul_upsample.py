"""Upsampling as a precomputed matmul, shared by the image/audio/video pyramids.

A pyramid level is upsampled to a *fixed* target resolution every step, so the
interpolation weights never change: the operation is one fixed linear map.
Bilinear/bicubic/trilinear are all separable, so that map factors into one small
matrix per spatial axis, ``A: (target, source)``, and the upsample becomes a
couple of matmuls.

The point is not the forward — ``F.interpolate``'s forward is a cheap gather.
It is the backward.  ``upsample_*_backward`` scatters every output element into
the source grid with ``atomicAdd``, and coarse pyramid levels are tiny, so the
whole avalanche funnels through a handful of addresses and serializes.  At
syn_res=256 an image pyramid's 1x1 level pushes ~31M atomic adds through 30
addresses, costing ~59 ms/backward against ~0.3 ms for the matmul.  The matmul's
backward is just another matmul: same reduction, expressed as a parallel tree
with no address collisions.  It is also deterministic, which the atomicAdd
kernels are not.

Each ``A`` should be built by pushing the identity basis through the very same
``F.interpolate`` call it replaces (by linearity, column *j* of ``A`` is what
basis vector *e_j* upsamples to).  That inherits PyTorch's exact coefficients,
sampling grid and edge handling instead of re-deriving them.  Agreement with
``F.interpolate`` is then ~1e-7 relative -- below the atomicAdd kernels' own
run-to-run spread -- differing only in floating-point reduction order.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor


@contextmanager
def fp32_matmul() -> Iterator[None]:
    """Force true fp32 matmuls, restoring the caller's TF32 setting on exit."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


# The entry points set float32_matmul_precision('high'), which would run these
# matmuls on tensor cores.  TF32 keeps fp32's exponent but only 10 mantissa
# bits, costing ~2e-4 relative accuracy -- three orders of magnitude worse than
# the ~1e-7 these operators are otherwise good for.  A context manager around
# the forward alone would not help: the backward runs later, during
# loss.backward(), under whatever the global flag is then.  Hence explicit
# autograd Functions that wrap both halves.
#
# ``ctx`` is FunctionCtx in forward but BackwardCFunction in backward, so there
# is no single accurate annotation for it.


class MatmulUpsample1D(torch.autograd.Function):
    """Contract the last axis of ``x`` with ``A: (target, source)``."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, A: Tensor) -> Tensor:
        ctx.save_for_backward(A)
        with fp32_matmul():
            return x @ A.T

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, None]:
        (A,) = ctx.saved_tensors
        with fp32_matmul():
            return grad_out @ A, None


class MatmulUpsample2D(torch.autograd.Function):
    """``A @ x @ A.T`` -- the same operator on both spatial axes of ``x``."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, A: Tensor) -> Tensor:
        ctx.save_for_backward(A)
        with fp32_matmul():
            return A @ (x @ A.T)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, None]:
        (A,) = ctx.saved_tensors
        with fp32_matmul():
            return A.T @ (grad_out @ A), None


class MatmulUpsample3D(torch.autograd.Function):
    """Trilinear upsample of ``(n, c, t, h, w)`` via one matrix per axis.

    ``A_s`` is applied to both spatial axes (pyramid levels are always square).
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, A_t: Tensor, A_s: Tensor) -> Tensor:
        ctx.save_for_backward(A_t, A_s)
        with fp32_matmul():
            x = torch.einsum("nctuv,Wv->nctuW", x, A_s)  # w -> W
            x = torch.einsum("nctuW,Hu->nctHW", x, A_s)  # h -> H
            return torch.einsum("nctHW,Tt->ncTHW", x, A_t)  # t -> T

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, None, None]:
        A_t, A_s = ctx.saved_tensors
        with fp32_matmul():
            g = torch.einsum("ncTHW,Tt->nctHW", grad_out, A_t)
            g = torch.einsum("nctHW,Hu->nctuW", g, A_s)
            g = torch.einsum("nctuW,Wv->nctuv", g, A_s)
        return g, None, None
