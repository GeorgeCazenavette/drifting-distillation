"""The drifting loss (paper Sec. 3.2, Algorithms 1-2).

Follows the full implementation of Deng et al., "Generative Modeling via
Drifting": the kernel is normalized over both the target and the synthetic
axis (their Alg. 2), features are rescaled by a stop-gradient scale ``S`` so
the mean pairwise distance is ~sqrt(C), the temperature is scaled to
``tau * sqrt(C)``, and each temperature's field is divided by its RMS before
summing (their App. A.6).

Positives are the real images of the class; negatives are the other synthetic
samples (all augmented views of all distilled images in the current step).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def _compute_V(
    d_pos: Tensor, d_neg: Tensor, phi_pos: Tensor, phi_neg: Tensor, T: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Drifting field for one temperature (Alg. 2).

    Args:
        d_pos: ``[locs, G, P]`` normalized synthetic-to-real distances.
        d_neg: ``[locs, G, G]`` normalized synthetic-to-synthetic distances
            (self-pairs masked to a large value).
        phi_pos: ``[locs, P, C]`` normalized real features.
        phi_neg: ``[locs, G, C]`` normalized synthetic features (detached).
        T: Effective temperature ``tau * sqrt(C)``.

    Returns:
        ``(V, V_pos, V_neg)``, each ``[locs, G, C]``.
    """
    P = d_pos.shape[-1]
    logit = torch.cat([-d_pos / T, -d_neg / T], dim=-1)  # [locs, G, P+G]
    A_row = logit.softmax(dim=-1)  # over targets
    A_col = logit.softmax(dim=-2)  # over synthetic samples
    A = (A_row * A_col).clamp(min=1e-6).sqrt()  # geometric mean

    A_pos = A[..., :P]
    A_neg = A[..., P:]
    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)

    V_pos = torch.matmul(W_pos, phi_pos)
    V_neg = torch.matmul(W_neg, phi_neg)
    return V_pos - V_neg, V_pos, V_neg


def single_feature_loss(
    gen_feat: Tensor, pos_feat: Tensor, G: int, P: int, temps: tuple[float, ...] = (0.5,)
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Drifting loss for one feature (Alg. 1 inner loop).

    Each spatial location of the feature is an independent drifting problem;
    only ``S`` and the per-temperature RMS ``lambda`` are shared across
    locations.

    Args:
        gen_feat: ``[G*locs, C]`` synthetic features (requires grad).
        pos_feat: ``[P*locs, C]`` real features.
        G, P: Number of synthetic / real samples.
        temps: Kernel temperatures; the normalized fields are summed.

    Returns:
        ``(loss, pos_field_mag, neg_field_mag, raw_field_rms)``.  The first
        three use the RMS-normalized field, so the loss is ~1 by construction;
        ``raw_field_rms`` is the per-dimension RMS of the field *before* that
        normalization (in ``S``-normalized feature units), which shrinks as the
        synthetic distribution approaches the real one.
    """
    C = gen_feat.shape[-1]
    sqrtC = C**0.5

    gen = rearrange(gen_feat, "(g l) c -> l g c", g=G)  # keeps grad
    pos = rearrange(pos_feat, "(p l) c -> l p c", p=P)
    locs = gen.shape[0]
    gen_det = gen.detach()
    arange_G = torch.arange(G, device=gen_feat.device)

    with torch.no_grad():
        d_pos = torch.cdist(gen_det, pos)  # [locs, G, P]
        d_neg = torch.cdist(gen_det, gen_det)  # [locs, G, G]

        # Feature normalization: mean pairwise distance -> sqrt(C).
        mean_dist = torch.cat([d_pos.reshape(-1), d_neg.reshape(-1)]).mean().clamp(min=1e-8)
        S = mean_dist / sqrtC

        d_neg[:, arange_G, arange_G] = 1e6  # mask self-pairs
        d_pos_n = d_pos / S
        d_neg_n = d_neg / S
        phi_pos = pos / S
        phi_neg = gen_det / S

        V = torch.zeros(locs, G, C, device=gen_feat.device, dtype=gen_feat.dtype)
        Vp = torch.zeros_like(V)
        Vn = torch.zeros_like(V)
        V_raw = torch.zeros_like(V)
        for tau in temps:
            V_t, Vp_t, Vn_t = _compute_V(d_pos_n, d_neg_n, phi_pos, phi_neg, tau * sqrtC)
            # Drift normalization: per-dimension RMS of the field -> 1.
            lam = (V_t.pow(2).sum(dim=-1).mean() / C).sqrt().clamp(min=1e-8)
            V_raw = V_raw + V_t
            V = V + V_t / lam
            Vp = Vp + Vp_t / lam
            Vn = Vn + Vn_t / lam

    # MSE(phi(x)/S, stopgrad(phi(x)/S + V)) -- the gradient flows through phi(x).
    phi_gen = rearrange(gen / S, "l g c -> (g l) c")
    target = phi_gen.detach() + rearrange(V, "l g c -> (g l) c")
    loss = F.mse_loss(phi_gen, target)

    with torch.no_grad():
        pos_mag = Vp.norm(dim=-1).mean()
        neg_mag = Vn.norm(dim=-1).mean()
        raw_rms = (V_raw.pow(2).sum(dim=-1).mean() / C).sqrt()
    return loss, pos_mag, neg_mag, raw_rms


def drifting_loss(
    syn: Tensor,
    extractor: nn.Module,
    pos_latents: dict[str, Tensor],
    loss_fn: Callable[[Tensor, Tensor, int, int], tuple[Tensor, Tensor, Tensor, Tensor]] = single_feature_loss,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sum of per-feature drifting losses.

    Args:
        syn: ``[G, ...]`` augmented synthetic samples (requires grad).
        extractor: Frozen feature extractor returning ``{name: [G*locs, C]}``.
        pos_latents: Precomputed real features ``{name: [P*locs, C]}``.
        loss_fn: Per-feature loss (``single_feature_loss`` with temperatures
            bound, optionally ``torch.compile``-d).

    Returns:
        ``(loss, mean_pos_field_mag, mean_neg_field_mag, mean_raw_field_rms)``.
    """
    G = syn.shape[0]
    gen_feats: dict[str, Tensor] = extractor(syn)

    total = syn.new_zeros(())
    pos_mag = syn.new_zeros(())
    neg_mag = syn.new_zeros(())
    raw_rms = syn.new_zeros(())
    for key, gf in gen_feats.items():
        pf = pos_latents[key]
        P = pf.shape[0] // (gf.shape[0] // G)
        l, pm, nm, rr = loss_fn(gf, pf, G, P)
        total = total + l
        pos_mag = pos_mag + pm
        neg_mag = neg_mag + nm
        raw_rms = raw_rms + rr
    n = max(len(gen_feats), 1)
    return total, pos_mag / n, neg_mag / n, raw_rms / n
