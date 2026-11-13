"""Coverage (Naeem et al., 2020): the fraction of real points whose k-NN ball
(radius = distance to the k-th nearest *real* neighbour) contains at least one
distilled point.  Distances are Euclidean on L2-normalised features (chord
distance on the unit sphere, i.e. monotone in cosine distance)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


@torch.no_grad()
def coverage(real: Tensor, synth: Tensor, k: int) -> float:
    """Coverage of ``real`` [N_r, D] by ``synth`` [N_g, D]; NaN if undefined."""
    n_r, n_g = real.shape[0], synth.shape[0]
    if n_g == 0 or n_r < k + 1:
        return math.nan
    real = F.normalize(real.float(), dim=-1)
    synth = F.normalize(synth.float(), dim=-1)
    d_rr = torch.cdist(real, real)
    d_rr.fill_diagonal_(math.inf)
    d_rg = torch.cdist(real, synth)  # [N_r, N_g]
    radii = d_rr.kthvalue(k, dim=1).values  # [N_r]
    in_ball = d_rg < rearrange(radii, "n -> n 1")
    return float(in_ball.any(dim=1).float().mean().item())


def per_class_coverage(feats: Tensor, labels: Tensor, qfeats: Tensor, qlabels: Tensor, k: int) -> dict[int, float]:
    """Coverage of each class's real val points by that class's distilled bank.
    ``feats``/``qfeats`` live on the compute device, labels on the CPU."""
    out = {}
    for c in sorted(int(c) for c in labels.unique().tolist()):
        s_feats = feats[(labels == c).to(feats.device)]
        q_feats = qfeats[(qlabels == c).to(qfeats.device)]
        out[c] = coverage(q_feats, s_feats, k)
    return out
