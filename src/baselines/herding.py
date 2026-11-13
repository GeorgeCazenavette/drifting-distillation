"""Herding (Welling, 2009): greedy per-class coreset selection in feature space.

Each step adds the sample that brings the running mean of the selected
features closest (L2) to the class mean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@torch.no_grad()
def extract_features(
    dataset: Dataset, extractor: nn.Module, device: str | torch.device, batch_size: int, num_workers: int
) -> tuple[Tensor, Tensor]:
    """One pass over ``dataset``; returns ``(feats [N, D], labels [N])`` on CPU.

    ``D`` concatenates every feature the extractor returns (in its key order).
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    feats, labels = [], []
    for x, y in tqdm(loader, desc="Extracting features"):
        out = extractor(x.to(device))
        feats.append(torch.cat(list(out.values()), dim=-1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def herding_select(features: Tensor, n_select: int) -> list[int]:
    """Indices of ``n_select`` rows of ``features [N, D]`` chosen by herding."""
    n = features.shape[0]
    n_select = min(n_select, n)

    mean_feat = features.mean(0, keepdim=True)
    running_sum = torch.zeros_like(mean_feat)
    used = torch.zeros(n, dtype=torch.bool)
    selected: list[int] = []
    for i in range(n_select):
        cost = (mean_feat - (features + running_sum) / (i + 1)).norm(2, dim=1)
        cost[used] = float("inf")
        idx = int(cost.argmin().item())
        selected.append(idx)
        running_sum += features[idx : idx + 1]
        used[idx] = True
    return selected


def select_per_class(feats: Tensor, labels: Tensor, n_select: int) -> dict[int, list[int]]:
    """``{class: dataset indices}`` of each class's herding coreset."""
    out = {}
    for cls in range(len(set(labels.tolist()))):
        idx = (labels == cls).nonzero().squeeze(1).tolist()
        out[cls] = [idx[j] for j in herding_select(feats[labels == cls], n_select)]
    return out
