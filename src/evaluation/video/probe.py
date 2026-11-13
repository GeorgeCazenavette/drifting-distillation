"""Linear probe and cosine k-NN on frozen features (shared by the video and audio evaluations).

Unlike the image evaluation's step-budget probe (``src/evaluation/linear_probe.py``),
this is the epoch-based probe used for the audio/video results: AdamW, cosine
LR annealed per epoch, plain un-augmented features.  With 5 samples per class
the whole support set fits in one batch, so every epoch is one full-batch step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@torch.no_grad()
def embed(extractor: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    """``(feats, labels)`` on CPU; every extractor output is concatenated along the last dim.

    Batches of ``[B, V, ...]`` multi-view samples (``loader.dataset.multi_view``)
    give ``feats [N, V, D]``.
    """
    multi_view = getattr(loader.dataset, "multi_view", False)
    feats, labels = [], []
    for x, y in tqdm(loader, desc="embed", dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        if multi_view:
            B = x.shape[0]
            x = rearrange(x, "b v ... -> (b v) ...")
        f = torch.cat(list(extractor(x).values()), dim=-1)
        if multi_view:
            f = rearrange(f, "(b v) d -> b v d", b=B)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def fit_linear_head(
    feats: Tensor,
    labels: Tensor,
    num_classes: int,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> nn.Linear:
    """Train ``nn.Linear(D, num_classes)`` on ``feats [N, D]``."""
    head = nn.Linear(feats.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    ds = TensorDataset(feats, labels)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, num_workers=0)
    for epoch in range(epochs):
        head.train()
        total_loss, n = 0.0, 0
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            loss = criterion(head(f), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * f.size(0)
            n += f.size(0)
        scheduler.step()
        if (epoch + 1) % max(epochs // 10, 1) == 0:
            print(f"  epoch {epoch + 1:>4}/{epochs}  loss={total_loss / n:.4f}")
    return head


@torch.inference_mode()
def linear_accuracy(head: nn.Linear, feats: Tensor, labels: Tensor, device: torch.device) -> float:
    """Top-1 accuracy (%) of ``head`` on ``feats [N, D]``."""
    head.eval()
    pred = head(feats.to(device)).argmax(dim=1)
    return 100.0 * (pred == labels.to(device)).float().mean().item()


@torch.no_grad()
def knn_accuracy(
    query_feats: Tensor,
    query_labels: Tensor,
    support_feats: Tensor,
    support_labels: Tensor,
    k: int = 1,
    device: torch.device | str = "cuda",
) -> float:
    """Top-1 accuracy (%) of cosine k-NN with uniform (majority) voting."""
    q = F.normalize(query_feats.to(device), dim=-1)
    s = F.normalize(support_feats.to(device), dim=-1)
    support_labels = support_labels.to(device)
    num_classes = int(support_labels.max().item()) + 1
    nn_labels = support_labels[(q @ s.T).topk(min(k, s.shape[0]), dim=1).indices]  # [Nq, k]
    votes = torch.zeros(q.shape[0], num_classes, device=device)
    votes.scatter_add_(1, nn_labels, torch.ones_like(nn_labels, dtype=torch.float))
    return 100.0 * (votes.argmax(dim=1) == query_labels.to(device)).float().mean().item()
