"""Linear probe on pre-extracted (frozen, unnormalized) features.

A single ``nn.Linear(D, C)`` is trained with AdamW for a fixed number of steps;
each step draws a batch uniformly with replacement from the bank, and the LR is
cosine-annealed to 0 per step.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def fit_linear_head(
    train_feats: Tensor,
    train_labels: Tensor,
    val_feats: Tensor,
    val_labels: Tensor,
    num_classes: int,
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Train the head and return ``{"acc1", "acc5"}`` on the val features."""
    head = nn.Linear(train_feats.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_feats, train_labels)
    bs = min(batch_size, len(train_ds))
    w = torch.ones(len(train_ds))
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        num_workers=0,
        drop_last=True,
        sampler=WeightedRandomSampler(w.double(), num_samples=steps * bs, replacement=True),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    t0 = time.time()
    head.train()
    total_loss, n = 0.0, 0
    for f, y in train_loader:
        f = f.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss = criterion(head(f), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * f.size(0)
        n += f.size(0)
    print(f"  {steps} steps  loss={total_loss / n:.4f}  ({time.time() - t0:.1f}s)")
    return evaluate_linear(head, val_feats, val_labels, device)


def _topk_correct(output: Tensor, target: Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    """Top-k accuracy (%) of a batch."""
    clamped = [min(k, output.size(1)) for k in topk]
    _, pred = output.topk(max(clamped), 1, True, True)
    correct = rearrange(pred, "n k -> k n").eq(target[None])
    return [100.0 * correct[:k].flatten().float().sum().item() / target.size(0) for k in clamped]


@torch.inference_mode()
def evaluate_linear(
    head: nn.Linear, val_feats: Tensor, val_labels: Tensor, device: torch.device, chunk_size: int = 4096
) -> dict[str, float]:
    """Top-1 / top-5 (%) of a linear head on cached features."""
    head.eval()
    correct1 = correct5 = 0
    n = val_feats.size(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        f = val_feats[start:end].to(device, non_blocking=True)
        y = val_labels[start:end].to(device, non_blocking=True)
        a1, a5 = _topk_correct(head(f), y)
        correct1 += a1 * (end - start) / 100.0
        correct5 += a5 * (end - start) / 100.0
    acc1, acc5 = 100.0 * correct1 / n, 100.0 * correct5 / n
    print(f"  linear val Acc@1: {acc1:.2f}%  Acc@5: {acc5:.2f}%  ({n} samples)")
    return {"acc1": acc1, "acc5": acc5}
