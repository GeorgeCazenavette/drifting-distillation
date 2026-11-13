"""1-NN (in general k-NN, uniform vote) classification of the real val set
against the distilled bank, with cosine similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


@torch.no_grad()
def knn_accuracy(
    query_feats: Tensor,
    query_labels: Tensor,
    support_feats: Tensor,
    support_labels: Tensor,
    k: int = 1,
    chunk_size: int = 1024,
) -> float:
    """Top-1 (%) of a uniform-vote cosine k-NN classifier."""
    device = support_feats.device
    n_query = query_feats.shape[0]
    k = min(k, support_feats.shape[0])
    num_classes = int(support_labels.max().item()) + 1
    query_feats = F.normalize(query_feats, dim=-1)
    support_feats = F.normalize(support_feats, dim=-1)
    support_t = rearrange(support_feats, "n d -> d n")

    correct = 0
    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        q = query_feats[start:end].to(device)
        topk = (q @ support_t).topk(k, dim=1)
        knn_labels = support_labels[topk.indices]
        gt = query_labels[start:end].to(device)
        votes = torch.zeros(end - start, num_classes, device=device)
        votes.scatter_add_(1, knn_labels, torch.ones_like(knn_labels, dtype=torch.float))
        correct += (votes.argmax(dim=1) == gt).sum().item()
    return 100.0 * correct / n_query
