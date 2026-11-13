"""Feature extraction for the feature-space evaluations (coverage, k-NN, linear).

Extractors (``src.extractors.get_extractor``) consume raw ``[0, 1]`` images and
return ``{name: Tensor[N, C]}``; the values are concatenated along the last dim.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from .data import real_dir

if TYPE_CHECKING:
    from .eval_coverage import CoverageArgs
    from .eval_linear import LinearArgs

CENTER_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])


def _flatten_features(out: dict[str, Tensor]) -> Tensor:
    """Flatten every value in ``out`` to ``[N, *]`` and concat along dim 1."""
    parts = []
    for v in out.values():
        if v.dim() > 2:
            v = rearrange(v, "n ... -> n (...)")
        parts.append(v)
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def embed_loader(extractor: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    """Embed every ``(image, label)`` batch.  Returns CPU ``(feats [N, D], labels [N])``."""
    feats: list[Tensor] = []
    labels: list[Tensor] = []
    for x, y in tqdm(loader, desc="embed", unit="batch", dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        feats.append(_flatten_features(extractor(x)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def query_features(
    extractor: nn.Module, args: CoverageArgs | LinearArgs, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Embeddings of the real val split (Resize 256, CenterCrop 224), computed
    once and cached at ``{cache_dir}/{backbone}_{mode}_{dataset}_val.pt``."""
    cache_path = os.path.join(args.cache_dir, f"{args.backbone}_{args.mode}_{args.dataset}_val.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached query embeddings from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        return cached["feats"], cached["labels"]

    dataset = ImageFolder(real_dir(args.data_root, args.dataset, "val"), transform=CENTER_TF)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=1 if args.num_workers > 0 else None,
    )
    print(f"Embedding query set ({len(dataset)} samples) ...")
    feats, labels = embed_loader(extractor, loader, device)
    os.makedirs(args.cache_dir, exist_ok=True)
    tmp = f"{cache_path}.tmp.{os.getpid()}"  # atomic: concurrent jobs may share the cache
    torch.save({"feats": feats, "labels": labels}, tmp)
    os.replace(tmp, cache_path)
    print(f"  cached to {cache_path}")
    return feats, labels


def support_bank(extractor: nn.Module, args: CoverageArgs | LinearArgs, device: torch.device) -> tuple[Tensor, Tensor]:
    """Augmented bank of the distilled set: the center crop of every image, followed by ``args.num_crops`` passes
    of RandomResizedCrop(224, scale=[min_scale, 1]) + flip.  The crops are drawn
    after ``torch.manual_seed(args.crop_seed)``, so the bank does not depend on
    ``--seed``."""
    support_ds = ImageFolder(args.data, transform=CENTER_TF)
    support_loader = DataLoader(
        support_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    t0 = time.time()
    base_feats, base_labels = embed_loader(extractor, support_loader, device)
    print(f"  centercrop: {tuple(base_feats.shape)} " f"({time.time() - t0:.1f}s)")

    aug_tf = T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(args.min_scale, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )
    aug_ds = ImageFolder(args.data, transform=aug_tf)
    aug_loader = DataLoader(
        aug_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    torch.manual_seed(args.crop_seed)
    all_feats, all_labels = [base_feats], [base_labels]
    for i in range(args.num_crops):
        t1 = time.time()
        f_aug, l_aug = embed_loader(extractor, aug_loader, device)
        all_feats.append(f_aug)
        all_labels.append(l_aug)
        print(f"  crop pass {i + 1}/{args.num_crops} ({time.time() - t1:.1f}s)")
    feats, labels = torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)
    print(f"  bank: {tuple(feats.shape)}")
    return feats, labels
