"""Coverage@k and k-NN of a distilled set in a frozen feature space.

Bank: each distilled image's center crop together with 10 RandomResizedCrops
(scale [0.08, 1]) + flip per image.  Query: the real val set (Resize 256,
CenterCrop 224; embeddings cached under ``--cache_dir``).  Coverage (k=5) is
computed per class with cosine (L2-normalised) distances and averaged over
classes; k-NN is 1-NN with cosine similarity against the whole bank.  Both are
seed-invariant: the crops are drawn from ``--crop_seed``.

    python -m src.evaluation.eval_coverage --data DIR --backbone siglip2 --out results/cov_siglip2.json
"""

from __future__ import annotations

import json
import math
import os

import torch
from tap import Tap

from ..extractors import get_extractor
from .coverage import per_class_coverage
from .data import NUM_CLASSES, check_distilled
from .features import query_features, support_bank
from .knn import knn_accuracy


class CoverageArgs(Tap):
    data: str  # Distilled ImageFolder dir ({class:04d}/*.png)
    out: str  # Results JSON path
    backbone: str = "siglip2"  # Eval space (paper: siglip2, mocov3, clip, dinov3, eupe, eva02, vit)
    mode: str = "cls"  # Extractor feature mode
    dataset: str = "imagenet-1k"  # imagenet-1k | imagenette | imagewoof
    data_root: str = "data"  # Holds the real val split (see src/evaluation/data.py)
    cache_dir: str = ".val_cache"  # Real val embedding cache
    k: int = 5  # Coverage neighbourhood size
    knn_k: int = 1  # k-NN neighbours
    num_crops: int = 10  # RandomResizedCrop passes appended to the center crops
    min_scale: float = 0.08  # RandomResizedCrop scale lower bound
    crop_seed: int = 0  # Seed of the crop passes
    seed: int = 42
    batch_size: int = 256
    num_workers: int = 8  # Also fixes which worker draws which crop


def main() -> None:
    args = CoverageArgs(explicit_bool=True).parse_args()
    assert args.dataset in NUM_CLASSES, args.dataset
    check_distilled(args.data, args.dataset)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    print(f"Loading extractor: backbone={args.backbone}, mode={args.mode}")
    extractor = get_extractor(args.backbone, mode=args.mode).to(device).eval()
    qfeats, qlabels = query_features(extractor, args, device)
    feats, labels = support_bank(extractor, args, device)

    qfeats_dev, feats_dev = qfeats.to(device), feats.to(device)
    per_class = per_class_coverage(feats_dev, labels, qfeats_dev, qlabels, args.k)
    valid = [v for v in per_class.values() if not math.isnan(v)]
    cov = sum(valid) / len(valid) if valid else math.nan
    knn = knn_accuracy(qfeats_dev, qlabels.to(device), feats_dev, labels.to(device), k=args.knn_k)
    print(f"\ncoverage@{args.k}: {cov:.4f}  ({len(valid)} classes)   {args.knn_k}-NN top-1: {knn:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "config": args.as_dict(),
                "coverage": cov,
                "knn_top1": knn,
                "n_valid_classes": len(valid),
                "bank_size": int(feats.shape[0]),
                "per_class_coverage": {str(c): v for c, v in per_class.items()},
            },
            f,
            indent=2,
        )
    print(f"Results saved to {args.out}")
    os._exit(0)  # leftover persistent DataLoader workers can hang interpreter shutdown


if __name__ == "__main__":
    main()
