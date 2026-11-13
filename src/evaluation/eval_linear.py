"""Augmented linear probe of a distilled set in a frozen feature space.

Bank: each distilled image's center crop together with
10 RandomResizedCrops (scale [0.7, 1]) + flip per image, embedded once.  A
linear head is trained on these frozen, unnormalized features (AdamW lr 1e-3,
wd 0, batch 256 sampled uniformly with replacement, cosine annealing to 0 over
``--probe_steps``: 4000 at IPC 10, 19600 at IPC 50) and evaluated on the real
val set (Resize 256, CenterCrop 224; embeddings cached under ``--cache_dir``).

    python -m src.evaluation.eval_linear --data DIR --backbone siglip2 --out results/linear_siglip2.json
"""

from __future__ import annotations

import json
import os

import torch
from tap import Tap

from ..extractors import get_extractor
from .data import NUM_CLASSES, check_distilled
from .features import query_features, support_bank
from .linear_probe import fit_linear_head

PROBE_STEPS = {10: 4000, 50: 19600}


class LinearArgs(Tap):
    data: str  # Distilled ImageFolder dir ({class:04d}/*.png)
    out: str  # Results JSON path
    backbone: str = "siglip2"  # Eval space (paper: siglip2, mocov3)
    mode: str = "cls"  # Extractor feature mode
    dataset: str = "imagenet-1k"  # imagenet-1k | imagenette | imagewoof
    data_root: str = "data"  # Holds the real val split (see src/evaluation/data.py)
    cache_dir: str = ".val_cache"  # Real val embedding cache
    probe_steps: int | None = None  # Optimizer steps (default: 4000 at IPC 10, 19600 at IPC 50)
    lr: float = 1e-3
    weight_decay: float = 0.0
    head_batch_size: int = 256
    num_crops: int = 10  # RandomResizedCrop passes appended to the center crops
    min_scale: float = 0.7  # RandomResizedCrop scale lower bound
    crop_seed: int = 0  # Seed of the crop passes
    seed: int = 42
    batch_size: int = 256  # Extractor batch size
    num_workers: int = 8  # Also fixes which worker draws which crop


def main() -> None:
    args = LinearArgs(explicit_bool=True).parse_args()
    assert args.dataset in NUM_CLASSES, args.dataset
    ipc = check_distilled(args.data, args.dataset)
    if args.probe_steps is None:
        if ipc not in PROBE_STEPS:
            raise SystemExit(f"no default --probe_steps for IPC {ipc}; pass one explicitly")
        args.probe_steps = PROBE_STEPS[ipc]
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    print(f"Loading extractor: backbone={args.backbone}, mode={args.mode}")
    extractor = get_extractor(args.backbone, mode=args.mode).to(device).eval()
    qfeats, qlabels = query_features(extractor, args, device)
    feats, labels = support_bank(extractor, args, device)

    num_classes = int(max(labels.max().item(), qlabels.max().item())) + 1
    metrics = fit_linear_head(
        feats,
        labels,
        qfeats,
        qlabels,
        num_classes=num_classes,
        steps=args.probe_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.head_batch_size,
        device=device,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "config": args.as_dict(),
                "acc1": metrics["acc1"],
                "acc5": metrics["acc5"],
                "bank_size": int(feats.shape[0]),
            },
            f,
            indent=2,
        )
    print(f"Results saved to {args.out}")
    os._exit(0)  # leftover persistent DataLoader workers can hang interpreter shutdown


if __name__ == "__main__":
    main()
