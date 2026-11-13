"""Hard-label evaluation on ImageNette / ImageWoof (Minimax Diffusion protocol).

Vendored from Minimax Diffusion's ``train.py`` (IDC lineage): ResNet-18 with
instance norm (GroupNorm with one group per channel), SGD lr 0.01 / momentum 0.9
/ wd 5e-4, MultiStepLR at [2E/3, 5E/6] with gamma 0.2, batch 64, CutMix (beta 1,
p 1), 224px, RRC(0.5) + flip + ColorJitter + PCA lighting, distilled images
cached in RAM.  Epochs follow the upstream ladder ``ipc_epoch``: E = 3000 / 2000
/ 1500 at IPC 1 / 10 / 50.  Validation (Resize 224, CenterCrop 224) every E/100
epochs; the reported number is the last one, ``final_top1``.

    python -m src.evaluation.train_dc --data DIR --dataset imagewoof --out results/dc.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import check_distilled, real_dir
from .dc_utils import (
    AverageMeter,
    ImageFolder,
    MultiEpochsDataLoader,
    ResNet,
    accuracy,
    rand_bbox,
    random_indices,
    transform_imagenet,
)


def ipc_epoch(ipc: int) -> int:
    """Training epochs, verbatim from IDC / ManifoldGD ``argument.py`` (factor 1)."""
    if ipc == 1:
        return 3000
    if ipc <= 10:
        return 2000
    if ipc <= 50:
        return 1500
    if ipc <= 200:
        return 1000
    if ipc <= 500:
        return 500
    return 300


def build_loaders(
    train_dir: str, val_dir: str, batch_size: int, workers: int, ipc: int
) -> tuple[MultiEpochsDataLoader, MultiEpochsDataLoader]:
    train_tf, test_tf = transform_imagenet(size=224)
    train_ds = ImageFolder(train_dir, train_tf, ipc=ipc, load_memory=True)
    val_ds = ImageFolder(val_dir, test_tf, ipc=-1, load_memory=False)
    assert train_ds.classes == val_ds.classes or len(train_ds.classes) == len(val_ds.classes)
    train_loader = MultiEpochsDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
    )
    val_loader = MultiEpochsDataLoader(
        val_ds,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    return train_loader, val_loader


def train(
    args: argparse.Namespace, model: nn.Module, train_loader: MultiEpochsDataLoader, val_loader: MultiEpochsDataLoader
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2
    )

    best_acc1, acc1 = 0, 0
    model = model.cuda()
    for epoch in range(1, args.epochs + 1):
        train_epoch(args, train_loader, model, criterion, optimizer, epoch)
        if epoch % args.val_every == 0:
            acc1, acc5, _ = validate(args, val_loader, model, criterion, epoch)
            if acc1 > best_acc1:
                best_acc1 = acc1
                print(f"Best accuracy (top-1 and 5): {best_acc1:.1f} {acc5:.1f}", flush=True)
        scheduler.step()
    return best_acc1, acc1


def train_epoch(
    args: argparse.Namespace,
    train_loader: MultiEpochsDataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
) -> None:
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if r < args.mix_p:
            # CutMix
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target)
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1.0 - ratio)
        else:
            output = model(input)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % args.val_every == 0:
        print(
            f'(Train) [Epoch {epoch}/{args.epochs}] {time.strftime("[%Y-%m-%d %H:%M:%S]")} '
            f"Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {losses.avg:.3f}",
            flush=True,
        )


def validate(
    args: argparse.Namespace, val_loader: MultiEpochsDataLoader, model: nn.Module, criterion: nn.Module, epoch: int
) -> tuple[float, float, float]:
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()
    for input, target in val_loader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))
    print(
        f'(Test ) [Epoch {epoch}/{args.epochs}] {time.strftime("[%Y-%m-%d %H:%M:%S]")} '
        f"Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {losses.avg:.3f}",
        flush=True,
    )
    return top1.avg, top5.avg, losses.avg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="Distilled ImageFolder dir ({class:04d}/*.png).")
    p.add_argument("--dataset", default="imagewoof", choices=["imagenette", "imagewoof"])
    p.add_argument("--data_root", default="data", help="Holds the real val split (see src/evaluation/data.py).")
    p.add_argument("--out", required=True, help="Results JSON path.")
    p.add_argument("--epochs", type=int, default=None, help="Default: ipc_epoch(ipc) = 3000/2000/1500.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    # Upstream never seeds this harness; we seed for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ipc = check_distilled(args.data, args.dataset)
    args.epochs = args.epochs or ipc_epoch(ipc)
    args.val_every = max(1, args.epochs // 100)
    args.lr, args.momentum, args.weight_decay, args.batch_size = 0.01, 0.9, 5e-4, 64
    args.mix_p, args.beta = 1.0, 1.0
    val_dir = real_dir(args.data_root, args.dataset, "val")
    print(f"[dc] train={args.data}  val={val_dir}", flush=True)

    train_loader, val_loader = build_loaders(args.data, val_dir, args.batch_size, args.workers, ipc)
    print(
        f"[dc] ResNet-18 (instance norm), SGD(lr={args.lr}, wd={args.weight_decay}), "
        f"epochs={args.epochs}, milestones={[2 * args.epochs // 3, 5 * args.epochs // 6]}, "
        f"val_every={args.val_every}, cutmix",
        flush=True,
    )

    model = ResNet(len(train_loader.dataset.classes), norm_type="instance")
    best_acc, last_acc = train(args, model, train_loader, val_loader)
    print(f"[dc] best_top1={best_acc:.3f} final_top1={last_acc:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"config": vars(args), "ipc": ipc, "epochs": args.epochs, "best_top1": best_acc, "final_top1": last_acc},
            f,
            indent=2,
        )
    print(f"[dc] wrote {args.out}", flush=True)
    # MultiEpochsDataLoader(persistent_workers=True) deadlocks at teardown; exit hard.
    os._exit(0)


if __name__ == "__main__":
    main()
