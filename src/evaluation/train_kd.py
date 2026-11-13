"""Knowledge-distillation evaluation (EDC recipe) of a distilled set.

A ResNet-18 student is trained from scratch; at every step the teacher(s) label
the exact (augmented, CutMix-ed) batch the student sees, and the student
minimizes ``MSE(z, mean_m f_m(x)) + 0.025 * CE(z, y)``.

Teachers:
  * ImageNet-1k: the 4-model EDC ensemble (torchvision ImageNet weights of
    resnet18, mobilenet_v2, shufflenet_v2_x0_5, alexnet); mean of raw logits.
  * ImageNette / ImageWoof: one ResNet-18 checkpoint of that dataset
    (default: the hosted converted RDED teacher, downloaded on first use;
    ``--teacher_weights`` overrides it with a torchvision resnet18 state_dict
    under ``"model"``, 10-class fc, sorted-wnid class order); RandAugment(2, 9)
    is added.

Recipe: AdamW lr 1e-3, wd 0.01 (weights only), cosine schedule with
deceleration zeta=2 (stepped per epoch), EMA 0.99 (parameters only),
RRC(0.5) + flip, CutMix(1.0); 300 epochs / batch 100 (ImageNet-1k) or 1000
epochs / batch 50 (ImageNette, ImageWoof).  Reported number: top-1 of the EMA
weights after the last epoch on the val set (Resize 256, CenterCrop 224),
``final_top1``.

    python -m src.evaluation.train_kd --data DIR --dataset imagenet-1k --out results/kd.json
    python -m src.evaluation.train_kd --data DIR --dataset imagewoof --out results/kd.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from ..download import teacher as hosted_teacher
from .data import NUM_CLASSES, check_distilled, real_dir
from .edc_utils import AverageMeter, ToFloatImage, accuracy, cutmix, get_parameters, preload_uint8_tensor

# EDC's relabel teacher pool; the first 4 form the ImageNet-1k ensemble.
EDC_TEACHERS = ["resnet18", "mobilenet_v2", "shufflenet_v2_x0_5", "alexnet"]


class EMAMODEL:
    def __init__(self, model: nn.Module):
        self.ema_model = deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def ema_step(self, decay_rate: float = 0.999, model: nn.Module | None = None) -> None:
        for param, ema_param in zip(model.parameters(), self.ema_model.parameters(), strict=True):
            ema_param.data.mul_(decay_rate).add_(param.data, alpha=1.0 - decay_rate)

    @torch.no_grad()
    def ema_swap(self, model: nn.Module | None = None) -> None:
        for param, ema_param in zip(self.ema_model.parameters(), model.parameters(), strict=True):
            tmp = param.data.detach()
            param.data = ema_param.detach()
            ema_param.data = tmp


class PreloadedImageFolder(datasets.ImageFolder):
    """ImageFolder that decodes every image once into a shared-memory uint8 tensor."""

    def __init__(self, root: str, transform: Callable[[Tensor], Tensor], preload_workers: int = 8):
        super().__init__(root)  # base transform stays None
        self.kd_transform = transform
        self.images = preload_uint8_tensor(self.samples, num_workers=preload_workers)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.kd_transform(self.images[index]), self.targets[index]


def build_teachers(teacher_weights: str | None = None, nclass: int = 1000) -> list[nn.Module]:
    """The EDC ImageNet ensemble, or one custom ResNet-18 expert."""
    if teacher_weights is not None:
        t = torchvision.models.resnet18(weights=None)
        if nclass != 1000:
            t.fc = nn.Linear(t.fc.in_features, nclass)
        ckpt = torch.load(teacher_weights, map_location="cpu", weights_only=True)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        t.load_state_dict(sd)
        t = t.cuda().eval()
        for p in t.parameters():
            p.requires_grad_(False)
        print(f"=> single expert teacher: resnet18 ({nclass}-class) from {teacher_weights}")
        return [t]
    print(f"=> teacher ensemble: {EDC_TEACHERS}")
    teachers = []
    for name in EDC_TEACHERS:
        t = torchvision.models.get_model(name, weights="IMAGENET1K_V1").cuda().eval()
        for p in t.parameters():
            p.requires_grad_(False)
        teachers.append(t)
    return teachers


@torch.no_grad()
def teacher_soft_label(teachers: list[nn.Module], images: Tensor) -> Tensor:
    """Mean of the teachers' raw logits (EDC's relabel aggregation)."""
    outs = [t(images) for t in teachers]
    return torch.stack(outs, 0).mean(0).float()


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="Distilled ImageFolder dir ({class:04d}/*.png).")
    p.add_argument("--dataset", default="imagenet-1k", choices=sorted(NUM_CLASSES))
    p.add_argument("--data_root", default="data", help="Holds the real val split (see src/evaluation/data.py).")
    p.add_argument("--out", required=True, help="Results JSON path.")
    p.add_argument(
        "--teacher_weights",
        default=None,
        help="10-class ResNet-18 teacher checkpoint for imagenette/imagewoof "
        "(default: the hosted teacher, downloaded on first use).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None, help="Default: 300 (imagenet-1k) / 1000 (others).")
    p.add_argument("--batch_size", type=int, default=None, help="Default: 100 (imagenet-1k) / 50 (others).")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    big = args.dataset == "imagenet-1k"
    if big and args.teacher_weights is not None:
        p.error("--teacher_weights is unused for imagenet-1k (it uses the EDC teacher ensemble)")
    if not big and args.teacher_weights is None:
        args.teacher_weights = hosted_teacher(args.dataset)
    args.nclass = NUM_CLASSES[args.dataset]
    args.epochs = args.epochs or (300 if big else 1000)
    args.batch_size = args.batch_size or (100 if big else 50)
    args.rand_aug = not big
    args.st = 2.0  # zeta: cosine deceleration
    args.ema_dr = 0.99
    args.ce_weight = 0.025
    args.adamw_lr = 1e-3
    args.adamw_weight_decay = 0.01
    args.min_scale_crops = 0.5
    args.cutmix = 1.0
    return args


def main() -> None:
    args = get_args()
    print("Run configuration:", json.dumps(vars(args), indent=2), flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ipc = check_distilled(args.data, args.dataset)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # RandAugment operates on the uint8 image, so it precedes float conversion.
    _pre = [transforms.RandAugment(num_ops=2, magnitude=9)] if args.rand_aug else []
    train_transform = transforms.Compose(
        _pre
        + [
            ToFloatImage(),
            transforms.RandomResizedCrop(
                224, scale=(args.min_scale_crops, 1.0), interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            normalize,
        ]
    )
    train_dataset = PreloadedImageFolder(args.data, train_transform, preload_workers=args.workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_dir = real_dir(args.data_root, args.dataset, "val")
    val_loader = DataLoader(
        datasets.ImageFolder(
            val_dir,
            transform=transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
            ),
        ),
        batch_size=int(args.batch_size / 4),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"load data successfully (val: {val_dir})")

    model = torchvision.models.resnet18(weights=None)
    if args.nclass != 1000:
        model.fc = nn.Linear(model.fc.in_features, args.nclass)
    model = model.cuda()
    model.train()
    ema_model = EMAMODEL(model)

    teachers = build_teachers(args.teacher_weights, args.nclass)

    optimizer = torch.optim.AdamW(get_parameters(model), lr=args.adamw_lr, weight_decay=args.adamw_weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lambda step: (
            0.5 * (1.0 + math.cos(math.pi * step / (args.st * args.epochs))) if step <= (args.st * args.epochs) else 0
        ),
        last_epoch=-1,
    )

    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch}", flush=True)
        train_loss, train_top1 = train(model, teachers, train_loader, optimizer, scheduler, ema_model, args, epoch)
        scheduler.step()

    ema_model.ema_swap(model)
    top1, top5 = validate(model, val_loader, args.epochs - 1)
    ema_model.ema_swap(model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "ipc": ipc,
                "epochs": args.epochs,
                "final_top1": top1,
                "final_top5": top5,
                "train_top1": train_top1,
                "train_loss": train_loss,
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"[kd] final_top1={top1:.3f}  wrote {args.out}", flush=True)
    os._exit(0)  # persistent DataLoader workers can deadlock at teardown


def train(
    model: nn.Module,
    teachers: list[nn.Module],
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    ema_model: EMAMODEL,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[float, float]:
    objs, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    t1 = time.time()
    for images, target in train_loader:
        images = images.cuda()
        target = target.cuda()

        # CutMix the batch, then label the *mixed* image on the fly.
        images = cutmix(images, args.cutmix)
        soft_label = teacher_soft_label(teachers, images)

        optimizer.zero_grad()
        output = model(images).float()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        loss = F.mse_loss(output, soft_label) + F.cross_entropy(output, target) * args.ce_weight
        loss.backward()
        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        optimizer.step()
        ema_model.ema_step(decay_rate=args.ema_dr, model=model)

    print(
        f"TRAIN Iter {epoch}: lr = {scheduler.get_last_lr()[0]:.6f},\tloss = {objs.avg:.6f},\tTop-1 err = {100 - top1.avg:.6f},\ttrain_time = {time.time() - t1:.6f}",
        flush=True,
    )
    return objs.avg, top1.avg


def validate(model: nn.Module, val_loader: DataLoader, epoch: int) -> tuple[float, float]:
    objs, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    loss_function = nn.CrossEntropyLoss()
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = loss_function(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
    print(
        f"TEST Iter {epoch}: loss = {objs.avg:.6f},\tTop-1 err = {100 - top1.avg:.6f},\tval_time = {time.time() - t1:.6f}"
    )
    return top1.avg, top5.avg


if __name__ == "__main__":
    main()
