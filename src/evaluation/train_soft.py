"""From-scratch ResNet-18 on a distilled ImageNet-1k set, with soft or hard labels.

Soft labels (``--alpha 0.9``, default): before training, a torchvision ResNet-18
teacher computes exactly one soft label per distilled image, the mean of its
softmax(logits / T) over 10 RandomResizedCrop(scale [min_crop, 1]) + flip views.
The student minimizes

    alpha * T^2 * CE(softmax(z / T), y_soft) + (1 - alpha) * CE_ls(z, y_hard)

Hard labels (``--alpha 0``): same recipe, only the label-smoothed CE; no teacher.

Recipe: AdamW lr 5e-4 (20-epoch linear warmup, cosine to 1e-5), weight decay
0.5 -> 0.1 (cosine), 1000 epochs, batch 50, EMA 0.9995, RRC(0.3) + flip +
TrivialAugmentWide, MixUp(0.2, p=0.5).  Reported number: top-1 of the
final-epoch (non-EMA) model on the ImageNet-1k val set (Resize 256, CenterCrop
224), ``results.final_val_accuracy_pct``.

    python -m src.evaluation.train_soft --data DIR --out results/soft.json
    python -m src.evaluation.train_soft --data DIR --out results/hard.json --alpha 0

Preemption: a checkpoint is written every 50 epochs and resumed automatically.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image

from .data import check_distilled, real_dir

_NORM = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_CKPT_EVERY = 50


def mixup_data(x: Tensor, y_soft: Tensor, alpha: float = 0.2) -> tuple[Tensor, Tensor, float, Tensor]:
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_mixed = lam * y_soft + (1 - lam) * y_soft[idx]
    return x_mixed, y_mixed, lam, idx


class _PreloadDecodeDS(Dataset):
    """Decode-only dataset used to parallel-load all images into RAM once."""

    def __init__(self, samples: list[tuple[str, int]]):
        self.samples = samples
        self._to_u8 = T.PILToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tensor:
        return self._to_u8(Image.open(self.samples[i][0]).convert("RGB"))


def _preload_images(image_folder: ImageFolder, num_workers: int) -> Tensor:
    """Decode every image once into a shared-memory uint8 (N,C,H,W) tensor.

    ``__getitem__`` rebuilds the PIL image via ``to_pil_image``, a lossless
    round-trip for uint8 RGB, so the transform sees byte-identical input.
    """
    from tqdm import tqdm

    loader = DataLoader(_PreloadDecodeDS(image_folder.samples), batch_size=64, shuffle=False, num_workers=num_workers)
    chunks = [b for b in tqdm(loader, desc="preload", unit="batch")]
    images = torch.cat(chunks).share_memory_()
    print(f"Preloaded {images.shape[0]} images " f"({images.nbytes / 1024 ** 3:.2f} GB) into RAM", flush=True)
    return images


class SoftLabelDataset(Dataset):
    """Preloaded ImageFolder yielding ``(image, soft_label, hard_label)``."""

    def __init__(
        self,
        image_folder: ImageFolder,
        soft_labels: Tensor,
        transform: Callable[[Image.Image], Tensor],
        preload_workers: int = 16,
    ):
        self.image_folder = image_folder
        self.soft_labels = soft_labels
        self.transform = transform
        self.images = _preload_images(image_folder, preload_workers)

    def __len__(self) -> int:
        return len(self.image_folder)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        hard_label = self.image_folder.samples[idx][1]
        img = self.transform(to_pil_image(self.images[idx]))
        return img, self.soft_labels[idx], hard_label


def compute_soft_labels(
    image_folder: ImageFolder,
    teacher: nn.Module,
    device: torch.device,
    temperature: float,
    num_crops: int,
    min_crop: float,
) -> Tensor:
    """One soft label per image: mean over ``num_crops`` RandomResizedCrop + flip
    views of the teacher's softmax(logits / T)."""
    aug_tf = T.Compose(
        [T.Resize(256), T.RandomResizedCrop(224, scale=(min_crop, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), _NORM]
    )
    soft_labels = []
    teacher.eval()
    with torch.no_grad():
        for path, _ in image_folder.samples:
            img = Image.open(path).convert("RGB")
            views = torch.stack([aug_tf(img) for _ in range(num_crops)])
            logits = teacher(views.to(device))
            soft_labels.append(F.softmax(logits / temperature, dim=1).mean(0).cpu())
    return torch.stack(soft_labels)


def soft_cross_entropy(pred: Tensor, target: Tensor, temperature: float = 1.0) -> Tensor:
    """Cross entropy with soft targets, with KD temperature scaling (T^2)."""
    log_probs = F.log_softmax(pred / temperature, dim=1)
    return -(target * log_probs).sum(dim=1).mean() * (temperature**2)


def _effective_lr(epoch: int, lr: float, epochs: int, warmup_epochs: int, min_lr: float) -> float:
    """Linear warmup for epochs [0, warmup_epochs), then cosine decay to min_lr."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return lr * float(epoch + 1) / float(max(1, warmup_epochs))
    if warmup_epochs > 0:
        t = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
    else:
        t = epoch / float(max(1, epochs - 1))
    t = min(1.0, max(0.0, t))
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(t * math.pi))


def _effective_wd(epoch: int, wd0: float, wd_final: float, epochs: int) -> float:
    """Cosine interpolation of weight decay from wd0 to wd_final over training."""
    t = min(1.0, max(0.0, epoch / float(max(1, epochs - 1))))
    return wd_final + (wd0 - wd_final) * 0.5 * (1.0 + math.cos(t * math.pi))


@torch.no_grad()
def _ema_update(ema_model: nn.Module, src_model: nn.Module, decay: float) -> None:
    for pe, ps in zip(ema_model.parameters(), src_model.parameters(), strict=True):
        pe.mul_(decay).add_(ps.detach(), alpha=1.0 - decay)
    for be, bs in zip(ema_model.buffers(), src_model.buffers(), strict=True):
        be.copy_(bs)


def _val_accuracy(
    model: nn.Module, val_loader: DataLoader, device: torch.device, num_classes: int
) -> tuple[float, float]:
    """Center-crop val; returns (acc1_pct, acc5_pct)."""
    model.eval()
    c1, c5, vt = 0, 0, 0
    maxk = min(5, int(num_classes))
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            c1 += (out.argmax(1) == labels).sum().item()
            _, pred = out.topk(maxk, dim=1, largest=True, sorted=True)
            c5 += pred.eq(rearrange(labels, "n -> n 1").expand_as(pred)).sum().item()
            vt += labels.size(0)
    return 100.0 * c1 / vt, 100.0 * c5 / vt


class _FiveCropStack:
    """Picklable replacement for a lambda (lambdas break DataLoader workers)."""

    def __call__(self, crops: tuple[Image.Image, ...]) -> Tensor:
        return torch.stack([_NORM(T.ToTensor()(c)) for c in crops])


def _tta_five_crop_accuracy(
    model: nn.Module, val_dir: str, device: torch.device, num_classes: int
) -> tuple[float, float]:
    """Five-crop TTA (center + 4 corners); returns (acc1_pct, acc5_pct)."""
    tta_ds = ImageFolder(val_dir, T.Compose([T.Resize(256), T.FiveCrop(224), _FiveCropStack()]))
    maxk = min(5, num_classes)
    tta_loader = DataLoader(tta_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    model.eval()
    c1, c5, vt = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tta_loader:
            bs = imgs.shape[0]
            imgs = rearrange(imgs, "b n c h w -> (b n) c h w").to(device)
            labels = labels.to(device)
            out = rearrange(model(imgs), "(b n) k -> b n k", b=bs).mean(dim=1)
            c1 += (out.argmax(1) == labels).sum().item()
            _, pred = out.topk(maxk, dim=1, largest=True, sorted=True)
            c5 += pred.eq(rearrange(labels, "n -> n 1").expand_as(pred)).sum().item()
            vt += labels.size(0)
    return 100.0 * c1 / vt, 100.0 * c5 / vt


def _save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="Distilled ImageFolder dir ({class:04d}/*.png).")
    p.add_argument("--data_root", default="data", help="Holds imagenet/val (see src/evaluation/data.py).")
    p.add_argument("--out", required=True, help="Results JSON path.")
    p.add_argument("--ckpt", default=None, help="Checkpoint path (default: <out stem>_ckpt.pt).")
    p.add_argument("--no_resume", action="store_true", help="Ignore an existing checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.9, help="Soft-loss weight; 0 = hard labels.")
    p.add_argument("--temperature", type=float, default=10.0, help="Teacher and student temperature T.")
    p.add_argument("--teacher_num_crops", type=int, default=10)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=0.5)
    p.add_argument("--wd_final", type=float, default=0.1)
    p.add_argument("--min_crop", type=float, default=0.3, help="RandomResizedCrop scale lower bound.")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.9995)
    args = p.parse_args()
    ckpt_path = args.ckpt or os.path.splitext(args.out)[0] + "_ckpt.pt"
    print("Run configuration:", json.dumps(vars(args), indent=2), flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    check_distilled(args.data, "imagenet-1k")
    val_dir = real_dir(args.data_root, "imagenet-1k", "val")

    raw_ds = ImageFolder(args.data)
    num_classes = len(raw_ds.classes)
    print(f"Train: {len(raw_ds)} images, {num_classes} classes", flush=True)

    if args.alpha == 0.0:
        print("Hard-label mode (--alpha 0): no teacher, no soft labels.", flush=True)
        soft_labels = torch.zeros(len(raw_ds), num_classes)
    else:
        teacher = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).to(device)
        print(
            f"Computing soft labels (teacher resnet18, T={args.temperature}, "
            f"num_crops={args.teacher_num_crops})...",
            flush=True,
        )
        soft_labels = compute_soft_labels(
            raw_ds, teacher, device, args.temperature, args.teacher_num_crops, args.min_crop
        )
        print(f"Soft labels computed: {soft_labels.shape}", flush=True)

    train_tf = T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(args.min_crop, 1.0)),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            _NORM,
        ]
    )
    val_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM])

    train_ds = SoftLabelDataset(raw_ds, soft_labels, train_tf)
    val_ds = ImageFolder(val_dir, val_tf)
    print(f"Val: {len(val_ds)} images ({val_dir})", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True
    )

    model = tvm.resnet18(weights=None)
    if num_classes != 1000:
        model.fc = nn.Linear(512, num_classes)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    ema_model = copy.deepcopy(model).eval()
    for _p in ema_model.parameters():
        _p.requires_grad_(False)

    start_epoch = 0
    if not args.no_resume and os.path.isfile(ckpt_path):
        print(f"Resuming from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        ema_model.load_state_dict(ckpt["ema_model"])
        start_epoch = ckpt["epoch"]

    for epoch in range(start_epoch, args.epochs):
        lr_now = _effective_lr(epoch, args.lr, args.epochs, args.warmup_epochs, args.min_lr)
        wd_now = _effective_wd(epoch, args.weight_decay, args.wd_final, args.epochs)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
            pg["weight_decay"] = wd_now

        model.train()
        correct, total, epoch_loss = 0, 0, 0.0
        for imgs, soft_tgt, hard_tgt in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            soft_tgt = soft_tgt.to(device, non_blocking=True)
            hard_tgt = hard_tgt.to(device, non_blocking=True)

            mix_lam, mix_idx = 1.0, None
            if torch.rand(1).item() < 0.5:
                imgs, soft_tgt, mix_lam, mix_idx = mixup_data(imgs, soft_tgt)

            opt.zero_grad()
            out = model(imgs)
            loss_soft = soft_cross_entropy(out, soft_tgt, temperature=args.temperature)
            if mix_idx is not None:
                loss_hard = mix_lam * ce_criterion(out, hard_tgt) + (1 - mix_lam) * ce_criterion(out, hard_tgt[mix_idx])
            else:
                loss_hard = ce_criterion(out, hard_tgt)
            loss = args.alpha * loss_soft + (1 - args.alpha) * loss_hard
            loss.backward()
            opt.step()
            _ema_update(ema_model, model, args.ema_decay)
            epoch_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == hard_tgt).sum().item()
            total += hard_tgt.size(0)

        if (epoch + 1) % _CKPT_EVERY == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch + 1}/{args.epochs}  train={100. * correct / total:.1f}%  "
                f"train_loss={epoch_loss / total:.4f}  lr={lr_now:.6f}",
                flush=True,
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "ema_model": ema_model.state_dict(),
                },
                ckpt_path,
            )

    final_val, final_val5 = _val_accuracy(model, val_loader, device, num_classes)
    print(f"\nFinal val @1={final_val:.2f}%  @5={final_val5:.2f}%", flush=True)
    ema_val, ema_val5 = _val_accuracy(ema_model, val_loader, device, num_classes)
    print(f"Final EMA val @1={ema_val:.2f}%  @5={ema_val5:.2f}%", flush=True)

    results = {
        "final_val_accuracy_pct": round(final_val, 4),
        "final_val_accuracy5_pct": round(final_val5, 4),
        "final_ema_val_acc_pct": round(ema_val, 4),
        "final_ema_val_acc5_pct": round(ema_val5, 4),
        "epochs": args.epochs,
    }
    payload = {
        "config": vars(args),
        "status": "completed",
        "results": results,
        "dataset": {"num_train": len(raw_ds), "num_val": len(val_ds), "num_classes": num_classes},
    }
    _save_json(args.out, payload)  # persisted before TTA, which can hit walltime/preemption

    print("Running 5-crop TTA on val set...", flush=True)
    tta, tta5 = _tta_five_crop_accuracy(model, val_dir, device, num_classes)
    print(f"TTA val @1={tta:.2f}%  @5={tta5:.2f}%", flush=True)
    tta_ema, tta_ema5 = _tta_five_crop_accuracy(ema_model, val_dir, device, num_classes)
    print(f"TTA EMA val @1={tta_ema:.2f}%  @5={tta_ema5:.2f}%", flush=True)
    results.update(
        {
            "tta_val_acc_pct": round(tta, 4),
            "tta_val_acc5_pct": round(tta5, 4),
            "tta_ema_val_acc_pct": round(tta_ema, 4),
            "tta_ema_val_acc5_pct": round(tta_ema5, 4),
        }
    )
    _save_json(args.out, payload)
    print(f"Saved results to {args.out}", flush=True)

    # persistent_workers DataLoaders can deadlock at interpreter teardown; exit hard.
    os._exit(0)


if __name__ == "__main__":
    main()
