"""Helpers for the EDC knowledge-distillation recipe (``train_kd``), copied from
EDC / SRe2L (``utils.py`` + ``utils_fkd.py``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from torch import Tensor


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)) -> list[Tensor]:
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = rearrange(pred, "n k -> k n")
    correct = pred.eq(rearrange(target, "n -> 1 n").expand_as(pred))

    res = []
    for k in topk:
        correct_k = rearrange(correct[:k], "k n -> (k n)").float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Weight decay on conv/linear weights only; biases and norm params get none."""
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find("weight") >= 0 and len(p.size()) > 1:
            group_weight_decay.append(p)
        else:
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)
    return [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.0)]


class ToFloatImage(torch.nn.Module):
    """uint8 CHW tensor -> float32 CHW in [0, 1] (``ToTensor`` for preloaded tensors)."""

    def forward(self, img: Tensor) -> Tensor:
        return img.to(torch.float32).div_(255.0)


class _PathDecodeDataset(torch.utils.data.Dataset):
    """sample index -> decoded uint8 CHW tensor (picklable, for the preload loader)."""

    def __init__(self, samples: list[tuple[str, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tensor:
        return torchvision.io.decode_image(self.samples[i][0], mode=torchvision.io.ImageReadMode.RGB)


def preload_uint8_tensor(samples: list[tuple[str, int]], num_workers: int = 8, batch_size: int = 64) -> Tensor:
    """Decode all images into one shared-memory uint8 (N,C,H,W) tensor, in parallel."""
    from tqdm import tqdm

    loader = torch.utils.data.DataLoader(
        _PathDecodeDataset(samples), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    chunks = [batch for batch in tqdm(loader, desc="preload", unit="batch")]
    images = torch.cat(chunks).share_memory_()
    print(f"Preloaded {images.shape[0]} images " f"({images.nbytes / 1024 ** 3:.2f} GB) into RAM", flush=True)
    return images


def rand_bbox(size: torch.Size, lam: float) -> tuple[np.int64, np.int64, np.int64, np.int64]:
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(images: Tensor, alpha: float = 1.0) -> Tensor:
    """In-place CutMix of a batch with a random permutation of itself."""
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images
