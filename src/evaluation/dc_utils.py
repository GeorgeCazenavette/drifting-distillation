"""Model, data and augmentation for ``train_dc``, vendored from Minimax Diffusion
(itself IDC / DatasetCondensation): the ImageNet-style ResNet with GroupNorm
"instance" norm, the RAM-preloading ImageFolder, the train/test transforms
(RRC + flip + ColorJitter + PCA lighting), and the multi-epoch DataLoader."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from einops import rearrange
from torch import Tensor

if TYPE_CHECKING:
    from PIL import Image

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------- model
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def normalization(inplanes: int, norm_type: str) -> nn.BatchNorm2d | nn.GroupNorm:
    if norm_type == "batch":
        return nn.BatchNorm2d(inplanes)
    if norm_type == "instance":
        return nn.GroupNorm(inplanes, inplanes)
    raise AssertionError(f"Check normalization type! {norm_type}")


class IntroBlock(nn.Module):
    """7x7/2 conv + norm + ReLU + 3x3/2 max-pool (the 'large' input stem)."""

    def __init__(self, planes: int, norm_type: str, nch: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(nch, planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = normalization(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.relu(self.bn1(self.conv1(x))))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, norm_type: str = "batch", stride: int = 1, downsample: nn.Module | None = None
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization(planes, norm_type)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normalization(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet(nn.Module):
    """ImageNet ResNet-18 at 224px ('large' stem, BasicBlock [2, 2, 2, 2])."""

    def __init__(self, num_classes: int, norm_type: str = "instance", nch: int = 3):
        super().__init__()
        self.norm_type = norm_type
        self.inplanes = 64
        self.layer0 = IntroBlock(self.inplanes, norm_type, nch=nch)
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                normalization(planes * block.expansion, self.norm_type),
            )
        layers = [block(self.inplanes, planes, norm_type=self.norm_type, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=self.norm_type))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.shape[-1])
        x = rearrange(x, "n c 1 1 -> n c")
        return self.fc(x)


# ----------------------------------------------------------------------------- augmentation
class Compose:
    def __init__(self, transforms: list[Callable[[Tensor], Tensor]]):
        self.transforms = transforms

    def __call__(self, img: Tensor) -> Tensor:
        for t in self.transforms:
            img = t(img)
        return img


class Lighting:
    """Lighting noise (AlexNet-style PCA-based noise)."""

    def __init__(
        self, alphastd: float, eigval: list[float], eigvec: list[list[float]], device: str | torch.device = "cpu"
    ):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, device=device)
        self.eigvec = torch.tensor(eigvec, device=device)

    def __call__(self, img: Tensor) -> Tensor:
        if self.alphastd == 0:
            return img
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(rearrange(alpha, "c -> 1 c").expand(3, 3))
            .mul(rearrange(self.eigval, "c -> 1 c").expand(3, 3))
            .sum(1)
            .squeeze()
        )
        return img + rearrange(rgb, "c -> c 1 1").expand_as(img)


class Grayscale:
    def __call__(self, img: Tensor) -> Tensor:
        gs = img.clone()
        gs[0].mul_(0.299).add_(gs[1], alpha=0.587).add_(gs[2], alpha=0.114)
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation:
    def __init__(self, var: float):
        self.var = var

    def __call__(self, img: Tensor) -> Tensor:
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness:
    def __init__(self, var: float):
        self.var = var

    def __call__(self, img: Tensor) -> Tensor:
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast:
    def __init__(self, var: float):
        self.var = var

    def __call__(self, img: Tensor) -> Tensor:
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter:
    def __init__(self, brightness: float = 0.4, contrast: float = 0.4, saturation: float = 0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img: Tensor) -> Tensor:
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))
        random.shuffle(self.transforms)
        return Compose(self.transforms)(img)


def transform_imagenet(size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Train: Resize+CenterCrop(size), RRC(size, 0.5-1), flip, ColorJitter, lighting.
    Test: Resize(size)+CenterCrop(size)."""
    resize = [transforms.Resize(size), transforms.CenterCrop(size)]
    jittering = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    lighting = Lighting(
        alphastd=0.1,
        eigval=[0.2175, 0.0188, 0.0045],
        eigvec=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]],
    )
    aug = [transforms.RandomResizedCrop(size, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(), jittering, lighting]
    normal_fn = [transforms.Normalize(mean=MEAN, std=STD)]
    train_transform = transforms.Compose(resize + [transforms.ToTensor()] + aug + normal_fn)
    test_transform = transforms.Compose(resize + [transforms.ToTensor()] + normal_fn)
    return train_transform, test_transform


# ----------------------------------------------------------------------------- data
class ImageFolder(datasets.DatasetFolder):
    """Sorted-subfolder ImageFolder that keeps the first ``ipc`` images per class
    and optionally caches the decoded PIL images in RAM."""

    def __init__(
        self,
        root: str,
        transform: Callable[[Image.Image], Tensor] | None = None,
        loader: Callable[[str], Image.Image] = datasets.folder.default_loader,
        load_memory: bool = False,
        ipc: int = -1,
    ):
        super().__init__(root, loader, IMG_EXTENSIONS, transform=transform)
        self.nclass = len(self.classes)
        if ipc > 0:
            self.samples = self._subset(ipc=ipc)
        self.targets = [s[1] for s in self.samples]
        self.load_memory = load_memory
        if self.load_memory:
            self.imgs = self._load_images()
        else:
            self.imgs = self.samples

    def _subset(self, ipc: int = 10) -> list[tuple[str, int]]:
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            idx_class[self.samples[i][1]].append(i)
        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc <= min_class

        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in np.arange(n):
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])
            if len(samples_subset) == ipc * self.nclass:
                break
        return samples_subset

    def _load_images(self) -> list[Image.Image]:
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            imgs.append(self.loader(path))
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end="\r")
        print(" " * 50, end="\r")
        return imgs

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, int]:
        sample = self.imgs[index] if self.load_memory else self.loader(self.samples[index][0])
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class _RepeatSampler:
    """Sampler that repeats forever."""

    def __init__(self, sampler: torch.utils.data.Sampler[list[int]]):
        self.sampler = sampler

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            yield from iter(self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """DataLoader whose workers and sampler persist across epochs."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()  # Init iterator and sampler once

        self.convert = None
        if self.dataset[0][0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        if self.dataset[0][0].device == torch.device("cpu"):
            self.device = "cpu"
        else:
            self.device = "cuda"

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for i in range(len(self)):
            data, target = next(self.iterator)
            if self.convert is not None:
                data = self.convert(data)
            yield data, target


# ----------------------------------------------------------------------------- training utils
def accuracy(output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)) -> list[Tensor]:
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = rearrange(pred, "n k -> k n")
    correct = pred.eq(rearrange(target, "n -> 1 n").expand_as(pred))

    res = []
    for k in topk:
        correct_k = rearrange(correct[:k], "k n -> (k n)").float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def random_indices(y: Tensor, device: str | torch.device = "cuda") -> Tensor:
    return torch.randperm(len(y)).to(device)


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
