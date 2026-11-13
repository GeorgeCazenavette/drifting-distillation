"""Real-image initialization via RDED's patch selection with factor 1.

Re-implements ``RDED/synthesize`` (Sun et al., "On the Diversity and Realism
of Distilled Dataset", CVPR 2024) for ``--factor 1 --num-crop 5 --mipc 300``,
the configuration whose images initialize our real-init variants:

1. sample ``mipc`` real images of the class;
2. take ``num_crop`` random square ``RandomResizedCrop``s (224 px) of each;
3. score every crop by the teacher's cross-entropy on the true class and keep
   the best crop per image;
4. keep the ``ipc`` crops with the lowest cross-entropy overall.

With factor 1 RDED stitches nothing, so each selected crop is one image.
Unlike RDED, sampling is seeded and the crops are returned as tensors rather
than round-tripped through JPEG.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from torch import Tensor

from ..datasets import class_image_paths
from ..download import TEACHERS, teacher as hosted_teacher

_IMAGENET_MEAN = rearrange(torch.tensor([0.485, 0.456, 0.406]), "c -> 1 c 1 1")
_IMAGENET_STD = rearrange(torch.tensor([0.229, 0.224, 0.225]), "c -> 1 c 1 1")


def load_teacher(teacher: str | None, device: str) -> torch.nn.Module:
    """Torchvision ResNet-18 teacher.

    ``None`` = ImageNet-1k ``IMAGENET1K_V1`` weights (RDED's ImageNet-1k
    observer); ``"imagenette"`` / ``"imagewoof"`` = that dataset's hosted teacher
    (downloaded on first use).  Otherwise a checkpoint path holding ``{"model": state_dict}``
    of a torchvision ResNet-18 whose ``fc`` is sized to the dataset's classes,
    in sorted-folder order (e.g. the ImageNette/ImageWoof ``*_wnid.pt``
    teachers).
    """
    if teacher is None:
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    else:
        if teacher in TEACHERS:
            teacher = hosted_teacher(teacher)
        sd = torch.load(teacher, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
        model = tvm.resnet18(weights=None, num_classes=sd["fc.weight"].shape[0])
        model.load_state_dict(sd)
    return model.to(device).eval()


@torch.no_grad()
def rded_select(
    root: str,
    class_idx: int,
    ipc: int,
    teacher: torch.nn.Module,
    device: str,
    seed: int = 0,
    mipc: int = 300,
    num_crop: int = 5,
    size: int = 224,
    batch_size: int = 300,
) -> Tensor:
    """Return ``ipc`` RDED-selected crops of class ``class_idx``: ``[ipc, 3, size, size]`` in ``[0, 1]``."""
    paths = class_image_paths(root, class_idx)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(paths), generator=g)[:mipc].tolist()

    cropper = T.RandomResizedCrop(size, ratio=(1, 1), antialias=True)
    to_tensor = T.ToTensor()

    # RandomResizedCrop draws from the global RNG; fork it so selection does
    # not perturb the distillation's random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        crops = torch.stack(
            [
                torch.stack([cropper(to_tensor(Image.open(paths[i]).convert("RGB"))) for _ in range(num_crop)])
                for i in order
            ]
        )  # [M, num_crop, 3, size, size]

    M = crops.shape[0]
    flat = rearrange(crops, "m k c h w -> (m k) c h w")
    ce = []
    for s in range(0, flat.shape[0], batch_size):
        x = flat[s : s + batch_size].to(device)
        logits = teacher((x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device))
        target = torch.full((x.shape[0],), class_idx, device=device)
        ce.append(F.cross_entropy(logits.float(), target, reduction="none").cpu())
    ce = rearrange(torch.cat(ce), "(m k) -> m k", k=num_crop)

    best_crop = ce.argmin(dim=1)  # best crop per image
    best_ce = ce[torch.arange(M), best_crop]
    keep = best_ce.argsort()[:ipc]  # lowest-CE images
    return crops[keep, best_crop[keep]]
