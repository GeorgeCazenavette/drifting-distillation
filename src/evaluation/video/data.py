"""Video evaluation datasets: distilled/coreset clip folders and multi-crop HMDB51 queries."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torchvision.io as tio
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from ...datasets.hmdb51 import HMDB51Dataset


def _resize_shorter(video: Tensor, res: int, crop_res: int) -> Tensor:
    """Bilinear (antialiased) resize of ``(T, C, H, W)`` so the shorter side is ``res``."""
    h, w = video.shape[-2], video.shape[-1]
    if h <= w:
        new_h, new_w = res, max(crop_res, int(w * res / h))
    else:
        new_h, new_w = max(crop_res, int(h * res / w)), res
    if new_h != h or new_w != w:
        video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    return video


class VideoTensorFolder(Dataset):
    """Class folders of ``clip_*.pt`` tensors ``(T, C, H, W)`` in ``[0, 1]``.

    Class index = sorted folder order.  Each clip gets the same spatial
    transform as the real videos (shorter side to ``res``, center crop to
    ``crop_res``), so a 224 px distilled clip is upsampled to 256 and
    center-cropped back to 224.  Returns ``(clip, label)``.
    """

    def __init__(self, root: str, res: int = 256, crop_res: int = 224):
        self.res = res
        self.crop_res = crop_res
        class_names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.samples: list[tuple[str, int]] = []
        for label, cname in enumerate(class_names):
            cdir = os.path.join(root, cname)
            for fname in sorted(os.listdir(cdir)):
                if fname.lower().endswith(".pt") and fname.startswith("clip_"):
                    self.samples.append((os.path.join(cdir, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path, label = self.samples[idx]
        clip = _resize_shorter(torch.load(path, map_location="cpu").float(), self.res, self.crop_res)
        h, w = clip.shape[-2], clip.shape[-1]
        top, left = (h - self.crop_res) // 2, (w - self.crop_res) // 2
        return clip[:, :, top : top + self.crop_res, left : left + self.crop_res], label


class MultiCropHMDB51(Dataset):
    """HMDB51 videos with ``num_crops`` spatial crops along the longer side.

    Frames are sampled by linspace over the whole video (one temporal view, as
    in distillation).  ``num_crops=3`` gives left/center/right (or
    top/center/bottom) crops; view 1 is then exactly the single-view center
    crop.  Returns ``(views [V, T, C, crop_res, crop_res], label)``.
    """

    multi_view = True

    def __init__(
        self,
        num_frames: int = 16,
        res: int = 256,
        crop_res: int = 224,
        data_root: str = "data",
        num_crops: int = 3,
        split: str = "train",
    ):
        if num_crops not in (1, 3):
            raise ValueError(f"num_crops must be 1 or 3, got {num_crops}")
        base = HMDB51Dataset(num_frames=num_frames, res=res, crop_res=crop_res, data_root=data_root, split=split)
        self.samples = [(base._video_paths[i], base._video_labels[i]) for i in base._subset_indices]
        self.num_frames = num_frames
        self.res = res
        self.crop_res = crop_res
        self.num_crops = num_crops

    def __len__(self) -> int:
        return len(self.samples)

    def _crops(self, h: int, w: int) -> list[tuple[int, int]]:
        c = self.crop_res
        if self.num_crops == 1:
            return [((h - c) // 2, (w - c) // 2)]
        if h <= w:
            return [((h - c) // 2, 0), ((h - c) // 2, (w - c) // 2), ((h - c) // 2, w - c)]
        return [(0, (w - c) // 2), ((h - c) // 2, (w - c) // 2), (h - c, (w - c) // 2)]

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path, label = self.samples[idx]
        video, _, _ = tio.read_video(path, pts_unit="sec")  # (T, H, W, C) uint8
        video = rearrange(video, "t h w c -> t c h w")
        indices = torch.linspace(0, video.shape[0] - 1, self.num_frames).long()
        video = _resize_shorter(video[indices].float() / 255.0, self.res, self.crop_res)
        c = self.crop_res
        views = [video[:, :, top : top + c, left : left + c] for top, left in self._crops(*video.shape[-2:])]
        return torch.stack(views, dim=0), int(label)
