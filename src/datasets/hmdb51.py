"""HMDB51, split 1.

Expects ``{data_root}/hmdb51/videos/<class>/*.avi`` (51 classes; class index =
sorted folder order) and the official ``testTrainMulti_7030_splits/``.  Each
video yields one ``[num_frames, 3, crop_res, crop_res]`` clip in ``[0, 1]``:
``num_frames`` frames spaced uniformly (linspace) over the whole video,
shorter side resized to ``res``, center-cropped to ``crop_res``.

``split="train"`` indexes the split-1 training videos (tag 1, 70 per class,
3570 total), which distillation and herding draw from and which the paper's
evaluation also used as its query set; ``split="test"`` indexes the split-1
test videos (tag 2, 30 per class, 1530 total).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torchvision.io as tio
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from ..download import ensure_hmdb51


class HMDB51Dataset(Dataset):
    def __init__(
        self,
        num_frames: int = 16,
        res: int = 256,
        crop_res: int = 224,
        data_root: str = "data",
        single_class: int | None = None,
        split: str = "train",
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        want_tag = 1 if split == "train" else 2
        self.num_frames = num_frames
        self.res = res
        self.crop_res = crop_res

        ensure_hmdb51(data_root)  # official release, downloaded on first use
        videos_dir = os.path.join(data_root, "hmdb51", "videos")
        splits_dir = os.path.join(data_root, "hmdb51", "testTrainMulti_7030_splits")
        self.class_names = sorted(d for d in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, d)))

        # Split files: "{class}_test_split1.txt", lines "{filename} {0|1|2}";
        # 1 = train, 2 = test, 0 = unused.
        self._video_paths: list[str] = []
        self._video_labels: list[int] = []
        for class_idx, class_name in enumerate(self.class_names):
            split_file = os.path.join(splits_dir, f"{class_name}_test_split1.txt")
            if not os.path.exists(split_file):
                continue
            with open(split_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fname, tag = line.rsplit(" ", 1)
                    if int(tag) != want_tag:
                        continue
                    full_path = os.path.join(videos_dir, class_name, fname)
                    if os.path.exists(full_path):
                        self._video_paths.append(full_path)
                        self._video_labels.append(class_idx)

        self._subset_indices = [
            i for i, lbl in enumerate(self._video_labels) if single_class is None or lbl == single_class
        ]

    def _transform(self, video: Tensor) -> Tensor:
        res, crop_res = self.res, self.crop_res
        video = video.float() / 255.0
        h, w = video.shape[-2], video.shape[-1]
        if h <= w:
            new_h, new_w = res, max(crop_res, int(w * res / h))
        else:
            new_h, new_w = max(crop_res, int(h * res / w)), res
        if new_h != h or new_w != w:
            video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        h, w = video.shape[-2], video.shape[-1]
        top = (h - crop_res) // 2
        left = (w - crop_res) // 2
        return video[:, :, top : top + crop_res, left : left + crop_res]

    def __len__(self) -> int:
        return len(self._subset_indices)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        video_idx = self._subset_indices[index]
        video, _, _ = tio.read_video(self._video_paths[video_idx], pts_unit="sec")  # (T, H, W, C) uint8
        indices = torch.linspace(0, video.shape[0] - 1, self.num_frames).long()
        video = rearrange(video[indices], "t h w c -> t c h w")
        return self._transform(video), self._video_labels[video_idx]
