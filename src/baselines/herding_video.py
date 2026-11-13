"""Herding coreset of real HMDB51 training clips (video baseline).

    python -m src.baselines.herding_video --data_root data --out_dir out/herding_hmdb51_ipc5

Selects ``num_clips`` clips per class with herding on the extractor's features
(default: the per-block mean tokens of VideoMAE-v2, the space the distillation
drifts in) and saves them as ``{out_dir}/{class:04d}/clip_{j:02d}.pt``,
``(T, 3, H, W)`` in ``[0, 1]`` -- the layout the video evaluation reads.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
from tap import Tap

from ..datasets.hmdb51 import HMDB51Dataset
from ..extractors import get_extractor
from .herding import extract_features, select_per_class


class HerdingVideoArgs(Tap):
    data_root: str = "data"  # Holds hmdb51/videos and hmdb51/testTrainMulti_7030_splits
    out_dir: str  # Coreset goes to {out_dir}/{class:04d}/clip_{j:02d}.pt
    num_clips: int = 5  # Clips per class
    backbone: Literal["videomaev2"] = "videomaev2"
    mode: Literal["mean_all"] = "mean_all"  # Herding feature space (concatenated per-block mean tokens)
    num_frames: int = 16
    res: int = 256
    crop_res: int = 224
    batch_size: int = 16
    num_workers: int = 8
    device: str = "cuda"


def main() -> None:
    args = HerdingVideoArgs(explicit_bool=True).parse_args()
    extractor = get_extractor(args.backbone, mode=args.mode).to(args.device).eval()
    ds = HMDB51Dataset(num_frames=args.num_frames, res=args.res, crop_res=args.crop_res, data_root=args.data_root)
    feats, labels = extract_features(ds, extractor, args.device, args.batch_size, args.num_workers)
    for cls, indices in select_per_class(feats, labels, args.num_clips).items():
        class_dir = os.path.join(args.out_dir, f"{cls:04d}")
        os.makedirs(class_dir, exist_ok=True)
        for j, i in enumerate(indices):
            torch.save(ds[i][0].contiguous(), os.path.join(class_dir, f"clip_{j:02d}.pt"))
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
