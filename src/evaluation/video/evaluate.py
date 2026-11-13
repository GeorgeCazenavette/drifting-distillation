"""Linear probe and 1-NN of a distilled (or coreset) HMDB51 clip set.

    python -m src.evaluation.video.evaluate --support_dir out/hmdb51_ipc5 --output_path out/hmdb51_ipc5.json

Features: VideoMAE-v2 ``mean`` (last-block mean token through ``fc_norm``).
Support: the ``clip_*.pt`` class folders.  Query: HMDB51 split 1, the held-out
test videos (``--split test``, 1530; the paper's table) or the training videos
that distillation and herding draw from (``--split train``, 3570); 16 linspace
frames each, center-cropped for the 1-NN
and averaged over 3 spatial crops for the linear probe.
Linear probe: 100 epochs, AdamW lr 1e-3, no weight decay, seed 42.

The query is embedded right after the extractor is built, inside a forked RNG
scope, so the head's initialization depends only on the seed and the support
set (as when the query features came from a cache).
"""

from __future__ import annotations

import json
import os
from typing import Literal

import torch
from tap import Tap
from torch.utils.data import DataLoader

from ...extractors import get_extractor
from .data import MultiCropHMDB51, VideoTensorFolder
from .probe import embed, fit_linear_head, knn_accuracy, linear_accuracy


class VideoEvalArgs(Tap):
    support_dir: str  # Class folders of clip_*.pt (class index = sorted folder order)
    data_root: str = "data"  # Holds hmdb51/
    split: Literal["train", "test"] = "test"  # Query videos: split-1 held-out test (the paper's table) or train
    output_path: str | None = None  # Write the results JSON here
    backbone: Literal["videomaev2"] = "videomaev2"
    mode: Literal["mean"] = "mean"
    num_frames: int = 16
    res: int = 256
    crop_res: int = 224
    num_crops: int = 3  # Query crops averaged for the linear probe (1-NN uses the center one)
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0
    head_batch_size: int = 256
    k: int = 1  # k of the cosine k-NN
    batch_size: int = 16
    num_workers: int = 8
    seed: int = 42
    device: str = "cuda"


def main() -> None:
    args = VideoEvalArgs(explicit_bool=True).parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    extractor = get_extractor(args.backbone, mode=args.mode).to(device).eval()

    with torch.random.fork_rng(devices=[device]):
        query_ds = MultiCropHMDB51(
            num_frames=args.num_frames,
            res=args.res,
            crop_res=args.crop_res,
            data_root=args.data_root,
            num_crops=args.num_crops,
            split=args.split,
        )
        query_feats, query_labels = embed(
            extractor,
            DataLoader(
                query_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False
            ),
            device,
        )  # [N, V, D]

    support_ds = VideoTensorFolder(args.support_dir, res=args.res, crop_res=args.crop_res)
    support_feats, support_labels = embed(
        extractor,
        DataLoader(
            support_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        ),
        device,
    )
    print(f"support {tuple(support_feats.shape)}, query {tuple(query_feats.shape)}")

    num_classes = int(max(support_labels.max().item(), query_labels.max().item())) + 1
    head = fit_linear_head(
        support_feats,
        support_labels,
        num_classes,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.head_batch_size,
        device=device,
    )
    results = {
        "linear_acc1": linear_accuracy(head, query_feats.mean(dim=1), query_labels, device),
        f"knn{args.k}_acc1": knn_accuracy(
            query_feats[:, query_feats.shape[1] // 2],
            query_labels,
            support_feats,
            support_labels,
            k=args.k,
            device=device,
        ),
    }
    print(json.dumps(results, indent=2))
    if args.output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump({**args.as_dict(), **results}, f, indent=2)


if __name__ == "__main__":
    main()
