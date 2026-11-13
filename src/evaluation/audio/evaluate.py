"""Linear probe and 1-NN of a distilled (or coreset) FMA-small clip set.

    python -m src.evaluation.audio.evaluate --support_dir out/fma_ipc5 --output_path out/fma_ipc5.json

Features: MusicFM-FMA ``mean`` (time-averaged last conformer block).
Support: class folders of ``wavs_*.pt`` (30 s center crop of each clip).
Query: the official FMA-small validation split (800 tracks).
Linear probe: 1000 epochs, AdamW lr 1e-3, no weight decay, seed 42.

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

from ...datasets.fma import FMADataset
from ...extractors import get_extractor
from ..video.probe import embed, fit_linear_head, knn_accuracy, linear_accuracy
from .data import WavTensorFolder


class AudioEvalArgs(Tap):
    support_dir: str  # Class folders of wavs_*.pt (class index = sorted folder order)
    data_root: str = "data/fma/fma_small_by_genre"  # Genre-organized fma_small; fma_metadata/ must be its sibling
    output_path: str | None = None  # Write the results JSON here
    backbone: Literal["musicfm-fma"] = "musicfm-fma"
    mode: Literal["mean"] = "mean"
    sample_rate: int = 16_000
    clip_samples: int = 480_000
    epochs: int = 1000
    lr: float = 0.001
    weight_decay: float = 0.0
    head_batch_size: int = 256
    k: int = 1  # k of the cosine k-NN
    batch_size: int = 64
    num_workers: int = 8
    seed: int = 42
    device: str = "cuda"


def main() -> None:
    args = AudioEvalArgs(explicit_bool=True).parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    extractor = get_extractor(args.backbone, mode=args.mode).to(device).eval()

    with torch.random.fork_rng(devices=[device]):
        query_ds = FMADataset(
            args.data_root, split="validation", sample_rate=args.sample_rate, clip_samples=args.clip_samples
        )
        query_feats, query_labels = embed(
            extractor,
            DataLoader(
                query_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False
            ),
            device,
        )

    support_ds = WavTensorFolder(args.support_dir, sample_rate=args.sample_rate, clip_samples=args.clip_samples)
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
        "linear_acc1": linear_accuracy(head, query_feats, query_labels, device),
        f"knn{args.k}_acc1": knn_accuracy(
            query_feats, query_labels, support_feats, support_labels, k=args.k, device=device
        ),
    }
    print(json.dumps(results, indent=2))
    if args.output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump({**args.as_dict(), **results}, f, indent=2)


if __name__ == "__main__":
    main()
