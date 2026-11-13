"""Herding coreset of real FMA-small training tracks (audio baseline).

    python -m src.baselines.herding_audio --data_root data/fma/fma_small_by_genre --out_dir out/herding_fma_ipc5

Selects ``num_clips`` tracks per genre with herding on the extractor's features
(default: the per-block mean features of MusicFM, the space the distillation
drifts in) and saves them as ``{out_dir}/{class:04d}/wavs_00000.pt`` holding
``{"wavs": [N, 1, clip_samples], "sample_rate": int}`` -- the layout the audio
evaluation reads -- plus ``{j:03d}.wav`` files.

Tracks longer than ``clip_samples`` are randomly cropped by the dataset, both
in the feature pass and again when a selected track is re-loaded for saving.
No seed is set: constructing MusicFM resets the global RNG
(``torch.manual_seed(142)``), which makes those crops reproducible.
"""

from __future__ import annotations

import os
from typing import Literal

import soundfile as sf
import torch
from tap import Tap

from ..datasets.fma import FMADataset
from ..extractors import get_extractor
from .herding import extract_features, select_per_class


class HerdingAudioArgs(Tap):
    data_root: str = "data/fma/fma_small_by_genre"  # Genre-organized fma_small; fma_metadata/ must be its sibling
    out_dir: str  # Coreset goes to {out_dir}/{class:04d}/wavs_00000.pt
    num_clips: int = 5  # Tracks per genre
    backbone: Literal["musicfm-fma"] = "musicfm-fma"
    mode: Literal["mean_all"] = "mean_all"  # Herding feature space (concatenated per-block mean features)
    sample_rate: int = 16_000
    clip_samples: int = 480_000
    batch_size: int = 64
    num_workers: int = 8
    device: str = "cuda"


def main() -> None:
    args = HerdingAudioArgs(explicit_bool=True).parse_args()
    extractor = get_extractor(args.backbone, mode=args.mode).to(args.device).eval()
    ds = FMADataset(args.data_root, split="training", sample_rate=args.sample_rate, clip_samples=args.clip_samples)
    feats, labels = extract_features(ds, extractor, args.device, args.batch_size, args.num_workers)
    for cls, indices in select_per_class(feats, labels, args.num_clips).items():
        class_dir = os.path.join(args.out_dir, f"{cls:04d}")
        os.makedirs(class_dir, exist_ok=True)
        wavs = torch.stack([ds[i][0] for i in indices])  # [N, 1, clip_samples]
        torch.save({"step": 0, "wavs": wavs, "sample_rate": args.sample_rate}, os.path.join(class_dir, "wavs_00000.pt"))
        for j, wav in enumerate(wavs):
            sf.write(os.path.join(class_dir, f"{j:03d}.wav"), wav.squeeze(0).numpy(), args.sample_rate, subtype="FLOAT")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
