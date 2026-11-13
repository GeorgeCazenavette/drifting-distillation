"""Class-wise audio distillation via drifting (FMA-small, MusicFM).

Same loop as :mod:`.optimize_image`, on waveforms: the distilled clips are
1-D pyramids of ``pyramid_samples`` samples, and each step drifts
``augs_per_batch`` random ``clip_samples``-long crops (plus noise) of every
clip toward the genre's real training tracks in the extractor's feature space.

    python -m src.distillation.optimize_audio \\
        --data_root data/fma/fma_small_by_genre --out_dir out/fma_ipc5 --classes 0 1 2

Output: ``{out_dir}/{class_idx:04d}/wavs_{steps:05d}.pt`` holding
``{"wavs": [N, 1, pyramid_samples], "sample_rate": int}`` (what the audio
evaluation reads), plus ``{i:03d}.wav`` files.  An interrupted class resumes
from ``checkpoint.pt``; finished classes are skipped.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from contextlib import AbstractContextManager

import soundfile as sf
import torch
import torch.nn as nn
from torch import Tensor

from ..args import AudioArgs
from ..augmentation import build_aug
from ..datasets.fma import FMADataset
from ..extractors import get_extractor
from ..pyramid import AudioPyramid
from .drifting import drifting_loss, single_feature_loss
from .optimize_image import precompute_pos_latents


def distill_class(
    args: AudioArgs,
    class_idx: int,
    extractor: nn.Module,
    loss_fn: Callable[[Tensor, Tensor, int, int], tuple[Tensor, Tensor, Tensor, Tensor]],
    autocast: Callable[[], AbstractContextManager[object]],
) -> None:
    class_dir = os.path.join(args.out_dir, f"{class_idx:04d}")
    if os.path.exists(os.path.join(class_dir, "done")):
        print(f"[class {class_idx}] already done, skipping.")
        return
    os.makedirs(class_dir, exist_ok=True)
    ckpt_path = os.path.join(class_dir, "checkpoint.pt")
    device = args.device

    torch.manual_seed(class_idx if args.seed is None else args.seed)
    scaler = torch.amp.GradScaler("cuda")

    pos_ds = FMADataset(
        args.data_root,
        split="training",
        sample_rate=args.sample_rate,
        clip_samples=args.clip_samples,
        single_class=class_idx,
    )
    pos_latents = precompute_pos_latents(extractor, pos_ds, nn.Identity(), autocast, args)

    syn = AudioPyramid(
        args.num_clips, target_samples=args.pyramid_samples, start_samples=args.start_samples, lr=args.lr, device=device
    )
    aug_syn = build_aug("audio", args.clip_samples, noise_std=args.noise_std).to(device)

    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        syn.load_from_dict(ckpt["syn"])
        start_step = ckpt["step"]
        print(f"[class {class_idx}] resumed from step {start_step}")

    for step in range(start_step, args.steps):
        x = syn.get_data()  # [N, 1, pyramid_samples] in (-1, 1)
        x = aug_syn(x.repeat(args.augs_per_batch, 1, 1))  # [N*augs, 1, clip_samples]

        with autocast():
            loss, pos_mag, neg_mag, _ = drifting_loss(x, extractor, pos_latents, loss_fn)

        syn.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(syn.optimizer)
        scaler.update()

        if (step + 1) % args.extend_every == 0:
            syn.extend_pyramid()

        if step % args.log_every == 0 or step == args.steps - 1:
            print(
                f"[class {class_idx}] step {step:5d}/{args.steps} | loss {loss.item():.5f}"
                f" | pos_field {pos_mag.item():.4f} | neg_field {neg_mag.item():.4f}"
                f" | levels {syn.num_levels()}",
                flush=True,
            )
        if (step + 1) % 1000 == 0 and step + 1 < args.steps:
            torch.save({"step": step + 1, "syn": syn.get_save_dict()}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    with torch.no_grad():
        wavs = syn.get_data().cpu()
    torch.save(
        {"step": args.steps, "wavs": wavs, "sample_rate": args.sample_rate},
        os.path.join(class_dir, f"wavs_{args.steps:05d}.pt"),
    )
    for i, wav in enumerate(wavs):
        sf.write(os.path.join(class_dir, f"{i:03d}.wav"), wav.squeeze(0).numpy(), args.sample_rate, subtype="FLOAT")
    open(os.path.join(class_dir, "done"), "w").close()
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    args = AudioArgs(explicit_bool=True).parse_args()

    extractor = get_extractor(args.backbone, mode=args.mode, compile=args.compile).to(args.device)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)

    autocast = functools.partial(torch.autocast, "cuda", dtype=torch.float16)
    loss_fn = functools.partial(
        torch.compile(single_feature_loss) if args.compile else single_feature_loss, temps=tuple(args.temps)
    )

    for c in args.classes:
        distill_class(args, c, extractor, loss_fn, autocast)


if __name__ == "__main__":
    main()
