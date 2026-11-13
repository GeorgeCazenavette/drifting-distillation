"""Class-wise video distillation via drifting (HMDB51, VideoMAE-v2).

Same loop as :mod:`.optimize_image`, on clips: the distilled clips are
spatiotemporal pyramids, each step drifts ``augs_per_batch`` augmented views
(flip + random resized crop + noise, identical across a clip's frames) toward
the class's real training videos in the extractor's feature space.

    python -m src.distillation.optimize_video \\
        --data_root data --out_dir out/hmdb51_ipc5 --classes 0 1 2

Output: ``{out_dir}/{class_idx:04d}/clip_{i:03d}.pt``, one ``(T, 3, H, W)``
tensor in ``[0, 1]`` per clip -- the layout the video evaluation reads.  An
interrupted class resumes from ``checkpoint.pt``; finished classes are skipped.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from contextlib import AbstractContextManager

import torch
import torch.nn as nn
from torch import Tensor

from ..args import VideoArgs
from ..augmentation import build_aug
from ..datasets.hmdb51 import HMDB51Dataset
from ..extractors import get_extractor
from ..pyramid import VideoPyramid
from .drifting import drifting_loss, single_feature_loss
from .optimize_image import precompute_pos_latents


def distill_class(
    args: VideoArgs,
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

    aug_real = build_aug("none", args.crop_res).to(device)
    aug_syn = build_aug("standard", args.crop_res, noise_std=args.noise_std).to(device)

    pos_ds = HMDB51Dataset(
        num_frames=args.num_frames,
        res=args.res,
        crop_res=args.crop_res,
        data_root=args.data_root,
        single_class=class_idx,
    )
    pos_latents = precompute_pos_latents(extractor, pos_ds, aug_real, autocast, args)

    syn = VideoPyramid(
        args.num_clips,
        num_frames=args.num_frames,
        syn_res=args.syn_res,
        start_res=args.start_res,
        start_frames=args.start_frames,
        lr=args.lr,
        decorrelate_color=args.decorrelate_color,
        growth=args.growth,
        device=device,
    )
    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        syn.load_from_dict(ckpt["syn"])
        start_step = ckpt["step"]
        print(f"[class {class_idx}] resumed from step {start_step}")

    for step in range(start_step, args.steps):
        x = syn.get_data()  # [N, T, 3, H, W] in [0, 1]
        x = aug_syn(x.repeat(args.augs_per_batch, 1, 1, 1, 1))  # [N*augs, T, 3, H, W]

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
        final = syn.get_data().cpu()
    for i, clip in enumerate(final):
        torch.save(clip.contiguous(), os.path.join(class_dir, f"clip_{i:03d}.pt"))
    open(os.path.join(class_dir, "done"), "w").close()
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    args = VideoArgs(explicit_bool=True).parse_args()

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
