"""Class-wise image distillation via drifting.

For each requested class, optimizes ``num_images`` synthetic images
(parameterized as Laplacian pyramids) so that their augmented views, embedded
by a frozen feature extractor, match the distribution of the class's real
images under the drifting loss.  Classes are independent, so a dataset is
distilled by splitting its classes across processes/GPUs.

    python -m src.distillation.optimize_image \\
        --data_dir data/imagenet/train --out_dir out/dinov3_ipc10 \\
        --backbone dinov3 --mode cls_grid --classes 0 1 2

Output: ``{out_dir}/{class_idx:04d}/{i:04d}.png`` -- an ImageFolder that the
evaluation scripts read directly.  An interrupted class resumes from
``{out_dir}/{class_idx:04d}/checkpoint.pt``; finished classes are skipped.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable
from contextlib import AbstractContextManager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvu
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..args import AudioArgs, ImageArgs, VideoArgs
from ..augmentation import build_aug
from ..datasets import ClassFolder
from ..download import ensure_path
from ..extractors import get_extractor
from ..pyramid import ImagePyramid, PixelParameterization
from .drifting import drifting_loss, single_feature_loss
from .rded_init import load_teacher, rded_select


@torch.no_grad()
def precompute_pos_latents(
    extractor: nn.Module,
    dataset: Dataset,
    aug_real: nn.Module,
    autocast: Callable[[], AbstractContextManager[object]],
    args: ImageArgs | VideoArgs | AudioArgs,
) -> dict[str, torch.Tensor]:
    """Features of every real image of the class: ``{name: [P*locs, C]}`` on the device."""
    loader = DataLoader(
        dataset,
        batch_size=args.pos_batch,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )
    feats: dict[str, list[torch.Tensor]] = {}
    for imgs, _ in tqdm(loader, desc="Pre-computing real features"):
        imgs = aug_real(imgs.to(args.device))
        with autocast():
            out = extractor(imgs)
        for k, f in out.items():
            feats.setdefault(k, []).append(f)
    return {k: torch.cat(v, dim=0) for k, v in feats.items()}


def distill_class(
    args: ImageArgs,
    class_idx: int,
    extractor: nn.Module,
    loss_fn: Callable[[Tensor, Tensor, int, int], tuple[Tensor, Tensor, Tensor, Tensor]],
    autocast: Callable[[], AbstractContextManager[object]],
    teacher: nn.Module | None = None,
) -> None:
    class_dir = os.path.join(args.out_dir, f"{class_idx:04d}")
    if os.path.exists(os.path.join(class_dir, "done")):
        print(f"[class {class_idx}] already done, skipping.")
        return
    os.makedirs(class_dir, exist_ok=True)
    ckpt_path = os.path.join(class_dir, "checkpoint.pt")
    device = args.device

    torch.manual_seed(class_idx if args.seed is None else args.seed)
    scaler = torch.amp.GradScaler("cuda")  # fp16 autocast + loss scaling, as in the paper runs

    scale, ratio = (args.min_crop, args.max_crop), (args.min_ratio, args.max_ratio)
    aug_real = build_aug("none", args.crop_res).to(device)
    aug_syn = build_aug(args.aug_syn, args.crop_res, noise_std=args.noise_std, scale=scale, ratio=ratio).to(device)

    pos_ds = ClassFolder(args.data_dir, class_idx, res=args.res)
    pos_latents = precompute_pos_latents(extractor, pos_ds, aug_real, autocast, args)

    # --- synthetic images ---------------------------------------------------
    init_mode = "zero" if args.init == "rded" else "noise"
    if args.parameterization == "pixel":
        # Match the pyramid's per-pixel init std, 1/sqrt(#initial levels).
        n_levels = int(math.log2(args.start_res)) + 1
        syn = PixelParameterization(
            args.num_images,
            syn_res=args.syn_res,
            lr=args.lr,
            init_mode=init_mode,
            decorrelate_color=args.decorrelate_color,
            device=device,
            init_std=n_levels**-0.5,
        )
    else:
        syn = ImagePyramid(
            args.num_images,
            syn_res=args.syn_res,
            start_res=args.start_res,
            lr=args.lr,
            init_mode=init_mode,
            tileable=args.tileable,
            decorrelate_color=args.decorrelate_color,
            device=device,
        )

    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        syn.load_from_dict(ckpt["syn"])
        start_step = ckpt["step"]
        print(f"[class {class_idx}] resumed from step {start_step}")
    elif args.init == "rded":
        seed = class_idx if args.seed is None else args.seed
        init = rded_select(args.data_dir, class_idx, args.num_images, teacher, device, seed=seed)
        # RDED crops are 224 px; resize to the pyramid resolution.
        init = F.interpolate(
            init, size=(args.syn_res, args.syn_res), mode="bilinear", align_corners=False, antialias=True
        ).clamp(0, 1)
        mse = syn.init_from_images(init, fit_steps=args.init_fit_steps, fit_lr=args.init_fit_lr)
        print(f"[class {class_idx}] fitted pyramid to RDED crops (MSE {mse:.6f})")

    # --- optimization -------------------------------------------------------
    for step in range(start_step, args.steps):
        x = syn.get_data()  # [N, 3, H, W] in [0, 1]
        x = aug_syn(x.repeat(args.augs_per_batch, 1, 1, 1))  # [N*augs, 3, crop, crop]

        with autocast():
            loss, pos_mag, neg_mag, _ = drifting_loss(x, extractor, pos_latents, loss_fn)

        syn.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(syn.optimizer)
        scaler.update()

        if args.parameterization == "pyramid" and (step + 1) % args.extend_every == 0:
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
    for i, img in enumerate(final):
        tvu.save_image(img, os.path.join(class_dir, f"{i:04d}.png"))
    open(os.path.join(class_dir, "done"), "w").close()
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    args = ImageArgs(explicit_bool=True).parse_args()
    ensure_path(args.data_dir)  # ImageNette / ImageWoof download on first use

    extractor = get_extractor(args.backbone, mode=args.mode, standardize=args.standardize, compile=args.compile).to(
        args.device
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)

    autocast = functools.partial(torch.autocast, "cuda", dtype=torch.float16)

    loss_fn = functools.partial(single_feature_loss, temps=tuple(args.temps))
    if args.compile:
        loss_fn = functools.partial(torch.compile(single_feature_loss), temps=tuple(args.temps))

    teacher = load_teacher(args.rded_teacher, args.device) if args.init == "rded" else None

    for c in args.classes:
        distill_class(args, c, extractor, loss_fn, autocast, teacher)


if __name__ == "__main__":
    main()
