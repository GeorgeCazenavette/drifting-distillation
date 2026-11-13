"""The distillation loop as a generator that streams previews to the app.

Uses the research code directly -- :class:`ImagePyramid`, ``build_aug``, the
drifting loss and the extractor registry -- for the one case the app needs: a
single uploaded image set treated as one distribution.  Compared to
``src/distillation/optimize_image.py`` it drops RDED init, checkpoint/resume
and ``torch.compile`` (compiling costs more warm-up than a short interactive
run saves), and adds bf16, optional positive subsampling and live previews.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from ..augmentation import build_aug
from ..distillation.drifting import drifting_loss, single_feature_loss
from ..extractors import BACKBONE_DEFAULTS, get_extractor
from ..pyramid import ImagePyramid
from .dataset import UploadedImages

IMAGE_BACKBONES = [b for b in BACKBONE_DEFAULTS if b not in ("videomaev2", "musicfm-fma")]
AUG_SCHEMES = ("standard", "resize")


def modes_for(backbone: str) -> tuple[str, ...]:
    """Feature modes an image backbone supports."""
    return ("cls_grid2", "cls") if backbone == "resnet18" else ("cls_grid", "cls_mean", "cls")


def check_crop_res(backbone: str, crop_res: int) -> None:
    """DINOv3 (RoPE) and ResNet-18 take any multiple of 16; the other ViTs have
    position embeddings fixed at 224 px."""
    if backbone.startswith("dinov3") or backbone == "resnet18":
        if crop_res % 16:
            raise ValueError(f"{backbone} needs crop_res to be a multiple of 16; got {crop_res}.")
    elif crop_res != 224:
        raise ValueError(f"{backbone} only supports crop_res=224; got {crop_res}.")


@dataclass
class DistillConfig:
    """Everything the app lets the user set for one run."""

    # extractor
    backbone: str = "dinov3"
    mode: str = "cls_grid"

    # synthetic set
    num_images: int = 10
    syn_res: int = 256
    start_res: int = 8
    extend_every: int = 200
    decorrelate_color: bool = True
    init_from_real: bool = False

    # optimization
    steps: int = 10000
    lr: float = 1e-3
    temp: float = 0.5
    augs_per_batch: int = 10
    pos_batch: int = 0  # positives per step; 0 = all uploaded images

    # data / augmentation
    res: int = 256
    crop_res: int = 224
    aug_syn: str = "standard"
    noise_std: float = 0.1
    min_crop: float = 0.08
    max_crop: float = 1.0
    min_ratio: float = 3 / 4
    max_ratio: float = 4 / 3

    # runtime
    precision: str = "amp"  # "amp" (fp16 autocast + loss scaling) or "bf16"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    preview_every: int = 25

    def validate(self) -> None:
        if self.backbone not in IMAGE_BACKBONES:
            raise ValueError(f"Unknown backbone {self.backbone!r}.")
        if self.mode not in modes_for(self.backbone):
            raise ValueError(f"{self.backbone} supports modes {modes_for(self.backbone)}; " f"got {self.mode!r}.")
        if self.aug_syn not in AUG_SCHEMES:
            raise ValueError(f"Unknown augmentation {self.aug_syn!r}.")
        if self.precision not in ("amp", "bf16"):
            raise ValueError(f"precision must be 'amp' or 'bf16', got {self.precision!r}.")
        check_crop_res(self.backbone, self.crop_res)
        if self.syn_res < self.crop_res:
            raise ValueError(f"syn_res ({self.syn_res}) must be >= crop_res ({self.crop_res}).")
        if self.start_res < 1 or self.start_res & (self.start_res - 1):
            raise ValueError(f"start_res must be a power of 2, got {self.start_res}.")
        if self.start_res > self.syn_res:
            raise ValueError(f"start_res ({self.start_res}) must be <= syn_res ({self.syn_res}).")
        if not 0.0 < self.min_crop <= self.max_crop <= 1.0:
            raise ValueError("crop scale must satisfy 0 < min_crop <= max_crop <= 1.")
        if not 0.0 < self.min_ratio <= self.max_ratio:
            raise ValueError("crop aspect ratio must satisfy 0 < min_ratio <= max_ratio.")
        if self.num_images < 1 or self.steps < 1 or self.augs_per_batch < 1:
            raise ValueError("num_images, steps and augs_per_batch must all be >= 1.")


@dataclass
class Update:
    """One streamed progress event."""

    phase: str  # "extract" | "init" | "step" | "done"
    message: str
    step: int = 0
    total_steps: int = 0
    loss: float | None = None
    field: float | None = None  # per-dim RMS of the un-normalized drifting field
    images: list[Image.Image] | None = None  # set on preview steps and at the end
    levels: int = 0


def _to_pil(imgs: Tensor) -> list[Image.Image]:
    return [TF.to_pil_image(img.clamp(0, 1).float().cpu()) for img in imgs]


def run_distillation(
    image_paths: list[Path], cfg: DistillConfig, out_dir: Path, stop: threading.Event | None = None
) -> Iterator[Update]:
    """Distill ``cfg.num_images`` images from the uploaded set.

    Yields an :class:`Update` while extracting features, every
    ``cfg.preview_every`` steps, and once at the end with the finished images
    (also written as PNGs into ``out_dir``).  Setting ``stop`` ends the loop
    early and still yields the final update.
    """
    cfg.validate()
    if not image_paths:
        raise ValueError("No images found.")
    if cfg.init_from_real and len(image_paths) < cfg.num_images:
        raise ValueError(f"init_from_real needs at least {cfg.num_images} images; " f"got {len(image_paths)}.")

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    device = cfg.device
    cuda = device.startswith("cuda")

    yield Update("extract", f"Loading {cfg.backbone} ({cfg.mode})…")
    extractor = get_extractor(cfg.backbone, mode=cfg.mode, standardize=cfg.backbone == "resnet18").to(device)
    if cfg.precision == "bf16":
        extractor.to(torch.bfloat16)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
    autocast = functools.partial(torch.autocast, "cuda", dtype=dtype) if cuda else contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and cfg.precision != "bf16")

    aug_real = build_aug("none", cfg.crop_res).to(device)
    aug_syn = build_aug(
        cfg.aug_syn,
        cfg.crop_res,
        noise_std=cfg.noise_std,
        scale=(cfg.min_crop, cfg.max_crop),
        ratio=(cfg.min_ratio, cfg.max_ratio),
    ).to(device)

    # --- real features, computed once ----------------------------------------
    ds = UploadedImages(image_paths, res=cfg.res)
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=cfg.num_workers,
        # fork, not the 3.14 default forkserver: workers are spawned from a
        # server request thread, where forkserver's handshake can fail.
        multiprocessing_context="fork" if cfg.num_workers > 0 else None,
    )
    feats: dict[str, list[Tensor]] = {}
    seen = 0
    for imgs in loader:
        imgs = aug_real(imgs.to(device))
        with torch.no_grad(), autocast():
            out = extractor(imgs)
        for k, f in out.items():
            feats.setdefault(k, []).append(f.cpu())
        seen += imgs.shape[0]
        yield Update("extract", f"Extracting features: {seen}/{len(ds)} images…")
    pos_latents = {k: torch.cat(v) for k, v in feats.items()}  # [P*locs, C]
    n_pos = len(ds)
    use_all_pos = cfg.pos_batch <= 0 or cfg.pos_batch >= n_pos
    if use_all_pos:
        pos_latents = {k: v.to(device) for k, v in pos_latents.items()}

    # --- synthetic images ------------------------------------------------------
    yield Update("init", "Initializing pyramid…")
    pyramid = ImagePyramid(
        cfg.num_images,
        syn_res=cfg.syn_res,
        start_res=cfg.start_res,
        lr=cfg.lr,
        init_mode="zero" if cfg.init_from_real else "noise",
        decorrelate_color=cfg.decorrelate_color,
        device=device,
    )
    if cfg.init_from_real:
        full = UploadedImages(image_paths, res=cfg.syn_res)
        targets = torch.stack([full[i] for i in range(cfg.num_images)]).to(device)
        mse = pyramid.init_from_images(targets)
        yield Update("init", f"Fitted pyramid to {cfg.num_images} uploaded images (MSE {mse:.5f}).")

    loss_fn = functools.partial(single_feature_loss, temps=(cfg.temp,))

    # --- optimization ---------------------------------------------------------
    for step in range(cfg.steps):
        if stop is not None and stop.is_set():
            break

        x = aug_syn(pyramid.get_data().repeat(cfg.augs_per_batch, 1, 1, 1))
        if use_all_pos:
            sampled = pos_latents
        else:
            # Rows are image-major ([P*locs, C]); pick whole images.  Features
            # can have different numbers of locations, so index each separately.
            idx = torch.randperm(n_pos)[: cfg.pos_batch]
            sampled = {
                k: rearrange(rearrange(v, "(p l) c -> p l c", p=n_pos)[idx], "p l c -> (p l) c").to(device)
                for k, v in pos_latents.items()
            }

        with autocast():
            loss, _, _, field = drifting_loss(x, extractor, sampled, loss_fn)

        pyramid.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(pyramid.optimizer)
        scaler.update()

        if cfg.extend_every and (step + 1) % cfg.extend_every == 0:
            pyramid.extend_pyramid()

        if step % cfg.preview_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                preview = _to_pil(pyramid.get_data())
            yield Update(
                "step",
                f"Step {step + 1}/{cfg.steps}",
                step=step + 1,
                total_steps=cfg.steps,
                loss=loss.item(),
                field=field.item(),
                images=preview,
                levels=pyramid.num_levels(),
            )

    # --- deliverable ----------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        final = _to_pil(pyramid.get_data())
    for i, img in enumerate(final):
        img.save(out_dir / f"{i:04d}.png")
    stopped = stop is not None and stop.is_set()
    yield Update(
        "done",
        f"{'Stopped' if stopped else 'Done'} — {len(final)} images at {cfg.syn_res}px.",
        step=cfg.steps,
        total_steps=cfg.steps,
        images=final,
        levels=pyramid.num_levels(),
    )
