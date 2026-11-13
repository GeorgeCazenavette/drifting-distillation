"""Arguments for image distillation (``python -m src.distillation.optimize_image``).

Defaults are the paper's ImageNet-1k recipe; see ``scripts/`` for the exact
command behind every result.
"""

from typing import Literal

from tap import Tap


class ImageArgs(Tap):
    # Data / output
    data_dir: str  # ImageFolder-style train dir (one subdir per class, sorted = class index)
    out_dir: str  # Distilled images go to {out_dir}/{class_idx:04d}/{i:04d}.png
    classes: list[int]  # Class indices to distill (processed one after another)
    num_images: int = 10  # Images per class (IPC)
    seed: int | None = None  # Random seed; None = class index (the paper's convention)

    # Feature extractor
    backbone: Literal[
        "resnet18", "vit", "clip", "siglip2", "eva02", "mocov3", "dinov3", "dinov3-small", "dinov3-large", "eupe"
    ] = "dinov3"
    mode: Literal["cls", "cls_mean", "cls_grid", "cls_grid2"] = (
        "cls_grid"  # Drifted features: CLS / CLS+mean patch / CLS+all patches (ViT); cls_grid2 = pooled + last two stages' locations (ResNet-18)
    )
    standardize: bool | None = None  # LayerNorm every ResNet latent (None = on for resnet18, off otherwise)

    # Initialization
    init: Literal["noise", "rded"] = "noise"  # noise, or RDED-selected real crops (factor 1) fitted into the pyramid
    rded_teacher: str | None = (
        None  # RDED observer: None = torchvision ResNet-18 (ImageNet-1k); "imagenette"/"imagewoof" = the hosted 10-class teacher (downloaded); or a checkpoint path
    )
    init_fit_steps: int = 300  # Steps fitting the pyramid to the init images
    init_fit_lr: float = 0.05  # LR for that fit

    # Optimization
    steps: int = 10000  # Paper: 10000 for noise init, 5000 for real init
    lr: float = 0.001  # Adam learning rate
    temps: list[float] = [0.5]  # Kernel temperature(s) tau
    augs_per_batch: int = 10  # Augmented views of each distilled image per step
    aug_syn: Literal["standard", "resize"] = (
        "standard"  # standard = flip + random resized crop + noise; resize = one deterministic view
    )
    noise_std: float = 0.1  # Gaussian-noise std in the standard augmentation
    min_crop: float = 0.08  # RandomResizedCrop area range
    max_crop: float = 1.0
    min_ratio: float = 3 / 4  # RandomResizedCrop aspect-ratio range
    max_ratio: float = 4 / 3
    res: int = 256  # Real images: resize + center crop to res ...
    crop_res: int = 224  # ... then center crop to crop_res (the extractor input size)

    # Parameterization
    parameterization: Literal["pyramid", "pixel"] = "pyramid"  # pixel = raw-pixel ablation
    syn_res: int = 256  # Distilled image resolution
    start_res: int = 8  # Finest initial pyramid level
    extend_every: int = 200  # Steps between added pyramid levels
    decorrelate_color: bool = True  # Fixed color decorrelation in the decoder
    tileable: bool = False  # Circular boundaries (seamless canvases)

    # Hardware
    device: str = "cuda"
    compile: bool = True  # torch.compile the extractor and the loss
    num_workers: int = 8
    pos_batch: int = 100  # Batch size for precomputing real features
    log_every: int = 100

    def process_args(self) -> None:
        if self.standardize is None:
            self.standardize = self.backbone == "resnet18"
        if self.start_res < 1 or self.start_res & (self.start_res - 1):
            raise ValueError(f"--start_res must be a power of 2, got {self.start_res}")
        if self.init == "rded" and self.parameterization != "pyramid":
            raise ValueError("--init rded requires --parameterization pyramid")
