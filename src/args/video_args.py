"""Arguments for video distillation (``python -m src.distillation.optimize_video``).

Defaults are the paper's HMDB51 / VideoMAE-v2 recipe.
"""

from typing import Literal

from tap import Tap


class VideoArgs(Tap):
    # Data / output
    data_root: str = "data"  # Holds hmdb51/videos and hmdb51/testTrainMulti_7030_splits
    out_dir: str  # Distilled clips go to {out_dir}/{class_idx:04d}/clip_{i:03d}.pt
    classes: list[int]  # HMDB51 class indices to distill (0-50, sorted folder order)
    num_clips: int = 5  # Clips per class
    seed: int | None = None  # Random seed; None = class index

    # Feature extractor
    backbone: Literal["videomaev2"] = "videomaev2"
    mode: Literal["mean_all"] = "mean_all"  # Drift the mean token of every transformer block

    # Optimization
    steps: int = 5000
    lr: float = 0.001  # Adam learning rate
    temps: list[float] = [0.5]  # Kernel temperature(s) tau
    augs_per_batch: int = 4  # Augmented views of each distilled clip per step
    noise_std: float = 0.1  # Gaussian-noise std of the flip + random-resized-crop + noise augmentation

    # Clips
    num_frames: int = 16  # Frames per clip (linspace over the whole real video)
    res: int = 256  # Real videos: shorter side resized to res ...
    crop_res: int = 224  # ... then center-cropped to crop_res (the extractor input size)
    syn_res: int = (
        256  # Distilled clip resolution; each step's random resized crop takes crop_res views (as for images)
    )
    start_res: int = 8  # Finest initial spatial pyramid level
    start_frames: int = 1  # Finest initial temporal pyramid level
    extend_every: int = 200  # Steps between growth steps
    growth: Literal["alternate", "joint"] = (
        "joint"  # joint = one level doubling both axes; alternate = a spatial level then a temporal level per growth step
    )
    decorrelate_color: bool = True  # Fixed color decorrelation in the decoder

    # Hardware
    device: str = "cuda"
    compile: bool = True  # torch.compile the extractor and the loss
    num_workers: int = 4
    pos_batch: int = 4  # Batch size for precomputing real features
    log_every: int = 10
