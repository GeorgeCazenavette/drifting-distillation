"""Arguments for audio distillation (``python -m src.distillation.optimize_audio``).

Defaults are the paper's FMA-small / MusicFM recipe.
"""

from typing import Literal

from tap import Tap


class AudioArgs(Tap):
    # Data / output
    data_root: str = "data/fma/fma_small_by_genre"  # Genre-organized fma_small; fma_metadata/ must be its sibling
    out_dir: str  # Distilled clips go to {out_dir}/{class_idx:04d}/wavs_{steps:05d}.pt (+ .wav files)
    classes: list[int]  # Genre indices to distill (0-7, see src.datasets.fma.GENRES)
    num_clips: int = 5  # Clips per class
    seed: int | None = None  # Random seed; None = class index

    # Feature extractor
    backbone: Literal["musicfm-fma"] = "musicfm-fma"
    mode: Literal["mean_all"] = "mean_all"  # Drift the time-averaged output of every conformer block

    # Optimization
    steps: int = 5000
    lr: float = 0.001  # Adam learning rate
    temps: list[float] = [0.5]  # Kernel temperature(s) tau
    augs_per_batch: int = 5  # Augmented views (random crops) of each distilled clip per step
    noise_std: float = 0.01  # Gaussian-noise std of the augmentation

    # Waveforms
    sample_rate: int = 16_000
    clip_samples: int = 480_000  # Extractor input length (30 s); real clips are cropped/padded to it
    pyramid_samples: int = 524_288  # Distilled waveform length (2^19); random 30 s crops are drifted
    start_samples: int = 1  # Finest initial pyramid level
    extend_every: int = 100  # Steps between added pyramid levels

    # Hardware
    device: str = "cuda"
    compile: bool = True  # torch.compile the extractor and the loss
    num_workers: int = 8
    pos_batch: int = 100  # Batch size for precomputing real features
    log_every: int = 10
