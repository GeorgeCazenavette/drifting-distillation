"""FMA-small (8 genres), official splits.

Expects a genre-organized copy of ``fma_small`` (``root/<Genre>/<id>.mp3``;
class index = position in ``GENRES``) with ``fma_metadata/`` as a sibling of
``root`` (or at ``metadata_root``).  Each track is decoded to mono float32,
resampled to ``sample_rate`` with ``scipy.signal.resample``, then randomly
cropped (if longer) or zero-padded (if shorter) to ``clip_samples``.  Returns
``(wav [1, clip_samples], label)``.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..download import ensure_path

# Truncated / corrupt tracks in fma_small.
_CORRUPT_FILES = {
    "099134.mp3",  # Electronic
    "133297.mp3",  # Experimental
    "098565.mp3",  # Hip-Hop
    "098567.mp3",  # Hip-Hop
    "098569.mp3",  # Hip-Hop
    "108925.mp3",  # Rock
}

GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]


def _load_split_ids(metadata_root: str, split: str) -> set[str]:
    """``{"000123.mp3", ...}`` of the fma_small tracks in ``split``.

    ``tracks.csv`` has three header rows; column 31 is the split, 32 the subset.
    """
    ids: set[str] = set()
    with open(os.path.join(metadata_root, "tracks.csv"), newline="") as f:
        for _ in range(3):
            next(f)
        for row in csv.reader(f):
            if len(row) < 33 or row[32].strip() != "small" or row[31].strip() != split:
                continue
            ids.add(f"{int(row[0].strip()):06d}.mp3")
    return ids


class FMADataset(Dataset):
    """Args:
    root: Genre-organized ``fma_small`` directory.
    split: ``"training"``, ``"validation"`` or ``"test"``.
    single_class: Keep only this genre index.
    """

    def __init__(
        self,
        root: str,
        split: str,
        sample_rate: int = 16_000,
        clip_samples: int = 480_000,
        single_class: int | None = None,
        metadata_root: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        ensure_path(root)  # fma_small + metadata download on first use
        if metadata_root is None:
            metadata_root = os.path.join(os.path.dirname(os.path.abspath(root)), "fma_metadata")
        split_ids = _load_split_ids(metadata_root, split)

        self.samples: list[tuple[str, int]] = []
        for label, genre in enumerate(GENRES):
            genre_dir = os.path.join(root, genre)
            if not os.path.isdir(genre_dir) or (single_class is not None and label != single_class):
                continue
            for fname in sorted(os.listdir(genre_dir)):
                if fname.endswith(".mp3") and fname not in _CORRUPT_FILES and fname in split_ids:
                    self.samples.append((os.path.join(genre_dir, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, path: str) -> Tensor:
        wav, sr = sf.read(path, always_2d=True)  # [T, C]
        wav = wav.mean(axis=1).astype(np.float32)  # mono
        if sr != self.sample_rate:
            wav = scipy.signal.resample(wav, int(len(wav) * self.sample_rate / sr)).astype(np.float32)
        return torch.from_numpy(wav)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path, label = self.samples[idx]
        wav = self._load_audio(path)
        T = wav.shape[0]
        if T >= self.clip_samples:
            start = torch.randint(0, T - self.clip_samples + 1, ()).item()
            wav = wav[start : start + self.clip_samples]
        else:
            wav = torch.cat([wav, torch.zeros(self.clip_samples - T)])
        return wav.unsqueeze(0), label
