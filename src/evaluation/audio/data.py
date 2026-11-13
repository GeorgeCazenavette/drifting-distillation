"""Distilled/coreset waveform folders for the audio evaluation."""

from __future__ import annotations

import os
import re

import torch
from torch import Tensor
from torch.utils.data import Dataset

_STEP_RE = re.compile(r"wavs_(\d+)\.pt$")


class WavTensorFolder(Dataset):
    """Class folders each holding ``wavs_{step:05d}.pt`` blobs ``{"wavs": [N, 1, T], "sample_rate"}``.

    Class index = sorted folder order; the highest-step blob of each class is
    used.  Clips are center-cropped (or zero-padded) to ``clip_samples``.
    Returns ``(wav [1, clip_samples], label)``.
    """

    def __init__(self, root: str, sample_rate: int = 16_000, clip_samples: int = 480_000):
        self.clip_samples = clip_samples
        class_names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.samples: list[tuple[Tensor, int]] = []
        for label, cname in enumerate(class_names):
            cdir = os.path.join(root, cname)
            blobs = [(int(m.group(1)), f) for f in os.listdir(cdir) if (m := _STEP_RE.match(f))]
            if not blobs:
                continue
            blob = torch.load(os.path.join(cdir, max(blobs)[1]), map_location="cpu")
            if int(blob.get("sample_rate", sample_rate)) != sample_rate:
                raise ValueError(f"{cdir}: sample rate {blob['sample_rate']} != {sample_rate}")
            for wav in blob["wavs"]:
                self.samples.append((wav, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        wav, label = self.samples[idx]  # [1, T]
        T = wav.shape[-1]
        if T >= self.clip_samples:
            start = (T - self.clip_samples) // 2
            wav = wav[..., start : start + self.clip_samples]
        else:
            wav = torch.cat([wav, torch.zeros(1, self.clip_samples - T)], dim=-1)
        return wav.contiguous(), label
