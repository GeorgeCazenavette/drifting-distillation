"""MusicFM (MusicFM25Hz, FMA checkpoint) feature extractor.

Modes:
    ``mean_all``  time-averaged output of every conformer block,
                  ``{"mean_layer{i}": [N, 1024]}`` (distillation and herding).
    ``mean``      time-averaged output of the last block, ``{"mean": [N, 1024]}``
                  (evaluation).

Input: ``[N, 1, T]`` waveforms in ``(-1, 1)`` at 16 kHz, linearly resampled
to MusicFM's 24 kHz inside the forward so gradients reach the waveform.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_MUSICFM_TARGET_SR = 24_000


class MusicFMExtractor(nn.Module):
    """``hidden_states[i + 1]`` is the output of conformer block ``i``.

    Weights (``{variant}_stats.json``, ``pretrained_{variant}.pt``) come from
    the ``minzwon/MusicFM`` hub repo, or from a local directory ``model_name``.
    """

    def __init__(
        self,
        model_name: str = "minzwon/MusicFM",
        variant: str = "fma",
        mode: str = "mean_all",
        src_sample_rate: int = 16_000,
    ):
        super().__init__()
        if mode not in ("mean", "mean_all"):
            raise ValueError(f"MusicFMExtractor: unknown mode {mode!r}; choose mean or mean_all.")
        stat_file, model_file = f"{variant}_stats.json", f"pretrained_{variant}.pt"
        if os.path.isdir(model_name):
            stat_path = os.path.join(model_name, stat_file)
            chkpt_path = os.path.join(model_name, model_file)
        else:
            from huggingface_hub import hf_hub_download

            stat_path = hf_hub_download(repo_id=model_name, filename=stat_file)
            chkpt_path = hf_hub_download(repo_id=model_name, filename=model_file)

        from ._musicfm import MusicFM25Hz

        self.model = MusicFM25Hz(stat_path=stat_path, model_path=chkpt_path)
        self.num_layers: int = self.model.conformer.config.num_hidden_layers  # 12
        self.src_sr = src_sample_rate
        self.mode = mode
        for p in self.model.parameters():
            p.requires_grad = False

    def _get_hidden_states(self, wav: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """MusicFM25Hz.get_predictions without its ``no_grad`` preprocessing.

        Runs in fp32: under fp16 autocast, ``log10`` underflows 1e-10 to zero
        (AmplitudeToDB emits -inf on silent frames) and the encoder overflows
        to NaN on near-uniform inputs (the zero-initialized pyramid).
        """
        with torch.autocast(wav.device.type, enabled=False):
            mel = self.model.preprocessor_melspec_2048.float()(wav.float())[..., :-1]
            mel = (mel - self.model.stat["melspec_2048_mean"]) / self.model.stat["melspec_2048_std"]
            # Breaks ties when every mel bin is identical (silence maps to the
            # AmplitudeToDB floor), which would otherwise give NaN.
            mel = mel + 1e-4 * torch.randn_like(mel)
            _, hidden_states = self.model.encoder(mel)
        return hidden_states

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.src_sr != _MUSICFM_TARGET_SR:
            x = F.interpolate(x, scale_factor=_MUSICFM_TARGET_SR / self.src_sr, mode="linear", align_corners=False)
        hidden_states = self._get_hidden_states(x.squeeze(1))
        if self.mode == "mean":
            return {"mean": hidden_states[self.num_layers].mean(dim=1)}
        return {f"mean_layer{i}": hidden_states[i + 1].mean(dim=1) for i in range(self.num_layers)}
