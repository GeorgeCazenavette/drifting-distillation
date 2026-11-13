# MIT License
#
# Copyright 2023 ByteDance Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""Vendored MusicFM (github.com/minzwon/musicfm), trimmed to the encoder forward.

Concatenates upstream ``modules/features.py``, ``modules/conv.py``,
``modules/random_quantizer.py`` and ``model/musicfm_25hz.py``.  Kept verbatim
except: the flash-attention conformer branch and the pre-training methods
(masking, tokenization, loss, ``no_grad`` preprocessing) are dropped.  The
random-projection quantizer, output projection and CLS token are unused by the
encoder but stay: the checkpoint loads ``strict=True``, and constructing them
consumes the global RNG exactly as upstream does (note the quantizer calls
``torch.manual_seed(142)``, resetting the global RNG at construction).
"""

from __future__ import annotations

import json
import random

import torch
import torchaudio
from einops import rearrange
from torch import Tensor, einsum, nn

# ── modules/features.py ──────────────────────────────────────────────


class MelSTFT(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 2048,
        hop_length: int = 240,
        n_mels: int = 128,
        is_db: bool = False,
    ):
        super().__init__()

        # spectrogram
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        # amplitude to decibel
        self.is_db = is_db
        if is_db:
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform: Tensor) -> Tensor:
        if self.is_db:
            return self.amplitude_to_db(self.mel_stft(waveform))
        else:
            return self.mel_stft(waveform)


# ── modules/conv.py ──────────────────────────────────────────────────


class Res2dModule(nn.Module):
    def __init__(self, idim: int, odim: int, stride: tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(idim, odim, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(odim)
        self.conv2 = nn.Conv2d(odim, odim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(odim)
        self.relu = nn.ReLU()

        # residual
        self.diff = False
        if (idim != odim) or (stride[0] > 1):
            self.conv3 = nn.Conv2d(idim, odim, 3, padding=1, stride=stride)
            self.bn3 = nn.BatchNorm2d(odim)
            self.diff = True

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.diff:
            x = self.bn3(self.conv3(x))
        out = x + out
        out = self.relu(out)
        return out


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        hdim (int): Hidden dimension.
        odim (int): Output dimension.
        strides (list): Sizes of strides.
        n_bands (int): Number of frequency bands.
    """

    def __init__(self, idim: int, hdim: int, odim: int, strides: list[int] = [2, 2], n_bands: int = 64):
        """Construct an Conv2dSubsampling object."""
        super().__init__()

        self.conv = nn.Sequential(
            Res2dModule(idim, hdim, (2, strides[0])),
            Res2dModule(hdim, hdim, (2, strides[1])),
        )
        self.linear = nn.Linear(hdim * n_bands // 2 // 2, odim)

    def forward(self, x: Tensor) -> Tensor:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, idim, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
        """

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, f, t)
        x = self.conv(x)
        x = rearrange(x, "b c f t -> b t (c f)")
        x = self.linear(x)
        return x


# ── modules/random_quantizer.py ──────────────────────────────────────


class RandomProjectionQuantizer(nn.Module):
    """
    Random projection and codebook lookup module

    Some code is borrowed from:
     https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/random_projection_quantizer.py
    But I did normalization using pre-computed global mean & variance instead of using layer norm.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_dim: int,
        codebook_size: int,
        seed: int = 142,
    ):
        super().__init__()

        # random seed
        torch.manual_seed(seed)

        # randomly initialized projection
        random_projection = torch.empty(input_dim, codebook_dim)
        nn.init.xavier_normal_(random_projection)
        self.register_buffer("random_projection", random_projection)

        # randomly initialized codebook
        codebook = torch.empty(codebook_size, codebook_dim)
        nn.init.normal_(codebook)
        self.register_buffer("codebook", codebook)

    def codebook_lookup(self, x: Tensor) -> Tensor:
        # reshape
        b = x.shape[0]
        x = rearrange(x, "b n e -> (b n) e")

        # L2 normalization
        normalized_x = nn.functional.normalize(x, dim=1, p=2)
        normalized_codebook = nn.functional.normalize(self.codebook, dim=1, p=2)

        # compute distances
        distances = torch.cdist(normalized_codebook, normalized_x)

        # get nearest
        nearest_indices = torch.argmin(distances, dim=0)

        # reshape
        xq = rearrange(nearest_indices, "(b n) -> b n", b=b)

        return xq

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        # always eval
        self.eval()

        # random projection [batch, length, input_dim] -> [batch, length, codebook_dim]
        x = einsum("b n d, d e -> b n e", x, self.random_projection)

        # codebook lookup
        xq = self.codebook_lookup(x)

        return xq


# ── model/musicfm_25hz.py ────────────────────────────────────────────


class MusicFM25Hz(nn.Module):
    """
    MusicFM

    Input: 128-band mel spectrogram
    Frontend: 2-layer Residual convolution
    Backend: 12-layer Conformer
    Quantizer: a codebook for mel spectrogram
    """

    def __init__(
        self,
        num_codebooks: int = 1,
        codebook_dim: int = 16,
        codebook_size: int = 4096,
        features: list[str] = ["melspec_2048"],
        hop_length: int = 240,
        n_mels: int = 128,
        conv_dim: int = 512,
        encoder_dim: int = 1024,
        encoder_depth: int = 12,
        mask_hop: float = 0.4,
        mask_prob: float = 0.6,
        stat_path: str = "./data/fma_stats.json",
        model_path: str | None = "./data/pretrained_fma.pt",
    ):
        super().__init__()

        # global variables
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.features = features

        # load feature mean / std stats
        with open(stat_path) as f:
            self.stat = json.load(f)

        # feature extractor
        self.preprocessor_melspec_2048 = MelSTFT(n_fft=2048, hop_length=hop_length, is_db=True)

        # random quantizer
        seed = 142
        for feature in self.features:
            for i in range(num_codebooks):
                setattr(
                    self,
                    f"quantizer_{feature}_{i}",
                    RandomProjectionQuantizer(n_mels * 4, codebook_dim, codebook_size, seed=seed + i),
                )

        # two residual convolution layers + one projection layer
        self.conv = Conv2dSubsampling(1, conv_dim, encoder_dim, strides=[2, 2], n_bands=n_mels)

        # Conformer
        from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
            Wav2Vec2ConformerConfig,
            Wav2Vec2ConformerEncoder,
        )

        config = Wav2Vec2ConformerConfig.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
        config.num_hidden_layers = encoder_depth
        config.hidden_size = encoder_dim

        self.conformer = Wav2Vec2ConformerEncoder(config)

        # projection
        self.linear = nn.Linear(encoder_dim, codebook_size)

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # cls token (used for sequence classification)
        random.seed(seed)
        self.cls_token = nn.Parameter(torch.randn(encoder_dim))

        # load model
        if model_path:
            S = torch.load(model_path)["state_dict"]
            SS = {k[6:]: v for k, v in S.items()}
            self.load_state_dict(SS, strict=True)

    def encoder(self, x: Tensor) -> tuple[dict[str, Tensor], tuple[Tensor, ...]]:
        """2-layer conv + w2v-conformer"""
        x = self.conv(x)
        out = self.conformer(x, output_hidden_states=True)
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        logits = {
            key: logits[:, :, i * self.codebook_size : (i + 1) * self.codebook_size]
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb
