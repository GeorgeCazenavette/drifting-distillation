"""Vendored MoCo-v3 ViT class.

From https://github.com/LTH14/rcg (MIT License, Copyright (c) 2023 Tianhong Li). Only the
``VisionTransformerMoCoV3`` class is kept; the ``vit_*`` factory functions and
``ConvStem`` are dropped because they are unused here. Loaded via
``VisionTransformerMoCoV3.from_pretrained("nyu-visionx/moco-v3-vit-{b,l}")``.
"""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from timm.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer


class VisionTransformerMoCoV3(VisionTransformer, PyTorchModelHubMixin):
    """timm ViT subclass with fixed 2-D sin-cos position embedding.

    The pos_embed is built once at construction for the configured
    ``img_size`` / ``patch_size`` and registered as a non-trainable parameter,
    so inputs must match that size.
    """

    def __init__(self, stop_grad_conv1: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.build_2d_sincos_position_embedding()

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            val = math.sqrt(6.0 / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature: float = 10000.0) -> None:
        h = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]
        w = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]

        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False
