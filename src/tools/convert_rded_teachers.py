"""Convert RDED's released ImageNette / ImageWoof ResNet-18s into our teacher files.

RDED (Sun et al., CVPR 2024) publishes its 10-class "observer" models in a
Google Drive folder linked from https://github.com/LINs-lab/RDED.  Download
``imagenet-nette_resnet18.pth`` and ``imagenet-woof_resnet18.pth`` from it, then

    python -m src.tools.convert_rded_teachers --src <download dir> --dst checkpoints/teachers

writes ``imagenette_resnet18_wnid.pt`` and ``imagewoof_resnet18_wnid.pt``
(``{"model": state_dict}``), the files the scripts expect.

Our datasets index classes in sorted-wnid (folder) order.  RDED's ImageNette
head already uses that order; its ImageWoof head does not, so its ``fc`` rows
are reordered: row ``k`` of the output is RDED's row ``WOOF_RDED_ROW[k]``.
Every other weight is copied unchanged.
"""

from __future__ import annotations

import argparse
import os

import torch

WOOF_RDED_ROW = [4, 6, 3, 5, 1, 0, 8, 9, 2, 7]
FILES = {  # dataset -> (RDED file name, fc row order or None)
    "imagenette": ("imagenet-nette_resnet18.pth", None),
    "imagewoof": ("imagenet-woof_resnet18.pth", WOOF_RDED_ROW),
}


def convert(path: str, rows: list[int] | None) -> dict:
    # The checkpoints also hold RDED's training args (an argparse.Namespace);
    # allow just that class so the load stays weights_only.
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    if rows is not None:
        sd["fc.weight"] = sd["fc.weight"][rows].clone()
        sd["fc.bias"] = sd["fc.bias"][rows].clone()
    return {"model": sd}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="directory holding RDED's .pth files")
    ap.add_argument("--dst", default="checkpoints/teachers")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    for dataset, (name, rows) in FILES.items():
        out = os.path.join(args.dst, f"{dataset}_resnet18_wnid.pt")
        torch.save(convert(os.path.join(args.src, name), rows), out)
        print(f"{name} -> {out}")


if __name__ == "__main__":
    main()
