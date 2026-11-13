"""Download one of our distilled datasets from Hugging Face and extract it.

    python -m src.tools.download_distilled --list
    python -m src.tools.download_distilled imagenet-1k_dinov3_noise_ipc10 --out distilled/dinov3_noise_ipc10

A tag is a config name of the Hugging Face dataset
(https://huggingface.co/datasets/georgecazenavette/drifting-distillation),
``{dataset}_{variant}_ipc{ipc}``; ``--list`` prints them all.  Files are
written as ``{out}/{class_idx:04d}/{name}``, the layout ``optimize_image``
writes, so an image set can be passed straight to the evaluations as ``--data``.
Videos are extracted as MP4 and audio as 16-bit WAV, the hosted formats.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "georgecazenavette/drifting-distillation"


def tag(entry: dict) -> str:
    return re.sub(r"[^\w.-]", "_", f"{entry['dataset']}_{entry['variant']}_ipc{entry['ipc']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", nargs="?", help="dataset tag, e.g. imagenet-1k_dinov3_noise_ipc10")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--list", action="store_true", help="print every tag and exit")
    args = ap.parse_args()

    manifest = json.load(open(hf_hub_download(REPO, "manifest.json", repo_type="dataset")))
    sets = {tag(e): e for e in manifest["sets"]}
    if args.list or not args.tag:
        for t, e in sets.items():
            print(f"{t:45s} {e['num_classes']:5d} classes x {e['ipc']:3d}  {e['bytes'] / 2**30:6.2f} GB")
        return
    if args.tag not in sets:
        ap.error(f"unknown tag {args.tag!r}; see --list")
    if not args.out:
        ap.error("--out is required")

    n = 0
    for f in sets[args.tag]["files"]:
        path = hf_hub_download(REPO, f["path"], repo_type="dataset")
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):  # one row group per class
            for r in pf.read_row_group(rg, columns=["class_idx", "name", "data"]).to_pylist():
                cdir = os.path.join(args.out, f"{r['class_idx']:04d}")
                os.makedirs(cdir, exist_ok=True)
                with open(os.path.join(cdir, r["name"]), "wb") as fh:
                    fh.write(r["data"])
                n += 1
        print(f"{f['path']}: done", flush=True)
    print(f"wrote {n} files to {args.out}")


if __name__ == "__main__":
    main()
