"""Pull one ImageNet class from the Hub, as an alternative to uploading a zip.

``evanarlian/imagenet_1k_resized_256`` stores 1.28M images at 256px across 52
parquet shards, each with 247 row groups of ~100 rows — and it is **sorted by
label**. So a single class lives in ~13 consecutive row groups of one shard, and
parquet's row-group statistics let us read exactly those over HTTP range
requests: ~17 MiB for a full 1300-image class instead of a 480 MB shard.

Images arrive as the dataset's own JPEG bytes (256px on the short side) and are
written straight to disk — no re-encode, so no second generation of artifacts.
"""

from __future__ import annotations

import functools
import json
import tempfile
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "evanarlian/imagenet_1k_resized_256"
_DATA = f"datasets/{REPO}/data"
NUM_CLASSES = 1000
CACHE_DIR = Path(tempfile.gettempdir()) / "drifting_imagenet_256"


@functools.lru_cache(maxsize=1)
def _fs() -> HfFileSystem:
    return HfFileSystem()


@functools.lru_cache(maxsize=1)
def _shards() -> list[str]:
    return sorted(f for f in _fs().ls(_DATA, detail=False) if "/train-" in f)


@functools.lru_cache(maxsize=64)
def _shard_labels(shard: str) -> tuple[int, int]:
    """(min, max) label in a shard, from its parquet footer."""
    with _fs().open(shard) as f:
        md = pq.ParquetFile(f).metadata
        col = md.schema.names.index("label")
        return (
            md.row_group(0).column(col).statistics.min,
            md.row_group(md.num_row_groups - 1).column(col).statistics.max,
        )


@functools.lru_cache(maxsize=1)
def class_names() -> list[str]:
    """The 1000 ClassLabel names, read from the dataset's parquet metadata."""
    with _fs().open(_shards()[0]) as f:
        meta = pq.ParquetFile(f).schema_arrow.metadata[b"huggingface"]
    return json.loads(meta)["info"]["features"]["label"]["names"]


def _shards_for(label: int) -> list[str]:
    """Shards containing this label; a class can straddle a shard boundary."""
    shards = _shards()
    # Labels are sorted and shards are equal-sized, so proportion lands within
    # one shard of the answer; walk from there.
    i = min(int(label / NUM_CLASSES * len(shards)), len(shards) - 1)
    while _shard_labels(shards[i])[0] > label and i > 0:
        i -= 1
    while _shard_labels(shards[i])[1] < label and i < len(shards) - 1:
        i += 1
    hits = [shards[i]]
    if _shard_labels(shards[i])[1] == label and i + 1 < len(shards):
        hits.append(shards[i + 1])
    return hits


def fetch_class(
    label: int,
    num_images: int,
    progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Download up to ``num_images`` images of ``label`` and return their paths.

    Files land in a per-label cache directory and are reused on later calls, so
    trying several classes (or re-running one) costs nothing the second time.
    """
    if not 0 <= label < NUM_CLASSES:
        raise ValueError(f"label must be in [0, {NUM_CLASSES}), got {label}.")

    dest = CACHE_DIR / f"{label:04d}"
    dest.mkdir(parents=True, exist_ok=True)
    have = sorted(dest.glob("*.jpg"))
    if len(have) >= num_images:
        return have[:num_images]

    written = {p.name for p in have}
    for shard in _shards_for(label):
        if len(written) >= num_images:
            break
        with _fs().open(shard) as f:
            pf = pq.ParquetFile(f)
            md = pf.metadata
            col = md.schema.names.index("label")
            groups = [
                i
                for i in range(md.num_row_groups)
                if md.row_group(i).column(col).statistics.min <= label <= md.row_group(i).column(col).statistics.max
            ]
            # One row group at a time (~100 rows) so a small request stops early.
            for gi in groups:
                if len(written) >= num_images:
                    break
                table = pf.read_row_groups([gi], columns=["image", "label"])
                labels = table.column("label").to_pylist()
                for row, row_label in enumerate(labels):
                    if row_label != label or len(written) >= num_images:
                        continue
                    img = table.column("image")[row].as_py()
                    name = Path(img["path"]).name
                    name = name if name.lower().endswith(".jpg") else f"{name}.jpg"
                    (dest / name).write_bytes(img["bytes"])
                    written.add(name)
                if progress is not None:
                    progress(min(len(written), num_images), num_images)

    return sorted(dest.glob("*.jpg"))[:num_images]
