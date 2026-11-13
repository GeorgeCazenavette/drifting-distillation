"""Automatic downloads: the 10-class teachers and the public datasets.

Each ``ensure_*`` is a no-op when the data is already in place, so the entry
points call them unconditionally.  ImageNet-1k cannot be downloaded
automatically (it needs registration at image-net.org).

    {root}/imagenette2/{train,val}/<wnid>/        official ImageNette archive
    {root}/imagewoof2/{train,val}/<wnid>/         official ImageWoof archive
    {root}/fma/fma_small_by_genre/<Genre>/*.mp3   fma_small sorted by tracks.csv genre_top
    {root}/fma/fma_metadata/                      FMA metadata (splits)
    {root}/hmdb51/videos/<class>/*.avi            official HMDB51 release (Serre lab)
    {root}/hmdb51/testTrainMulti_7030_splits/     official splits

``python -m src.tools.download_data`` fetches them ahead of time.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

from tqdm import tqdm

HF_REPO = "georgecazenavette/drifting-distillation"
TEACHERS = ("imagenette", "imagewoof")

ARCHIVES = {
    "imagenette": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", "imagenette2"),
    "imagewoof": ("https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz", "imagewoof2"),
}
FMA_SMALL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
HMDB51_REPO = "Serrelab/hmdb51"  # the authors' upload of the official .rar files


def teacher(dataset: str) -> str:
    """Local path of the ImageNette/ImageWoof ResNet-18 teacher, downloaded from
    Hugging Face on first use (RDED's observers, converted to sorted-wnid order)."""
    if dataset not in TEACHERS:
        raise ValueError(f"no hosted teacher for {dataset!r}; choose from {TEACHERS}")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, f"teachers/{dataset}_resnet18_wnid.pt", repo_type="dataset")


def _fetch(url: str, dst: str) -> str:
    """Download ``url`` to ``dst`` (via a .part file, so an interrupted download restarts)."""
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    part = dst + ".part"
    with urllib.request.urlopen(url) as r, open(part, "wb") as f, tqdm(
        total=int(r.headers.get("Content-Length", 0)) or None,
        unit="B",
        unit_scale=True,
        desc=f"downloading {os.path.basename(dst)}",
    ) as bar:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    os.replace(part, dst)
    return dst


def ensure_archive(dataset: str, root: str) -> str:
    """ImageNette / ImageWoof under ``root``; returns ``{root}/imagenette2`` etc."""
    url, name = ARCHIVES[dataset]
    out = os.path.join(root, name)
    if os.path.isdir(os.path.join(out, "val")):
        return out
    tgz = _fetch(url, os.path.join(root, name + ".tgz"))
    print(f"extracting {tgz}")
    with tarfile.open(tgz) as t:
        t.extractall(root, filter="data")
    os.remove(tgz)
    return out


def ensure_fma(root: str) -> str:
    """fma_small sorted into genre folders, plus fma_metadata; returns the genre root."""
    fma = os.path.join(root, "fma")
    by_genre = os.path.join(fma, "fma_small_by_genre")
    meta = os.path.join(fma, "fma_metadata")
    if os.path.isdir(by_genre) and os.path.isfile(os.path.join(meta, "tracks.csv")):
        return by_genre
    if not os.path.isfile(os.path.join(meta, "tracks.csv")):
        z = _fetch(FMA_METADATA, os.path.join(fma, "fma_metadata.zip"))
        print(f"extracting {z}")
        with zipfile.ZipFile(z) as f:
            f.extractall(fma)
        os.remove(z)
    genre = {}
    with open(os.path.join(meta, "tracks.csv"), newline="") as f:
        rows = csv.reader(f)
        for _ in range(3):  # three header rows
            next(rows)
        for row in rows:  # column 32: subset, 40: genre_top
            if len(row) > 40 and row[32].strip() == "small":
                genre[f"{int(row[0]):06d}.mp3"] = row[40].strip()
    z = _fetch(FMA_SMALL, os.path.join(fma, "fma_small.zip"))
    print(f"sorting {z} into genre folders")
    tmp = by_genre + ".part"
    with zipfile.ZipFile(z) as f:
        for info in tqdm(f.infolist(), desc="extracting"):
            name = os.path.basename(info.filename)
            if name in genre:
                os.makedirs(os.path.join(tmp, genre[name]), exist_ok=True)
                with f.open(info) as src, open(os.path.join(tmp, genre[name], name), "wb") as dst:
                    shutil.copyfileobj(src, dst)
    os.replace(tmp, by_genre)
    os.remove(z)
    return by_genre


def _unrar(archive: str, dst: str) -> None:
    """Extract a .rar with whichever of unrar / 7z / bsdtar is installed."""
    os.makedirs(dst, exist_ok=True)
    for cmd in (
        ["unrar", "x", "-inul", "-o+", archive, dst + os.sep],
        ["7z", "x", "-y", f"-o{dst}", archive],
        ["bsdtar", "-xf", archive, "-C", dst],
    ):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            return
    raise RuntimeError(
        "HMDB51 is distributed as .rar archives; install one of unrar, 7z or "
        "bsdtar (e.g. `apt install unrar`, `brew install sevenzip`, or "
        "`conda install -c conda-forge libarchive`) and rerun."
    )


def ensure_hmdb51(root: str) -> str:
    """Official HMDB51 videos and split files under ``{root}/hmdb51``; returns that dir."""
    out = os.path.join(root, "hmdb51")
    videos, splits = os.path.join(out, "videos"), os.path.join(out, "testTrainMulti_7030_splits")
    if os.path.isdir(videos) and os.path.isdir(splits):
        return out
    from huggingface_hub import hf_hub_download

    dl = os.path.join(out, "download")
    get = lambda name: hf_hub_download(HMDB51_REPO, name, repo_type="dataset", local_dir=dl)
    if not os.path.isdir(splits):
        _unrar(get("test_train_splits.rar"), out)
    if not os.path.isdir(videos):
        print("extracting HMDB51 (51 class archives)")
        classes = os.path.join(dl, "classes")
        _unrar(get("hmdb51_org.rar"), classes)
        tmp = videos + ".part"
        for r in tqdm(sorted(os.listdir(classes)), desc="extracting"):
            _unrar(os.path.join(classes, r), tmp)
        os.replace(tmp, videos)
    shutil.rmtree(dl)
    return out


def ensure_path(path: str) -> None:
    """Download whatever dataset ``path`` points into, if it is a known layout that is
    missing: ``.../imagenette2/...``, ``.../imagewoof2/...``, ``.../fma/fma_small_by_genre``."""
    if os.path.exists(path):
        return
    parts = os.path.normpath(path).split(os.sep)
    for dataset, (_, name) in ARCHIVES.items():
        if name in parts:
            ensure_archive(dataset, os.sep.join(parts[: parts.index(name)]) or ".")
            return
    if parts[-2:] == ["fma", "fma_small_by_genre"]:
        ensure_fma(os.sep.join(parts[:-2]) or ".")
