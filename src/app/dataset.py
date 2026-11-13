"""User image sets: safe zip extraction, URL download, and a flat image dataset."""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def extract_zip(zip_path: str | Path, dest: str | Path) -> list[Path]:
    """Extract the image files from a zip into ``dest`` and return their paths.

    Subdirectories are flattened: every image anywhere in the archive counts as
    one sample of the same distribution.  Entries with absolute or ``..`` paths
    are skipped (a zip is user input), as are macOS resource forks and dotfiles.

    Args:
        zip_path: Path to the uploaded ``.zip``.
        dest: Directory to extract into (created if missing).

    Returns:
        Sorted list of extracted image paths.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                continue
            if name.suffix.lower() not in IMAGE_EXTS:
                continue
            if name.name.startswith(".") or "__MACOSX" in name.parts:
                continue
            # Flatten, keeping the parent dir in the name so images from
            # different subfolders can't collide.
            flat = "__".join(name.parts[-2:]) if len(name.parts) > 1 else name.name
            target = dest / flat
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out.append(target)

    return sorted(out)


_PIL_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif", "GIF": ".gif"}


def _fetch_image(url: str, target_stem: Path, max_bytes: int, timeout: float) -> Path:
    """Download one URL, check that it decodes as an image, and save it."""
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs are allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "drifting-distillation-demo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"larger than {max_bytes // 2**20} MB")
    with Image.open(io.BytesIO(data)) as img:
        img.verify()  # raises on non-images
        fmt = img.format
    target = target_stem.with_suffix(_PIL_EXT.get(fmt, ".img"))
    target.write_bytes(data)
    return target


def download_urls(
    urls: list[str],
    dest: str | Path,
    max_bytes: int = 20 * 2**20,
    timeout: float = 15.0,
    workers: int = 16,
    progress: Callable[[int, int], object] | None = None,
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Download image URLs into ``dest`` in parallel.

    Every URL that fails (bad scheme, network error, too large, not an image)
    is skipped and reported rather than aborting the batch.

    Args:
        urls: Image URLs; blank lines and duplicates are ignored.
        dest: Directory to save into (created if missing).
        max_bytes: Per-file size cap.
        timeout: Per-request timeout in seconds.
        workers: Concurrent downloads.
        progress: Optional ``progress(done, total)`` callback.

    Returns:
        ``(paths, failures)``: saved image paths in URL order, and
        ``(url, reason)`` for every URL that was skipped.
    """
    urls = list(dict.fromkeys(u.strip() for u in urls if u.strip()))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    results: dict[int, Path] = {}
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_image, u, dest / f"url_{i:05d}", max_bytes, timeout): i for i, u in enumerate(urls)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # user input: report, don't abort
                failures.append((urls[i], f"{type(exc).__name__}: {exc}"))
            if progress is not None:
                progress(done, len(urls))
    return [results[i] for i in sorted(results)], failures


class UploadedImages(Dataset):
    """Resize + center-crop the uploaded images to ``res``, as raw ``[0, 1]``.

    Matches the research pipeline's real-image preprocessing (``Resize(res)`` →
    ``CenterCrop(res)`` → ``ToTensor``); the per-step augmentation does the
    final crop to ``crop_res``.
    """

    def __init__(self, paths: list[Path], res: int = 256):
        self.paths = list(paths)
        self.transform = T.Compose(
            [
                T.Resize(res),
                T.CenterCrop(res),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.paths[index]) as img:
            return self.transform(img.convert("RGB"))
