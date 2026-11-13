"""Gradio app: distill a small synthetic image set out of your own photos.

Upload a zip of images of one thing (a dog breed, a painter, your product
shots), pick a backbone, and watch a handful of synthetic images form as the
drifting loss pushes them toward the feature distribution of the uploads.
"""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import torch
from huggingface_hub import auth_check
from huggingface_hub.errors import GatedRepoError
from PIL import Image

from ..extractors import BACKBONE_DEFAULTS
from . import imagenet_hub
from .dataset import download_urls, extract_zip
from .distill import AUG_SCHEMES, IMAGE_BACKBONES, DistillConfig, Update, modes_for, run_distillation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PREVIEW_UPLOADS = 12
MAX_URLS = 2000
EXAMPLES_DIR = Path(__file__).parent / "examples"
GOLDFISH = 1  # ImageNet class index of the goldfish example
# 42 Qi Baishi paintings (Art Institute of Chicago + WikiArt), downloaded on
# demand rather than bundled.
BAISHI_URLS = EXAMPLES_DIR / "qi_baishi_urls.txt"
CROP_CHOICES = [160, 192, 224, 256]  # free for DINOv3 / ResNet-18; the other ViTs need 224
SYN_CHOICES = [224, 256, 320, 384, 512, 1024, 2048]
# Gradio fixes a plot block at `height` with overflow hidden, but sizes the chart
# to fill it, so the title and caption push the x-axis out of view.  Give the
# chart its own height and let the block grow around it.
CSS = "#field-plot { height: auto !important; }" " #field-plot .vega-embed { height: 240px !important; }"

# Settings presets.  Each one sets every option in the form (all but the seed
# and preview cadence), so switching between them never leaves a stale value.
DEFAULTS = DistillConfig()
PRESETS = {
    "Default": DEFAULTS,
    # The paper's art canvases (scripts/distill_art.sh): SigLIP2 cls_grid, one
    # 2048px image, 100 square crops per step covering 0.38-4.8% of its area.
    "Art canvas": replace(
        DEFAULTS,
        backbone="siglip2",
        mode="cls_grid",
        num_images=1,
        augs_per_batch=100,
        syn_res=2048,
        min_crop=0.0038,
        max_crop=0.0479,
        min_ratio=1.0,
        max_ratio=1.0,
    ),
}
PRESET_INFO = {
    "Default": "DINOv3 ViT-B, 10 images at 256px, 10k steps.",
    "Art canvas": "The paper's painting recipe: one 2048px SigLIP2 canvas, 10k steps.",
}
# Form fields, in the order the run handler receives them.
FORM_FIELDS = (
    "backbone",
    "mode",
    "num_images",
    "steps",
    "augs_per_batch",
    "lr",
    "syn_res",
    "crop_res",
    "res",
    "aug_syn",
    "noise_std",
    "min_crop",
    "max_crop",
    "min_ratio",
    "max_ratio",
    "temp",
    "start_res",
    "extend_every",
    "decorrelate_color",
    "init_from_real",
    "pos_batch",
    "precision",
    "seed",
    "preview_every",
)
PRESET_FIELDS = tuple(f for f in FORM_FIELDS if f not in ("seed", "preview_every"))
_CAST = {f.name: {"int": int, "float": float, "bool": bool, "str": str}[f.type] for f in fields(DistillConfig)}


def contact_sheet(images: list, cell: int = 192) -> Image.Image | None:
    """Tile the synthetic images into one PIL image.

    A gr.Gallery writes (and the browser fetches) one file per image per
    update; over a few thousand steps that is thousands of temp files and
    hundreds of MB pushed down the event stream. One sheet is one file.
    """
    if not images:
        return None
    cols = min(5, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, img in enumerate(images):
        sheet.paste(img.resize((cell, cell), Image.LANCZOS), ((i % cols) * cell, (i // cols) * cell))
    return sheet


def imagenet_choices() -> list[tuple[str, int]]:
    """Dropdown choices for the 1000 classes; degrades if the Hub is unreachable."""
    try:
        names = imagenet_hub.class_names()
    except Exception as exc:  # offline / Hub down
        print(f"[imagenet] class names unavailable ({exc}); using numeric labels")
        names = [f"class {i}" for i in range(imagenet_hub.NUM_CLASSES)]
    return [(f"{i} — {name}", i) for i, name in enumerate(names)]


def describe_device() -> str:
    if DEVICE == "cpu":
        return "**No GPU detected** — this will be far too slow to watch; run it on a GPU."
    return f"Running on **{torch.cuda.get_device_name(0)}**."


# ── upload ────────────────────────────────────────────────────────────


def on_upload(zip_file: str | None, workdir: str | None) -> tuple[str, list[str], str | None, list[str]]:
    """Extract the uploaded zip and show what was found."""
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    if zip_file is None:
        return describe_device(), [], None, []

    workdir = tempfile.mkdtemp(prefix="drifting_")
    try:
        paths = extract_zip(zip_file, Path(workdir) / "images")
    except Exception as exc:  # a zip is user input
        shutil.rmtree(workdir, ignore_errors=True)
        raise gr.Error(f"Could not read that zip: {exc}") from exc

    if not paths:
        shutil.rmtree(workdir, ignore_errors=True)
        raise gr.Error("No images found in the zip (png/jpg/webp/bmp/tif).")

    status = f"**{len(paths)} images** loaded. {describe_device()}"
    return status, [str(p) for p in paths[:MAX_PREVIEW_UPLOADS]], workdir, [str(p) for p in paths]


def load_imagenet_class(
    label: int, num_images: int, workdir: str | None, progress: gr.Progress = gr.Progress()
) -> tuple[str, list[str], str, list[str]]:
    """Pull one ImageNet class from the Hub into the positive set."""
    label = int(label)
    try:
        name = imagenet_hub.class_names()[label]
    except Exception:
        name = f"class {label}"

    progress(0.0, desc=f"Fetching {name}…")
    try:
        paths = imagenet_hub.fetch_class(
            label,
            int(num_images),
            progress=lambda done, total: progress(done / total, desc=f"{name}: {done}/{total}"),
        )
    except Exception as exc:
        raise gr.Error(f"Could not fetch class {label} from the Hub: {exc}") from exc

    workdir = workdir or tempfile.mkdtemp(prefix="drifting_")
    status = f"**{len(paths)} images** of *{name}* (ImageNet class {label}). " f"{describe_device()}"
    return (status, [str(p) for p in paths[:MAX_PREVIEW_UPLOADS]], workdir, [str(p) for p in paths])


def load_urls(
    text: str, workdir: str | None, progress: gr.Progress = gr.Progress()
) -> tuple[str, list[str], str, list[str]]:
    """Download a pasted list of image URLs (one per line) into the positive set."""
    urls = [u for u in (text or "").splitlines() if u.strip()]
    if not urls:
        raise gr.Error("Paste at least one image URL (one per line).")
    if len(urls) > MAX_URLS:
        raise gr.Error(f"At most {MAX_URLS} URLs per set; got {len(urls)}.")
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    workdir = tempfile.mkdtemp(prefix="drifting_")

    progress(0.0, desc=f"Downloading {len(urls)} URLs…")
    paths, failures = download_urls(
        urls,
        Path(workdir) / "images",
        progress=lambda done, total: progress(done / total, desc=f"Downloaded {done}/{total}"),
    )
    for url, reason in failures:
        print(f"[urls] skipped {url}: {reason}", flush=True)
    if not paths:
        shutil.rmtree(workdir, ignore_errors=True)
        raise gr.Error(f"None of the {len(failures)} URLs gave an image " f"(first error: {failures[0][1]}).")

    status = f"**{len(paths)} images** downloaded."
    if failures:
        status += f" Skipped {len(failures)} URL(s) that failed or weren't images (see the server log)."
    status += f" {describe_device()}"
    return (status, [str(p) for p in paths[:MAX_PREVIEW_UPLOADS]], workdir, [str(p) for p in paths])


# ── run ───────────────────────────────────────────────────────────────
#
# The distillation runs in a background thread, not inside the request. A
# browser stream is not a reliable 20-minute pipe -- ours was being cut at
# ~600 s by something between the browser and the server, and Gradio cancels
# the event when that happens, killing the run with it. With the work detached,
# a dropped stream costs only the view: press Reconnect (or reload) to attach
# to the job again. One GPU, so one job at a time.


@dataclass
class Job:
    """A detached distillation run and its latest state."""

    cfg: DistillConfig
    out_dir: Path
    stop: threading.Event = field(default_factory=threading.Event)
    updates: queue.Queue[Update | None] = field(default_factory=queue.Queue)
    status: str = "Starting…"
    sheet: Image.Image | None = None
    rows: list = field(default_factory=list)
    archive: str | None = None
    error: BaseException | None = None
    done: bool = False
    step: int = 0


_JOB: Job | None = None


def _worker(job: Job, paths: list[str]) -> None:
    """Run the distillation to completion, whatever the browser is doing."""
    t0 = time.time()
    try:
        for update in run_distillation([Path(p) for p in paths], job.cfg, job.out_dir, stop=job.stop):
            if update.images is not None:
                # Overwrite the deliverable on every preview, so the images so
                # far survive even a hard kill.
                for i, img in enumerate(update.images):
                    img.save(job.out_dir / f"{i:04d}.png")
                # Full resolution, capped so the sheet stays <= 2048 px wide
                # (one 2048 px art canvas is shown at full size).
                cols = min(5, len(update.images))
                job.sheet = contact_sheet(update.images, cell=min(job.cfg.syn_res, 2048 // cols))
            if update.loss is not None:
                job.rows.append({"step": update.step, "field": update.field})
                job.step = update.step
                print(
                    f"[job] step {update.step}/{update.total_steps} "
                    f"t={time.time() - t0:7.1f}s field={update.field:.4f}",
                    flush=True,
                )
            job.status = update.message
            if update.phase == "step":
                job.status = f"{update.message} · field {update.field:.4f} · " f"{update.levels} pyramid levels"
            job.updates.put(update)

        job.archive = shutil.make_archive(str(job.out_dir.with_name("distilled_images")), "zip", root_dir=job.out_dir)
        verb = "Stopped" if job.stop.is_set() else "Done"
        job.status = f"**{verb}.** Images at {job.cfg.syn_res}px — download below."
        print(f"[job] {verb.lower()} at step {job.step} " f"after {time.time() - t0:.1f}s", flush=True)
    except BaseException as exc:  # noqa: BLE001 - reported to the UI
        job.error = exc
        job.status = f"**Failed:** {type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        job.done = True
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        job.updates.put(None)  # wake any attached stream


def _stream(job: Job) -> Iterator[tuple[str, Image.Image | None, pd.DataFrame, dict[str, Any]]]:
    """Yield the job's state as it advances; safe to start, drop and restart."""
    while True:
        yield (
            job.status,
            job.sheet,
            pd.DataFrame(job.rows),
            gr.update(value=job.archive, visible=True) if job.archive else gr.update(),
        )
        if job.done and job.updates.empty():
            break
        try:
            job.updates.get(timeout=1.0)
        except queue.Empty:
            continue  # re-yield: keeps the view current

    if job.error is not None:
        raise gr.Error(f"{type(job.error).__name__}: {job.error}")


def gated_out(backbone: str) -> bool:
    """True if the Hugging Face token can't access ``backbone``'s gated weights.

    Only a definite "gated" answer counts: offline use with cached weights, or
    a Hub hiccup, leaves the backbone as chosen.
    """
    try:
        auth_check(BACKBONE_DEFAULTS[backbone])
    except GatedRepoError:
        return True
    except Exception:
        return False
    return False


def run(
    paths: list[str], workdir: str | None, *values: int | float | str | bool
) -> Iterator[tuple[str, Image.Image | None, pd.DataFrame, dict[str, Any]]]:
    """Start a detached run and stream it: status, preview, field curve, zip.

    ``values`` are the form fields, in ``FORM_FIELDS`` order.
    """
    global _JOB

    if _JOB is not None and not _JOB.done:
        raise gr.Error("A run is already going — press Reconnect to watch it, " "or Stop to end it.")
    if not paths:
        raise gr.Error("Load some images first (zip, example, ImageNet class or URLs).")

    cfg = DistillConfig(device=DEVICE, **{k: _CAST[k](v) for k, v in zip(FORM_FIELDS, values, strict=True)})
    try:
        cfg.validate()
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    if cfg.backbone.startswith("dinov3") and gated_out(cfg.backbone):
        # DINOv3 weights are gated on Hugging Face; DINOv2 is open.
        gr.Warning(
            f"No access to {BACKBONE_DEFAULTS[cfg.backbone]} (DINOv3 is gated: "
            "accept its license on Hugging Face and log in with `hf auth login`). "
            "Falling back to DINOv2 ViT-B at 224px crops.",
            duration=None,
        )
        print(f"[run] {cfg.backbone} is gated for this token; using dinov2", flush=True)
        cfg = replace(cfg, backbone="dinov2", crop_res=224)

    out_dir = Path(workdir or tempfile.mkdtemp(prefix="drifting_")) / "distilled"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    _JOB = Job(cfg=cfg, out_dir=out_dir)
    threading.Thread(target=_worker, args=(_JOB, paths), daemon=True).start()
    yield from _stream(_JOB)


def reconnect() -> Iterator[tuple[str, Image.Image | None, pd.DataFrame, dict[str, Any]]]:
    """Re-attach to the running (or finished) job after a dropped stream."""
    if _JOB is None:
        raise gr.Error("No run to reconnect to.")
    yield from _stream(_JOB)


def request_stop() -> str:
    if _JOB is None or _JOB.done:
        return "Nothing running."
    _JOB.stop.set()
    return "Stopping after this step…"


# ── UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Dataset Distillation via Drifting", delete_cache=(1800, 1800)) as demo:
    workdir_state = gr.State(None)
    paths_state = gr.State([])

    gr.Markdown(
        "# Dataset Distillation via Drifting\n"
        "Upload a zip of images of **one** subject and distill them into a handful of "
        "synthetic images. Each step decodes a multi-scale image pyramid, augments it, "
        "and drifts its features toward the feature distribution of your uploads."
    )

    with gr.Row():
        with gr.Column(scale=2):
            # ── 1. images ──
            zip_in = gr.File(label="Image set (.zip)", file_types=[".zip"], type="filepath")
            with gr.Row():
                example_btn = gr.Button("Example: 1300 ImageNet goldfish (downloads)", size="sm")
                baishi_btn = gr.Button("Example: 42 Qi Baishi paintings (downloads)", size="sm")
            upload_gallery = gr.Gallery(label="Your images", columns=6, height=140, show_label=True, preview=False)

            with gr.Accordion("…or pull an ImageNet class from the Hub", open=False):
                gr.Markdown(
                    f"Reads one class straight out of "
                    f"[`{imagenet_hub.REPO}`](https://huggingface.co/datasets/"
                    f"{imagenet_hub.REPO}) — a few MB of parquet range reads, "
                    "not a full shard. Needs network access."
                )
                class_in = gr.Dropdown(imagenet_choices(), value=1, label="ImageNet class", filterable=True)
                class_n_in = gr.Slider(50, 1300, value=1300, step=50, label="Images to pull")
                load_btn = gr.Button("Load class")

            with gr.Accordion("…or paste image URLs", open=False):
                urls_in = gr.Textbox(
                    lines=6,
                    max_lines=20,
                    label="Image URLs, one per line",
                    placeholder="https://example.com/photo1.jpg\nhttps://example.com/photo2.png",
                )
                urls_btn = gr.Button("Download images")

            # ── 2. settings ──
            preset_in = gr.Radio(list(PRESETS), value="Default", label="Preset", info=PRESET_INFO["Default"])

            d = DEFAULTS
            with gr.Row():
                backbone_in = gr.Dropdown(IMAGE_BACKBONES, value=d.backbone, label="Backbone")
                mode_in = gr.Dropdown(list(modes_for(d.backbone)), value=d.mode, label="Feature mode")
            num_images_in = gr.Slider(1, 32, value=d.num_images, step=1, label="Synthetic images")
            steps_in = gr.Slider(
                100, 10000, value=d.steps, step=100, label="Steps", info="The paper's noise-initialized runs use 10000."
            )
            augs_in = gr.Slider(1, 128, value=d.augs_per_batch, step=1, label="Augmentations per step")
            lr_in = gr.Number(value=d.lr, label="Learning rate")

            with gr.Accordion("Resolution", open=False):
                syn_res_in = gr.Dropdown(SYN_CHOICES, value=d.syn_res, label="Synthetic resolution")
                crop_res_in = gr.Dropdown(CROP_CHOICES, value=d.crop_res, label="Crop fed to the backbone")
                res_in = gr.Slider(128, 512, value=d.res, step=32, label="Resize of your images before cropping")

            with gr.Accordion("Augmentation", open=False):
                aug_in = gr.Dropdown(list(AUG_SCHEMES), value=d.aug_syn, label="Synthetic augmentation")
                noise_in = gr.Slider(0.0, 0.5, value=d.noise_std, step=0.01, label="Gaussian noise std")
                with gr.Row():
                    min_crop_in = gr.Slider(0.001, 1.0, value=d.min_crop, step=0.0001, label="Min crop area")
                    max_crop_in = gr.Slider(0.001, 1.0, value=d.max_crop, step=0.0001, label="Max crop area")
                with gr.Row():
                    min_ratio_in = gr.Slider(0.25, 4.0, value=d.min_ratio, step=0.01, label="Min aspect ratio (w/h)")
                    max_ratio_in = gr.Slider(0.25, 4.0, value=d.max_ratio, step=0.01, label="Max aspect ratio (w/h)")

            with gr.Accordion("Advanced", open=False):
                temp_in = gr.Slider(0.05, 2.0, value=d.temp, step=0.05, label="Kernel temperature τ")
                start_res_in = gr.Dropdown([1, 2, 4, 8, 16, 32], value=d.start_res, label="Initial pyramid resolution")
                extend_in = gr.Slider(
                    0, 2000, value=d.extend_every, step=50, label="Grow pyramid every N steps (0 = never)"
                )
                decorr_in = gr.Checkbox(value=d.decorrelate_color, label="Color decorrelation")
                init_real_in = gr.Checkbox(value=d.init_from_real, label="Initialize from your first N images")
                pos_batch_in = gr.Slider(0, 512, value=d.pos_batch, step=8, label="Real images per step (0 = all)")
                precision_in = gr.Dropdown(["amp", "bf16"], value=d.precision, label="Precision")
                seed_in = gr.Number(value=d.seed, precision=0, label="Seed")
                preview_in = gr.Slider(1, 200, value=10, step=1, label="Preview every N steps")

            # ── 3. run ──
            with gr.Row():
                run_btn = gr.Button("Distill", variant="primary")
                reconnect_btn = gr.Button("Reconnect")
                stop_btn = gr.Button("Stop")

        with gr.Column(scale=3):
            status_out = gr.Markdown(describe_device())
            syn_view = gr.Image(
                label="Synthetic images", height=520, interactive=False, buttons=["download", "fullscreen"]
            )
            field_plot = gr.LinePlot(
                pd.DataFrame({"step": [], "field": []}),
                x="step",
                y="field",
                height=260,
                elem_id="field-plot",
                x_title="step",
                y_title="field RMS",
                label="Drifting field magnitude",
                caption="Per-dimension RMS of the drifting field before "
                "normalization; falls as the synthetic set "
                "matches your images.",
            )
            download_out = gr.File(label="Distilled images (.zip)", visible=False)

    form = dict(
        zip(
            FORM_FIELDS,
            [
                backbone_in,
                mode_in,
                num_images_in,
                steps_in,
                augs_in,
                lr_in,
                syn_res_in,
                crop_res_in,
                res_in,
                aug_in,
                noise_in,
                min_crop_in,
                max_crop_in,
                min_ratio_in,
                max_ratio_in,
                temp_in,
                start_res_in,
                extend_in,
                decorr_in,
                init_real_in,
                pos_batch_in,
                precision_in,
                seed_in,
                preview_in,
            ],
            strict=True,
        )
    )
    image_outputs = [status_out, upload_gallery, workdir_state, paths_state]
    run_outputs = [status_out, syn_view, field_plot, download_out]

    zip_in.change(on_upload, [zip_in, workdir_state], image_outputs)

    def load_goldfish(
        workdir: str | None, progress: gr.Progress = gr.Progress()
    ) -> tuple[str, list[str], str, list[str]]:
        return load_imagenet_class(GOLDFISH, 1300, workdir, progress)

    example_btn.click(load_goldfish, workdir_state, image_outputs)
    # Fill the URL box (so the list is visible and editable), then download it.
    baishi_btn.click(lambda: BAISHI_URLS.read_text(), None, urls_in).then(
        load_urls, [urls_in, workdir_state], image_outputs
    )
    urls_btn.click(load_urls, [urls_in, workdir_state], image_outputs)
    load_btn.click(load_imagenet_class, [class_in, class_n_in, workdir_state], image_outputs)

    def on_preset(name: str) -> list[dict[str, Any]]:
        cfg = PRESETS[name]
        return [gr.update(info=PRESET_INFO[name])] + [gr.update(value=getattr(cfg, f)) for f in PRESET_FIELDS]

    preset_in.input(on_preset, preset_in, [preset_in] + [form[f] for f in PRESET_FIELDS])

    # Mode choices depend on the backbone; all ViTs but DINOv3 are fixed at 224px.
    def on_backbone(b: str, mode: str, crop: int) -> tuple[dict[str, Any], dict[str, Any]]:
        modes = modes_for(b)
        free = b.startswith("dinov3") or b == "resnet18"
        return (
            gr.update(choices=list(modes), value=mode if mode in modes else modes[0]),
            gr.update(value=crop if free else 224),
        )

    backbone_in.change(on_backbone, [backbone_in, mode_in, crop_res_in], [mode_in, crop_res_in])

    run_btn.click(
        run, [paths_state, workdir_state] + list(form.values()), run_outputs, concurrency_limit=1
    )  # one run at a time: they share one GPU
    reconnect_btn.click(reconnect, None, run_outputs)
    stop_btn.click(request_stop, None, status_out)


def main() -> None:
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )
    demo.queue().launch(server_name="0.0.0.0", theme=theme, css=CSS, show_error=True)


if __name__ == "__main__":
    main()
