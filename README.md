# Dataset Distillation via Drifting

Code for *Dataset Distillation via Drifting* (NeurIPS 2026).

We distill a dataset one class at a time: a handful of synthetic images,
parameterized as Laplacian pyramids, are optimized so that they *together with
their augmentations* match the distribution of the real class in the feature
space of a frozen encoder, using the drifting field of
[Deng et al.](https://arxiv.org/abs/2602.04770) as the training signal.

This repository contains exactly what is needed to reproduce the paper:
ImageNet-1k / ImageNette / ImageWoof image distillation with every backbone in
the paper, the from-scratch, knowledge-distillation and feature-space
evaluations, audio (FMA) and video (HMDB51) distillation with the herding
baseline, and the painting canvases.

## Setup

```bash
conda create -n drifting -c conda-forge python=3.14 libarchive && conda activate drifting
pip install -r requirements.txt
```

(`libarchive` provides `bsdtar`, used to unpack HMDB51's `.rar` archives; `unrar`
or `7z`, if installed, work as well.)

Everything is run from the repository root as `python -m src.<module>`; there is
nothing to install.

**Data.** Image datasets are read as ImageFolder directories (one
subdirectory per class; class index = sorted subdirectory order). Everything
except ImageNet-1k is downloaded into `data/` automatically the first time it
is used (or ahead of time with
`python -m src.tools.download_data imagenette imagewoof fma hmdb51 teachers`):

| dataset | location | source |
|---|---|---|
| ImageNet-1k | `data/imagenet/{train,val}` (wnid folders) | **manual**: [image-net.org](https://image-net.org) |
| ImageNette | `data/imagenette2/{train,val}` | automatic ([imagenette2.tgz](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz)) |
| ImageWoof | `data/imagewoof2/{train,val}` | automatic ([imagewoof2.tgz](https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz)) |
| FMA-small | `data/fma/fma_small_by_genre`, `data/fma/fma_metadata` | automatic ([FMA](https://github.com/mdeff/fma), 7.2 GB) |
| HMDB51 | `data/hmdb51/{videos,testTrainMulti_7030_splits}` | automatic ([official release](https://huggingface.co/datasets/Serrelab/hmdb51), 2.1 GB) |
| paintings | `data/paintings/<painter>/*.jpg` | any collection |

**Weights.** Backbones download from torchvision / Hugging Face / timm on first
use (DINOv3 requires accepting its license on Hugging Face; EuPE loads through
`torch.hub` from GitHub). The 10-class ResNet-18 teachers for ImageNette and
ImageWoof (used for RDED initialization and knowledge-distillation evaluation)
are [RDED](https://github.com/LINs-lab/RDED)'s released models, converted to our
class order (`src/tools/convert_rded_teachers.py`) and hosted with the distilled
data; they are downloaded automatically.

## Distillation

```bash
python -m src.distillation.optimize_image \
    --data_dir data/imagenet/train --out_dir outputs/dinov3_ipc10 \
    --backbone dinov3 --mode cls_grid --num_images 10 --steps 10000 \
    --classes 0 1 2
```

writes `outputs/dinov3_ipc10/{class:04d}/{i:04d}.png`. Classes are independent,
so a dataset is distilled by splitting `--classes` across GPUs or jobs
(`scripts/slurm_array.sh`); finished classes are skipped and interrupted ones
resume. One class at IPC 10 takes about 15 minutes on a modern GPU, and every
image experiment fits in 24 GB.

Defaults are the paper's recipe: Adam (lr 1e-3), kernel temperature 0.5, 10
augmented views per image per step (flip, random resized crop, Gaussian
noise), all real images of the class as positives, a 256 px pyramid starting
at 8x8 with one level added every 200 steps, and fixed color decorrelation.

- `--backbone`: `resnet18`, `vit`, `clip`, `siglip2`, `eva02`, `mocov3`, `dinov3`, `dinov3-small`, `dinov3-large`, `eupe`
- `--mode`: `cls_grid` (CLS + every patch token; ViTs), `cls_grid2` (pooled + last two stages; ResNet-18), `cls`, `cls_mean`
- `--init`: `noise`, or `rded`: start from real crops chosen by RDED's selection (factor 1), fitted into the pyramid
- `--steps`: 10000 for noise init, 5000 for RDED init (as in the paper)

**Real initialization.** `--init rded` re-implements the patch selection of
[RDED](https://github.com/LINs-lab/RDED) with factor 1: 300 random real images,
5 random square crops each, scored by a ResNet-18 teacher's cross-entropy; the
lowest-loss crop of each image competes, and the best `num_images` are kept.
The teacher is torchvision's ImageNet ResNet-18 by default; for ImageNette /
ImageWoof pass `--rded_teacher imagenette` / `--rded_teacher imagewoof` to use
the dataset's 10-class teacher.

**Paper commands.**

| script | experiments |
|---|---|
| `scripts/distill_imagenet.sh` | every "Ours" row on ImageNet-1k (main tables, appendix backbones, coverage matrix) |
| `scripts/ablations.sh` | appendix sweeps: augmentations per step, backbone size, temperature, drifted features, parameterization |
| `scripts/distill_imagenette_imagewoof.sh` | ImageNette / ImageWoof |
| `scripts/distill_art.sh` | 2048x2048 painting canvases |

Note: the paper states 5000 distillation iterations; the noise-initialized
runs behind the tables in fact used 10000 (RDED-initialized runs used 5000).
The scripts use the settings that were actually run.

## Distilled data

Every distilled set from the paper is on Hugging Face at
[`georgecazenavette/drifting-distillation`](https://huggingface.co/datasets/georgecazenavette/drifting-distillation)
(browse it at [drifting-distillation.github.io/data](https://drifting-distillation.github.io/data/)).
To download one set by its tag and extract it as `{class:04d}/{name}`:

```bash
python -m src.tools.download_distilled --list
python -m src.tools.download_distilled imagenet-1k_dinov3_noise_ipc10 --out distilled/dinov3_noise_ipc10
```

## Evaluation

Every evaluation takes a distilled set as an ImageFolder (`--data`) and reads
the real validation split from `--data_root` (default `data/`). Defaults are the
paper's protocols (appendix, *Evaluation Protocols*); results are written as JSON.
`scripts/evaluate.sh SET DATASET` runs all of them for one set.

| evaluation | command |
|---|---|
| hard-label from scratch (ImageNet-1k) | `python -m src.evaluation.train_soft --data SET --alpha 0 --out hard.json` |
| soft-label from scratch (ImageNet-1k) | `python -m src.evaluation.train_soft --data SET --out soft.json` |
| knowledge distillation, EDC recipe (ImageNet-1k, 4 teachers) | `python -m src.evaluation.train_kd --data SET --dataset imagenet-1k --out kd.json` |
| knowledge distillation (ImageNette / ImageWoof) | `python -m src.evaluation.train_kd --data SET --dataset imagewoof --out kd.json` |
| hard-label from scratch (ImageNette / ImageWoof, Minimax protocol) | `python -m src.evaluation.train_dc --data SET --dataset imagewoof --out hard.json` |
| coverage@5 and 1-NN in a feature space | `python -m src.evaluation.eval_coverage --data SET --backbone siglip2 --out coverage.json` |
| augmented linear probe | `python -m src.evaluation.eval_linear --data SET --backbone siglip2 --out linear.json` |

All from-scratch evaluations train a ResNet-18 and report final-epoch top-1 on
the validation set; the paper reports mean ± std over `--seed 42 43 44`.
Coverage, 1-NN and the linear probe draw their crops from a fixed seed, so one
run suffices. The feature-space evaluations cache the real validation
embeddings in `.val_cache/` (one ~10-minute pass per backbone).

## Audio and video

`scripts/audio_video.sh` runs the paper's audio and video experiments:
5 samples per class, drifting the mean token of every layer, the herding
coreset baseline in the same feature space, and linear-probe / cosine 1-NN
evaluation on mean-pooled final-layer features.

| | data | backbone | distillation |
|---|---|---|---|
| audio | FMA-small, 8 genres (`data/fma/fma_small_by_genre`) | MusicFM-FMA | `python -m src.distillation.optimize_audio --data_root data/fma/fma_small_by_genre --classes $(seq 0 7) --out_dir outputs/fma/ours_ipc5` |
| video | HMDB51, split 1 (`data/hmdb51`) | VideoMAE-v2-Base | `python -m src.distillation.optimize_video --data_root data --classes $(seq 0 50) --out_dir outputs/hmdb51/ours_ipc5` |

Herding: `python -m src.baselines.herding_{audio,video}`; evaluation:
`python -m src.evaluation.{audio,video}.evaluate --support_dir <set>`. Video
distillation needs an 80 GB GPU.

## Interactive demo

```bash
python -m src.app            # http://localhost:7860
```

Upload a zip of images of one subject (or paste a list of image URLs, or pull
any ImageNet class from the Hub) and watch a handful of synthetic images form. The demo runs the same
pyramid, augmentation, drifting loss and extractors as the distillation
script, on a single image set treated as one distribution. It needs a GPU;
on a shared machine, pick an idle one with `CUDA_VISIBLE_DEVICES`.

* The run is detached from the browser: if the page's event stream drops
  (long runs through proxies often do), the job keeps going and **Reconnect**
  re-attaches. The PNGs so far are rewritten on every preview, so even a
  killed server leaves the latest images on disk.
* *Pull an ImageNet class* range-reads one class out of
  [`evanarlian/imagenet_1k_resized_256`](https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256)
  (a few MB per class, cached locally), so no ImageNet download is needed.
* Two example image sets, both downloaded on demand: *1300 ImageNet goldfish*
  (through the Hub loader above) and *42 Qi Baishi paintings* (the image URLs
  in `src/app/examples/qi_baishi_urls.txt`, from the Art Institute of Chicago
  and WikiArt). Pair the latter with the *Art canvas* preset.
* DINOv3 weights are gated on Hugging Face. Without access (accept the license
  on the model page, then `hf auth login`), the demo warns and falls back to
  DINOv2 ViT-B (`--backbone dinov2`, which is not used in the paper).
* Presets set every option at once: *Default* (DINOv3 ViT-B, 10 images,
  10k steps) and *Art canvas*, the paper's painting recipe (one 2048px
  SigLIP2 canvas, 100 square crops per step, as in `scripts/distill_art.sh`).

## License

The code is released under the [MIT License](LICENSE), except for the
third-party files below, which keep their original terms.

**Third-party code.**

| file | derived from | license |
|---|---|---|
| `src/extractors/audio/_musicfm.py` | [MusicFM](https://github.com/minzwon/musicfm) | MIT (notice in the file) |
| `src/extractors/image/_moco_vit.py` | [RCG](https://github.com/LTH14/rcg) MoCo-v3 ViT | MIT (notice in the file) |
| `src/evaluation/train_dc.py`, `src/evaluation/dc_utils.py` | [Minimax Diffusion](https://github.com/vimar-gu/MinimaxDiffusion) (IDC lineage) | no license stated upstream |
| `src/evaluation/edc_utils.py` | [EDC](https://github.com/shaoshitong/EDC) / [SRe2L](https://github.com/VILA-Lab/SRe2L) | no license stated upstream |

The ImageNette / ImageWoof teachers are RDED's models (Apache-2.0), hosted with
the distilled data.

## Citation

```bibtex
@inproceedings{cazenavette2026drifting,
  title={Dataset Distillation via Drifting},
  author={Cazenavette, George and Quan, Angelina and Daras, Giannis and Torralba, Antonio and Sitzmann, Vincent},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
