#!/usr/bin/env bash
# ImageNet-1k distillation commands for every "Ours" row of the paper
# (main from-scratch table, feature-space table, appendix backbone table,
# inter-model coverage matrix).
#
# Each command distills all 1000 classes in one process.  Classes are
# independent: to parallelize, give each GPU/job a slice of --classes,
# e.g. --classes $(seq 0 99), --classes $(seq 100 199), ...
# (see scripts/slurm_array.sh).  Finished classes are skipped on rerun.
#
# Noise-initialized runs use 10000 steps; RDED-initialized runs use 5000.
set -euo pipefail
DATA=${DATA:-data/imagenet/train}
OUT=${OUT:-outputs}
CLASSES=${CLASSES:-$(seq 0 999)}

for IPC in 10 50; do
  # --- main table: ResNet-18, DINOv3, MoCo-v3 -------------------------------
  python -m src.distillation.optimize_image --data_dir $DATA --classes $CLASSES --num_images $IPC --out_dir $OUT/imagenet-1k/resnet18_noise_ipc$IPC --backbone resnet18 --mode cls_grid2 --steps 10000
  python -m src.distillation.optimize_image --data_dir $DATA --classes $CLASSES --num_images $IPC --out_dir $OUT/imagenet-1k/resnet18_rded_ipc$IPC --backbone resnet18 --mode cls_grid2 --init rded --steps 5000
  for BB in dinov3 mocov3; do
    python -m src.distillation.optimize_image --data_dir $DATA --classes $CLASSES --num_images $IPC --out_dir $OUT/imagenet-1k/${BB}_noise_ipc$IPC --backbone $BB --mode cls_grid --steps 10000
    python -m src.distillation.optimize_image --data_dir $DATA --classes $CLASSES --num_images $IPC --out_dir $OUT/imagenet-1k/${BB}_rded_ipc$IPC --backbone $BB --mode cls_grid --init rded --steps 5000
  done

  # --- appendix backbones (+ EuPE, a row of the coverage matrix) ---------------
  for BB in vit clip siglip2 eva02 eupe; do
    python -m src.distillation.optimize_image --data_dir $DATA --classes $CLASSES --num_images $IPC --out_dir $OUT/imagenet-1k/${BB}_noise_ipc$IPC --backbone $BB --mode cls_grid --steps 10000
    python -m src.distillation.optimize_image --data_dir $DATA --classes $CLASSES --num_images $IPC --out_dir $OUT/imagenet-1k/${BB}_rded_ipc$IPC --backbone $BB --mode cls_grid --init rded --steps 5000
  done
done
