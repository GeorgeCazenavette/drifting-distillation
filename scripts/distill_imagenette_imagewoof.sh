#!/usr/bin/env bash
# ImageNette / ImageWoof (official splits) distillation, DINOv3 and ResNet-18.
# RDED initialization scores crops with the dataset's 10-class ResNet-18 teacher
# (downloaded from Hugging Face on first use; the same teacher serves the KD eval).
# The datasets themselves are downloaded into $ROOT on first use.
set -euo pipefail
ROOT=${ROOT:-data}
OUT=${OUT:-outputs}

for DS in imagenette imagewoof; do
  DATA=$ROOT/${DS}2/train
  for IPC in 10 50; do
    for BB in dinov3 resnet18; do
      MODE=$([ $BB = resnet18 ] && echo cls_grid2 || echo cls_grid)
      python -m src.distillation.optimize_image --data_dir $DATA --classes $(seq 0 9) --num_images $IPC --out_dir $OUT/$DS/${BB}_noise_ipc$IPC --backbone $BB --mode $MODE --steps 10000
      python -m src.distillation.optimize_image --data_dir $DATA --classes $(seq 0 9) --num_images $IPC --out_dir $OUT/$DS/${BB}_rded_ipc$IPC --backbone $BB --mode $MODE --init rded --rded_teacher $DS --steps 5000
    done
  done
done
