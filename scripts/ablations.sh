#!/usr/bin/env bash
# Appendix hyperparameter sweeps.  Each changes one setting of the default
# configuration (DINOv3, noise init, ImageNet-1k, IPC 10, 10000 steps), which
# is the dinov3_noise_ipc10 run of scripts/distill_imagenet.sh.
set -euo pipefail
DATA=${DATA:-data/imagenet/train}
OUT=${OUT:-outputs}/imagenet-1k/ablations
CLASSES=${CLASSES:-$(seq 0 999)}
BASE="--data_dir $DATA --classes $CLASSES --num_images 10 --backbone dinov3 --mode cls_grid --steps 10000"

# Augmented views per step ("None" = one deterministic resized view).
python -m src.distillation.optimize_image $BASE --out_dir $OUT/augs_none --aug_syn resize --augs_per_batch 1
for A in 1 2 5; do
  python -m src.distillation.optimize_image $BASE --out_dir $OUT/augs_$A --augs_per_batch $A
done

# Backbone size (ViT-S / ViT-L; ViT-B is the default).
python -m src.distillation.optimize_image $BASE --out_dir $OUT/dinov3_small --backbone dinov3-small
python -m src.distillation.optimize_image $BASE --out_dir $OUT/dinov3_large --backbone dinov3-large

# Kernel temperature (default 0.5).
python -m src.distillation.optimize_image $BASE --out_dir $OUT/temp_0.05 --temps 0.05
python -m src.distillation.optimize_image $BASE --out_dir $OUT/temp_5 --temps 5

# Drifted features: CLS only / CLS + mean patch (default: CLS + every patch).
python -m src.distillation.optimize_image $BASE --out_dir $OUT/mode_cls --mode cls
python -m src.distillation.optimize_image $BASE --out_dir $OUT/mode_cls_mean --mode cls_mean

# Parameterization: no color decorrelation / raw pixels instead of the pyramid.
# (The pixel run used lr 0.01 and 5000 steps.)
python -m src.distillation.optimize_image $BASE --out_dir $OUT/no_decorrelation --decorrelate_color False
python -m src.distillation.optimize_image $BASE --out_dir $OUT/pixel --parameterization pixel --lr 0.01 --steps 5000
