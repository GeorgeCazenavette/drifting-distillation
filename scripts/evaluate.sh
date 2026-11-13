#!/usr/bin/env bash
# All evaluations of one distilled set.  Usage:
#   scripts/evaluate.sh outputs/imagenet-1k/dinov3_noise_ipc10 imagenet-1k
#   scripts/evaluate.sh outputs/imagewoof/dinov3_noise_ipc10  imagewoof
# Real data is read from data/ (data/imagenet/val, data/imagewoof2/val, ...).
# From-scratch results are means over seeds 42-44 in the paper; coverage, 1-NN
# and the linear probe are seed-invariant (crops are drawn from a fixed seed).
set -euo pipefail
SET=$1
DS=$2
RES=${RES:-$SET/eval}
mkdir -p $RES

if [ "$DS" = imagenet-1k ]; then
  for SEED in 42 43 44; do
    python -m src.evaluation.train_soft --data $SET --out $RES/hard_seed$SEED.json --alpha 0 --seed $SEED
    python -m src.evaluation.train_soft --data $SET --out $RES/soft_seed$SEED.json --seed $SEED
    python -m src.evaluation.train_kd --data $SET --dataset imagenet-1k --out $RES/kd_seed$SEED.json --seed $SEED
  done
  # Feature-space metrics (main table: siglip2, mocov3; coverage matrix adds the rest).
  for BB in siglip2 mocov3 clip dinov3 eupe eva02 vit; do
    python -m src.evaluation.eval_coverage --data $SET --backbone $BB --out $RES/coverage_$BB.json
  done
  for BB in siglip2 mocov3; do
    python -m src.evaluation.eval_linear --data $SET --backbone $BB --out $RES/linear_$BB.json
  done
else  # imagenette / imagewoof
  for SEED in 42 43 44; do
    python -m src.evaluation.train_dc --data $SET --dataset $DS --out $RES/hard_seed$SEED.json --seed $SEED
    python -m src.evaluation.train_kd --data $SET --dataset $DS --out $RES/kd_seed$SEED.json --seed $SEED
  done
fi
