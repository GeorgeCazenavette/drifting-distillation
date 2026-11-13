#!/usr/bin/env bash
# Example: distill ImageNet-1k across a Slurm job array, 100 classes per task.
#   sbatch --array=0-9 scripts/slurm_array.sh --backbone dinov3 --mode cls_grid --num_images 10 --steps 10000 --out_dir outputs/imagenet-1k/dinov3_noise_ipc10
# Adapt the #SBATCH lines to your cluster.  Any GPU with 24 GB suffices for
# 256 px images at IPC 10/50.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
DATA=${DATA:-data/imagenet/train}
PER_TASK=${PER_TASK:-100}
START=$((SLURM_ARRAY_TASK_ID * PER_TASK))
python -m src.distillation.optimize_image --data_dir $DATA --classes $(seq $START $((START + PER_TASK - 1))) "$@"
