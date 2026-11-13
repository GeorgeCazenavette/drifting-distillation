#!/usr/bin/env bash
# "An Image is Worth a Thousand Paintings": distill a painter's collection into
# one 2048x2048 canvas with SigLIP2.  PAINTINGS holds one subfolder per painter
# (e.g. paintings/claude_monet/*.jpg); CLASS picks the painter (sorted order).
# Each step drifts 100 random square crops covering 0.38%-4.8% of the canvas
# area (~126-448 px), i.e. roughly the scale of a 224 px view of one painting.
set -euo pipefail
PAINTINGS=${PAINTINGS:-data/paintings}
CLASS=${CLASS:-0}
OUT=${OUT:-outputs}/art

python -m src.distillation.optimize_image --data_dir $PAINTINGS --classes $CLASS --out_dir $OUT --backbone siglip2 --mode cls_grid --num_images 1 --augs_per_batch 100 --syn_res 2048 --min_crop 0.0038 --max_crop 0.0479 --min_ratio 1 --max_ratio 1 --steps 10000
