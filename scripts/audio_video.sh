#!/usr/bin/env bash
# Audio (FMA-small, MusicFM) and video (HMDB51, VideoMAE-v2) distillation at
# 5 clips per class, the herding baseline in the same feature space, and the
# linear-probe / 1-NN evaluation of both (paper Table "audio-video").
# The paper's video runs used one 80 GB H100 per class.
set -euo pipefail
ROOT=${ROOT:-data}
OUT=${OUT:-outputs}
FMA=$ROOT/fma/fma_small_by_genre

# Audio: 8 genres
python -m src.distillation.optimize_audio --data_root $FMA --classes $(seq 0 7) --out_dir $OUT/fma/ours_ipc5
python -m src.baselines.herding_audio --data_root $FMA --out_dir $OUT/fma/herding_ipc5
python -m src.evaluation.audio.evaluate --data_root $FMA --support_dir $OUT/fma/ours_ipc5 --output_path $OUT/fma/ours_ipc5.json
python -m src.evaluation.audio.evaluate --data_root $FMA --support_dir $OUT/fma/herding_ipc5 --output_path $OUT/fma/herding_ipc5.json

# Video: 51 classes
python -m src.distillation.optimize_video --data_root $ROOT --classes $(seq 0 50) --out_dir $OUT/hmdb51/ours_ipc5
python -m src.baselines.herding_video --data_root $ROOT --out_dir $OUT/hmdb51/herding_ipc5
python -m src.evaluation.video.evaluate --data_root $ROOT --support_dir $OUT/hmdb51/ours_ipc5 --output_path $OUT/hmdb51/ours_ipc5.json
python -m src.evaluation.video.evaluate --data_root $ROOT --support_dir $OUT/hmdb51/herding_ipc5 --output_path $OUT/hmdb51/herding_ipc5.json
