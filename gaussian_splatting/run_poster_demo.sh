#!/usr/bin/env bash
set -euo pipefail

ENV=/home/woom/miniconda3/envs/gsplat
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
ROOT=/home/woom/gaussian_splatting_demo
DATA="$ROOT/data/nerfstudio/poster"
OUT="$ROOT/outputs"

mkdir -p "$ROOT"
cd "$ROOT"

if [[ ! -f "$DATA/transforms.json" ]]; then
  "$ENV/bin/ns-download-data" nerfstudio --capture-name=poster --save-dir "$ROOT/data"
fi

# 3k iterations are enough for the first visual/PLY validation on an 8 GB GPU.
"$ENV/bin/ns-train" splatfacto \
  --data "$DATA" \
  --output-dir "$OUT" \
  --max-num-iterations 3000 \
  --vis tensorboard
