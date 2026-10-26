#!/usr/bin/env bash
set -euo pipefail

ENV=/home/woom/miniconda3/envs/gsplat
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
ROOT=/home/woom/gaussian_splatting_demo
DATA="$ROOT/data/blender/lego"
OUT="$ROOT/outputs"

mkdir -p "$ROOT"
cd "$ROOT"

if [[ ! -f "$DATA/transforms_train.json" ]]; then
  "$ENV/bin/ns-download-data" blender --save-dir "$ROOT/data"
fi

"$ENV/bin/ns-train" splatfacto \
  --data "$DATA" \
  --output-dir "$OUT" \
  --max-num-iterations 3000 \
  --vis tensorboard \
  blender-data
