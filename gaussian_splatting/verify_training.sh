#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/home/woom/miniconda3/envs/gsplat
export PATH="$CUDA_HOME/bin:/usr/bin:/bin"
CUDA_TARGET="$CUDA_HOME/targets/x86_64-linux"
export CPATH="$CUDA_TARGET/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_TARGET/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_TARGET/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_TARGET/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST=12.0
ROOT=/home/woom/gaussian_splatting_demo
DATA="$ROOT/tandt_db/db/drjohnson"
mkdir -p "$DATA/images_4"
for input in "$DATA"/images/*.jpg; do
  output="$DATA/images_4/$(basename "$input")"
  if [[ ! -s "$output" ]]; then
    ffmpeg -nostdin -loglevel error -i "$input" -vf 'scale=trunc(iw/4):trunc(ih/4)' -q:v 2 "$output"
  fi
done
ns-train splatfacto --data "$DATA" --output-dir "$ROOT/outputs" \
  --max-num-iterations "${GS_ITERATIONS:-100}" --steps-per-save 100 \
  --vis tensorboard --pipeline.model.num-downscales 0 \
  colmap --colmap-path sparse/0 --downscale-factor 4
