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
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ROOT=/home/woom/gaussian_splatting_demo/lab_video_20260929
python /mnt/c/Users/woom/Desktop/woom/gaussian_splatting/select_lab_model.py
MODEL="$(cat "$ROOT/selected_model.txt")"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
ns-train splatfacto --data "$ROOT" --output-dir /home/woom/gaussian_splatting_demo/outputs \
 --experiment-name lab_video --timestamp "$STAMP" --max-num-iterations "${GS_ITERATIONS:-1000}" \
 --steps-per-save 100 --vis tensorboard --pipeline.model.num-downscales 0 \
 --pipeline.datamanager.camera-res-scale-factor 0.5 \
 colmap --colmap-path "$MODEL" --downscale-factor 1
CONFIG="/home/woom/gaussian_splatting_demo/outputs/lab_video/splatfacto/$STAMP/config.yml"
ns-export gaussian-splat --load-config "$CONFIG" --output-dir "$ROOT/exports/$STAMP"
python /mnt/c/Users/woom/Desktop/woom/gaussian_splatting/render_preview.py "$CONFIG" "$ROOT/exports/$STAMP/previews"
printf '%s\n' "$CONFIG" > "$ROOT/latest_config.txt"
