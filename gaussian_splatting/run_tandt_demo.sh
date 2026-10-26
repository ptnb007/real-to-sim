#!/usr/bin/env bash
set -euo pipefail

ENV=/home/woom/miniconda3/envs/gsplat
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST=12.0
ROOT=/home/woom/gaussian_splatting_demo
ARCHIVE="$ROOT/tandt_db.zip"
EXTRACTED="$ROOT/tandt_db"
OUT="$ROOT/outputs"
ITERATIONS="${GS_ITERATIONS:-3000}"

mkdir -p "$ROOT"

if [[ ! -f "$ARCHIVE" ]]; then
  curl --fail --location --retry 3 \
    https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip \
    --output "$ARCHIVE"
fi

if [[ ! -d "$EXTRACTED" ]]; then
  mkdir -p "$EXTRACTED"
  "$ENV/bin/python" -m zipfile -e "$ARCHIVE" "$EXTRACTED"
fi

# The archive contains multiple COLMAP scenes. Use the smallest valid scene for
# the first 8 GB VRAM validation and print the selection for traceability.
DATA="$(find "$EXTRACTED" -type d -path '*/sparse/0' -printf '%h\n' | sed 's#/sparse$##' | sort | head -n 1)"
if [[ -z "$DATA" ]]; then
  echo "No COLMAP scene found after extracting $ARCHIVE" >&2
  exit 2
fi
echo "SELECTED_COLMAP_SCENE=$DATA"

"$ENV/bin/ns-train" splatfacto \
  --data "$DATA" \
  --output-dir "$OUT" \
  --max-num-iterations "$ITERATIONS" \
  --vis tensorboard \
  colmap \
  --colmap-path sparse/0
