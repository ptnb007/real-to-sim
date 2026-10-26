#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script inside the configured Ubuntu/WSL environment."
  exit 1
fi

if [[ ! -f pyproject.toml || ! -d vlfm ]]; then
  echo "Run this script from the VLFM repository root."
  exit 1
fi

scene="data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
if [[ ! -f "$scene" ]]; then
  echo "HM3D val scene is missing: $scene"
  echo "Download hm3d_val_v0.2 with your Matterport credentials first."
  exit 2
fi

# WSLg exposes a Mesa EGL device that is not CUDA-indexed. Device -1 selects
# the default EGL renderer and avoids Habitat-Sim's CUDA/EGL device matching.
python -m vlfm.run \
  habitat.dataset.data_path=data/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz \
  habitat.simulator.habitat_sim_v0.gpu_device_id=-1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.verbose=False \
  "$@"
