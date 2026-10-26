#!/usr/bin/env bash
set -euo pipefail

# Reproduction environment for VLFM on RTX 50-series GPUs under WSL2.
# Run from the repository root after Ubuntu and NVIDIA GPU support work in WSL.

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script must run inside Linux/WSL2."
  exit 1
fi

if [[ ! -f pyproject.toml || ! -d vlfm ]]; then
  echo "Run this script from the VLFM repository root."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable in WSL. Finish WSL/NVIDIA setup first."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Install Miniconda in WSL, then rerun this script."
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx vlfm; then
  conda create -n vlfm python=3.9 cmake=3.26 ninja -y \
    --override-channels -c conda-forge
fi
conda activate vlfm

# Habitat 0.2.4 matches the VLFM release. Use the display/Bullet variant in
# WSLg; the headless build insists on matching an EGL device to CUDA device 0.
conda install -y habitat-sim=0.2.4 withbullet \
  "habitat-sim-mutex=1.0=display_bullet" \
  --override-channels -c conda-forge -c aihabitat

python -m pip install --upgrade "pip<25" "setuptools==69.5.1" wheel

# PyTorch 1.12/CUDA 11.3 from the paper predates Blackwell. PyTorch 2.7 with
# CUDA 12.8 is the first official wheel line supporting RTX 50-series GPUs.
python -m pip install \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

# GroundingDINO's deformable-attention extension needs a CUDA compiler and
# development headers matching the PyTorch CUDA 12.8 runtime.
conda install -y cuda-nvcc=12.8.61 cuda-libraries-dev=12.8.0 \
  --override-channels -c nvidia -c conda-forge

python -m pip install \
  numpy==1.26.4 \
  flask requests seaborn open3d imageio-ffmpeg==0.6.0 \
  transformers==4.26.0 timm==0.4.12 \
  opencv-python==4.5.5.64 \
  spacy==3.7.5 \
  salesforce-lavis==1.0.2

python -m pip install \
  habitat-lab==0.2.420230405 \
  habitat-baselines==0.2.420230405

python -m pip install \
  "git+https://github.com/naokiyokoyama/frontier_exploration.git" \
  "git+https://github.com/ChaoningZhang/MobileSAM.git" \
  "git+https://github.com/naokiyokoyama/depth_camera_filtering.git"

# Use the source revision already downloaded next to VLFM. CUDA extensions are
# compiled against the PyTorch/CUDA runtime selected above.
CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 \
  python -m pip install -e GroundingDINO --no-build-isolation
python -m pip install -e . --no-deps

# The Habitat downloader creates scene_datasets/hm3d as a symlink into its
# versioned-data directory, so only create the parent directory here.
mkdir -p data/scene_datasets
python -m vlfm.utils.generate_dummy_policy
python scripts/check_reproduction_assets.py --allow-missing-scenes

python - <<'PY'
import torch
import habitat_sim
import habitat
import vlfm

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("habitat-sim import: OK")
print("habitat-lab import: OK")
print("vlfm import: OK")
PY

echo "VLFM base environment is ready."
