#!/usr/bin/env bash
set -euo pipefail

CONDA=/home/woom/miniconda3/bin/conda
ENV=/home/woom/miniconda3/envs/gsplat

if [[ ! -x "$ENV/bin/python" ]]; then
  "$CONDA" create -y -n gsplat --override-channels -c conda-forge python=3.10 pip
fi

"$CONDA" install -y -n gsplat --override-channels \
  -c nvidia/label/cuda-12.8.0 -c conda-forge cuda-toolkit=12.8

"$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV/bin/python" -m pip install \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/python" -m pip install nerfstudio==1.1.5
# Nerfstudio 1.1.5 pins gsplat 1.4.0, which predates Blackwell sm_120.
# Current gsplat main also contains the CUDA 12.8 cudaEventCreateWithFlags fix.
"$ENV/bin/python" -m pip install --no-deps --upgrade \
  git+https://github.com/nerfstudio-project/gsplat.git@main

export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"

"$ENV/bin/python" - <<'PY'
import torch
from importlib.metadata import version
print("NERFSTUDIO", version("nerfstudio"))
print("TORCH", torch.__version__)
print("CUDA", torch.version.cuda, torch.cuda.is_available())
print("GPU", torch.cuda.get_device_name(0))
print("CAPABILITY", torch.cuda.get_device_capability(0))
PY

nvcc --version | tail -n 1

"$ENV/bin/ns-train" splatfacto --help >/dev/null
echo "GSPLAT_ENV_VERIFIED"
