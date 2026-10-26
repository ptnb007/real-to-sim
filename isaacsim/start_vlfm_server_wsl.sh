#!/usr/bin/env bash
set -euo pipefail

cd /home/woom/vlfm
source /home/woom/miniconda3/etc/profile.d/conda.sh
conda activate vlfm

echo "Starting VLFM visual services..."
bash scripts/launch_vlm_servers.sh

echo "Starting Isaac Sim bridge for target: ${VLFM_GOAL:-person}"
exec python scripts/run_isaac_vlfm_server.py --goal "${VLFM_GOAL:-person}" --port 12190
