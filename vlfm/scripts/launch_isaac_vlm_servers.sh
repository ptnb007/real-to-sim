#!/usr/bin/env bash
set -euo pipefail

export VLFM_PYTHON="${VLFM_PYTHON:-$(which python)}"
export MOBILE_SAM_CHECKPOINT="${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}"
export GROUNDING_DINO_CONFIG="${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
export GROUNDING_DINO_WEIGHTS="${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}"
export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-12181}"
export BLIP2ITM_PORT="${BLIP2ITM_PORT:-12182}"
export SAM_PORT="${SAM_PORT:-12183}"

session_name="vlfm_isaac_servers"
if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "VLFM Isaac visual services are already running in tmux session ${session_name}."
  exit 0
fi

tmux new-session -d -s "${session_name}"
tmux split-window -v -t "${session_name}:0"
tmux split-window -h -t "${session_name}:0.0"
tmux send-keys -t "${session_name}:0.0" "${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t "${session_name}:0.1" "${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t "${session_name}:0.2" "${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m

echo "Started GroundingDINO, BLIP2-ITM and MobileSAM for Isaac Sim."
echo "Wait about 90 seconds before starting Isaac Sim."
echo "Monitor with: tmux attach -t ${session_name}"
