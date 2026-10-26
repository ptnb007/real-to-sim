#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

exec bash scripts/run_val_mini_wsl.sh \
  'habitat_baselines.eval.video_option=["disk"]' \
  habitat_baselines.video_dir=results/val_mini_video \
  "$@"
