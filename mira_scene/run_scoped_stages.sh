#!/usr/bin/env bash
set -u
ROOT=/mnt/c/Users/woom/Desktop/woom/mira_scene/results/lab_between_dividers
/home/woom/mira_envs/geometry-rtx5060/bin/python \
  /mnt/c/Users/woom/Desktop/woom/mira_scene/run_real_mira_stages.py \
  --root "$ROOT"
