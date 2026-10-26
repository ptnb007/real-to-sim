@echo off
title InternNav iPlanner local planner
wsl.exe -d Ubuntu-24.04 -- bash -lc "source /home/woom/miniconda3/etc/profile.d/conda.sh && conda activate vlfm && cd /mnt/c/Users/woom/Desktop/woom/third_party/NavDP/baselines/iplanner && python iplanner_server.py --port 8888 --device cuda:0"
pause
