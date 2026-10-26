@echo off
title VLFM policy server for Isaac Sim
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd /home/woom/vlfm && source /home/woom/miniconda3/etc/profile.d/conda.sh && conda activate vlfm && bash scripts/launch_isaac_vlm_servers.sh && python scripts/run_isaac_vlfm_server.py --goal person --port 12190"
pause
