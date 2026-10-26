#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/woom/miniconda3/envs/gsplat/bin:/usr/bin:/bin
export QT_QPA_PLATFORM=offscreen
ROOT=/home/woom/gaussian_splatting_demo/lab_video_20260929
VIDEO=/mnt/c/Users/woom/Desktop/新建文件夹/48cc438700516ba4b1bd916b4b8f829d.mp4
mkdir -p "$ROOT/images" "$ROOT/sparse"
if [[ ! -f "$ROOT/frames.done" ]]; then
 ffmpeg -nostdin -loglevel error -i "$VIDEO" -vf fps=4 -q:v 2 "$ROOT/images/frame_%04d.jpg"
 touch "$ROOT/frames.done"
fi
colmap feature_extractor --database_path "$ROOT/database.db" --image_path "$ROOT/images" \
 --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
 --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads 4
colmap sequential_matcher --database_path "$ROOT/database.db" \
 --SiftMatching.use_gpu 0 --SiftMatching.num_threads 4 --SequentialMatching.overlap 10
colmap mapper --database_path "$ROOT/database.db" --image_path "$ROOT/images" \
 --output_path "$ROOT/sparse" --Mapper.num_threads 4
python /mnt/c/Users/woom/Desktop/woom/gaussian_splatting/select_lab_model.py
