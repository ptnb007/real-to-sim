#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/woom/miniconda3/envs/gsplat/bin:/usr/bin:/bin
export QT_QPA_PLATFORM=offscreen
SOURCE=/home/woom/gaussian_splatting_demo/lab_video_20260929
ROOT=/home/woom/gaussian_splatting_demo/lab_enhanced_no_robot_20260930
PHOTOS=/mnt/c/Users/woom/Desktop/woom/gaussian_splatting/input_photos_20260930
rm -rf "$ROOT"
mkdir -p "$ROOT/images/video" "$ROOT/images/photos" "$ROOT/masks/video" "$ROOT/masks/photos" "$ROOT/sparse"
cp "$SOURCE"/images/*.jpg "$ROOT/images/video/"
cp /home/woom/gaussian_splatting_demo/robot_masks_v2/*.png "$ROOT/masks/video/"
for photo in "$PHOTOS"/*.jpg; do
  name="photo_$(basename "$photo")"
  cp "$photo" "$ROOT/images/photos/$name"
  python - "$photo" "$ROOT/masks/photos/${name%.jpg}.png" <<'PY'
import sys
from PIL import Image
src, dst = sys.argv[1:]
im = Image.open(src)
Image.new('L', im.size, 255).save(dst)
PY
done
colmap feature_extractor --database_path "$ROOT/database.db" --image_path "$ROOT/images" \
  --ImageReader.single_camera 0 --ImageReader.single_camera_per_folder 1 --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads 4
colmap exhaustive_matcher --database_path "$ROOT/database.db" \
  --SiftMatching.use_gpu 0 --SiftMatching.num_threads 4
colmap mapper --database_path "$ROOT/database.db" --image_path "$ROOT/images" \
  --output_path "$ROOT/sparse" --Mapper.num_threads 4
GS_DATA_ROOT="$ROOT" python /mnt/c/Users/woom/Desktop/woom/gaussian_splatting/select_lab_model.py
