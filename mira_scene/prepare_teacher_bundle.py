"""Assemble the real RGB-D files and provenance for sharing with the instructor."""
from pathlib import Path
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results/real_lab_mira"
CASE = BASE / "cases/white_divider"
DEPTH = CASE / "depth/moge"
OUT = ROOT / "teacher_bundle_iphone17pro_1x"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

files = {
    "01_rgb_original.png": CASE / "input/scene.png",
    "02_depth_preview_relative_not_meters.png": BASE / "rgbd_pointcloud/lab_depth_relative_preview.png",
    "03_depth_raw_moge_v1.npy": DEPTH / "depth.npy",
    "04_valid_pixel_mask.npy": DEPTH / "valid_mask.npy",
    "05_camera_intrinsics_moge_estimated.npy": DEPTH / "intrinsics.npy",
    "06_camera_space_xyz_map_moge.npy": DEPTH / "camera_pts_map.npy",
    "07_fullscene_rgb_pointcloud_moge.ply": DEPTH / "camera_pts.ply",
    "08_white_divider_rgb_pointcloud.ply": BASE / "rgbd_pointcloud/white_divider.ply",
    "09_camera_metadata_iphone17pro_main_1x.json": CASE / "camera_metadata_iphone17pro_1x.json",
    "10_moge_horizontal_fov_radians.txt": DEPTH / "fov_x_rad.txt",
}
for name, source in files.items():
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, OUT / name)

readme = """# 实验室 RGB-D / 点云数据（iPhone 17 Pro 主摄 1×）

## 文件说明
- `01_rgb_original.png`：原始 RGB 输入，1707×1280。
- `02_depth_preview_relative_not_meters.png`：便于查看的彩色深度预览；颜色仅作可视化。
- `03_depth_raw_moge_v1.npy`：每像素浮点深度数组，形状 1280×1707。
- `04_valid_pixel_mask.npy`：深度有效像素掩码，与深度数组同尺寸。
- `05_camera_intrinsics_moge_estimated.npy`：MoGe v1 从图像估计出的内参矩阵 K（像素单位）。
- `06_camera_space_xyz_map_moge.npy`：MoGe v1 生成的逐像素相机坐标 XYZ 图。
- `07_fullscene_rgb_pointcloud_moge.ply`：全图 RGB 着色点云（约 2.18M 点）。
- `08_white_divider_rgb_pointcloud.ply`：SAM3 白色折叠挡板实例的 RGB 点云（57,460 点）。
- `09_camera_metadata_iphone17pro_main_1x.json`：用户确认的设备/镜头和名义内参估计。
- `10_moge_horizontal_fov_radians.txt`：MoGe 输出的水平视场估计，单位弧度。

## 重要限制（请与文件一起转发）
本包当前的深度与点云来自 MoGe v1，输出尺度未经米制标定；文件中的数值不能解释为米，PLY 坐标也不是米。深度彩色 PNG 不是数值深度源。iPhone 17 Pro 主摄 1× 的名义内参依据标称 24 mm 等效焦距推算；由于原图不含 EXIF，也没有棋盘格/标定板实测，仍属估值，不是精确相机标定。要达到绝对距离尺度，还需已知场景尺寸/距离用于尺度校准，或提供可靠的米制深度源。

## 本次参数
图像尺寸：1707×1280；名义 K：fx=1183.6, fy=1183.4, cx=853.5, cy=640.0。MoGe v1 图像估计 K：fx=1223.0074, fy=1223.0076, cx=853.5, cy=640.0。两者都应视作估计值，不要将其描述成硬件实测值。
"""
(OUT / "00_README_先看此文件.md").write_text(readme, encoding="utf-8")

manifest = {
    "scene": "real lab photo / white folding divider",
    "files": list(files.keys()) + ["00_README_先看此文件.md"],
    "depth_metric": False,
    "depth_units": "relative / unscaled",
    "camera_intrinsics": "nominal estimate for iPhone 17 Pro rear main 1x; not physically calibrated",
    "pointcloud_units": "relative / unscaled",
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

zip_path = ROOT / "teacher_bundle_iphone17pro_1x.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
    for path in OUT.iterdir():
        zf.write(path, arcname=f"{OUT.name}/{path.name}")
print(OUT)
print(zip_path)
