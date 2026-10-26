from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image


CASE_DIR = Path("/home/woom/Mira-Scene/work/lab_no_robot")
RESULT_DIR = Path("/mnt/c/Users/woom/Desktop/woom/mira_scene/results/lab_no_robot_objects")
NAMES = ["divider", "desk", "ottoman", "poster_stand"]


def main() -> None:
    points_map = np.load(CASE_DIR / "depth/moge/camera_pts_map.npy")
    valid_map = np.load(CASE_DIR / "depth/moge/valid_mask.npy").astype(bool)
    colors = np.asarray(Image.open(CASE_DIR / "input/scene.png").convert("RGB"), dtype=np.float32) / 255.0
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    figure = plt.figure(figsize=(12, 8), dpi=160)
    axis = figure.add_subplot(111, projection="3d")

    for index, name in enumerate(NAMES):
        mask = np.asarray(Image.open(CASE_DIR / f"input/mask_{index:03d}.png").convert("L")) > 127
        valid = mask & valid_map & np.isfinite(points_map).all(axis=-1)
        xyz = points_map[valid].astype(np.float64)
        rgb = colors[valid].astype(np.float64)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        cloud.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(str(RESULT_DIR / f"{index:03d}_{name}.ply"), cloud, write_ascii=False)

        count = min(6000, len(xyz))
        if count:
            chosen = rng.choice(len(xyz), count, replace=False)
            sample = xyz[chosen]
            axis.scatter(sample[:, 0], sample[:, 2], -sample[:, 1], s=0.35, c=rgb[chosen], label=name)
        print(f"{name}: {len(xyz)} points")

    axis.set_xlabel("camera X")
    axis.set_ylabel("depth Z")
    axis.set_zlabel("up")
    axis.set_title("Object-level point clouds from Mira-Scene depth")
    axis.legend(loc="upper right")
    axis.view_init(elev=22, azim=-65)
    axis.set_box_aspect((1.6, 1.4, 1.0))
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "object_pointcloud_preview.png", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
