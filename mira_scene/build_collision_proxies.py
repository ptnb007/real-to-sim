from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


ROOT = Path("/mnt/c/Users/woom/Desktop/woom/mira_scene/results/lab_no_robot_objects")
OBJECTS = [
    ("000_divider.ply", "divider", [105, 160, 255, 180]),
    ("001_desk.ply", "desk", [80, 190, 120, 180]),
    ("002_ottoman.ply", "ottoman", [245, 190, 60, 180]),
    ("003_poster_stand.ply", "poster_stand", [235, 85, 85, 180]),
]


def main() -> None:
    scene = trimesh.Scene()
    records = []

    for filename, name, color in OBJECTS:
        cloud = o3d.io.read_point_cloud(str(ROOT / filename))
        points = np.asarray(cloud.points)
        if len(points) < 10:
            continue

        # Trim extreme monocular-depth outliers before fitting a closed proxy.
        lower = np.quantile(points, 0.02, axis=0)
        upper = np.quantile(points, 0.98, axis=0)
        trimmed = points[np.all((points >= lower) & (points <= upper), axis=1)]
        fitted = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(trimmed)).get_oriented_bounding_box(robust=True)
        extent = np.maximum(np.asarray(fitted.extent), [0.04, 0.04, 0.04])
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(fitted.R)
        transform[:3, 3] = np.asarray(fitted.center)

        mesh = trimesh.creation.box(extents=extent, transform=transform)
        mesh.visual.face_colors = color
        scene.add_geometry(mesh, node_name=name, geom_name=f"{name}_collision")
        records.append((name, extent.tolist(), transform[:3, 3].tolist()))

    target = ROOT / "lab_collision_proxies.glb"
    target.write_bytes(scene.export(file_type="glb"))
    for name, extent, center in records:
        print(f"{name}: extent={np.round(extent, 3).tolist()} center={np.round(center, 3).tolist()}")
    print(target)


if __name__ == "__main__":
    main()
