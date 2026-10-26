"""Preserve a source PLY and apply a conservative statistical outlier filter."""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


SOURCE = Path(__file__).resolve().parent / "results/real_lab_mira/cases/robot_divider_scene/depth/moge/robot.ply"
OUT = SOURCE.parents[2] / "pointcloud_cleaning"
K_NEIGHBORS = 20
STD_RATIO = 3.0
CHUNK = 40_000


def load_ply(path: Path):
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Invalid PLY: missing end_header")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Expected a binary little-endian PLY")
        count = int(next(row.split()[-1] for row in header if row.startswith("element vertex ")))
        props = [row.split()[-1] for row in header if row.startswith("property ")]
        if not all(k in props for k in ("red", "green", "blue")):
            raise ValueError("Expected RGB vertex colors")
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")]
                         + ([ ("alpha", "u1") ] if "alpha" in props else []))
        values = np.frombuffer(stream.read(), dtype=dtype, count=count)
    xyz = np.column_stack([values[k] for k in ("x", "y", "z")]).astype(np.float32, copy=True)
    colors = np.column_stack([values[k] for k in ("red", "green", "blue")]).astype(np.uint8, copy=True)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz, colors, finite


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rows = np.empty(len(xyz), dtype=dtype)
    for key, axis in zip(("x", "y", "z"), range(3)):
        rows[key] = xyz[:, axis]
    rows["red"], rows["green"], rows["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment SOR: k=20, threshold=global_mean+3*global_std\n"
              f"element vertex {len(rows)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(rows.tobytes())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clone = OUT / "robot_original_clone.ply"
    filtered = OUT / "robot_sor_k20_sigma3.ply"
    report_path = OUT / "sor_report.json"
    shutil.copy2(SOURCE, clone)
    xyz, colors, finite = load_ply(SOURCE)
    valid_xyz, valid_colors = xyz[finite], colors[finite]
    tree = cKDTree(valid_xyz, compact_nodes=True, balanced_tree=True)
    means = np.empty(len(valid_xyz), dtype=np.float32)
    for start in range(0, len(valid_xyz), CHUNK):
        stop = min(start + CHUNK, len(valid_xyz))
        distances, _ = tree.query(valid_xyz[start:stop], k=K_NEIGHBORS + 1, workers=-1)
        means[start:stop] = distances[:, 1:].mean(axis=1)
        print(f"SOR_QUERY {stop}/{len(valid_xyz)}", flush=True)
    global_mean = float(means.mean())
    global_std = float(means.std())
    threshold = global_mean + STD_RATIO * global_std
    keep = means <= threshold
    write_ply(filtered, valid_xyz[keep], valid_colors[keep])
    report = {
        "source": str(SOURCE), "original_clone": str(clone), "filtered": str(filtered),
        "method": "statistical outlier removal", "neighbors": K_NEIGHBORS,
        "std_ratio": STD_RATIO, "threshold": threshold, "mean_neighbor_distance": global_mean,
        "std_neighbor_distance": global_std, "input_points": int(len(xyz)),
        "finite_points": int(finite.sum()), "kept_points": int(keep.sum()),
        "removed_points": int((~keep).sum()), "manual_screen_space_crop": False,
        "note": "No guessed screen-space crop was applied; compare filtered and source clouds for detail loss."
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
