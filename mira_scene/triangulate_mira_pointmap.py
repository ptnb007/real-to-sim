"""Build a conservative colored triangle mesh from Mira's RGB-D point map.

This is a single-view surface mesh, not a watertight reconstruction. It only
connects neighboring valid image pixels and rejects large depth discontinuities.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def read_ply(path: Path):
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Expected binary little-endian PLY")
        count = int(next(x.split()[-1] for x in header if x.startswith("element vertex ")))
        props = [x.split()[-1] for x in header if x.startswith("property ")]
        has_color = all(c in props for c in ("red", "green", "blue"))
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
                         + ([ ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1") ]
                            if has_color and "alpha" in props else
                            ([ ("red", "u1"), ("green", "u1"), ("blue", "u1") ] if has_color else [])))
        data = np.frombuffer(f.read(), dtype=dtype, count=count)
    xyz = np.column_stack([data[k] for k in ("x", "y", "z")]).astype(np.float32)
    colors = (np.column_stack([data[k] for k in ("red", "green", "blue")]).astype(np.uint8)
              if has_color else np.full((count, 3), 185, np.uint8))
    return xyz, colors


def build(pointmap: np.ndarray, valid: np.ndarray, rgb: np.ndarray, step: int,
          max_edge_ratio: float):
    h, w = valid.shape
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    p = pointmap[yy, xx]
    ok = valid[yy, xx] & np.isfinite(p).all(axis=-1)
    # Keep flattened image grid, including holes; vertex IDs remain image aligned.
    flat = p.reshape(-1, 3)
    rgbflat = rgb[yy, xx].reshape(-1, 3)
    okflat = ok.ravel()
    grid_h, grid_w = ok.shape
    edges = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0 = slice(0, grid_h - dy)
        y1 = slice(dy, grid_h)
        if dx >= 0:
            x0, x1 = slice(0, grid_w - dx), slice(dx, grid_w)
        else:
            x0, x1 = slice(-dx, grid_w), slice(0, grid_w + dx)
        a = np.arange(grid_h * grid_w).reshape(grid_h, grid_w)[y0, x0]
        b = np.arange(grid_h * grid_w).reshape(grid_h, grid_w)[y1, x1]
        good = ok[y0, x0] & ok[y1, x1]
        pa, pb = p[y0, x0], p[y1, x1]
        dist = np.linalg.norm(pa - pb, axis=-1)
        local = np.linalg.norm(pa, axis=-1)
        good &= dist < np.maximum(0.012, local * max_edge_ratio)
        edges.append(np.stack([a[good], b[good]], axis=1))
    edge = np.concatenate(edges, axis=0) if edges else np.empty((0, 2), np.int64)
    # A face is accepted only if all three edges were accepted.
    edge_keys = set((np.minimum(edge[:, 0], edge[:, 1]) * len(flat) +
                     np.maximum(edge[:, 0], edge[:, 1])).tolist())
    faces = []
    ids = np.arange(grid_h * grid_w).reshape(grid_h, grid_w)
    for y in range(grid_h - 1):
        for x in range(grid_w - 1):
            a, b, c, d = int(ids[y, x]), int(ids[y, x + 1]), int(ids[y + 1, x]), int(ids[y + 1, x + 1])
            for tri in ((a, b, c), (b, d, c)):
                if not (okflat[list(tri)].all()):
                    continue
                pairs = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
                if all((min(i, j) * len(flat) + max(i, j)) in edge_keys for i, j in pairs):
                    faces.append(tri)
    if not faces:
        raise RuntimeError("No valid faces survived; increase max_edge_ratio or use denser sampling")
    faces = np.asarray(faces, dtype=np.int32)
    used = np.unique(faces)
    remap = np.full(len(flat), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return flat[used], rgbflat[used], remap[faces]


def write_ply(path: Path, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray):
    with path.open("wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\ncomment Mira MoGe RGB-D single-view surface mesh; relative scale, not metric\n"
                 f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode("ascii"))
        v = np.empty(len(vertices), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        for i, axis in enumerate(("x", "y", "z")):
            v[axis] = vertices[:, i]
        v["red"], v["green"], v["blue"] = colors.T
        f.write(v.tobytes())
        for tri in faces:
            f.write(struct.pack("<Biii", 3, *map(int, tri)))


def write_obj(path: Path, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray):
    with path.open("w", encoding="ascii") as f:
        f.write("# Mira MoGe RGB-D single-view surface mesh; relative scale, not metric\n")
        for p, c in zip(vertices, colors):
            f.write(f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g} {c[0]/255:.5f} {c[1]/255:.5f} {c[2]/255:.5f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--rgb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--max-edge-ratio", type=float, default=0.025)
    args = ap.parse_args()
    import importlib.util
    spec = importlib.util.find_spec("PIL")
    if spec is None:
        raise RuntimeError("Pillow is required to load RGB and validity images")
    from PIL import Image
    pointmap = np.load(args.pointmap).astype(np.float32)
    valid = np.load(args.valid).astype(bool)
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    vertices, colors, faces = build(pointmap, valid, rgb, args.step, args.max_edge_ratio)
    args.out.mkdir(parents=True, exist_ok=True)
    write_ply(args.out / "mira_rgbd_surface_mesh.ply", vertices, colors, faces)
    write_obj(args.out / "mira_rgbd_surface_mesh.obj", vertices, colors, faces)
    print(f"vertices={len(vertices)} faces={len(faces)} output={args.out}")


if __name__ == "__main__":
    main()
