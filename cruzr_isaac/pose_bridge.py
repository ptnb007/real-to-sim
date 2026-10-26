"""Renderer-neutral rigid Gaussian binding contract. SI metres, wxyz quaternions.

For each link: T_world_asset = T_world_link @ T_link_asset. Gaussian positions
and covariance must BOTH rotate. SH/view-dependent colour is renderer-owned.
No original CRUZR manifest schema is guessed here.
"""
import json
import math
from pathlib import Path


def matrix(position, quaternion):
    p = [float(v) for v in position]
    q = [float(v) for v in quaternion]
    if len(p) != 3 or len(q) != 4 or not all(math.isfinite(v) for v in p + q):
        raise ValueError("Expected finite xyz position and wxyz quaternion")
    norm = math.sqrt(sum(v*v for v in q))
    if norm < 1e-12:
        raise ValueError("Zero quaternion")
    w, x, y, z = [v/norm for v in q]
    return [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), p[0]],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), p[1]],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), p[2]],
        [0., 0., 0., 1.],
    ]


def multiply(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def validate_rigid(t):
    if len(t) != 4 or any(len(row) != 4 for row in t):
        raise ValueError("Binding must be a 4x4 matrix")
    if not all(math.isfinite(float(v)) for row in t for v in row):
        raise ValueError("Non-finite binding")
    if any(abs(t[3][i] - [0, 0, 0, 1][i]) > 1e-6 for i in range(4)):
        raise ValueError("Invalid homogeneous bottom row")
    for i in range(3):
        for j in range(3):
            if abs(sum(t[k][i]*t[k][j] for k in range(3)) - (i == j)) > 1e-5:
                raise ValueError("Binding must contain a rigid rotation, no scale/shear")
    a,b,c = [r[:3] for r in t[:3]]
    det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if abs(det-1) > 1e-5:
        raise ValueError("Binding rotation must be right-handed")


def load_bindings(path, require_assets=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "cruzr-isaac-bindings/v1":
        raise ValueError("Explicit normalized binding schema required; original manifest needs review")
    links = data.get("links", [])
    if not links:
        raise ValueError("No link bindings")
    names, prims = set(), set()
    for item in links:
        name, prim = item["name"], item["prim_path"]
        if name in names or prim in prims or not prim.startswith("/"):
            raise ValueError("Duplicate name/prim or non-absolute prim path")
        names.add(name); prims.add(prim)
        validate_rigid(item["T_link_asset"])
        scale = float(item["asset_unit_to_m"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Positive asset_unit_to_m required")
        asset = (path.parent / item["ply"]).resolve()
        if require_assets and not asset.is_file():
            raise FileNotFoundError(asset)
        item["ply"] = str(asset)
    return links


def make_frame(frame, time_s, bindings, poses):
    result = []
    for item in bindings:
        p, q = poses[item["name"]]
        result.append({
            "name": item["name"], "prim_path": item["prim_path"],
            "ply": item["ply"], "asset_unit_to_m": item["asset_unit_to_m"],
            "T_world_asset": multiply(matrix(p, q), item["T_link_asset"]),
        })
    return {"schema": "cruzr-isaac-poses/v1", "frame": frame,
            "simulation_time_s": time_s, "length_unit": "m",
            "matrix_layout": "row-major; column vectors", "links": result}


def transform_gaussian(mean, covariance, t, unit_to_m):
    """Reference math for a renderer adapter, not a PLY parser or renderer."""
    out_mean = [sum(t[i][j]*mean[j]*unit_to_m for j in range(3))+t[i][3] for i in range(3)]
    out_cov = [[sum(t[i][k]*covariance[k][l]*t[j][l]*unit_to_m**2
                    for k in range(3) for l in range(3)) for j in range(3)] for i in range(3)]
    return out_mean, out_cov
