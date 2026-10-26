"""Create scoped surface meshes from completed Mira/MoGe point maps."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from triangulate_mira_pointmap import build, write_obj, write_ply

ROOT = HERE / "results" / "lab_between_dividers"

for case in ("tables_purple", "white_boundary"):
    base = ROOT / "cases" / case
    pointmap = np.load(base / "depth" / "moge" / "camera_pts_map.npy").astype(np.float32)
    valid = np.load(base / "depth" / "moge" / "valid_mask.npy").astype(bool)
    rgb = np.asarray(Image.open(base / "input" / "scene.png").convert("RGB"))
    masks = []
    for path in sorted((base / "input").glob("mask_*.png")):
        masks.append(np.asarray(Image.open(path).convert("L")) > 127)
    floor_path = base / "input" / "floor_mask.png"
    if floor_path.exists():
        masks.append(np.asarray(Image.open(floor_path).convert("L")) > 127)
    scope = np.logical_or.reduce(masks)
    valid &= scope
    vertices, colors, faces = build(pointmap, valid, rgb, step=2, max_edge_ratio=0.025)
    out = ROOT / "surface_mesh" / case
    out.mkdir(parents=True, exist_ok=True)
    write_ply(out / "mira_scoped_surface.ply", vertices, colors, faces)
    write_obj(out / "mira_scoped_surface.obj", vertices, colors, faces)
    print(case, "vertices", len(vertices), "faces", len(faces), "out", out)
