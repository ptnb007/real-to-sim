"""Create a clearly labeled visualization of the current relative depth map."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "results/real_lab_mira/cases/white_divider"
depth_path = CASE / "depth/moge/depth.npy"
mask_path = CASE / "depth/moge/valid_mask.npy"
output = ROOT / "results/real_lab_mira/rgbd_pointcloud/lab_depth_relative_preview.png"

depth = np.load(depth_path).astype(np.float32)
valid = np.load(mask_path).astype(bool) & np.isfinite(depth) & (depth > 0)
lo, hi = np.percentile(depth[valid], [2, 98])
normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
colored = plt.get_cmap("turbo")(normalized)[..., :3]
colored[~valid] = 0

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.imshow(colored)
ax.axis("off")
sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(lo, hi))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("MoGe v1 predicted depth (relative / unscaled; NOT meters)")
ax.set_title("Lab RGB-D depth preview — relative scale, not metric")
fig.tight_layout()
fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
print(output)
