"""Render the MoGe depth map paired with robot_divider_scene's source photo."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CASE = Path(__file__).resolve().parent / "results/real_lab_mira/cases/robot_divider_scene"
PACK = Path(__file__).resolve().parent.parent / "Mira_OneDrive_分享包"
DEPTH_PATH = CASE / "depth/moge/depth.npy"
VALID_PATH = CASE / "depth/moge/valid_mask.npy"
IMAGE_PATH = CASE / "input/scene.jpg"
OUTPUT_PATH = PACK / "00_原图与匹配深度图/robot_divider_depth_relative_not_metric.png"

depth = np.load(DEPTH_PATH).astype(np.float32)
valid = np.load(VALID_PATH).astype(bool) & np.isfinite(depth)
image = Image.open(IMAGE_PATH)
if image.size != (depth.shape[1], depth.shape[0]):
    raise ValueError(f"Photo/depth size mismatch: photo={image.size}, depth={depth.shape[::-1]}")

values = depth[valid]
lo, hi = np.percentile(values, [1, 99])
shown = np.ma.masked_where(~valid, np.clip(depth, lo, hi))
fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
im = ax.imshow(shown, cmap="turbo", vmin=lo, vmax=hi)
ax.set_title("Robot-divider lab photo — MoGe depth preview (relative scale; not metric)", fontsize=13)
ax.axis("off")
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Predicted depth (relative / unscaled; NOT meters)")
fig.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"saved={OUTPUT_PATH}; shape={depth.shape}; scale=relative/unscaled; clip_percentiles=1,99")
