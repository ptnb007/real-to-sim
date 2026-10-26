from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    source = Path("/home/woom/Mira-Scene/work/lab_no_robot/depth/moge/depth.npy")
    target = Path("/mnt/c/Users/woom/Desktop/woom/mira_scene/results/lab_no_robot_depth")
    target.mkdir(parents=True, exist_ok=True)

    depth = np.load(source)
    valid = np.isfinite(depth) & (depth > 0)
    near, far = np.percentile(depth[valid], [2, 98])

    fig, ax = plt.subplots(figsize=(12, 9), dpi=160)
    image = ax.imshow(depth, cmap="turbo_r", vmin=near, vmax=far)
    ax.set_title("Mira-Scene / MoGe Depth Preview (meters)")
    ax.axis("off")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Depth (m)")
    fig.tight_layout()
    fig.savefig(target / "depth_preview.png", bbox_inches="tight")
    plt.close(fig)

    np.save(target / "depth.npy", depth)
    print(target / "depth_preview.png")


if __name__ == "__main__":
    main()
