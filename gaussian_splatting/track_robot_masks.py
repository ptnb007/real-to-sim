#!/usr/bin/env python3
"""Write manually verified, piecewise-interpolated Galaxea exclusion masks."""

from pathlib import Path
import argparse
import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("masks", type=Path)
    parser.add_argument("overlays", type=Path)
    args = parser.parse_args()

    paths = sorted(args.images.glob("*.jpg"))
    if not paths:
        raise SystemExit("No JPG frames found")
    args.masks.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)

    # Bounding boxes are x, y, width, height in the 1280x720 source frames.
    # The robot remains partly visible at the left edge through frame 22,
    # exits between frames 23-27, and leaves the image after frame 57.
    segments = [
        [(0, (930, 45, 330, 510)), (8, (670, 20, 350, 520)),
         (16, (330, 0, 320, 500)), (18, (0, 0, 270, 500)),
         (22, (0, 0, 300, 620))],
        [(28, (0, 80, 350, 620)), (32, (70, 20, 390, 650)),
         (36, (250, 10, 390, 660)), (40, (440, 0, 400, 675)),
         (44, (690, 0, 390, 680)), (48, (1010, 0, 270, 680)),
         (51, (790, 80, 490, 640))],
        [(52, (0, 90, 360, 630)), (57, (0, 110, 280, 590))],
    ]

    def box_for(frame_index: int):
        for keyframes in segments:
            if keyframes[0][0] <= frame_index <= keyframes[-1][0]:
                for (ia, a), (ib, b) in zip(keyframes, keyframes[1:]):
                    if ia <= frame_index <= ib:
                        t = (frame_index - ia) / max(1, ib - ia)
                        return tuple((1 - t) * av + t * bv for av, bv in zip(a, b))
        return None

    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        overlay = image.copy()
        tracked = box_for(index)
        if tracked is not None:
            x, y, w, h = tracked
            # Use a deliberately generous exclusion margin. Thin exposed edges of
            # the arm/base are otherwise enough for a multi-view splat model to
            # reconstruct a recognizable translucent robot.
            margin_x = max(60, int(w * 0.35))
            margin_y = max(50, int(h * 0.18))
            x0 = max(0, int(x) - margin_x)
            y0 = max(0, int(y) - margin_y)
            x1 = min(width, int(x + w) + margin_x)
            y1 = min(height, int(y + h) + margin_y)
            mask[y0:y1, x0:x1] = 0
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 4)
        cv2.imwrite(str(args.masks / f"{path.stem}.png"), mask)
        cv2.imwrite(str(args.overlays / path.name), overlay)


if __name__ == "__main__":
    main()
