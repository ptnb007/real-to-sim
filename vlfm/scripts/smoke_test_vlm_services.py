#!/usr/bin/env python3
"""Send one synthetic inference request to every VLFM model service."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.grounding_dino import GroundingDINOClient
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client


def main() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[64:192, 64:192] = 255

    gdino = GroundingDINOClient().predict(image, "chair .")
    blip_score = BLIP2ITMClient().cosine(image, "a white square")
    sam_mask = MobileSAMClient().segment_bbox(image, [64, 64, 192, 192])
    yolo = YOLOv7Client().predict(image)

    print(f"GroundingDINO detections: {len(gdino.boxes)}")
    print(f"BLIP2-ITM cosine: {blip_score:.6f}")
    print(f"MobileSAM mask pixels: {int(sam_mask.sum())}")
    print(f"YOLOv7 detections: {len(yolo.boxes)}")
    print("All four VLFM services completed inference.")


if __name__ == "__main__":
    main()
