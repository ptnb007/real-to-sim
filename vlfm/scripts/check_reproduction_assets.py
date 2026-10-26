#!/usr/bin/env python3
"""Check the local assets needed for a VLFM HM3D evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_ASSETS = {
    "MobileSAM": ROOT / "data" / "mobile_sam.pt",
    "GroundingDINO": ROOT / "data" / "groundingdino_swint_ogc.pth",
    "YOLOv7": ROOT / "data" / "yolov7-e6e.pt",
    "PointNav": ROOT / "data" / "pointnav_weights.pth",
    "HM3D ObjectNav episodes": (
        ROOT / "data" / "datasets" / "objectnav" / "hm3d" / "v1" / "val" / "val.json.gz"
    ),
    "GroundingDINO source": ROOT / "GroundingDINO" / "groundingdino",
    "YOLOv7 source": ROOT / "yolov7" / "models",
}

SCENE_ROOT = ROOT / "data" / "scene_datasets" / "hm3d"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-missing-scenes",
        action="store_true",
        help="Do not fail when only the credential-gated HM3D scene files are absent.",
    )
    args = parser.parse_args()
    missing = []
    print("VLFM reproduction assets")
    for label, path in PUBLIC_ASSETS.items():
        exists = path.exists()
        print(f"[{'OK' if exists else 'MISSING'}] {label}: {path.relative_to(ROOT)}")
        if not exists:
            missing.append(label)

    scene_files = list(SCENE_ROOT.rglob("*.glb")) if SCENE_ROOT.exists() else []
    if scene_files:
        print(f"[OK] HM3D scenes: {len(scene_files)} .glb files")
    else:
        print("[MISSING] HM3D scenes: Matterport-authenticated download is still required")
        if not args.allow_missing_scenes:
            missing.append("HM3D scenes")

    if missing:
        print("\nMissing:", ", ".join(missing))
        return 1

    if args.allow_missing_scenes and not scene_files:
        print("\nAll public assets are present; HM3D scenes remain pending.")
    else:
        print("\nAll required assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
