"""Validate that the reconstructed lab and Galaxea R1 Pro compose correctly."""

import json
import sys
from pathlib import Path

sys.argv.append("--/app/vulkan=false")
from isaacsim import SimulationApp

APP = SimulationApp({"headless": True, "multi_gpu": False})

from pxr import Usd, UsdGeom, UsdPhysics


HERE = Path(__file__).resolve().parent
SCENE = HERE / "isaac_usd" / "lab_with_r1pro.usda"
REPORT = HERE / "isaac_usd" / "lab_with_r1pro_verification.json"


stage = Usd.Stage.Open(str(SCENE), load=Usd.Stage.LoadAll)
if stage is None:
    raise RuntimeError(f"Cannot open stage: {SCENE}")

robot_root = stage.GetPrimAtPath("/World/R1Pro")
colliders = stage.GetPrimAtPath("/World/Colliders")

if not robot_root.IsValid():
    raise RuntimeError("Missing /World/R1Pro reference")
if not colliders.IsValid():
    raise RuntimeError("The lab collision proxy is missing")

visual_prims = [
    prim for prim in stage.Traverse()
    if prim.GetPath().HasPrefix("/World")
    and not prim.GetPath().HasPrefix("/World/R1Pro")
    and prim.IsA(UsdGeom.Gprim)
]
if not visual_prims:
    raise RuntimeError("The visual lab geometry is missing")

articulation_prims = [
    prim for prim in Usd.PrimRange(robot_root) if prim != robot_root
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
]
if not articulation_prims:
    descendants = [str(prim.GetPath()) for prim in Usd.PrimRange(robot_root) if prim != robot_root]
    raise RuntimeError(
        "R1 Pro reference has no composed articulation. Descendants: "
        + json.dumps(descendants[:30])
    )
articulation = articulation_prims[0]

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
box = bbox_cache.ComputeWorldBound(robot_root).ComputeAlignedBox()
minimum = box.GetMin()
maximum = box.GetMax()

joint_count = sum(1 for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint))
collision_count = sum(
    1 for prim in Usd.PrimRange(colliders)
    if prim != colliders and prim.HasAPI(UsdPhysics.CollisionAPI)
)

result = {
    "status": "passed",
    "scene": str(SCENE),
    "robot": "Galaxea R1 Pro",
    "robot_root": str(robot_root.GetPath()),
    "articulation_root": str(articulation.GetPath()),
    "joint_count": joint_count,
    "collision_proxy_count": collision_count,
    "robot_world_bounds_m": {
        "min": [float(v) for v in minimum],
        "max": [float(v) for v in maximum],
    },
    "visual_environment_loaded": True,
    "environment_geometry_count": len(visual_prims),
    "physics_scene_loaded": stage.GetPrimAtPath("/World/PhysicsScene").IsValid(),
}
REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("LAB_R1PRO_COMPOSITION_VERIFIED " + json.dumps(result, ensure_ascii=False))
APP.close()
