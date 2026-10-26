"""Validate Galaxea R1 Pro composition in the newest focus_v4 lab scene."""

import json
import sys
from pathlib import Path

sys.argv.append("--/app/vulkan=false")
from isaacsim import SimulationApp

APP = SimulationApp({"headless": True, "multi_gpu": False})

from pxr import Usd, UsdGeom, UsdPhysics


HERE = Path(__file__).resolve().parent
SCENE = HERE / "focus_v4" / "focused_zone_with_r1pro.usda"
REPORT = HERE / "focus_v4" / "focused_zone_with_r1pro_verification.json"

try:
    stage = Usd.Stage.Open(str(SCENE), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Cannot open stage: {SCENE}")

    robot_root = stage.GetPrimAtPath("/World/R1Pro")
    collision_root = stage.GetPrimAtPath("/World/Collision")
    if not robot_root.IsValid() or not collision_root.IsValid():
        raise RuntimeError("Robot reference or focus_v4 collision layer is missing")

    articulations = [
        prim for prim in Usd.PrimRange(robot_root)
        if prim != robot_root and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not articulations:
        raise RuntimeError("Galaxea R1 Pro articulation did not compose")

    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    ).ComputeWorldBound(robot_root).ComputeAlignedBox()
    minimum, maximum = bbox.GetMin(), bbox.GetMax()

    collision_count = sum(
        1 for prim in Usd.PrimRange(collision_root)
        if prim != collision_root and prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    environment_geometry_count = sum(
        1 for prim in stage.Traverse()
        if prim.GetPath().HasPrefix("/World")
        and not prim.GetPath().HasPrefix("/World/R1Pro")
        and prim.IsA(UsdGeom.Gprim)
    )
    joint_count = sum(
        1 for prim in Usd.PrimRange(robot_root) if prim.IsA(UsdPhysics.Joint)
    )

    result = {
        "status": "passed",
        "scene_version": "focus_v4 (latest)",
        "scene": str(SCENE),
        "robot": "Galaxea R1 Pro",
        "robot_spawn_xyz_m": [0.25, -0.55, 0.0],
        "articulation_root": str(articulations[0].GetPath()),
        "joint_count": joint_count,
        "collision_proxy_count": collision_count,
        "environment_geometry_count": environment_geometry_count,
        "physics_scene_loaded": stage.GetPrimAtPath("/World/PhysicsScene").IsValid(),
        "robot_world_bounds_m": {
            "min": [float(v) for v in minimum],
            "max": [float(v) for v in maximum],
        },
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FOCUS_V4_R1PRO_VERIFIED " + json.dumps(result, ensure_ascii=False))
finally:
    APP.close()
