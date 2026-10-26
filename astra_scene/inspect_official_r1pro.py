import json
import sys
from pathlib import Path

sys.argv.append("--/app/vulkan=false")
from isaacsim import SimulationApp

APP = SimulationApp({"headless": True, "multi_gpu": False})

import omni.usd
from pxr import UsdPhysics
from isaacsim.core.utils.stage import open_stage

HERE = Path(__file__).resolve().parent
SCENE = HERE / "focus_v4" / "focused_zone_with_official_r1pro.usda"

try:
    open_stage(str(SCENE))
    stage = omni.usd.get_context().get_stage()
    roots = []
    joints = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            if "wheel" in name or "steer" in name:
                joints.append(str(prim.GetPath()))
    result = {"articulation_roots": roots, "base_joints": joints}
    (HERE / "focus_v4" / "official_r1pro_inspection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("OFFICIAL_R1PRO_INSPECTION " + json.dumps(result))
finally:
    APP.close()
