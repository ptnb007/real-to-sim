"""Smoke-test a fresh fixed-base R1 Pro import without touching the source USD."""
from pathlib import Path
import sys

sys.argv.append("--/app/vulkan=false")
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "multi_gpu": False})
import omni.kit.commands
import omni.usd
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from pxr import Gf, UsdGeom

here = Path(__file__).resolve().parent
fixed_asset = here / "fixed_base_r1pro.usda"
urdf_text = (here / "r1pro_grasp_demo.urdf").read_text(encoding="utf-8")
urdf_text = urdf_text.replace(
    "package://r1pro_urdf/meshes/",
    "../../third_party/galaxea_urdf/R1Pro/urdf_r1pro_g1z_2026/meshes/",
)
urdf = here / "r1pro_fixed_base_import_test.urdf"
urdf.write_text(urdf_text, encoding="utf-8")
context = omni.usd.get_context()
context.new_stage()
for _ in range(30):
    app.update()
status, config = omni.kit.commands.execute("URDFCreateImportConfig")
config.fix_base = True
config.merge_fixed_joints = False
config.import_inertia_tensor = True
status, robot_path = omni.kit.commands.execute(
    "URDFParseAndImportFile", urdf_path=str(urdf), import_config=config
)
print(f"URDF_IMPORT_RESULT status={status} path={robot_path}", flush=True)
stage = context.get_stage()
robot_prim = stage.GetPrimAtPath(robot_path)
if not robot_prim.IsValid():
    raise RuntimeError(f"Imported robot prim not found at {robot_path}")
xf = UsdGeom.Xformable(robot_prim)
robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.55, 0.48, 0.01))
robot_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0.70710678, 0.0, 0.0, -0.70710678))
robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.0))
robot_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0))
if not stage.GetRootLayer().Export(str(fixed_asset)):
    raise RuntimeError(f"Could not export fixed-base robot USD: {fixed_asset}")
print(f"EXPORTED_FIXED_BASE_ASSET {fixed_asset}", flush=True)
robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.55, 0.48, 0.01))
robot_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0.70710678, 0.0, 0.0, -0.70710678))
world = World(stage_units_in_meters=1.0)
world.reset()
robot = Robot(prim_path=str(robot_path), name="r1pro_fixed_base_test")
robot.initialize()
print(f"URDF_DOF_NAMES {robot.dof_names}", flush=True)
print(f"URDF_BASE_POSE {robot.get_world_pose()}", flush=True)
for _ in range(60):
    world.step(render=False)
print(f"URDF_AFTER_60_BASE {robot.get_world_pose()}", flush=True)
print(f"URDF_AFTER_60_Q {np.asarray(robot.get_joint_positions()).tolist()}", flush=True)
urdf.unlink(missing_ok=True)
app.close()
