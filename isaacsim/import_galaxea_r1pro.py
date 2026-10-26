"""Import and validate the official Galaxea R1 Pro 2026 URDF in Isaac Sim 5.1.

This is deliberately limited to the first milestone: import the wheeled dual-arm
robot, advance physics, render one frame, save a reusable USD stage, and write a
machine-readable verification record. Chassis and arm motion are tested by later
scripts so an import failure cannot be confused with a controller failure.
"""

import json
import os
import sys
import traceback
from pathlib import Path

# Let Isaac Sim resolve package://r1pro_urdf/... mesh references from the
# official ROS package. This must be present before the URDF extension starts.
_EARLY_WORKSPACE = Path(__file__).resolve().parent.parent
os.environ["ROS_PACKAGE_PATH"] = str(
    _EARLY_WORKSPACE / "third_party" / "galaxea_urdf" / "R1Pro"
)

# This workstation uses the D3D12 workaround already validated by the Jetbot and
# A2D scenes. It must be supplied before SimulationApp is constructed.
sys.argv.append("--/app/vulkan=false")

from isaacsim import SimulationApp


APP = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "multi_gpu": False,
        "width": 960,
        "height": 540,
    }
)

import numpy as np
import omni.kit.commands
import omni.usd
from PIL import Image
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from pxr import Gf, PhysicsSchemaTools


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
URDF = (
    WORKSPACE
    / "third_party"
    / "galaxea_urdf"
    / "R1Pro"
    / "urdf_r1pro_g1z_2026"
    / "urdf"
    / "r1pro_2026.urdf"
)
SCENE = HERE / "galaxea_r1pro_scene.usda"
PREVIEW = HERE / "galaxea_r1pro_preview.png"
REPORT = HERE / "galaxea_r1pro_verification.json"


def main() -> None:
    if not URDF.is_file():
        raise FileNotFoundError(f"Official R1 Pro URDF not found: {URDF}")

    print(f"[R1PRO] Loading URDF: {URDF}", flush=True)
    world = World(stage_units_in_meters=1.0)
    # Use a local physics plane. World.add_default_ground_plane() asks the
    # online asset service for a visual material and can stall offline.
    PhysicsSchemaTools.addGroundPlane(
        omni.usd.get_context().get_stage(),
        "/World/GroundPlane",
        "Z",
        20.0,
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.45, 0.45, 0.45),
    )

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Isaac Sim did not create the URDF import configuration")

    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.distance_scale = 1.0

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(URDF),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status or not prim_path:
        raise RuntimeError(f"R1 Pro URDF import failed: status={status}, prim={prim_path}")
    print(f"[R1PRO] Imported articulation at {prim_path}", flush=True)

    camera = Camera(prim_path="/World/R1ProValidationCamera", resolution=(960, 540))
    set_camera_view(
        eye=np.array([4.2, 4.2, 2.8]),
        target=np.array([0.0, 0.0, 1.0]),
        camera_prim_path="/World/R1ProValidationCamera",
    )
    set_camera_view(
        eye=np.array([4.2, 4.2, 2.8]),
        target=np.array([0.0, 0.0, 1.0]),
    )

    world.reset()
    camera.initialize()
    robot = Articulation(prim_path)
    robot.initialize()

    if not robot.is_physics_handle_valid():
        raise RuntimeError(f"Imported prim is not a valid articulation: {prim_path}")
    print(f"[R1PRO] Physics articulation valid with {robot.num_dof} DOF", flush=True)

    initial_positions = np.asarray(robot.get_joint_positions(), dtype=float)
    for _ in range(90):
        world.step(render=True)
    final_positions = np.asarray(robot.get_joint_positions(), dtype=float)
    print("[R1PRO] Advanced 90 physics/render frames", flush=True)

    if robot.num_dof < 24:
        raise RuntimeError(f"Unexpectedly low R1 Pro DOF count: {robot.num_dof}")
    if not np.isfinite(final_positions).all():
        raise RuntimeError("R1 Pro joint state contains non-finite values")

    # Capture before save_as_stage. In Isaac Sim 5.1 saving/reloading the
    # current stage invalidates the camera render product.
    for _ in range(5):
        world.step(render=True)
    rgba = camera.get_rgba()
    preview_saved = rgba is not None and bool(np.asarray(rgba).size)
    if preview_saved:
        Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(PREVIEW)
        print(f"[R1PRO] Saved preview: {PREVIEW}", flush=True)
    else:
        print("[R1PRO] Headless D3D12 camera returned no frame; GUI preview remains available", flush=True)

    if not omni.usd.get_context().save_as_stage(str(SCENE)):
        raise RuntimeError(f"Failed to save R1 Pro stage: {SCENE}")
    print(f"[R1PRO] Saved stage: {SCENE}", flush=True)

    names = list(robot.dof_names)
    categories = {
        "steering": [name for name in names if "steer" in name.lower()],
        "wheels": [name for name in names if "wheel" in name.lower()],
        "torso": [name for name in names if "torso" in name.lower()],
        "left_arm": [name for name in names if "left_arm" in name.lower()],
        "right_arm": [name for name in names if "right_arm" in name.lower()],
        "grippers": [name for name in names if "gripper" in name.lower()],
    }
    result = {
        "status": "passed",
        "milestone": "official Galaxea R1 Pro URDF import and physics/render validation",
        "isaac_sim_version": "5.1",
        "source_urdf": str(URDF),
        "articulation_prim": str(prim_path),
        "dof": int(robot.num_dof),
        "joint_names": names,
        "joint_groups": categories,
        "initial_joint_positions": initial_positions.tolist(),
        "final_joint_positions": final_positions.tolist(),
        "simulation_frames": 90,
        "scene": str(SCENE),
        "preview": str(PREVIEW) if preview_saved else None,
        "preview_note": None if preview_saved else "Open the saved USD in the Isaac Sim GUI for visual inspection.",
        "scope_limit": "Import only; chassis and arm motion are separate validations.",
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("GALAXEA_R1PRO_IMPORT_VERIFIED " + json.dumps(result, ensure_ascii=False), flush=True)


exit_code = 0
try:
    main()
except BaseException:
    exit_code = 1
    traceback.print_exc()
finally:
    APP.close()
if exit_code:
    raise SystemExit(exit_code)
