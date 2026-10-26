"""Create and verify a minimal Isaac Sim scene containing an NVIDIA Jetbot."""

import sys

# Isaac Sim 5.1's Python experience forces Vulkan. On this Windows/RTX 5060
# host the compatibility checker succeeds with D3D12 while Vulkan's RTX scene
# database crashes during startup, so override the backend before SimulationApp
# parses the command line.
sys.argv.append("--/app/vulkan=false")

from isaacsim import SimulationApp

# Keep the first validation light enough for an 8 GB GPU.
simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "multi_gpu": False,
        "width": 960,
        "height": 540,
    }
)

import json
import os

import carb
import numpy as np
from PIL import Image
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
import omni.usd


def main() -> int:
    output = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "jetbot_scene.usda")
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac Sim online asset root is unavailable")

    robot_asset = assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
    robot_path = "/World/Jetbot"
    add_reference_to_stage(usd_path=robot_asset, prim_path=robot_path)
    robot = world.scene.add(Robot(prim_path=robot_path, name="jetbot"))
    camera = Camera(
        prim_path="/World/VerificationCamera",
        position=np.array([0.0, 0.0, 2.0]),
        frequency=20,
        resolution=(640, 640),
        orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True),
    )

    world.reset()
    camera.initialize()
    for _ in range(60):
        world.step(render=True)

    dof = int(robot.num_dof)
    if dof != 2:
        raise RuntimeError(f"Jetbot articulation check failed: expected 2 DOF, got {dof}")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    preview = os.path.splitext(output)[0] + "_preview.png"
    rgba = camera.get_rgba()
    if rgba is None or rgba.size == 0:
        raise RuntimeError("Jetbot verification image was not generated")
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(preview)

    saved = omni.usd.get_context().save_as_stage(output)
    if not saved:
        raise RuntimeError(f"Could not save stage to {output}")

    result = {
        "status": "passed",
        "robot": "NVIDIA Jetbot",
        "prim_path": robot_path,
        "degrees_of_freedom": dof,
        "asset": robot_asset,
        "scene": output,
        "preview": preview,
    }
    result_path = os.path.splitext(output)[0] + "_verification.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    carb.log_info("JETBOT_IMPORT_OK " + json.dumps(result))
    print("JETBOT_IMPORT_OK " + json.dumps(result), flush=True)
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
