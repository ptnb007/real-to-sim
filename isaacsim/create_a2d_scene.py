"""Build and validate the official A2D asset in Isaac Sim 5.1."""
import sys
sys.argv.append('--/app/vulkan=false')
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'multi_gpu': False, 'width': 960, 'height': 540})
import json
from pathlib import Path
import numpy as np
from PIL import Image
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
import omni.usd

try:
    output = Path(__file__).resolve().parent
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    asset = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Agibot/A2D/A2D.usd'
    add_reference_to_stage(asset, '/World/A2D')
    robot = world.scene.add(Robot(prim_path='/World/A2D', name='a2d', position=np.array([0., 0., 0.05])))
    camera = Camera(prim_path='/World/DemoCamera', resolution=(640, 640))
    set_camera_view(eye=np.array([3.5, 3.5, 2.5]), target=np.array([0., 0., 0.9]), camera_prim_path='/World/DemoCamera')
    set_camera_view(eye=np.array([3.5, 3.5, 2.5]), target=np.array([0., 0., 0.9]))
    world.reset()
    camera.initialize()
    initial = robot.get_joint_positions().copy()
    for _ in range(60):
        world.step(render=True)
    positions = robot.get_joint_positions()
    if robot.num_dof < 14 or not np.isfinite(positions).all():
        raise RuntimeError('Invalid A2D articulation')
    rgba = camera.get_rgba()
    if rgba is None or not rgba.size:
        raise RuntimeError('No rendered camera frame')
    Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(output / 'a2d_preview.png')
    world.reset()
    if not omni.usd.get_context().save_as_stage(str(output / 'a2d_scene.usda')):
        raise RuntimeError('Scene save failed')
    result = {'status': 'passed', 'asset': asset, 'dof': int(robot.num_dof), 'joint_names': list(robot.dof_names), 'initial_positions': initial.tolist(), 'simulation_frames': 60}
    (output / 'a2d_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('A2D_VERIFIED ' + json.dumps(result), flush=True)
finally:
    app.close()
